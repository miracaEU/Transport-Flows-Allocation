from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define constants for file paths used throughout the project
BASE_DIR = "/soge-home/projects/mistral/miraca/"

OUT_CAR = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "data_passenger_OD", "cars_OD_pairs_reduced")
OUT_BUS = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "data_passenger_OD", "buses_OD_pairs_reduced")
OUT_CARS_SINGLE = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "data_passenger_OD", "cars_OD_merged_300_reduced.parquet")
OUT_BUSES_SINGLE = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "data_passenger_OD", "buses_OD_merged_300_reduced.parquet")

def _load_mode_dir(input_dir: Path, value_col: str) -> pd.DataFrame:
    """Load and normalize passenger OD Parquet dataset with robust fallback."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Warning: Path does not exist: {input_path}")
        return pd.DataFrame()
    
    dataset = ds.dataset(str(input_path), format="parquet")

    # Only request columns by name to avoid Expression meta-function issues.
    requested_cols = [c for c in ("origin_node", "dest_node", "origin_nuts3", "dest_nuts3", value_col)
                      if c in dataset.schema.names]
    if not requested_cols:
        return pd.DataFrame()

    # Prefer dataset.to_table; if it fails with ArrowInvalid (e.g., int8 index overflow),
    # fallback to reading individual Parquet files in parallel and concatenating.
    try:
        tbl = dataset.to_table(columns=requested_cols, use_threads=True)
    except pa.ArrowInvalid as e:
        print(f"Dataset scanner failed ({e}), falling back to parallel per-file reading...")
        
        def read_and_normalize(fpath):
            """Read a single parquet file and normalize its schema."""
            try:
                t = pq.read_table(fpath, columns=requested_cols)
                if t.num_rows == 0:
                    return None
                
                # Normalize schema: decode dictionaries and cast to consistent types
                arrays = []
                fields = []
                string_cols = {"origin_node", "dest_node", "origin_nuts3", "dest_nuts3"}
                
                for col_name in t.column_names:
                    arr = t[col_name]
                    if pa.types.is_dictionary(arr.type):
                        arr = pc.dictionary_decode(arr)
                    if col_name in string_cols:
                        arr = pc.cast(arr, pa.string())
                    elif col_name == value_col:
                        arr = pc.cast(arr, pa.float64())
                    arrays.append(arr)
                    fields.append(pa.field(col_name, arr.type))
                
                return pa.Table.from_arrays(arrays, schema=pa.schema(fields))
            except Exception:
                return None
        
        # Collect all parquet file paths
        file_paths = list(input_path.rglob('*.parquet'))
        print(f"Found {len(file_paths):,} parquet files, reading in parallel...")
        
        # Read files in parallel using ThreadPoolExecutor (good for I/O-bound tasks)
        tables = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(read_and_normalize, fpath): fpath for fpath in file_paths}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    tables.append(result)
                # Progress indicator every 1000 files
                if i % 1000 == 0:
                    print(f"  Processed {i:,}/{len(file_paths):,} files, {len(tables):,} valid tables...")
        
        print(f"Loaded {len(tables):,} valid tables, concatenating...")
        if not tables:
            return pd.DataFrame()
        tbl = pa.concat_tables(tables)

    # Decode dictionary-encoded columns and cast to appropriate types in one pass.
    arrays = {}
    string_cols = {"origin_node", "dest_node", "origin_nuts3", "dest_nuts3"}
    
    for name in requested_cols:
        arr = tbl[name]
        # Decode dictionaries
        if pa.types.is_dictionary(arr.type):
            arr = pc.dictionary_decode(arr)
        # Cast string columns
        if name in string_cols:
            arr = pc.cast(arr, pa.string())
        # Cast value column to float64
        elif name == value_col:
            arr = pc.cast(arr, pa.float64())
        arrays[name] = arr

    # Build a normalized table from arrays.
    norm_tbl = pa.Table.from_arrays(list(arrays.values()), names=list(arrays.keys()))

    # Apply filters post-materialization: drop UK/TR and non-positive/small values.
    mask = None
    if {"origin_nuts3", "dest_nuts3"}.issubset(arrays.keys()):
        o, d = norm_tbl["origin_nuts3"], norm_tbl["dest_nuts3"]
        # Combine all UK/TR checks in one expression
        bad = pc.or_(
            pc.or_(pc.starts_with(o, "UK"), pc.starts_with(o, "TR")),
            pc.or_(pc.starts_with(d, "UK"), pc.starts_with(d, "TR"))
        )
        mask = pc.invert(bad)
    
    if value_col in arrays:
        vpos = pc.greater(norm_tbl[value_col], 0.1)
        mask = vpos if mask is None else pc.and_(mask, vpos)

    if mask is not None:
        norm_tbl = norm_tbl.filter(mask)

    df = norm_tbl.to_pandas(types_mapper=None)
    # Ensure numeric vehicles are int64 via ceil.
    if value_col in df.columns:
        df[value_col] = np.ceil(pd.to_numeric(df[value_col], errors="coerce")).astype("int64")
    return df

def _print_percentiles(s: pd.Series, label: str) -> None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        print(f"[{label}] No data for percentiles")
        return
    qlist = [i/100 for i in range(5, 100, 5)]
    qvals = s.quantile(qlist).to_dict()
    pretty = ", ".join([f"P{int(k*100)}={int(v) if float(v).is_integer() else round(float(v),2)}"
                        for k, v in qvals.items()])
    print(f"[{label}] Percentiles (5..95): {pretty}")

def _process_mode(mode_name: str, input_dir: str, output_path: str, value_col: str) -> None:
    """Load, analyze, and save a single mode's OD data."""
    t0 = time.time()
    df = _load_mode_dir(Path(input_dir), value_col)
    elapsed = time.time() - t0
    
    if not df.empty and value_col in df.columns:
        _print_percentiles(df[value_col], mode_name)
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not df.empty:
        df.to_parquet(out_path, index=False)
        print(f"[{mode_name}] Saved: {out_path} | rows={len(df):,} cols={len(df.columns)} time={elapsed:.1f}s")
    else:
        print(f"[{mode_name}] No data to save. time={elapsed:.1f}s")

def save_modes(cars_dir: str, buses_dir: str, cars_out: str, buses_out: str) -> None:
    """Process and save both buses and cars OD datasets."""
    _process_mode("buses", buses_dir, buses_out, "buses_vehicles")
    _process_mode("cars", cars_dir, cars_out, "cars_vehicles")

# Run
save_modes(OUT_CAR, OUT_BUS, OUT_CARS_SINGLE, OUT_BUSES_SINGLE)