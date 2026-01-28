from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd
import numpy as np
import os 
import sys

# Define constants for file paths used throughout the project
BASE_DIR = "/soge-home/projects/mistral/miraca/"

OUT_RAIL = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "data_passenger_OD", "rail_OD_pairs_100")
OUT_RAIL_SINGLE = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "data_passenger_OD", "rail_OD_merged_100.parquet")


def _load_mode_dir(input_dir: Path, value_col: str) -> pd.DataFrame:
    dataset = ds.dataset(str(input_dir), format="parquet")
    schema = dataset.schema

    cols = ["origin_node","dest_node","origin_nuts3","dest_nuts3", value_col]
    cols = [c for c in cols if c in schema.names]
    if value_col not in schema.names:
        candidates = [c for c in schema.names if c.endswith("_trips") or c == "trips"]
        if len(candidates) != 1:
            return pd.DataFrame()
        value_col = candidates[0]
        cols = [c for c in ["origin_node","dest_node","origin_nuts3","dest_nuts3", value_col] if c in schema.names]

    # Decode dictionary-encoded columns to strings
    tables: list[pa.Table] = []
    for frag in dataset.get_fragments():
        t = frag.to_table(columns=cols)
        for c in ("origin_node", "dest_node", "origin_nuts3", "dest_nuts3"):
            if c in t.schema.names and pa.types.is_dictionary(t[c].type):
                decoded = pc.dictionary_decode(t[c])
                t = t.set_column(t.schema.get_field_index(c), pa.field(c, decoded.type), decoded)
        tables.append(t)

    if not tables:
        return pd.DataFrame()
    try:
        tbl = pa.concat_tables(tables, promote_options="default")
    except TypeError:
        tbl = pa.concat_tables(tables, promote=True)

    # To pandas first, then filter UK/TR and clean
    df = tbl.to_pandas(types_mapper=lambda t: pd.ArrowDtype(t))

    for nuts in ("origin_nuts3", "dest_nuts3"):
        if nuts in df.columns:
            df[nuts] = df[nuts].astype(str)

    if {"origin_nuts3","dest_nuts3"}.issubset(df.columns):
        mask_bad = df["origin_nuts3"].str.startswith(("UK","TR")) | df["dest_nuts3"].str.startswith(("UK","TR"))
        df = df[~mask_bad].copy()

    if value_col in df.columns:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df[df[value_col] > 0.4].copy()
        df[value_col] = np.ceil(df[value_col]).astype("int64")

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

def save_modes(rail_dir: Path, rail_out: Path) -> None:
    rail = _load_mode_dir(Path(rail_dir), "trips")

    if not rail.empty and "trips" in rail.columns:
        _print_percentiles(rail["trips"], "rail")

    rail_out = Path(rail_out); rail_out.parent.mkdir(parents=True, exist_ok=True)

    if not rail.empty:
        rail.to_parquet(rail_out, index=False)
        print(f"[rail] Saved: {rail_out} | rows={len(rail)} cols={len(rail.columns)}")
    else:
        print("[rail] No data to save.")

# Run
save_modes(OUT_RAIL, OUT_RAIL_SINGLE)