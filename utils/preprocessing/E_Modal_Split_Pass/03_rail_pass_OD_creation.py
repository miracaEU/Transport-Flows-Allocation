# Purpose:
#   - Build station-to-station OD flows for trips.
#   - Map ETIS zones to NUTS3, convert trips to trips using occupancy,
#     downscale NUTS3 OD to stations by population weights, prune tiny flows,
#     apply population growth ratios, and save parquet outputs.
#
# Key inputs:
#   - TRAIN_OD_FILE: ETIS OD CSVs.
#   - ETIS_NUTS_MAP: Excel mapping ETISZONE3 -> NUTS3.
#   - POP_TSV: EUROSTAT population time series (for growth ratios).
#   - nodes_with_population.parquet: nodes with population_node and population_NUTS3.
#
# Main steps:
#   1) load_etis_mapping: read Excel and normalize columns (ETISZONE3, NUTS3).
#   2) process_od_data: read OD, rename trips column, map ETIS->NUTS3, compute trips via occupancy.
#   3) downscale_nuts3_od_to_nodes: distribute NUTS3 OD to nodes using population weights.
#   4) apply_population_growth: adjust by NUTS3 growth ratios (2010->2021).
#   5) process_mode/main: orchestrate for cars and buses and write parquet files.
#
# Assumptions:
#   - nodes_df has columns: node_id, NUTS3, population_node, population_NUTS3.
#   - Zero/negative population nodes are filtered out before downscaling.
#   - ETIS OD has ORIGINZONE_3_ID and DESTINATIONZONE_3_ID fields.

import os
import pandas as pd
from pathlib import Path
import pandas as pd
import numpy as np
import time
import pyarrow as pa
import pyarrow.parquet as pq
import sys

# Define constants for file paths used throughout the project
BASE_DIR = "/soge-home/projects/mistral/miraca/"

# Population density TSV file
POP_TSV = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "population_data", "pop_variation_EUROSTAT.tsv")

# Add Rail OD file (ETIS rail tripsengers)
RAIL_OD_FILE = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "ETISplus", "TXT", "etis_2010_modelled", "p_transport_rail.csv")
# Output directory for rail (per-pair parquet like cars/buses)
OUT_RAIL = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "data_passsenger_OD", "rail_OD_pairs_100")

# ETIS to NUTS3 mapping file
ETIS_NUTS_MAP = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "ETISplus", "Additional", "ETIS2010_Zones_N2006_V2", "ETIS 2006 V2", "ETIS2006_LEVEL_ALL.xls")

# Load nodes_with_population parquet
NODES_WITH_POP = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "population_data", "rail_nodes_with_population.parquet")
nodes_df = pd.read_parquet(NODES_WITH_POP)
def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _areas_map(nodes_df: pd.DataFrame, area_col: str = "area") -> dict:
    a = nodes_df.set_index("node_id")[area_col] if area_col in nodes_df.columns else None
    if a is None:
        return {}
    # Keep node_id keys as strings; ensure numeric areas with fallback 1.0
    return pd.to_numeric(a, errors="coerce").fillna(1.0).to_dict()

def _optimize_chunk_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "trips" in d:
        d["trips"] = pd.to_numeric(d["trips"], errors="coerce").fillna(0).astype("int32")
    # Keep NUTS as strings to avoid per-chunk dictionary issues
    if "origin_nuts3" in d.columns: d["origin_nuts3"] = d["origin_nuts3"].astype(str)
    if "dest_nuts3" in d.columns:   d["dest_nuts3"]   = d["dest_nuts3"].astype(str)
    # Keep node IDs as strings (do not categorical-encode)
    for col in ("origin_node", "dest_node"):
        if col in d.columns:
            d[col] = d[col].astype(str)
    return d

def _downscale_pair_to_nodes(o_nuts: str,
                             d_nuts: str,
                             base_val: float,
                             nuts_nodes: dict[str, np.ndarray],
                             area_map: dict,
                             small_flow_threshold: float) -> pd.DataFrame:
    """
    Downscale one NUTS3 pair to node-node flows, remove diagonals by construction,
    redistribute 'small' flows within the pair, and return a compact DataFrame.

    Threshold for 'small' flows computed per pair: min(1, 5th percentile of pair flows).
    """
    if o_nuts not in nuts_nodes or d_nuts not in nuts_nodes:
        return pd.DataFrame(columns=["origin_node","dest_node","origin_nuts3","dest_nuts3","trips"])

    # Origin/Dest arrays
    o_rec = nuts_nodes[o_nuts]  # (node_id, pop_node, pop_nuts)
    d_rec = nuts_nodes[d_nuts]

    o_ids = o_rec["node_id"]; o_pn = o_rec["population_node"].astype(float); o_pN = float(o_rec["population_NUTS3"][0])
    d_ids = d_rec["node_id"]; d_pn = d_rec["population_node"].astype(float); d_pN = float(d_rec["population_NUTS3"][0])

    if o_pN <= 0 or d_pN <= 0 or base_val <= 0:
        return pd.DataFrame(columns=["origin_node","dest_node","origin_nuts3","dest_nuts3","trips"])

    same = (o_nuts == d_nuts)

    # Weights matrix
    if same:
        # Avoid diagonal by excluding each node's own population from its group denominator
        denom_o = np.maximum(o_pN - o_pn, 1e-9)            # shape (O,)
        denom_d = np.maximum(d_pN - d_pn, 1e-9)            # shape (D,)
        w_o = o_pn / denom_o                                # shape (O,)
        w_d = d_pn / denom_d                                # shape (D,)
    else:
        w_o = o_pn / o_pN
        w_d = d_pn / d_pN

    W = np.outer(w_o, w_d)                                  # shape (O,D)

    # Zero true diagonal pairs if same NUTS3 and node ids match
    if same:
        # Build mask of matches between origin nodes and dest nodes
        d_index = {nid: j for j, nid in enumerate(d_ids)}
        diag_idx = [ (i, d_index[nid]) for i, nid in enumerate(o_ids) if nid in d_index ]
        if diag_idx:
            ii, jj = zip(*diag_idx)
            W[np.array(ii), np.array(jj)] = 0.0

    flows = base_val * W                                    # trips for this NUTS-pair

    # Compute pair threshold and redistribute small flows to remaining ones using area weights
    v = flows.ravel()
    pos = v[v > 0]
    if pos.size == 0:
        return pd.DataFrame(columns=["origin_node","dest_node","origin_nuts3","dest_nuts3","trips"])
    p5 = float(np.quantile(pos, 0.05))
    thr = float(small_flow_threshold)

    # If no flow reaches the threshold, drop the entire pair
    if flows.max() < thr:
        return pd.DataFrame(columns=["origin_node","dest_node","origin_nuts3","dest_nuts3","trips"])

    # Otherwise, remove and redistribute small flows within the pair
#    small_mask = flows < thr
#    if np.any(small_mask):
#        transfer = float(flows[small_mask].sum())
#        flows[small_mask] = 0.0
#        ao = np.array([area_map.get(str(n), 1.0) for n in o_ids], dtype=float)
#        ad = np.array([area_map.get(str(n), 1.0) for n in d_ids], dtype=float)
#        A = np.outer(ao, ad)
#        recip_mask = flows > 0
#        Wrec = A * recip_mask
#        wsum = Wrec.sum()
#        if wsum > 0 and transfer > 0:
#            flows += (transfer * (Wrec / wsum))

    # If no flow reaches the threshold, skip removal to avoid wiping the pair
    if flows.max() >= thr:
        small_mask = flows < thr
        if np.any(small_mask):
            transfer = float(flows[small_mask].sum())
            flows[small_mask] = 0.0
            # Recipients weights = area(o) + area(d); node_id keys are strings
            ao = np.array([area_map.get(str(n), 1.0) for n in o_ids], dtype=float)  # (O,)
            ad = np.array([area_map.get(str(n), 1.0) for n in d_ids], dtype=float)  # (D,)
            A = np.outer(ao, ad)
            # Only consider recipients where flows > 0 after small removal
            recip_mask = flows > 0
            Wrec = A * recip_mask
            wsum = Wrec.sum()
            if wsum > 0 and transfer > 0:
                flows += (transfer * (Wrec / wsum))

    # Build compact DataFrame, dropping zeros
    oi, dj = np.nonzero(flows > 0)
    if oi.size == 0:
        return pd.DataFrame(columns=["origin_node","dest_node","origin_nuts3","dest_nuts3","trips"])
    out = pd.DataFrame({
        "origin_node": o_ids[oi],
        "dest_node": d_ids[dj],
        "origin_nuts3": o_nuts,
        "dest_nuts3": d_nuts,
        "trips": flows[oi, dj]
    })
    return out

def downscale_streaming_write(od_df: pd.DataFrame,
                              nodes_df: pd.DataFrame,
                              origin_col: str,
                              dest_col: str,
                              veh_col: str,
                              out_path: Path,
                              growth_ratio: dict[str, float] | None = None,
                              progress_every: int = 100,
                              small_flow_threshold: float = 1.0) -> None:
    """
    Memory-lean path: iterate pair-by-pair, downscale to nodes, remove diagonals,
    redistribute small flows per pair, and write incrementally to Parquet.
    """
    # Prep nodes by NUTS and areas
    nodes_ok = nodes_df[(nodes_df["population_node"] > 0) & (nodes_df["population_NUTS3"] > 0)].copy()
    nuts_nodes = {k: g[["node_id","population_node","population_NUTS3"]].to_records(index=False)
                  for k, g in nodes_ok.groupby("NUTS3")}
    area_map = _areas_map(nodes_df, area_col="area")
    # Aggregate OD by NUTS3 pair
    pairs = (od_df[[origin_col, dest_col, veh_col]]
             .groupby([origin_col, dest_col], as_index=False)[veh_col]
             .sum())
    total = len(pairs)
    if total == 0:
        empty = pd.DataFrame(columns=["origin_node","dest_node","origin_nuts3","dest_nuts3",veh_col])
        empty.to_parquet(out_path, index=False)
        return

    # Use a directory to store per-pair files
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()   

    for i, (o_nuts, d_nuts, base_val) in enumerate(pairs[[origin_col, dest_col, veh_col]].itertuples(index=False, name=None), 1):
        # Apply growth per pair
        if growth_ratio is not None:
            grow = ((growth_ratio.get(o_nuts, 1.0) + growth_ratio.get(d_nuts, 1.0)) / 2.0)
            base_val = base_val * float(grow)

        chunk = _downscale_pair_to_nodes(o_nuts, d_nuts, base_val, nuts_nodes, area_map, small_flow_threshold)
        if not chunk.empty:
            chunk = chunk.rename(columns={"trips": veh_col})
            chunk[veh_col] = np.ceil(pd.to_numeric(chunk[veh_col], errors="coerce")).astype("int32")
            chunk = _optimize_chunk_for_parquet(chunk)

            # Safe filename: origin_dest.parquet
            fname = f"{o_nuts}__{d_nuts}.parquet"
            pair_path = out_dir / fname
            chunk.to_parquet(pair_path, index=False)

        if (i % max(1, progress_every) == 0) or i == total:
            elapsed = time.time() - start
            rate = max(elapsed / i, 1e-6)
            eta = rate * (total - i)
            print(f"Done {o_nuts} → {d_nuts}, {i}/{total} pairs | elapsed {_fmt_hms(elapsed)} | ETA {_fmt_hms(eta)}", flush=True)

def apply_population_growth(df: pd.DataFrame,
                            origin_nuts_col: str,
                            dest_nuts_col: str,
                            veh_col: str,
                            growth_ratio: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    o_ratio = df[origin_nuts_col].map(growth_ratio).fillna(1.0)
    d_ratio = df[dest_nuts_col].map(growth_ratio).fillna(1.0)
    adj = (o_ratio + d_ratio) / 2.0
    df[veh_col] = df[veh_col] * adj
    return df

def load_population_growth_ratios(tsv_path: str) -> dict[str, float]:
    df = pd.read_csv(tsv_path, sep=r"\s+", engine="python")
    geo_col = next((c for c in df.columns if c.lower().startswith("geo")), "geo")
    year_cols = [c for c in df.columns if c.isdigit()]
    for y in ["2010", "2021"]:
        if y not in year_cols:
            raise KeyError(f"Year {y} not found in population TSV.")
    df = df[[geo_col, "2010", "2021"]].copy()
    df = df[df[geo_col].str.len() == 5]  # NUTS3 codes
    df["2010"] = pd.to_numeric(df["2010"], errors="coerce")
    df["2021"] = pd.to_numeric(df["2021"], errors="coerce")
    df["ratio_2021_2010"] = np.where(df["2010"] > 0, df["2021"] / df["2010"], 1.0)
    return dict(zip(df[geo_col], df["ratio_2021_2010"]))


def load_etis_mapping(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    # Decide reader based on extension
    ext = p.suffix.lower()
    if ext in (".xls", ".xlsx"):
        # Read first sheet (or a specific one if needed)
        try:
            mp = pd.read_excel(path, sheet_name=0, engine=None)
        except Exception:
            # Fallback engine
            mp = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    else:
        # Fallback to CSV behavior if not Excel
        mp = pd.read_csv(path)

    # Standardize column names (case-insensitive)
    rename_map = {}
    for c in mp.columns:
        cl = c.lower()
        if cl in ("nuts_id", "nuts3"):
            rename_map[c] = "NUTS3"
        if cl in ("etiszone3_id", "etiszone3", "etis_zone3_id"):
            rename_map[c] = "ETISZONE3"
    mp = mp.rename(columns=rename_map)

    if "NUTS3" not in mp.columns or "ETISZONE3" not in mp.columns:
        raise KeyError(f"Required columns not found. Columns: {list(mp.columns)}")

    mp["NUTS3"] = mp["NUTS3"].astype(str)
    mp["ETISZONE3"] = mp["ETISZONE3"].astype(str)
    mp = mp.drop_duplicates(subset=["ETISZONE3"])
    return mp[["ETISZONE3", "NUTS3"]].copy()

def translate_etis_to_nuts(od_df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Map ETIS zone IDs to NUTS3 for origin and destination.
    Expects od_df to have ORIGINZONE_3_ID and DESTINATIONZONE_3_ID (or similar),
    and mapping with columns ETISZONE3, NUTS3.
    """
    df = od_df.copy()

    # Pick origin/dest ETIS columns robustly
    origin_col = next((c for c in df.columns if c.upper().startswith("ORIGINZONE_3")), None)
    dest_col   = next((c for c in df.columns if c.upper().startswith("DEST_ZONE_3")), None)
    if origin_col is None or dest_col is None:
        # Fallback names
        origin_col = origin_col or "ORIGINZONE_3_ID"
        dest_col   = dest_col or "DEST_ZONE_3_ID"
        if origin_col not in df.columns or dest_col not in df.columns:
            raise KeyError(f"ETIS OD missing origin/dest zone columns. Available: {list(df.columns)}")

    # Normalize types
    df[origin_col] = df[origin_col].astype(str)
    df[dest_col]   = df[dest_col].astype(str)

    # Prepare mapping
    mp = mapping.copy()
    mp["ETISZONE3"] = mp["ETISZONE3"].astype(str)
    mp["NUTS3"]     = mp["NUTS3"].astype(str)

    # Map to NUTS3
    df = df.merge(mp.rename(columns={"ETISZONE3": "ORIGINZONE_3_ID", "NUTS3": "ORIGIN_NUTS3_ID"}),
                  left_on=origin_col, right_on="ORIGINZONE_3_ID", how="left")
    df = df.merge(mp.rename(columns={"ETISZONE3": "DEST_ZONE_3_ID", "NUTS3": "DESTINATION_NUTS3_ID"}),
                  left_on=dest_col, right_on="DEST_ZONE_3_ID", how="left")

    # Keep required columns
    if df["ORIGIN_NUTS3_ID"].isna().any() or df["DESTINATION_NUTS3_ID"].isna().any():
        # Drop rows without mapping
        df = df.dropna(subset=["ORIGIN_NUTS3_ID", "DESTINATION_NUTS3_ID"])

    return df

def _redistribute_small_within_pairs(od_df: pd.DataFrame,
                                     origin_col: str = 'ORIGIN_NUTS3_ID',
                                     dest_col: str = 'DESTINATION_NUTS3_ID',
                                     veh_col: str = 'trips',
                                     threshold_year: float = 1.0) -> pd.DataFrame:
    """
    For rows within the same (origin NUTS3, destination NUTS3) group:
      - Identify entries with value < threshold_year (e.g., 1 trips/year).
      - Sum those 'small' values and redistribute proportionally to the remaining
        entries in the group (by their value), then drop the small entries.
    """
    df = od_df.copy()
    df[veh_col] = pd.to_numeric(df[veh_col], errors='coerce').fillna(0.0)

    def _fix_group(g: pd.DataFrame) -> pd.DataFrame:
        vals = g[veh_col].to_numpy(dtype=float)
        small_mask = vals < threshold_year
        if not np.any(small_mask):
            return g
        transfer = float(vals[small_mask].sum())
        keep_mask = ~small_mask
        if transfer > 0 and np.any(keep_mask):
            weights = np.maximum(g.loc[keep_mask, veh_col].to_numpy(dtype=float), 1e-9)
            wsum = float(weights.sum())
            if wsum > 0:
                g.loc[keep_mask, veh_col] = g.loc[keep_mask, veh_col].to_numpy(dtype=float) + transfer * (weights / wsum)
        # Drop the small entries
        return g.loc[keep_mask]

    fixed = []
    for (_, _), grp in df.groupby([origin_col, dest_col], as_index=False):
        fixed.append(_fix_group(grp))
    return pd.concat(fixed, ignore_index=True) if fixed else df

def process_od_data(od_file: Path,
                    mapping_file: Path,
                    trips_col: str) -> pd.DataFrame:
    od_df = pd.read_csv(od_file)

    if trips_col not in od_df.columns:
        raise KeyError(f"Trips column '{trips_col}' not found. Available: {list(od_df.columns)}")
    if trips_col != 'trips':
        od_df = od_df.rename(columns={trips_col: 'trips'})

    # Map ETIS zones to NUTS3 first
    mapping = load_etis_mapping(str(mapping_file))
    od_df = translate_etis_to_nuts(od_df, mapping)

    # Redistribute entries < 1 trips/year within the same NUTS3 pair
    od_df = _redistribute_small_within_pairs(
        od_df,
        origin_col='ORIGIN_NUTS3_ID',
        dest_col='DESTINATION_NUTS3_ID',
        veh_col='trips',
        threshold_year=1.0
    )

    # Aggregate to pair totals 
    od_df = (od_df[['ORIGIN_NUTS3_ID', 'DESTINATION_NUTS3_ID', 'trips']]
             .groupby(['ORIGIN_NUTS3_ID', 'DESTINATION_NUTS3_ID'], as_index=False)['trips']
             .sum())

    return od_df[['ORIGIN_NUTS3_ID', 'DESTINATION_NUTS3_ID', 'trips']]

def _optimize_chunk_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "trips" in d:
        d["trips"] = pd.to_numeric(d["trips"], errors="coerce").fillna(0).astype("int32")
    # Keep NUTS as strings to avoid per-chunk dictionary issues
    if "origin_nuts3" in d.columns: d["origin_nuts3"] = d["origin_nuts3"].astype(str)
    if "dest_nuts3" in d.columns:   d["dest_nuts3"]   = d["dest_nuts3"].astype(str)
    # Nodes can stay categorical
    for col in ("origin_node", "dest_node"):
        if col in d.columns:
            d[col] = d[col].astype("category")
    return d

def count_pair_rows(out_dir: Path) -> int:
    """Count total OD rows across all per-pair parquet files."""
    out_dir = Path(out_dir)
    total = 0
    for f in out_dir.glob("*.parquet"):
        try:
            # fast metadata read
            import pyarrow.parquet as pq
            total += pq.ParquetFile(f).metadata.num_rows
        except Exception:
            # fallback to pandas
            total += len(pd.read_parquet(f))
    return total


def process_mode(trips_file: Path,
                 trips_col: str,
                 out_file: Path,
                 mode_name: str,
                 growth_ratios: dict[str, float],
                 nodes_df: pd.DataFrame) -> None:
    print(f"[{mode_name}] Loading + converting OD...", flush=True)
    od_nuts = process_od_data(trips_file, ETIS_NUTS_MAP, trips_col)
    origin_col = "ORIGIN_NUTS3_ID"
    dest_col = "DESTINATION_NUTS3_ID"
    # Mode-specific small flow threshold: keep <1 trips/year redistribution logic
    small_thr = 100.0
    downscale_streaming_write(
        od_df=od_nuts,
        nodes_df=nodes_df,
        origin_col=origin_col,
        dest_col=dest_col,
        veh_col="trips",
        out_path=out_file,
        growth_ratio=growth_ratios,
        progress_every=100,
        small_flow_threshold=small_thr
    )
    total_rows = count_pair_rows(out_file)
    print(f"[{mode_name}] Total OD rows written: {total_rows}", flush=True)

def main():
    # Load population growth ratios
    growth_ratios = load_population_growth_ratios(POP_TSV)
    print("Population growth ratios loaded.")

    # Process rail trips 
    # Use a representative average rail occupancy (seats per train or tripsengers per carriage equivalent).
    # If your ETIS "p_transport_rail.csv" is trips of tripsengers, set occupancy similar to buses or a domain estimate.
    process_mode(RAIL_OD_FILE, "p_transport_rail_trips",  OUT_RAIL, "rail", growth_ratios, nodes_df)
    print("Rail trips processed.")

if __name__ == "__main__":
    main()