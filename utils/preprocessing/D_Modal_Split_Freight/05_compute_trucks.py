import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd

# Data sources for spatial assignment
ROAD_NODES_PARQUET = Path("/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_road_nodes_TENT.parquet")  # EPSG:3857, column 'id'
NUTS3_PARQUET = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/admin/NUTS/NUTS3.parquet")  # EPSG:3035

# Inputs
ROAD_OD_FILE = Path("/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/road_OD_final.parquet")
# Output
OUT_FILE = Path("/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/road_OD_with_trucks_300.parquet")

# Truck capacity (tons) by commodity: mean and std
CAPACITY_STATS = {
    "IL": (16.0, 0.8),
    "MI": (16.5, 0.66),
    "FB": (13.0, 0.4),
    "CH": (16.0, 0.5),
    "MT": (12.0, 0.35),
    "PW": (13.0, 0.35),
    "WW": (8.0, 0.25),
    "CO": (9.5, 0.25),
}

def _assign_nuts3_to_od(od_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign origin_nuts3 and dest_nuts3 by spatially joining node points (from_id/to_id)
    to NUTS3 polygons. Nodes are provided in EPSG:3857, NUTS3 in EPSG:3035.
    """
    if od_df.empty:
        return od_df

    # Load geodata
    nodes = gpd.read_parquet(ROAD_NODES_PARQUET)
    nuts3 = gpd.read_parquet(NUTS3_PARQUET)

    # Ensure geometry and CRS
    if nodes.crs is None:
        nodes = nodes.set_crs(3857)
    elif nodes.crs.to_epsg() != 3857:
        nodes = nodes.to_crs(3857)

    if nuts3.crs is None:
        nuts3 = nuts3.set_crs(3035)
    elif nuts3.crs.to_epsg() != 3035:
        nuts3 = nuts3.to_crs(3035)

    # Project nodes to NUTS3 CRS for join
    nodes_3035 = nodes.to_crs(3035)

    # Minimal columns
    nid_col = "id" if "id" in nodes_3035.columns else ("node_id" if "node_id" in nodes_3035.columns else None)
    nuts_id_col = next((c for c in nuts3.columns if c.upper().startswith("NUTS3")), None)
    if nid_col is None:
        raise KeyError("Road nodes must contain 'id' or 'node_id'.")
    if nuts_id_col is None:
        # fallback common names
        nuts_id_col = next((c for c in nuts3.columns if c.upper() in ("NUTS_ID", "NUTS3_CODE", "NUTS3")), None)
    if nuts_id_col is None:
        raise KeyError(f"NUTS3 parquet must contain a NUTS3 code column. Available: {list(nuts3.columns)}")

    nodes_3035[nid_col] = nodes_3035[nid_col].astype(str)
    nuts3[nuts_id_col] = nuts3[nuts_id_col].astype(str)

    # Spatial join nodes -> NUTS3
    node_nuts = gpd.sjoin(
        nodes_3035[[nid_col, "geometry"]],
        nuts3[[nuts_id_col, "geometry"]],
        how="left",
        predicate="within"
    ).rename(columns={nuts_id_col: "NUTS3"})

    # Build mappings
    id_to_nuts = node_nuts.set_index(nid_col)["NUTS3"].to_dict()

    d = od_df.copy()
    d["from_id"] = d["from_id"].astype(str)
    d["to_id"] = d["to_id"].astype(str)
    d["origin_nuts3"] = d["from_id"].map(id_to_nuts)
    d["dest_nuts3"] = d["to_id"].map(id_to_nuts)
    return d

def _sample_capacity(commodity: str, size: int) -> np.ndarray:
    mean, std = CAPACITY_STATS.get(commodity, (16.0, 0.8))
    # Truncated to avoid non-positive capacities
    vals = np.random.normal(loc=mean, scale=std, size=size)
    return np.clip(vals, a_min=max(0.1, mean - 5*std), a_max=mean + 5*std)

def _redistribute_small_and_self_loops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redistribute rows with trucks per year < 1 and self-loops (from_id == to_id)
    within the same (origin_nuts3, dest_nuts3) pair, proportionally to recipients' total_tons_per_year.
    If NUTS3 columns are missing, skip redistribution.
    """
    if not {"origin_nuts3", "dest_nuts3", "from_id", "to_id", "total_tons_per_year", "trucks_per_year"}.issubset(df.columns):
        return df

    d = df.copy()
    d["total_tons_per_year"] = pd.to_numeric(d["total_tons_per_year"], errors="coerce").fillna(0.0)
    d["trucks_per_year"] = pd.to_numeric(d["trucks_per_year"], errors="coerce").fillna(0.0)

    def _apply_group(g: pd.DataFrame) -> pd.DataFrame:
        small = g["trucks_per_year"] <300.0
        diag = g["from_id"].astype(str) == g["to_id"].astype(str)
        donors = g.loc[small | diag].copy()
        recip  = g.loc[~(small | diag)].copy()

        transfer = float(donors["trucks_per_year"].sum())
        donors["trucks_per_year"] = 0.0

        if transfer > 0 and not recip.empty:
            w = pd.to_numeric(recip["total_tons_per_year"], errors="coerce").fillna(0.0).to_numpy()
            wsum = float(w.sum())
            if wsum <= 0.0:
                add = np.full(len(recip), transfer / len(recip))
            else:
                add = transfer * (w / wsum)
            recip["trucks_per_year"] = np.ceil(pd.to_numeric(recip["trucks_per_year"], errors="coerce").fillna(0.0) + add).astype("int32")
        donors["trucks_per_year"] = donors["trucks_per_year"].astype("int32")
        return pd.concat([recip, donors], ignore_index=True)

    return d.groupby(["origin_nuts3", "dest_nuts3"], group_keys=False).apply(_apply_group).reset_index(drop=True)

def compute_trucks_per_od(od_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Input:
      od_df columns: from_id, to_id, value, origin_sector
      - value is in thousand tons per year (ths tons/year)
      - origin_sector is the commodity code (e.g., IL, MI, ...)
    Output:
      DataFrame grouped by (from_id, to_id) with:
        - total_tons_per_year
        - share_<commodity>
        - trucks_<commodity> (Monte Carlo on capacity)
        - trucks_per_year
    """
    np.random.seed(seed)

    # Clean and ensure types
    df = od_df.copy()
    df["from_id"] = df["from_id"].astype(str)
    df["to_id"] = df["to_id"].astype(str)
    df["origin_sector"] = df["origin_sector"].astype(str).str.upper().str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)  # ths tons/year
    df["tons_per_year"] = (df["value"] * 1000.0)

    # Keep only positive flows
    df = df[df["tons_per_year"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["from_id","to_id","total_tons_per_year","trucks_per_year"])

        # Aggregate totals per (from_id, to_id, commodity) in tons/year
    by_pair_comm = (
        df.groupby(["from_id", "to_id", "origin_sector"], as_index=False)["tons_per_year"]
          .sum()
          .rename(columns={"tons_per_year": "tons_per_year_comm"})
    )

    # Totals per pair (tons/year)
    pair_totals = (
        by_pair_comm.groupby(["from_id", "to_id"], as_index=False)["tons_per_year_comm"]
        .sum()
        .rename(columns={"tons_per_year_comm": "total_tons_per_year"})
    )

    # Attach NUTS3 (optional)
    if {"origin_nuts3","dest_nuts3"}.issubset(df.columns):
        pair_nuts = df.groupby(["from_id","to_id"], as_index=False)[["origin_nuts3","dest_nuts3"]].first()
        pair_totals = pair_totals.merge(pair_nuts, on=["from_id","to_id"], how="left")

    # 1) Compute initial shares per commodity (before any capping)
    od_shares = by_pair_comm.merge(pair_totals[["from_id","to_id","total_tons_per_year"]], on=["from_id","to_id"], how="left")
    od_shares["share"] = np.where(od_shares["total_tons_per_year"] > 0,
                                  od_shares["tons_per_year_comm"] / od_shares["total_tons_per_year"], 0.0)

    # 2) Cap totals (p99.999) and redistribute excess to non-capped pairs within same NUTS3
    pair_totals["total_tons_per_year"] = pd.to_numeric(pair_totals["total_tons_per_year"], errors="coerce").fillna(0.0)
    cap = float(pair_totals["total_tons_per_year"].quantile(0.99999))
    print(cap)
    group_cols = ["origin_nuts3","dest_nuts3"] if {"origin_nuts3","dest_nuts3"}.issubset(pair_totals.columns) else None

    def _cap_group(g: pd.DataFrame) -> pd.DataFrame:
        w = g["total_tons_per_year"].to_numpy(dtype=float)

        # Identify donors and initial excess
        donors = w > cap
        if not donors.any():
            return g
        excess = float((w[donors] - cap).sum())
        w[donors] = cap

        # Iteratively distribute excess to recipients with headroom
        while excess > 1e-9:
            recip_mask = w < cap
            if not recip_mask.any():
                # No headroom left; stop
                break
            recip_w = w[recip_mask]
            headroom = cap - recip_w
            total_weight = float(recip_w.sum())
            if total_weight <= 0.0:
                # Equal split across headroom
                add = np.minimum(headroom, excess / recip_w.shape[0])
            else:
                # Proportional to current weight
                add = excess * (recip_w / total_weight)
                # Respect headroom
                add = np.minimum(add, headroom)

            added = float(add.sum())
            w[recip_mask] = recip_w + add
            excess -= added
            # Loop continues; if any excess remains after hitting headroom caps, try another round

        g = g.copy()
        g["total_tons_per_year"] = w
        return g
    pair_totals = (pair_totals.groupby(group_cols, group_keys=False).apply(_cap_group)
                   if group_cols else _cap_group(pair_totals))

    # 3) Recompute per-commodity tons using shares * capped totals (shares stay the same, sum to 1)
    capped_tot_map = pair_totals.set_index(["from_id","to_id"])["total_tons_per_year"]
    od_shares["capped_total"] = od_shares.set_index(["from_id","to_id"]).index.map(capped_tot_map).astype(float).fillna(0.0)
    od_shares["tons_per_year_comm"] = od_shares["share"] * od_shares["capped_total"]

    # 4) Compute trucks from capped per-commodity tons
    capacities = _sample_capacity_series(od_shares["origin_sector"])
    od_shares["trucks"] = od_shares["tons_per_year_comm"] / capacities

    # Pivot shares and trucks to wide per pair
    share_piv = (od_shares.groupby(["from_id","to_id","origin_sector"], as_index=False)
                 .agg(share=("share","sum")))
    trucks_piv = (od_shares.groupby(["from_id","to_id","origin_sector"], as_index=False)
                  .agg(trucks=("trucks","sum")))

    share_piv = share_piv.pivot_table(index=["from_id","to_id"], columns="origin_sector", values="share", aggfunc="sum").fillna(0.0)
    trucks_piv = trucks_piv.pivot_table(index=["from_id","to_id"], columns="origin_sector", values="trucks", aggfunc="sum").fillna(0.0)

    share_piv.columns = [f"share_{c}" for c in share_piv.columns]
    trucks_piv.columns = [f"trucks_{c}" for c in trucks_piv.columns]

    out = pair_totals.set_index(["from_id","to_id"]).join(share_piv).join(trucks_piv).reset_index()

    # Round trucks and sum
    truck_cols = [c for c in out.columns if c.startswith("trucks_")]
    for c in truck_cols:
        out[c] = np.ceil(pd.to_numeric(out[c], errors="coerce").fillna(0.0)).astype("int32")
    out["trucks_per_year"] = out[truck_cols].sum(axis=1).astype("int32")

    # Keep origin/dest NUTS3 if present
    for c in ("origin_nuts3", "dest_nuts3"):
        if c in df.columns:
            out[c] = df.groupby(["from_id","to_id"])[c].first().reindex(out.set_index(["from_id","to_id"]).index).values

    return out

def _sample_capacity_series(comms: pd.Series) -> np.ndarray:
    vals = np.empty(len(comms), dtype=float)
    # Vectorized by grouping commodities
    idx_by_comm = {}
    for i, c in enumerate(comms.astype(str).values):
        idx_by_comm.setdefault(c, []).append(i)
    for c, idxs in idx_by_comm.items():
        vals[idxs] = _sample_capacity(c, size=len(idxs))
    return vals


def main():
    # Load road OD parquet produced earlier (contains commodity in origin_sector)
    try:
        od = pd.read_parquet(ROAD_OD_FILE)
    except Exception as e:
        raise RuntimeError(f"Failed to read road OD: {e}")

    # Assign NUTS3 by location of from_id/to_id
    od = _assign_nuts3_to_od(od)

    result = compute_trucks_per_od(od, seed=42)

    # Redistribute small (<1) and self-loop rows within NUTS3 pairs (if NUTS3 available)
    result = _redistribute_small_and_self_loops(result)

    # Continue with removing zero-truck rows before saving
    result["trucks_per_year"] = pd.to_numeric(result["trucks_per_year"], errors="coerce").fillna(0)
    result = result[result["trucks_per_year"] > 0].copy()

    # Minimal output + keep share by commodity
    share_cols = [c for c in result.columns if c.startswith("share_")]
    minimal_cols = ["from_id", "to_id", "origin_nuts3", "dest_nuts3", "total_tons_per_year", "trucks_per_year"] + share_cols
    minimal = result[minimal_cols].copy()

    minimal["from_id"] = minimal["from_id"].astype("category")
    minimal["to_id"] = minimal["to_id"].astype("category")

    # Save compressed Parquet
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        tbl = pa.Table.from_pandas(minimal, preserve_index=False)
        pq.write_table(
            tbl,
            OUT_FILE.as_posix(),
            compression="zstd",
            use_dictionary=True
        )
    except Exception:
        minimal.to_parquet(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}")

if __name__ == "__main__":
    main()