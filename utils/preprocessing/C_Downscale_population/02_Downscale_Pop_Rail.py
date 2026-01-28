# Purpose:
#   1. Optionally crop a global population raster to a fixed EUROPE_BOUNDS region (cached).
#   2. Load road network nodes (parquet), decode geometries (WKB/WKT) into a GeoDataFrame.
#   3. Normalize nodes and NUTS3 CRS to EPSG:4326.
#   4. Sample/cumulate raster population values to nearest node (blockwise or whole window).
#   5. Spatially join nodes to NUTS3; fill missing via nearest centroid in projected CRS.
#   6. Aggregate node populations to NUTS3 totals; filter out zero-pop nodes.
#   7. Cache resulting enriched node dataset (population_node, population_NUTS3).

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import xy
from rasterio.crs import CRS as RCRS
from pyproj import Transformer
from pyproj import CRS as PCRS
from scipy.spatial import cKDTree
from shapely import wkt, wkb
from pathlib import Path
import argparse
import sys

print('hello world')
EUROPE_BOUNDS = {
    "xmin": -12,
    "xmax": 32,
    "ymin": 35,
    "ymax": 72,
}

BASE_DIR = "/soge-home/projects/mistral/miraca/"
NUTS3_PARQUET = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "admin", "NUTS", "NUTS3_2010.parquet")
POP_RASTER = os.path.join(BASE_DIR, "incoming_data", "spatial_data", "population_data", "GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0.tif")

NODES_PARQUET= Path("/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_railways_stations_TENT.parquet") 
OUT_PARQUET= Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/population_data/rail_nodes_with_population.parquet")
CROPPED_RASTER_PATH = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/population_data/GHS_POP_E2020_EUROPE_CROPPED.tif")

def downscale_nuts3_od_to_nodes(od_df: pd.DataFrame,
                                nodes_df: pd.DataFrame,
                                origin_col: str,
                                dest_col: str,
                                veh_col: str) -> pd.DataFrame:
    nuts_groups = nodes_df.groupby("NUTS3")

    nuts_nodes = {
        k: g[["node_id", "population_node", "population_NUTS3"]].to_records(index=False)
        for k, g in nuts_groups
    }
    rows = []
    for r in od_df.itertuples():
        o_nuts = getattr(r, origin_col)
        d_nuts = getattr(r, dest_col)
        base_val = getattr(r, veh_col)
        if o_nuts not in nuts_nodes or d_nuts not in nuts_nodes:
            continue
        o_nodes = nuts_nodes[o_nuts]
        d_nodes = nuts_nodes[d_nuts]
        same_nuts = o_nuts == d_nuts
        for o_node, o_pop_node, o_pop_nuts in o_nodes:
            for d_node, d_pop_node, d_pop_nuts in d_nodes:
                if o_node == d_node:
                    continue
                if o_pop_nuts <= 0 or d_pop_nuts <= 0:
                    continue
                if same_nuts:
                    denom_o = max(o_pop_nuts - o_pop_node, 1e-9)
                    denom_d = max(d_pop_nuts - d_pop_node, 1e-9)
                    w = (o_pop_node / denom_o) * (d_pop_node / denom_d)
                else:
                    w = (o_pop_node / o_pop_nuts) * (d_pop_node / d_pop_nuts)
                val = base_val * w
                if val > 0:
                    rows.append((o_node, d_node, o_nuts, d_nuts, val))
    if not rows:
        return pd.DataFrame(columns=["origin_node", "dest_node", "origin_nuts3", "dest_nuts3", "trains"])
    out = pd.DataFrame(rows, columns=["origin_node", "dest_node", "origin_nuts3", "dest_nuts3", "trains"])
    return out

def _ensure_cropped_raster(pop_raster: str | Path, cropped_path: Path) -> Path:
    cropped_path = Path(cropped_path)
    if cropped_path.exists():
        return cropped_path
    src_path = Path(pop_raster)
    cropped_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        r_crs = src.crs
        if r_crs is None:
            raise ValueError("Population raster has no CRS.")
        is_wgs84 = False
        try:
            is_wgs84 = RCRS.from_epsg(4326).equals(r_crs)
        except Exception:
            is_wgs84 = False
        if not is_wgs84:
            s = r_crs.to_string()
            is_wgs84 = ("WGS 84" in s) and getattr(r_crs, "is_geographic", True)

        if is_wgs84:
            xmin, xmax = EUROPE_BOUNDS["xmin"], EUROPE_BOUNDS["xmax"]
            ymin, ymax = EUROPE_BOUNDS["ymin"], EUROPE_BOUNDS["ymax"]
        else:
            transformer = Transformer.from_crs("EPSG:4326", r_crs, always_xy=True)
            xmin, ymin = transformer.transform(EUROPE_BOUNDS["xmin"], EUROPE_BOUNDS["ymin"])
            xmax, ymax = transformer.transform(EUROPE_BOUNDS["xmax"], EUROPE_BOUNDS["ymax"])

        window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        data = src.read(1, window=window)
        transform_win = src.window_transform(window)

        profile = src.profile.copy()
        profile.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform_win,
            "compress": profile.get("compress", "lzw")
        })

        with rasterio.open(cropped_path, "w", **profile) as dst:
            dst.write(data, 1)
    return cropped_path

def load_nodes_population(nodes_parquet: str = NODES_PARQUET,
                          nuts3_parquet: str = NUTS3_PARQUET,
                          pop_raster: str = POP_RASTER,
                          nodes_crs: str | int | None = None,
                          nuts3_code_col_candidates = ("NUTS_ID","nuts_id","NUTS3","nuts3"),
                          node_lon_col_candidates = ("lon","longitude","x"),
                          node_lat_col_candidates = ("lat","latitude","y"),
                          chunk_blockwise: bool = True) -> pd.DataFrame:

    nodes = pd.read_parquet(nodes_parquet)
    if "node_id" not in nodes.columns and "id" in nodes.columns:
        nodes = nodes.rename(columns={"id": "node_id"})

    def _build_geodataframe(nodes_df: pd.DataFrame) -> gpd.GeoDataFrame:
        if "geometry" not in nodes_df.columns:
            raise ValueError("Missing geometry column.")
        series = nodes_df["geometry"]
        first = series.iloc[0]
        if isinstance(first, (bytes, bytearray, memoryview)):
            decoded = [wkb.loads(bytes(g)) for g in series]
            return gpd.GeoDataFrame(nodes_df.drop(columns=["geometry"]), geometry=decoded)
        if isinstance(first, str) and first.strip().upper().startswith(("POINT","LINESTRING","MULTI","POLYGON")):
            decoded = [wkt.loads(g) for g in series]
            return gpd.GeoDataFrame(nodes_df.drop(columns=["geometry"]), geometry=decoded)
        return gpd.GeoDataFrame(nodes_df, geometry="geometry")

    gnodes = _build_geodataframe(nodes)

    if nodes_crs:
        gnodes = gnodes.set_crs(nodes_crs)
    elif gnodes.crs is None:
        x_min, x_max = gnodes.geometry.x.min(), gnodes.geometry.x.max()
        y_min, y_max = gnodes.geometry.y.min(), gnodes.geometry.y.max()
        if (abs(x_min) <= 180 and abs(x_max) <= 180 and abs(y_min) <= 90 and abs(y_max) <= 90):
            gnodes = gnodes.set_crs(4326)
        else:
            if max(abs(x_min), abs(x_max)) < 4000000 and max(abs(y_min), abs(y_max)) < 6000000:
                gnodes = gnodes.set_crs(3035)
            else:
                if max(abs(x_min), abs(x_max)) < 20040000:
                    gnodes = gnodes.set_crs(3857)
                else:
                    raise ValueError("Cannot infer CRS; supply nodes_crs parameter.")

    if gnodes.crs.to_epsg() != 4326:
        gnodes = gnodes.to_crs(4326)

    nuts3 = gpd.read_parquet(nuts3_parquet)
    if nuts3.crs is None:
        nuts3 = nuts3.set_crs(4326)
    elif PCRS.from_user_input(nuts3.crs).to_epsg() != 4326:
        nuts3 = nuts3.to_crs(4326)

    nuts_code_col = next((c for c in nuts3_code_col_candidates if c in nuts3.columns), None)
    if not nuts_code_col:
        raise ValueError("Cannot find NUTS3 code column in NUTS3 parquet.")
    nuts3 = nuts3[[nuts_code_col, "geometry"]].rename(columns={nuts_code_col: "NUTS3"})

    node_lon = gnodes.geometry.x.values
    node_lat = gnodes.geometry.y.values
    pop_accum = np.zeros(len(gnodes), dtype="float64")

    if CROPPED_RASTER_PATH.exists():
        print(f"[Raster] Using existing cropped raster: {CROPPED_RASTER_PATH}", flush=True)
        cropped_raster = CROPPED_RASTER_PATH
    else:
        print(f"[Raster] Cropped raster not found. Creating: {CROPPED_RASTER_PATH}", flush=True)
        cropped_raster = _ensure_cropped_raster(POP_RASTER, CROPPED_RASTER_PATH)

    with rasterio.open(cropped_raster) as src:
        r_crs = src.crs
        if r_crs is None:
            raise ValueError("Population raster has no CRS.")

        try:
            is_wgs84 = RCRS.from_epsg(4326).equals(r_crs)
        except Exception:
            s = r_crs.to_string()
            is_wgs84 = ("WGS 84" in s) and getattr(r_crs, "is_geographic", True)

        if is_wgs84:
            xmin, xmax = EUROPE_BOUNDS["xmin"], EUROPE_BOUNDS["xmax"]
            ymin, ymax = EUROPE_BOUNDS["ymin"], EUROPE_BOUNDS["ymax"]
            window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
            kdtree = cKDTree(np.vstack([node_lon, node_lat]).T)
            bounds_filter = lambda xs, ys: (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
        else:
            transformer = Transformer.from_crs("EPSG:4326", r_crs, always_xy=True)
            xmin_r, ymin_r = transformer.transform(EUROPE_BOUNDS["xmin"], EUROPE_BOUNDS["ymin"])
            xmax_r, ymax_r = transformer.transform(EUROPE_BOUNDS["xmax"], EUROPE_BOUNDS["ymax"])
            window = from_bounds(xmin_r, ymin_r, xmax_r, ymax_r, transform=src.transform)
            node_xr, node_yr = transformer.transform(node_lon, node_lat)
            kdtree = cKDTree(np.vstack([node_xr, node_yr]).T)
            bounds_filter = lambda xs, ys: (xs >= xmin_r) & (xs <= xmax_r) & (ys >= ymin_r) & (ys <= ymax_r)

        if not chunk_blockwise:
            data = src.read(1, window=window, masked=True)
            transform_win = src.window_transform(window)
            valid = (~data.mask) & (data.data > 0)
            rows, cols = np.where(valid)
            if rows.size:
                xs, ys = xy(transform_win, rows, cols, offset="center")
                xs = np.asarray(xs); ys = np.asarray(ys)
                m = bounds_filter(xs, ys)
                if m.any():
                    xs_f = xs[m]; ys_f = ys[m]
                    vals = data.data[rows[m], cols[m]]
                    _, idxs = kdtree.query(np.vstack([xs_f, ys_f]).T, k=1)
                    np.add.at(pop_accum, idxs, vals)
        else:
            win_bounds = rasterio.windows.bounds(window, src.transform)
            for _, bw in src.block_windows(1):
                b = rasterio.windows.bounds(bw, src.transform)
                if b[0] > win_bounds[2] or b[2] < win_bounds[0] or b[1] > win_bounds[3] or b[3] < win_bounds[1]:
                    continue
                block = src.read(1, window=bw, masked=True)
                if block.mask.all():
                    continue
                valid = (~block.mask) & (block.data > 0)
                rows, cols = np.where(valid)
                if not rows.size:
                    continue
                t_block = src.window_transform(bw)
                xs, ys = xy(t_block, rows, cols, offset="center")
                xs = np.asarray(xs); ys = np.asarray(ys)
                m = bounds_filter(xs, ys)
                if not m.any():
                    continue
                xs_f = xs[m]; ys_f = ys[m]
                vals = block.data[rows[m], cols[m]]
                _, idxs = kdtree.query(np.vstack([xs_f, ys_f]).T, k=1)
                np.add.at(pop_accum, idxs, vals)

        gnodes["population_node"] = pop_accum

    print(f"[Nodes] Total population assigned to nodes: {gnodes['population_node'].sum()}", flush=True)

    gnodes["population_node"] = pd.to_numeric(gnodes["population_node"], errors="coerce").fillna(0.0)

    # Spatial join NUTS3 polygons
    gnodes = gpd.sjoin(gnodes, nuts3, predicate="within", how="left")

    # Compute NUTS3 totals BEFORE filtering
    nuts_totals = gnodes.groupby("NUTS3", dropna=False)["population_node"].sum().rename("population_NUTS3")
    gnodes = gnodes.merge(nuts_totals, on="NUTS3", how="left")

    # Filter nonzero node + NUTS total
    gnodes = gnodes[(gnodes["population_node"] > 0) & (gnodes["population_NUTS3"] > 0)].copy()
    print(f"Nodes with nonzero population : {len(gnodes)}", flush=True)

    # Keep geometry
    gnodes = gnodes[["node_id", "NUTS3", "population_node", "population_NUTS3", "geometry"]]
    if not isinstance(gnodes, gpd.GeoDataFrame):
        gnodes = gpd.GeoDataFrame(gnodes, geometry="geometry", crs=4326)
    return gnodes


def merge_low_pop(gdf: gpd.GeoDataFrame, min_pop: float = 1000.0, max_dist_m: float = 1000.0) -> gpd.GeoDataFrame:
    if "population_node" not in gdf.columns:
        raise KeyError("population_node column is required.")
    g = gdf.copy()
    g["population_node"] = pd.to_numeric(g["population_node"], errors="coerce").fillna(0.0)

    low_mask = g["population_node"] < min_pop
    high_mask = g["population_node"] >= min_pop
    if not low_mask.any() or not high_mask.any():
        print("[merge<min] no candidates.")
        return g

    gproj = g.to_crs(3035)
    low_idx = gproj.index[low_mask]
    high_idx = gproj.index[high_mask]
    lx = gproj.loc[low_idx, "geometry"].x.to_numpy()
    ly = gproj.loc[low_idx, "geometry"].y.to_numpy()
    hx = gproj.loc[high_idx, "geometry"].x.to_numpy()
    hy = gproj.loc[high_idx, "geometry"].y.to_numpy()

    low_valid = np.isfinite(lx) & np.isfinite(ly)
    high_valid = np.isfinite(hx) & np.isfinite(hy)
    if not low_valid.any() or not high_valid.any():
        print("[merge<min] invalid coordinates.")
        return g

    lx, ly = lx[low_valid], ly[low_valid]
    hx, hy = hx[high_valid], hy[high_valid]
    low_idx_arr = low_idx.to_numpy()[low_valid]
    high_idx_arr = high_idx.to_numpy()[high_valid]

    tree = cKDTree(np.column_stack([hx, hy]))
    dist, nn_pos = tree.query(np.column_stack([lx, ly]), k=1)
    within = dist <= max_dist_m
    if not np.any(within):
        print(f"[merge<min] none within {int(max_dist_m)} m.")
        return g

    low_sel = low_idx_arr[within]
    high_sel = high_idx_arr[np.asarray(nn_pos)[within]]

    transfers = (
        pd.DataFrame({"low": low_sel, "high": high_sel})
        .assign(pop=g.loc[low_sel, "population_node"].to_numpy())
        .groupby("high", as_index=False)["pop"].sum()
    )
    g.loc[transfers["high"].to_numpy(), "population_node"] += transfers["pop"].to_numpy()
    g = g.drop(index=low_sel)
    print(f"[merge<min] moved {int(len(low_sel))} nodes into nearest >={int(min_pop)} within {int(max_dist_m)} m")
    return g

def consolidate_close_nodes(gdf: gpd.GeoDataFrame, radius_m: float = 1000.0) -> gpd.GeoDataFrame:
    if "population_node" not in gdf.columns:
        raise KeyError("population_node column is required.")
    g = gdf.copy()
    g["population_node"] = pd.to_numeric(g["population_node"], errors="coerce").fillna(0.0)

    if g.empty:
        return g

    gproj = g.to_crs(3035)
    px = gproj.geometry.x.to_numpy()
    py = gproj.geometry.y.to_numpy()
    valid = np.isfinite(px) & np.isfinite(py)
    if not valid.any():
        print("[consolidate<1km] invalid coordinates.")
        return g

    coords = np.column_stack([px[valid], py[valid]])
    idx_map = gproj.index.to_numpy()[valid]

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=radius_m)
    if not pairs:
        print(f"[consolidate<1km] no pairs within {int(radius_m)} m.")
        return g

    parent = {i: i for i in range(coords.shape[0])}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for a, b in pairs:
        union(a, b)

    clusters = {}
    for i in range(coords.shape[0]):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    drops = []
    transfers = []
    pop_series = pd.to_numeric(g["population_node"], errors="coerce").fillna(0.0)

    for members in clusters.values():
        if len(members) < 2:
            continue
        member_idx = [idx_map[m] for m in members]
        pops = pop_series.loc[member_idx].to_numpy()
        keep_pos = int(np.argmax(pops))
        keep_idx = member_idx[keep_pos]
        other_idxs = [i for j, i in enumerate(member_idx) if j != keep_pos]
        sum_others = float(np.sum(pop_series.loc[other_idxs].to_numpy())) if other_idxs else 0.0
        if sum_others > 0:
            transfers.append((keep_idx, sum_others))
        drops.extend(other_idxs)

    if transfers:
        tdf = pd.DataFrame(transfers, columns=["keep", "pop_add"]).groupby("keep", as_index=False)["pop_add"].sum()
        g.loc[tdf["keep"].to_numpy(), "population_node"] += tdf["pop_add"].to_numpy()
    if drops:
        g = g.drop(index=pd.Index(drops).intersection(g.index))
        print(f"[consolidate<1km] collapsed {len(drops)} nodes into keepers")
    return g

def main():
    # Load with geometry
    gdf = load_nodes_population(NODES_PARQUET)
    min_pop = 200.0
    max_dist_m = 6000.0  
    radius_m = 750.0
 
    # Merge low-pop nodes into nearest high-pop within threshold
    gdf = merge_low_pop(gdf, min_pop, max_dist_m)
    # Consolidate clusters closer than radius_m
    gdf = consolidate_close_nodes(gdf, radius_m)
 
    print(f"[save] nodes with population after merging/consolidation: {len(gdf)}", flush=True)

    # Save keeping geometry if desired, or select minimal columns
    gdf.to_parquet(OUT_PARQUET, index=False)
    print(f"[save] {OUT_PARQUET} | nodes: {len(gdf)}")

#try:
#    job_num = int(sys.argv[1])
#    total_num = int(sys.argv[2])
#except IndexError:
#    print(f"Usage: python {__file__} <start_num> <end_num>")

#vector_indices = np.arange(job_num,25,total_num)

#def main():
#    # Load once (expensive)
#    base_gdf = load_nodes_population(NODES_PARQUET)
#    print(f"[load] nodes: {len(base_gdf)}", flush=True)
#
#    # Grids
#    min_pops = list(range(100, 1001, 200))
#    radii_m  = 100+(job_num-5*np.floor((job_num)/5))*200
#    radij_m  = 1000+(np.floor(job_num/5))*2000
#    radij_m  = list(range(100, 1001, 250))      # 100..1000
#    radii_m  = list(range(1000, 10001, 2500))     # 1000..10000
#
#    out_dir = Path(OUT_PARQUET).parent
#    out_dir.mkdir(parents=True, exist_ok=True)
#
#    total_runs = len(min_pops)
#    run = 0
#    radius_m = radii_m
#    dist_m   = radij_m
#    for min_pop in min_pops:
#         for radius_m in radii_m:
#    #         for dist_m in radij_m:
#        run += 1
#    print(f"[{run}/{total_runs}] min_pop={min_pop}, radius_m={radius_m}, dist_m={dist_m}", flush=True)
#
#    # Work on a copy to keep base_gdf intact
#    gdf = base_gdf.copy()
#
#    # Merge low-pop nodes into nearest high-pop within threshold (use radius_m as max_dist)
#    gdf = merge_low_pop(gdf, min_pop=min_pop, max_dist_m=dist_m)
#    # Consolidate clusters closer than radius_m
#    gdf = consolidate_close_nodes(gdf, radius_m=radius_m)

#    # Build variant filename
#    out_file = Path(OUT_PARQUET).with_name(
#        f"{Path(OUT_PARQUET).stem}_min{min_pop}_rad{radius_m}_dist{dist_m}{Path(OUT_PARQUET).suffix}"
#    )
#    gdf.to_parquet(out_file, index=False)
#    print(f"[save] {out_file} | nodes: {len(gdf)}", flush=True)
#
#    print("[done] all combinations saved.", flush=True)


if __name__ == "__main__":
    main()
    print("Nodes population data loaded and saved.", flush=True)
    
