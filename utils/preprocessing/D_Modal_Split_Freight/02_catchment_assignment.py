#Catchment assignment of industries to nearest large facilities (ports, IWW ports, airports).

#What this script does:
#- Load updated Ports, IWW Nodes, and Airports GeoParquets (WKB geometry) and Industries with flows.
#- Normalize CRS, build centroids and coordinates; compute projected x/y in meters (EPSG:3035).
#- Load TENT road network (nodes/edges), compute edge travel time, and build an undirected igraph graph.
#- Build a KDTree on road nodes (for nearest-node lookup) and assign each facility/industry to a closest road node.
#- Aggregate facilities by code (port_code/icao): representative location per code and total “size” per code.
#- For each industry, select the best facility by a size-over-distance score among k nearest facilities.
#- Save the industry table with closest facilities (port, IWW port, airport) and their closest road nodes.

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkb import loads
import igraph as ig
from scipy.spatial import cKDTree
import sys
from pathlib import Path

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
ports_df = pd.read_parquet('/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_ports_TENT.parquet')
airports_df = pd.read_parquet('/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_airports_TENT.parquet')
iww_nodes_df = pd.read_parquet('/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_iww_nodes_TENT.parquet')
iww_ports_df = iww_nodes_df[iww_nodes_df['feature'] == 'port'].copy()

industries_df = pd.read_parquet('/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industries_w_outputs/industries_with_flows_threshold_0.25.parquet')

# Filter invalid industries
industries_df = industries_df[industries_df['area'] > 0].copy()

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Reproject / geometry setup
# -------------------------------------------------------------------

def prep_from_crs(df: pd.DataFrame, source_epsg: str) -> gpd.GeoDataFrame:
    """Decode WKB, set source CRS, project to EPSG:3035 for distance calculations."""
    df = df.copy()
    # Check if geometry is already Shapely objects or WKB bytes
    if len(df) > 0 and pd.notna(df['geometry'].iloc[0]):
        first_geom = df['geometry'].iloc[0]
        if isinstance(first_geom, bytes):
            # Decode WKB
            df['geometry'] = df['geometry'].apply(lambda b: loads(b) if pd.notna(b) else None)
    # If already a GeoDataFrame with CRS, preserve it
    if isinstance(df, gpd.GeoDataFrame) and df.crs is not None:
        gdf = df
    else:
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=source_epsg)
    # Force projection to EPSG:3035 for distance calculations
    if gdf.crs.to_string() != "EPSG:3035":
        gdf = gdf.to_crs("EPSG:3035", inplace=False)
    gdf['centroid'] = gdf.geometry.centroid
    gdf['x_m'] = gdf['centroid'].x
    gdf['y_m'] = gdf['centroid'].y
    # Also compute lon/lat for reference (reproject centroid to 4326)
    gdf_4326 = gdf.set_geometry(gdf['centroid']).to_crs("EPSG:4326")
    gdf['longitude'] = gdf_4326.geometry.x
    gdf['latitude'] = gdf_4326.geometry.y
    print(f"prep_from_crs: {source_epsg} -> EPSG:3035, CRS now: {gdf.crs.to_string()} (bounds={gdf.total_bounds})", flush=True)
    return gdf

# Use explicit source CRS for these layers (all now in EPSG:3035 from 01_flow_check)
ports_df     = prep_from_crs(ports_df,     "EPSG:3035")
iww_ports_df = prep_from_crs(iww_ports_df, "EPSG:3035")
airports_df  = prep_from_crs(airports_df,  "EPSG:3035")
industries_df = prep_from_crs(industries_df, "EPSG:3035")

# -------------------------------------------------------------------
# Size definitions
# -------------------------------------------------------------------
EPS = 1e-9
INDUSTRY_SIZE_MODE = 'sum'   # options: 'sum','out','in'
FACILITY_SIZE_MODE = 'sum'   # options: 'sum','out','in'

out_cols = [c for c in industries_df.columns if c.startswith('outgoing_')]
in_cols = [c for c in industries_df.columns if c.startswith('incoming_')]
industries_df['total_out'] = industries_df[out_cols].sum(axis=1) if out_cols else 0.0
industries_df['total_in'] = industries_df[in_cols].sum(axis=1) if in_cols else 0.0

if INDUSTRY_SIZE_MODE == 'out':
    industries_df['industry_size'] = industries_df['total_out']
elif INDUSTRY_SIZE_MODE == 'in':
    industries_df['industry_size'] = industries_df['total_in']
else:
    industries_df['industry_size'] = industries_df['total_out'] + industries_df['total_in']
industries_df['industry_size'] = industries_df['industry_size'].clip(lower=0) + EPS

# -------------------------------------------------------------------
# Build facility dataframes with size aggregation
# -------------------------------------------------------------------

def build_fac_df(df: gpd.GeoDataFrame, code_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['agg_size', 'code', 'latitude', 'longitude', 'x_m', 'y_m'])

    for col in ('latitude', 'longitude', 'x_m', 'y_m', code_col):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in facility dataframe.")

    df = df.copy()
    in_col, out_col = 'OBS_VALUE_IN', 'OBS_VALUE_OUT'
    in_vals = pd.to_numeric(df.get(in_col, 0), errors='coerce').fillna(0.0)
    out_vals = pd.to_numeric(df.get(out_col, 0), errors='coerce').fillna(0.0)
    if FACILITY_SIZE_MODE == 'out':
        size_vals = out_vals
    elif FACILITY_SIZE_MODE == 'in':
        size_vals = in_vals
    else:
        size_vals = in_vals + out_vals
    df['facility_size'] = size_vals.clip(lower=0)  # Don't add EPS yet - need to filter zeros first

    # Representative record per code (largest facility_size)
    rep_idx = df.groupby(code_col)['facility_size'].idxmax()
    reps = df.loc[rep_idx, ['facility_size', code_col, 'latitude', 'longitude', 'x_m', 'y_m']].copy()
    reps.rename(columns={'facility_size': 'rep_size'}, inplace=True)
    reps.set_index(code_col, inplace=True)
    reps.index.name = 'code'

    sums = df.groupby(code_col)['facility_size'].sum().rename('agg_size')
    sums.index.name = 'code'

    fac = reps.join(sums).reset_index()
    # Add EPS only to non-zero values for distance calculations
    fac['agg_size'] = np.where(fac['agg_size'] > 0, fac['agg_size'] + EPS, 0.0)
    return fac[['agg_size', 'code', 'latitude', 'longitude', 'x_m', 'y_m']]

# Build per-mode facility frames
airports_fac_df = build_fac_df(airports_df, 'icao')
ports_fac_df = build_fac_df(ports_df, 'port_code')
iww_fac_df = build_fac_df(iww_ports_df, 'port_code')

# Filter out facilities with zero OBS_VALUE (no Eurostat data)
# Only assign industries to ports/airports with actual traffic data
print(f"Before filtering: Airports={len(airports_fac_df)}, Ports={len(ports_fac_df)}, IWW={len(iww_fac_df)}", flush=True)

airports_fac_df = airports_fac_df[airports_fac_df['agg_size'] > 0].copy()
ports_fac_df = ports_fac_df[ports_fac_df['agg_size'] > 0].copy()
iww_fac_df = iww_fac_df[iww_fac_df['agg_size'] > 0].copy()

print(f"After filtering (OBS_VALUE > 0): Airports={len(airports_fac_df)}, Ports={len(ports_fac_df)}, IWW={len(iww_fac_df)}", flush=True)

def prep_fac_for_distance(fac_df: pd.DataFrame) -> pd.DataFrame:
    if fac_df is None or fac_df.empty:
        return pd.DataFrame(columns=['code', 'size', 'x_m', 'y_m'])
    df = fac_df[['code', 'agg_size', 'x_m', 'y_m']].copy()
    df.rename(columns={'agg_size': 'size'}, inplace=True)
    return df.dropna(subset=['x_m', 'y_m'])

airports_rad = prep_fac_for_distance(airports_fac_df)
ports_rad = prep_fac_for_distance(ports_fac_df)
iww_rad = prep_fac_for_distance(iww_fac_df)

# Optional: warn if only a single candidate remains (explains RUMLK)
for name, fac in [('ports', ports_rad), ('airports', airports_rad), ('iww', iww_rad)]:
    if len(fac) <= 1:
        print(f"Warning: {name} candidate list has {len(fac)} row(s). All industries will be assigned to the only code available.", flush=True)

def select_by_size_over_distance(ix: float, iy: float, fac: pd.DataFrame, k_members: int, size_factor: float):
    """
    Return best_code maximizing size / distance^size_factor among k nearest by distance.
    ix, iy in meters (EPSG:3035), fac has columns ['code','size','x_m','y_m'] in meters.
    """
    if fac is None or fac.empty or ix is None or iy is None:
        return None

    dx = fac['x_m'].to_numpy(dtype=np.float64) - ix
    dy = fac['y_m'].to_numpy(dtype=np.float64) - iy
    dist = np.sqrt(dx*dx + dy*dy) / 1000.0  # km

    n = dist.size
    if n == 0:
        return None

    k = min(k_members, n)
    idx_k = np.argpartition(dist, k - 1)[:k]
    idx_k = idx_k[np.argsort(dist[idx_k])]

    dist_k = dist[idx_k]
    dist_k = np.where(dist_k <= 0, EPS, dist_k)
    size_k = fac['size'].to_numpy(dtype=np.float64)[idx_k]
    scores = (size_k ** size_factor / dist_k) 

    if not np.isfinite(scores).any():
        return None

    gidx = idx_k[int(np.nanargmax(scores))]
    return fac['code'].iloc[gidx]

# -------------------------------------------------------------------
# Assignment loop
# -------------------------------------------------------------------
k_members = 5
size_factors = [0.5]
output_base = Path('/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industries_w_outputs/industries_with_closest_ports_airports_euclidean.parquet')

n_ind = len(industries_df)
sel_idx = industries_df.index

for size_factor in size_factors:
    res_air_code = np.empty(n_ind, dtype=object)
    res_port_code = np.empty(n_ind, dtype=object)
    res_iww_code = np.empty(n_ind, dtype=object)

    for j, (idx, row) in enumerate(industries_df.iterrows()):
        ix = row.x_m; iy = row.y_m
        if pd.isna(ix) or pd.isna(iy):
            res_air_code[j] = None
            res_port_code[j] = None
            res_iww_code[j] = None
            continue
        
        res_air_code[j] = select_by_size_over_distance(ix, iy, airports_rad, k_members, size_factor)
        res_port_code[j] = select_by_size_over_distance(ix, iy, ports_rad, k_members, size_factor)
        res_iww_code[j] = select_by_size_over_distance(ix, iy, iww_rad, k_members, size_factor)

    print(f"Completed assignment (size_factor={size_factor})", flush=True)

    out_df = industries_df.copy()
    out_df.loc[sel_idx, 'closest_airport'] = res_air_code
    out_df.loc[sel_idx, 'closest_port'] = res_port_code
    out_df.loc[sel_idx, 'closest_iww_port'] = res_iww_code

    out_df = out_df.filter(items=[
        'geometry','id','nuts2','eprtr_sectors','eea_activities','area','area_nuts2',
        'outgoing_CH','outgoing_CO','outgoing_FB','outgoing_IL','outgoing_MI','outgoing_MT','outgoing_PW',
        'incoming_CH','incoming_CO','incoming_FB','incoming_IL','incoming_MI','incoming_MT','incoming_PW',
        'longitude','latitude','closest_port','closest_airport','closest_iww_port'
    ])

    sf_tag = str(size_factor).replace('.', '_')
    out_path = output_base.with_name(f"{output_base.stem}_sf{sf_tag}{output_base.suffix}")
    out_df.to_parquet(out_path, index=False)
    print(f"Saved results (size_factor={size_factor}) to {out_path}", flush=True)
