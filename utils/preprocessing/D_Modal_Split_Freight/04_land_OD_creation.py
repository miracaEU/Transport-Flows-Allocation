import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import shapely.wkb as wkb
import shapely.wkt as wkt
import geopandas as gpd
import sys

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
road_nodes_file = '/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_road_nodes_TENT.parquet'
rail_nodes_file = '/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_railways_stations_TENT.parquet'
iww_nodes_file = '/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_iww_nodes_TENT.parquet'
industries_file = '/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industries_w_outputs/industries_with_closest_ports_airports_euclidean.parquet'
intermodal_file = '/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/EUROSTAT/intermodal/Modal split of inland freight transport.csv'
flows_file = "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industries_w_outputs/industrial_OD_threshold_0.25_airmar_assigned.parquet"
maritime_air_land_file = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/intermediate_ODs/OD_ports_airports_industries.parquet'

# Output files
rail_od_file = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/intermediate_ODs/rail_OD.parquet'
road_od_file = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/intermediate_ODs/road_OD.parquet'
iww_od_file = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/intermediate_ODs/iww_OD.parquet'
station_industry_od_file = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/intermediate_ODs/rail_station_industryy_road_OD.parquet'
iwwport_industry_od_file = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/intermediate_ODs/iwwport_industryy_road_OD.parquet'

mar_road_od_file = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/intermediate_ODs/OD_ports_airports_industries_road.parquet'
mar_rail_od_file = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/intermediate_ODs/OD_ports_airports_industries_rail.parquet'
mar_iww_od_file = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/intermediate_ODs/OD_ports_airports_industries_iww.parquet'

rail_od_file_final = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/rail_OD_final.parquet'
road_od_file_final = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/road_OD_final.parquet'
iww_od_file_final = '/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/iww_OD_final.parquet'

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_nodes(path: str, name: str) -> gpd.GeoDataFrame:
    """Load node parquet (try GeoPandas), decode WKB/WKT if needed, set CRS."""
    try:
        gdf = gpd.read_parquet(path)
    except Exception:
        df = pd.read_parquet(path)
        if 'geometry' not in df.columns:
            raise ValueError(f"{name}: geometry column missing.")
        def to_geom(val):
            if val is None or (isinstance(val, float) and pd.isna(val)): return None
            if isinstance(val, (bytes, bytearray)):
                try: return wkb.loads(val)
                except: return None
            if isinstance(val, str):
                try: return wkt.loads(val)
                except: return None
            return val
        df['geometry'] = df['geometry'].apply(to_geom)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    # Infer CRS if missing
    if gdf.crs is None:
        max_abs = max(gdf.geometry.x.abs().max(), gdf.geometry.y.abs().max())
        if max_abs > 400:  # looks projected
            gdf.set_crs("EPSG:3035", inplace=True)
        else:
            gdf.set_crs("EPSG:4326", inplace=True)
    # Convert to WGS84 for nearest queries (simple Euclidean on lat/lon)
    if gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(4326)
    # Use centroid if not point
    geom_pts = gdf.geometry.apply(lambda g: g if g is None or g.geom_type == "Point" else g.centroid)
    gdf['longitude'] = geom_pts.apply(lambda g: g.x if g else np.nan)
    gdf['latitude'] = geom_pts.apply(lambda g: g.y if g else np.nan)
    return gdf

def ensure_industry_coords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {'x_m','y_m'}.issubset(df.columns):
        df['coord_x'] = df['x_m']
        df['coord_y'] = df['y_m']
    elif {'longitude','latitude'}.issubset(df.columns):
        df['coord_x'] = df['longitude']
        df['coord_y'] = df['latitude']
    else:
        if 'geometry' not in df.columns:
            raise ValueError("Industries lack geometry/longitude/latitude/x_m/y_m.")
        def to_geom(val):
            if isinstance(val, (bytes, bytearray)):
                try: return wkb.loads(val)
                except: return None
            if isinstance(val, str):
                try: return wkt.loads(val)
                except: return None
            return val
        df['geometry'] = df['geometry'].apply(to_geom)
        df['coord_x'] = df['geometry'].apply(lambda g: g.x if g else np.nan)
        df['coord_y'] = df['geometry'].apply(lambda g: g.y if g else np.nan)
    if df['coord_x'].isna().any() or df['coord_y'].isna().any():
        print("Warning: Some industry coordinates NaN.", flush=True)
    return df

def normalize_id_series(s: pd.Series) -> pd.Series:
    """Return a string ID series with no trailing '.0' for integer-like floats."""
    s_num = pd.to_numeric(s, errors='coerce')
    s_str = s.astype(str).str.strip()
    mask_num = s_num.notna()
    mask_int = mask_num & np.isclose(s_num % 1, 0)
    # Integers: cast to int then str -> '5182094'
    s_str.loc[mask_int] = s_num.loc[mask_int].astype(np.int64).astype(str)
    s_str.loc[mask_num] = s_num.loc[mask_num].astype(np.int64).astype(str)
    return s_str.fillna('')

def build_od(df: pd.DataFrame, o_col: str, d_col: str, v_col: str,
             extra_group_cols: list[str] | None = None) -> pd.DataFrame:
    groups = [o_col, d_col]
    if extra_group_cols:
        groups.extend(extra_group_cols)
    od = (df.groupby(groups, dropna=False)[v_col]
            .sum()
            .reset_index()
            .rename(columns={o_col: 'from_id', d_col: 'to_id', v_col: 'value'}))
    od = od[od['value'] > 0]
    return od

def _haversine_km(lat1, lon1, lat2, lon2):
    # Vectorized haversine (inputs are numpy arrays)
    R = 6371.0088
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))
# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
industries_df = pd.read_parquet(industries_file)
industries_df = ensure_industry_coords(industries_df)

industries_unique = industries_df.drop_duplicates('id').copy()
# Create normalized ID
industries_unique['id'] = normalize_id_series(industries_unique['id'])

# Dict lookups (avoid index duplication issues)
ind_x_map = dict(zip(industries_unique['id'], industries_unique['coord_x']))
ind_y_map = dict(zip(industries_unique['id'], industries_unique['coord_y']))

road_nodes = load_nodes(road_nodes_file, "road_nodes")
rail_nodes = load_nodes(rail_nodes_file, "rail_nodes")
iww_nodes = load_nodes(iww_nodes_file, "iww_nodes")
iww_ports = iww_nodes[iww_nodes['feature'] == 'port'].copy()

# Set unified coord_x / coord_y
for nd in (road_nodes, rail_nodes, iww_ports):
    nd['coord_x'] = nd['longitude']
    nd['coord_y'] = nd['latitude']

flows = pd.read_parquet(flows_file)

# Ensure id dtype alignment
flows['origin_id'] = flows['origin_id'].astype(str)
flows['destination_id'] = flows['destination_id'].astype(str)

# Normalize IDs to remove trailing '.0'
flows['origin_id'] = normalize_id_series(flows['origin_id'])
flows['destination_id'] = normalize_id_series(flows['destination_id'])
industries_unique['id'] = normalize_id_series(industries_unique['id'])

# Merge industry lon/lat onto flows to define o_/d_ coordinates
flows = flows.merge(
    industries_unique[['id','longitude','latitude']].rename(
        columns={'id':'origin_id','longitude':'o_lon','latitude':'o_lat'}
    ),
    on='origin_id', how='left'
)
flows = flows.merge(
    industries_unique[['id','longitude','latitude']].rename(
        columns={'id':'destination_id','longitude':'d_lon','latitude':'d_lat'}
    ),
    on='destination_id', how='left'
)

# Drop rows with missing coords (avoid (0,0) fallback)
miss = flows[['o_lon','o_lat','d_lon','d_lat']].isna().any(axis=1)
if int(miss.sum()) > 0:
    print(f"Dropping {int(miss.sum())} ODs with missing industry lon/lat.", flush=True)
flows = flows.loc[~miss].copy()

# Fix land calculation (use column names, not Series)
def s(df, col):  # safe series by column name
    return pd.to_numeric(df[col], errors='coerce').fillna(0.0) if isinstance(col, str) and col in df.columns else 0.0

total_col = 'value'
air_col = 'air_flow'
mar_col = 'maritime_flow'
flows['land'] = s(flows, total_col) - s(flows, air_col) - s(flows, mar_col)
flows['land'] = flows['land'].clip(lower=0)


# ------------------------------------------------------------------
# Intermodal shares (vectorized)
# ------------------------------------------------------------------

ISO3_TO_ISO2 = {
    'AUT':'AT','BEL':'BE','BGR':'BG','HRV':'HR','CYP':'CY','CZE':'CZ','DNK':'DK','EST':'EE','FIN':'FI',
    'FRA':'FR','DEU':'DE','GRC':'GR','HUN':'HU','IRL':'IE','ITA':'IT','LVA':'LV','LTU':'LT','LUX':'LU',
    'MLT':'MT','NLD':'NL','POL':'PL','PRT':'PT','ROU':'RO','SVK':'SK','SVN':'SI','ESP':'ES','SWE':'SE',
    'NOR':'NO','CHE':'CH','ISL':'IS','GBR':'GB','ALB':'AL','SRB':'RS','MKD':'MK','MNE':'ME','BIH':'BA',
    'MDA':'MD','UKR':'UA','BLR':'BY','RUS':'RU','TUR':'TR'
}

def normalize_intermodal_df(ic: pd.DataFrame) -> pd.DataFrame:
    # Detect headers flexibly
    geo_col = next((c for c in ['go\\TI_PRIOD','geo\\TIME_PERIOD','geo','GEO'] if c in ic.columns), None)
    mode_col = next((c for c in ['tra_od','mode','MODE'] if c in ic.columns), None)
    val_col = next((c for c in ['2021','VALUE','value'] if c in ic.columns), None)
    if not (geo_col and mode_col and val_col):
        return pd.DataFrame(columns=['country_code','mode','share'])

    ic = ic[[geo_col, mode_col, val_col]].copy()
    ic.rename(columns={geo_col:'country_code', mode_col:'mode', val_col:'share'}, inplace=True)

    # Normalize country_code to ISO2
    cc = ic['country_code'].astype(str).str.strip().str.upper().str.split('_').str[0]
    cc = cc.replace({'UK':'GB','EL':'GR'})
    cc = cc.map(lambda s: ISO3_TO_ISO2.get(s, s))
    name_map = {
        'GERMANY':'DE','FRANCE':'FR','ITALY':'IT','SPAIN':'ES','PORTUGAL':'PT','BELGIUM':'BE','NETHERLANDS':'NL',
        'POLAND':'PL','AUSTRIA':'AT','CZECHIA':'CZ','CZECH REPUBLIC':'CZ','SLOVAKIA':'SK','SLOVENIA':'SI',
        'HUNGARY':'HU','ROMANIA':'RO','BULGARIA':'BG','CROATIA':'HR','GREECE':'GR','IRELAND':'IE','DENMARK':'DK',
        'SWEDEN':'SE','FINLAND':'FI','LUXEMBOURG':'LU','LITHUANIA':'LT','LATVIA':'LV','ESTONIA':'EE','NORWAY':'NO',
        'SWITZERLAND':'CH','ICELAND':'IS','UNITED KINGDOM':'GB','MOLDOVA':'MD','SERBIA':'RS','ALBANIA':'AL',
        'BOSNIA AND HERZEGOVINA':'BA','MONTENEGRO':'ME','NORTH MACEDONIA':'MK','BELARUS':'BY','UKRAINE':'UA',
        'RUSSIA':'RU','TURKEY':'TR','TURKIYE':'TR'
    }
    cc = cc.map(lambda s: name_map.get(s, s))
    ic['country_code'] = cc

    ic['mode'] = ic['mode'].astype(str).str.upper().str.strip()
    ic['share'] = pd.to_numeric(ic['share'], errors='coerce')
    ic = ic.dropna(subset=['country_code','share'])
    ic = ic[ic['mode'].isin(['ROAD','RAIL','IWW'])]
    return ic

intermodal_csv_raw = pd.read_csv(intermodal_file)
intermodal_csv = normalize_intermodal_df(intermodal_csv_raw)

if intermodal_csv.empty:
    print("Warning: Intermodal share columns missing/unsupported; defaulting 100% ROAD.", flush=True)

# Ensure flows have ISO2 countries for compute_mode_share
if 'origin_country' not in flows.columns and 'origin_nuts2' in flows.columns:
    flows['origin_country'] = flows['origin_nuts2'].astype(str).str[:2].str.upper()
if 'destination_country' not in flows.columns and 'destination_nuts2' in flows.columns:
    flows['destination_country'] = flows['destination_nuts2'].astype(str).str[:2].str.upper()

# Build per-country share maps once (avoid per-row filtering)
piv = (intermodal_csv
       .pivot_table(index='country_code', columns='mode', values='share', aggfunc='mean')
       .fillna(0))
ROAD_MAP = piv['ROAD'].to_dict() if 'ROAD' in piv.columns else {}
RAIL_MAP = piv['RAIL'].to_dict() if 'RAIL' in piv.columns else {}
IWW_MAP = piv['IWW'].to_dict() if 'IWW' in piv.columns else {}

def compute_mode_share(row):
    # Base shares from intermodal data
    if not {'origin_country','destination_country'}.issubset(flows.columns) or intermodal_csv.empty:
        return 1.0, 0.0, 0.0
    oc = str(row.get('origin_country', '')).upper().strip()
    dc = str(row.get('destination_country', '')).upper().strip()
    rs = (ROAD_MAP.get(oc, 0.0) + ROAD_MAP.get(dc, 0.0)) / 2.0
    ls = (RAIL_MAP.get(oc, 0.0) + RAIL_MAP.get(dc, 0.0)) / 2.0
    ws = (IWW_MAP.get(oc, 0.0) + IWW_MAP.get(dc, 0.0)) / 2.0
    tot = rs + ls + ws
    if not np.isfinite(tot) or tot <= 0:
        road_share, rail_share, iww_share = 1.0, 0.0, 0.0
    else:
        road_share, rail_share, iww_share = rs/tot, ls/tot, ws/tot

    return road_share, rail_share, iww_share

# ------------------------------------------------------------------
# KD-Trees (simple Euclidean in lat/lon space)
# ------------------------------------------------------------------

rail_tree = cKDTree(rail_nodes[['coord_y','coord_x']].to_numpy())
road_tree = cKDTree(road_nodes[['coord_y','coord_x']].to_numpy())
iww_tree = cKDTree(iww_ports[['coord_y','coord_x']].to_numpy())

# Map each rail station to its nearest road node
station_xy = np.column_stack([rail_nodes['coord_y'].to_numpy(), rail_nodes['coord_x'].to_numpy()])
_, nearest_road_idx = road_tree.query(station_xy)
rail_nodes['nearest_road_id'] = road_nodes.iloc[nearest_road_idx]['id'].values
# Build lookup dicts
station2road = dict(zip(rail_nodes['id'].astype(str), rail_nodes['nearest_road_id'].astype(str)))

# Also map each road node to its nearest rail station (for converting road-node ODs to station ODs)
road_xy = np.column_stack([road_nodes['coord_y'].to_numpy(), road_nodes['coord_x'].to_numpy()])
_, nearest_station_idx = rail_tree.query(road_xy)
road_nodes['nearest_station_id'] = rail_nodes.iloc[nearest_station_idx]['id'].values
road2station = dict(zip(road_nodes['id'].astype(str), road_nodes['nearest_station_id'].astype(str)))

# Map each iww node to its nearest road node
iww_xy = np.column_stack([iww_ports['coord_y'].to_numpy(), iww_ports['coord_x'].to_numpy()])
_, nearest_road_iww_idx = road_tree.query(iww_xy)
iww_ports['nearest_road_id'] = road_nodes.iloc[nearest_road_iww_idx]['id'].astype(str).values
iww2road = dict(zip(iww_ports['node_id'].astype(str), iww_ports['nearest_road_id']))

_, nearest_iww_for_road_idx = iww_tree.query(road_xy)
road_nodes['nearest_iww_id'] = iww_ports.iloc[nearest_iww_for_road_idx]['node_id'].values
road2iww = dict(zip(road_nodes['id'].astype(str), road_nodes['nearest_iww_id'].astype(str)))

# Now compute shares (compute_mode_share uses rail_tree/iww_tree)
flows[['road_share','rail_share','iww_share']] = flows.apply(compute_mode_share, axis=1, result_type='expand')
flows['road_flow'] = flows['land'] * flows['road_share']
flows['rail_flow'] = flows['land'] * flows['rail_share']
flows['iww_flow']  = flows['land'] * flows['iww_share']

# ------------------------------------------------------------------
# Nearest node assignment (vectorized, no Dask)
# ------------------------------------------------------------------
o_xy = np.column_stack([flows['o_lat'].to_numpy(float), flows['o_lon'].to_numpy(float)])
d_xy = np.column_stack([flows['d_lat'].to_numpy(float), flows['d_lon'].to_numpy(float)])

_, o_rail_idx = rail_tree.query(o_xy)
_, d_rail_idx = rail_tree.query(d_xy)
_, o_road_idx = road_tree.query(o_xy)
_, d_road_idx = road_tree.query(d_xy)
_, o_iww_idx  = iww_tree.query(o_xy)
_, d_iww_idx  = iww_tree.query(d_xy)

flows['from_rail_id'] = rail_nodes.iloc[o_rail_idx]['id'].astype(str).values
flows['to_rail_id'] = rail_nodes.iloc[d_rail_idx]['id'].astype(str).values
flows['from_road_id'] = road_nodes.iloc[o_road_idx]['id'].astype(str).values
flows['to_road_id'] = road_nodes.iloc[d_road_idx]['id'].astype(str).values
flows['from_iww_id'] = iww_ports.iloc[o_iww_idx]['node_id'].astype(str).values
flows['to_iww_id'] = iww_ports.iloc[d_iww_idx]['node_id'].astype(str).values


# Zero assigned mode flows where from_id == to_id (remove self-loops)
mask_rail_loop = flows['from_rail_id'] == flows['to_rail_id']
mask_road_loop = flows['from_road_id'] == flows['to_road_id']
mask_iww_loop  = flows['from_iww_id']  == flows['to_iww_id']
flows.loc[mask_rail_loop, 'rail_flow'] = 0.0
flows.loc[mask_road_loop, 'road_flow'] = 0.0
flows.loc[mask_iww_loop,  'iww_flow']  = 0.0

# Map station -> nearest road node using station ids
flows['origin_station_road_id'] = flows['from_rail_id'].map(station2road).astype(str)
flows['destination_station_road_id'] = flows['to_rail_id'].map(station2road).astype(str)

flows['origin_iww_road_id'] = flows['from_iww_id'].map(iww2road).astype(str)
flows['destination_iww_road_id'] = flows['to_iww_id'].map(iww2road).astype(str)

# Drop shares (if kept only for debugging)
flows = flows.drop(columns=['road_share','rail_share', 'iww_share'], errors='ignore')

print(flows.head(20), flush=True)
# ------------------------------------------------------------------
# Build OD matrices
# ------------------------------------------------------------------
need_sector = 'origin_sector' in flows.columns

od_matrix_rail = build_od(
    flows, 'from_rail_id', 'to_rail_id', 'rail_flow',
    extra_group_cols=(['origin_sector'] if need_sector else None)
)
od_matrix_rail.rename(columns={'from_rail_id':'from_id', 'to_rail_id':'to_id', 'rail_flow':'value'}, inplace=True)

od_matrix_road = build_od(
    flows, 'from_road_id', 'to_road_id', 'road_flow',
    extra_group_cols=(['origin_sector'] if need_sector else None)
)
od_matrix_road.rename(columns={'from_road_id':'from_id', 'to_road_id':'to_id', 'road_flow':'value'}, inplace=True)

od_matrix_iww = build_od(
    flows, 'from_iww_id', 'to_iww_id', 'iww_flow',
    extra_group_cols=(['origin_sector'] if need_sector else None)
)
od_matrix_iww.rename(columns={'from_iww_id':'from_id', 'to_iww_id':'to_id', 'iww_flow':'value'}, inplace=True)

od_industry_to_station = build_od(
    flows, 'from_road_id', 'origin_station_road_id', 'rail_flow',
    extra_group_cols=(['origin_sector'] if need_sector else None)
)
od_industry_to_station.rename(columns={'from_road_id':'from_id', 'origin_station_road_id':'to_id', 'rail_flow':'value'}, inplace=True)

od_station_to_industry = build_od(
    flows, 'destination_station_road_id', 'to_road_id', 'rail_flow',
    extra_group_cols=(['origin_sector'] if need_sector else None)
)
od_station_to_industry.rename(columns={'destination_station_road_id':'from_id', 'to_road_id':'to_id', 'rail_flow':'value'}, inplace=True)

od_iww_port_to_industry = build_od(
    flows, 'origin_iww_road_id', 'to_road_id', 'iww_flow',
    extra_group_cols=(['origin_sector'] if need_sector else None)
)
od_iww_port_to_industry.rename(columns={'origin_iww_road_id':'from_id', 'to_road_id':'to_id', 'iww_flow':'value'}, inplace=True)

od_iww_industry_to_port = build_od(
    flows, 'from_road_id', 'destination_iww_road_id', 'iww_flow',
    extra_group_cols=(['origin_sector'] if need_sector else None)
)
od_iww_industry_to_port.rename(columns={'from_road_id':'from_id', 'destination_iww_road_id':'to_id', 'iww_flow':'value'}, inplace=True)
# ------------------------------------------------------------------

od_matrix_station_industry = pd.concat(
    [od_industry_to_station, od_station_to_industry],
    ignore_index=True
)

od_matrix_iwwport_industry = pd.concat(
    [od_iww_port_to_industry, od_iww_industry_to_port],
    ignore_index=True
)

if 'value' in od_matrix_station_industry.columns:
    od_matrix_station_industry = od_matrix_station_industry[od_matrix_station_industry['value'] > 0]

od_matrix_station_industry.to_parquet(station_industry_od_file, index=False)

if 'value' in od_matrix_iwwport_industry.columns:
    od_matrix_port_industry_iww = od_matrix_iwwport_industry[od_matrix_iwwport_industry['value'] > 0]

od_matrix_port_industry_iww.to_parquet(iwwport_industry_od_file, index=False)

# ------------------------------------------------------------------
# Merge all land-related OD (industry/ports/airports/IWW + road/rail legs)
# Apply ROAD/RAIL/IWW factors to maritime_air_land_od,
# split into road-node ODs and station-to-station ODs
# ------------------------------------------------------------------

try:
    maritime_air_land_od = pd.read_parquet(maritime_air_land_file)
except Exception:
    maritime_air_land_od = pd.DataFrame(columns=['from_id','to_id','value','origin_sector'])
    print("Warning: maritime_air_land file missing.", flush=True)

# Set unified coord_x / coord_y
for nd in (road_nodes, rail_nodes):
    nd['coord_x'] = nd['longitude']
    nd['coord_y'] = nd['latitude']

def ensure_node_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OD has from_id/to_id as strings."""
    for c in ['from_id','to_id']:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' missing in OD dataframe.")
    df = df.copy()
    df['from_id'] = df['from_id'].astype(str)
    df['to_id'] = df['to_id'].astype(str)
    return df

def add_country_and_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Map from_id/to_id -> node country via road_nodes and compute ROAD/RAIL/IWW shares.
       Adjust RAIL/IWW shares by proximity from nearest station/port to the industry (road node)."""
    df = df.copy()
    # Build node->ISO2 country map
    if 'iso_a2' in road_nodes.columns:
        iso2 = road_nodes['iso_a2'].astype(str).str.upper().str.strip()
    elif 'iso_a3' in road_nodes.columns:
        iso3 = road_nodes['iso_a3'].astype(str).str.upper().str.strip()
        # use global ISO3_TO_ISO2
        iso2 = iso3.map(ISO3_TO_ISO2)
    else:
        print("Warning: road_nodes lacks country codes; defaulting external ODs to 100% ROAD.", flush=True)
        df['road_share2'] = 1.0; df['rail_share2'] = 0.0; df['iww_share2'] = 0.0
        return df

    node2country = dict(zip(road_nodes['id'].astype(str), iso2))
    df['origin_country'] = df['from_id'].astype(str).map(node2country)
    df['destination_country'] = df['to_id'].astype(str).map(node2country)

    if {'country_code','mode','share'}.issubset(intermodal_csv.columns) and not intermodal_csv.empty:
        ic = intermodal_csv.copy()  # already normalized
        piv = ic.pivot_table(index='country_code', columns='mode', values='share', aggfunc='first').fillna(0)
        road_share_map = piv['ROAD'].to_dict() if 'ROAD' in piv.columns else {}
        rail_share_map = piv['RAIL'].to_dict() if 'RAIL' in piv.columns else {}
        iww_share_map  = piv['IWW'].to_dict()  if 'IWW'  in piv.columns else {}

        o_r = df['origin_country'].map(road_share_map).fillna(0.0)
        d_r = df['destination_country'].map(road_share_map).fillna(0.0)
        o_l = df['origin_country'].map(rail_share_map).fillna(0.0)
        d_l = df['destination_country'].map(rail_share_map).fillna(0.0)
        o_w = df['origin_country'].map(iww_share_map).fillna(0.0)
        d_w = df['destination_country'].map(iww_share_map).fillna(0.0)

        road_share = (o_r + d_r) / 2.0
        rail_share = (o_l + d_l) / 2.0
        iww_share  = (o_w + d_w) / 2.0

        total = road_share + rail_share + iww_share
        total = total.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df['road_share2'] = np.where(total > 0, road_share / total, 1.0)
        df['rail_share2'] = np.where(total > 0, rail_share / total, 0.0)
        df['iww_share2']  = np.where(total > 0, iww_share  / total, 0.0)

        # --- Proximity multipliers (distance from facility -> industry road node) ---
        # Build coordinate maps
        road_lon = dict(zip(road_nodes['id'].astype(str), road_nodes['longitude']))
        road_lat = dict(zip(road_nodes['id'].astype(str), road_nodes['latitude']))
        st_lon   = dict(zip(rail_nodes['id'].astype(str),   rail_nodes['longitude']))
        st_lat   = dict(zip(rail_nodes['id'].astype(str),   rail_nodes['latitude']))
        iww_lon  = dict(zip(iww_ports['node_id'].astype(str), iww_ports['longitude']))
        iww_lat  = dict(zip(iww_ports['node_id'].astype(str), iww_ports['latitude']))

        # Nearest facility per road node (built earlier)
        # road2station, road2iww exist at module scope
        df['from_station'] = df['from_id'].map(road2station)
        df['to_station']   = df['to_id'].map(road2station)
        df['from_iww']     = df['from_id'].map(road2iww)
        df['to_iww']       = df['to_id'].map(road2iww)

        # Distances (km) from station/port -> industry road node
        fr_lat = df['from_id'].map(road_lat).astype(float); fr_lon = df['from_id'].map(road_lon).astype(float)
        to_lat = df['to_id'].map(road_lat).astype(float);   to_lon = df['to_id'].map(road_lon).astype(float)

        fr_st_lat = df['from_station'].map(st_lat).astype(float); fr_st_lon = df['from_station'].map(st_lon).astype(float)
        to_st_lat = df['to_station'].map(st_lat).astype(float);   to_st_lon = df['to_station'].map(st_lon).astype(float)

        fr_iw_lat = df['from_iww'].map(iww_lat).astype(float);    fr_iw_lon = df['from_iww'].map(iww_lon).astype(float)
        to_iw_lat = df['to_iww'].map(iww_lat).astype(float);      to_iw_lon = df['to_iww'].map(iww_lon).astype(float)

        rail_km_from = _haversine_km(fr_lat.to_numpy(), fr_lon.to_numpy(), fr_st_lat.to_numpy(), fr_st_lon.to_numpy())
        rail_km_to   = _haversine_km(to_lat.to_numpy(), to_lon.to_numpy(), to_st_lat.to_numpy(), to_st_lon.to_numpy())
        iww_km_from  = _haversine_km(fr_lat.to_numpy(), fr_lon.to_numpy(), fr_iw_lat.to_numpy(), fr_iw_lon.to_numpy())
        iww_km_to    = _haversine_km(to_lat.to_numpy(), to_lon.to_numpy(), to_iw_lat.to_numpy(), to_iw_lon.to_numpy())

        # Use min of origin/destination proximity
        rail_km = np.nanmin(np.vstack([rail_km_from, rail_km_to]), axis=0)
        iww_km  = np.nanmin(np.vstack([iww_km_from,  iww_km_to]),  axis=0)

        # Bucketed multipliers
        def _factor(arr):
            conds = [arr <= 20, arr <= 40, arr <= 60, arr <= 80, arr <= 100]
            choices = [5.0, 4.0, 3.0, 2.0, 1.0]
            return np.select(conds, choices, default=1.0).astype(float)

        rail_fac = _factor(rail_km)
        iww_fac  = _factor(iww_km)

        # Adjust and renormalize
        road_adj = df['road_share2'].to_numpy()
        rail_adj = df['rail_share2'].to_numpy() * rail_fac
        iww_adj  = df['iww_share2'].to_numpy()  * iww_fac
        tot = road_adj + rail_adj + iww_adj

        df['road_share2'] = road_adj / tot
        df['rail_share2'] = rail_adj / tot
        df['iww_share2']  = iww_adj  / tot

    else:
        print("Warning: intermodal CSV missing needed columns; defaulting to 100% ROAD.", flush=True)
        df['road_share2'] = 1.0
        df['rail_share2'] = 0.0
        df['iww_share2'] = 0.0
    return df

def split_modes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['road_value'] = df['value'].clip(lower=0) * df['road_share2']
    df['rail_value'] = df['value'].clip(lower=0) * df['rail_share2']
    df['iww_value'] = df['value'].clip(lower=0) * df['iww_share2']
    return df

def build_road_od(df: pd.DataFrame, extra_group_cols: list[str] | None = None) -> pd.DataFrame:
    return build_od(df, 'from_id', 'to_id', 'road_value', extra_group_cols=extra_group_cols)

def build_rail_station_od(df: pd.DataFrame, extra_group_cols: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    df['from_id'] = df['from_id'].map(road2station)
    df['to_id'] = df['to_id'].map(road2station)
    return build_od(df, 'from_id', 'to_id', 'rail_value', extra_group_cols=extra_group_cols)

def build_iww_port_od(df: pd.DataFrame, extra_group_cols: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    # mar_land_df has road-node IDs; convert them to nearest IWW port IDs
    df['from_id'] = df['from_id'].map(road2iww)
    df['to_id'] = df['to_id'].map(road2iww)
    # drop rows that didn't map (prevents null from_id/to_id in the OD)
    df = df.dropna(subset=['from_id','to_id'])
    return build_od(df, 'from_id', 'to_id', 'iww_value', extra_group_cols=extra_group_cols)

# Prepare and split both external ODs
need_sector_ext = 'origin_sector' in maritime_air_land_od.columns
extra_cols = ['origin_sector'] if need_sector_ext else None
maritime_air_land_od = ensure_node_cols(maritime_air_land_od)
maritime_air_land_od = add_country_and_shares(maritime_air_land_od)
mar_land_df = split_modes(maritime_air_land_od)


# Reassign external rail/iww self-loops (same station/port at both ends) to roads
_st_from = mar_land_df['from_id'].map(road2station)
_st_to   = mar_land_df['to_id'].map(road2station)
mask_ext_rail_self = _st_from.notna() & (_st_from == _st_to)

_pw_from = mar_land_df['from_id'].map(road2iww)
_pw_to   = mar_land_df['to_id'].map(road2iww)
mask_ext_iww_self = _pw_from.notna() & (_pw_from == _pw_to)

if mask_ext_rail_self.any():
    mar_land_df.loc[mask_ext_rail_self, 'road_value'] = (
        mar_land_df.loc[mask_ext_rail_self, 'road_value'].fillna(0.0) +
        mar_land_df.loc[mask_ext_rail_self, 'rail_value'].fillna(0.0)
    )
    mar_land_df.loc[mask_ext_rail_self, 'rail_value'] = 0.0

if mask_ext_iww_self.any():
    mar_land_df.loc[mask_ext_iww_self, 'road_value'] = (
        mar_land_df.loc[mask_ext_iww_self, 'road_value'].fillna(0.0) +
        mar_land_df.loc[mask_ext_iww_self, 'iww_value'].fillna(0.0)
    )
    mar_land_df.loc[mask_ext_iww_self, 'iww_value'] = 0.0

# Build derived ODs
mar_road_od = build_road_od(mar_land_df, extra_group_cols=extra_cols)
mar_rail_od = build_rail_station_od(mar_land_df, extra_group_cols=extra_cols)
mar_iww_od = build_iww_port_od(mar_land_df, extra_group_cols=extra_cols)
mar_road_od.to_parquet(mar_road_od_file, index=False)
mar_rail_od.to_parquet(mar_rail_od_file, index=False)
mar_iww_od.to_parquet(mar_iww_od_file, index=False)

# Self-loops
mask_rail_loop = flows['from_rail_id'] == flows['to_rail_id']
mask_road_loop = flows['from_road_id'] == flows['to_road_id']
mask_iww_loop  = flows['from_iww_id']  == flows['to_iww_id']

# Assign rail/iww self-loop flows to road, then zero the rail/iww amounts
if mask_rail_loop.any():
    flows.loc[mask_rail_loop, 'road_flow'] = (
        flows.loc[mask_rail_loop, 'road_flow'].fillna(0.0) +
        flows.loc[mask_rail_loop, 'rail_flow'].fillna(0.0)
    )
    flows.loc[mask_rail_loop, 'rail_flow'] = 0.0

if mask_iww_loop.any():
    flows.loc[mask_iww_loop, 'road_flow'] = (
        flows.loc[mask_iww_loop, 'road_flow'].fillna(0.0) +
        flows.loc[mask_iww_loop, 'iww_flow'].fillna(0.0)
    )
    flows.loc[mask_iww_loop, 'iww_flow'] = 0.0

# Keep road self-loops handling as-is (zero if you want to drop them)
flows.loc[mask_road_loop, 'road_flow'] = 0.0


# Combine with existing ODs
# od_matrix_rail/road and od_matrix_station_industry already have from_id/to_id/value
od_matrix_rail.to_parquet(rail_od_file, index=False)
rail_od_all = pd.concat([od_matrix_rail, mar_rail_od], ignore_index=True)
rail_od_all = rail_od_all[rail_od_all['value'] > 0]
rail_od_all.to_parquet(rail_od_file_final, index=False)

od_matrix_iww.to_parquet(iww_od_file, index=False)
iww_od_all = pd.concat([od_matrix_iww, mar_iww_od], ignore_index=True)
iww_od_all = iww_od_all[iww_od_all['value'] > 0]
iww_od_all.to_parquet(iww_od_file_final, index=False)

road_od_all = pd.concat(
    [od_matrix_road, od_matrix_station_industry, od_matrix_port_industry_iww, mar_road_od],
    ignore_index=True
)
road_od_all = road_od_all[road_od_all['value'] > 0]
road_od_all.to_parquet(road_od_file_final, index=False)

print("Saved final combined rail and road ODs with external IWW/Maritime splits.", flush=True)


# Save the final shape of the flows_df
flows_df = flows.filter(items=[
    'origin_id', 'destination_id', 'origin_nuts2', 'destination_nuts2', 'origin_sector', 'destination_sector', 'origin_country', 'destination_country',
    'origin_port', 'destination_port', 'origin_airport', 'destination_airport', 'origin_iww_port', 'destination_iww_port',
    'total_flow', 'road_flow', 'rail_flow', 'iww_flow', 'air_flow', 'maritime_flow'
])

flows_df.to_parquet('/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/flows_with_modes.parquet', index=False)
