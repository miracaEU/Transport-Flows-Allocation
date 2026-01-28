# Air and maritime OD creation

# What this script does:
# - Load updated ports, airports, and industries-with-closest-facilities GeoParquets.
# - Normalize geometries and CRS; compute lat/lon and projected coordinates for industries.
# - Load OD flows, compute Euclidean distances between industry origins/destinations in EPSG:3035.
# - Map each industry to its closest airport/port and their capacities (Eurostat totals).
# - Apply distance and capacity gates to assign portions of flows to air and maritime modes.
# - Subtract assigned air/maritime flows to leave residual as land; export OD matrices for air, maritime, and land connectors.


from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkb import loads
from shapely.wkb import loads
import sys 
# ---------------------------
# Helpers
# ---------------------------

def wkb_to_gdf(df: pd.DataFrame, crs: str) -> gpd.GeoDataFrame:
    """Decode WKB -> geometry and set CRS."""
    df = df.copy()
    df['geometry'] = df['geometry'].apply(lambda b: loads(b) if pd.notna(b) else None)
    return gpd.GeoDataFrame(df, geometry='geometry', crs=crs)

# ---------------------------
# Load inputs
# ---------------------------

ports_df_raw = pd.read_parquet('/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_ports_TENT.parquet')
airports_df_raw = pd.read_parquet('/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_airports_TENT.parquet')

industries_df = pd.read_parquet('/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industries_w_outputs/industries_with_closest_ports_airports_euclidean.parquet')
industries_df = industries_df[pd.to_numeric(industries_df['area'], errors='coerce').fillna(0) > 0].copy()

# ---------------------------
# CRS and geometry normalization (explicit CRS)
# ---------------------------
# Ports/airports are now stored in EPSG:3035 (from 02_catchment_assignment.py)
# Industries are stored in EPSG:3035 as well
ports_gdf = wkb_to_gdf(ports_df_raw, crs="EPSG:3035")
airports_gdf = wkb_to_gdf(airports_df_raw, crs="EPSG:3035")
industries_gdf = wkb_to_gdf(industries_df, crs="EPSG:3035")

# Compute longitude/latitude if not already present
for gdf, name in [(ports_gdf, 'ports'), (airports_gdf, 'airports'), (industries_gdf, 'industries')]:
    if 'longitude' not in gdf.columns or 'latitude' not in gdf.columns:
        # Compute centroid and project to WGS84 for lon/lat
        gdf_4326 = gdf.set_geometry(gdf.geometry.centroid).to_crs("EPSG:4326")
        gdf['longitude'] = gdf_4326.geometry.x
        gdf['latitude'] = gdf_4326.geometry.y
        print(f"Computed lon/lat for {name}", flush=True)

# Use the geodataframes with computed coordinates
ports_df = ports_gdf
airports_df = airports_gdf
industries_df = industries_gdf

# Unique industry index and maps for later joining
_ind = (
    industries_df.sort_values('area', ascending=False)
    .drop_duplicates(subset='id')
    .set_index('id')
)

# ---------------------------
# Load flows and compute distances
# ---------------------------
flow_path = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industrial_OD/industrial_OD_threshold_0.25.parquet")
flows_df = pd.read_parquet(flow_path)

# Ensure correct dtypes for mapping
_ind.index = _ind.index.astype(flows_df['origin_id'].dtype, copy=False)

# Countries for extra‑EU centroid lookup
COUNTRIES_SHP = "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/admin/ne_10m/ne_10m_admin_0_countries.shp"

countries_gdf = gpd.read_file(COUNTRIES_SHP)
# Compute centroid in projected CRS first, then get lat/lon
countries_projected = countries_gdf.to_crs("EPSG:3035")
countries_projected["centroid_geom"] = countries_projected.geometry.centroid
# Convert centroids back to WGS84 for lon/lat
countries_gdf = countries_projected.to_crs("EPSG:4326")
countries_gdf["centroid"] = countries_gdf["centroid_geom"]
# Pick ISO2 column or derive from ISO3
iso2_col = None
for c in ["ISO_A2", "iso_a2", "ISO_A2_EH"]:
    if c in countries_gdf.columns:
        iso2_col = c
        break
if iso2_col is None:
    iso3_col = next((c for c in ["ISO_A3", "ISO_A3_EH", "ADM0_A3"] if c in countries_gdf.columns), None)
    if iso3_col:
        iso3_vals = countries_gdf[iso3_col].astype(str).str.upper()
        iso3_to_iso2 = {
            'AUT':'AT','BEL':'BE','BGR':'BG','HRV':'HR','CYP':'CY','CZE':'CZ','DNK':'DK','EST':'EE','FIN':'FI',
            'FRA':'FR','DEU':'DE','GRC':'GR','HUN':'HU','IRL':'IE','ITA':'IT','LVA':'LV','LTU':'LT','LUX':'LU',
            'MLT':'MT','NLD':'NL','POL':'PL','PRT':'PT','ROU':'RO','SVK':'SK','SVN':'SI','ESP':'ES','SWE':'SE',
            'GBR':'GB','NOR':'NO','CHE':'CH','ISL':'IS','ALB':'AL','SRB':'RS','MKD':'MK','MNE':'ME','BIH':'BA',
            'MDA':'MD','UKR':'UA','BLR':'BY','RUS':'RU','TUR':'TR','GIB':'GI','AND':'AD','LIE':'LI','MCO':'MC',
            'KOS':'XK','XKX':'XK'
        }
        countries_gdf["ISO2_TMP"] = iso3_vals.map(iso3_to_iso2)
        iso2_col = "ISO2_TMP"
countries_gdf["centroid"] = countries_gdf.geometry.centroid
country_centroid_map_lon = countries_gdf.set_index(iso2_col)["centroid"].x.to_dict() if iso2_col else {}
country_centroid_map_lat = countries_gdf.set_index(iso2_col)["centroid"].y.to_dict() if iso2_col else {}

# ---------------------------
# Extra-EU handling and mappings
# ---------------------------
# Identify extra‑European (code chars 2:4 == '00')
origin_extra = flows_df['origin_nuts2'].str[2:4].eq('00')
destination_extra = flows_df['destination_nuts2'].str[2:4].eq('00')

# Drop flows where both origin and destination are extra‑EU
both_extra = origin_extra & destination_extra
removed_rows = int(both_extra.sum())
if removed_rows:
    print(f"Removing {removed_rows} flows with both origin and destination outside EU.", flush=True)
flows_df = flows_df.loc[~both_extra].copy()

# Recompute masks after filtering
origin_extra = flows_df['origin_nuts2'].str[2:4].eq('00')
destination_extra = flows_df['destination_nuts2'].str[2:4].eq('00')
in_origin = ~origin_extra
in_destination = ~destination_extra

# Ensure country codes exist for centroid mapping (fallback if missing)
if 'origin_country' not in flows_df.columns:
    flows_df['origin_country'] = flows_df['origin_nuts2'].str[:2]
if 'destination_country' not in flows_df.columns:
    flows_df['destination_country'] = flows_df['destination_nuts2'].str[:2]

for col in [
    'origin_port','origin_airport','destination_port','destination_airport',
    'origin_node_port','origin_node_airport','destination_node_port','destination_node_airport',
    'origin_node','destination_node'
]:
    if col in flows_df.columns:
        flows_df[col] = flows_df[col].astype('string')

# Build industry -> code/node maps (string)
port_map = _ind['closest_port'].astype('string', errors='ignore')
airport_map = _ind['closest_airport'].astype('string', errors='ignore')
node_port_map = _ind['closest_node_port'].astype('string', errors='ignore')
node_airport_map = _ind['closest_node_airport'].astype('string', errors='ignore')
node_map = _ind['closest_node'].astype('string', errors='ignore')

# Assign placeholders for extra‑EU rows
flows_df.loc[origin_extra, 'origin_port'] = (flows_df.loc[origin_extra, 'origin_country'] + 'XXX').astype('string')
flows_df.loc[origin_extra, 'origin_airport'] = (flows_df.loc[origin_extra, 'origin_country'] + 'XX').astype('string')
flows_df.loc[destination_extra, 'destination_port'] = (flows_df.loc[destination_extra, 'destination_country'] + 'XXX').astype('string')
flows_df.loc[destination_extra, 'destination_airport'] = (flows_df.loc[destination_extra, 'destination_country'] + 'XX').astype('string')

# Assign actual codes for in‑EU using maps
flows_df.loc[in_origin, 'origin_port'] = flows_df.loc[in_origin, 'origin_id'].map(port_map).astype('string')
flows_df.loc[in_origin, 'origin_airport'] = flows_df.loc[in_origin, 'origin_id'].map(airport_map).astype('string')
flows_df.loc[in_destination, 'destination_port'] = flows_df.loc[in_destination, 'destination_id'].map(port_map).astype('string')
flows_df.loc[in_destination, 'destination_airport'] = flows_df.loc[in_destination, 'destination_id'].map(airport_map).astype('string')
flows_df.loc[in_origin, 'origin_node_port'] = flows_df.loc[in_origin, 'origin_id'].map(node_port_map).astype('string')
flows_df.loc[in_origin, 'origin_node_airport'] = flows_df.loc[in_origin, 'origin_id'].map(node_airport_map).astype('string')
flows_df.loc[in_destination, 'destination_node_port'] = flows_df.loc[in_destination, 'destination_id'].map(node_port_map).astype('string')
flows_df.loc[in_destination, 'destination_node_airport'] = flows_df.loc[in_destination, 'destination_id'].map(node_airport_map).astype('string')


# Assign port coordinates: in‑EU from actual port centroids, extra‑EU from country centroid
port_lon_map = ports_df.dropna(subset=['port_code']).drop_duplicates('port_code').set_index('port_code')['longitude'].to_dict()
port_lat_map = ports_df.dropna(subset=['port_code']).drop_duplicates('port_code').set_index('port_code')['latitude'].to_dict()

for c in ['origin_port_lon','origin_port_lat','destination_port_lon','destination_port_lat']:
    if c not in flows_df.columns:
        flows_df[c] = np.nan

# In‑EU mapping by port code
flows_df.loc[in_origin, 'origin_port_lon'] = flows_df.loc[in_origin, 'origin_port'].map(port_lon_map)
flows_df.loc[in_origin, 'origin_port_lat'] = flows_df.loc[in_origin, 'origin_port'].map(port_lat_map)
flows_df.loc[in_destination, 'destination_port_lon'] = flows_df.loc[in_destination, 'destination_port'].map(port_lon_map)
flows_df.loc[in_destination, 'destination_port_lat'] = flows_df.loc[in_destination, 'destination_port'].map(port_lat_map)

# Extra‑EU mapping by country centroid
if country_centroid_map_lon and country_centroid_map_lat:
    flows_df.loc[origin_extra, 'origin_port_lon'] = flows_df.loc[origin_extra, 'origin_country'].map(country_centroid_map_lon)
    flows_df.loc[origin_extra, 'origin_port_lat'] = flows_df.loc[origin_extra, 'origin_country'].map(country_centroid_map_lat)
    flows_df.loc[destination_extra, 'destination_port_lon'] = flows_df.loc[destination_extra, 'destination_country'].map(country_centroid_map_lon)
    flows_df.loc[destination_extra, 'destination_port_lat'] = flows_df.loc[destination_extra, 'destination_country'].map(country_centroid_map_lat)

# Industry road node ids for land OD connectors
flows_df.loc[in_origin, 'origin_node'] = flows_df.loc[in_origin, 'origin_id'].map(node_map).astype('string')
flows_df.loc[in_destination, 'destination_node'] = flows_df.loc[in_destination, 'destination_id'].map(node_map).astype('string')

# ---------------------------
# Facility capacities (Eurostat totals aggregated per code)
# ---------------------------
def _caps(df: gpd.GeoDataFrame, key_col: str, out_col: str, in_col: str):
    df_local = df.dropna(subset=[key_col]).copy()
    df_local[key_col] = df_local[key_col].astype(str)
    caps = df_local.groupby(key_col, as_index=True)[[out_col, in_col]].sum()
    return caps[out_col], caps[in_col]

airport_out_map, airport_in_map = _caps(airports_df, 'icao', 'OBS_VALUE_OUT', 'OBS_VALUE_IN')
port_out_map, port_in_map       = _caps(ports_df,    'port_code', 'OBS_VALUE_OUT', 'OBS_VALUE_IN')

# Assign capacities (transfer to ths tons)
flows_df['origin_airport_capacity']      = pd.to_numeric(flows_df['origin_airport'].astype(str).map(airport_out_map), errors='coerce') / 1000.0
flows_df['destination_airport_capacity'] = pd.to_numeric(flows_df['destination_airport'].astype(str).map(airport_in_map), errors='coerce') / 1000.0
flows_df['origin_port_capacity']         = pd.to_numeric(flows_df['origin_port'].astype(str).map(port_out_map), errors='coerce') / 1000.0
flows_df['destination_port_capacity']    = pd.to_numeric(flows_df['destination_port'].astype(str).map(port_in_map), errors='coerce') / 1000.0

# Very large capacity for extra‑EU placeholders
PLACEHOLDER_CAP = 10**10
# Airports placeholders end with 'XX'
mask_o_air_placeholder = flows_df['origin_airport'].fillna('').str.endswith('XX')
mask_d_air_placeholder = flows_df['destination_airport'].fillna('').str.endswith('XX')
flows_df.loc[mask_o_air_placeholder, 'origin_airport_capacity'] = PLACEHOLDER_CAP
flows_df.loc[mask_d_air_placeholder, 'destination_airport_capacity'] = PLACEHOLDER_CAP
# Ports placeholders end with 'XXX' (FIX)
mask_o_port_placeholder = flows_df['origin_port'].fillna('').str.endswith('XXX')
mask_d_port_placeholder = flows_df['destination_port'].fillna('').str.endswith('XXX')
flows_df.loc[mask_o_port_placeholder, 'origin_port_capacity'] = PLACEHOLDER_CAP
flows_df.loc[mask_d_port_placeholder, 'destination_port_capacity'] = PLACEHOLDER_CAP

for cap_col in [
    'origin_airport_capacity','destination_airport_capacity',
    'origin_port_capacity','destination_port_capacity'
]:
    flows_df[cap_col] = pd.to_numeric(flows_df[cap_col], errors='coerce').fillna(0.0)


# ---------------------------
# Maritime assignment (distance gating)
# ---------------------------
DISTANCE_THRESHOLD_KM = 100
mask_distance_ok = flows_df['distance'].ge(DISTANCE_THRESHOLD_KM)
# Save original total
flows_df['total_flow'] = flows_df['value']

for col in [
    'cumulated_port_origin','cumulated_port_origin_od',
    'cumulated_port_destination','cumulated_port_destination_od',
    'origin_maritime_assigned','destination_maritime_assigned','maritime_flow'
]:
    flows_df[col] = 0.0

maritime_subset_idx = flows_df.index[mask_distance_ok]
if len(maritime_subset_idx) > 0:
    mar_sub = flows_df.loc[maritime_subset_idx, ['origin_port','destination_port','value','distance']].copy()

    # Origin-side
    mar_sub = mar_sub.sort_values(by=['origin_port','destination_port','distance'], ascending=[True, True, False])
    mar_sub['cum_origin'] = mar_sub.groupby('origin_port')['value'].cumsum()
    mar_sub['cum_origin_od'] = mar_sub.groupby(['origin_port','destination_port'])['value'].cumsum()

    # Destination-side
    mar_sub = mar_sub.sort_values(by=['destination_port','origin_port','distance'], ascending=[True, True, False])
    mar_sub['cum_destination'] = mar_sub.groupby('destination_port')['value'].cumsum()
    mar_sub['cum_destination_od'] = mar_sub.groupby(['destination_port','origin_port'])['value'].cumsum()

    flows_df.loc[maritime_subset_idx, 'cumulated_port_origin'] = mar_sub['cum_origin']
    flows_df.loc[maritime_subset_idx, 'cumulated_port_origin_od'] = mar_sub['cum_origin_od']
    flows_df.loc[maritime_subset_idx, 'cumulated_port_destination'] = mar_sub['cum_destination']
    flows_df.loc[maritime_subset_idx, 'cumulated_port_destination_od'] = mar_sub['cum_destination_od']

    flows_df.loc[maritime_subset_idx, 'origin_maritime_assigned'] = (
        flows_df.loc[maritime_subset_idx, 'cumulated_port_origin'] <
        flows_df.loc[maritime_subset_idx, 'origin_port_capacity']
    ).astype(int)

    flows_df.loc[maritime_subset_idx, 'destination_maritime_assigned'] = (
        flows_df.loc[maritime_subset_idx, 'cumulated_port_destination'] <
        flows_df.loc[maritime_subset_idx, 'destination_port_capacity']
    ).astype(int)

    flows_df.loc[maritime_subset_idx, 'maritime_flow'] = np.where(
        (flows_df.loc[maritime_subset_idx, 'origin_maritime_assigned'] == 1) &
        (flows_df.loc[maritime_subset_idx, 'destination_maritime_assigned'] == 1),
        flows_df.loc[maritime_subset_idx, 'value'],
        0.0
    )

# Subtract maritime from remaining value (residual = land)
flows_df['value'] = flows_df['value'] - flows_df['maritime_flow']

# Save updated flows with assignments
flows_output_file = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industries_w_outputs/industrial_OD_threshold_0.25_airmar_assigned.parquet")
flows_output_file.parent.mkdir(parents=True, exist_ok=True)
flows_df.to_parquet(flows_output_file, index=False)

# ---------------------------
# Air assignment (sector + distance gating)
# ---------------------------
allowed_air_sectors = {'COMMERCIAL', 'FOOD AND BEVERAGE'}
mask_air_sector = flows_df['origin_sector'].isin(allowed_air_sectors)

# Init columns
for col in [
    'cumulated_airport_origin','cumulated_airport_destination',
    'cumulated_airport_origin_od','cumulated_airport_destination_od',
    'origin_air_assigned','destination_air_assigned','air_flow'
]:
    flows_df[col] = 0.0

air_subset_idx = flows_df.index[mask_air_sector & mask_distance_ok]

print(f"Applied distance threshold {DISTANCE_THRESHOLD_KM} km for air & maritime eligibility", flush=True)

if len(air_subset_idx) > 0:
    air_sub = flows_df.loc[air_subset_idx, ['origin_airport','destination_airport','value','distance']].copy()

    # Origin-side: cumsum by airport, farthest first
    air_sub = air_sub.sort_values(by=['origin_airport','distance'], ascending=[True, False])
    air_sub['cum_origin'] = air_sub.groupby('origin_airport')['value'].cumsum()
    air_sub['cum_origin_od'] = air_sub.groupby(['origin_airport','destination_airport'])['value'].cumsum()

    # Destination-side
    air_sub = air_sub.sort_values(by=['destination_airport','distance'], ascending=[True, False])
    air_sub['cum_destination'] = air_sub.groupby('destination_airport')['value'].cumsum()
    air_sub['cum_destination_od'] = air_sub.groupby(['destination_airport','origin_airport'])['value'].cumsum()

    flows_df.loc[air_subset_idx, 'cumulated_airport_origin'] = air_sub['cum_origin']
    flows_df.loc[air_subset_idx, 'cumulated_airport_origin_od'] = air_sub['cum_origin_od']
    flows_df.loc[air_subset_idx, 'cumulated_airport_destination'] = air_sub['cum_destination']
    flows_df.loc[air_subset_idx, 'cumulated_airport_destination_od'] = air_sub['cum_destination_od']

    flows_df.loc[air_subset_idx, 'origin_air_assigned'] = (
        flows_df.loc[air_subset_idx, 'cumulated_airport_origin'] <
        flows_df.loc[air_subset_idx, 'origin_airport_capacity']
    ).astype(int)

    flows_df.loc[air_subset_idx, 'destination_air_assigned'] = (
        flows_df.loc[air_subset_idx, 'cumulated_airport_destination'] <
        flows_df.loc[air_subset_idx, 'destination_airport_capacity']
    ).astype(int)

    flows_df.loc[air_subset_idx, 'air_flow'] = np.where(
        (flows_df.loc[air_subset_idx, 'origin_air_assigned'] == 1) &
        (flows_df.loc[air_subset_idx, 'destination_air_assigned'] == 1),
        flows_df.loc[air_subset_idx, 'value'],
        0.0
    )
else:
    flows_df[['origin_air_assigned','destination_air_assigned','air_flow']] = 0.0

# Disallowed sectors or short-distance flows -> zero air
flows_df.loc[~(mask_air_sector & mask_distance_ok),
             ['origin_air_assigned','destination_air_assigned','air_flow',
              'cumulated_airport_origin','cumulated_airport_destination',
              'cumulated_airport_origin_od','cumulated_airport_destination_od']] = 0.0

flows_df['value'] = flows_df['value'] - flows_df['air_flow']


# ---------------------------
# Build OD matrices
# ---------------------------
print('Computing ODs', flush=True)

# Maritime OD by port codes
od_matrix_maritime = flows_df.groupby(
    ['origin_port', 'destination_port', 'origin_sector'], as_index=False
)['maritime_flow'].sum().rename(columns={
    'origin_port': 'from_id',
    'destination_port': 'to_id',
    'maritime_flow': 'value'
})[['from_id', 'to_id', 'value','origin_sector']]

# Air OD by airport codes
od_matrix_air = flows_df.groupby(
    ['origin_airport', 'destination_airport', 'origin_sector'], as_index=False
)['air_flow'].sum().rename(columns={
    'origin_airport': 'from_id',
    'destination_airport': 'to_id',
    'air_flow': 'value'
})[['from_id', 'to_id', 'value','origin_sector']]

# Land connector ODs (industry node <-> facility node)
od_industry_to_port = flows_df.groupby(
    ['origin_node', 'origin_node_port', 'origin_sector'], as_index=False
)['maritime_flow'].sum().rename(columns={
    'origin_node': 'from_id',
    'origin_node_port': 'to_id',
    'maritime_flow': 'value'
})[['from_id', 'to_id', 'value','origin_sector']]

od_port_to_industry = flows_df.groupby(
    ['destination_node_port', 'destination_node', 'origin_sector'], as_index=False
)['maritime_flow'].sum().rename(columns={
    'destination_node_port': 'from_id',
    'destination_node': 'to_id',
    'maritime_flow': 'value'
})[['from_id', 'to_id', 'value','origin_sector']]

od_industry_to_airport = flows_df.groupby(
    ['origin_node', 'origin_node_airport', 'origin_sector'], as_index=False
)['air_flow'].sum().rename(columns={
    'origin_node': 'from_id',
    'origin_node_airport': 'to_id',
    'air_flow': 'value'
})[['from_id', 'to_id', 'value','origin_sector']]

od_airport_to_industry = flows_df.groupby(
    ['destination_node_airport', 'destination_node', 'origin_sector'], as_index=False
)['air_flow'].sum().rename(columns={
    'destination_node_airport': 'from_id',
    'destination_node': 'to_id',
    'air_flow': 'value'
})[['from_id', 'to_id', 'value','origin_sector']]

# Combine land connectors
od_land = pd.concat(
    [od_industry_to_port, od_port_to_industry, od_industry_to_airport, od_airport_to_industry],
    ignore_index=True
)

# Keep positive values and ensure ids are strings
def _clean_od(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['from_id', 'to_id', 'value'])
    df = df[df['value'] > 0]
    df['from_id'] = df['from_id'].astype(str)
    df['to_id'] = df['to_id'].astype(str)
    return df

od_matrix_maritime = _clean_od(od_matrix_maritime)
od_matrix_air = _clean_od(od_matrix_air)
od_land = _clean_od(od_land)

# Keep positive values and ensure ids are strings
for _df in (od_matrix_maritime, od_matrix_air, od_land):
    _df['value'] = pd.to_numeric(_df['value'], errors='coerce').fillna(0)
    _df.dropna(subset=['from_id', 'to_id'], inplace=True)
    _df = _df[_df['value'] > 0]
    _df['from_id'] = _df['from_id'].astype(str)
    _df['to_id'] = _df['to_id'].astype(str)

# Save OD matrices
output_file_maritime = Path('/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/maritime_OD.parquet')
output_file_air = Path('/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/air_OD.parquet')
output_file_land = Path('/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/intermediate_ODs/OD_ports_airports_industries.parquet')
for p in (output_file_maritime, output_file_air, output_file_land):
    p.parent.mkdir(parents=True, exist_ok=True)

od_matrix_maritime.to_parquet(output_file_maritime, index=False)
od_matrix_air.to_parquet(output_file_air, index=False)
od_land.to_parquet(output_file_land, index=False)

print("Saved OD matrices", flush=True)

# Aggregate assigned flows onto facilities and attach to ports/airports
def _sum_by(df: pd.DataFrame, from_col: str, to_col: str, val_col: str = 'value'):
    if df.empty:
        return {}, {}
    out = df.groupby(from_col, as_index=True)[val_col].sum().to_dict()
    inn = df.groupby(to_col, as_index=True)[val_col].sum().to_dict()
    return out, inn

mar_out_map, mar_in_map = _sum_by(od_matrix_maritime, 'from_id', 'to_id')
air_out_map, air_in_map = _sum_by(od_matrix_air, 'from_id', 'to_id')

# Ports: distribute totals from maritime OD by area (same as OBS_VALUE distribution)
# First map totals by port_code
ports_df['port_total_out'] = ports_df['port_code'].astype(str).map(mar_out_map).fillna(0.0)
ports_df['port_total_in'] = ports_df['port_code'].astype(str).map(mar_in_map).fillna(0.0)

# Identify placeholder ports (extra-EU) - they end with 'XXX'
is_placeholder = ports_df['port_code'].astype(str).str.endswith('XXX')

# Calculate total area per port_code (only for real ports, not placeholders)
ports_df['port_code_area'] = ports_df.groupby('port_code')['area'].transform('sum')

# Distribute flows proportionally by area within each port (only for real EU ports)
# For placeholders, keep the total (they represent country-level aggregates, not terminals)
ports_df['total_outgoing_flow'] = np.where(
    is_placeholder,
    ports_df['port_total_out'],  # Placeholders: keep total
    np.where(
        ports_df['port_code_area'] > 0,
        ports_df['port_total_out'] * (ports_df['area'] / ports_df['port_code_area']),  # Real ports: distribute by area
        0.0
    )
)
ports_df['total_incoming_flow'] = np.where(
    is_placeholder,
    ports_df['port_total_in'],  # Placeholders: keep total
    np.where(
        ports_df['port_code_area'] > 0,
        ports_df['port_total_in'] * (ports_df['area'] / ports_df['port_code_area']),  # Real ports: distribute by area
        0.0
    )
)

# Drop temporary columns
ports_df = ports_df.drop(columns=['port_total_out', 'port_total_in', 'port_code_area'])

# Airports: totals from air OD (use ICAO)
airports_df['air_out'] = airports_df['icao'].astype(str).map(air_out_map).fillna(0.0)
airports_df['air_in']  = airports_df['icao'].astype(str).map(air_in_map).fillna(0.0)
airports_df['total_outgoing_flow'] = airports_df['air_out']
airports_df['total_incoming_flow'] = airports_df['air_in']
# Drop air_out and air_in columns
airports_df = airports_df.drop(columns=['air_out', 'air_in'])

# Save updated ports and airports with assigned flows
ports_output_file = Path("/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_ports_TENT.parquet")
ports_df.to_parquet(ports_output_file, index=False)
airports_output_file = Path("/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_airports_TENT.parquet")
airports_df.to_parquet(airports_output_file, index=False)

print("Updated ports and airports with assigned flows", flush=True)