"""
Downscale industrial/commercial flows to site-to-site OD pairs by:
- Loading industries and commercial sites (GeoParquet) and normalizing to EPSG:3035.
- Normalizing flow matrix (Excel) indexed/columned by <ISO2><NUTS2/00><PRODUCT>.
- For each product pair and country pair, selecting matching sites by NUTS2.
- Distributing flows proportionally by site area within NUTS2, with special handling for intra-NUTS2.
- Writing per-commodity, per-country-pair OD parquet files with distances (EPSG:4258 haversine).
"""

import warnings
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import sys
import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

CRS = "EPSG:3035"

# ---------------------------
# Helpers
# ---------------------------
def read_gdf_auto(path: Path) -> gpd.GeoDataFrame:
    """Read GeoDataFrame from Parquet if possible, else fall back to generic/explicit GPKG."""
    p = str(path).lower()
    try:
        if p.endswith((".parquet", ".pq")):
            return gpd.read_parquet(path)
        return gpd.read_file(path)
    except Exception as e:
        # Fallback to explicit GPKG (handles misnamed files)
        print(f"Warning: failed to read as Parquet/auto for {path}. Falling back to GPKG. ({e})", flush=True)
        try:
            return gpd.read_file(path, driver="GPKG")
        except Exception as e2:
            raise RuntimeError(f"Unable to read {path} as Parquet or GPKG") from e2

def ensure_xy_4258(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure x_4258/y_4258 exist; compute from geometry centroid in EPSG:4258."""
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    needs = ('x_4258' not in gdf.columns) or ('y_4258' not in gdf.columns) or gdf['x_4258'].isna().any() or gdf['y_4258'].isna().any()
    if needs:
        g4258 = gdf.to_crs("EPSG:4258")
        cent = g4258.geometry.centroid
        gdf['x_4258'] = cent.x
        gdf['y_4258'] = cent.y
    return gdf

def _iso2_lookup_field(gdf: gpd.GeoDataFrame) -> str:
    for c in ('ISO_A2', 'ISO_A2_EH'):
        if c in gdf.columns:
            return c
    raise KeyError("ISO2 field not found in countries_gdf (expected ISO_A2 or ISO_A2_EH)")

def create_dummy_industry(nuts2_code, sector, countries_gdf, area_col='area_nuts2'):
    """Create a 1-area dummy site at the centroid of the ISO2 country (from first two chars of nuts2)."""
    iso2 = (nuts2_code or '')[:2]
    # EU alias: EL -> GR (Natural Earth uses GR)
    if iso2 == 'EL':
        iso2 = 'GR'
    iso_col = _iso2_lookup_field(countries_gdf)
    row = countries_gdf[countries_gdf[iso_col] == iso2]

    if row.empty:
        geom_3035 = None
        x4258 = None
        y4258 = None
    else:
        row_4258 = row.to_crs("EPSG:4258")
        cent_4258 = row_4258.geometry.centroid.iloc[0]
        x4258, y4258 = cent_4258.x, cent_4258.y
        row_3035 = row.to_crs(CRS)
        geom_3035 = row_3035.geometry.centroid.iloc[0]

    data = {
        'id': 0,
        'nuts2': nuts2_code,
        'eprtr_sectors': sector if sector else 'UNKNOWN',
        'area': 1.0,
        area_col: 1.0,
        'x_4258': x4258,
        'y_4258': y4258,
    }
    return gpd.GeoDataFrame([data], geometry=[geom_3035], crs=CRS)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2.0)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0)**2
    c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
    return R * c

def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def ensure_area_nuts2(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure per-site 'area' and per-NUTS2 total 'area_nuts2' exist."""
    if df.empty:
        return df
    g = df.copy()
    if 'area' not in g.columns or g['area'].isna().any():
        # best-effort fallback; compute from geometry if possible, else set 1.0
        try:
            g['area'] = g.geometry.area
        except Exception:
            g['area'] = 1.0
        g['area'] = g['area'].fillna(1.0).replace(0, 1.0)
    if 'area_nuts2' not in g.columns:
        nuts2_sum = g.groupby('nuts2', as_index=False)['area'].sum().rename(columns={'area': 'area_nuts2'})
        g = g.merge(nuts2_sum, on='nuts2', how='left')
    return g

def extract_nuts2(code) -> str:
    s = str(code)
    if len(s) >= 4 and (s[3].isdigit() or s[2].isdigit()):
        return s[:4]
    return s[:2] + '00'

def filter_by_country(df: pd.DataFrame, iso2: str) -> pd.DataFrame:
    return df[df['nuts2'].str.startswith(iso2)].copy()

def select_industries(product_code, iso2, industries, commercial, product_mapping):
    if product_code == 'CO':
        df = filter_by_country(commercial, iso2)
        df.loc[:, 'eprtr_sectors'] = 'COMMERCIAL'
        return df
    sector = product_mapping.get(product_code)
    if sector:
        return industries[(industries['eprtr_sectors'] == sector) & (industries['nuts2'].str.startswith(iso2))].copy()
    return filter_by_country(industries, iso2)

# ---------------------------
# IO paths
# ---------------------------

output_dir = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industrial_OD/country_to_country")
industries_file = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/land_use/processed_sites/industries/industries_europe_NUTS2.parquet")
commercial_file = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/land_use/processed_sites/commercial/commercial_europe_NUTS2.parquet")
countries_path = r"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/admin/ne_10m/ne_10m_admin_0_countries.shp"
flow_xlsx = Path(r"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/NUTS_3_flows_surender/Step2_Deriving SUT using revised mapping/nuts2_eu_global_trade_v2.xlsx")
flow_sheet = "Sheet1"

# ---------------------------
# Read inputs
# ---------------------------

# HAving some issues with reading the industries and commercial files directly as parquet, so reading in a robust way
print(f"Reading industries file: {industries_file}")
industries = read_gdf_auto(industries_file)
print(f"Reading commercial file: {commercial_file}")
commercial = read_gdf_auto(commercial_file)

# Ensure the CRS is consistent
industries = industries.set_crs(CRS, allow_override=True) if industries.crs is None else industries.to_crs(CRS)
commercial = commercial.set_crs(CRS, allow_override=True) if commercial.crs is None else commercial.to_crs(CRS)

# Ensure 4258 coordinates for distance (commercial may lack them)
industries = ensure_xy_4258(industries)
commercial = ensure_xy_4258(commercial)

# Calculate total area per nuts2 region for industries
nuts2_areas = industries.groupby('nuts2')['area'].sum().rename('area_nuts2').reset_index()
industries = industries.merge(nuts2_areas, on='nuts2', how='left')

# Product code -> sector mapping
product_mapping = {
    "IL": "INTENSIVE LIVESTOCK",
    "MI": "MINERALS",
    "FB": "FOOD AND BEVERAGE",
    "CH": "CHEMICALS",
    "PW": "PAPER AND WOOD",
    "MT": "METALS",
    "CO": "COMMERCIAL",
}

# Country list for iteration (ISO2)
country_codes = [
    'AT','BE','BG','CZ','CY','DK','DE','EL',
    'EE','ES','FI','FR','GR','HR','HU',
    'IE','IT','LT','LU','LV','MT','NL',
    'NO','PL','PT','RO','SI','SK','SE',
    'US','CH','RU','TR','CA','MX','AR',
    'BR','ZA','AU','SA','ID','CN','IN',
    'JP','KR'
]

# Parse job args
try:
    job_num = int(sys.argv[1])
    total_num = int(sys.argv[2])
except (IndexError, ValueError):
    print(f"Usage: python {Path(__file__).name} <start_num> <stride>")
    sys.exit(1)

# Vectorized indexing into 4D space: sector_o x sector_d x country_o x country_d
vector_indices = np.arange(
    job_num,
    len(country_codes) * len(country_codes) * len(product_mapping) * len(product_mapping),
    total_num
)
indices_sectors_destinations = np.floor(vector_indices / (len(country_codes) * len(country_codes) * len(product_mapping)))
indices_sectors_origins = np.floor(
    (vector_indices - indices_sectors_destinations * (len(country_codes) * len(country_codes) * len(product_mapping)))
    / (len(country_codes) * len(country_codes))
)
indices_destinations = np.floor(
    (vector_indices
     - indices_sectors_destinations * (len(country_codes) * len(country_codes) * len(product_mapping))
     - indices_sectors_origins * (len(country_codes) * len(country_codes)))
    / len(country_codes)
)
indices_origins = np.floor(
    (vector_indices
     - indices_sectors_destinations * (len(country_codes) * len(country_codes) * len(product_mapping))
     - indices_sectors_origins * (len(country_codes) * len(country_codes))
     - indices_destinations * len(country_codes))
)

# Cast to int
indices_sectors_origins = indices_sectors_origins.astype(int)
indices_sectors_destinations = indices_sectors_destinations.astype(int)
indices_origins = indices_origins.astype(int)
indices_destinations = indices_destinations.astype(int)

# Filtered product codes and country codes
product_mapping_items = list(product_mapping.items())
filtered_product_mapping_origins = [product_mapping_items[i][0] for i in indices_sectors_origins]
filtered_product_mapping_destinations = [product_mapping_items[i][0] for i in indices_sectors_destinations]
filtered_country_codes_origins = [country_codes[i] for i in indices_origins]
filtered_country_codes_destinations = [country_codes[i] for i in indices_destinations]

# Load flow matrix and normalize labels
flow_matrix = pd.read_excel(flow_xlsx, sheet_name=flow_sheet, index_col=0, header=0)
flow_matrix.index = flow_matrix.index.astype(str)
flow_matrix.columns = flow_matrix.columns.astype(str)

# Load countries (for centroids)
countries_gdf = gpd.read_file(countries_path)

# One-time normalization for strings and missing values
industries['nuts2'] = industries['nuts2'].astype(str).str.strip()
commercial['nuts2'] = commercial['nuts2'].astype(str).str.strip()
industries = industries.dropna(subset=['nuts2', 'eprtr_sectors'])
commercial = commercial.dropna(subset=['nuts2'])

# Assign unique IDs to commercial to avoid collision with industries
max_industrial_id = int(industries['id'].max()) if len(industries) else 0
commercial['id'] = range(max_industrial_id + 1, max_industrial_id + 1 + len(commercial))
commercial['eprtr_sectors'] = 'COMMERCIAL'

# ---------------------------
# Main loop
# ---------------------------

for idx, _ in enumerate(vector_indices):
    product_code_origins = filtered_product_mapping_origins[idx]
    product_code_destinations = filtered_product_mapping_destinations[idx]
    origin_country = filtered_country_codes_origins[idx]
    destination_country = filtered_country_codes_destinations[idx]

    # Output dir/file
    output_dir_commodity = output_dir / f"{product_code_origins}_{product_code_destinations}"
    output_dir_commodity.mkdir(parents=True, exist_ok=True)
    output_file = output_dir_commodity / f"flows_{product_code_origins}_{product_code_destinations}_{origin_country}_{destination_country}.parquet"
    # if output_file.exists():
    #     print(f"File exists, skipping: {output_file}", flush=True)
    #     continue

    # Select candidate industries
    filtered_industries_origins = select_industries(product_code_origins, origin_country, industries, commercial, product_mapping)
    filtered_industries_destinations = select_industries(product_code_destinations, destination_country, industries, commercial, product_mapping)

    # Ensure dummies if empty
    if filtered_industries_origins.empty:
        print(f"No industries for {product_code_origins} in {origin_country}. Creating dummy.", flush=True)
        filtered_industries_origins = create_dummy_industry(origin_country + '00', product_mapping.get(product_code_origins), countries_gdf, area_col='area_nuts2')
    if filtered_industries_destinations.empty:
        print(f"No industries for {product_code_destinations} in {destination_country}. Creating dummy.", flush=True)
        filtered_industries_destinations = create_dummy_industry(destination_country + '00', product_mapping.get(product_code_destinations), countries_gdf, area_col='area_nuts2')

    # Subset flow matrix for this origin/destination
    row_filter = (flow_matrix.index.str[:2] == origin_country) & (flow_matrix.index.str[-2:] == product_code_origins)
    column_filter = (flow_matrix.columns.str[:2] == destination_country) & (flow_matrix.columns.str[-2:] == product_code_destinations)
    flow_subset = flow_matrix.loc[row_filter, column_filter]

    if flow_subset.empty:
        print(f"No flows for {product_code_origins}->{product_code_destinations} from {origin_country} to {destination_country}. Skipping.", flush=True)
        continue

    # Prepare area cols and groups
    filtered_industries_origins = ensure_area_nuts2(filtered_industries_origins)
    filtered_industries_destinations = ensure_area_nuts2(filtered_industries_destinations)
    origins_by_nuts2 = {k: v for k, v in filtered_industries_origins.groupby('nuts2')}
    dests_by_nuts2 = {k: v for k, v in filtered_industries_destinations.groupby('nuts2')}

    updates = {}

    # Iterate submatrix and compute allocations
    for o_idx, row in flow_subset.iterrows():
        nuts2_o = extract_nuts2(o_idx)
        origin_industries = origins_by_nuts2.get(nuts2_o)
        if origin_industries is None or origin_industries.empty:
            continue

        for d_idx, flow_value in row.items():
            if not flow_value or pd.isna(flow_value):
                continue
            nuts2_d = extract_nuts2(d_idx)
            destination_industries = dests_by_nuts2.get(nuts2_d)
            if destination_industries is None or destination_industries.empty:
                continue

            origin_ind = origin_industries[origin_industries['area_nuts2'] != 0]
            dest_ind = destination_industries[destination_industries['area_nuts2'] != 0]
            if origin_ind.empty or dest_ind.empty:
                continue

            for _, origin in origin_ind.iterrows():
                for _, destination in dest_ind.iterrows():
                    key = (
                        origin['id'], destination['id'],
                        origin['nuts2'], destination['nuts2'],
                        origin['eprtr_sectors'], destination['eprtr_sectors']
                    )

                    if nuts2_o == nuts2_d:
                        # Intra-NUTS2: avoid self-pairs and adjust denominator
                        if origin['id'] == destination['id']:
                            continue
                        denom_o = origin['area_nuts2']
                        denom_d = destination['area_nuts2']
                        if denom_o == 0 or denom_d == 0:
                            continue
                        if destination['area_nuts2'] == destination['area']:
                            value = flow_value * (origin['area'] / denom_o)
                        else:
                            if (destination['area_nuts2'] - destination['area']) == 0:
                                continue
                            value = flow_value * (origin['area'] / denom_o) * (destination['area'] / (denom_d - destination['area']))
                        if origin['area_nuts2'] == origin['area']:
                            value = flow_value * (destination['area'] / denom_d)
                        else:
                            if (origin['area_nuts2'] - origin['area']) == 0:
                                continue
                            value = flow_value * (destination['area'] / denom_d) * (origin['area'] / (denom_o - origin['area']))
                    else:
                        value = (
                            flow_value *
                            (origin['area'] / origin['area_nuts2']) *
                            (destination['area'] / destination['area_nuts2'])
                        )

                    if isinstance(value, (int, float)):
                        updates[key] = updates.get(key, 0.0) + float(value)

    # Write out updates for this pair
    if not updates:
        print(f"No valid updates for {product_code_origins}->{product_code_destinations} {origin_country}->{destination_country}.", flush=True)
        continue

    updates_df = pd.DataFrame.from_dict(updates, orient='index', columns=['value'])
    level_names = ['origin_id', 'destination_id', 'origin_nuts2', 'destination_nuts2', 'origin_sector', 'destination_sector']
    updates_df.index = pd.MultiIndex.from_tuples(updates_df.index, names=level_names[:len(updates_df.index[0])])
    updates_df = updates_df.reset_index()

    # Distances (vectorized if coords available)
    if {'x_4258','y_4258'}.issubset(filtered_industries_origins.columns) and {'x_4258','y_4258'}.issubset(filtered_industries_destinations.columns):
        o_coords = filtered_industries_origins[['id','y_4258','x_4258']].drop_duplicates().rename(columns={'y_4258':'orig_lat','x_4258':'orig_lon'})
        d_coords = filtered_industries_destinations[['id','y_4258','x_4258']].drop_duplicates().rename(columns={'y_4258':'dest_lat','x_4258':'dest_lon'})
        updates_df = updates_df.merge(o_coords, left_on='origin_id', right_on='id', how='left').drop(columns=['id'])
        updates_df = updates_df.merge(d_coords, left_on='destination_id', right_on='id', how='left').drop(columns=['id'])
        mask_coords = updates_df[['orig_lat','orig_lon','dest_lat','dest_lon']].notna().all(axis=1)
        updates_df['distance'] = 10**6
        updates_df.loc[mask_coords, 'distance'] = haversine_vec(
            updates_df.loc[mask_coords, 'orig_lat'].values,
            updates_df.loc[mask_coords, 'orig_lon'].values,
            updates_df.loc[mask_coords, 'dest_lat'].values,
            updates_df.loc[mask_coords, 'dest_lon'].values,
        )
    else:
        updates_df['distance'] = 10**6

    # Rescale to preserve total flow, drop tiny flows
    total_flow = updates_df['value'].sum()
    filtered_updates = updates_df[updates_df['value'] >= 0.001].copy()
    considered_flow = filtered_updates['value'].sum()
    if considered_flow > 0:
        scale = total_flow / considered_flow
        filtered_updates['value'] *= scale
    updates_df = filtered_updates.dropna().reset_index(drop=True)

    if updates_df.empty:
        print(f"All data filtered out for {product_code_origins}->{product_code_destinations} {origin_country}->{destination_country}.", flush=True)
        continue

    updates_df.to_parquet(output_file, index=False)
    print(f"Saved flows: {output_file} (min={updates_df['value'].min():.6f}, n={len(updates_df)})", flush=True)