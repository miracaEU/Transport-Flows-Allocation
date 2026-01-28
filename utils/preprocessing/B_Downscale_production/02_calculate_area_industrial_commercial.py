import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path

#European Industrial and Commercial Area Analysis
#-------------------------------
#Processes industrial anc commercial location data across Europe by:
#1. Loading industrial, commercial points, NUTS2 regions, and land use polygons
#2. Deduplicating nearby points
#3. Distributing areas from polygon sources (industrial, farmland, mining and commercial)
#4. Aggregating results by NUTS2 regions and industry types


# Constants
CRS = "EPSG:3035"
OUTPUT_BASE = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/land_use/processed_sites/industries")
OUTPUT_COUNTRIES = OUTPUT_BASE / "countries" / "NUTS2"
OUTPUT_COUNTRIES.mkdir(parents=True, exist_ok=True)

# Helper functions
def safe_read_gdf(path, crs=CRS):
    gdf = gpd.read_parquet(path) if str(path).lower().endswith(".parquet") else gpd.read_file(path)
    return gdf.to_crs(crs)

def deduplicate_points_by_buffer(gdf, buffer_dist=5):
    """Return GeoDataFrame with one representative per cluster of points within buffer_dist."""
    if gdf.empty:
        return gdf.copy()
    bufs = gdf.geometry.buffer(buffer_dist)
    dissolved = bufs.unary_union
    unique_rows = []
    geoms = dissolved.geoms if hasattr(dissolved, "geoms") else [dissolved]
    for geom in geoms:
        matches = gdf[gdf.geometry.intersects(geom)]
        if not matches.empty:
            unique_rows.append(matches.iloc[0])
    if not unique_rows:
        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
    return gpd.GeoDataFrame(unique_rows, columns=gdf.columns).set_crs(gdf.crs)

def distribute_area(sources, targets, country_industries, group_col='eea_activities', buffer=100, drop_others=False):
    """Distribute polygon 'area' from sources to point targets within buffer.
       If drop_others=True, remove duplicates in country_industries not selected."""
    if sources.empty or targets.empty:
        return
    for _, src in sources.iterrows():
        buf = src.geometry.buffer(buffer)
        matches = targets[targets.geometry.intersects(buf)]
        if matches.empty:
            continue
        matches = matches.reset_index()  # preserve original indices in 'index'
        # choose one per activity
        unique = matches.groupby(group_col).first()
        indices_to_keep = unique['index']
        if drop_others:
            to_remove = matches.loc[~matches['index'].isin(indices_to_keep), 'index']
            to_remove = to_remove[to_remove.isin(country_industries.index)]
            if not to_remove.empty:
                country_industries.drop(index=to_remove, inplace=True, errors='ignore')
        selected = matches.set_index('index').loc[indices_to_keep]
        n = len(selected)
        shared_area = src.get('area', 0) / n if n > 0 else 0
        for idx in selected.index:
            if idx in country_industries.index:
                country_industries.at[idx, 'area'] = country_industries.at[idx, 'area'] + shared_area

# Load datasets
industries = safe_read_gdf("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/buildings/european_industries_location.parquet")
nuts2 = safe_read_gdf("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/admin/NUTS/NUTS2_v2016.parquet")
countries = gpd.read_file(r"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/admin/ne_10m/ne_10m_admin_0_countries.shp").to_crs(CRS)

# Ensure point geometries
def ensure_point_geometry(gdf):
    def extract_first_point(geom):
        if geom is None:
            return None
        if geom.geom_type == 'MultiPoint':
            return geom.geoms[0]
        if geom.geom_type == 'Point':
            return geom
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")
    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.apply(extract_first_point)
    if not all(gdf.geometry.geom_type == 'Point'):
        raise ValueError("Not all geometries are Point geometries after conversion.")
    return gdf

industries = ensure_point_geometry(industries)

# Spatial join to assign NUTS2
industries = gpd.sjoin(industries, nuts2[['geometry', 'NUTS_ID']], how='left', predicate='intersects', lsuffix='ind', rsuffix='nuts2')
industries = industries.reset_index(drop=True).rename(columns={'NUTS_ID': 'nuts2'})

# Prepare country lists (kept as in original)
country_list = ['austria','belgium','bulgaria','czechia','cyprus','denmark','germany',
                'estonia','spain','findland','france','greece','croatia','hungary',
                'ireland','italy','latvia','luxembourg','lithuania','malta','netherlands',
                'norway','poland','portugal','romania','slovenia','slovakia','sweden']
CountryCode_list = ['AT','BE','BG','CZ','CY','DK','DE',
                    'EE','ES','FI','FR','GR','HR','HU',
                    'IE','IT','LT','LU','LV','MT','NL',
                    'NO','PL','PT','RO','SI','SK','SE']
ISO3_list=['AUT','BEL','BGR','CZE','CYP','DNK','DEU',
           'EST','ESP','FIN','FRA','GRC','HRV','HUN',
           'IRL','ITA','LTU','LUX','LVA','MLT','NLD',
           'NOR','POL','PRT','ROU','SVN','SVK','SWE']

def main():
    # Iterate countries
    for country_name, iso3, country_code in zip(country_list, ISO3_list, CountryCode_list):
        print(f"Processing industries for {country_name} (ISO3: {iso3})")
        country_industries = industries[industries['countryCode'] == country_code].copy()
        print(f"Initial industries: {len(country_industries)}")
        if country_industries.empty:
            print(f"No industries for {country_name}, skipping.")
            continue

        # Deduplicate by proximity
        country_industries = deduplicate_points_by_buffer(country_industries, buffer_dist=5)
        country_industries = country_industries.reset_index(drop=True)
        print(f"After deduplication: {len(country_industries)}")

        # Create categories
        agricultural_industries = country_industries[(country_industries['eprtr_sectors'] == 'INTENSIVE LIVESTOCK')].copy()
        mining_industries = country_industries[(country_industries['eea_activities'] == 'Extractive industry')].copy()
        other_industries = country_industries[(country_industries['eprtr_sectors'] != 'INTENSIVE LIVESTOCK') & (country_industries['eea_activities'] != 'Extractive industry')].copy()

        print(f"Counts: agricultural={len(agricultural_industries)}, mining={len(mining_industries)}, other={len(other_industries)}")

        # Country boundary for naming files / lookups
        country_row = countries[countries['ISO_A3_EH'] == iso3]
        if country_row.empty:
            print(f"No country boundary found for {country_name} ({iso3}). Skipping.")
            continue

        # Load and prepare polygon sources (mining, farmland, industrial)
        mining_gpkg = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/land_use/mining/global_mining_polygons_v2.gpkg")
        # mining is global; ensure CRS already set above
        mining_gdf = gpd.read_file(mining_gpkg).to_crs(CRS)
        mining_gdf = mining_gdf[mining_gdf.geometry.is_valid]
        mining_gdf['area'] = mining_gdf.geometry.area

        farmland_path = Path(f"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/land_use/farmland/{country_row['SOVEREIGNT'].iloc[0].lower()}-farmland.gpkg")
        if not farmland_path.exists():
            print(f"Farmland file not found for {country_name}; skipping farmland/mining distribution.")
            farmland_gdf = gpd.GeoDataFrame(columns=['geometry'], crs=CRS)
        else:
            farmland_gdf = gpd.read_file(farmland_path).to_crs(CRS)
            farmland_gdf = farmland_gdf[farmland_gdf.geometry.is_valid]
            farmland_gdf['area'] = farmland_gdf.geometry.area

        industrial_path = Path(f"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/land_use/industrial/{country_row['SOVEREIGNT'].iloc[0].lower()}-industrial.gpkg")
        if not industrial_path.exists():
            print(f"Industrial landuse file not found for {country_name}; skipping industrial distribution.")
            industrial_gdf = gpd.GeoDataFrame(columns=['geometry'], crs=CRS)
        else:
            industrial_gdf = gpd.read_file(industrial_path).to_crs(CRS)
            industrial_gdf = industrial_gdf[industrial_gdf.geometry.is_valid]
            industrial_gdf['area'] = industrial_gdf.geometry.area

        # Ensure 'area' column exists on country_industries
        if 'area' not in country_industries.columns:
            country_industries['area'] = 0.0

        # Distribute areas:
        # - other_industries <- industrial_gdf (do not drop others)
        distribute_area(industrial_gdf, other_industries, country_industries, drop_others=False)
        # - agricultural <- farmland_gdf (drop others)
        distribute_area(farmland_gdf, agricultural_industries, country_industries, drop_others=True)
        # - mining <- mining_gdf (drop others)
        distribute_area(mining_gdf, mining_industries, country_industries, drop_others=True)

        # Filter and aggregate
        industries_with_large_area = country_industries[country_industries['area'] > 1]
        sector_area_sums = country_industries.groupby(['nuts2', 'eprtr_sectors'])['area'].sum().rename('area_nuts2')
        country_industries = country_industries.merge(sector_area_sums.reset_index(), on=['nuts2', 'eprtr_sectors'], how='left')

        # Normalize column name
        if 'area_nuts2_new' in country_industries.columns and 'area_nuts2' not in country_industries.columns:
            country_industries = country_industries.rename(columns={'area_nuts2_new': 'area_nuts2'})

        # Save per-country parquet
        out_path = OUTPUT_COUNTRIES / f"industries_NUTS2_{country_name.lower()}.parquet"
        country_industries.to_parquet(out_path, index=False)
        print(f"Saved processed industries for {country_name} -> {out_path}")

    # Merge all per-country outputs into one GeoDataFrame
    gdfs = []
    for file in OUTPUT_COUNTRIES.glob("industries_NUTS2*.parquet"):
        print(f"Reading file: {file.name}")
        gdf = gpd.read_parquet(file)
        if not gdf.empty:
            gdfs.append(gdf)
    if gdfs:
        industries_merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        industries_merged = industries_merged.set_crs(CRS)
        merged_out = OUTPUT_BASE / "industries_europe_NUTS2.parquet"
        industries_merged.to_parquet(merged_out, index=False)
        print(f"Merged industries saved to {merged_out}")
    else:
        print("No per-country industry files found to merge.")

    # -----------------------------
    # Commercial processing (refactored: consistent CRS, centroid, correct save)
    commercial_folder = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/land_use/commercial")
    nuts2 = nuts2.to_crs(CRS)  # already loaded above
    all_features = []

    for file in commercial_folder.glob("*.gpkg"):
        print(f"Processing commercial file: {file.name}")
        gdf = gpd.read_file(file).to_crs(CRS)
        gdf['area'] = gdf.geometry.area
        gdf = gpd.sjoin(gdf, nuts2[['geometry', 'NUTS_ID']], how='left', predicate='intersects')
        gdf = gdf.rename(columns={'NUTS_ID': 'nuts2'})
        total_area_per_nuts2 = gdf.groupby('nuts2')['area'].sum().reset_index().rename(columns={'area': 'area_nuts2'})
        gdf = gdf.merge(total_area_per_nuts2, on='nuts2', how='left')
        gdf['geometry'] = gdf.geometry.centroid
        all_features.append(gdf[['geometry', 'area', 'nuts2', 'area_nuts2']])

    if all_features:
        all_features_gdf = gpd.GeoDataFrame(pd.concat(all_features, ignore_index=True), crs=CRS)
        commercial_out = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/land_use/processed_sites/commercial/commercial_europe_NUTS2.parquet")
        all_features_gdf.to_parquet(commercial_out, index=False)
        print(f"Commercial processed and saved to {commercial_out}")
    else:
        print("No commercial features processed.")

if __name__ == "__main__":
    main()