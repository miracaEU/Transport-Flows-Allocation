# This script processes and matches OpenStreetMap (OSM) transport infrastructure data with TEN-T (Trans-European Transport Network) corridors.
# It assigns TEN-T corridor attributes to OSM elements based on spatial proximity and name similarity.
# The processed data is saved as GeoParquet and GeoPackage files.

import pandas as pd
import geopandas as gpd
from shapely import wkb
from shapely.geometry import Polygon, MultiPolygon
from difflib import SequenceMatcher
from rapidfuzz import fuzz
from shapely.wkb import loads as load_wkb
from shapely.wkt import loads as load_wkt

CRS_WGS84 = "EPSG:4326"   # Latitude/Longitude
CRS_WGS84_IWW = "EPSG:3857"   # Latitude/Longitude
CRS_UTM = "EPSG:3034"    # UTM Zone 32N ETRS89

#we first load each tansport infrastructure layer
# Load the OSM parquet files
file_paths_OSM = {
    "road_edges": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/roads/europe_road_edges.parquet",
    "road_nodes": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/roads/europe_road_nodes.parquet",
    "railway_edges": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/railways/europe_railways_edges.parquet",
    "railway_nodes": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/railways/europe_railways_nodes.parquet",
    "railway_stations": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/railways/europe_train_stations.parquet",
    "ports": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/ports/europe_ports.parquet",
    "port_terminals": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/ports/europe_port_terminals.parquet",
    "iww_nodes": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/iww/europe_IWW_nodes.parquet",
    "iww_edges": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/iww/europe_IWW_edges.parquet",
    "airports": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/airports/europe_airports.parquet",
    "intermodal": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/intermodal/europe_intermodal_terminals.parquet",
}


#now the TENT corridors
file_paths_TENT = {
    "road": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/roads_GL2024_ETCs.parquet",
    "railways": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/railways_GL2024_ETCs.parquet",
    "ports": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/ports_GL2024_ETCs.parquet",
    "airports": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/airports_GL2024_ETCs.parquet",
    "iww": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/iww_GL2024_ETCs.parquet",
    "urban_nodes": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/urban_nodes_GL2024_ETCs.parquet",
}

# Define the paths where we need to save the data
file_paths_save_OSM = {
    "road_edges": "/soge-home/projects/mistral/miraca/processed_data/europe_road_edges_TENT.parquet",
    "road_nodes": "/soge-home/projects/mistral/miraca/processed_data/europe_road_nodes_TENT.parquet",
    "railway_edges": "/soge-home/projects/mistral/miraca/processed_data/europe_railways_edges_TENT.parquet",
    "railway_nodes": "/soge-home/projects/mistral/miraca/processed_data/europe_railways_nodes_TENT.parquet",
    "railway_stations": "/soge-home/projects/mistral/miraca/processed_data/europe_railways_stations_TENT.parquet",
    "ports": "/soge-home/projects/mistral/miraca/processed_data/europe_ports_TENT.parquet",
    "port_areas": "/soge-home/projects/mistral/miraca/processed_data/europe_port_areas_TENT.gpkg",
    "iww_nodes": "/soge-home/projects/mistral/miraca/processed_data/europe_IWW_nodes_TENT.parquet",
    "iww_edges": "/soge-home/projects/mistral/miraca/processed_data/europe_IWW_edges_TENT.parquet",
    "airports": "/soge-home/projects/mistral/miraca/processed_data/europe_airports_TENT.parquet",
    "airport_areas": "/soge-home/projects/mistral/miraca/processed_data/europe_airport_areas_TENT.parquet",
    "intermodal": "/soge-home/projects/mistral/miraca/processed_data/europe_intermodal_terminals_TENT.parquet",
}

#now the TENT corridors
file_paths_save_TENT = {
    "road": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/roads_GL2024_ETCs_2.parquet",
    "railways": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/railways_GL2024_ETCs_2.parquet",
    "ports": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/ports_GL2024_ETCs_2.parquet",
    "airports": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/airports_GL2024_ETCs_2.parquet",
    "iww": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/iww_GL2024_ETCs_2.parquet",
    "urban_nodes": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/TENT_data_input/urban_nodes_GL2024_ETCs_2.parquet",
}

def load_and_reproject(df, crs_original, crs_target):
    # Check if 'geometry' column exists
    if 'geometry' not in df.columns:
        raise ValueError("DataFrame does not contain a 'geometry' column.")

    # Convert geometry if needed
    if isinstance(df['geometry'].iloc[0], bytes):  # Likely WKB format
        df['geometry'] = df['geometry'].apply(load_wkb)
    elif isinstance(df['geometry'].iloc[0], str):  # Likely WKT format
        df['geometry'] = df['geometry'].apply(load_wkt)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs_original)
    return gdf.to_crs(crs_target)
    
def assign_TENT_corridor_by_buffer(els_OSM, els_TENT, buff):
    # Save original projection
    OSM_pr = els_OSM.crs
    TENT_pr = els_TENT.crs

    # Convert WKB to Shapely geometries if necessary
    for df in [els_OSM, els_TENT]:
        df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x)

    # Ensure GeoDataFrame format
    els_OSM = gpd.GeoDataFrame(els_OSM, geometry='geometry', crs=OSM_pr)
    els_TENT = gpd.GeoDataFrame(els_TENT, geometry='geometry', crs=TENT_pr)

    # Reproject to EPSG:3034
    CRS_UTM = "EPSG:3034"
    els_OSM = els_OSM.to_crs(CRS_UTM)
    els_TENT = els_TENT.to_crs(CRS_UTM)

    # Explode MultiLineStrings, but only if they exist (avoid exploding points)
    if 'MultiLineString' in els_OSM.geom_type.unique():
        els_OSM = els_OSM.explode(index_parts=True).reset_index(drop=True)  # Keep track of original MultiLineString

    # Function to apply spatial join for attributes
    def assign_attribute(attr_name):
        nonlocal els_OSM  # Ensure changes persist in main function
        # Ensure attr_name exists before join
        if attr_name not in els_TENT.columns:
            return
        
        # Create buffer
        buffered_TENT = els_TENT.copy()
        buffered_TENT['buffer'] = buffered_TENT['geometry'].buffer(buff)
        buffered_TENT = buffered_TENT[[attr_name, 'buffer']].dropna(subset=[attr_name])

        # Perform spatial join (find overlapping segments)
        nodes_within_buffer = gpd.sjoin(els_OSM, buffered_TENT.set_geometry('buffer'), how='left', predicate='intersects')

        # Assign attr_name based on proportion of overlap
        if attr_name in nodes_within_buffer.columns:
            if 'Point' in els_OSM.geom_type.unique():
                # Assign directly for points
                els_OSM.loc[nodes_within_buffer.index, attr_name] = nodes_within_buffer[attr_name].astype(str)
            else:
                # Calculate proportion of each segment within a corridor
                nodes_within_buffer['overlap_length'] = nodes_within_buffer.geometry.length

                # Assign the corridor(s) with the largest overlap
                def assign_most_overlap(group):
                    if group.empty:
                        return None  # No corridor found

                    sorted_corridors = group.sort_values(by='overlap_length', ascending=False)

                    # Return only the corridor with the highest overlap
                    return str(sorted_corridors.iloc[0][attr_name])  # Select the top row

                corridor_assignments = nodes_within_buffer.groupby(level=0).apply(assign_most_overlap)
                els_OSM[attr_name] = els_OSM.index.map(corridor_assignments)#

        # Assign None to columns for rows where attr_name is 'NULL' or NaN or empty
        mask = els_OSM[attr_name].isna() | (els_OSM[attr_name] == 'nan') | (els_OSM[attr_name] == '')
        els_OSM.loc[mask, attr_name] = None

    # Assign attributes
    assign_attribute('CORRIDORS')
    assign_attribute('RAILWAYS_A')
    assign_attribute('GIS_STATUS')

    # Convert back to original CRS
    els_OSM = els_OSM.to_crs(OSM_pr)
    els_OSM=gpd.GeoDataFrame(els_OSM, geometry='geometry', crs=OSM_pr)

     # Compute 'osm_match' variable
    def compute_osm_match():
        buffered_TENT = els_TENT.copy()
        buffered_TENT['buffer'] = buffered_TENT['geometry'].buffer(buff)

        # Spatial join to check if any els_OSM is within the buffer
        matches = gpd.sjoin(buffered_TENT.set_geometry('buffer'), els_OSM, how='left', predicate='intersects')

        # Assign osm_match = 1 if there is at least one match, otherwise 0
        match_flag = matches.groupby(matches.index).size() > 0
        els_TENT['osm_match'] = match_flag.astype(int)

    compute_osm_match()

    return els_TENT, els_OSM

def assign_TENT_corridor_by_name(els_OSM, els_TENT, tag_OSM, tag_TENT, sim_thr):

    # Save original projection
    OSM_pr = els_OSM.crs
    corridor_map = {}

    # Convert WKB to Shapely geometries if necessary
    els_OSM['geometry'] = els_OSM['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x)

    # Ensure name columns do not contain NaN values
    els_OSM[tag_OSM] = els_OSM[tag_OSM].fillna("").astype(str)
    els_TENT[tag_TENT] = els_TENT[tag_TENT].fillna("").astype(str)

    # Initialize osm_match dictionary
    osm_match_map = {idx: 0 for idx in els_TENT.index}

    # Pre-filter empty names
    els_OSM_filtered = els_OSM[els_OSM[tag_OSM] != ""]
    els_TENT_filtered = els_TENT[els_TENT[tag_TENT] != ""]
    
    # Vectorized approach: build a list for rapid matching
    osm_names = els_OSM_filtered[tag_OSM].tolist()
    osm_indices = els_OSM_filtered.index.tolist()
    
    # Iterate over TENT corridors once
    for tent_idx, tent_row in els_TENT_filtered.iterrows():
        tent_name = tent_row[tag_TENT]
        corridor = tent_row['CORRIDORS']
        match_found = False

        # Vectorized similarity calculation using rapidfuzz
        for i, osm_name in enumerate(osm_names):
            if osm_name in tent_name:
                corridor_map[osm_indices[i]] = corridor
                match_found = True
            elif fuzz.ratio(osm_name, tent_name) > sim_thr * 100:  # rapidfuzz returns 0-100
                corridor_map[osm_indices[i]] = corridor
                match_found = True

        # Assign osm_match = 1 if a match was found for this TENT corridor
        if match_found:
            osm_match_map[tent_idx] = 1

    # Assign corridor values to OSM
    els_OSM.loc[list(corridor_map.keys()), 'CORRIDORS'] = list(map(str, corridor_map.values()))

    # Assign None to 'CORRIDORS' for rows where tag_OSM is empty
    els_OSM.loc[els_OSM[tag_OSM].isna() | (els_OSM[tag_OSM] == 'nan') | (els_OSM[tag_OSM] == ''), 'CORRIDORS'] = None

    # Assign osm_match to TENT
    els_TENT['osm_match'] = els_TENT.index.map(osm_match_map)

    # Convert back to original CRS
    els_OSM = els_OSM.to_crs(OSM_pr)
    els_OSM = gpd.GeoDataFrame(els_OSM, geometry='geometry', crs=OSM_pr)

    return els_TENT, els_OSM 

def filter_roads_tent(road_nodes, road_edges):
    # Create copies to avoid modifying original data
    road_edges_filtered = road_edges.copy()
    road_nodes_filtered = road_nodes.copy()
    # Assign None to 'CORRIDORS' where the road type is not motorway or trunk
    road_edges_filtered.loc[~road_edges_filtered['tag_highway'].isin(['motorway', 'trunk', 'primary']), 'CORRIDORS'] = None
    # Identify nodes connected to non-motorway/non-trunk edges
    non_corridor_nodes = set(road_edges_filtered.loc[road_edges_filtered['CORRIDORS'].isna(), ['from_id', 'to_id']].values.flatten())
    # Assign None to 'CORRIDORS' for nodes that are only connected to non-motorway/non-trunk roads
    road_nodes_filtered.loc[road_nodes_filtered['id'].isin(non_corridor_nodes), 'CORRIDORS'] = None
    
    return road_nodes_filtered, road_edges_filtered

# Function to calculate similarity
def is_similar(str1, str2, threshold):
    if pd.isna(str1) or pd.isna(str2):
        return False
    return SequenceMatcher(None, str1, str2).ratio() >= threshold

# filter out similar elements
def drop_similar(els_1, els_2, els_1_col, els_2_col, threshold):
    els_2_to_keep = els_2.copy()
    for port_idx, port_row in els_2.iterrows():
        port_name = port_row[els_2_col]
        # Check if there's a similar name in els_1
        if any(is_similar(port_name, terminal_name, threshold) for terminal_name in els_1[els_1_col]):
            els_2_to_keep = els_2_to_keep.drop(port_idx)
    return els_2_to_keep

# Function to merge ports and terminals, prioritizing terminals
def merge_ports_terminals(ports, terminals):
    # Convert terminals to points using their centroid
    terminals = terminals.copy()
    # terminals['geometry'] = terminals.geometry.centroid  # Replace geometry with centroid

    # fix the names and countries
    ports['port_name'] = ports['name'].apply(lambda x: x.split('_')[0] if pd.notnull(x) else None)
    ports['country'] = ports['name'].apply(lambda x: x.split('_')[1] if pd.notnull(x) and '_' in x else None)

    # Remove similar ports (assuming drop_similar function exists)
    filtered_ports = drop_similar(terminals, ports, els_1_col='port_name', els_2_col='name', threshold=0.8)

    # Ensure both have the same columns
    all_columns = set(filtered_ports.columns).union(set(terminals.columns))
    for col in all_columns:
        if col not in filtered_ports.columns:
            filtered_ports[col] = None
        if col not in terminals.columns:
            terminals[col] = None

    # Remove empty DataFrames before merging
    non_empty_frames = [df for df in [filtered_ports, terminals] if not df.empty]
    # Merge points and polygons into a single GeoDataFrame
    nodes = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else pd.DataFrame()

    # Assign unique IDs
    nodes['id'] = range(1, len(nodes) + 1)
    nodes['infra']='port'

    for idx, row in nodes.iterrows():
        if pd.isnull(row['iso3']):    # Check if iso3 is NULL
            country = row['country']  # Get the country value from the current row
            # Search for the row in els_OSM where 'Country' matches and get the iso3 value
            matching_row = nodes[nodes['country'] == country]
        
            if not matching_row.empty:  # If a matching row is found
                # Assign the iso3 value to the 'iso3' column
                nodes.at[idx, 'iso3'] = matching_row['iso3'].values[0]

    return nodes

def main():
    # Load the OSM layers and reproject to UTM
    dataframes_OSM = {}
    for key, path in file_paths_OSM.items():
        # Convert to GeoDataFrame and reproject to UTM directly
        if key in ["iww_ports", "iww_locks", "iww_channels"]:            
            dataframes_OSM[key] = load_and_reproject(pd.read_parquet(path), CRS_WGS84_IWW, CRS_UTM)
        else:
            dataframes_OSM[key] = load_and_reproject(pd.read_parquet(path), CRS_WGS84, CRS_UTM)

    # Load the TENT layers and ensure they are in UTM
    dataframes_TENT = {}
    for key, path in file_paths_TENT.items():
        dataframes_TENT[key] = load_and_reproject(pd.read_parquet(path), CRS_UTM, CRS_UTM)

    print('Data loaded', flush=True)

    # Perform the corridor assignment analysis (no changes here)
    #dataframes_TENT["road"], dataframes_OSM["road_edges"] = assign_TENT_corridor_by_buffer(dataframes_OSM["road_edges"], dataframes_TENT["road"], buff=200)
    #_, dataframes_OSM["road_nodes"] = assign_TENT_corridor_by_buffer(dataframes_OSM["road_nodes"], dataframes_TENT["road"], buff=200)
    #print('Road assigned')
    #dataframes_TENT["railways"], dataframes_OSM["railway_edges"] = assign_TENT_corridor_by_buffer(dataframes_OSM["railway_edges"], dataframes_TENT["railways"], buff=1000)
    #_, dataframes_OSM["railway_nodes"] = assign_TENT_corridor_by_buffer(dataframes_OSM["railway_nodes"], dataframes_TENT["railways"], buff=1000)
    #_, dataframes_OSM["railway_stations"] = assign_TENT_corridor_by_buffer(dataframes_OSM["railway_stations"], dataframes_TENT["railways"], buff=1000)
    #print('Rail assigned')
    dataframes_TENT["iww"], dataframes_OSM["iww_edges"] = assign_TENT_corridor_by_buffer(dataframes_OSM["iww_edges"], dataframes_TENT["iww"], buff=1000)
    _, dataframes_OSM["iww_nodes"] = assign_TENT_corridor_by_buffer(dataframes_OSM["iww_nodes"], dataframes_TENT["iww"], buff=1000)
    print('IWW assigned', flush=True)
    _, dataframes_OSM["intermodal"] = assign_TENT_corridor_by_buffer(dataframes_OSM["intermodal"], dataframes_TENT["iww"], buff=10000)
    _, dataframes_OSM["intermodal"] = assign_TENT_corridor_by_buffer(dataframes_OSM["intermodal"], dataframes_TENT["road"], buff=10000)
    print('Intermodal assigned',   flush=True)
    dataframes_TENT["ports"], aux_ports = assign_TENT_corridor_by_buffer(dataframes_OSM["ports"], dataframes_TENT["ports"], buff=10000)
    _, aux_terminals = assign_TENT_corridor_by_name(dataframes_OSM["port_terminals"], dataframes_TENT["ports"], 'port_name', 'DESCRIPTIO', 0.8)
    dataframes_TENT["airports"], dataframes_OSM["airports"] = assign_TENT_corridor_by_name(dataframes_OSM["airports"], dataframes_TENT["airports"], 'icao', 'ICAO_CODE', 0.8)
    dataframes_OSM["ports"] = merge_ports_terminals(aux_ports, aux_terminals)
    print('Ports and airports assigned', flush=True)

    # we filter out the roads that are not motorways or trunks
    # dataframes_OSM["road_nodes"], dataframes_OSM["road_edges"] = filter_roads_tent(dataframes_OSM["road_nodes"], dataframes_OSM["road_edges"])
    dataframes_OSM["port_areas"]= gpd.GeoDataFrame(dataframes_OSM["ports"])
    dataframes_OSM["airport_areas"]= gpd.GeoDataFrame(dataframes_OSM["airports"])

    # Convert areas to points (apply centroid transformation)
    dataframes_OSM['ports']['geometry'] = dataframes_OSM['ports']['geometry'].apply(lambda geom: geom.centroid if isinstance(geom, (Polygon, MultiPolygon)) else geom)
    dataframes_OSM['airports']['geometry'] = dataframes_OSM['airports']['geometry'].apply(lambda geom: geom.centroid if isinstance(geom, (Polygon, MultiPolygon)) else geom)

    #Ensure it's a GeoDataFrame
    for key, osm_df in dataframes_OSM.items():
    # Convert to GeoDataFrame
        dataframes_OSM[key] = gpd.GeoDataFrame(osm_df)

        # Drop unwanted columns safely
        columns_to_remove = ['geometry_bbox.xmin', 'geometry_bbox.xmax', 'geometry_bbox.ymin', 'geometry_bbox.ymax',
                            'fid', 'Continent_', 'lat', 'lon', 'centroid_x', 'centroid_y', 'layer', 'Geometry','name']
        
        columns_in_df = dataframes_OSM[key].columns
        columns_to_drop = [col for col in columns_to_remove if col in columns_in_df]

        if columns_to_drop:
            dataframes_OSM[key] = dataframes_OSM[key].drop(columns=columns_to_drop)

        # Reproject to WGS84 for saving (Ensure CRS is valid)
        dataframes_OSM[key] = dataframes_OSM[key].to_crs(CRS_WGS84_IWW)
        
    # Save the dataframes as GeoDataFrame (OSM) - write parquet except GeoPackage outputs while debugging
    for key, path in file_paths_save_OSM.items():
        gdf = dataframes_OSM.get(key)
        if gdf is None or not isinstance(gdf, gpd.GeoDataFrame):
            print(f"Warning: {key} not found, skipping...")
            continue

        if str(path).lower().endswith('.gpkg'):
            # Write GeoPackage (use layer name = key)
            gdf.to_file(path, layer=key, driver="GPKG")
            print(f"Successfully saved {key} as GeoPackage to {path}", flush=True)
        else:
            # Default: parquet
            gdf.to_parquet(path, index=False)
            print(f"Successfully saved {key} as GeoParquet to {path}", flush=True)
    # Save the dataframes as GeoDataFrame (TENT)
    for key, path in file_paths_save_TENT.items():
        gdf = dataframes_TENT.get(key)
        if gdf is None or not isinstance(gdf, gpd.GeoDataFrame):
            print(f"Warning: {key} not found in dataframes_TENT, skipping...")
            continue
        gdf.to_parquet(path, index=False)  # Use GeoPandas' native function

if __name__ == "__main__":
    main()