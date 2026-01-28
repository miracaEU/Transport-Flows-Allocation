# This code processes IWW (Inland Waterway) data to create a network of nodes and edges.
# It assigns features (e.g., ports, locks, intermodal terminals) to the nearest nodes,
# generates edges from channel geometries, and saves the results as GeoParquet files.

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.wkb import loads as load_wkb
from shapely.wkt import loads as load_wkt

# Define CRS constants
CRS_WGS84 = "EPSG:4326"   # Latitude/Longitude
CRS_WGS84_IWW = "EPSG:3857"   # Latitude/Longitude
CRS_UTM = "EPSG:3034"    # UTM Zone 32N ETRS89

# File paths for the datasets
file_paths = {
    "iww_ports": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/iww/europe_IWW_ports.parquet",
    "iww_channels": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/iww/europe_IWW_channels.parquet",
    "intermodal": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/intermodal/europe_intermodal_terminals.parquet",
    "iww_locks": "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/iww/europe_IWW_locks.parquet"
}


# Utility function to load and reproject datasets
def load_and_reproject(file_path, crs_original, crs_target):
    """
    Loads a dataset, converts geometry to GeoDataFrame, and reprojects to the target CRS.
    """
    df = pd.read_parquet(file_path)
    if 'geometry' not in df.columns:
        raise ValueError(f"Dataset {file_path} does not contain a 'geometry' column.")
    if isinstance(df['geometry'].iloc[0], bytes):  # WKB format
        df['geometry'] = df['geometry'].apply(load_wkb)
    elif isinstance(df['geometry'].iloc[0], str):  # WKT format
        df['geometry'] = df['geometry'].apply(load_wkt)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs_original)
    return gdf.to_crs(crs_target)

def assign_nearest_feature(nodes_gdf, feature_gdf, feature_name, max_distance=2000):
    """
    Assigns each feature (port, lock, etc.) to only one closest node.
    """

    feature_gdf = feature_gdf.to_crs(nodes_gdf.crs)
    nearest_nodes = gpd.sjoin_nearest(feature_gdf, nodes_gdf, how="left", distance_col="distance")
    nearest_nodes = nearest_nodes[nearest_nodes["distance"] <= max_distance]
    nearest_nodes = nearest_nodes.sort_values(by=["distance"]).drop_duplicates(subset="index_right")
    nearest_nodes['feature'] = feature_name
    if 'BB_PORT_NA' not in nearest_nodes.columns:
        nearest_nodes['BB_PORT_NA'] = None
    nodes_gdf = nodes_gdf.merge(nearest_nodes[["index_right", "feature", "BB_PORT_NA"]], 
                                left_index=True, right_on="index_right", how="left", suffixes=("", "_new"))
    nodes_gdf["feature"] = nodes_gdf["feature"].combine_first(nodes_gdf["feature_new"])
    nodes_gdf = nodes_gdf.drop(columns=["feature_new", "index_right"])

    return nodes_gdf


def create_edges_from_channels(iww_channels_gdf, iww_nodes):
    """
    Creates edges based on channel geometries by breaking the LineStrings into smaller segments
    and assigns the closest nodes as from/to.
    """
    edges = []
    
    for _, row in iww_channels_gdf.iterrows():
        channel = row['geometry']
        if isinstance(channel, LineString):
            num_segments = len(channel.coords) - 1
            for i in range(num_segments):
                segment = LineString([channel.coords[i], channel.coords[i+1]])
                start_point = Point(segment.coords[0]) 
                end_point = Point(segment.coords[1])  
                start_node = iww_nodes.iloc[iww_nodes.geometry.distance(start_point).idxmin()]
                end_node = iww_nodes.iloc[iww_nodes.geometry.distance(end_point).idxmin()]
                edge = {
                    "from_node": start_node["node_id"],
                    "to_node": end_node["node_id"],
                    "geometry": segment
                }
                edges.append(edge)
        else:
            continue
    return edges


def main():
    # Load and reproject datasets
    iww_ports_gdf = load_and_reproject(pd.read_parquet(file_paths["iww_ports"]), CRS_WGS84_IWW, CRS_UTM)
    iww_channels_gdf = load_and_reproject(pd.read_parquet(file_paths["iww_channels"]), CRS_WGS84_IWW, CRS_UTM)
    iww_locks_gdf = load_and_reproject(pd.read_parquet(file_paths["iww_locks"]), CRS_WGS84_IWW, CRS_UTM)
    intermodal_gdf = load_and_reproject(pd.read_parquet(file_paths["intermodal"]), CRS_WGS84, CRS_UTM)

    # Check if 'geometry' exists after loading and reprojecting
    if 'geometry' not in iww_ports_gdf.columns or 'geometry' not in iww_channels_gdf.columns:
        raise ValueError("Missing geometry column in one of the input GeoDataFrames.")

    # Filter intermodal points of interest
    intermodal_wasser_gdf = intermodal_gdf[intermodal_gdf['LEGENDE'].isin(['Wasserstraße / Straße'])]
    intermodal_tri_gdf = intermodal_gdf[intermodal_gdf['LEGENDE'].isin(['Trimodal'])]

    # Extract unique nodes from channel edges
    unique_nodes = set()
    for line in iww_channels_gdf.geometry:
        if isinstance(line, LineString):
            unique_nodes.update(Point(coord) for coord in line.coords)

    # Create iww_nodes GeoDataFrame with geometry
    iww_nodes = gpd.GeoDataFrame(geometry=list(unique_nodes), crs=CRS_UTM)
    iww_nodes["node_id"] = range(len(iww_nodes))

    # Assign nearest nodes to elements within 100m
    iww_nodes["feature"] = None
    iww_nodes = assign_nearest_feature(iww_nodes, iww_ports_gdf, "port")
    iww_nodes = assign_nearest_feature(iww_nodes, iww_locks_gdf, "lock")
    iww_nodes = assign_nearest_feature(iww_nodes, intermodal_wasser_gdf, "Road-IWW intermodal")
    iww_nodes = assign_nearest_feature(iww_nodes, intermodal_tri_gdf, "Trimodal")

    # Clean up
    iww_nodes.drop(columns="dist", errors="ignore", inplace=True)
    iww_nodes.reset_index(drop=True, inplace=True)
    iww_nodes["node_id"] = iww_nodes.index
    iww_nodes = iww_nodes.loc[:, ~iww_nodes.columns.duplicated()]

    # Generate edge list from channel geometries
    edge_list = create_edges_from_channels(iww_channels_gdf, iww_nodes)

    # Convert the list of dictionaries into a GeoDataFrame
    if edge_list:
        iww_edges = gpd.GeoDataFrame(edge_list, crs=CRS_UTM)
    else:
        iww_edges = gpd.GeoDataFrame(columns=["from_node", "to_node", "geometry"], crs=CRS_UTM)

    # Convert nodes back to WGS84 and save results
    iww_nodes = iww_nodes.to_crs(CRS_WGS84)
    iww_edges = iww_edges.to_crs(CRS_WGS84)
    iww_edges.to_parquet("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/iww/europe_IWW_edges.parquet")
    iww_nodes.to_parquet("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/iww/europe_IWW_nodes.parquet")

if __name__ == "__main__":
    main()