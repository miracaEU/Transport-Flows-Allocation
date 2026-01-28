# This script creates a multimodal transportation network by connecting different modes of transport
# (e.g., road, rail, inland waterways, sea, and air) based on proximity and intermodal terminals.
# The resulting network is saved as a GeoPackage file.

import os
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
tqdm.pandas()
from utils_MIRACA import *
from shapely import wkb
import re

CRS_WGS84 = "EPSG:4326"   # Latitude/Longitude
CRS_WGS84_IWW = "EPSG:3857"   # Latitude/Longitude
CRS_UTM = "EPSG:3034"    # UTM

data_path = "/soge-home/projects/mistral/miraca/"
config = load_config(data_path)
incoming_data_path = config['paths']['incoming_data']
processed_data_path = config['paths']['processed_data']

file_paths_save = {
    "road_edges": "/soge-home/projects/mistral/miraca/processed_data/europe_road_edges_TENT.parquet",
    "road_nodes": "/soge-home/projects/mistral/miraca/processed_data/europe_road_nodes_TENT.parquet",
    "railway_edges": "/soge-home/projects/mistral/miraca/processed_data/europe_railways_edges_TENT.parquet",
    "railway_nodes": "/soge-home/projects/mistral/miraca/processed_data/europe_railways_nodes_TENT.parquet",
    "ports": "/soge-home/projects/mistral/miraca/processed_data/europe_ports_TENT.parquet",
    "iww_nodes": "/soge-home/projects/mistral/miraca/processed_data/europe_IWW_nodes_TENT.parquet",
    "iww_edges": "/soge-home/projects/mistral/miraca/processed_data/europe_IWW_edges_TENT.parquet",
    "airports": "/soge-home/projects/mistral/miraca/processed_data/europe_airports_TENT.parquet",
    "intermodal": "/soge-home/projects/mistral/miraca/processed_data/europe_intermodal_terminals_TENT.parquet",
}

def get_mode_dataframe(mode,rail_status=["open"],intermodal_connection=False):
    
    if mode == "air":
        nodes =  pd.read_parquet(os.path.join(
                                processed_data_path,
                                "processed_unimodal",
                                "europe_airports_TENT.parquet"))  
        # Assign unique integer IDs based on 'name', as aiports have no 'id' column
        nodes['id'] = pd.factorize(nodes['int_name'])[0] + 1

    elif mode == "sea":
        # we will merge the terminals and ports in a node layer, giving importance to terminals,
        # if terminals are not present for the given port
        nodes =  pd.read_parquet(os.path.join(
            processed_data_path,
            "processed_unimodal",
            "europe_ports_TENT.parquet"))  
        # nodes['id'] = pd.factorize(nodes['port_name'])[0] + 1

    elif mode == "IWW":
        iww_edges = pd.read_parquet(os.path.join(
            processed_data_path,
            "processed_unimodal",
            "europe_IWW_edges_TENT.parquet"))
        iww_node_ids = list(set(iww_edges["from_id"].values.tolist() + iww_edges["to_id"].values.tolist()))
        nodes = pd.read_parquet(os.path.join(
                                processed_data_path,
                                "processed_unimodal",
                                "europe_IWW_nodes_TENT.parquet"))
        nodes = nodes[(nodes["id"].isin(iww_node_ids))]
        degree_df = iww_edges[["from_id","to_id"]].stack().value_counts().rename_axis('id').reset_index(name='degree')
        nodes = pd.merge(nodes,degree_df,how="left",on=["id"])

    elif mode == "rail":
        rail_edges = pd.read_parquet(os.path.join(
            processed_data_path,
            "processed_unimodal",
            "europe_railways_edges_TENT.parquet"))
        rail_node_ids = list(set(rail_edges["from_id"].values.tolist() + rail_edges["to_id"].values.tolist()))
        nodes = pd.read_parquet(os.path.join(
                                processed_data_path,
                                "processed_unimodal",
                                "europe_railways_nodes_TENT.parquet"))
        nodes['id'] = nodes['id'].str.extract(r'_(\d+)$').fillna(nodes['id'])  # Keep original if no match
        rail_node_ids = [re.search(r'_(\d+)$', item).group(1) if re.search(r'_(\d+)$', item) else item for item in rail_node_ids]
        nodes = nodes[(nodes["id"].isin(rail_node_ids))]
        degree_df = rail_edges[["from_id","to_id"]].stack().value_counts().rename_axis('id').reset_index(name='degree')
        nodes = pd.merge(nodes,degree_df,how="left",on=["id"])
    elif mode == "road": 
        road_edges = pd.read_parquet(os.path.join(
            processed_data_path,
            "processed_unimodal",
            "europe_road_edges_TENT.parquet"))
        road_node_ids = list(set(road_edges["from_id"].values.tolist() + road_edges["to_id"].values.tolist()))
        nodes = pd.read_parquet(os.path.join(
                                processed_data_path,
                                "processed_unimodal",
                                "europe_road_nodes_TENT.parquet"))
        nodes['id'] = nodes['id'].str.extract(r'_(\d+)$').fillna(nodes['id'])  # Keep original if no match
        road_node_ids = [re.search(r'_(\d+)$', item).group(1) if re.search(r'_(\d+)$', item) else item for item in road_node_ids]
        nodes = nodes[(nodes["id"].isin(road_node_ids))]
        degree_df = road_edges[["from_id","to_id"]].stack().value_counts().rename_axis('id').reset_index(name='degree')
        nodes = pd.merge(nodes,degree_df,how="left",on=["id"])

    # Convert WKT to geometries
    nodes['geometry'] = nodes['geometry'].apply(wkb.loads)
    # Convert to GeoDataFrame
    nodes = gpd.GeoDataFrame(nodes, geometry='geometry', crs="EPSG:3034")
    if 'Country' in nodes.columns:
        nodes.rename(columns={'Country': 'country'}, inplace=True)
    elif 'country' in nodes.columns:
        pass
    else:
        admin_layer_path = os.path.join(
            incoming_data_path,
            "spatial/admin",
            "ne_110m_admin_0_countries.shp")
        nodes = add_country(nodes,admin_layer_path)

    if 'iso_a3' in nodes.columns:
        nodes.rename(columns={'iso_a3': 'iso3'}, inplace=True)
    elif 'iso_3' in nodes.columns:
        nodes.rename(columns={'iso_3': 'iso3'}, inplace=True)
    elif 'iso3' in nodes.columns:
        pass
    else:
        admin_layer_path = os.path.join(
            incoming_data_path,
            "spatial/admin",
            "ne_110m_admin_0_countries.shp")
        nodes = add_iso3(nodes,admin_layer_path)
    
    return nodes

def main():    
    intermodal_gpd = pd.read_parquet(os.path.join(
        processed_data_path,
        "processed_unimodal",
        "europe_intermodal_terminals_TENT.parquet"))
    # Convert WKT to geometries
    intermodal_gpd['geometry'] = intermodal_gpd['geometry'].apply(wkb.loads)
    # Convert to GeoDataFrame
    intermodal_gpd = gpd.GeoDataFrame(intermodal_gpd, geometry='geometry', crs="EPSG:3034")  

    epsg_meters = 3395 # To convert geometries to measure distances in meters
    from_modes = ["IWW","IWW","sea","sea","rail", "air","air","IWW"]
    to_modes = ["rail","road","rail","road","road","rail","road","sea"]

    multi_df = []
    for idx,(f_m,t_m) in enumerate(zip(from_modes,to_modes)):
        print(f"Creating edges from {f_m} to {t_m}")
        f_df = get_mode_dataframe(f_m)
        t_df = get_mode_dataframe(t_m)
        rail_to_IWW=False
        rail_to_road=False
        road_to_IWW=False
        if f_m=="IWW" and t_m=="rail":
            rail_to_IWW=True
        elif f_m=="IWW" and t_m=="road":
            road_to_IWW=True
        elif  f_m=="rail" and t_m=="road":
            rail_to_road=True

        f_t_df = create_edges_from_nearest_node_joins(
                            f_df,
                            t_df,
                            'id', 'id',
                            'iso3', 'iso3',
                            f_m,t_m,
                            rail_to_road,
                            rail_to_IWW,
                            road_to_IWW,
                            intermodal_gpd)

        if len(f_t_df) > 0:
            multi_df.append(f_t_df)
            c_t_df = f_t_df[["from_id","to_id","from_infra",
                        "to_infra","from_iso3","to_iso3",
                        "link_type","length_m","geometry"]].copy()
            c_t_df.columns = ["to_id","from_id",
                        "to_infra","from_infra",
                        "to_iso3",
                        "from_iso3",
                        "link_type","length_m","geometry"]
            multi_df.append(c_t_df)

    multi_df = gpd.GeoDataFrame(
                    pd.concat(multi_df,axis=0,ignore_index=True),
                    geometry="geometry",crs=f"EPSG:{epsg_meters}")
    multi_df = multi_df.to_crs(epsg=4326)
    multi_df["id"] = multi_df.index.values.tolist()
    multi_df["id"] = multi_df.progress_apply(lambda x:f"intermodale_{x.id}",axis=1)
    
    
    multi_df.to_file(os.path.join(
                            processed_data_path,
                            "Europe_multimodal.gpkg"
                                ), 
                            layer="edges",
                            driver="GPKG"
                            )

if __name__ == '__main__':
    main()