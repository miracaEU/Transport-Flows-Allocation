"""Functions for preprocessing network data
    Original Code from Raghav Pant
    Adjusted to MIRACA (paneuropean study case) by Alberto Fernandez-Perez
"""
import os
import json
import network as ntx
import numpy as np
import pandas as pd
import igraph as ig
import networkx
import geopandas as gpd
from shapely.geometry import shape, mapping, LineString
from scipy.spatial import cKDTree
from tqdm import tqdm
from haversine import haversine
tqdm.pandas()
from math import sin, cos, asin, sqrt, radians
from geopy.distance import geodesic
import glob
import rasterio
from rasterio.features import shapes

def calculate_nearest_distances(df_1, df_2, distance_threshold):
    """
    Calculate the minimum distance from each point in `df_1` to the nearest point in `df_2`
    using spatial indexing to speed up the process.
    """
    # Create a spatial index for the df_2 terminals
    df_2_sindex = df_2.sindex
    min_distances = []
    
    for _, from_row in df_1.iterrows():
        # Get the geometry of the df_1 point
        from_point = from_row.geometry
        # Use the spatial index to find possible nearest neighbors in the df_2 DataFrame
        possible_matches_index = list(df_2_sindex.nearest(from_point))  # Adjust num_results for optimization
        # Calculate the geodesic distance for the closest possible matches
        min_distance = float('inf')
        for idx in possible_matches_index:
            to_point = df_2.iloc[idx].geometry
            # Ensure coordinates are extracted correctly
            from_coords = (from_point.y, from_point.x)  # (latitude, longitude)
            to_coords = to_point.iloc[0].coords[0]        # (latitude, longitude)
            to_coords = (to_coords[1], to_coords[0])
            try:
                # Calculate the geodesic distance and catch any errors
                dist = geodesic(from_coords, to_coords).km
            except Exception as e:
                # Print the error and the problematic coordinates
                print(f"Error calculating distance between {from_coords} and {to_coords}: {e}")
                continue  # Skip this iteration and continue with the next
            
            if dist < min_distance:
                min_distance = dist
        
        min_distances.append(min_distance)
    
    # Add the calculated minimum distances to the dataframe
    df_1['min_distance'] = min_distances
    
    # Filter based on distance threshold
    return df_1[df_1['min_distance'] <= distance_threshold]


def create_edges_from_nearest_node_joins(from_df,to_df,
                    from_id_column,to_id_column,
                    from_iso_column,to_iso_column,
                    from_mode,to_mode,
                    rail_to_road,
                    rail_to_IWW,
                    road_to_IWW,
                    intermodal,
                    distance_threshold=1000):
    # Ensure the EPSG is correct for haversine distance
    from_df = from_df.to_crs(epsg=4326)
    to_df = to_df.to_crs(epsg=4326)
    intermodal = intermodal.to_crs(epsg=4326)

    # Now use the function to filter the dataframes
    if rail_to_road:
        from_df = calculate_nearest_distances(from_df, intermodal, distance_threshold)
        from_df = from_df[(from_df['degree'] == 1) | (from_df['min_distance'] <= distance_threshold)]
        print('Intermodal terminals with rail connections to road:', len(from_df.index))

    if road_to_IWW or rail_to_IWW:
        from_df = calculate_nearest_distances(from_df, intermodal, distance_threshold)
        from_df = from_df[from_df['min_distance'] <= distance_threshold]
        print('Intermodal terminals with IWW connections to road/rail:', len(from_df.index))

    from_df.rename(columns={from_id_column:"from_id",from_iso_column:"from_iso3"},inplace=True)
    to_df.rename(columns={to_id_column:"to_id",to_iso_column:"to_iso3"},inplace=True)
    from_df["from_infra"] = from_mode
    to_df["to_infra"] = to_mode

    from_to_df = ckdnearest(from_df[["from_id","from_iso3","from_infra","geometry"]],
                            to_df[["to_id","to_iso3","to_infra","geometry"]])
    from_to_df["link_type"] = f"{from_mode}-{to_mode}"
    from_to_df = from_to_df[from_to_df["dist"] <= distance_threshold]

    if len(from_to_df.index) > 0:
        from_to_df.rename(columns = {"geometry":"from_geometry","dist":"length_m"},inplace=True)
        to_df.rename(columns = {"geometry":"to_geometry"},inplace=True)
        from_to_df = pd.merge(from_to_df,to_df[["to_id","to_geometry"]],how="left",on=["to_id"])
        from_to_df["geometry"] = from_to_df.progress_apply(
                                    lambda x:LineString([x.from_geometry,x.to_geometry]),
                                    axis=1)
        from_to_df.drop(["from_geometry","to_geometry"],axis=1,inplace=True)
    
    return from_to_df


def get_line_status(x):
    if "abandon" in x:
        return "abandoned"
    else:
        return "open"

def add_attributes(dataframe,columns_attributes):
    for column_name,attribute_value in columns_attributes.items():
        dataframe[column_name] = attribute_value

    return dataframe


# Function to calculate similarity
def is_similar(str1, str2, threshold):
    if pd.isna(str1) or pd.isna(str2):
        return False
    return SequenceMatcher(None, str1, str2).ratio() >= threshold

# filter out similar elements
def drop_similar(els_1, els_2, terminal_col, port_col, threshold):
    els_2_to_keep = els_2.copy()
    for port_idx, port_row in els_2.iterrows():
        port_name = port_row[port_col]
        # Check if there's a similar name in els_1
        if any(is_similar(port_name, terminal_name, threshold) for terminal_name in els_1[terminal_col]):
            els_2_to_keep = els_2_to_keep.drop(port_idx)
    return els_2_to_keep


# Function to assign the country name based on geometry
def add_country(df,admin_layer_path):
    world = gpd.read_file(admin_layer_path)
    # Ensure the GeoDataFrame is in the same CRS as the world boundaries
    df = df.to_crs(world.crs)
    # Spatial join to get country name based on geometry
    joined = gpd.sjoin(df, world[['geometry', 'ADMIN']], how="left", predicate='intersects')
    # Assign country name
    df['country'] = joined['ADMIN']
    
    return df

# Function to assign the ISO3 code based on geometry
def add_iso3(df,admin_layer_path):
    world = gpd.read_file(admin_layer_path)
    # Ensure the GeoDataFrame is in the same CRS as the world boundaries
    df = df.to_crs(world.crs)
    # Spatial join to get country name based on geometry
    joined = gpd.sjoin(df, world[['geometry', 'ISO_A3']], how="left", predicate='intersects')
    # Assign ISO3 code
    df['iso3'] = joined['ISO_A3']
    
    return df



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

def link_nodes_to_nearest_edge(network, condition=None, tolerance=1e-9):
    """Link nodes to all edges within some distance"""
    new_node_geoms = []
    new_edge_geoms = []
    for node in tqdm(
        network.nodes.itertuples(index=False), desc="link", total=len(network.nodes)
    ):
        # for each node, find edges within
        edge = ntx.nearest_edge(node.geometry, network.edges)
        if condition is not None and not condition(node, edge):
            continue
        # add nodes at points-nearest
        point = ntx.nearest_point_on_line(node.geometry, edge.geometry)
        if point != node.geometry:
            new_node_geoms.append(point)
            # add edges linking
            line = LineString([node.geometry, point])
            new_edge_geoms.append(line)

    new_nodes = ntx.matching_gdf_from_geoms(network.nodes, new_node_geoms)
    all_nodes = ntx.concat_dedup([network.nodes, new_nodes])

    new_edges = ntx.matching_gdf_from_geoms(network.edges, new_edge_geoms)
    all_edges = ntx.concat_dedup([network.edges, new_edges])

    # split edges as necessary after new node creation
    unsplit = ntx.Network(nodes=all_nodes, edges=all_edges)

    # this step is typically the majority of processing time
    split = ntx.split_edges_at_nodes(unsplit,tolerance=tolerance)

    return split
def convert_json_geopandas(df,epsg=4326):
    layer_dict = []    
    for key, value in df.items():
        if key == "features":
            for feature in value:
                if any(feature["geometry"]["coordinates"]):
                    d1 = {"geometry":shape(feature["geometry"])}
                    d1.update(feature["properties"])
                    layer_dict.append(d1)

    return gpd.GeoDataFrame(pd.DataFrame(layer_dict),geometry="geometry", crs=f"EPSG:{epsg}")

def components(edges,nodes,
                node_id_column="id",edge_id_column="id",
                from_node_column="from_id",to_node_column="to_id"):
    G = networkx.Graph()
    G.add_nodes_from(
        (getattr(n, node_id_column), {"geometry": n.geometry}) for n in nodes.itertuples()
    )
    G.add_edges_from(
        (getattr(e,from_node_column), getattr(e,to_node_column), 
            {edge_id_column: getattr(e,edge_id_column), "geometry": e.geometry})
        for e in edges.itertuples()
    )
    components = networkx.connected_components(G)
    for num, c in enumerate(components):
        print(f"Component {num} has {len(c)} nodes")
        edges.loc[(edges[from_node_column].isin(c) | edges[to_node_column].isin(c)), "component"] = num
        nodes.loc[nodes[node_id_column].isin(c), "component"] = num

    return edges, nodes

def add_lines(x,from_nodes_df,to_nodes_df,from_nodes_id,to_nodes_id):
    from_point = from_nodes_df[from_nodes_df[from_nodes_id] == x[from_nodes_id]]
    to_point = to_nodes_df[to_nodes_df[to_nodes_id] == x[to_nodes_id]] 
    return LineString([from_point.geometry.values[0],to_point.geometry.values[0]])

def ckdnearest(gdA, gdB):
    """Taken from https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
    """
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf

def gdf_geom_clip(gdf_in, clip_geom):
    """Filter a dataframe to contain only features within a clipping geometry

    Parameters
    ---------
    gdf_in
        geopandas dataframe to be clipped in
    province_geom
        shapely geometry of province for what we do the calculation

    Returns
    -------
    filtered dataframe
    """
    return gdf_in.loc[gdf_in['geometry'].apply(lambda x: x.within(clip_geom))].reset_index(drop=True)

def get_nearest_values(x,input_gdf,column_name):
    polygon_index = input_gdf.distance(x.geometry).sort_values().index[0]
    return input_gdf.loc[polygon_index,column_name]

def extract_gdf_values_containing_nodes(x, input_gdf, column_name):
    a = input_gdf.loc[list(input_gdf.geometry.contains(x.geometry))]
    if len(a.index) > 0:
        return a[column_name].values[0]
    else:
        polygon_index = input_gdf.distance(x.geometry).sort_values().index[0]
        return input_gdf.loc[polygon_index,column_name]

def load_config(file_path=None):
    if file_path:
        script_dir = file_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = os.path.join(script_dir, '..', '..')  # Move two levels up

    config_path = os.path.join(script_dir, 'config.json')
    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    
    return config
    
    
def create_network_from_nodes_and_edges(nodes,edges,node_edge_prefix,
                        snap_distance=None,geometry_precision=False,by=None):
    edges.columns = map(str.lower, edges.columns)
    if "id" in edges.columns.values.tolist():
        edges.rename(columns={"id": "e_id"}, inplace=True)

    # Deal with empty edges (drop)
    empty_idx = edges.geometry.apply(lambda e: e is None or e.is_empty)
    if empty_idx.sum():
        empty_edges = edges[empty_idx]
        print(f"Found {len(empty_edges)} empty edges.")
        print(empty_edges)
        edges = edges[~empty_idx].copy()

    network = ntx.Network(nodes, edges)
    print("* Done with network creation")

    network = ntx.split_multilinestrings(network)
    print("* Done with splitting multilines")
    if geometry_precision is True:
        network = ntx.round_geometries(network, precision=5)
        print("* Done with rounding off geometries")

    if nodes is not None:
        if snap_distance is not None:
            # network = ntx.link_nodes_to_edges_within(network, snap_distance, tolerance=1e-10)
            network = link_nodes_to_nearest_edge(network, tolerance=1e-9)
            print ('* Done with joining nodes to edges')
        else:
            network = ntx.snap_nodes(network)
            print ('* Done with snapping nodes to edges')
        # network.nodes = ntx.drop_duplicate_geometries(network.nodes)
        # print ('* Done with dropping same geometries')

        # network = ntx.split_edges_at_nodes(network,tolerance=9e-10)
        # print ('* Done with splitting edges at nodes')

    network = ntx.add_endpoints(network)   
    print ('* Done with adding endpoints')

    network.nodes = ntx.drop_duplicate_geometries(network.nodes)
    print ('* Done with dropping same geometries')

    network = ntx.split_edges_at_nodes(network,tolerance=1e-9)
    print ('* Done with splitting edges at nodes')
    
    network = ntx.add_ids(network, 
                            edge_prefix=f"{node_edge_prefix}e", 
                            node_prefix=f"{node_edge_prefix}n")
    network = ntx.add_topology(network, id_col='id')
    print ('* Done with network topology')

    if by is not None:
        network = ntx.merge_edges(network,by=by)
        print ('* Done with merging network')

    network.edges.rename(columns={'from_id':'from_node',
                                'to_id':'to_node',
                                'id':'edge_id'},
                                inplace=True)
    network.nodes.rename(columns={'id':'node_id'},inplace=True)
    
    return network

def network_od_path_estimations(graph,
    source, target, cost_criteria,path_id_column):
    """Estimate the paths, distances, times, and costs for given OD pair

    Parameters
    ---------
    graph
        igraph network structure
    source
        String/Float/Integer name of Origin node ID
    source
        String/Float/Integer name of Destination node ID
    tonnage : float
        value of tonnage
    vehicle_weight : float
        unit weight of vehicle
    cost_criteria : str
        name of generalised cost criteria to be used: min_gcost or max_gcost
    time_criteria : str
        name of time criteria to be used: min_time or max_time
    fixed_cost : bool

    Returns
    -------
    edge_path_list : list[list]
        nested lists of Strings/Floats/Integers of edge ID's in routes
    path_dist_list : list[float]
        estimated distances of routes
    path_time_list : list[float]
        estimated times of routes
    path_gcost_list : list[float]
        estimated generalised costs of routes

    """
    paths = graph.get_shortest_paths(source, target, weights=cost_criteria, output="epath")


    edge_path_list = []
    path_gcost_list = []
    # for p in range(len(paths)):
    for path in paths:
        edge_path = []
        path_gcost = 0
        if path:
            for n in path:
                edge_path.append(graph.es[n][path_id_column])
                path_gcost += graph.es[n][cost_criteria]

        edge_path_list.append(edge_path)
        path_gcost_list.append(path_gcost)

    
    return edge_path_list, path_gcost_list

def haversine_distance(point1, point2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1 = point1.bounds[0], point1.bounds[1]
    lon2, lat2 = point2.bounds[0], point2.bounds[1]

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers
    # print('Distance from beginning to end of route in km: ',round((c * r), 2),'\n')
    return c * r

def modify_distance(x):
    if x["length"] < 355 and x["distance"] < 40075:
        return x["distance"]
    else:
        start = x.geometry.coords[0]
        end = x.geometry.coords[-1]
        return haversine(
                    (
                        round(start[1],2),
                        round(start[0],2)
                    ),
                    (
                        round(end[1],2),
                        round(end[0],2)
                    )
                )

def match_ports(df1,df2,df1_id_column,df2_id_column,cutoff_distance):
    # Find the nearest ports that match and the ones which do not 
    matches = ckdnearest(df1,
                        df2)
    matches = matches.sort_values(by="dist",ascending=True)
    matches["matches"] = np.where(matches["dist"] <= cutoff_distance,"Y","N")

    selection = matches[matches["dist"] <= cutoff_distance]
    selection = selection.drop_duplicates(subset=df2_id_column,keep='first')
    matched_ids = list(set(selection[df1_id_column].values.tolist()))
    return matches, df1[~(df1[df1_id_column].isin(matched_ids))]


def create_igraph_from_dataframe(graph_dataframe, directed=False, simple=False):
    graph = ig.Graph.TupleList(
        graph_dataframe.itertuples(index=False),
        edge_attrs=list(graph_dataframe.columns)[2:],
        directed=directed
    )
    if simple:
        graph.simplify()

    es, vs, simple = graph.es, graph.vs, graph.is_simple()
    d = "directed" if directed else "undirected"
    s = "simple" if simple else "multi"
    print(
        "Created {}, {} {}: {} edges, {} nodes.".format(
            s, d, "igraph", len(es), len(vs)))

    return graph


def process_rasters_to_parquet(input_folder, value_to_extract, output_parquet):
    all_vectors = []
    input_files = glob.glob(os.path.join(input_folder, "*.TIF")) + glob.glob(os.path.join(input_folder, "*.tif"))
    
    if not input_files:
        print(f"No raster files found in folder: {input_folder}")
        return

    print(f"Found {len(input_files)} raster files.")

    for file in input_files:
        try:
            print(f"Processing file: {file}")
            with rasterio.open(file) as src:
                raster = src.read(1)  # Read the first band
                mask = raster == value_to_extract
                
                if not mask.any():
                    print(f"No pixels found with value {value_to_extract} in {file}. Skipping.")
                    continue
                
                results = shapes(raster, mask=mask, transform=src.transform)
                geoms = [({"geometry": geom, "properties": {"value": val}}) for geom, val in results]
                
                gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)
                all_vectors.append(gdf)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if all_vectors:
        merged_gdf = gpd.GeoDataFrame(pd.concat(all_vectors, ignore_index=True))
        merged_gdf.to_parquet(output_parquet, index=False)
        print(f"Saved merged vector data to {output_parquet}")
    else:
        print("No rasters processed.")