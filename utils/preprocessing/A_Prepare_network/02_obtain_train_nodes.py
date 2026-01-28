# This code processes a GeoParquet file containing railway nodes to filter out non-station nodes 
# and save the filtered data to a new Parquet file.

import geopandas as gpd

def main():
    # Define file paths
    input_path = "/soge-home/users/cenv0972/AFP_python/open-gira/results/europe-latest_filter-rail/nodes.gpq"
    output_path = "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/railways/europe_railways_nodes.parquet"

    # Load the GeoParquet file
    gdf = gpd.read_parquet(input_path)

    # Filter nodes where 'station' is False
    filtered_gdf = gdf[gdf['station'] == False]

    # Save to Parquet format
    filtered_gdf.to_parquet(output_path)

    print(f"Filtered data saved to: {output_path}")

if __name__ == "__main__":
    main()