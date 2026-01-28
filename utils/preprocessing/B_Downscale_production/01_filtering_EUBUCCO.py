# This script processes building data from the EUBUCCO dataset, filtering it based on specific criteria 
# and saving the results for further analysis. The script is designed to handle large datasets by dividing 
# the workload into jobs, making it suitable for parallel processing. It filters buildings based on their 
# type (e.g., industrial) and ensures CRS consistency for spatial analysis.

from pathlib import Path
import geopandas as gpd

# Define the buildings folder path
buildings_folder = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/buildings")
filtered_folder = buildings_folder / "filtered"
filtered_folder.mkdir(parents=True, exist_ok=True)

# Iterate only .gpkg files (skips directories)
for file in buildings_folder.glob("*.gpkg"):
    print(f"Processing file: {file.name}")
    iso3 = file.stem.split("-")[1] if "-" in file.stem else file.stem
    buildings = gpd.read_file(file).to_crs("EPSG:3035")
    buildings = buildings[ buildings['type_source'].isna() | (buildings['type_source'] == 'industrial') ]
    output_file = filtered_folder / f"{iso3}.parquet"
    buildings.to_parquet(output_file, index=False)
    print(f"Saved filtered buildings for ISO3: {iso3} to {output_file}")