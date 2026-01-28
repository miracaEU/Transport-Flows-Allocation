# This script automates the process of downloading OpenStreetMap (OSM) data for European countries 
# from the Geofabrik website. The downloaded data is saved in the specified directories for further 
# processing. It organizes the data into categories such as industrial, commercial, residential, 
# and farmland land use. 

import os
from pyrosm import OSM
import requests
from tqdm import tqdm

# Define the base directories
DATA_DIR = "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/osm_data"
LAND_USE_DIR = "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/land_use"
INDUSTRIAL_DIR = os.path.join(LAND_USE_DIR, "industrial")
COMMERCIAL_DIR = os.path.join(LAND_USE_DIR, "commercial")
RESIDENTIAL_DIR = os.path.join(LAND_USE_DIR, "residential")
FARMLAND_DIR = os.path.join(LAND_USE_DIR, "farmland")
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the osm_data directory exists
os.makedirs(INDUSTRIAL_DIR, exist_ok=True)  
os.makedirs(COMMERCIAL_DIR, exist_ok=True)  
os.makedirs(RESIDENTIAL_DIR, exist_ok=True)  
os.makedirs(FARMLAND_DIR, exist_ok=True)  

# European countries list
countries = ['cyprus','denmark','germany',
                'estonia','spain','finland','france','greece','croatia','hungary',
                'ireland-and-northern-ireland','italy','latvia','luxembourg','lithuania','malta','netherlands',
                'norway','poland','portugal','romania','slovenia','slovakia','sweden']

# Base URL for downloading OSM data
BASE_URL = "https://download.geofabrik.de/europe/"

# Function to download the .osm.pbf file
def download_file(url, out_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(out_path, 'wb') as file, tqdm(
        desc=os.path.basename(out_path),
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    # Loop through all countries
    landuse_types = [
        ("industrial", INDUSTRIAL_DIR),
        ("commercial", COMMERCIAL_DIR),
        ("residential", RESIDENTIAL_DIR),
        ("farmland", FARMLAND_DIR),
    ]

    for country in countries:
        file_name = f"{country}-latest.osm.pbf"
        file_url = f"{BASE_URL}{file_name}"
        pbf_path = os.path.join(DATA_DIR, file_name)

        # Prepare output paths for each landuse
        output_paths = {lu: os.path.join(dirpath, f"{country}-{lu}.gpkg") for lu, dirpath in landuse_types}

        # Download the .osm.pbf file if it doesn't already exist
        if not os.path.exists(pbf_path):
            print(f"Downloading {file_name}...")
            try:
                download_file(file_url, pbf_path)
                print(f"Downloaded {file_name} to {pbf_path}")
            except Exception as e:
                print(f"Failed to download {file_name}: {e}")
                continue
        else:
            print(f"{file_name} already exists at {pbf_path}")

        # Create a single OSM instance per PBF and process requested landuse types
        try:
            osm = OSM(pbf_path)
        except Exception as e:
            print(f"Failed to open OSM file {pbf_path}: {e}")
            continue

        for landuse, _dir in landuse_types:
            out_path = output_paths[landuse]
            if os.path.exists(out_path):
                print(f"{file_name} already exists at {out_path}")
                continue

            print(f"Processing {file_name} for landuse='{landuse}'...")
            try:
                lu_gdf = osm.get_landuse(custom_filter={"landuse": [landuse]})
                lu_gdf.to_file(out_path, driver="GPKG")
                print(f"Filtered {landuse} data saved to {out_path}")
            except Exception as e:
                print(f"Failed to process {landuse} for {file_name}: {e}")

if __name__ == "__main__":
    main()
    