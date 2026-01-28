# This script processes and merges industrial and commercial data at the NUTS2 level.
# It combines the datasets, assigns unique IDs, and calculates outgoing and incoming flows for each sector.
# The script also filters and redistributes flows based on specified thresholds and saves the results as Parquet files.

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

# Define the directory containing the output files
output_dir = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industrial_OD/country_to_country") 

# Define the paths for industrial and commercial data
industries_file = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/land_use/processed_sites/industries/industries_europe_NUTS2.parquet")
commercial_file = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/land_use/processed_sites/commercial/commercial_europe_NUTS2.parquet")

# Read industrial and commercial data
print(f"Reading industries file: {industries_file}", flush=True)
industries = gpd.read_parquet(industries_file)

print(f"Reading commercial file: {commercial_file}", flush=True)
commercial = gpd.read_file(commercial_file)  # Read as GeoPackage

# Ensure consistent CRS for merging
industries = industries.to_crs("EPSG:3035")
commercial = commercial.to_crs("EPSG:3035")

# Combine industrial and commercial data into one GeoDataFrame
combined_gdf = gpd.GeoDataFrame(pd.concat([industries, commercial], ignore_index=True), crs="EPSG:3035")

# Remove the 'OBJECTID' column if it exists
if 'OBJECTID' in combined_gdf.columns:
    combined_gdf = combined_gdf.drop(columns=['OBJECTID'])

files = list(output_dir.glob("**/flows_*.parquet"))
if not files:
    raise FileNotFoundError(f"No files matching 'flows_*.parquet' found in {output_dir} or its subfolders.")

# Initialize a list to store all DataFrames
all_flows = []
for file in files:
    try:
        all_flows.append(pd.read_parquet(file))
    except Exception as e:
        print(f"Error reading file {file}: {e}")

# Check if any files were successfully read
if not all_flows:
    raise ValueError("No valid Parquet files were read. The 'all_flows' list is empty.")

# Combine all DataFrames in the list into a single DataFrame
all_flows_df = pd.concat(all_flows, ignore_index=True)

# Ensure output dirs exist
flow_output_file = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industrial_OD//industrial_OD.parquet")
flow_output_file.parent.mkdir(parents=True, exist_ok=True)

# Save combined flow file
print(f"Saving flows DataFrame to: {flow_output_file}", flush=True)
all_flows_df.to_parquet(flow_output_file, index=False)

# Create outgoing flow columns for each commodity type
outgoing_flows = (
    all_flows_df.groupby(["origin_id", "origin_sector"])["value"]
    .sum()
    .unstack(fill_value=0)
    .add_prefix("outgoing_")
)

# Create incoming flow columns for each commodity type
incoming_flows = (
    all_flows_df.groupby(["destination_id", "origin_sector"])["value"]
    .sum()
    .unstack(fill_value=0)
    .add_prefix("incoming_")
)

# Merge outgoing and incoming flows into the combined GeoDataFrame
combined_gdf = combined_gdf.merge(outgoing_flows, left_on="id", right_index=True, how="left")
combined_gdf = combined_gdf.merge(incoming_flows, left_on="id", right_index=True, how="left")

# Fill NaN values with 0 for the new flow columns
flow_columns = [col for col in combined_gdf.columns if col.startswith("outgoing_") or col.startswith("incoming_")]
combined_gdf[flow_columns] = combined_gdf[flow_columns].fillna(0)

# Remove industries where all flow columns are 0
combined_gdf = combined_gdf[combined_gdf[flow_columns].sum(axis=1) > 0]

# Save the updated GeoDataFrame to a Parquet file
output_file = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industries_w_outputs/industries_with_flows.parquet")
output_file.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving updated GeoDataFrame with flows to: {output_file}", flush=True)
combined_gdf.to_parquet(output_file, index=False)

# Add origin and destination country columns
all_flows_df['origin_country'] = all_flows_df['origin_nuts2'].str[:2]
all_flows_df['destination_country'] = all_flows_df['destination_nuts2'].str[:2]

# Thresholds to process
thresholds = [0.25, 0.1, 0.05, 0.01]
print(f"Total number of rows before filtering: {len(all_flows_df)}", flush=True)

for threshold in thresholds:
    print(f"Processing threshold: {threshold}", flush=True)

    # Filter out flows below the threshold
    filtered_flows_df = all_flows_df[all_flows_df['value'] >= threshold].copy()

    # Compute total flow before redistribution
    total_flow_before = all_flows_df['value'].sum()
    print(f"Total flow before redistribution: {total_flow_before}", flush=True)

    # Country-pair totals before/after
    country_pair_totals_before = all_flows_df.groupby(['origin_country', 'destination_country'])['value'].sum()
    country_pair_totals_after = filtered_flows_df.groupby(['origin_country', 'destination_country'])['value'].sum()

    # Scaling factors (vectorized)
    scaling_factors = (country_pair_totals_before / country_pair_totals_after).replace([np.inf, -np.inf, np.nan], 0.0)
    scaling_factors = scaling_factors.rename('scale').reset_index()
    filtered_flows_df = filtered_flows_df.merge(scaling_factors, on=['origin_country', 'destination_country'], how='left')
    filtered_flows_df['scale'] = filtered_flows_df['scale'].fillna(0.0)
    filtered_flows_df['value'] = filtered_flows_df['value'] * filtered_flows_df['scale']
    filtered_flows_df = filtered_flows_df.drop(columns='scale')

    # Remove NUTS2 pairs whose total (post-redistribution) is still below threshold
    country_pair_totals = filtered_flows_df.groupby(['origin_nuts2', 'destination_nuts2'])['value'].sum()
    keep_pairs = country_pair_totals[country_pair_totals >= threshold].index
    filtered_flows_df = filtered_flows_df.set_index(['origin_nuts2', 'destination_nuts2']).loc[keep_pairs].reset_index()

    # Total flow after redistribution
    total_flow_after = filtered_flows_df['value'].sum()
    print(f"Total flow after redistribution: {total_flow_after} for threshold {threshold}", flush=True)

    # Percentage of total flow removed
    flow_removed_percentage = ((total_flow_before - total_flow_after) / total_flow_before) * 100.0
    print(f"Percentage of total flow removed: {flow_removed_percentage:.2f}% for threshold {threshold}", flush=True)

    # Final number of rows
    final_row_count = len(filtered_flows_df)
    print(f"Final number of rows: {final_row_count} for threshold {threshold}", flush=True)

    # Rows where both origin_id and destination_id are 0 (use AND)
    zero_id_rows = filtered_flows_df[(filtered_flows_df['origin_id'] == 0) & (filtered_flows_df['destination_id'] == 0)].shape[0]
    print(f"Number of rows where both origin_id and destination_id are 0: {zero_id_rows} for threshold {threshold}", flush=True)

    # Save the filtered flows to a new file
    thr_flow_out = Path(f"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industrial_OD/industrial_OD_threshold_{threshold}.parquet")
    thr_flow_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving filtered flows to: {thr_flow_out}", flush=True)
    filtered_flows_df.to_parquet(thr_flow_out, index=False)

    # Create outgoing flow columns for each commodity type
    outgoing_flows = (
        filtered_flows_df.groupby(["origin_id", "origin_sector"])["value"]
        .sum()
        .unstack(fill_value=0)
        .add_prefix("outgoing_")
    )

    # Create incoming flow columns for each commodity type (use destination_sector)
    incoming_flows = (
        filtered_flows_df.groupby(["destination_id", "destination_sector"])["value"]
        .sum()
        .unstack(fill_value=0)
        .add_prefix("incoming_")
    )

    # Merge outgoing and incoming flows into the combined GeoDataFrame
    combined_gdf_copy = combined_gdf.copy()
    combined_gdf_copy = combined_gdf_copy.merge(outgoing_flows, left_on="id", right_index=True, how="left")
    combined_gdf_copy = combined_gdf_copy.merge(incoming_flows, left_on="id", right_index=True, how="left")

    # Fill NaN values with 0 for the new flow columns
    flow_columns = [c for c in combined_gdf_copy.columns if c.startswith("outgoing_") or c.startswith("incoming_")]
    combined_gdf_copy[flow_columns] = combined_gdf_copy[flow_columns].fillna(0)

    # Remove industries where all flow columns are 0
    combined_gdf_copy = combined_gdf_copy[combined_gdf_copy[flow_columns].sum(axis=1) > 0]

    # Save the updated GeoDataFrame to a Parquet file (fix path typo)
    industries_out = Path(f"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industries_w_outputs/industries_with_flows_threshold_{threshold}.parquet")
    industries_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving updated GeoDataFrame with flows to: {industries_out}", flush=True)
    combined_gdf_copy.to_parquet(industries_out, index=False)

print("Processing complete.", flush=True)


