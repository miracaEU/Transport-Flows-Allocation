import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

# Load passenger flows
flow_file = Path("/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/rail_passenger_OD.parquet")
flows_df = pd.read_parquet(flow_file)

# Load population nodes
pop_nodes_file = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/networks/transport/railways/europe_train_stations.parquet")
print(f"Reading population nodes from: {pop_nodes_file}", flush=True)
pop_nodes_gdf = gpd.read_parquet(pop_nodes_file)

# Load country boundaries
countries_path = r"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/admin/ne_10m/ne_10m_admin_0_countries.shp"
europe_shape = gpd.read_file(countries_path)

# Define the bounding box for Europe
europe_bounds = {
    "xmin": -12,  # Westernmost longitude
    "xmax": 32,   # Easternmost longitude
    "ymin": 35,   # Southernmost latitude
    "ymax": 72    # Northernmost latitude
}

# Filter the GeoDataFrame to include only points within Europe
europe_shape = europe_shape.cx[europe_bounds["xmin"]:europe_bounds["xmax"], europe_bounds["ymin"]:europe_bounds["ymax"]]

# Ensure flows have origin_node and dest_node
if 'origin_node' not in flows_df.columns or 'dest_node' not in flows_df.columns:
    raise ValueError("flows_df must contain 'origin_node' and 'dest_node' columns.")

# Ensure trips column exists
trip_col = 'trips' if 'trips' in flows_df.columns else 'value'
if trip_col not in flows_df.columns:
    raise ValueError(f"flows_df must contain '{trip_col}' column.")

# Ensure population nodes have id and geometry
if 'id' not in pop_nodes_gdf.columns:
    # Try to find an ID column
    id_candidates = [c for c in pop_nodes_gdf.columns if c.lower() in ['id', 'node_id', 'pop_id']]
    if id_candidates:
        pop_nodes_gdf = pop_nodes_gdf.rename(columns={id_candidates[0]: 'id'})
    else:
        raise ValueError("pop_nodes_gdf must contain an 'id' column.")

# Ensure geometry is in lat/lon (EPSG:4326)
if pop_nodes_gdf.crs is None:
    pop_nodes_gdf.set_crs("EPSG:4326", inplace=True)
elif pop_nodes_gdf.crs.to_epsg() != 4326:
    pop_nodes_gdf = pop_nodes_gdf.to_crs("EPSG:4326")

# Extract x, y coordinates
pop_nodes_gdf['x'] = pop_nodes_gdf.geometry.x
pop_nodes_gdf['y'] = pop_nodes_gdf.geometry.y

# Aggregate outgoing flows by origin_node
outgoing_flows = flows_df.groupby('origin_node', as_index=False)[trip_col].sum()
outgoing_flows = outgoing_flows.rename(columns={trip_col: 'outgoing_trips'})

# Aggregate incoming flows by dest_node
incoming_flows = flows_df.groupby('dest_node', as_index=False)[trip_col].sum()
incoming_flows = incoming_flows.rename(columns={trip_col: 'incoming_trips'})

# Merge with population nodes
pop_nodes_gdf = pop_nodes_gdf.merge(outgoing_flows, left_on='id', right_on='origin_node', how='left')
pop_nodes_gdf = pop_nodes_gdf.merge(incoming_flows, left_on='id', right_on='dest_node', how='left')

# Fill NaN values with 0
pop_nodes_gdf['outgoing_trips'] = pop_nodes_gdf['outgoing_trips'].fillna(0.0)/365
pop_nodes_gdf['incoming_trips'] = pop_nodes_gdf['incoming_trips'].fillna(0.0)/365

print(f"Total outgoing trips: {pop_nodes_gdf['outgoing_trips'].sum():.0f}", flush=True)
print(f"Total incoming trips: {pop_nodes_gdf['incoming_trips'].sum():.0f}", flush=True)

# Define the resolution of the heatmap grid
heatmap_resolution = 0.15  # Degrees (adjust as needed)

# Update the grid bounds to match Europe
xmin, xmax, ymin, ymax = europe_bounds["xmin"], europe_bounds["xmax"], europe_bounds["ymin"], europe_bounds["ymax"]
x_coords = np.arange(xmin, xmax, heatmap_resolution)
y_coords = np.arange(ymin, ymax, heatmap_resolution)
x_grid, y_grid = np.meshgrid(x_coords, y_coords)

# Output directory
out_dir = Path("/soge-home/projects/mistral/miraca/incoming_data/plots/heatmaps/passengers")
out_dir.mkdir(parents=True, exist_ok=True)

# Generate heatmaps for outgoing and incoming trips
for flow_type in ['outgoing_trips', 'incoming_trips']:
    print(f"Generating heatmap for: {flow_type}", flush=True)
    
    # Initialize an empty grid for the heatmap
    heatmap = np.zeros(x_grid.shape)
    
    # Populate the heatmap grid with flow values
    for _, row in pop_nodes_gdf.iterrows():
        x_idx = np.searchsorted(x_coords, row['x']) - 1
        y_idx = np.searchsorted(y_coords, row['y']) - 1
        if 0 <= x_idx < heatmap.shape[1] and 0 <= y_idx < heatmap.shape[0]:
            heatmap[y_idx, x_idx] += row[flow_type]
    
    # Plot the heatmap
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Plot the shape of Europe
    europe_shape.boundary.plot(ax=ax, color="black", linewidth=0.5)
    
    # Determine vmax based on data
    vmax_val = np.percentile(heatmap[heatmap > 0], 95) if (heatmap > 0).any() else heatmap.max()
    
    # Plot the heatmap
    c = ax.imshow(
        heatmap,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        cmap="ocean_r",  # Colormap for passenger flows
        interpolation='bicubic',
        vmin=0.0,
        vmax=vmax_val,
        alpha=0.8  # Transparency
    )
    
    # Add a colorbar
    cbar = plt.colorbar(c, ax=ax, shrink=0.8)
    cbar.set_label(f"{flow_type.replace('_', ' ').title()} (trips/day)", fontsize=12)
    
    # Add title and labels
    ax.set_title(f"Heatmap of {flow_type.replace('_', ' ').title()} (Railways)", fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    
    # Save the heatmap
    heatmap_file = out_dir / f"{flow_type}_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved heatmap for {flow_type} to: {heatmap_file}", flush=True)

print("Passenger heatmap generation complete!", flush=True)
