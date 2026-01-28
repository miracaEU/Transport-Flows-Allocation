import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from matplotlib.colors import LogNorm  # Import LogNorm
import pandas as pd

flow_file = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industrial_OD/industrial_OD.parquet")
flows_df = pd.read_parquet(flow_file)
# Load nuts2 boundaries
nuts2_path = r"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/admin/NUTS/NUTS2_v2016.parquet"
nuts2 = gpd.read_parquet(nuts2_path)
# Load the combined GeoDataFrame
output_file = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/industries_w_outputs/industries_with_flows.parquet")
print(f"Reading GeoDataFrame from: {output_file}", flush=True)
combined_gdf = gpd.read_parquet(output_file)

# Load country boundaries
countries_path = r"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/admin/ne_10m/ne_10m_admin_0_countries.shp"
europe_shape = gpd.read_file(countries_path)

# Define the resolution of the heatmap grid
heatmap_resolution = 0.15  # Degrees (adjust as needed)
# Define the bounding box for Europe
europe_bounds = {
    "xmin": -12,  # Westernmost longitude
    "xmax": 32,   # Easternmost longitude
    "ymin": 35,   # Southernmost latitude
    "ymax": 72    # Northernmost latitude
}

# Filter the GeoDataFrame to include only points within Europe
europe_shape = europe_shape.cx[europe_bounds["xmin"]:europe_bounds["xmax"], europe_bounds["ymin"]:europe_bounds["ymax"]]

# Update the grid bounds to match Europe
xmin, xmax, ymin, ymax = europe_bounds["xmin"], europe_bounds["xmax"], europe_bounds["ymin"], europe_bounds["ymax"]
x_coords = np.arange(xmin, xmax, heatmap_resolution)
y_coords = np.arange(ymin, ymax, heatmap_resolution)
x_grid, y_grid = np.meshgrid(x_coords, y_coords)

# Generate heatmaps for each outgoing and incoming commodity type
flow_columns = [col for col in combined_gdf.columns if col.startswith("outgoing_") or col.startswith("incoming_")]

# Generate heatmaps for each outgoing and incoming commodity type
for flow_column in flow_columns:
    print(f"Generating heatmap for: {flow_column}", flush=True)

    # Initialize an empty grid for the heatmap
    heatmap = np.zeros(x_grid.shape)
    # Populate the heatmap grid with flow values
    for _, row in combined_gdf.iterrows():
        x_idx = np.searchsorted(x_coords, row['x_4258']) - 1
        y_idx = np.searchsorted(y_coords, row['y_4258']) - 1       
        if 0 <= x_idx < heatmap.shape[1] and 0 <= y_idx < heatmap.shape[0]:
            heatmap[y_idx, x_idx] += row[flow_column]

    # Plot the heatmap
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))  # Increase figure size for a larger map

    # Plot the shape of Europe
    europe_shape.boundary.plot(ax=ax, color="black", linewidth=0.5)
    if flow_column == 'outgoing_COMMERCIAL':
    
        # Plot the heatmap
        c = ax.imshow(
            heatmap,
            extent=(xmin, xmax, ymin, ymax),
            origin="lower",
            cmap="ocean_r",  # Colormap
            interpolation='bicubic',
            vmin=0.0, 
            vmax=50,  
            alpha=0.8  # Transparency
        )
    else:
        # Plot the heatmap
        c = ax.imshow(
            heatmap,
            extent=(xmin, xmax, ymin, ymax),
            origin="lower",
            cmap="ocean_r",  # Colormap
            interpolation='bicubic',
            vmin=0.0, 
            vmax=250,  
            alpha=0.8  # Transparency
        )

    # Add a colorbar with reduced size
    cbar = plt.colorbar(c, ax=ax, shrink=0.8)  # Shrink the color bar to 80% of its default size
    cbar.set_label(f"{flow_column} ", fontsize=12)

    # Add title and labels
    ax.set_title(f"Heatmap of {flow_column}", fontsize=16)  # Increase title font size
    ax.set_xlabel("Longitude", fontsize=14)  # Increase label font size
    ax.set_ylabel("Latitude", fontsize=14)  # Increase label font size

    # Save the heatmap to a file
    heatmap_file = Path(f"/soge-home/projects/mistral/miraca/incoming_data/plots/heatmaps/{flow_column}_heatmap.png")
    heatmap_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(heatmap_file, dpi=300)
    plt.close(fig)

    print(f"Saved heatmap for {flow_column} to: {heatmap_file}", flush=True)

# Detect value column in flows_df

val_col = 'total_flow'

# Ensure required NUTS2 code columns exist
if not {'origin_nuts2', 'destination_nuts2'}.issubset(flows_df.columns):
    raise ValueError("flows_df must contain 'origin_nuts2' and 'destination_nuts2' columns.")

# Normalize codes as strings
flows_df['origin_nuts2'] = flows_df['origin_nuts2'].astype(str).str.strip()
flows_df['destination_nuts2'] = flows_df['destination_nuts2'].astype(str).str.strip()

# Detect the NUTS2 code column in the polygons
nuts_code_col = 'NUTS_ID'
nuts2 = nuts2[[nuts_code_col, 'geometry']].drop_duplicates(subset=[nuts_code_col]).copy()
nuts2[nuts_code_col] = nuts2[nuts_code_col].astype(str).str.strip()

# Aggregate totals
out_tot = (
    flows_df.groupby('origin_nuts2', dropna=False)[val_col]
    .sum()
    .reset_index()
    .rename(columns={'origin_nuts2': nuts_code_col, val_col: 'OUT_TOTAL'})
)
in_tot = (
    flows_df.groupby('destination_nuts2', dropna=False)[val_col]
    .sum()
    .reset_index()
    .rename(columns={'destination_nuts2': nuts_code_col, val_col: 'IN_TOTAL'})
)

# Merge with NUTS2 polygons
nuts2_out = nuts2.merge(out_tot, on=nuts_code_col, how='left')
nuts2_in = nuts2.merge(in_tot, on=nuts_code_col, how='left')
nuts2_out['OUT_TOTAL'] = pd.to_numeric(nuts2_out['OUT_TOTAL'], errors='coerce').fillna(0.0)
nuts2_in['IN_TOTAL'] = pd.to_numeric(nuts2_in['IN_TOTAL'], errors='coerce').fillna(0.0)

# ----- Plot maps -----
out_dir = Path("/soge-home/projects/mistral/miraca/incoming_data/plots/heatmaps/nuts2_maps")
out_dir.mkdir(parents=True, exist_ok=True)

# Outgoing map
fig, ax = plt.subplots(1, 1, figsize=(14, 12))
try:
    # requires mapclassify
    nuts2_out.plot(ax=ax, column='OUT_TOTAL', cmap='OrRd', linewidth=0.2, edgecolor='black',
                   legend=True, scheme='quantiles', k=7)
except Exception:
    nuts2_out.plot(ax=ax, column='OUT_TOTAL', cmap='OrRd', linewidth=0.2, edgecolor='black',
                   legend=True)
ax.set_title("Total Outgoing Flow by NUTS2")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
out_png = out_dir / "nuts2_outgoing.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_png}", flush=True)

# Incoming map
fig, ax = plt.subplots(1, 1, figsize=(14, 12))
try:
    nuts2_in.plot(ax=ax, column='IN_TOTAL', cmap='PuBu', linewidth=0.2, edgecolor='black',
                  legend=True, scheme='quantiles', k=7)
except Exception:
    nuts2_in.plot(ax=ax, column='IN_TOTAL', cmap='PuBu', linewidth=0.2, edgecolor='black',
                  legend=True)
ax.set_title("Total Incoming Flow by NUTS2")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
in_png = out_dir / "nuts2_incoming.png"
plt.savefig(in_png, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {in_png}", flush=True)

# Save aggregated GeoParquet
nuts2_out.to_parquet(out_dir / "nuts2_outgoing.parquet", index=False)
nuts2_in.to_parquet(out_dir / "nuts2_incoming.parquet", index=False)
print(f"Saved aggregated GeoParquet to: {out_dir}", flush=True)

    