import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import glob
import matplotlib.pyplot as plt

AIRPORTS_PARQUET = Path("/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_airports_TENT.parquet")
PASSENGERS_DIR   = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/EUROSTAT/airports/Passengers/Air passenger transport routes between partner airports and main airports/")
OUT_OD_PARQUET   = Path("/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/airports_passenger_OD.parquet")

def load_airports() -> pd.DataFrame:
    gdf = gpd.read_parquet(AIRPORTS_PARQUET)
    # Expect columns: ICAO (or similar), name, geometry
    icao_col = next((c for c in ["icao","ICAO","icao_code","airport_icao"] if c in gdf.columns), None)
    if icao_col is None:
        raise KeyError("Airports parquet must contain an ICAO code column (e.g., 'ICAO').")
    gdf["ICAO"] = gdf[icao_col].astype(str)
    return gdf

def read_passenger_file(path: str) -> pd.DataFrame:
    # Try multiple separators (Eurostat files vary)
    for sep in ["\t", ";", ",", " "]:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str)
            # Heuristic: require airp_pr\TIME_PERIOD present
            if not any("airp_pr" in c.lower() for c in df.columns):
                continue
            return df
        except Exception:
            continue
    # Fallback: pandas auto-detect
    return pd.read_csv(path, dtype=str)

def load_passengers_2021_sum() -> pd.DataFrame:
    files = glob.glob(str(PASSENGERS_DIR / "*.tsv"))
    if not files:
        raise FileNotFoundError(f"No TSV files found in {PASSENGERS_DIR}")
    dfs = [read_passenger_file(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Identify the TIME_PERIOD column
    rep_col = next((c for c in df.columns if "airp_pr" in c.lower()), None)
    if rep_col is None and "airp_pr\\TIME_PERIOD" in df.columns:
        rep_col = "airp_pr\\TIME_PERIOD"
    if rep_col is None:
        raise KeyError("Could not find 'airp_pr\\TIME_PERIOD' column.")

    # Quarter columns: accept "2021-Q1..Q4" or "2021Q1..Q4"
    qcols = [c for c in df.columns if c.startswith("2021-")]
    qcols = sorted(set(qcols))
    if not qcols:
        raise KeyError("No 2021 quarterly columns found (expected 2021-Q1..Q4 or 2021Q1..Q4).")

    # Numeric values and row-wise sum
    for c in qcols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["value_2021_sum"] = df[qcols].sum(axis=1)

    route_token = df[rep_col].astype(str).str.split(",").str[-1].str.strip()
    icao_pairs = route_token.str.extract(r"^[A-Z]{2}_([A-Z0-9]{4})_[A-Z]{2}_([A-Z0-9]{4})$", expand=True)
    df["ICAO_OUT"] = icao_pairs[0].astype(str)
    df["ICAO_IN"]  = icao_pairs[1].astype(str)

    # Keep only rows with valid ICAOs parsed
    df = df[df["ICAO_OUT"].str.len() == 4]
    df = df[df["ICAO_IN"].str.len() == 4]

    print(df.head(), flush=True)

    # Optional filter: keep only relevant unit/measure if needed
    # df = df[df["unit"].astype(str) == "PAS"] if "unit" in df.columns else df

    # Aggregate to OD by sum
    od = (
        df.groupby(["ICAO_OUT", "ICAO_IN"], as_index=False)["value_2021_sum"]
          .sum()
          .rename(columns={"ICAO_OUT": "from_id", "ICAO_IN": "to_id", "value_2021_sum": "value"})
    )
    return od

def build_airport_passenger_od_2021():
    airports_gdf = load_airports()
    od = load_passengers_2021_sum()

    # Keep only ODs where both ICAOs exist in airports list
    valid = set(airports_gdf["ICAO"].astype(str))
    od = od[od["from_id"].isin(valid) & od["to_id"].isin(valid)].copy()

    # Remove zero-value rows before saving
    od["value"] = pd.to_numeric(od["value"], errors="coerce").fillna(0.0)
    od = od[od["value"] > 0].copy()

    OUT_OD_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    od.to_parquet(OUT_OD_PARQUET, index=False)
    print(f"[air-OD] Saved {OUT_OD_PARQUET} with {len(od)} rows.", flush=True)
    
    return airports_gdf, od

# -------------------------------------------------------------------
# GeoBubble plotting function
# -------------------------------------------------------------------

def create_geobubble_plot(gdf, data_col, title, output_file, xlim=None, ylim=None):
    """Create a bubble map for airport passenger flows."""
    if gdf is None or gdf.empty:
        print(f"No data to plot for {title}")
        return

    # To WGS84
    try:
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    # Drop invalid geometries
    gdf = gdf[gdf['geometry'].notna() & (~gdf['geometry'].is_empty)].copy()
    if gdf.empty:
        print(f"All geometries invalid/empty for {title}")
        return

    # Values and bubble sizes
    gdf[data_col] = pd.to_numeric(gdf[data_col], errors='coerce').fillna(0.0)
    gdf['bubble_size'] = gdf[data_col] / 1e6

    valid_tent_types = ['A','B','C','E','G','I','J','K','L']
    gdf['TENT_type'] = gdf.get('CORRIDORS', pd.Series(index=gdf.index, dtype=object)).fillna('Non-assigned').astype(str).str[0]
    gdf['TENT_type'] = gdf['TENT_type'].where(gdf['TENT_type'].isin(valid_tent_types), 'Non-assigned')

    corridor_colors = ['#0080C0','#E91E8C','#00A651','#FF69B4','#FFD700','#00BFFF','#8B4789','#8B4513','#228B22']
    color_map = {k: v for k, v in zip(valid_tent_types, corridor_colors)}
    color_map['Non-assigned'] = '#b3b3b3'
    gdf['color'] = gdf['TENT_type'].map(color_map).fillna('#b3b3b3')

    max_bs = gdf['bubble_size'].max()
    gdf['normalized_bubble_size'] = (50 if (max_bs == 0 or np.isnan(max_bs)) else (gdf['bubble_size'] / max_bs) * 600)

    # Load Europe shape
    countries_path = r"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/admin/ne_10m/ne_10m_admin_0_countries.shp"
    europe_shape = gpd.read_file(countries_path)
    europe_bounds = {"xmin": -12, "xmax": 32, "ymin": 35, "ymax": 72}
    europe_shape = europe_shape.cx[europe_bounds["xmin"]:europe_bounds["xmax"], europe_bounds["ymin"]:europe_bounds["ymax"]]

    fig, ax = plt.subplots(1, 1, figsize=(13, 11))
    europe_shape.plot(ax=ax, color='#f2f2f2', edgecolor='black', linewidth=0.3, alpha=0.8)

    # Representative points for polygons/lines
    try:
        is_point = gdf.geometry.geom_type.isin(['Point', 'MultiPoint'])
    except Exception:
        is_point = pd.Series(False, index=gdf.index)

    rep_points = gdf.geometry.where(is_point, gdf.geometry.representative_point())
    xs = rep_points.x.values
    ys = rep_points.y.values

    ax.scatter(xs, ys, s=gdf['normalized_bubble_size'].values, c=gdf['color'].tolist(), alpha=0.8, linewidths=0.35, edgecolors='black')

    ax.set_xlim(xlim if xlim is not None else (-12, 32))
    ax.set_ylim(ylim if ylim is not None else (35, 72))
    ax.set_aspect('auto')
    ax.autoscale(enable=False)

    title_fs, label_fs, legend_title_fs, legend_label_fs, tick_fs = 18, 18, 16, 14, 13
    ax.set_title(title, fontsize=title_fs, pad=16)
    ax.set_xlabel("Longitude", fontsize=label_fs, labelpad=10)
    ax.set_ylabel("Latitude", fontsize=label_fs, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)

    legend_order = valid_tent_types + ['Non-assigned']
    color_handles = [
        plt.Line2D([0],[0], marker='o', markerfacecolor=color_map[t], markeredgecolor='black',
                   markeredgewidth=0.5, color='w', label=t, markersize=12)
        for t in legend_order if t in color_map
    ]
    color_legend = ax.legend(handles=color_handles, title="TENT Corridor", loc='upper right',
                             title_fontsize=legend_title_fs, fontsize=legend_label_fs, frameon=True)

    if max_bs == 0 or np.isnan(max_bs):
        size_vals = [1, 2, 3]
    else:
        size_vals = [max_bs * f for f in (0.1, 0.5, 1.0)]
    size_labels = [f"{v:,.0f}M" for v in size_vals]

    def scale_marker(v):
        if max_bs == 0 or np.isnan(max_bs):
            return 10
        return 10 + (v / max_bs) * 30

    size_handles = [
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#808080', markeredgecolor='black',
                   markeredgewidth=0.5, markersize=scale_marker(v), label=l)
        for v, l in zip(size_vals, size_labels)
    ]

    ax.add_artist(color_legend)
    size_legend = ax.legend(handles=size_handles, title="Passengers", loc='lower left',
                           title_fontsize=legend_title_fs, fontsize=legend_label_fs, frameon=True)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved GeoBubble plot to {out_path}")

if __name__ == "__main__":
    airports_gdf, od = build_airport_passenger_od_2021()
    
    # Aggregate outgoing and incoming flows
    outgoing = od.groupby('from_id', as_index=False)['value'].sum().rename(columns={'value': 'outgoing_passengers'})
    incoming = od.groupby('to_id', as_index=False)['value'].sum().rename(columns={'value': 'incoming_passengers'})
    
    # Merge with airports
    airports_gdf = airports_gdf.merge(outgoing, left_on='ICAO', right_on='from_id', how='left')
    airports_gdf = airports_gdf.merge(incoming, left_on='ICAO', right_on='to_id', how='left')
    airports_gdf['outgoing_passengers'] = airports_gdf['outgoing_passengers'].fillna(0.0)
    airports_gdf['incoming_passengers'] = airports_gdf['incoming_passengers'].fillna(0.0)
    
    print(f"Total outgoing passengers: {airports_gdf['outgoing_passengers'].sum():.0f}", flush=True)
    print(f"Total incoming passengers: {airports_gdf['incoming_passengers'].sum():.0f}", flush=True)
    
    # Output directory for plots
    plot_dir = Path("/soge-home/projects/mistral/miraca/incoming_data/plots/heatmaps/airports")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate bubble plots
    create_geobubble_plot(
        airports_gdf[airports_gdf['outgoing_passengers'] > 0].copy(),
        'outgoing_passengers',
        'Outgoing Air Passenger Flows by Airport',
        plot_dir / 'outgoing_passengers_bubble.png'
    )
    
    create_geobubble_plot(
        airports_gdf[airports_gdf['incoming_passengers'] > 0].copy(),
        'incoming_passengers',
        'Incoming Air Passenger Flows by Airport',
        plot_dir / 'incoming_passengers_bubble.png'
    )
    
    print("GeoBubble plot generation complete!", flush=True)
