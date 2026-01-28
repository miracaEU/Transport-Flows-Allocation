import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from shapely.geometry import Point
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Inputs
PORTS_PARQUET = Path("/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_ports_TENT.parquet")
MAR_EDGES_GPKG = Path("/soge-home/projects/mistral/incoming_data/verschuur-2022-global-port-supply-chains/Network/edges_maritime_corrected.gpkg")
MAR_NODES_GPKG = Path("/soge-home/projects/mistral/incoming_data/verschuur-2022-global-port-supply-chains/Network/nodes_maritime.gpkg")
PASSENGERS_TSV = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/EUROSTAT/ports/Passengers/Passengers transported fromto the main ports by direction.tsv")

OUT_OD_PARQUET = Path("/soge-home/projects/mistral/miraca/processed_data/lifelines_OD/maritime_passenger_OD.parquet")

def load_ports() -> gpd.GeoDataFrame:
    ports = gpd.read_parquet(PORTS_PARQUET)
    if ports.crs is None:
        ports = ports.set_crs(4326)
    elif ports.crs.to_epsg() != 4326:
        ports = ports.to_crs(4326)
    # Expect columns: port_code, port_name, country, geometry
    needed = {"port_code", "geometry"}
    miss = [c for c in needed if c not in ports.columns]
    if miss:
        raise KeyError(f"Ports parquet missing columns {miss}")
    ports["port_code"] = ports["port_code"].astype(str)
    return ports

def load_maritime_network():
    edges = gpd.read_file(MAR_EDGES_GPKG).to_crs(4326)
    nodes = gpd.read_file(MAR_NODES_GPKG).to_crs(4326)
    # Expect node id col and edge endpoints
    nid_col = "id" if "id" in nodes.columns else ("node_id" if "node_id" in nodes.columns else None)
    if nid_col is None:
        raise KeyError("nodes_maritime.gpkg must contain 'id' or 'node_id'.")
    sid = "from_id" if "from_id" in edges.columns else ("source" if "source" in edges.columns else None)
    tid = "to_id"   if "to_id"   in edges.columns else ("target" if "target" in edges.columns else None)
    if sid is None or tid is None:
        raise KeyError("edges_maritime_corrected.gpkg must contain 'from_id'/'to_id' or 'source'/'target'.")
    # Build weight from straight-line distance between endpoints (km) if no length
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1; dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    # Compute fallback weights
    w = pd.Series(np.nan, index=edges.index, dtype=float)
    if "length" in edges.columns:
        w = pd.to_numeric(edges["length"], errors="coerce")
    bad = ~np.isfinite(w) | (w <= 0)
    # endpoints from geometry
    def endpoints(geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == "LineString":
            (x1, y1) = geom.coords[0]; (x2, y2) = geom.coords[-1]
            return (y1, x1, y2, x2)
        if geom.geom_type == "MultiLineString":
            ls = list(geom.geoms)[0]
            (x1, y1) = ls.coords[0]; (x2, y2) = ls.coords[-1]
            return (y1, x1, y2, x2)
        return None
    ep = edges["geometry"].apply(endpoints)
    hv = ep.apply(lambda t: _haversine(*t) if isinstance(t, tuple) else np.nan)
    w[bad] = hv[bad]
    w = w.replace([np.inf, -np.inf], np.nan).fillna(1.0).astype(float).clip(lower=1e-3)
    edges["weight_km"] = w
    # Build graph
    G = nx.Graph()
    for _, e in edges.iterrows():
        u = str(e[sid]); v = str(e[tid])
        G.add_edge(u, v, weight=float(e["weight_km"]))
    # Node coords
    nodes["id_str"] = nodes[nid_col].astype(str)
    coords = dict(zip(nodes["id_str"], zip(nodes.geometry.x.astype(float), nodes.geometry.y.astype(float))))
    return G, nodes, coords, nid_col

def load_passengers() -> pd.DataFrame:
    df = pd.read_csv(PASSENGERS_TSV, sep="\t", dtype=str)
    # Drop interleaved flag columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    # Replace Eurostat missing placeholder ":" with NaN
    df = df.replace(":", np.nan)

    if "direct" not in df.columns or "tra_cov" not in df.columns:
        raise KeyError("Passenger TSV must contain 'direct' and 'tra_cov'.")

    # Keep only 2019 quarters (Q1..Q4) and sum them
    year_cols = [c for c in df.columns if c.startswith("2019-") and c.endswith(tuple(["Q1","Q2","Q3","Q4"]))]
    year_cols = sorted(set(year_cols) | {c for c in df.columns if c.upper().startswith("2019-Q")})
    if not year_cols:
        raise KeyError("No 2019 quarterly columns found (expected 2019-Q1..Q4 or 2019Q1..Q4).")

    # Filter INTL_IEU28 and sum across 2019 quarters
    df = df[df["tra_cov"].astype(str) == "INTL_IEU28"].copy()
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["value_2019_sum"] = df[year_cols].sum(axis=1)

    # Extract port_code from rep_mar\TIME_PERIOD last 5 chars
    rep_col = next((c for c in df.columns if "rep_mar" in c.lower()), None)
    if rep_col is None and "rep_mar\\TIME_PERIOD" in df.columns:
        rep_col = "rep_mar\\TIME_PERIOD"
    if rep_col is None:
        raise KeyError("Could not find 'rep_mar\\TIME_PERIOD' column in passenger TSV.")
    df["port_code"] = df[rep_col].astype(str).str[-5:]

    # Aggregate by port and direction using the 2019 sum
    df["direct"] = df["direct"].astype(str)
    agg = (
        df.groupby(["port_code", "direct"], as_index=False)["value_2019_sum"]
          .sum()
          .rename(columns={"value_2019_sum": "value"})
    )

    wide = agg.pivot_table(index="port_code", columns="direct", values="value", fill_value=0.0).reset_index()
    wide.columns = ["port_code"] + [str(c).upper() for c in wide.columns.tolist()[1:]]
    for c in ("IN", "OUT"):
        if c not in wide.columns:
            wide[c] = 0.0
    return wide

def match_ports_to_network(ports_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame, nid_col: str) -> pd.DataFrame:
    # Nearest maritime node per port (EPSG:3857 for metric nearest)
    ports_proj = ports_gdf.to_crs(3857)
    nodes_proj = nodes_gdf.to_crs(3857)
    # Build KDTree
    nx_arr = nodes_proj.geometry.x.to_numpy(); ny_arr = nodes_proj.geometry.y.to_numpy()
    px_arr = ports_proj.geometry.x.to_numpy(); py_arr = ports_proj.geometry.y.to_numpy()
    # Simple brute-force nearest (fallback without scipy)
    nearest_idx = []
    for x, y in zip(px_arr, py_arr):
        d2 = (nx_arr - x)**2 + (ny_arr - y)**2
        j = int(np.argmin(d2))
        nearest_idx.append(nodes_proj.index[j])
    map_df = pd.DataFrame({
        "port_code": ports_gdf["port_code"].astype(str).to_numpy(),
        "nearest_node_id": nodes_gdf.loc[nearest_idx, nid_col].astype(str).to_numpy()
    })
    return map_df

def compute_port_distances(G: nx.Graph, port_to_node: pd.DataFrame) -> pd.DataFrame:
    # All-pairs shortest path distances (km) between port nodes in the maritime network
    codes = port_to_node["port_code"].astype(str).tolist()
    node_ids = port_to_node["nearest_node_id"].astype(str).tolist()
    # Map code -> node id
    code2node = dict(zip(codes, node_ids))
    pairs = []
    for i, ci in enumerate(codes):
        u = code2node[ci]
        for j, cj in enumerate(codes):
            if i == j:
                continue
            v = code2node[cj]
            try:
                d = nx.shortest_path_length(G, source=u, target=v, weight="weight")
            except nx.NetworkXNoPath:
                d = np.inf
            pairs.append((ci, cj, d))
    dist_df = pd.DataFrame(pairs, columns=["origin_port", "dest_port", "dist_km"])
    dist_df["dist_km"] = pd.to_numeric(dist_df["dist_km"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return dist_df.dropna(subset=["dist_km"])

def radiation_od(ports_df: pd.DataFrame, dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Radiation model (Simini et al. 2012):
    T_ij = O_i * (m_i * n_j) / ((m_i + s_ij) * (m_i + n_j))
    Here:
      O_i = OUT_i (origin emissions)
      n_j = IN_j  (destination attraction)
      m_i = OUT_i (proxy mass at i)
      s_ij = sum of attractions n_k for all ports k with dist(i,k) < dist(i,j) and k != i,j
    """
    # Prepare OUT/IN per port
    ports_df = ports_df.copy()
    ports_df["OUT"] = pd.to_numeric(ports_df["OUT"], errors="coerce").fillna(0.0)
    ports_df["IN"]  = pd.to_numeric(ports_df["IN"],  errors="coerce").fillna(0.0)
    # Index for quick lookup
    out_map = ports_df.set_index("port_code")["OUT"].to_dict()
    in_map  = ports_df.set_index("port_code")["IN"].to_dict()

    # Build distance lists per origin
    dist_df = dist_df.copy()
    dist_df = dist_df[dist_df["dist_km"] > 0]
    # Group by origin, sort by distance
    od_rows = []
    for o, g in dist_df.groupby("origin_port"):
        O_i = float(out_map.get(o, 0.0))
        m_i = O_i
        if O_i <= 0:
            continue
        g = g.sort_values("dist_km")
        # cumulative sum of destination IN excluding current j
        # s_ij for j is sum of IN for all k with dist < dist(o,j)
        in_vals = g["dest_port"].map(in_map).astype(float).fillna(0.0).to_numpy()
        dvals = g["dist_km"].to_numpy()
        # Compute s_ij using cumulative IN over sorted distances
        cumsum_in = np.cumsum(in_vals)
        for idx in range(len(g)):
            j = g.iloc[idx]["dest_port"]
            n_j = float(in_map.get(j, 0.0))
            if n_j <= 0:
                continue
            s_ij = float(cumsum_in[idx-1]) if idx > 0 else 0.0
            denom = (m_i + s_ij) * (m_i + n_j)
            Tij = 0.0 if denom <= 0 else O_i * (m_i * n_j) / denom
            if Tij > 0:
                od_rows.append((o, j, Tij))
    od_df = pd.DataFrame(od_rows, columns=["from_id", "to_id", "value"])
    return od_df

def build_ports_passenger_od():
    # Load
    ports_gdf = load_ports()
    G, nodes_gdf, _, nid_col = load_maritime_network()
    passengers = load_passengers()  # port_code, IN, OUT

    # Match ports to maritime nodes
    mapping = match_ports_to_network(ports_gdf, nodes_gdf, nid_col)
    # Compute network distances between ports
    dist_df = compute_port_distances(G, mapping)

    # Radiation model OD
    od_df = radiation_od(passengers, dist_df)

    # Remove zero-value rows before saving
    od_df["value"] = pd.to_numeric(od_df["value"], errors="coerce").fillna(0.0)
    od_df = od_df[od_df["value"] > 0].copy()

    # Attach names if available
    name_map = ports_gdf.set_index("port_code").get("port_name", pd.Series(index=[])).to_dict() if "port_name" in ports_gdf.columns else {}
    od_df["origin_name"] = od_df["from_id"].map(name_map)
    od_df["destination_name"] = od_df["to_id"].map(name_map)

    OUT_OD_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    od_df.to_parquet(OUT_OD_PARQUET, index=False)
    print(f"[ports-OD] Saved {OUT_OD_PARQUET} with {len(od_df)} rows.", flush=True)
    
    return ports_gdf, od_df

# -------------------------------------------------------------------
# GeoBubble plotting function
# -------------------------------------------------------------------

def create_geobubble_plot(gdf, data_col, title, output_file, xlim=None, ylim=None):
    """Create a bubble map for port passenger flows."""
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
    ports_gdf, od_df = build_ports_passenger_od()
    
    # Aggregate outgoing and incoming flows
    outgoing = od_df.groupby('from_id', as_index=False)['value'].sum().rename(columns={'value': 'outgoing_passengers'})
    incoming = od_df.groupby('to_id', as_index=False)['value'].sum().rename(columns={'value': 'incoming_passengers'})
    
    # Merge with ports
    ports_gdf = ports_gdf.merge(outgoing, left_on='port_code', right_on='from_id', how='left')
    ports_gdf = ports_gdf.merge(incoming, left_on='port_code', right_on='to_id', how='left')
    ports_gdf['outgoing_passengers'] = ports_gdf['outgoing_passengers'].fillna(0.0)
    ports_gdf['incoming_passengers'] = ports_gdf['incoming_passengers'].fillna(0.0)
    
    print(f"Total outgoing passengers: {ports_gdf['outgoing_passengers'].sum():.0f}", flush=True)
    print(f"Total incoming passengers: {ports_gdf['incoming_passengers'].sum():.0f}", flush=True)
    
    # Output directory for plots
    plot_dir = Path("/soge-home/projects/mistral/miraca/incoming_data/plots/heatmaps/ports")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate bubble plots
    create_geobubble_plot(
        ports_gdf[ports_gdf['outgoing_passengers'] > 0].copy(),
        'outgoing_passengers',
        'Outgoing Maritime Passenger Flows by Port (2019)',
        plot_dir / 'outgoing_passengers_bubble.png'
    )
    
    create_geobubble_plot(
        ports_gdf[ports_gdf['incoming_passengers'] > 0].copy(),
        'incoming_passengers',
        'Incoming Maritime Passenger Flows by Port (2019)',
        plot_dir / 'incoming_passengers_bubble.png'
    )
    
    print("GeoBubble plot generation complete!", flush=True)