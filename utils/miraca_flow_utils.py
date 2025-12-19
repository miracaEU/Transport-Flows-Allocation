
import warnings
from itertools import chain
from collections import defaultdict
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import igraph as ig
import uuid
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines

def plot_bubble_map(ax, ports_gdf, flow_column, corridor_col, country_geometry, country_name, 
                    flow_type, mode, outpath, scale_factor=100, flow_values = [100, 10000, 100000, 1000000], transport_mode='maritime'):
    """
    Create a bubble map visualization of port flows.
    
    Parameters:
    -----------
    ports_gdf : GeoDataFrame
        Ports data with geometry and flow columns
    flow_column : str
        Name of column containing flow values
    country_geometry : GeoDataFrame
        Country boundary to plot
    country_name : str
        Name of country for title
    flow_type : str
        'Inbound' or 'Outbound'
    mode : str
        'Freight' or 'Passenger'
    outpath : Path
        Directory to save output
    scale_factor : float
        Divisor for bubble size scaling (default: 100)
    flow_values : list
        List of flow values for legend bubbles
    transport_mode : str
        'maritime' (default) or 'air' (for transport mode in title/filename)
    """
    if ports_gdf[flow_column].sum() == 0:
        print(f"No {transport_mode.lower()} {mode.lower()} {flow_type.lower()} flows to visualize")
        return
    
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
        created_fig = True
    
    # Plot country boundary
    country_geometry.plot(ax=ax, facecolor='lightgray', edgecolor='black', linewidth=1, alpha=0.5)
    
    # Filter ports with flow
    ports_with_flow = ports_gdf[ports_gdf[flow_column] > 0].copy()
    
    # Use logarithmic scale for bubble sizes
    ports_with_flow['bubble_size'] = np.log10(ports_with_flow[flow_column] + 1) * scale_factor
    
    # Robust corridor column detection (case-insensitive, plural/singular)
    corridor_col_candidates = ['CORRIDORS', 'CORRIDOR']
    corridor_col = next((c for c in corridor_col_candidates if c in ports_gdf.columns), None)
    if corridor_col is None:
        # Try case-insensitive match
        corridor_col = next((c for c in ports_gdf.columns if c.upper() in [col.upper() for col in corridor_col_candidates]), None)
    
    # Only proceed with corridor logic if corridor_col is valid
    if corridor_col:
        ports_with_flow[corridor_col] = ports_with_flow[corridor_col].fillna('Not-in-corridor')
        def corridor_key(val):
            nullish = [None, '', 'nan', 'NA', 'N/A', 'null', 'None']
            nullish_upper = set(str(v).upper() for v in nullish if v is not None)
            if pd.isna(val) or (isinstance(val, str) and val.strip().upper() in nullish_upper) or val == 'Not-in-corridor':
                return 'U'
            return val[0] if isinstance(val, str) and len(val) > 0 else 'U'
        ports_with_flow['corr_key'] = ports_with_flow[corridor_col].apply(corridor_key)
    else:
        ports_with_flow['corr_key'] = 'U'

    # Color by corridor using fixed palette/order as in plot_edges_by_flow_thickness
    corridor_order = ['A','B','C','E','G','I','J','K','L','U']
    corridor_legend = [
        'Baltic-Adriatic',
        'North Sea-Adriatic',
        'Mediterranean',
        'Scandinavian-Mediterranean',
        'Atlantic',
        'Rhine-Danube',
        'Baltic-Aegean',
        'W. Balkans-E. Mediterranean',
        'North Sea-Mediterranean',
        'Not-in-corridor'
    ]
    corridor_palette = [
        '#0080C0',  # Blue
        '#E91E8C',  # Magenta/Pink
        '#00A651',  # Green
        '#FF69B4',  # Light Pink
        '#FFD700',  # Yellow/Gold
        '#00BFFF',  # Cyan/Light Blue
        '#8B4789',  # Purple
        '#8B4513',  # Brown
        '#228B22',  # Dark Green
        '#b3b3b3'   # Gray for 'U' (Not-in-corridor)
    ]
    fixed_colors = {k: corridor_palette[i] for i, k in enumerate(corridor_order)}
    default_color = '#b3b3b3'
    unique_corrs = ports_with_flow['corr_key'].dropna().unique() if 'corr_key' in ports_with_flow.columns else []
    ordered_corrs = [k for k in corridor_order if k in unique_corrs] + [k for k in unique_corrs if k not in corridor_order]
    corridor_colors = {k: fixed_colors.get(k, default_color) for k in ordered_corrs}

    # Plot by corridor key
    legend_labels = {k: corridor_legend[i] for i, k in enumerate(corridor_order)}
    for corr in ordered_corrs:
        corr_ports = ports_with_flow[ports_with_flow['corr_key'] == corr]
        ax.scatter(
            corr_ports.geometry.x,
            corr_ports.geometry.y,
            s=corr_ports['bubble_size'],
            c=[corridor_colors[corr]],
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            label=legend_labels.get(corr, corr)
        )
    
    # Create size legend
    unit_label = 'ths tons/year' if mode == 'Freight' else 'trips/year'
    legend_sizes = [np.log10(v + 1) * scale_factor for v in flow_values]
    
    # Add size legend
    legend_elements = []
    for val, size in zip(flow_values, legend_sizes):
        legend_elements.append(plt.scatter([], [], s=size, c='gray', alpha=0.6, 
                                          edgecolors='black', linewidth=0.5,
                                          label=f'{val:,} {unit_label}'))
    
    # Corridor legend handles (all possible, not just present)
    corridor_handles = [mlines.Line2D([], [], color=fixed_colors.get(k, default_color), marker='o', linestyle='None', markersize=10, label=legend_labels.get(k, k)) for k in corridor_order]
    # Add 'Not-in-corridor' explicitly
    if 'U' not in corridor_order:
        corridor_handles.append(mlines.Line2D([], [], color=default_color, marker='o', linestyle='None', markersize=10, label='Not-in-corridor'))
    # Size legend handles (all flow values)
    size_legend_handles = [plt.Line2D([], [], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(s), label=f'{v:,} {unit_label}') for v, s in zip(flow_values, legend_sizes)]
    # Size legend at bottom left
    size_legend = ax.legend(handles=size_legend_handles, title='Flow Volume', loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=True)
    ax.add_artist(size_legend)
    # Corridor legend just above size legend
    legend1 = ax.legend(handles=corridor_handles, title='Corridor', loc='lower left', bbox_to_anchor=(0.01, 0.20), frameon=True)
    ax.add_artist(legend1)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.draw()
    
    # Add port name and ICAO labels if available
    name_col_candidates = ['port_name', 'name', 'NAME', 'Port_Name']
    name_col = next((c for c in name_col_candidates if c in ports_with_flow.columns), None)
    icao_col_candidates = ['icao', 'ICAO', 'IATA', 'iata']
    icao_col = next((c for c in icao_col_candidates if c in ports_with_flow.columns), None)
    
    for idx, row in ports_with_flow.iterrows():
        label = None
        if name_col and icao_col:
            label = f"{row[name_col]} ({row[icao_col]})"
        elif name_col:
            label = str(row[name_col])
        elif icao_col:
            label = str(row[icao_col])
        if label:
            ax.annotate(
                label,
                xy=(row.geometry.x, row.geometry.y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
                ha='left'
            )
    
    unit = 'thousand tons' if mode == 'Freight' else 'trips'
    # Capitalize transport mode for title
    tm_title = transport_mode.capitalize()
    ax.set_title(f'{tm_title} {mode} {flow_type} Flows - {country_name}\n(Bubble size = flow volume in {unit}, log scale)', fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Save or show, always use bbox_inches='tight' to avoid legend clipping
    filename = f'{transport_mode.lower()}_bubbles_{country_name.replace(" ", "_")}.png'
    if created_fig:
        plt.savefig(outpath / filename, dpi=300, bbox_inches='tight')
        plt.show()




def build_igraph_from_edges(edges_gdf, src_col, dst_col, nodes_gdf, node_id_col, weight_col=None):
    """
    Build undirected igraph from edges and nodes GeoDataFrames.
    
    Parameters:
    - edges_gdf: GeoDataFrame with network edges
    - src_col, dst_col: column names for source and destination nodes in edges_gdf
    - nodes_gdf: GeoDataFrame with network nodes
    - node_id_col: column name for node IDs in nodes_gdf
    - weight_col: column name for edge weights (e.g., 'length')
    
    Returns:
    - g: igraph.Graph object
    - node_index: dict mapping node IDs to igraph indices
    - edges_view: filtered edges_gdf with valid endpoints
    """
    # Ensure node_id is string for matching
    nodes_gdf['node_id'] = nodes_gdf['node_id'].astype(str)

    # Ensure from_node and to_node are string for matching
    edges_gdf['from_node'] = edges_gdf['from_node'].astype(str)
    edges_gdf['to_node'] = edges_gdf['to_node'].astype(str)
    node_set = set(nodes_gdf['node_id'])
    # Remove edges with missing node references
    edges_gdf = edges_gdf[
        edges_gdf['from_node'].isin(node_set) & edges_gdf['to_node'].isin(node_set)
    ].copy()

    # Remove duplicate node IDs, keep first occurrence
    nodes_gdf = nodes_gdf.drop_duplicates(subset=[node_id_col], keep='first').copy()
    # Build node index from the full nodes_gdf
    nodes = nodes_gdf[node_id_col].astype(str).to_numpy()
    node_index = {n: i for i, n in enumerate(nodes)}

    # Remove duplicate edges (same src and dst, undirected)
    edges_gdf['_src'] = edges_gdf[src_col].astype(str)
    edges_gdf['_dst'] = edges_gdf[dst_col].astype(str)
    # For undirected, sort src/dst so (a,b) == (b,a)
    edges_gdf['_edge_key'] = edges_gdf.apply(lambda row: tuple(sorted([row['_src'], row['_dst']])), axis=1)
    edges_gdf = edges_gdf.drop_duplicates(subset=['_edge_key'], keep='first').copy()

    # Keep only edges whose endpoints exist in node_index
    s = edges_gdf['_src']
    t = edges_gdf['_dst']
    mask = s.isin(node_index) & t.isin(node_index)
    edges_view = edges_gdf[mask].copy()

    e_idx = list(zip(s[mask].map(node_index).to_numpy(), t[mask].map(node_index).to_numpy()))
    g = ig.Graph(n=len(nodes), edges=e_idx, directed=False)

    # Weights
    if weight_col and (weight_col in edges_view.columns):
        w = pd.to_numeric(edges_view[weight_col], errors='coerce').fillna(1.0).astype(float).tolist()
    else:
        w = [1.0] * len(e_idx)
    g.es['weight'] = w

    # Store original node ids
    g.vs['node_id'] = nodes.tolist()
    # Remove helper columns before returning
    if '_src' in edges_view.columns:
        edges_view = edges_view.drop(columns=['_src', '_dst', '_edge_key'], errors='ignore')
    return g, node_index, edges_view


def assemble_od_paths(od_df, g, node_index, weight_attr='weight', edge_id_attr='edge_id'):
    """
    Calculate shortest paths for all OD pairs and return DataFrame with paths.
    
    Parameters:
    - od_df: DataFrame with 'from_node' and 'to_node' columns
    - g: igraph.Graph object
    - node_index: dict mapping node IDs to igraph indices
    - weight_attr: attribute name for edge weights
    - edge_id_attr: attribute name for edge IDs
    
    Returns:
    - DataFrame with columns: from_node, to_node, edge_path (list of edge IDs), cost
    """
    rows = []
    # Connected components to skip unreachable pairs
    comps = g.connected_components()
    membership = np.asarray(comps.membership, dtype=int)
    w = np.asarray(g.es[weight_attr], dtype=float)

    for s, grp in od_df.groupby('from_node'):
        if s not in node_index:
            continue
        s_idx = int(node_index[s])

        # Map destinations to indices
        dest_names_all = grp['to_node'].astype(str).tolist()
        keep_idxs = []
        keep_names = []
        for d in dest_names_all:
            ti = node_index.get(d, None)
            if ti is None:
                continue
            if membership[ti] == membership[s_idx]:
                keep_idxs.append(int(ti))
                keep_names.append(d)
        if not keep_idxs:
            continue

        epaths = g.get_shortest_paths(s_idx, to=keep_idxs, weights=weight_attr, output='epath')
        for dname, epath in zip(keep_names, epaths):
            if not epath:
                rows.append({'from_node': s, 'to_node': dname, 'edge_path': [], 'cost': 0.0})
                continue
            # Convert igraph edge indices -> edge_id
            eids = [g.es[e][edge_id_attr] for e in epath]
            cost = float(np.sum(w[np.asarray(epath, dtype=int)])) if epath else 0.0
            rows.append({'from_node': s, 'to_node': dname, 'edge_path': eids, 'cost': cost})
    return pd.DataFrame(rows)


def add_flow_to_edges(arr, eids, v):
    """Add flow value v to edges specified by edge IDs in eids."""
    if v <= 0 or not isinstance(eids, (list, tuple)) or len(eids) == 0:
        return
    np.add.at(arr, np.asarray(eids, dtype=int), float(v))

def plot_edges_by_flow_thickness(
    ax, edges_plot, flow_col='flow', corridors_col='CORRIDORS',
    lw_min=0.1, lw_max=4, scale='linear', scale_div=250.0, legend_ticks=None
):
    """
    Plot network edges with variable thickness based on flow volume.
    
    Parameters:
    - ax: matplotlib axis
    - edges_plot: GeoDataFrame with edges to plot
    - flow_col: column name for flow values
    - corridors_col: column name for corridor classification
    - lw_min, lw_max: min and max linewidths
    - scale: 'linear' or 'log' scaling
    - scale_div: divisor for flow values (for nicer legend numbers)
    """
    import numpy as np  # Ensure np is always defined in local scope for closures
    df = edges_plot.copy()
    if df is None or len(df) == 0:
        print("No edges to plot (empty input).")
        return

    # Find CORRIDORS column (case-insensitive)
    if corridors_col not in df.columns:
        alt = next((c for c in df.columns if c.upper() == 'CORRIDORS'), None)
        corridors_col = alt if alt else None

    # Corridor key (first char) with NULL -> 'U'
    if corridors_col:
        raw = df[corridors_col].astype(str).str.strip()
        nullish = df[corridors_col].isna() | raw.str.upper().isin(['NULL', 'NONE', 'nan', 'NA', 'N/A'])
        up = raw.str.upper()
        df['corr_key'] = up.str[0]
        df.loc[nullish, 'corr_key'] = 'U'
    else:
        df['corr_key'] = 'U'

  
    # Legends
    # Use same corridor order, names, and colors as plot_bubble_map
    corridor_order = ['A','B','C','E','G','I','J','K','L','U']
    corridor_legend = [
        'Baltic-Adriatic',
        'North Sea-Adriatic',
        'Mediterranean',
        'Scandinavian-Mediterranean',
        'Atlantic',
        'Rhine-Danube',
        'Baltic-Aegean',
        'W. Balkans-E. Mediterranean',
        'North Sea-Mediterranean',
        'Not-in-corridor'
    ]
    corridor_palette = [
        '#0080C0',  # Blue
        '#E91E8C',  # Magenta/Pink
        '#00A651',  # Green
        '#FF69B4',  # Light Pink
        '#FFD700',  # Yellow/Gold
        '#00BFFF',  # Cyan/Light Blue
        '#8B4789',  # Purple
        '#8B4513',  # Brown
        '#228B22',  # Dark Green
        '#b3b3b3'   # Gray for 'U' (Not-in-corridor)
    ]
    fixed_colors = {k: corridor_palette[i] for i, k in enumerate(corridor_order)}
    legend_labels = {k: corridor_legend[i] for i, k in enumerate(corridor_order)}

    default_color = '#b3b3b3'

    keys = sorted(df['corr_key'].unique())
    color_map = {k: fixed_colors.get(k, default_color) for k in keys}
    keys_for_plot = (['U'] if 'U' in keys else []) + [k for k in keys if k != 'U']
    
    # Source values (RAW flow)
    flows_raw = pd.to_numeric(df[flow_col], errors='coerce').fillna(0.0)
    
    scale_div = float(scale_div)
    vals = flows_raw / scale_div
    if vals.empty:
        print("No flow values to scale.")
        return

    vmin = float(vals.min())
    vmax = float(vals.max())

    # Build mapper
    if vmax <= vmin:
        vmax = vmin + 1.0
    def map_to_lw(x):
        t = (x - vmin) / max(1e-12, (vmax - vmin))
        t = float(np.clip(t, 0.0, 1.0))
        return lw_min + t * (lw_max - lw_min)

    # Compute linewidths
    df['lw'] = vals.apply(map_to_lw)

    # Plot by corridor (keeping U under), using LineCollection for variable linewidths
    for k in keys_for_plot:
        kdf = df[df['corr_key'] == k].copy()
        if not kdf.empty:
            z = 1 if k == 'U' else 2
            kdf['lw'] = pd.to_numeric(kdf['lw'], errors='coerce').astype(float)
            # Extract line geometries and linewidths
            lines = kdf.geometry.values
            # Convert to list of xy pairs for LineCollection
            line_segments = [np.array(line.xy).T for line in lines if line is not None]
            linewidths = kdf['lw'].values[:len(line_segments)]
            # Only plot if there are valid lines
            if line_segments:
                lc = LineCollection(line_segments, colors=color_map[k], linewidths=linewidths, alpha=0.9, zorder=z)
                ax.add_collection(lc)

    
    # Corridor legend handles (all possible, not just present)
    corridor_handles = [mlines.Line2D([], [], color=fixed_colors.get(k, default_color), lw=3, label=legend_labels.get(k, k)) for k in corridor_order]
    # Add 'Not-in-corridor' explicitly if not in order
    if 'U' not in corridor_order:
        corridor_handles.append(mlines.Line2D([], [], color=default_color, lw=3, label='Not-in-corridor'))
    # Size legend (flow thickness)
    def fmt_tick(x):
        x = float(x)
        if x >= 1_000_000:
            return f"{x/1_000_000:.1f}M"
        if x >= 1000:
            return f"{int(x/1000)}k"
        return f"{int(x)}"
    if legend_ticks is not None:
        legend_ticks = list(legend_ticks)
    else:
        fmin = float(max(flows_raw[flows_raw > 0].min(), 1e-9)) if (scale.lower() == 'log' and (flows_raw > 0).any()) else float(flows_raw.min())
        fmax = float(flows_raw.max())
        legend_ticks = [fmin, fmax] if fmax <= fmin else (np.geomspace(fmin, fmax, 4).tolist() if (scale.lower() == 'log' and fmin > 0) else np.linspace(fmin, fmax, 4).tolist())
    th_handles = [mlines.Line2D([], [], color='gray', lw=map_to_lw(f), label=fmt_tick(f)) for f in legend_ticks]

    # Set legend title based on flow_col
    if flow_col.lower() in ['flow', 'tons', 'freight', 'weight']:
        legend_title = 'Flow (ths tons/year)'
    elif flow_col.lower() in ['trips', 'passengers', 'pass', 'passenger']:
        legend_title = 'Flow (Pass/year)'
    else:
        legend_title = f'Flow ({flow_col})'
    # Place size legend at bottom left
    size_legend = ax.legend(handles=th_handles, title=legend_title, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=True)
    ax.add_artist(size_legend)
    # Place corridor legend just above size legend
    color_legend = ax.legend(handles=corridor_handles, title='Corridor', loc='lower left', bbox_to_anchor=(0.01, 0.20), frameon=True)
    ax.add_artist(color_legend)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.draw()
    return ax


    
def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the columns {candidates} found in dataframe.")
    return None

def ensure_metric_length(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Ensure geometry is projected for length in meters
    if edges_gdf.crs is None or edges_gdf.crs.to_epsg() in (4326,):
        edg_m = edges_gdf.to_crs("EPSG:3857").copy()
    else:
        edg_m = edges_gdf.copy()
    edg_m['length_m'] = edg_m.geometry.length.astype(float)
    return edg_m

def compute_travel_time_kh(edges_gdf: gpd.GeoDataFrame, speed_col='tag_maxspeed') -> gpd.GeoDataFrame:
    edg_m = ensure_metric_length(edges_gdf)
    sp_col = speed_col
    if sp_col is None:
        speeds = pd.Series(80.0, index=edges_gdf.index, dtype=float)
    else:
        speeds = pd.to_numeric(edges_gdf[sp_col], errors='coerce')
        speeds = speeds.where((speeds > 0) & speeds.notna(), 80.0).astype(float)
    # time [hours] = meters / (km/h * 1000m/km)
    tt_hours = edg_m['length_m'] / (speeds * 1000.0)
    tt_hours = tt_hours.replace([np.inf, -np.inf], np.nan).fillna(edg_m['length_m'] / (80.0 * 1000.0))
    edges_gdf = edges_gdf.copy()
    edges_gdf['travel_time'] = tt_hours.astype(float)
    return edges_gdf

def map_stations_to_nodes(stations_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame,
                          station_id_col_candidates=('id','station_id'),
                          node_id_col='id') -> dict[str, str]:
    # Snap each station to nearest graph node (EPSG:4326 assumed)
    from scipy.spatial import cKDTree
    sx = stations_gdf.geometry.x.to_numpy()
    sy = stations_gdf.geometry.y.to_numpy()
    nx = nodes_gdf.geometry.x.to_numpy()
    ny = nodes_gdf.geometry.y.to_numpy()
    tree = cKDTree(np.column_stack([ny, nx]))  # query on [lat, lon]
    _, idx = tree.query(np.column_stack([sy, sx]))

    st_id_col = pick_col(stations_gdf, list(station_id_col_candidates))
    stations_ids = stations_gdf[st_id_col].astype(str).to_numpy()
    node_ids = nodes_gdf.iloc[idx][node_id_col].astype(str).to_numpy()
    return dict(zip(stations_ids, node_ids))

def edge_operable_hours(ty: pd.Series, mode) -> np.ndarray:
    s = ty.astype(str).str.upper()
    has_f = s.str.contains('FREIGHT', na=False)
    has_p = s.str.contains('PASSENGER', na=False)
    # Default to mixed (6 h/day) for null/unknown
    hours = np.full(len(s), 6.0, dtype=float)
    if mode == 'Freight':
        hours = np.where(has_f & ~has_p, 18.0, hours)  # freight-only
        hours = np.where(~has_f & has_p, 3.0, hours)   # passengers-only
        hours = np.where(has_f & has_p, 6.0, hours)    # mixed
    else:
        hours = np.where(has_f & ~has_p, 3.0, hours)  # freight-only
        hours = np.where(~has_f & has_p, 18.0, hours)   # passengers-only
        hours = np.where(has_f & has_p, 12.0, hours)    # mixed
    return hours.astype(float)

def compute_edge_capacity(edges_view: pd.DataFrame, tt_col='travel_time', train_tons = 500, mode='Freight') -> np.ndarray:
    min_trains_per_hour=1/10
    max_trains_per_hour=6
    days_per_year=300
    if tt_col not in edges_view.columns:
        raise KeyError(f"Missing '{tt_col}' in edges_view.")
    tt = pd.to_numeric(edges_view[tt_col], errors='coerce').astype(float).values
    tt = np.clip(tt, 1e-4, None)  # avoid zero
    # pick plausible rail typology column (fix: avoid list('RAILWAYS_A'))
    tcol = pick_col(edges_view, ['RAILWAYS_A', 'railways_a', 'RAILWAY_TYP', 'rail_type', 'type'], required=False)
    if tcol is None:
        # Default to mixed (18 h/day)
        hours = np.full(len(edges_view), 18.0, dtype=float)
    else:
        hours = edge_operable_hours(edges_view[tcol], mode)
    trains = np.floor(hours / tt)
    trains = np.maximum(trains, min_trains_per_hour*hours)
    trains = np.minimum(trains, max_trains_per_hour*hours)
    train_tons_vec = np.asarray(train_tons, dtype=float)
    if train_tons_vec.size == 1:
        train_tons_vec = np.full(len(edges_view), float(train_tons_vec), dtype=float)
    raw_capacity = trains * train_tons_vec * days_per_year
    # 10th-degree smoothing: average of 5 previous, 5 following, and itself
    if len(raw_capacity) > 1:
        smoothed = np.convolve(raw_capacity, np.ones(11)/11, mode='same')
        # For edges at the start/end, average over available window
        for i in range(len(raw_capacity)):
            left = max(0, i-5)
            right = min(len(raw_capacity), i+6)
            smoothed[i] = np.mean(raw_capacity[left:right])
        return smoothed.astype(float)
    else:
        return raw_capacity


def compute_edge_capacity_cars(edges_view: pd.DataFrame, tt_col='travel_time', share_cars=1, share_trucks=0, share_buses=0) -> np.ndarray:
    days_per_year = 300
    hours_per_day = 24
    # Set base capacity per tag_highway
    highway_capacities = {
        'motorway': 3000,
        'trunk': 2500,
        'primary': 2000,
        'secondary': 1500
    }
    if 'tag_highway' in edges_view.columns:
        tag_hw = edges_view['tag_highway'].astype(str).str.lower()
        base_capacity = tag_hw.map(highway_capacities).fillna(1800).astype(float)
    else:
        base_capacity = np.full(len(edges_view), 500.0, dtype=float)

    # Joint capacity using shares and PCU equivalents
    # PCU factors
    pcu_cars = 1.0
    pcu_trucks = 3.0
    pcu_buses = 3.0
    avg_pcu = share_cars * pcu_cars + share_trucks * pcu_trucks + share_buses * pcu_buses
    # Vehicles per hour (joint capacity)
    vehicles_per_hour = base_capacity / avg_pcu
    # Total vehicles per year
    raw_capacity = vehicles_per_hour * hours_per_day * days_per_year
    return raw_capacity.astype(float)
    

def ensure_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a 1-D Series for col even if duplicate column names exist or selection became a DataFrame."""
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        # keep only first occurrence
        first = obj.iloc[:, 0]
        # rebuild df without subsequent duplicates of col
        keep = []
        seen = 0
        for c in df.columns:
            if c == col:
                seen += 1
                if seen > 1:
                    continue
            keep.append(c)
        df.drop(columns=[c for c in df.columns if c == col][1:], inplace=True, errors='ignore')
        return first
    return obj


def find_minimal_flows_along_overcapacity_paths(over_capacity_ods,
                                                network_dataframe,
                                                over_capacity_edges,
                                                edge_id_paths,
                                                edge_id_column,
                                                flow_column):
    # Build per-edge dataframe listing path indexes that traverse each over‑capacity edge
    over_capacity_edges_df = pd.DataFrame(
        [(edge_id, edge_id_paths.get(edge_id, [])) for edge_id in over_capacity_edges],
        columns=[edge_id_column, "path_indexes"]
    )

    if over_capacity_edges_df.empty:
        oc = over_capacity_ods.copy()
        flow_series = ensure_series(oc, flow_column)
        oc["min_flows"] = 0.0
        oc["residual_flows"] = pd.to_numeric(flow_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return oc

    # Merge only residual_capacity (added_flow was dropped earlier)
    over_capacity_edges_df = over_capacity_edges_df.merge(
        network_dataframe[[edge_id_column, "residual_capacity"]],
        how="left",
        on=edge_id_column
    )

    # Flows along paths for each edge
    over_capacity_edges_df["edge_path_flow"] = over_capacity_edges_df.apply(
        lambda x: over_capacity_ods[over_capacity_ods.index.isin(x.path_indexes)][flow_column].values,
        axis=1
    )

    # Recompute per-edge added_flow as sum of path flows
    over_capacity_edges_df["added_flow"] = over_capacity_edges_df["edge_path_flow"].apply(
        lambda arr: float(np.sum(arr)) if len(arr) else 0.0
    )

    def _corr(x):
        flows = np.asarray(x.edge_path_flow, dtype=float)
        if flows.size == 0 or x.added_flow <= 0 or not np.isfinite(x.added_flow):
            return [0.0] * int(flows.size)
        cap = float(x.residual_capacity) if np.isfinite(x.residual_capacity) else 0.0
        ratio = cap / float(x.added_flow)
        return list(ratio * flows)

    over_capacity_edges_df["edge_path_flow_cor"] = over_capacity_edges_df.apply(_corr, axis=1)

    def _pairs(x):
        pis = list(x.path_indexes)
        vals = list(x.edge_path_flow_cor)
        m = min(len(pis), len(vals))
        return list(zip(pis[:m], vals[:m]))

    over_capacity_edges_df["path_flow_tuples"] = over_capacity_edges_df.apply(_pairs, axis=1)

    flat = list(chain.from_iterable(over_capacity_edges_df["path_flow_tuples"]))
    if len(flat) == 0:
        oc = over_capacity_ods.copy()
        flow_series = ensure_series(oc, flow_column)
        flow_vals = pd.to_numeric(flow_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        oc["min_flows"] = 0.0
        oc["residual_flows"] = flow_vals
        return oc

    min_flows = pd.DataFrame(flat, columns=["path_indexes", "min_flows"])
    min_flows["min_flows"] = pd.to_numeric(min_flows["min_flows"], errors="coerce").fillna(0.0)
    min_flows = (min_flows.sort_values("min_flows")
                           .drop_duplicates(subset=["path_indexes"], keep="first"))

    oc = over_capacity_ods.copy()
    oc = oc.merge(min_flows, how="left", on="path_indexes")
    oc["min_flows"] = pd.to_numeric(oc["min_flows"], errors="coerce").fillna(0.0)

    flow_series = ensure_series(oc, flow_column)
    flow_vals = pd.to_numeric(flow_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    min_vals = oc["min_flows"].to_numpy(dtype=float)
    oc["residual_flows"] = np.maximum(flow_vals - min_vals, 0.0)

    return oc

def get_path_indexes_for_edges(edge_ids_with_paths,selected_edge_list):
    return list(
            set(
                list(
                    chain.from_iterable([
                        path_idx for path_key,path_idx in edge_ids_with_paths.items() if path_key in selected_edge_list
                                        ]
                                        )
                    )
                )
            )

def get_flow_paths_indexes_of_edges(flow_dataframe,path_criteria):
    edge_path_index = defaultdict(list)
    for v in flow_dataframe.itertuples():
        for k in getattr(v,path_criteria):
            edge_path_index[k].append(v.Index)

    del flow_dataframe
    return edge_path_index

def get_flow_on_edges(save_paths_df, edge_id_column, edge_path_column, flow_column):
    # Vectorized accumulation via explode + groupby (faster than Python loops)
    df = save_paths_df[[edge_path_column, flow_column]].copy()
    df[flow_column] = pd.to_numeric(df[flow_column], errors="coerce").fillna(0.0).astype(float)
    # Ensure list dtype; replace non-list with empty list
    ep = df[edge_path_column].apply(lambda x: x if isinstance(x, list) else [])
    df = df.assign(**{edge_path_column: ep})
    if df.empty or (df[edge_path_column].map(len) == 0).all():
        return pd.DataFrame(columns=[edge_id_column, flow_column])
    exploded = df.explode(edge_path_column, ignore_index=True)
    exploded.dropna(subset=[edge_path_column], inplace=True)
    exploded[edge_id_column] = pd.to_numeric(exploded[edge_path_column], errors="coerce").astype("Int64")
    exploded = exploded.dropna(subset=[edge_id_column])
    exploded[edge_id_column] = exploded[edge_id_column].astype(int)
    agg = exploded.groupby(edge_id_column, as_index=False)[flow_column].sum()
    return agg

def update_flow_and_overcapacity(od_dataframe, network_dataframe, flow_column,
                                 edge_id_column="id", network_capacity_column="capacity",
                                 subtract=False, drop_added=True):
    edge_flows = get_flow_on_edges(od_dataframe, edge_id_column, "edge_path", flow_column)
    edge_flows.set_index(edge_id_column, inplace=True)
    ndx = network_dataframe.set_index(edge_id_column)
    added = edge_flows[flow_column].reindex(ndx.index).fillna(0.0).to_numpy(dtype=float)
    if subtract:
        # Cap the amount subtracted to the available capacity
        available = ndx[network_capacity_column].to_numpy(dtype=float) - ndx[flow_column].to_numpy(dtype=float)
        capped_added = np.minimum(added, np.maximum(available, 0.0))
    else:
        before = ndx[flow_column].to_numpy(dtype=float)
        ndx[flow_column] = before + added
        after = ndx[flow_column].to_numpy(dtype=float)
    ndx["over_capacity"] = ndx[network_capacity_column].to_numpy(dtype=float) - ndx[flow_column].to_numpy(dtype=float)
    network_dataframe = ndx.reset_index()
    return network_dataframe

def create_igraph_from_dataframe(graph_dataframe, directed=False, simple=False):
    # This might be in Snkit or Snail or Open-GIRA
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

def network_od_path_estimations_multiattribute(graph,
    source, target, cost_criteria, path_id_column, attribute_list=None, weights=None):
    # suppress igraph reachability warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Couldn't reach some vertices.*", category=RuntimeWarning)
        paths = graph.get_shortest_paths(source, target,
                                         weights=(weights if weights is not None else cost_criteria),
                                         output="epath")

    rows = []
    if attribute_list is None:
        for path in paths:
            rec = { 'edge_path': [], cost_criteria: 0.0 }
            if path:
                for eidx in path:
                    rec['edge_path'].append(graph.es[eidx][path_id_column])
                    # cost sum uses edge attribute; acceptable since disabled edges produce empty paths
                    rec[cost_criteria] += float(graph.es[eidx][cost_criteria])
            rows.append(rec)
    else:
        for path in paths:
            rec = {'edge_path': [], cost_criteria: 0.0}
            for a in attribute_list:
                rec[a] = 0.0
            if path:
                for eidx in path:
                    rec['edge_path'].append(graph.es[eidx][path_id_column])
                    rec[cost_criteria] += float(graph.es[eidx][cost_criteria])
                    for a in attribute_list:
                        rec[a] += float(graph.es[eidx][a])
            rows.append(rec)
    return pd.DataFrame(rows)

def network_od_paths_assembly_multiattributes(points_dataframe, graph,
                                cost_criteria, path_id_column,
                                origin_id_column, destination_id_column,
                                attribute_list=None, store_edge_path=True, weights=None,
                                path_cache=None):
    """Assemble OD paths and their costs; returns a DataFrame."""
    save_paths_df = []
    points_dataframe = points_dataframe.set_index(origin_id_column)
    origins = list(set(points_dataframe.index.values.tolist()))

    # Components on the graph passed in (already the base graph)
    try:
        comps = graph.components()
    except AttributeError:
        comps = graph.connected_components()
    name2comp = {}
    if 'name' in graph.vs.attributes():
        name2comp = dict(zip(graph.vs['name'], comps.membership))

    # Simple cache to avoid recomputing identical (origin, destination) paths across iterations
    if path_cache is None:
        path_cache = {}

    for origin in origins:
        try:
            destinations_all = list(set(points_dataframe.loc[[origin], destination_id_column].values.tolist()))
        except Exception:
            continue

        # filter destinations in the same component as origin
        destinations = destinations_all
        if name2comp:
            co = name2comp.get(origin, None)
            if co is None:
                destinations = []
            else:
                destinations = [d for d in destinations_all if name2comp.get(d, None) == co]
        if not destinations:
            continue

        # Try to reuse cached paths for each (origin, destination)
        cached_rows = []
        to_compute = []
        for dst in destinations:
            key = (origin, dst)
            if key in path_cache:
                cached_rows.append(path_cache[key])
            else:
                to_compute.append(dst)

        df_cached = pd.DataFrame(cached_rows) if cached_rows else pd.DataFrame(columns=(['edge_path', cost_criteria] + (list(attribute_list) if attribute_list else [])))

        df_new = pd.DataFrame(columns=df_cached.columns) if df_cached is not None else pd.DataFrame()
        if to_compute:
            df_new = network_od_path_estimations_multiattribute(
                graph, origin, to_compute, cost_criteria, path_id_column,
                attribute_list=attribute_list, weights=weights
            )
            # store to cache
            for i, dst in enumerate(to_compute):
                rec = df_new.iloc[i].to_dict()
                path_cache[(origin, dst)] = rec

        # Concatenate cached and new results, handling empty DataFrames
        if df_cached.empty and df_new.empty:
            df = pd.DataFrame(columns=['edge_path', cost_criteria])
        elif df_cached.empty:
            df = df_new.copy()
        elif df_new.empty:
            df = df_cached.copy()
        else:
            df = pd.concat([df_cached, df_new], axis=0, ignore_index=True)
        
        df[origin_id_column] = origin
        # Ensure destination alignment
        if len(df) == len(destinations):
            df[destination_id_column] = destinations
        else:
            # fallback: rebuild from cache order
            df[destination_id_column] = [d for d in destinations if (origin, d) in path_cache] + to_compute
        save_paths_df.append(df)

    if not save_paths_df:
        base_cols = [origin_id_column, destination_id_column, cost_criteria]
        if attribute_list:
            base_cols += list(attribute_list)
        cols = (['edge_path'] if store_edge_path else []) + base_cols
        return pd.DataFrame(columns=cols)

    out = pd.concat(save_paths_df, axis=0, ignore_index=True)
    if not store_edge_path and 'edge_path' in out.columns:
        out = out.drop(columns=['edge_path'])
    return out

def od_flow_allocation_capacity_constrained(flow_ods,
                                            network_dataframe,
                                            flow_column,
                                            cost_column,
                                            path_id_column,
                                            attribute_list=None,
                                            origin_id_column="origin_id",
                                            destination_id_column="destination_id",
                                            network_capacity_column="capacity",
                                            directed=False,
                                            simple=False,
                                            store_edge_path=True,
                                            graph_base: ig.Graph | None = None,
                                            track_progress=True,
                                            early_stop_share: float | None = None):

    total_target = float(flow_ods[flow_column].sum())
    network_dataframe["over_capacity"] = network_dataframe[network_capacity_column] - network_dataframe[flow_column]
    capacity_ods = []
    unassigned_paths = []
    progress_records = []
    graph_nodes = set(graph_base.vs['name'])
    iter_id = 0
    cum_assigned = 0.0
    last_cum_assigned = -1.0
    # Global path cache reused across iterations
    global_path_cache = {}

    while len(flow_ods.index) > 0:
        reachable_mask = (flow_ods[origin_id_column].isin(graph_nodes)) & (flow_ods[destination_id_column].isin(graph_nodes))
        dropped = flow_ods[~reachable_mask]
        if not dropped.empty:
            unassigned_paths.append(dropped)
        flow_ods = flow_ods[reachable_mask].copy()
        if len(flow_ods.index) == 0:
            break

        flows_only = flow_ods[[origin_id_column, destination_id_column, flow_column]].copy()

        ndx = network_dataframe.set_index(path_id_column)
        tt_by_eid = ndx[cost_column]
        oc_by_eid = ndx["over_capacity"]
        base_ids = np.asarray(graph_base.es['edge_id'], dtype=int)
        allowed = oc_by_eid.reindex(base_ids).to_numpy() > 0.0    # threshold removed
        w_vec = np.where(allowed, tt_by_eid.reindex(base_ids).to_numpy(), np.inf)

        paths_df = network_od_paths_assembly_multiattributes(
            flow_ods, graph_base,
            cost_column,
            path_id_column,
            origin_id_column,
            destination_id_column,
            attribute_list=attribute_list,
            store_edge_path=store_edge_path,
            weights=w_vec,
            path_cache=global_path_cache
        )

        flow_ods = paths_df.merge(flows_only, on=[origin_id_column, destination_id_column], how='left')
        flow_vals = flow_ods[flow_column]
        if isinstance(flow_vals, pd.DataFrame):
            flow_vals = flow_vals.iloc[:, 0]
        flow_ods[flow_column] = pd.to_numeric(flow_vals, errors='coerce').fillna(0.0)

        unreach = flow_ods[flow_ods[cost_column] == 0]
        if not unreach.empty:
            unassigned_paths.append(unreach)
        flow_ods = flow_ods[flow_ods[cost_column] > 0]
        if len(flow_ods.index) == 0:
            break

        network_dataframe["residual_capacity"] = network_dataframe["over_capacity"]
        network_dataframe = update_flow_and_overcapacity(
            flow_ods,
            network_dataframe,
            flow_column,
            edge_id_column=path_id_column,
            network_capacity_column=network_capacity_column,
            subtract=True,
            drop_added=True
        )

        over_capacity_edges = network_dataframe[network_dataframe["over_capacity"] < 0.0][path_id_column].tolist()
        if len(over_capacity_edges) > 0:
            for eid in over_capacity_edges:
                cap_val = network_dataframe.loc[network_dataframe[path_id_column] == eid, network_capacity_column].values
                overcap_val = network_dataframe.loc[network_dataframe[path_id_column] == eid, "over_capacity"].values

        finalized_dfs = []

        if len(over_capacity_edges) > 0:
            edge_id_paths = get_flow_paths_indexes_of_edges(flow_ods, "edge_path")
            edge_paths_overcapacity = get_path_indexes_for_edges(edge_id_paths, over_capacity_edges)

            cap_ods_nc = flow_ods[~flow_ods.index.isin(edge_paths_overcapacity)].copy()
            if store_edge_path is False and "edge_path" in cap_ods_nc.columns:
                cap_ods_nc.drop(["edge_path"], axis=1, inplace=True)
            capacity_ods.append(cap_ods_nc)
            finalized_dfs.append(cap_ods_nc)

            over_capacity_ods = flow_ods[flow_ods.index.isin(edge_paths_overcapacity)].copy()
            over_capacity_ods["path_indexes"] = over_capacity_ods.index.astype(int)

            over_capacity_ods = find_minimal_flows_along_overcapacity_paths(
                over_capacity_ods,
                network_dataframe,
                over_capacity_edges,
                edge_id_paths,
                path_id_column,
                flow_column
            )

            cap_ods = over_capacity_ods.copy()
            drop_cap = [c for c in ["path_indexes", flow_column, "residual_flows"] if c in cap_ods.columns]
            cap_ods.drop(columns=drop_cap, inplace=True)
            if "min_flows" in cap_ods.columns:
                cap_ods.rename(columns={"min_flows": flow_column}, inplace=True)
            if store_edge_path is False and "edge_path" in cap_ods.columns:
                cap_ods.drop(["edge_path"], axis=1, inplace=True)
            capacity_ods.append(cap_ods)
            finalized_dfs.append(cap_ods)

            orig_flow = pd.to_numeric(over_capacity_ods[flow_column], errors="coerce").fillna(0.0).to_numpy()
            resid_flow = pd.to_numeric(over_capacity_ods.get("residual_flows", 0.0), errors="coerce").fillna(0.0).to_numpy()
            safe_orig = np.where(orig_flow > 0, orig_flow, np.nan)
            over_capacity_ods["residual_ratio"] = np.nan_to_num(resid_flow / safe_orig, nan=0.0, posinf=0.0, neginf=0.0)

            if "residual_flows" in over_capacity_ods.columns:
                residual_df = over_capacity_ods[["edge_path"]].copy()
                residual_df[flow_column] = over_capacity_ods["residual_flows"].astype(float)
                # Debug: print residual_df before subtraction
                if 'edge_path' in residual_df.columns:
                    edge_paths = residual_df['edge_path'].head(10).tolist()
                    # If edge_paths are lists, flatten and print unique values
                    if edge_paths and isinstance(edge_paths[0], list):
                        flat_edge_paths = set(x for sublist in edge_paths for x in sublist)
                        
                # Print over_capacity before subtraction for a few edges
                sample_edges = []
                if 'edge_path' in residual_df.columns:
                    if edge_paths and isinstance(edge_paths[0], list):
                        sample_edges = list(flat_edge_paths)[:5]

                network_dataframe = update_flow_and_overcapacity(
                    residual_df,
                    network_dataframe,
                    flow_column,
                    path_id_column,
                    network_capacity_column=network_capacity_column,
                    subtract=True,
                    drop_added=True
                )

            # Keep any OD with remaining residual flow to be reconsidered next iteration
            flow_ods = over_capacity_ods[pd.to_numeric(over_capacity_ods.get("residual_flows", 0.0), errors="coerce").fillna(0.0) > 0.0].copy()
            if not flow_ods.empty:
                if "residual_flows" in flow_ods.columns:
                    flow_ods[flow_column] = pd.to_numeric(flow_ods["residual_flows"], errors="coerce").fillna(0.0)
                drop_cols_next = ["min_flows", "residual_flows", "path_indexes"]
                flow_ods.drop(columns=[c for c in drop_cols_next if c in flow_ods.columns], inplace=True)
                if attribute_list is not None:
                    flow_ods = flow_ods[[origin_id_column, destination_id_column, flow_column]]
            else:
                flow_ods = pd.DataFrame()
            del over_capacity_ods
        else:
            if store_edge_path is False and not flow_ods.empty:
                f = flow_ods.drop(["edge_path"], axis=1).copy()
            else:
                f = flow_ods.copy()
            capacity_ods.append(f)
            finalized_dfs.append(f)
            network_dataframe.drop(["residual_capacity"], axis=1, inplace=True, errors='ignore')
            flow_ods = pd.DataFrame()

        if track_progress:
            iter_assigned = sum(df[flow_column].sum() for df in finalized_dfs)
            cum_assigned += iter_assigned
            remaining = max(total_target - cum_assigned, 0.0)
            share = (cum_assigned / total_target) if total_target > 0 else 0.0

            print(f"[progress] iter={iter_id} "
                  f"assigned_iter={iter_assigned:.3f} "
                  f"cum={cum_assigned:.3f} ({share:.4f}) "
                  f"remaining={remaining:.3f} "
                  f"overcap_edges={len(over_capacity_edges)} "
                  f"next_flows={len(flow_ods)}",
                  flush=True)

            progress_records.append({
                "iteration": iter_id,
                "assigned_flow_iter": iter_assigned,
                "cumulative_assigned_flow": cum_assigned,
                "remaining_flow": remaining,
                "cumulative_share": share,
                "remaining_share": (remaining / total_target) if total_target > 0 else 0.0,
                "over_capacity_edges": len(over_capacity_edges),
                "next_flows_len": len(flow_ods)
            })

        # Stagnation guard: break if no progress (no flow assigned and flow_ods unchanged)
        if iter_assigned <= 100 and (last_cum_assigned >= 0 and abs(cum_assigned - last_cum_assigned) <= 100):
            print(f"[progress] No progress in iter={iter_id}; breaking to avoid infinite loop.", flush=True)
            break
        last_cum_assigned = cum_assigned

        if early_stop_share is not None:
            current_share = (cum_assigned / total_target) if total_target > 0 else 0.0
            if current_share >= early_stop_share:
                if not flow_ods.empty:
                    fin = flow_ods.drop(columns=["edge_path"]).copy() if (store_edge_path is False and "edge_path" in flow_ods.columns) else flow_ods.copy()
                    capacity_ods.append(fin)
                print(f"Early stop triggered at share {current_share:.3f} (threshold {early_stop_share:.3f}).", flush=True)
                break

        iter_id += 1

    progress_df = pd.DataFrame(progress_records) if track_progress else pd.DataFrame()
    return capacity_ods, unassigned_paths, network_dataframe, progress_df

def od_road_creation(od_cars, od_buses, od_trucks, pcu_cars, pcu_buses, pcu_trucks):

    # Prepare OD dataframes with PCU-weighted flows
    od_cars_pcu = od_cars[['from_node', 'to_node', 'cars_vehicles']].copy()
    od_cars_pcu['pcu_flow'] = od_cars_pcu['cars_vehicles'] * pcu_cars
    od_cars_pcu['pcu_flow_cars'] = od_cars_pcu['pcu_flow']
    od_cars_pcu['pcu_flow_trucks'] = 0.0
    od_cars_pcu['pcu_flow_buses'] = 0.0

    od_trucks_pcu = od_trucks[['from_node', 'to_node', 'trucks_per_year']].copy()
    od_trucks_pcu['pcu_flow'] = od_trucks_pcu['trucks_per_year'] * pcu_trucks
    od_trucks_pcu['pcu_flow_cars'] = 0.0
    od_trucks_pcu['pcu_flow_trucks'] = od_trucks_pcu['pcu_flow']
    od_trucks_pcu['pcu_flow_buses'] = 0.0

    od_buses_pcu = od_buses[['from_node', 'to_node', 'buses_vehicles']].copy()
    od_buses_pcu['pcu_flow'] = od_buses_pcu['buses_vehicles'] * pcu_buses
    od_buses_pcu['pcu_flow_cars'] = 0.0
    od_buses_pcu['pcu_flow_trucks'] = 0.0
    od_buses_pcu['pcu_flow_buses'] = od_buses_pcu['pcu_flow']

    # Merge all OD flows into a single DataFrame
    od_all_pcu = pd.concat([
        od_cars_pcu[['from_node', 'to_node', 'pcu_flow', 'pcu_flow_cars', 'pcu_flow_trucks', 'pcu_flow_buses']],
        od_trucks_pcu[['from_node', 'to_node', 'pcu_flow', 'pcu_flow_cars', 'pcu_flow_trucks', 'pcu_flow_buses']],
        od_buses_pcu[['from_node', 'to_node', 'pcu_flow', 'pcu_flow_cars', 'pcu_flow_trucks', 'pcu_flow_buses']]
    ], axis=0, ignore_index=True)

    # Aggregate by OD pair
    vehicles_od_pcu = od_all_pcu.groupby(['from_node', 'to_node'], as_index=False).agg({
        'pcu_flow': 'sum',
        'pcu_flow_cars': 'sum',
        'pcu_flow_trucks': 'sum',
        'pcu_flow_buses': 'sum'
    })

    # Calculate shares based on PCU flows
    vehicles_od_pcu['share_cars'] = vehicles_od_pcu['pcu_flow_cars'] / vehicles_od_pcu['pcu_flow']
    vehicles_od_pcu['share_trucks'] = vehicles_od_pcu['pcu_flow_trucks'] / vehicles_od_pcu['pcu_flow']
    vehicles_od_pcu['share_buses'] = vehicles_od_pcu['pcu_flow_buses'] / vehicles_od_pcu['pcu_flow']

    # Keep only pcu_flow and shares
    vehicles_od_pcu = vehicles_od_pcu[['from_node', 'to_node', 'pcu_flow', 'share_cars', 'share_trucks', 'share_buses']]

    return vehicles_od_pcu