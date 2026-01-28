# Flow check and assignment:
# - Read and normalize ports, inland waterway (IWW) nodes/ports, and airports geodata.
# - Match port names to UN/LOCODE with country-filtered fuzzy matching.
# - Join Eurostat IN/OUT totals (sea, IWW, air), distribute to terminals by area.
# - Save updated datasets and generate GeoBubble plots per mode (IN/OUT) over Europe.

import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkb import loads
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
import re
import sys
# -------------------------------------------------------------------
# Paths and base data
# -------------------------------------------------------------------
countries_path = r"/soge-home/projects/mistral/miraca/incoming_data/spatial_data/admin/ne_10m/ne_10m_admin_0_countries.shp"
europe_shape = gpd.read_file(countries_path)

ports_parquet = "/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_ports_TENT.parquet"
iww_nodes_parquet = "/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_iww_nodes_TENT.parquet"
ports_out_parquet = "/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_ports_TENT.parquet"
iww_nodes_out_parquet = "/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_iww_nodes_TENT.parquet"

airports_parquet = "/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_airports_TENT.parquet"
airports_out_parquet = "/soge-home/projects/mistral/miraca/processed_data/processed_unimodal/europe_airports_TENT.parquet"

eurostat_ports_csv = "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/EUROSTAT/ports/Freight/Gross weight of goods transported tofrom main ports by direction and type of traffic.csv"
eurostat_iww_csv = "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/EUROSTAT/ports/Inland/Freight loaded and unloaded in ports for inland waterway transport.csv"
unloc_folder = Path("/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/EUROSTAT/ports")

air_routes_folder = "/soge-home/projects/mistral/miraca/incoming_data/spatial_data/data_industrial_OD/EUROSTAT/airports/Freight/Freight and mail air transport routes (aricrafts per quarter) between partner airports and main airports/"

# -------------------------------------------------------------------
# Load ports and IWW nodes, set CRS
# -------------------------------------------------------------------
ports_df = pd.read_parquet(ports_parquet)
iww_nodes_df = pd.read_parquet(iww_nodes_parquet)

# Convert WKB -> geometry and define CRS (EPSG:3857 as in source)
ports_df['geometry'] = ports_df['geometry'].apply(lambda b: loads(b) if pd.notna(b) else None)
ports_df = gpd.GeoDataFrame(ports_df, geometry='geometry', crs="EPSG:3857")

iww_nodes_df['geometry'] = iww_nodes_df['geometry'].apply(lambda b: loads(b) if pd.notna(b) else None)
iww_nodes_df = gpd.GeoDataFrame(iww_nodes_df, geometry='geometry', crs="EPSG:3857")
iww_nodes_df = iww_nodes_df.rename(columns={"BB_PORT_NA": "port_name"})
# Keep text before '/' if present and strip
iww_nodes_df['port_name'] = iww_nodes_df['port_name'].astype(str).str.split('/').str[0].str.strip()

# IWW ports subset
iww_ports_df = iww_nodes_df[iww_nodes_df['feature'] == 'port'].copy()

# -------------------------------------------------------------------
# Eurostat Ports and IWW data
# -------------------------------------------------------------------
transported_df_ports = pd.read_csv(eurostat_ports_csv, sep=',', encoding='ISO-8859-1')
transported_df_ports['tra_cov'] = transported_df_ports['tra_cov'].astype(str).str.strip()
transported_df_ports['rep_mar'] = transported_df_ports['rep_mar'].astype(str).str.strip()
transported_df_ports = transported_df_ports[transported_df_ports['tra_cov'] == 'TOTAL']
transported_df_ports['port_code'] = transported_df_ports['rep_mar'].str[-5:]
database_codes_unique_all = sorted(set(c for c in transported_df_ports['port_code'].unique().tolist() if isinstance(c, str) and len(c) >= 5))

transported_df_iww_ports = pd.read_csv(eurostat_iww_csv, sep=',', encoding='ISO-8859-1')
transported_df_iww_ports['tra_meas'] = transported_df_iww_ports['tra_meas'].astype(str).str.strip()
transported_df_iww_ports['port_iww'] = transported_df_iww_ports['port_iww'].astype(str).str.strip()
transported_df_iww_ports['port_code'] = transported_df_iww_ports['port_iww'].str[-5:]
database_codes_unique_iww = sorted(set(c for c in transported_df_iww_ports['port_code'].unique().tolist() if isinstance(c, str) and len(c) >= 5))

# Combine unique codes (preserve uniqueness)
database_codes_unique_all = list(dict.fromkeys(database_codes_unique_all + database_codes_unique_iww))

# -------------------------------------------------------------------
# UN/LOCODE lookup (codes + names)
# -------------------------------------------------------------------
codes, names = [], []
parts = sorted(unloc_folder.glob("2022-2 UNLOCODE CodeListPart*.csv"))
for p in parts:
    try:
        dfp = pd.read_csv(p, encoding='ISO-8859-1')
        codes.extend((dfp['Column1'].astype(str) + dfp['Column2'].astype(str)).tolist())
        names.extend(dfp['Column4'].astype(str).tolist())
    except Exception as e:
        print(f"Warning: failed reading {p}: {e}", flush=True)

order_in_codes = [i for i, code in enumerate(codes) if code in database_codes_unique_all]
ports_in_database = [(codes[i], names[i]) for i in order_in_codes]
name_to_code = {name: code for code, name in ports_in_database}
_db_names = list(name_to_code.keys())

# ISO3 -> ISO2 for country consistency in fuzzy match
ISO3_TO_ISO2 = {
    'AUT':'AT','BEL':'BE','BGR':'BG','HRV':'HR','CYP':'CY','CZE':'CZ','DNK':'DK','EST':'EE','FIN':'FI',
    'FRA':'FR','DEU':'DE','GRC':'GR','HUN':'HU','IRL':'IE','ITA':'IT','LVA':'LV','LTU':'LT','LUX':'LU',
    'MLT':'MT','NLD':'NL','POL':'PL','PRT':'PT','ROU':'RO','SVK':'SK','SVN':'SI','ESP':'ES','SWE':'SE',
    'GBR':'GB','NOR':'NO','CHE':'CH','ISL':'IS','ALB':'AL','SRB':'RS','MKD':'MK','MNE':'ME','BIH':'BA',
    'MDA':'MD','UKR':'UA','BLR':'BY','RUS':'RU','TUR':'TR','GIB':'GI','AND':'AD','LIE':'LI','MCO':'MC'
}

# Stopwords to remove from multi-word port names before matching (case-insensitive)
_PORT_STOPWORDS = {
    'port', 'porto', 'sea', 'ports', 'puerto', 'häfen', 'riverine', 'river', 'de', "d'", 'di', 'le', 'la', 'of', 'and', 'the', 'harbor', 'harbour', 'haven',
}

europe_shape = gpd.read_file(countries_path)

# Build ISO2 -> country geometry (in EPSG:3857) for spatial validation
try:
    europe_shape_3857 = europe_shape if (europe_shape.crs and europe_shape.crs.to_string() == "EPSG:3857") else europe_shape.to_crs("EPSG:3857")
except Exception:
    europe_shape_3857 = europe_shape.copy()
iso2_candidates = ['ISO_A2', 'ISO_A2_EH', 'CNTR_ID', 'CNTR_CODE']
_iso2_col = next((c for c in iso2_candidates if c in europe_shape_3857.columns), None)
_COUNTRY_GEOMS = {}
if _iso2_col:
    tmp = europe_shape_3857[[_iso2_col, 'geometry']].dropna(subset=[_iso2_col]).copy()
    tmp[_iso2_col] = tmp[_iso2_col].astype(str).str.upper().str.strip()
    # dissolve to one geometry per ISO2
    dissolved = tmp.dissolve(by=_iso2_col, as_index=False)
    _COUNTRY_GEOMS = {row[_iso2_col]: row.geometry for _, row in dissolved.iterrows()}

def _geom_in_iso2(geom, iso2: str) -> bool:
    """Return True if geom lies within the ISO2 country polygon (using representative point for non-points)."""
    if geom is None or not iso2 or iso2.upper() not in _COUNTRY_GEOMS:
        return True  # no check possible -> do not block
    try:
        rep = geom if getattr(geom, "geom_type", "") in ("Point", "MultiPoint") else geom.representative_point()
        return _COUNTRY_GEOMS[iso2.upper()].contains(rep)
    except Exception:
        return True
    
def _clean_port_name_text(name: str) -> str:
    """Remove generic words from multi-word port names (case-insensitive)."""
    if not isinstance(name, str):
        return name
    s = re.sub(r'[\/,\-]+', ' ', name).strip()
    s = re.sub(r'\s+', ' ', s)
    tokens = s.split()
    if len(tokens) <= 1:
        return s
    # Remove the phrase 'puerto de' first (case-insensitive)
    s = re.sub(r'\bpuerto\s+de\b', ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s).strip()
    # Token-level stopword removal (case-insensitive)
    kept = [t for t in s.split() if t.lower() not in _PORT_STOPWORDS]
    return ' '.join(kept) if kept else s

def fuzzy_match_port_name(row, threshold=80):
    """
    Match a port name to UN/LOCODE with country restriction (ISO2 from iso3) and fuzzy fallback.
    Returns tuple: (port_code, match_method, match_score, matched_name)
    """

    port_name = row.get('port_name')
    if pd.isna(port_name):
        return pd.Series([np.nan, 'no_name', 0, ''])

    # Clean generic words case-insensitively
    s = _clean_port_name_text(str(port_name).strip())
    s_low = s.lower()

    # Restrict candidates by country (code[:2] == ISO2)
    candidates = _db_names
    iso3 = row.get('iso3')
    iso2 = None
    if pd.notna(iso3):
        iso2 = ISO3_TO_ISO2.get(str(iso3).upper())
        if iso2:
            candidates = [n for n in _db_names if str(name_to_code[n])[:2].upper() == iso2]
    
    # Clean candidate names for consistent matching
    candidates_cleaned = [_clean_port_name_text(c) for c in candidates]

    # Helper: geometry-country validation for a candidate name
    # Skip geometry check if we already have country match via ISO3 -> ISO2
    def _geo_ok(name: str) -> bool:
        code = name_to_code.get(name)
        if not code or len(code) < 2:
            return True
        # If iso2 is known and matches code country, trust it (skip geometry check)
        if iso2 and code[:2].upper() == iso2.upper():
            return True
        # Otherwise, do geometry validation
        return _geom_in_iso2(row.get('geometry'), code[:2])

    # 1) Exact match (case-insensitive) - check cleaned candidates
    for i, candidate_clean in enumerate(candidates_cleaned):
        if candidate_clean.lower() == s_low:
            original_name = candidates[i]
            if _geo_ok(original_name):
                code = name_to_code.get(original_name, np.nan)
                return pd.Series([code, 'exact_match', 100, original_name])
    
    # 2) Containment (case-insensitive) - bidirectional check on cleaned names
    # Check if cleaned candidate is contained in cleaned port_name OR vice versa
    # Use word boundaries to avoid matching "gent" in "aregenteau"
    def _is_word_match(short: str, long: str) -> bool:
        """Check if short is a whole word in long (not just a substring)"""
        pattern = r'\b' + re.escape(short) + r'\b'
        return bool(re.search(pattern, long, re.IGNORECASE))
    
    contained_indices = []
    for i, candidate_clean in enumerate(candidates_cleaned):
        if not candidate_clean:
            continue
        c_low = candidate_clean.lower()
        # Check word-boundary match in both directions
        if len(s_low) <= len(c_low):
            if _is_word_match(s_low, c_low):
                contained_indices.append(i)
        else:
            if _is_word_match(c_low, s_low):
                contained_indices.append(i)
    
    contained_geo = [i for i in contained_indices if _geo_ok(candidates[i])]
    if contained_geo:
        # Choose the longest cleaned candidate
        best_idx = max(contained_geo, key=lambda i: len(candidates_cleaned[i]))
        best_name = candidates[best_idx]
        code = name_to_code.get(best_name, np.nan)
        return pd.Series([code, 'containment_geo', 100, best_name])
    elif contained_indices:
        # fallback to contained without geo filter if nothing passed geo check
        best_idx = max(contained_indices, key=lambda i: len(candidates_cleaned[i]))
        best_name = candidates[best_idx]
        code = name_to_code.get(best_name)
        if code and _geom_in_iso2(row.get('geometry'), code[:2]):
            return pd.Series([code, 'containment', 100, best_name])

    # 3) Fuzzy matching (case-insensitive via lowercase comparison on cleaned names)
    # Convert cleaned candidates to lowercase for matching
    candidates_cleaned_lower = [c.lower() for c in candidates_cleaned]
    results = process.extract(s_low, candidates_cleaned_lower, scorer=fuzz.partial_ratio, limit=10)
    
    for matched_lower, score, idx in results:
        if score >= threshold:
            # Get the original candidate name from the index
            original_name = candidates[idx]
            if _geo_ok(original_name):
                code = name_to_code.get(original_name, np.nan)
                return pd.Series([code, 'fuzzy', score, original_name])

    # No match found - return best fuzzy match even if below threshold
    if results:
        matched_lower, best_score, idx = results[0]
        original_name = candidates[idx]
        return pd.Series([np.nan, 'no_match', best_score, original_name])
    
    return pd.Series([np.nan, 'no_candidates', 0, ''])
    

# Normalize port names (example: Ghent -> Gent)
def _normalize_name(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r'^Ghent$', 'Gent', regex=True)
    )

ports_df.loc[ports_df['port_name'].notna(), 'port_name'] = _normalize_name(ports_df.loc[ports_df['port_name'].notna(), 'port_name'])
iww_ports_df.loc[iww_ports_df['port_name'].notna(), 'port_name'] = _normalize_name(iww_ports_df.loc[iww_ports_df['port_name'].notna(), 'port_name'])

# Apply matching
ports_df[['port_code', 'match_method', 'match_score', 'matched_name']] = ports_df.apply(fuzzy_match_port_name, axis=1)
iww_ports_df[['port_code', 'match_method', 'match_score', 'matched_name']] = iww_ports_df.apply(fuzzy_match_port_name, axis=1)

# Print matching statistics for ports
print("\n=== MARITIME PORTS MATCHING RESULTS ===")
unmatched_ports = ports_df[ports_df['port_code'].isna()].copy()
matched_ports = ports_df[ports_df['port_code'].notna()].copy()

print(f"Total ports: {len(ports_df)}")
print(f"Matched: {len(matched_ports)} ({100*len(matched_ports)/len(ports_df):.1f}%)")
print(f"Unmatched: {len(unmatched_ports)} ({100*len(unmatched_ports)/len(ports_df):.1f}%)")

if len(matched_ports) > 0:
    print(f"\nMatch method breakdown:")
    print(matched_ports['match_method'].value_counts().to_string())
    print(f"\nAverage match score: {matched_ports['match_score'].mean():.1f}")

if len(unmatched_ports) > 0:
    print(f"\n=== UNMATCHED MARITIME PORTS (showing up to 50) ===")
    unmatched_display = unmatched_ports[['port_name', 'iso3', 'match_method', 'match_score', 'matched_name']].head(50)
    
    # Diagnose why matches failed
    perfect_matches = unmatched_display[unmatched_display['match_score'] == 100].copy()
    if len(perfect_matches) > 0:
        print(f"\n*** {len(perfect_matches)} ports have PERFECT matches (100 score) but failed - investigating... ***\n")
        
        for idx, row in perfect_matches.head(10).iterrows():
            port_name = row['port_name']
            matched_name = row['matched_name']
            iso3 = row['iso3']
            iso2 = ISO3_TO_ISO2.get(str(iso3).upper(), '??')
            
            # Check if matched name is in UN/LOCODE
            code_found = name_to_code.get(matched_name, 'NOT_IN_UNLOCODE')
            in_eurostat = code_found in database_codes_unique_all if code_found != 'NOT_IN_UNLOCODE' else False
            
            # Get the country code from the matched code
            matched_country = code_found[:2] if code_found != 'NOT_IN_UNLOCODE' and len(code_found) >= 2 else '??'
            
            print(f"Port: {port_name:25s} | Match: {matched_name:25s} | Code: {code_found:8s} | Port country: {iso2} | Code country: {matched_country} | In EUROSTAT: {in_eurostat}")
        
        print(f"\nConclusion: Ports with score=100 are failing because:")
        print(f"  - Either their UN/LOCODE code is NOT in EUROSTAT database")
        print(f"  - Or geometry validation failed (port outside expected country)")
    
    print(f"\n=== ALL UNMATCHED PORTS ===")
    for idx, row in unmatched_display.iterrows():
        port_name = str(row['port_name']) if row['port_name'] is not None else 'N/A'
        iso3 = str(row['iso3']) if row['iso3'] is not None else 'N/A'
        matched_name = str(row['matched_name']) if row['matched_name'] is not None else 'N/A'
        match_score = row['match_score'] if pd.notna(row['match_score']) else 0
        print(f"Port: {port_name:30s} | Country: {iso3:3s} | Best match: {matched_name:30s} | Score: {match_score:.0f}")

# In-DB flags (ports_df vs Eurostat 'ports'; iww uses its own mapping later)
ports_df['in_transported_db'] = ports_df['port_code'].isin(transported_df_ports['port_code'])

# Synthesize codes for unmatched entries (iso2 + 3-letter abbreviation of name)
def _synthesize_codes(df):
    if 'iso3' not in df.columns:
        return df

    def _abbr3(name: str) -> str:
        s = ''.join(ch for ch in str(name) if ch.isalpha()).upper()
        L = len(s)
        if L == 0: return 'XXX'
        if L == 1: return s[0] + 'XX'
        if L == 2: return s[0] + s[1] + 'X'
        return s[0] + s[L // 2] + s[-1]

    iso2 = df['iso3'].astype(str).str.upper().map(ISO3_TO_ISO2)
    abbr = df['port_name'].fillna('').apply(_abbr3)
    synth = (iso2.fillna('') + abbr).where(iso2.notna() & abbr.ne(''))
    mask = df['port_code'].isna()
    df.loc[mask, 'port_code'] = synth[mask]
    df.loc[mask, 'in_transported_db'] = False
    return df

ports_df = _synthesize_codes(ports_df)
iww_ports_df = _synthesize_codes(iww_ports_df)

# -------------------------------------------------------------------
# Sea ports: build yearly totals and map to terminals
# -------------------------------------------------------------------
# Normalize quarter columns (strip trailing spaces in headers)
transported_df_ports.columns = transported_df_ports.columns.str.strip()
q_cols = [c for c in transported_df_ports.columns if c.startswith('2021-Q')]
for c in q_cols:
    transported_df_ports[c] = pd.to_numeric(transported_df_ports[c].replace({':': 0, ': ': 0}), errors='coerce').fillna(0.0)

transported_df_ports['OBS_VALUE'] = transported_df_ports[q_cols].sum(axis=1) * 1000.0  # to tons
transported_df_ports['direct'] = transported_df_ports['direct'].astype(str).str.strip()

# IN/OUT dicts
in_cargo = transported_df_ports[transported_df_ports['direct'] == 'IN']
out_cargo = transported_df_ports[transported_df_ports['direct'] == 'OUT']
in_cargo_mapping = in_cargo.set_index('port_code')['OBS_VALUE'].to_dict()
out_cargo_mapping = out_cargo.set_index('port_code')['OBS_VALUE'].to_dict()

# Map to matched terminals
ports_df['OBS_VALUE_IN'] = ports_df['port_code'].map(in_cargo_mapping)
ports_df['OBS_VALUE_OUT'] = ports_df['port_code'].map(out_cargo_mapping)

# Exclude Warehouse/Storage from sea ports and set their IN/OUT to zero
excluded_land_uses = ['Warehouse', 'Storage']
filtered_ports_df = ports_df[~ports_df['land_use'].isin(excluded_land_uses)].copy()
excluded_ports_df = ports_df[ports_df['land_use'].isin(excluded_land_uses)].copy()
excluded_ports_df.loc[:, 'OBS_VALUE_IN'] = 0.0
excluded_ports_df.loc[:, 'OBS_VALUE_OUT'] = 0.0

# Re-map on filtered
filtered_ports_df['OBS_VALUE_IN'] = filtered_ports_df['port_code'].map(in_cargo_mapping)
filtered_ports_df['OBS_VALUE_OUT'] = filtered_ports_df['port_code'].map(out_cargo_mapping)

# Drop rows with NaN area; sum per port_name
filtered_ports_df = filtered_ports_df.dropna(subset=['area'])
filtered_ports_df['port_total_area'] = filtered_ports_df.groupby('port_name')['area'].transform('sum')

# Split with and without data (IN); keep copies to avoid chained assignments
ports_with_data = filtered_ports_df[(filtered_ports_df['OBS_VALUE_IN'].notna()) & (filtered_ports_df['OBS_VALUE_IN'] != 0)].copy()
ports_without_data = filtered_ports_df[(filtered_ports_df['OBS_VALUE_IN'].isna()) | (filtered_ports_df['OBS_VALUE_IN'] == 0)].copy()

# Distribute for those with data, proportionally by area within the port
ports_with_data.loc[:, 'OBS_VALUE_IN'] = ports_with_data['OBS_VALUE_IN'] * (ports_with_data['area'] / ports_with_data['port_total_area'])
ports_with_data.loc[:, 'OBS_VALUE_OUT'] = ports_with_data['OBS_VALUE_OUT'] * (ports_with_data['area'] / ports_with_data['port_total_area'])

print(f"Ports with data: {len(ports_with_data)}, Ports without data: {len(ports_without_data)}, Excluded ports: {len(excluded_ports_df)}")

# For ports_without_data, compute 10th percentile of value/area ratio from ports with data
# then multiply by each port's area
if len(ports_with_data) > 0 and (ports_with_data['area'] > 0).any():
    # Compute value per unit area for ports with data
    ports_with_data_positive_area = ports_with_data[ports_with_data['area'] > 0].copy()
    value_per_area_in = (ports_with_data_positive_area['OBS_VALUE_IN'] / ports_with_data_positive_area['area']).dropna()
    value_per_area_out = (ports_with_data_positive_area['OBS_VALUE_OUT'] / ports_with_data_positive_area['area']).dropna()
    
    # 10th percentile of value/area ratios
    q10_ratio_in = value_per_area_in.quantile(0.10) if len(value_per_area_in) > 0 else 0.0
    q10_ratio_out = value_per_area_out.quantile(0.10) if len(value_per_area_out) > 0 else 0.0
    
    # Assign values based on area × 10th percentile ratio
    ports_without_data.loc[:, 'OBS_VALUE_IN'] = ports_without_data['area'] * q10_ratio_in
    ports_without_data.loc[:, 'OBS_VALUE_OUT'] = ports_without_data['area'] * q10_ratio_out
else:
    ports_without_data.loc[:, 'OBS_VALUE_IN'] = 0.0
    ports_without_data.loc[:, 'OBS_VALUE_OUT'] = 0.0

# Combine all back
ports_df = pd.concat([ports_with_data, ports_without_data, excluded_ports_df], ignore_index=True)

# Convert to EPSG:3035 for consistency and add lon/lat coordinates
ports_df = ports_df.to_crs("EPSG:3035")

# Remove any existing longitude/latitude columns that might be incorrect
for col in ['longitude', 'latitude', 'lon', 'lat', 'centroid']:
    if col in ports_df.columns:
        ports_df = ports_df.drop(columns=[col])

# Compute centroid in projected CRS, then get lon/lat from EPSG:4326
centroid_3035 = ports_df.geometry.centroid
gdf_centroids_4326 = gpd.GeoDataFrame(geometry=centroid_3035, crs="EPSG:3035").to_crs("EPSG:4326")
ports_df['longitude'] = gdf_centroids_4326.geometry.x
ports_df['latitude'] = gdf_centroids_4326.geometry.y

print(f"Ports CRS: {ports_df.crs}, Sample lon/lat: {ports_df[['longitude', 'latitude']].head()}", flush=True)
ports_df.to_parquet(ports_out_parquet)

# -------------------------------------------------------------------
# IWW ports: map values
# -------------------------------------------------------------------
# IWW Eurostat: columns 'FR_NLD' (unloaded) and 'FR_LD' (loaded) via tra_meas
transported_df_iww_ports['OBS_VALUE'] = pd.to_numeric(transported_df_iww_ports['2021'], errors='coerce').fillna(0.0)
in_cargo_iww = transported_df_iww_ports[transported_df_iww_ports['tra_meas'] == 'FR_NLD']
out_cargo_iww = transported_df_iww_ports[transported_df_iww_ports['tra_meas'] == 'FR_LD']
in_cargo_mapping_iww = in_cargo_iww.set_index('port_code')['OBS_VALUE'].to_dict()
out_cargo_mapping_iww = out_cargo_iww.set_index('port_code')['OBS_VALUE'].to_dict()

iww_ports_df['OBS_VALUE_IN'] = iww_ports_df['port_code'].map(in_cargo_mapping_iww)
iww_ports_df['OBS_VALUE_OUT'] = iww_ports_df['port_code'].map(out_cargo_mapping_iww)

# Fill missing capacities with 25th quantile across ports that have capacity (> 0)
cap_in_pos = iww_ports_df['OBS_VALUE_IN'].dropna()
cap_in_pos = cap_in_pos[cap_in_pos > 0]
cap_out_pos = iww_ports_df['OBS_VALUE_OUT'].dropna()
cap_out_pos = cap_out_pos[cap_out_pos > 0]

q10_in = cap_in_pos.quantile(0.10) if len(cap_in_pos) else np.nan
q10_out = cap_out_pos.quantile(0.10) if len(cap_out_pos) else np.nan

# Fallback: if no positive capacities, do not change NaNs
if not np.isnan(q10_in):
    iww_ports_df['OBS_VALUE_IN'] = iww_ports_df['OBS_VALUE_IN'].fillna(q10_in)
    iww_ports_df.loc[iww_ports_df['OBS_VALUE_IN'] == 0, 'OBS_VALUE_IN'] = q10_in
if not np.isnan(q10_out):
    iww_ports_df['OBS_VALUE_OUT'] = iww_ports_df['OBS_VALUE_OUT'].fillna(q10_out)
    iww_ports_df.loc[iww_ports_df['OBS_VALUE_OUT'] == 0, 'OBS_VALUE_OUT'] = q10_out

# Ports without any capacity set feature to None
mask_no_trade = iww_ports_df['OBS_VALUE_IN'].isna() & iww_ports_df['OBS_VALUE_OUT'].isna()
iww_ports_df.loc[mask_no_trade, 'feature'] = None

# Merge back to nodes (overwrite by dropping existing first)
cols_to_overwrite = ['port_code', 'OBS_VALUE_IN', 'OBS_VALUE_OUT']
iww_nodes_updated = iww_nodes_df.drop(columns=cols_to_overwrite, errors='ignore').merge(
    iww_ports_df[['port_name'] + cols_to_overwrite],
    on='port_name',
    how='left'
)
# Remove any merge-suffix columns (_x/_y)
suffix_cols = [c for c in iww_nodes_updated.columns if c.endswith('_x') or c.endswith('_y')]
if suffix_cols:
    iww_nodes_updated = iww_nodes_updated.drop(columns=suffix_cols)
    
# Convert any dict-typed object columns to string (robust GeoParquet write)
for col in iww_nodes_updated.select_dtypes(include='object').columns:
    if iww_nodes_updated[col].apply(lambda x: isinstance(x, dict)).any():
        iww_nodes_updated[col] = iww_nodes_updated[col].apply(str)

iww_nodes_updated = iww_nodes_updated.drop_duplicates()

# Convert to EPSG:3035 for consistency and add lon/lat coordinates
iww_nodes_updated = iww_nodes_updated.to_crs("EPSG:3035")

# Remove any existing longitude/latitude columns that might be incorrect
for col in ['longitude', 'latitude', 'lon', 'lat', 'centroid']:
    if col in iww_nodes_updated.columns:
        iww_nodes_updated = iww_nodes_updated.drop(columns=[col])

# Compute centroid in projected CRS, then get lon/lat from EPSG:4326
centroid_3035 = iww_nodes_updated.geometry.centroid
gdf_centroids_4326 = gpd.GeoDataFrame(geometry=centroid_3035, crs="EPSG:3035").to_crs("EPSG:4326")
iww_nodes_updated['longitude'] = gdf_centroids_4326.geometry.x
iww_nodes_updated['latitude'] = gdf_centroids_4326.geometry.y

iww_nodes_updated.to_parquet(iww_nodes_out_parquet)


# -------------------------------------------------------------------
# Airports: read routes and aggregate IN/OUT per ICAO
# -------------------------------------------------------------------
airports_df = pd.read_parquet(airports_parquet)
airports_df['geometry'] = airports_df['geometry'].apply(lambda b: loads(b) if pd.notna(b) else None)
airports_df = gpd.GeoDataFrame(airports_df, geometry='geometry', crs="EPSG:3857")

# Read all TSV files (Eurostat routes data)
tsv_files = [os.path.join(air_routes_folder, fn) for fn in os.listdir(air_routes_folder) if fn.lower().endswith('.tsv')]
transported_df_airports = pd.concat(
    [pd.read_csv(f, sep='\t', encoding='ISO-8859-1') for f in tsv_files],
    ignore_index=True
)
# Clean headers and values
transported_df_airports.columns = (
    transported_df_airports.columns.str.strip().str.replace(r'[\\/]+', '_', regex=True)
)
transported_df_airports.replace(':', pd.NA, inplace=True)

# Filter for unit='T' (tons) - check if 'unit' column exists
if 'unit' in transported_df_airports.columns:
    transported_df_airports = transported_df_airports[transported_df_airports['unit'] == 'T'].copy()
    print(f"Filtered airports data for unit='T': {len(transported_df_airports)} rows", flush=True)

# Find the composite origin-destination column (it may be prefixed: 'freq,unit,tra_meas,airp_pr_TIME_PERIOD')
pair_col = next(
    (c for c in transported_df_airports.columns
     if 'airp_pr_time_period' in c.lower().replace('\\', '_').replace('/', '_')),
    None
)
if pair_col is None:
    # Fallback: any 'airp_pr*' object column
    pair_col = next(
        (c for c in transported_df_airports.columns
         if c.lower().startswith('airp_pr') and transported_df_airports[c].dtype == object),
        None
)
if pair_col is None:
    raise KeyError(f"Could not find any column containing 'airp_pr\\TIME_PERIOD'. Available: {list(transported_df_airports.columns)}")

# Normalize the composite field: take last comma segment, then collapse commas/spaces to "_"
transported_df_airports[pair_col] = (
    transported_df_airports[pair_col].astype(str)
    .str.split(',').str[-1]
    .str.replace(r'[\s,;]+', '_', regex=True)
    .str.replace(r'__+', '_', regex=True)
    .str.strip('_')
)

def _extract_origin_dest_icao(cell):
    if pd.isna(cell):
        return pd.Series([np.nan, np.nan])
    s = str(cell).strip()
    # Split normalized string and keep code-like tokens, ignoring country prefixes (e.g., AT, AE)
    toks = [t for t in s.split('_') if t]
    codes = [t.upper() for t in toks if t.isalnum() and 3 <= len(t) <= 4]
    if len(codes) < 2:
        return pd.Series([np.nan, np.nan])
    o, d = codes[-2], codes[-1]
    return pd.Series([o, d])

# Extract identifiers and compute totals
transported_df_airports[['originICAO','destinationICAO']] = (
    transported_df_airports[pair_col].apply(_extract_origin_dest_icao)
)

q_cols_air = [c for c in transported_df_airports.columns if c.startswith('2021-Q')]
for c in q_cols_air:
    transported_df_airports[c] = pd.to_numeric(transported_df_airports[c], errors='coerce').fillna(0.0)
transported_df_airports['total_value'] = transported_df_airports[q_cols_air].sum(axis=1) # to tons (factor as per source)

airports_df = airports_df.drop(columns=['OBS_VALUE_OUT', 'OBS_VALUE_IN'], errors='ignore')
# Remove any merge-suffix columns (_x/_y)
suffix_cols = [c for c in airports_df.columns if c.endswith('_x') or c.endswith('_y')]
if suffix_cols:
    airports_df = airports_df.drop(columns=suffix_cols)

# Aggregate IN/OUT by ICAO
obs_value_out = transported_df_airports.groupby('originICAO', as_index=False)['total_value'] \
    .sum().rename(columns={'originICAO': 'icao', 'total_value': 'OBS_VALUE_OUT'})
obs_value_in = transported_df_airports.groupby('destinationICAO', as_index=False)['total_value'] \
    .sum().rename(columns={'destinationICAO': 'icao', 'total_value': 'OBS_VALUE_IN'})

airports_df = airports_df.merge(obs_value_out, on='icao', how='left').merge(obs_value_in, on='icao', how='left')

airports_df['OBS_VALUE_OUT'] = airports_df['OBS_VALUE_OUT'].fillna(0.0)
airports_df['OBS_VALUE_IN'] = airports_df['OBS_VALUE_IN'].fillna(0.0)

# Drop rows with missing/blank ICAO
airports_df = airports_df[airports_df['icao'].notna() & (airports_df['icao'].astype(str).str.strip() != '')].copy()


# Flag if airport appears in routes
airports_df['in_origin_destination'] = (
    airports_df['icao'].isin(transported_df_airports['originICAO']) |
    airports_df['icao'].isin(transported_df_airports['destinationICAO'])
)

# Keep only airports with any flow and set geometry to centroid (vectorized)
airports_df.geometry = airports_df.geometry.centroid

# Rename index column if present
if '_indez_level_0_' in airports_df.columns:
    airports_df = airports_df.rename(columns={'_indez_level_0_': 'id'})

# Convert to EPSG:3035 for consistency and add lon/lat coordinates
airports_df = airports_df.to_crs("EPSG:3035")

# Remove any existing longitude/latitude columns that might be incorrect
for col in ['longitude', 'latitude', 'lon', 'lat', 'centroid']:
    if col in airports_df.columns:
        airports_df = airports_df.drop(columns=[col])

# Geometry is already centroid from line above, compute lon/lat from EPSG:4326
gdf_4326 = airports_df.to_crs("EPSG:4326")
airports_df['longitude'] = gdf_4326.geometry.x
airports_df['latitude'] = gdf_4326.geometry.y

airports_df.to_parquet(airports_out_parquet)

print("Flow check and assignment completed successfully.", flush=True)

# -------------------------------------------------------------------
# Plotting helpers and outputs
# -------------------------------------------------------------------
def create_geobubble_plot(gdf, data, title, output_file, xlim=None, ylim=None):
    if gdf is None or gdf.empty:
        print(f"No data to plot for {title}")
        return

    # To WGS84
    try:
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("Modal:")
    except Exception:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    # Drop invalid geometries
    gdf = gdf[gdf['geometry'].notna() & (~gdf['geometry'].is_empty)].copy()
    if gdf.empty:
        print(f"All geometries invalid/empty for {title}")
        return

    # Values and bubble sizes
    gdf[data] = pd.to_numeric(gdf[data], errors='coerce').fillna(0.0)
    gdf['bubble_size'] = gdf[data] / 1e6

    valid_tent_types = ['A','B','C','E','G','I','J','K','L']
    gdf['TENT_type'] = gdf.get('CORRIDORS', pd.Series(index=gdf.index, dtype=object)).fillna('Non-assigned').astype(str).str[0]
    gdf['TENT_type'] = gdf['TENT_type'].where(gdf['TENT_type'].isin(valid_tent_types), 'Non-assigned')

    corridor_colors = ['#0080C0','#E91E8C','#00A651','#FF69B4','#FFD700','#00BFFF','#8B4789','#8B4513','#228B22']
    color_map = {k: v for k, v in zip(valid_tent_types, corridor_colors)}
    color_map['Non-assigned'] = '#b3b3b3'
    gdf['color'] = gdf['TENT_type'].map(color_map).fillna('#b3b3b3')

    max_bs = gdf['bubble_size'].max()
    gdf['normalized_bubble_size'] = (50 if (max_bs == 0 or np.isnan(max_bs)) else (gdf['bubble_size'] / max_bs) * 600)

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
    size_labels = [f"{v:,.0f}" for v in size_vals]

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
    size_legend = ax.legend(handles=size_handles, title="Flow (M tons)", loc='lower left',
                           title_fontsize=legend_title_fs, fontsize=legend_label_fs, frameon=True)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved GeoBubble plot to {out_path}")

# -------------------------------------------------------------------
# Aggregate per code and plot GeoBubbles
# -------------------------------------------------------------------
aggregated_ports = ports_df.groupby('port_code', as_index=False).agg({
    'OBS_VALUE_IN': 'sum',
    'OBS_VALUE_OUT': 'sum',
    'geometry': 'first',
    'port_name': 'first',
    'CORRIDORS': 'first'
})
ports_df = gpd.GeoDataFrame(aggregated_ports, geometry='geometry', crs=ports_df.crs)

# Ports IN/OUT
create_geobubble_plot(
    ports_df, data='OBS_VALUE_IN',
    title="GeoBubble Plot for Ports (IN)",
    output_file="/soge-home/projects/mistral/miraca/incoming_data/plots/aiport_port_in_out/ports_geobubble_IN.png"
)
create_geobubble_plot(
    ports_df, data='OBS_VALUE_OUT',
    title="GeoBubble Plot for Ports (OUT)",
    output_file="/soge-home/projects/mistral/miraca/incoming_data/plots/aiport_port_in_out/ports_geobubble_OUT.png"
)

# Airports IN/OUT
create_geobubble_plot(
    airports_df, data='OBS_VALUE_IN',
    title="GeoBubble Plot for Airports (IN)",
    output_file="/soge-home/projects/mistral/miraca/incoming_data/plots/aiport_port_in_out/airports_geobubble_IN.png"
)
create_geobubble_plot(
    airports_df, data='OBS_VALUE_OUT',
    title="GeoBubble Plot for Airports (OUT)",
    output_file="/soge-home/projects/mistral/miraca/incoming_data/plots/aiport_port_in_out/airports_geobubble_OUT.png"
)

# IWW Ports IN/OUT (only 'port' features)
iww_ports_only = iww_ports_df[iww_ports_df['feature'] == 'port'].copy()
create_geobubble_plot(
    iww_ports_only, data='OBS_VALUE_IN',
    title="GeoBubble Plot for Inland Waterway Ports (IN)",
    output_file="/soge-home/projects/mistral/miraca/incoming_data/plots/aiport_port_in_out/iww_ports_geobubble_IN.png"
)
create_geobubble_plot(
    iww_ports_only, data='OBS_VALUE_OUT',
    title="GeoBubble Plot for Inland Waterway Ports (OUT)",
    output_file="/soge-home/projects/mistral/miraca/incoming_data/plots/aiport_port_in_out/iww_ports_geobubble_OUT.png"
)