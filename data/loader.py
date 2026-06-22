"""
Data Loader for OR Facility Layout Optimization
=================================================
Parses the client's Excel file into clean, validated data structures.

Data Sheets:
  1. Flow        — 30×30 asymmetric department interaction matrix
  2. Coordinates  — 197 candidate locations (x, y, fixed_dist, type)
  3. Distribution Substations — 137 zones with demand, cost, capacity, + 137×137 distance matrix
  4. Demand       — 20 products with throughput and storage requirements
"""

import numpy as np
import pandas as pd
import openpyxl
import os
import json


class FacilityData:
    """Container for all parsed data from the Excel file."""

    def __init__(self):
        # Problem 1: Facility Layout
        self.departments = []          # List of 30 department names
        self.flow_matrix = None        # 30×30 NumPy array
        
        # Problem 1: Candidate Locations
        self.locations = None          # DataFrame: id, x, y, fixed_dist, type
        self.location_distance_matrix = None  # 197×197 Euclidean distance matrix
        
        # Problem 2: Distribution Substations
        self.zones = None              # DataFrame: zone_id, demand, cost, capacity
        self.zone_distance_matrix = None  # 137×137 NumPy array (inf for +infinity)
        self.zone_names = []           # List of zone labels
        
        # Problem 3: Storage/Demand
        self.products = None           # DataFrame: product_id, throughput, storage

    def summary(self):
        """Print a summary of all loaded data."""
        print("=" * 60)
        print("FACILITY LAYOUT OPTIMIZATION - DATA SUMMARY")
        print("=" * 60)
        
        print(f"\n[P1] Facility Layout")
        print(f"   Departments: {len(self.departments)}")
        print(f"   Flow matrix: {self.flow_matrix.shape}")
        print(f"   Total flow:  {self.flow_matrix.sum():,.0f}")
        print(f"   Max flow:    {self.flow_matrix.max():,.0f}")
        print(f"   Asymmetric:  {not np.allclose(self.flow_matrix, self.flow_matrix.T)}")
        
        print(f"\n[P1] Candidate Locations")
        print(f"   Locations:   {len(self.locations)}")
        print(f"   X range:     [{self.locations['x'].min():.1f}, {self.locations['x'].max():.1f}]")
        print(f"   Y range:     [{self.locations['y'].min():.1f}, {self.locations['y'].max():.1f}]")
        print(f"   Types:       {self.locations['type'].nunique()}")
        
        print(f"\n[P2] Substation Location")
        print(f"   Zones:       {len(self.zones)}")
        print(f"   Total demand:{self.zones['demand'].sum():,.0f} (x1000 KWh)")
        print(f"   Avg cost:    {self.zones['cost'].mean():,.0f} (x$10)")
        print(f"   Avg capacity:{self.zones['capacity'].mean():,.0f} (x100 KWh)")
        n_inf = np.isinf(self.zone_distance_matrix).sum()
        total = self.zone_distance_matrix.size
        print(f"   Distance matrix: {self.zone_distance_matrix.shape}")
        print(f"   +infinity entries: {n_inf} ({100*n_inf/total:.1f}%)")
        
        print(f"\n[P3] Storage Layout")
        print(f"   Products:    {len(self.products)}")
        print(f"   Total throughput: {self.products['throughput'].sum():,.0f} ops/period")
        print(f"   Total storage:    {self.products['storage'].sum():,.0f} slots")
        print(f"   Slot size:   1.5m x 1.5m x 1.5m")
        
        print(f"\n{'=' * 60}")


def load_data(filepath):
    """
    Load and parse the Excel file into a FacilityData object.
    
    Parameters
    ----------
    filepath : str
        Path to the Excel file (Group 1.xlsx)
    
    Returns
    -------
    FacilityData
        Parsed and validated data container
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Excel file not found: {filepath}")
    
    wb = openpyxl.load_workbook(filepath, data_only=True)
    data = FacilityData()
    
    # ================================================================
    # SHEET 1: Flow Matrix (30×30)
    # ================================================================
    ws = wb['Flow ']
    
    # Extract department names from row 3, columns C onwards
    row3 = list(ws.iter_rows(min_row=3, max_row=3))[0]
    data.departments = []
    for cell in row3:
        if cell.value is not None and cell.column >= 3:  # Column C onwards
            data.departments.append(str(cell.value).strip())
    
    n_depts = len(data.departments)
    assert n_depts == 30, f"Expected 30 departments, got {n_depts}"
    
    # Extract flow values from rows 4-33, columns C-AF (3 to 32)
    data.flow_matrix = np.zeros((n_depts, n_depts), dtype=float)
    for i, row in enumerate(ws.iter_rows(min_row=4, max_row=4 + n_depts - 1)):
        for j in range(n_depts):
            val = row[j + 2].value  # Column C is index 2
            if val is not None:
                data.flow_matrix[i, j] = float(val)
    
    # ================================================================
    # SHEET 2: Coordinates (197 locations)
    # ================================================================
    ws = wb['Coordinates ']
    
    locations_data = []
    for row in ws.iter_rows(min_row=5, max_row=201):
        loc_id = row[1].value   # Column B
        x = row[2].value        # Column C
        y = row[3].value        # Column D
        fixed_dist = row[4].value  # Column E
        loc_type = row[5].value    # Column F
        
        if loc_id is not None:
            locations_data.append({
                'id': int(loc_id),
                'x': float(x),
                'y': float(y),
                'fixed_dist': float(fixed_dist) if fixed_dist else 0.0,
                'type': str(loc_type).strip() if loc_type else 'Unknown'
            })
    
    data.locations = pd.DataFrame(locations_data)
    assert len(data.locations) == 197, f"Expected 197 locations, got {len(data.locations)}"
    
    # Compute Euclidean distance matrix for locations
    coords = data.locations[['x', 'y']].values
    n_locs = len(coords)
    data.location_distance_matrix = np.zeros((n_locs, n_locs))
    for i in range(n_locs):
        for j in range(n_locs):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            data.location_distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)
    
    # ================================================================
    # SHEET 3: Distribution Substations (137 zones)
    # ================================================================
    ws = wb['Distribution Substations']
    
    zones_data = []
    zone_names = []
    
    for row in ws.iter_rows(min_row=7, max_row=7 + 280):  # generous upper bound
        zone_id = row[1].value   # Column B
        if zone_id is None or not str(zone_id).startswith('Z-'):
            continue
        
        demand = row[2].value     # Column C
        cost = row[3].value       # Column D
        capacity = row[4].value   # Column E
        
        zones_data.append({
            'zone_id': str(zone_id),
            'demand': float(demand) if demand else 0.0,
            'cost': float(cost) if cost else 0.0,
            'capacity': float(capacity) if capacity else 0.0,
        })
        zone_names.append(str(zone_id))
    
    data.zones = pd.DataFrame(zones_data)
    data.zone_names = zone_names
    n_zones = len(data.zones)
    
    # Extract distance matrix (columns I onwards, starting from row 7)
    data.zone_distance_matrix = np.zeros((n_zones, n_zones))
    
    for i, row in enumerate(ws.iter_rows(min_row=7, max_row=7 + n_zones - 1)):
        # Verify row label matches
        row_label = row[8].value  # Column I (index 8)
        if row_label is None:
            continue
        
        for j in range(n_zones):
            val = row[9 + j].value  # Column J onwards (index 9)
            if val is None:
                data.zone_distance_matrix[i, j] = np.inf
            elif isinstance(val, str) and 'inf' in val.lower():
                data.zone_distance_matrix[i, j] = np.inf
            else:
                try:
                    data.zone_distance_matrix[i, j] = float(val)
                except (ValueError, TypeError):
                    data.zone_distance_matrix[i, j] = np.inf
    
    # ================================================================
    # SHEET 4: Demand (20 products)
    # ================================================================
    ws = wb['Demand']
    
    products_data = []
    for row in ws.iter_rows(min_row=7, max_row=26):
        product_id = row[1].value   # Column B
        throughput = row[2].value   # Column C
        storage = row[3].value     # Column D
        
        if product_id is not None:
            products_data.append({
                'product_id': str(product_id).strip(),
                'throughput': int(throughput) if throughput else 0,
                'storage': int(storage) if storage else 0,
            })
    
    data.products = pd.DataFrame(products_data)
    assert len(data.products) == 20, f"Expected 20 products, got {len(data.products)}"
    
    wb.close()
    return data


def validate_data(data):
    """
    Run validation checks on the parsed data.
    
    Returns a list of (level, message) tuples where level is 'INFO', 'WARN', or 'ERROR'.
    """
    checks = []
    
    # Flow matrix checks
    checks.append(('INFO', f"Flow matrix shape: {data.flow_matrix.shape}"))
    
    if np.any(data.flow_matrix < 0):
        checks.append(('ERROR', 'Flow matrix contains negative values'))
    else:
        checks.append(('INFO', 'Flow matrix: all values non-negative [OK]'))
    
    diag = np.diag(data.flow_matrix)
    if np.any(diag != 0):
        checks.append(('WARN', f'Flow matrix diagonal has non-zero values: {np.count_nonzero(diag)} entries'))
    else:
        checks.append(('INFO', 'Flow matrix: diagonal is all zeros [OK]'))
    
    if np.allclose(data.flow_matrix, data.flow_matrix.T):
        checks.append(('INFO', 'Flow matrix is symmetric'))
    else:
        asymmetry = np.abs(data.flow_matrix - data.flow_matrix.T).max()
        checks.append(('INFO', f'Flow matrix is ASYMMETRIC (max diff: {asymmetry:.0f})'))
    
    # Location checks
    checks.append(('INFO', f"Locations: {len(data.locations)}, Types: {data.locations['type'].nunique()}"))
    
    # Distance matrix checks
    n_inf = np.isinf(data.zone_distance_matrix).sum()
    total = data.zone_distance_matrix.size
    checks.append(('INFO', f"Zone distance matrix: {n_inf} infinity entries ({100*n_inf/total:.1f}%)"))
    
    # Check diagonal (should be infinity or zero for same-zone)
    diag_zone = np.diag(data.zone_distance_matrix)
    inf_on_diag = np.isinf(diag_zone).sum()
    checks.append(('INFO', f"Zone distance diagonal: {inf_on_diag} infinity, {np.count_nonzero(~np.isinf(diag_zone))} finite"))
    
    # Demand checks
    checks.append(('INFO', f"Products: {len(data.products)}"))
    checks.append(('INFO', f"Total throughput: {data.products['throughput'].sum():,} ops/period"))
    checks.append(('INFO', f"Total storage: {data.products['storage'].sum():,} slots"))
    
    return checks


def export_clean_data(data, output_dir='output'):
    """Export parsed data to clean CSV files for easy inspection."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Department names
    pd.DataFrame({'department': data.departments}).to_csv(
        os.path.join(output_dir, 'departments.csv'), index=False)
    
    # Flow matrix
    flow_df = pd.DataFrame(
        data.flow_matrix,
        index=data.departments,
        columns=data.departments
    )
    flow_df.to_csv(os.path.join(output_dir, 'flow_matrix.csv'))
    
    # Locations
    data.locations.to_csv(
        os.path.join(output_dir, 'locations.csv'), index=False)
    
    # Location distance matrix
    np.savetxt(
        os.path.join(output_dir, 'location_distances.csv'),
        data.location_distance_matrix, delimiter=',', fmt='%.4f')
    
    # Zones
    data.zones.to_csv(
        os.path.join(output_dir, 'zones.csv'), index=False)
    
    # Zone distance matrix (replace inf with string for readability)
    zone_dist_df = pd.DataFrame(
        data.zone_distance_matrix,
        index=data.zone_names,
        columns=data.zone_names
    )
    zone_dist_df.to_csv(os.path.join(output_dir, 'zone_distances.csv'))
    
    # Products
    data.products.to_csv(
        os.path.join(output_dir, 'products.csv'), index=False)
    
    print(f"[OK] Clean data exported to '{output_dir}/' directory")
    print(f"  - departments.csv")
    print(f"  - flow_matrix.csv ({data.flow_matrix.shape[0]}x{data.flow_matrix.shape[1]})")
    print(f"  - locations.csv ({len(data.locations)} rows)")
    print(f"  - location_distances.csv ({data.location_distance_matrix.shape})")
    print(f"  - zones.csv ({len(data.zones)} rows)")
    print(f"  - zone_distances.csv ({data.zone_distance_matrix.shape})")
    print(f"  - products.csv ({len(data.products)} rows)")


# ================================================================
# CLI entry point
# ================================================================
if __name__ == '__main__':
    import sys
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'Group 1.xlsx'
    
    print(f"Loading data from: {filepath}")
    print("-" * 40)
    
    data = load_data(filepath)
    data.summary()
    
    print("\nRunning validation checks...")
    checks = validate_data(data)
    for level, msg in checks:
        icon = {'INFO': '[OK]', 'WARN': '[!!]', 'ERROR': '[XX]'}[level]
        print(f"  {icon} [{level}] {msg}")
    
    print("\nExporting clean data...")
    export_clean_data(data)
