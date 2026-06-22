"""
Visualization Module for OR Facility Layout Optimization
=========================================================
Generates publication-quality figures for input data analysis.
All figures saved to output/figures/ directory.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import seaborn as sns
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import load_data


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.facecolor': 'white',
        'axes.facecolor': '#fafafa',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_locations(data, output_dir):
    """
    2D scatter plot of all 197 candidate locations, colored by type.
    """
    fig, ax = plt.subplots(figsize=(14, 11))
    
    locs = data.locations.copy()
    
    # Group rare types into "Other"
    type_counts = locs['type'].value_counts()
    rare_types = type_counts[type_counts < 3].index
    locs['type_grouped'] = locs['type'].apply(
        lambda x: 'Other' if x in rare_types else x)
    
    # Color palette
    unique_types = sorted(locs['type_grouped'].unique())
    n_types = len(unique_types)
    palette = sns.color_palette('tab20', n_types)
    color_map = dict(zip(unique_types, palette))
    
    # Plot each type
    for loc_type in unique_types:
        subset = locs[locs['type_grouped'] == loc_type]
        marker = 'o' if loc_type != 'Other' else 'x'
        size = 60 if loc_type != 'Other' else 30
        ax.scatter(subset['x'], subset['y'],
                   c=[color_map[loc_type]], label=f"{loc_type} ({len(subset)})",
                   s=size, marker=marker, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('197 Candidate Locations (Colored by Type)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
              title='Location Type', title_fontsize=9)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'locations_scatter.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_flow_heatmap(data, output_dir):
    """
    Heatmap of the 30x30 department flow matrix.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Shorten department names for readability
    short_names = [d.split('. ', 1)[-1][:25] for d in data.departments]
    
    flow = data.flow_matrix.copy()
    # Mask diagonal (zeros)
    mask = np.eye(30, dtype=bool)
    
    sns.heatmap(flow, ax=ax,
                xticklabels=short_names, yticklabels=short_names,
                cmap='YlOrRd', mask=mask,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Flow (units/period)', 'shrink': 0.8},
                fmt='.0f')
    
    ax.set_title('Department Flow Matrix (30 x 30, Asymmetric)')
    ax.set_xlabel('To Department')
    ax.set_ylabel('From Department')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'flow_heatmap.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_flow_top_pairs(data, output_dir):
    """
    Bar chart of top 20 highest-flow department pairs.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    short_names = [d.split('. ', 1)[-1][:20] for d in data.departments]
    
    # Get top flow pairs
    pairs = []
    for i in range(30):
        for j in range(30):
            if i != j:
                total = data.flow_matrix[i, j] + data.flow_matrix[j, i]
                pairs.append((short_names[i], short_names[j], 
                              data.flow_matrix[i, j], total))
    
    # Sort by total bidirectional flow, deduplicate
    seen = set()
    unique_pairs = []
    for a, b, one_way, total in sorted(pairs, key=lambda x: -x[3]):
        key = tuple(sorted([a, b]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((a, b, total))
    
    top20 = unique_pairs[:20]
    labels = [f"{a} <-> {b}" for a, b, _ in top20]
    values = [v for _, _, v in top20]
    
    colors = sns.color_palette('viridis', 20)
    bars = ax.barh(range(len(top20)), values, color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Total Bidirectional Flow (units/period)')
    ax.set_title('Top 20 Highest-Flow Department Pairs')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                f'{val:,.0f}', va='center', fontsize=7)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'flow_top_pairs.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_zone_distance_heatmap(data, output_dir):
    """
    Heatmap of the 137x137 zone distance matrix with infinity highlighted.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    dist = data.zone_distance_matrix.copy()
    
    # Replace inf with NaN for visualization (will show as white)
    dist_vis = np.where(np.isinf(dist), np.nan, dist)
    
    im = ax.imshow(dist_vis, cmap='plasma', aspect='auto', interpolation='nearest')
    
    # Mark infinity positions
    inf_mask = np.isinf(dist)
    inf_y, inf_x = np.where(inf_mask)
    ax.scatter(inf_x, inf_y, c='red', s=0.5, marker='s', alpha=0.5, label='Infinity')
    
    ax.set_title(f'Zone Distance Matrix (137 x 137)\n{inf_mask.sum()} infinity entries shown in red')
    ax.set_xlabel('Zone Index')
    ax.set_ylabel('Zone Index')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Distance (meters)')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'zone_distance_heatmap.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_zone_demand_capacity(data, output_dir):
    """
    Scatter plot of zone demand vs capacity with construction cost as size.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    zones = data.zones
    
    # Left: Demand vs Capacity
    ax = axes[0]
    scatter = ax.scatter(zones['demand'], zones['capacity'],
                         c=zones['cost'], cmap='RdYlGn_r',
                         s=50, alpha=0.7, edgecolors='grey', linewidth=0.3)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Construction Cost (x$10)')
    ax.set_xlabel('Expected Demand (x1000 KWh)')
    ax.set_ylabel('Substation Capacity (x100 KWh)')
    ax.set_title('Zone Demand vs Capacity')
    
    # Add diagonal line (demand = capacity, after unit alignment)
    # Demand is in 1000 KWh, capacity is in 100 KWh
    # So capacity needs to be 10x demand for self-sufficiency
    ax.axline((0, 0), slope=10, color='red', linestyle='--', alpha=0.5, 
              label='Self-sufficient (cap = 10x demand)')
    ax.legend(fontsize=8)
    
    # Right: Distribution of costs
    ax = axes[1]
    ax.hist(zones['cost'], bins=25, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(zones['cost'].mean(), color='red', linestyle='--', 
               label=f"Mean: {zones['cost'].mean():,.0f}")
    ax.axvline(zones['cost'].median(), color='orange', linestyle='--',
               label=f"Median: {zones['cost'].median():,.0f}")
    ax.set_xlabel('Construction Cost (x$10)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Substation Construction Costs')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'zone_demand_capacity.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_products(data, output_dir):
    """
    Combined bar chart of product throughput and storage requirements.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    products = data.products.sort_values('throughput', ascending=True)
    
    # Left: Throughput
    ax = axes[0]
    colors_tp = sns.color_palette('Blues_d', len(products))
    ax.barh(range(len(products)), products['throughput'], color=colors_tp)
    ax.set_yticks(range(len(products)))
    ax.set_yticklabels(products['product_id'], fontsize=8)
    ax.set_xlabel('Throughput (operations/period)')
    ax.set_title('Product Throughput')
    
    # Add value labels
    for i, (_, row) in enumerate(products.iterrows()):
        ax.text(row['throughput'] + 100, i, f"{row['throughput']:,}", 
                va='center', fontsize=7)
    
    # Right: Storage
    products_s = data.products.sort_values('storage', ascending=True)
    ax = axes[1]
    colors_st = sns.color_palette('Oranges_d', len(products_s))
    ax.barh(range(len(products_s)), products_s['storage'], color=colors_st)
    ax.set_yticks(range(len(products_s)))
    ax.set_yticklabels(products_s['product_id'], fontsize=8)
    ax.set_xlabel('Storage Slots Required')
    ax.set_title('Product Storage Requirements')
    
    for i, (_, row) in enumerate(products_s.iterrows()):
        ax.text(row['storage'] + 30, i, f"{row['storage']:,}", 
                va='center', fontsize=7)
    
    plt.suptitle('Product Demand Analysis (20 Products)', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'products_analysis.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_throughput_vs_storage(data, output_dir):
    """
    Scatter plot: throughput vs storage per product.
    Useful for identifying high-priority products for layout.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    p = data.products
    
    ax.scatter(p['throughput'], p['storage'], s=100, c='steelblue',
               edgecolors='white', linewidth=1, alpha=0.8, zorder=5)
    
    # Label each point
    for _, row in p.iterrows():
        ax.annotate(row['product_id'].replace('Product-', 'P'),
                     (row['throughput'], row['storage']),
                     textcoords='offset points', xytext=(8, 5),
                     fontsize=8, color='#333')
    
    ax.set_xlabel('Throughput (operations/period)')
    ax.set_ylabel('Storage Slots Required')
    ax.set_title('Product Throughput vs Storage\n(Top-right = highest layout priority)')
    
    # Highlight quadrants
    med_tp = p['throughput'].median()
    med_st = p['storage'].median()
    ax.axvline(med_tp, color='grey', linestyle=':', alpha=0.5)
    ax.axhline(med_st, color='grey', linestyle=':', alpha=0.5)
    
    ax.text(p['throughput'].max() * 0.95, p['storage'].max() * 0.95,
            'HIGH PRIORITY', ha='right', fontsize=9, color='red', alpha=0.6, fontweight='bold')
    ax.text(p['throughput'].min() * 1.1, p['storage'].min() * 1.1,
            'LOW PRIORITY', ha='left', fontsize=9, color='green', alpha=0.6, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'throughput_vs_storage.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_location_fixed_distance(data, output_dir):
    """
    Scatter plot of locations with fixed-travel distance shown as marker size.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    locs = data.locations
    
    scatter = ax.scatter(locs['x'], locs['y'],
                         c=locs['fixed_dist'], cmap='coolwarm',
                         s=locs['fixed_dist'] * 0.5 + 10,
                         alpha=0.7, edgecolors='grey', linewidth=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Fixed-Travel Distance')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Candidate Locations: Fixed-Travel Distance\n(Size & color = distance overhead)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'locations_fixed_dist.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


# ================================================================
# CLI entry point
# ================================================================
if __name__ == '__main__':
    setup_style()
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'Group 1.xlsx'
    output_dir = os.path.join('output', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from: {filepath}")
    data = load_data(filepath)
    
    print(f"\nGenerating visualizations -> {output_dir}/")
    print("-" * 40)
    
    plot_locations(data, output_dir)
    plot_flow_heatmap(data, output_dir)
    plot_flow_top_pairs(data, output_dir)
    plot_zone_distance_heatmap(data, output_dir)
    plot_zone_demand_capacity(data, output_dir)
    plot_products(data, output_dir)
    plot_throughput_vs_storage(data, output_dir)
    plot_location_fixed_distance(data, output_dir)
    
    print("-" * 40)
    print(f"Done! {8} figures generated.")
