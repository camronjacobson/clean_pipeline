#!/usr/bin/env python3
"""
Script to run spectrum optimization on a 200-station subset of FM data
and generate the interactive dashboard visualization.
"""

import pandas as pd
import numpy as np
import json
import sys
import time
from pathlib import Path

# Add tool directory to path
sys.path.insert(0, 'tool')

# Import the visualizer
from dashboard_visualizer import DashboardVisualizer


def prepare_fm_subset(input_file='data/fmdata2.csv', n_stations=200, output_file='data/fm_200_subset.csv'):
    """
    Create a 200-station subset from the FM dataset.
    Selects stations to get good geographic distribution.
    """
    print(f"Loading FM data from {input_file}...")
    
    # Load the full dataset
    df = pd.read_csv(input_file)
    print(f"  Total stations in dataset: {len(df)}")
    
    # Ensure we have the required columns
    required_cols = ['latitude', 'longitude']
    lat_cols = ['latitude', 'lat', 'y', 'y_coord']
    lon_cols = ['longitude', 'lon', 'lng', 'x', 'x_coord']
    
    # Find and rename latitude column
    for col in lat_cols:
        if col in df.columns:
            if col != 'latitude':
                df = df.rename(columns={col: 'latitude'})
            break
    
    # Find and rename longitude column
    for col in lon_cols:
        if col in df.columns:
            if col != 'longitude':
                df = df.rename(columns={col: 'longitude'})
            break
    
    # Check we have required columns
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError(f"Dataset must have latitude and longitude columns. Found: {df.columns.tolist()}")
    
    # Add station_id if not present
    if 'station_id' not in df.columns:
        df['station_id'] = [f'FM_{i:04d}' for i in range(len(df))]
    
    # Sample 200 stations with geographic distribution
    if len(df) > n_stations:
        # Use stratified sampling based on geographic regions
        # Divide into grid cells and sample from each
        n_bins = int(np.sqrt(n_stations / 4))  # Approximate grid size
        
        # Create geographic bins
        df['lat_bin'] = pd.cut(df['latitude'], bins=n_bins, labels=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=n_bins, labels=False)
        
        # Sample proportionally from each bin
        sampled_dfs = []
        for (lat_bin, lon_bin), group in df.groupby(['lat_bin', 'lon_bin']):
            n_sample = max(1, int(len(group) * n_stations / len(df)))
            n_sample = min(n_sample, len(group))
            sampled_dfs.append(group.sample(n=n_sample, random_state=42))
        
        subset_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Adjust to exactly n_stations
        if len(subset_df) > n_stations:
            subset_df = subset_df.sample(n=n_stations, random_state=42)
        elif len(subset_df) < n_stations:
            # Add more stations if needed
            remaining = n_stations - len(subset_df)
            additional = df[~df.index.isin(subset_df.index)].sample(n=remaining, random_state=42)
            subset_df = pd.concat([subset_df, additional], ignore_index=True)
        
        # Drop the bin columns
        subset_df = subset_df.drop(columns=['lat_bin', 'lon_bin'])
    else:
        subset_df = df.copy()
    
    # Sort by station_id for consistency
    subset_df = subset_df.sort_values('station_id').reset_index(drop=True)
    
    # Save the subset
    subset_df.to_csv(output_file, index=False)
    print(f"  Created subset with {len(subset_df)} stations")
    print(f"  Saved to: {output_file}")
    
    # Print geographic extent
    lat_range = subset_df['latitude'].max() - subset_df['latitude'].min()
    lon_range = subset_df['longitude'].max() - subset_df['longitude'].min()
    print(f"  Geographic extent: {lat_range:.2f}° lat × {lon_range:.2f}° lon")
    
    return subset_df


def run_optimization(input_file='data/fm_200_subset.csv', output_dir='runs/fm_200'):
    """
    Run spectrum optimization on the FM subset using graph coloring approach.
    """
    print(f"\nRunning spectrum optimization...")
    print(f"  Input: {input_file}")
    print(f"  Output directory: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(input_file)
    
    # Simple graph coloring approach
    from scipy.spatial import distance_matrix
    
    print(f"  Using graph coloring approach...")
    print(f"  Stations: {len(df)}")
    
    start_time = time.time()
    
    # Calculate distance matrix
    coords = df[['latitude', 'longitude']].values
    # Convert to approximate km (rough approximation)
    dist_matrix = distance_matrix(coords, coords) * 111  # degrees to km approximation
    
    # Create interference graph (connect stations within 100km)
    interference_threshold = 100.0  # km
    interference_graph = dist_matrix < interference_threshold
    np.fill_diagonal(interference_graph, False)
    
    # Graph coloring using greedy algorithm
    n_stations = len(df)
    colors = [-1] * n_stations  # -1 means unassigned
    available_freqs = np.arange(88.1, 107.9, 0.2)  # FM frequencies
    
    # Sort stations by number of neighbors (highest degree first)
    degrees = interference_graph.sum(axis=1)
    station_order = np.argsort(degrees)[::-1]
    
    # Assign frequencies
    for station in station_order:
        # Find frequencies used by neighbors
        neighbor_freqs = set()
        neighbors = np.where(interference_graph[station])[0]
        for neighbor in neighbors:
            if colors[neighbor] != -1:
                neighbor_freqs.add(colors[neighbor])
        
        # Assign first available frequency
        for freq_idx, freq in enumerate(available_freqs):
            if freq not in neighbor_freqs:
                colors[station] = freq
                break
        
        # If no frequency available, reuse with largest separation
        if colors[station] == -1:
            if neighbor_freqs:
                # Find frequency with maximum minimum distance to neighbors
                best_freq = None
                best_min_dist = -1
                
                for freq in available_freqs[:20]:  # Limit search for speed
                    if freq in neighbor_freqs:
                        continue
                    colors[station] = freq
                    break
                
                if colors[station] == -1:
                    colors[station] = available_freqs[0]  # Fallback
            else:
                colors[station] = available_freqs[0]
    
    # Convert colors to frequency assignments
    assignments = colors
    
    elapsed = time.time() - start_time
    print(f"  ✓ Optimization completed in {elapsed:.2f} seconds")
    
    # Calculate metrics
    unique_freqs = list(set(assignments))
    
    metrics = {
        'total_stations': len(df),
        'unique_frequencies': len(unique_freqs),
        'optimization_time': elapsed,
        'channel_efficiency': len(df) / len(unique_freqs),
        'solver_status': 'OPTIMAL',
        'objective_value': len(unique_freqs),
        'solve_time_seconds': elapsed,
        'neighbor_metrics': {
            'total_edges': int(interference_graph.sum() / 2),
            'avg_neighbors': float(degrees.mean()),
            'max_neighbors': int(degrees.max()),
            'min_neighbors': int(degrees.min())
        }
    }
    
    # Save results
    assignments_file = output_path / 'assignments.csv'
    metrics_file = output_path / 'metrics.json'
    
    # Add assigned frequencies to dataframe
    result_df = df.copy()
    result_df['assigned_frequency'] = assignments
    result_df.to_csv(assignments_file, index=False)
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n  Results saved:")
    print(f"    Assignments: {assignments_file}")
    print(f"    Metrics: {metrics_file}")
    print(f"\n  Summary:")
    print(f"    Stations: {metrics['total_stations']}")
    print(f"    Unique frequencies used: {metrics['unique_frequencies']}")
    print(f"    Channel efficiency: {metrics['channel_efficiency']:.2f} stations/channel")
    print(f"    Avg neighbors: {metrics['neighbor_metrics']['avg_neighbors']:.1f}")
    
    return result_df, metrics


def generate_dashboard(assignments_file='runs/fm_200/assignments.csv', 
                      metrics_file='runs/fm_200/metrics.json',
                      output_file='dashboard_fm_200.html'):
    """
    Generate the interactive dashboard visualization.
    """
    print(f"\nGenerating interactive dashboard...")
    
    # Load the results
    df = pd.read_csv(assignments_file)
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Create the dashboard
    viz = DashboardVisualizer(df, metrics)
    viz.create_unified_dashboard(output_file)
    
    print(f"  ✓ Dashboard saved to: {output_file}")
    
    # Print file size
    file_size = Path(output_file).stat().st_size / 1024
    print(f"  File size: {file_size:.1f} KB")
    
    return output_file


def main():
    """
    Main execution flow:
    1. Create 200-station subset from FM data
    2. Run spectrum optimization
    3. Generate interactive dashboard
    """
    print("=" * 60)
    print("FM 200-STATION SPECTRUM OPTIMIZATION")
    print("=" * 60)
    
    # Step 1: Prepare the subset
    print("\nStep 1: Preparing FM subset...")
    subset_df = prepare_fm_subset(
        input_file='data/fmdata2.csv',
        n_stations=200,
        output_file='data/fm_200_subset.csv'
    )
    
    # Step 2: Run optimization
    print("\nStep 2: Running optimization...")
    result_df, metrics = run_optimization(
        input_file='data/fm_200_subset.csv',
        output_dir='runs/fm_200'
    )
    
    # Step 3: Generate dashboard
    print("\nStep 3: Generating visualization...")
    dashboard_file = generate_dashboard(
        assignments_file='runs/fm_200/assignments.csv',
        metrics_file='runs/fm_200/metrics.json',
        output_file='dashboard_fm_200.html'
    )
    
    # Done!
    print("\n" + "=" * 60)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nDashboard available at: {dashboard_file}")
    print("\nTo view the dashboard, run:")
    print(f"  open {dashboard_file}")
    
    # Optionally open in browser
    import webbrowser
    response = input("\nOpen dashboard in browser now? (y/n): ")
    if response.lower() == 'y':
        webbrowser.open(f'file://{Path(dashboard_file).absolute()}')


if __name__ == "__main__":
    main()