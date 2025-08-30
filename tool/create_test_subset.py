#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import json

def create_test_subset(input_file='data/apps_13_ready_with_zipcodes.parquet', 
                       output_name='test_subset_200', 
                       n_stations=200):
    '''Create representative subset for testing'''
    
    # Load full dataset
    df = pd.read_parquet(input_file)
    print(f'Full dataset: {len(df)} stations, {df.zipcode.nunique()} zipcodes')
    
    # Strategy 1: Geographic clustering - get stations from different regions
    # Divide into geographic quadrants and sample from each
    lat_median = df['latitude'].median()
    lon_median = df['longitude'].median()
    
    quadrants = {
        'NE': df[(df.latitude >= lat_median) & (df.longitude >= lon_median)],
        'NW': df[(df.latitude >= lat_median) & (df.longitude < lon_median)],
        'SE': df[(df.latitude < lat_median) & (df.longitude >= lon_median)],
        'SW': df[(df.latitude < lat_median) & (df.longitude < lon_median)]
    }
    
    # Sample proportionally from each quadrant
    samples = []
    for name, quad_df in quadrants.items():
        n_sample = min(len(quad_df), n_stations // 4)
        if n_sample > 0:
            sample = quad_df.sample(n=n_sample, random_state=42)
            samples.append(sample)
            print(f'Quadrant {name}: {n_sample} stations')
    
    subset = pd.concat(samples).reset_index(drop=True)
    
    # Add some guaranteed interference cases (find closest station pairs)
    # This ensures we test interference constraints
    for i in range(5):  # Add 5 close pairs
        idx = np.random.randint(0, len(subset)-1)
        station = subset.iloc[idx]
        # Find nearest station
        distances = np.sqrt((subset.latitude - station.latitude)**2 + 
                          (subset.longitude - station.longitude)**2)
        distances[idx] = np.inf  # Exclude self
        nearest_idx = distances.argmin()
        print(f'Close pair: stations {idx} and {nearest_idx}, distance: {distances[nearest_idx]*111:.1f} km')
    
    # Analyze subset characteristics
    zipcode_counts = subset.zipcode.value_counts().head(10)
    zipcode_dict = {str(k): int(v) for k, v in zipcode_counts.items()}
    
    stats = {
        'total_stations': int(len(subset)),
        'unique_zipcodes': int(subset.zipcode.nunique()),
        'zipcode_list': zipcode_dict,
        'directional_stations': int((subset.azimuth_deg != 0).sum()) if 'azimuth_deg' in subset else 0,
        'omnidirectional_stations': int((subset.azimuth_deg == 0).sum()) if 'azimuth_deg' in subset else 0,
        'lat_range': [float(subset.latitude.min()), float(subset.latitude.max())],
        'lon_range': [float(subset.longitude.min()), float(subset.longitude.max())],
        'power_range': [float(subset.power_watts.min()), float(subset.power_watts.max())] if 'power_watts' in subset else [0, 0]
    }
    
    # Save subset
    subset.to_csv(f'data/{output_name}.csv', index=False)
    subset.to_parquet(f'data/{output_name}.parquet', index=False)
    
    # Save stats
    with open(f'data/{output_name}_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f'\nSubset saved as data/{output_name}.csv and .parquet')
    print(f'Statistics saved as data/{output_name}_stats.json')
    print(f'\nSubset characteristics:')
    for key, value in stats.items():
        if key != 'zipcode_list':
            print(f'  {key}: {value}')
    
    return subset, stats

if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    create_test_subset(n_stations=n)