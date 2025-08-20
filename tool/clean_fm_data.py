#!/usr/bin/env python3
"""
Clean FM data: remove duplicates and create subset with beamwidth based on azimuth.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def clean_and_prepare_fm_data(input_file: str, output_base: str = None, 
                              subset_size: int = 2000,
                              directional_beamwidth: float = 120.0):
    """
    Clean FM data by removing duplicates and adding beamwidth.
    
    Args:
        input_file: Input CSV file path
        output_base: Base name for output files
        subset_size: Number of stations for subset
        directional_beamwidth: Beamwidth for directional stations
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original data: {len(df)} stations")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Check for duplicates based on station_id
    if 'station_id' in df.columns:
        duplicates = df.duplicated(subset=['station_id'], keep='first')
        num_duplicates = duplicates.sum()
        print(f"\nFound {num_duplicates} duplicate station_ids")
        
        if num_duplicates > 0:
            # Show some duplicate examples
            dup_ids = df[duplicates]['station_id'].head(5).tolist()
            print(f"Example duplicate IDs: {dup_ids}")
            
        # Remove duplicates
        df_clean = df[~duplicates].copy()
    else:
        # Check for duplicates based on lat/lon if no station_id
        print("No station_id column found, checking lat/lon duplicates...")
        duplicates = df.duplicated(subset=['latitude', 'longitude'], keep='first')
        num_duplicates = duplicates.sum()
        print(f"Found {num_duplicates} duplicate locations")
        df_clean = df[~duplicates].copy()
    
    print(f"After removing duplicates: {len(df_clean)} stations")
    
    # Create subset
    if len(df_clean) > subset_size:
        print(f"\nCreating subset of {subset_size} stations...")
        # Use random sampling for better geographic distribution
        df_subset = df_clean.sample(n=subset_size, random_state=42)
        df_subset = df_subset.sort_index()  # Maintain some order
    else:
        df_subset = df_clean
        print(f"Data has {len(df_clean)} stations, using all")
    
    # Add beamwidth based on azimuth
    print("\nAdding beamwidth based on azimuth values...")
    if 'azimuth_deg' in df_subset.columns:
        # Count azimuth distribution
        zero_azimuth = (df_subset['azimuth_deg'] == 0).sum()
        non_zero_azimuth = (df_subset['azimuth_deg'] != 0).sum()
        
        # Add beamwidth column
        df_subset['beamwidth_deg'] = df_subset['azimuth_deg'].apply(
            lambda az: 360.0 if az == 0.0 else directional_beamwidth
        )
        
        print(f"  - Omnidirectional (azimuth=0, beamwidth=360°): {zero_azimuth}")
        print(f"  - Directional (azimuth≠0, beamwidth={directional_beamwidth}°): {non_zero_azimuth}")
    else:
        print("  - No azimuth_deg column found, adding default beamwidth=360°")
        df_subset['beamwidth_deg'] = 360.0
    
    # Generate output filenames
    if output_base is None:
        input_path = Path(input_file)
        output_base = input_path.stem
    
    # Save cleaned full dataset
    clean_file = f"data/{output_base}_clean.csv"
    df_clean.to_csv(clean_file, index=False)
    print(f"\nSaved cleaned full dataset: {clean_file}")
    
    # Save subset with beamwidth
    subset_file = f"data/{output_base}_subset_{subset_size}.csv"
    df_subset.to_csv(subset_file, index=False)
    print(f"Saved subset with beamwidth: {subset_file}")
    
    # Print statistics
    print("\n" + "="*50)
    print("CLEANING SUMMARY")
    print("="*50)
    print(f"Original stations: {len(df)}")
    print(f"Duplicates removed: {num_duplicates}")
    print(f"Clean stations: {len(df_clean)}")
    print(f"Subset size: {len(df_subset)}")
    
    if 'frequency_mhz' in df_subset.columns:
        freq_stats = df_subset['frequency_mhz'].describe()
        print(f"\nFrequency range: {freq_stats['min']:.1f} - {freq_stats['max']:.1f} MHz")
    
    return subset_file, df_subset


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_fm_data.py <input_csv> [subset_size] [beamwidth]")
        print("Example: python clean_fm_data.py data/fmdata2.csv 2000 120")
        sys.exit(1)
    
    input_file = sys.argv[1]
    subset_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    beamwidth = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    
    clean_and_prepare_fm_data(input_file, subset_size=subset_size, 
                              directional_beamwidth=beamwidth)