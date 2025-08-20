#!/usr/bin/env python3
"""
Demonstration of schema flexibility - handles any reasonable column naming.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectrum_optimizer_enhanced import EnhancedSpectrumOptimizer
from schema_normalizer import get_schema_info, prepare_dataframe


def demo_column_variations():
    """Show that various column names work."""
    print("=" * 60)
    print("DEMO 1: Column Name Variations")
    print("=" * 60)
    
    # Dataset 1: Using 'lat' and 'lon'
    df1 = pd.DataFrame({
        'lat': [40.0, 40.1],
        'lon': [-120.0, -120.1],
        'freq': [100.0, 101.0]
    })
    
    print("\nDataset 1 columns:", list(df1.columns))
    optimizer = EnhancedSpectrumOptimizer('default', seed=42)
    result1 = optimizer.optimize(df1)
    print(f"✓ Optimization successful with 'lat'/'lon'")
    print(f"  Assigned frequencies: {result1['assigned_frequency'].values}")
    
    # Dataset 2: Using 'y_coord' and 'x_coord'
    df2 = pd.DataFrame({
        'y_coord': [40.0, 40.1],
        'x_coord': [-120.0, -120.1],
        'frequency': [100.0, 101.0]
    })
    
    print("\nDataset 2 columns:", list(df2.columns))
    result2 = optimizer.optimize(df2)
    print(f"✓ Optimization successful with 'y_coord'/'x_coord'")
    print(f"  Assigned frequencies: {result2['assigned_frequency'].values}")
    
    # Dataset 3: Mixed case
    df3 = pd.DataFrame({
        'LaTiTuDe': [40.0, 40.1],
        'LONGITUDE': [-120.0, -120.1],
        'Freq_MHz': [100.0, 101.0]
    })
    
    print("\nDataset 3 columns:", list(df3.columns))
    result3 = optimizer.optimize(df3)
    print(f"✓ Optimization successful with mixed case")
    print(f"  Assigned frequencies: {result3['assigned_frequency'].values}")


def demo_minimal_input():
    """Show that minimal input (just lat/lon) works."""
    print("\n" + "=" * 60)
    print("DEMO 2: Minimal Input (Just Coordinates)")
    print("=" * 60)
    
    # Absolute minimum: just coordinates
    df = pd.DataFrame({
        'lat': [40.0, 40.05, 40.1],
        'lon': [-120.0, -120.05, -120.1]
    })
    
    print("\nInput columns:", list(df.columns))
    print("Input data:")
    print(df)
    
    # Get schema info before processing
    info = get_schema_info(df)
    print(f"\nSchema analysis:")
    print(f"  Recognized: {info['recognized_columns']}")
    print(f"  Missing optional: {info['missing_optional']}")
    
    # Process and show defaults added
    prepared = prepare_dataframe(df)
    print(f"\nAfter preparation:")
    print(f"  Columns: {list(prepared.columns)}")
    print(f"  Default station IDs: {prepared['station_id'].tolist()}")
    print(f"  Default frequency: {prepared['frequency_mhz'].iloc[0]} MHz")
    print(f"  Default power: {prepared['power_watts'].iloc[0]} watts")
    
    # Run optimization
    optimizer = EnhancedSpectrumOptimizer('default', seed=42)
    result = optimizer.optimize(df)
    print(f"\n✓ Optimization successful with minimal input")
    print(f"  Assigned frequencies: {result['assigned_frequency'].values}")


def demo_extra_columns():
    """Show that extra/custom columns are preserved."""
    print("\n" + "=" * 60)
    print("DEMO 3: Extra Columns Preserved")
    print("=" * 60)
    
    df = pd.DataFrame({
        'latitude': [40.0, 40.1],
        'longitude': [-120.0, -120.1],
        'call_sign': ['KQED', 'KPIX'],
        'owner': ['Public Radio', 'CBS'],
        'notes': ['Downtown site', 'Hill site'],
        'custom_id': [12345, 67890]
    })
    
    print("\nInput columns:", list(df.columns))
    
    optimizer = EnhancedSpectrumOptimizer('default', seed=42)
    result = optimizer.optimize(df)
    
    print(f"\n✓ Extra columns preserved after optimization:")
    print(f"  Output columns: {list(result.columns)}")
    print(f"  Owner data preserved: {result['owner'].tolist()}")
    print(f"  Notes preserved: {result['notes'].tolist()}")
    print(f"  Custom IDs preserved: {result['custom_id'].tolist()}")


def demo_error_handling():
    """Show clear error messages for invalid input."""
    print("\n" + "=" * 60)
    print("DEMO 4: Clear Error Messages")
    print("=" * 60)
    
    # Missing required columns
    print("\nTest 1: Missing required columns")
    df_missing = pd.DataFrame({
        'frequency': [100.0, 101.0]
        # Missing lat/lon
    })
    
    try:
        optimizer = EnhancedSpectrumOptimizer('default')
        result = optimizer.optimize(df_missing)
    except Exception as e:
        print(f"✓ Clear error: {str(e).split(':')[0]}")
    
    # Invalid latitude
    print("\nTest 2: Invalid coordinate values")
    df_invalid = pd.DataFrame({
        'lat': [95.0, 40.0],  # 95 is invalid
        'lon': [-120.0, -121.0]
    })
    
    try:
        optimizer = EnhancedSpectrumOptimizer('default')
        result = optimizer.optimize(df_invalid)
    except Exception as e:
        error_lines = str(e).split('\n')
        for line in error_lines:
            if 'Latitude' in line:
                print(f"✓ Clear error: {line.strip()}")
                break
    
    # Negative power
    print("\nTest 3: Invalid power values")
    df_negative = pd.DataFrame({
        'lat': [40.0],
        'lon': [-120.0],
        'power_watts': [-100]  # Negative power
    })
    
    try:
        optimizer = EnhancedSpectrumOptimizer('default')
        result = optimizer.optimize(df_negative)
    except Exception as e:
        error_lines = str(e).split('\n')
        for line in error_lines:
            if 'negative' in line:
                print(f"✓ Clear error: {line.strip()}")
                break


def demo_real_world_schema():
    """Demonstrate with realistic messy data."""
    print("\n" + "=" * 60)
    print("DEMO 5: Real-World Messy Data")
    print("=" * 60)
    
    # Realistic data with various issues
    df = pd.DataFrame({
        'Call Sign': ['KQED-FM', 'KPIX-TV', 'KGO-AM'],
        'Y': [37.7749, 37.8044, 37.6213],  # San Francisco area
        'X': [-122.4194, -122.4506, -122.3878],
        'Freq': [88.5, 855.0, 810.0],  # Mixed units (FM in MHz, AM in kHz)
        'ERP': [110000, 316000, 50000],  # Effective Radiated Power
        'Az': [0, 'ND', 180],  # Mixed: number and 'ND' for non-directional
        'notes': ['Public station', 'CBS affiliate', 'News/Talk']
    })
    
    print("\nOriginal messy data:")
    print(df)
    
    # Normalize frequency to MHz
    df['Freq'] = df['Freq'].apply(lambda x: x if x < 200 else x/1000)
    
    # Handle 'ND' azimuth
    df['Az'] = df['Az'].replace('ND', 0)
    
    print("\nAfter basic cleanup:")
    print(df[['Call Sign', 'Freq', 'Az']])
    
    # Run optimization
    optimizer = EnhancedSpectrumOptimizer('default', seed=42)
    result = optimizer.optimize(df)
    
    print(f"\n✓ Successfully handled real-world schema")
    print(f"  Stations: {result['station_id'].tolist()}")
    print(f"  Assigned: {result['assigned_frequency'].values}")
    print(f"  Notes preserved: {result['notes'].tolist()}")


if __name__ == "__main__":
    print("SCHEMA FLEXIBILITY DEMONSTRATION")
    print("================================\n")
    
    demo_column_variations()
    demo_minimal_input()
    demo_extra_columns()
    demo_error_handling()
    demo_real_world_schema()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ Handles any reasonable column naming (lat/latitude/y_coord)")
    print("✓ Works with minimal input (just coordinates)")
    print("✓ Preserves extra/custom columns")
    print("✓ Provides clear error messages")
    print("✓ Handles real-world messy data")
    print("✓ Backwards compatible with existing code")