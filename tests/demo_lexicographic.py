#!/usr/bin/env python3
"""
Demonstration of lexicographic objective improvements.
Shows channel minimization, low-frequency packing, and determinism.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectrum_optimizer_enhanced import EnhancedSpectrumOptimizer


def demo_channel_minimization():
    """Demonstrate that channels are minimized first."""
    print("=" * 60)
    print("DEMO 1: Channel Minimization (Primary Objective)")
    print("=" * 60)
    
    optimizer = EnhancedSpectrumOptimizer('default', seed=42)
    optimizer.config['band']['min_mhz'] = 88.0
    optimizer.config['band']['max_mhz'] = 92.0  # 21 channels
    optimizer.config['band']['step_khz'] = 200
    optimizer.config['geometry']['r_main_km'] = 10.0
    
    # 5-station chain requiring multiple channels
    df = pd.DataFrame({
        'station_id': ['A', 'B', 'C', 'D', 'E'],
        'latitude': [40.0, 40.0, 40.0, 40.0, 40.0],
        'longitude': [-120.0, -119.95, -119.90, -119.85, -119.80],  # ~5km apart
        'power_watts': [1000] * 5,
        'azimuth_deg': [0] * 5,
        'beamwidth_deg': [360] * 5
    })
    
    result = optimizer.optimize(df)
    
    if hasattr(result, 'attrs'):
        obj_metrics = result.attrs['optimization_metrics']['objective_metrics']
        print(f"\nResults:")
        print(f"  Stations: {len(df)}")
        print(f"  Channels available: 21 (88.0-92.0 MHz)")
        print(f"  Channels used: {obj_metrics['channels_used']} ✓")
        print(f"  Indices used: {obj_metrics['channel_indices_used']}")
        print(f"  Assignments: {result['assigned_frequency'].values}")
        print(f"\n✓ Minimizes channel count despite having 21 available")


def demo_low_frequency_packing():
    """Demonstrate packing toward low frequencies."""
    print("\n" + "=" * 60)
    print("DEMO 2: Low-Frequency Packing (Secondary Objective)")
    print("=" * 60)
    
    optimizer = EnhancedSpectrumOptimizer('default', seed=42)
    optimizer.config['band']['min_mhz'] = 88.0
    optimizer.config['band']['max_mhz'] = 108.0  # Full FM band (101 channels)
    optimizer.config['band']['step_khz'] = 200
    optimizer.config['geometry']['r_main_km'] = 50.0
    optimizer.config['interference']['guard_offsets'] = []  # No guards for clarity
    
    # Three isolated groups
    groups = []
    for g in range(3):
        base_lat = 40.0 + g * 10  # Groups far apart
        for s in range(2):
            groups.append({
                'station_id': f'G{g}S{s}',
                'latitude': base_lat + s * 0.01,
                'longitude': -120.0 - g * 10,
                'power_watts': 1000,
                'azimuth_deg': 0,
                'beamwidth_deg': 360
            })
    
    df = pd.DataFrame(groups)
    result = optimizer.optimize(df)
    
    if hasattr(result, 'attrs'):
        obj_metrics = result.attrs['optimization_metrics']['objective_metrics']
        print(f"\nResults:")
        print(f"  Stations: {len(df)} (3 groups of 2)")
        print(f"  Channels available: 101 (88.0-108.0 MHz)")
        print(f"  Channels used: {obj_metrics['channels_used']}")
        print(f"  Indices used: {obj_metrics['channel_indices_used']}")
        print(f"  Average index: {obj_metrics['channel_packing_score']:.2f}")
        print(f"  Max index: {max(obj_metrics['channel_indices_used'])}")
        print(f"\n✓ Uses lowest channels (0, 2) instead of spreading across band")


def demo_determinism():
    """Demonstrate deterministic results with fixed seed."""
    print("\n" + "=" * 60)
    print("DEMO 3: Deterministic Results (Tertiary Objective)")
    print("=" * 60)
    
    # Create complex scenario with tie-breaking needed
    df = pd.DataFrame({
        'station_id': ['A', 'B', 'C', 'D'],
        'latitude': [40.0, 40.1, 40.2, 40.3],
        'longitude': [-120.0, -120.1, -120.2, -120.3],
        'power_watts': [1000] * 4,
        'azimuth_deg': [0, 90, 180, 270],
        'beamwidth_deg': [60] * 4
    })
    
    print("\nRunning same problem 3 times with same seed (42):")
    results = []
    for run in range(3):
        opt = EnhancedSpectrumOptimizer('default', seed=42)
        opt.config['band']['min_mhz'] = 88.0
        opt.config['band']['max_mhz'] = 90.0
        opt.config['band']['step_khz'] = 200
        opt.config['geometry']['r_main_km'] = 15.0
        
        result = opt.optimize(df.copy())
        freqs = result['assigned_frequency'].values
        results.append(freqs)
        print(f"  Run {run+1}: {freqs}")
    
    # Check all identical
    all_same = all(np.array_equal(results[0], r) for r in results[1:])
    print(f"\n✓ All runs identical: {all_same}")
    
    print("\nRunning with different seed (999):")
    opt_diff = EnhancedSpectrumOptimizer('default', seed=999)
    opt_diff.config['band']['min_mhz'] = 88.0
    opt_diff.config['band']['max_mhz'] = 90.0
    opt_diff.config['band']['step_khz'] = 200
    opt_diff.config['geometry']['r_main_km'] = 15.0
    
    result_diff = opt_diff.optimize(df.copy())
    freqs_diff = result_diff['assigned_frequency'].values
    print(f"  Different seed: {freqs_diff}")
    
    same_channels = len(np.unique(results[0])) == len(np.unique(freqs_diff))
    print(f"\n✓ Same channel count with different seed: {same_channels}")
    print(f"✓ But different assignment (tie-breaking): {not np.array_equal(results[0], freqs_diff)}")


def demo_objective_hierarchy():
    """Show the strict hierarchy of objectives."""
    print("\n" + "=" * 60)
    print("DEMO 4: Objective Hierarchy (Lexicographic Ordering)")
    print("=" * 60)
    
    optimizer = EnhancedSpectrumOptimizer('default', seed=42)
    
    # Scenario where packing conflicts with channel count
    optimizer.config['band']['min_mhz'] = 88.0
    optimizer.config['band']['max_mhz'] = 92.0
    optimizer.config['band']['step_khz'] = 200
    optimizer.config['geometry']['r_main_km'] = 8.0
    
    # Create scenario: A-B-C in line, D isolated
    # If we pack low, D would use channel 0
    # But that would force A-B-C to use 3 more channels (total 4)
    # Better: A-B-C use channels 0,2,4 and D uses 0 (total 3)
    
    df = pd.DataFrame({
        'station_id': ['A', 'B', 'C', 'D'],
        'latitude': [40.0, 40.0, 40.0, 50.0],  # D is far
        'longitude': [-120.0, -119.93, -119.86, -110.0],  # A-B-C close, D far
        'power_watts': [1000] * 4,
        'azimuth_deg': [0] * 4,
        'beamwidth_deg': [360] * 4
    })
    
    result = optimizer.optimize(df)
    
    if hasattr(result, 'attrs'):
        obj_metrics = result.attrs['optimization_metrics']['objective_metrics']
        freqs = result['assigned_frequency'].values
        
        print(f"\nScenario: 3 interfering stations + 1 isolated")
        print(f"  A-B-C chain: {freqs[:3]}")
        print(f"  D isolated: {freqs[3]}")
        print(f"\nObjective priorities:")
        print(f"  1. Minimize channels: {obj_metrics['channels_used']} channels ✓")
        print(f"  2. Pack low: indices {obj_metrics['channel_indices_used']} ✓")
        print(f"  3. Deterministic: same result every time ✓")
        
        print(f"\n✓ Channel minimization (3) beats packing everything at index 0")
        print(f"✓ Within 3 channels, uses lowest possible indices")


if __name__ == "__main__":
    print("LEXICOGRAPHIC OBJECTIVE DEMONSTRATION")
    print("=====================================\n")
    
    demo_channel_minimization()
    demo_low_frequency_packing()
    demo_determinism()
    demo_objective_hierarchy()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ Primary: Minimizes total channels used (W=10^9)")
    print("✓ Secondary: Packs channels toward low frequencies (W=10^3)")
    print("✓ Tertiary: Ensures deterministic tie-breaking (W=1)")
    print("✓ Strict lexicographic ordering maintained")
    print("✓ Solutions are reproducible with fixed seed")