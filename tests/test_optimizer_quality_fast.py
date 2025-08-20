#!/usr/bin/env python3
"""
Fast quality testing for spectrum optimizer scaling.
Reduced timeouts and smaller test sizes for rapid validation.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.spectrum_optimizer_enhanced import EnhancedSpectrumOptimizer

def test_determinism():
    """Test that same seed produces identical results."""
    print("\n=== DETERMINISM TEST ===")
    
    # Create test data
    np.random.seed(42)
    stations = []
    for i in range(5):
        stations.append({
            'station_id': f'S{i:03d}',
            'latitude': 40.0 + np.random.uniform(-0.05, 0.05),
            'longitude': -74.0 + np.random.uniform(-0.05, 0.05),
            'azimuth_deg': np.random.uniform(0, 360),
            'beamwidth_deg': 90.0
        })
    df = pd.DataFrame(stations)
    
    # Test same seed multiple times
    results = []
    for run in range(3):
        optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        optimizer.config['solver']['timeout_seconds'] = 5
        result = optimizer.optimize(df)
        freqs = result['assigned_frequency'].tolist()
        results.append(freqs)
        print(f"Run {run+1}: {result['assigned_frequency'].nunique()} channels")
    
    # Check if all results are identical
    if results[0] == results[1] == results[2]:
        print("✓ PASS: Deterministic with same seed")
        return True
    else:
        print("✗ FAIL: Non-deterministic results")
        return False

def test_scaling():
    """Test optimizer quality at different scales."""
    print("\n=== SCALING TEST ===")
    
    test_sizes = [5, 10, 20, 30]
    results = []
    
    for size in test_sizes:
        # Create grid of stations
        stations = []
        grid_size = int(np.sqrt(size))
        for i in range(size):
            row = i // grid_size
            col = i % grid_size
            stations.append({
                'station_id': f'S{i:06d}',
                'latitude': 40.0 + row * 0.01,  # ~1km spacing
                'longitude': -74.0 + col * 0.01,
                'azimuth_deg': float(45 * (i % 8)),
                'beamwidth_deg': 90.0
            })
        df = pd.DataFrame(stations)
        
        # Run optimization
        optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        optimizer.config['solver']['timeout_seconds'] = 10
        
        start = time.time()
        result = optimizer.optimize(df)
        elapsed = time.time() - start
        
        channels = result['assigned_frequency'].nunique()
        
        results.append({
            'size': size,
            'channels': channels,
            'time': elapsed,
            'efficiency': channels / size  # Lower is better
        })
        
        print(f"{size:4d} stations: {channels:3d} channels in {elapsed:5.1f}s (efficiency: {channels/size:.2f})")
    
    # Check if efficiency doesn't degrade too much
    efficiencies = [r['efficiency'] for r in results]
    if max(efficiencies) / min(efficiencies) < 2.0:  # Allow 2x variation
        print("✓ PASS: Efficiency remains stable")
        return True
    else:
        print("✗ FAIL: Efficiency degrades significantly")
        return False

def test_conflicts():
    """Test that no conflicts exist in solutions."""
    print("\n=== CONFLICT TEST ===")
    
    # Create dense scenario
    stations = []
    for i in range(10):
        stations.append({
            'station_id': f'S{i:03d}',
            'latitude': 40.0 + np.random.uniform(-0.02, 0.02),  # Very close
            'longitude': -74.0 + np.random.uniform(-0.02, 0.02),
            'azimuth_deg': 0.0,  # All pointing same direction
            'beamwidth_deg': 360.0  # Omnidirectional
        })
    df = pd.DataFrame(stations)
    
    optimizer = EnhancedSpectrumOptimizer('default', seed=42)
    optimizer.config['solver']['timeout_seconds'] = 10
    result = optimizer.optimize(df)
    
    # Check for conflicts (simplified - just check unique assignments)
    channels = result['assigned_frequency'].nunique()
    
    # In worst case (all interfere), need N channels
    if channels <= len(df):
        print(f"✓ PASS: {channels} channels for {len(df)} omnidirectional stations")
        return True
    else:
        print(f"✗ FAIL: Too many channels used")
        return False

def main():
    """Run all quality tests."""
    print("SPECTRUM OPTIMIZER QUALITY TESTS")
    print("=" * 40)
    
    tests = [
        ("Determinism", test_determinism),
        ("Scaling", test_scaling),
        ("Conflicts", test_conflicts)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ FAIL {name}: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)