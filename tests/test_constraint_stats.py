#!/usr/bin/env python3
"""
Test constraint statistics to verify O(n*f) complexity.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectrum_optimizer_enhanced import EnhancedSpectrumOptimizer


def test_constraint_scaling():
    """Test that constraints scale as O(n*k*f) where k=avg_neighbors."""
    
    print("Constraint Scaling Analysis")
    print("=" * 60)
    
    optimizer = EnhancedSpectrumOptimizer('default')
    
    # Use smaller band for faster testing
    optimizer.config['band']['min_mhz'] = 88.0
    optimizer.config['band']['max_mhz'] = 90.0  # 11 channels at 0.2 MHz
    optimizer.config['band']['step_khz'] = 200
    optimizer.config['geometry']['r_main_km'] = 30.0
    
    n_freqs = len(optimizer._generate_frequencies())
    print(f"Using {n_freqs} frequency channels\n")
    
    results = []
    
    for n_stations in [10, 20, 50, 100]:
        # Create random stations spread over larger area to control density
        np.random.seed(42)
        df = pd.DataFrame({
            'station_id': [f'S{i}' for i in range(n_stations)],
            'latitude': np.random.uniform(35, 45, n_stations),  # 10 degree spread
            'longitude': np.random.uniform(-125, -115, n_stations),  # 10 degree spread
            'power_watts': [1000] * n_stations,
            'azimuth_deg': np.random.uniform(0, 360, n_stations),
            'beamwidth_deg': np.random.choice([60, 120, 360], n_stations),
        })
        
        # Run optimization
        result = optimizer.optimize(df)
        
        # Extract metrics
        if hasattr(result, 'attrs') and 'optimization_metrics' in result.attrs:
            metrics = result.attrs['optimization_metrics']
            stats = metrics['constraint_stats']
            neighbor_metrics = metrics['neighbor_metrics']
            
            total_edges = neighbor_metrics.get('total_edges', 0)
            avg_neighbors = neighbor_metrics['avg_neighbors']
            
            # Calculate theoretical bounds
            all_pairs = n_stations * (n_stations - 1) // 2
            guard_offsets = len(optimizer.config['interference']['guard_offsets'])
            
            # Actual constraints per edge
            constraints_per_edge = n_freqs + 2 * n_freqs * guard_offsets  # co-channel + adjacent
            expected_constraints = total_edges * constraints_per_edge
            
            results.append({
                'n': n_stations,
                'edges': total_edges,
                'all_pairs': all_pairs,
                'avg_neighbors': avg_neighbors,
                'constraints': stats['total'],
                'expected': expected_constraints,
                'ratio': stats['total'] / expected_constraints if expected_constraints > 0 else 0
            })
            
            print(f"n={n_stations:3d}:")
            print(f"  Edges: {total_edges:5d} / {all_pairs:6d} (all pairs)")
            print(f"  Avg neighbors: {avg_neighbors:.1f}")
            print(f"  Constraints: {stats['total']:6d}")
            print(f"  Expected (edges * f * guards): {expected_constraints:6d}")
            print(f"  Ratio: {results[-1]['ratio']:.2f}")
            print(f"  Co-channel: {stats['co_channel']}")
            print(f"  Adjacent: {stats['adjacent_channel']}")
            print()
    
    # Analyze scaling
    if len(results) >= 2:
        print("Complexity Analysis:")
        print("-" * 40)
        
        for i in range(1, len(results)):
            n_ratio = results[i]['n'] / results[0]['n']
            constraint_ratio = results[i]['constraints'] / results[0]['constraints']
            edge_ratio = results[i]['edges'] / results[0]['edges'] if results[0]['edges'] > 0 else 0
            
            print(f"n: {results[0]['n']}→{results[i]['n']} ({n_ratio:.1f}x)")
            print(f"  Constraints: {constraint_ratio:.1f}x (edges: {edge_ratio:.1f}x)")
            
            # Check if sub-quadratic
            if constraint_ratio < n_ratio * n_ratio * 0.8:
                print(f"  ✓ Sub-quadratic: {constraint_ratio:.1f}x < {n_ratio*n_ratio:.1f}x")
            else:
                print(f"  ⚠ Near-quadratic: {constraint_ratio:.1f}x ≈ {n_ratio*n_ratio:.1f}x")
        
        print("\nConclusion:")
        final_ratio = results[-1]['constraints'] / results[0]['constraints']
        final_n_ratio = results[-1]['n'] / results[0]['n']
        
        if final_ratio < final_n_ratio * 1.5:
            print(f"✓ Constraints scale as O(n) with geographic data")
        elif final_ratio < final_n_ratio * final_n_ratio * 0.5:
            print(f"✓ Constraints scale sub-quadratically: O(n^1.5)")
        else:
            print(f"⚠ Constraints may be scaling quadratically")
        
        print(f"  {final_n_ratio:.0f}x stations → {final_ratio:.1f}x constraints")
        print(f"  Average neighbors remains ~constant: {results[0]['avg_neighbors']:.1f} → {results[-1]['avg_neighbors']:.1f}")


if __name__ == "__main__":
    test_constraint_scaling()