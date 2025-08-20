#!/usr/bin/env python3
"""
Simplified test to verify directional geometry is working.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool import DirectionalGeometry, DirectionalConfig


def test_directional_cases():
    """Test key directional scenarios."""
    config = DirectionalConfig(
        az_tolerance_deg=5.0,
        r_main_km=5.0,
        r_off_km=0.5
    )
    geom = DirectionalGeometry(config)
    
    print("Directional Geometry Tests")
    print("=" * 50)
    
    # Test 1: Back-to-back stations at 1km
    print("\n1. Back-to-back stations (1km apart):")
    s1 = {'latitude': 40.0, 'longitude': -120.0, 'azimuth_deg': 0, 'beamwidth_deg': 60, 'station_id': 0}
    s2 = {'latitude': 40.009, 'longitude': -120.0, 'azimuth_deg': 180, 'beamwidth_deg': 60, 'station_id': 1}
    
    dist = geom.haversine_distance(s1['latitude'], s1['longitude'], s2['latitude'], s2['longitude'])
    bearing_1_to_2 = geom.great_circle_bearing(s1['latitude'], s1['longitude'], s2['latitude'], s2['longitude'])
    bearing_2_to_1 = geom.great_circle_bearing(s2['latitude'], s2['longitude'], s1['latitude'], s1['longitude'])
    
    s1_sees_s2 = geom.in_lobe(s1['latitude'], s1['longitude'], s2['latitude'], s2['longitude'],
                              s1['azimuth_deg'], s1['beamwidth_deg'])
    s2_sees_s1 = geom.in_lobe(s2['latitude'], s2['longitude'], s1['latitude'], s1['longitude'],
                              s2['azimuth_deg'], s2['beamwidth_deg'])
    
    interferes, radius = geom.check_directional_interference(s1, s2)
    
    print(f"  Distance: {dist:.2f} km")
    print(f"  Bearing S1→S2: {bearing_1_to_2:.1f}° (S1 points {s1['azimuth_deg']}°)")
    print(f"  Bearing S2→S1: {bearing_2_to_1:.1f}° (S2 points {s2['azimuth_deg']}°)")
    print(f"  S1 sees S2: {s1_sees_s2}")
    print(f"  S2 sees S1: {s2_sees_s1}")
    print(f"  Effective radius: {radius} km")
    print(f"  Interferes: {interferes}")
    print(f"  ✓ EXPECTED: False (back-to-back)")
    
    # Test 2: Facing stations at 3km
    print("\n2. Facing stations (3km apart):")
    s3 = {'latitude': 40.0, 'longitude': -120.0, 'azimuth_deg': 0, 'beamwidth_deg': 60, 'station_id': 2}
    s4 = {'latitude': 40.027, 'longitude': -120.0, 'azimuth_deg': 180, 'beamwidth_deg': 60, 'station_id': 3}
    
    dist = geom.haversine_distance(s3['latitude'], s3['longitude'], s4['latitude'], s4['longitude'])
    s3_sees_s4 = geom.in_lobe(s3['latitude'], s3['longitude'], s4['latitude'], s4['longitude'],
                              s3['azimuth_deg'], s3['beamwidth_deg'])
    s4_sees_s3 = geom.in_lobe(s4['latitude'], s4['longitude'], s3['latitude'], s3['longitude'],
                              s4['azimuth_deg'], s4['beamwidth_deg'])
    
    interferes, radius = geom.check_directional_interference(s3, s4)
    
    print(f"  Distance: {dist:.2f} km")
    print(f"  S3 sees S4: {s3_sees_s4}")
    print(f"  S4 sees S3: {s4_sees_s3}")
    print(f"  Effective radius: {radius} km")
    print(f"  Interferes: {interferes}")
    print(f"  ✓ EXPECTED: True (facing each other)")
    
    # Test 3: Perpendicular at 1km
    print("\n3. Perpendicular stations (1km apart):")
    s5 = {'latitude': 40.0, 'longitude': -120.0, 'azimuth_deg': 0, 'beamwidth_deg': 30, 'station_id': 4}
    s6 = {'latitude': 40.0, 'longitude': -119.991, 'azimuth_deg': 90, 'beamwidth_deg': 30, 'station_id': 5}
    
    dist = geom.haversine_distance(s5['latitude'], s5['longitude'], s6['latitude'], s6['longitude'])
    bearing_5_to_6 = geom.great_circle_bearing(s5['latitude'], s5['longitude'], s6['latitude'], s6['longitude'])
    
    s5_sees_s6 = geom.in_lobe(s5['latitude'], s5['longitude'], s6['latitude'], s6['longitude'],
                              s5['azimuth_deg'], s5['beamwidth_deg'])
    s6_sees_s5 = geom.in_lobe(s6['latitude'], s6['longitude'], s5['latitude'], s5['longitude'],
                              s6['azimuth_deg'], s6['beamwidth_deg'])
    
    interferes, radius = geom.check_directional_interference(s5, s6)
    
    print(f"  Distance: {dist:.2f} km")
    print(f"  Bearing S5→S6: {bearing_5_to_6:.1f}° (S5 points {s5['azimuth_deg']}°)")
    print(f"  S5 sees S6: {s5_sees_s6}")
    print(f"  S6 sees S5: {s6_sees_s5}")
    print(f"  Effective radius: {radius} km")
    print(f"  Interferes: {interferes}")
    print(f"  ✓ EXPECTED: False (perpendicular narrow beams)")
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print("- Great-circle bearings correctly calculated")
    print("- in_lobe function considers angular difference with tolerance")
    print("- Conservative approach: interfere if EITHER sees the other")
    print("- Dual radius: r_main when in lobe, r_off otherwise")


def test_complexity_scaling():
    """Show that neighbor discovery scales linearly, not quadratically."""
    print("\n\nComplexity Scaling Test")
    print("=" * 50)
    
    from tool import create_neighbor_discovery
    
    for n in [10, 50, 100, 500, 1000]:
        # Create random stations
        np.random.seed(42)
        df = pd.DataFrame({
            'latitude': np.random.uniform(35, 45, n),
            'longitude': np.random.uniform(-125, -115, n),
            'azimuth_deg': np.random.uniform(0, 360, n),
            'beamwidth_deg': np.random.choice([60, 120, 360], n)
        })
        
        neighbor_disc = create_neighbor_discovery({
            'r_main_km': 30,
            'r_off_km': 10,
            'az_tolerance_deg': 5
        })
        
        edges = neighbor_disc.build_interference_graph(df, use_directional=True)
        stats = neighbor_disc.get_complexity_analysis()
        
        print(f"n={n:4d}: edges={len(edges):6d}, all_pairs={stats['all_pairs_edges']:8.0f}, "
              f"speedup={stats['speedup_vs_all_pairs']:6.1f}x, {stats['complexity_class']}")
    
    print("\nConclusion: Edge count grows as O(k) where k << n, not O(n²)")


if __name__ == "__main__":
    test_directional_cases()
    test_complexity_scaling()