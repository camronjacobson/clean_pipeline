#!/usr/bin/env python3
"""
Test script for shapefile overlay functionality.
Tests visualization with and without shapefiles, measuring performance impact.
"""

import sys
import time
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'tool')

from visualizer_enhanced_v2 import EnhancedVisualizer, create_visualization


def test_with_shapefiles():
    """Test map generation with shapefile overlays."""
    print("=" * 60)
    print("TESTING SHAPEFILE OVERLAY FUNCTIONALITY")
    print("=" * 60)
    
    # Load test data
    if not Path('runs/am_test/assignments.csv').exists():
        print("Error: Run optimization first to generate test data")
        return
    
    df = pd.read_csv('runs/am_test/assignments.csv')
    metrics = json.load(open('runs/am_test/metrics.json'))
    
    print(f"\nTest data: {len(df)} stations")
    
    # Test 1: Map WITHOUT shapefiles
    print("\n1. Testing WITHOUT shapefiles...")
    start = time.time()
    
    viz_plain = EnhancedVisualizer(df, metrics, None)
    viz_plain.create_interactive_map('map_plain.html')
    
    time_plain = time.time() - start
    size_plain = Path('map_plain.html').stat().st_size / 1024
    
    print(f"   ✓ Plain map created")
    print(f"   - File size: {size_plain:.1f} KB")
    print(f"   - Generation time: {time_plain:.2f} seconds")
    
    # Test 2: Map WITH shapefiles
    print("\n2. Testing WITH shapefiles...")
    
    # List available shapefiles
    shapefiles = []
    shapefile_dir = Path('shapefiles')
    if shapefile_dir.exists():
        for ext in ['*.geojson', '*.shp']:
            shapefiles.extend([str(f) for f in shapefile_dir.glob(ext)])
    
    # Filter out invalid test file
    shapefiles = [f for f in shapefiles if 'invalid' not in f]
    
    print(f"   Found {len(shapefiles)} shapefiles:")
    for sf in shapefiles:
        print(f"     - {Path(sf).name}")
    
    start = time.time()
    
    viz_overlays = EnhancedVisualizer(df, metrics, shapefiles)
    viz_overlays.create_interactive_map('map_with_overlays.html')
    
    time_overlays = time.time() - start
    size_overlays = Path('map_with_overlays.html').stat().st_size / 1024
    
    print(f"   ✓ Map with overlays created")
    print(f"   - File size: {size_overlays:.1f} KB")
    print(f"   - Generation time: {time_overlays:.2f} seconds")
    print(f"   - Shapefile layers added: {len(viz_overlays.shapefiles)}")
    
    # Compare performance
    print("\n3. Performance Comparison:")
    print(f"   - Size overhead: {size_overlays - size_plain:.1f} KB ({(size_overlays/size_plain - 1)*100:.1f}% increase)")
    print(f"   - Time overhead: {time_overlays - time_plain:.2f} seconds ({(time_overlays/time_plain - 1)*100:.1f}% increase)")
    
    # Generate report with regional statistics
    print("\n4. Regional Statistics:")
    report = viz_overlays.generate_frequency_report()
    
    if report.get('regional_distribution'):
        for layer, regions in report['regional_distribution'].items():
            print(f"\n   {layer}:")
            for region, count in sorted(regions.items(), key=lambda x: x[1], reverse=True):
                print(f"     {region}: {count} stations")
    else:
        print("   No stations found within shapefile regions")
    
    # Test 3: Test with invalid shapefile (error handling)
    print("\n5. Testing error handling with invalid shapefile...")
    
    invalid_shapefiles = ['shapefiles/invalid_test.geojson', 'nonexistent.shp']
    
    try:
        viz_invalid = EnhancedVisualizer(df, metrics, invalid_shapefiles)
        viz_invalid.create_interactive_map('map_invalid_test.html')
        print("   ✓ Gracefully handled invalid shapefiles")
        print(f"   - Valid shapefiles loaded: {len(viz_invalid.shapefiles)}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Test choropleth with different datasets
    print("\n6. Testing choropleth with FM data...")
    
    if Path('runs/fm_test/assignments.csv').exists():
        df_fm = pd.read_csv('runs/fm_test/assignments.csv')
        
        viz_fm = EnhancedVisualizer(df_fm, None, shapefiles)
        viz_fm.create_interactive_map('map_fm_with_overlays.html')
        
        size_fm = Path('map_fm_with_overlays.html').stat().st_size / 1024
        print(f"   ✓ FM map with overlays created")
        print(f"   - File size: {size_fm:.1f} KB")
        print(f"   - Stations: {len(df_fm)}")
        
        # Check regional distribution
        report_fm = viz_fm.generate_frequency_report()
        if report_fm.get('regional_distribution'):
            total_in_regions = sum(
                sum(regions.values()) 
                for regions in report_fm['regional_distribution'].values()
            )
            print(f"   - Stations in shapefile regions: {total_in_regions}/{len(df_fm)}")


def test_layer_control():
    """Test layer control functionality."""
    print("\n" + "=" * 60)
    print("TESTING LAYER CONTROL")
    print("=" * 60)
    
    # Create a map with multiple layers
    print("\nCreating map with multiple toggleable layers...")
    
    # Generate test data with different densities
    import numpy as np
    np.random.seed(42)
    
    # California stations
    df_ca = pd.DataFrame({
        'station_id': [f'CA_{i:03d}' for i in range(50)],
        'latitude': np.random.uniform(32.5, 42.0, 50),
        'longitude': np.random.uniform(-124.5, -114.0, 50),
        'assigned_frequency': np.random.choice([88.1, 92.3, 96.5, 100.7], 50),
        'power_watts': np.random.choice([1000, 5000, 10000], 50)
    })
    
    # Load all shapefiles
    shapefiles = [
        'shapefiles/california_regions.geojson',
        'shapefiles/us_states_west.geojson',
        'shapefiles/bea_regions.geojson'
    ]
    
    viz = EnhancedVisualizer(df_ca, None, shapefiles)
    viz.create_interactive_map('map_layer_control.html')
    
    print(f"✓ Created map with {len(shapefiles)} toggleable layers")
    print("✓ Open 'map_layer_control.html' to test:")
    print("  - Toggle each shapefile layer on/off")
    print("  - View station density choropleth")
    print("  - Check tooltips on regions")
    
    # Generate statistics
    report = viz.generate_frequency_report()
    print(f"\n✓ Map includes:")
    print(f"  - {report['total_stations']} stations")
    print(f"  - {report['unique_frequencies']} unique frequencies")
    print(f"  - {len(report.get('shapefile_layers', []))} shapefile layers")


def test_performance_scaling():
    """Test performance with increasing numbers of shapefiles and stations."""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE SCALING")
    print("=" * 60)
    
    import numpy as np
    np.random.seed(42)
    
    # Test with increasing station counts
    station_counts = [10, 50, 100, 500]
    shapefiles = ['shapefiles/california_regions.geojson', 'shapefiles/us_states_west.geojson']
    
    print("\nPerformance with increasing stations:")
    print("Stations | No Overlays | With Overlays | Overhead")
    print("-" * 50)
    
    for n in station_counts:
        # Generate synthetic data
        df = pd.DataFrame({
            'station_id': [f'S_{i:04d}' for i in range(n)],
            'latitude': np.random.uniform(32.5, 42.0, n),
            'longitude': np.random.uniform(-124.5, -114.0, n),
            'assigned_frequency': np.random.choice([88.1, 92.3, 96.5, 100.7, 104.9], n)
        })
        
        # Without overlays
        start = time.time()
        viz_plain = EnhancedVisualizer(df, None, None)
        viz_plain.create_interactive_map(f'perf_test_{n}_plain.html')
        time_plain = time.time() - start
        
        # With overlays
        start = time.time()
        viz_overlay = EnhancedVisualizer(df, None, shapefiles)
        viz_overlay.create_interactive_map(f'perf_test_{n}_overlay.html')
        time_overlay = time.time() - start
        
        overhead_pct = ((time_overlay / time_plain) - 1) * 100
        
        print(f"{n:8d} | {time_plain:10.3f}s | {time_overlay:13.3f}s | {overhead_pct:+7.1f}%")
    
    # Clean up test files
    for n in station_counts:
        Path(f'perf_test_{n}_plain.html').unlink(missing_ok=True)
        Path(f'perf_test_{n}_overlay.html').unlink(missing_ok=True)


if __name__ == "__main__":
    print("SHAPEFILE OVERLAY SYSTEM TEST SUITE")
    print("=" * 60)
    
    # Run all tests
    test_with_shapefiles()
    test_layer_control()
    test_performance_scaling()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("\nGenerated maps:")
    print("  - map_plain.html (no overlays)")
    print("  - map_with_overlays.html (with all shapefiles)")
    print("  - map_layer_control.html (toggleable layers)")
    print("  - map_fm_with_overlays.html (FM data with overlays)")
    print("\nOpen these files in a browser to explore:")
    print("  - Toggle shapefile layers on/off")
    print("  - View station density by region")
    print("  - Check regional boundaries and labels")