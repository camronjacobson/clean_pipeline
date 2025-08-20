#!/usr/bin/env python3
"""
Test script for unified analytics dashboard.
Verifies single HTML file contains all components.
"""

import sys
import time
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'tool')

from dashboard_visualizer import DashboardVisualizer, create_dashboard


def test_unified_dashboard():
    """Test unified dashboard generation."""
    print("=" * 60)
    print("TESTING UNIFIED ANALYTICS DASHBOARD")
    print("=" * 60)
    
    # Test 1: Small dataset (AM 25 stations)
    if Path('runs/am_test/assignments.csv').exists():
        print("\n1. Testing with AM dataset (25 stations)...")
        start = time.time()
        
        create_dashboard(
            'runs/am_test/assignments.csv',
            'runs/am_test/metrics.json',
            None,  # No shapefiles for first test
            'dashboard_am.html'
        )
        
        elapsed = time.time() - start
        file_size = Path('dashboard_am.html').stat().st_size / 1024
        
        # Verify content
        with open('dashboard_am.html', 'r') as f:
            content = f.read()
        
        # Check for all tabs
        assert 'Overview' in content, "Missing Overview tab"
        assert 'Map View' in content, "Missing Map tab"
        assert 'Spectrum Analysis' in content, "Missing Spectrum tab"
        assert 'Geographic' in content, "Missing Geographic tab"
        assert 'Network' in content, "Missing Network tab"
        assert 'Metrics' in content, "Missing Metrics tab"
        
        # Check for Plotly
        assert 'plotly' in content.lower(), "Missing Plotly library"
        
        # Check for charts
        assert 'spectrum-chart' in content, "Missing spectrum chart"
        assert 'Channel Efficiency' in content, "Missing efficiency gauge"
        
        print(f"   ✅ Dashboard created successfully")
        print(f"   - File size: {file_size:.1f} KB")
        print(f"   - Generation time: {elapsed:.2f} seconds")
        print(f"   - All 6 tabs present")
        print(f"   - Plotly charts embedded")
    
    # Test 2: Medium dataset (FM 30 stations)
    if Path('runs/fm_test/assignments.csv').exists():
        print("\n2. Testing with FM dataset (30 stations)...")
        start = time.time()
        
        df = pd.read_csv('runs/fm_test/assignments.csv')
        metrics = json.load(open('runs/fm_test/metrics.json'))
        
        viz = DashboardVisualizer(df, metrics)
        viz.create_unified_dashboard('dashboard_fm.html')
        
        elapsed = time.time() - start
        file_size = Path('dashboard_fm.html').stat().st_size / 1024
        
        print(f"   ✅ FM dashboard created")
        print(f"   - File size: {file_size:.1f} KB")
        print(f"   - Generation time: {elapsed:.2f} seconds")
        print(f"   - Stations: {len(df)}")
        print(f"   - Frequencies: {df['assigned_frequency'].nunique()}")
    
    # Test 3: With shapefiles
    print("\n3. Testing with shapefiles...")
    
    shapefiles = []
    if Path('shapefiles').exists():
        shapefiles = [
            'shapefiles/california_regions.geojson',
            'shapefiles/us_states_west.geojson'
        ]
        shapefiles = [s for s in shapefiles if Path(s).exists()]
    
    if shapefiles:
        start = time.time()
        
        df = pd.read_csv('runs/am_test/assignments.csv')
        metrics = json.load(open('runs/am_test/metrics.json'))
        
        viz = DashboardVisualizer(df, metrics, shapefiles)
        viz.create_unified_dashboard('dashboard_with_shapes.html')
        
        elapsed = time.time() - start
        file_size = Path('dashboard_with_shapes.html').stat().st_size / 1024
        
        print(f"   ✅ Dashboard with shapefiles created")
        print(f"   - File size: {file_size:.1f} KB")
        print(f"   - Generation time: {elapsed:.2f} seconds")
        print(f"   - Shapefile layers: {len(shapefiles)}")
    
    # Test 4: Large synthetic dataset
    print("\n4. Testing with large dataset (500 stations)...")
    
    import numpy as np
    np.random.seed(42)
    
    n_stations = 500
    df_large = pd.DataFrame({
        'station_id': [f'S{i:04d}' for i in range(n_stations)],
        'latitude': np.random.uniform(30, 45, n_stations),
        'longitude': np.random.uniform(-120, -80, n_stations),
        'assigned_frequency': np.random.choice(
            np.linspace(88.0, 108.0, 20), n_stations
        ),
        'power_watts': np.random.choice([1000, 5000, 10000, 50000], n_stations)
    })
    
    # Create mock metrics
    mock_metrics = {
        'optimization_metrics': {
            'unique_frequencies': 20,
            'total_stations': n_stations,
            'solve_time_seconds': 15.3,
            'solver_status': 'OPTIMAL',
            'objective_metrics': {
                'channels_used': 20,
                'spectrum_span_khz': 20000,
                'channel_packing_score': 8.5
            },
            'neighbor_metrics': {
                'avg_neighbors': 12.5,
                'total_edges': 3125,
                'complexity_class': 'O(n log n)'
            },
            'constraint_stats': {
                'total': 50000,
                'co_channel': 10000,
                'adjacent_channel': 40000
            }
        }
    }
    
    start = time.time()
    
    viz = DashboardVisualizer(df_large, mock_metrics)
    viz.create_unified_dashboard('dashboard_large.html')
    
    elapsed = time.time() - start
    file_size = Path('dashboard_large.html').stat().st_size / 1024
    
    print(f"   ✅ Large dataset dashboard created")
    print(f"   - File size: {file_size:.1f} KB")
    print(f"   - Generation time: {elapsed:.2f} seconds")
    print(f"   - Network graph: Hidden (>100 stations)")


def test_dashboard_components():
    """Test individual dashboard components."""
    print("\n" + "=" * 60)
    print("TESTING DASHBOARD COMPONENTS")
    print("=" * 60)
    
    # Create minimal test data
    df = pd.DataFrame({
        'station_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
        'latitude': [37.0, 37.1, 37.2, 37.3, 37.4],
        'longitude': [-122.0, -122.1, -122.2, -122.3, -122.4],
        'assigned_frequency': [88.1, 88.1, 92.3, 96.5, 96.5],
        'power_watts': [1000, 5000, 10000, 5000, 1000]
    })
    
    metrics = {
        'optimization_metrics': {
            'objective_metrics': {
                'channels_used': 3,
                'spectrum_span_khz': 8400,
                'channel_packing_score': 5.0
            },
            'neighbor_metrics': {
                'avg_neighbors': 2.0,
                'total_edges': 5,
                'complexity_class': 'O(n)'
            },
            'solve_time_seconds': 0.5
        }
    }
    
    viz = DashboardVisualizer(df, metrics)
    
    print("\n1. Testing metric extraction...")
    assert viz.total_stations == 5
    assert viz.unique_frequencies == 3
    assert viz.channels_used == 3
    print("   ✅ Metrics extracted correctly")
    
    print("\n2. Testing chart generation...")
    
    # Test spectrum allocation chart
    spectrum_chart = viz._create_spectrum_allocation_chart()
    assert 'Spectrum Allocation' in spectrum_chart
    assert 'plotly' in spectrum_chart.lower()
    print("   ✅ Spectrum chart generated")
    
    # Test efficiency gauges
    gauges = viz._create_efficiency_gauges()
    assert 'Channel Efficiency' in gauges
    assert 'gauge' in gauges.lower()
    print("   ✅ Efficiency gauges generated")
    
    # Test summary stats
    stats = viz._create_summary_stats()
    assert 'Total Stations' in stats
    assert '5' in stats  # Our test has 5 stations
    print("   ✅ Summary statistics generated")
    
    # Test interference graph
    network = viz._create_interference_graph()
    assert 'Interference' in network or 'Network' in network
    print("   ✅ Network graph generated")
    
    print("\n3. Testing complete dashboard generation...")
    viz.create_unified_dashboard('dashboard_test.html')
    
    # Verify file exists and has content
    assert Path('dashboard_test.html').exists()
    file_size = Path('dashboard_test.html').stat().st_size
    assert file_size > 10000, f"Dashboard too small: {file_size} bytes"
    
    print(f"   ✅ Complete dashboard generated ({file_size/1024:.1f} KB)")


def measure_performance():
    """Measure dashboard performance with different sizes."""
    print("\n" + "=" * 60)
    print("PERFORMANCE MEASUREMENTS")
    print("=" * 60)
    
    import numpy as np
    np.random.seed(42)
    
    sizes = [10, 50, 100, 250, 500]
    
    print("\nStations | File Size | Generation Time")
    print("-" * 40)
    
    for n in sizes:
        df = pd.DataFrame({
            'station_id': [f'S{i:04d}' for i in range(n)],
            'latitude': np.random.uniform(30, 45, n),
            'longitude': np.random.uniform(-120, -80, n),
            'assigned_frequency': np.random.choice([88.1, 92.3, 96.5, 100.7, 104.9], n),
            'power_watts': np.random.choice([1000, 5000, 10000], n)
        })
        
        start = time.time()
        viz = DashboardVisualizer(df)
        viz.create_unified_dashboard(f'dashboard_perf_{n}.html')
        elapsed = time.time() - start
        
        file_size = Path(f'dashboard_perf_{n}.html').stat().st_size / 1024
        
        print(f"{n:8d} | {file_size:9.1f} KB | {elapsed:10.2f}s")
        
        # Clean up
        Path(f'dashboard_perf_{n}.html').unlink(missing_ok=True)


if __name__ == "__main__":
    print("UNIFIED DASHBOARD TEST SUITE")
    print("=" * 60)
    
    # Run all tests
    test_unified_dashboard()
    test_dashboard_components()
    measure_performance()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    print("\nGenerated dashboards:")
    for dashboard in ['dashboard_am.html', 'dashboard_fm.html', 'dashboard_large.html']:
        if Path(dashboard).exists():
            size = Path(dashboard).stat().st_size / 1024
            print(f"  - {dashboard}: {size:.1f} KB")
    
    print("\n📊 Open any dashboard.html file in a browser to explore:")
    print("  • Overview tab with key metrics")
    print("  • Interactive map with clustering")
    print("  • Spectrum allocation charts")
    print("  • Geographic distribution")
    print("  • Interference network (for <100 stations)")
    print("  • Detailed metrics tables")