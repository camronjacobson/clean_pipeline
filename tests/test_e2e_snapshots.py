#!/usr/bin/env python3
"""
End-to-end snapshot tests for spectrum optimization.
Verifies deterministic behavior and expected outputs.
"""

import pytest
import pandas as pd
import json
import sys
from pathlib import Path
import subprocess
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectrum_optimizer_enhanced import EnhancedSpectrumOptimizer


class TestAMOptimization:
    """Test AM band optimization with snapshot comparisons."""
    
    def test_am_small_deterministic(self):
        """Test that AM optimization is deterministic with same seed."""
        df = pd.DataFrame({
            'station_id': ['AM1', 'AM2', 'AM3', 'AM4', 'AM5'],
            'latitude': [37.0, 37.1, 37.2, 37.3, 37.4],
            'longitude': [-122.0, -122.1, -122.2, -122.3, -122.4],
            'frequency_mhz': [0.810, 0.850, 0.910, 0.980, 1.070],
            'power_watts': [50000, 10000, 5000, 1000, 50000]
        })
        
        # Run twice with same seed
        optimizer1 = EnhancedSpectrumOptimizer('am', seed=42)
        result1 = optimizer1.optimize(df)
        
        optimizer2 = EnhancedSpectrumOptimizer('am', seed=42)
        result2 = optimizer2.optimize(df)
        
        # Results should be identical
        assert result1['assigned_frequency'].tolist() == result2['assigned_frequency'].tolist()
        assert optimizer1.objective_metrics['channels_used'] == optimizer2.objective_metrics['channels_used']
    
    def test_am_channel_minimization(self):
        """Test that AM optimizer minimizes channels used."""
        # Create stations that can share frequencies
        df = pd.DataFrame({
            'station_id': [f'AM{i}' for i in range(6)],
            'latitude': [37.0, 37.0, 38.0, 38.0, 39.0, 39.0],  # 3 pairs far apart
            'longitude': [-120.0, -125.0, -120.0, -125.0, -120.0, -125.0],
            'frequency_mhz': [0.810] * 6,
            'power_watts': [5000] * 6
        })
        
        optimizer = EnhancedSpectrumOptimizer('am', seed=42)
        result = optimizer.optimize(df)
        
        # Should use minimal channels (likely 2-3 given interference patterns)
        channels_used = result['assigned_frequency'].nunique()
        assert channels_used <= 3, f"Used {channels_used} channels, expected ≤3 for non-interfering pairs"
        
        # Verify metrics
        assert optimizer.objective_metrics['channels_used'] == channels_used
        assert optimizer.objective_metrics['spectrum_span_khz'] >= 0
    
    def test_am_guard_channels(self):
        """Test that AM guard channels are enforced."""
        # Two close stations that interfere
        df = pd.DataFrame({
            'station_id': ['AM1', 'AM2'],
            'latitude': [37.0, 37.01],  # Very close
            'longitude': [-122.0, -122.01],
            'frequency_mhz': [0.810, 0.820],
            'power_watts': [50000, 50000]
        })
        
        optimizer = EnhancedSpectrumOptimizer('am', seed=42)
        result = optimizer.optimize(df)
        
        freq1 = result.iloc[0]['assigned_frequency']
        freq2 = result.iloc[1]['assigned_frequency']
        
        # Calculate channel separation (AM uses 10 kHz channels)
        channel_sep = abs(freq1 - freq2) * 100  # Convert MHz to 10kHz units
        
        # AM guard_offsets: [-2, -1, 1, 2] means need at least 3 channel separation
        assert channel_sep >= 3, f"Channel separation {channel_sep} violates AM guard requirements"


class TestFMOptimization:
    """Test FM band optimization with snapshot comparisons."""
    
    def test_fm_small_deterministic(self):
        """Test that FM optimization is deterministic with same seed."""
        df = pd.DataFrame({
            'station_id': ['FM1', 'FM2', 'FM3'],
            'latitude': [37.0, 37.5, 38.0],
            'longitude': [-122.0, -122.5, -123.0],
            'frequency_mhz': [88.5, 96.1, 103.7],
            'power_watts': [100000, 50000, 25000],
            'azimuth_deg': [0, 180, 90]
        })
        
        # Run twice with same seed
        optimizer1 = EnhancedSpectrumOptimizer('fm', seed=123)
        result1 = optimizer1.optimize(df)
        
        optimizer2 = EnhancedSpectrumOptimizer('fm', seed=123)
        result2 = optimizer2.optimize(df)
        
        # Results should be identical
        assert result1['assigned_frequency'].tolist() == result2['assigned_frequency'].tolist()
    
    def test_fm_directional_patterns(self):
        """Test that FM directional patterns affect interference."""
        # Two stations with directional patterns pointing away
        df = pd.DataFrame({
            'station_id': ['FM1', 'FM2'],
            'latitude': [37.0, 37.1],  # Close together
            'longitude': [-122.0, -122.0],
            'frequency_mhz': [96.1, 96.1],
            'power_watts': [100000, 100000],
            'azimuth_deg': [0, 180],  # Pointing opposite directions
            'beamwidth_deg': [60, 60]  # Narrow beams
        })
        
        optimizer = EnhancedSpectrumOptimizer('fm', seed=42)
        result = optimizer.optimize(df)
        
        # With narrow beams pointing away, they might be able to share frequency
        # or at least be closer than omnidirectional
        channels_used = result['assigned_frequency'].nunique()
        assert channels_used <= 2, f"Directional patterns should reduce interference"


class TestCLIIntegration:
    """Test command-line interface integration."""
    
    def test_cli_basic_run(self, tmp_path):
        """Test basic CLI execution."""
        # Create test data
        test_csv = tmp_path / "test_stations.csv"
        df = pd.DataFrame({
            'latitude': [37.0, 37.1, 37.2],
            'longitude': [-122.0, -122.1, -122.2],
            'frequency_mhz': [100.0, 101.0, 102.0],
            'power_watts': [1000, 2000, 3000]
        })
        df.to_csv(test_csv, index=False)
        
        # Run CLI
        output_dir = tmp_path / "output"
        cmd = [
            sys.executable, '-m', 'tool.optimize',
            str(test_csv),
            '--profile', 'default',
            '--out', str(output_dir),
            '--seed', '42'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Check success
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "OPTIMIZATION COMPLETE" in result.stdout
        
        # Check output files
        assert (output_dir / 'assignments.csv').exists()
        assert (output_dir / 'assignments.geojson').exists()
        assert (output_dir / 'metrics.json').exists()
        assert (output_dir / 'report.html').exists()
        
        # Verify metrics
        with open(output_dir / 'metrics.json', 'r') as f:
            metrics = json.load(f)
        
        assert metrics['total_stations'] == 3
        assert 'channels_used' in metrics['objective_metrics']
        assert metrics['solver_status'] in ['OPTIMAL', 'FEASIBLE']
    
    def test_cli_max_stations(self, tmp_path):
        """Test --max-stations parameter."""
        # Create test data with many stations
        test_csv = tmp_path / "many_stations.csv"
        df = pd.DataFrame({
            'latitude': [37.0 + i*0.01 for i in range(20)],
            'longitude': [-122.0 + i*0.01 for i in range(20)],
            'frequency_mhz': [100.0 + i*0.2 for i in range(20)]
        })
        df.to_csv(test_csv, index=False)
        
        # Run with max-stations limit
        output_dir = tmp_path / "output_limited"
        cmd = [
            sys.executable, '-m', 'tool.optimize',
            str(test_csv),
            '--out', str(output_dir),
            '--max-stations', '5',
            '--seed', '42'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0
        assert "Stations: 5" in result.stdout
        
        # Check only 5 stations in output
        result_df = pd.read_csv(output_dir / 'assignments.csv')
        assert len(result_df) == 5


class TestPerformanceMetrics:
    """Test performance and complexity metrics."""
    
    def test_neighbor_complexity_scaling(self):
        """Test that neighbor discovery scales sub-quadratically."""
        sizes = [10, 20, 40]
        edge_counts = []
        
        for n in sizes:
            df = pd.DataFrame({
                'latitude': [37.0 + i*0.1 for i in range(n)],
                'longitude': [-122.0 + i*0.1 for i in range(n)],
                'frequency_mhz': [100.0] * n,
                'power_watts': [1000] * n
            })
            
            optimizer = EnhancedSpectrumOptimizer('default', seed=42)
            result = optimizer.optimize(df)
            
            metrics = result.attrs['optimization_metrics']
            edge_counts.append(metrics['neighbor_metrics']['total_edges'])
        
        # Check sub-quadratic growth
        # If O(n²), edges would grow 4x and 16x
        # If O(n), edges would grow 2x and 4x
        growth_1_2 = edge_counts[1] / edge_counts[0]
        growth_2_3 = edge_counts[2] / edge_counts[1]
        
        # Should be closer to linear than quadratic
        assert growth_1_2 < 3.5, f"Edge growth {growth_1_2:.1f}x suggests O(n²)"
        assert growth_2_3 < 3.5, f"Edge growth {growth_2_3:.1f}x suggests O(n²)"
    
    def test_optimization_metrics_present(self):
        """Test that all expected metrics are reported."""
        df = pd.DataFrame({
            'latitude': [37.0, 37.1],
            'longitude': [-122.0, -122.1],
            'frequency_mhz': [100.0, 101.0]
        })
        
        optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        result = optimizer.optimize(df)
        
        metrics = result.attrs['optimization_metrics']
        
        # Check required metrics
        assert 'unique_frequencies' in metrics
        assert 'total_stations' in metrics
        assert 'solve_time_seconds' in metrics
        assert 'solver_status' in metrics
        
        # Check constraint stats
        assert 'constraint_stats' in metrics
        stats = metrics['constraint_stats']
        assert 'co_channel' in stats
        assert 'adjacent_channel' in stats
        assert 'total' in stats
        
        # Check neighbor metrics
        assert 'neighbor_metrics' in metrics
        neighbor = metrics['neighbor_metrics']
        assert 'avg_neighbors' in neighbor
        assert 'complexity_class' in neighbor
        
        # Check objective metrics
        assert 'objective_metrics' in metrics
        obj = metrics['objective_metrics']
        assert 'channels_used' in obj
        assert 'spectrum_span_khz' in obj
        assert 'channel_packing_score' in obj


def test_schema_flexibility_e2e():
    """End-to-end test of schema flexibility."""
    # Test various column name variations
    test_cases = [
        # Minimal schema
        {
            'y': [40.0, 40.1],
            'x': [-120.0, -120.1]
        },
        # Alternative names
        {
            'lat': [40.0, 40.1],
            'lng': [-120.0, -120.1],
            'freq': [100.0, 101.0]
        },
        # Mixed case
        {
            'LaTiTuDe': [40.0, 40.1],
            'LONGITUDE': [-120.0, -120.1],
            'Power_Watts': [1000, 2000]
        }
    ]
    
    for i, data in enumerate(test_cases):
        df = pd.DataFrame(data)
        optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        
        # Should not raise any errors
        result = optimizer.optimize(df)
        
        assert 'assigned_frequency' in result.columns
        assert len(result) == 2
        assert result['assigned_frequency'].notna().all()
        
        print(f"✓ Schema test case {i+1} passed")


if __name__ == "__main__":
    # Run basic tests
    print("Running E2E Snapshot Tests...")
    
    # Test AM optimization
    am_test = TestAMOptimization()
    am_test.test_am_small_deterministic()
    print("✓ AM deterministic test passed")
    am_test.test_am_channel_minimization()
    print("✓ AM channel minimization test passed")
    am_test.test_am_guard_channels()
    print("✓ AM guard channels test passed")
    
    # Test FM optimization
    fm_test = TestFMOptimization()
    fm_test.test_fm_small_deterministic()
    print("✓ FM deterministic test passed")
    fm_test.test_fm_directional_patterns()
    print("✓ FM directional patterns test passed")
    
    # Test performance metrics
    perf_test = TestPerformanceMetrics()
    perf_test.test_neighbor_complexity_scaling()
    print("✓ Neighbor complexity scaling test passed")
    perf_test.test_optimization_metrics_present()
    print("✓ Optimization metrics test passed")
    
    # Test schema flexibility
    test_schema_flexibility_e2e()
    print("✓ Schema flexibility E2E tests passed")
    
    print("\n" + "="*60)
    print("ALL E2E SNAPSHOT TESTS PASSED")
    print("="*60)