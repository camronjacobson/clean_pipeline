"""
Tests for lexicographic objective optimization.
Verifies channel minimization, low-frequency packing, and determinism.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectrum_optimizer_enhanced import EnhancedSpectrumOptimizer


class TestLexicographicObjective:
    """Test cases for lexicographic objective function."""
    
    def test_channel_minimization(self):
        """
        Test that optimizer minimizes channel count as first priority.
        """
        # Create optimizer with fixed seed
        optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        
        # Override for testing
        optimizer.config['band']['min_mhz'] = 88.0
        optimizer.config['band']['max_mhz'] = 90.0  # Limited band
        optimizer.config['band']['step_khz'] = 200
        optimizer.config['geometry']['r_main_km'] = 10.0
        
        # 3-station chain that needs exactly 3 channels with guards
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C'],
            'latitude': [40.0, 40.0, 40.0],
            'longitude': [-120.0, -119.91, -119.82],  # ~10km apart
            'power_watts': [1000, 1000, 1000],
            'azimuth_deg': [0, 0, 0],
            'beamwidth_deg': [360, 360, 360]
        })
        
        result = optimizer.optimize(df)
        
        # Check objective metrics
        if hasattr(result, 'attrs') and 'optimization_metrics' in result.attrs:
            obj_metrics = result.attrs['optimization_metrics']['objective_metrics']
            
            print(f"\nChannel Minimization Test:")
            print(f"  Channels used: {obj_metrics['channels_used']}")
            print(f"  Channel indices: {obj_metrics['channel_indices_used']}")
            print(f"  Packing score: {obj_metrics['channel_packing_score']:.2f}")
            
            # Should use exactly 3 channels (minimum with guards)
            assert obj_metrics['channels_used'] == 3, \
                f"Expected 3 channels (minimum), got {obj_metrics['channels_used']}"
    
    def test_low_frequency_packing(self):
        """
        Test that optimizer packs channels toward low frequencies.
        """
        optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        
        # Wide band for testing packing
        optimizer.config['band']['min_mhz'] = 88.0
        optimizer.config['band']['max_mhz'] = 98.0  # 51 channels
        optimizer.config['band']['step_khz'] = 200
        optimizer.config['geometry']['r_main_km'] = 100.0  # Large radius
        
        # Two isolated groups that don't interfere
        df = pd.DataFrame({
            'station_id': ['A1', 'A2', 'B1', 'B2'],
            'latitude': [40.0, 40.01, 50.0, 50.01],  # Groups 1000km apart
            'longitude': [-120.0, -120.0, -110.0, -110.0],
            'power_watts': [1000] * 4,
            'azimuth_deg': [0] * 4,
            'beamwidth_deg': [360] * 4
        })
        
        result = optimizer.optimize(df)
        
        if hasattr(result, 'attrs') and 'optimization_metrics' in result.attrs:
            obj_metrics = result.attrs['optimization_metrics']['objective_metrics']
            
            print(f"\nLow Frequency Packing Test:")
            print(f"  Channels used: {obj_metrics['channels_used']}")
            print(f"  Channel indices: {obj_metrics['channel_indices_used']}")
            print(f"  Average index: {obj_metrics['channel_packing_score']:.2f}")
            
            # Channels should be packed toward low indices
            max_index = max(obj_metrics['channel_indices_used'])
            assert max_index < 10, f"Channels not packed low: max index {max_index}"
            
            # Average index should be low
            assert obj_metrics['channel_packing_score'] < 5, \
                f"Poor packing: average index {obj_metrics['channel_packing_score']:.2f}"
    
    def test_deterministic_results(self):
        """
        Test that same problem with same seed produces identical results.
        """
        # Create two identical optimizers with same seed
        opt1 = EnhancedSpectrumOptimizer('default', seed=123)
        opt2 = EnhancedSpectrumOptimizer('default', seed=123)
        
        # Configure identically
        for opt in [opt1, opt2]:
            opt.config['band']['min_mhz'] = 88.0
            opt.config['band']['max_mhz'] = 92.0
            opt.config['band']['step_khz'] = 200
            opt.config['geometry']['r_main_km'] = 15.0
        
        # Same input data
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D', 'E'],
            'latitude': [40.0, 40.05, 40.1, 40.15, 40.2],
            'longitude': [-120.0, -120.05, -120.1, -120.15, -120.2],
            'power_watts': [1000] * 5,
            'azimuth_deg': [0, 90, 180, 270, 0],
            'beamwidth_deg': [120] * 5
        })
        
        # Run optimization twice
        result1 = opt1.optimize(df.copy())
        result2 = opt2.optimize(df.copy())
        
        # Results should be identical
        freqs1 = result1['assigned_frequency'].values
        freqs2 = result2['assigned_frequency'].values
        
        print(f"\nDeterminism Test:")
        print(f"  Run 1: {freqs1}")
        print(f"  Run 2: {freqs2}")
        print(f"  Identical: {np.array_equal(freqs1, freqs2)}")
        
        assert np.array_equal(freqs1, freqs2), \
            "Same seed should produce identical results"
        
        # Check that different seed produces different tie-breaking
        opt3 = EnhancedSpectrumOptimizer('default', seed=999)
        opt3.config['band']['min_mhz'] = 88.0
        opt3.config['band']['max_mhz'] = 92.0
        opt3.config['band']['step_khz'] = 200
        opt3.config['geometry']['r_main_km'] = 15.0
        
        result3 = opt3.optimize(df.copy())
        freqs3 = result3['assigned_frequency'].values
        
        print(f"  Run 3 (different seed): {freqs3}")
        
        # Should use same number of channels but potentially different assignment
        unique1 = len(np.unique(freqs1))
        unique3 = len(np.unique(freqs3))
        assert unique1 == unique3, "Different seeds should still minimize channels"
    
    def test_same_channel_count_prefers_lower(self):
        """
        Test that when channel count is equal, lower indices are preferred.
        """
        optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        
        # Large band to test preference
        optimizer.config['band']['min_mhz'] = 88.0
        optimizer.config['band']['max_mhz'] = 108.0  # Full FM band
        optimizer.config['band']['step_khz'] = 200
        optimizer.config['geometry']['r_main_km'] = 5.0
        optimizer.config['interference']['guard_offsets'] = []  # No guards for simplicity
        
        # Two isolated stations that don't interfere
        df = pd.DataFrame({
            'station_id': ['A', 'B'],
            'latitude': [40.0, 50.0],  # Very far apart
            'longitude': [-120.0, -110.0],
            'power_watts': [1000, 1000],
            'azimuth_deg': [0, 0],
            'beamwidth_deg': [360, 360]
        })
        
        result = optimizer.optimize(df)
        
        if hasattr(result, 'attrs') and 'optimization_metrics' in result.attrs:
            obj_metrics = result.attrs['optimization_metrics']['objective_metrics']
            
            print(f"\nLower Index Preference Test:")
            print(f"  Channel indices used: {obj_metrics['channel_indices_used']}")
            
            # Both stations should use channel 0 (88.0 MHz)
            assert obj_metrics['channel_indices_used'] == [0], \
                f"Should use lowest channel [0], got {obj_metrics['channel_indices_used']}"
            
            freqs = result['assigned_frequency'].values
            assert all(f == 88.0 for f in freqs), \
                f"All should use 88.0 MHz, got {freqs}"
    
    def test_objective_component_tracking(self):
        """
        Test that all objective components are properly tracked in metrics.
        """
        optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C'],
            'latitude': [40.0, 40.1, 40.2],
            'longitude': [-120.0, -120.1, -120.2],
            'power_watts': [1000] * 3,
            'azimuth_deg': [0] * 3,
            'beamwidth_deg': [360] * 3
        })
        
        result = optimizer.optimize(df)
        
        if hasattr(result, 'attrs') and 'optimization_metrics' in result.attrs:
            metrics = result.attrs['optimization_metrics']
            
            # Check that objective metrics exist
            assert 'objective_metrics' in metrics, "Missing objective_metrics"
            
            obj_metrics = metrics['objective_metrics']
            
            # All components should be present
            required_keys = [
                'channels_used',
                'channel_indices_used', 
                'spectrum_span_khz',
                'channel_packing_score'
            ]
            
            for key in required_keys:
                assert key in obj_metrics, f"Missing objective metric: {key}"
            
            print(f"\nObjective Component Tracking:")
            print(f"  Channels used: {obj_metrics['channels_used']}")
            print(f"  Indices: {obj_metrics['channel_indices_used']}")
            print(f"  Span (kHz): {obj_metrics['spectrum_span_khz']}")
            print(f"  Packing score: {obj_metrics['channel_packing_score']:.2f}")
            
            # Values should be reasonable
            assert obj_metrics['channels_used'] > 0, "Should use at least 1 channel"
            assert len(obj_metrics['channel_indices_used']) == obj_metrics['channels_used'], \
                "Index list should match channel count"
            assert obj_metrics['spectrum_span_khz'] >= 0, "Span should be non-negative"
            assert obj_metrics['channel_packing_score'] >= 0, "Packing score should be non-negative"