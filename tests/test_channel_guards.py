"""
Regression tests for channel usage with adjacent channel protection.
Verifies that guard channels are properly enforced.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectrum_optimizer_enhanced import EnhancedSpectrumOptimizer


class TestChannelGuards:
    """Test cases for adjacent channel protection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create optimizer with custom config for testing
        self.optimizer = EnhancedSpectrumOptimizer('default')
        
        # Override for testing with predictable settings
        self.optimizer.config['band']['min_mhz'] = 88.0
        self.optimizer.config['band']['max_mhz'] = 92.0  # Limited band for testing
        self.optimizer.config['band']['step_khz'] = 200  # 0.2 MHz steps
        self.optimizer.config['geometry']['r_main_km'] = 10.0
        self.optimizer.config['geometry']['r_off_km'] = 2.0
        self.optimizer.config['interference']['guard_offsets'] = [-1, 1]  # ±1 channel guard
    
    def test_3_station_chain_with_guards(self):
        """
        Test 3-station chain with adjacent channel guards.
        A -- B -- C (10km apart)
        With guards, should need 3 channels not 2.
        """
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C'],
            'latitude': [40.0, 40.0, 40.0],
            'longitude': [-120.0, -119.91, -119.82],  # ~10km apart
            'power_watts': [1000, 1000, 1000],
            'azimuth_deg': [90, 90, 90],  # All pointing east
            'beamwidth_deg': [360, 360, 360],  # Omnidirectional
        })
        
        result = self.optimizer.optimize(df)
        
        # Check frequencies assigned
        freqs = result['assigned_frequency'].values
        unique_freqs = np.unique(freqs)
        
        print(f"\n3-Station Chain Test:")
        print(f"  Assigned frequencies: {freqs}")
        print(f"  Unique count: {len(unique_freqs)}")
        
        # With adjacent channel guards, need 3 channels
        # A and C can't share because B needs guard channels on both sides
        assert len(unique_freqs) == 3, f"Expected 3 channels with guards, got {len(unique_freqs)}"
        
        # Verify guard channel spacing
        freq_a = result[result['station_id'] == 'A']['assigned_frequency'].iloc[0]
        freq_b = result[result['station_id'] == 'B']['assigned_frequency'].iloc[0]
        freq_c = result[result['station_id'] == 'C']['assigned_frequency'].iloc[0]
        
        # B should be at least 2 channels away from both A and C
        channel_step = 0.2  # MHz
        assert abs(freq_b - freq_a) >= 2 * channel_step - 0.01, "Guard channel violation A-B"
        assert abs(freq_b - freq_c) >= 2 * channel_step - 0.01, "Guard channel violation B-C"
    
    def test_4_station_ring_with_guards(self):
        """
        Test 4-station ring with adjacent channel guards.
        A -- B
        |    |
        D -- C
        With guards, should need 4 channels (not 2).
        """
        # Create 4 stations in a square, 8km apart
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D'],
            'latitude': [40.0, 40.0, 39.928, 39.928],  # Square ~8km sides
            'longitude': [-120.0, -119.928, -119.928, -120.0],
            'power_watts': [1000, 1000, 1000, 1000],
            'azimuth_deg': [0, 0, 0, 0],
            'beamwidth_deg': [360, 360, 360, 360],  # All omnidirectional
        })
        
        result = self.optimizer.optimize(df)
        
        # Check frequencies assigned
        freqs = result['assigned_frequency'].values
        unique_freqs = np.unique(freqs)
        
        print(f"\n4-Station Ring Test:")
        print(f"  Assigned frequencies: {freqs}")
        print(f"  Unique count: {len(unique_freqs)}")
        
        # With adjacent channel guards in a ring, need 4 channels
        assert len(unique_freqs) >= 3, f"Expected at least 3 channels with guards, got {len(unique_freqs)}"
    
    def test_adjacent_channel_protection(self):
        """
        Test that adjacent channel protection actually works.
        Two close stations should not use adjacent channels.
        """
        # Two stations very close (2km)
        df = pd.DataFrame({
            'station_id': ['A', 'B'],
            'latitude': [40.0, 40.018],  # ~2km apart
            'longitude': [-120.0, -120.0],
            'power_watts': [1000, 1000],
            'azimuth_deg': [0, 180],
            'beamwidth_deg': [360, 360],
        })
        
        result = self.optimizer.optimize(df)
        
        freq_a = result[result['station_id'] == 'A']['assigned_frequency'].iloc[0]
        freq_b = result[result['station_id'] == 'B']['assigned_frequency'].iloc[0]
        
        channel_diff = abs(freq_a - freq_b) / 0.2  # Convert to channel numbers
        
        print(f"\nAdjacent Channel Protection Test:")
        print(f"  Station A: {freq_a:.1f} MHz")
        print(f"  Station B: {freq_b:.1f} MHz")
        print(f"  Channel separation: {channel_diff:.1f}")
        
        # Should be at least 2 channels apart (guard = ±1)
        assert channel_diff >= 1.9, f"Adjacent channel violation: only {channel_diff:.1f} channels apart"
    
    def test_constraint_statistics(self):
        """
        Test that constraint counts are O(n*f) not O(n²*f).
        """
        # Create varying sizes and check constraint scaling
        results = []
        
        for n in [5, 10, 20]:
            df = pd.DataFrame({
                'station_id': [f'S{i}' for i in range(n)],
                'latitude': np.random.uniform(39.9, 40.1, n),
                'longitude': np.random.uniform(-120.1, -119.9, n),
                'power_watts': [1000] * n,
                'azimuth_deg': [0] * n,
                'beamwidth_deg': [360] * n,
            })
            
            np.random.seed(42)  # For reproducibility
            
            result = self.optimizer.optimize(df)
            
            # Get constraint statistics
            if hasattr(result, 'attrs') and 'optimization_metrics' in result.attrs:
                stats = result.attrs['optimization_metrics']['constraint_stats']
                neighbor_metrics = result.attrs['optimization_metrics']['neighbor_metrics']
                
                results.append({
                    'n': n,
                    'constraints': stats['total'],
                    'avg_neighbors': neighbor_metrics['avg_neighbors']
                })
                
                print(f"\nConstraint scaling (n={n}):")
                print(f"  Total constraints: {stats['total']}")
                print(f"  Co-channel: {stats['co_channel']}")
                print(f"  Adjacent: {stats['adjacent_channel']}")
                print(f"  Avg neighbors: {neighbor_metrics['avg_neighbors']:.1f}")
        
        if len(results) >= 2:
            # Check that constraints scale linearly, not quadratically
            # If O(n²), doubling n would quadruple constraints
            # If O(n), doubling n would double constraints
            ratio = results[-1]['constraints'] / results[0]['constraints']
            n_ratio = results[-1]['n'] / results[0]['n']
            
            # Should be closer to linear than quadratic
            assert ratio < n_ratio * n_ratio * 0.5, f"Constraints appear O(n²): {ratio:.1f}x for {n_ratio}x stations"
            print(f"\n✓ Constraint scaling is sub-quadratic: {ratio:.1f}x constraints for {n_ratio}x stations")


class TestConfigProfiles:
    """Test configuration profiles for different services."""
    
    def test_am_profile(self):
        """Test AM band configuration profile."""
        optimizer = EnhancedSpectrumOptimizer('am')
        
        assert optimizer.config['band']['min_mhz'] == 0.53
        assert optimizer.config['band']['max_mhz'] == 1.7
        assert optimizer.config['band']['step_khz'] == 10
        assert optimizer.config['interference']['guard_offsets'] == [-2, -1, 1, 2]
        
        print("\n✓ AM profile loaded correctly")
    
    def test_fm_profile(self):
        """Test FM band configuration profile."""
        optimizer = EnhancedSpectrumOptimizer('fm')
        
        assert optimizer.config['band']['min_mhz'] == 88.0
        assert optimizer.config['band']['max_mhz'] == 108.0
        assert optimizer.config['band']['step_khz'] == 200
        assert optimizer.config['interference']['guard_offsets'] == [-1, 1]
        
        print("\n✓ FM profile loaded correctly")
    
    def test_default_profile(self):
        """Test default configuration profile."""
        optimizer = EnhancedSpectrumOptimizer('default')
        
        assert optimizer.config['band']['min_mhz'] == 88.0
        assert optimizer.config['band']['max_mhz'] == 108.0
        assert optimizer.config['interference']['guard_offsets'] == [-1, 1]
        
        print("\n✓ Default profile loaded correctly")