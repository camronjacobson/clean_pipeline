"""
Test suite for channel packing optimization.
Tests that optimizer minimizes channels used and prefers lower indices.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from .test_adapter import SpectrumOptimizer
import config


class TestChannelPacking:
    """Test cases for efficient channel packing and frequency assignment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = SpectrumOptimizer(
            freq_params={
                'min_freq': 88.0,
                'max_freq': 108.0,
                'channel_step': 0.2,
                'band_name': 'FM'
            }
        )
    
    def test_minimal_channels_used(self):
        """
        Test that optimizer uses minimum number of channels.
        For k-colorable graph, should use exactly k channels.
        """
        # Create a bipartite graph (2-colorable)
        # Two groups that don't interfere internally but do across groups
        group1 = [(40.0 + i*0.2, -120.0) for i in range(3)]  # Vertical line, far apart
        group2 = [(40.05 + i*0.2, -119.995) for i in range(3)]  # Parallel line, close
        
        stations = []
        for i, (lat, lon) in enumerate(group1):
            stations.append({
                'station_id': f'G1_{i}',
                'latitude': lat,
                'longitude': lon,
                'power_watts': 5000,
                'azimuth_deg': 90,  # Point east toward group 2
                'beamwidth_deg': 60,
                'frequency_mhz': 90.0 + i*0.2
            })
        
        for i, (lat, lon) in enumerate(group2):
            stations.append({
                'station_id': f'G2_{i}',
                'latitude': lat,
                'longitude': lon,
                'power_watts': 5000,
                'azimuth_deg': 270,  # Point west toward group 1
                'beamwidth_deg': 60,
                'frequency_mhz': 93.0 + i*0.2
            })
        
        df = pd.DataFrame(stations)
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Should use exactly 2 frequencies (bipartite graph)
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 2, f"Bipartite graph should use exactly 2 frequencies, got {unique_freqs}"
    
    def test_prefer_lower_frequency_indices(self):
        """
        Test that optimizer prefers lower frequency indices.
        Given choice, should pack into lower frequencies.
        """
        # Create isolated stations that don't interfere
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C'],
            'latitude': [40.0, 45.0, 50.0],  # Very far apart
            'longitude': [-120.0, -125.0, -130.0],
            'power_watts': [1000, 1000, 1000],
            'azimuth_deg': [0, 0, 0],
            'beamwidth_deg': [360, 360, 360],
            'frequency_mhz': [100.0, 102.0, 104.0]  # Start with high frequencies
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # All should share the lowest possible frequency
        assigned_freqs = result['assigned_frequency'].values
        assert len(set(assigned_freqs)) == 1, "Non-interfering stations should share frequency"
        
        # Should use the minimum frequency in the band
        min_possible = self.optimizer.freq_params['min_freq']
        assert assigned_freqs[0] == min_possible, f"Should use lowest frequency {min_possible}, got {assigned_freqs[0]}"
    
    def test_contiguous_channel_assignment(self):
        """
        Test that channels are assigned contiguously when possible.
        Should not have gaps in frequency assignments.
        """
        # Create a chain that needs exactly 3 colors
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D', 'E', 'F'],
            'latitude': [40.0, 40.009, 40.018, 40.027, 40.036, 40.045],  # Chain
            'longitude': [-120.0] * 6,
            'power_watts': [5000] * 6,
            'azimuth_deg': [0] * 6,
            'beamwidth_deg': [360] * 6,
            'frequency_mhz': [90.0 + i*2 for i in range(6)]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Get unique frequencies and sort them
        unique_freqs = sorted(result['assigned_frequency'].unique())
        
        # Check that frequencies are contiguous (no gaps)
        if len(unique_freqs) > 1:
            freq_diffs = np.diff(unique_freqs)
            channel_step = self.optimizer.freq_params['channel_step']
            
            # All differences should be multiples of channel_step with no gaps
            for diff in freq_diffs:
                assert abs(diff - channel_step) < 0.01 or abs(diff - 2*channel_step) < 0.01, \
                    f"Frequencies should be contiguous, found gap of {diff} MHz"
    
    def test_spectrum_span_minimization(self):
        """
        Test that optimizer minimizes the span of used spectrum.
        Should pack frequencies tightly together.
        """
        # Create stations that need 3 frequencies
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C'],
            'latitude': [40.0, 40.009, 40.0],  # Triangle
            'longitude': [-120.0, -119.995, -119.99],
            'power_watts': [5000, 5000, 5000],
            'azimuth_deg': [0, 0, 0],
            'beamwidth_deg': [360, 360, 360],
            'frequency_mhz': [88.0, 98.0, 108.0]  # Spread across band initially
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Calculate spectrum span
        freqs = result['assigned_frequency'].values
        span = max(freqs) - min(freqs)
        
        # For 3 frequencies with 0.2 MHz spacing, minimum span is 0.4 MHz
        expected_span = 0.4  # 2 * channel_step
        assert span <= expected_span + 0.01, f"Spectrum span {span} should be minimized to ~{expected_span} MHz"
    
    def test_reuse_maximization(self):
        """
        Test that optimizer maximizes frequency reuse.
        Non-interfering stations should share frequencies.
        """
        # Create two isolated groups
        group1_lats = [40.0 + i*0.001 for i in range(5)]  # Close cluster
        group2_lats = [50.0 + i*0.001 for i in range(5)]  # Far cluster
        
        stations = []
        for i, lat in enumerate(group1_lats):
            stations.append({
                'station_id': f'G1_{i}',
                'latitude': lat,
                'longitude': -120.0,
                'power_watts': 100,  # Low power
                'azimuth_deg': 0,
                'beamwidth_deg': 360,
                'frequency_mhz': 90.0 + i*0.2
            })
        
        for i, lat in enumerate(group2_lats):
            stations.append({
                'station_id': f'G2_{i}',
                'latitude': lat,
                'longitude': -120.0,
                'power_watts': 100,  # Low power
                'azimuth_deg': 0,
                'beamwidth_deg': 360,
                'frequency_mhz': 95.0 + i*0.2
            })
        
        df = pd.DataFrame(stations)
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Count frequency reuse
        freq_counts = result['assigned_frequency'].value_counts()
        max_reuse = freq_counts.max()
        
        # Each frequency used within a group should be reused in the other group
        assert max_reuse >= 2, f"Isolated groups should reuse frequencies, max reuse is {max_reuse}"
    
    def test_greedy_vs_optimal_packing(self):
        """
        Test case where greedy packing differs from optimal.
        Exposes if optimizer uses naive greedy algorithm.
        """
        # Create a graph where greedy first-fit gives suboptimal result
        # Pentagon graph - needs 3 colors, but greedy might use more
        angles = [0, 72, 144, 216, 288]  # Pentagon angles
        radius = 0.01  # About 1km
        
        stations = []
        for i, angle in enumerate(angles):
            lat = 40.0 + radius * np.cos(np.radians(angle))
            lon = -120.0 + radius * np.sin(np.radians(angle))
            stations.append({
                'station_id': f'P{i}',
                'latitude': lat,
                'longitude': lon,
                'power_watts': 5000,
                'azimuth_deg': 0,
                'beamwidth_deg': 360,
                'frequency_mhz': 90.0 + i*0.2
            })
        
        df = pd.DataFrame(stations)
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Pentagon needs exactly 3 colors
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 3, f"Pentagon graph needs exactly 3 colors, got {unique_freqs}"
    
    def test_channel_boundary_cases(self):
        """
        Test edge cases at channel boundaries.
        Ensure proper handling of band edges.
        """
        # Create stations that might push against band edges
        df = pd.DataFrame({
            'station_id': ['LOW1', 'LOW2', 'HIGH1', 'HIGH2'],
            'latitude': [40.0, 40.1, 45.0, 45.1],
            'longitude': [-120.0, -120.0, -125.0, -125.0],
            'power_watts': [1000] * 4,
            'azimuth_deg': [0] * 4,
            'beamwidth_deg': [360] * 4,
            'frequency_mhz': [88.0, 88.2, 107.8, 108.0]  # At band edges
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Check all assignments are within band
        min_freq = self.optimizer.freq_params['min_freq']
        max_freq = self.optimizer.freq_params['max_freq']
        
        for freq in result['assigned_frequency'].values:
            assert min_freq <= freq <= max_freq, f"Frequency {freq} outside band [{min_freq}, {max_freq}]"
    
    def test_large_scale_packing_efficiency(self):
        """
        Test packing efficiency with larger number of stations.
        Should maintain good packing even at scale.
        """
        # Create 20 stations in a grid
        stations = []
        for i in range(4):
            for j in range(5):
                stations.append({
                    'station_id': f'S_{i}_{j}',
                    'latitude': 40.0 + i*0.01,  # ~1km spacing
                    'longitude': -120.0 + j*0.01,
                    'power_watts': 1000,
                    'azimuth_deg': 0,
                    'beamwidth_deg': 360,
                    'frequency_mhz': 90.0 + (i*5 + j)*0.2
                })
        
        df = pd.DataFrame(stations)
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Check packing efficiency
        unique_freqs = result['assigned_frequency'].nunique()
        total_stations = len(result)
        avg_reuse = total_stations / unique_freqs
        
        # For a 4x5 grid, we expect reasonable reuse
        assert avg_reuse >= 1.5, f"Poor packing efficiency: {avg_reuse:.2f} average reuse"
        
        # Check spectrum span is reasonable
        freq_span = result['assigned_frequency'].max() - result['assigned_frequency'].min()
        max_expected_span = unique_freqs * self.optimizer.freq_params['channel_step']
        
        assert freq_span <= max_expected_span * 1.5, \
            f"Spectrum span {freq_span} too large for {unique_freqs} channels"