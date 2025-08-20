"""
Test suite for toy graphs with known optimal solutions.
These tests expose weaknesses in the current optimizer implementation.
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


class TestToyGraphs:
    """Test cases using small graphs with known optimal solutions."""
    
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
    
    def test_3_station_chain(self):
        """
        Test a 3-station chain: A -- B -- C
        Optimal: A and C share frequency, B gets different frequency.
        Total channels used: 2
        """
        # Create stations in a line, 10km apart
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C'],
            'latitude': [40.0, 40.0, 40.0],
            'longitude': [-120.0, -119.91, -119.82],  # ~10km apart
            'power_watts': [1000, 1000, 1000],
            'azimuth_deg': [90, 90, 90],  # All pointing east
            'beamwidth_deg': [360, 360, 360],  # Omnidirectional
            'frequency_mhz': [90.0, 92.0, 94.0]  # Initial frequencies (to be optimized)
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Check that exactly 2 unique frequencies are used
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 2, f"Expected 2 unique frequencies, got {unique_freqs}"
        
        # Check that A and C share frequency (they don't interfere)
        freq_a = result[result['station_id'] == 'A']['assigned_frequency'].iloc[0]
        freq_c = result[result['station_id'] == 'C']['assigned_frequency'].iloc[0]
        freq_b = result[result['station_id'] == 'B']['assigned_frequency'].iloc[0]
        
        assert freq_a == freq_c, f"Stations A and C should share frequency, got A:{freq_a}, C:{freq_c}"
        assert freq_b != freq_a, f"Station B should have different frequency from A, got B:{freq_b}, A:{freq_a}"
    
    def test_4_station_ring(self):
        """
        Test a 4-station ring: A -- B
                               |    |
                               D -- C
        Optimal: A,C share one frequency, B,D share another.
        Total channels used: 2
        """
        # Create 4 stations in a square, 10km apart
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D'],
            'latitude': [40.0, 40.0, 39.91, 39.91],  # Square pattern
            'longitude': [-120.0, -119.91, -119.91, -120.0],
            'power_watts': [1000, 1000, 1000, 1000],
            'azimuth_deg': [0, 0, 0, 0],
            'beamwidth_deg': [360, 360, 360, 360],  # All omnidirectional
            'frequency_mhz': [90.0, 92.0, 94.0, 96.0]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Check that exactly 2 unique frequencies are used
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 2, f"Expected 2 unique frequencies for 4-station ring, got {unique_freqs}"
        
        # Check the diagonal pairs share frequencies
        freq_a = result[result['station_id'] == 'A']['assigned_frequency'].iloc[0]
        freq_b = result[result['station_id'] == 'B']['assigned_frequency'].iloc[0]
        freq_c = result[result['station_id'] == 'C']['assigned_frequency'].iloc[0]
        freq_d = result[result['station_id'] == 'D']['assigned_frequency'].iloc[0]
        
        # A and C should share (diagonal), B and D should share (diagonal)
        assert freq_a == freq_c, f"Diagonal stations A and C should share frequency"
        assert freq_b == freq_d, f"Diagonal stations B and D should share frequency"
        assert freq_a != freq_b, f"Adjacent stations should not share frequency"
    
    def test_2x3_grid(self):
        """
        Test a 2x3 grid:  A -- B -- C
                          |    |    |
                          D -- E -- F
        Optimal: 3 colors needed (like graph coloring)
        Pattern: A,E share; B,D,F share; C shares with A,E
        Total channels used: 3
        """
        # Create 6 stations in a 2x3 grid
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D', 'E', 'F'],
            'latitude': [40.0, 40.0, 40.0, 39.91, 39.91, 39.91],
            'longitude': [-120.0, -119.91, -119.82, -120.0, -119.91, -119.82],
            'power_watts': [1000] * 6,
            'azimuth_deg': [0] * 6,
            'beamwidth_deg': [360] * 6,
            'frequency_mhz': [90.0 + i*2 for i in range(6)]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Check that exactly 3 unique frequencies are used
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 3, f"Expected 3 unique frequencies for 2x3 grid, got {unique_freqs}"
    
    def test_isolated_stations(self):
        """
        Test stations that are far apart (>100km).
        All should share the same frequency.
        """
        # Create 5 stations very far apart
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D', 'E'],
            'latitude': [40.0, 42.0, 44.0, 46.0, 48.0],  # ~200km apart each
            'longitude': [-120.0, -122.0, -124.0, -126.0, -128.0],
            'power_watts': [1000] * 5,
            'azimuth_deg': [0] * 5,
            'beamwidth_deg': [360] * 5,
            'frequency_mhz': [90.0 + i*2 for i in range(5)]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # All stations should share the same frequency
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 1, f"Expected 1 unique frequency for isolated stations, got {unique_freqs}"
    
    def test_dense_cluster(self):
        """
        Test a dense cluster where all stations interfere.
        Each should get a unique frequency.
        """
        # Create 5 stations very close together (1km apart)
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D', 'E'],
            'latitude': [40.0, 40.009, 40.018, 40.027, 40.036],  # ~1km apart
            'longitude': [-120.0, -120.0, -120.0, -120.0, -120.0],
            'power_watts': [5000] * 5,  # High power
            'azimuth_deg': [0] * 5,
            'beamwidth_deg': [360] * 5,
            'frequency_mhz': [90.0 + i*0.2 for i in range(5)]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # All stations should have unique frequencies
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 5, f"Expected 5 unique frequencies for dense cluster, got {unique_freqs}"
    
    def test_star_topology(self):
        """
        Test star topology: one central station with 4 satellites.
        Center needs unique frequency, satellites can share.
        Optimal: 2 frequencies
        """
        # Create star topology
        df = pd.DataFrame({
            'station_id': ['CENTER', 'N', 'E', 'S', 'W'],
            'latitude': [40.0, 40.09, 40.0, 39.91, 40.0],  # 10km from center
            'longitude': [-120.0, -120.0, -119.91, -120.0, -120.09],
            'power_watts': [2000, 1000, 1000, 1000, 1000],
            'azimuth_deg': [0] * 5,
            'beamwidth_deg': [360] * 5,
            'frequency_mhz': [90.0 + i*2 for i in range(5)]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Should use exactly 2 frequencies
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 2, f"Expected 2 frequencies for star topology, got {unique_freqs}"
        
        # Center should have unique frequency
        center_freq = result[result['station_id'] == 'CENTER']['assigned_frequency'].iloc[0]
        satellite_freqs = result[result['station_id'] != 'CENTER']['assigned_frequency'].unique()
        
        assert len(satellite_freqs) == 1, "All satellites should share the same frequency"
        assert center_freq != satellite_freqs[0], "Center should have different frequency from satellites"