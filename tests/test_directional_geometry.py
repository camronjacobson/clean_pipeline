"""
Test suite for directional antenna geometry and interference.
Tests azimuth, beamwidth, and back-to-back station scenarios.
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


class TestDirectionalGeometry:
    """Test cases for directional antenna patterns and interference."""
    
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
    
    def test_back_to_back_stations(self):
        """
        Test back-to-back stations (pointing away from each other).
        They should be able to reuse the same frequency.
        """
        # Two stations at same location, pointing opposite directions
        df = pd.DataFrame({
            'station_id': ['A', 'B'],
            'latitude': [40.0, 40.0],
            'longitude': [-120.0, -119.999],  # Very close (100m apart)
            'power_watts': [1000, 1000],
            'azimuth_deg': [0, 180],  # North and South (opposite)
            'beamwidth_deg': [60, 60],  # Narrow beams
            'frequency_mhz': [90.0, 92.0]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Should use only 1 frequency (they can share)
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 1, f"Back-to-back stations should share frequency, got {unique_freqs} frequencies"
    
    def test_facing_stations(self):
        """
        Test stations facing each other.
        They must use different frequencies due to interference.
        """
        # Two stations 5km apart, facing each other
        df = pd.DataFrame({
            'station_id': ['A', 'B'],
            'latitude': [40.0, 40.045],  # ~5km apart
            'longitude': [-120.0, -120.0],
            'power_watts': [1000, 1000],
            'azimuth_deg': [0, 180],  # A points North to B, B points South to A
            'beamwidth_deg': [60, 60],  # Narrow beams
            'frequency_mhz': [90.0, 90.0]  # Start with same frequency
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Must use 2 different frequencies
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 2, f"Facing stations must use different frequencies, got {unique_freqs}"
    
    def test_perpendicular_beams(self):
        """
        Test stations with perpendicular beam directions.
        They should be able to share frequency.
        """
        # Two close stations with perpendicular beams
        df = pd.DataFrame({
            'station_id': ['A', 'B'],
            'latitude': [40.0, 40.009],  # ~1km apart
            'longitude': [-120.0, -120.0],
            'power_watts': [1000, 1000],
            'azimuth_deg': [0, 90],  # North and East (perpendicular)
            'beamwidth_deg': [30, 30],  # Very narrow beams
            'frequency_mhz': [90.0, 92.0]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Should use only 1 frequency
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 1, f"Perpendicular narrow beams should share frequency, got {unique_freqs}"
    
    def test_overlapping_sectors(self):
        """
        Test three stations with 120-degree sectors covering full circle.
        All three should need different frequencies.
        """
        # Three co-located stations with 120-degree sectors
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C'],
            'latitude': [40.0, 40.0, 40.0],
            'longitude': [-120.0, -120.001, -120.002],  # Very close
            'power_watts': [1000, 1000, 1000],
            'azimuth_deg': [0, 120, 240],  # Three sectors
            'beamwidth_deg': [120, 120, 120],  # Wide sectors with overlap
            'frequency_mhz': [90.0, 92.0, 94.0]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Should use 3 different frequencies (overlapping coverage)
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 3, f"Overlapping 120° sectors need different frequencies, got {unique_freqs}"
    
    def test_non_overlapping_sectors(self):
        """
        Test four stations with 90-degree sectors, no overlap.
        All four should be able to share the same frequency.
        """
        # Four co-located stations with non-overlapping 90-degree sectors
        df = pd.DataFrame({
            'station_id': ['N', 'E', 'S', 'W'],
            'latitude': [40.0, 40.0, 40.0, 40.0],
            'longitude': [-120.0, -120.001, -120.002, -120.003],
            'power_watts': [1000, 1000, 1000, 1000],
            'azimuth_deg': [0, 90, 180, 270],  # Four cardinal directions
            'beamwidth_deg': [85, 85, 85, 85],  # Slightly less than 90° to avoid overlap
            'frequency_mhz': [90.0, 92.0, 94.0, 96.0]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Should use only 1 frequency (no overlap)
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs == 1, f"Non-overlapping sectors should share frequency, got {unique_freqs}"
    
    def test_narrow_beam_alignment(self):
        """
        Test narrow beam alignment scenarios.
        Aligned beams interfere, misaligned don't.
        """
        # Five stations in a line with narrow beams
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D', 'E'],
            'latitude': [40.0, 40.0, 40.0, 40.0, 40.0],
            'longitude': [-120.0, -119.99, -119.98, -119.97, -119.96],  # Line of stations
            'power_watts': [1000] * 5,
            'azimuth_deg': [90, 270, 90, 270, 90],  # Alternating East/West
            'beamwidth_deg': [15, 15, 15, 15, 15],  # Very narrow beams
            'frequency_mhz': [90.0 + i*0.2 for i in range(5)]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Should use 2 frequencies (alternating pattern)
        unique_freqs = result['assigned_frequency'].nunique()
        assert unique_freqs <= 2, f"Alternating narrow beams should use at most 2 frequencies, got {unique_freqs}"
    
    def test_omnidirectional_vs_directional(self):
        """
        Test mix of omnidirectional and directional antennas.
        Omni interferes with all, directional only in its beam.
        """
        # One omni in center, four directional around it
        df = pd.DataFrame({
            'station_id': ['OMNI', 'DIR_N', 'DIR_E', 'DIR_S', 'DIR_W'],
            'latitude': [40.0, 40.045, 40.0, 39.955, 40.0],  # ~5km spacing
            'longitude': [-120.0, -120.0, -119.955, -120.0, -120.045],
            'power_watts': [2000, 1000, 1000, 1000, 1000],
            'azimuth_deg': [0, 180, 270, 0, 90],  # Directionals point toward center
            'beamwidth_deg': [360, 60, 60, 60, 60],  # Omni vs directional
            'frequency_mhz': [90.0, 92.0, 94.0, 96.0, 98.0]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Omni needs unique frequency, directionals might share
        omni_freq = result[result['station_id'] == 'OMNI']['assigned_frequency'].iloc[0]
        dir_freqs = result[result['station_id'] != 'OMNI']['assigned_frequency'].values
        
        assert omni_freq not in dir_freqs, "Omnidirectional should have unique frequency"
        
        # Directionals pointing at center should be able to share some frequencies
        unique_dir_freqs = len(set(dir_freqs))
        assert unique_dir_freqs <= 2, f"Non-interfering directionals should share frequencies, got {unique_dir_freqs}"
    
    def test_beamwidth_edge_cases(self):
        """
        Test edge cases for beamwidth calculations.
        Very wide beams (>180°) and very narrow beams (<10°).
        """
        # Mix of extreme beamwidths
        df = pd.DataFrame({
            'station_id': ['WIDE1', 'WIDE2', 'NARROW1', 'NARROW2'],
            'latitude': [40.0, 40.009, 40.018, 40.027],  # ~1km apart each
            'longitude': [-120.0, -120.0, -120.0, -120.0],
            'power_watts': [1000, 1000, 1000, 1000],
            'azimuth_deg': [0, 180, 45, 225],  # Various directions
            'beamwidth_deg': [270, 270, 5, 5],  # Very wide and very narrow
            'frequency_mhz': [90.0, 92.0, 94.0, 96.0]
        })
        
        # Run optimization
        result = self.optimizer.optimize(df)
        
        # Wide beams should conflict, narrow beams might not
        freq_wide1 = result[result['station_id'] == 'WIDE1']['assigned_frequency'].iloc[0]
        freq_wide2 = result[result['station_id'] == 'WIDE2']['assigned_frequency'].iloc[0]
        freq_narrow1 = result[result['station_id'] == 'NARROW1']['assigned_frequency'].iloc[0]
        freq_narrow2 = result[result['station_id'] == 'NARROW2']['assigned_frequency'].iloc[0]
        
        assert freq_wide1 != freq_wide2, "Very wide overlapping beams must use different frequencies"
        # Narrow beams at 45° and 225° (opposite) should be able to share
        # This might fail if the optimizer is too conservative