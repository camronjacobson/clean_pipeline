"""
Property-based tests for spectrum optimizer.
Tests invariants and properties that should hold under perturbations.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, List
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from .test_adapter import SpectrumOptimizer
import config


class TestPropertyBased:
    """Property-based tests for robustness and consistency."""
    
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
        # Set seed for reproducibility
        np.random.seed(42)
        random.seed(42)
    
    def test_azimuth_rotation_invariance(self):
        """
        Property: Rotating all azimuths by same amount shouldn't change interference.
        Tests that relative angles matter, not absolute.
        """
        # Create base configuration
        base_df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D'],
            'latitude': [40.0, 40.01, 40.0, 39.99],  # Square
            'longitude': [-120.0, -120.0, -119.99, -120.0],
            'power_watts': [1000] * 4,
            'azimuth_deg': [0, 90, 180, 270],  # Pointing outward
            'beamwidth_deg': [60] * 4,
            'frequency_mhz': [90.0, 92.0, 94.0, 96.0]
        })
        
        # Test with different global rotations
        rotations = [0, 45, 90, 180, 270]
        results = []
        
        for rotation in rotations:
            df_rotated = base_df.copy()
            df_rotated['azimuth_deg'] = (df_rotated['azimuth_deg'] + rotation) % 360
            
            result = self.optimizer.optimize(df_rotated)
            unique_freqs = result['assigned_frequency'].nunique()
            results.append(unique_freqs)
        
        # All rotations should give same number of frequencies
        assert len(set(results)) == 1, f"Azimuth rotation changed result: {results}"
    
    def test_beamwidth_monotonicity(self):
        """
        Property: Increasing beamwidth should not decrease interference.
        Wider beams = more potential interference = more frequencies needed.
        """
        # Create configuration with variable beamwidth
        base_df = pd.DataFrame({
            'station_id': ['A', 'B', 'C'],
            'latitude': [40.0, 40.009, 40.018],  # Line of stations
            'longitude': [-120.0, -120.0, -120.0],
            'power_watts': [1000] * 3,
            'azimuth_deg': [90, 270, 90],  # Alternating directions
            'beamwidth_deg': [30, 30, 30],  # Start narrow
            'frequency_mhz': [90.0, 92.0, 94.0]
        })
        
        beamwidths = [10, 30, 60, 120, 180, 360]
        frequencies_needed = []
        
        for bw in beamwidths:
            df = base_df.copy()
            df['beamwidth_deg'] = bw
            
            result = self.optimizer.optimize(df)
            unique_freqs = result['assigned_frequency'].nunique()
            frequencies_needed.append(unique_freqs)
        
        # Check monotonicity (non-decreasing)
        for i in range(1, len(frequencies_needed)):
            assert frequencies_needed[i] >= frequencies_needed[i-1], \
                f"Beamwidth increase decreased frequencies: {beamwidths[i-1]}°→{beamwidths[i]}° " \
                f"gave {frequencies_needed[i-1]}→{frequencies_needed[i]} frequencies"
    
    def test_power_scaling_consistency(self):
        """
        Property: Scaling all power by same factor shouldn't change topology.
        Tests that relative power matters for interference.
        """
        base_df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D', 'E'],
            'latitude': [40.0, 40.009, 40.018, 40.027, 40.036],
            'longitude': [-120.0] * 5,
            'power_watts': [100, 500, 1000, 500, 100],  # Variable power
            'azimuth_deg': [0] * 5,
            'beamwidth_deg': [360] * 5,
            'frequency_mhz': [90.0 + i*0.2 for i in range(5)]
        })
        
        # Test with different power scales
        scales = [0.1, 1.0, 10.0, 100.0]
        results = []
        
        for scale in scales:
            df = base_df.copy()
            df['power_watts'] = df['power_watts'] * scale
            
            result = self.optimizer.optimize(df)
            # Get the assignment pattern (which stations share frequencies)
            assignment = result.set_index('station_id')['assigned_frequency'].to_dict()
            # Normalize to pattern (relative assignment)
            pattern = []
            freq_map = {}
            next_id = 0
            for station in ['A', 'B', 'C', 'D', 'E']:
                freq = assignment[station]
                if freq not in freq_map:
                    freq_map[freq] = next_id
                    next_id += 1
                pattern.append(freq_map[freq])
            results.append(tuple(pattern))
        
        # All scales should give similar interference pattern
        # (May not be exactly same due to distance thresholds)
        unique_patterns = len(set(results))
        assert unique_patterns <= 2, f"Power scaling changed topology too much: {results}"
    
    def test_location_perturbation_stability(self):
        """
        Property: Small location changes should not drastically change solution.
        Tests solution stability/continuity.
        """
        base_df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D'],
            'latitude': [40.0, 40.1, 40.2, 40.3],  # Well separated
            'longitude': [-120.0, -120.1, -120.2, -120.3],
            'power_watts': [1000] * 4,
            'azimuth_deg': [0] * 4,
            'beamwidth_deg': [360] * 4,
            'frequency_mhz': [90.0, 92.0, 94.0, 96.0]
        })
        
        # Get baseline result
        baseline = self.optimizer.optimize(base_df)
        baseline_freqs = baseline['assigned_frequency'].nunique()
        
        # Test with small perturbations
        for _ in range(10):
            df = base_df.copy()
            # Add small random perturbations (< 100m)
            df['latitude'] += np.random.normal(0, 0.0001, len(df))
            df['longitude'] += np.random.normal(0, 0.0001, len(df))
            
            result = self.optimizer.optimize(df)
            perturbed_freqs = result['assigned_frequency'].nunique()
            
            # Should not change by more than 1 frequency
            assert abs(perturbed_freqs - baseline_freqs) <= 1, \
                f"Small perturbation caused large change: {baseline_freqs}→{perturbed_freqs}"
    
    def test_frequency_assignment_determinism(self):
        """
        Property: Same input should always produce same output.
        Tests deterministic behavior.
        """
        df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D', 'E', 'F'],
            'latitude': [40.0 + i*0.01 for i in range(6)],
            'longitude': [-120.0] * 6,
            'power_watts': [1000] * 6,
            'azimuth_deg': [i * 60 for i in range(6)],
            'beamwidth_deg': [45] * 6,
            'frequency_mhz': [90.0 + i*0.2 for i in range(6)]
        })
        
        # Run multiple times
        results = []
        for _ in range(5):
            result = self.optimizer.optimize(df.copy())
            # Extract assignment as sorted list
            assignment = result.sort_values('station_id')['assigned_frequency'].tolist()
            results.append(tuple(assignment))
        
        # All runs should give exact same result
        unique_results = len(set(results))
        assert unique_results == 1, f"Non-deterministic results: {unique_results} different outcomes"
    
    def test_constraint_satisfaction_under_permutation(self):
        """
        Property: Reordering stations shouldn't violate constraints.
        Tests that solution quality is independent of input order.
        """
        base_df = pd.DataFrame({
            'station_id': ['A', 'B', 'C', 'D', 'E'],
            'latitude': [40.0, 40.01, 40.02, 40.03, 40.04],
            'longitude': [-120.0, -120.01, -120.0, -120.01, -120.0],
            'power_watts': [1000, 2000, 1000, 2000, 1000],
            'azimuth_deg': [0, 90, 180, 270, 0],
            'beamwidth_deg': [90, 90, 90, 90, 90],
            'frequency_mhz': [90.0, 92.0, 94.0, 96.0, 98.0]
        })
        
        # Test different permutations
        permutations = [
            [0, 1, 2, 3, 4],  # Original
            [4, 3, 2, 1, 0],  # Reverse
            [2, 0, 4, 1, 3],  # Random
        ]
        
        results = []
        for perm in permutations:
            df = base_df.iloc[perm].reset_index(drop=True)
            result = self.optimizer.optimize(df)
            
            # Check number of unique frequencies
            unique_freqs = result['assigned_frequency'].nunique()
            results.append(unique_freqs)
        
        # All permutations should use same number of frequencies
        assert len(set(results)) == 1, f"Order dependency detected: {results}"
    
    def test_incremental_station_addition(self):
        """
        Property: Adding stations should not decrease frequency reuse.
        More stations = more opportunities for reuse (if well-placed).
        """
        # Start with 2 stations
        stations = [
            {'station_id': 'A', 'latitude': 40.0, 'longitude': -120.0},
            {'station_id': 'B', 'latitude': 45.0, 'longitude': -125.0},  # Far apart
        ]
        
        reuse_rates = []
        
        # Incrementally add stations
        for i in range(3, 8):
            # Add a new station far from others
            stations.append({
                'station_id': chr(65 + i - 1),  # C, D, E, ...
                'latitude': 40.0 + i * 5,
                'longitude': -120.0 - i * 5
            })
            
            df = pd.DataFrame(stations)
            df['power_watts'] = 1000
            df['azimuth_deg'] = 0
            df['beamwidth_deg'] = 360
            df['frequency_mhz'] = 90.0
            
            result = self.optimizer.optimize(df)
            
            # Calculate reuse rate
            unique_freqs = result['assigned_frequency'].nunique()
            reuse_rate = len(result) / unique_freqs
            reuse_rates.append(reuse_rate)
        
        # Reuse rate should generally increase or stay same
        # (Allow small decrease due to optimization boundaries)
        for i in range(1, len(reuse_rates)):
            assert reuse_rates[i] >= reuse_rates[i-1] - 0.2, \
                f"Reuse rate decreased: {reuse_rates[i-1]:.2f}→{reuse_rates[i]:.2f}"
    
    def test_symmetry_preservation(self):
        """
        Property: Symmetric configurations should produce symmetric solutions.
        """
        # Create symmetric star configuration
        center = (40.0, -120.0)
        radius = 0.05  # ~5km
        
        stations = [{'station_id': 'CENTER', 'latitude': center[0], 'longitude': center[1]}]
        
        # Add symmetric satellites
        for i in range(6):
            angle = i * 60  # Hexagonal symmetry
            lat = center[0] + radius * np.cos(np.radians(angle))
            lon = center[1] + radius * np.sin(np.radians(angle))
            stations.append({
                'station_id': f'S{i}',
                'latitude': lat,
                'longitude': lon
            })
        
        df = pd.DataFrame(stations)
        df['power_watts'] = 1000
        df['azimuth_deg'] = 0
        df['beamwidth_deg'] = 360
        df['frequency_mhz'] = 90.0
        
        result = self.optimizer.optimize(df)
        
        # All satellites should get same frequency (due to symmetry)
        satellite_freqs = result[result['station_id'].str.startswith('S')]['assigned_frequency'].values
        assert len(set(satellite_freqs)) == 1, f"Symmetric satellites got different frequencies: {satellite_freqs}"
        
        # Center should have different frequency
        center_freq = result[result['station_id'] == 'CENTER']['assigned_frequency'].iloc[0]
        assert center_freq != satellite_freqs[0], "Center should have different frequency from satellites"
    
    def test_distance_threshold_boundary(self):
        """
        Property: Stations just inside/outside interference threshold.
        Tests boundary condition handling.
        """
        # Assuming ~30km interference threshold for these powers
        threshold_km = 30
        
        # Create pairs at various distances
        distances_km = [29, 30, 31, 50, 100]  # Around threshold
        
        for dist_km in distances_km:
            # Convert to degrees (approximate)
            dist_deg = dist_km / 111.0
            
            df = pd.DataFrame({
                'station_id': ['A', 'B'],
                'latitude': [40.0, 40.0 + dist_deg],
                'longitude': [-120.0, -120.0],
                'power_watts': [5000, 5000],  # High power
                'azimuth_deg': [0, 180],  # Facing each other
                'beamwidth_deg': [360, 360],
                'frequency_mhz': [90.0, 90.0]
            })
            
            result = self.optimizer.optimize(df)
            unique_freqs = result['assigned_frequency'].nunique()
            
            # Stations within threshold should need 2 frequencies
            # Stations beyond should share 1 frequency
            if dist_km <= threshold_km:
                assert unique_freqs == 2, f"Close stations ({dist_km}km) should use different frequencies"
            else:
                assert unique_freqs == 1, f"Distant stations ({dist_km}km) should share frequency"