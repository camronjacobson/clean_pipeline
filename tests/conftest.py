"""
Pytest configuration and shared fixtures for spectrum optimizer tests.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def mock_spectrum_optimizer():
    """
    Create a mock SpectrumOptimizer that implements the expected interface.
    This allows tests to run even if the actual implementation has issues.
    """
    class MockSpectrumOptimizer:
        def __init__(self, freq_params):
            self.freq_params = freq_params
        
        def optimize(self, df):
            """
            Naive implementation for testing:
            - Assigns frequencies greedily
            - Checks basic interference (distance-based)
            """
            result = df.copy()
            n_stations = len(df)
            
            # Simple greedy coloring based on distance
            # This will fail many tests, exposing optimizer weaknesses
            assigned = {}
            available_freqs = np.arange(
                self.freq_params['min_freq'],
                self.freq_params['max_freq'],
                self.freq_params['channel_step']
            )
            
            for i, row in df.iterrows():
                used_freqs = set()
                
                # Check interference with already assigned stations
                for j, other in df.iterrows():
                    if j >= i or j not in assigned:
                        continue
                    
                    # Simple distance check (10km threshold)
                    dist = self._haversine(
                        row['latitude'], row['longitude'],
                        other['latitude'], other['longitude']
                    )
                    
                    if dist < 10:  # 10km interference threshold
                        used_freqs.add(assigned[j])
                
                # Assign first available frequency
                for freq in available_freqs:
                    if freq not in used_freqs:
                        assigned[i] = freq
                        break
                else:
                    # No frequency available, assign None
                    assigned[i] = None
            
            # Add assignments to result
            result['assigned_frequency'] = [assigned.get(i, None) for i in range(n_stations)]
            
            return result
        
        def _haversine(self, lat1, lon1, lat2, lon2):
            """Calculate distance in km between two points."""
            R = 6371  # Earth radius in km
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
    
    return MockSpectrumOptimizer


@pytest.fixture
def sample_am_data():
    """Generate sample AM band data for testing."""
    return pd.DataFrame({
        'station_id': [f'AM_{i}' for i in range(10)],
        'latitude': np.linspace(35, 45, 10),
        'longitude': np.linspace(-125, -115, 10),
        'frequency_mhz': np.linspace(0.53, 1.7, 10),  # AM band in MHz
        'power_watts': [1000] * 10,
        'azimuth_deg': [0] * 10,
        'beamwidth_deg': [360] * 10,
        'x_coord': np.linspace(-125, -115, 10),
        'y_coord': np.linspace(35, 45, 10),
        'area_type': ['urban'] * 3 + ['suburban'] * 4 + ['rural'] * 3
    })


@pytest.fixture
def sample_fm_data():
    """Generate sample FM band data for testing."""
    return pd.DataFrame({
        'station_id': [f'FM_{i}' for i in range(10)],
        'latitude': np.linspace(35, 45, 10),
        'longitude': np.linspace(-125, -115, 10),
        'frequency_mhz': np.linspace(88, 108, 10),  # FM band
        'power_watts': [5000] * 10,
        'azimuth_deg': np.random.randint(0, 360, 10),
        'beamwidth_deg': [120] * 5 + [360] * 5,  # Mix of directional and omni
        'x_coord': np.linspace(-125, -115, 10),
        'y_coord': np.linspace(35, 45, 10),
        'area_type': ['urban'] * 3 + ['suburban'] * 4 + ['rural'] * 3
    })


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for full pipeline"
    )
    config.addinivalue_line(
        "markers", "property: Property-based tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 10 seconds"
    )
    config.addinivalue_line(
        "markers", "requires_data: Tests that require the actual data files"
    )