"""
Tests for schema flexibility and column name variations.
Verifies the tool handles any reasonable input schema.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from schema_normalizer import (
    normalize_columns, 
    add_default_columns,
    validate_input,
    prepare_dataframe,
    get_schema_info,
    SchemaError
)
from spectrum_optimizer_enhanced import EnhancedSpectrumOptimizer


class TestColumnNormalization:
    """Test column name normalization."""
    
    def test_latitude_variations(self):
        """Test various latitude column names."""
        variations = ['lat', 'latitude', 'y', 'y_coord', 'LAT', 'Latitude']
        
        for col_name in variations:
            df = pd.DataFrame({
                col_name: [40.0, 41.0],
                'lon': [-120.0, -121.0]
            })
            
            normalized = normalize_columns(df)
            assert 'latitude' in normalized.columns, f"Failed to normalize '{col_name}' to 'latitude'"
            assert normalized['latitude'].iloc[0] == 40.0
    
    def test_longitude_variations(self):
        """Test various longitude column names."""
        variations = ['lon', 'longitude', 'lng', 'x', 'x_coord', 'LON', 'Long']
        
        for col_name in variations:
            df = pd.DataFrame({
                'lat': [40.0, 41.0],
                col_name: [-120.0, -121.0]
            })
            
            normalized = normalize_columns(df)
            assert 'longitude' in normalized.columns, f"Failed to normalize '{col_name}' to 'longitude'"
            assert normalized['longitude'].iloc[0] == -120.0
    
    def test_frequency_variations(self):
        """Test various frequency column names."""
        variations = ['freq_mhz', 'frequency_mhz', 'frequency', 'center_freq_mhz', 'freq']
        
        for col_name in variations:
            df = pd.DataFrame({
                'lat': [40.0],
                'lon': [-120.0],
                col_name: [100.5]
            })
            
            normalized = normalize_columns(df)
            assert 'frequency_mhz' in normalized.columns, f"Failed to normalize '{col_name}'"
    
    def test_mixed_case_handling(self):
        """Test that mixed case column names are handled."""
        df = pd.DataFrame({
            'LaT': [40.0],
            'LoNgItUdE': [-120.0],
            'FREQ_MHz': [100.0],
            'Power_Watts': [1000]
        })
        
        normalized = normalize_columns(df)
        
        assert 'latitude' in normalized.columns
        assert 'longitude' in normalized.columns
        assert 'frequency_mhz' in normalized.columns
        assert 'power_watts' in normalized.columns
    
    def test_unrecognized_columns_preserved(self):
        """Test that unrecognized columns are preserved in non-strict mode."""
        df = pd.DataFrame({
            'lat': [40.0],
            'lon': [-120.0],
            'custom_field': ['value'],
            'another_field': [123]
        })
        
        normalized = normalize_columns(df, strict=False)
        
        assert 'custom_field' in normalized.columns
        assert 'another_field' in normalized.columns
        assert normalized['custom_field'].iloc[0] == 'value'
    
    def test_strict_mode_raises_error(self):
        """Test that strict mode raises error for unrecognized columns."""
        df = pd.DataFrame({
            'lat': [40.0],
            'lon': [-120.0],
            'unknown_column': [123]
        })
        
        with pytest.raises(SchemaError) as exc_info:
            normalize_columns(df, strict=True)
        
        assert 'Unrecognized columns' in str(exc_info.value)


class TestDefaultColumns:
    """Test adding default columns."""
    
    def test_add_missing_optional_columns(self):
        """Test that missing optional columns are added with defaults."""
        df = pd.DataFrame({
            'latitude': [40.0, 41.0],
            'longitude': [-120.0, -121.0]
        })
        
        with_defaults = add_default_columns(df)
        
        # Check all optional columns added
        assert 'station_id' in with_defaults.columns
        assert 'frequency_mhz' in with_defaults.columns
        assert 'power_watts' in with_defaults.columns
        assert 'azimuth_deg' in with_defaults.columns
        assert 'beamwidth_deg' in with_defaults.columns
        
        # Check default values
        assert with_defaults['station_id'].tolist() == ['S0', 'S1']
        assert all(with_defaults['frequency_mhz'] == 100.0)
        assert all(with_defaults['power_watts'] == 1000.0)
        assert all(with_defaults['azimuth_deg'] == 0.0)
        assert all(with_defaults['beamwidth_deg'] == 360.0)
    
    def test_existing_columns_not_overwritten(self):
        """Test that existing columns are not overwritten."""
        df = pd.DataFrame({
            'latitude': [40.0],
            'longitude': [-120.0],
            'power_watts': [5000]  # Custom value
        })
        
        with_defaults = add_default_columns(df)
        
        assert with_defaults['power_watts'].iloc[0] == 5000  # Not overwritten


class TestInputValidation:
    """Test input validation."""
    
    def test_missing_required_columns(self):
        """Test that missing required columns are detected."""
        df = pd.DataFrame({
            'frequency_mhz': [100.0]
            # Missing latitude and longitude
        })
        
        is_valid, errors = validate_input(df)
        
        assert not is_valid
        assert any('Missing required columns' in e for e in errors)
    
    def test_invalid_latitude_range(self):
        """Test latitude range validation."""
        df = pd.DataFrame({
            'latitude': [95.0],  # Invalid
            'longitude': [-120.0]
        })
        
        is_valid, errors = validate_input(df)
        
        assert not is_valid
        assert any('Latitude values out of range' in e for e in errors)
    
    def test_invalid_longitude_range(self):
        """Test longitude range validation."""
        df = pd.DataFrame({
            'latitude': [40.0],
            'longitude': [200.0]  # Invalid
        })
        
        is_valid, errors = validate_input(df)
        
        assert not is_valid
        assert any('Longitude values out of range' in e for e in errors)
    
    def test_negative_power_values(self):
        """Test that negative power values are detected."""
        df = pd.DataFrame({
            'latitude': [40.0],
            'longitude': [-120.0],
            'power_watts': [-100]  # Invalid
        })
        
        is_valid, errors = validate_input(df)
        
        assert not is_valid
        assert any('negative power values' in e for e in errors)
    
    def test_duplicate_station_ids(self):
        """Test that duplicate station IDs are detected."""
        df = pd.DataFrame({
            'latitude': [40.0, 41.0],
            'longitude': [-120.0, -121.0],
            'station_id': ['A', 'A']  # Duplicate
        })
        
        is_valid, errors = validate_input(df)
        
        assert not is_valid
        assert any('duplicate station IDs' in e for e in errors)
    
    def test_valid_input_passes(self):
        """Test that valid input passes validation."""
        df = pd.DataFrame({
            'latitude': [40.0, 41.0],
            'longitude': [-120.0, -121.0],
            'station_id': ['A', 'B'],
            'frequency_mhz': [100.0, 101.0],
            'power_watts': [1000, 2000]
        })
        
        is_valid, errors = validate_input(df)
        
        assert is_valid
        assert len(errors) == 0


class TestEndToEndFlexibility:
    """Test end-to-end optimization with various schemas."""
    
    def test_minimal_schema(self):
        """Test optimization with minimal required columns."""
        df = pd.DataFrame({
            'lat': [40.0, 40.1, 40.2],
            'lon': [-120.0, -120.1, -120.2]
        })
        
        optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        result = optimizer.optimize(df)
        
        assert 'assigned_frequency' in result.columns
        assert len(result) == 3
        assert result['assigned_frequency'].notna().all()
    
    def test_alternative_column_names(self):
        """Test optimization with alternative column names."""
        df = pd.DataFrame({
            'y_coord': [40.0, 40.1],
            'x_coord': [-120.0, -120.1],
            'freq': [100.0, 101.0],
            'az': [0, 90],
            'beam': [60, 120]
        })
        
        optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        result = optimizer.optimize(df)
        
        assert 'assigned_frequency' in result.columns
        assert len(result) == 2
    
    def test_mixed_schema(self):
        """Test with mixed recognized and unrecognized columns."""
        df = pd.DataFrame({
            'latitude': [40.0, 41.0],
            'x': [-120.0, -121.0],  # Alternative for longitude
            'call_sign': ['KQED', 'KPIX'],  # Alternative for station_id
            'custom_field': ['A', 'B'],  # Unrecognized
            'notes': ['Station A', 'Station B']  # Unrecognized
        })
        
        optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        result = optimizer.optimize(df)
        
        # Original columns preserved
        assert 'custom_field' in result.columns
        assert 'notes' in result.columns
        assert result['custom_field'].tolist() == ['A', 'B']
        
        # Optimization completed
        assert 'assigned_frequency' in result.columns
    
    def test_prepare_dataframe_complete_pipeline(self):
        """Test the complete preparation pipeline."""
        df = pd.DataFrame({
            'y': [40.0, 91.0],  # Second value invalid
            'x': [-120.0, -121.0],
            'freq': [100.0, -1.0],  # Second value invalid
            'power': [1000, 1e10]  # Second value unrealistic
        })
        
        # Should raise error for invalid data
        with pytest.raises(SchemaError) as exc_info:
            prepare_dataframe(df)
        
        error_msg = str(exc_info.value)
        assert 'validation failed' in error_msg.lower()
    
    def test_comma_separated_azimuth(self):
        """Test handling of comma-separated azimuth values."""
        df = pd.DataFrame({
            'latitude': [40.0, 41.0],
            'longitude': [-120.0, -121.0],
            'azimuth_deg': ['123,255', '45']  # First has multiple values
        })
        
        prepared = prepare_dataframe(df)
        
        # Should take first value
        assert prepared['azimuth_deg'].iloc[0] == 123.0
        assert prepared['azimuth_deg'].iloc[1] == 45.0


class TestSchemaInfo:
    """Test schema information retrieval."""
    
    def test_get_schema_info(self):
        """Test getting schema information."""
        df = pd.DataFrame({
            'lat': [40.0],
            'custom_column': [123],
            'frequency': [100.0]
        })
        
        info = get_schema_info(df)
        
        assert info['total_columns'] == 3
        assert 'latitude' in info['recognized_columns']
        assert 'frequency_mhz' in info['recognized_columns']
        assert 'custom_column' in info['unrecognized_columns']
        assert 'longitude' in info['missing_required']
        assert 'power_watts' in info['missing_optional']