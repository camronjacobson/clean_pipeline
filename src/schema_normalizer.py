"""
Schema normalization and validation for spectrum optimization.
Handles column name variations and ensures data integrity.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Column name synonyms mapping to standard names
COLUMN_SYNONYMS = {
    'latitude': ['lat', 'latitude', 'y', 'y_coord', 'y_coordinate', 'lat_deg'],
    'longitude': ['lon', 'longitude', 'lng', 'long', 'x', 'x_coord', 'x_coordinate', 'lon_deg', 'long_deg'],
    'station_id': ['station_id', 'id', 'call_sign', 'callsign', 'name', 'station_name', 'site_id'],
    'frequency_mhz': ['center_freq_mhz', 'freq_mhz', 'frequency_mhz', 'frequency', 'freq', 'center_frequency'],
    'bandwidth_khz': ['bandwidth_khz', 'bw_khz', 'bandwidth', 'bw', 'channel_width'],
    'power_watts': ['power_watts', 'power', 'erp', 'erp_watts', 'tx_power', 'transmit_power'],
    'azimuth_deg': ['azimuth_deg', 'azimuth', 'az', 'bearing', 'direction', 'heading'],
    'beamwidth_deg': ['beamwidth_deg', 'beam', 'beamwidth', 'bw_deg', 'beam_width', 'pattern_width']
}

# Required columns for operation
REQUIRED_COLUMNS = ['latitude', 'longitude']

# Optional columns with defaults
OPTIONAL_DEFAULTS = {
    'station_id': lambda df: [f'S{i}' for i in range(len(df))],
    'frequency_mhz': lambda df: 100.0,
    'bandwidth_khz': lambda df: 200.0,
    'power_watts': lambda df: 1000.0,
    'azimuth_deg': lambda df: 0.0,
    'beamwidth_deg': lambda df: 360.0
}


class SchemaError(Exception):
    """Custom exception for schema-related errors."""
    pass


def normalize_columns(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Normalize column names to standard format.
    
    Args:
        df: Input DataFrame
        strict: If True, raise error for unrecognized columns
        
    Returns:
        DataFrame with normalized column names
        
    Raises:
        SchemaError: If strict mode and unrecognized columns exist
    """
    df = df.copy()
    normalized_cols = {}
    unrecognized = []
    
    # Track which columns have been normalized
    used_columns = set()
    
    for col in df.columns:
        col_lower = col.lower().strip()
        found = False
        
        # Check against each standard column's synonyms
        for standard, synonyms in COLUMN_SYNONYMS.items():
            if col_lower in [s.lower() for s in synonyms]:
                if standard in normalized_cols.values():
                    logger.warning(f"Duplicate mapping for {standard}: {col} ignored")
                else:
                    normalized_cols[col] = standard
                    used_columns.add(col)
                found = True
                break
        
        if not found and col not in used_columns:
            # Keep unrecognized columns as-is unless strict mode
            if strict:
                unrecognized.append(col)
            else:
                normalized_cols[col] = col
    
    if strict and unrecognized:
        raise SchemaError(f"Unrecognized columns in strict mode: {unrecognized}")
    
    # Apply renaming
    df = df.rename(columns=normalized_cols)
    
    logger.info(f"Normalized {len(normalized_cols)} columns")
    if normalized_cols:
        logger.debug(f"Column mappings: {normalized_cols}")
    
    return df


def add_default_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add default values for missing optional columns.
    
    Args:
        df: Input DataFrame with normalized columns
        
    Returns:
        DataFrame with all optional columns filled
    """
    df = df.copy()
    
    for col, default_func in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            default_value = default_func(df)
            df[col] = default_value
            logger.info(f"Added default column '{col}' with value(s): {default_value if not isinstance(default_value, list) else 'generated'}")
    
    return df


def validate_input(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has required columns and valid data.
    
    Args:
        df: DataFrame to validate (should be normalized first)
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Check required columns
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        errors.append(f"Missing required columns: {missing_required}")
        return False, errors  # Can't continue without required columns
    
    # Validate data types and ranges
    
    # Latitude validation
    if 'latitude' in df.columns:
        lat_values = df['latitude']
        if not pd.api.types.is_numeric_dtype(lat_values):
            errors.append("Latitude must be numeric")
        elif lat_values.isna().any():
            errors.append(f"Found {lat_values.isna().sum()} missing latitude values")
        elif not ((-90 <= lat_values.min()) and (lat_values.max() <= 90)):
            errors.append(f"Latitude values out of range [-90, 90]: min={lat_values.min():.2f}, max={lat_values.max():.2f}")
    
    # Longitude validation
    if 'longitude' in df.columns:
        lon_values = df['longitude']
        if not pd.api.types.is_numeric_dtype(lon_values):
            errors.append("Longitude must be numeric")
        elif lon_values.isna().any():
            errors.append(f"Found {lon_values.isna().sum()} missing longitude values")
        elif not ((-180 <= lon_values.min()) and (lon_values.max() <= 180)):
            errors.append(f"Longitude values out of range [-180, 180]: min={lon_values.min():.2f}, max={lon_values.max():.2f}")
    
    # Frequency validation (if present)
    if 'frequency_mhz' in df.columns:
        freq_values = df['frequency_mhz']
        if pd.api.types.is_numeric_dtype(freq_values):
            if (freq_values <= 0).any():
                errors.append(f"Found {(freq_values <= 0).sum()} non-positive frequency values")
            elif freq_values.max() > 10000:  # Sanity check for MHz
                errors.append(f"Frequency values seem too high (max={freq_values.max():.1f} MHz)")
    
    # Power validation (if present)
    if 'power_watts' in df.columns:
        power_values = df['power_watts']
        if pd.api.types.is_numeric_dtype(power_values):
            if (power_values < 0).any():
                errors.append(f"Found {(power_values < 0).sum()} negative power values")
            elif power_values.max() > 1e7:  # 10 MW sanity check
                errors.append(f"Power values seem too high (max={power_values.max():.0f} watts)")
    
    # Azimuth validation (if present)
    if 'azimuth_deg' in df.columns:
        az_values = df['azimuth_deg']
        if pd.api.types.is_numeric_dtype(az_values):
            # Handle special case of comma-separated values (like "123,255")
            if az_values.dtype == object:
                logger.warning("Azimuth contains non-numeric values, may be comma-separated")
            elif not ((0 <= az_values.min()) and (az_values.max() <= 360)):
                # Normalize to 0-360 range
                df['azimuth_deg'] = az_values % 360
                logger.info("Normalized azimuth values to 0-360 range")
    
    # Beamwidth validation (if present)
    if 'beamwidth_deg' in df.columns:
        bw_values = df['beamwidth_deg']
        if pd.api.types.is_numeric_dtype(bw_values):
            if (bw_values <= 0).any():
                errors.append(f"Found {(bw_values <= 0).sum()} non-positive beamwidth values")
            elif (bw_values > 360).any():
                errors.append(f"Found {(bw_values > 360).sum()} beamwidth values > 360°")
    
    # Check for duplicate station IDs
    if 'station_id' in df.columns:
        duplicates = df['station_id'].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate station IDs")
    
    # Check for sufficient data
    if len(df) == 0:
        errors.append("DataFrame is empty")
    elif len(df) == 1:
        logger.warning("Only one station in dataset")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def prepare_dataframe(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Complete pipeline to prepare DataFrame for optimization.
    
    Args:
        df: Raw input DataFrame
        strict: If True, enforce strict validation
        
    Returns:
        Normalized and validated DataFrame ready for optimization
        
    Raises:
        SchemaError: If validation fails
    """
    # Step 1: Normalize column names
    df = normalize_columns(df, strict=strict)
    
    # Step 2: Add default columns
    df = add_default_columns(df)
    
    # Step 3: Validate
    is_valid, errors = validate_input(df)
    
    if not is_valid:
        error_msg = "Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise SchemaError(error_msg)
    
    # Step 4: Final adjustments
    # Ensure station_id is string type
    if 'station_id' in df.columns:
        df['station_id'] = df['station_id'].astype(str)
    
    # Handle multi-valued azimuth (e.g., "123,255")
    if 'azimuth_deg' in df.columns and df['azimuth_deg'].dtype == object:
        # Take first value if comma-separated
        df['azimuth_deg'] = df['azimuth_deg'].apply(
            lambda x: float(str(x).split(',')[0]) if isinstance(x, str) else float(x)
        )
    
    logger.info(f"Prepared DataFrame with {len(df)} stations and {len(df.columns)} columns")
    
    return df


def get_schema_info(df: pd.DataFrame) -> Dict:
    """
    Get information about the DataFrame schema.
    
    Returns:
        Dictionary with schema information
    """
    normalized_df = normalize_columns(df.copy())
    
    info = {
        'total_columns': len(df.columns),
        'recognized_columns': [],
        'unrecognized_columns': [],
        'missing_required': [],
        'missing_optional': []
    }
    
    # Check recognized columns
    for col in normalized_df.columns:
        if col in COLUMN_SYNONYMS:
            info['recognized_columns'].append(col)
        else:
            info['unrecognized_columns'].append(col)
    
    # Check missing columns
    for col in REQUIRED_COLUMNS:
        if col not in normalized_df.columns:
            info['missing_required'].append(col)
    
    for col in OPTIONAL_DEFAULTS:
        if col not in normalized_df.columns:
            info['missing_optional'].append(col)
    
    return info