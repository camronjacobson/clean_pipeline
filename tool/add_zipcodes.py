#!/usr/bin/env python3
"""
Add zipcode column to dataset using reverse geocoding with geopy/Nominatim.
"""

import pandas as pd
import numpy as np
import time
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from tqdm import tqdm

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
except ImportError:
    print("Error: geopy not installed. Install with: pip install geopy")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ZipcodeGeocoder:
    """Handles zipcode geocoding with caching and rate limiting."""
    
    def __init__(self, delay: float = 1.0, user_agent: str = "spectrum_optimizer_geocoder"):
        """
        Initialize geocoder with Nominatim.
        
        Args:
            delay: Delay between requests in seconds (respect Nominatim's rate limits)
            user_agent: User agent string for Nominatim
        """
        self.delay = delay
        self.cache: Dict[Tuple[float, float], str] = {}
        
        # Initialize Nominatim geocoder
        self.geolocator = Nominatim(user_agent=user_agent)
        logger.info(f"Initialized Nominatim geocoder with {delay}s delay between requests")
    
    def get_zipcode(self, lat: float, lon: float) -> str:
        """
        Get zipcode for coordinates using Nominatim reverse geocoding.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Zipcode string or '00000' if not found
        """
        # Check cache first (round to 5 decimal places for cache key)
        cache_key = (round(lat, 5), round(lon, 5))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Validate coordinates
        if pd.isna(lat) or pd.isna(lon):
            zipcode = '00000'
            self.cache[cache_key] = zipcode
            return zipcode
        
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logger.debug(f"Invalid coordinates: ({lat}, {lon})")
            zipcode = '00000'
            self.cache[cache_key] = zipcode
            return zipcode
        
        # Respect rate limits
        time.sleep(self.delay)
        
        try:
            # Perform reverse geocoding
            location = self.geolocator.reverse(f"{lat}, {lon}", exactly_one=True, timeout=10)
            
            if location and location.raw and 'address' in location.raw:
                address = location.raw['address']
                
                # Try different keys for postal code
                for key in ['postcode', 'postal_code', 'zipcode']:
                    if key in address:
                        zipcode = str(address[key])
                        # Extract just the 5-digit zipcode if it contains extra info
                        if '-' in zipcode:
                            zipcode = zipcode.split('-')[0]
                        if len(zipcode) > 5:
                            zipcode = zipcode[:5]
                        self.cache[cache_key] = zipcode
                        return zipcode
                
                # If no postal code found, log the location for debugging
                logger.debug(f"No postal code found for ({lat}, {lon}): {address.get('country', 'Unknown country')}")
                
        except GeocoderTimedOut:
            logger.warning(f"Timeout for coordinates ({lat}, {lon})")
        except GeocoderServiceError as e:
            logger.warning(f"Service error for ({lat}, {lon}): {e}")
        except Exception as e:
            logger.error(f"Unexpected error for ({lat}, {lon}): {e}")
        
        # Default to '00000' if no zipcode found
        zipcode = '00000'
        self.cache[cache_key] = zipcode
        return zipcode


def add_zipcodes(input_file: Path, output_format: str = 'both', 
                 delay: float = 1.0, max_rows: Optional[int] = None) -> None:
    """
    Add zipcode column to dataset using reverse geocoding.
    
    Args:
        input_file: Path to input CSV or Parquet file
        output_format: Output format ('csv', 'parquet', or 'both')
        delay: Delay between geocoding requests
        max_rows: Maximum number of rows to process (for testing)
    """
    # Load data
    logger.info(f"Loading data from {input_file}")
    
    if input_file.suffix.lower() == '.parquet':
        df = pd.read_parquet(input_file)
    elif input_file.suffix.lower() in ['.csv', '.txt']:
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    # Limit rows if specified
    if max_rows and len(df) > max_rows:
        logger.info(f"Limiting to {max_rows} rows (from {len(df)})")
        df = df.head(max_rows)
    
    logger.info(f"Processing {len(df)} rows")
    
    # Check for required columns
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError("Dataset must have 'latitude' and 'longitude' columns")
    
    # Skip if zipcode already exists
    if 'zipcode' in df.columns:
        logger.warning("Dataset already has 'zipcode' column. Overwriting...")
    
    # Initialize geocoder
    geocoder = ZipcodeGeocoder(delay=delay)
    
    # Get unique coordinates to minimize API calls
    unique_coords = df[['latitude', 'longitude']].drop_duplicates()
    logger.info(f"Processing {len(unique_coords)} unique coordinate pairs")
    
    # Estimate time
    estimated_time = len(unique_coords) * delay
    logger.info(f"Estimated time: {estimated_time/60:.1f} minutes (at {delay}s per request)")
    
    # Process coordinates with progress bar
    zipcode_map = {}
    found_count = 0
    missing_count = 0
    
    with tqdm(total=len(unique_coords), desc="Geocoding") as pbar:
        for idx, row in unique_coords.iterrows():
            lat, lon = row['latitude'], row['longitude']
            zipcode = geocoder.get_zipcode(lat, lon)
            
            zipcode_map[(lat, lon)] = zipcode
            
            if zipcode != '00000':
                found_count += 1
            else:
                missing_count += 1
            
            pbar.update(1)
            pbar.set_postfix({'found': found_count, 'missing': missing_count, 'cached': len(geocoder.cache)})
    
    # Add zipcode column to dataframe
    logger.info("Adding zipcode column to dataset")
    df['zipcode'] = df.apply(lambda row: zipcode_map.get((row['latitude'], row['longitude']), '00000'), axis=1)
    
    # Generate output filename
    output_stem = input_file.stem + '_with_zipcodes'
    output_dir = input_file.parent
    
    # Save output files
    if output_format in ['parquet', 'both']:
        output_path = output_dir / f"{output_stem}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved Parquet file: {output_path}")
    
    if output_format in ['csv', 'both']:
        output_path = output_dir / f"{output_stem}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved CSV file: {output_path}")
    
    # Print statistics
    total_rows = len(df)
    unique_zipcodes = df['zipcode'].nunique()
    valid_zipcodes = len(df[df['zipcode'] != '00000'])
    
    # Top zipcodes
    zipcode_counts = df[df['zipcode'] != '00000']['zipcode'].value_counts().head(10)
    
    print("\n" + "="*50)
    print("ZIPCODE GEOCODING STATISTICS")
    print("="*50)
    print(f"Total rows: {total_rows:,}")
    print(f"Unique coordinates: {len(unique_coords):,}")
    print(f"Zipcodes found: {found_count:,} ({100*found_count/len(unique_coords):.1f}%)")
    print(f"Zipcodes missing: {missing_count:,} ({100*missing_count/len(unique_coords):.1f}%)")
    print(f"Unique zipcodes: {unique_zipcodes:,}")
    print(f"Valid zipcode rows: {valid_zipcodes:,} ({100*valid_zipcodes/total_rows:.1f}%)")
    print(f"Cache entries: {len(geocoder.cache):,}")
    
    if len(zipcode_counts) > 0:
        print("\nTop 10 zipcodes:")
        for zipcode, count in zipcode_counts.items():
            print(f"  {zipcode}: {count:,} stations")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Add zipcode column to dataset using reverse geocoding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool uses the Nominatim geocoding service to find zipcodes from coordinates.
Note: Nominatim has rate limits (1 request per second), so processing large datasets may take time.

Examples:
  %(prog)s data/stations.csv
  %(prog)s data/stations.parquet --output parquet
  %(prog)s data/stations.csv --delay 1.5 --max-rows 100  # Test with 100 rows
        """
    )
    
    parser.add_argument('input_file', type=Path, help='Input CSV or Parquet file')
    parser.add_argument('--output', choices=['csv', 'parquet', 'both'], default='both',
                        help='Output format (default: both)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests in seconds (default: 1.0, respect Nominatim limits)')
    parser.add_argument('--max-rows', type=int, metavar='N',
                        help='Process only first N rows (for testing)')
    
    args = parser.parse_args()
    
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    if args.delay < 1.0:
        logger.warning("Delay less than 1.0s may violate Nominatim's usage policy")
    
    try:
        add_zipcodes(
            args.input_file,
            output_format=args.output,
            delay=args.delay,
            max_rows=args.max_rows
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()