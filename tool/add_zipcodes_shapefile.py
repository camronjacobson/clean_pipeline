#!/usr/bin/env python3
"""
Add zipcode column to dataset using shapefile-based spatial join.
Much faster than API-based geocoding.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_zipcodes_shapefile(input_file: Path, shapefile_path: Path, output_format: str = 'both',
                           max_rows: Optional[int] = None) -> None:
    """
    Add zipcode column to dataset using shapefile spatial join.
    
    Args:
        input_file: Path to input CSV or Parquet file
        shapefile_path: Path to ZCTA shapefile
        output_format: Output format ('csv', 'parquet', or 'both')
        max_rows: Maximum number of rows to process (for testing)
    """
    start_time = time.time()
    
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
    
    # Load ZCTA shapefile
    logger.info(f"Loading ZCTA shapefile from {shapefile_path}")
    zcta_gdf = gpd.read_file(shapefile_path)
    
    # ZCTA shapefiles typically have ZCTA5CE20 or ZCTA5CE10 column for zipcode
    zipcode_col = None
    for col in ['ZCTA5CE20', 'ZCTA5CE10', 'ZCTA5CE', 'GEOID10', 'GEOID20']:
        if col in zcta_gdf.columns:
            zipcode_col = col
            break
    
    if not zipcode_col:
        logger.error(f"Could not find zipcode column in shapefile. Available columns: {zcta_gdf.columns.tolist()}")
        raise ValueError("Could not find zipcode column in shapefile")
    
    logger.info(f"Using column '{zipcode_col}' for zipcodes")
    
    # Convert dataframe to GeoDataFrame
    logger.info("Creating geometry from coordinates")
    geometry = [Point(lon, lat) if pd.notna(lat) and pd.notna(lon) else None 
                for lat, lon in zip(df['latitude'], df['longitude'])]
    
    geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Ensure CRS match
    if zcta_gdf.crs != geo_df.crs:
        logger.info(f"Reprojecting from {geo_df.crs} to {zcta_gdf.crs}")
        geo_df = geo_df.to_crs(zcta_gdf.crs)
    
    # Perform spatial join
    logger.info("Performing spatial join to find zipcodes...")
    joined = gpd.sjoin(geo_df, zcta_gdf[[zipcode_col, 'geometry']], 
                       how='left', predicate='within')
    
    # Add zipcode column
    df['zipcode'] = joined[zipcode_col].fillna('00000')
    
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
    elapsed_time = time.time() - start_time
    total_rows = len(df)
    valid_zipcodes = len(df[df['zipcode'] != '00000'])
    unique_zipcodes = df['zipcode'].nunique()
    
    # Top zipcodes
    zipcode_counts = df[df['zipcode'] != '00000']['zipcode'].value_counts().head(10)
    
    print("\n" + "="*50)
    print("ZIPCODE SHAPEFILE JOIN STATISTICS")
    print("="*50)
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Total rows: {total_rows:,}")
    print(f"Valid zipcode rows: {valid_zipcodes:,} ({100*valid_zipcodes/total_rows:.1f}%)")
    print(f"Missing zipcode rows: {total_rows - valid_zipcodes:,} ({100*(total_rows - valid_zipcodes)/total_rows:.1f}%)")
    print(f"Unique zipcodes: {unique_zipcodes:,}")
    
    if len(zipcode_counts) > 0:
        print("\nTop 10 zipcodes:")
        for zipcode, count in zipcode_counts.items():
            print(f"  {zipcode}: {count:,} stations")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Add zipcode column to dataset using shapefile spatial join',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool uses shapefile-based spatial join to find zipcodes from coordinates.
This is much faster than API-based geocoding (seconds vs minutes).

Examples:
  %(prog)s data/stations.csv
  %(prog)s data/stations.parquet --output parquet
  %(prog)s data/stations.csv --shapefile shapefiles/custom_zcta.shp
        """
    )
    
    parser.add_argument('input_file', type=Path, help='Input CSV or Parquet file')
    parser.add_argument('--shapefile', type=Path, 
                        default=Path('shapefiles/tl_2020_us_zcta520.shp'),
                        help='Path to ZCTA shapefile (default: shapefiles/tl_2020_us_zcta520.shp)')
    parser.add_argument('--output', choices=['csv', 'parquet', 'both'], default='both',
                        help='Output format (default: both)')
    parser.add_argument('--max-rows', type=int, metavar='N',
                        help='Process only first N rows (for testing)')
    
    args = parser.parse_args()
    
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    if not args.shapefile.exists():
        logger.error(f"Shapefile not found: {args.shapefile}")
        sys.exit(1)
    
    try:
        add_zipcodes_shapefile(
            args.input_file,
            args.shapefile,
            output_format=args.output,
            max_rows=args.max_rows
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()