#!/usr/bin/env python3
"""
Chunk large zipcodes to ensure no zipcode has more than 100 stations.
Creates synthetic zipcode suffixes for chunks (e.g., 12345-A, 12345-B, etc.)
"""

import pandas as pd
import numpy as np
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def chunk_large_zipcodes(input_file: Path, max_stations: int = 100, 
                         output_format: str = 'both') -> None:
    """
    Chunk zipcodes that have more than max_stations stations.
    
    Args:
        input_file: Path to input CSV or Parquet file with zipcode column
        max_stations: Maximum stations per zipcode chunk (default: 100)
        output_format: Output format ('csv', 'parquet', or 'both')
    """
    # Load data
    logger.info(f"Loading data from {input_file}")
    
    if input_file.suffix.lower() == '.parquet':
        df = pd.read_parquet(input_file)
    elif input_file.suffix.lower() in ['.csv', '.txt']:
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    logger.info(f"Loaded {len(df)} stations")
    
    # Check for zipcode column
    if 'zipcode' not in df.columns:
        raise ValueError("Dataset must have 'zipcode' column. Run add_zipcodes first.")
    
    # Save original zipcode for reference
    df['original_zipcode'] = df['zipcode'].copy()
    
    # Count stations per zipcode
    zipcode_counts = df['zipcode'].value_counts()
    large_zipcodes = zipcode_counts[zipcode_counts > max_stations]
    
    if len(large_zipcodes) == 0:
        logger.info(f"No zipcodes have more than {max_stations} stations. No chunking needed.")
        # Still save with the chunked suffix for consistency
        output_stem = input_file.stem.replace('_with_zipcodes', '') + '_chunked'
        output_dir = input_file.parent
        
        if output_format in ['parquet', 'both']:
            output_path = output_dir / f"{output_stem}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved Parquet file: {output_path}")
        
        if output_format in ['csv', 'both']:
            output_path = output_dir / f"{output_stem}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved CSV file: {output_path}")
        return
    
    logger.info(f"Found {len(large_zipcodes)} zipcodes with more than {max_stations} stations")
    
    # Create new zipcode column with chunks
    new_zipcodes = []
    total_chunks_created = 0
    
    # Process each row
    for idx, row in df.iterrows():
        zipcode = row['zipcode']
        
        if zipcode in large_zipcodes.index:
            # This zipcode needs chunking
            # Get all stations with this zipcode
            zipcode_mask = df['zipcode'] == zipcode
            zipcode_df = df[zipcode_mask]
            
            # Assign chunk letters based on position
            # Find position of this station within its zipcode group
            zipcode_positions = np.where(zipcode_mask)[0]
            current_position = np.where(zipcode_positions == idx)[0][0]
            
            # Calculate chunk number (0-based)
            chunk_num = current_position // max_stations
            
            # Convert to letter suffix (A, B, C, ...)
            # If more than 26 chunks needed, use AA, AB, etc.
            if chunk_num < 26:
                suffix = chr(65 + chunk_num)  # A-Z
            else:
                # For more than 26 chunks, use AA, AB, AC, etc.
                first_letter = chr(65 + (chunk_num // 26) - 1)
                second_letter = chr(65 + (chunk_num % 26))
                suffix = first_letter + second_letter
            
            new_zipcode = f"{zipcode}-{suffix}"
            new_zipcodes.append(new_zipcode)
            
            # Track if this is a new chunk
            if current_position % max_stations == 0:
                total_chunks_created += 1
        else:
            # This zipcode doesn't need chunking
            new_zipcodes.append(zipcode)
    
    # Update the zipcode column
    df['zipcode'] = new_zipcodes
    
    # Generate output filename
    output_stem = input_file.stem.replace('_with_zipcodes', '') + '_chunked'
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
    new_zipcode_counts = df['zipcode'].value_counts()
    max_after = new_zipcode_counts.max()
    
    print("\n" + "="*60)
    print("ZIPCODE CHUNKING STATISTICS")
    print("="*60)
    print(f"Total stations: {len(df):,}")
    print(f"Original unique zipcodes: {len(zipcode_counts):,}")
    print(f"Zipcodes needing chunking: {len(large_zipcodes):,}")
    print(f"Total chunks created: {total_chunks_created:,}")
    print(f"New unique zipcode chunks: {len(new_zipcode_counts):,}")
    print(f"Max stations per original zipcode: {zipcode_counts.max():,}")
    print(f"Max stations per chunk: {max_after:,}")
    
    if len(large_zipcodes) > 0:
        print("\nLarge zipcodes that were chunked:")
        for zipcode, count in large_zipcodes.head(10).items():
            chunks_needed = (count + max_stations - 1) // max_stations  # Ceiling division
            chunk_codes = df[df['original_zipcode'] == zipcode]['zipcode'].unique()
            print(f"  {zipcode}: {count:,} stations → {chunks_needed} chunks {sorted(chunk_codes)[:5]}")
            if len(chunk_codes) > 5:
                print(f"    ... and {len(chunk_codes) - 5} more chunks")
    
    # Verify no chunk exceeds max_stations
    if max_after > max_stations:
        logger.warning(f"WARNING: Some chunks still exceed {max_stations} stations!")
        over_limit = new_zipcode_counts[new_zipcode_counts > max_stations]
        print(f"\nChunks exceeding limit:")
        for chunk, count in over_limit.items():
            print(f"  {chunk}: {count} stations")
    else:
        print(f"\n✅ SUCCESS: All zipcode chunks have ≤ {max_stations} stations")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Chunk large zipcodes to limit stations per zipcode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool chunks zipcodes that have too many stations, creating synthetic
zipcode suffixes (e.g., 12345-A, 12345-B) to ensure better optimization performance.

Examples:
  %(prog)s data/stations_with_zipcodes.parquet
  %(prog)s data/stations_with_zipcodes.parquet --max-stations 50
  %(prog)s data/stations.csv --output parquet
        """
    )
    
    parser.add_argument('input_file', type=Path, help='Input CSV or Parquet file with zipcodes')
    parser.add_argument('--max-stations', type=int, default=100,
                        help='Maximum stations per zipcode chunk (default: 100)')
    parser.add_argument('--output', choices=['csv', 'parquet', 'both'], default='both',
                        help='Output format (default: both)')
    
    args = parser.parse_args()
    
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    if args.max_stations < 1:
        logger.error("max-stations must be at least 1")
        sys.exit(1)
    
    try:
        chunk_large_zipcodes(
            args.input_file,
            max_stations=args.max_stations,
            output_format=args.output
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()