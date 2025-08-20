#!/usr/bin/env python3
"""
Add beamwidth column to FM data based on azimuth values.
Stations with azimuth=0 are assumed omnidirectional (360°).
Stations with non-zero azimuth are assumed directional (120° default).
"""

import pandas as pd
import sys
from pathlib import Path

def add_beamwidth_column(input_file: str, output_file: str = None, 
                         directional_beamwidth: float = 120.0):
    """
    Add beamwidth_deg column based on azimuth values.
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path (defaults to input_with_beamwidth.csv)
        directional_beamwidth: Beamwidth for directional stations (default 120°)
    """
    # Read data
    df = pd.read_csv(input_file)
    
    # Add beamwidth column
    # If azimuth is 0, assume omnidirectional (360°)
    # Otherwise, assume directional with specified beamwidth
    df['beamwidth_deg'] = df['azimuth_deg'].apply(
        lambda az: 360.0 if az == 0.0 else directional_beamwidth
    )
    
    # Output file name
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_with_beamwidth.csv"
    
    # Save
    df.to_csv(output_file, index=False)
    
    # Print summary
    omni_count = (df['beamwidth_deg'] == 360.0).sum()
    dir_count = (df['beamwidth_deg'] != 360.0).sum()
    
    print(f"Added beamwidth column to {len(df)} stations:")
    print(f"  - Omnidirectional (360°): {omni_count}")
    print(f"  - Directional ({directional_beamwidth}°): {dir_count}")
    print(f"Saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_beamwidth.py <input_csv> [output_csv] [beamwidth_deg]")
        print("Example: python add_beamwidth.py data/fm_200_subset.csv")
        print("         python add_beamwidth.py data/fm_200_subset.csv output.csv 60")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    beamwidth = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    
    add_beamwidth_column(input_file, output_file, beamwidth)