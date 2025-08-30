#!/usr/bin/env python3
"""
Diagnostic script to identify and fix catastrophic optimizer failures.
Analyzes interference density problems and tests solutions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tool.neighbors import create_neighbor_discovery
from src.spectrum_optimizer_enhanced import EnhancedSpectrumOptimizer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_interference_problem(data_file='data/test_subset_200.csv'):
    """Diagnose why interference graphs are too dense"""
    print('\n' + '='*80)
    print('INTERFERENCE DENSITY DIAGNOSIS')
    print('='*80)
    
    # Load data
    if Path(data_file).exists():
        df = pd.read_csv(data_file)
    else:
        print(f"Warning: {data_file} not found, using smaller test set")
        data_file = 'data/test_subset_50_with_zipcodes.parquet'
        df = pd.read_parquet(data_file)
    
    print(f'\nAnalyzing: {data_file}')
    print(f'Total stations: {len(df)}')
    
    # Check geographic spread
    lat_range = df['latitude'].max() - df['latitude'].min()
    lon_range = df['longitude'].max() - df['longitude'].min()
    print(f'Geographic spread: {lat_range:.2f}° lat x {lon_range:.2f}° lon')
    
    # Calculate actual distances between all pairs (sample for performance)
    sample_size = min(len(df), 50)
    df_sample = df.head(sample_size)
    distances = []
    
    for i in range(len(df_sample)):
        for j in range(i+1, len(df_sample)):
            lat1, lon1 = df_sample.iloc[i][['latitude', 'longitude']]
            lat2, lon2 = df_sample.iloc[j][['latitude', 'longitude']]
            # Approximate distance in km (simple formula)
            dist = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111
            distances.append(dist)
    
    if distances:
        print(f'\nDistance statistics for {sample_size} stations (km):')
        print(f'  Min: {np.min(distances):.1f}')
        print(f'  Max: {np.max(distances):.1f}')
        print(f'  Mean: {np.mean(distances):.1f}')
        print(f'  Median: {np.median(distances):.1f}')
        print(f'  25th percentile: {np.percentile(distances, 25):.1f}')
        print(f'  75th percentile: {np.percentile(distances, 75):.1f}')
    
    # Test different interference radii
    print('\n' + '-'*60)
    print('Testing different interference radii (100 stations):')
    print('-'*60)
    
    test_df = df.head(100) if len(df) >= 100 else df
    
    for radius in [10, 20, 30, 40, 50, 60, 100]:
        config = {
            'geometry': {
                'r_main_km': radius,
                'r_off_km': radius/3,
                'az_tolerance_deg': 5.0
            }
        }
        
        discovery = create_neighbor_discovery(config)
        edges = discovery.build_interference_graph(test_df)
        avg_neighbors = len(edges) * 2 / len(test_df) if len(test_df) > 0 else 0
        
        # Estimate minimum colors needed (clique number approximation)
        min_colors = int(avg_neighbors) + 1
        feasible = "✓ FEASIBLE" if min_colors <= 100 else "✗ INFEASIBLE"
        
        print(f'  Radius {radius:3d}km: {len(edges):4d} edges, '
              f'avg {avg_neighbors:5.1f} neighbors/station, '
              f'min freqs ~{min_colors:3d} {feasible}')
    
    return distances

def test_with_reasonable_radius(data_file='data/test_subset_200.csv'):
    """Test optimizer with more reasonable interference radius"""
    print('\n' + '='*80)
    print('TESTING WITH ADJUSTED PARAMETERS')
    print('='*80)
    
    # Load data
    if Path(data_file).exists():
        df = pd.read_csv(data_file)
    else:
        data_file = 'data/test_subset_50_with_zipcodes.parquet'
        df = pd.read_parquet(data_file)
    
    # Create custom config with smaller radius
    config = {
        'band': {
            'min_mhz': 88.0,
            'max_mhz': 108.0,
            'step_khz': 200
        },
        'geometry': {
            'r_main_km': 20,  # Reduced from 60
            'r_off_km': 8,    # Reduced from 15
            'az_tolerance_deg': 5.0
        },
        'interference': {
            'guard_offsets': [-1, 1]
        },
        'solver': {
            'timeout_seconds': 30,
            'num_workers': 4
        },
        'weights': {
            'w_span': 100,
            'w_count': 10,
            'w_surplus': 1
        }
    }
    
    # Save test config
    Path('config/profiles').mkdir(parents=True, exist_ok=True)
    with open('config/profiles/test_reduced.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f'\nCreated test profile: config/profiles/test_reduced.yaml')
    print(f'  Main interference radius: {config["geometry"]["r_main_km"]}km')
    print(f'  Off-axis radius: {config["geometry"]["r_off_km"]}km')
    
    # Run optimizer with reduced radius
    print(f'\nRunning optimizer on {min(50, len(df))} stations...')
    optimizer = EnhancedSpectrumOptimizer('test_reduced')
    result = optimizer.optimize(df.head(50))
    
    # Check results
    if result is not None and 'assigned_frequency' in result.columns:
        unique_freqs = result['assigned_frequency'].nunique()
        efficiency = len(result)/unique_freqs if unique_freqs > 0 else 0
        
        print(f'\nResults with {config["geometry"]["r_main_km"]}km radius:')
        print(f'  Stations processed: {len(result)}')
        print(f'  Unique frequencies used: {unique_freqs}')
        print(f'  Efficiency: {efficiency:.2f} stations/freq')
        
        # Check frequency distribution
        freq_counts = result['assigned_frequency'].value_counts()
        print(f'  Max stations on single freq: {freq_counts.max()}')
        print(f'  Min stations on single freq: {freq_counts.min()}')
        
        # Verify no co-channel violations
        violations = check_interference_violations(
            result, 
            config['geometry']['r_main_km']
        )
        print(f'  Interference violations: {violations}')
        
        return result
    else:
        print("  ERROR: Optimizer returned no valid result")
        return None

def check_interference_violations(df, radius_km):
    """Check for co-channel interference violations"""
    if 'assigned_frequency' not in df.columns:
        return -1
    
    violations = 0
    violation_details = []
    
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            lat1, lon1 = df.iloc[i][['latitude', 'longitude']]
            lat2, lon2 = df.iloc[j][['latitude', 'longitude']]
            dist = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111
            
            if dist <= radius_km:
                freq1 = df.iloc[i]['assigned_frequency']
                freq2 = df.iloc[j]['assigned_frequency']
                
                if freq1 == freq2:
                    violations += 1
                    if len(violation_details) < 5:  # Show first few violations
                        violation_details.append(
                            f'    Violation {violations}: stations {i},{j} '
                            f'at {dist:.1f}km both on {freq1} MHz'
                        )
    
    if violation_details:
        print('\n  Sample violations:')
        for detail in violation_details:
            print(detail)
    
    return violations

def analyze_current_config():
    """Analyze the current FM configuration"""
    print('\n' + '='*80)
    print('CURRENT FM CONFIGURATION ANALYSIS')
    print('='*80)
    
    config_path = 'config/profiles/fm.yaml'
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f'\nCurrent FM profile settings:')
        if 'geometry' in config:
            geom = config['geometry']
            print(f'  Main interference radius: {geom.get("r_main_km", "N/A")}km')
            print(f'  Off-axis radius: {geom.get("r_off_km", "N/A")}km')
            print(f'  Azimuth tolerance: {geom.get("az_tolerance_deg", "N/A")}°')
            
            if geom.get('r_main_km', 0) > 40:
                print('\n  ⚠️  WARNING: Main radius > 40km will create very dense graphs!')
                print('     Recommended: 15-25km for FM band')
        
        if 'solver' in config:
            solver = config['solver']
            print(f'\n  Solver timeout: {solver.get("timeout_seconds", "N/A")}s')
            print(f'  Worker threads: {solver.get("num_workers", "N/A")}')
    else:
        print(f"  Config file not found: {config_path}")

def suggest_fixes():
    """Suggest specific fixes for the problems"""
    print('\n' + '='*80)
    print('RECOMMENDED FIXES')
    print('='*80)
    
    print('\n1. UPDATE config/profiles/fm.yaml:')
    print('   Change interference radius from 60km to 20km:')
    print('   ```yaml')
    print('   geometry:')
    print('     r_main_km: 20      # was 60')
    print('     r_off_km: 8        # was 15')
    print('     az_tolerance_deg: 5.0')
    print('   ```')
    
    print('\n2. FIX FALLBACK in src/spectrum_optimizer_enhanced.py:')
    print('   Find line: result["assigned_frequency"] = frequencies[0]')
    print('   Replace with:')
    print('   ```python')
    print('   # Use round-robin assignment as emergency fallback')
    print('   for idx in range(len(result)):')
    print('       result.iloc[idx, result.columns.get_loc("assigned_frequency")] = \\')
    print('           frequencies[idx % len(frequencies)]')
    print('   logger.warning(f"Used round-robin fallback for {len(result)} stations")')
    print('   ```')
    
    print('\n3. INCREASE SOLVER TIMEOUT for dense problems:')
    print('   In config files, set:')
    print('   ```yaml')
    print('   solver:')
    print('     timeout_seconds: 120  # was 60')
    print('   ```')
    
    print('\n4. CONSIDER GEOGRAPHIC CHUNKING:')
    print('   Instead of processing all stations in a zipcode together,')
    print('   chunk by actual geographic regions to reduce density.')

def main():
    """Run all diagnostics"""
    print('\n' + '='*80)
    print(' FM SPECTRUM OPTIMIZER DIAGNOSTIC TOOL')
    print('='*80)
    
    try:
        # Analyze current configuration
        analyze_current_config()
        
        # Diagnose interference density
        diagnose_interference_problem()
        
        # Test with adjusted parameters
        test_with_reasonable_radius()
        
        # Provide fix recommendations
        suggest_fixes()
        
        print('\n' + '='*80)
        print(' DIAGNOSTIC COMPLETE')
        print('='*80)
        print('\nRun this script again after applying fixes to verify improvements.')
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())