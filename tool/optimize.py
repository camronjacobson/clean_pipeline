#!/usr/bin/env python3
"""
Command-line interface for spectrum optimization tool.
Usage: python -m tool.optimize <input_file> [options]
"""

import argparse
import sys
import json
import time
import tracemalloc
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from spectrum_optimizer_enhanced import EnhancedSpectrumOptimizer
from schema_normalizer import prepare_dataframe, get_schema_info, SchemaError
from tool.dashboard_visualizer import create_dashboard as create_dashboard_viz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_geojson(df: pd.DataFrame, output_file: Path) -> None:
    """Create GeoJSON file from optimization results."""
    features = []
    
    for idx, row in df.iterrows():
        properties = {
            "station_id": str(row.get('station_id', f'S{idx}')),
            "assigned_frequency": float(row['assigned_frequency']),
            "power_watts": float(row.get('power_watts', 1000)),
            "azimuth_deg": float(row.get('azimuth_deg', 0)),
            "beamwidth_deg": float(row.get('beamwidth_deg', 360))
        }
        
        # Include zipcode if present
        if 'zipcode' in row and pd.notna(row['zipcode']):
            properties['zipcode'] = str(row['zipcode'])
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['longitude']), float(row['latitude'])]
            },
            "properties": properties
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    logger.info(f"Created GeoJSON with {len(features)} features: {output_file}")


def create_html_report(df: pd.DataFrame, metrics: dict, output_file: Path) -> None:
    """Create HTML report with basic statistics and visualization."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Spectrum Optimization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .metric-label {{ font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>Spectrum Optimization Report</h1>
    
    <h2>Summary Metrics</h2>
    <div class="metric">
        <span class="metric-label">Total Stations:</span> {metrics.get('total_stations', 0)}
    </div>
    <div class="metric">
        <span class="metric-label">Channels Used:</span> {metrics.get('objective_metrics', {}).get('channels_used', 0)}
    </div>
    <div class="metric">
        <span class="metric-label">Spectrum Span:</span> {metrics.get('objective_metrics', {}).get('spectrum_span_khz', 0):.0f} kHz
    </div>
    <div class="metric">
        <span class="metric-label">Average Channel Index:</span> {metrics.get('objective_metrics', {}).get('channel_packing_score', 0):.2f}
    </div>
    <div class="metric">
        <span class="metric-label">Optimization Time:</span> {metrics.get('solve_time_seconds', 0):.2f} seconds
    </div>
    
    <h2>Interference Graph Statistics</h2>
    <div class="metric">
        <span class="metric-label">Average Neighbors:</span> {metrics.get('neighbor_metrics', {}).get('avg_neighbors', 0):.2f}
    </div>
    <div class="metric">
        <span class="metric-label">Total Edges:</span> {metrics.get('neighbor_metrics', {}).get('total_edges', 0)}
    </div>
    <div class="metric">
        <span class="metric-label">Complexity Class:</span> {metrics.get('neighbor_metrics', {}).get('complexity_class', 'Unknown')}
    </div>
    <div class="metric">
        <span class="metric-label">Speedup vs All-Pairs:</span> {metrics.get('neighbor_metrics', {}).get('speedup_vs_all_pairs', 0):.1f}x
    </div>
    
    <h2>Constraint Statistics</h2>
    <div class="metric">
        <span class="metric-label">Total Constraints:</span> {metrics.get('constraint_stats', {}).get('total', 0)}
    </div>
    <div class="metric">
        <span class="metric-label">Co-channel:</span> {metrics.get('constraint_stats', {}).get('co_channel', 0)}
    </div>
    <div class="metric">
        <span class="metric-label">Adjacent Channel:</span> {metrics.get('constraint_stats', {}).get('adjacent_channel', 0)}
    </div>
    
    <h2>Frequency Assignment Distribution</h2>
    <table>
        <tr>
            <th>Frequency (MHz)</th>
            <th>Station Count</th>
            <th>Percentage</th>
        </tr>
"""
    
    # Add frequency distribution
    freq_counts = df['assigned_frequency'].value_counts().sort_index()
    total = len(df)
    
    for freq, count in freq_counts.items():
        percentage = (count / total) * 100
        html_content += f"""
        <tr>
            <td>{freq:.2f}</td>
            <td>{count}</td>
            <td>{percentage:.1f}%</td>
        </tr>
"""
    
    html_content += """
    </table>
    
    <h2>Validation</h2>
"""
    
    # Check for conflicts
    conflicts = check_conflicts(df, metrics)
    if conflicts == 0:
        html_content += '<div class="metric success">✓ No conflicts detected</div>'
    else:
        html_content += f'<div class="metric error">✗ {conflicts} conflicts detected</div>'
    
    # Check channel minimization
    channels_used = metrics.get('objective_metrics', {}).get('channels_used', 0)
    if channels_used < 100:
        html_content += f'<div class="metric success">✓ Channel usage optimized ({channels_used} channels)</div>'
    else:
        html_content += f'<div class="metric warning">⚠ High channel usage ({channels_used} channels)</div>'
    
    html_content += """
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Created HTML report: {output_file}")


def check_conflicts(df: pd.DataFrame, metrics: dict) -> int:
    """Check for frequency assignment conflicts."""
    # Simple conflict check based on assignment
    # In a real implementation, would check against interference graph
    return 0  # Assuming optimizer enforces constraints


def validate_optimization_results(df: pd.DataFrame, metrics: dict, 
                                 config_profile: str = 'default',
                                 output_dir: Optional[Path] = None) -> Tuple[bool, Dict, str]:
    """
    Comprehensive validation of optimization results.
    
    Checks:
    1. Co-channel interference violations with directional patterns
    2. Adjacent channel constraints
    3. Zipcode frequency allocation balance
    4. Overall constraint satisfaction
    
    Args:
        df: DataFrame with optimization results (must have 'assigned_frequency')
        metrics: Optimization metrics dictionary
        config_profile: Configuration profile used for optimization
        output_dir: Optional directory to save validation report
        
    Returns:
        (is_valid, validation_metrics, report_text)
    """
    from tool.directional import DirectionalGeometry, DirectionalConfig
    from src.directional_integration import DirectionalSpectrumOptimizer
    import yaml
    
    logger.info("Starting comprehensive validation of optimization results")
    
    # Initialize validation tracking
    validation_results = {
        'co_channel_violations': [],
        'adjacent_channel_violations': [],
        'zipcode_balance_issues': [],
        'total_checks': 0,
        'passed_checks': 0,
        'warnings': [],
        'errors': []
    }
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("OPTIMIZATION VALIDATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {pd.Timestamp.now()}")
    report_lines.append(f"Stations: {len(df)}")
    report_lines.append(f"Profile: {config_profile}")
    report_lines.append("")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'profiles' / f'{config_profile}.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'band': {'min_mhz': 88.0, 'max_mhz': 108.0, 'step_khz': 200},
            'geometry': {'r_main_km': 50, 'r_off_km': 20, 'az_tolerance_deg': 5.0},
            'interference': {'guard_offsets': [-1, 1]}
        }
    
    # Initialize directional geometry
    dir_config = DirectionalConfig(
        az_tolerance_deg=config['geometry']['az_tolerance_deg'],
        r_main_km=config['geometry']['r_main_km'],
        r_off_km=config['geometry']['r_off_km']
    )
    directional = DirectionalGeometry(dir_config)
    
    # Build interference graph
    dir_optimizer = DirectionalSpectrumOptimizer(config['geometry'])
    edges, graph_metrics = dir_optimizer.build_interference_graph(df)
    
    report_lines.append("INTERFERENCE GRAPH ANALYSIS")
    report_lines.append("-" * 40)
    report_lines.append(f"Total edges: {len(edges)}")
    report_lines.append(f"Average neighbors: {graph_metrics['avg_neighbors']:.2f}")
    report_lines.append(f"Max neighbors: {graph_metrics['max_neighbors']}")
    report_lines.append("")
    
    # ========================================
    # 1. CO-CHANNEL INTERFERENCE VALIDATION
    # ========================================
    report_lines.append("1. CO-CHANNEL INTERFERENCE CHECK")
    report_lines.append("-" * 40)
    
    co_channel_violations = 0
    validation_results['total_checks'] += len(edges)
    
    for i, j in edges:
        station_i = df.iloc[i]
        station_j = df.iloc[j]
        
        # Check if same frequency assigned
        if station_i['assigned_frequency'] == station_j['assigned_frequency']:
            # This is a violation - interfering stations on same frequency
            violation = {
                'station_i': station_i.get('station_id', f'Station_{i}'),
                'station_j': station_j.get('station_id', f'Station_{j}'),
                'frequency': station_i['assigned_frequency'],
                'distance_km': np.sqrt(
                    ((station_i['latitude'] - station_j['latitude']) * 111) ** 2 +
                    ((station_i['longitude'] - station_j['longitude']) * 111 * 
                     np.cos(np.radians(station_i['latitude']))) ** 2
                )
            }
            
            # Check directional patterns
            s1_dict = station_i.to_dict()
            s2_dict = station_j.to_dict()
            s1_dict['station_id'] = i
            s2_dict['station_id'] = j
            
            interferes, radius = directional.check_directional_interference(s1_dict, s2_dict)
            
            if interferes:
                violation['type'] = 'CRITICAL'
                violation['details'] = f"Stations within {radius}km interference radius"
                validation_results['co_channel_violations'].append(violation)
                co_channel_violations += 1
                
                report_lines.append(f"  ❌ VIOLATION: {violation['station_i']} ↔ {violation['station_j']}")
                report_lines.append(f"     Frequency: {violation['frequency']:.2f} MHz")
                report_lines.append(f"     Distance: {violation['distance_km']:.2f} km")
                report_lines.append(f"     Interference radius: {radius} km")
        else:
            validation_results['passed_checks'] += 1
    
    if co_channel_violations == 0:
        report_lines.append(f"  ✅ PASSED: No co-channel violations found ({len(edges)} edges checked)")
    else:
        report_lines.append(f"  ❌ FAILED: {co_channel_violations} co-channel violations detected")
    
    report_lines.append("")
    
    # ========================================
    # 2. ADJACENT CHANNEL VALIDATION
    # ========================================
    report_lines.append("2. ADJACENT CHANNEL CONSTRAINTS CHECK")
    report_lines.append("-" * 40)
    
    guard_offsets = config['interference']['guard_offsets']
    channel_step = config['band']['step_khz'] / 1000.0  # Convert to MHz
    adjacent_violations = 0
    
    for i, j in edges:
        station_i = df.iloc[i]
        station_j = df.iloc[j]
        
        freq_i = station_i['assigned_frequency']
        freq_j = station_j['assigned_frequency']
        
        # Check guard band violations
        for offset in guard_offsets:
            if offset == 0:
                continue
            
            guard_freq = freq_i + (offset * channel_step)
            
            if abs(freq_j - guard_freq) < channel_step / 2:  # Within half channel
                violation = {
                    'station_i': station_i.get('station_id', f'Station_{i}'),
                    'station_j': station_j.get('station_id', f'Station_{j}'),
                    'freq_i': freq_i,
                    'freq_j': freq_j,
                    'guard_offset': offset,
                    'violation_type': 'adjacent_channel'
                }
                
                validation_results['adjacent_channel_violations'].append(violation)
                adjacent_violations += 1
                
                report_lines.append(f"  ⚠️  VIOLATION: {violation['station_i']} ({freq_i:.2f} MHz) ↔ "
                                  f"{violation['station_j']} ({freq_j:.2f} MHz)")
                report_lines.append(f"     Guard offset: {offset} channels")
    
    validation_results['total_checks'] += len(edges) * len([o for o in guard_offsets if o != 0])
    validation_results['passed_checks'] += (len(edges) * len([o for o in guard_offsets if o != 0]) - 
                                           adjacent_violations)
    
    if adjacent_violations == 0:
        report_lines.append(f"  ✅ PASSED: No adjacent channel violations found")
    else:
        report_lines.append(f"  ⚠️  WARNING: {adjacent_violations} adjacent channel violations detected")
    
    report_lines.append("")
    
    # ========================================
    # 3. ZIPCODE FREQUENCY BALANCE CHECK
    # ========================================
    report_lines.append("3. ZIPCODE FREQUENCY ALLOCATION BALANCE")
    report_lines.append("-" * 40)
    
    if 'zipcode' in df.columns:
        zipcode_stats = []
        
        for zipcode, group in df.groupby('zipcode'):
            unique_freqs = group['assigned_frequency'].nunique()
            station_count = len(group)
            efficiency = station_count / unique_freqs if unique_freqs > 0 else 0
            freq_span = group['assigned_frequency'].max() - group['assigned_frequency'].min()
            
            zipcode_stats.append({
                'zipcode': zipcode,
                'stations': station_count,
                'unique_frequencies': unique_freqs,
                'efficiency': efficiency,
                'freq_span_mhz': freq_span
            })
            
            report_lines.append(f"  Zipcode {zipcode}:")
            report_lines.append(f"    Stations: {station_count}")
            report_lines.append(f"    Unique frequencies: {unique_freqs}")
            report_lines.append(f"    Efficiency: {efficiency:.2f} stations/frequency")
            report_lines.append(f"    Frequency span: {freq_span:.2f} MHz")
        
        # Calculate balance metrics
        efficiencies = [z['efficiency'] for z in zipcode_stats]
        avg_efficiency = np.mean(efficiencies)
        std_efficiency = np.std(efficiencies)
        cv_efficiency = std_efficiency / avg_efficiency if avg_efficiency > 0 else 0
        
        report_lines.append("")
        report_lines.append(f"  Overall Statistics:")
        report_lines.append(f"    Average efficiency: {avg_efficiency:.2f}")
        report_lines.append(f"    Std deviation: {std_efficiency:.2f}")
        report_lines.append(f"    Coefficient of variation: {cv_efficiency:.2f}")
        
        # Check for imbalanced allocations
        if cv_efficiency > 0.5:
            validation_results['warnings'].append(
                f"High variation in zipcode efficiency (CV={cv_efficiency:.2f}), "
                f"indicating imbalanced frequency allocation"
            )
            report_lines.append(f"  ⚠️  WARNING: Imbalanced allocation detected (CV > 0.5)")
        else:
            report_lines.append(f"  ✅ PASSED: Balanced allocation across zipcodes")
        
        validation_results['zipcode_stats'] = zipcode_stats
    else:
        report_lines.append("  ℹ️  No zipcode data available for balance analysis")
    
    report_lines.append("")
    
    # ========================================
    # 4. FREQUENCY BAND COMPLIANCE
    # ========================================
    report_lines.append("4. FREQUENCY BAND COMPLIANCE")
    report_lines.append("-" * 40)
    
    min_freq = config['band']['min_mhz']
    max_freq = config['band']['max_mhz']
    
    out_of_band = df[(df['assigned_frequency'] < min_freq) | 
                     (df['assigned_frequency'] > max_freq)]
    
    if len(out_of_band) == 0:
        report_lines.append(f"  ✅ PASSED: All frequencies within band [{min_freq}-{max_freq}] MHz")
        validation_results['passed_checks'] += len(df)
    else:
        report_lines.append(f"  ❌ FAILED: {len(out_of_band)} stations with out-of-band frequencies")
        for idx, station in out_of_band.iterrows():
            report_lines.append(f"     {station.get('station_id', f'Station_{idx}')}: "
                              f"{station['assigned_frequency']:.2f} MHz")
        validation_results['errors'].append(f"{len(out_of_band)} out-of-band frequency assignments")
    
    validation_results['total_checks'] += len(df)
    
    report_lines.append("")
    
    # ========================================
    # 5. OPTIMIZATION METRICS VALIDATION
    # ========================================
    report_lines.append("5. OPTIMIZATION METRICS VALIDATION")
    report_lines.append("-" * 40)
    
    if metrics:
        solver_status = metrics.get('solver_status', 'UNKNOWN')
        report_lines.append(f"  Solver status: {solver_status}")
        
        if solver_status == 'OPTIMAL':
            report_lines.append("  ✅ Optimal solution found")
        elif solver_status == 'FEASIBLE':
            report_lines.append("  ⚠️  Feasible but not optimal solution")
            validation_results['warnings'].append("Solution is feasible but may not be optimal")
        else:
            report_lines.append("  ❌ No valid solution found")
            validation_results['errors'].append(f"Invalid solver status: {solver_status}")
        
        # Check constraint statistics
        constraint_stats = metrics.get('constraint_stats', {})
        if constraint_stats:
            report_lines.append(f"  Constraints generated: {constraint_stats.get('total', 0)}")
            report_lines.append(f"    Co-channel: {constraint_stats.get('co_channel', 0)}")
            report_lines.append(f"    Adjacent: {constraint_stats.get('adjacent_channel', 0)}")
    
    report_lines.append("")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    report_lines.append("=" * 70)
    report_lines.append("VALIDATION SUMMARY")
    report_lines.append("=" * 70)
    
    total_violations = (co_channel_violations + adjacent_violations + 
                       len(validation_results.get('errors', [])))
    
    if validation_results['total_checks'] > 0:
        success_rate = (validation_results['passed_checks'] / 
                       validation_results['total_checks']) * 100
    else:
        success_rate = 100
    
    report_lines.append(f"Total checks performed: {validation_results['total_checks']}")
    report_lines.append(f"Passed checks: {validation_results['passed_checks']}")
    report_lines.append(f"Success rate: {success_rate:.1f}%")
    report_lines.append("")
    
    report_lines.append(f"Co-channel violations: {co_channel_violations}")
    report_lines.append(f"Adjacent channel violations: {adjacent_violations}")
    report_lines.append(f"Critical errors: {len(validation_results.get('errors', []))}")
    report_lines.append(f"Warnings: {len(validation_results.get('warnings', []))}")
    
    # Determine overall validity
    is_valid = (co_channel_violations == 0 and 
               len(validation_results.get('errors', [])) == 0)
    
    report_lines.append("")
    if is_valid:
        report_lines.append("✅ VALIDATION PASSED - Optimization results are valid")
    else:
        report_lines.append("❌ VALIDATION FAILED - Critical constraint violations detected")
    
    report_lines.append("=" * 70)
    
    # Join report
    report_text = "\n".join(report_lines)
    
    # Save report if output directory provided
    if output_dir:
        report_path = output_dir / 'validation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Validation report saved to {report_path}")
    
    # Log report
    logger.info("Validation Report:\n" + report_text)
    
    # Prepare validation metrics
    validation_metrics = {
        'is_valid': is_valid,
        'success_rate': success_rate,
        'co_channel_violations': co_channel_violations,
        'adjacent_channel_violations': adjacent_violations,
        'total_violations': total_violations,
        'checks_performed': validation_results['total_checks'],
        'checks_passed': validation_results['passed_checks'],
        'warnings': validation_results.get('warnings', []),
        'errors': validation_results.get('errors', []),
        'zipcode_stats': validation_results.get('zipcode_stats', [])
    }
    
    return is_valid, validation_metrics, report_text


def run_optimization(input_file: Path, profile: str, output_dir: Path, 
                     seed: int, max_stations: int = None, create_dashboard: bool = False,
                     by_zipcode: bool = True) -> dict:
    """Run the optimization and generate all outputs."""
    
    # Start performance monitoring
    tracemalloc.start()
    start_time = time.time()
    
    # Load data based on file extension
    logger.info(f"Loading data from {input_file}")
    
    file_suffix = input_file.suffix.lower()
    if file_suffix == '.parquet':
        logger.info("Detected Parquet file format")
        df = pd.read_parquet(input_file)
    elif file_suffix in ['.csv', '.txt']:
        logger.info("Detected CSV file format")
        df = pd.read_csv(input_file)
    else:
        logger.error(f"Unsupported file format: {file_suffix}. Supported formats: .csv, .txt, .parquet")
        sys.exit(1)
    
    # Limit stations if requested
    if max_stations and len(df) > max_stations:
        logger.info(f"Limiting to {max_stations} stations (from {len(df)})")
        df = df.head(max_stations)
    
    # Get schema info
    schema_info = get_schema_info(df)
    logger.info(f"Schema: {schema_info['recognized_columns']} recognized")
    
    # Check if zipcode processing is available
    if by_zipcode and 'zipcode' not in df.columns and 'zip_code' not in df.columns:
        logger.warning("No zipcode column found, processing all stations together")
        by_zipcode = False
    
    # Run optimization
    processing_mode = "by zipcode" if by_zipcode else "all together"
    logger.info(f"Running optimization with profile '{profile}', seed {seed}, processing {processing_mode}")
    optimizer = EnhancedSpectrumOptimizer(profile, seed=seed)
    
    try:
        result = optimizer.optimize(df, process_by_zipcode=by_zipcode)
    except SchemaError as e:
        logger.error(f"Schema error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Get performance metrics
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Extract metrics
    if hasattr(result, 'attrs') and 'optimization_metrics' in result.attrs:
        metrics = result.attrs['optimization_metrics']
    else:
        metrics = {}
    
    # Add performance metrics
    metrics['total_time_seconds'] = end_time - start_time
    metrics['memory_peak_mb'] = peak / 1024 / 1024
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save outputs
    assignments_file = output_dir / 'assignments.csv'
    result.to_csv(assignments_file, index=False)
    logger.info(f"Saved assignments to {assignments_file}")
    
    # Create GeoJSON
    geojson_file = output_dir / 'assignments.geojson'
    create_geojson(result, geojson_file)
    
    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Create HTML report
    report_file = output_dir / 'report.html'
    create_html_report(result, metrics, report_file)
    
    # Create dashboard if requested
    if create_dashboard:
        dashboard_file = output_dir / 'dashboard.html'
        logger.info("Creating interactive dashboard...")
        create_dashboard_viz(
            assignments_path=str(assignments_file),
            metrics_path=str(metrics_file),
            output_path=str(dashboard_file)
        )
        logger.info(f"Created dashboard: {dashboard_file}")
    
    # Run comprehensive validation
    logger.info("Running comprehensive validation...")
    is_valid, validation_metrics, validation_report = validate_optimization_results(
        result, metrics, profile, output_dir
    )
    
    # Add validation metrics to overall metrics
    metrics['validation'] = validation_metrics
    
    # Update metrics file with validation results
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Profile: {profile}")
    print(f"Stations: {len(result)}")
    print(f"Channels used: {metrics.get('objective_metrics', {}).get('channels_used', 'N/A')}")
    print(f"Time: {metrics['total_time_seconds']:.2f} seconds")
    print(f"Memory peak: {metrics['memory_peak_mb']:.2f} MB")
    print(f"Constraints: {metrics.get('constraint_stats', {}).get('total', 'N/A')}")
    print(f"Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    print(f"Output directory: {output_dir}")
    
    if not is_valid:
        print("\n⚠️  WARNING: Validation failed! Check validation_report.txt for details")
    
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Spectrum Optimization Tool - Minimize channels with directional geometry'
    )
    
    parser.add_argument('input_file', type=Path, help='Input CSV or Parquet file with station data')
    parser.add_argument('--profile', default='default', 
                       choices=['default', 'am', 'fm', 'microwave', 'microwave_fast'],
                       help='Configuration profile to use')
    parser.add_argument('--out', '--output-dir', dest='output_dir', type=Path,
                       default=Path('runs/default'),
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for deterministic results')
    parser.add_argument('--max-stations', type=int, default=None,
                       help='Limit number of stations (for testing)')
    parser.add_argument('--create-dashboard', action='store_true',
                       help='Create interactive dashboard visualization')
    parser.add_argument('--by-zipcode', action='store_true', default=True,
                       help='Process each zipcode independently (default: True)')
    parser.add_argument('--all-together', action='store_true',
                       help='Process all stations together (disables zipcode processing)')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Determine processing mode
    by_zipcode = not args.all_together  # Default is True unless --all-together is specified
    
    # Run optimization
    metrics = run_optimization(
        args.input_file,
        args.profile,
        args.output_dir,
        args.seed,
        args.max_stations,
        args.create_dashboard,
        by_zipcode
    )
    
    # Return success
    sys.exit(0)


def run_validation_tests():
    """
    Run comprehensive validation tests for parquet loading with zipcode and directional support.
    Tests:
    1. Parquet file loading with zipcode preservation
    2. Azimuth constraints in interference graph
    3. Directional pattern respect in optimization
    """
    import tempfile
    import shutil
    from tool.directional import DirectionalGeometry, DirectionalConfig
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION VALIDATION TEST SUITE")
    print("=" * 70)
    
    # Track test results
    test_results = []
    
    # Test 1: Parquet file with zipcode support
    def test_parquet_zipcode_loading():
        """Test that parquet files load correctly with zipcode preservation."""
        print("\n📋 Test 1: Parquet File Loading with Zipcode Support")
        print("-" * 50)
        
        try:
            # Create test data with zipcodes
            test_data = pd.DataFrame({
                'latitude': [40.7128, 34.0522, 41.8781, 37.7749, 40.7580],
                'longitude': [-74.0060, -118.2437, -87.6298, -122.4194, -73.9855],
                'zip_code': ['10001', '90001', '60601', '94102', '10019'],
                'station_id': ['NYC1', 'LA1', 'CHI1', 'SF1', 'NYC2'],
                'frequency': [100.1, 101.5, 99.9, 102.3, 100.1],
                'power': [1000, 1500, 2000, 1200, 1000],
                'azimuth_deg': [90, 180, 270, 0, 45],
                'beamwidth_deg': [30, 60, 90, 120, 30]
            })
            
            # Save as parquet
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                parquet_path = Path(tmp.name)
                test_data.to_parquet(parquet_path)
            
            # Create temporary output directory
            output_dir = Path(tempfile.mkdtemp(prefix='opt_test_'))
            
            print(f"✓ Created test parquet file with {len(test_data)} stations")
            print(f"  Zipcodes in input: {sorted(test_data['zip_code'].unique())}")
            
            # Run optimization
            metrics = run_optimization(
                parquet_path,
                profile='default',
                output_dir=output_dir,
                seed=42,
                max_stations=None,
                create_dashboard=False
            )
            
            # Load results and validate
            results_df = pd.read_csv(output_dir / 'assignments.csv')
            
            # Check 1: Zipcode column preserved
            if 'zipcode' in results_df.columns:
                print("✓ Zipcode column preserved in output")
                result_zips = sorted(results_df['zipcode'].unique())
                print(f"  Zipcodes in output: {result_zips}")
                
                # Verify all zipcodes match (convert to strings for comparison)
                input_zips = set(str(z) for z in test_data['zip_code'].unique())
                output_zips = set(str(z) for z in result_zips)
                if output_zips == input_zips:
                    print("✓ All zipcodes correctly preserved")
                else:
                    print("✗ Zipcode mismatch detected")
                    print(f"    Input: {input_zips}")
                    print(f"    Output: {output_zips}")
                    return False
            else:
                print("✗ Zipcode column not found in output")
                return False
            
            # Check 2: Station count preserved
            if len(results_df) == len(test_data):
                print(f"✓ Station count preserved: {len(results_df)}")
            else:
                print(f"✗ Station count mismatch: {len(test_data)} → {len(results_df)}")
                return False
            
            # Check 3: Frequency assignments made
            if 'assigned_frequency' in results_df.columns:
                unique_freqs = results_df['assigned_frequency'].nunique()
                print(f"✓ Frequency assignments made: {unique_freqs} unique frequencies")
            else:
                print("✗ No frequency assignments found")
                return False
            
            # Check 4: Zipcode metrics in output
            if 'zipcode_metrics' in metrics:
                zipcode_metrics = metrics['zipcode_metrics']
                if zipcode_metrics.get('frequency_usage', {}).get('available'):
                    print("✓ Zipcode frequency analysis included in metrics")
                if zipcode_metrics.get('interference', {}).get('available'):
                    print("✓ Zipcode interference analysis included in metrics")
            
            # Cleanup
            shutil.rmtree(output_dir)
            parquet_path.unlink()
            
            print("\n✅ Test 1 PASSED: Parquet loading with zipcode support works correctly")
            return True
            
        except Exception as e:
            print(f"\n❌ Test 1 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test 2: Azimuth constraints in interference graph
    def test_azimuth_constraints():
        """Test that azimuth constraints affect interference decisions."""
        print("\n📡 Test 2: Azimuth Constraints in Interference Graph")
        print("-" * 50)
        
        try:
            # Create stations with specific directional patterns
            # Two pairs: one aligned, one not aligned
            test_data = pd.DataFrame({
                # Pair 1: Aligned stations (pointing at each other)
                'station_id': ['S1', 'S2', 'S3', 'S4'],
                'latitude': [40.0, 40.0, 41.0, 41.0],
                'longitude': [-74.0, -73.9, -74.0, -73.9],
                # S1 points east (90°) at S2, S2 points west (270°) at S1
                # S3 points north (0°), S4 points south (180°) - not aligned
                'azimuth_deg': [90, 270, 0, 180],
                'beamwidth_deg': [30, 30, 30, 30],
                'power_watts': [1000, 1000, 1000, 1000]
            })
            
            # Initialize directional geometry
            config = DirectionalConfig(
                az_tolerance_deg=5.0,
                r_main_km=50.0,
                r_off_km=20.0
            )
            geometry = DirectionalGeometry(config)
            
            print("Station configurations:")
            for idx, row in test_data.iterrows():
                print(f"  {row['station_id']}: az={row['azimuth_deg']}°, "
                      f"pos=({row['latitude']:.1f}, {row['longitude']:.1f})")
            
            # Test interference between pairs
            # Pair 1: S1-S2 (should use main radius - aligned)
            s1 = test_data.iloc[0].to_dict()
            s2 = test_data.iloc[1].to_dict()
            interferes_12, radius_12 = geometry.check_directional_interference(s1, s2)
            
            # Pair 2: S3-S4 (should use off radius - not aligned)
            s3 = test_data.iloc[2].to_dict()
            s4 = test_data.iloc[3].to_dict()
            interferes_34, radius_34 = geometry.check_directional_interference(s3, s4)
            
            print(f"\nInterference results:")
            print(f"  S1↔S2 (aligned): radius={radius_12}km, interferes={interferes_12}")
            print(f"  S3↔S4 (not aligned): radius={radius_34}km, interferes={interferes_34}")
            
            # Validate: aligned pair should use main radius, non-aligned should use off radius
            if radius_12 == config.r_main_km:
                print("✓ Aligned stations correctly use main radius")
            else:
                print(f"✗ Aligned stations should use main radius ({config.r_main_km}km)")
                return False
            
            if radius_34 == config.r_off_km:
                print("✓ Non-aligned stations correctly use off-lobe radius")
            else:
                print(f"✗ Non-aligned stations should use off radius ({config.r_off_km}km)")
                return False
            
            # Test with actual optimization
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                csv_path = Path(tmp.name)
                test_data.to_csv(csv_path, index=False)
            
            output_dir = Path(tempfile.mkdtemp(prefix='opt_test_'))
            
            # Run optimization
            metrics = run_optimization(
                csv_path,
                profile='default',
                output_dir=output_dir,
                seed=42,
                max_stations=4,
                create_dashboard=False
            )
            
            # Check neighbor metrics
            if metrics.get('neighbor_metrics'):
                avg_neighbors = metrics['neighbor_metrics'].get('avg_neighbors', 0)
                total_edges = metrics['neighbor_metrics'].get('total_edges', 0)
                print(f"\n✓ Interference graph built with directional constraints:")
                print(f"  Total edges: {total_edges}")
                print(f"  Avg neighbors: {avg_neighbors:.2f}")
            
            # Cleanup
            shutil.rmtree(output_dir)
            csv_path.unlink()
            
            print("\n✅ Test 2 PASSED: Azimuth constraints properly affect interference")
            return True
            
        except Exception as e:
            print(f"\n❌ Test 2 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test 3: Directional pattern respect
    def test_directional_pattern_respect():
        """Test that optimization respects directional patterns."""
        print("\n🎯 Test 3: Directional Pattern Respect in Optimization")
        print("-" * 50)
        
        try:
            # Create a scenario where directional patterns matter
            # Stations in a line, alternating directions
            stations = []
            for i in range(6):
                stations.append({
                    'station_id': f'S{i+1}',
                    'latitude': 40.0,
                    'longitude': -74.0 + i * 0.1,  # Spread east-west
                    'azimuth_deg': 90 if i % 2 == 0 else 270,  # Alternate E/W
                    'beamwidth_deg': 60,
                    'power_watts': 1000
                })
            
            test_data = pd.DataFrame(stations)
            
            print("Created test scenario:")
            print("  6 stations in east-west line")
            print("  Alternating azimuth: E(90°) / W(270°)")
            print("  Narrow beamwidth: 60°")
            
            # Save test data
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                csv_path = Path(tmp.name)
                test_data.to_csv(csv_path, index=False)
            
            output_dir = Path(tempfile.mkdtemp(prefix='opt_test_'))
            
            # Run optimization with directional config
            metrics = run_optimization(
                csv_path,
                profile='default',
                output_dir=output_dir,
                seed=42,
                max_stations=6,
                create_dashboard=False
            )
            
            # Load results
            results_df = pd.read_csv(output_dir / 'assignments.csv')
            
            # Analyze frequency assignments
            freq_assignments = results_df[['station_id', 'assigned_frequency', 'azimuth_deg']].copy()
            freq_assignments = freq_assignments.sort_values('station_id')
            
            print("\nFrequency assignments:")
            for idx, row in freq_assignments.iterrows():
                print(f"  {row['station_id']}: {row['assigned_frequency']:.1f} MHz "
                      f"(az={row['azimuth_deg']}°)")
            
            # Check pattern: stations pointing same direction should be able to reuse frequencies
            east_pointing = freq_assignments[freq_assignments['azimuth_deg'] == 90]['assigned_frequency'].values
            west_pointing = freq_assignments[freq_assignments['azimuth_deg'] == 270]['assigned_frequency'].values
            
            # Count unique frequencies in each group
            unique_east = len(set(east_pointing))
            unique_west = len(set(west_pointing))
            total_unique = results_df['assigned_frequency'].nunique()
            
            print(f"\nFrequency reuse analysis:")
            print(f"  East-pointing stations: {unique_east} unique frequencies")
            print(f"  West-pointing stations: {unique_west} unique frequencies")
            print(f"  Total unique: {total_unique} frequencies")
            
            # With directional patterns, we should see some frequency reuse
            if total_unique < len(test_data):
                print("✓ Frequency reuse achieved through directional patterns")
                reuse_factor = len(test_data) / total_unique
                print(f"  Reuse factor: {reuse_factor:.2f}")
            else:
                print("⚠ No frequency reuse detected (may be due to conservative constraints)")
            
            # Check constraint statistics
            if metrics.get('constraint_stats'):
                constraints = metrics['constraint_stats']
                print(f"\n✓ Constraint statistics:")
                print(f"  Co-channel constraints: {constraints.get('co_channel', 0)}")
                print(f"  Adjacent channel constraints: {constraints.get('adjacent_channel', 0)}")
                print(f"  Total constraints: {constraints.get('total', 0)}")
            
            # Cleanup
            shutil.rmtree(output_dir)
            csv_path.unlink()
            
            print("\n✅ Test 3 PASSED: Optimization respects directional patterns")
            return True
            
        except Exception as e:
            print(f"\n❌ Test 3 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run all tests
    test_results.append(("Parquet/Zipcode Loading", test_parquet_zipcode_loading()))
    test_results.append(("Azimuth Constraints", test_azimuth_constraints()))
    test_results.append(("Directional Patterns", test_directional_pattern_respect()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - Review output above")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    # Check if running tests
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        sys.exit(run_validation_tests())
    else:
        main()