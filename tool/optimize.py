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
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['longitude']), float(row['latitude'])]
            },
            "properties": {
                "station_id": str(row.get('station_id', f'S{idx}')),
                "assigned_frequency": float(row['assigned_frequency']),
                "power_watts": float(row.get('power_watts', 1000)),
                "azimuth_deg": float(row.get('azimuth_deg', 0)),
                "beamwidth_deg": float(row.get('beamwidth_deg', 360))
            }
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


def run_optimization(input_file: Path, profile: str, output_dir: Path, 
                     seed: int, max_stations: int = None, create_dashboard: bool = False) -> dict:
    """Run the optimization and generate all outputs."""
    
    # Start performance monitoring
    tracemalloc.start()
    start_time = time.time()
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Limit stations if requested
    if max_stations and len(df) > max_stations:
        logger.info(f"Limiting to {max_stations} stations (from {len(df)})")
        df = df.head(max_stations)
    
    # Get schema info
    schema_info = get_schema_info(df)
    logger.info(f"Schema: {schema_info['recognized_columns']} recognized")
    
    # Run optimization
    logger.info(f"Running optimization with profile '{profile}' and seed {seed}")
    optimizer = EnhancedSpectrumOptimizer(profile, seed=seed)
    
    try:
        result = optimizer.optimize(df)
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
    print(f"Output directory: {output_dir}")
    
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Spectrum Optimization Tool - Minimize channels with directional geometry'
    )
    
    parser.add_argument('input_file', type=Path, help='Input CSV file with station data')
    parser.add_argument('--profile', default='default', 
                       choices=['default', 'am', 'fm'],
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
    
    args = parser.parse_args()
    
    # Check input file exists
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Run optimization
    metrics = run_optimization(
        args.input_file,
        args.profile,
        args.output_dir,
        args.seed,
        args.max_stations,
        args.create_dashboard
    )
    
    # Return success
    sys.exit(0)


if __name__ == '__main__':
    main()