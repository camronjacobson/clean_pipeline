"""
Enhanced spectrum optimizer with geometric neighbors and guard channel constraints.
Uses directional geometry from Task 1 and implements proper CP-SAT constraints.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from ortools.sat.python import cp_model
from typing import Dict, List, Tuple, Optional, Set
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tool import create_neighbor_discovery
from src.directional_integration import DirectionalSpectrumOptimizer
from src.schema_normalizer import prepare_dataframe, validate_input, SchemaError

logger = logging.getLogger(__name__)


class EnhancedSpectrumOptimizer:
    """
    Spectrum optimizer with geometric neighbors and adjacent channel protection.
    """
    
    def __init__(self, config_profile: str = 'default', seed: int = 42):
        """
        Initialize with configuration profile.
        
        Args:
            config_profile: Name of config profile (default, am, fm)
            seed: Random seed for deterministic tie-breaking
        """
        self.config = self._load_config(config_profile)
        self.seed = seed
        
        # Create directional geometry components
        self.directional_optimizer = DirectionalSpectrumOptimizer({
            'az_tolerance_deg': self.config['geometry']['az_tolerance_deg'],
            'r_main_km': self.config['geometry']['r_main_km'],
            'r_off_km': self.config['geometry']['r_off_km'],
            'max_search_radius_km': max(
                self.config['geometry']['r_main_km'] * 1.5,
                100.0
            )
        })
        
        # Statistics tracking
        self.constraint_stats = {
            'co_channel': 0,
            'adjacent_channel': 0,
            'total': 0,
            'skipped_invalid': 0
        }
        
        # Objective component tracking
        self.objective_metrics = {
            'channels_used': 0,
            'channel_indices_used': [],
            'spectrum_span_khz': 0,
            'channel_packing_score': 0
        }
        
        # Zipcode analysis tracking
        self.zipcode_metrics = {}
        
        logger.info(f"EnhancedSpectrumOptimizer initialized with profile: {config_profile}, seed: {seed}")
    
    def _load_config(self, profile: str) -> Dict:
        """Load configuration from YAML profile."""
        config_dir = Path(__file__).parent.parent / 'config' / 'profiles'
        config_file = config_dir / f'{profile}.yaml'
        
        if not config_file.exists():
            logger.warning(f"Config profile {profile} not found, using default")
            config_file = config_dir / 'default.yaml'
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            # Return default configuration
            return {
                'band': {'min_mhz': 88.0, 'max_mhz': 108.0, 'step_khz': 200},
                'geometry': {'r_main_km': 50, 'r_off_km': 20, 'az_tolerance_deg': 5.0},
                'interference': {'guard_offsets': [-1, 1]},
                'solver': {'timeout_seconds': 60, 'num_workers': 4},
                'weights': {'w_span': 100, 'w_count': 10, 'w_surplus': 1}
            }
    
    def optimize(self, stations_df: pd.DataFrame, process_by_zipcode: bool = True) -> pd.DataFrame:
        """
        Optimize frequency assignments using geometric neighbors and guard channels.
        
        Args:
            stations_df: DataFrame with station data (any reasonable column naming)
            process_by_zipcode: If True and zipcode column exists, process each zipcode independently
            
        Returns:
            DataFrame with 'assigned_frequency' column added
        """
        # Normalize and validate input
        try:
            stations_df = prepare_dataframe(stations_df, strict=False)
        except SchemaError as e:
            logger.error(f"Schema validation failed: {e}")
            raise
        
        start_time = time.time()
        
        # Check if we should process by zipcode
        if process_by_zipcode and 'zipcode' in stations_df.columns:
            logger.info(f"Processing {len(stations_df)} stations by zipcode")
            return self._optimize_by_zipcode(stations_df, start_time)
        else:
            logger.info(f"Processing all {len(stations_df)} stations together")
            return self._optimize_all_stations(stations_df, start_time)
    
    def _optimize_by_zipcode(self, stations_df: pd.DataFrame, start_time: float) -> pd.DataFrame:
        """
        Optimize each zipcode independently for better scalability.
        
        Args:
            stations_df: DataFrame with station data including zipcode column
            start_time: Start time for tracking total optimization time
            
        Returns:
            DataFrame with frequency assignments for all zipcodes
        """
        # Group stations by zipcode
        zipcode_groups = stations_df.groupby('zipcode')
        num_zipcodes = len(zipcode_groups)
        
        logger.info(f"Found {num_zipcodes} zipcodes to process independently")
        
        # Results storage
        all_results = []
        zipcode_solve_times = {}
        zipcode_stats = {}
        
        # Process each zipcode independently
        for zipcode, group_df in zipcode_groups:
            zipcode_start = time.time()
            group_df = group_df.copy()
            num_stations = len(group_df)
            
            logger.info(f"Processing zipcode {zipcode} with {num_stations} stations")
            
            try:
                # Optimize this zipcode independently
                result_df = self._optimize_single_group(group_df, f"zipcode_{zipcode}")
                
                # Track metrics for this zipcode
                zipcode_solve_time = time.time() - zipcode_start
                zipcode_solve_times[str(zipcode)] = zipcode_solve_time
                
                if 'assigned_frequency' in result_df.columns:
                    unique_freqs = result_df['assigned_frequency'].nunique()
                    zipcode_stats[str(zipcode)] = {
                        'stations': num_stations,
                        'unique_frequencies': unique_freqs,
                        'solve_time': zipcode_solve_time,
                        'efficiency': num_stations / unique_freqs if unique_freqs > 0 else 0
                    }
                    logger.info(f"  Zipcode {zipcode}: {unique_freqs} frequencies for {num_stations} stations "
                              f"(efficiency: {zipcode_stats[str(zipcode)]['efficiency']:.2f})")
                
                all_results.append(result_df)
                
            except Exception as e:
                logger.error(f"Failed to optimize zipcode {zipcode}: {e}")
                # Assign a default frequency for failed zipcodes
                group_df['assigned_frequency'] = self.config['band']['min_mhz']
                all_results.append(group_df)
        
        # Combine all results
        result = pd.concat(all_results, ignore_index=True)
        
        # Calculate overall metrics
        total_time = time.time() - start_time
        
        # Store zipcode-specific metrics
        self.zipcode_metrics = {
            'processing_mode': 'by_zipcode',
            'num_zipcodes_processed': num_zipcodes,
            'zipcode_stats': zipcode_stats,
            'solve_times': zipcode_solve_times,
            'total_solve_time': total_time,
            'avg_solve_time_per_zipcode': np.mean(list(zipcode_solve_times.values())) if zipcode_solve_times else 0,
            'max_solve_time': max(zipcode_solve_times.values()) if zipcode_solve_times else 0,
            'min_solve_time': min(zipcode_solve_times.values()) if zipcode_solve_times else 0
        }
        
        # Calculate frequency reuse across zipcodes
        if 'assigned_frequency' in result.columns:
            total_unique_freqs = result['assigned_frequency'].nunique()
            avg_freqs_per_zip = np.mean([s['unique_frequencies'] for s in zipcode_stats.values()]) if zipcode_stats else 0
            
            self.zipcode_metrics['frequency_reuse'] = {
                'total_unique_frequencies': total_unique_freqs,
                'avg_frequencies_per_zipcode': avg_freqs_per_zip,
                'frequency_reuse_factor': avg_freqs_per_zip / total_unique_freqs if total_unique_freqs > 0 else 0
            }
            
            logger.info(f"Optimization complete: {total_unique_freqs} total unique frequencies")
            logger.info(f"Average {avg_freqs_per_zip:.1f} frequencies per zipcode")
            logger.info(f"Frequency reuse factor: {self.zipcode_metrics['frequency_reuse']['frequency_reuse_factor']:.2f}")
        
        # Add metrics to result
        result.attrs = {
            'optimization_metrics': {
                'solve_time_seconds': total_time,
                'processing_mode': 'by_zipcode',
                'num_zipcodes': num_zipcodes,
                'zipcode_metrics': self.zipcode_metrics.copy(),
                'constraint_stats': self.constraint_stats.copy(),
                'objective_metrics': self.objective_metrics.copy()
            }
        }
        
        logger.info(f"Total optimization time: {total_time:.2f}s for {num_zipcodes} zipcodes")
        
        return result
    
    def _optimize_single_group(self, group_df: pd.DataFrame, group_name: str) -> pd.DataFrame:
        """
        Optimize a single group of stations (e.g., one zipcode).
        
        Args:
            group_df: DataFrame with stations to optimize
            group_name: Name of the group for logging
            
        Returns:
            DataFrame with frequency assignments
        """
        n_stations = len(group_df)
        
        # Build interference graph for this group only
        edges, neighbor_metrics = self.directional_optimizer.build_interference_graph(group_df)
        
        if len(edges) > 0:
            logger.debug(f"  {group_name}: {len(edges)} edges, avg neighbors: {neighbor_metrics['avg_neighbors']:.1f}")
        
        # Generate frequency channels
        frequencies = self._generate_frequencies()
        n_freqs = len(frequencies)
        
        # Create CP-SAT model
        model = cp_model.CpModel()
        
        # Variables: x[i,f] = 1 if station i uses frequency f
        x = {}
        for i in range(n_stations):
            for f in range(n_freqs):
                x[i, f] = model.NewBoolVar(f'x_{i}_{f}')
        
        # Variables: y[f] = 1 if frequency f is used by ANY station
        y = {}
        for f in range(n_freqs):
            y[f] = model.NewBoolVar(f'y_{f}')
        
        # Constraint 1: Each station gets exactly one frequency
        for i in range(n_stations):
            model.Add(sum(x[i, f] for f in range(n_freqs)) == 1)
        
        # Constraint 2: Link y[f] to x[i,f]
        for f in range(n_freqs):
            model.Add(sum(x[i, f] for i in range(n_stations)) <= n_stations * y[f])
            model.Add(sum(x[i, f] for i in range(n_stations)) >= y[f])
        
        # Constraint 3: Add interference constraints
        self._add_interference_constraints(model, x, edges, n_freqs)
        
        # Objective: Minimize channels used (simpler for per-zipcode optimization)
        model.Minimize(sum(y[f] for f in range(n_freqs)))
        
        # Solve with shorter timeout for individual zipcodes
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = min(10, self.config['solver']['timeout_seconds'])
        solver.parameters.num_search_workers = self.config['solver']['num_workers']
        solver.parameters.random_seed = self.seed
        
        status = solver.Solve(model)
        
        # Process results
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            result = group_df.copy()
            assignments = []
            
            for i in range(n_stations):
                for f in range(n_freqs):
                    if solver.Value(x[i, f]) == 1:
                        assignments.append(frequencies[f])
                        break
                else:
                    assignments.append(frequencies[0])  # Fallback
            
            result['assigned_frequency'] = assignments
            return result
        else:
            # If optimization fails, assign default frequency
            logger.warning(f"  {group_name}: No solution found, using default frequency")
            result = group_df.copy()
            result['assigned_frequency'] = self.config['band']['min_mhz']
            return result
    
    def _optimize_all_stations(self, stations_df: pd.DataFrame, start_time: float) -> pd.DataFrame:
        """
        Original optimization method that processes all stations together.
        """
        # Reset statistics
        self.constraint_stats = {
            'co_channel': 0,
            'adjacent_channel': 0,
            'total': 0,
            'skipped_invalid': 0
        }
        
        # Build geometric neighbors using directional geometry
        edges, neighbor_metrics = self.directional_optimizer.build_interference_graph(stations_df)
        logger.info(f"Found {len(edges)} interference edges (avg neighbors: {neighbor_metrics['avg_neighbors']:.1f})")
        
        # Analyze zipcode-based interference patterns
        self.zipcode_metrics['interference'] = self.analyze_zipcode_interference(stations_df, edges)
        
        # Generate frequency channels
        frequencies = self._generate_frequencies()
        n_freqs = len(frequencies)
        n_stations = len(stations_df)
        
        logger.info(f"Using {n_freqs} frequency channels from {frequencies[0]:.2f} to {frequencies[-1]:.2f} MHz")
        
        # Create CP-SAT model
        model = cp_model.CpModel()
        
        # Variables: x[i,f] = 1 if station i uses frequency f
        logger.info(f"Creating {n_stations * n_freqs} decision variables...")
        x = {}
        for i in range(n_stations):
            for f in range(n_freqs):
                x[i, f] = model.NewBoolVar(f'x_{i}_{f}')
        
        # Variables: y[f] = 1 if frequency f is used by ANY station
        y = {}
        for f in range(n_freqs):
            y[f] = model.NewBoolVar(f'y_{f}')
        logger.info(f"Created {n_stations * n_freqs + n_freqs} variables")
        
        # Constraint 1: Each station gets exactly one frequency
        logger.info("Adding assignment constraints...")
        for i in range(n_stations):
            model.Add(sum(x[i, f] for f in range(n_freqs)) == 1)
        
        # Constraint 2: Link y[f] to x[i,f] - y[f]=1 if and only if any x[i,f]=1
        logger.info("Adding channel usage tracking constraints...")
        for f in range(n_freqs):
            # If any station uses f, then y[f] must be 1
            model.Add(sum(x[i, f] for i in range(n_stations)) <= n_stations * y[f])
            # Force y[f]=0 if no station uses f (tighter formulation)
            model.Add(sum(x[i, f] for i in range(n_stations)) >= y[f])
        
        # Constraint 3: Add interference constraints for geometric neighbors only
        logger.info(f"Adding interference constraints for {len(edges)} edges...")
        self._add_interference_constraints(model, x, edges, n_freqs)
        
        # Objective: Lexicographic minimization via weights
        logger.info("Setting optimization objective...")
        self._add_lexicographic_objective(model, x, y, n_stations, n_freqs)
        
        # Solve
        logger.info(f"Starting solver with timeout={self.config['solver']['timeout_seconds']}s, workers={self.config['solver']['num_workers']}")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config['solver']['timeout_seconds']
        solver.parameters.num_search_workers = self.config['solver']['num_workers']
        solver.parameters.random_seed = self.seed  # Ensure deterministic results
        
        # Add solution callback for progress monitoring
        class SolutionCallback(cp_model.CpSolverSolutionCallback):
            def __init__(self):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self.solution_count = 0
                self.start_time = time.time()
                
            def on_solution_callback(self):
                self.solution_count += 1
                elapsed = time.time() - self.start_time
                if self.solution_count == 1 or self.solution_count % 10 == 0:
                    logger.info(f"Found solution {self.solution_count} after {elapsed:.1f}s")
        
        callback = SolutionCallback()
        status = solver.Solve(model, callback)
        
        # Extract solution
        result = stations_df.copy()
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            assignments = []
            for i in range(n_stations):
                for f in range(n_freqs):
                    if solver.Value(x[i, f]) == 1:
                        assignments.append(frequencies[f])
                        break
                else:
                    assignments.append(frequencies[0])  # Fallback
            
            result['assigned_frequency'] = assignments
            
            # Extract y[f] values and calculate objective metrics
            channels_used = []
            channel_indices_used = []
            for f in range(n_freqs):
                if solver.Value(y[f]) == 1:
                    channels_used.append(frequencies[f])
                    channel_indices_used.append(f)
            
            # Calculate objective component metrics
            self.objective_metrics['channels_used'] = len(channels_used)
            self.objective_metrics['channel_indices_used'] = channel_indices_used
            
            if channels_used:
                min_freq = min(channels_used)
                max_freq = max(channels_used)
                self.objective_metrics['spectrum_span_khz'] = (max_freq - min_freq) * 1000
                self.objective_metrics['channel_packing_score'] = sum(channel_indices_used) / len(channel_indices_used)
            else:
                self.objective_metrics['spectrum_span_khz'] = 0
                self.objective_metrics['channel_packing_score'] = 0
            
            # Calculate metrics
            unique_freqs = result['assigned_frequency'].nunique()
            elapsed = time.time() - start_time
            
            logger.info(f"Optimization complete: {unique_freqs} unique frequencies used in {elapsed:.1f}s")
            logger.info(f"Constraints added: {self.constraint_stats['total']} "
                       f"(co-channel: {self.constraint_stats['co_channel']}, "
                       f"adjacent: {self.constraint_stats['adjacent_channel']}, "
                       f"skipped: {self.constraint_stats['skipped_invalid']})")
            
            # Analyze zipcode-based frequency usage
            self.zipcode_metrics['frequency_usage'] = self.analyze_zipcode_frequency_usage(result)
            
            # Add metrics to result
            result.attrs = {
                'optimization_metrics': {
                    'unique_frequencies': unique_freqs,
                    'total_stations': n_stations,
                    'solve_time_seconds': elapsed,
                    'solver_status': solver.StatusName(status),
                    'constraint_stats': self.constraint_stats.copy(),
                    'neighbor_metrics': neighbor_metrics,
                    'objective_metrics': self.objective_metrics.copy(),
                    'zipcode_metrics': self.zipcode_metrics.copy(),
                    'config_profile': self.config
                }
            }
        else:
            logger.error(f"Solver failed with status: {solver.StatusName(status)}")
            # Use round-robin assignment as emergency fallback to avoid all stations on same frequency
            logger.warning(f"Using round-robin fallback assignment for {len(result)} stations")
            for idx in range(len(result)):
                result.iloc[idx, result.columns.get_loc('assigned_frequency')] = \
                    frequencies[idx % len(frequencies)]
        
        return result
    
    def _generate_frequencies(self) -> List[float]:
        """Generate list of available frequencies from config."""
        min_freq = self.config['band']['min_mhz']
        max_freq = self.config['band']['max_mhz']
        step_mhz = self.config['band']['step_khz'] / 1000.0
        
        frequencies = []
        freq = min_freq
        while freq <= max_freq:
            frequencies.append(freq)
            freq += step_mhz
        
        return frequencies
    
    def _add_interference_constraints(self, model: cp_model.CpModel, x: Dict,
                                     edges: List[Tuple[int, int]], n_freqs: int):
        """
        Add co-channel and adjacent channel constraints for geometric neighbors.
        
        Args:
            model: CP-SAT model
            x: Decision variables
            edges: List of (i, j) geometric neighbor pairs
            n_freqs: Number of frequency channels
        """
        guard_offsets = self.config['interference']['guard_offsets']
        
        for i, j in edges:
            # Co-channel constraint: stations i and j cannot use same frequency
            for f in range(n_freqs):
                model.Add(x[i, f] + x[j, f] <= 1)
                self.constraint_stats['co_channel'] += 1
            
            # Adjacent channel constraints based on guard_offsets
            for offset in guard_offsets:
                if offset == 0:
                    continue  # Skip co-channel (already handled)
                
                for f in range(n_freqs):
                    f_adjacent = f + offset
                    
                    # Check if adjacent frequency is valid (in band)
                    if 0 <= f_adjacent < n_freqs:
                        # Station i at frequency f cannot coexist with station j at f+offset
                        model.Add(x[i, f] + x[j, f_adjacent] <= 1)
                        # Symmetric: station j at frequency f cannot coexist with station i at f+offset
                        model.Add(x[j, f] + x[i, f_adjacent] <= 1)
                        self.constraint_stats['adjacent_channel'] += 2
                    else:
                        self.constraint_stats['skipped_invalid'] += 1
        
        self.constraint_stats['total'] = (
            self.constraint_stats['co_channel'] + 
            self.constraint_stats['adjacent_channel']
        )
    
    def _add_lexicographic_objective(self, model: cp_model.CpModel, x: Dict, y: Dict,
                                    n_stations: int, n_freqs: int):
        """
        Add lexicographic objective function with strict weight separation.
        
        Priority order:
        1. Minimize number of channels used (W1 = 10^9)
        2. Pack channels toward low frequencies (W2 = 10^3)
        3. Deterministic tie-breaking (W3 = 1)
        """
        # Calculate safe weights for lexicographic ordering
        W1 = 10**9  # Dominates everything - minimize channel count
        W2 = 10**3  # Secondary - pack toward low frequencies
        W3 = 1      # Tertiary - deterministic tie-breaking
        
        # Build objective
        objective = 0
        
        for f in range(n_freqs):
            # Primary: minimize total channels used
            objective += W1 * y[f]
            
            # Secondary: prefer lower frequency indices
            objective += W2 * f * y[f]
            
            # Tertiary: deterministic station assignment for tie-breaking
            for i in range(n_stations):
                # Create stable hash for deterministic ordering
                # Scale down to avoid overflow and keep as integer
                stable_hash = (hash((f, i, self.seed)) % 100)
                # Multiply by small factor for tie-breaking
                objective += x[i, f] * stable_hash
        
        model.Minimize(objective)
    
    def analyze_zipcode_interference(self, stations_df: pd.DataFrame, 
                                    edges: List[Tuple[int, int]]) -> Dict:
        """
        Analyze interference patterns grouped by zipcode.
        
        Args:
            stations_df: DataFrame with station data including zipcode column
            edges: List of (i, j) interference edges
            
        Returns:
            Dictionary with zipcode-based interference statistics
        """
        zipcode_stats = {}
        
        # Check if zipcode column exists
        if 'zipcode' not in stations_df.columns:
            logger.info("No zipcode column found, skipping zipcode analysis")
            return {'available': False, 'message': 'No zipcode data available'}
        
        # Create station index to zipcode mapping
        idx_to_zip = {idx: str(zip_val) for idx, zip_val in enumerate(stations_df['zipcode'])}
        
        # Group stations by zipcode
        zipcode_groups = stations_df.groupby('zipcode').groups
        
        for zipcode, station_indices in zipcode_groups.items():
            zipcode = str(zipcode)
            station_indices = list(station_indices)
            
            # Initialize stats for this zipcode
            stats = {
                'station_count': len(station_indices),
                'internal_edges': 0,  # Edges within the same zipcode
                'external_edges': 0,  # Edges to other zipcodes
                'connected_zipcodes': set(),  # Set of zipcodes this one connects to
                'avg_interference_degree': 0,
                'stations': station_indices
            }
            
            # Count interference edges for stations in this zipcode
            edge_count = {idx: 0 for idx in station_indices}
            
            for i, j in edges:
                zip_i = idx_to_zip.get(i)
                zip_j = idx_to_zip.get(j)
                
                # Check if either station is in this zipcode
                if i in station_indices:
                    edge_count[i] += 1
                    if zip_j == zipcode:
                        stats['internal_edges'] += 0.5  # Count once per edge pair
                    else:
                        stats['external_edges'] += 1
                        if zip_j:
                            stats['connected_zipcodes'].add(zip_j)
                
                if j in station_indices and i not in station_indices:
                    edge_count[j] += 1
                    stats['external_edges'] += 1
                    if zip_i:
                        stats['connected_zipcodes'].add(zip_i)
            
            # Calculate average interference degree
            if station_indices:
                stats['avg_interference_degree'] = sum(edge_count.values()) / len(station_indices)
            
            # Convert set to list for JSON serialization
            stats['connected_zipcodes'] = list(stats['connected_zipcodes'])
            stats['internal_edges'] = int(stats['internal_edges'])  # Convert from float
            
            zipcode_stats[zipcode] = stats
        
        # Calculate overall statistics
        total_internal = sum(s['internal_edges'] for s in zipcode_stats.values())
        total_external = sum(s['external_edges'] for s in zipcode_stats.values()) / 2  # Divide by 2 as counted from both sides
        total_edges = len(edges)
        
        return {
            'available': True,
            'by_zipcode': zipcode_stats,
            'summary': {
                'total_zipcodes': len(zipcode_stats),
                'total_internal_edges': int(total_internal),
                'total_external_edges': int(total_external),
                'total_edges': total_edges,
                'internal_ratio': total_internal / total_edges if total_edges > 0 else 0,
                'external_ratio': total_external / total_edges if total_edges > 0 else 0
            }
        }
    
    def analyze_zipcode_frequency_usage(self, result_df: pd.DataFrame) -> Dict:
        """
        Analyze frequency usage patterns by zipcode after optimization.
        
        Args:
            result_df: DataFrame with optimization results including assigned_frequency
            
        Returns:
            Dictionary with zipcode-based frequency usage statistics
        """
        if 'zipcode' not in result_df.columns:
            return {'available': False, 'message': 'No zipcode data available'}
        
        if 'assigned_frequency' not in result_df.columns:
            return {'available': False, 'message': 'No frequency assignments available'}
        
        zipcode_freq_stats = {}
        
        # Group by zipcode
        for zipcode, group in result_df.groupby('zipcode'):
            zipcode = str(zipcode)
            
            # Get frequency distribution for this zipcode
            freq_counts = group['assigned_frequency'].value_counts()
            
            stats = {
                'station_count': len(group),
                'unique_frequencies': len(freq_counts),
                'frequency_distribution': freq_counts.to_dict(),
                'most_used_frequency': float(freq_counts.idxmax()) if not freq_counts.empty else None,
                'frequency_reuse_ratio': 1.0 - (len(freq_counts) / len(group)) if len(group) > 0 else 0,
                'min_frequency': float(group['assigned_frequency'].min()),
                'max_frequency': float(group['assigned_frequency'].max()),
                'frequency_span_mhz': float(group['assigned_frequency'].max() - group['assigned_frequency'].min())
            }
            
            zipcode_freq_stats[zipcode] = stats
        
        # Calculate cross-zipcode frequency conflicts
        freq_zipcode_map = {}
        for zipcode, group in result_df.groupby('zipcode'):
            for freq in group['assigned_frequency'].unique():
                if freq not in freq_zipcode_map:
                    freq_zipcode_map[freq] = []
                freq_zipcode_map[freq].append(str(zipcode))
        
        # Identify shared frequencies
        shared_frequencies = {freq: zips for freq, zips in freq_zipcode_map.items() if len(zips) > 1}
        
        return {
            'available': True,
            'by_zipcode': zipcode_freq_stats,
            'shared_frequencies': shared_frequencies,
            'summary': {
                'total_zipcodes': len(zipcode_freq_stats),
                'avg_frequencies_per_zipcode': np.mean([s['unique_frequencies'] for s in zipcode_freq_stats.values()]),
                'max_frequencies_in_zipcode': max([s['unique_frequencies'] for s in zipcode_freq_stats.values()]) if zipcode_freq_stats else 0,
                'min_frequencies_in_zipcode': min([s['unique_frequencies'] for s in zipcode_freq_stats.values()]) if zipcode_freq_stats else 0,
                'frequencies_shared_across_zipcodes': len(shared_frequencies)
            }
        }
        
        logger.debug(f"Lexicographic objective weights: W1={W1}, W2={W2}, W3={W3}")


def create_enhanced_optimizer(config_profile: str = 'default') -> EnhancedSpectrumOptimizer:
    """
    Factory function to create enhanced optimizer with specified profile.
    
    Args:
        config_profile: Name of configuration profile (default, am, fm)
        
    Returns:
        Configured EnhancedSpectrumOptimizer instance
    """
    return EnhancedSpectrumOptimizer(config_profile)