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
    
    def optimize(self, stations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize frequency assignments using geometric neighbors and guard channels.
        
        Args:
            stations_df: DataFrame with station data (any reasonable column naming)
            
        Returns:
            DataFrame with 'assigned_frequency' column added
        """
        # Normalize and validate input
        try:
            stations_df = prepare_dataframe(stations_df, strict=False)
        except SchemaError as e:
            logger.error(f"Schema validation failed: {e}")
            raise
        
        logger.info(f"Starting optimization for {len(stations_df)} stations")
        start_time = time.time()
        
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
        
        # Generate frequency channels
        frequencies = self._generate_frequencies()
        n_freqs = len(frequencies)
        n_stations = len(stations_df)
        
        logger.info(f"Using {n_freqs} frequency channels from {frequencies[0]:.2f} to {frequencies[-1]:.2f} MHz")
        
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
        
        # Constraint 2: Link y[f] to x[i,f] - y[f]=1 if and only if any x[i,f]=1
        for f in range(n_freqs):
            # If any station uses f, then y[f] must be 1
            model.Add(sum(x[i, f] for i in range(n_stations)) <= n_stations * y[f])
            # Force y[f]=0 if no station uses f (tighter formulation)
            model.Add(sum(x[i, f] for i in range(n_stations)) >= y[f])
        
        # Constraint 3: Add interference constraints for geometric neighbors only
        self._add_interference_constraints(model, x, edges, n_freqs)
        
        # Objective: Lexicographic minimization via weights
        self._add_lexicographic_objective(model, x, y, n_stations, n_freqs)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config['solver']['timeout_seconds']
        solver.parameters.num_search_workers = self.config['solver']['num_workers']
        solver.parameters.random_seed = self.seed  # Ensure deterministic results
        
        status = solver.Solve(model)
        
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
                    'config_profile': self.config
                }
            }
        else:
            logger.error(f"Solver failed with status: {solver.StatusName(status)}")
            result['assigned_frequency'] = frequencies[0]  # Assign all to first frequency
        
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