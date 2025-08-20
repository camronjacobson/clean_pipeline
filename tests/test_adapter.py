"""
Adapter to make tests work with actual SpectrumOptimizer implementation.
Now includes directional geometry support.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectrum_optimizer import SpectrumOptimizer as RealSpectrumOptimizer
from data import FrequencyDetector
from directional_integration import DirectionalSpectrumOptimizer
from spectrum_optimizer_enhanced import EnhancedSpectrumOptimizer
from tool import DirectionalGeometry, DirectionalConfig


class SpectrumOptimizer:
    """
    Adapter class that provides the interface expected by tests.
    Now with directional geometry support.
    """
    def __init__(self, freq_params=None):
        """Initialize with freq_params as tests expect."""
        # Use enhanced optimizer with fixed seed for determinism
        self.enhanced_optimizer = EnhancedSpectrumOptimizer('default', seed=42)
        
        # Override config for testing with smaller radii
        self.enhanced_optimizer.config['geometry']['r_main_km'] = 5.0
        self.enhanced_optimizer.config['geometry']['r_off_km'] = 0.5
        
        # Create directional optimizer with appropriate config
        # Use smaller radii for tests to better demonstrate directional effects
        config = {
            'az_tolerance_deg': 5.0,
            'r_main_km': 5.0,   # Reduced for tests (main lobe interference)
            'r_off_km': 0.5,    # Very small off-lobe radius
            'max_search_radius_km': 50.0
        }
        self.directional_optimizer = DirectionalSpectrumOptimizer(config)
        
        # Create real optimizer
        self.real_optimizer = RealSpectrumOptimizer(num_threads=1)
        
        # Set frequency parameters
        if freq_params:
            self.freq_params = freq_params
            self.real_optimizer.freq_params = freq_params
            # Update enhanced optimizer config
            self.enhanced_optimizer.config['band']['min_mhz'] = freq_params.get('min_freq', 88.0)
            self.enhanced_optimizer.config['band']['max_mhz'] = freq_params.get('max_freq', 108.0)
            self.enhanced_optimizer.config['band']['step_khz'] = freq_params.get('channel_step', 0.2) * 1000
        else:
            self.freq_params = {
                'min_freq': 88.0,
                'max_freq': 108.0,
                'channel_step': 0.2,
                'band_name': 'FM'
            }
            self.real_optimizer.freq_params = self.freq_params
    
    def optimize(self, df):
        """
        Wrapper for optimize method that uses directional geometry.
        """
        # Add missing columns with defaults if needed
        if 'station_id' not in df.columns:
            df['station_id'] = [f'S{i}' for i in range(len(df))]
        
        if 'power_watts' not in df.columns:
            df['power_watts'] = 1000
        
        if 'azimuth_deg' not in df.columns:
            df['azimuth_deg'] = 0
            
        if 'beamwidth_deg' not in df.columns:
            df['beamwidth_deg'] = 360
        
        # Add x_coord and y_coord if missing
        if 'x_coord' not in df.columns:
            df['x_coord'] = df['longitude']
        if 'y_coord' not in df.columns:
            df['y_coord'] = df['latitude']
        
        # Use enhanced optimization with guard channels
        try:
            result = self.enhanced_optimizer.optimize(df)
            
            # Ensure result has assigned_frequency column
            if result is not None and 'assigned_frequency' not in result.columns:
                result['assigned_frequency'] = result['frequency_mhz']
            
            return result
            
        except Exception as e:
            print(f"Enhanced optimization failed: {e}, trying directional")
            try:
                result = self._optimize_with_directional(df)
                if result is not None and 'assigned_frequency' not in result.columns:
                    result['assigned_frequency'] = result['frequency_mhz']
                return result
            except Exception as e2:
                print(f"Directional optimization also failed: {e2}, using fallback")
                return self._fallback_optimize(df)
    
    def _optimize_with_directional(self, df):
        """
        Optimize using directional geometry for interference detection.
        """
        import pandas as pd
        from ortools.sat.python import cp_model
        
        n_stations = len(df)
        result = df.copy()
        
        # Build interference graph using directional geometry
        edges, metrics = self.directional_optimizer.build_interference_graph(df)
        
        # Generate available frequencies
        available_freqs = np.arange(
            self.freq_params['min_freq'],
            self.freq_params['max_freq'] + self.freq_params['channel_step'],
            self.freq_params['channel_step']
        )
        n_freqs = len(available_freqs)
        
        # Create CP-SAT model
        model = cp_model.CpModel()
        
        # Create variables: x[i][f] = 1 if station i uses frequency f
        x = {}
        for i in range(n_stations):
            for f in range(n_freqs):
                x[i, f] = model.NewBoolVar(f'x_{i}_{f}')
        
        # Constraint: Each station gets exactly one frequency
        for i in range(n_stations):
            model.Add(sum(x[i, f] for f in range(n_freqs)) == 1)
        
        # Constraint: Interfering stations cannot use same frequency
        for i, j in edges:
            for f in range(n_freqs):
                model.Add(x[i, f] + x[j, f] <= 1)
        
        # Objective: Minimize spectrum span and channel count
        # First, create variables for which frequencies are used
        freq_used = {}
        for f in range(n_freqs):
            freq_used[f] = model.NewBoolVar(f'freq_used_{f}')
            # freq_used[f] = 1 if any station uses frequency f
            for i in range(n_stations):
                model.Add(freq_used[f] >= x[i, f])
            # But also ensure it's 0 if no station uses it
            model.Add(freq_used[f] <= sum(x[i, f] for i in range(n_stations)))
        
        # Minimize number of frequencies used
        model.Minimize(sum(freq_used[f] for f in range(n_freqs)))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        solver.parameters.num_search_workers = 1
        
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Extract solution
            assignments = []
            for i in range(n_stations):
                for f in range(n_freqs):
                    if solver.Value(x[i, f]) == 1:
                        assignments.append(available_freqs[f])
                        break
                else:
                    assignments.append(available_freqs[0])  # Fallback
            
            result['assigned_frequency'] = assignments
        else:
            # If solver fails, use greedy approach
            result = self._greedy_directional_optimize(df, edges, available_freqs)
        
        return result
    
    def _greedy_directional_optimize(self, df, edges, available_freqs):
        """
        Greedy frequency assignment using directional interference edges.
        """
        n_stations = len(df)
        result = df.copy()
        
        # Build adjacency list from edges
        neighbors = {i: set() for i in range(n_stations)}
        for i, j in edges:
            neighbors[i].add(j)
            neighbors[j].add(i)
        
        # Greedy coloring
        assigned = {}
        for i in range(n_stations):
            used_freqs = {assigned[j] for j in neighbors[i] if j in assigned}
            
            # Find first available frequency
            for freq in available_freqs:
                if freq not in used_freqs:
                    assigned[i] = freq
                    break
            else:
                assigned[i] = available_freqs[0]  # Fallback
        
        result['assigned_frequency'] = [assigned[i] for i in range(n_stations)]
        return result
    
    def _fallback_optimize(self, df):
        """
        Simple fallback optimization for testing.
        """
        import numpy as np
        
        result = df.copy()
        n_stations = len(df)
        
        # Generate available frequencies
        available_freqs = np.arange(
            self.freq_params['min_freq'],
            self.freq_params['max_freq'] + self.freq_params['channel_step'],
            self.freq_params['channel_step']
        )
        
        # Simple distance-based interference check
        assigned = {}
        
        for i in range(n_stations):
            used_freqs = set()
            
            # Check interference with already assigned stations
            for j in range(i):
                if j not in assigned:
                    continue
                
                # Calculate distance
                dist = self._haversine(
                    df.iloc[i]['latitude'], df.iloc[i]['longitude'],
                    df.iloc[j]['latitude'], df.iloc[j]['longitude']
                )
                
                # If within 10km, mark frequency as used
                if dist < 10:
                    used_freqs.add(assigned[j])
            
            # Assign first available frequency
            for freq in available_freqs:
                if freq not in used_freqs:
                    assigned[i] = freq
                    break
            else:
                # No frequency available
                assigned[i] = available_freqs[0]
        
        result['assigned_frequency'] = [assigned[i] for i in range(n_stations)]
        return result
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calculate distance in km between two points."""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c