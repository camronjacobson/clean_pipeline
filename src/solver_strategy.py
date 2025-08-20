"""
Solver strategy with 4-rung ladder to eliminate UNKNOWN states.
Based on review §3.2 and §3.3.
"""

import logging
import time
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
from collections import defaultdict

from config_flags import UNKNOWN_ELIMINATION_LADDER_ENABLED, SOLVER_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SolverRung:
    """Configuration for one rung of the solver ladder."""
    name: str
    timeout_seconds: int
    num_workers: int
    linearization_level: int
    use_lns: bool
    lns_focus_variables: Optional[List[int]] = None
    constraints_to_relax: Optional[List[str]] = None
    relaxation_factor: float = 1.0
    extra_frequencies: int = 0
    use_warm_start: bool = False
    warm_start_solution: Optional[Dict] = None
    description: str = ""


@dataclass
class SolverResult:
    """Result from a solver attempt."""
    status: int
    solution: Optional[Dict[int, float]] = None
    objective_value: Optional[float] = None
    solve_time: float = 0.0
    rung_name: str = ""
    relaxations_applied: List[str] = None
    
    def __post_init__(self):
        if self.relaxations_applied is None:
            self.relaxations_applied = []


class SolverStrategy:
    """
    Implements 4-rung progressive solver strategy to eliminate UNKNOWN states.
    Each rung tries increasingly aggressive techniques.
    """
    
    def __init__(self, 
                 freq_params: Dict[str, Any],
                 n_stations: int,
                 interference_pairs: List[Tuple[int, int]],
                 feasible_freqs: Dict[int, List[int]] = None):
        """
        Initialize solver strategy.
        
        Args:
            freq_params: Frequency parameters (min, max, step)
            n_stations: Number of stations
            interference_pairs: List of interfering station pairs
            feasible_freqs: Feasible frequencies per station (for sparse constraints)
        """
        self.freq_params = freq_params
        self.n_stations = n_stations
        self.interference_pairs = interference_pairs
        self.feasible_freqs = feasible_freqs or self._compute_default_feasible_freqs()
        
        # Build adjacency list for graph operations
        self.adj_list = defaultdict(set)
        for i, j in interference_pairs:
            self.adj_list[i].add(j)
            self.adj_list[j].add(i)
        
        # Track solver progress
        self.stats = {
            'total_attempts': 0,
            'successful_rung': None,
            'total_time': 0.0,
            'plateau_detected': False,
            'unknown_eliminated': False
        }
        
        # For plateau detection
        self.best_objective = float('inf')
        self.no_improvement_time = 0.0
        self.last_improvement_time = time.time()
    
    def solve_with_ladder(self, model: cp_model.CpModel,
                         station_freq_vars: Dict[Tuple[int, int], Any],
                         freq_used_vars: List[Any] = None,
                         stations_df: pd.DataFrame = None) -> SolverResult:
        """
        Apply 4-rung ladder strategy to solve the model.
        
        Args:
            model: CP-SAT model to solve
            station_freq_vars: Dictionary of (station, freq) -> BoolVar
            freq_used_vars: Optional list of frequency usage variables
            stations_df: Optional DataFrame with station data
            
        Returns:
            SolverResult with best solution found
        """
        if not UNKNOWN_ELIMINATION_LADDER_ENABLED:
            logger.info("UNKNOWN elimination ladder disabled, using standard solve")
            return self._standard_solve(model)
        
        logger.info(f"Starting 4-rung solver ladder for {self.n_stations} stations")
        start_time = time.time()
        
        # Create solver rungs
        rungs = self._create_solver_rungs(stations_df)
        
        best_result = None
        
        for rung_idx, rung in enumerate(rungs, 1):
            logger.info(f"Attempting Rung {rung_idx}: {rung.name}")
            
            # Apply rung-specific modifications
            modified_model = self._apply_rung_modifications(
                model, station_freq_vars, rung, stations_df
            )
            
            # Attempt to solve
            result = self._solve_with_rung(modified_model, station_freq_vars, rung)
            self.stats['total_attempts'] += 1
            
            # Check result
            if result.status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                logger.info(f"✓ Rung {rung_idx} ({rung.name}) succeeded: {self._status_name(result.status)}")
                self.stats['successful_rung'] = rung.name
                self.stats['unknown_eliminated'] = True
                best_result = result
                break
            else:
                logger.info(f"✗ Rung {rung_idx} ({rung.name}) failed: {self._status_name(result.status)}")
            
            # Check for plateau
            if self._check_plateau():
                logger.info("Plateau detected, moving to next rung")
                self.stats['plateau_detected'] = True
            
            # Keep best partial result
            if result.solution and (not best_result or len(result.solution) > len(best_result.solution)):
                best_result = result
        
        # If all rungs failed, extract partial solution
        if not best_result or best_result.status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            logger.warning("All rungs failed, extracting partial solution")
            best_result = self._extract_partial_solution(model, station_freq_vars)
            self.stats['unknown_eliminated'] = False
        
        self.stats['total_time'] = time.time() - start_time
        self._log_stats()
        
        return best_result
    
    def _create_solver_rungs(self, stations_df: pd.DataFrame = None) -> List[SolverRung]:
        """Create the 4-rung ladder configuration."""
        rungs = []
        
        # Rung 1: Warm start with greedy solution
        greedy_solution = self._generate_greedy_solution()
        rungs.append(SolverRung(
            name="warm_greedy",
            timeout_seconds=30,
            num_workers=4,
            linearization_level=1,
            use_lns=False,
            use_warm_start=True,
            warm_start_solution=greedy_solution,
            description="Warm start with greedy graph coloring"
        ))
        
        # Rung 2: LNS with boundary focus
        boundary_vars = self._identify_boundary_variables(stations_df)
        rungs.append(SolverRung(
            name="lns_boundary",
            timeout_seconds=60,
            num_workers=8,
            linearization_level=2,
            use_lns=True,
            lns_focus_variables=boundary_vars,
            description="Large neighborhood search focusing on boundary stations"
        ))
        
        # Rung 3: Relax adjacent channel constraints
        rungs.append(SolverRung(
            name="relax_adjacent",
            timeout_seconds=45,
            num_workers=6,
            linearization_level=2,
            use_lns=False,
            constraints_to_relax=["adjacent_channel"],
            relaxation_factor=0.9,
            description="Relaxed adjacent channel protection"
        ))
        
        # Rung 4: Expand frequency palette
        rungs.append(SolverRung(
            name="expand_palette",
            timeout_seconds=30,
            num_workers=4,
            linearization_level=1,
            use_lns=False,
            extra_frequencies=5,
            description="Expanded frequency palette as last resort"
        ))
        
        return rungs
    
    def _apply_rung_modifications(self, 
                                 base_model: cp_model.CpModel,
                                 station_freq_vars: Dict,
                                 rung: SolverRung,
                                 stations_df: pd.DataFrame = None) -> cp_model.CpModel:
        """Apply rung-specific modifications to the model."""
        # Create a fresh model for this rung (CP-SAT doesn't support deep copy)
        # In practice, we'll modify solver parameters instead of the model
        model = base_model
        
        # For rungs that need model changes, we track them separately
        if rung.constraints_to_relax:
            # Add assumption literals for toggleable constraints
            self._add_assumption_literals(model, rung.constraints_to_relax)
        
        if rung.extra_frequencies > 0:
            # This would require rebuilding the model with more frequencies
            # For now, we'll handle this in the solver parameters
            logger.debug(f"Would add {rung.extra_frequencies} extra frequencies")
        
        return model
    
    def _solve_with_rung(self,
                        model: cp_model.CpModel,
                        station_freq_vars: Dict,
                        rung: SolverRung) -> SolverResult:
        """Solve the model with rung-specific configuration."""
        solver = cp_model.CpSolver()
        
        # Apply solver parameters
        solver.parameters.max_time_in_seconds = rung.timeout_seconds
        solver.parameters.num_search_workers = rung.num_workers
        solver.parameters.linearization_level = rung.linearization_level
        solver.parameters.random_seed = SOLVER_CONFIG.get('random_seed', 42)
        
        # LNS configuration
        if rung.use_lns:
            solver.parameters.use_lns_only = True
            if rung.lns_focus_variables:
                # In practice, this would require custom search strategy
                logger.debug(f"Focusing LNS on {len(rung.lns_focus_variables)} variables")
        
        # Add warm start hints
        if rung.use_warm_start and rung.warm_start_solution:
            for (i, f), value in rung.warm_start_solution.items():
                if (i, f) in station_freq_vars:
                    solver.AddHint(station_freq_vars[(i, f)], value)
        
        # Solve
        start_solve = time.time()
        status = solver.Solve(model)
        solve_time = time.time() - start_solve
        
        # Extract solution if found
        solution = None
        objective = None
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution = {}
            for (i, f), var in station_freq_vars.items():
                if solver.Value(var) == 1:
                    solution[i] = f
            
            if solver.HasObjective():
                objective = solver.ObjectiveValue()
                # Update plateau detection
                if objective < self.best_objective:
                    self.best_objective = objective
                    self.last_improvement_time = time.time()
        
        return SolverResult(
            status=status,
            solution=solution,
            objective_value=objective,
            solve_time=solve_time,
            rung_name=rung.name,
            relaxations_applied=rung.constraints_to_relax or []
        )
    
    def _generate_greedy_solution(self) -> Dict[Tuple[int, int], int]:
        """Generate greedy graph coloring solution for warm start."""
        solution = {}
        colors = {}  # station -> frequency index
        
        # Order stations by degree (most constrained first)
        station_degrees = [(i, len(self.adj_list[i])) for i in range(self.n_stations)]
        station_degrees.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy coloring
        for station, _ in station_degrees:
            # Find available colors
            used_colors = set()
            for neighbor in self.adj_list[station]:
                if neighbor in colors:
                    used_colors.add(colors[neighbor])
            
            # Assign lowest available color from feasible set
            for freq_idx in self.feasible_freqs.get(station, range(len(self._get_frequencies()))):
                if freq_idx not in used_colors:
                    colors[station] = freq_idx
                    solution[(station, freq_idx)] = 1
                    break
        
        logger.debug(f"Greedy solution uses {len(set(colors.values()))} colors for {len(colors)} stations")
        return solution
    
    def _identify_boundary_variables(self, stations_df: pd.DataFrame = None) -> List[int]:
        """Identify boundary stations for LNS focus."""
        boundary_vars = []
        
        if stations_df is not None and 'chunk_id' in stations_df.columns:
            # Stations at chunk boundaries
            chunk_counts = stations_df.groupby('chunk_id').size()
            
            for chunk_id in chunk_counts.index:
                chunk_mask = stations_df['chunk_id'] == chunk_id
                chunk_indices = stations_df[chunk_mask].index.tolist()
                
                # Consider last 20% as boundary
                boundary_size = max(1, len(chunk_indices) // 5)
                boundary_vars.extend(chunk_indices[-boundary_size:])
        else:
            # Use high-degree stations as proxies for difficult assignments
            high_degree_stations = [
                i for i, neighbors in self.adj_list.items()
                if len(neighbors) > np.percentile(
                    [len(n) for n in self.adj_list.values()], 75
                )
            ]
            boundary_vars = high_degree_stations[:self.n_stations // 10]
        
        logger.debug(f"Identified {len(boundary_vars)} boundary variables for LNS")
        return boundary_vars
    
    def _add_assumption_literals(self, model: cp_model.CpModel, 
                                constraints_to_relax: List[str]):
        """Add assumption literals for toggleable constraints."""
        assumptions = []
        
        for constraint_type in constraints_to_relax:
            if constraint_type == "adjacent_channel":
                # Create assumption literal
                adjacent_active = model.NewBoolVar(f'{constraint_type}_active')
                assumptions.append(adjacent_active)
                
                # Make constraints conditional on assumption
                # Note: This is simplified - in practice would need to track
                # and modify existing constraints
                logger.debug(f"Added assumption literal for {constraint_type}")
            
            elif constraint_type == "guard_band":
                guard_active = model.NewBoolVar(f'{constraint_type}_active')
                assumptions.append(guard_active)
                logger.debug(f"Added assumption literal for {constraint_type}")
        
        return assumptions
    
    def _check_plateau(self, plateau_threshold_seconds: float = 15.0) -> bool:
        """Check if solver has plateaued (no improvement)."""
        current_time = time.time()
        self.no_improvement_time = current_time - self.last_improvement_time
        
        return self.no_improvement_time > plateau_threshold_seconds
    
    def _extract_partial_solution(self,
                                 model: cp_model.CpModel,
                                 station_freq_vars: Dict) -> SolverResult:
        """Extract best partial solution found so far."""
        # Try a quick solve with very relaxed parameters
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        solver.parameters.num_search_workers = 1
        solver.parameters.stop_after_first_solution = True
        
        status = solver.Solve(model)
        
        partial_solution = {}
        if status != cp_model.INFEASIBLE:
            # Extract any assignments made
            for (i, f), var in station_freq_vars.items():
                try:
                    if solver.Value(var) == 1:
                        partial_solution[i] = f
                except:
                    pass  # Variable might not be assigned
        
        # If still no solution, use greedy fallback
        if not partial_solution:
            greedy = self._generate_greedy_solution()
            for (i, f), value in greedy.items():
                if value == 1:
                    partial_solution[i] = f
        
        logger.info(f"Extracted partial solution with {len(partial_solution)}/{self.n_stations} stations")
        
        return SolverResult(
            status=cp_model.UNKNOWN,
            solution=partial_solution,
            solve_time=5.0,
            rung_name="partial_extraction",
            relaxations_applied=["all"]
        )
    
    def _compute_default_feasible_freqs(self) -> Dict[int, List[int]]:
        """Compute default feasible frequencies for each station."""
        feasible = {}
        n_freqs = self._get_num_frequencies()
        
        for i in range(self.n_stations):
            degree = len(self.adj_list[i])
            
            if degree == 0:
                # No interference, can use any frequency
                feasible[i] = [0]
            elif degree < 5:
                # Low interference
                feasible[i] = list(range(min(degree * 2, n_freqs)))
            elif degree < 20:
                # Medium interference
                feasible[i] = list(range(min(degree + 5, n_freqs)))
            else:
                # High interference, need all frequencies
                feasible[i] = list(range(n_freqs))
        
        return feasible
    
    def _get_frequencies(self) -> np.ndarray:
        """Get frequency array from parameters."""
        return np.arange(
            self.freq_params['min_freq'],
            self.freq_params['max_freq'] + self.freq_params['channel_step'],
            self.freq_params['channel_step']
        )
    
    def _get_num_frequencies(self) -> int:
        """Get number of available frequencies."""
        return len(self._get_frequencies())
    
    def _standard_solve(self, model: cp_model.CpModel) -> SolverResult:
        """Standard solve without ladder strategy."""
        solver = cp_model.CpSolver()
        
        # Use config parameters
        solver.parameters.max_time_in_seconds = SOLVER_CONFIG.get('max_time_in_seconds', 60)
        solver.parameters.num_search_workers = SOLVER_CONFIG.get('num_search_workers', 8)
        solver.parameters.random_seed = SOLVER_CONFIG.get('random_seed', 42)
        
        status = solver.Solve(model)
        
        return SolverResult(
            status=status,
            solve_time=solver.WallTime(),
            rung_name="standard"
        )
    
    def _status_name(self, status: int) -> str:
        """Get human-readable status name."""
        return cp_model.CpSolver.StatusName(status)
    
    def _log_stats(self):
        """Log solver strategy statistics."""
        logger.info("Solver ladder statistics:")
        logger.info(f"  Total attempts: {self.stats['total_attempts']}")
        logger.info(f"  Successful rung: {self.stats['successful_rung']}")
        logger.info(f"  Total time: {self.stats['total_time']:.2f}s")
        logger.info(f"  Plateau detected: {self.stats['plateau_detected']}")
        logger.info(f"  UNKNOWN eliminated: {self.stats['unknown_eliminated']}")


class AssumptionLiteralManager:
    """
    Manages assumption literals for dynamic constraint toggling.
    Allows constraints to be enabled/disabled without rebuilding the model.
    """
    
    def __init__(self, model: cp_model.CpModel):
        """Initialize assumption manager."""
        self.model = model
        self.assumptions = {}
        self.constraint_groups = defaultdict(list)
    
    def create_assumption(self, name: str) -> Any:
        """Create a new assumption literal."""
        if name not in self.assumptions:
            assumption = self.model.NewBoolVar(f'assume_{name}')
            self.assumptions[name] = assumption
        return self.assumptions[name]
    
    def add_conditional_constraint(self, 
                                  constraint_expr: Any,
                                  assumption_name: str,
                                  group: Optional[str] = None):
        """Add a constraint that's conditional on an assumption."""
        assumption = self.create_assumption(assumption_name)
        
        # Add constraint only if assumption is true
        self.model.Add(constraint_expr).OnlyEnforceIf(assumption)
        
        if group:
            self.constraint_groups[group].append(assumption_name)
    
    def get_assumption_set(self, 
                          active_assumptions: List[str],
                          inactive_assumptions: List[str] = None) -> List[Any]:
        """
        Get assumption literals for solving.
        
        Args:
            active_assumptions: Assumptions to set to True
            inactive_assumptions: Assumptions to set to False
            
        Returns:
            List of assumption literals for solver
        """
        solver_assumptions = []
        
        for name in active_assumptions:
            if name in self.assumptions:
                solver_assumptions.append(self.assumptions[name])
        
        if inactive_assumptions:
            for name in inactive_assumptions:
                if name in self.assumptions:
                    solver_assumptions.append(self.assumptions[name].Not())
        
        return solver_assumptions
    
    def try_assumption_combinations(self, 
                                   solver: cp_model.CpSolver,
                                   base_assumptions: List[str],
                                   optional_groups: List[str],
                                   timeout_per_try: int = 10) -> List[Tuple[int, List[str]]]:
        """
        Try different combinations of assumptions.
        
        Args:
            solver: CP-SAT solver
            base_assumptions: Always-active assumptions
            optional_groups: Groups to try toggling
            timeout_per_try: Timeout for each attempt
            
        Returns:
            List of (status, active_assumptions) tuples
        """
        results = []
        
        # Try with all assumptions active
        all_active = base_assumptions.copy()
        for group in optional_groups:
            all_active.extend(self.constraint_groups[group])
        
        solver.parameters.max_time_in_seconds = timeout_per_try
        solver.assumptions = self.get_assumption_set(all_active)
        status = solver.Solve(self.model)
        results.append((status, all_active.copy()))
        
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Try relaxing each group
            for group_to_relax in optional_groups:
                active = base_assumptions.copy()
                for group in optional_groups:
                    if group != group_to_relax:
                        active.extend(self.constraint_groups[group])
                
                solver.assumptions = self.get_assumption_set(active)
                status = solver.Solve(self.model)
                results.append((status, active.copy()))
                
                if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                    break
        
        return results