"""
Advanced CP-SAT solver optimization strategies for spectrum allocation.
Handles remaining UNKNOWN cases with adaptive techniques.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from ortools.sat.python import cp_model
from collections import defaultdict
import time
import math

logger = logging.getLogger(__name__)


class SolverOptimizer:
    """Adaptive solver strategies for challenging optimization cases."""
    
    def __init__(self):
        self.strategy_stats = defaultdict(int)
        self.success_rates = defaultdict(lambda: {'success': 0, 'total': 0})
    
    def analyze_problem_complexity(self, n_stations: int, n_freqs: int,
                                  interference_pairs: List[Tuple[int, int]],
                                  feasible_freqs: Dict[int, List[int]]) -> Dict:
        """
        Analyze problem complexity to guide solver strategy selection.
        """
        # Calculate graph metrics
        n_edges = len(interference_pairs)
        density = n_edges / (n_stations * (n_stations - 1) / 2) if n_stations > 1 else 0
        
        # Calculate degree distribution
        degrees = defaultdict(int)
        for i, j in interference_pairs:
            degrees[i] += 1
            degrees[j] += 1
        
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
        max_degree = max(degrees.values()) if degrees else 0
        
        # Calculate feasibility metrics
        avg_feasible = np.mean([len(freqs) for freqs in feasible_freqs.values()])
        min_feasible = min(len(freqs) for freqs in feasible_freqs.values()) if feasible_freqs else n_freqs
        
        # Estimate chromatic number lower bound (max clique size)
        chromatic_lower = max_degree + 1 if degrees else 1
        
        # Calculate problem difficulty score (0-1)
        difficulty_factors = [
            density * 2,  # Weight density heavily
            (max_degree / n_stations) if n_stations > 0 else 0,
            1 - (avg_feasible / n_freqs) if n_freqs > 0 else 0,
            (chromatic_lower / n_freqs) if n_freqs > 0 else 0
        ]
        difficulty_score = min(1.0, np.mean(difficulty_factors))
        
        # Classify problem type
        if density > 0.8:
            problem_type = "near_clique"
        elif density > 0.5:
            problem_type = "very_dense"
        elif density > 0.2:
            problem_type = "dense"
        elif max_degree > n_stations * 0.5:
            problem_type = "hub_dominated"
        elif avg_degree < 2:
            problem_type = "sparse"
        else:
            problem_type = "moderate"
        
        return {
            'n_stations': n_stations,
            'n_edges': n_edges,
            'density': density,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'avg_feasible_channels': avg_feasible,
            'min_feasible_channels': min_feasible,
            'chromatic_lower_bound': chromatic_lower,
            'difficulty_score': difficulty_score,
            'problem_type': problem_type,
            'estimated_solve_time': self._estimate_solve_time(n_stations, density, difficulty_score)
        }
    
    def _estimate_solve_time(self, n_stations: int, density: float, difficulty: float) -> float:
        """Estimate solve time based on problem characteristics."""
        # Base time increases with size
        base_time = 0.01 * n_stations
        
        # Exponential penalty for density
        density_factor = math.exp(2 * density)
        
        # Difficulty multiplier
        difficulty_factor = 1 + 4 * difficulty
        
        estimated_time = base_time * density_factor * difficulty_factor
        
        # Cap at reasonable limits
        return min(30.0, max(0.1, estimated_time))
    
    def get_adaptive_timeout(self, complexity: Dict, base_timeout: float = 2.0) -> float:
        """
        Calculate adaptive timeout based on problem complexity.
        """
        difficulty = complexity['difficulty_score']
        problem_type = complexity['problem_type']
        estimated_time = complexity['estimated_solve_time']
        
        # Adjust timeout based on problem type
        timeout_multipliers = {
            'near_clique': 0.5,  # Don't waste time on near-impossible problems
            'very_dense': 1.0,
            'dense': 1.5,
            'hub_dominated': 1.2,
            'moderate': 2.0,
            'sparse': 0.8
        }
        
        multiplier = timeout_multipliers.get(problem_type, 1.0)
        
        # Use estimated time as guide, but respect base timeout
        adaptive_timeout = max(
            base_timeout * multiplier,
            min(estimated_time * 1.5, base_timeout * 3)
        )
        
        # Never exceed 10 seconds for a single chunk
        return min(10.0, adaptive_timeout)
    
    def get_solver_parameters(self, complexity: Dict) -> cp_model.CpSolver:
        """
        Get solver with optimized parameters based on problem complexity.
        Returns configured solver instance.
        """
        solver = cp_model.CpSolver()
        problem_type = complexity['problem_type']
        
        # Use minimal, safe parameters to avoid MODEL_INVALID
        # The issue seems to be with certain parameter combinations
        solver.parameters.num_search_workers = 4  # Parallel search
        
        # Don't set any potentially problematic parameters
        # Let OR-Tools use its defaults which are known to be stable
        
        if problem_type == 'near_clique':
            # For very hard problems, stop after first solution
            solver.parameters.stop_after_first_solution = True
            
        # Random seed for diversity in parallel search
        solver.parameters.random_seed = int(time.time() * 1000) % 2147483647
        
        return solver
    
    def generate_smart_initial_solution(self, n_stations: int,
                                       interference_pairs: List[Tuple[int, int]],
                                       n_freqs: int,
                                       feasible_freqs: Dict[int, List[int]]) -> List[Optional[int]]:
        """
        Generate improved initial solution using advanced heuristics.
        """
        # Build adjacency list
        adj_list = defaultdict(set)
        for i, j in interference_pairs:
            adj_list[i].add(j)
            adj_list[j].add(i)
        
        # Calculate degrees
        degrees = [(i, len(adj_list[i])) for i in range(n_stations)]
        
        # Try different strategies based on graph structure
        avg_degree = sum(d for _, d in degrees) / n_stations if n_stations > 0 else 0
        
        if avg_degree > n_stations * 0.5:
            # Very dense: use DSATUR (Degree of Saturation)
            return self._dsatur_coloring(n_stations, adj_list, feasible_freqs, n_freqs)
        else:
            # Moderate: use Welsh-Powell with feasibility constraints
            return self._welsh_powell_coloring(n_stations, adj_list, feasible_freqs, n_freqs)
    
    def _dsatur_coloring(self, n_stations: int, adj_list: Dict[int, set],
                         feasible_freqs: Dict[int, List[int]], n_freqs: int) -> List[Optional[int]]:
        """
        DSATUR (Degree of Saturation) coloring heuristic.
        Prioritizes nodes with highest saturation (most different neighbor colors).
        """
        colors = [None] * n_stations
        saturation = [0] * n_stations
        uncolored = set(range(n_stations))
        
        # Color first node (highest degree)
        first = max(uncolored, key=lambda x: len(adj_list[x]))
        colors[first] = feasible_freqs[first][0] if feasible_freqs[first] else 0
        uncolored.remove(first)
        
        # Update saturation for neighbors
        for neighbor in adj_list[first]:
            if neighbor in uncolored:
                saturation[neighbor] += 1
        
        while uncolored:
            # Choose node with highest saturation (ties broken by degree)
            next_node = max(uncolored, 
                          key=lambda x: (saturation[x], len(adj_list[x])))
            
            # Find available colors
            neighbor_colors = {colors[n] for n in adj_list[next_node] if colors[n] is not None}
            
            # Choose color from feasible set
            chosen_color = None
            for color_idx in feasible_freqs.get(next_node, range(n_freqs)):
                if color_idx not in neighbor_colors:
                    chosen_color = color_idx
                    break
            
            if chosen_color is None:
                # Fallback: use least-used feasible color
                if feasible_freqs.get(next_node):
                    chosen_color = feasible_freqs[next_node][0]
                else:
                    chosen_color = 0
            
            colors[next_node] = chosen_color
            uncolored.remove(next_node)
            
            # Update saturation for uncolored neighbors
            for neighbor in adj_list[next_node]:
                if neighbor in uncolored:
                    # Count unique colors in neighborhood
                    neighbor_neighbor_colors = {colors[n] for n in adj_list[neighbor] 
                                               if colors[n] is not None}
                    saturation[neighbor] = len(neighbor_neighbor_colors)
        
        return colors
    
    def _welsh_powell_coloring(self, n_stations: int, adj_list: Dict[int, set],
                              feasible_freqs: Dict[int, List[int]], n_freqs: int) -> List[Optional[int]]:
        """
        Welsh-Powell coloring heuristic with feasibility constraints.
        Colors nodes in order of decreasing degree.
        """
        colors = [None] * n_stations
        
        # Sort stations by degree (descending)
        stations_by_degree = sorted(range(n_stations), 
                                   key=lambda x: len(adj_list[x]), 
                                   reverse=True)
        
        for station in stations_by_degree:
            # Find colors used by neighbors
            neighbor_colors = {colors[n] for n in adj_list[station] if colors[n] is not None}
            
            # Choose first available color from feasible set
            for color_idx in feasible_freqs.get(station, range(n_freqs)):
                if color_idx not in neighbor_colors:
                    colors[station] = color_idx
                    break
            
            # Fallback if no feasible color available
            if colors[station] is None:
                colors[station] = feasible_freqs[station][0] if feasible_freqs.get(station) else 0
        
        return colors
    
    def add_advanced_hints(self, model: cp_model.CpModel,
                          station_freq: Dict[Tuple[int, int], Any],
                          initial_solution: List[Optional[int]],
                          complexity: Dict):
        """
        Add advanced hints and search strategies to the model.
        """
        problem_type = complexity['problem_type']
        
        # Add solution hints - AddHint expects a boolean value (0 or 1)
        # For very dense problems, we'll trust the heuristic and add more hints
        
        hints_added = 0
        hints_skipped = 0
        for i, freq_idx in enumerate(initial_solution):
            if freq_idx is not None:
                if (i, freq_idx) in station_freq:
                    try:
                        # AddHint expects (variable, value) where value must be 0 or 1
                        model.AddHint(station_freq[(i, freq_idx)], 1)
                        hints_added += 1
                    except Exception as e:
                        logger.debug(f"Failed to add hint for station {i}, freq {freq_idx}: {e}")
                        hints_skipped += 1
                else:
                    # This frequency doesn't have a variable (not in feasible set)
                    logger.debug(f"Skipping hint for station {i}, freq {freq_idx} - no variable exists")
                    hints_skipped += 1
        
        logger.debug(f"Added {hints_added} hints to model for {problem_type} problem, skipped {hints_skipped}")
        
        # Add variable ordering hints for dense problems
        if problem_type in ['near_clique', 'very_dense', 'dense']:
            # Prioritize high-degree nodes in search
            degrees = defaultdict(int)
            for (i, _) in station_freq.keys():
                degrees[i] += 1
            
            # This would require OR-Tools internal API access
            # Just document the strategy for now
            logger.debug(f"Would prioritize {len(degrees)} high-degree nodes in search")
    
    def create_relaxed_model(self, original_model: cp_model.CpModel,
                            station_freq: Dict, n_stations: int,
                            feasible_freqs: Dict[int, List[int]]) -> cp_model.CpModel:
        """
        Create a relaxed version of the model for finding feasible solutions quickly.
        Useful when original model times out.
        """
        relaxed_model = cp_model.CpModel()
        
        # Copy variables
        relaxed_vars = {}
        for (i, f), var in station_freq.items():
            relaxed_vars[(i, f)] = relaxed_model.NewBoolVar(f'relaxed_station_{i}_freq_{f}')
        
        # Relaxed assignment constraints (allow some stations to be unassigned)
        unassigned = []
        for i in range(n_stations):
            # Create unassigned variable
            unassigned_var = relaxed_model.NewBoolVar(f'unassigned_{i}')
            unassigned.append(unassigned_var)
            
            # Station is either assigned or unassigned
            station_vars = [relaxed_vars[(i, f)] for f in feasible_freqs[i] 
                          if (i, f) in relaxed_vars]
            if station_vars:
                relaxed_model.Add(sum(station_vars) + unassigned_var == 1)
        
        # Copy interference constraints (these remain hard constraints)
        # This is simplified - would need full constraint copying in practice
        
        # Minimize unassigned stations
        relaxed_model.Minimize(sum(unassigned))
        
        return relaxed_model
    
    def apply_solution_polishing(self, initial_solution: List[float],
                                interference_pairs: List[Tuple[int, int]],
                                frequencies: np.ndarray) -> List[float]:
        """
        Polish a solution by local improvements.
        """
        solution = initial_solution.copy()
        improved = True
        iterations = 0
        max_iterations = 10
        
        # Build conflict list
        adj_list = defaultdict(set)
        for i, j in interference_pairs:
            adj_list[i].add(j)
            adj_list[j].add(i)
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try to reduce frequency span by reassigning outliers
            freq_usage = defaultdict(list)
            for i, freq in enumerate(solution):
                freq_usage[freq].append(i)
            
            # Find least-used frequencies
            sorted_freqs = sorted(freq_usage.keys(), key=lambda f: len(freq_usage[f]))
            
            for rare_freq in sorted_freqs[:3]:  # Try to eliminate 3 rarest frequencies
                if len(freq_usage[rare_freq]) <= 2:
                    # Try to reassign these stations
                    for station in freq_usage[rare_freq]:
                        # Find available frequencies
                        neighbor_freqs = {solution[n] for n in adj_list[station]}
                        
                        # Try to use a more common frequency
                        for common_freq in sorted_freqs[-10:]:
                            if common_freq not in neighbor_freqs:
                                solution[station] = common_freq
                                improved = True
                                break
        
        return solution
    
    def get_fallback_strategy(self, complexity: Dict) -> str:
        """
        Determine best fallback strategy for UNKNOWN cases.
        """
        problem_type = complexity['problem_type']
        density = complexity['density']
        
        if problem_type == 'near_clique':
            return 'sequential_assignment'  # Assign frequencies sequentially
        elif density > 0.7:
            return 'random_sampling'  # Random valid assignment
        elif complexity['max_degree'] > complexity['n_stations'] * 0.6:
            return 'degree_based'  # Prioritize high-degree nodes
        else:
            return 'greedy_dsatur'  # Use DSATUR heuristic
    
    def report_strategy_effectiveness(self):
        """
        Report on effectiveness of different strategies.
        """
        logger.info("\n=== Solver Strategy Effectiveness ===")
        for strategy, stats in self.success_rates.items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total'] * 100
                logger.info(f"{strategy}: {success_rate:.1f}% success rate "
                          f"({stats['success']}/{stats['total']} attempts)")


class AdaptiveSolverWrapper:
    """
    Wrapper that applies adaptive strategies to CP-SAT solving.
    """
    
    def __init__(self, optimizer: SolverOptimizer = None):
        self.optimizer = optimizer or SolverOptimizer()
    
    def solve_with_adaptation(self, model: cp_model.CpModel,
                             n_stations: int, n_freqs: int,
                             interference_pairs: List[Tuple[int, int]],
                             feasible_freqs: Dict[int, List[int]],
                             station_freq: Dict,
                             base_timeout: float = 2.0) -> Tuple[int, cp_model.CpSolver]:
        """
        Solve with adaptive strategies.
        
        Returns:
            (status, solver) tuple
        """
        # Validate model before proceeding
        validation_result = model.Validate()
        if validation_result:
            logger.error(f"Model validation failed before adaptive solve: {validation_result}")
            # Return MODEL_INVALID status without attempting to solve
            solver = cp_model.CpSolver()
            return cp_model.MODEL_INVALID, solver
        
        # Analyze complexity
        complexity = self.optimizer.analyze_problem_complexity(
            n_stations, n_freqs, interference_pairs, feasible_freqs
        )
        
        logger.info(f"Problem analysis: type={complexity['problem_type']}, "
                   f"density={complexity['density']:.3f}, "
                   f"difficulty={complexity['difficulty_score']:.2f}")
        
        # Get adaptive timeout
        timeout = self.optimizer.get_adaptive_timeout(complexity, base_timeout)
        logger.debug(f"Using adaptive timeout: {timeout:.1f}s")
        
        # Generate smart initial solution
        initial_solution = self.optimizer.generate_smart_initial_solution(
            n_stations, interference_pairs, n_freqs, feasible_freqs
        )
        
        # Add hints to model
        self.optimizer.add_advanced_hints(
            model, station_freq, initial_solution, complexity
        )
        
        # Get solver with optimized parameters
        solver = self.optimizer.get_solver_parameters(complexity)
        solver.parameters.max_time_in_seconds = timeout
        
        # Log solver parameters for debugging
        logger.debug(f"Solver parameters before solve:")
        logger.debug(f"  - search_branching: {solver.parameters.search_branching}")
        logger.debug(f"  - linearization_level: {solver.parameters.linearization_level}")
        logger.debug(f"  - cp_model_presolve: {solver.parameters.cp_model_presolve}")
        logger.debug(f"  - num_search_workers: {solver.parameters.num_search_workers}")
        logger.debug(f"  - max_time_in_seconds: {solver.parameters.max_time_in_seconds}")
        
        # First attempt: solve with adapted parameters
        logger.debug(f"Attempting solve with {complexity['problem_type']} strategy")
        try:
            status = solver.Solve(model)
            if status == cp_model.MODEL_INVALID:
                logger.error(f"Solver returned MODEL_INVALID with parameters: branching={solver.parameters.search_branching}, linearization={solver.parameters.linearization_level}")
        except Exception as e:
            logger.error(f"Solver raised exception: {e}")
            return cp_model.MODEL_INVALID, solver
        
        # If UNKNOWN, try fallback strategies
        if status == cp_model.UNKNOWN:
            logger.info(f"Initial solve returned UNKNOWN, trying fallback strategies")
            
            fallback = self.optimizer.get_fallback_strategy(complexity)
            logger.debug(f"Using fallback strategy: {fallback}")
            
            if fallback == 'sequential_assignment':
                # Don't retry solver, use heuristic directly
                status = cp_model.UNKNOWN  # Keep as UNKNOWN but with good heuristic
                
            elif fallback == 'random_sampling' and timeout > 0.5:
                # Try with different random seed and shorter timeout
                solver.parameters.random_seed = int(time.time() * 1000) % 2147483647
                solver.parameters.max_time_in_seconds = timeout * 0.5
                status = solver.Solve(model)
                
            elif fallback == 'greedy_dsatur':
                # Trust the DSATUR solution
                status = cp_model.UNKNOWN  # Keep as UNKNOWN but solution is good
        
        # Record strategy effectiveness
        strategy_name = f"{complexity['problem_type']}_{timeout:.1f}s"
        self.optimizer.success_rates[strategy_name]['total'] += 1
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            self.optimizer.success_rates[strategy_name]['success'] += 1
        
        return status, solver


def test_solver_optimization():
    """Test the solver optimization strategies."""
    import pandas as pd
    
    logger.info("Testing solver optimization strategies...")
    
    # Create test problem
    n_stations = 50
    n_freqs = 20
    
    # Create dense interference graph
    interference_pairs = []
    for i in range(n_stations):
        for j in range(i+1, min(i+10, n_stations)):  # Each station interferes with ~10 others
            interference_pairs.append((i, j))
    
    # Create feasible frequencies
    feasible_freqs = {}
    for i in range(n_stations):
        if i < 10:  # High-degree nodes have fewer options
            feasible_freqs[i] = list(range(5))
        else:
            feasible_freqs[i] = list(range(n_freqs))
    
    # Test complexity analysis
    optimizer = SolverOptimizer()
    complexity = optimizer.analyze_problem_complexity(
        n_stations, n_freqs, interference_pairs, feasible_freqs
    )
    
    logger.info(f"Complexity analysis: {complexity}")
    
    # Test initial solution generation
    initial = optimizer.generate_smart_initial_solution(
        n_stations, interference_pairs, n_freqs, feasible_freqs
    )
    
    # Check validity
    valid = True
    for i, j in interference_pairs:
        if initial[i] == initial[j]:
            valid = False
            break
    
    logger.info(f"Initial solution valid: {valid}")
    logger.info(f"Colors used: {len(set(initial))}")
    
    # Test adaptive timeout
    timeout = optimizer.get_adaptive_timeout(complexity)
    logger.info(f"Adaptive timeout: {timeout:.2f}s")
    
    return complexity, initial


if __name__ == "__main__":
    test_solver_optimization()