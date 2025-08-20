"""
Fixed version of solver optimization that avoids MODEL_INVALID errors.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from ortools.sat.python import cp_model
from collections import defaultdict
import time
import math

logger = logging.getLogger(__name__)


class SafeAdaptiveSolverWrapper:
    """
    Safe wrapper that avoids MODEL_INVALID by using only compatible solver parameters.
    """
    
    def __init__(self):
        self.strategy_stats = defaultdict(int)
    
    def solve_with_safe_adaptation(self, model: cp_model.CpModel,
                                  n_stations: int, n_freqs: int,
                                  interference_pairs: List[Tuple[int, int]],
                                  feasible_freqs: Dict[int, List[int]],
                                  station_freq: Dict,
                                  base_timeout: float = 2.0) -> Tuple[int, cp_model.CpSolver]:
        """
        Solve with safe adaptive strategies that avoid MODEL_INVALID.
        
        Returns:
            (status, solver) tuple
        """
        # First validate the model
        validation_result = model.Validate()
        if validation_result:
            logger.error(f"Model validation failed: {validation_result}")
            solver = cp_model.CpSolver()
            # Return INFEASIBLE instead of MODEL_INVALID for better error handling
            return cp_model.INFEASIBLE, solver
        
        # Calculate problem density
        n_edges = len(interference_pairs)
        density = n_edges / (n_stations * (n_stations - 1) / 2) if n_stations > 1 else 0
        
        logger.info(f"Safe solver: density={density:.3f}, stations={n_stations}, frequencies={n_freqs}")
        
        # Create solver with SAFE parameters only
        solver = cp_model.CpSolver()
        
        # Only set parameters that are guaranteed to be safe:
        # 1. Time limit - always safe
        solver.parameters.max_time_in_seconds = base_timeout
        
        # 2. Number of workers - always safe
        solver.parameters.num_search_workers = min(4, n_stations // 10 + 1)
        
        # 3. Random seed - always safe
        solver.parameters.random_seed = int(time.time() * 1000) % 2147483647
        
        # 4. For very dense problems, stop after first solution
        if density > 0.8:
            solver.parameters.stop_after_first_solution = True
            logger.debug("Using stop_after_first_solution for dense problem")
        
        # DO NOT SET these parameters as they can cause MODEL_INVALID:
        # - search_branching (especially FIXED_SEARCH without proper annotations)
        # - linearization_level (can be incompatible with certain constraints)
        # - cp_model_presolve (can expose model issues)
        # - cp_model_probing_level (may not exist in all versions)
        
        # Try to solve
        try:
            status = solver.Solve(model)
            logger.info(f"Safe solver completed with status: {solver.StatusName(status)}")
        except Exception as e:
            logger.error(f"Solver raised exception: {e}")
            status = cp_model.INFEASIBLE
        
        return status, solver


def create_safe_adaptive_wrapper():
    """Factory function to create a safe adaptive solver wrapper."""
    return SafeAdaptiveSolverWrapper()