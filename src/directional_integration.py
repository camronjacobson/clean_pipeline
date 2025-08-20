"""
Integration module to connect directional geometry with spectrum optimizer.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# Add tool to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tool import (
    DirectionalGeometry,
    DirectionalConfig,
    NeighborDiscovery,
    create_neighbor_discovery
)

logger = logging.getLogger(__name__)


class DirectionalSpectrumOptimizer:
    """
    Enhanced spectrum optimizer with directional geometry support.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with configuration.
        
        Args:
            config: Configuration dictionary with optional keys:
                - az_tolerance_deg: Angular tolerance for lobe calculations
                - r_main_km: Main lobe interference radius
                - r_off_km: Off-lobe interference radius
                - max_search_radius_km: Maximum search radius for KDTree
        """
        self.config = config or {}
        
        # Set default values
        self.config.setdefault('az_tolerance_deg', 5.0)
        self.config.setdefault('r_main_km', 30.0)
        self.config.setdefault('r_off_km', 10.0)
        self.config.setdefault('max_search_radius_km', 50.0)
        
        # Create neighbor discovery with directional geometry
        self.neighbor_discovery = create_neighbor_discovery(self.config)
        
        logger.info(f"DirectionalSpectrumOptimizer initialized with config: {self.config}")
    
    def build_interference_graph(self, stations_df: pd.DataFrame) -> Tuple[List[Tuple[int, int]], Dict]:
        """
        Build interference graph using directional geometry.
        
        Args:
            stations_df: DataFrame with station data
            
        Returns:
            (edges, metrics) where edges is list of (i, j) interference pairs
            and metrics contains neighbor statistics
        """
        logger.info(f"Building interference graph for {len(stations_df)} stations")
        
        # Ensure required columns exist
        required_cols = ['latitude', 'longitude']
        if not all(col in stations_df.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns: {required_cols}")
        
        # Add default azimuth/beamwidth if missing
        if 'azimuth_deg' not in stations_df.columns:
            stations_df['azimuth_deg'] = 0
        if 'beamwidth_deg' not in stations_df.columns:
            stations_df['beamwidth_deg'] = 360
        
        # Build graph with directional geometry
        edges = self.neighbor_discovery.build_interference_graph(
            stations_df, use_directional=True
        )
        
        # Get statistics
        stats = self.neighbor_discovery.stats
        complexity = self.neighbor_discovery.get_complexity_analysis()
        cache_stats = self.neighbor_discovery.directional.get_cache_stats()
        
        metrics = {
            'total_stations': stats.total_stations,
            'total_edges': len(edges),
            'avg_neighbors': stats.avg_neighbors,
            'max_neighbors': stats.max_neighbors,
            'min_neighbors': stats.min_neighbors,
            'complexity_class': complexity['complexity_class'],
            'edge_density': complexity['edge_density'],
            'speedup_vs_all_pairs': complexity['speedup_vs_all_pairs'],
            'cache_hit_rate': cache_stats['hit_rate'],
            'directional_config': {
                'r_main_km': self.config['r_main_km'],
                'r_off_km': self.config['r_off_km'],
                'az_tolerance_deg': self.config['az_tolerance_deg']
            }
        }
        
        logger.info(f"Graph built: {len(edges)} edges, avg_neighbors={stats.avg_neighbors:.1f}, "
                   f"complexity={complexity['complexity_class']}")
        
        return edges, metrics
    
    def optimize_with_directional(self, stations_df: pd.DataFrame,
                                 freq_params: Dict) -> pd.DataFrame:
        """
        Run optimization with directional interference.
        
        Args:
            stations_df: Input station data
            freq_params: Frequency parameters (min_freq, max_freq, channel_step)
            
        Returns:
            DataFrame with assigned frequencies
        """
        # Build interference graph
        edges, metrics = self.build_interference_graph(stations_df)
        
        # Import the actual optimizer
        from spectrum_optimizer import SpectrumOptimizer
        
        # Create optimizer instance
        optimizer = SpectrumOptimizer(num_threads=1)
        optimizer.freq_params = freq_params
        
        # Replace the interference detection with our directional version
        # This is done by monkey-patching the method
        original_method = optimizer.detect_interference
        
        def directional_interference(df, i, j):
            """Enhanced interference detection using directional geometry."""
            station1 = {
                'latitude': df.iloc[i]['latitude'],
                'longitude': df.iloc[i]['longitude'],
                'azimuth_deg': df.iloc[i].get('azimuth_deg', 0),
                'beamwidth_deg': df.iloc[i].get('beamwidth_deg', 360),
                'station_id': i
            }
            station2 = {
                'latitude': df.iloc[j]['latitude'],
                'longitude': df.iloc[j]['longitude'],
                'azimuth_deg': df.iloc[j].get('azimuth_deg', 0),
                'beamwidth_deg': df.iloc[j].get('beamwidth_deg', 360),
                'station_id': j
            }
            
            interferes, _ = self.neighbor_discovery.directional.check_directional_interference(
                station1, station2
            )
            return interferes
        
        # Patch the method
        optimizer.detect_interference = directional_interference
        
        # Run optimization
        result = optimizer.optimize_spectrum(stations_df)
        
        # Add metrics to result
        if result is not None:
            result.attrs = getattr(result, 'attrs', {})
            result.attrs['directional_metrics'] = metrics
        
        return result


def integrate_directional_into_optimizer(spectrum_optimizer_instance, config: Optional[Dict] = None):
    """
    Integrate directional geometry into existing SpectrumOptimizer instance.
    
    Args:
        spectrum_optimizer_instance: Instance of SpectrumOptimizer
        config: Directional configuration
    """
    # Create directional components
    directional_opt = DirectionalSpectrumOptimizer(config)
    
    # Store original method
    original_build_graph = getattr(
        spectrum_optimizer_instance, 
        'build_interference_graph',
        None
    )
    
    # Replace with directional version
    def enhanced_build_graph(df):
        """Enhanced graph building with directional geometry."""
        edges, metrics = directional_opt.build_interference_graph(df)
        
        # Store metrics for later use
        if not hasattr(spectrum_optimizer_instance, 'directional_metrics'):
            spectrum_optimizer_instance.directional_metrics = {}
        spectrum_optimizer_instance.directional_metrics.update(metrics)
        
        return edges
    
    # Patch the instance
    spectrum_optimizer_instance.build_interference_graph = enhanced_build_graph
    spectrum_optimizer_instance.directional_optimizer = directional_opt
    
    logger.info("Directional geometry integrated into SpectrumOptimizer")
    
    return spectrum_optimizer_instance