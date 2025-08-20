"""
KDTree-based neighbor discovery with directional geometry support.
Ensures O(n log n) complexity, not O(n²).
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict

from .directional import DirectionalGeometry, DirectionalConfig

logger = logging.getLogger(__name__)


@dataclass
class NeighborStats:
    """Statistics about neighbor discovery."""
    total_stations: int = 0
    total_neighbors: int = 0
    avg_neighbors: float = 0.0
    max_neighbors: int = 0
    min_neighbors: int = 0
    neighbor_histogram: Dict[int, int] = None
    
    def __post_init__(self):
        if self.neighbor_histogram is None:
            self.neighbor_histogram = {}


class NeighborDiscovery:
    """
    Efficient neighbor discovery using KDTree with directional geometry.
    """
    
    def __init__(self, directional_geometry: Optional[DirectionalGeometry] = None,
                 max_search_radius_km: float = 50.0):
        """
        Initialize neighbor discovery.
        
        Args:
            directional_geometry: DirectionalGeometry instance for lobe calculations
            max_search_radius_km: Maximum search radius for KDTree queries
        """
        self.directional = directional_geometry or DirectionalGeometry()
        self.max_search_radius = max_search_radius_km
        
        # Statistics
        self.stats = NeighborStats()
        
        logger.info(f"NeighborDiscovery initialized with max_search_radius={max_search_radius_km}km")
    
    def find_neighbors(self, stations_df: pd.DataFrame, 
                      use_directional: bool = True) -> Dict[int, Set[int]]:
        """
        Find neighbors for all stations using KDTree.
        
        Args:
            stations_df: DataFrame with columns:
                - latitude, longitude
                - azimuth_deg, beamwidth_deg (optional)
                - station_id or index used as ID
            use_directional: Whether to use directional geometry
            
        Returns:
            Dictionary mapping station index to set of neighbor indices
        """
        n_stations = len(stations_df)
        logger.info(f"Finding neighbors for {n_stations} stations using KDTree")
        
        # Prepare coordinates for KDTree (convert to radians for better accuracy)
        coords_rad = np.radians(stations_df[['latitude', 'longitude']].values)
        
        # Build KDTree
        tree = cKDTree(coords_rad)
        
        # Convert search radius to radians (approximate)
        # 1 degree ≈ 111 km at equator
        search_radius_deg = self.max_search_radius / 111.0
        search_radius_rad = np.radians(search_radius_deg)
        
        # Find potential neighbors using KDTree
        neighbors = defaultdict(set)
        neighbor_counts = []
        
        for idx in range(n_stations):
            # Query KDTree for all stations within max search radius
            station_coord = coords_rad[idx]
            potential_indices = tree.query_ball_point(station_coord, search_radius_rad)
            
            # Remove self from potential neighbors
            potential_indices = [i for i in potential_indices if i != idx]
            
            if use_directional and 'azimuth_deg' in stations_df.columns:
                # Apply directional filtering
                station_neighbors = self._filter_by_directional(
                    idx, potential_indices, stations_df
                )
            else:
                # Simple distance-based filtering
                station_neighbors = self._filter_by_distance(
                    idx, potential_indices, stations_df
                )
            
            neighbors[idx] = station_neighbors
            neighbor_counts.append(len(station_neighbors))
            
            if idx % 100 == 0 and idx > 0:
                avg_so_far = np.mean(neighbor_counts)
                logger.debug(f"Processed {idx}/{n_stations} stations, "
                           f"avg neighbors so far: {avg_so_far:.1f}")
        
        # Update statistics
        self._update_stats(neighbor_counts, n_stations)
        
        logger.info(f"Neighbor discovery complete: avg={self.stats.avg_neighbors:.1f}, "
                   f"max={self.stats.max_neighbors}, total_edges={self.stats.total_neighbors}")
        
        return dict(neighbors)
    
    def _filter_by_directional(self, station_idx: int, potential_indices: List[int],
                              stations_df: pd.DataFrame) -> Set[int]:
        """
        Filter potential neighbors using directional geometry.
        """
        station_neighbors = set()
        station_data = stations_df.iloc[station_idx]
        
        # Prepare station1 dict
        station1 = {
            'latitude': station_data['latitude'],
            'longitude': station_data['longitude'],
            'azimuth_deg': station_data.get('azimuth_deg', 0),
            'beamwidth_deg': station_data.get('beamwidth_deg', 360),
            'station_id': station_idx
        }
        
        for other_idx in potential_indices:
            other_data = stations_df.iloc[other_idx]
            
            # Prepare station2 dict
            station2 = {
                'latitude': other_data['latitude'],
                'longitude': other_data['longitude'],
                'azimuth_deg': other_data.get('azimuth_deg', 0),
                'beamwidth_deg': other_data.get('beamwidth_deg', 360),
                'station_id': other_idx
            }
            
            # Check directional interference
            interferes, effective_radius = self.directional.check_directional_interference(
                station1, station2
            )
            
            if interferes:
                station_neighbors.add(other_idx)
        
        return station_neighbors
    
    def _filter_by_distance(self, station_idx: int, potential_indices: List[int],
                          stations_df: pd.DataFrame) -> Set[int]:
        """
        Filter potential neighbors using simple distance threshold.
        """
        station_neighbors = set()
        station_data = stations_df.iloc[station_idx]
        
        lat1 = station_data['latitude']
        lon1 = station_data['longitude']
        
        # Use the main radius as default threshold
        threshold_km = self.directional.config.r_main_km
        
        for other_idx in potential_indices:
            other_data = stations_df.iloc[other_idx]
            lat2 = other_data['latitude']
            lon2 = other_data['longitude']
            
            # Calculate distance
            distance = self.directional.haversine_distance(
                lat1, lon1, lat2, lon2, station_idx, other_idx
            )
            
            if distance <= threshold_km:
                station_neighbors.add(other_idx)
        
        return station_neighbors
    
    def build_interference_graph(self, stations_df: pd.DataFrame,
                                use_directional: bool = True) -> List[Tuple[int, int]]:
        """
        Build interference graph as edge list.
        
        Returns:
            List of (station1_idx, station2_idx) tuples representing interference edges
        """
        neighbors = self.find_neighbors(stations_df, use_directional)
        
        # Convert to edge list (avoiding duplicates)
        edges = []
        for station_idx, neighbor_set in neighbors.items():
            for neighbor_idx in neighbor_set:
                if station_idx < neighbor_idx:  # Avoid duplicates
                    edges.append((station_idx, neighbor_idx))
        
        logger.info(f"Built interference graph with {len(edges)} edges")
        return edges
    
    def _update_stats(self, neighbor_counts: List[int], n_stations: int) -> None:
        """Update neighbor statistics."""
        self.stats.total_stations = n_stations
        self.stats.total_neighbors = sum(neighbor_counts)
        self.stats.avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 0
        self.stats.max_neighbors = max(neighbor_counts) if neighbor_counts else 0
        self.stats.min_neighbors = min(neighbor_counts) if neighbor_counts else 0
        
        # Create histogram
        histogram = defaultdict(int)
        for count in neighbor_counts:
            histogram[count] += 1
        self.stats.neighbor_histogram = dict(histogram)
    
    def get_complexity_analysis(self) -> Dict:
        """
        Analyze computational complexity based on actual neighbor counts.
        """
        n = self.stats.total_stations
        avg_k = self.stats.avg_neighbors
        total_edges = self.stats.total_neighbors
        
        # Theoretical complexities
        all_pairs_edges = n * (n - 1) / 2  # O(n²)
        actual_density = total_edges / all_pairs_edges if all_pairs_edges > 0 else 0
        
        return {
            'n_stations': n,
            'avg_neighbors': avg_k,
            'total_edges': total_edges,
            'all_pairs_edges': all_pairs_edges,
            'edge_density': actual_density,
            'complexity_class': self._classify_complexity(avg_k, n),
            'speedup_vs_all_pairs': all_pairs_edges / total_edges if total_edges > 0 else float('inf')
        }
    
    def _classify_complexity(self, avg_neighbors: float, n_stations: int) -> str:
        """Classify the complexity based on neighbor statistics."""
        if n_stations == 0:
            return "O(1)"
        
        ratio = avg_neighbors / n_stations
        
        if ratio < 0.01:
            return "O(1) - sparse"
        elif ratio < 0.1:
            return "O(log n) - very sparse"
        elif ratio < 0.5:
            return "O(√n) - moderately sparse"
        elif ratio < 0.9:
            return "O(n) - dense"
        else:
            return "O(n²) - very dense"


def create_neighbor_discovery(config: Optional[Dict] = None) -> NeighborDiscovery:
    """
    Create NeighborDiscovery instance with configuration.
    
    Args:
        config: Optional configuration dictionary with keys:
            - max_search_radius_km
            - az_tolerance_deg
            - r_main_km
            - r_off_km
    """
    config = config or {}
    
    # Create directional geometry
    directional_config = DirectionalConfig(
        az_tolerance_deg=config.get('az_tolerance_deg', 5.0),
        r_main_km=config.get('r_main_km', 30.0),
        r_off_km=config.get('r_off_km', 10.0)
    )
    directional = DirectionalGeometry(directional_config)
    
    # Create neighbor discovery
    max_radius = max(
        config.get('max_search_radius_km', 50.0),
        directional_config.r_main_km * 1.5  # Ensure we search far enough
    )
    
    return NeighborDiscovery(directional, max_radius)