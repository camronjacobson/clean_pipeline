"""
Directional geometry calculations for spectrum optimization.
Implements great-circle bearings, lobe calculations, and dual-radius interference.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DirectionalConfig:
    """Configuration for directional interference calculations."""
    az_tolerance_deg: float = 5.0  # Angular tolerance for lobe calculations
    r_main_km: float = 30.0  # Main lobe interference radius
    r_off_km: float = 10.0  # Off-lobe interference radius
    cache_size: int = 1000000  # Maximum cache entries


class DirectionalGeometry:
    """
    Handles directional antenna geometry calculations with caching.
    """
    
    def __init__(self, config: Optional[DirectionalConfig] = None):
        """Initialize with configuration."""
        self.config = config or DirectionalConfig()
        
        # Initialize caches
        self._distance_cache: Dict[Tuple[int, int], float] = {}
        self._bearing_cache: Dict[Tuple[int, int], float] = {}
        self._in_lobe_cache: Dict[Tuple[int, int, float, float], bool] = {}
        
        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"DirectionalGeometry initialized with r_main={self.config.r_main_km}km, "
                   f"r_off={self.config.r_off_km}km, az_tolerance={self.config.az_tolerance_deg}°")
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float,
                          station_id1: Optional[int] = None, station_id2: Optional[int] = None) -> float:
        """
        Calculate great-circle distance between two points in kilometers.
        Uses caching with station IDs when available.
        """
        # Create cache key
        if station_id1 is not None and station_id2 is not None:
            cache_key = (min(station_id1, station_id2), max(station_id1, station_id2))
            if cache_key in self._distance_cache:
                self._cache_hits += 1
                return self._distance_cache[cache_key]
        
        self._cache_misses += 1
        
        # Calculate distance
        R = 6371.0  # Earth radius in km
        
        # Convert to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to handle numerical errors
        
        distance = R * c
        
        # Cache result
        if station_id1 is not None and station_id2 is not None:
            self._manage_cache_size(self._distance_cache)
            self._distance_cache[cache_key] = distance
        
        return distance
    
    def great_circle_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float,
                           station_id1: Optional[int] = None, station_id2: Optional[int] = None) -> float:
        """
        Calculate initial bearing from point 1 to point 2 along great circle.
        Returns bearing in degrees (0-360).
        """
        # Create cache key
        if station_id1 is not None and station_id2 is not None:
            cache_key = (station_id1, station_id2)  # Direction matters for bearing
            if cache_key in self._bearing_cache:
                self._cache_hits += 1
                return self._bearing_cache[cache_key]
        
        self._cache_misses += 1
        
        # Convert to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        # Calculate bearing
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        
        bearing_rad = np.arctan2(y, x)
        bearing_deg = (np.degrees(bearing_rad) + 360) % 360  # Normalize to 0-360
        
        # Cache result
        if station_id1 is not None and station_id2 is not None:
            self._manage_cache_size(self._bearing_cache)
            self._bearing_cache[cache_key] = bearing_deg
        
        return bearing_deg
    
    def angular_difference(self, angle1: float, angle2: float) -> float:
        """
        Calculate minimum angular difference between two angles.
        Returns value in range [0, 180].
        """
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)
    
    def in_lobe(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float,
                azimuth_deg: float, beamwidth_deg: float,
                from_id: Optional[int] = None, to_id: Optional[int] = None) -> bool:
        """
        Check if station 'to' is within the main lobe of station 'from'.
        
        Args:
            from_lat, from_lon: Coordinates of transmitting station
            to_lat, to_lon: Coordinates of receiving station
            azimuth_deg: Azimuth of transmitting station (0-360)
            beamwidth_deg: Beamwidth of transmitting station
            from_id, to_id: Optional station IDs for caching
            
        Returns:
            True if 'to' is within main lobe of 'from'
        """
        # Create cache key
        cache_key = (from_id, to_id, azimuth_deg, beamwidth_deg) if from_id is not None and to_id is not None else None
        if cache_key and cache_key in self._in_lobe_cache:
            self._cache_hits += 1
            return self._in_lobe_cache[cache_key]
        
        self._cache_misses += 1
        
        # Calculate bearing from 'from' to 'to'
        bearing = self.great_circle_bearing(from_lat, from_lon, to_lat, to_lon, from_id, to_id)
        
        # Calculate angular difference
        ang_diff = self.angular_difference(bearing, azimuth_deg)
        
        # Check if within lobe (with tolerance)
        half_beamwidth = beamwidth_deg / 2.0
        result = ang_diff <= (half_beamwidth + self.config.az_tolerance_deg)
        
        # Cache result
        if cache_key:
            self._manage_cache_size(self._in_lobe_cache)
            self._in_lobe_cache[cache_key] = result
        
        return result
    
    def check_directional_interference(self, station1: Dict, station2: Dict) -> Tuple[bool, float]:
        """
        Check if two stations interfere based on directional patterns.
        Uses conservative approach: interference if EITHER station has the other in its lobe.
        
        Args:
            station1, station2: Dictionaries with keys:
                - latitude, longitude
                - azimuth_deg, beamwidth_deg
                - station_id (optional)
                
        Returns:
            (interferes, effective_radius_km)
        """
        # Extract station data
        lat1, lon1 = station1['latitude'], station1['longitude']
        lat2, lon2 = station2['latitude'], station2['longitude']
        
        # Get station IDs if available
        id1 = station1.get('station_id', station1.get('id'))
        id2 = station2.get('station_id', station2.get('id'))
        
        # Calculate distance
        distance = self.haversine_distance(lat1, lon1, lat2, lon2, id1, id2)
        
        # Check if either station has the other in its main lobe
        az1 = station1.get('azimuth_deg', 0)
        bw1 = station1.get('beamwidth_deg', 360)
        az2 = station2.get('azimuth_deg', 0)
        bw2 = station2.get('beamwidth_deg', 360)
        
        # Station 1 has station 2 in its lobe?
        s1_has_s2 = self.in_lobe(lat1, lon1, lat2, lon2, az1, bw1, id1, id2)
        
        # Station 2 has station 1 in its lobe?
        s2_has_s1 = self.in_lobe(lat2, lon2, lat1, lon1, az2, bw2, id2, id1)
        
        # Determine effective radius (conservative: use main radius if EITHER is in lobe)
        if s1_has_s2 or s2_has_s1:
            effective_radius = self.config.r_main_km
            logger.debug(f"Stations {id1}-{id2}: Using main radius {effective_radius}km "
                        f"(s1_has_s2={s1_has_s2}, s2_has_s1={s2_has_s1})")
        else:
            effective_radius = self.config.r_off_km
            logger.debug(f"Stations {id1}-{id2}: Using off-lobe radius {effective_radius}km")
        
        # Check interference
        interferes = distance <= effective_radius
        
        return interferes, effective_radius
    
    def _manage_cache_size(self, cache: Dict) -> None:
        """Manage cache size to prevent unbounded growth."""
        if len(cache) >= self.config.cache_size:
            # Remove 10% of oldest entries
            num_to_remove = self.config.cache_size // 10
            for _ in range(num_to_remove):
                if cache:
                    cache.pop(next(iter(cache)))
    
    def get_cache_stats(self) -> Dict:
        """Return cache statistics."""
        total_queries = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_queries * 100) if total_queries > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'distance_cache_size': len(self._distance_cache),
            'bearing_cache_size': len(self._bearing_cache),
            'in_lobe_cache_size': len(self._in_lobe_cache)
        }
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self._distance_cache.clear()
        self._bearing_cache.clear()
        self._in_lobe_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("All caches cleared")


def create_directional_geometry_from_config(config_dict: Dict) -> DirectionalGeometry:
    """
    Create DirectionalGeometry instance from configuration dictionary.
    
    Args:
        config_dict: Dictionary with optional keys:
            - az_tolerance_deg
            - r_main_km
            - r_off_km
            - cache_size
    """
    config = DirectionalConfig(
        az_tolerance_deg=config_dict.get('az_tolerance_deg', 5.0),
        r_main_km=config_dict.get('r_main_km', 30.0),
        r_off_km=config_dict.get('r_off_km', 10.0),
        cache_size=config_dict.get('cache_size', 1000000)
    )
    return DirectionalGeometry(config)