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
    
    def analyze_zipcode_frequency_reuse(self, stations_df: pd.DataFrame, 
                                       edges: List[Tuple[int, int]]) -> Dict:
        """
        Analyze frequency reuse opportunities across zipcodes considering directional patterns.
        
        Args:
            stations_df: DataFrame with station data including 'zipcode' and 'assigned_frequency'
            edges: List of interference edges from build_interference_graph
            
        Returns:
            Dictionary with detailed zipcode frequency reuse analysis
        """
        # Check if required columns exist
        if 'zipcode' not in stations_df.columns:
            return {
                'available': False,
                'message': 'No zipcode data available for reuse analysis'
            }
        
        if 'assigned_frequency' not in stations_df.columns:
            return {
                'available': False,
                'message': 'No frequency assignments available for reuse analysis'
            }
        
        logger.info("Analyzing zipcode frequency reuse with directional patterns")
        
        # Create station index to zipcode mapping
        idx_to_zip = {idx: str(row['zipcode']) for idx, row in stations_df.iterrows()}
        zip_to_indices = {}
        for idx, zipcode in idx_to_zip.items():
            if zipcode not in zip_to_indices:
                zip_to_indices[zipcode] = []
            zip_to_indices[zipcode].append(idx)
        
        # Build zipcode interference matrix
        zipcode_interference = {}
        for zip1 in zip_to_indices:
            zipcode_interference[zip1] = set()
            
        # Analyze interference between zipcodes
        for i, j in edges:
            zip_i = idx_to_zip.get(i)
            zip_j = idx_to_zip.get(j)
            
            if zip_i and zip_j and zip_i != zip_j:
                zipcode_interference[zip_i].add(zip_j)
                zipcode_interference[zip_j].add(zip_i)
        
        # Analyze frequency usage per zipcode
        zipcode_freq_usage = {}
        for zipcode, indices in zip_to_indices.items():
            stations_subset = stations_df.iloc[indices]
            freq_counts = stations_subset['assigned_frequency'].value_counts()
            
            zipcode_freq_usage[zipcode] = {
                'station_count': len(indices),
                'frequencies_used': set(stations_subset['assigned_frequency'].values),
                'unique_freq_count': len(freq_counts),
                'frequency_distribution': freq_counts.to_dict(),
                'efficiency': len(indices) / max(len(freq_counts), 1)
            }
        
        # Identify non-interfering zipcode pairs
        non_interfering_pairs = []
        for zip1 in zip_to_indices:
            for zip2 in zip_to_indices:
                if zip1 < zip2:  # Avoid duplicates
                    if zip2 not in zipcode_interference[zip1]:
                        non_interfering_pairs.append((zip1, zip2))
        
        # Analyze reuse opportunities
        reuse_opportunities = []
        total_potential_savings = 0
        
        for zip1, zip2 in non_interfering_pairs:
            freqs1 = zipcode_freq_usage[zip1]['frequencies_used']
            freqs2 = zipcode_freq_usage[zip2]['frequencies_used']
            
            # Calculate overlap and potential for reuse
            freq_overlap = freqs1.intersection(freqs2)
            unique_to_zip1 = freqs1 - freqs2
            unique_to_zip2 = freqs2 - freqs1
            
            # Calculate potential frequency savings if perfect reuse
            current_total_freqs = len(freqs1.union(freqs2))
            optimal_freqs = max(len(freqs1), len(freqs2))
            potential_savings = current_total_freqs - optimal_freqs
            
            if potential_savings > 0:
                # Analyze directional patterns for better understanding
                stations1 = stations_df.iloc[zip_to_indices[zip1]]
                stations2 = stations_df.iloc[zip_to_indices[zip2]]
                
                # Calculate average directional characteristics
                avg_azimuth1 = stations1.get('azimuth_deg', pd.Series([0])).mean()
                avg_azimuth2 = stations2.get('azimuth_deg', pd.Series([0])).mean()
                avg_beamwidth1 = stations1.get('beamwidth_deg', pd.Series([360])).mean()
                avg_beamwidth2 = stations2.get('beamwidth_deg', pd.Series([360])).mean()
                
                # Calculate geographic separation
                center1_lat = stations1['latitude'].mean()
                center1_lon = stations1['longitude'].mean()
                center2_lat = stations2['latitude'].mean()
                center2_lon = stations2['longitude'].mean()
                
                # Approximate distance
                distance_km = np.sqrt(
                    ((center2_lat - center1_lat) * 111) ** 2 +
                    ((center2_lon - center1_lon) * 111 * np.cos(np.radians(center1_lat))) ** 2
                )
                
                reuse_opportunities.append({
                    'zipcode_pair': (zip1, zip2),
                    'current_frequencies': current_total_freqs,
                    'optimal_frequencies': optimal_freqs,
                    'potential_savings': potential_savings,
                    'freq_overlap': len(freq_overlap),
                    'distance_km': round(distance_km, 2),
                    'directional_info': {
                        'avg_azimuth_diff': abs(avg_azimuth1 - avg_azimuth2),
                        'avg_beamwidth': (avg_beamwidth1 + avg_beamwidth2) / 2,
                        'likely_directional': avg_beamwidth1 < 180 or avg_beamwidth2 < 180
                    },
                    'reuse_score': potential_savings / current_total_freqs if current_total_freqs > 0 else 0
                })
                
                total_potential_savings += potential_savings
        
        # Sort opportunities by potential savings
        reuse_opportunities.sort(key=lambda x: x['potential_savings'], reverse=True)
        
        # Calculate interference statistics per zipcode
        zipcode_isolation_scores = {}
        for zipcode in zip_to_indices:
            interfering_count = len(zipcode_interference[zipcode])
            total_other_zips = len(zip_to_indices) - 1
            isolation_score = 1.0 - (interfering_count / max(total_other_zips, 1))
            
            zipcode_isolation_scores[zipcode] = {
                'interfering_zipcodes': interfering_count,
                'total_other_zipcodes': total_other_zips,
                'isolation_score': round(isolation_score, 3),
                'non_interfering': list(set(zip_to_indices.keys()) - 
                                       zipcode_interference[zipcode] - {zipcode})
            }
        
        # Identify zipcode clusters (groups that don't interfere)
        zipcode_clusters = self._find_zipcode_clusters(
            list(zip_to_indices.keys()),
            zipcode_interference
        )
        
        # Calculate overall metrics
        total_frequencies_used = len(set(stations_df['assigned_frequency'].values))
        avg_efficiency = np.mean([z['efficiency'] for z in zipcode_freq_usage.values()])
        
        # Find the theoretical minimum frequencies needed
        max_freq_per_cluster = []
        for cluster in zipcode_clusters:
            cluster_freqs = set()
            for zipcode in cluster:
                cluster_freqs.update(zipcode_freq_usage[zipcode]['frequencies_used'])
            max_freq_per_cluster.append(len(cluster_freqs))
        
        theoretical_min = max(max_freq_per_cluster) if max_freq_per_cluster else total_frequencies_used
        
        return {
            'available': True,
            'summary': {
                'total_zipcodes': len(zip_to_indices),
                'total_stations': len(stations_df),
                'total_interference_edges': len(edges),
                'non_interfering_pairs': len(non_interfering_pairs),
                'total_frequencies_used': total_frequencies_used,
                'theoretical_minimum_frequencies': theoretical_min,
                'potential_frequency_savings': total_potential_savings,
                'average_efficiency': round(avg_efficiency, 3),
                'reuse_improvement_potential': round(
                    (total_frequencies_used - theoretical_min) / max(total_frequencies_used, 1),
                    3
                )
            },
            'zipcode_frequency_usage': zipcode_freq_usage,
            'zipcode_interference_matrix': {
                zip_code: list(interfering) 
                for zip_code, interfering in zipcode_interference.items()
            },
            'zipcode_isolation_scores': zipcode_isolation_scores,
            'reuse_opportunities': reuse_opportunities[:10],  # Top 10 opportunities
            'zipcode_clusters': zipcode_clusters,
            'recommendations': self._generate_reuse_recommendations(
                reuse_opportunities, 
                zipcode_freq_usage,
                zipcode_isolation_scores
            )
        }
    
    def _find_zipcode_clusters(self, zipcodes: List[str], 
                               interference_matrix: Dict[str, set]) -> List[List[str]]:
        """
        Find clusters of zipcodes that don't interfere with each other.
        Uses a greedy graph coloring approach.
        """
        clusters = []
        assigned = set()
        
        for zipcode in zipcodes:
            if zipcode in assigned:
                continue
            
            # Start new cluster
            cluster = [zipcode]
            assigned.add(zipcode)
            
            # Try to add non-interfering zipcodes
            for other_zip in zipcodes:
                if other_zip in assigned:
                    continue
                
                # Check if this zipcode interferes with any in cluster
                can_add = True
                for cluster_zip in cluster:
                    if other_zip in interference_matrix[cluster_zip]:
                        can_add = False
                        break
                
                if can_add:
                    cluster.append(other_zip)
                    assigned.add(other_zip)
            
            clusters.append(cluster)
        
        return clusters
    
    def _generate_reuse_recommendations(self, opportunities: List[Dict],
                                       freq_usage: Dict,
                                       isolation_scores: Dict) -> List[str]:
        """Generate actionable recommendations for frequency reuse improvement."""
        recommendations = []
        
        if not opportunities:
            recommendations.append(
                "No significant frequency reuse opportunities found. "
                "Current allocation may already be optimal given interference constraints."
            )
            return recommendations
        
        # Top opportunity
        if opportunities:
            top = opportunities[0]
            recommendations.append(
                f"Highest impact: Zipcodes {top['zipcode_pair'][0]} and "
                f"{top['zipcode_pair'][1]} could share frequencies, "
                f"saving {top['potential_savings']} channels "
                f"(currently using {top['current_frequencies']} combined)."
            )
        
        # Find most isolated zipcodes
        most_isolated = sorted(
            isolation_scores.items(),
            key=lambda x: x[1]['isolation_score'],
            reverse=True
        )[:3]
        
        if most_isolated and most_isolated[0][1]['isolation_score'] > 0.5:
            isolated_zips = ', '.join([z[0] for z in most_isolated])
            recommendations.append(
                f"Zipcodes {isolated_zips} have high isolation scores and "
                f"are good candidates for frequency reuse with multiple other regions."
            )
        
        # Check for directional opportunities
        directional_opportunities = [
            opp for opp in opportunities 
            if opp['directional_info']['likely_directional']
        ]
        
        if directional_opportunities:
            recommendations.append(
                f"Found {len(directional_opportunities)} zipcode pairs where "
                f"directional antenna patterns could enable better frequency reuse. "
                f"Consider adjusting antenna azimuths to minimize inter-zipcode interference."
            )
        
        # Overall efficiency
        total_savings = sum(opp['potential_savings'] for opp in opportunities)
        if total_savings > 10:
            recommendations.append(
                f"Total potential frequency savings across all non-interfering "
                f"zipcode pairs: {total_savings} channels. This represents a "
                f"{(total_savings / max(len(freq_usage), 1)):.1%} potential reduction "
                f"in spectrum usage."
            )
        
        return recommendations
    
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