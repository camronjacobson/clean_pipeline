#!/usr/bin/env python3
"""
Enhanced visualization for spectrum optimization results with shapefile overlay support.
Creates interactive maps with frequency-based coloring, clustering, and geographic boundaries.
"""

import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd
import json
import numpy as np
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors
from typing import Dict, Optional, Tuple, List, Union
import logging

# Geo imports for shapefile support
try:
    import geopandas as gpd
    from shapely.geometry import Point, shape
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not available. Shapefile support disabled.")

logger = logging.getLogger(__name__)


class EnhancedVisualizer:
    """
    Creates interactive visualizations of spectrum assignments with optional shapefile overlays.
    Handles both small (<100 stations) and large (30,000+) datasets.
    """
    
    def __init__(self, assignments_df: pd.DataFrame, metrics: Optional[Dict] = None,
                 shapefile_paths: Optional[List[str]] = None):
        """
        Initialize visualizer with assignment data and optional shapefiles.
        
        Args:
            assignments_df: DataFrame with columns: latitude, longitude, assigned_frequency
            metrics: Optional metrics dictionary from optimization
            shapefile_paths: Optional list of paths to shapefiles/GeoJSON files
        """
        self.df = self._normalize_dataframe(assignments_df)
        self.metrics = metrics or {}
        self.freq_colors = self._generate_frequency_colors()
        self.shapefiles = self._load_shapefiles(shapefile_paths)
        
        # Calculate map bounds
        self.center_lat = self.df['latitude'].mean()
        self.center_lon = self.df['longitude'].mean()
        self.bounds = self._calculate_bounds()
        
        logger.info(f"Visualizer initialized with {len(self.df)} stations, "
                   f"{len(self.freq_colors)} unique frequencies, "
                   f"{len(self.shapefiles)} shapefiles")
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names for consistency."""
        df = df.copy()
        
        # Handle various column name variations
        lat_cols = ['latitude', 'lat', 'y', 'y_coord']
        lon_cols = ['longitude', 'lon', 'lng', 'x', 'x_coord']
        
        for col in lat_cols:
            if col in df.columns:
                df['latitude'] = df[col]
                break
        
        for col in lon_cols:
            if col in df.columns:
                df['longitude'] = df[col]
                break
        
        # Ensure required columns exist
        required = ['latitude', 'longitude', 'assigned_frequency']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    def _load_shapefiles(self, paths: Optional[List[str]]) -> Dict[str, gpd.GeoDataFrame]:
        """
        Load and cache shapefiles if provided.
        
        Args:
            paths: List of paths to shapefile or GeoJSON files
            
        Returns:
            Dictionary of {name: GeoDataFrame}
        """
        if not paths or not GEOPANDAS_AVAILABLE:
            return {}
        
        shapes = {}
        for path in paths:
            try:
                path_obj = Path(path)
                if not path_obj.exists():
                    logger.warning(f"Shapefile not found: {path}")
                    continue
                
                name = path_obj.stem
                
                # Load based on extension
                if path.endswith('.geojson') or path.endswith('.json'):
                    shapes[name] = gpd.read_file(path)
                    logger.info(f"Loaded GeoJSON: {name} ({len(shapes[name])} features)")
                elif path.endswith('.shp'):
                    shapes[name] = gpd.read_file(path)
                    logger.info(f"Loaded Shapefile: {name} ({len(shapes[name])} features)")
                else:
                    logger.warning(f"Unsupported file type: {path}")
                    
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")
                
        return shapes
    
    def _calculate_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate map bounds with padding."""
        lat_min, lat_max = self.df['latitude'].min(), self.df['latitude'].max()
        lon_min, lon_max = self.df['longitude'].min(), self.df['longitude'].max()
        
        # Add 5% padding
        lat_padding = (lat_max - lat_min) * 0.05
        lon_padding = (lon_max - lon_min) * 0.05
        
        return ((lat_min - lat_padding, lon_min - lon_padding),
                (lat_max + lat_padding, lon_max + lon_padding))
    
    def _generate_frequency_colors(self) -> Dict[float, str]:
        """
        Map frequencies to colors using spectrum colormap.
        Returns dict of {frequency: hex_color}
        """
        unique_freqs = sorted(self.df['assigned_frequency'].unique())
        n_freqs = len(unique_freqs)
        
        if n_freqs == 0:
            return {}
        
        # Use rainbow colormap for good frequency distinction
        cmap = cm.get_cmap('rainbow')
        norm = colors.Normalize(vmin=0, vmax=max(n_freqs-1, 1))
        
        freq_colors = {}
        for i, freq in enumerate(unique_freqs):
            rgba = cmap(norm(i))
            # Convert to hex
            freq_colors[freq] = colors.rgb2hex(rgba[:3])
        
        logger.debug(f"Generated {n_freqs} frequency colors")
        return freq_colors
    
    def create_interactive_map(self, output_path: str = "report.html") -> folium.Map:
        """
        Create interactive map with appropriate visualization strategy and optional overlays.
        
        Args:
            output_path: Path to save HTML file
            
        Returns:
            Folium map object
        """
        # Determine initial zoom level based on geographic extent
        lat_range = self.df['latitude'].max() - self.df['latitude'].min()
        lon_range = self.df['longitude'].max() - self.df['longitude'].min()
        
        if max(lat_range, lon_range) > 10:
            zoom_start = 5  # Country level
        elif max(lat_range, lon_range) > 2:
            zoom_start = 7  # State level
        else:
            zoom_start = 10  # City level
        
        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap',
            prefer_canvas=True  # Better performance for many markers
        )
        
        # Fit to bounds
        m.fit_bounds([
            [self.bounds[0][0], self.bounds[0][1]],
            [self.bounds[1][0], self.bounds[1][1]]
        ])
        
        # Add shapefile overlays if available (BEFORE stations for proper layering)
        if self.shapefiles:
            self._add_shapefile_overlays(m)
            self._add_density_choropleth(m)
            logger.info(f"Added {len(self.shapefiles)} shapefile layers")
        else:
            logger.info("No shapefiles provided - showing stations only")
        
        # Choose visualization strategy based on station count
        n_stations = len(self.df)
        
        if n_stations < 100:
            logger.info(f"Using individual markers for {n_stations} stations")
            self._add_individual_markers(m)
        elif n_stations < 1000:
            logger.info(f"Using clustered markers for {n_stations} stations")
            self._add_clustered_markers(m)
        else:
            logger.info(f"Using heatmap + clusters for {n_stations} stations")
            self._add_heatmap_layer(m)
            self._add_clustered_markers(m)
        
        # Add layer control if we have multiple layers
        if self.shapefiles or n_stations >= 1000:
            folium.LayerControl(collapsed=False).add_to(m)
        
        # Add frequency legend
        self._add_frequency_legend(m)
        
        # Add title
        self._add_map_title(m)
        
        # Save map
        m.save(output_path)
        logger.info(f"Map saved to {output_path}")
        
        return m
    
    def _add_shapefile_overlays(self, map_obj: folium.Map) -> None:
        """
        Add optional boundary overlays with layer control.
        
        Args:
            map_obj: Folium map to add overlays to
        """
        for name, gdf in self.shapefiles.items():
            # Create feature group for this shapefile
            fg = folium.FeatureGroup(name=name.replace('_', ' ').title(), show=False)
            
            # Simplify geometry for performance if needed
            gdf_simplified = gdf.copy()
            if len(gdf) > 100:
                # Only simplify if many features
                gdf_simplified['geometry'] = gdf_simplified.geometry.simplify(0.01)
            
            # Determine style based on shapefile name
            if 'region' in name.lower():
                style = {
                    'fillColor': 'lightblue',
                    'color': 'darkblue',
                    'weight': 2,
                    'fillOpacity': 0.1,
                    'opacity': 0.7
                }
            elif 'state' in name.lower():
                style = {
                    'fillColor': 'none',
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0,
                    'opacity': 0.5,
                    'dashArray': '5, 5'
                }
            else:
                style = {
                    'fillColor': 'none',
                    'color': 'gray',
                    'weight': 1,
                    'fillOpacity': 0,
                    'opacity': 0.4
                }
            
            # Add each shape
            for idx, row in gdf_simplified.iterrows():
                try:
                    # Convert to GeoJSON
                    geo_json = folium.GeoJson(
                        data=row['geometry'].__geo_interface__,
                        style_function=lambda x, s=style: s
                    )
                    
                    # Add label if available
                    label_cols = ['NAME', 'name', 'Name', 'STATE_ABBR', 'REGION_ID']
                    label = None
                    for col in label_cols:
                        if col in row and pd.notna(row[col]):
                            label = str(row[col])
                            break
                    
                    if label:
                        geo_json.add_child(folium.Tooltip(label))
                    
                    geo_json.add_to(fg)
                    
                except Exception as e:
                    logger.debug(f"Could not add feature {idx} from {name}: {e}")
            
            fg.add_to(map_obj)
            logger.info(f"Added overlay '{name}' with {len(gdf)} features")
    
    def _add_density_choropleth(self, map_obj: folium.Map, shapefile_name: Optional[str] = None) -> None:
        """
        Color regions by station density.
        
        Args:
            map_obj: Folium map
            shapefile_name: Name of shapefile to use for regions (auto-detect if None)
        """
        if not self.shapefiles or not GEOPANDAS_AVAILABLE:
            return
        
        # Auto-detect appropriate shapefile
        if shapefile_name is None:
            # Prefer regions or states
            for name in ['california_regions', 'us_states_west', 'bea_regions']:
                if name in self.shapefiles:
                    shapefile_name = name
                    break
            
            # Use first available if no preferred found
            if shapefile_name is None:
                shapefile_name = list(self.shapefiles.keys())[0]
        
        if shapefile_name not in self.shapefiles:
            logger.warning(f"Shapefile '{shapefile_name}' not found for choropleth")
            return
        
        gdf = self.shapefiles[shapefile_name].copy()
        
        # Calculate stations per region
        station_counts = []
        for region_idx, region in gdf.iterrows():
            count = 0
            for _, station in self.df.iterrows():
                point = Point(station['longitude'], station['latitude'])
                try:
                    if region['geometry'].contains(point):
                        count += 1
                except:
                    pass  # Skip invalid geometries
            station_counts.append(count)
        
        gdf['station_count'] = station_counts
        
        # Only create choropleth if we have stations in regions
        if gdf['station_count'].sum() > 0:
            # Create choropleth
            choropleth = folium.Choropleth(
                geo_data=gdf.to_json(),
                name='Station Density',
                data=gdf,
                columns=[gdf.index, 'station_count'],
                key_on='feature.id',
                fill_color='YlOrRd',
                fill_opacity=0.3,
                line_opacity=0.2,
                legend_name='Stations per Region',
                show=False,
                highlight=True
            ).add_to(map_obj)
            
            # Add tooltips with counts
            choropleth.geojson.add_child(
                folium.features.GeoJsonTooltip(
                    fields=['station_count'],
                    aliases=['Stations:'],
                    labels=True,
                    sticky=True
                )
            )
            
            logger.info(f"Added station density choropleth using '{shapefile_name}'")
    
    def _add_individual_markers(self, map_obj: folium.Map) -> None:
        """For <100 stations: individual markers with detailed popups."""
        for idx, row in self.df.iterrows():
            freq = row['assigned_frequency']
            color = self.freq_colors.get(freq, '#808080')
            
            # Create detailed popup
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4 style="margin: 5px 0;">Station Details</h4>
                <table style="width: 100%;">
                    <tr><td><b>ID:</b></td><td>{row.get('station_id', f'Station {idx}')}</td></tr>
                    <tr><td><b>Frequency:</b></td><td>{freq:.2f} MHz</td></tr>
                    <tr><td><b>Latitude:</b></td><td>{row['latitude']:.4f}</td></tr>
                    <tr><td><b>Longitude:</b></td><td>{row['longitude']:.4f}</td></tr>
                    <tr><td><b>Power:</b></td><td>{row.get('power_watts', 'N/A')} W</td></tr>
                    <tr><td><b>Azimuth:</b></td><td>{row.get('azimuth_deg', 'Omni')}°</td></tr>
                    <tr><td><b>Beamwidth:</b></td><td>{row.get('beamwidth_deg', 360)}°</td></tr>
                </table>
            </div>
            """
            
            # Add circle marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Station: {row.get('station_id', idx)}<br>Freq: {freq:.2f} MHz",
                color='black',
                fillColor=color,
                fillOpacity=0.8,
                weight=1
            ).add_to(map_obj)
    
    def _add_clustered_markers(self, map_obj: folium.Map) -> None:
        """For 100+ stations: clustered markers with zoom details."""
        # Create marker cluster
        marker_cluster = MarkerCluster(
            name='Stations',
            options={
                'maxClusterRadius': 50,
                'spiderfyOnMaxZoom': True,
                'showCoverageOnHover': False,
                'zoomToBoundsOnClick': True,
                'spiderfyDistanceMultiplier': 2
            }
        ).add_to(map_obj)
        
        # Add markers to cluster
        for idx, row in self.df.iterrows():
            freq = row['assigned_frequency']
            
            # Simple popup for performance
            popup_html = f"""
            <b>{row.get('station_id', f'Station {idx}')}</b><br>
            Frequency: {freq:.2f} MHz<br>
            Power: {row.get('power_watts', 'N/A')} W<br>
            Location: ({row['latitude']:.3f}, {row['longitude']:.3f})
            """
            
            # Map frequency to folium icon color
            icon_color = self._freq_to_icon_color(freq)
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"{row.get('station_id', idx)}: {freq:.1f} MHz",
                icon=folium.Icon(color=icon_color, icon='radio', prefix='fa')
            ).add_to(marker_cluster)
    
    def _add_heatmap_layer(self, map_obj: folium.Map) -> None:
        """Add heatmap layer for large datasets."""
        # Prepare heatmap data (lat, lon, weight)
        heat_data = []
        
        for _, row in self.df.iterrows():
            # Weight by power if available
            weight = 1.0
            if 'power_watts' in row:
                # Normalize power to 0-1 scale
                weight = min(row['power_watts'] / 100000, 1.0)
            
            heat_data.append([row['latitude'], row['longitude'], weight])
        
        # Add heatmap
        HeatMap(
            heat_data,
            name='Station Density',
            min_opacity=0.3,
            max_zoom=10,
            radius=15,
            blur=20,
            gradient={
                0.0: 'blue',
                0.25: 'cyan',
                0.5: 'lime',
                0.75: 'yellow',
                1.0: 'red'
            }
        ).add_to(map_obj)
    
    def _freq_to_icon_color(self, freq: float) -> str:
        """Convert frequency to folium icon color."""
        # Folium icon colors are limited
        folium_colors = ['red', 'blue', 'green', 'purple', 'orange', 
                        'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
                        'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue',
                        'lightgreen', 'gray', 'black', 'lightgray']
        
        unique_freqs = sorted(self.freq_colors.keys())
        if freq in unique_freqs:
            idx = unique_freqs.index(freq) % len(folium_colors)
            return folium_colors[idx]
        return 'gray'
    
    def _add_frequency_legend(self, map_obj: folium.Map) -> None:
        """Add frequency color legend to map."""
        if not self.freq_colors:
            return
        
        # Create legend HTML
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: auto;
                    background-color: white; z-index: 1000; 
                    border: 2px solid grey; border-radius: 5px;
                    padding: 10px; font-size: 14px;">
        <h4 style="margin: 0 0 5px 0;">Frequency Assignments</h4>
        '''
        
        # Add up to 20 frequencies to legend
        for i, (freq, color) in enumerate(sorted(self.freq_colors.items())[:20]):
            count = (self.df['assigned_frequency'] == freq).sum()
            legend_html += f'''
            <div style="margin: 2px 0;">
                <span style="background-color: {color}; 
                            width: 20px; height: 10px; 
                            display: inline-block; margin-right: 5px;
                            border: 1px solid black;"></span>
                <span>{freq:.1f} MHz ({count})</span>
            </div>
            '''
        
        if len(self.freq_colors) > 20:
            legend_html += f'<div style="margin-top: 5px; font-style: italic;">... and {len(self.freq_colors)-20} more</div>'
        
        legend_html += '</div>'
        
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def _add_map_title(self, map_obj: folium.Map) -> None:
        """Add title and statistics to map."""
        shapefile_info = f" | Overlays: {len(self.shapefiles)}" if self.shapefiles else ""
        
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: auto; height: auto;
                    background-color: white; z-index: 1000;
                    border: 2px solid grey; border-radius: 5px;
                    padding: 10px; font-size: 16px;">
        <h3 style="margin: 0;">Spectrum Optimization Results</h3>
        <div style="font-size: 14px; margin-top: 5px;">
            Stations: {len(self.df)} | 
            Frequencies: {len(self.freq_colors)} | 
            Efficiency: {len(self.df)/max(len(self.freq_colors), 1):.1f} stations/channel
            {shapefile_info}
        </div>
        </div>
        '''
        
        map_obj.get_root().html.add_child(folium.Element(title_html))
    
    def generate_frequency_report(self) -> Dict:
        """Generate statistics about frequency usage and geography."""
        report = {
            'total_stations': len(self.df),
            'unique_frequencies': len(self.freq_colors),
            'frequency_distribution': {},
            'geographic_span': {
                'lat_range': float(self.df['latitude'].max() - self.df['latitude'].min()),
                'lon_range': float(self.df['longitude'].max() - self.df['longitude'].min())
            },
            'shapefile_layers': list(self.shapefiles.keys()) if self.shapefiles else []
        }
        
        # Count stations per frequency
        for freq in sorted(self.freq_colors.keys()):
            count = (self.df['assigned_frequency'] == freq).sum()
            report['frequency_distribution'][float(freq)] = int(count)
        
        # Add regional statistics if shapefiles available
        if self.shapefiles and GEOPANDAS_AVAILABLE:
            report['regional_distribution'] = {}
            for name, gdf in self.shapefiles.items():
                region_counts = {}
                for region_idx, region in gdf.iterrows():
                    count = 0
                    for _, station in self.df.iterrows():
                        point = Point(station['longitude'], station['latitude'])
                        try:
                            if region['geometry'].contains(point):
                                count += 1
                        except:
                            pass
                    if count > 0:
                        region_label = str(region.get('NAME', region.get('name', f'Region_{region_idx}')))
                        region_counts[region_label] = count
                
                if region_counts:
                    report['regional_distribution'][name] = region_counts
        
        return report


def create_visualization(assignments_path: str, metrics_path: Optional[str] = None,
                         shapefile_paths: Optional[List[str]] = None,
                         output_path: str = "interactive_map.html") -> None:
    """
    Convenience function to create visualization from files.
    
    Args:
        assignments_path: Path to assignments CSV
        metrics_path: Optional path to metrics JSON
        shapefile_paths: Optional list of shapefile/GeoJSON paths
        output_path: Where to save HTML map
    """
    # Load data
    df = pd.read_csv(assignments_path)
    
    metrics = None
    if metrics_path and Path(metrics_path).exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    # Create visualization
    viz = EnhancedVisualizer(df, metrics, shapefile_paths)
    viz.create_interactive_map(output_path)
    
    # Generate report
    report = viz.generate_frequency_report()
    print(f"Map created: {output_path}")
    print(f"Stations: {report['total_stations']}")
    print(f"Unique frequencies: {report['unique_frequencies']}")
    print(f"Geographic span: {report['geographic_span']['lat_range']:.2f}° lat, "
          f"{report['geographic_span']['lon_range']:.2f}° lon")
    
    if report.get('shapefile_layers'):
        print(f"Shapefile layers: {', '.join(report['shapefile_layers'])}")
    
    if report.get('regional_distribution'):
        for layer, regions in report['regional_distribution'].items():
            print(f"\nStation distribution in {layer}:")
            for region, count in sorted(regions.items(), key=lambda x: x[1], reverse=True):
                print(f"  {region}: {count} stations")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualizer_enhanced_v2.py <assignments.csv> [metrics.json] [output.html] [shapefile1] [shapefile2] ...")
        sys.exit(1)
    
    assignments = sys.argv[1]
    metrics = None
    output = "interactive_map.html"
    shapefiles = []
    
    # Parse arguments
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg.endswith('.json') and metrics is None:
            metrics = arg
        elif arg.endswith('.html'):
            output = arg
        elif arg.endswith(('.shp', '.geojson', '.json')):
            shapefiles.append(arg)
    
    create_visualization(assignments, metrics, shapefiles if shapefiles else None, output)