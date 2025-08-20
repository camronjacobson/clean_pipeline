#!/usr/bin/env python3
"""
Enhanced visualization for spectrum optimization results.
Creates interactive maps with frequency-based coloring and clustering.
"""

import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd
import json
import numpy as np
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnhancedVisualizer:
    """
    Creates interactive visualizations of spectrum assignments.
    Handles both small (<100 stations) and large (30,000+) datasets.
    """
    
    def __init__(self, assignments_df: pd.DataFrame, metrics: Optional[Dict] = None):
        """
        Initialize visualizer with assignment data.
        
        Args:
            assignments_df: DataFrame with columns: latitude, longitude, assigned_frequency
            metrics: Optional metrics dictionary from optimization
        """
        self.df = self._normalize_dataframe(assignments_df)
        self.metrics = metrics or {}
        self.freq_colors = self._generate_frequency_colors()
        
        # Calculate map bounds
        self.center_lat = self.df['latitude'].mean()
        self.center_lon = self.df['longitude'].mean()
        self.bounds = self._calculate_bounds()
        
        logger.info(f"Visualizer initialized with {len(self.df)} stations, "
                   f"{len(self.freq_colors)} unique frequencies")
    
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
        Create interactive map with appropriate visualization strategy.
        
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
        
        # Add frequency legend
        self._add_frequency_legend(m)
        
        # Add layer control if multiple layers
        if n_stations >= 1000:
            folium.LayerControl().add_to(m)
        
        # Add title
        self._add_map_title(m)
        
        # Save map
        m.save(output_path)
        logger.info(f"Map saved to {output_path}")
        
        return m
    
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
            
            # Add directional indicator for narrow beams
            if 'azimuth_deg' in row and 'beamwidth_deg' in row:
                if row['beamwidth_deg'] < 360:
                    self._add_beam_indicator(map_obj, row)
    
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
    
    def _add_beam_indicator(self, map_obj: folium.Map, station: pd.Series) -> None:
        """Add directional beam indicator for stations with narrow beamwidth."""
        # Calculate beam edges
        azimuth = station['azimuth_deg']
        beamwidth = station['beamwidth_deg']
        
        # Create a simple line showing beam direction
        # (In a full implementation, would draw sector)
        distance = 0.01  # degrees
        end_lat = station['latitude'] + distance * np.cos(np.radians(azimuth))
        end_lon = station['longitude'] + distance * np.sin(np.radians(azimuth))
        
        folium.PolyLine(
            locations=[
                [station['latitude'], station['longitude']],
                [end_lat, end_lon]
            ],
            color='red',
            weight=2,
            opacity=0.5
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
        </div>
        </div>
        '''
        
        map_obj.get_root().html.add_child(folium.Element(title_html))
    
    def generate_frequency_report(self) -> Dict:
        """Generate statistics about frequency usage."""
        report = {
            'total_stations': len(self.df),
            'unique_frequencies': len(self.freq_colors),
            'frequency_distribution': {},
            'geographic_span': {
                'lat_range': float(self.df['latitude'].max() - self.df['latitude'].min()),
                'lon_range': float(self.df['longitude'].max() - self.df['longitude'].min())
            }
        }
        
        # Count stations per frequency
        for freq in sorted(self.freq_colors.keys()):
            count = (self.df['assigned_frequency'] == freq).sum()
            report['frequency_distribution'][float(freq)] = int(count)
        
        return report


def create_visualization(assignments_path: str, metrics_path: Optional[str] = None,
                         output_path: str = "interactive_map.html") -> None:
    """
    Convenience function to create visualization from files.
    
    Args:
        assignments_path: Path to assignments CSV
        metrics_path: Optional path to metrics JSON
        output_path: Where to save HTML map
    """
    # Load data
    df = pd.read_csv(assignments_path)
    
    metrics = None
    if metrics_path and Path(metrics_path).exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    # Create visualization
    viz = EnhancedVisualizer(df, metrics)
    viz.create_interactive_map(output_path)
    
    # Generate report
    report = viz.generate_frequency_report()
    print(f"Map created: {output_path}")
    print(f"Stations: {report['total_stations']}")
    print(f"Unique frequencies: {report['unique_frequencies']}")
    print(f"Geographic span: {report['geographic_span']['lat_range']:.2f}° lat, "
          f"{report['geographic_span']['lon_range']:.2f}° lon")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualizer_enhanced.py <assignments.csv> [metrics.json] [output.html]")
        sys.exit(1)
    
    assignments = sys.argv[1]
    metrics = sys.argv[2] if len(sys.argv) > 2 else None
    output = sys.argv[3] if len(sys.argv) > 3 else "interactive_map.html"
    
    create_visualization(assignments, metrics, output)