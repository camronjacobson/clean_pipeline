#!/usr/bin/env python3
"""
Fixed Enhanced Visualizer that properly embeds map in dashboard.
Creates interactive maps with frequency-based coloring that display correctly.
"""

import folium
from folium.plugins import MarkerCluster
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
    Fixed version that properly embeds maps in dashboards.
    """
    
    def __init__(self, assignments_df: pd.DataFrame, metrics: Optional[Dict] = None):
        """Initialize visualizer with assignment data."""
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
        """Map frequencies to colors using spectrum colormap."""
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
            freq_colors[freq] = colors.rgb2hex(rgba[:3])
        
        return freq_colors
    
    def _calculate_zoom_level(self) -> int:
        """Calculate appropriate zoom level based on geographic extent."""
        lat_range = self.df['latitude'].max() - self.df['latitude'].min()
        lon_range = self.df['longitude'].max() - self.df['longitude'].min()
        
        max_range = max(lat_range, lon_range)
        
        if max_range > 10:
            return 5  # Country level
        elif max_range > 2:
            return 7  # State level
        elif max_range > 0.5:
            return 10  # City level
        else:
            return 12  # Neighborhood level
    
    def _create_inline_map(self) -> str:
        """Create map without iframe complications - directly embedded Leaflet."""
        stations_geojson = {
            'type': 'FeatureCollection',
            'features': []
        }
        
        # Create color mapping for frequencies
        freq_to_color = {}
        unique_freqs = sorted(self.df['assigned_frequency'].unique())
        colors_palette = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
                         '#00FFFF', '#FFA500', '#800080', '#008000', '#000080']
        
        for i, freq in enumerate(unique_freqs):
            freq_to_color[freq] = colors_palette[i % len(colors_palette)]
        
        # Build GeoJSON features
        for idx, row in self.df.iterrows():
            freq = row['assigned_frequency']
            stations_geojson['features'].append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['longitude'], row['latitude']]
                },
                'properties': {
                    'station_id': row.get('station_id', f'Station {idx}'),
                    'frequency': freq,
                    'color': freq_to_color[freq]
                }
            })
        
        # Calculate bounds for map
        lat_min, lat_max = self.df['latitude'].min(), self.df['latitude'].max()
        lon_min, lon_max = self.df['longitude'].min(), self.df['longitude'].max()
        
        return f"""
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        
        <div id="leaflet-map" style="width: 100%; height: 600px; border: 1px solid #ccc;"></div>
        
        <script>
            // Initialize map
            var map = L.map('leaflet-map').setView([{self.center_lat}, {self.center_lon}], {self._calculate_zoom_level()});
            
            // Add tile layer
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors',
                maxZoom: 18
            }}).addTo(map);
            
            // Station data
            var stationsData = {json.dumps(stations_geojson)};
            
            // Create frequency color mapping
            var freqColors = {json.dumps(freq_to_color)};
            
            // Add stations to map
            L.geoJSON(stationsData, {{
                pointToLayer: function(feature, latlng) {{
                    return L.circleMarker(latlng, {{
                        radius: 8,
                        fillColor: feature.properties.color,
                        color: '#000',
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    }});
                }},
                onEachFeature: function(feature, layer) {{
                    layer.bindPopup(
                        '<b>Station:</b> ' + feature.properties.station_id + '<br>' +
                        '<b>Frequency:</b> ' + feature.properties.frequency.toFixed(2) + ' MHz'
                    );
                }}
            }}).addTo(map);
            
            // Fit map to bounds
            map.fitBounds([[{lat_min}, {lon_min}], [{lat_max}, {lon_max}]]);
            
            // Add legend
            var legend = L.control({{position: 'bottomright'}});
            legend.onAdd = function(map) {{
                var div = L.DomUtil.create('div', 'info legend');
                div.style.backgroundColor = 'white';
                div.style.padding = '10px';
                div.style.border = '2px solid gray';
                div.style.borderRadius = '5px';
                
                div.innerHTML = '<h4 style="margin: 0 0 5px 0;">Frequencies</h4>';
                
                for (var freq in freqColors) {{
                    div.innerHTML += '<div style="margin: 2px 0;">' +
                        '<span style="background:' + freqColors[freq] + '; width: 20px; height: 10px; ' +
                        'display: inline-block; margin-right: 5px; border: 1px solid black;"></span>' +
                        '<span>' + parseFloat(freq).toFixed(1) + ' MHz</span></div>';
                }}
                
                return div;
            }};
            legend.addTo(map);
        </script>
        """
    
    def create_unified_dashboard(self, output_path: str = "dashboard.html") -> None:
        """Generate dashboard with properly embedded map."""
        
        # Generate the inline map
        map_html = self._create_inline_map()
        
        # Create dashboard metrics
        total_stations = len(self.df)
        unique_frequencies = len(self.freq_colors)
        efficiency = total_stations / max(unique_frequencies, 1)
        
        # Frequency distribution stats
        freq_counts = self.df['assigned_frequency'].value_counts()
        
        # Create spectrum utilization chart
        spectrum_chart = self._create_spectrum_chart()
        
        # Create metrics panel
        metrics_panel = self._create_metrics_panel()
        
        # Build complete dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spectrum Optimization Dashboard</title>
            <meta charset="utf-8">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #f5f5f5;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                
                .header h1 {{
                    margin: 0 0 10px 0;
                    font-size: 2.5em;
                }}
                
                .header p {{
                    opacity: 0.9;
                    font-size: 1.2em;
                }}
                
                .container {{
                    max-width: 1400px;
                    margin: 20px auto;
                    padding: 0 20px;
                }}
                
                .tab-container {{
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                
                .tab-buttons {{
                    display: flex;
                    background: #f8f9fa;
                    border-bottom: 2px solid #dee2e6;
                }}
                
                .tab-button {{
                    flex: 1;
                    padding: 15px 20px;
                    background: none;
                    border: none;
                    cursor: pointer;
                    font-size: 1em;
                    font-weight: 500;
                    color: #6c757d;
                    transition: all 0.3s;
                }}
                
                .tab-button:hover {{
                    background: #e9ecef;
                }}
                
                .tab-button.active {{
                    color: #667eea;
                    background: white;
                    border-bottom: 3px solid #667eea;
                    margin-bottom: -2px;
                }}
                
                .tab-content {{
                    display: none;
                    padding: 30px;
                    min-height: 600px;
                }}
                
                .tab-content.active {{
                    display: block;
                }}
                
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                
                .stat-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                    margin: 10px 0;
                }}
                
                .stat-label {{
                    color: #6c757d;
                    font-size: 0.9em;
                }}
                
                .freq-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                
                .freq-table th, .freq-table td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #dee2e6;
                }}
                
                .freq-table th {{
                    background: #f8f9fa;
                    font-weight: 600;
                }}
                
                .freq-table tr:hover {{
                    background: #f8f9fa;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Spectrum Optimization Dashboard</h1>
                <p>{total_stations} Stations | {unique_frequencies} Frequencies | {efficiency:.1f} Stations/Channel</p>
            </div>
            
            <div class="container">
                <div class="tab-container">
                    <div class="tab-buttons">
                        <button class="tab-button active" onclick="showTab('map')">Interactive Map</button>
                        <button class="tab-button" onclick="showTab('spectrum')">Spectrum Analysis</button>
                        <button class="tab-button" onclick="showTab('metrics')">Metrics</button>
                    </div>
                    
                    <div id="map" class="tab-content active">
                        <h2>Station Map</h2>
                        <p style="margin: 10px 0; color: #6c757d;">
                            Click markers for station details | Zoom to explore | Colors represent frequency assignments
                        </p>
                        {map_html}
                    </div>
                    
                    <div id="spectrum" class="tab-content">
                        <h2>Spectrum Analysis</h2>
                        {spectrum_chart}
                    </div>
                    
                    <div id="metrics" class="tab-content">
                        <h2>Optimization Metrics</h2>
                        {metrics_panel}
                    </div>
                </div>
            </div>
            
            <script>
                function showTab(tabName) {{
                    // Hide all tabs
                    document.querySelectorAll('.tab-content').forEach(tab => {{
                        tab.classList.remove('active');
                    }});
                    
                    // Remove active from all buttons
                    document.querySelectorAll('.tab-button').forEach(btn => {{
                        btn.classList.remove('active');
                    }});
                    
                    // Show selected tab
                    document.getElementById(tabName).classList.add('active');
                    
                    // Mark button as active
                    event.target.classList.add('active');
                }}
            </script>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(dashboard_html)
        
        print(f"Dashboard created: {output_path}")
        print(f"Total stations: {total_stations}")
        print(f"Unique frequencies: {unique_frequencies}")
        print(f"Channel efficiency: {efficiency:.2f} stations/channel")
    
    def _create_spectrum_chart(self) -> str:
        """Create spectrum utilization chart."""
        freq_counts = self.df['assigned_frequency'].value_counts().sort_index()
        
        chart_html = """
        <div class="stats-grid">
        """
        
        # Add frequency distribution table
        chart_html += """
        <div style="grid-column: 1 / -1;">
            <h3>Frequency Distribution</h3>
            <table class="freq-table">
                <thead>
                    <tr>
                        <th>Frequency (MHz)</th>
                        <th>Station Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        total = len(self.df)
        for freq, count in freq_counts.items():
            percentage = (count / total) * 100
            chart_html += f"""
                <tr>
                    <td>{freq:.2f}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """
        
        chart_html += """
                </tbody>
            </table>
        </div>
        </div>
        """
        
        return chart_html
    
    def _create_metrics_panel(self) -> str:
        """Create metrics summary panel."""
        total_stations = len(self.df)
        unique_frequencies = len(self.freq_colors)
        efficiency = total_stations / max(unique_frequencies, 1)
        
        # Geographic span
        lat_range = self.df['latitude'].max() - self.df['latitude'].min()
        lon_range = self.df['longitude'].max() - self.df['longitude'].min()
        
        metrics_html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Stations</div>
                <div class="stat-value">{total_stations:,}</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Unique Frequencies</div>
                <div class="stat-value">{unique_frequencies}</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Channel Efficiency</div>
                <div class="stat-value">{efficiency:.2f}</div>
                <div class="stat-label">stations per channel</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Geographic Coverage</div>
                <div class="stat-value">{lat_range:.1f}° × {lon_range:.1f}°</div>
                <div class="stat-label">latitude × longitude</div>
            </div>
        </div>
        """
        
        if self.metrics:
            metrics_html += """
            <h3 style="margin-top: 30px;">Optimization Details</h3>
            <div class="stats-grid">
            """
            
            if 'solve_time_seconds' in self.metrics:
                metrics_html += f"""
                <div class="stat-card">
                    <div class="stat-label">Solve Time</div>
                    <div class="stat-value">{self.metrics['solve_time_seconds']:.2f}s</div>
                </div>
                """
            
            if 'solver_status' in self.metrics:
                metrics_html += f"""
                <div class="stat-card">
                    <div class="stat-label">Solver Status</div>
                    <div class="stat-value" style="font-size: 1.2em;">{self.metrics['solver_status']}</div>
                </div>
                """
            
            metrics_html += "</div>"
        
        return metrics_html


def test_fixed_visualizer():
    """Test the fixed visualizer with sample data."""
    import webbrowser
    
    # Check for test data
    test_path = Path("runs/fm_subset_100/assignments.csv")
    if not test_path.exists():
        test_path = Path("runs/am_test/assignments.csv")
    
    if not test_path.exists():
        print("No test data found. Creating sample data...")
        # Create sample data
        np.random.seed(42)
        n_stations = 50
        
        df = pd.DataFrame({
            'station_id': [f'S{i:03d}' for i in range(n_stations)],
            'latitude': np.random.uniform(35, 45, n_stations),
            'longitude': np.random.uniform(-125, -115, n_stations),
            'assigned_frequency': np.random.choice([88.1, 91.5, 94.7, 98.3, 101.1], n_stations)
        })
        
        metrics = {
            'solve_time_seconds': 1.23,
            'solver_status': 'OPTIMAL',
            'unique_frequencies': 5
        }
    else:
        print(f"Using test data from {test_path}")
        df = pd.read_csv(test_path)
        
        metrics_path = test_path.parent / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}
    
    # Create visualizer and generate dashboard
    viz = EnhancedVisualizer(df, metrics)
    output_path = "test_dashboard_fixed.html"
    viz.create_unified_dashboard(output_path)
    
    # Open in browser
    full_path = Path(output_path).absolute()
    print(f"\nOpening dashboard in browser: {full_path}")
    webbrowser.open(f'file://{full_path}')
    
    return True


if __name__ == "__main__":
    test_fixed_visualizer()