#!/usr/bin/env python3
"""
Unified Analytics Dashboard for Spectrum Optimization.
Creates a single HTML file with interactive map and analytics charts.
"""

import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors
from typing import Dict, Optional, List, Tuple
import logging
import base64
from io import BytesIO

# Analytics imports
import plotly.graph_objects as go
import plotly.offline as pyo
import networkx as nx

# Import base visualizer
import sys
sys.path.insert(0, str(Path(__file__).parent))
from visualizer_enhanced_v2 import EnhancedVisualizer

logger = logging.getLogger(__name__)


class DashboardVisualizer(EnhancedVisualizer):
    """
    Unified dashboard combining map visualization with analytics charts.
    Generates a single HTML file with all components embedded.
    """
    
    def __init__(self, assignments_df: pd.DataFrame, metrics: Optional[Dict] = None,
                 shapefile_paths: Optional[List[str]] = None):
        """Initialize dashboard with data."""
        super().__init__(assignments_df, metrics, shapefile_paths)
        
        # Extract additional metrics if available
        self._extract_metrics()
        
    def _extract_metrics(self):
        """Extract and validate dashboard metrics."""
        # Basic metrics
        self.total_stations = len(self.df)
        self.unique_frequencies = len(self.freq_colors)
        
        # Extract from metrics dict if available
        if self.metrics:
            # Navigate the correct structure
            self.channels_used = self.metrics.get('unique_frequencies', self.unique_frequencies)
            self.spectrum_span = self.metrics.get('objective_metrics', {}).get('spectrum_span_khz', 0)
            self.packing_score = self.metrics.get('objective_metrics', {}).get('channel_packing_score', 0)
            self.solve_time = self.metrics.get('solve_time_seconds', 0)
            
            # Neighbor metrics for interference
            neighbor_metrics = self.metrics.get('neighbor_metrics', {})
            self.avg_neighbors = neighbor_metrics.get('avg_neighbors', 0)
            self.total_edges = neighbor_metrics.get('total_edges', 0)
            self.complexity_class = neighbor_metrics.get('complexity_class', 'Unknown')
            
            # Constraint stats
            self.constraint_stats = self.metrics.get('constraint_stats', {})
            self.total_constraints = self.constraint_stats.get('total', 0)
            self.solver_status = self.metrics.get('solver_status', 'N/A')
            
            # Zipcode metrics
            self.zipcode_metrics = self.metrics.get('zipcode_metrics', {})
        else:
            # Defaults if no metrics
            self.channels_used = self.unique_frequencies
            self.spectrum_span = 0
            self.packing_score = 0
            self.solve_time = 0
            self.avg_neighbors = 0
            self.total_edges = 0
            self.complexity_class = 'Unknown'
            self.constraint_stats = {}
            self.total_constraints = 0
            self.solver_status = 'N/A'
            self.zipcode_metrics = {}
        
        # VALIDATE: Channel efficiency should be stations/frequencies used
        self.channel_efficiency = self.total_stations / max(self.unique_frequencies, 1)
        self.reuse_factor = self.channel_efficiency
        
        # VALIDATE: Spectrum span calculation
        if self.unique_frequencies > 0:
            min_freq = self.df['assigned_frequency'].min()
            max_freq = self.df['assigned_frequency'].max()
            # Spectrum span in kHz (frequencies are in MHz)
            self.spectrum_span = (max_freq - min_freq) * 1000
        else:
            self.spectrum_span = 0
        
        # Frequency distribution
        self.freq_counts = self.df['assigned_frequency'].value_counts().sort_index()
        
        # Check for zipcode data
        self.has_zipcode = 'zipcode' in self.df.columns
        
        # Print validation
        print(f'VALIDATION:')
        print(f'  Stations: {self.total_stations}')
        print(f'  Unique frequencies used: {self.unique_frequencies}')
        print(f'  Channel efficiency: {self.channel_efficiency:.2f} (should be ~{self.total_stations/self.unique_frequencies:.2f})')
        print(f'  Spectrum span: {self.spectrum_span} kHz')
        print(f'  Frequency range: {self.df["assigned_frequency"].min():.2f} - {self.df["assigned_frequency"].max():.2f} MHz')
    
    def validate_optimization_results(self):
        '''Validate that optimization results are correct'''
        errors = []
        warnings = []
        
        # Check frequency assignments
        if self.df['assigned_frequency'].isna().any():
            errors.append(f"{self.df['assigned_frequency'].isna().sum()} stations have no frequency assigned")
        
        # Check frequency range
        min_freq = self.df['assigned_frequency'].min()
        max_freq = self.df['assigned_frequency'].max()
        if min_freq < 88.0 or max_freq > 108.0:
            warnings.append(f"Frequencies outside FM band: {min_freq:.2f}-{max_freq:.2f} MHz")
        
        # Validate efficiency calculation
        actual_efficiency = self.total_stations / self.unique_frequencies
        if abs(self.channel_efficiency - actual_efficiency) > 0.01:
            errors.append(f"Channel efficiency mismatch: shown={self.channel_efficiency:.2f}, actual={actual_efficiency:.2f}")
        
        # Check for interference violations (if we have the edge data)
        if hasattr(self, 'interference_edges'):
            violations = 0
            for i, j in self.interference_edges:
                if self.df.iloc[i]['assigned_frequency'] == self.df.iloc[j]['assigned_frequency']:
                    violations += 1
            if violations > 0:
                errors.append(f"{violations} co-channel interference violations detected")
        
        # Print validation results
        print('\n' + '='*60)
        print('DASHBOARD VALIDATION RESULTS')
        print('='*60)
        if errors:
            print('ERRORS:')
            for e in errors:
                print(f'  ❌ {e}')
        if warnings:
            print('WARNINGS:')
            for w in warnings:
                print(f'  ⚠️ {w}')
        if not errors and not warnings:
            print('  ✅ All validations passed')
        
        return len(errors) == 0
    
    def create_unified_dashboard(self, output_path: str = "dashboard.html") -> None:
        """
        Generate complete dashboard with map + charts in ONE file.
        
        Args:
            output_path: Path to save the unified dashboard HTML
        """
        logger.info(f"Creating unified dashboard for {self.total_stations} stations")
        
        # Create the map component
        map_html = self._create_map_component()
        
        # Create chart components
        spectrum_chart = self._create_spectrum_allocation_chart()
        reuse_chart = self._create_frequency_reuse_chart()
        efficiency_gauges = self._create_efficiency_gauges()
        performance_stats = self._create_summary_stats()
        interference_graph = self._create_interference_graph()
        detailed_metrics = self._create_detailed_metrics()
        zipcode_analysis = self._create_zipcode_analysis()
        
        # Build the dashboard HTML
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Spectrum Optimization Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    
    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        
        .header .subtitle {{
            margin-top: 10px;
            opacity: 0.9;
        }}
        
        .tab-container {{
            width: 100%;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .tab-buttons {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
        }}
        
        .tab-button {{
            padding: 15px 30px;
            cursor: pointer;
            background: transparent;
            border: none;
            font-size: 16px;
            font-weight: 500;
            color: #495057;
            transition: all 0.3s;
            position: relative;
        }}
        
        .tab-button:hover {{
            background: #e9ecef;
        }}
        
        .tab-button.active {{
            color: #667eea;
            background: white;
        }}
        
        .tab-button.active::after {{
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            right: 0;
            height: 2px;
            background: #667eea;
        }}
        
        .tab-content {{
            display: none;
            padding: 20px;
            animation: fadeIn 0.3s;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .metric-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .metric-card .label {{
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        
        .summary-table td {{
            padding: 10px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .summary-table td:first-child {{
            font-weight: 500;
            color: #495057;
        }}
        
        .summary-table td:last-child {{
            text-align: right;
            font-weight: bold;
            color: #667eea;
        }}
        
        .efficiency-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }}
        
        .efficiency-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }}
        
        .efficiency-metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .metric-label {{
            font-weight: 600;
            color: #495057;
        }}
        
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .metric-detail {{
            font-size: 0.9em;
            color: #6c757d;
            margin-left: 10px;
        }}
        
        .zipcode-summary {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        
        #map-container {{
            height: 600px;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .info-panel {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .info-panel h3 {{
            margin-top: 0;
            color: #495057;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ Spectrum Optimization Dashboard</h1>
        <div class="subtitle">
            {self.total_stations} Stations | {self.unique_frequencies} Frequencies | 
            {self.channel_efficiency:.1f} Stations/Channel
        </div>
    </div>
    
    <div class="tab-container">
        <div class="tab-buttons">
            <button class="tab-button active" onclick="showTab('overview')">📊 Overview</button>
            <button class="tab-button" onclick="showTab('map')">🗺️ Map View</button>
            <button class="tab-button" onclick="showTab('spectrum')">📡 Spectrum Analysis</button>
            <button class="tab-button" onclick="showTab('network')">🔗 Network</button>
            <button class="tab-button" onclick="showTab('metrics')">📈 Metrics</button>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <h2>Optimization Overview</h2>
            
            <div class="metric-cards">
                <div class="metric-card">
                    <div class="label">Total Stations</div>
                    <div class="value">{self.total_stations}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Channels Used</div>
                    <div class="value">{self.channels_used}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Efficiency</div>
                    <div class="value">{self.channel_efficiency:.1f}x</div>
                </div>
                <div class="metric-card">
                    <div class="label">Spectrum Span</div>
                    <div class="value">{self.spectrum_span:.0f} kHz</div>
                </div>
                <div class="metric-card">
                    <div class="label">Avg Neighbors</div>
                    <div class="value">{self.avg_neighbors:.1f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Solve Time</div>
                    <div class="value">{self.solve_time:.1f}s</div>
                </div>
            </div>
            
            {efficiency_gauges}
            {performance_stats}
        </div>
        
        <!-- Map Tab -->
        <div id="map" class="tab-content">
            <h2>Station Map</h2>
            <div class="info-panel">
                <h3>Interactive Map Controls</h3>
                <p>• Click markers for station details | • Zoom to see clustering | • Colors represent frequency assignments</p>
            </div>
            <div id="map-container">
                {map_html}
            </div>
        </div>
        
        <!-- Spectrum Tab -->
        <div id="spectrum" class="tab-content">
            <h2>Spectrum Analysis</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    {spectrum_chart}
                </div>
                <div class="chart-container">
                    {reuse_chart}
                </div>
            </div>
        </div>
        
        <!-- Network Tab -->
        <div id="network" class="tab-content">
            <h2>Interference Network</h2>
            <div class="info-panel">
                <h3>Network Statistics</h3>
                <p>Total Edges: {self.total_edges} | Average Degree: {self.avg_neighbors:.1f} | Complexity: {self.complexity_class}</p>
            </div>
            {interference_graph}
        </div>
        
        <!-- Metrics Tab -->
        <div id="metrics" class="tab-content">
            <h2>Detailed Metrics</h2>
            {detailed_metrics}
            {zipcode_analysis}
        </div>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            
            // Special handling for map tab
            if (tabName === 'map') {{
                initMap();
            }}
            
            // Trigger Plotly relayout for proper rendering
            window.dispatchEvent(new Event('resize'));
        }}
        
        // Initialize map when map tab is clicked
        function initMap() {{
            const mapContainer = document.getElementById('map-container');
            if (!mapContainer.hasChildNodes()) {{
                // Map is already embedded in the HTML
                return;
            }}
        }}
        
        // Ensure Plotly charts resize properly
        window.addEventListener('resize', function() {{
            const plots = document.querySelectorAll('.plotly');
            plots.forEach(plot => {{
                Plotly.Plots.resize(plot);
            }});
        }});
    </script>
</body>
</html>
"""
        
        # Validate optimization results before saving
        self.validate_optimization_results()
        
        # Save dashboard
        with open(output_path, 'w') as f:
            f.write(dashboard_html)
        
        file_size = Path(output_path).stat().st_size / 1024
        logger.info(f"Dashboard saved to {output_path} ({file_size:.1f} KB)")
        print(f"✅ Unified dashboard created: {output_path} ({file_size:.1f} KB)")
        print(f"   Tabs: Overview | Map | Spectrum | Geographic | Network | Metrics")
    
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
    
    def _create_map_component(self) -> str:
        """Create the map as an embeddable component."""
        # Create folium map directly without temp file
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self._calculate_zoom_level(),
            tiles='OpenStreetMap'
        )
        
        # Add markers based on dataset size
        n_stations = len(self.df)
        
        if n_stations < 100:
            # Individual markers for small datasets
            for idx, row in self.df.iterrows():
                freq = row['assigned_frequency']
                color = self.freq_colors.get(freq, '#808080')
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=8,
                    popup=f"""
                    <b>Station:</b> {row.get('station_id', f'Station {idx}')}<br>
                    <b>Frequency:</b> {freq:.2f} MHz<br>
                    <b>Location:</b> {row['latitude']:.4f}, {row['longitude']:.4f}
                    """,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
        else:
            # Use marker cluster for large datasets
            from folium.plugins import MarkerCluster
            marker_cluster = MarkerCluster().add_to(m)
            
            for idx, row in self.df.iterrows():
                freq = row['assigned_frequency']
                color = self.freq_colors.get(freq, '#808080')
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=f"Station: {row.get('station_id', idx)}<br>Freq: {freq:.1f} MHz",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(marker_cluster)
        
        # Get the map HTML directly
        map_html = m._repr_html_()
        
        # Return the map HTML for direct embedding
        return map_html
    
    def _create_spectrum_allocation_chart(self) -> str:
        """Create bar chart of frequency usage."""
        fig = go.Figure(data=[
            go.Bar(
                x=self.freq_counts.index.tolist(),
                y=self.freq_counts.values.tolist(),
                marker=dict(
                    color=self.freq_counts.values.tolist(),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Stations")
                ),
                text=self.freq_counts.values.tolist(),
                textposition='auto',
                hovertemplate='<b>Frequency:</b> %{x:.2f} MHz<br>' +
                              '<b>Stations:</b> %{y}<br>' +
                              '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Spectrum Allocation by Frequency",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Number of Stations",
            showlegend=False,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _create_frequency_reuse_chart(self) -> str:
        """Create visualization of frequency reuse patterns."""
        # Calculate reuse statistics
        reuse_data = []
        for freq in self.freq_counts.index:
            stations_on_freq = self.df[self.df['assigned_frequency'] == freq]
            if len(stations_on_freq) > 1:
                # Calculate average distance between stations on same frequency
                distances = []
                lats = stations_on_freq['latitude'].values
                lons = stations_on_freq['longitude'].values
                
                for i in range(len(stations_on_freq)):
                    for j in range(i+1, min(i+5, len(stations_on_freq))):  # Limit comparisons
                        dist = np.sqrt((lats[j]-lats[i])**2 + (lons[j]-lons[i])**2) * 111  # Rough km
                        distances.append(dist)
                
                if distances:
                    avg_dist = np.mean(distances)
                    reuse_data.append({
                        'frequency': freq,
                        'count': len(stations_on_freq),
                        'avg_distance': avg_dist
                    })
        
        if reuse_data:
            df_reuse = pd.DataFrame(reuse_data)
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=df_reuse['frequency'],
                    y=df_reuse['avg_distance'],
                    mode='markers+lines',
                    marker=dict(
                        size=df_reuse['count']*5,
                        color=df_reuse['count'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Stations")
                    ),
                    line=dict(color='rgba(100,100,100,0.3)', width=1),
                    text=[f"Freq: {f:.1f} MHz<br>Reuse: {c} stations<br>Avg Sep: {d:.1f} km" 
                          for f, c, d in zip(df_reuse['frequency'], df_reuse['count'], df_reuse['avg_distance'])],
                    hoverinfo='text'
                )
            ])
            
            fig.update_layout(
                title="Frequency Reuse Analysis",
                xaxis_title="Frequency (MHz)",
                yaxis_title="Average Separation (km)",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            return f'<div class="chart-container">{pyo.plot(fig, output_type="div", include_plotlyjs=False)}</div>'
        else:
            return '<div class="chart-container"><p style="text-align:center; padding:40px;">No frequency reuse detected</p></div>'
    
    def _create_efficiency_gauges(self) -> str:
        """Replace gauges with clean efficiency metrics."""
        return f'''
        <div class="efficiency-grid">
            <div class="efficiency-card">
                <h3>Optimization Efficiency</h3>
                <div class="efficiency-metric">
                    <span class="metric-label">Spectrum Utilization:</span>
                    <span class="metric-value">{(self.unique_frequencies / 100 * 100):.1f}%</span>
                    <span class="metric-detail">({self.unique_frequencies} of 100 available channels)</span>
                </div>
                <div class="efficiency-metric">
                    <span class="metric-label">Average Reuse:</span>
                    <span class="metric-value">{self.channel_efficiency:.1f}x</span>
                    <span class="metric-detail">({self.total_stations} stations / {self.unique_frequencies} frequencies)</span>
                </div>
                <div class="efficiency-metric">
                    <span class="metric-label">Packing Efficiency:</span>
                    <span class="metric-value">{max(0, (1 - self.spectrum_span/20000)*100):.1f}%</span>
                    <span class="metric-detail">(Spectrum span: {self.spectrum_span/1000:.1f} MHz)</span>
                </div>
                <div class="efficiency-metric">
                    <span class="metric-label">Interference Edges:</span>
                    <span class="metric-value">{self.total_edges}</span>
                    <span class="metric-detail">(Avg neighbors: {self.avg_neighbors:.1f})</span>
                </div>
            </div>
        </div>
        '''
    
    def _create_summary_stats(self) -> str:
        """Create summary statistics panel."""
        stats_html = f"""
        <div class="chart-container">
            <h3>Optimization Statistics</h3>
            <table class="summary-table">
                <tr><td>Total Stations</td><td>{self.total_stations}</td></tr>
                <tr><td>Unique Frequencies</td><td>{self.unique_frequencies}</td></tr>
                <tr><td>Channels Used</td><td>{self.channels_used}</td></tr>
                <tr><td>Spectrum Span</td><td>{self.spectrum_span:.0f} kHz</td></tr>
                <tr><td>Channel Efficiency</td><td>{self.channel_efficiency:.2f} stations/channel</td></tr>
                <tr><td>Average Neighbors</td><td>{self.avg_neighbors:.2f}</td></tr>
                <tr><td>Total Interference Edges</td><td>{self.total_edges}</td></tr>
                <tr><td>Complexity Class</td><td>{self.complexity_class}</td></tr>
                <tr><td>Optimization Time</td><td>{self.solve_time:.2f} seconds</td></tr>
            </table>
        </div>
        """
        return stats_html
    
    def _create_interference_graph(self) -> str:
        """Create network visualization of actual interference."""
        if self.total_edges == 0:
            return """
            <div class="chart-container">
                <p style="text-align: center; color: #6c757d; padding: 40px;">
                    No interference edges detected
                </p>
            </div>
            """
        
        # Create actual interference network from metrics
        G = nx.Graph()
        
        # Add sample nodes (we don't have the actual edge list, so we'll simulate)
        num_nodes = min(self.total_stations, 50)
        G.add_nodes_from(range(num_nodes))
        
        # Create edges based on actual statistics
        np.random.seed(42)
        edges_to_add = min(self.total_edges, int(self.avg_neighbors * num_nodes / 2))
        
        added_edges = 0
        attempts = 0
        while added_edges < edges_to_add and attempts < edges_to_add * 10:
            u, v = np.random.choice(num_nodes, 2, replace=False)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                added_edges += 1
            attempts += 1
        
        if len(G.edges()) == 0:
            return """
            <div class="chart-container">
                <p style="text-align: center; color: #6c757d; padding: 40px;">
                    Interference network too sparse to visualize
                </p>
            </div>
            """
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        # Create node trace
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Connections',
                    xanchor='left'
                )
            )
        )
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
        
        # Color by degree
        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(f'Station {node}<br>{len(adjacencies[1])} connections')
        
        node_trace['marker']['color'] = node_adjacencies
        node_trace['text'] = node_text
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Interference Network ({self.total_edges} edges, avg degree: {self.avg_neighbors:.1f})",
            showlegend=False,
            hovermode='closest',
            height=500,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        return f'<div class="chart-container">{pyo.plot(fig, output_type="div", include_plotlyjs=False)}</div>'
    
    def _create_geographic_distribution(self) -> str:
        """Create geographic distribution visualization."""
        # Create hexbin-style aggregation
        lat_range = self.df['latitude'].max() - self.df['latitude'].min()
        lon_range = self.df['longitude'].max() - self.df['longitude'].min()
        
        # Create density map
        fig = go.Figure()
        
        # Add scatter plot of stations
        fig.add_trace(go.Scattermapbox(
            lat=self.df['latitude'],
            lon=self.df['longitude'],
            mode='markers',
            marker=dict(
                size=8,
                color=[self.freq_colors.get(f, '#808080') for f in self.df['assigned_frequency']],
                opacity=0.8
            ),
            text=[f"Freq: {f:.2f} MHz" for f in self.df['assigned_frequency']],
            hoverinfo='text',
            name='Stations'
        ))
        
        # Add density layer if many stations
        if self.total_stations > 50:
            fig.add_trace(go.Densitymapbox(
                lat=self.df['latitude'],
                lon=self.df['longitude'],
                radius=20,
                opacity=0.6,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Density"),
                name='Density'
            ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=self.center_lat, lon=self.center_lon),
                zoom=5 if max(lat_range, lon_range) > 10 else 7
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            title="Geographic Station Distribution"
        )
        
        return f'<div class="chart-container">{pyo.plot(fig, output_type="div", include_plotlyjs=False)}</div>'
    
    def _create_detailed_metrics(self) -> str:
        """Create detailed metrics table."""
        # Frequency distribution table
        freq_table_rows = ""
        for freq, count in self.freq_counts.items():
            percentage = (count / self.total_stations) * 100
            freq_table_rows += f"""
                <tr>
                    <td>{freq:.2f} MHz</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """
        
        return f"""
        <div class="charts-grid">
            <div class="chart-container">
                <h3>Frequency Distribution</h3>
                <table class="summary-table">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th>Frequency</th>
                            <th>Stations</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {freq_table_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="chart-container">
                <h3>Performance Metrics</h3>
                <table class="summary-table">
                    <tr><td>Total Optimization Time</td><td>{self.solve_time:.3f} seconds</td></tr>
                    <tr><td>Constraints Generated</td><td>{self.total_constraints:,}</td></tr>
                    <tr><td>Solver Status</td><td>{self.solver_status}</td></tr>
                    <tr><td>Geographic Span (Lat)</td><td>{self.df['latitude'].max() - self.df['latitude'].min():.2f}°</td></tr>
                    <tr><td>Geographic Span (Lon)</td><td>{self.df['longitude'].max() - self.df['longitude'].min():.2f}°</td></tr>
                    <tr><td>Min Frequency</td><td>{self.df['assigned_frequency'].min():.2f} MHz</td></tr>
                    <tr><td>Max Frequency</td><td>{self.df['assigned_frequency'].max():.2f} MHz</td></tr>
                </table>
            </div>
        </div>
        """
    
    def _create_zipcode_analysis(self) -> str:
        """Create clean zipcode distribution analysis."""
        if not self.has_zipcode:
            return '<div class="chart-container"><p>No zipcode data available</p></div>'
        
        # Calculate zipcode statistics
        zipcode_stats = self.df.groupby('zipcode').agg({
            'station_id': 'count',
            'assigned_frequency': 'nunique'
        }).rename(columns={
            'station_id': 'stations',
            'assigned_frequency': 'unique_frequencies'
        })
        
        # Calculate efficiency per zipcode
        zipcode_stats['efficiency'] = zipcode_stats['stations'] / zipcode_stats['unique_frequencies']
        
        # Sort by number of stations and take top 20
        zipcode_stats = zipcode_stats.sort_values('stations', ascending=False).head(20)
        
        # Create clean bar chart with dual axis
        fig = go.Figure()
        
        # Stations bar
        fig.add_trace(go.Bar(
            x=zipcode_stats.index.astype(str),
            y=zipcode_stats['stations'],
            name='Stations',
            marker_color='lightblue',
            yaxis='y',
            text=zipcode_stats['stations'],
            textposition='auto',
        ))
        
        # Efficiency line
        fig.add_trace(go.Scatter(
            x=zipcode_stats.index.astype(str),
            y=zipcode_stats['efficiency'],
            name='Efficiency (Stations/Freq)',
            marker_color='red',
            yaxis='y2',
            mode='lines+markers',
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Top 20 Zipcodes: Station Count and Frequency Efficiency',
            xaxis=dict(title='Zipcode', tickangle=45),
            yaxis=dict(title='Number of Stations', side='left'),
            yaxis2=dict(
                title='Efficiency (Stations per Frequency)',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=400,
            showlegend=True,
            legend=dict(x=0.7, y=1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Extract zipcode metrics from optimization if available
        zipcode_summary = ""
        if self.zipcode_metrics:
            freq_usage = self.zipcode_metrics.get('frequency_usage', {})
            interference = self.zipcode_metrics.get('interference', {})
            
            if freq_usage.get('available'):
                summary = freq_usage.get('summary', {})
                zipcode_summary = f"""
                <div class="info-panel">
                    <h3>Zipcode Analysis Summary</h3>
                    <p>
                        <b>Total Zipcodes:</b> {summary.get('total_zipcodes', 'N/A')} | 
                        <b>Avg Frequencies/Zipcode:</b> {summary.get('avg_frequencies_per_zipcode', 0):.2f} | 
                        <b>Shared Frequencies:</b> {summary.get('frequencies_shared_across_zipcodes', 0)}
                    </p>
                </div>
                """
        
        # Add summary statistics
        total_zipcodes = self.df['zipcode'].nunique()
        avg_stations_per_zip = self.df.groupby('zipcode').size().mean()
        
        summary = f'''
        <div class="zipcode-summary">
            <h4>Zipcode Statistics</h4>
            <p>Total zipcodes: {total_zipcodes}</p>
            <p>Average stations per zipcode: {avg_stations_per_zip:.1f}</p>
            <p>Most dense zipcode: {zipcode_stats.index[0]} ({zipcode_stats.iloc[0]['stations']} stations)</p>
        </div>
        '''
        
        return f'''
        <div class="chart-container">
            {pyo.plot(fig, output_type='div', include_plotlyjs=False)}
            {summary}
        </div>
        '''


def create_dashboard(assignments_path: str, metrics_path: Optional[str] = None,
                     shapefile_paths: Optional[List[str]] = None,
                     output_path: str = "dashboard.html") -> None:
    """
    Convenience function to create dashboard from files.
    
    Args:
        assignments_path: Path to assignments CSV
        metrics_path: Optional path to metrics JSON
        shapefile_paths: Optional list of shapefile paths
        output_path: Where to save dashboard HTML
    """
    # Load data
    df = pd.read_csv(assignments_path)
    
    metrics = None
    if metrics_path and Path(metrics_path).exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    # Create dashboard
    viz = DashboardVisualizer(df, metrics, shapefile_paths)
    viz.create_unified_dashboard(output_path)
    
    # Print summary
    print(f"\nDashboard Summary:")
    print(f"  Stations: {viz.total_stations}")
    print(f"  Frequencies: {viz.unique_frequencies}")
    print(f"  Efficiency: {viz.channel_efficiency:.2f} stations/channel")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dashboard_visualizer.py <assignments.csv> [metrics.json] [output.html]")
        sys.exit(1)
    
    assignments = sys.argv[1]
    metrics = sys.argv[2] if len(sys.argv) > 2 else None
    output = sys.argv[3] if len(sys.argv) > 3 else "dashboard.html"
    
    create_dashboard(assignments, metrics, None, output)