#!/usr/bin/env python3
"""
Create test shapefiles for demonstration.
Creates simple US states and regions for testing overlay functionality.
"""

import geopandas as gpd
from shapely.geometry import Polygon, Point
import pandas as pd
import json

def create_california_regions():
    """Create simplified California regions for testing."""
    
    # Define simplified California regions (rough approximations)
    regions = {
        'Northern California': Polygon([
            (-124.5, 40.0), (-124.5, 42.0), (-120.0, 42.0), 
            (-120.0, 40.0), (-124.5, 40.0)
        ]),
        'Bay Area': Polygon([
            (-123.5, 37.0), (-123.5, 38.5), (-121.5, 38.5),
            (-121.5, 37.0), (-123.5, 37.0)
        ]),
        'Central Valley': Polygon([
            (-121.5, 36.0), (-121.5, 40.0), (-119.5, 40.0),
            (-119.5, 36.0), (-121.5, 36.0)
        ]),
        'Southern California': Polygon([
            (-120.0, 32.5), (-120.0, 36.0), (-115.0, 36.0),
            (-115.0, 32.5), (-120.0, 32.5)
        ])
    }
    
    # Create GeoDataFrame
    df = pd.DataFrame([
        {'NAME': name, 'geometry': geom, 'REGION_ID': i}
        for i, (name, geom) in enumerate(regions.items())
    ])
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    
    # Save as GeoJSON
    gdf.to_file('shapefiles/california_regions.geojson', driver='GeoJSON')
    print(f"Created: shapefiles/california_regions.geojson with {len(gdf)} regions")
    
    return gdf


def create_us_states_simplified():
    """Create simplified US states for testing."""
    
    # Create a few simplified state boundaries for demonstration
    states = {
        'California': Polygon([
            (-124.5, 32.5), (-124.5, 42.0), (-114.0, 42.0),
            (-114.0, 32.5), (-124.5, 32.5)
        ]),
        'Nevada': Polygon([
            (-120.0, 35.0), (-120.0, 42.0), (-114.0, 42.0),
            (-114.0, 35.0), (-120.0, 35.0)
        ]),
        'Oregon': Polygon([
            (-124.5, 42.0), (-124.5, 46.0), (-116.5, 46.0),
            (-116.5, 42.0), (-124.5, 42.0)
        ]),
        'Washington': Polygon([
            (-124.7, 46.0), (-124.7, 49.0), (-117.0, 49.0),
            (-117.0, 46.0), (-124.7, 46.0)
        ]),
        'Arizona': Polygon([
            (-114.8, 31.3), (-114.8, 37.0), (-109.0, 37.0),
            (-109.0, 31.3), (-114.8, 31.3)
        ])
    }
    
    df = pd.DataFrame([
        {'NAME': name, 'STATE_ABBR': name[:2].upper(), 'geometry': geom, 'STATE_ID': i}
        for i, (name, geom) in enumerate(states.items())
    ])
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    
    # Save as GeoJSON
    gdf.to_file('shapefiles/us_states_west.geojson', driver='GeoJSON')
    print(f"Created: shapefiles/us_states_west.geojson with {len(gdf)} states")
    
    return gdf


def create_bea_regions():
    """Create simplified BEA economic regions."""
    
    # Create simplified BEA-like regions
    regions = {
        'Pacific': Polygon([
            (-124.7, 32.5), (-124.7, 49.0), (-114.0, 49.0),
            (-114.0, 32.5), (-124.7, 32.5)
        ]),
        'Mountain': Polygon([
            (-114.0, 31.3), (-114.0, 49.0), (-104.0, 49.0),
            (-104.0, 31.3), (-114.0, 31.3)
        ]),
        'Southwest': Polygon([
            (-109.0, 28.0), (-109.0, 37.0), (-94.0, 37.0),
            (-94.0, 28.0), (-109.0, 28.0)
        ])
    }
    
    df = pd.DataFrame([
        {'NAME': name, 'BEA_REGION': name, 'geometry': geom, 'REGION_ID': i}
        for i, (name, geom) in enumerate(regions.items())
    ])
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    
    # Save as both GeoJSON and Shapefile
    gdf.to_file('shapefiles/bea_regions.geojson', driver='GeoJSON')
    print(f"Created: shapefiles/bea_regions.geojson with {len(gdf)} regions")
    
    # Also create as shapefile
    gdf.to_file('shapefiles/bea_regions.shp')
    print(f"Created: shapefiles/bea_regions.shp with {len(gdf)} regions")
    
    return gdf


def create_invalid_shapefile():
    """Create an invalid shapefile for error testing."""
    
    # Create a file that looks like GeoJSON but has invalid geometry
    invalid_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"NAME": "Invalid Region"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[180, 90], [181, 91], [182, 92]]]  # Invalid coords
                }
            }
        ]
    }
    
    with open('shapefiles/invalid_test.geojson', 'w') as f:
        json.dump(invalid_data, f)
    
    print("Created: shapefiles/invalid_test.geojson (intentionally invalid for testing)")


if __name__ == "__main__":
    print("Creating test shapefiles...")
    
    # Create various test shapefiles
    create_california_regions()
    create_us_states_simplified()
    create_bea_regions()
    create_invalid_shapefile()
    
    print("\nTest shapefiles created successfully!")
    print("Files created in: shapefiles/")
    print("  - california_regions.geojson")
    print("  - us_states_west.geojson")
    print("  - bea_regions.geojson")
    print("  - bea_regions.shp")
    print("  - invalid_test.geojson (for error testing)")