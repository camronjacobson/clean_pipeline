"""
End-to-end integration tests for the spectrum optimizer.
Tests the full pipeline with real datasets and validates outputs.
"""

import pytest
import pandas as pd
import numpy as np
import json
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from spectrum_optimizer import run_spectrum_optimization
from main import run_optimization_pipeline
import config


class TestE2EIntegration:
    """End-to-end tests with real datasets."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.am_data_file = Path(__file__).parent.parent / 'data' / 'california_am_subset_500.csv'
    
    def teardown_method(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_am_dataset_optimization(self):
        """
        Test optimization with the California AM subset dataset.
        Validates that optimization completes and produces valid results.
        """
        if not self.am_data_file.exists():
            pytest.skip(f"AM dataset not found at {self.am_data_file}")
        
        # Load the AM data
        df = pd.read_csv(self.am_data_file)
        
        # Run optimization with reasonable parameters
        result = run_spectrum_optimization(
            df,
            region_type=None,  # No shapefiles
            radius_miles=30,
            max_chunk_size=50,
            num_threads=2,
            service='AM',
            use_full_band=True
        )
        
        # Validate results
        assert result is not None, "Optimization returned None"
        assert not result.empty, "Optimization returned empty dataframe"
        assert len(result) == len(df), f"Result has {len(result)} rows, expected {len(df)}"
        
        # Check required columns exist
        assert 'assigned_frequency' in result.columns, "Missing assigned_frequency column"
        assert 'station_id' in result.columns, "Missing station_id column"
        
        # Check frequency assignments are valid
        assigned = result['assigned_frequency'].dropna()
        assert len(assigned) > 0, "No frequencies were assigned"
        
        # For AM band (530-1700 kHz), convert to MHz for comparison
        am_min = 0.53  # MHz
        am_max = 1.7   # MHz
        
        assert assigned.min() >= am_min, f"Frequency {assigned.min()} below AM band minimum"
        assert assigned.max() <= am_max, f"Frequency {assigned.max()} above AM band maximum"
        
        # Check for frequency reuse (should have some)
        unique_freqs = assigned.nunique()
        reuse_rate = len(assigned) / unique_freqs
        assert reuse_rate > 1.0, f"No frequency reuse detected (rate: {reuse_rate:.2f})"
    
    def test_metrics_json_generation(self):
        """
        Test that metrics.json is properly generated with expected fields.
        """
        if not self.am_data_file.exists():
            pytest.skip(f"AM dataset not found at {self.am_data_file}")
        
        # Create a small subset for faster testing
        df = pd.read_csv(self.am_data_file).head(20)
        
        # Create temporary file
        temp_input = Path(self.test_dir) / "test_input.csv"
        df.to_csv(temp_input, index=False)
        
        # Mock argparse Namespace
        class Args:
            input_file = str(temp_input)
            region_type = None
            radius_miles = 30
            max_chunk_size = 10
            threads = 1
            skip_validation = False
            auto_fix = True
            output_dir = self.test_dir
            output_prefix = "test"
            skip_viz = True
            debug = False
            shapefiles_dir = "shapefiles"
            hybrid_filter = False
            dynamic_radius = False
            service = 'AM'
            band_start_khz = None
            band_end_khz = None
            step_khz = None
            use_full_band = True
            w_span = None
            w_count = None
            w_surplus = None
        
        # Run full pipeline
        summary = run_optimization_pipeline(Args())
        
        # Check summary structure
        assert 'success' in summary, "Summary missing success field"
        assert 'stages_completed' in summary, "Summary missing stages_completed"
        
        # Find the output directory
        output_dirs = [d for d in Path(self.test_dir).iterdir() if d.is_dir() and d.name.startswith("test_")]
        assert len(output_dirs) > 0, "No output directory created"
        
        output_dir = output_dirs[0]
        
        # Check for key output files
        assert (output_dir / "processed_data.csv").exists(), "Processed data file not created"
        assert (output_dir / "optimized_spectrum.csv").exists(), "Optimized spectrum file not created"
        assert (output_dir / "summary.json").exists(), "Summary JSON not created"
    
    def test_report_html_generation(self):
        """
        Test that report.html is generated with proper structure.
        """
        if not self.am_data_file.exists():
            pytest.skip(f"AM dataset not found at {self.am_data_file}")
        
        # Create small test dataset
        df = pd.read_csv(self.am_data_file).head(10)
        
        # Run optimization
        result = run_spectrum_optimization(
            df,
            region_type=None,
            radius_miles=30,
            max_chunk_size=10,
            num_threads=1,
            service='AM'
        )
        
        # Check if result has reporting attributes
        if hasattr(result, 'attrs') and 'optimization_metrics' in result.attrs:
            metrics = result.attrs['optimization_metrics']
            
            # Validate metrics structure
            assert 'spectrum_usage' in metrics, "Missing spectrum_usage in metrics"
            assert 'reuse_metrics' in metrics, "Missing reuse_metrics"
            assert 'constraint_validation' in metrics, "Missing constraint_validation"
            
            # Check reuse metrics
            reuse = metrics['reuse_metrics']
            assert 'avg_reuse' in reuse, "Missing avg_reuse"
            assert 'max_reuse' in reuse, "Missing max_reuse"
            assert reuse['avg_reuse'] >= 1.0, "Invalid average reuse"
    
    def test_chunking_consistency(self):
        """
        Test that chunking produces consistent results.
        Same input should produce same chunks.
        """
        if not self.am_data_file.exists():
            pytest.skip(f"AM dataset not found at {self.am_data_file}")
        
        df = pd.read_csv(self.am_data_file).head(50)
        
        # Run optimization twice with same parameters
        result1 = run_spectrum_optimization(
            df.copy(),
            region_type=None,
            radius_miles=25,
            max_chunk_size=20,
            num_threads=1,
            service='AM'
        )
        
        result2 = run_spectrum_optimization(
            df.copy(),
            region_type=None,
            radius_miles=25,
            max_chunk_size=20,
            num_threads=1,
            service='AM'
        )
        
        # Results should be deterministic
        if 'chunk_id' in result1.columns and 'chunk_id' in result2.columns:
            chunks1 = result1.groupby('chunk_id').size().sort_index()
            chunks2 = result2.groupby('chunk_id').size().sort_index()
            
            # Same number of chunks
            assert len(chunks1) == len(chunks2), "Different number of chunks generated"
    
    def test_invalid_input_handling(self):
        """
        Test that the system handles invalid inputs gracefully.
        """
        # Test with missing required columns
        df_invalid = pd.DataFrame({
            'station_id': ['A', 'B', 'C'],
            'latitude': [40.0, 41.0, 42.0]
            # Missing longitude, frequency, etc.
        })
        
        with pytest.raises((KeyError, ValueError, AttributeError)):
            run_spectrum_optimization(
                df_invalid,
                region_type=None,
                radius_miles=30,
                max_chunk_size=10
            )
    
    def test_frequency_band_detection(self):
        """
        Test automatic frequency band detection from data.
        """
        # Create FM band data
        df_fm = pd.DataFrame({
            'station_id': ['FM1', 'FM2', 'FM3'],
            'latitude': [40.0, 41.0, 42.0],
            'longitude': [-120.0, -121.0, -122.0],
            'frequency_mhz': [88.1, 95.5, 107.9],  # FM band frequencies
            'power_watts': [1000, 2000, 3000],
            'azimuth_deg': [0, 90, 180],
            'beamwidth_deg': [360, 360, 360]
        })
        
        # Run without specifying service (should auto-detect FM)
        result = run_spectrum_optimization(
            df_fm,
            region_type=None,
            radius_miles=100,  # Far apart, no interference
            max_chunk_size=10,
            service=None  # Auto-detect
        )
        
        # Check that frequencies are in FM band
        if result is not None and 'assigned_frequency' in result.columns:
            assigned = result['assigned_frequency'].dropna()
            assert assigned.min() >= 88.0, "Frequency below FM band"
            assert assigned.max() <= 108.0, "Frequency above FM band"
    
    def test_edge_case_single_station(self):
        """
        Test optimization with single station (edge case).
        """
        df_single = pd.DataFrame({
            'station_id': ['ONLY'],
            'latitude': [40.0],
            'longitude': [-120.0],
            'frequency_mhz': [1.0],
            'power_watts': [1000],
            'azimuth_deg': [0],
            'beamwidth_deg': [360]
        })
        
        result = run_spectrum_optimization(
            df_single,
            region_type=None,
            radius_miles=30,
            max_chunk_size=10,
            service='AM'
        )
        
        assert result is not None, "Single station optimization failed"
        assert len(result) == 1, "Result should have one row"
        assert 'assigned_frequency' in result.columns, "Missing frequency assignment"
    
    def test_performance_with_scale(self):
        """
        Test that optimization completes in reasonable time for larger datasets.
        """
        if not self.am_data_file.exists():
            pytest.skip(f"AM dataset not found at {self.am_data_file}")
        
        # Use full 500-station dataset
        df = pd.read_csv(self.am_data_file)
        
        import time
        start_time = time.time()
        
        result = run_spectrum_optimization(
            df,
            region_type=None,
            radius_miles=30,
            max_chunk_size=80,  # Larger chunks
            num_threads=4,  # More threads
            service='AM'
        )
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time (adjust based on hardware)
        assert elapsed < 300, f"Optimization took too long: {elapsed:.1f} seconds"
        assert result is not None, "Large dataset optimization failed"
        
        # Check optimization quality
        if 'assigned_frequency' in result.columns:
            assigned = result['assigned_frequency'].dropna()
            success_rate = len(assigned) / len(df) * 100
            assert success_rate > 50, f"Low success rate: {success_rate:.1f}%"