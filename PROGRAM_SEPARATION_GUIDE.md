# Program Separation Guide - Two Optimization Systems

This directory contains **TWO SEPARATE** spectrum optimization programs that serve different purposes. Here's a clear breakdown:

---

## 🔵 PROGRAM 1: Main Pipeline (src/main.py)
**Primary Entry Point:** `src/main.py`
**Purpose:** Geographic-agnostic spectrum optimization with region support

### Key Files for Program 1:
```
src/
├── main.py                    # Main entry point
├── config.py                  # Configuration settings
├── data.py                    # Data processing
├── spectrum_optimizer.py      # Core optimization logic
├── visualizer_simple.py       # Visualization
├── graph_build.py            # Graph construction
├── solver_optimization.py     # Solver optimization
├── production_fixes.py       # Production fixes
├── streaming.py              # Streaming utilities
├── safe_boundary_reconciliation.py
├── safe_chunk_integration.py
├── chunk_diagnostics.py
├── chunk_retry_controller.py
├── two_pass_optimizer.py
├── objective_linearization.py
├── interference_predicates.py
├── metrics_integration.py
├── otel_metrics.py
├── reporting.py
├── schemas.py
└── config_flags.py

config/
├── bands.yaml                # Frequency band configurations
└── profiles/                 # AM/FM profiles
    ├── am.yaml
    ├── fm.yaml
    └── default.yaml

shapefiles/                   # Geographic boundary files
├── bea_regions.*            # BEA region shapefiles
├── pea_regions.geojson      # PEA regions
├── california_regions.geojson
└── us_states_west.geojson
```

### How to Run Program 1:
```bash
# Basic usage
python src/main.py data/fmdata2.csv --output-dir outputs/run1

# With geographic regions
python src/main.py data/fmdata2.csv \
  --output-dir outputs/run1 \
  --region-type bea \
  --radius 30 \
  --max-chunk-size 50

# With environment variables
USE_PROPER_OBJECTIVE=true python src/main.py data/fmdata2.csv \
  --output-dir outputs/test_run \
  --region-type bea \
  --radius 30
```

### Program 1 Features:
- Geographic region support (BEA, PEA, states)
- Proximity-based chunking
- Multi-threaded optimization
- Automatic data validation
- HTML/CSV report generation

---

## 🔴 PROGRAM 2: Enhanced Tool (tool/optimize.py)
**Primary Entry Point:** `tool/optimize.py`
**Purpose:** Enhanced spectrum optimization with flexible schema support

### Key Files for Program 2:
```
tool/
├── optimize.py               # Main entry point
├── dashboard_visualizer.py   # Dashboard creation
├── dashboard_visualizer_fixed.py
├── directional.py           # Directional antenna support
├── neighbors.py             # Neighbor analysis
├── scalable_visualizer.py   # Scalable visualization
├── visualizer_enhanced.py    # Enhanced visualizations
└── visualizer_enhanced_v2.py

src/                         # Shared with Program 1 but uses:
├── spectrum_optimizer_enhanced.py  # Enhanced optimizer
├── schema_normalizer.py           # Schema normalization
├── directional_integration.py     # Directional antenna integration
└── solver_optimization_fixed.py   # Fixed solver version
```

### How to Run Program 2:
```bash
# Basic usage
python -m tool.optimize data/fmdata2.csv

# With output directory
python -m tool.optimize data/fmdata2.csv --output-dir runs/fm_test

# With performance monitoring
python -m tool.optimize data/fmdata2.csv \
  --output-dir runs/fm_test \
  --memory-profile \
  --verbose

# Create dashboard
python -m tool.optimize data/fmdata2.csv \
  --output-dir runs/fm_test \
  --create-dashboard
```

### Program 2 Features:
- Flexible schema support (auto-detection)
- Memory profiling
- Enhanced visualizations with dashboards
- GeoJSON export
- HTML reports with metrics
- Directional antenna support

---

## 📊 YOUR DATA FILE: fmdata2.csv

Your `fmdata2.csv` file is located at: `data/fmdata2.csv`

### To run with Program 1 (Main Pipeline):
```bash
# Simple run
python src/main.py data/fmdata2.csv --output-dir outputs/fmdata2_run

# With all features
USE_PROPER_OBJECTIVE=true python src/main.py data/fmdata2.csv \
  --output-dir outputs/fmdata2_run \
  --region-type bea \
  --radius 30 \
  --max-chunk-size 50 \
  --threads 8
```

### To run with Program 2 (Enhanced Tool):
```bash
# Simple run
python -m tool.optimize data/fmdata2.csv

# With output and dashboard
python -m tool.optimize data/fmdata2.csv \
  --output-dir runs/fmdata2_enhanced \
  --create-dashboard \
  --verbose
```

---

## 🔍 Quick Identification Tips

### If you see these patterns, it's Program 1:
- Uses `src/main.py`
- References `USE_PROPER_OBJECTIVE` environment variable
- Has `--region-type` flag
- Outputs to timestamped directories (e.g., `run_20250819_224121`)

### If you see these patterns, it's Program 2:
- Uses `tool/optimize.py` or `python -m tool.optimize`
- Has `--create-dashboard` flag
- References `schema_normalizer`
- Has memory profiling options

---

## 📁 Test Files & Runs

### Test Data Files:
- `data/fmdata2.csv` - Main FM dataset (your file)
- `data/fm_200_subset.csv` - 200 station subset
- `data/fmdata2_subset_100.csv` - 100 station subset
- `data/california_am_subset_500.csv` - AM band test data
- `data/fm_directional_*.csv` - Directional antenna tests

### Previous Run Outputs:
- `outputs/` - Program 1 outputs
- `runs/` - Program 2 outputs
- Various HTML files in root - Test visualizations

---

## ⚡ Quick Start Recommendations

1. **For production use with geographic regions:** Use Program 1
2. **For experimentation and visualization:** Use Program 2
3. **For your fmdata2.csv file:** Either program works, choose based on features needed

Need help? Check the README.md for dependencies and setup instructions.