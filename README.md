# Spectrum Optimization Pipeline - Clean Copy

This is a clean copy of all necessary files for the spectrum optimization pipeline.

## Directory Structure
- `src/` - All Python source files
- `config/` - Configuration files (bands.yaml)
- `shapefiles/` - Geographic boundary files
- `data/` - Sample data files

## Running the Pipeline

```bash
USE_PROPER_OBJECTIVE=true python src/main.py <input_csv> \
  --output-dir <output_dir> \
  --region-type <bea|pea> \
  --radius <miles> \
  --max-chunk-size <size> \
  --threads <num>
```

## Example Command
```bash
USE_PROPER_OBJECTIVE=true python src/main.py data/your_dataset.csv \
  --output-dir outputs/test_run \
  --region-type bea \
  --radius 30 \
  --max-chunk-size 50 \
  --threads 8
```

## Required Environment Variables
- `USE_PROPER_OBJECTIVE=true` - Enable optimized frequency assignment
- `SOLVER_TIME_LIMIT=30` - Optional: Solver timeout in seconds

## Python Dependencies
- pandas
- numpy
- ortools
- scipy
- scikit-learn
- geopandas
- shapely
- plotly
- folium
