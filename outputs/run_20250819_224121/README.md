# Spectrum Optimization Results

Generated: 20250819_224121

## Configuration

- **Input File**: fmdata2_test_subset.csv
- **Region Type**: None
- **Radius**: 30.0 miles
- **Total Stations**: 9

## Results

- **Successfully Optimized**: 9 (100.0%)
- **Regions**: N/A
- **Frequency Band**: FM

## Frequency Reuse

- **Maximum Global Reuse**: N/A stations
- **Average Global Reuse**: 0.0 stations

## Files

- `processed_data.csv` - Validated and processed input data
- `optimized_spectrum.csv` - Optimization results with frequency assignments
- `summary.json` - Machine-readable summary
- `visualizations/` - Interactive visualizations and plots
- `logs/` - Detailed execution logs

## Viewing Results

1. Open `visualizations/spectrum_dashboard.html` for comprehensive analytics
2. Review `optimized_spectrum.csv` for detailed frequency assignments
3. Check logs for optimization details and any warnings

## Column Descriptions

### Key Columns in Results:
- `assigned_frequency` - Assigned frequency in MHz
- Additional columns from original data preserved
- `optimization_status` - Success/failed/reused
- `chunk_id` - Proximity chunk used for optimization

## Next Steps

To further analyze results:
```python
import pandas as pd
results = pd.read_csv('optimized_spectrum.csv')
# Analysis code here
```
