# Aethalometer Data Analysis Notebook

This directory contains Jupyter notebooks for analyzing aethalometer data using the modular analysis system.

## Available Notebooks

### `aethalometer_data_analysis.ipynb`
A comprehensive notebook that demonstrates:

- **Data Loading**: Uses both direct pandas and the modular `AethalometerPKLLoader`
- **Data Inspection**: Detailed examination of DataFrame structure and contents
- **Statistical Analysis**: Basic descriptive statistics and correlation analysis
- **Visualization**: Time series plots using the modular plotting utilities
- **Advanced Analysis**: Source apportionment and quality assessment
- **Error Handling**: Robust fallback methods for different data formats

## Getting Started

### Prerequisites
Make sure you have installed the required dependencies:
```bash
pip install -r ../requirements.txt

# Test the complete system (optional but recommended)
python ../test_system.py
```

### Running the Notebook

#### In VS Code (Recommended)
```bash
code notebooks/aethalometer_data_analysis.ipynb
```

#### Alternative Methods
```bash
# Option 1: Jupyter in browser
jupyter notebook notebooks/aethalometer_data_analysis.ipynb

# Option 2: JupyterLab
jupyter lab notebooks/aethalometer_data_analysis.ipynb
```

## Data Sources

The sample notebook is configured to work with:
- **Primary**: `/Users/ahzs645/Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University/Research/Grad/UC Davis Ann/NASA MAIA/Data/Aethelometry Data/Kyan Data/Mergedcleaned and uncleaned MA350 data20250707030704/df_uncleaned_Jacros_API_and_OG.pkl`

To use your own data:
1. Edit the `data_path` variable in the second code cell
2. Ensure your data is in pickle format (.pkl)
3. Run the notebook cells in order

## Features Demonstrated

### ✅ Modular System Integration
- Import and use all analysis components
- Automatic format detection for different data types
- Error handling and fallback methods

### ✅ Data Quality Assessment
- Missing data detection
- Outlier identification
- Data range validation
- Correlation analysis

### ✅ Visualization
- Time series plotting with the `AethalometerPlotter`
- Multiple plot types (line, scatter, distribution)
- Automatic styling and formatting

### ✅ Advanced Analytics
- Source apportionment analysis
- Black carbon concentration analysis
- Statistical summaries

## Customization

### Adding New Analysis
To add your own analysis modules:
1. Create new cells in the notebook
2. Import additional modules from `src/analysis/`
3. Follow the existing patterns for error handling

### Different Data Formats
The notebook supports:
- Standard aethalometer format
- JPL repository format
- Auto-detection of format type
- Custom column mapping

### Export Options
Results can be saved in various formats:
- JSON (using `utils.file_io.save_results_to_json`)
- CSV export of processed data
- PNG/PDF plots using matplotlib

## Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```python
   # Add this to your notebook if needed:
   import sys
   sys.path.insert(0, '../src')
   ```

2. **Data Loading Errors**
   - Check file path exists
   - Verify pickle file format
   - Try both loading methods (direct pandas vs modular)

3. **Plotting Issues**
   - Ensure matplotlib backend is set correctly
   - Check if data has datetime index for time series plots
   - Use fallback plotting methods if needed

### Getting Help

- Check the main project README: `../README.md`
- Look at example scripts in `../examples/`
- Review the modular system documentation: `../MODULAR_SYSTEM_SUMMARY.md`

---

**🎯 This notebook demonstrates the successful integration of the modular aethalometer analysis system with Jupyter notebooks!**
