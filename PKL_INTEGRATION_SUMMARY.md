# PKL Data Cleaning Integration

This document describes the integration of the PKL data cleaning pipeline into the aethmodular project structure.

## What Was Integrated

The PKL data cleaning pipeline has been successfully integrated from the original `PKL data cleaning pipeline/` folder into the main project structure:

### Files Integrated:

1. **External Calibration Module**: 
   - Original: `PKL data cleaning pipeline/calibration.py`
   - New location: `src/external/calibration.py`
   - **Status**: Preserved as-is for easy updates from external source

2. **PKL Cleaning Pipeline**:
   - Original: `PKL data cleaning pipeline/data_cleaning_pipeline_no_aethpy.py`
   - New location: `src/data/qc/pkl_cleaning.py`
   - **Status**: Refactored and integrated with configurable paths

## Key Improvements

### 🔧 Configurable Data Directory

The major improvement is that the data directory path is now **configurable** instead of hardcoded:

**Before:**
```python
# Hardcoded path
df = calibration.readall_BCdata_from_dir("../JPL_aeth/", ...)
```

**After:**
```python
# Configurable path
cleaner = PKLDataCleaner(data_directory="/your/custom/path/")
df = cleaner.load_and_clean_data(...)
```

### 📁 Project Structure Integration

The PKL cleaning functionality is now properly integrated into the existing QC (Quality Control) module structure:

```
src/data/qc/
├── __init__.py                 # Updated with PKL imports
├── pkl_cleaning.py            # Main PKL cleaning module
├── missing_data.py            # Existing QC modules
├── quality_classifier.py
└── ...

src/external/
└── calibration.py             # External calibration (unchanged)
```

## Usage Examples

### 1. Class-Based Approach (Recommended)

```python
from src.data.qc import PKLDataCleaner

# Configure with custom data directory
cleaner = PKLDataCleaner(data_directory="/path/to/your/pkl/data/")

# Load and clean data
df_cleaned = cleaner.load_and_clean_data()
```

### 2. Function-Based Approach

```python
from src.data.qc import load_and_clean_pkl_data

# Direct function call with custom path
df_cleaned = load_and_clean_pkl_data(directory_path="/path/to/your/pkl/data/")
```

### 3. Individual Cleaning Steps

```python
from src.data.qc import PKLDataCleaner

cleaner = PKLDataCleaner(data_directory="/your/path/")

# Apply individual cleaning steps
df_step1 = cleaner.clean_by_status(df)
df_step2 = cleaner.clean_optical_saturation(df_step1)
df_step3 = cleaner.clean_flow_range(df_step2)
```

## Configuration Options

The PKL cleaning pipeline now supports multiple configuration approaches:

### Environment Variable
```bash
export PKL_DATA_PATH="/opt/data/aethalometer/pkl/"
```

```python
import os
data_dir = os.getenv('PKL_DATA_PATH', '../JPL_aeth/')
cleaner = PKLDataCleaner(data_directory=data_dir)
```

### Command Line Arguments
```bash
python pkl_cleaning_example.py /path/to/data/
```

### Configuration File
```python
import json
with open("config.json") as f:
    config = json.load(f)
cleaner = PKLDataCleaner(data_directory=config["pkl_data_path"])
```

## Available Resources

### 📓 Jupyter Notebook
- **Location**: `notebooks/pkl_data_cleaning_demo.ipynb`
- **Purpose**: Interactive demonstration of PKL cleaning with configurable paths
- **Features**: Step-by-step examples, configuration options, quality assessment

### 🐍 Python Example
- **Location**: `examples/pkl_cleaning_example_configurable.py`
- **Purpose**: Command-line example script
- **Usage**: `python pkl_cleaning_example_configurable.py [data_directory]`

### 📚 Documentation
- **PKL Cleaning README**: `src/data/qc/PKL_CLEANING_README.md`
- **Module Documentation**: Comprehensive docstrings in `src/data/qc/pkl_cleaning.py`

## Integration Benefits

### ✅ Modular Design
- PKL cleaning is now a proper module within the QC framework
- Can be combined with other QC tools (missing data analysis, quality classification, etc.)

### ✅ Configurable Paths
- No more hardcoded directory paths
- Supports multiple deployment environments
- Easy to adapt for different data locations

### ✅ External Calibration Preserved
- External calibration script kept unchanged in `src/external/`
- Easy to update when new versions are available
- No risk of accidentally modifying external code

### ✅ Consistent API
- Follows the same patterns as other QC modules
- Integrated with the main package `__init__.py`
- Comprehensive error handling and reporting

## Migration from Original

If you were using the original PKL cleaning pipeline:

**Old way:**
```python
# Had to modify paths inside the script
import sys
sys.path.append("PKL data cleaning pipeline/")
from data_cleaning_pipeline_no_aethpy import *
```

**New way:**
```python
# Clean import with configurable path
from src.data.qc import PKLDataCleaner
cleaner = PKLDataCleaner(data_directory="your/path/")
df = cleaner.load_and_clean_data()
```

## Next Steps

1. **Test with your data**: Update the data directory path and run the examples
2. **Explore the notebook**: Use `notebooks/pkl_data_cleaning_demo.ipynb` for interactive analysis
3. **Integrate with other QC tools**: Combine PKL cleaning with other quality control modules
4. **Configure for your environment**: Set up environment variables or config files for your deployment

The PKL data cleaning pipeline is now fully integrated and ready for use! 🎉
