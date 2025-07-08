# FTIR and Aethalometer Analysis - Modular Structure

This repository contains a complete refactoring of the FTIR and Aethalometer analysis code from a monolithic structure to a modular, maintainable architecture.

## 🎯 Overview

The original codebase had several issues:
- Large, monolithic functions doing too many things
- Mixed responsibilities (data loading, processing, analysis, visualization)
- Difficult to test and maintain
- Limited reusability of components

This new structure solves these problems by:
- ✅ Breaking down large functions into focused, single-purpose modules
- ✅ Separating concerns (data, analysis, visualization, configuration)
- ✅ Making components easily testable and reusable
- ✅ Providing clear interfaces and consistent error handling

## 📁 Project Structure

```
src/
├── config/           # Configuration settings
│   ├── settings.py   # Global settings, paths, constants
│   └── plotting.py   # Plot styling and configuration
├── core/             # Core base classes and utilities
│   ├── base.py       # Base classes for analyzers and loaders
│   └── exceptions.py # Custom exceptions
├── data/             # Data loading and processing
│   ├── loaders/      # Data loading modules
│   │   └── database.py
│   └── processors/   # Data processing modules
│       └── validation.py
├── analysis/         # Analysis modules
│   ├── correlations/ # Correlation analysis
│   │   └── pearson.py
│   ├── statistics/   # Statistical analysis
│   │   └── descriptive.py
│   └── ftir/         # FTIR-specific analysis
│       ├── fabs_ec_analyzer.py
│       └── oc_ec_analyzer.py
└── utils/            # Utility functions
    └── file_io.py

examples/             # Usage examples
├── basic_usage.py
└── before_after_comparison.py
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Basic Example
```bash
cd examples
python basic_usage.py
```

### 3. Run Comprehensive Enhanced Demo
```bash
cd examples
python comprehensive_enhanced_demo.py
```

## 📖 Usage Examples

### Aethalometer Data Smoothening

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis.aethalometer.smoothening import ONASmoothing, CMASmoothing, DEMASmoothing

# Load aethalometer data
# data should have columns like 'IR BCc', 'IR ATN1', etc.

# Apply different smoothening methods
ona = ONASmoothing(delta_atn_threshold=0.05)
cma = CMASmoothing(window_size=15)
dema = DEMASmoothing(alpha=0.2)

# Analyze with each method
ona_results = ona.analyze(data, wavelength='IR')
cma_results = cma.analyze(data, wavelength='IR') 
dema_results = dema.analyze(data, wavelength='IR')

print(f"ONA noise reduction: {ona_results['improvement_metrics']['noise_reduction_percent']:.1f}%")
print(f"CMA noise reduction: {cma_results['improvement_metrics']['noise_reduction_percent']:.1f}%")
print(f"DEMA noise reduction: {dema_results['improvement_metrics']['noise_reduction_percent']:.1f}%")
```

### 9AM-to-9AM Period Processing

```python
from analysis.aethalometer.period_processor import NineAMPeriodProcessor

# Process aethalometer data into 9AM-to-9AM periods
processor = NineAMPeriodProcessor()
period_results = processor.analyze(data, date_column='timestamp')

# Check period quality
for period in period_results['period_classifications']:
    print(f"Period {period['date_label']}: {period['quality']} "
          f"({period['data_completeness']['completeness_percentage']:.1f}% complete)")

# Get high-quality periods only
good_periods = processor.get_quality_filtered_periods(period_results, min_quality='good')
```

### Enhanced MAC Analysis

```python
from analysis.ftir.enhanced_mac_analyzer import EnhancedMACAnalyzer

# Comprehensive MAC analysis with all 4 methods
mac_analyzer = EnhancedMACAnalyzer()
mac_results = mac_analyzer.analyze(ftir_data)

# Compare all methods
for method, results in mac_results['mac_results'].items():
    if 'error' not in results:
        print(f"{method}: MAC = {results['mac_value']:.2f} m²/g")

# Get best method recommendation
best_method = mac_results['method_comparison']['recommendations']['best_overall']
print(f"Recommended method: {best_method}")
```

### Ethiopian Seasonal Analysis

```python
from analysis.seasonal.ethiopian_seasons import EthiopianSeasonAnalyzer

# Analyze seasonal patterns specific to Ethiopian climate
seasonal_analyzer = EthiopianSeasonAnalyzer()
seasonal_results = seasonal_analyzer.analyze(data, date_column='timestamp')

# Compare seasons
for season, stats in seasonal_results['seasonal_statistics'].items():
    if 'error' not in stats:
        fabs_mean = stats['statistics'].get('fabs', {}).get('mean', 0)
        print(f"{stats['name']}: Mean Fabs = {fabs_mean:.1f} Mm⁻¹")

# Get climate insights
insights = seasonal_results['climate_analytics']['climate_insights']
for insight in insights:
    print(f"• {insight}")
```

### Basic FTIR Analysis (Original)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.loaders.database import FTIRHIPSLoader
from analysis.ftir.fabs_ec_analyzer import FabsECAnalyzer

# Load data
loader = FTIRHIPSLoader("path/to/database.db")
data = loader.load("ETAD")

# Analyze Fabs-EC relationship
analyzer = FabsECAnalyzer()
results = analyzer.analyze(data)

# Display results
print(f"Valid samples: {results['sample_info']['valid_samples']}")
print(f"MAC mean: {results['mac_statistics']['mac_mean']:.2f}")
print(f"Correlation: {results['correlations']['pearson_r']:.3f}")
```

### Working with Individual Components

```python
# Use individual statistical functions
from analysis.statistics.descriptive import calculate_basic_statistics
from analysis.correlations.pearson import calculate_pearson_correlation

# Calculate statistics
mac_values = data['fabs'] / data['ec_ftir']
mac_stats = calculate_basic_statistics(mac_values, 'mac')

# Calculate correlations
correlation = calculate_pearson_correlation(data['fabs'], data['ec_ftir'])
```

## 🔧 Migration Guide

### From Old Monolithic Code

**Old way:**
```python
# Old monolithic function
def analyze_fabs_ec_relationship(df):
    # 100+ lines of mixed responsibilities
    # - Data validation
    # - Calculations
    # - Statistics
    # - Plotting
    # - Result formatting
```

**New way:**
```python
# New modular approach
analyzer = FabsECAnalyzer()
results = analyzer.analyze(df)

# Or use individual components
from data.processors.validation import validate_columns_exist
from analysis.correlations.pearson import calculate_pearson_correlation

validate_columns_exist(df, ['fabs', 'ec_ftir'])
correlation = calculate_pearson_correlation(df['fabs'], df['ec_ftir'])
```

## 🧪 Key Benefits

### 1. **Testability**
```python
# Test individual components
def test_mac_calculation():
    analyzer = FabsECAnalyzer()
    test_data = pd.DataFrame({'fabs': [10, 20], 'ec_ftir': [2, 4]})
    mac_values = analyzer.get_mac_values(test_data)
    assert mac_values.tolist() == [5.0, 5.0]
```

### 2. **Reusability**
```python
# Use the same correlation function for different analyses
from analysis.correlations.pearson import calculate_pearson_correlation

# Use for Fabs-EC analysis
fabs_ec_corr = calculate_pearson_correlation(data['fabs'], data['ec_ftir'])

# Use for OC-EC analysis
oc_ec_corr = calculate_pearson_correlation(data['oc_ftir'], data['ec_ftir'])
```

### 3. **Error Handling**
```python
# Consistent error handling across modules
from core.exceptions import DataValidationError, InsufficientDataError

try:
    results = analyzer.analyze(data)
except DataValidationError as e:
    print(f"Data validation error: {e}")
except InsufficientDataError as e:
    print(f"Insufficient data: {e}")
```

### 4. **Configuration Management**
```python
# Centralized configuration
from config.settings import QUALITY_THRESHOLDS, ETHIOPIAN_SEASONS

# Use consistent thresholds across all analyses
ec_threshold = QUALITY_THRESHOLDS['ec_ftir']
```

## 📊 Available Analyzers

### FTIR Analyzers
- **`FabsECAnalyzer`**: Analyzes Fabs-EC relationship and calculates MAC
- **`OCECAnalyzer`**: Analyzes OC-EC relationship and ratios
- **`EnhancedMACAnalyzer`**: Advanced MAC calculation with 4 different methods and physical constraint validation

### Aethalometer Analyzers
- **`ONASmoothing`**: Optimized Noise-reduction Algorithm with adaptive time-averaging
- **`CMASmoothing`**: Centered Moving Average for noise reduction
- **`DEMASmoothing`**: Double Exponentially Weighted Moving Average with minimal lag
- **`NineAMPeriodProcessor`**: 9AM-to-9AM period alignment and quality classification

### Seasonal Analysis
- **`EthiopianSeasonAnalyzer`**: Climate-specific seasonal analysis for Ethiopian conditions

### Visualization Tools
- **`TimeSeriesPlotter`**: Comprehensive time series visualization
- **`StatisticalPlotter`**: Statistical analysis plots
- **`ComparisonPlotter`**: Method comparison visualizations

### Creating New Analyzers
```python
from core.base import BaseAnalyzer
from data.processors.validation import validate_columns_exist

class MyAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("MyAnalyzer")
        self.required_columns = ['col1', 'col2']
    
    def analyze(self, data):
        validate_columns_exist(data, self.required_columns)
        # Your analysis logic here
        return results
```

## 🛠️ Development

### Adding New Modules

1. **New Data Loader**: Add to `src/data/loaders/`
2. **New Analysis Method**: Add to appropriate `src/analysis/` subdirectory
3. **New Utility**: Add to `src/utils/`

### Following the Pattern

All analyzers should:
- Inherit from `BaseAnalyzer`
- Implement the `analyze()` method
- Use validation functions from `data.processors.validation`
- Return structured results dictionaries
- Handle errors gracefully

## 📋 Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0

## 🤝 Contributing

1. Follow the existing modular structure
2. Add comprehensive docstrings
3. Include type hints
4. Write unit tests for new components
5. Update examples when adding new features

## 📝 Migration Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run basic example: `python examples/basic_usage.py`
- [ ] Compare old vs new: `python examples/before_after_comparison.py`
- [ ] Identify your key analyses to migrate
- [ ] Create new analyzers following the pattern
- [ ] Test new analyzers against old results
- [ ] Gradually replace old code with new modular components

## 📚 Documentation

- See `MIGRATION_GUIDE.md` for detailed migration instructions
- Check `Implementation/` folder for design documentation
- Review `examples/` for usage patterns

## 🔍 Troubleshooting

**Import errors**: Make sure to add the src directory to your Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
```

**Missing dependencies**: Install requirements:
```bash
pip install -r requirements.txt
```

**Database path issues**: Use absolute paths or check file existence:
```python
from pathlib import Path
db_path = Path("path/to/database.db")
assert db_path.exists(), f"Database not found: {db_path}"
```

---

This modular structure makes your code more maintainable, testable, and reusable. Start with the examples and gradually migrate your existing analyses to take advantage of the new architecture!
