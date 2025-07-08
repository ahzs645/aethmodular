# Migration Guide: From Monolithic to Modular Structure

## Overview
This guide helps you migrate from the existing monolithic code structure to the new modular system.

## Installation and Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Optional: Install as Package
```bash
pip install -e .
```

## Migration Steps

### Step 1: Basic Usage (Start Here)
Replace your current analysis workflow with the new modular system:

**Old way (from your existing code):**
```python
# Old monolithic approach
from data_loader import load_ftir_hips_data
from ftir_analysis import analyze_fabs_ec_relationship

# Load data
data = load_ftir_hips_data("database.db", "ETAD")

# Run analysis (large, complex function)
results = analyze_fabs_ec_relationship(data)
```

**New way (modular approach):**
```python
# New modular approach
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.loaders.database import FTIRHIPSLoader
from analysis.ftir.fabs_ec_analyzer import FabsECAnalyzer

# Load data
loader = FTIRHIPSLoader("database.db")
data = loader.load("ETAD")

# Run analysis (focused, testable functions)
analyzer = FabsECAnalyzer()
results = analyzer.analyze(data)
```

### Step 2: Understanding the New Structure

#### Core Components:
- **`src/config/`**: Configuration settings, thresholds, plotting styles
- **`src/core/`**: Base classes and common interfaces
- **`src/data/`**: Data loading and processing
- **`src/analysis/`**: Analysis modules organized by domain
- **`src/utils/`**: Utility functions

#### Key Benefits:
1. **Focused functions**: Each function has a single responsibility
2. **Easy testing**: Test individual components in isolation
3. **Reusable components**: Use the same statistics/correlation functions across analyses
4. **Better error handling**: Specific exceptions for different error types

### Step 3: Running Your First Analysis

Try the basic example:
```bash
cd examples
python basic_usage.py
```

This will:
1. Load data from your database
2. Run Fabs-EC analysis
3. Display results
4. Save results to JSON

### Step 4: Migrating Your Existing Analyses

#### A. Break Down Large Functions
**Example: Your existing `analyze_fabs_ec_relationship` function**

Old (monolithic):
```python
def analyze_fabs_ec_relationship(df):
    # 100+ lines doing:
    # - Data validation
    # - Correlation calculation
    # - Regression analysis
    # - MAC calculation
    # - Statistics generation
    # - Results formatting
```

New (modular):
```python
# Use the new FabsECAnalyzer class
analyzer = FabsECAnalyzer()
results = analyzer.analyze(df)

# Or use individual components:
from src.data.processors.validation import validate_columns_exist
from src.analysis.correlations.pearson import calculate_pearson_correlation
from src.analysis.statistics.descriptive import calculate_basic_statistics

validate_columns_exist(df, ['fabs', 'ec_ftir'])
correlations = calculate_pearson_correlation(df['fabs'], df['ec_ftir'])
stats = calculate_basic_statistics(df['fabs'] / df['ec_ftir'], 'mac')
```

#### B. Migration Mapping

| Old Function | New Module | Notes |
|-------------|------------|-------|
| `load_ftir_hips_data()` | `src/data/loaders/database.py` | Now a class with better error handling |
| `calculate_mac_methods()` | `src/analysis/ftir/fabs_ec_analyzer.py` | Integrated into analyzer |
| `analyze_oc_ec_relationship()` | Create new analyzer following same pattern | |
| Correlation calculations | `src/analysis/correlations/pearson.py` | Separate, reusable module |
| Statistical calculations | `src/analysis/statistics/descriptive.py` | Separate, reusable module |

### Step 5: Creating New Analyzers

Follow the `FabsECAnalyzer` pattern to create new analysis modules:

```python
# src/analysis/ftir/oc_ec_analyzer.py
from ...core.base import BaseAnalyzer
from ...data.processors.validation import validate_columns_exist, get_valid_data_mask
from ...analysis.correlations.pearson import calculate_pearson_correlation

class OCECAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("OCECAnalyzer")
        self.required_columns = ['oc_ftir', 'ec_ftir']
    
    def analyze(self, data):
        # Validation
        validate_columns_exist(data, self.required_columns)
        valid_mask = get_valid_data_mask(data, self.required_columns)
        
        # Analysis
        clean_data = data[valid_mask]
        correlations = calculate_pearson_correlation(
            clean_data['oc_ftir'], 
            clean_data['ec_ftir']
        )
        
        return {
            'sample_info': {'valid_samples': len(clean_data)},
            'correlations': correlations
        }
```

### Step 6: Advanced Usage

#### A. Custom Configuration
Modify `src/config/settings.py` for your specific needs:
```python
# Add your custom thresholds
QUALITY_THRESHOLDS = {
    'your_parameter': {'min_value': 0.0, 'max_value': 100.0},
}
```

#### B. Custom Loaders
Create specialized loaders for different data sources:
```python
# src/data/loaders/csv_loader.py
from ...core.base import BaseLoader

class CSVLoader(BaseLoader):
    def load(self, file_path):
        # Your CSV loading logic
        pass
```

#### C. Custom Analyzers
Create domain-specific analyzers:
```python
# src/analysis/seasonal/seasonal_analyzer.py
from ...core.base import BaseAnalyzer

class SeasonalAnalyzer(BaseAnalyzer):
    def analyze(self, data):
        # Your seasonal analysis logic
        pass
```

## Benefits You'll See Immediately

1. **Easier Debugging**: Problems are isolated to small, focused functions
2. **Faster Development**: Reuse existing components instead of rewriting
3. **Better Testing**: Test individual components in isolation
4. **Cleaner Code**: Single responsibility principle applied throughout
5. **Better Documentation**: Each module has clear purpose and documentation

## Common Issues and Solutions

### Issue 1: Import Errors
**Problem**: `ModuleNotFoundError` when importing from `src/`
**Solution**: Add the src directory to your Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
```

### Issue 2: Missing Dependencies
**Problem**: `ImportError` for pandas, numpy, etc.
**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

### Issue 3: Database Path Issues
**Problem**: Database not found errors
**Solution**: Update paths in your scripts or use absolute paths:
```python
from pathlib import Path
db_path = Path.home() / "path/to/your/database.db"
```

## Next Steps

1. **Start with one analysis**: Migrate your most important analysis first
2. **Test thoroughly**: Compare results with your old code
3. **Gradually expand**: Add more analyses as you become comfortable
4. **Customize as needed**: Modify the structure to fit your specific needs

## Support

If you encounter issues:
1. Check the example files in `examples/`
2. Review the implementation guide in `Implementation/`
3. Look at the modular structure documentation in `Implementation/modular_structure.md`

Remember: This is a gradual migration. Keep your old code working while you transition to the new system!
