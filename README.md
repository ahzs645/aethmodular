# ETAD - Enhanced Aethalometer Data Analysis System

This repository contains a complete, production-ready aethalometer data analysis system with advanced analytics capabilities, quality assessment, machine learning, and statistical analysis.

## 🎯 Overview

The Enhanced Aethalometer Data Analysis (ETAD) system is a modular, scalable platform designed for comprehensive analysis of aethalometer measurements. It transforms raw black carbon absorption data into actionable insights through:

- ✅ **Robust data quality assessment** - Completeness analysis, missing data patterns, period classification
- ✅ **Advanced statistical analysis** - Distribution fitting, outlier detection, comparative statistics  
- ✅ **Machine learning capabilities** - Regression modeling, clustering, time series forecasting
- ✅ **Production-ready infrastructure** - Performance monitoring, memory optimization, parallel processing
- ✅ **Comprehensive smoothening algorithms** - ONA, CMA, DEMA implementations
- ✅ **Time series analysis** - Trend detection, seasonality analysis, stationarity testing
- ✅ **Correlation and seasonal studies** - Multi-variable analysis, Ethiopian seasonal patterns
- ✅ **Jupyter notebook interface** - Interactive analysis with visualization and reporting

## 🚀 Quick Start

### 1. Installation and Verification
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python examples/verify_installation.py
```

### 2. Quick Start Examples
```bash
# Basic usage example
python examples/basic_usage.py

# Production pipeline demo
python examples/production_pipeline_demo.py

# Comprehensive advanced analytics demo
python examples/comprehensive_advanced_analytics_demo.py
```

### 3. Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python tests/test_quality_analysis.py
python tests/test_advanced_analytics.py
```

## � Key Features

### 🔍 Data Quality Assessment
- **Completeness Analysis**: Assess data completeness by periods (daily, 9am-9am)
- **Missing Data Patterns**: Identify systematic gaps, maintenance periods, outages
- **Period Classification**: Classify data quality (Excellent, Good, Moderate, Poor)
- **Quality Metrics**: Calculate quality scores and thresholds

### 📈 Statistical Analysis
- **Distribution Fitting**: Test normal, lognormal, exponential, gamma distributions
- **Outlier Detection**: Multiple methods (IQR, Z-score, Modified Z-score, Isolation Forest)
- **Comparative Statistics**: Period comparison with t-tests, KS tests, Mann-Whitney
- **Normality Testing**: Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling tests

### 🤖 Machine Learning
- **Regression Models**: Random Forest, Linear, SVR, Gradient Boosting
- **Time Series Forecasting**: Exponential smoothing, Holt's method, linear trends
- **Clustering Analysis**: K-means, hierarchical clustering with optimal cluster detection
- **Predictive Analytics**: Model training, validation, and performance assessment

### 📉 Time Series Analysis
- **Trend Detection**: Linear and non-linear trend identification
- **Seasonality Analysis**: Decomposition and seasonal strength assessment
- **Stationarity Testing**: ADF tests and stationarity assessment
- **Pattern Recognition**: Periodic pattern detection and characterization

### 🔧 Production Infrastructure
- **Performance Monitoring**: Execution time tracking, memory usage, error handling
- **Memory Optimization**: DataFrame optimization, batch processing, caching
- **Parallel Processing**: Multi-core processing, async operations, pipeline orchestration
- **Logging System**: Structured logging with multiple handlers and levels

## 📖 Usage Examples

### Quality Analysis
```python
from src.analysis.quality.completeness_analyzer import CompletenessAnalyzer
from src.analysis.quality.missing_data_analyzer import MissingDataAnalyzer

# Analyze data completeness
completeness = CompletenessAnalyzer()
results = completeness.analyze_completeness(data, period_type='daily')

# Analyze missing data patterns
missing_analyzer = MissingDataAnalyzer()
patterns = missing_analyzer.analyze_missing_patterns(data)
```

### Advanced Analytics
```python
from src.analysis.advanced.statistical_analysis import StatisticalComparator
from src.analysis.advanced.ml_analysis import MLModelTrainer

# Compare statistical properties between periods
comparator = StatisticalComparator()
comparison = comparator.compare_periods(data1, data2)

# Train machine learning model
trainer = MLModelTrainer()
model_results = trainer.train_regression_model(
    data, target_column='BC', model_type='random_forest'
)
```

### Aethalometer Smoothening
```python
from src.analysis.aethalometer.smoothening import ONASmoothing, CMASmoothing

# Apply smoothening algorithms
ona = ONASmoothing(delta_atn_threshold=0.05)
cma = CMASmoothing(window_size=15)

# Smooth the data
smoothed_bc = ona.smooth(data['BC'])
smoothed_bc_cma = cma.smooth(data['BC'])
```

### Production Pipeline
```python
from examples.production_pipeline_demo import ETADProductionPipeline

# Initialize complete analysis pipeline
pipeline = ETADProductionPipeline()

# Run comprehensive analysis
results = pipeline.run_complete_analysis(data)

# Access results
quality_results = results['quality_assessment']
ml_results = results['machine_learning']
summary = results['executive_summary']
```

## 🧪 Testing and Validation

### Test Suite
The system includes comprehensive testing:

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific modules
python tests/test_quality_analysis.py
python tests/test_advanced_analytics.py
python tests/test_integration_performance.py
```

### Test Coverage
- **Unit Tests**: Individual module functionality
- **Integration Tests**: Cross-module workflows
- **Performance Tests**: Memory usage and execution time
- **Data Quality Tests**: Realistic data scenarios

## 🔧 Configuration

### Analysis Configuration
```python
analysis_config = {
    'quality_analysis': {
        'period_type': 'daily',
        'completeness_threshold': 0.8
    },
    'machine_learning': {
        'regression_models': ['random_forest', 'linear'],
        'clustering': True,
        'forecasting': True
    },
    'statistical_analysis': {
        'outlier_methods': ['iqr', 'zscore'],
        'distribution_tests': ['normal', 'lognormal']
    }
}
```

### Quality Thresholds
```python
# Customizable quality classification
quality_thresholds = {
    'Excellent': 10,      # ≤ 10 missing minutes per day
    'Good': 60,           # ≤ 60 missing minutes
    'Moderate': 240,      # ≤ 240 missing minutes
    'Poor': float('inf') # > 240 missing minutes
}
```

## 📊 Performance Features

### Memory Optimization
- **DataFrame Optimization**: Automatic dtype optimization
- **Batch Processing**: Handle large datasets efficiently  
- **Caching**: Intelligent result caching
- **Memory Monitoring**: Track memory usage patterns

### Parallel Processing
- **Multi-core Processing**: Utilize all available CPU cores
- **Async Operations**: Non-blocking operations for I/O
- **Pipeline Orchestration**: Coordinate complex workflows
- **Background Tasks**: Long-running analysis tasks

### Monitoring and Logging
- **Performance Metrics**: Execution time and resource usage
- **Error Handling**: Graceful error recovery and reporting
- **Structured Logging**: Comprehensive activity logging
- **Progress Tracking**: Real-time analysis progress

## 📈 Advanced Analytics Capabilities

### Statistical Methods
- **Hypothesis Testing**: t-tests, KS tests, Mann-Whitney U
- **Distribution Analysis**: Multiple distribution fitting
- **Correlation Analysis**: Pearson and Spearman correlations
- **Outlier Detection**: Multiple detection algorithms

### Machine Learning
- **Supervised Learning**: Regression and classification models
- **Unsupervised Learning**: Clustering and pattern detection
- **Time Series ML**: Forecasting and trend analysis
- **Model Validation**: Cross-validation and performance metrics

### Time Series Analysis
- **Decomposition**: Trend, seasonal, residual components
- **Stationarity**: Statistical tests and transformations
- **Forecasting**: Multiple forecasting methods
- **Pattern Detection**: Periodic and seasonal patterns

## 📁 Example Outputs

### Quality Assessment Report
```
Data Quality Assessment:
├── Completeness: 96.8%
├── Missing Periods: 3 gaps identified
├── Quality Classification:
│   ├── Excellent: 18 days (64.3%)
│   ├── Good: 7 days (25.0%)
│   ├── Moderate: 3 days (10.7%)
│   └── Poor: 0 days (0.0%)
└── Recommendations: Regular monitoring suggested
```

### Statistical Analysis Summary
```
Statistical Analysis Results:
├── Distribution Analysis:
│   ├── BC: Best fit = lognormal (AIC: 1247.3)
│   └── UV_abs: Best fit = normal (AIC: 1891.2)
├── Outlier Detection:
│   ├── BC: 23 outliers (1.2%) detected
│   └── Consensus outliers: 8 points
└── Period Comparison:
    ├── T-test: Significant difference (p<0.001)
    └── KS-test: Distributions differ (p<0.05)
```

### Machine Learning Results
```
ML Analysis Results:
├── Regression Model (Random Forest):
│   ├── Training R²: 0.943
│   ├── Test R²: 0.891
│   └── Top Features: [UV_abs, IR_abs, temperature]
├── Clustering Analysis:
│   ├── Optimal clusters: 3
│   ├── Silhouette score: 0.672
│   └── Cluster sizes: [341, 189, 127] points
└── Forecasting (24h):
    ├── Method: Holt Linear Trend
    ├── RMSE: 0.234
    └── Trend: 0.0012 increase/hour
```

## 🚀 Production Deployment

### System Requirements
- **Python**: 3.8+ 
- **Memory**: 4GB+ RAM recommended
- **Storage**: SSD recommended for large datasets
- **CPU**: Multi-core processor for parallel processing

### Dependencies
```bash
# Core dependencies
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0

# Machine learning
scikit-learn>=1.0.0

# Visualization  
matplotlib>=3.4.0
seaborn>=0.11.0

# System monitoring
psutil>=5.8.0

# Data handling
openpyxl>=3.0.0
xlrd>=2.0.0
```

### Deployment Checklist
- [ ] Install all dependencies
- [ ] Run verification script
- [ ] Configure analysis parameters
- [ ] Set up logging directories
- [ ] Test with sample data
- [ ] Run integration tests
- [ ] Configure monitoring
- [ ] Set up automated backups

## 📚 Documentation

### API Documentation
Each module includes comprehensive docstrings with:
- Parameter descriptions and types
- Return value specifications  
- Usage examples
- Error handling information

### Migration Guide
See `MIGRATION_GUIDE.md` and `ENHANCED_MIGRATION_GUIDE.md` for:
- Step-by-step migration instructions
- Code comparison examples
- Best practices and recommendations
- Troubleshooting guide

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-repo/aethmodular.git
cd aethmodular

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Code formatting
black src/ tests/ examples/
```

### Code Standards
- **Type Hints**: Use type hints for all functions
- **Docstrings**: Comprehensive documentation
- **Testing**: Unit tests for all modules
- **Error Handling**: Graceful error handling
- **Logging**: Structured logging throughout

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ethiopian Environment, Forest and Climate Change Commission
- Atmospheric measurement community
- Open source scientific Python ecosystem
- Contributors and maintainers

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check existing documentation
- Review example code
- Run verification script for troubleshooting

---

**ETAD - Enhanced Aethalometer Data Analysis System**: Production-ready, scientifically rigorous, and built for scale.

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
