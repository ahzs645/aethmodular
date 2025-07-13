# Aethdata-Analysis Visualization Types for AethModular Templates

## Overview
This document extracts all visualization types from the aethdata-analysis repository to create templatable components for aethmodular. The visualizations are organized by category with implementation details and template structures.

## 📊 Core Visualization Categories

### 1. Time Series Visualizations

#### 1.1 Basic Time Series Plots
**Source**: `src/utils/plotting.py` - `AethalometerPlotter.plot_time_series()`
**Purpose**: Plot multiple BC concentration time series with automatic styling

```python
# Template Structure
{
    "type": "time_series",
    "parameters": {
        "data": "pd.DataFrame with datetime index",
        "columns": "List of column names to plot",
        "title": "Plot title",
        "save_path": "Optional save location"
    },
    "features": [
        "Auto-detection of BC columns",
        "Color cycling for multiple series",
        "NaN handling",
        "Date formatting"
    ]
}
```

#### 1.2 Smoothening Comparison Plots
**Source**: `src/visualization/time_series.py` - `TimeSeriesPlotter.plot_smoothening_comparison()`
**Purpose**: Compare original vs smoothed data using different methods (ONA, CMA, DEMA)

```python
# Template Structure
{
    "type": "smoothening_comparison", 
    "parameters": {
        "original_data": "np.ndarray",
        "smoothed_results": "Dict[method_name, smoothed_data]",
        "timestamps": "pd.Series",
        "title": "Comparison title"
    },
    "layout": "Vertical subplots (original + each method)",
    "methods": ["ONA", "CMA", "DEMA"]
}
```

#### 1.3 Diurnal Pattern Analysis
**Source**: `src/visualization/time_series.py` - `TimeSeriesPlotter.plot_diurnal_patterns()`
**Purpose**: Show 24-hour patterns with missing data analysis

```python
# Template Structure
{
    "type": "diurnal_patterns",
    "parameters": {
        "data": "pd.DataFrame",
        "date_column": "Datetime column name", 
        "value_columns": "List of columns to analyze",
        "missing_data_analysis": "Boolean flag"
    },
    "outputs": [
        "Hourly mean patterns",
        "Missing data by hour",
        "Statistical error bars"
    ]
}
```

### 2. Heatmap Visualizations

#### 2.1 Weekly Pattern Heatmaps
**Source**: `src/visualization/time_series.py` - `TimeSeriesPlotter.plot_weekly_heatmap()`
**Purpose**: Day-of-week × hour patterns visualization

```python
# Template Structure
{
    "type": "weekly_heatmap",
    "parameters": {
        "data": "pd.DataFrame",
        "date_column": "Datetime column",
        "value_column": "Value to analyze",
        "missing_data": "Boolean - show missing vs actual values"
    },
    "dimensions": "7 days × 24 hours",
    "color_maps": {
        "missing_data": "Reds",
        "actual_values": "viridis"
    }
}
```

#### 2.2 Seasonal Heatmaps
**Source**: `src/visualization/time_series.py` - `TimeSeriesPlotter.plot_seasonal_heatmap()`
**Purpose**: Month × year patterns with data coverage

```python
# Template Structure
{
    "type": "seasonal_heatmap",
    "parameters": {
        "data": "pd.DataFrame",
        "date_column": "Datetime column",
        "value_column": "Value to analyze", 
        "missing_data": "Boolean flag"
    },
    "dimensions": "12 months × N years",
    "features": [
        "Missing data percentage calculation",
        "Automatic year detection",
        "Month name formatting"
    ]
}
```

### 3. Scientific Analysis Plots

#### 3.1 MAC Analysis Visualization
**Source**: `src/analysis/ftir/enhanced_mac_analyzer.py` + plotting functions
**Purpose**: Mass Absorption Cross-section analysis with 4 methods

```python
# Template Structure
{
    "type": "mac_analysis",
    "methods": {
        "method_1": "Individual MAC mean",
        "method_2": "Ratio of means", 
        "method_3": "Linear regression with intercept",
        "method_4": "Linear regression through origin"
    },
    "plot_types": [
        "Scatter plots (EC vs Fabs)",
        "Regression lines",
        "Method comparison charts",
        "Uncertainty visualization"
    ],
    "parameters": {
        "fabs_data": "Absorption data",
        "ec_data": "Element carbon data",
        "confidence_intervals": "Boolean"
    }
}
```

#### 3.2 Correlation Analysis Plots
**Source**: Multiple files - correlation functions
**Purpose**: Multi-variable correlation visualization

```python
# Template Structure
{
    "type": "correlation_analysis",
    "plot_types": [
        "Scatter plots with color coding",
        "Correlation matrices",
        "Feature importance plots"
    ],
    "parameters": {
        "data": "pd.DataFrame",
        "target_variable": "Primary variable",
        "feature_variables": "List of features",
        "color_variable": "Optional color coding variable"
    }
}
```

#### 3.3 Machine Learning Visualization
**Source**: `Only Python/Elemental Comparison ECFTIR.py`
**Purpose**: ML model performance and feature analysis

```python
# Template Structure
{
    "type": "ml_analysis",
    "plot_types": [
        "Feature importance plots",
        "Model performance metrics",
        "Prediction vs actual plots",
        "Residual analysis plots"
    ],
    "models": [
        "Random Forest",
        "Linear Regression", 
        "Ridge Regression",
        "Support Vector Regression"
    ],
    "parameters": {
        "X": "Feature matrix",
        "y": "Target variable",
        "model_results": "Trained model results"
    }
}
```

### 4. Quality Assessment Plots

#### 4.1 Period Quality Overview
**Source**: `src/visualization/time_series.py` - `TimeSeriesPlotter.plot_period_quality_overview()`
**Purpose**: 9AM-to-9AM period quality classification

```python
# Template Structure
{
    "type": "period_quality",
    "parameters": {
        "period_results": "Dict from NineAMPeriodProcessor",
        "quality_categories": ["Excellent", "Good", "Moderate", "Poor"]
    },
    "plot_types": [
        "Quality timeline",
        "Quality distribution pie chart",
        "Data coverage bar chart"
    ]
}
```

#### 4.2 Data Completeness Analysis
**Source**: `Only Python/data completeness.py`
**Purpose**: Missing data patterns and completeness assessment

```python
# Template Structure
{
    "type": "data_completeness",
    "plot_types": [
        "Missing data timeline",
        "Hourly missing patterns",
        "Daily missing summaries",
        "Monthly/yearly heatmaps"
    ],
    "parameters": {
        "data": "pd.DataFrame",
        "datetime_column": "Timestamp column",
        "analysis_columns": "Columns to analyze"
    }
}
```

### 5. Statistical Visualizations

#### 5.1 Distribution Analysis
**Source**: Various statistical analysis files
**Purpose**: Distribution fitting and comparison

```python
# Template Structure
{
    "type": "distribution_analysis",
    "plot_types": [
        "Histograms with fitted distributions",
        "Q-Q plots",
        "Box plots",
        "Violin plots"
    ],
    "parameters": {
        "data": "Data arrays",
        "distributions": "List of distributions to fit",
        "comparison_groups": "Optional grouping variable"
    }
}
```

#### 5.2 Seasonal Analysis Plots
**Source**: `src/analysis/seasonal/ethiopian_seasons.py`
**Purpose**: Ethiopian seasonal pattern analysis

```python
# Template Structure
{
    "type": "seasonal_analysis",
    "plot_types": [
        "Seasonal box plots",
        "Monthly trend lines",
        "Seasonal comparison charts",
        "Climate pattern heatmaps"
    ],
    "seasons": ["Bega", "Belg", "Kiremt"],
    "parameters": {
        "data": "Seasonal analysis results",
        "climate_variables": "List of variables to analyze"
    }
}
```

## 🔧 Implementation Templates for AethModular

### Base Visualization Class Template

```python
class BaseVisualizationTemplate:
    """Base template for all visualization types"""
    
    def __init__(self, viz_type: str, config: dict):
        self.viz_type = viz_type
        self.config = config
        self.setup_styling()
    
    def setup_styling(self):
        """Configure matplotlib/seaborn styling"""
        pass
    
    def validate_parameters(self, **kwargs):
        """Validate required parameters"""
        pass
    
    def create_plot(self, **kwargs):
        """Main plotting method - to be overridden"""
        raise NotImplementedError
    
    def save_plot(self, fig, path: str):
        """Save plot with consistent formatting"""
        pass
```

### Specific Template Examples

#### Time Series Template
```python
class TimeSeriesTemplate(BaseVisualizationTemplate):
    """Template for time series visualizations"""
    
    REQUIRED_PARAMS = ['data', 'columns']
    OPTIONAL_PARAMS = ['title', 'save_path', 'figsize']
    
    def create_plot(self, **kwargs):
        # Implementation based on extracted code
        pass
```

#### Heatmap Template
```python
class HeatmapTemplate(BaseVisualizationTemplate):
    """Template for heatmap visualizations"""
    
    REQUIRED_PARAMS = ['data', 'x_column', 'y_column', 'value_column']
    OPTIONAL_PARAMS = ['cmap', 'title', 'missing_data_mode']
    
    def create_plot(self, **kwargs):
        # Implementation based on extracted code
        pass
```

#### Scientific Analysis Template
```python
class ScientificAnalysisTemplate(BaseVisualizationTemplate):
    """Template for scientific analysis plots"""
    
    PLOT_TYPES = ['regression', 'correlation', 'mac_analysis', 'ml_results']
    
    def create_plot(self, plot_type: str, **kwargs):
        # Route to specific analysis plot type
        pass
```

## 📁 Template Organization Structure

```
aethmodular/
├── templates/
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── base_template.py
│   │   ├── time_series/
│   │   │   ├── basic_time_series.py
│   │   │   ├── smoothening_comparison.py
│   │   │   └── diurnal_patterns.py
│   │   ├── heatmaps/
│   │   │   ├── weekly_heatmap.py
│   │   │   └── seasonal_heatmap.py
│   │   ├── scientific/
│   │   │   ├── mac_analysis.py
│   │   │   ├── correlation_plots.py
│   │   │   └── ml_visualization.py
│   │   ├── quality/
│   │   │   ├── period_quality.py
│   │   │   └── data_completeness.py
│   │   └── statistical/
│   │       ├── distribution_analysis.py
│   │       └── seasonal_analysis.py
│   └── config/
│       ├── plot_styles.json
│       ├── color_schemes.json
│       └── default_parameters.json
```

## 🎯 Usage Examples

### Basic Usage
```python
from aethmodular.templates.visualization import TimeSeriesTemplate

# Create time series plot
template = TimeSeriesTemplate('basic_time_series', config)
fig = template.create_plot(
    data=aethalometer_data,
    columns=['IR BCc', 'Blue BCc'],
    title='BC Concentrations Over Time'
)
```

### Advanced Configuration
```python
from aethmodular.templates.visualization import HeatmapTemplate

# Create weekly pattern heatmap
template = HeatmapTemplate('weekly_pattern', config)
fig = template.create_plot(
    data=processed_data,
    date_column='timestamp',
    value_column='IR BCc',
    missing_data=True,
    cmap='Reds'
)
```

## 🔄 Migration Strategy

1. **Extract Core Functions**: Copy visualization functions from aethdata-analysis
2. **Create Template Classes**: Wrap functions in template classes
3. **Standardize Interfaces**: Ensure consistent parameter naming
4. **Add Configuration**: Make plots customizable through config files
5. **Test Integration**: Verify templates work with aethmodular data structures
6. **Documentation**: Create usage examples for each template type

## 📋 Implementation Checklist

- [ ] Extract all visualization functions from aethdata-analysis
- [ ] Create base template class structure
- [ ] Implement specific template classes for each category
- [ ] Create configuration system for styling and parameters
- [ ] Add validation and error handling
- [ ] Write comprehensive tests for each template
- [ ] Create usage documentation and examples
- [ ] Set up template discovery and registration system

This extraction provides a complete foundation for creating templatable visualizations in aethmodular while preserving all the analytical capabilities from aethdata-analysis.