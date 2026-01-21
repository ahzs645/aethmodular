# AethModular Visualization Templates

A flexible, template-based visualization system for aethalometer data analysis. This system provides consistent, reusable visualization components that can be easily customized and applied across different analysis workflows.

## 🚀 Quick Start

```python
from aethmodular.visualization.templates import VisualizationTemplateFactory
import pandas as pd

# Load your aethalometer data
data = pd.read_pickle('your_aethalometer_data.pkl')

# Create a time series plot
template = VisualizationTemplateFactory.create_template('time_series')
fig = template.create_plot(
    data=data,
    columns=['IR BCc', 'Blue BCc'],
    title='BC Concentrations Over Time'
)

# Save the plot
template.save_plot(fig, 'bc_timeseries.png')
```

## 📊 Available Templates

### Time Series Templates
- **`time_series`**: Basic time series plots with automatic BC column detection
- **`smoothening_comparison`**: Compare original vs smoothed data using different methods
- **`diurnal_patterns`**: 24-hour pattern analysis with missing data visualization

### Heatmap Templates  
- **`weekly_heatmap`**: Day-of-week × hour pattern visualization
- **`seasonal_heatmap`**: Month × year pattern analysis

### Scientific Analysis Templates
- **`mac_analysis`**: Mass Absorption Cross-section analysis with 4 methods
- **`correlation_analysis`**: Correlation matrix visualization
- **`scatter_plot`**: Scatter plots with optional regression lines

## 🎨 Configuration System

### Pre-defined Styles
```python
# Use different pre-configured styles
configs = {
    'default': {'figsize': [12, 8], 'style': 'whitegrid'},
    'publication': {'figsize': [10, 6], 'style': 'white', 'dpi': 600},
    'presentation': {'figsize': [16, 10], 'font_size': 16},
    'notebook': {'figsize': [12, 6], 'dpi': 150}
}

# Create template with specific style
template = VisualizationTemplateFactory.create_template('time_series', configs['publication'])
```

### Custom Configuration
```python
custom_config = {
    'figsize': (15, 8),
    'style': 'darkgrid',
    'color_palette': 'Set2',
    'font_size': 14,
    'dpi': 300,
    'save_format': 'pdf'
}

template = VisualizationTemplateFactory.create_template('time_series', custom_config)
```

## 📋 Detailed Usage Examples

### 1. Time Series Visualization

```python
# Basic time series
template = VisualizationTemplateFactory.create_template('time_series')
fig = template.create_plot(
    data=df,  # DataFrame with datetime index
    columns=['IR BCc', 'Blue BCc', 'Green BCc'],
    title='Multi-wavelength BC Analysis',
    ylabel='BC Concentration (μg/m³)'
)
```

### 2. Smoothening Comparison

```python
# Compare different smoothening methods
template = VisualizationTemplateFactory.create_template('smoothening_comparison')
fig = template.create_plot(
    original_data=raw_data,
    smoothed_results={
        'ONA': ona_smoothed,
        'CMA': cma_smoothed,
        'DEMA': dema_smoothed
    },
    timestamps=timestamps,
    title='Smoothening Methods Comparison'
)
```

### 3. Diurnal Pattern Analysis

```python
# Analyze 24-hour patterns with missing data
template = VisualizationTemplateFactory.create_template('diurnal_patterns')
fig = template.create_plot(
    data=df,
    date_column='timestamp',
    value_columns=['IR BCc', 'Blue BCc'],
    missing_data_analysis=True
)
```

### 4. Weekly Pattern Heatmap

```python
# Create day-of-week × hour heatmap
template = VisualizationTemplateFactory.create_template('weekly_heatmap')
fig = template.create_plot(
    data=df,
    date_column='timestamp',
    value_column='IR BCc',
    missing_data=False,  # Show actual values
    title='Weekly BC Pattern'
)
```

### 5. Seasonal Analysis

```python
# Long-term seasonal patterns
template = VisualizationTemplateFactory.create_template('seasonal_heatmap')
fig = template.create_plot(
    data=df,
    date_column='timestamp',
    value_column='IR BCc',
    missing_data=True,  # Show data coverage
    title='Seasonal Data Coverage'
)
```

### 6. MAC Analysis

```python
# Mass Absorption Cross-section analysis
template = VisualizationTemplateFactory.create_template('mac_analysis')
fig = template.create_plot(
    fabs_data=absorption_data,
    ec_data=elemental_carbon_data,
    title='MAC Analysis: Multiple Methods'
)
```

### 7. Correlation Analysis

```python
# Multi-variable correlation matrix
template = VisualizationTemplateFactory.create_template('correlation_analysis')
fig = template.create_plot(
    data=df,
    columns=['IR BCc', 'Blue BCc', 'PM2.5', 'NO2'],
    title='Air Quality Correlations',
    method='pearson'
)
```

## 🛠️ Advanced Features

### Custom Template Registration

```python
from aethmodular.visualization.templates import BaseVisualizationTemplate

class CustomTemplate(BaseVisualizationTemplate):
    def validate_parameters(self, **kwargs):
        # Custom validation
        return True
    
    def create_plot(self, **kwargs):
        # Custom plotting logic
        fig, ax = plt.subplots(figsize=self.config['figsize'])
        # ... your plotting code ...
        return fig

# Register your custom template
VisualizationTemplateFactory.register_template('my_custom_plot', CustomTemplate)

# Use it like any other template
template = VisualizationTemplateFactory.create_template('my_custom_plot')
```

### Convenience Functions

```python
from aethmodular.visualization.templates import create_plot

# Quick plotting without explicit template creation
fig = create_plot(
    'scatter_plot',
    data=df,
    x_column='IR BCc',
    y_column='Blue BCc',
    add_regression=True,
    title='IR vs Blue BC'
)
```

### Template Discovery

```python
# List all available templates
templates = VisualizationTemplateFactory.list_templates()
print(f"Available: {templates}")

# Get templates by category
categories = VisualizationTemplateFactory.list_templates_by_category()
for category, template_list in categories.items():
    print(f"{category}: {template_list}")

# Get detailed information about a template
info = VisualizationTemplateFactory.get_template_info('time_series')
print(f"Required parameters: {info['required_params']}")
print(f"Optional parameters: {info['optional_params']}")
```

## 🎯 Template Parameters

### Time Series Template
**Required**: `data` (DataFrame with datetime index)  
**Optional**: `columns`, `title`, `ylabel`, `xlabel`

### Smoothening Comparison Template
**Required**: `original_data`, `smoothed_results`, `timestamps`  
**Optional**: `title`

### Diurnal Patterns Template
**Required**: `data`, `date_column`, `value_columns`  
**Optional**: `missing_data_analysis`

### Heatmap Templates
**Required**: `data`, `date_column`, `value_column`  
**Optional**: `missing_data`, `title`

### MAC Analysis Template
**Required**: `fabs_data`, `ec_data`  
**Optional**: `title`

### Correlation Analysis Template
**Required**: `data`  
**Optional**: `columns`, `title`, `method`, `mask_upper`

### Scatter Plot Template
**Required**: `data`, `x_column`, `y_column`  
**Optional**: `color_column`, `title`, `add_regression`

## 🎨 Color Schemes

Built-in color schemes for different analysis types:

```python
# Access color schemes
from aethmodular.visualization.templates.config_utils import config_manager

# BC analysis colors
bc_colors = config_manager.get_color_scheme('bc_analysis')

# Seasonal colors
seasonal_colors = config_manager.get_color_scheme('seasonal')

# Quality assessment colors
quality_colors = config_manager.get_color_scheme('quality')

# Spectral wavelength colors
spectral_colors = config_manager.get_color_scheme('spectral')
```

## 📁 Directory Structure

```
src/visualization/templates/
├── __init__.py                    # Package initialization
├── base_template.py              # Abstract base class
├── time_series_templates.py      # Time series specific templates
├── heatmap_templates.py          # Heatmap templates
├── scientific_templates.py       # Scientific analysis templates
├── factory.py                    # Template factory and registry
├── config_utils.py              # Configuration management
├── examples.py                   # Usage examples
├── config/
│   ├── plot_styles.json         # Pre-defined plot styles
│   ├── color_schemes.json       # Color scheme definitions
│   └── default_parameters.json  # Default parameters per template
└── README.md                     # This documentation
```

## 🔧 Installation and Dependencies

The templates system requires:
- `matplotlib` >= 3.5.0
- `pandas` >= 1.3.0  
- `numpy` >= 1.20.0
- `seaborn` >= 0.11.0
- `scikit-learn` >= 1.0.0 (optional, for advanced MAC analysis)

## 🤝 Contributing

To add new templates:

1. Create a new template class inheriting from `BaseVisualizationTemplate`
2. Implement required methods: `validate_parameters()` and `create_plot()`
3. Register your template with the factory
4. Add examples and documentation
5. Update configuration files if needed

## 📄 License

This visualization template system is part of the AethModular project and follows the same licensing terms.

## 🆘 Support

For issues, questions, or feature requests:
1. Check the examples in `examples.py`
2. Review template parameter requirements
3. Verify your data format matches template expectations
4. Submit issues with minimal reproducible examples

---

**Quick Reference**: Use `VisualizationTemplateFactory.list_templates()` to see all available templates and `create_plot()` for quick visualizations.
