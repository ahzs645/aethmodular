"""
Usage Examples for AethModular Visualization Templates

This module provides comprehensive examples of how to use the 
visualization template system with different types of data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from .factory import VisualizationTemplateFactory, create_plot

def create_sample_data(n_days: int = 30) -> pd.DataFrame:
    """Create sample aethalometer data for examples"""
    # Create datetime index
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_days * 24, freq='H')
    
    # Generate realistic BC data with diurnal patterns
    hours = np.array([d.hour for d in dates])
    
    # Base concentrations with diurnal variation
    base_ir = 2.0 + 1.5 * np.sin((hours - 6) * np.pi / 12) ** 2
    base_blue = 1.8 + 1.2 * np.sin((hours - 6) * np.pi / 12) ** 2
    
    # Add random noise and weekly patterns
    days_of_week = np.array([d.dayofweek for d in dates])
    weekend_factor = 1 + 0.3 * np.isin(days_of_week, [5, 6])  # Higher on weekends
    
    ir_bc = base_ir * weekend_factor + np.random.normal(0, 0.3, len(dates))
    blue_bc = base_blue * weekend_factor + np.random.normal(0, 0.25, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'IR BCc': np.maximum(0, ir_bc),  # Ensure non-negative
        'Blue BCc': np.maximum(0, blue_bc),
        'fabs_370': np.maximum(0, ir_bc * 8.5 + np.random.normal(0, 1, len(dates))),
        'ec_ftir': np.maximum(0, ir_bc * 0.8 + np.random.normal(0, 0.2, len(dates)))
    })
    
    # Add some missing data
    missing_indices = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'IR BCc'] = np.nan
    
    return df

def example_time_series():
    """Example: Basic time series plot"""
    print("Example 1: Basic Time Series Plot")
    print("-" * 40)
    
    # Create sample data
    data = create_sample_data(7)  # 7 days of data
    data.set_index('timestamp', inplace=True)
    
    # Create time series plot
    template = VisualizationTemplateFactory.create_template('time_series')
    fig = template.create_plot(
        data=data,
        columns=['IR BCc', 'Blue BCc'],
        title='BC Concentrations Over Time - 1 Week',
        ylabel='BC Concentration (μg/m³)'
    )
    
    print("✓ Time series plot created successfully")
    return fig

def example_smoothening_comparison():
    """Example: Smoothening methods comparison"""
    print("\nExample 2: Smoothening Comparison")
    print("-" * 40)
    
    # Create sample data
    data = create_sample_data(3)  # 3 days
    original = data['IR BCc'].values
    timestamps = data['timestamp']
    
    # Create example smoothed data (simulated)
    def moving_average(arr, window):
        return np.convolve(arr, np.ones(window)/window, mode='same')
    
    smoothed_results = {
        'Moving Average (3h)': moving_average(original, 3),
        'Moving Average (6h)': moving_average(original, 6),
        'Moving Average (12h)': moving_average(original, 12)
    }
    
    # Create comparison plot
    template = VisualizationTemplateFactory.create_template('smoothening_comparison')
    fig = template.create_plot(
        original_data=original,
        smoothed_results=smoothed_results,
        timestamps=timestamps,
        title='Smoothening Methods Comparison - IR BC'
    )
    
    print("✓ Smoothening comparison plot created successfully")
    return fig

def example_diurnal_patterns():
    """Example: Diurnal pattern analysis"""
    print("\nExample 3: Diurnal Pattern Analysis")
    print("-" * 40)
    
    # Create sample data with more pronounced diurnal patterns
    data = create_sample_data(30)  # 30 days for better statistics
    
    # Create diurnal pattern plot
    template = VisualizationTemplateFactory.create_template('diurnal_patterns')
    fig = template.create_plot(
        data=data,
        date_column='timestamp',
        value_columns=['IR BCc', 'Blue BCc'],
        missing_data_analysis=True
    )
    
    print("✓ Diurnal pattern plot created successfully")
    return fig

def example_weekly_heatmap():
    """Example: Weekly pattern heatmap"""
    print("\nExample 4: Weekly Pattern Heatmap")
    print("-" * 40)
    
    # Create sample data
    data = create_sample_data(21)  # 3 weeks
    
    # Create weekly heatmap
    template = VisualizationTemplateFactory.create_template('weekly_heatmap')
    fig = template.create_plot(
        data=data,
        date_column='timestamp',
        value_column='IR BCc',
        missing_data=False,  # Show actual values
        title='Weekly Pattern: IR BC Concentrations'
    )
    
    print("✓ Weekly heatmap created successfully")
    return fig

def example_seasonal_heatmap():
    """Example: Seasonal pattern heatmap"""
    print("\nExample 5: Seasonal Pattern Heatmap")
    print("-" * 40)
    
    # Create longer-term sample data
    data = create_sample_data(365)  # 1 year
    
    # Create seasonal heatmap
    template = VisualizationTemplateFactory.create_template('seasonal_heatmap')
    fig = template.create_plot(
        data=data,
        date_column='timestamp',
        value_column='IR BCc',
        missing_data=False,
        title='Seasonal Pattern: IR BC Concentrations (2023)'
    )
    
    print("✓ Seasonal heatmap created successfully")
    return fig

def example_mac_analysis():
    """Example: MAC analysis"""
    print("\nExample 6: MAC Analysis")
    print("-" * 40)
    
    # Create sample data
    data = create_sample_data(30)
    
    # Remove NaN values for MAC analysis
    clean_data = data.dropna()
    
    # Create MAC analysis plot
    template = VisualizationTemplateFactory.create_template('mac_analysis')
    fig = template.create_plot(
        fabs_data=clean_data['fabs_370'].values,
        ec_data=clean_data['ec_ftir'].values,
        title='MAC Analysis: 370nm Absorption vs EC'
    )
    
    print("✓ MAC analysis plot created successfully")
    return fig

def example_correlation_analysis():
    """Example: Correlation analysis"""
    print("\nExample 7: Correlation Analysis")
    print("-" * 40)
    
    # Create sample data
    data = create_sample_data(30)
    
    # Select numeric columns for correlation
    numeric_columns = ['IR BCc', 'Blue BCc', 'fabs_370', 'ec_ftir']
    
    # Create correlation plot
    template = VisualizationTemplateFactory.create_template('correlation_analysis')
    fig = template.create_plot(
        data=data,
        columns=numeric_columns,
        title='Correlation Matrix: BC and Related Variables'
    )
    
    print("✓ Correlation analysis plot created successfully")
    return fig

def example_custom_config():
    """Example: Using custom configuration"""
    print("\nExample 8: Custom Configuration")
    print("-" * 40)
    
    # Create sample data
    data = create_sample_data(7)
    data.set_index('timestamp', inplace=True)
    
    # Custom configuration for publication-ready plots
    custom_config = {
        'figsize': (10, 6),
        'style': 'white',
        'font_size': 14,
        'dpi': 600,
        'save_format': 'pdf'
    }
    
    # Create template with custom config
    template = VisualizationTemplateFactory.create_template('time_series', custom_config)
    fig = template.create_plot(
        data=data,
        columns=['IR BCc'],
        title='BC Concentrations - Publication Style',
        ylabel='BC (μg/m³)'
    )
    
    print("✓ Custom configuration plot created successfully")
    return fig

def example_convenience_function():
    """Example: Using convenience function"""
    print("\nExample 9: Convenience Function")
    print("-" * 40)
    
    # Create sample data
    data = create_sample_data(5)
    
    # Use convenience function for quick plotting
    fig = create_plot(
        'scatter_plot',
        data=data,
        x_column='IR BCc',
        y_column='Blue BCc',
        title='IR BC vs Blue BC',
        add_regression=True
    )
    
    print("✓ Convenience function plot created successfully")
    return fig

def run_all_examples():
    """Run all examples"""
    print("AethModular Visualization Templates - Examples")
    print("=" * 50)
    
    examples = [
        example_time_series,
        example_smoothening_comparison,
        example_diurnal_patterns,
        example_weekly_heatmap,
        example_seasonal_heatmap,
        example_mac_analysis,
        example_correlation_analysis,
        example_custom_config,
        example_convenience_function
    ]
    
    figures = []
    for example_func in examples:
        try:
            fig = example_func()
            figures.append(fig)
        except Exception as e:
            print(f"❌ Error in {example_func.__name__}: {str(e)}")
    
    print(f"\n✅ Successfully created {len(figures)} example plots")
    return figures

def demo_template_info():
    """Demonstrate template discovery and information"""
    print("\nTemplate System Information")
    print("=" * 30)
    
    # List all available templates
    templates = VisualizationTemplateFactory.list_templates()
    print(f"Available templates: {templates}")
    
    # Show templates by category
    categories = VisualizationTemplateFactory.list_templates_by_category()
    print(f"\nTemplates by category:")
    for category, template_list in categories.items():
        print(f"  {category}: {template_list}")
    
    # Show detailed info for one template
    print(f"\nDetailed info for 'time_series' template:")
    info = VisualizationTemplateFactory.get_template_info('time_series')
    for key, value in info.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    demo_template_info()
    figures = run_all_examples()
