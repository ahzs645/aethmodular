"""
Usage Examples for AethModular Visualization Templates

This module provides comprehensive examples of how to use the 
visualization template system with different types of data.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
from .factory import VisualizationTemplateFactory, create_plot

def create_sample_data(n_days: int = 30) -> pd.DataFrame:
    """
    PLACEHOLDER: This function was removed to avoid sample data generation.
    Replace with your actual data loading function.
    
    Expected DataFrame structure:
    - timestamp: DateTime column
    - IR BCc: IR wavelength BC concentrations
    - Blue BCc: Blue wavelength BC concentrations  
    - fabs_370: Absorption coefficients (optional)
    - ec_ftir: FTIR EC measurements (optional)
    """
    raise NotImplementedError(
        "This function requires actual data. Please load your aethalometer data "
        "with columns: ['timestamp', 'IR BCc', 'Blue BCc', ...]"
    )

def example_time_series():
    """Example: Basic time series plot"""
    print("Example 1: Basic Time Series Plot")
    print("-" * 40)
    
    # PLACEHOLDER: Replace with your actual data
    print("❌ This example requires actual aethalometer data.")
    print("   Please load your data and pass it to the visualization functions.")
    return None
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
    
    # PLACEHOLDER: Replace with your actual data
    print("❌ This example requires actual aethalometer data.")
    print("   Please load your data and pass it to the visualization functions.")
    return None
    
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
    
    # PLACEHOLDER: Replace with your actual data
    print("❌ This example requires actual aethalometer data.")
    print("   Please load your data and pass it to the visualization functions.")
    return None
    
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
    
    # PLACEHOLDER: Replace with your actual data
    print("❌ This example requires actual aethalometer data.")
    print("   Please load your data and pass it to the visualization functions.")
    return None
    
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
    
    # PLACEHOLDER: Replace with your actual data
    print("❌ This example requires actual aethalometer data.")
    print("   Please load your data and pass it to the visualization functions.")
    return None
    
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
    
    # PLACEHOLDER: Replace with your actual data
    print("❌ This example requires actual aethalometer data.")
    print("   Please load your data and pass it to the visualization functions.")
    return None
    
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
    
    # PLACEHOLDER: Replace with your actual data
    print("❌ This example requires actual aethalometer data.")
    print("   Please load your data and pass it to the visualization functions.")
    return None
    
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
    
    # PLACEHOLDER: Replace with your actual data
    print("❌ This example requires actual aethalometer data.")
    print("   Please load your data and pass it to the visualization functions.")
    return None
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
    
    # PLACEHOLDER: Replace with your actual data
    print("❌ This example requires actual aethalometer data.")
    print("   Please load your data and pass it to the visualization functions.")
    return None
    
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
