"""
Time Series Visualization Templates

This module contains specialized templates for time series visualizations
including basic time series plots, smoothening comparisons, and diurnal patterns.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .base_template import BaseVisualizationTemplate

class TimeSeriesTemplate(BaseVisualizationTemplate):
    """Template for basic time series visualizations"""
    
    REQUIRED_PARAMS = ['data']
    OPTIONAL_PARAMS = ['columns', 'title', 'ylabel', 'xlabel']
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate parameters for time series plotting"""
        data = kwargs.get('data')
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Parameter 'data' must be a pandas DataFrame")
        
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            raise ValueError("DataFrame index must be datetime type")
        
        return True
    
    def create_plot(self, **kwargs) -> plt.Figure:
        """Create basic time series plot"""
        self.validate_parameters(**kwargs)
        
        data = kwargs['data']
        columns = kwargs.get('columns') or self._auto_detect_bc_columns(data)
        title = kwargs.get('title', 'Time Series Plot')
        ylabel = kwargs.get('ylabel', 'Concentration (μg/m³)')
        xlabel = kwargs.get('xlabel', 'Date/Time')
        
        fig, ax = plt.subplots(figsize=self.config['figsize'])
        
        # Plot each column
        colors = self._get_color_palette(len(columns))
        for i, col in enumerate(columns):
            if col in data.columns:
                ax.plot(data.index, data[col], 
                       label=col, color=colors[i], linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig


class SmootheningComparisonTemplate(BaseVisualizationTemplate):
    """Template for smoothening method comparison plots"""
    
    REQUIRED_PARAMS = ['original_data', 'smoothed_results', 'timestamps']
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate smoothening comparison parameters"""
        required = ['original_data', 'smoothed_results', 'timestamps']
        for param in required:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
        return True
    
    def create_plot(self, **kwargs) -> plt.Figure:
        """Create smoothening methods comparison plot"""
        self.validate_parameters(**kwargs)
        
        original_data = kwargs['original_data']
        smoothed_results = kwargs['smoothed_results']
        timestamps = kwargs['timestamps']
        title = kwargs.get('title', 'Smoothening Methods Comparison')
        
        n_methods = len(smoothed_results)
        fig, axes = plt.subplots(n_methods + 1, 1, 
                                figsize=(15, 4 * (n_methods + 1)), 
                                sharex=True)
        
        if n_methods == 0:
            axes = [axes]
        
        # Plot original data
        axes[0].plot(timestamps, original_data, 'k-', alpha=0.7, 
                    linewidth=0.8, label='Original')
        axes[0].set_ylabel('BC (ng/m³)')
        axes[0].set_title('Original Data')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot each smoothened method
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (method, smoothed_data) in enumerate(smoothed_results.items()):
            ax = axes[i + 1]
            
            # Plot original (faded) and smoothed
            ax.plot(timestamps, original_data, 'k-', alpha=0.3, 
                   linewidth=0.5, label='Original')
            ax.plot(timestamps, smoothed_data, color=colors[i % len(colors)], 
                   linewidth=1.5, label=f'{method} Smoothed')
            
            ax.set_ylabel('BC (ng/m³)')
            ax.set_title(f'{method} Smoothening')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        return fig


class DiurnalPatternTemplate(BaseVisualizationTemplate):
    """Template for diurnal pattern analysis"""
    
    REQUIRED_PARAMS = ['data', 'date_column', 'value_columns']
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate diurnal pattern parameters"""
        data = kwargs.get('data')
        date_column = kwargs.get('date_column')
        value_columns = kwargs.get('value_columns')
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame")
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")
        
        return True
    
    def create_plot(self, **kwargs) -> plt.Figure:
        """Create diurnal pattern analysis plot"""
        self.validate_parameters(**kwargs)
        
        data = kwargs['data'].copy()
        date_column = kwargs['date_column']
        value_columns = kwargs['value_columns']
        missing_data_analysis = kwargs.get('missing_data_analysis', True)
        
        # Ensure datetime column
        data = self._ensure_datetime_index(data, date_column)
        
        # Extract hour
        data['hour'] = data[date_column].dt.hour
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Hourly means with error bars
        for col in value_columns:
            if col in data.columns:
                hourly_stats = data.groupby('hour')[col].agg(['mean', 'std', 'count'])
                
                axes[0].errorbar(hourly_stats.index, hourly_stats['mean'], 
                               yerr=hourly_stats['std'], 
                               marker='o', capsize=5, label=col)
        
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Mean Concentration')
        axes[0].set_title('Diurnal Patterns (24-hour cycle)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(range(0, 24, 2))
        
        # Plot 2: Missing data analysis
        if missing_data_analysis:
            missing_by_hour = data.groupby('hour')[value_columns].apply(
                lambda x: x.isnull().sum() / len(x) * 100
            )
            
            for col in value_columns:
                if col in missing_by_hour.columns:
                    axes[1].plot(missing_by_hour.index, missing_by_hour[col], 
                               marker='s', label=f'{col} Missing %')
            
            axes[1].set_xlabel('Hour of Day')
            axes[1].set_ylabel('Missing Data (%)')
            axes[1].set_title('Missing Data by Hour')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        return fig
