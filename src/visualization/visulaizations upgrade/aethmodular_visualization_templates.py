"""
AethModular Visualization Templates
Implementation of templatable visualization system based on aethdata-analysis extraction

This module provides a flexible template system for creating consistent visualizations
across different aethalometer analysis workflows.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from pathlib import Path
import json

# ============================================================================
# Base Template System
# ============================================================================

class BaseVisualizationTemplate(ABC):
    """
    Abstract base class for all visualization templates
    Provides consistent interface and shared functionality
    """
    
    def __init__(self, template_name: str, config: Optional[Dict] = None):
        self.template_name = template_name
        self.config = config or self._load_default_config()
        self.setup_styling()
    
    def _load_default_config(self) -> Dict:
        """Load default configuration for the template"""
        return {
            'figsize': (12, 8),
            'style': 'whitegrid',
            'color_palette': 'Set1',
            'font_size': 12,
            'save_format': 'png',
            'dpi': 300
        }
    
    def setup_styling(self):
        """Configure matplotlib/seaborn styling"""
        sns.set_style(self.config.get('style', 'whitegrid'))
        plt.rcParams['figure.figsize'] = self.config.get('figsize', (12, 8))
        plt.rcParams['font.size'] = self.config.get('font_size', 12)
        plt.rcParams['axes.labelsize'] = self.config.get('font_size', 12) + 2
        plt.rcParams['axes.titlesize'] = self.config.get('font_size', 12) + 4
    
    @abstractmethod
    def validate_parameters(self, **kwargs) -> bool:
        """Validate required parameters for the template"""
        pass
    
    @abstractmethod
    def create_plot(self, **kwargs) -> plt.Figure:
        """Create the visualization - must be implemented by subclasses"""
        pass
    
    def save_plot(self, fig: plt.Figure, path: str, **kwargs):
        """Save plot with consistent formatting"""
        save_kwargs = {
            'dpi': self.config.get('dpi', 300),
            'bbox_inches': 'tight',
            'format': self.config.get('save_format', 'png')
        }
        save_kwargs.update(kwargs)
        fig.savefig(path, **save_kwargs)

# ============================================================================
# Time Series Templates
# ============================================================================

class TimeSeriesTemplate(BaseVisualizationTemplate):
    """Template for time series visualizations"""
    
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
        columns = kwargs.get('columns') or self._auto_detect_columns(data)
        title = kwargs.get('title', 'Time Series Plot')
        ylabel = kwargs.get('ylabel', 'Concentration (μg/m³)')
        xlabel = kwargs.get('xlabel', 'Date/Time')
        
        fig, ax = plt.subplots(figsize=self.config['figsize'])
        
        # Plot each column
        colors = plt.cm.Set1(np.linspace(0, 1, len(columns)))
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
    
    def _auto_detect_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect BC columns for plotting"""
        bc_columns = [col for col in data.columns if 'BC' in str(col) and 'c' in str(col)]
        return bc_columns[:5]  # Limit to first 5 columns


class SmootheningComparisonTemplate(TimeSeriesTemplate):
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
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])
        
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

# ============================================================================
# Heatmap Templates
# ============================================================================

class HeatmapTemplate(BaseVisualizationTemplate):
    """Base template for heatmap visualizations"""
    
    def _prepare_heatmap_data(self, data: pd.DataFrame, 
                             date_column: str, value_column: str,
                             x_grouper: str, y_grouper: str,
                             missing_data: bool = False) -> pd.DataFrame:
        """Prepare data for heatmap visualization"""
        df = data.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Add grouping columns
        if x_grouper == 'hour':
            df['x_group'] = df[date_column].dt.hour
        elif x_grouper == 'day_of_week':
            df['x_group'] = df[date_column].dt.day_name()
        elif x_grouper == 'month':
            df['x_group'] = df[date_column].dt.month
        elif x_grouper == 'year':
            df['x_group'] = df[date_column].dt.year
        
        if y_grouper == 'day_of_week':
            df['y_group'] = df[date_column].dt.day_name()
        elif y_grouper == 'month':
            df['y_group'] = df[date_column].dt.month
        elif y_grouper == 'year':
            df['y_group'] = df[date_column].dt.year
        
        # Create pivot table
        if missing_data:
            # Calculate missing data percentage
            grouped = df.groupby(['y_group', 'x_group']).agg({
                value_column: ['count', 'size']
            }).reset_index()
            
            grouped.columns = ['y_group', 'x_group', 'valid_count', 'total_count']
            grouped['missing_percent'] = (1 - grouped['valid_count'] / grouped['total_count']) * 100
            heatmap_data = grouped.pivot(index='y_group', columns='x_group', values='missing_percent')
        else:
            # Calculate mean values
            heatmap_data = df.pivot_table(index='y_group', columns='x_group', 
                                        values=value_column, aggfunc='mean')
        
        return heatmap_data


class WeeklyHeatmapTemplate(HeatmapTemplate):
    """Template for weekly pattern heatmaps (day-of-week × hour)"""
    
    REQUIRED_PARAMS = ['data', 'date_column', 'value_column']
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate weekly heatmap parameters"""
        for param in self.REQUIRED_PARAMS:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
        return True
    
    def create_plot(self, **kwargs) -> plt.Figure:
        """Create weekly pattern heatmap"""
        self.validate_parameters(**kwargs)
        
        data = kwargs['data']
        date_column = kwargs['date_column']
        value_column = kwargs['value_column']
        missing_data = kwargs.get('missing_data', False)
        title = kwargs.get('title')
        
        # Prepare heatmap data
        heatmap_data = self._prepare_heatmap_data(
            data, date_column, value_column, 
            x_grouper='hour', y_grouper='day_of_week', 
            missing_data=missing_data
        )
        
        # Set colors and labels based on mode
        if missing_data:
            if not title:
                title = f'Weekly Missing Data Pattern: {value_column}'
            cmap = 'Reds'
            cbar_label = 'Missing Data (%)'
        else:
            if not title:
                title = f'Weekly Pattern: {value_column} (Mean Values)'
            cmap = 'viridis'
            cbar_label = f'Mean {value_column}'
        
        # Reorder days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 8))
        
        sns.heatmap(heatmap_data, annot=False, cmap=cmap, ax=ax,
                   cbar_kws={'label': cbar_label}, fmt='.1f')
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Hour of Day', fontsize=14)
        ax.set_ylabel('Day of Week', fontsize=14)
        
        # Format x-axis
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        
        plt.tight_layout()
        return fig


class SeasonalHeatmapTemplate(HeatmapTemplate):
    """Template for seasonal heatmaps (month × year)"""
    
    REQUIRED_PARAMS = ['data', 'date_column', 'value_column']
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate seasonal heatmap parameters"""
        for param in self.REQUIRED_PARAMS:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
        return True
    
    def create_plot(self, **kwargs) -> plt.Figure:
        """Create seasonal pattern heatmap"""
        self.validate_parameters(**kwargs)
        
        data = kwargs['data']
        date_column = kwargs['date_column']
        value_column = kwargs['value_column']
        missing_data = kwargs.get('missing_data', False)
        title = kwargs.get('title')
        
        # Prepare heatmap data
        heatmap_data = self._prepare_heatmap_data(
            data, date_column, value_column,
            x_grouper='year', y_grouper='month',
            missing_data=missing_data
        )
        
        # Set colors and labels
        if missing_data:
            if not title:
                title = f'Seasonal Missing Data: {value_column}'
            cmap = 'Reds'
            cbar_label = 'Missing Data (%)'
        else:
            if not title:
                title = f'Seasonal Pattern: {value_column}'
            cmap = 'viridis'
            cbar_label = f'Mean {value_column}'
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(heatmap_data.columns) * 0.8), 10))
        
        sns.heatmap(heatmap_data, annot=True, cmap=cmap, ax=ax,
                   cbar_kws={'label': cbar_label}, fmt='.1f')
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Month', fontsize=14)
        
        # Format y-axis with month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_yticklabels(month_names)
        
        plt.tight_layout()
        return fig

# ============================================================================
# Scientific Analysis Templates
# ============================================================================

class MACAnalysisTemplate(BaseVisualizationTemplate):
    """Template for MAC (Mass Absorption Cross-section) analysis"""
    
    REQUIRED_PARAMS = ['fabs_data', 'ec_data']
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate MAC analysis parameters"""
        fabs_data = kwargs.get('fabs_data')
        ec_data = kwargs.get('ec_data')
        
        if fabs_data is None or ec_data is None:
            raise ValueError("Both 'fabs_data' and 'ec_data' are required")
        
        if len(fabs_data) != len(ec_data):
            raise ValueError("fabs_data and ec_data must have same length")
        
        return True
    
    def create_plot(self, **kwargs) -> plt.Figure:
        """Create MAC analysis plot with multiple methods"""
        self.validate_parameters(**kwargs)
        
        fabs_data = np.array(kwargs['fabs_data'])
        ec_data = np.array(kwargs['ec_data'])
        title = kwargs.get('title', 'MAC Analysis - Multiple Methods')
        
        # Filter valid data
        valid_mask = (fabs_data > 0) & (ec_data > 0) & np.isfinite(fabs_data) & np.isfinite(ec_data)
        fabs_clean = fabs_data[valid_mask]
        ec_clean = ec_data[valid_mask]
        
        if len(fabs_clean) < 5:
            raise ValueError("Insufficient valid data points for MAC analysis")
        
        # Calculate MAC using different methods
        mac_methods = self._calculate_mac_methods(fabs_clean, ec_clean)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (method_name, method_data) in enumerate(mac_methods.items()):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(ec_clean, fabs_clean, alpha=0.6, s=30)
            
            # Add regression line if available
            if 'predictions' in method_data:
                x_line = np.linspace(ec_clean.min(), ec_clean.max(), 100)
                if method_name == 'Linear Regression (Origin)':
                    y_line = method_data['mac_value'] * x_line
                else:
                    y_line = (method_data['mac_value'] * x_line + 
                             method_data.get('intercept', 0))
                ax.plot(x_line, y_line, color=colors[i], linewidth=2, 
                       label=f"MAC = {method_data['mac_value']:.2f}")
            
            ax.set_xlabel('EC (μg/m³)')
            ax.set_ylabel('Fabs (Mm⁻¹)')
            ax.set_title(f"{method_name}\nMAC = {method_data['mac_value']:.2f} m²/g")
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            if 'r_squared' in method_data:
                ax.text(0.05, 0.95, f"R² = {method_data['r_squared']:.3f}", 
                       transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def _calculate_mac_methods(self, fabs: np.ndarray, ec: np.ndarray) -> Dict[str, Dict]:
        """Calculate MAC using 4 different methods"""
        from sklearn.linear_model import LinearRegression
        from scipy import stats
        
        methods = {}
        
        # Method 1: Individual MAC mean
        individual_mac = fabs / ec
        methods['Individual MAC Mean'] = {
            'mac_value': np.mean(individual_mac),
            'std': np.std(individual_mac)
        }
        
        # Method 2: Ratio of means
        methods['Ratio of Means'] = {
            'mac_value': np.mean(fabs) / np.mean(ec)
        }
        
        # Method 3: Linear regression with intercept
        reg_intercept = LinearRegression().fit(ec.reshape(-1, 1), fabs)
        y_pred_intercept = reg_intercept.predict(ec.reshape(-1, 1))
        methods['Linear Regression (Intercept)'] = {
            'mac_value': reg_intercept.coef_[0],
            'intercept': reg_intercept.intercept_,
            'r_squared': reg_intercept.score(ec.reshape(-1, 1), fabs),
            'predictions': y_pred_intercept
        }
        
        # Method 4: Linear regression through origin
        reg_origin = LinearRegression(fit_intercept=False).fit(ec.reshape(-1, 1), fabs)
        y_pred_origin = reg_origin.predict(ec.reshape(-1, 1))
        methods['Linear Regression (Origin)'] = {
            'mac_value': reg_origin.coef_[0],
            'r_squared': reg_origin.score(ec.reshape(-1, 1), fabs),
            'predictions': y_pred_origin
        }
        
        return methods

# ============================================================================
# Template Factory and Registry
# ============================================================================

class VisualizationTemplateFactory:
    """Factory for creating visualization templates"""
    
    _templates = {
        'time_series': TimeSeriesTemplate,
        'smoothening_comparison': SmootheningComparisonTemplate,
        'diurnal_patterns': DiurnalPatternTemplate,
        'weekly_heatmap': WeeklyHeatmapTemplate,
        'seasonal_heatmap': SeasonalHeatmapTemplate,
        'mac_analysis': MACAnalysisTemplate,
    }
    
    @classmethod
    def create_template(cls, template_type: str, config: Optional[Dict] = None) -> BaseVisualizationTemplate:
        """Create a visualization template of the specified type"""
        if template_type not in cls._templates:
            available = ', '.join(cls._templates.keys())
            raise ValueError(f"Unknown template type '{template_type}'. Available: {available}")
        
        template_class = cls._templates[template_type]
        return template_class(template_type, config)
    
    @classmethod
    def register_template(cls, template_type: str, template_class):
        """Register a new template type"""
        cls._templates[template_type] = template_class
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List all available template types"""
        return list(cls._templates.keys())

# ============================================================================
# Usage Examples
# ============================================================================

def example_usage():
    """Example usage of the visualization templates"""
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'IR BCc': np.random.lognormal(2, 0.5, 1000),
        'Blue BCc': np.random.lognormal(1.8, 0.6, 1000),
        'fabs': np.random.lognormal(3, 0.4, 1000),
        'ec_ftir': np.random.lognormal(1.5, 0.5, 1000)
    })
    sample_data.set_index('timestamp', inplace=True)
    
    # Example 1: Basic time series
    ts_template = VisualizationTemplateFactory.create_template('time_series')
    fig1 = ts_template.create_plot(
        data=sample_data,
        columns=['IR BCc', 'Blue BCc'],
        title='BC Concentrations Over Time'
    )
    ts_template.save_plot(fig1, 'time_series_example.png')
    
    # Example 2: Weekly heatmap
    weekly_template = VisualizationTemplateFactory.create_template('weekly_heatmap')
    sample_data_reset = sample_data.reset_index()
    fig2 = weekly_template.create_plot(
        data=sample_data_reset,
        date_column='timestamp',
        value_column='IR BCc',
        missing_data=False
    )
    weekly_template.save_plot(fig2, 'weekly_heatmap_example.png')
    
    # Example 3: MAC analysis
    mac_template = VisualizationTemplateFactory.create_template('mac_analysis')
    fig3 = mac_template.create_plot(
        fabs_data=sample_data['fabs'].values,
        ec_data=sample_data['ec_ftir'].values,
        title='MAC Analysis Example'
    )
    mac_template.save_plot(fig3, 'mac_analysis_example.png')
    
    print("Example plots created successfully!")

if __name__ == "__main__":
    example_usage()
