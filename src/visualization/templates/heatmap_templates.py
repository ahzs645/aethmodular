"""
Heatmap Visualization Templates

This module contains specialized templates for heatmap visualizations
including weekly patterns and seasonal analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from .base_template import BaseVisualizationTemplate

class HeatmapTemplate(BaseVisualizationTemplate):
    """Base template for heatmap visualizations"""
    
    def _prepare_heatmap_data(self, data: pd.DataFrame, 
                             date_column: str, value_column: str,
                             x_grouper: str, y_grouper: str,
                             missing_data: bool = False) -> pd.DataFrame:
        """Prepare data for heatmap visualization"""
        df = data.copy()
        
        # Ensure datetime
        df = self._ensure_datetime_index(df, date_column)
        
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
