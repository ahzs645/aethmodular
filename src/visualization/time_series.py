"""Time series visualization for aethalometer and FTIR data"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import seaborn as sns
from datetime import datetime

class TimeSeriesPlotter:
    """
    Comprehensive time series plotting for aethalometer and FTIR data
    
    Features:
    - Temporal comparison plots: Stacked time series for different methods
    - Diurnal pattern analysis: Hour-by-hour missing data patterns  
    - Weekly pattern heatmaps: Day-of-week × hour missing data visualization
    - Seasonal heatmaps: Month × year missing data patterns
    """
    
    def __init__(self, style: str = 'whitegrid', figsize: Tuple[int, int] = (12, 8)):
        self.style = style
        self.figsize = figsize
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup consistent plotting style"""
        sns.set_style(self.style)
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['legend.fontsize'] = 12
    
    def plot_smoothening_comparison(self, original_data: np.ndarray, 
                                  smoothed_results: Dict[str, np.ndarray],
                                  timestamps: pd.Series,
                                  title: str = "Smoothening Methods Comparison") -> plt.Figure:
        """
        Plot comparison of original vs smoothed data using different methods
        
        Parameters:
        -----------
        original_data : np.ndarray
            Original BC data
        smoothed_results : Dict[str, np.ndarray]
            Dictionary of smoothed data from different methods
        timestamps : pd.Series
            Timestamp data
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(len(smoothed_results) + 1, 1, 
                                figsize=(15, 4 * (len(smoothed_results) + 1)), 
                                sharex=True)
        
        if len(smoothed_results) == 0:
            axes = [axes]
        
        # Plot original data
        axes[0].plot(timestamps, original_data, 'k-', alpha=0.7, linewidth=0.8, label='Original')
        axes[0].set_ylabel('BC (ng/m³)')
        axes[0].set_title('Original Data')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot each smoothened method
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (method, smoothed_data) in enumerate(smoothed_results.items()):
            ax = axes[i + 1]
            
            # Plot original (faded) and smoothed
            ax.plot(timestamps, original_data, 'k-', alpha=0.3, linewidth=0.5, label='Original')
            ax.plot(timestamps, smoothed_data, color=colors[i % len(colors)], 
                   linewidth=1.5, label=f'{method} Smoothed')
            
            ax.set_ylabel('BC (ng/m³)')
            ax.set_title(f'{method} Smoothening')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        return fig
    
    def plot_diurnal_patterns(self, data: pd.DataFrame, date_column: str,
                             value_columns: List[str],
                             missing_data_analysis: bool = True) -> plt.Figure:
        """
        Plot diurnal (hourly) patterns and missing data analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with datetime and value columns
        date_column : str
            Name of datetime column
        value_columns : List[str]
            Columns to analyze
        missing_data_analysis : bool
            Whether to include missing data analysis
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        # Ensure datetime column
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
        
        data['hour'] = data[date_column].dt.hour
        
        n_cols = len(value_columns)
        n_rows = 2 if missing_data_analysis else 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows))
        
        if n_cols == 1:
            axes = axes.reshape(-1, 1) if n_rows > 1 else [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(value_columns):
            # Plot 1: Average values by hour
            hourly_stats = data.groupby('hour')[col].agg(['mean', 'std', 'count']).reset_index()
            
            ax1 = axes[0, i] if n_rows > 1 else axes[0] if n_cols == 1 else axes[i]
            
            # Plot mean with error bars
            ax1.errorbar(hourly_stats['hour'], hourly_stats['mean'], 
                        yerr=hourly_stats['std'], capsize=3, capthick=2,
                        marker='o', markersize=6, linewidth=2, label=f'Mean ± Std')
            
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel(f'{col}')
            ax1.set_title(f'Diurnal Pattern: {col}')
            ax1.set_xticks(range(0, 24, 2))
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            if missing_data_analysis:
                # Plot 2: Missing data by hour
                ax2 = axes[1, i]
                
                # Calculate missing data percentage by hour
                total_by_hour = data.groupby('hour').size()
                valid_by_hour = data.groupby('hour')[col].count()
                missing_percent = (1 - valid_by_hour / total_by_hour) * 100
                
                bars = ax2.bar(missing_percent.index, missing_percent.values, 
                              alpha=0.7, color='red')
                
                ax2.set_xlabel('Hour of Day')
                ax2.set_ylabel('Missing Data (%)')
                ax2.set_title(f'Missing Data by Hour: {col}')
                ax2.set_xticks(range(0, 24, 2))
                ax2.grid(True, alpha=0.3)
                
                # Add percentage labels on bars
                for bar, pct in zip(bars, missing_percent.values):
                    if pct > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_weekly_heatmap(self, data: pd.DataFrame, date_column: str,
                           value_column: str, missing_data: bool = True) -> plt.Figure:
        """
        Create heatmap showing day-of-week × hour patterns
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with datetime and value columns
        date_column : str
            Name of datetime column
        value_column : str
            Column to analyze
        missing_data : bool
            Whether to show missing data (True) or actual values (False)
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        # Prepare data
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
        
        data['hour'] = data[date_column].dt.hour
        data['day_of_week'] = data[date_column].dt.day_name()
        
        # Create pivot table
        if missing_data:
            # Calculate missing data percentage
            pivot_data = data.groupby(['day_of_week', 'hour']).agg({
                value_column: ['count', 'size']
            }).reset_index()
            
            pivot_data.columns = ['day_of_week', 'hour', 'valid_count', 'total_count']
            pivot_data['missing_percent'] = (1 - pivot_data['valid_count'] / pivot_data['total_count']) * 100
            
            heatmap_data = pivot_data.pivot(index='day_of_week', columns='hour', values='missing_percent')
            title = f'Missing Data Heatmap: {value_column} (% Missing)'
            cmap = 'Reds'
            cbar_label = 'Missing Data (%)'
        else:
            # Calculate mean values
            pivot_data = data.groupby(['day_of_week', 'hour'])[value_column].mean().reset_index()
            heatmap_data = pivot_data.pivot(index='day_of_week', columns='hour', values=value_column)
            title = f'Weekly Pattern Heatmap: {value_column} (Mean Values)'
            cmap = 'viridis'
            cbar_label = f'Mean {value_column}'
        
        # Reorder days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 8))
        
        sns.heatmap(heatmap_data, annot=False, cmap=cmap, ax=ax,
                   cbar_kws={'label': cbar_label}, fmt='.1f')
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Hour of Day', fontsize=14)
        ax.set_ylabel('Day of Week', fontsize=14)
        
        # Format x-axis to show all hours
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        
        plt.tight_layout()
        return fig
    
    def plot_seasonal_heatmap(self, data: pd.DataFrame, date_column: str,
                             value_column: str, missing_data: bool = True) -> plt.Figure:
        """
        Create heatmap showing month × year patterns
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with datetime and value columns
        date_column : str
            Name of datetime column
        value_column : str
            Column to analyze
        missing_data : bool
            Whether to show missing data (True) or actual values (False)
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        # Prepare data
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
        
        data['year'] = data[date_column].dt.year
        data['month'] = data[date_column].dt.month
        
        # Create pivot table
        if missing_data:
            # Calculate missing data percentage
            pivot_data = data.groupby(['year', 'month']).agg({
                value_column: ['count', 'size']
            }).reset_index()
            
            pivot_data.columns = ['year', 'month', 'valid_count', 'total_count']
            pivot_data['missing_percent'] = (1 - pivot_data['valid_count'] / pivot_data['total_count']) * 100
            
            heatmap_data = pivot_data.pivot(index='month', columns='year', values='missing_percent')
            title = f'Seasonal Missing Data: {value_column} (% Missing)'
            cmap = 'Reds'
            cbar_label = 'Missing Data (%)'
        else:
            # Calculate mean values
            pivot_data = data.groupby(['year', 'month'])[value_column].mean().reset_index()
            heatmap_data = pivot_data.pivot(index='month', columns='year', values=value_column)
            title = f'Seasonal Pattern: {value_column} (Mean Values)'
            cmap = 'viridis'
            cbar_label = f'Mean {value_column}'
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(heatmap_data.columns) * 0.8), 10))
        
        sns.heatmap(heatmap_data, annot=True, cmap=cmap, ax=ax,
                   cbar_kws={'label': cbar_label}, fmt='.1f')
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Month', fontsize=14)
        
        # Format y-axis to show month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_yticklabels(month_names)
        
        plt.tight_layout()
        return fig
    
    def plot_period_quality_overview(self, period_results: Dict[str, Any]) -> plt.Figure:
        """
        Plot overview of 9AM-to-9AM period quality classification
        
        Parameters:
        -----------
        period_results : Dict
            Results from NineAMPeriodProcessor
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        classifications = period_results['period_classifications']
        
        if not classifications:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No period data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            return fig
        
        # Extract data
        dates = [pd.to_datetime(c['date_label']) for c in classifications]
        qualities = [c['quality'] for c in classifications]
        completeness = [c['data_completeness']['completeness_percentage'] for c in classifications]
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Quality over time
        quality_colors = {'excellent': 'green', 'good': 'orange', 'poor': 'red'}
        colors = [quality_colors[q] for q in qualities]
        
        ax1.scatter(dates, range(len(dates)), c=colors, s=50, alpha=0.7)
        ax1.set_ylabel('Period Index')
        ax1.set_title('Period Quality Classification Over Time')
        
        # Add legend
        for quality, color in quality_colors.items():
            ax1.scatter([], [], c=color, label=quality.capitalize())
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Completeness percentage over time
        ax2.plot(dates, completeness, 'b-', marker='o', markersize=4, linewidth=1)
        ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
        ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.set_ylabel('Completeness (%)')
        ax2.set_title('Data Completeness Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: Quality distribution
        quality_counts = {}
        for q in qualities:
            quality_counts[q] = quality_counts.get(q, 0) + 1
        
        wedges, texts, autotexts = ax3.pie(quality_counts.values(), labels=quality_counts.keys(),
                                         colors=[quality_colors[q] for q in quality_counts.keys()],
                                         autopct='%1.1f%%', startangle=90)
        ax3.set_title('Overall Quality Distribution')
        
        # Format x-axis for time plots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
