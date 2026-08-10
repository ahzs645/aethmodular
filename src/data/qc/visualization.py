"""
Quality Control Visualization Module

This module provides comprehensive visualization tools for data quality assessment
results, including missing data patterns, quality distributions, seasonal patterns,
and filter sample overlaps.

Author: AethModular Team
Created: 2025-01-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import calendar

logger = logging.getLogger(__name__)


class QualityVisualizer:
    """
    Comprehensive visualization tools for quality assessment results.
    
    This class provides methods to:
    - Plot missing data patterns and statistics
    - Visualize quality distributions and trends
    - Create seasonal and temporal pattern plots
    - Generate filter sample overlap visualizations
    - Create calendar views of data availability
    """
    
    def __init__(self, style: str = 'whitegrid', figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualization settings.
        
        Parameters:
        -----------
        style : str, default 'whitegrid'
            Seaborn plotting style
        figsize : tuple, default (12, 6)
            Default figure size
        """
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
        
        # Define consistent color schemes
        self.quality_colors = {
            'Excellent': '#1976d2',   # Blue
            'Good': '#388e3c',        # Green  
            'Moderate': '#f57c00',    # Orange
            'Poor': '#d32f2f',        # Red
            'No Data': '#9e9e9e',     # Gray
            'Missing': '#9e9e9e'      # Gray
        }
        
        self.season_colors = {
            'Dry Season': '#8d6e63',         # Brown
            'Belg Rainy Season': '#4caf50',  # Green
            'Kiremt Rainy Season': '#2196f3', # Blue
            'Winter': '#607d8b',             # Blue gray
            'Spring': '#4caf50',             # Green
            'Summer': '#ff9800',             # Orange
            'Autumn': '#795548'              # Brown
        }
    
    def plot_missing_patterns(self, missing_analysis: Dict, 
                             show_partial_only: bool = True) -> None:
        """
        Plot comprehensive missing data patterns.
        
        Parameters:
        -----------
        missing_analysis : dict
            Results from MissingDataAnalyzer.analyze_missing_patterns()
        show_partial_only : bool, default True
            If True, focus plots on partial missing days (exclude full gaps)
        """
        logger.info("Creating missing data pattern plots...")
        
        # Get data
        missing_per_day = missing_analysis['daily_patterns']['missing_per_day']
        partial_missing = missing_analysis['daily_patterns']['partial_missing_days']
        missing_per_hour = missing_analysis['temporal_patterns']['missing_per_hour']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Daily missing counts
        if len(missing_per_day) > 0:
            if show_partial_only and len(partial_missing) > 0:
                partial_missing.plot(kind='bar', ax=axes[0, 0], color='skyblue')
                axes[0, 0].set_title("Missing Minutes per Day (Partial Days Only)")
            else:
                missing_per_day.plot(kind='bar', ax=axes[0, 0], color='lightcoral')
                axes[0, 0].set_title("Missing Minutes per Day (All Days)")
            
            axes[0, 0].set_xlabel("Date")
            axes[0, 0].set_ylabel("Missing Minutes")
            axes[0, 0].tick_params(axis='x', rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                           transform=axes[0, 0].transAxes, fontsize=14)
            axes[0, 0].set_title("Daily Missing Pattern")
        
        # 2. Hourly missing pattern
        if len(missing_per_hour) > 0:
            missing_per_hour.plot.bar(ax=axes[0, 1], color='orange')
            axes[0, 1].set_title("Missing Samples per Hour")
            axes[0, 1].set_xlabel("Hour of Day")
            axes[0, 1].set_ylabel("Missing Count")
            axes[0, 1].tick_params(axis='x', rotation=0)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Data', ha='center', va='center',
                           transform=axes[0, 1].transAxes, fontsize=14)
            axes[0, 1].set_title("Hourly Missing Pattern")
        
        # 3. Missing distribution histogram
        if len(partial_missing) > 0:
            # Focus on reasonable range for better visualization
            plot_data = partial_missing[partial_missing <= 300]  # Up to 5 hours
            
            sns.histplot(plot_data, bins=30, kde=True, ax=axes[1, 0], color='lightgreen')
            axes[1, 0].axvline(partial_missing.median(), color='red', linestyle='--',
                              label=f"Median: {partial_missing.median():.0f} min")
            axes[1, 0].axvline(partial_missing.quantile(0.9), color='orange', linestyle='--',
                              label=f"90th %: {partial_missing.quantile(0.9):.0f} min")
            axes[1, 0].set_title("Distribution of Missing Minutes (Partial Days)")
            axes[1, 0].set_xlabel("Missing Minutes")
            axes[1, 0].set_ylabel("Number of Days")
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No Partial Missing Days', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title("Missing Distribution")
        
        # 4. Monthly heatmap
        missing_idx = missing_analysis['missing_indices']
        if len(missing_idx) > 0:
            miss_df = pd.DataFrame(index=missing_idx)
            miss_df['year'] = miss_df.index.year
            miss_df['month'] = miss_df.index.month
            
            pivot_my = miss_df.groupby(['month', 'year']).size().unstack(fill_value=0)
            
            if not pivot_my.empty:
                sns.heatmap(pivot_my, cmap='Reds', ax=axes[1, 1], cbar_kws={'label': 'Missing Minutes'},
                           yticklabels=['Jan','Feb','Mar','Apr','May','Jun',
                                       'Jul','Aug','Sep','Oct','Nov','Dec'])
                axes[1, 1].set_title("Missing Minutes by Month & Year")
                axes[1, 1].set_xlabel("Year")
                axes[1, 1].set_ylabel("Month")
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                               transform=axes[1, 1].transAxes, fontsize=14)
                axes[1, 1].set_title("Monthly Pattern")
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Data', ha='center', va='center',
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title("Monthly Pattern")
        
        plt.tight_layout()
        plt.show()
    
    def plot_quality_distribution(self, quality_series: pd.Series, 
                                 title: str = "Data Quality Distribution") -> None:
        """
        Plot quality distribution with statistics.
        
        Parameters:
        -----------
        quality_series : pd.Series
            Series with quality labels
        title : str
            Plot title
        """
        logger.info("Creating quality distribution plot...")
        
        quality_counts = quality_series.value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar chart
        colors = [self.quality_colors.get(q, '#cccccc') for q in quality_counts.index]
        bars = ax1.bar(quality_counts.index, quality_counts.values, color=colors)
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        ax1.set_title(title)
        ax1.set_ylabel("Number of Periods")
        ax1.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        total = sum(quality_counts.values)
        for i, (quality, count) in enumerate(quality_counts.items()):
            pct = count / total * 100
            ax1.text(i, count/2, f'{pct:.1f}%', ha='center', va='center', 
                    color='white', fontweight='bold')
        
        # Pie chart
        ax2.pie(quality_counts.values, labels=quality_counts.index, 
               colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Quality Distribution")
        
        plt.tight_layout()
        plt.show()
    
    def plot_seasonal_patterns(self, seasonal_analysis: Dict) -> None:
        """
        Plot seasonal missing data patterns.
        
        Parameters:
        -----------
        seasonal_analysis : dict
            Results from SeasonalPatternAnalyzer.analyze_seasonal_missing_patterns()
        """
        logger.info("Creating seasonal pattern plots...")
        
        seasonal_patterns = seasonal_analysis['seasonal_patterns']
        overall_pattern = seasonal_analysis['overall_pattern']
        
        if not seasonal_patterns:
            logger.warning("No seasonal patterns to plot")
            return
        
        # Calculate global max for consistent color scaling
        all_patterns = list(seasonal_patterns.values()) + [overall_pattern]
        valid_patterns = [p for p in all_patterns if not p.empty]
        
        if not valid_patterns:
            logger.warning("No valid patterns to plot")
            return
        
        vmax = max(p.max().max() for p in valid_patterns)
        
        n_seasons = len(seasonal_patterns)
        fig, axes = plt.subplots(1, n_seasons + 1, figsize=(5 * (n_seasons + 1), 6))
        
        if n_seasons == 0:
            axes.text(0.5, 0.5, 'No Seasonal Data', ha='center', va='center',
                     transform=axes.transAxes, fontsize=14)
            plt.show()
            return
        
        # Ensure axes is always a list
        if n_seasons == 0:
            axes = [axes]
        elif n_seasons == 1:
            axes = [axes[0], axes[1]]
        
        # Plot each season
        for i, (season, pattern) in enumerate(seasonal_patterns.items()):
            if not pattern.empty and i < len(axes) - 1:
                sns.heatmap(pattern, ax=axes[i], cmap='YlOrRd', vmin=0, vmax=vmax,
                           xticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                           yticklabels=[f'{h:02d}:00' for h in range(24)],
                           cbar_kws={'label': 'Missing Fraction'})
                axes[i].set_title(f"{season}")
                axes[i].set_xlabel("Day of Week")
                if i == 0:
                    axes[i].set_ylabel("Hour of Day")
        
        # Overall pattern
        if not overall_pattern.empty and len(axes) > n_seasons:
            sns.heatmap(overall_pattern, ax=axes[-1], cmap='YlOrRd', vmin=0, vmax=vmax,
                       xticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                       yticklabels=False, cbar_kws={'label': 'Missing Fraction'})
            axes[-1].set_title("Overall Pattern")
            axes[-1].set_xlabel("Day of Week")
        
        plt.suptitle("Weekly Diurnal Missing Data Patterns", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_filter_overlap_summary(self, overlap_results: Dict) -> None:
        """
        Plot summary of filter sample and quality overlap.
        
        Parameters:
        -----------
        overlap_results : dict
            Results from FilterSampleMapper.map_to_quality_periods()
        """
        logger.info("Creating filter overlap summary plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Quality distribution of filter periods
        filter_quality_table = overlap_results.get('filter_quality_table', pd.DataFrame())
        
        if not filter_quality_table.empty:
            colors = [self.quality_colors.get(q, '#cccccc') for q in filter_quality_table.index]
            
            bars = axes[0, 0].bar(filter_quality_table.index, filter_quality_table['Count'], color=colors)
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{int(height)}', ha='center', va='bottom')
            
            axes[0, 0].set_title("Quality Distribution of Filter Sample Periods")
            axes[0, 0].set_ylabel("Number of Periods")
            axes[0, 0].grid(axis='y', alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Filter Quality Data', ha='center', va='center',
                           transform=axes[0, 0].transAxes, fontsize=14)
        
        # 2. Overlap summary
        overlap_data = {
            'Total Filter Periods': len(overlap_results.get('filter_periods', [])),
            'High Quality Overlaps': len(overlap_results.get('overlap_periods', [])),
            'Excellent Overlaps': len(overlap_results.get('excellent_overlaps', [])),
            'Good Overlaps': len(overlap_results.get('good_overlaps', []))
        }
        
        bars = axes[0, 1].bar(overlap_data.keys(), overlap_data.values(), 
                             color=['#757575', '#4caf50', '#2196f3', '#8bc34a'])
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom')
        
        axes[0, 1].set_title("Filter Sample - Quality Overlap Summary")
        axes[0, 1].set_ylabel("Number of Periods")
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Timeline view
        filter_periods = overlap_results.get('filter_periods', pd.DatetimeIndex([]))
        overlap_periods = overlap_results.get('overlap_periods', pd.DatetimeIndex([]))
        
        if len(filter_periods) > 0:
            filter_series = pd.Series(1, index=filter_periods)
            overlap_series = pd.Series(1, index=overlap_periods)
            
            # Group by month for timeline
            filter_monthly = filter_series.groupby(filter_series.index.to_period('M')).count()
            hq_monthly = overlap_series.groupby(overlap_series.index.to_period('M')).count()
            
            x_pos = range(len(filter_monthly))
            width = 0.35
            
            axes[1, 0].bar([x - width/2 for x in x_pos], filter_monthly.values, 
                          width, label='All Filter Samples', color='#90a4ae')
            axes[1, 0].bar([x + width/2 for x in x_pos], 
                          hq_monthly.reindex(filter_monthly.index, fill_value=0).values,
                          width, label='High Quality Overlaps', color='#4caf50')
            
            axes[1, 0].set_title("Monthly Filter Samples and High-Quality Overlaps")
            axes[1, 0].set_xlabel("Month")
            axes[1, 0].set_ylabel("Number of Samples")
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels([str(p) for p in filter_monthly.index], rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Filter Period Data', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=14)
        
        # 4. Quality percentages pie chart
        if not filter_quality_table.empty:
            colors = [self.quality_colors.get(q, '#cccccc') for q in filter_quality_table.index]
            axes[1, 1].pie(filter_quality_table['Count'], labels=filter_quality_table.index,
                          colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title("Filter Period Quality Distribution")
        else:
            axes[1, 1].text(0.5, 0.5, 'No Data for Pie Chart', ha='center', va='center',
                           transform=axes[1, 1].transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    def plot_quality_timeline(self, quality_series: pd.Series, 
                             window: str = 'M') -> None:
        """
        Plot quality trends over time.
        
        Parameters:
        -----------
        quality_series : pd.Series
            Quality classification series
        window : str, default 'M'
            Aggregation window ('D', 'W', 'M', 'Q')
        """
        logger.info("Creating quality timeline plot...")
        
        if len(quality_series) == 0:
            logger.warning("No quality data to plot")
            return
        
        # Create quality indicator dataframe
        quality_df = pd.DataFrame({'quality': quality_series})
        quality_df['excellent'] = (quality_df['quality'] == 'Excellent').astype(int)
        quality_df['good'] = (quality_df['quality'] == 'Good').astype(int)
        quality_df['high_quality'] = (quality_df['quality'].isin(['Excellent', 'Good'])).astype(int)
        
        # Resample by window
        resampled = quality_df.resample(window).agg({
            'excellent': 'sum',
            'good': 'sum', 
            'high_quality': 'sum',
            'quality': 'count'
        })
        
        # Calculate percentages
        resampled['excellent_pct'] = resampled['excellent'] / resampled['quality'] * 100
        resampled['good_pct'] = resampled['good'] / resampled['quality'] * 100
        resampled['high_quality_pct'] = resampled['high_quality'] / resampled['quality'] * 100
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Stacked bar chart of counts
        resampled[['excellent', 'good']].plot(kind='bar', stacked=True, ax=ax1,
                                             color=[self.quality_colors['Excellent'], 
                                                   self.quality_colors['Good']])
        ax1.set_title(f"Quality Counts by {window}")
        ax1.set_ylabel("Number of Periods")
        ax1.legend(['Excellent', 'Good'])
        ax1.tick_params(axis='x', rotation=45)
        
        # Line plot of percentages
        resampled['high_quality_pct'].plot(ax=ax2, marker='o', color='green', linewidth=2)
        ax2.set_title(f"High Quality Percentage by {window}")
        ax2.set_ylabel("High Quality Percentage")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


def plot_missing_summary(missing_analysis: Dict) -> None:
    """
    Convenience function for plotting missing data summary.
    
    Parameters:
    -----------
    missing_analysis : dict
        Results from missing data analysis
    """
    visualizer = QualityVisualizer()
    visualizer.plot_missing_patterns(missing_analysis)


def plot_quality_summary(quality_series: pd.Series, title: str = "Data Quality Summary") -> None:
    """
    Convenience function for plotting quality distribution.
    
    Parameters:
    -----------
    quality_series : pd.Series
        Quality classification series
    title : str
        Plot title
    """
    visualizer = QualityVisualizer()
    visualizer.plot_quality_distribution(quality_series, title)


if __name__ == "__main__":
    # Example usage
    print("Quality Control Visualization Module")
    print("Use QualityVisualizer class or convenience functions")
