"""
Seasonal Pattern Analysis Module

This module provides tools for analyzing seasonal and diurnal patterns in missing data
and quality metrics. It supports different seasonal definitions and can identify
temporal patterns in data quality.

Author: AethModular Team
Created: 2025-01-12
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Callable, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SeasonalPatternAnalyzer:
    """
    Analyzer for seasonal and diurnal patterns in data quality and missingness.
    
    This class provides methods to:
    - Analyze seasonal missing data patterns
    - Create diurnal (hourly) pattern analysis
    - Support custom season definitions
    - Generate weekly pattern analysis
    """
    
    def __init__(self, season_mapping_func: Optional[Callable] = None):
        """
        Initialize the seasonal pattern analyzer.
        
        Parameters:
        -----------
        season_mapping_func : callable, optional
            Function to map month numbers to season names.
            Default uses Ethiopian seasons.
        """
        self.season_mapping_func = season_mapping_func or self._default_season_mapping
        
    @staticmethod
    def _default_season_mapping(month: int) -> str:
        """Default Ethiopian season mapping."""
        if month in [10, 11, 12, 1, 2]:
            return 'Dry Season'
        elif month in [3, 4, 5]:
            return 'Belg Rainy Season'
        else:  # months 6–9
            return 'Kiremt Rainy Season'
    
    @staticmethod
    def _meteorological_season_mapping(month: int) -> str:
        """Standard meteorological season mapping."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # months 9, 10, 11
            return 'Autumn'
    
    def analyze_seasonal_missing_patterns(self, missing_analysis: Dict, 
                                        exclude_full_days: bool = True) -> Dict:
        """
        Analyze seasonal patterns in missing data.
        
        Parameters:
        -----------
        missing_analysis : dict
            Results from MissingDataAnalyzer.analyze_missing_patterns()
        exclude_full_days : bool, default True
            Whether to exclude fully missing days from pattern analysis
            
        Returns:
        --------
        dict
            Seasonal pattern analysis results
        """
        logger.info("Analyzing seasonal missing data patterns...")
        
        missing_idx = missing_analysis['missing_indices']
        expected_idx = missing_analysis['temporal_patterns']['expected_idx']
        
        if exclude_full_days:
            # Exclude full missing days for cleaner patterns
            missing_per_day = missing_analysis['daily_patterns']['missing_per_day']
            expected_per_day = missing_analysis['daily_patterns']['expected_per_day']
            full_days_missing = missing_per_day[missing_per_day >= expected_per_day].index
            partial_missing_idx = missing_idx[~missing_idx.normalize().isin(full_days_missing)]
        else:
            partial_missing_idx = missing_idx
        
        # Create DataFrame for analysis
        miss_df = pd.DataFrame(index=expected_idx)
        miss_df['is_missing'] = 0
        if len(partial_missing_idx) > 0:
            miss_df.loc[partial_missing_idx, 'is_missing'] = 1
        
        # Add time features
        miss_df['Hour'] = miss_df.index.hour
        miss_df['DayOfWeek'] = miss_df.index.dayofweek
        miss_df['Month'] = miss_df.index.month
        miss_df['Year'] = miss_df.index.year
        miss_df['Season'] = miss_df['Month'].map(self.season_mapping_func)
        
        # Create pivot tables for different patterns
        patterns = self._create_pattern_pivots(miss_df)
        
        return {
            'seasonal_patterns': patterns['seasonal'],
            'yearly_patterns': patterns['yearly'],
            'overall_pattern': patterns['overall'],
            'seasonal_summary': self._summarize_seasonal_patterns(miss_df),
            'seasons': miss_df['Season'].unique(),
            'years': sorted(miss_df['Year'].unique()),
            'analysis_df': miss_df,
            'excluded_full_days': exclude_full_days
        }
    
    def _create_pattern_pivots(self, miss_df: pd.DataFrame) -> Dict:
        """Create pivot tables for different temporal patterns."""
        patterns = {}
        
        # Seasonal patterns (hour × day of week)
        seasons = miss_df['Season'].unique()
        patterns['seasonal'] = {}
        for season in seasons:
            patterns['seasonal'][season] = miss_df[miss_df['Season'] == season]\
                .pivot_table(index='Hour', columns='DayOfWeek', 
                           values='is_missing', aggfunc='mean')
        
        # Yearly patterns (hour × day of week)
        years = sorted(miss_df['Year'].unique())
        patterns['yearly'] = {}
        for year in years:
            patterns['yearly'][year] = miss_df[miss_df['Year'] == year]\
                .pivot_table(index='Hour', columns='DayOfWeek', 
                           values='is_missing', aggfunc='mean')
        
        # Overall pattern (hour × day of week)
        patterns['overall'] = miss_df.pivot_table(
            index='Hour', columns='DayOfWeek', 
            values='is_missing', aggfunc='mean'
        )
        
        return patterns
    
    def _summarize_seasonal_patterns(self, miss_df: pd.DataFrame) -> Dict:
        """Summarize seasonal patterns with statistics."""
        summary = {}
        
        for season in miss_df['Season'].unique():
            season_data = miss_df[miss_df['Season'] == season]
            
            summary[season] = {
                'total_points': len(season_data),
                'missing_points': season_data['is_missing'].sum(),
                'missing_rate': season_data['is_missing'].mean(),
                'peak_missing_hour': season_data.groupby('Hour')['is_missing'].mean().idxmax(),
                'peak_missing_rate': season_data.groupby('Hour')['is_missing'].mean().max(),
                'best_hour': season_data.groupby('Hour')['is_missing'].mean().idxmin(),
                'best_hour_rate': season_data.groupby('Hour')['is_missing'].mean().min()
            }
        
        return summary
    
    def analyze_diurnal_patterns(self, missing_analysis: Dict, 
                                quality_series: Optional[pd.Series] = None) -> Dict:
        """
        Analyze diurnal (hourly) patterns in missing data and quality.
        
        Parameters:
        -----------
        missing_analysis : dict
            Results from missing data analysis
        quality_series : pd.Series, optional
            Quality classification series for additional analysis
            
        Returns:
        --------
        dict
            Diurnal pattern analysis
        """
        logger.info("Analyzing diurnal patterns...")
        
        missing_per_hour = missing_analysis['temporal_patterns']['missing_per_hour']
        
        # Calculate daily coverage for fraction calculation
        n_days = missing_analysis['timeline']['duration_days']
        missing_fraction_by_hour = missing_per_hour / n_days
        
        results = {
            'missing_count_by_hour': missing_per_hour,
            'missing_fraction_by_hour': missing_fraction_by_hour,
            'peak_missing_hour': missing_per_hour.idxmax() if len(missing_per_hour) > 0 else None,
            'best_hour': missing_per_hour.idxmin() if len(missing_per_hour) > 0 else None
        }
        
        # Add quality analysis if provided
        if quality_series is not None:
            quality_hourly = self._analyze_quality_by_hour(quality_series)
            results['quality_by_hour'] = quality_hourly
        
        return results
    
    def _analyze_quality_by_hour(self, quality_series: pd.Series) -> Dict:
        """Analyze quality patterns by hour of day."""
        # Create hourly quality summary
        quality_by_hour = {}
        
        for hour in range(24):
            hour_periods = quality_series[quality_series.index.hour == hour]
            if len(hour_periods) > 0:
                quality_counts = hour_periods.value_counts()
                quality_by_hour[hour] = {
                    'total_periods': len(hour_periods),
                    'quality_distribution': quality_counts.to_dict(),
                    'excellent_rate': quality_counts.get('Excellent', 0) / len(hour_periods),
                    'good_rate': quality_counts.get('Good', 0) / len(hour_periods),
                    'high_quality_rate': (quality_counts.get('Excellent', 0) + 
                                        quality_counts.get('Good', 0)) / len(hour_periods)
                }
        
        return quality_by_hour
    
    def analyze_weekly_patterns(self, missing_analysis: Dict) -> Dict:
        """
        Analyze weekly patterns in missing data.
        
        Parameters:
        -----------
        missing_analysis : dict
            Results from missing data analysis
            
        Returns:
        --------
        dict
            Weekly pattern analysis
        """
        logger.info("Analyzing weekly patterns...")
        
        missing_per_weekday = missing_analysis['temporal_patterns']['missing_per_weekday']
        
        # Map weekday numbers to names
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']
        
        weekday_summary = {}
        for weekday, count in missing_per_weekday.items():
            weekday_summary[weekday_names[weekday]] = {
                'missing_count': count,
                'weekday_number': weekday
            }
        
        # Find patterns
        n_weeks = missing_analysis['timeline']['duration_days'] / 7
        missing_rate_by_weekday = missing_per_weekday / n_weeks
        
        return {
            'missing_by_weekday': missing_per_weekday,
            'missing_rate_by_weekday': missing_rate_by_weekday,
            'weekday_summary': weekday_summary,
            'worst_weekday': weekday_names[missing_per_weekday.idxmax()] if len(missing_per_weekday) > 0 else None,
            'best_weekday': weekday_names[missing_per_weekday.idxmin()] if len(missing_per_weekday) > 0 else None
        }
    
    def compare_seasonal_quality(self, quality_series: pd.Series) -> Dict:
        """
        Compare quality across seasons.
        
        Parameters:
        -----------
        quality_series : pd.Series
            Quality classification series
            
        Returns:
        --------
        dict
            Seasonal quality comparison
        """
        logger.info("Comparing seasonal quality...")
        
        # Add season information to quality series
        quality_df = pd.DataFrame({'quality': quality_series})
        quality_df['month'] = quality_df.index.month
        quality_df['season'] = quality_df['month'].map(self.season_mapping_func)
        
        seasonal_comparison = {}
        
        for season in quality_df['season'].unique():
            season_data = quality_df[quality_df['season'] == season]
            quality_counts = season_data['quality'].value_counts()
            
            seasonal_comparison[season] = {
                'total_periods': len(season_data),
                'quality_distribution': quality_counts.to_dict(),
                'excellent_rate': quality_counts.get('Excellent', 0) / len(season_data),
                'good_rate': quality_counts.get('Good', 0) / len(season_data),
                'high_quality_rate': (quality_counts.get('Excellent', 0) + 
                                    quality_counts.get('Good', 0)) / len(season_data),
                'poor_rate': quality_counts.get('Poor', 0) / len(season_data)
            }
        
        # Find best and worst seasons
        best_season = max(seasonal_comparison.keys(), 
                         key=lambda s: seasonal_comparison[s]['high_quality_rate'])
        worst_season = min(seasonal_comparison.keys(), 
                          key=lambda s: seasonal_comparison[s]['high_quality_rate'])
        
        return {
            'seasonal_comparison': seasonal_comparison,
            'best_season': best_season,
            'worst_season': worst_season,
            'season_ranking': sorted(seasonal_comparison.keys(), 
                                   key=lambda s: seasonal_comparison[s]['high_quality_rate'], 
                                   reverse=True)
        }


def analyze_seasonal_patterns(missing_analysis: Dict, 
                            season_mapping: Optional[Callable] = None,
                            exclude_full_days: bool = True) -> Dict:
    """
    Convenience function for seasonal pattern analysis.
    
    Parameters:
    -----------
    missing_analysis : dict
        Results from missing data analysis
    season_mapping : callable, optional
        Custom season mapping function
    exclude_full_days : bool, default True
        Whether to exclude fully missing days
        
    Returns:
    --------
    dict
        Seasonal pattern analysis results
    """
    analyzer = SeasonalPatternAnalyzer(season_mapping)
    return analyzer.analyze_seasonal_missing_patterns(missing_analysis, exclude_full_days)


def get_diurnal_summary(missing_analysis: Dict) -> Dict:
    """
    Get summary of diurnal missing patterns.
    
    Parameters:
    -----------
    missing_analysis : dict
        Results from missing data analysis
        
    Returns:
    --------
    dict
        Diurnal pattern summary
    """
    analyzer = SeasonalPatternAnalyzer()
    return analyzer.analyze_diurnal_patterns(missing_analysis)


if __name__ == "__main__":
    # Example usage
    print("Seasonal Pattern Analysis Module")
    print("Use SeasonalPatternAnalyzer class or convenience functions")
