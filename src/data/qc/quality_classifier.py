"""
Quality Classification Module

This module provides tools for classifying data quality periods based on missing data
patterns and other quality metrics. It supports different classification schemes and
time period definitions.

Author: AethModular Team
Created: 2025-01-12
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QualityClassifier:
    """
    Classifier for data quality periods based on missing data and other metrics.
    
    This class provides methods to:
    - Classify quality based on missing data thresholds
    - Support different time period definitions (daily, 9am-to-9am)
    - Apply custom quality criteria
    - Generate quality summaries and statistics
    """
    
    def __init__(self, quality_thresholds: Optional[Dict] = None):
        """
        Initialize the quality classifier.
        
        Parameters:
        -----------
        quality_thresholds : dict, optional
            Dictionary defining quality thresholds:
            {'excellent': 10, 'good': 60, 'moderate': 240}
            Values represent maximum missing minutes for each category.
        """
        self.quality_thresholds = quality_thresholds or {
            'excellent': 10,    # ≤10 minutes missing
            'good': 60,         # 11-60 minutes missing  
            'moderate': 240,    # 61-240 minutes missing
            # >240 minutes = poor
        }
        
        self.quality_labels = ['Excellent', 'Good', 'Moderate', 'Poor']
        
    def classify_daily_periods(self, missing_analysis: Dict) -> pd.Series:
        """
        Classify data quality by daily periods (midnight to midnight).
        
        Parameters:
        -----------
        missing_analysis : dict
            Results from MissingDataAnalyzer.analyze_missing_patterns()
            
        Returns:
        --------
        pd.Series
            Series with date index and quality labels
        """
        logger.info("Classifying daily quality periods...")
        
        missing_per_day = missing_analysis['daily_patterns']['missing_per_day']
        
        # Filter out full missing days for partial classification
        expected_per_day = missing_analysis['daily_patterns']['expected_per_day']
        partial_missing = missing_per_day[missing_per_day < expected_per_day]
        
        if len(partial_missing) == 0:
            logger.warning("No partial missing days found for classification")
            return pd.Series(dtype='object')
        
        quality_series = partial_missing.apply(self._classify_quality_by_missing)
        
        # Convert index to datetime for consistency
        quality_series.index = pd.to_datetime(quality_series.index)
        
        self._log_quality_distribution(quality_series, "daily")
        return quality_series
    
    def classify_9am_to_9am_periods(self, missing_analysis: Dict) -> pd.Series:
        """
        Classify data quality by 9am-to-9am periods (filter sampling periods).
        
        Parameters:
        -----------
        missing_analysis : dict
            Results from MissingDataAnalyzer.analyze_missing_patterns()
            
        Returns:
        --------
        pd.Series
            Series with datetime index (9am start times) and quality labels
        """
        logger.info("Classifying 9am-to-9am quality periods...")
        
        missing_idx = missing_analysis['missing_indices']
        
        if len(missing_idx) == 0:
            logger.info("No missing data found - all periods are Excellent")
            # Create a series covering the full period with Excellent quality
            start = missing_analysis['timeline']['start']
            end = missing_analysis['timeline']['end']
            nine_am_starts = pd.date_range(
                start.normalize() + pd.Timedelta(hours=9),
                end.normalize() + pd.Timedelta(hours=9),
                freq='D'
            )
            return pd.Series('Excellent', index=nine_am_starts)
        
        # Map each missing timestamp to its 9am-to-9am period start
        nine_am_periods = missing_idx.map(lambda ts: 
            ts.normalize() + pd.Timedelta(hours=9) if ts.hour < 9 
            else ts.normalize() + pd.Timedelta(hours=9) + pd.Timedelta(days=1)
        )
        
        # Count missing minutes per 9am-to-9am period
        missing_per_period = pd.Series(1, index=nine_am_periods).groupby(level=0).count()
        
        # Filter out full missing periods (1440 minutes = 24 hours)
        partial_missing = missing_per_period[missing_per_period < 1440]
        
        if len(partial_missing) == 0:
            logger.warning("No partial missing periods found for classification")
            return pd.Series(dtype='object')
        
        quality_series = partial_missing.apply(self._classify_quality_by_missing)
        
        # Ensure datetime index
        quality_series.index = pd.DatetimeIndex(quality_series.index)
        quality_series = quality_series.sort_index()
        
        self._log_quality_distribution(quality_series, "9am-to-9am")
        return quality_series
    
    def _classify_quality_by_missing(self, missing_minutes: int) -> str:
        """
        Classify quality based on missing minutes.
        
        Parameters:
        -----------
        missing_minutes : int
            Number of missing minutes
            
        Returns:
        --------
        str
            Quality label
        """
        if missing_minutes <= self.quality_thresholds['excellent']:
            return 'Excellent'
        elif missing_minutes <= self.quality_thresholds['good']:
            return 'Good'
        elif missing_minutes <= self.quality_thresholds['moderate']:
            return 'Moderate'
        else:
            return 'Poor'
    
    def _log_quality_distribution(self, quality_series: pd.Series, period_type: str) -> None:
        """Log quality distribution statistics."""
        quality_counts = quality_series.value_counts()
        logger.info(f"{period_type.title()} quality distribution:")
        for quality in self.quality_labels:
            count = quality_counts.get(quality, 0)
            pct = count / len(quality_series) * 100 if len(quality_series) > 0 else 0
            logger.info(f"  {quality}: {count} periods ({pct:.1f}%)")
    
    def get_high_quality_periods(self, quality_series: pd.Series, 
                                levels: Optional[list] = None) -> pd.DatetimeIndex:
        """
        Get periods with high quality data.
        
        Parameters:
        -----------
        quality_series : pd.Series
            Quality classification series
        levels : list, optional
            Quality levels to include. Default is ['Excellent', 'Good']
            
        Returns:
        --------
        pd.DatetimeIndex
            Index of high quality periods
        """
        if levels is None:
            levels = ['Excellent', 'Good']
        
        high_quality_mask = quality_series.isin(levels)
        return quality_series[high_quality_mask].index
    
    def get_quality_summary(self, quality_series: pd.Series) -> Dict:
        """
        Get comprehensive quality summary statistics.
        
        Parameters:
        -----------
        quality_series : pd.Series
            Quality classification series
            
        Returns:
        --------
        dict
            Summary statistics
        """
        quality_counts = quality_series.value_counts()
        total_periods = len(quality_series)
        
        summary = {
            'total_periods': total_periods,
            'quality_distribution': quality_counts.to_dict(),
            'quality_percentages': (quality_counts / total_periods * 100).round(1).to_dict(),
            'high_quality_count': quality_counts.get('Excellent', 0) + quality_counts.get('Good', 0),
            'usable_percentage': ((quality_counts.get('Excellent', 0) + quality_counts.get('Good', 0)) / total_periods * 100).round(1) if total_periods > 0 else 0
        }
        
        # Add period information
        if len(quality_series) > 0:
            summary.update({
                'period_start': quality_series.index.min(),
                'period_end': quality_series.index.max(),
                'period_span_days': (quality_series.index.max() - quality_series.index.min()).days
            })
        
        return summary
    
    def apply_custom_criteria(self, df: pd.DataFrame, 
                            quality_series: pd.Series,
                            criteria_func: callable) -> pd.Series:
        """
        Apply custom quality criteria to existing classifications.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original data for additional quality checks
        quality_series : pd.Series
            Existing quality classifications
        criteria_func : callable
            Function that takes (df_subset, current_quality) and returns new quality
            
        Returns:
        --------
        pd.Series
            Updated quality series
        """
        logger.info("Applying custom quality criteria...")
        
        updated_quality = quality_series.copy()
        
        for period_start in quality_series.index:
            # Get data for this period (assuming 24-hour periods)
            period_end = period_start + pd.Timedelta(days=1)
            period_data = df[(df.index >= period_start) & (df.index < period_end)]
            
            current_quality = quality_series.loc[period_start]
            new_quality = criteria_func(period_data, current_quality)
            
            if new_quality != current_quality:
                updated_quality.loc[period_start] = new_quality
        
        changes = (updated_quality != quality_series).sum()
        logger.info(f"Custom criteria changed {changes} period classifications")
        
        return updated_quality
    
    def filter_by_quality(self, quality_series: pd.Series, 
                         min_quality: str = 'Good') -> pd.DatetimeIndex:
        """
        Filter periods by minimum quality level.
        
        Parameters:
        -----------
        quality_series : pd.Series
            Quality classification series
        min_quality : str, default 'Good'
            Minimum quality level ('Excellent', 'Good', 'Moderate', 'Poor')
            
        Returns:
        --------
        pd.DatetimeIndex
            Index of periods meeting minimum quality
        """
        quality_hierarchy = {'Excellent': 4, 'Good': 3, 'Moderate': 2, 'Poor': 1}
        min_score = quality_hierarchy.get(min_quality, 3)
        
        valid_mask = quality_series.map(quality_hierarchy).fillna(0) >= min_score
        return quality_series[valid_mask].index


def classify_periods(missing_analysis: Dict, 
                    period_type: str = 'daily',
                    quality_thresholds: Optional[Dict] = None) -> pd.Series:
    """
    Convenience function for quality classification.
    
    Parameters:
    -----------
    missing_analysis : dict
        Results from missing data analysis
    period_type : str, default 'daily'
        Type of periods: 'daily' or '9am_to_9am'
    quality_thresholds : dict, optional
        Custom quality thresholds
        
    Returns:
    --------
    pd.Series
        Quality classification series
    """
    classifier = QualityClassifier(quality_thresholds)
    
    if period_type == 'daily':
        return classifier.classify_daily_periods(missing_analysis)
    elif period_type == '9am_to_9am':
        return classifier.classify_9am_to_9am_periods(missing_analysis)
    else:
        raise ValueError("period_type must be 'daily' or '9am_to_9am'")


def get_quality_periods(quality_series: pd.Series, 
                       quality_levels: Optional[list] = None) -> pd.DatetimeIndex:
    """
    Get periods matching specified quality levels.
    
    Parameters:
    -----------
    quality_series : pd.Series
        Quality classification series
    quality_levels : list, optional
        Quality levels to include. Default is ['Excellent', 'Good']
        
    Returns:
    --------
    pd.DatetimeIndex
        Matching periods
    """
    if quality_levels is None:
        quality_levels = ['Excellent', 'Good']
    
    return quality_series[quality_series.isin(quality_levels)].index


if __name__ == "__main__":
    # Example usage
    print("Quality Classification Module")
    print("Use QualityClassifier class or convenience functions")
