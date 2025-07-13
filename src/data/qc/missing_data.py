"""
Missing Data Analysis Module

This module provides tools for analyzing missing data patterns in time series data,
particularly for aethalometer measurements. It identifies gaps, calculates statistics,
and categorizes missing periods.

Author: AethModular Team
Created: 2025-01-12
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MissingDataAnalyzer:
    """
    Analyzer for missing data patterns in time series data.
    
    This class provides methods to:
    - Identify missing timestamps in expected timeline
    - Calculate missing data statistics
    - Categorize missing periods (full days vs partial)
    - Analyze temporal patterns of missingness
    """
    
    def __init__(self):
        """Initialize the missing data analyzer."""
        self.results = None
        
    def analyze_missing_patterns(self, df: pd.DataFrame, freq: str = 'min') -> Dict:
        """
        Comprehensive analysis of missing data patterns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with datetime index
        freq : str, default 'min'
            Expected frequency of the data (e.g., 'min', 'H', 'D')
            
        Returns:
        --------
        dict
            Dictionary containing comprehensive missing data analysis:
            - timeline: Basic timeline statistics
            - daily_patterns: Daily missing data patterns
            - temporal_patterns: Hourly and temporal patterns
            - missing_indices: Actual missing timestamps
        """
        logger.info("Starting comprehensive missing data analysis...")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        
        # Build expected timeline
        start, end = df.index.min(), df.index.max()
        expected_idx = pd.date_range(start, end, freq=freq)
        actual_idx = df.index.unique().sort_values()
        missing_idx = expected_idx.difference(actual_idx)
        
        # Calculate basic statistics
        missing_count = len(missing_idx)
        missing_pct = (missing_count / len(expected_idx)) * 100
        
        # Analyze daily patterns
        daily_analysis = self._analyze_daily_patterns(missing_idx, freq)
        
        # Analyze temporal patterns
        temporal_analysis = self._analyze_temporal_patterns(missing_idx, expected_idx, actual_idx)
        
        # Store results
        self.results = {
            'timeline': {
                'start': start,
                'end': end,
                'duration_days': (end - start).days + 1,
                'expected_points': len(expected_idx),
                'actual_points': len(actual_idx),
                'missing_points': missing_count,
                'missing_percentage': missing_pct,
                'frequency': freq
            },
            'daily_patterns': daily_analysis,
            'temporal_patterns': temporal_analysis,
            'missing_indices': missing_idx
        }
        
        logger.info(f"Missing data analysis complete. {missing_pct:.2f}% data missing.")
        return self.results
    
    def _analyze_daily_patterns(self, missing_idx: pd.DatetimeIndex, freq: str) -> Dict:
        """
        Analyze missing data patterns by day.
        
        Parameters:
        -----------
        missing_idx : pd.DatetimeIndex
            Index of missing timestamps
        freq : str
            Data frequency
            
        Returns:
        --------
        dict
            Daily pattern analysis results
        """
        if len(missing_idx) == 0:
            return {
                'missing_per_day': pd.Series(dtype=int),
                'full_days_missing': pd.Series(dtype=int),
                'partial_missing_days': pd.Series(dtype=int),
                'n_full_missing_days': 0,
                'n_partial_missing_days': 0
            }
        
        # Count missing points per day
        missing_per_day = pd.Series(1, index=missing_idx).groupby(missing_idx.date).count()
        
        # Determine expected points per day based on frequency
        if freq == 'min':
            expected_per_day = 1440  # 24 * 60 minutes
        elif freq == 'H':
            expected_per_day = 24   # 24 hours
        elif freq == 'D':
            expected_per_day = 1    # 1 day
        else:
            # Try to estimate based on frequency string
            try:
                freq_minutes = pd.Timedelta(freq).total_seconds() / 60
                expected_per_day = int(1440 / freq_minutes)
            except:
                logger.warning(f"Unknown frequency {freq}, assuming minute data")
                expected_per_day = 1440
        
        # Categorize missing days
        full_days_missing = missing_per_day[missing_per_day >= expected_per_day]
        partial_missing = missing_per_day[missing_per_day < expected_per_day]
        
        return {
            'missing_per_day': missing_per_day,
            'full_days_missing': full_days_missing,
            'partial_missing_days': partial_missing,
            'n_full_missing_days': len(full_days_missing),
            'n_partial_missing_days': len(partial_missing),
            'expected_per_day': expected_per_day
        }
    
    def _analyze_temporal_patterns(self, missing_idx: pd.DatetimeIndex, 
                                 expected_idx: pd.DatetimeIndex,
                                 actual_idx: pd.DatetimeIndex) -> Dict:
        """
        Analyze temporal patterns of missing data.
        
        Parameters:
        -----------
        missing_idx : pd.DatetimeIndex
            Missing timestamps
        expected_idx : pd.DatetimeIndex
            Expected complete timeline
        actual_idx : pd.DatetimeIndex
            Actual data timestamps
            
        Returns:
        --------
        dict
            Temporal pattern analysis
        """
        if len(missing_idx) == 0:
            return {
                'missing_per_hour': pd.Series(dtype=int),
                'missing_per_weekday': pd.Series(dtype=int),
                'missing_per_month': pd.Series(dtype=int),
                'missing_idx': missing_idx,
                'expected_idx': expected_idx,
                'actual_idx': actual_idx
            }
        
        # Hourly patterns
        missing_per_hour = pd.Series(1, index=missing_idx).groupby(missing_idx.hour).count()
        
        # Weekly patterns
        missing_per_weekday = pd.Series(1, index=missing_idx).groupby(missing_idx.dayofweek).count()
        
        # Monthly patterns
        missing_per_month = pd.Series(1, index=missing_idx).groupby(missing_idx.month).count()
        
        return {
            'missing_per_hour': missing_per_hour,
            'missing_per_weekday': missing_per_weekday,
            'missing_per_month': missing_per_month,
            'missing_idx': missing_idx,
            'expected_idx': expected_idx,
            'actual_idx': actual_idx
        }
    
    def get_missing_periods(self, min_duration: Optional[str] = None) -> pd.DataFrame:
        """
        Get continuous missing periods.
        
        Parameters:
        -----------
        min_duration : str, optional
            Minimum duration to include (e.g., '1H', '30min')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with start, end, and duration of missing periods
        """
        if self.results is None:
            raise ValueError("Must run analyze_missing_patterns() first")
        
        missing_idx = self.results['missing_indices']
        
        if len(missing_idx) == 0:
            return pd.DataFrame(columns=['start', 'end', 'duration', 'n_points'])
        
        # Find continuous periods
        periods = []
        current_start = missing_idx[0]
        current_end = missing_idx[0]
        
        for i in range(1, len(missing_idx)):
            # Check if this timestamp continues the current period
            expected_next = current_end + pd.Timedelta(self.results['timeline']['frequency'])
            
            if missing_idx[i] <= expected_next + pd.Timedelta('1min'):  # Allow small tolerance
                current_end = missing_idx[i]
            else:
                # End current period, start new one
                periods.append({
                    'start': current_start,
                    'end': current_end,
                    'duration': current_end - current_start,
                    'n_points': len(missing_idx[(missing_idx >= current_start) & (missing_idx <= current_end)])
                })
                current_start = missing_idx[i]
                current_end = missing_idx[i]
        
        # Add final period
        periods.append({
            'start': current_start,
            'end': current_end,
            'duration': current_end - current_start,
            'n_points': len(missing_idx[(missing_idx >= current_start) & (missing_idx <= current_end)])
        })
        
        periods_df = pd.DataFrame(periods)
        
        # Filter by minimum duration if specified
        if min_duration:
            min_td = pd.Timedelta(min_duration)
            periods_df = periods_df[periods_df['duration'] >= min_td]
        
        return periods_df.sort_values('start')
    
    def exclude_full_missing_days(self, missing_idx: Optional[pd.DatetimeIndex] = None) -> pd.DatetimeIndex:
        """
        Remove timestamps from fully missing days to focus on partial missing patterns.
        
        Parameters:
        -----------
        missing_idx : pd.DatetimeIndex, optional
            Missing timestamps. If None, uses stored results.
            
        Returns:
        --------
        pd.DatetimeIndex
            Missing timestamps excluding full missing days
        """
        if missing_idx is None:
            if self.results is None:
                raise ValueError("Must run analyze_missing_patterns() first or provide missing_idx")
            missing_idx = self.results['missing_indices']
        
        if self.results is None:
            raise ValueError("Must run analyze_missing_patterns() first to get daily patterns")
        
        # Get full missing days
        full_days_missing = self.results['daily_patterns']['full_days_missing'].index
        
        # Filter out missing timestamps from full missing days
        partial_missing_idx = missing_idx[~missing_idx.normalize().isin(full_days_missing)]
        
        logger.info(f"Excluded {len(missing_idx) - len(partial_missing_idx)} timestamps from {len(full_days_missing)} full missing days")
        
        return partial_missing_idx
    
    def summary_stats(self) -> Dict:
        """
        Get summary statistics of missing data analysis.
        
        Returns:
        --------
        dict
            Summary statistics
        """
        if self.results is None:
            raise ValueError("Must run analyze_missing_patterns() first")
        
        timeline = self.results['timeline']
        daily = self.results['daily_patterns']
        
        partial_missing = daily['partial_missing_days']
        
        summary = {
            'total_timespan_days': timeline['duration_days'],
            'missing_percentage': timeline['missing_percentage'],
            'full_missing_days': daily['n_full_missing_days'],
            'partial_missing_days': daily['n_partial_missing_days'],
            'total_missing_points': timeline['missing_points']
        }
        
        if len(partial_missing) > 0:
            summary.update({
                'partial_missing_stats': {
                    'mean_missing_per_day': partial_missing.mean(),
                    'median_missing_per_day': partial_missing.median(),
                    'max_missing_per_day': partial_missing.max(),
                    'std_missing_per_day': partial_missing.std()
                }
            })
        
        return summary


def analyze_missing_patterns(df: pd.DataFrame, freq: str = 'min') -> Dict:
    """
    Convenience function for missing data analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    freq : str, default 'min'
        Expected data frequency
        
    Returns:
    --------
    dict
        Missing data analysis results
    """
    analyzer = MissingDataAnalyzer()
    return analyzer.analyze_missing_patterns(df, freq)


def get_data_gaps(df: pd.DataFrame, freq: str = 'min', min_duration: str = '1H') -> pd.DataFrame:
    """
    Get significant data gaps in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    freq : str, default 'min'
        Expected data frequency
    min_duration : str, default '1H'
        Minimum gap duration to include
        
    Returns:
    --------
    pd.DataFrame
        DataFrame of significant gaps
    """
    analyzer = MissingDataAnalyzer()
    analyzer.analyze_missing_patterns(df, freq)
    return analyzer.get_missing_periods(min_duration)


if __name__ == "__main__":
    # Example usage
    print("Missing Data Analysis Module")
    print("Use MissingDataAnalyzer class or convenience functions")
