"""Data completeness analysis for aethalometer data"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ...core.base import BaseAnalyzer


class CompletenessAnalyzer(BaseAnalyzer):
    """
    Analyzer for data completeness and quality classification
    
    Provides functionality to analyze missing data patterns,
    classify data quality periods, and generate completeness reports.
    """
    
    def __init__(self):
        super().__init__("CompletenessAnalyzer")
        
        # Quality classification thresholds (missing minutes per day/period)
        self.quality_thresholds = {
            'Excellent': 10,      # ≤ 10 missing minutes
            'Good': 60,           # ≤ 60 missing minutes (1 hour)
            'Moderate': 240,      # ≤ 240 missing minutes (4 hours)
            'Poor': float('inf') # > 240 missing minutes
        }
    
    def analyze_completeness(self, data: pd.DataFrame, 
                           period_type: str = 'daily') -> Dict[str, Any]:
        """
        Analyze data completeness for the given dataset
        
        Parameters:
        -----------
        data : pd.DataFrame
            Aethalometer data with datetime index
        period_type : str
            Analysis period ('daily' or '9am_to_9am')
            
        Returns:
        --------
        Dict[str, Any]
            Completeness analysis results
        """
        print(f"🔍 Analyzing data completeness ({period_type} periods)...")
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime_local' in data.columns:
                data = data.set_index('datetime_local')
            else:
                raise ValueError("Data must have datetime index or 'datetime_local' column")
        
        # Get expected timeline
        start_time, end_time = data.index.min(), data.index.max()
        expected_timeline = pd.date_range(start_time, end_time, freq='min')
        
        # Identify missing timestamps
        actual_timestamps = data.index.unique().sort_values()
        missing_timestamps = expected_timeline.difference(actual_timestamps)
        
        # Calculate basic statistics
        total_expected = len(expected_timeline)
        total_actual = len(actual_timestamps)
        total_missing = len(missing_timestamps)
        
        print(f"  • Expected points: {total_expected:,}")
        print(f"  • Actual points: {total_actual:,}")
        print(f"  • Missing points: {total_missing:,} ({total_missing/total_expected*100:.2f}%)")
        
        # Analyze by periods
        if period_type == 'daily':
            missing_per_period = self._analyze_daily_missing(missing_timestamps)
            period_quality = self._classify_daily_quality(missing_per_period)
        elif period_type == '9am_to_9am':
            missing_per_period = self._analyze_9am_missing(missing_timestamps)
            period_quality = self._classify_9am_quality(missing_per_period)
        else:
            raise ValueError("period_type must be 'daily' or '9am_to_9am'")
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(missing_timestamps)
        
        # Generate column-level analysis
        column_completeness = self._analyze_column_completeness(data)
        
        results = {
            'period_type': period_type,
            'analysis_summary': {
                'total_expected_points': total_expected,
                'total_actual_points': total_actual,
                'total_missing_points': total_missing,
                'overall_completeness_percent': (total_actual / total_expected) * 100,
                'analysis_period': {
                    'start': start_time,
                    'end': end_time,
                    'duration_days': (end_time - start_time).days + 1
                }
            },
            'period_analysis': {
                'missing_per_period': missing_per_period,
                'quality_classification': period_quality
            },
            'temporal_patterns': temporal_patterns,
            'column_completeness': column_completeness
        }
        
        return results
    
    def _analyze_daily_missing(self, missing_timestamps: pd.DatetimeIndex) -> pd.Series:
        """Analyze missing data by calendar day (midnight to midnight)"""
        if len(missing_timestamps) == 0:
            return pd.Series(dtype=int)
        
        # Group by date
        missing_per_day = pd.Series(1, index=missing_timestamps).groupby(
            missing_timestamps.date
        ).count()
        
        return missing_per_day
    
    def _analyze_9am_missing(self, missing_timestamps: pd.DatetimeIndex) -> pd.Series:
        """Analyze missing data by 9AM-to-9AM periods"""
        if len(missing_timestamps) == 0:
            return pd.Series(dtype=int)
        
        # Map each timestamp to its 9AM period start
        period_starts = missing_timestamps.map(lambda ts: 
            ts.normalize() + pd.Timedelta(hours=9) if ts.hour >= 9 
            else ts.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=9)
        )
        
        missing_per_period = pd.Series(1, index=period_starts).groupby(level=0).count()
        
        return missing_per_period
    
    def _classify_daily_quality(self, missing_per_day: pd.Series) -> pd.Series:
        """Classify quality for daily periods"""
        return missing_per_day.apply(self._classify_period_quality)
    
    def _classify_9am_quality(self, missing_per_period: pd.Series) -> pd.Series:
        """Classify quality for 9AM-to-9AM periods"""
        return missing_per_period.apply(self._classify_period_quality)
    
    def _classify_period_quality(self, missing_minutes: int) -> str:
        """Classify a single period's quality based on missing minutes"""
        for quality, threshold in self.quality_thresholds.items():
            if missing_minutes <= threshold:
                return quality
        return 'Poor'  # Fallback
    
    def _analyze_temporal_patterns(self, missing_timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """Analyze temporal patterns in missing data"""
        if len(missing_timestamps) == 0:
            return {
                'missing_by_hour': pd.Series(dtype=int),
                'missing_by_month': pd.Series(dtype=int),
                'missing_by_weekday': pd.Series(dtype=int),
                'largest_gaps': []
            }
        
        # Missing by hour of day
        missing_by_hour = pd.Series(1, index=missing_timestamps).groupby(
            missing_timestamps.hour
        ).count()
        
        # Missing by month
        missing_by_month = pd.Series(1, index=missing_timestamps).groupby(
            missing_timestamps.month
        ).count()
        
        # Missing by weekday
        missing_by_weekday = pd.Series(1, index=missing_timestamps).groupby(
            missing_timestamps.dayofweek
        ).count()
        
        # Find largest gaps
        largest_gaps = self._find_largest_gaps(missing_timestamps)
        
        return {
            'missing_by_hour': missing_by_hour,
            'missing_by_month': missing_by_month,
            'missing_by_weekday': missing_by_weekday,
            'largest_gaps': largest_gaps
        }
    
    def _find_largest_gaps(self, missing_timestamps: pd.DatetimeIndex, 
                          top_n: int = 10) -> list:
        """Find the largest continuous gaps in data"""
        if len(missing_timestamps) == 0:
            return []
        
        # Sort timestamps
        sorted_missing = missing_timestamps.sort_values()
        
        # Find continuous gaps (sequences of consecutive minutes)
        gaps = []
        gap_start = sorted_missing[0]
        gap_end = sorted_missing[0]
        
        for i in range(1, len(sorted_missing)):
            current_time = sorted_missing[i]
            expected_time = gap_end + pd.Timedelta(minutes=1)
            
            if current_time == expected_time:
                # Continue current gap
                gap_end = current_time
            else:
                # End current gap and start new one
                gap_duration = (gap_end - gap_start).total_seconds() / 60 + 1  # +1 for inclusive
                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration_minutes': int(gap_duration)
                })
                gap_start = current_time
                gap_end = current_time
        
        # Don't forget the last gap
        gap_duration = (gap_end - gap_start).total_seconds() / 60 + 1
        gaps.append({
            'start': gap_start,
            'end': gap_end,
            'duration_minutes': int(gap_duration)
        })
        
        # Sort by duration and return top N
        gaps_sorted = sorted(gaps, key=lambda x: x['duration_minutes'], reverse=True)
        return gaps_sorted[:top_n]
    
    def _analyze_column_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze completeness for each column"""
        bc_columns = [col for col in data.columns if col.endswith('BCc')]
        atn_columns = [col for col in data.columns if col.endswith('ATN1')]
        
        column_stats = {}
        
        for col in bc_columns + atn_columns:
            if col in data.columns:
                total_rows = len(data)
                missing_count = data[col].isna().sum()
                completeness_pct = ((total_rows - missing_count) / total_rows) * 100
                
                column_stats[col] = {
                    'total_rows': total_rows,
                    'missing_count': int(missing_count),
                    'completeness_percent': round(completeness_pct, 2)
                }
        
        return column_stats
    
    def get_quality_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of quality classification results"""
        period_quality = analysis_results['period_analysis']['quality_classification']
        
        if len(period_quality) == 0:
            return {'total_periods': 0}
        
        quality_counts = period_quality.value_counts()
        total_periods = len(period_quality)
        
        summary = {
            'total_periods': total_periods,
            'quality_distribution': {}
        }
        
        for quality in ['Excellent', 'Good', 'Moderate', 'Poor']:
            count = quality_counts.get(quality, 0)
            percentage = (count / total_periods) * 100 if total_periods > 0 else 0
            summary['quality_distribution'][quality] = {
                'count': int(count),
                'percentage': round(percentage, 1)
            }
        
        return summary
    
    def print_completeness_report(self, analysis_results: Dict[str, Any]):
        """Print a formatted completeness report"""
        print("\n" + "="*80)
        print("DATA COMPLETENESS ANALYSIS REPORT")
        print("="*80)
        
        # Summary statistics
        summary = analysis_results['analysis_summary']
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  • Analysis Period: {summary['analysis_period']['start'].date()} to {summary['analysis_period']['end'].date()}")
        print(f"  • Duration: {summary['analysis_period']['duration_days']} days")
        print(f"  • Expected Data Points: {summary['total_expected_points']:,}")
        print(f"  • Actual Data Points: {summary['total_actual_points']:,}")
        print(f"  • Missing Data Points: {summary['total_missing_points']:,}")
        print(f"  • Overall Completeness: {summary['overall_completeness_percent']:.2f}%")
        
        # Quality distribution
        quality_summary = self.get_quality_summary(analysis_results)
        if quality_summary['total_periods'] > 0:
            print(f"\n🏆 QUALITY DISTRIBUTION ({analysis_results['period_type']} periods):")
            for quality, stats in quality_summary['quality_distribution'].items():
                print(f"  • {quality}: {stats['count']} periods ({stats['percentage']:.1f}%)")
        
        # Temporal patterns
        patterns = analysis_results['temporal_patterns']
        if len(patterns['largest_gaps']) > 0:
            print(f"\n⏱️  LARGEST DATA GAPS:")
            for i, gap in enumerate(patterns['largest_gaps'][:5], 1):
                print(f"  {i}. {gap['start']} to {gap['end']} ({gap['duration_minutes']} minutes)")
        
        # Column completeness
        col_completeness = analysis_results['column_completeness']
        if col_completeness:
            print(f"\n📋 COLUMN COMPLETENESS:")
            for col, stats in col_completeness.items():
                print(f"  • {col}: {stats['completeness_percent']:.1f}% ({stats['missing_count']:,} missing)")
        
        print("\n" + "="*80)
