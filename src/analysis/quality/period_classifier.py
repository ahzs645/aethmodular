"""Period quality classifier based on data completeness"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from ...core.base import BaseAnalyzer


class PeriodClassifier(BaseAnalyzer):
    """
    Classifier for data quality periods based on completeness and other criteria
    
    Provides advanced classification logic beyond simple missing data counts,
    including pattern recognition and quality trend analysis.
    """
    
    def __init__(self, custom_thresholds: Optional[Dict[str, int]] = None):
        super().__init__("PeriodClassifier")
        
        # Default quality thresholds (missing minutes)
        self.thresholds = custom_thresholds or {
            'Excellent': 10,      # ≤ 10 missing minutes
            'Good': 60,           # ≤ 60 missing minutes  
            'Moderate': 240,      # ≤ 240 missing minutes
            'Poor': float('inf') # > 240 missing minutes
        }
        
        # Additional quality factors
        self.quality_factors = {
            'max_consecutive_missing': 30,  # Max consecutive missing minutes for Good
            'gap_penalty_threshold': 5,     # Penalize periods with many small gaps
            'weekend_tolerance': 1.5        # Weekend periods get more tolerance
        }
    
    def classify_periods(self, data: pd.DataFrame, 
                        period_type: str = '9am_to_9am',
                        include_quality_factors: bool = True) -> Dict[str, Any]:
        """
        Classify data quality for analysis periods
        
        Parameters:
        -----------
        data : pd.DataFrame
            Aethalometer data with datetime index
        period_type : str
            Period type ('daily' or '9am_to_9am')
        include_quality_factors : bool
            Whether to include advanced quality factors beyond missing data
            
        Returns:
        --------
        Dict[str, Any]
            Classification results with quality labels and metadata
        """
        print(f"🏷️  Classifying {period_type} periods...")
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime_local' in data.columns:
                data = data.set_index('datetime_local')
            else:
                raise ValueError("Data must have datetime index")
        
        # Generate expected timeline and find missing data
        start_time, end_time = data.index.min(), data.index.max()
        expected_timeline = pd.date_range(start_time, end_time, freq='min')
        actual_timestamps = data.index.unique().sort_values()
        missing_timestamps = expected_timeline.difference(actual_timestamps)
        
        # Classify periods based on type
        if period_type == 'daily':
            classifications = self._classify_daily_periods(
                missing_timestamps, start_time, end_time, include_quality_factors
            )
        elif period_type == '9am_to_9am':
            classifications = self._classify_9am_periods(
                missing_timestamps, start_time, end_time, include_quality_factors
            )
        else:
            raise ValueError("period_type must be 'daily' or '9am_to_9am'")
        
        # Generate summary statistics
        summary = self._generate_classification_summary(classifications)
        
        results = {
            'period_type': period_type,
            'classifications': classifications,
            'summary': summary,
            'thresholds_used': self.thresholds.copy(),
            'quality_factors_applied': include_quality_factors
        }
        
        return results
    
    def _classify_daily_periods(self, missing_timestamps: pd.DatetimeIndex,
                               start_time: pd.Timestamp, end_time: pd.Timestamp,
                               include_quality_factors: bool) -> pd.DataFrame:
        """Classify daily periods (midnight to midnight)"""
        # Generate all dates in range
        date_range = pd.date_range(start_time.date(), end_time.date(), freq='D')
        
        classifications = []
        
        for date in date_range:
            period_start = pd.Timestamp(date)
            period_end = period_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
            
            # Find missing data in this period
            period_missing = missing_timestamps[
                (missing_timestamps >= period_start) & 
                (missing_timestamps <= period_end)
            ]
            
            # Basic classification
            missing_count = len(period_missing)
            base_quality = self._get_base_quality(missing_count)
            
            # Advanced quality factors
            if include_quality_factors and len(period_missing) > 0:
                quality_adjustments = self._analyze_quality_factors(
                    period_missing, period_start, period_end
                )
                final_quality = self._adjust_quality(base_quality, quality_adjustments)
            else:
                quality_adjustments = {}
                final_quality = base_quality
            
            classifications.append({
                'period_start': period_start,
                'period_end': period_end,
                'missing_minutes': missing_count,
                'base_quality': base_quality,
                'final_quality': final_quality,
                'quality_adjustments': quality_adjustments,
                'is_weekend': period_start.weekday() >= 5
            })
        
        return pd.DataFrame(classifications)
    
    def _classify_9am_periods(self, missing_timestamps: pd.DatetimeIndex,
                             start_time: pd.Timestamp, end_time: pd.Timestamp,
                             include_quality_factors: bool) -> pd.DataFrame:
        """Classify 9AM-to-9AM periods"""
        # Generate 9AM period starts
        first_9am = start_time.normalize() + pd.Timedelta(hours=9)
        if start_time.hour < 9:
            first_9am -= pd.Timedelta(days=1)
        
        last_9am = end_time.normalize() + pd.Timedelta(hours=9)
        if end_time.hour < 9:
            last_9am -= pd.Timedelta(days=1)
        
        period_starts = pd.date_range(first_9am, last_9am, freq='D')
        
        classifications = []
        
        for period_start in period_starts:
            period_end = period_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
            
            # Find missing data in this 9AM-to-9AM period
            period_missing = missing_timestamps[
                (missing_timestamps >= period_start) & 
                (missing_timestamps <= period_end)
            ]
            
            # Basic classification
            missing_count = len(period_missing)
            base_quality = self._get_base_quality(missing_count)
            
            # Advanced quality factors
            if include_quality_factors and len(period_missing) > 0:
                quality_adjustments = self._analyze_quality_factors(
                    period_missing, period_start, period_end
                )
                final_quality = self._adjust_quality(base_quality, quality_adjustments)
            else:
                quality_adjustments = {}
                final_quality = base_quality
            
            classifications.append({
                'period_start': period_start,
                'period_end': period_end,
                'missing_minutes': missing_count,
                'base_quality': base_quality,
                'final_quality': final_quality,
                'quality_adjustments': quality_adjustments,
                'is_weekend': period_start.weekday() >= 5
            })
        
        return pd.DataFrame(classifications)
    
    def _get_base_quality(self, missing_minutes: int) -> str:
        """Get base quality classification from missing minutes"""
        for quality, threshold in self.thresholds.items():
            if missing_minutes <= threshold:
                return quality
        return 'Poor'
    
    def _analyze_quality_factors(self, missing_timestamps: pd.DatetimeIndex,
                                period_start: pd.Timestamp, 
                                period_end: pd.Timestamp) -> Dict[str, Any]:
        """Analyze additional quality factors beyond simple missing count"""
        factors = {}
        
        if len(missing_timestamps) == 0:
            return factors
        
        # 1. Maximum consecutive missing minutes
        consecutive_gaps = self._find_consecutive_gaps(missing_timestamps)
        max_consecutive = max([gap['duration'] for gap in consecutive_gaps]) if consecutive_gaps else 0
        factors['max_consecutive_missing'] = max_consecutive
        
        # 2. Number of separate gaps
        factors['number_of_gaps'] = len(consecutive_gaps)
        
        # 3. Gap distribution pattern
        if len(consecutive_gaps) > 0:
            gap_durations = [gap['duration'] for gap in consecutive_gaps]
            factors['gap_pattern'] = {
                'mean_gap_duration': np.mean(gap_durations),
                'std_gap_duration': np.std(gap_durations),
                'small_gaps_count': sum(1 for d in gap_durations if d <= 5),
                'large_gaps_count': sum(1 for d in gap_durations if d > 30)
            }
        
        # 4. Weekend adjustment factor
        is_weekend = period_start.weekday() >= 5
        factors['is_weekend'] = is_weekend
        
        return factors
    
    def _find_consecutive_gaps(self, missing_timestamps: pd.DatetimeIndex) -> List[Dict[str, Any]]:
        """Find consecutive gaps in missing timestamps"""
        if len(missing_timestamps) == 0:
            return []
        
        sorted_missing = missing_timestamps.sort_values()
        gaps = []
        
        gap_start = sorted_missing[0]
        gap_end = sorted_missing[0]
        
        for i in range(1, len(sorted_missing)):
            current_time = sorted_missing[i]
            expected_time = gap_end + pd.Timedelta(minutes=1)
            
            if current_time == expected_time:
                gap_end = current_time
            else:
                # End current gap
                duration = int((gap_end - gap_start).total_seconds() / 60) + 1
                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration': duration
                })
                gap_start = current_time
                gap_end = current_time
        
        # Add final gap
        duration = int((gap_end - gap_start).total_seconds() / 60) + 1
        gaps.append({
            'start': gap_start,
            'end': gap_end,
            'duration': duration
        })
        
        return gaps
    
    def _adjust_quality(self, base_quality: str, quality_factors: Dict[str, Any]) -> str:
        """Adjust quality classification based on additional factors"""
        if not quality_factors:
            return base_quality
        
        # Define quality levels for easier manipulation
        quality_levels = ['Excellent', 'Good', 'Moderate', 'Poor']
        current_level = quality_levels.index(base_quality)
        
        adjustments = 0
        
        # Penalize for large consecutive gaps
        max_consecutive = quality_factors.get('max_consecutive_missing', 0)
        if max_consecutive > self.quality_factors['max_consecutive_missing']:
            adjustments += 1  # Downgrade one level
        
        # Penalize for many small gaps (indicates intermittent issues)
        if 'gap_pattern' in quality_factors:
            small_gaps = quality_factors['gap_pattern']['small_gaps_count']
            if small_gaps > self.quality_factors['gap_penalty_threshold']:
                adjustments += 1
        
        # Weekend tolerance (reduce penalties)
        if quality_factors.get('is_weekend', False):
            adjustments = max(0, adjustments - 1)
        
        # Apply adjustments
        final_level = min(len(quality_levels) - 1, current_level + adjustments)
        return quality_levels[final_level]
    
    def _generate_classification_summary(self, classifications: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for classifications"""
        if len(classifications) == 0:
            return {'total_periods': 0}
        
        # Quality distribution
        quality_counts = classifications['final_quality'].value_counts()
        total_periods = len(classifications)
        
        # Missing data statistics
        missing_stats = {
            'mean_missing_minutes': classifications['missing_minutes'].mean(),
            'median_missing_minutes': classifications['missing_minutes'].median(),
            'max_missing_minutes': classifications['missing_minutes'].max(),
            'periods_with_no_missing': (classifications['missing_minutes'] == 0).sum()
        }
        
        # Weekend vs weekday analysis
        weekend_analysis = {}
        if 'is_weekend' in classifications.columns:
            weekend_periods = classifications[classifications['is_weekend']]
            weekday_periods = classifications[~classifications['is_weekend']]
            
            weekend_analysis = {
                'weekend_periods': len(weekend_periods),
                'weekday_periods': len(weekday_periods),
                'weekend_quality_dist': weekend_periods['final_quality'].value_counts().to_dict() if len(weekend_periods) > 0 else {},
                'weekday_quality_dist': weekday_periods['final_quality'].value_counts().to_dict() if len(weekday_periods) > 0 else {}
            }
        
        summary = {
            'total_periods': total_periods,
            'quality_distribution': {
                quality: {
                    'count': int(quality_counts.get(quality, 0)),
                    'percentage': round((quality_counts.get(quality, 0) / total_periods) * 100, 1)
                }
                for quality in ['Excellent', 'Good', 'Moderate', 'Poor']
            },
            'missing_data_statistics': {
                k: round(v, 1) if isinstance(v, float) else int(v)
                for k, v in missing_stats.items()
            },
            'weekend_analysis': weekend_analysis
        }
        
        return summary
    
    def get_excellent_periods(self, classification_results: Dict[str, Any]) -> pd.DataFrame:
        """Get all periods classified as 'Excellent' quality"""
        classifications = classification_results['classifications']
        return classifications[classifications['final_quality'] == 'Excellent'].copy()
    
    def get_usable_periods(self, classification_results: Dict[str, Any],
                          min_quality: str = 'Good') -> pd.DataFrame:
        """
        Get all periods with quality at or above specified minimum
        
        Parameters:
        -----------
        classification_results : Dict[str, Any]
            Results from classify_periods
        min_quality : str
            Minimum quality level ('Excellent', 'Good', 'Moderate')
            
        Returns:
        --------
        pd.DataFrame
            Filtered periods meeting quality criteria
        """
        quality_hierarchy = ['Excellent', 'Good', 'Moderate', 'Poor']
        min_level = quality_hierarchy.index(min_quality)
        usable_qualities = quality_hierarchy[:min_level + 1]
        
        classifications = classification_results['classifications']
        return classifications[classifications['final_quality'].isin(usable_qualities)].copy()
    
    def print_classification_report(self, classification_results: Dict[str, Any]):
        """Print a formatted classification report"""
        print("\n" + "="*80)
        print("PERIOD QUALITY CLASSIFICATION REPORT")
        print("="*80)
        
        summary = classification_results['summary']
        period_type = classification_results['period_type']
        
        print(f"\n📊 CLASSIFICATION SUMMARY ({period_type} periods):")
        print(f"  • Total periods analyzed: {summary['total_periods']}")
        
        # Quality distribution
        print(f"\n🏆 QUALITY DISTRIBUTION:")
        for quality, stats in summary['quality_distribution'].items():
            print(f"  • {quality}: {stats['count']} periods ({stats['percentage']:.1f}%)")
        
        # Missing data statistics
        missing_stats = summary['missing_data_statistics']
        print(f"\n📉 MISSING DATA STATISTICS:")
        print(f"  • Mean missing minutes per period: {missing_stats['mean_missing_minutes']:.1f}")
        print(f"  • Median missing minutes per period: {missing_stats['median_missing_minutes']:.1f}")
        print(f"  • Maximum missing minutes: {missing_stats['max_missing_minutes']}")
        print(f"  • Periods with no missing data: {missing_stats['periods_with_no_missing']}")
        
        # Weekend analysis
        if summary['weekend_analysis'] and summary['weekend_analysis']['weekend_periods'] > 0:
            weekend_info = summary['weekend_analysis']
            print(f"\n📅 WEEKEND vs WEEKDAY ANALYSIS:")
            print(f"  • Weekend periods: {weekend_info['weekend_periods']}")
            print(f"  • Weekday periods: {weekend_info['weekday_periods']}")
        
        # Quality factors info
        if classification_results['quality_factors_applied']:
            print(f"\n⚙️  QUALITY FACTORS APPLIED:")
            print(f"  • Maximum consecutive missing threshold: {self.quality_factors['max_consecutive_missing']} minutes")
            print(f"  • Gap penalty threshold: {self.quality_factors['gap_penalty_threshold']} gaps")
            print(f"  • Weekend tolerance factor: {self.quality_factors['weekend_tolerance']}x")
        
        print("\n" + "="*80)
