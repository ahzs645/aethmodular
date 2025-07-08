"""Missing data pattern analyzer for identifying systematic issues"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ...core.base import BaseAnalyzer


class MissingDataAnalyzer(BaseAnalyzer):
    """
    Advanced analyzer for missing data patterns and systematic issues
    
    Identifies patterns in missing data that may indicate systematic problems,
    maintenance periods, or instrument issues.
    """
    
    def __init__(self):
        super().__init__("MissingDataAnalyzer")
    
    def analyze_missing_patterns(self, data: pd.DataFrame, 
                                detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of missing data patterns
        
        Parameters:
        -----------
        data : pd.DataFrame
            Aethalometer data with datetime index
        detailed_analysis : bool
            Whether to perform detailed pattern recognition
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including patterns and recommendations
        """
        print("🔍 Analyzing missing data patterns...")
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime_local' in data.columns:
                data = data.set_index('datetime_local')
            else:
                raise ValueError("Data must have datetime index")
        
        # Generate missing data map
        missing_map = self._create_missing_data_map(data)
        
        # Identify missing periods
        missing_periods = self._identify_missing_periods(data)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(missing_periods)
        
        # Column-specific analysis
        column_patterns = self._analyze_column_patterns(data)
        
        results = {
            'missing_data_map': missing_map,
            'missing_periods': missing_periods,
            'temporal_patterns': temporal_patterns,
            'column_patterns': column_patterns
        }
        
        if detailed_analysis:
            # Advanced pattern recognition
            advanced_patterns = self._perform_advanced_pattern_analysis(missing_periods)
            systematic_issues = self._identify_systematic_issues(missing_periods, temporal_patterns)
            maintenance_periods = self._identify_potential_maintenance(missing_periods)
            
            results.update({
                'advanced_patterns': advanced_patterns,
                'systematic_issues': systematic_issues,
                'maintenance_periods': maintenance_periods,
                'recommendations': self._generate_recommendations(results)
            })
        
        return results
    
    def _create_missing_data_map(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive missing data map"""
        start_time, end_time = data.index.min(), data.index.max()
        expected_timeline = pd.date_range(start_time, end_time, freq='min')
        actual_timestamps = data.index.unique().sort_values()
        missing_timestamps = expected_timeline.difference(actual_timestamps)
        
        # Basic statistics
        total_expected = len(expected_timeline)
        total_missing = len(missing_timestamps)
        
        # Daily missing summary
        if len(missing_timestamps) > 0:
            missing_by_date = pd.Series(1, index=missing_timestamps).groupby(
                missing_timestamps.date
            ).count()
        else:
            missing_by_date = pd.Series(dtype=int)
        
        return {
            'total_expected_points': total_expected,
            'total_missing_points': total_missing,
            'missing_percentage': (total_missing / total_expected) * 100,
            'missing_timestamps': missing_timestamps,
            'missing_by_date': missing_by_date,
            'analysis_period': {
                'start': start_time,
                'end': end_time,
                'duration_days': (end_time - start_time).days + 1
            }
        }
    
    def _identify_missing_periods(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify continuous missing periods"""
        start_time, end_time = data.index.min(), data.index.max()
        expected_timeline = pd.date_range(start_time, end_time, freq='min')
        actual_timestamps = data.index.unique().sort_values()
        missing_timestamps = expected_timeline.difference(actual_timestamps)
        
        if len(missing_timestamps) == 0:
            return []
        
        # Find continuous missing periods
        missing_periods = []
        sorted_missing = missing_timestamps.sort_values()
        
        period_start = sorted_missing[0]
        period_end = sorted_missing[0]
        
        for i in range(1, len(sorted_missing)):
            current_time = sorted_missing[i]
            expected_next = period_end + pd.Timedelta(minutes=1)
            
            if current_time == expected_next:
                # Continue current period
                period_end = current_time
            else:
                # End current period and record it
                duration_minutes = int((period_end - period_start).total_seconds() / 60) + 1
                missing_periods.append({
                    'start': period_start,
                    'end': period_end,
                    'duration_minutes': duration_minutes,
                    'duration_hours': duration_minutes / 60,
                    'day_of_week': period_start.strftime('%A'),
                    'hour_of_day': period_start.hour,
                    'is_weekend': period_start.weekday() >= 5
                })
                
                # Start new period
                period_start = current_time
                period_end = current_time
        
        # Add final period
        duration_minutes = int((period_end - period_start).total_seconds() / 60) + 1
        missing_periods.append({
            'start': period_start,
            'end': period_end,
            'duration_minutes': duration_minutes,
            'duration_hours': duration_minutes / 60,
            'day_of_week': period_start.strftime('%A'),
            'hour_of_day': period_start.hour,
            'is_weekend': period_start.weekday() >= 5
        })
        
        return missing_periods
    
    def _analyze_temporal_patterns(self, missing_periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in missing data"""
        if not missing_periods:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(missing_periods)
        
        # Hour of day patterns
        hour_patterns = df.groupby('hour_of_day').agg({
            'duration_minutes': ['count', 'mean', 'sum']
        }).round(2)
        
        # Day of week patterns
        dow_patterns = df.groupby('day_of_week').agg({
            'duration_minutes': ['count', 'mean', 'sum']
        }).round(2)
        
        # Weekend vs weekday
        weekend_comparison = {
            'weekend_periods': len(df[df['is_weekend']]),
            'weekday_periods': len(df[~df['is_weekend']]),
            'weekend_total_minutes': df[df['is_weekend']]['duration_minutes'].sum(),
            'weekday_total_minutes': df[~df['is_weekend']]['duration_minutes'].sum()
        }
        
        # Duration distribution
        duration_stats = {
            'mean_duration_minutes': df['duration_minutes'].mean(),
            'median_duration_minutes': df['duration_minutes'].median(),
            'max_duration_minutes': df['duration_minutes'].max(),
            'min_duration_minutes': df['duration_minutes'].min(),
            'std_duration_minutes': df['duration_minutes'].std()
        }
        
        # Categorize by duration
        duration_categories = {
            'short_gaps_1_5min': len(df[df['duration_minutes'] <= 5]),
            'medium_gaps_6_30min': len(df[(df['duration_minutes'] > 5) & (df['duration_minutes'] <= 30)]),
            'long_gaps_31_120min': len(df[(df['duration_minutes'] > 30) & (df['duration_minutes'] <= 120)]),
            'very_long_gaps_120min_plus': len(df[df['duration_minutes'] > 120])
        }
        
        return {
            'hour_patterns': hour_patterns.to_dict() if not hour_patterns.empty else {},
            'day_of_week_patterns': dow_patterns.to_dict() if not dow_patterns.empty else {},
            'weekend_comparison': weekend_comparison,
            'duration_statistics': duration_stats,
            'duration_categories': duration_categories
        }
    
    def _analyze_column_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns for each column"""
        bc_columns = [col for col in data.columns if col.endswith('BCc')]
        atn_columns = [col for col in data.columns if col.endswith('ATN1')]
        
        column_analysis = {}
        
        for col in bc_columns + atn_columns:
            if col not in data.columns:
                continue
            
            # Basic missing statistics
            total_rows = len(data)
            missing_mask = data[col].isna()
            missing_count = missing_mask.sum()
            missing_percentage = (missing_count / total_rows) * 100
            
            # Find missing periods for this column
            missing_data = data[missing_mask]
            if len(missing_data) > 0:
                # Group consecutive missing periods
                missing_periods = self._find_column_missing_periods(data, col)
            else:
                missing_periods = []
            
            column_analysis[col] = {
                'total_missing_count': int(missing_count),
                'missing_percentage': round(missing_percentage, 2),
                'missing_periods': missing_periods,
                'largest_gap_minutes': max([p['duration_minutes'] for p in missing_periods]) if missing_periods else 0,
                'number_of_gaps': len(missing_periods)
            }
        
        return column_analysis
    
    def _find_column_missing_periods(self, data: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
        """Find missing periods for a specific column"""
        missing_mask = data[column].isna()
        missing_indices = data.index[missing_mask]
        
        if len(missing_indices) == 0:
            return []
        
        periods = []
        period_start = missing_indices[0]
        period_end = missing_indices[0]
        
        for i in range(1, len(missing_indices)):
            current_time = missing_indices[i]
            expected_next = period_end + pd.Timedelta(minutes=1)
            
            # Allow small gaps (up to 5 minutes) to be considered continuous
            if current_time <= expected_next + pd.Timedelta(minutes=5):
                period_end = current_time
            else:
                # End current period
                duration_minutes = int((period_end - period_start).total_seconds() / 60) + 1
                periods.append({
                    'start': period_start,
                    'end': period_end,
                    'duration_minutes': duration_minutes
                })
                period_start = current_time
                period_end = current_time
        
        # Add final period
        duration_minutes = int((period_end - period_start).total_seconds() / 60) + 1
        periods.append({
            'start': period_start,
            'end': period_end,
            'duration_minutes': duration_minutes
        })
        
        return periods
    
    def _perform_advanced_pattern_analysis(self, missing_periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform advanced pattern recognition on missing data"""
        if not missing_periods:
            return {}
        
        df = pd.DataFrame(missing_periods)
        
        # Detect recurring patterns
        recurring_patterns = self._detect_recurring_patterns(df)
        
        # Analyze clustering
        temporal_clustering = self._analyze_temporal_clustering(df)
        
        # Seasonal analysis
        seasonal_patterns = self._analyze_seasonal_patterns(df)
        
        return {
            'recurring_patterns': recurring_patterns,
            'temporal_clustering': temporal_clustering,
            'seasonal_patterns': seasonal_patterns
        }
    
    def _detect_recurring_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect recurring patterns in missing data"""
        patterns = {}
        
        # Same hour pattern
        hour_counts = df['hour_of_day'].value_counts()
        if len(hour_counts) > 0:
            most_common_hour = hour_counts.index[0]
            patterns['most_problematic_hour'] = {
                'hour': int(most_common_hour),
                'occurrences': int(hour_counts.iloc[0]),
                'percentage_of_gaps': round((hour_counts.iloc[0] / len(df)) * 100, 1)
            }
        
        # Same day of week pattern
        dow_counts = df['day_of_week'].value_counts()
        if len(dow_counts) > 0:
            most_common_dow = dow_counts.index[0]
            patterns['most_problematic_day'] = {
                'day': most_common_dow,
                'occurrences': int(dow_counts.iloc[0]),
                'percentage_of_gaps': round((dow_counts.iloc[0] / len(df)) * 100, 1)
            }
        
        return patterns
    
    def _analyze_temporal_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal clustering of missing periods"""
        if len(df) < 2:
            return {}
        
        # Calculate time differences between consecutive missing periods
        df_sorted = df.sort_values('start')
        time_diffs = []
        
        for i in range(1, len(df_sorted)):
            prev_end = df_sorted.iloc[i-1]['end']
            current_start = df_sorted.iloc[i]['start']
            diff_hours = (current_start - prev_end).total_seconds() / 3600
            time_diffs.append(diff_hours)
        
        if time_diffs:
            clustering_stats = {
                'mean_gap_between_periods_hours': np.mean(time_diffs),
                'median_gap_between_periods_hours': np.median(time_diffs),
                'min_gap_hours': np.min(time_diffs),
                'max_gap_hours': np.max(time_diffs),
                'clustered_periods': sum(1 for diff in time_diffs if diff < 24)  # Within 24 hours
            }
        else:
            clustering_stats = {}
        
        return clustering_stats
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in missing data"""
        if len(df) == 0:
            return {}
        
        # Group by month
        months = pd.to_datetime(df['start']).dt.month
        month_counts = months.value_counts().sort_index()
        
        # Ethiopian seasons mapping
        ethiopian_seasons = {
            'Dry_Season': [10, 11, 12, 1, 2],
            'Belg_Rainy': [3, 4, 5],
            'Kiremt_Rainy': [6, 7, 8, 9]
        }
        
        season_counts = {}
        for season, season_months in ethiopian_seasons.items():
            season_count = sum(month_counts.get(month, 0) for month in season_months)
            season_counts[season] = season_count
        
        return {
            'monthly_distribution': month_counts.to_dict(),
            'ethiopian_seasonal_distribution': season_counts
        }
    
    def _identify_systematic_issues(self, missing_periods: List[Dict[str, Any]], 
                                   temporal_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential systematic issues based on patterns"""
        issues = []
        
        if not missing_periods:
            return issues
        
        # Check for regular maintenance patterns
        if 'hour_patterns' in temporal_patterns:
            hour_data = temporal_patterns['hour_patterns']
            # Look for consistent gaps at the same hour
            for hour, stats in hour_data.items():
                if stats['duration_minutes']['count'] >= 5:  # At least 5 occurrences
                    issues.append({
                        'type': 'recurring_hourly_gap',
                        'hour': hour,
                        'occurrences': stats['duration_minutes']['count'],
                        'severity': 'medium',
                        'description': f'Recurring gaps at hour {hour} ({stats["duration_minutes"]["count"]} times)'
                    })
        
        # Check for very long gaps (potential instrument failures)
        df = pd.DataFrame(missing_periods)
        very_long_gaps = df[df['duration_minutes'] > 720]  # More than 12 hours
        
        for _, gap in very_long_gaps.iterrows():
            issues.append({
                'type': 'extended_outage',
                'start': gap['start'],
                'duration_hours': gap['duration_hours'],
                'severity': 'high',
                'description': f'Extended outage: {gap["duration_hours"]:.1f} hours starting {gap["start"]}'
            })
        
        # Check for weekend-specific issues
        weekend_stats = temporal_patterns.get('weekend_comparison', {})
        if weekend_stats:
            weekend_ratio = weekend_stats.get('weekend_total_minutes', 0) / max(weekend_stats.get('weekday_total_minutes', 1), 1)
            if weekend_ratio > 2:  # Weekends have significantly more missing data
                issues.append({
                    'type': 'weekend_maintenance_pattern',
                    'severity': 'low',
                    'description': 'Higher missing data on weekends suggests maintenance schedule'
                })
        
        return issues
    
    def _identify_potential_maintenance(self, missing_periods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify periods that might be planned maintenance"""
        if not missing_periods:
            return []
        
        maintenance_periods = []
        
        for period in missing_periods:
            # Criteria for potential maintenance:
            # 1. Duration between 1-8 hours
            # 2. Starts during business hours (8 AM - 6 PM) or early morning
            # 3. Occurs on weekdays or weekends
            
            duration_hours = period['duration_hours']
            start_hour = period['hour_of_day']
            is_weekend = period['is_weekend']
            
            # Likely maintenance windows
            is_maintenance_duration = 1 <= duration_hours <= 8
            is_maintenance_time = (8 <= start_hour <= 18) or (0 <= start_hour <= 6)
            
            if is_maintenance_duration and is_maintenance_time:
                confidence = 'medium'
                if is_weekend:
                    confidence = 'high'  # Weekend maintenance more likely
                
                maintenance_periods.append({
                    'start': period['start'],
                    'end': period['end'],
                    'duration_hours': duration_hours,
                    'confidence': confidence,
                    'reasons': [
                        f'Duration ({duration_hours:.1f}h) typical for maintenance',
                        f'Time ({start_hour}:00) typical for maintenance',
                        'Weekend timing' if is_weekend else 'Weekday timing'
                    ]
                })
        
        return maintenance_periods
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on missing data analysis"""
        recommendations = []
        
        # Basic completeness recommendations
        missing_map = analysis_results['missing_data_map']
        missing_pct = missing_map['missing_percentage']
        
        if missing_pct > 20:
            recommendations.append("🚨 High missing data rate (>20%). Consider instrument maintenance or replacement.")
        elif missing_pct > 10:
            recommendations.append("⚠️ Moderate missing data rate (>10%). Review data collection procedures.")
        elif missing_pct > 5:
            recommendations.append("📊 Acceptable missing data rate (<10%), but monitor trends.")
        else:
            recommendations.append("✅ Excellent data completeness (<5% missing).")
        
        # Pattern-based recommendations
        if 'systematic_issues' in analysis_results:
            issues = analysis_results['systematic_issues']
            
            if any(issue['type'] == 'extended_outage' for issue in issues):
                recommendations.append("🔧 Extended outages detected. Schedule comprehensive instrument inspection.")
            
            if any(issue['type'] == 'recurring_hourly_gap' for issue in issues):
                recommendations.append("⏰ Recurring hourly gaps suggest scheduled process interference. Review sampling protocols.")
            
            if any(issue['type'] == 'weekend_maintenance_pattern' for issue in issues):
                recommendations.append("📅 Weekend maintenance pattern detected. Document maintenance schedule for data users.")
        
        # Maintenance recommendations
        if 'maintenance_periods' in analysis_results:
            maintenance_periods = analysis_results['maintenance_periods']
            if len(maintenance_periods) > 0:
                recommendations.append(f"🔧 {len(maintenance_periods)} potential maintenance periods identified. Consider maintenance log integration.")
        
        return recommendations
    
    def print_missing_data_report(self, analysis_results: Dict[str, Any]):
        """Print comprehensive missing data analysis report"""
        print("\n" + "="*80)
        print("MISSING DATA PATTERN ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        missing_map = analysis_results['missing_data_map']
        print(f"\n📊 OVERVIEW:")
        print(f"  • Analysis period: {missing_map['analysis_period']['start'].date()} to {missing_map['analysis_period']['end'].date()}")
        print(f"  • Duration: {missing_map['analysis_period']['duration_days']} days")
        print(f"  • Missing data: {missing_map['total_missing_points']:,} points ({missing_map['missing_percentage']:.2f}%)")
        
        # Missing periods summary
        missing_periods = analysis_results['missing_periods']
        if missing_periods:
            print(f"  • Number of missing periods: {len(missing_periods)}")
            durations = [p['duration_minutes'] for p in missing_periods]
            print(f"  • Average gap duration: {np.mean(durations):.1f} minutes")
            print(f"  • Longest gap: {np.max(durations):.0f} minutes ({np.max(durations)/60:.1f} hours)")
        
        # Temporal patterns
        if 'temporal_patterns' in analysis_results and analysis_results['temporal_patterns']:
            patterns = analysis_results['temporal_patterns']
            
            if 'duration_categories' in patterns:
                cats = patterns['duration_categories']
                print(f"\n⏱️  GAP DURATION BREAKDOWN:")
                print(f"  • Short gaps (≤5 min): {cats['short_gaps_1_5min']}")
                print(f"  • Medium gaps (6-30 min): {cats['medium_gaps_6_30min']}")
                print(f"  • Long gaps (31-120 min): {cats['long_gaps_31_120min']}")
                print(f"  • Very long gaps (>120 min): {cats['very_long_gaps_120min_plus']}")
        
        # Systematic issues
        if 'systematic_issues' in analysis_results:
            issues = analysis_results['systematic_issues']
            if issues:
                print(f"\n🚨 SYSTEMATIC ISSUES DETECTED:")
                for issue in issues:
                    print(f"  • {issue['description']} (Severity: {issue['severity']})")
        
        # Maintenance periods
        if 'maintenance_periods' in analysis_results:
            maintenance = analysis_results['maintenance_periods']
            if maintenance:
                high_confidence = [m for m in maintenance if m['confidence'] == 'high']
                print(f"\n🔧 POTENTIAL MAINTENANCE PERIODS:")
                print(f"  • Total identified: {len(maintenance)}")
                print(f"  • High confidence: {len(high_confidence)}")
        
        # Recommendations
        if 'recommendations' in analysis_results:
            recommendations = analysis_results['recommendations']
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"  {rec}")
        
        print("\n" + "="*80)
