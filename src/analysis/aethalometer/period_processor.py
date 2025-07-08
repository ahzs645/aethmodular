"""9AM-to-9AM period processing for aethalometer data"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from ...core.base import BaseAnalyzer
from ...data.processors.validation import validate_columns_exist

class NineAMPeriodProcessor(BaseAnalyzer):
    """
    Processor for 9AM-to-9AM period alignment and quality classification
    
    This class handles:
    - Filter sample alignment: Matching aethalometer data to 24-hour filter periods
    - Quality classification: Excellent/Good/Poor period identification
    - Daily aggregation methods: Multiple approaches for daily averaging
    """
    
    def __init__(self):
        super().__init__("NineAMPeriodProcessor")
        self.quality_thresholds = {
            'excellent': 10,  # ≤10 minutes missing
            'good': 60,       # ≤60 minutes missing
            'poor': float('inf')  # >60 minutes missing
        }
    
    def analyze(self, data: pd.DataFrame, date_column: str = 'timestamp') -> Dict[str, Any]:
        """
        Process aethalometer data into 9AM-to-9AM periods
        
        Parameters:
        -----------
        data : pd.DataFrame
            Aethalometer data with timestamp column
        date_column : str
            Name of the timestamp/datetime column
            
        Returns:
        --------
        Dict[str, Any]
            Results including period summaries and quality classifications
        """
        validate_columns_exist(data, [date_column])
        
        # Ensure datetime column
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
        
        # Sort by timestamp
        data = data.sort_values(date_column).reset_index(drop=True)
        
        # Define 9AM-to-9AM periods
        periods = self._define_9am_periods(data, date_column)
        
        # Classify each period
        period_classifications = self._classify_periods(data, periods, date_column)
        
        # Calculate aggregations for each period
        period_aggregations = self._calculate_period_aggregations(
            data, periods, date_column
        )
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(period_classifications)
        
        results = {
            'processing_info': {
                'total_periods': len(periods),
                'date_range': {
                    'start': periods[0]['start'].strftime('%Y-%m-%d %H:%M:%S') if periods else None,
                    'end': periods[-1]['end'].strftime('%Y-%m-%d %H:%M:%S') if periods else None
                },
                'quality_thresholds': self.quality_thresholds
            },
            'periods': periods,
            'period_classifications': period_classifications,
            'period_aggregations': period_aggregations,
            'summary_statistics': summary_stats
        }
        
        return results
    
    def _define_9am_periods(self, data: pd.DataFrame, date_column: str) -> List[Dict[str, Any]]:
        """Define 9AM-to-9AM periods based on data range"""
        if len(data) == 0:
            return []
        
        start_time = data[date_column].min()
        end_time = data[date_column].max()
        
        # Find first 9AM after start time
        first_9am = start_time.replace(hour=9, minute=0, second=0, microsecond=0)
        if first_9am <= start_time:
            first_9am += timedelta(days=1)
        
        periods = []
        current_start = first_9am
        
        while current_start < end_time:
            current_end = current_start + timedelta(days=1)
            
            periods.append({
                'period_id': len(periods) + 1,
                'start': current_start,
                'end': current_end,
                'date_label': current_start.strftime('%Y-%m-%d')
            })
            
            current_start = current_end
        
        return periods
    
    def _classify_periods(self, data: pd.DataFrame, periods: List[Dict], 
                         date_column: str) -> List[Dict[str, Any]]:
        """Classify each period based on data completeness"""
        classifications = []
        
        for period in periods:
            # Extract period data
            period_mask = (data[date_column] >= period['start']) & \
                         (data[date_column] < period['end'])
            period_data = data[period_mask]
            
            # Calculate expected vs actual data points
            # Assuming 1-minute data
            expected_minutes = 24 * 60  # 1440 minutes in 24 hours
            actual_minutes = len(period_data)
            missing_minutes = expected_minutes - actual_minutes
            missing_percentage = (missing_minutes / expected_minutes) * 100
            
            # Classify quality
            if missing_minutes <= self.quality_thresholds['excellent']:
                quality = 'excellent'
            elif missing_minutes <= self.quality_thresholds['good']:
                quality = 'good'
            else:
                quality = 'poor'
            
            # Calculate hourly missing patterns
            hourly_missing = self._calculate_hourly_missing_pattern(
                period_data, period, date_column
            )
            
            classifications.append({
                'period_id': period['period_id'],
                'date_label': period['date_label'],
                'start': period['start'],
                'end': period['end'],
                'quality': quality,
                'data_completeness': {
                    'expected_minutes': expected_minutes,
                    'actual_minutes': actual_minutes,
                    'missing_minutes': missing_minutes,
                    'missing_percentage': round(missing_percentage, 2),
                    'completeness_percentage': round(100 - missing_percentage, 2)
                },
                'hourly_missing_pattern': hourly_missing
            })
        
        return classifications
    
    def _calculate_hourly_missing_pattern(self, period_data: pd.DataFrame, 
                                        period: Dict, date_column: str) -> Dict[int, int]:
        """Calculate missing data pattern by hour of day"""
        hourly_missing = {}
        
        for hour in range(24):
            hour_start = period['start'] + timedelta(hours=hour)
            hour_end = hour_start + timedelta(hours=1)
            
            hour_mask = (period_data[date_column] >= hour_start) & \
                       (period_data[date_column] < hour_end)
            hour_data = period_data[hour_mask]
            
            expected_per_hour = 60  # 60 minutes
            actual_per_hour = len(hour_data)
            missing_per_hour = expected_per_hour - actual_per_hour
            
            hourly_missing[hour] = missing_per_hour
        
        return hourly_missing
    
    def _calculate_period_aggregations(self, data: pd.DataFrame, periods: List[Dict], 
                                     date_column: str) -> List[Dict[str, Any]]:
        """Calculate different aggregation methods for each period"""
        aggregations = []
        
        # Get all numeric columns (BC values)
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for period in periods:
            # Extract period data
            period_mask = (data[date_column] >= period['start']) & \
                         (data[date_column] < period['end'])
            period_data = data[period_mask]
            
            if len(period_data) == 0:
                # No data for this period
                aggregations.append({
                    'period_id': period['period_id'],
                    'date_label': period['date_label'],
                    'aggregation_methods': {},
                    'sample_count': 0
                })
                continue
            
            # Calculate different aggregation methods
            methods = {}
            
            for col in numeric_columns:
                if col in period_data.columns:
                    valid_data = period_data[col].dropna()
                    
                    if len(valid_data) > 0:
                        methods[col] = {
                            'mean': float(valid_data.mean()),
                            'median': float(valid_data.median()),
                            'std': float(valid_data.std()),
                            'min': float(valid_data.min()),
                            'max': float(valid_data.max()),
                            'count': int(len(valid_data)),
                            'sum': float(valid_data.sum()),
                            # Weighted by data completeness
                            'completeness_weighted_mean': float(
                                valid_data.mean() * (len(valid_data) / 1440)
                            )
                        }
                    else:
                        methods[col] = {
                            'mean': np.nan, 'median': np.nan, 'std': np.nan,
                            'min': np.nan, 'max': np.nan, 'count': 0,
                            'sum': np.nan, 'completeness_weighted_mean': np.nan
                        }
            
            aggregations.append({
                'period_id': period['period_id'],
                'date_label': period['date_label'],
                'start': period['start'],
                'end': period['end'],
                'aggregation_methods': methods,
                'sample_count': len(period_data)
            })
        
        return aggregations
    
    def _generate_summary_statistics(self, classifications: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics across all periods"""
        if not classifications:
            return {}
        
        # Count periods by quality
        quality_counts = {'excellent': 0, 'good': 0, 'poor': 0}
        completeness_values = []
        missing_minutes_values = []
        
        for classification in classifications:
            quality_counts[classification['quality']] += 1
            completeness_values.append(
                classification['data_completeness']['completeness_percentage']
            )
            missing_minutes_values.append(
                classification['data_completeness']['missing_minutes']
            )
        
        total_periods = len(classifications)
        
        # Calculate overall statistics
        summary = {
            'total_periods': total_periods,
            'quality_distribution': {
                'excellent': {
                    'count': quality_counts['excellent'],
                    'percentage': round(quality_counts['excellent'] / total_periods * 100, 1)
                },
                'good': {
                    'count': quality_counts['good'],
                    'percentage': round(quality_counts['good'] / total_periods * 100, 1)
                },
                'poor': {
                    'count': quality_counts['poor'],
                    'percentage': round(quality_counts['poor'] / total_periods * 100, 1)
                }
            },
            'completeness_statistics': {
                'mean_completeness': round(np.mean(completeness_values), 2),
                'median_completeness': round(np.median(completeness_values), 2),
                'min_completeness': round(np.min(completeness_values), 2),
                'max_completeness': round(np.max(completeness_values), 2),
                'std_completeness': round(np.std(completeness_values), 2)
            },
            'missing_data_statistics': {
                'mean_missing_minutes': round(np.mean(missing_minutes_values), 1),
                'median_missing_minutes': round(np.median(missing_minutes_values), 1),
                'min_missing_minutes': int(np.min(missing_minutes_values)),
                'max_missing_minutes': int(np.max(missing_minutes_values)),
                'std_missing_minutes': round(np.std(missing_minutes_values), 1)
            }
        }
        
        return summary
    
    def get_quality_filtered_periods(self, results: Dict[str, Any], 
                                   min_quality: str = 'good') -> List[Dict]:
        """
        Get periods that meet minimum quality criteria
        
        Parameters:
        -----------
        results : Dict
            Results from analyze() method
        min_quality : str
            Minimum quality level ('excellent', 'good', 'poor')
            
        Returns:
        --------
        List[Dict]
            Filtered periods meeting quality criteria
        """
        quality_hierarchy = {'excellent': 3, 'good': 2, 'poor': 1}
        min_level = quality_hierarchy.get(min_quality, 1)
        
        filtered_periods = []
        for classification in results['period_classifications']:
            period_level = quality_hierarchy.get(classification['quality'], 1)
            if period_level >= min_level:
                filtered_periods.append(classification)
        
        return filtered_periods
    
    def create_daily_summary_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create a pandas DataFrame with daily summaries"""
        if not results['period_aggregations']:
            return pd.DataFrame()
        
        rows = []
        for agg in results['period_aggregations']:
            row = {
                'date': agg['date_label'],
                'period_id': agg['period_id'],
                'sample_count': agg['sample_count']
            }
            
            # Add aggregated values for each column
            for col, methods in agg['aggregation_methods'].items():
                for method, value in methods.items():
                    row[f"{col}_{method}"] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Add quality information
        quality_map = {cls['period_id']: cls['quality'] 
                      for cls in results['period_classifications']}
        completeness_map = {cls['period_id']: cls['data_completeness']['completeness_percentage'] 
                           for cls in results['period_classifications']}
        
        df['quality'] = df['period_id'].map(quality_map)
        df['completeness_percentage'] = df['period_id'].map(completeness_map)
        
        return df
