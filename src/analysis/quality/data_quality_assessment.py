# src/analysis/quality/data_quality_assessment.py
"""
Data quality assessment module for aethalometer data
Extracted from notebook for cleaner usage
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class QualityAssessmentResult:
    """Container for quality assessment results"""
    
    dataset_name: str
    total_periods: int
    excellent_periods: int
    excellent_periods_df: pd.DataFrame
    data_completeness: float
    missing_points: int
    time_range: Tuple[datetime, datetime]
    quality_threshold: int
    
    @property
    def excellent_percentage(self) -> float:
        """Percentage of excellent quality periods"""
        if self.total_periods == 0:
            return 0.0
        return (self.excellent_periods / self.total_periods) * 100

class DataQualityAssessor:
    """
    Comprehensive data quality assessment for aethalometer data
    """
    
    def __init__(self, quality_threshold: int = 10):
        """
        Initialize quality assessor
        
        Parameters:
        -----------
        quality_threshold : int
            Maximum missing minutes per 24h period for "excellent" quality
        """
        self.quality_threshold = quality_threshold
    
    def assess_dataset_quality(self, 
                             aethalometer_df: pd.DataFrame, 
                             dataset_name: str) -> QualityAssessmentResult:
        """
        Comprehensive quality assessment for 24-hour periods (9am-to-9am)
        
        Parameters:
        -----------
        aethalometer_df : pd.DataFrame
            Aethalometer data with datetime index
        dataset_name : str
            Name of the dataset for logging
            
        Returns:
        --------
        QualityAssessmentResult
            Comprehensive quality assessment results
        """
        
        print(f"🔍 Analyzing {dataset_name} data quality...")
        print(f"📊 Quality threshold: ≤{self.quality_threshold} missing minutes per 24h period")
        
        # Validate input
        if not isinstance(aethalometer_df.index, pd.DatetimeIndex):
            raise ValueError(f"Invalid index type: {type(aethalometer_df.index)}")
        
        if len(aethalometer_df) == 0:
            raise ValueError("Empty DataFrame provided")
        
        # Basic statistics
        df_start = aethalometer_df.index.min()
        df_end = aethalometer_df.index.max()
        actual_points = len(aethalometer_df.index.unique())
        
        print(f"📅 Time range: {df_start} to {df_end}")
        print(f"📊 Actual data points: {actual_points:,}")
        
        # Calculate expected points based on 1-minute resolution
        total_minutes = int((df_end - df_start).total_seconds() / 60) + 1
        print(f"📊 Expected data points (1-min resolution): {total_minutes:,}")
        
        # Calculate missing points
        missing_points = max(0, total_minutes - actual_points)
        completeness = (actual_points / total_minutes) * 100
        
        print(f"⚠️ Missing data points: {missing_points:,}")
        print(f"📊 Data completeness: {completeness:.1f}%")
        
        # Assess 24-hour periods
        excellent_periods_df = self._assess_24h_periods(aethalometer_df, df_start, df_end)
        
        # Create result object
        result = QualityAssessmentResult(
            dataset_name=dataset_name,
            total_periods=self._count_total_periods(df_start, df_end),
            excellent_periods=len(excellent_periods_df),
            excellent_periods_df=excellent_periods_df,
            data_completeness=completeness,
            missing_points=missing_points,
            time_range=(df_start, df_end),
            quality_threshold=self.quality_threshold
        )
        
        self._print_quality_summary(result)
        
        return result
    
    def _assess_24h_periods(self, 
                          aethalometer_df: pd.DataFrame, 
                          df_start: datetime, 
                          df_end: datetime) -> pd.DataFrame:
        """Assess 24-hour periods from 9am to 9am"""
        
        # Create all possible 9am-to-9am periods
        first_9am = df_start.normalize() + pd.Timedelta(hours=9)
        if df_start.hour < 9:
            first_9am -= pd.Timedelta(days=1)
        
        last_9am = df_end.normalize() + pd.Timedelta(hours=9)
        if df_end.hour < 9:
            last_9am -= pd.Timedelta(days=1)
        
        all_period_starts = pd.date_range(first_9am, last_9am, freq='D')
        
        print(f"📅 Analyzing {len(all_period_starts)} 24-hour periods...")
        
        # Assess each period
        excellent_periods_list = []
        
        for period_start in all_period_starts:
            period_end = period_start + pd.Timedelta(days=1)
            
            # Get data for this period (inclusive start, exclusive end)
            period_data = aethalometer_df.loc[period_start:period_end]
            actual_minutes = len(period_data)
            expected_minutes = 1440  # 24 hours * 60 minutes
            missing_minutes = max(0, expected_minutes - actual_minutes)
            
            # Check if this period qualifies as excellent
            if missing_minutes <= self.quality_threshold:
                excellent_periods_list.append({
                    'start_time': period_start,
                    'end_time': period_end,
                    'missing_minutes': missing_minutes,
                    'actual_minutes': actual_minutes,
                    'completeness_pct': (actual_minutes / expected_minutes) * 100
                })
        
        return pd.DataFrame(excellent_periods_list)
    
    def _count_total_periods(self, df_start: datetime, df_end: datetime) -> int:
        """Count total possible 24-hour periods"""
        
        first_9am = df_start.normalize() + pd.Timedelta(hours=9)
        if df_start.hour < 9:
            first_9am -= pd.Timedelta(days=1)
        
        last_9am = df_end.normalize() + pd.Timedelta(hours=9)
        if df_end.hour < 9:
            last_9am -= pd.Timedelta(days=1)
        
        return len(pd.date_range(first_9am, last_9am, freq='D'))
    
    def _print_quality_summary(self, result: QualityAssessmentResult):
        """Print formatted quality summary"""
        
        print(f"✅ Quality assessment complete for {result.dataset_name}")
        print(f"📊 Total 24h periods: {result.total_periods}")
        print(f"🌟 Excellent periods: {result.excellent_periods}")
        print(f"📈 Excellence rate: {result.excellent_percentage:.1f}%")
        
        if len(result.excellent_periods_df) > 0:
            print(f"📅 Excellent periods range: {result.excellent_periods_df['start_time'].min()} to {result.excellent_periods_df['start_time'].max()}")
            print(f"📊 Missing minutes distribution:")
            
            missing_dist = result.excellent_periods_df['missing_minutes']
            print(f"   0 minutes missing: {(missing_dist == 0).sum()} periods")
            print(f"   1-5 minutes missing: {((missing_dist >= 1) & (missing_dist <= 5)).sum()} periods")
            print(f"   6-10 minutes missing: {((missing_dist >= 6) & (missing_dist <= 10)).sum()} periods")
        else:
            print("❌ No excellent quality periods found")

class MultiDatasetQualityAssessor:
    """
    Quality assessment for multiple datasets
    """
    
    def __init__(self, quality_threshold: int = 10):
        """
        Initialize multi-dataset assessor
        
        Parameters:
        -----------
        quality_threshold : int
            Quality threshold for assessments
        """
        self.assessor = DataQualityAssessor(quality_threshold)
        self.results = {}
    
    def assess_all_datasets(self, 
                           datasets: Dict[str, pd.DataFrame]) -> Dict[str, QualityAssessmentResult]:
        """
        Assess quality for all datasets
        
        Parameters:
        -----------
        datasets : dict
            Dictionary of dataset_name -> DataFrame
            
        Returns:
        --------
        dict
            Dictionary of dataset_name -> QualityAssessmentResult
        """
        
        print("\n" + "="*80)
        print("🔍 MULTI-DATASET QUALITY ASSESSMENT")
        print("="*80)
        
        results = {}
        
        for dataset_name, df in datasets.items():
            if 'ftir' in dataset_name.lower():
                print(f"\n⚠️ Skipping FTIR dataset {dataset_name} (not time-series)")
                continue
            
            print(f"\n{'='*60}")
            
            try:
                result = self.assessor.assess_dataset_quality(df, dataset_name)
                results[dataset_name] = result
            except Exception as e:
                print(f"❌ Failed to assess {dataset_name}: {e}")
            
            print("="*60)
        
        self.results = results
        self._print_comparative_summary()
        
        return results
    
    def _print_comparative_summary(self):
        """Print comparative summary across datasets"""
        
        if not self.results:
            return
        
        print(f"\n" + "="*80)
        print("📊 COMPARATIVE QUALITY SUMMARY")
        print("="*80)
        
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Dataset': name,
                'Total Periods': result.total_periods,
                'Excellent Periods': result.excellent_periods,
                'Excellence Rate (%)': f"{result.excellent_percentage:.1f}%",
                'Data Completeness (%)': f"{result.data_completeness:.1f}%",
                'Missing Points': f"{result.missing_points:,}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Find best dataset
        if len(summary_data) > 1:
            best_dataset = max(self.results.items(), key=lambda x: x[1].excellent_percentage)
            print(f"\n🏆 Best quality dataset: {best_dataset[0]} ({best_dataset[1].excellent_percentage:.1f}% excellent)")
        
        print("="*80)
    
    def get_excellent_periods_for_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Get excellent periods DataFrame for a specific dataset
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
            
        Returns:
        --------
        pd.DataFrame or None
            DataFrame with excellent periods
        """
        
        if dataset_name in self.results:
            return self.results[dataset_name].excellent_periods_df
        else:
            print(f"⚠️ Dataset '{dataset_name}' not found in assessment results")
            return None
    
    def get_best_quality_dataset(self) -> Optional[str]:
        """
        Get the name of the best quality dataset
        
        Returns:
        --------
        str or None
            Name of best quality dataset
        """
        
        if not self.results:
            return None
        
        best_dataset = max(self.results.items(), key=lambda x: x[1].excellent_percentage)
        return best_dataset[0]
    
    def filter_datasets_by_quality(self, 
                                 min_excellence_rate: float = 50.0) -> List[str]:
        """
        Filter datasets by minimum excellence rate
        
        Parameters:
        -----------
        min_excellence_rate : float
            Minimum excellence rate percentage
            
        Returns:
        --------
        list
            List of dataset names meeting the criteria
        """
        
        qualified_datasets = []
        
        for name, result in self.results.items():
            if result.excellent_percentage >= min_excellence_rate:
                qualified_datasets.append(name)
        
        print(f"📊 Datasets with ≥{min_excellence_rate}% excellence rate: {qualified_datasets}")
        
        return qualified_datasets

# Convenience functions for notebook usage
def assess_single_dataset(df: pd.DataFrame, 
                        dataset_name: str, 
                        quality_threshold: int = 10) -> QualityAssessmentResult:
    """
    Convenience function to assess a single dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Aethalometer data with datetime index
    dataset_name : str
        Name of the dataset
    quality_threshold : int
        Quality threshold
        
    Returns:
    --------
    QualityAssessmentResult
        Quality assessment results
    """
    
    assessor = DataQualityAssessor(quality_threshold)
    return assessor.assess_dataset_quality(df, dataset_name)

def assess_multiple_datasets(datasets: Dict[str, pd.DataFrame],
                           quality_threshold: int = 10) -> Dict[str, QualityAssessmentResult]:
    """
    Convenience function to assess multiple datasets
    
    Parameters:
    -----------
    datasets : dict
        Dictionary of dataset_name -> DataFrame
    quality_threshold : int
        Quality threshold
        
    Returns:
    --------
    dict
        Dictionary of dataset_name -> QualityAssessmentResult
    """
    
    assessor = MultiDatasetQualityAssessor(quality_threshold)
    return assessor.assess_all_datasets(datasets)