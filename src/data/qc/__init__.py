"""
Quality Control (QC) Module for Aethalometer Data

This package provides modular tools for data quality assessment:

- missing_data: Analysis of missing data patterns and gaps
- quality_classifier: Classification of data quality periods
- seasonal_patterns: Seasonal and temporal pattern analysis  
- filter_mapping: Mapping filter samples to quality periods
- visualization: Plotting and visualization tools
- reports: Comprehensive quality reports

Example usage:
    from src.data.qc import missing_data, quality_classifier
    
    # Analyze missing data
    missing_analysis = missing_data.analyze_missing_patterns(df)
    
    # Classify quality periods
    quality_periods = quality_classifier.classify_periods(missing_analysis)
"""

from .missing_data import MissingDataAnalyzer
from .quality_classifier import QualityClassifier
from .seasonal_patterns import SeasonalPatternAnalyzer
from .filter_mapping import FilterSampleMapper
from .visualization import QualityVisualizer
from .reports import QualityReportGenerator

# Convenience function for quick analysis
def quick_quality_check(df, freq='min'):
    """
    Quick quality check with basic statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    freq : str, default 'min'
        Expected data frequency
    """
    analyzer = MissingDataAnalyzer()
    missing_analysis = analyzer.analyze_missing_patterns(df, freq=freq)
    
    stats = missing_analysis['timeline']
    print(f"📊 Quick Quality Check")
    print(f"Time range: {stats['start']} to {stats['end']} ({stats['duration_days']} days)")
    print(f"Expected points: {stats['expected_points']:,}")
    print(f"Actual points: {stats['actual_points']:,}")
    print(f"Missing: {stats['missing_points']:,} ({stats['missing_percentage']:.2f}%)")
    
    daily_stats = missing_analysis['daily_patterns']
    print(f"Full missing days: {daily_stats['n_full_missing_days']}")
    print(f"Partial missing days: {daily_stats['n_partial_missing_days']}")

__all__ = [
    'MissingDataAnalyzer',
    'QualityClassifier', 
    'SeasonalPatternAnalyzer',
    'FilterSampleMapper',
    'QualityVisualizer',
    'QualityReportGenerator',
    'quick_quality_check'
]
