"""
Quality Assessment Reports Module

This module provides tools for generating comprehensive quality assessment reports,
combining all analysis components into unified reports with statistics, visualizations,
and export capabilities.

Author: AethModular Team
Created: 2025-01-12
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, List, Union
from datetime import datetime, timedelta
import json

from .missing_data import MissingDataAnalyzer
from .quality_classifier import QualityClassifier
from .seasonal_patterns import SeasonalPatternAnalyzer
from .filter_mapping import FilterSampleMapper
from .visualization import QualityVisualizer

logger = logging.getLogger(__name__)


class QualityReportGenerator:
    """
    Generator for comprehensive quality assessment reports.
    
    This class provides methods to:
    - Run complete quality assessment workflows
    - Generate summary statistics and reports
    - Export results in multiple formats
    - Create standardized quality reports
    """
    
    def __init__(self, output_dir: str = 'outputs/quality_assessment'):
        """
        Initialize the report generator.
        
        Parameters:
        -----------
        output_dir : str, default 'outputs/quality_assessment'
            Directory for saving outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.missing_analyzer = MissingDataAnalyzer()
        self.quality_classifier = QualityClassifier()
        self.seasonal_analyzer = SeasonalPatternAnalyzer()
        self.filter_mapper = FilterSampleMapper()
        self.visualizer = QualityVisualizer()
        
        # Store results
        self.results = {}
        
    def generate_complete_report(self, df: pd.DataFrame, 
                               db_path: Optional[str] = None,
                               site_code: str = 'ETAD',
                               period_type: str = '9am_to_9am',
                               freq: str = 'min') -> Dict:
        """
        Generate a complete quality assessment report.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Aethalometer data with datetime index
        db_path : str, optional
            Path to filter sample database
        site_code : str, default 'ETAD'
            Site code for filter samples
        period_type : str, default '9am_to_9am'
            Quality classification period type
        freq : str, default 'min'
            Expected data frequency
            
        Returns:
        --------
        dict
            Comprehensive analysis results
        """
        logger.info("Generating complete quality assessment report...")
        
        report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Missing data analysis
        logger.info("Step 1: Analyzing missing data patterns...")
        missing_analysis = self.missing_analyzer.analyze_missing_patterns(df, freq=freq)
        
        # 2. Quality classification
        logger.info("Step 2: Classifying quality periods...")
        if period_type == 'daily':
            quality_series = self.quality_classifier.classify_daily_periods(missing_analysis)
        else:
            quality_series = self.quality_classifier.classify_9am_to_9am_periods(missing_analysis)
        
        # 3. Seasonal pattern analysis
        logger.info("Step 3: Analyzing seasonal patterns...")
        seasonal_analysis = self.seasonal_analyzer.analyze_seasonal_missing_patterns(missing_analysis)
        
        # 4. Filter mapping (if database provided)
        overlap_results = None
        filter_export_df = None
        
        if db_path and Path(db_path).exists():
            logger.info("Step 4: Mapping filter samples to quality periods...")
            etad_data, ftir_data = self.filter_mapper.load_filter_data_from_db(db_path, site_code)
            overlap_results = self.filter_mapper.map_to_quality_periods(quality_series, etad_data, ftir_data)
            
            # Export filter data
            filter_export_path = self.output_dir / f'filter_samples_quality_{period_type}_{report_timestamp}.csv'
            filter_export_df = self.filter_mapper.export_quality_filtered_periods(
                include_all_quality=True, output_path=str(filter_export_path)
            )
        
        # 5. Generate visualizations
        logger.info("Step 5: Generating visualizations...")
        self._generate_all_visualizations(missing_analysis, quality_series, 
                                        seasonal_analysis, overlap_results)
        
        # 6. Compile comprehensive results
        results = {
            'metadata': {
                'report_timestamp': report_timestamp,
                'dataset_info': {
                    'start_date': missing_analysis['timeline']['start'],
                    'end_date': missing_analysis['timeline']['end'],
                    'duration_days': missing_analysis['timeline']['duration_days'],
                    'data_frequency': freq,
                    'period_type': period_type
                },
                'analysis_settings': {
                    'site_code': site_code,
                    'quality_thresholds': self.quality_classifier.quality_thresholds,
                    'database_used': db_path is not None
                }
            },
            'missing_analysis': missing_analysis,
            'quality_classification': {
                'quality_series': quality_series,
                'summary': self.quality_classifier.get_quality_summary(quality_series)
            },
            'seasonal_analysis': seasonal_analysis,
            'filter_mapping': overlap_results,
            'filter_export': filter_export_df,
            'summary_statistics': self._generate_summary_statistics(
                missing_analysis, quality_series, overlap_results
            )
        }
        
        # Store results
        self.results = results
        
        # 7. Export report
        self._export_report(results, report_timestamp)
        
        logger.info("Complete quality assessment report generated successfully!")
        
        return results
    
    def _generate_all_visualizations(self, missing_analysis: Dict, 
                                   quality_series: pd.Series,
                                   seasonal_analysis: Dict,
                                   overlap_results: Optional[Dict]) -> None:
        """Generate all visualization plots."""
        try:
            # Missing patterns
            self.visualizer.plot_missing_patterns(missing_analysis)
            
            # Quality distribution
            self.visualizer.plot_quality_distribution(quality_series)
            
            # Seasonal patterns
            self.visualizer.plot_seasonal_patterns(seasonal_analysis)
            
            # Quality timeline
            self.visualizer.plot_quality_timeline(quality_series)
            
            # Filter overlap (if available)
            if overlap_results:
                self.visualizer.plot_filter_overlap_summary(overlap_results)
                
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
    
    def _generate_summary_statistics(self, missing_analysis: Dict, 
                                   quality_series: pd.Series,
                                   overlap_results: Optional[Dict]) -> Dict:
        """Generate comprehensive summary statistics."""
        timeline = missing_analysis['timeline']
        quality_summary = self.quality_classifier.get_quality_summary(quality_series)
        
        summary = {
            'data_coverage': {
                'total_expected_points': timeline['expected_points'],
                'actual_data_points': timeline['actual_points'],
                'missing_points': timeline['missing_points'],
                'data_completeness_percentage': 100 - timeline['missing_percentage'],
                'missing_percentage': timeline['missing_percentage']
            },
            'quality_assessment': {
                'total_classified_periods': quality_summary['total_periods'],
                'high_quality_periods': quality_summary['high_quality_count'],
                'usable_percentage': quality_summary['usable_percentage'],
                'quality_distribution': quality_summary['quality_distribution']
            },
            'temporal_patterns': {
                'full_missing_days': missing_analysis['daily_patterns']['n_full_missing_days'],
                'partial_missing_days': missing_analysis['daily_patterns']['n_partial_missing_days'],
                'peak_missing_hour': self._get_peak_missing_hour(missing_analysis),
                'best_hour': self._get_best_hour(missing_analysis)
            }
        }
        
        # Add filter-specific statistics if available
        if overlap_results:
            filter_stats = overlap_results.get('filter_quality_stats', {})
            summary['filter_analysis'] = {
                'total_filter_periods': filter_stats.get('total_filter_periods', 0),
                'high_quality_overlaps': filter_stats.get('high_quality_periods', 0),
                'usable_for_comparison': filter_stats.get('high_quality_periods', 0),
                'filter_usability_percentage': filter_stats.get('usable_percentage', 0)
            }
        
        return summary
    
    def _get_peak_missing_hour(self, missing_analysis: Dict) -> Optional[int]:
        """Get hour with most missing data."""
        missing_per_hour = missing_analysis['temporal_patterns']['missing_per_hour']
        return missing_per_hour.idxmax() if len(missing_per_hour) > 0 else None
    
    def _get_best_hour(self, missing_analysis: Dict) -> Optional[int]:
        """Get hour with least missing data."""
        missing_per_hour = missing_analysis['temporal_patterns']['missing_per_hour']
        return missing_per_hour.idxmin() if len(missing_per_hour) > 0 else None
    
    def _export_report(self, results: Dict, timestamp: str) -> None:
        """Export complete report to files."""
        try:
            # Export summary as JSON
            summary_path = self.output_dir / f'quality_report_summary_{timestamp}.json'
            summary_data = {
                'metadata': results['metadata'],
                'summary_statistics': results['summary_statistics']
            }
            
            # Convert pandas objects to serializable format
            summary_serializable = self._make_json_serializable(summary_data)
            
            with open(summary_path, 'w') as f:
                json.dump(summary_serializable, f, indent=2, default=str)
            
            logger.info(f"Summary report exported to: {summary_path}")
            
            # Export quality series as CSV
            if 'quality_series' in results['quality_classification']:
                quality_path = self.output_dir / f'quality_periods_{timestamp}.csv'
                quality_df = pd.DataFrame({
                    'period_start': results['quality_classification']['quality_series'].index,
                    'quality': results['quality_classification']['quality_series'].values
                })
                quality_df.to_csv(quality_path, index=False)
                logger.info(f"Quality periods exported to: {quality_path}")
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert pandas/numpy objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def generate_quick_summary(self, df: pd.DataFrame, freq: str = 'min') -> Dict:
        """
        Generate a quick quality summary without full analysis.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with datetime index
        freq : str, default 'min'
            Expected data frequency
            
        Returns:
        --------
        dict
            Quick summary statistics
        """
        logger.info("Generating quick quality summary...")
        
        # Basic missing analysis
        missing_analysis = self.missing_analyzer.analyze_missing_patterns(df, freq)
        
        # Basic quality classification
        quality_series = self.quality_classifier.classify_daily_periods(missing_analysis)
        
        # Quick summary
        timeline = missing_analysis['timeline']
        quality_summary = self.quality_classifier.get_quality_summary(quality_series)
        
        summary = {
            'data_span': f"{timeline['start'].date()} to {timeline['end'].date()}",
            'duration_days': timeline['duration_days'],
            'data_completeness': f"{100 - timeline['missing_percentage']:.1f}%",
            'total_quality_periods': quality_summary['total_periods'],
            'high_quality_periods': quality_summary['high_quality_count'],
            'usable_percentage': f"{quality_summary['usable_percentage']:.1f}%",
            'quality_breakdown': quality_summary['quality_distribution']
        }
        
        # Print summary
        print("\n📊 Quick Quality Assessment Summary")
        print("=" * 40)
        print(f"Data span: {summary['data_span']}")
        print(f"Duration: {summary['duration_days']} days")
        print(f"Data completeness: {summary['data_completeness']}")
        print(f"Quality periods analyzed: {summary['total_quality_periods']}")
        print(f"High-quality periods: {summary['high_quality_periods']} ({summary['usable_percentage']})")
        print("\nQuality breakdown:")
        for quality, count in summary['quality_breakdown'].items():
            print(f"  {quality}: {count} periods")
        
        return summary
    
    def export_usable_periods_only(self, min_quality: str = 'Good',
                                  output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Export only the periods usable for analysis.
        
        Parameters:
        -----------
        min_quality : str, default 'Good'
            Minimum quality level
        output_path : str, optional
            Path to save CSV
            
        Returns:
        --------
        pd.DataFrame
            Usable periods
        """
        if not self.results:
            raise ValueError("Must generate report first")
        
        quality_series = self.results['quality_classification']['quality_series']
        
        # Filter by quality
        quality_hierarchy = {'Excellent': 4, 'Good': 3, 'Moderate': 2, 'Poor': 1}
        min_score = quality_hierarchy.get(min_quality, 3)
        
        usable_mask = quality_series.map(quality_hierarchy).fillna(0) >= min_score
        usable_periods = quality_series[usable_mask]
        
        # Create export DataFrame
        export_df = pd.DataFrame({
            'period_start': usable_periods.index,
            'period_end': usable_periods.index + pd.Timedelta(days=1),
            'quality': usable_periods.values
        })
        
        if output_path:
            export_df.to_csv(output_path, index=False)
            logger.info(f"Usable periods exported to: {output_path}")
        
        logger.info(f"Found {len(export_df)} usable periods with {min_quality}+ quality")
        
        return export_df


def create_comprehensive_report(df: pd.DataFrame, 
                              db_path: Optional[str] = None,
                              site_code: str = 'ETAD',
                              output_dir: str = 'outputs/quality_assessment',
                              period_type: str = '9am_to_9am') -> Dict:
    """
    Convenience function for creating a comprehensive quality assessment report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Aethalometer data with datetime index
    db_path : str, optional
        Path to filter sample database
    site_code : str, default 'ETAD'
        Site code for filter samples
    output_dir : str, default 'outputs/quality_assessment'
        Directory to save outputs
    period_type : str, default '9am_to_9am'
        Quality classification period type
        
    Returns:
    --------
    dict
        Comprehensive analysis results
    """
    generator = QualityReportGenerator(output_dir)
    return generator.generate_complete_report(df, db_path, site_code, period_type)


def quick_assessment(df: pd.DataFrame, freq: str = 'min') -> Dict:
    """
    Convenience function for quick quality assessment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    freq : str, default 'min'
        Expected data frequency
        
    Returns:
    --------
    dict
        Quick summary statistics
    """
    generator = QualityReportGenerator()
    return generator.generate_quick_summary(df, freq)


if __name__ == "__main__":
    # Example usage
    print("Quality Assessment Reports Module")
    print("Use QualityReportGenerator class or convenience functions")
