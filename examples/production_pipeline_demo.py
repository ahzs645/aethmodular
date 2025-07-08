"""
Production-Ready ETAD Analysis Pipeline
=======================================

This example demonstrates a complete production-ready analysis pipeline
that combines all ETAD capabilities in a realistic workflow.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import os
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.insert(0, str(src_path))

# Import all necessary modules
try:
    # Core infrastructure
    from src.core.base import BaseAnalyzer
    from src.core.monitoring import monitor_performance, handle_errors
    from src.core.parallel_processing import ParallelProcessor
    from src.utils.memory_optimization import MemoryOptimizer, DataFrameOptimizer
    from src.utils.logging.logger import setup_logging, get_logger
    
    # Quality analysis
    from src.analysis.quality.completeness_analyzer import CompletenessAnalyzer
    from src.analysis.quality.missing_data_analyzer import MissingDataAnalyzer
    from src.analysis.quality.period_classifier import PeriodClassifier
    
    # Advanced analytics
    from src.analysis.advanced.statistical_analysis import (
        StatisticalComparator, DistributionAnalyzer, OutlierDetector
    )
    from src.analysis.advanced.ml_analysis import (
        MLModelTrainer, PredictiveAnalyzer, ClusterAnalyzer
    )
    from src.analysis.advanced.time_series_analysis import (
        TimeSeriesAnalyzer, TrendDetector, SeasonalAnalyzer
    )
    
    # Traditional analysis modules
    from src.analysis.aethalometer.smoothening import ONASmoothing, CMASmoothing, DEMASmoothing
    from src.analysis.correlations.pearson import PearsonAnalyzer
    from src.analysis.seasonal.ethiopian_seasons import EthiopianSeasonAnalyzer
    
    # Configuration
    from src.config.plotting import PlottingConfig
    from src.config.settings import AnalysisSettings
    
    MODULES_AVAILABLE = True
    print("✅ All ETAD modules imported successfully!")
    
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"❌ Import error: {e}")
    print("Please run: pip install -r requirements.txt")


class ETADProductionPipeline(BaseAnalyzer):
    """
    Production-ready ETAD analysis pipeline that combines all capabilities
    """
    
    def __init__(self, config_file=None):
        super().__init__("ETADProductionPipeline")
        
        # Setup logging
        try:
            setup_logging()
            self.logger = get_logger(__name__)
            self.logger.info("Initializing ETAD Production Pipeline")
        except:
            print("⚠️  Logging setup failed, using print statements")
            self.logger = None
        
        # Initialize components
        self.memory_optimizer = MemoryOptimizer()
        self.parallel_processor = ParallelProcessor()
        
        # Analysis modules
        self.quality_modules = {
            'completeness': CompletenessAnalyzer(),
            'missing_data': MissingDataAnalyzer(),
            'classifier': PeriodClassifier()
        }
        
        self.statistical_modules = {
            'comparator': StatisticalComparator(),
            'distribution': DistributionAnalyzer(),
            'outlier_detector': OutlierDetector()
        }
        
        self.ml_modules = {
            'trainer': MLModelTrainer(),
            'predictor': PredictiveAnalyzer(),
            'clusterer': ClusterAnalyzer()
        }
        
        self.ts_modules = {
            'analyzer': TimeSeriesAnalyzer(),
            'trend_detector': TrendDetector(),
            'seasonal_analyzer': SeasonalAnalyzer()
        }
        
        # Smoothening methods
        self.smoothening_methods = {
            'ona': ONASmoothing(),
            'cma': CMASmoothing(),
            'dema': DEMASmoothing()
        }
        
        # Other analyzers
        self.correlation_analyzer = PearsonAnalyzer()
        self.seasonal_analyzer = EthiopianSeasonAnalyzer()
        
        # Results storage
        self.results = {}
        
        self._log("Pipeline initialized successfully")
    
    def _log(self, message, level='info'):
        """Helper method for logging"""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    @monitor_performance
    @handle_errors
    def run_complete_analysis(self, data, analysis_config=None):
        """
        Run complete ETAD analysis pipeline
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw aethalometer data
        analysis_config : dict, optional
            Configuration for analysis steps
            
        Returns:
        --------
        dict
            Complete analysis results
        """
        self._log("Starting complete ETAD analysis pipeline")
        
        if analysis_config is None:
            analysis_config = self._get_default_config()
        
        # Stage 1: Data preparation and optimization
        self._log("Stage 1: Data preparation and memory optimization")
        optimized_data = self._prepare_data(data)
        
        # Stage 2: Quality assessment
        self._log("Stage 2: Quality assessment")
        quality_results = self._run_quality_analysis(optimized_data, analysis_config)
        
        # Stage 3: Data preprocessing (smoothening, outlier handling)
        self._log("Stage 3: Data preprocessing")
        processed_data = self._preprocess_data(optimized_data, quality_results, analysis_config)
        
        # Stage 4: Statistical analysis
        self._log("Stage 4: Statistical analysis")
        statistical_results = self._run_statistical_analysis(processed_data, analysis_config)
        
        # Stage 5: Time series analysis
        self._log("Stage 5: Time series analysis")
        ts_results = self._run_time_series_analysis(processed_data, analysis_config)
        
        # Stage 6: Machine learning analysis
        self._log("Stage 6: Machine learning analysis")
        ml_results = self._run_ml_analysis(processed_data, analysis_config)
        
        # Stage 7: Correlation and seasonal analysis
        self._log("Stage 7: Correlation and seasonal analysis")
        correlation_results = self._run_correlation_analysis(processed_data, analysis_config)
        
        # Stage 8: Generate comprehensive report
        self._log("Stage 8: Generating comprehensive report")
        final_report = self._generate_final_report(
            optimized_data, processed_data, quality_results,
            statistical_results, ts_results, ml_results, correlation_results
        )
        
        self._log("Complete analysis pipeline finished successfully")
        return final_report
    
    def _get_default_config(self):
        """Get default analysis configuration"""
        return {
            'quality_analysis': {
                'period_type': 'daily',
                'completeness_threshold': 0.8
            },
            'preprocessing': {
                'smoothening_method': 'ona',
                'outlier_handling': 'flag',
                'outlier_methods': ['iqr', 'zscore']
            },
            'statistical_analysis': {
                'comparison_tests': ['ttest', 'kstest'],
                'distribution_tests': ['normal', 'lognormal'],
                'outlier_detection': True
            },
            'time_series': {
                'trend_detection': True,
                'seasonality_analysis': True,
                'stationarity_tests': True
            },
            'machine_learning': {
                'regression_models': ['random_forest', 'linear'],
                'clustering': True,
                'forecasting': True,
                'forecast_periods': 24
            },
            'correlation_analysis': {
                'methods': ['pearson', 'spearman'],
                'variables': ['BC', 'UV_abs', 'IR_abs']
            }
        }
    
    @monitor_performance
    def _prepare_data(self, data):
        """Prepare and optimize data for analysis"""
        # Memory optimization
        df_optimizer = DataFrameOptimizer()
        optimized_data = df_optimizer.optimize_dataframe(data)
        
        # Basic data validation
        if not isinstance(optimized_data.index, pd.DatetimeIndex):
            if 'datetime_local' in optimized_data.columns:
                optimized_data = optimized_data.set_index('datetime_local')
            else:
                raise ValueError("Data must have datetime index or 'datetime_local' column")
        
        # Sort by time
        optimized_data = optimized_data.sort_index()
        
        self._log(f"Data prepared: {len(optimized_data):,} points, {len(optimized_data.columns)} variables")
        return optimized_data
    
    @monitor_performance
    def _run_quality_analysis(self, data, config):
        """Run comprehensive quality analysis"""
        results = {}
        
        # Completeness analysis
        try:
            results['completeness'] = self.quality_modules['completeness'].analyze_completeness(
                data, period_type=config['quality_analysis']['period_type']
            )
        except Exception as e:
            self._log(f"Completeness analysis failed: {e}", 'warning')
            results['completeness'] = {'error': str(e)}
        
        # Missing data patterns
        try:
            results['missing_patterns'] = self.quality_modules['missing_data'].analyze_missing_patterns(data)
        except Exception as e:
            self._log(f"Missing data analysis failed: {e}", 'warning')
            results['missing_patterns'] = {'error': str(e)}
        
        # Period classification
        try:
            results['period_classification'] = self.quality_modules['classifier'].classify_periods(
                data, period_type=config['quality_analysis']['period_type']
            )
        except Exception as e:
            self._log(f"Period classification failed: {e}", 'warning')
            results['period_classification'] = {'error': str(e)}
        
        return results
    
    @monitor_performance
    def _preprocess_data(self, data, quality_results, config):
        """Preprocess data based on quality analysis"""
        processed_data = data.copy()
        
        # Apply smoothening if configured
        smoothening_method = config['preprocessing']['smoothening_method']
        if smoothening_method in self.smoothening_methods:
            try:
                smoother = self.smoothening_methods[smoothening_method]
                for column in ['BC', 'UV_abs', 'IR_abs']:
                    if column in processed_data.columns:
                        smoothed = smoother.smooth(processed_data[column])
                        processed_data[f'{column}_smoothed'] = smoothed
                
                self._log(f"Applied {smoothening_method} smoothening")
            except Exception as e:
                self._log(f"Smoothening failed: {e}", 'warning')
        
        # Outlier detection and handling
        if config['preprocessing']['outlier_handling'] == 'flag':
            try:
                outlier_results = self.statistical_modules['outlier_detector'].detect_outliers(
                    processed_data, 
                    methods=config['preprocessing']['outlier_methods']
                )
                
                # Flag outliers in data
                for var, detection in outlier_results.get('outlier_detection', {}).items():
                    if 'summary' in detection:
                        outlier_indices = set()
                        for method, result in detection.items():
                            if isinstance(result, dict) and 'outlier_indices' in result:
                                outlier_indices.update(result['outlier_indices'])
                        
                        # Create outlier flag column
                        processed_data[f'{var}_outlier'] = processed_data.index.isin(outlier_indices)
                
                self._log(f"Outlier detection completed")
            except Exception as e:
                self._log(f"Outlier detection failed: {e}", 'warning')
        
        return processed_data
    
    @monitor_performance
    def _run_statistical_analysis(self, data, config):
        """Run statistical analysis"""
        results = {}
        
        # Split data for comparison
        mid_point = len(data) // 2
        period1 = data.iloc[:mid_point]
        period2 = data.iloc[mid_point:]
        
        # Period comparison
        try:
            results['period_comparison'] = self.statistical_modules['comparator'].compare_periods(
                period1, period2,
                comparison_tests=config['statistical_analysis']['comparison_tests']
            )
        except Exception as e:
            self._log(f"Period comparison failed: {e}", 'warning')
            results['period_comparison'] = {'error': str(e)}
        
        # Distribution analysis
        try:
            results['distributions'] = {}
            for var in ['BC', 'UV_abs', 'IR_abs']:
                if var in data.columns:
                    results['distributions'][var] = self.statistical_modules['distribution'].analyze_distribution(
                        data[var].dropna(),
                        distributions=config['statistical_analysis']['distribution_tests']
                    )
        except Exception as e:
            self._log(f"Distribution analysis failed: {e}", 'warning')
            results['distributions'] = {'error': str(e)}
        
        return results
    
    @monitor_performance
    def _run_time_series_analysis(self, data, config):
        """Run time series analysis"""
        results = {}
        
        # Basic time series analysis
        if config['time_series']['trend_detection']:
            try:
                results['bc_analysis'] = self.ts_modules['analyzer'].analyze_time_series(data['BC'])
            except Exception as e:
                self._log(f"Time series analysis failed: {e}", 'warning')
                results['bc_analysis'] = {'error': str(e)}
        
        # Trend detection
        if config['time_series']['trend_detection']:
            try:
                results['trends'] = {}
                for var in ['BC', 'UV_abs']:
                    if var in data.columns:
                        results['trends'][var] = self.ts_modules['trend_detector'].detect_trends(data[var])
            except Exception as e:
                self._log(f"Trend detection failed: {e}", 'warning')
                results['trends'] = {'error': str(e)}
        
        # Seasonality analysis
        if config['time_series']['seasonality_analysis']:
            try:
                results['seasonality'] = {}
                for var in ['BC', 'UV_abs']:
                    if var in data.columns and len(data[var].dropna()) > 48:
                        results['seasonality'][var] = self.ts_modules['seasonal_analyzer'].analyze_seasonality(data[var])
            except Exception as e:
                self._log(f"Seasonality analysis failed: {e}", 'warning')
                results['seasonality'] = {'error': str(e)}
        
        return results
    
    @monitor_performance
    def _run_ml_analysis(self, data, config):
        """Run machine learning analysis"""
        results = {}
        
        # Prepare clean data for ML
        ml_data = data[['BC', 'UV_abs', 'IR_abs', 'temperature', 'humidity']].dropna()
        
        if len(ml_data) < 100:
            self._log("Insufficient data for ML analysis", 'warning')
            return {'error': 'Insufficient data'}
        
        # Regression modeling
        try:
            results['regression'] = {}
            for model_type in config['machine_learning']['regression_models']:
                results['regression'][model_type] = self.ml_modules['trainer'].train_regression_model(
                    ml_data,
                    target_column='BC',
                    feature_columns=['UV_abs', 'IR_abs', 'temperature'],
                    model_type=model_type
                )
        except Exception as e:
            self._log(f"Regression modeling failed: {e}", 'warning')
            results['regression'] = {'error': str(e)}
        
        # Clustering
        if config['machine_learning']['clustering']:
            try:
                results['clustering'] = self.ml_modules['clusterer'].perform_clustering(
                    ml_data,
                    columns=['BC', 'UV_abs', 'IR_abs'],
                    n_clusters=3
                )
            except Exception as e:
                self._log(f"Clustering failed: {e}", 'warning')
                results['clustering'] = {'error': str(e)}
        
        # Forecasting
        if config['machine_learning']['forecasting']:
            try:
                results['forecasting'] = self.ml_modules['predictor'].forecast_time_series(
                    data['BC'].dropna(),
                    forecast_periods=config['machine_learning']['forecast_periods'],
                    method='holt'
                )
            except Exception as e:
                self._log(f"Forecasting failed: {e}", 'warning')
                results['forecasting'] = {'error': str(e)}
        
        return results
    
    @monitor_performance
    def _run_correlation_analysis(self, data, config):
        """Run correlation and seasonal analysis"""
        results = {}
        
        # Correlation analysis
        try:
            variables = config['correlation_analysis']['variables']
            available_vars = [var for var in variables if var in data.columns]
            
            if len(available_vars) >= 2:
                results['correlations'] = self.correlation_analyzer.calculate_correlations(
                    data[available_vars]
                )
        except Exception as e:
            self._log(f"Correlation analysis failed: {e}", 'warning')
            results['correlations'] = {'error': str(e)}
        
        # Ethiopian seasonal analysis
        try:
            results['ethiopian_seasons'] = self.seasonal_analyzer.analyze_seasonal_patterns(
                data, variables=['BC', 'UV_abs']
            )
        except Exception as e:
            self._log(f"Seasonal analysis failed: {e}", 'warning')
            results['ethiopian_seasons'] = {'error': str(e)}
        
        return results
    
    def _generate_final_report(self, original_data, processed_data, quality_results,
                              statistical_results, ts_results, ml_results, correlation_results):
        """Generate comprehensive final report"""
        
        report = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'data_period': {
                    'start': original_data.index.min().isoformat(),
                    'end': original_data.index.max().isoformat(),
                    'duration_days': (original_data.index.max() - original_data.index.min()).days + 1
                },
                'data_summary': {
                    'original_points': len(original_data),
                    'processed_points': len(processed_data),
                    'variables': list(original_data.columns)
                }
            },
            'quality_assessment': quality_results,
            'statistical_analysis': statistical_results,
            'time_series_analysis': ts_results,
            'machine_learning': ml_results,
            'correlations_and_seasonality': correlation_results,
            'executive_summary': self._create_executive_summary(
                quality_results, statistical_results, ts_results, ml_results
            )
        }
        
        return report
    
    def _create_executive_summary(self, quality_results, statistical_results, ts_results, ml_results):
        """Create executive summary of key findings"""
        summary = {
            'data_quality': 'Unknown',
            'key_findings': [],
            'recommendations': [],
            'analysis_completeness': {}
        }
        
        # Assess data quality
        if 'completeness' in quality_results and 'analysis_summary' in quality_results['completeness']:
            completeness = quality_results['completeness']['analysis_summary'].get('overall_completeness_percent', 0)
            if completeness >= 95:
                summary['data_quality'] = 'Excellent'
            elif completeness >= 85:
                summary['data_quality'] = 'Good' 
            elif completeness >= 70:
                summary['data_quality'] = 'Moderate'
            else:
                summary['data_quality'] = 'Poor'
            
            summary['key_findings'].append(f"Data completeness: {completeness:.1f}%")
        
        # Check analysis completeness
        analyses = {
            'quality_assessment': 'error' not in str(quality_results),
            'statistical_analysis': 'error' not in str(statistical_results),
            'time_series_analysis': 'error' not in str(ts_results),
            'machine_learning': 'error' not in str(ml_results)
        }
        
        summary['analysis_completeness'] = analyses
        completed_count = sum(analyses.values())
        total_count = len(analyses)
        
        summary['key_findings'].append(f"Analysis completion: {completed_count}/{total_count} modules")
        
        # Add recommendations based on results
        if summary['data_quality'] in ['Poor', 'Moderate']:
            summary['recommendations'].append("Investigate data collection issues to improve completeness")
        
        if completed_count < total_count:
            summary['recommendations'].append("Address module errors to enable complete analysis")
        
        summary['recommendations'].append("Regular monitoring and quality assessment recommended")
        summary['recommendations'].append("Consider seasonal patterns in interpretation")
        
        return summary


def generate_demo_data(n_days=14):
    """Generate realistic demo data for pipeline testing"""
    np.random.seed(42)
    
    start_date = datetime(2023, 6, 1)  # Ethiopian dry season
    end_date = start_date + timedelta(days=n_days)
    timeline = pd.date_range(start_date, end_date, freq='min', inclusive='left')
    
    n_points = len(timeline)
    t_hours = np.arange(n_points) / 60
    t_days = t_hours / 24
    
    # Generate realistic black carbon with patterns
    bc_base = (
        4.0 +  # Baseline higher in dry season
        2.0 * np.sin(2 * np.pi * t_hours / 24) +  # Daily pattern
        0.8 * np.sin(2 * np.pi * t_days / 7) +    # Weekly pattern
        0.02 * t_days  # Slight trend
    )
    
    bc = np.maximum(bc_base + np.random.normal(0, 0.4, n_points), 0.1)
    
    # Correlated measurements
    uv_abs = 4.5 * bc + np.random.normal(0, 0.6, n_points)
    ir_abs = 3.2 * bc + np.random.normal(0, 0.5, n_points)
    
    # Environmental variables
    temperature = 25 + 8 * np.sin(2 * np.pi * t_hours / 24) + np.random.normal(0, 2, n_points)
    humidity = 30 + 15 * np.sin(2 * np.pi * t_hours / 24 + np.pi) + np.random.normal(0, 3, n_points)
    
    # Create DataFrame
    data = pd.DataFrame({
        'BC': bc,
        'UV_abs': uv_abs,
        'IR_abs': ir_abs,
        'temperature': temperature,
        'humidity': np.clip(humidity, 0, 100),
        'wind_speed': np.maximum(1 + np.random.exponential(1.5, n_points), 0.1)
    }, index=timeline)
    
    # Introduce realistic gaps
    gap_rate = 0.01
    random_gaps = np.random.random(n_points) < gap_rate
    
    # Maintenance windows
    for day in range(0, n_days, 7):  # Weekly maintenance
        maint_start = start_date + timedelta(days=day, hours=9)
        maint_end = maint_start + timedelta(hours=1)
        maint_mask = (data.index >= maint_start) & (data.index < maint_end)
        random_gaps |= maint_mask
    
    return data[~random_gaps].copy()


def main():
    """Run the production pipeline demonstration"""
    print("🚀 ETAD Production Pipeline Demonstration")
    print("=" * 60)
    
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available. Please install dependencies.")
        return
    
    # Initialize pipeline
    print("\n🔧 Initializing production pipeline...")
    pipeline = ETADProductionPipeline()
    
    # Generate demo data
    print("\n📊 Generating realistic demo data...")
    demo_data = generate_demo_data(n_days=10)
    print(f"   Generated {len(demo_data):,} data points over {(demo_data.index.max() - demo_data.index.min()).days + 1} days")
    
    # Run complete analysis
    print("\n🔬 Running complete analysis pipeline...")
    try:
        results = pipeline.run_complete_analysis(demo_data)
        
        print("\n📋 Analysis Results Summary:")
        print("=" * 40)
        
        # Print executive summary
        if 'executive_summary' in results:
            summary = results['executive_summary']
            print(f"Data Quality: {summary['data_quality']}")
            print(f"Analysis Modules Completed: {sum(summary['analysis_completeness'].values())}/{len(summary['analysis_completeness'])}")
            
            print(f"\nKey Findings:")
            for finding in summary['key_findings']:
                print(f"  • {finding}")
            
            print(f"\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        # Print detailed results summary
        print(f"\n📊 Detailed Results:")
        for section, section_results in results.items():
            if section == 'metadata':
                continue
            elif section == 'executive_summary':
                continue
            
            has_error = 'error' in str(section_results)
            status = "❌" if has_error else "✅"
            print(f"  {status} {section.replace('_', ' ').title()}")
        
        print(f"\n🎉 Production pipeline completed successfully!")
        print(f"   Full results available in pipeline.results")
        
        # Save results if possible
        try:
            import json
            with open('etad_analysis_results.json', 'w') as f:
                # Convert datetime objects for JSON serialization
                json_safe_results = results.copy()
                json.dump(json_safe_results, f, indent=2, default=str)
            print(f"   Results saved to etad_analysis_results.json")
        except:
            print(f"   Note: Could not save results to file")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
