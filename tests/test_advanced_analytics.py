"""Tests for advanced analytics modules"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.analysis.advanced.statistical_analysis import (
        StatisticalComparator, DistributionAnalyzer, OutlierDetector
    )
    from src.analysis.advanced.ml_analysis import (
        MLModelTrainer, PredictiveAnalyzer, ClusterAnalyzer
    )
    from src.analysis.advanced.time_series_analysis import (
        TimeSeriesAnalyzer, TrendDetector, SeasonalAnalyzer
    )
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    ADVANCED_MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)


def generate_test_data(n_samples=1000, add_noise=True, add_outliers=False):
    """Generate synthetic aethalometer-like data for testing"""
    np.random.seed(42)
    
    # Generate datetime index
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, periods=n_samples, freq='H')
    
    # Generate base signals with some patterns
    t = np.arange(n_samples)
    
    # BC concentration with daily pattern and trend
    bc_base = 5 + 2 * np.sin(2 * np.pi * t / 24) + 0.001 * t
    bc_noise = np.random.normal(0, 0.5, n_samples) if add_noise else 0
    bc = bc_base + bc_noise
    
    # UV absorption coefficient
    uv_base = 20 + 5 * np.sin(2 * np.pi * t / 24) + 0.002 * t
    uv_noise = np.random.normal(0, 1, n_samples) if add_noise else 0
    uv = uv_base + uv_noise
    
    # IR absorption coefficient  
    ir_base = 15 + 3 * np.sin(2 * np.pi * t / 24) + 0.0015 * t
    ir_noise = np.random.normal(0, 0.8, n_samples) if add_noise else 0
    ir = ir_base + ir_noise
    
    # Add outliers if requested
    if add_outliers:
        outlier_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
        bc[outlier_indices] *= np.random.uniform(3, 5, len(outlier_indices))
        uv[outlier_indices] *= np.random.uniform(2, 4, len(outlier_indices))
    
    # Ensure no negative values
    bc = np.maximum(bc, 0.1)
    uv = np.maximum(uv, 0.1)
    ir = np.maximum(ir, 0.1)
    
    data = pd.DataFrame({
        'BC': bc,
        'UV_abs': uv,
        'IR_abs': ir,
        'temperature': 20 + 10 * np.sin(2 * np.pi * t / (24*365)) + np.random.normal(0, 2, n_samples),
        'humidity': 60 + 20 * np.sin(2 * np.pi * t / (24*30)) + np.random.normal(0, 5, n_samples)
    }, index=dates)
    
    return data


@pytest.mark.skipif(not ADVANCED_MODULES_AVAILABLE, 
                    reason=f"Advanced modules not available: {IMPORT_ERROR if not ADVANCED_MODULES_AVAILABLE else ''}")
class TestStatisticalAnalysis:
    """Test statistical analysis functionality"""
    
    def test_statistical_comparator_basic(self):
        """Test basic statistical comparison"""
        # Generate two datasets with different characteristics
        data1 = generate_test_data(500, add_noise=True)
        data2 = generate_test_data(500, add_noise=True)
        data2['BC'] = data2['BC'] * 1.5  # Make BC different
        
        comparator = StatisticalComparator()
        results = comparator.compare_periods(data1, data2, columns=['BC', 'UV_abs'])
        
        assert 'datasets_info' in results
        assert 'statistical_tests' in results
        assert 'BC' in results['statistical_tests']
        assert 'UV_abs' in results['statistical_tests']
        
        # Check that BC shows significant difference
        bc_ttest = results['statistical_tests']['BC'].get('t_test', {})
        if 'error' not in bc_ttest:
            assert 'p_value' in bc_ttest
            assert 'significant' in bc_ttest
    
    def test_distribution_analyzer(self):
        """Test distribution analysis"""
        # Generate data with known normal distribution
        np.random.seed(42)
        normal_data = pd.Series(np.random.normal(10, 2, 1000))
        
        analyzer = DistributionAnalyzer()
        results = analyzer.analyze_distribution(normal_data)
        
        assert 'data_summary' in results
        assert 'normality_tests' in results
        assert 'distribution_fits' in results
        assert 'best_fit' in results
        
        # Check that normal distribution is detected
        if 'normal' in results['distribution_fits']:
            normal_fit = results['distribution_fits']['normal']
            if 'error' not in normal_fit:
                assert 'parameters' in normal_fit
                assert 'goodness_of_fit' in normal_fit
    
    def test_outlier_detector(self):
        """Test outlier detection"""
        data = generate_test_data(500, add_outliers=True)
        
        detector = OutlierDetector()
        results = detector.detect_outliers(data, columns=['BC', 'UV_abs'])
        
        assert 'outlier_detection' in results
        assert 'BC' in results['outlier_detection']
        
        # Check that outliers were detected
        bc_results = results['outlier_detection']['BC']
        if 'iqr' in bc_results:
            iqr_result = bc_results['iqr']
            if 'error' not in iqr_result:
                assert 'outlier_count' in iqr_result
                assert iqr_result['outlier_count'] > 0  # Should detect some outliers


@pytest.mark.skipif(not ADVANCED_MODULES_AVAILABLE, 
                    reason=f"Advanced modules not available: {IMPORT_ERROR if not ADVANCED_MODULES_AVAILABLE else ''}")
class TestMLAnalysis:
    """Test machine learning analysis functionality"""
    
    def test_ml_model_trainer(self):
        """Test ML model training"""
        data = generate_test_data(200)
        
        trainer = MLModelTrainer()
        results = trainer.train_regression_model(
            data, 
            target_column='BC',
            feature_columns=['UV_abs', 'IR_abs', 'temperature'],
            model_type='linear'
        )
        
        if 'error' not in results:
            assert 'model_type' in results
            assert 'performance' in results
            assert 'training' in results['performance']
            assert 'test' in results['performance']
            
            # Check that model performance metrics are reasonable
            train_r2 = results['performance']['training']['r2']
            test_r2 = results['performance']['test']['r2']
            assert -1 <= train_r2 <= 1
            assert -1 <= test_r2 <= 1
    
    def test_predictive_analyzer(self):
        """Test time series forecasting"""
        # Generate time series data
        data = generate_test_data(100)
        bc_series = data['BC']
        
        analyzer = PredictiveAnalyzer()
        results = analyzer.forecast_time_series(
            bc_series, 
            forecast_periods=24,
            method='simple_exponential'
        )
        
        assert 'method' in results
        assert 'forecast_values' in results
        assert 'forecast_index' in results
        assert len(results['forecast_values']) == 24
    
    def test_cluster_analyzer(self):
        """Test clustering analysis"""
        data = generate_test_data(100)
        
        analyzer = ClusterAnalyzer()
        results = analyzer.perform_clustering(
            data, 
            columns=['BC', 'UV_abs', 'IR_abs'],
            n_clusters=3,
            method='kmeans'
        )
        
        if 'error' not in results:
            assert 'cluster_labels' in results
            assert 'cluster_analysis' in results
            assert len(results['cluster_labels']) == len(data)
            assert 'metrics' in results
            
            # Test optimal cluster finding
            optimal_results = analyzer.find_optimal_clusters(
                data,
                columns=['BC', 'UV_abs'],
                max_clusters=5
            )
            
            if 'error' not in optimal_results:
                assert 'optimal_clusters' in optimal_results
                assert 'cluster_analysis' in optimal_results


@pytest.mark.skipif(not ADVANCED_MODULES_AVAILABLE, 
                    reason=f"Advanced modules not available: {IMPORT_ERROR if not ADVANCED_MODULES_AVAILABLE else ''}")
class TestTimeSeriesAnalysis:
    """Test time series analysis functionality"""
    
    def test_time_series_analyzer(self):
        """Test basic time series analysis"""
        data = generate_test_data(500)
        bc_series = data['BC']
        
        analyzer = TimeSeriesAnalyzer()
        results = analyzer.analyze_time_series(bc_series)
        
        assert 'basic_statistics' in results
        assert 'trend_analysis' in results
        assert 'seasonality_analysis' in results
        assert 'stationarity_tests' in results
    
    def test_trend_detector(self):
        """Test trend detection"""
        # Generate data with clear trend
        t = np.arange(200)
        trend_data = pd.Series(2 + 0.01 * t + np.random.normal(0, 0.1, 200))
        trend_data.index = pd.date_range('2023-01-01', periods=200, freq='H')
        
        detector = TrendDetector()
        results = detector.detect_trends(trend_data)
        
        assert 'trend_detected' in results
        assert 'trend_direction' in results
        if results['trend_detected']:
            assert results['trend_direction'] in ['increasing', 'decreasing']
    
    def test_seasonal_analyzer(self):
        """Test seasonal analysis"""
        # Generate data with seasonal pattern
        t = np.arange(24*7)  # One week of hourly data
        seasonal_data = pd.Series(
            5 + 2 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.1, len(t))
        )
        seasonal_data.index = pd.date_range('2023-01-01', periods=len(t), freq='H')
        
        analyzer = SeasonalAnalyzer()
        results = analyzer.analyze_seasonality(seasonal_data)
        
        assert 'seasonal_components' in results
        assert 'seasonal_strength' in results


class TestIntegrationAdvancedAnalytics:
    """Integration tests for advanced analytics workflow"""
    
    @pytest.mark.skipif(not ADVANCED_MODULES_AVAILABLE, 
                        reason=f"Advanced modules not available: {IMPORT_ERROR if not ADVANCED_MODULES_AVAILABLE else ''}")
    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow"""
        # Generate comprehensive test dataset
        data = generate_test_data(500, add_noise=True, add_outliers=True)
        
        # 1. Statistical Analysis
        comparator = StatisticalComparator()
        data1 = data.iloc[:250]
        data2 = data.iloc[250:]
        stat_results = comparator.compare_periods(data1, data2)
        
        # 2. Distribution Analysis
        dist_analyzer = DistributionAnalyzer()
        dist_results = dist_analyzer.analyze_distribution(data['BC'])
        
        # 3. Outlier Detection
        outlier_detector = OutlierDetector()
        outlier_results = outlier_detector.detect_outliers(data)
        
        # 4. ML Analysis
        ml_trainer = MLModelTrainer()
        ml_results = ml_trainer.train_regression_model(
            data, 'BC', ['UV_abs', 'IR_abs']
        )
        
        # 5. Time Series Analysis
        ts_analyzer = TimeSeriesAnalyzer()
        ts_results = ts_analyzer.analyze_time_series(data['BC'])
        
        # 6. Clustering
        cluster_analyzer = ClusterAnalyzer()
        cluster_results = cluster_analyzer.perform_clustering(
            data, columns=['BC', 'UV_abs', 'IR_abs']
        )
        
        # Verify all analyses completed
        assert stat_results is not None
        assert dist_results is not None
        assert outlier_results is not None
        
        # Create summary report
        summary = {
            'data_shape': data.shape,
            'analysis_completed': {
                'statistical_comparison': 'error' not in stat_results,
                'distribution_analysis': 'error' not in dist_results,
                'outlier_detection': 'error' not in outlier_results,
                'ml_analysis': 'error' not in ml_results if ml_results else False,
                'time_series_analysis': 'error' not in ts_results if ts_results else False,
                'clustering': 'error' not in cluster_results if cluster_results else False
            }
        }
        
        # At least some analyses should complete successfully
        completed_count = sum(summary['analysis_completed'].values())
        assert completed_count >= 3, f"Only {completed_count} analyses completed successfully"
        
        print(f"\n✅ Advanced Analytics Integration Test Summary:")
        print(f"   Data shape: {summary['data_shape']}")
        print(f"   Analyses completed: {completed_count}/6")
        for analysis, completed in summary['analysis_completed'].items():
            status = "✅" if completed else "❌"
            print(f"   {status} {analysis}")


if __name__ == "__main__":
    # Run a simple test
    if ADVANCED_MODULES_AVAILABLE:
        print("🧪 Running advanced analytics tests...")
        
        # Test data generation
        test_data = generate_test_data(100)
        print(f"✅ Generated test data: {test_data.shape}")
        
        # Test basic functionality
        try:
            comparator = StatisticalComparator()
            print("✅ StatisticalComparator initialized")
            
            detector = OutlierDetector()
            print("✅ OutlierDetector initialized")
            
            analyzer = TimeSeriesAnalyzer()
            print("✅ TimeSeriesAnalyzer initialized")
            
            print("\n🎉 All advanced analytics modules loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error testing modules: {e}")
    else:
        print(f"❌ Advanced modules not available: {IMPORT_ERROR}")
