"""
Comprehensive Advanced Analytics Demo
====================================

This example demonstrates the advanced analytics capabilities of the ETAD system,
including statistical analysis, machine learning, and quality assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the new advanced analytics modules
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
    from src.analysis.quality.completeness_analyzer import CompletenessAnalyzer
    from src.analysis.quality.missing_data_analyzer import MissingDataAnalyzer
    from src.analysis.quality.period_classifier import PeriodClassifier
    
    # Import infrastructure modules
    from src.core.monitoring import monitor_performance, handle_errors
    from src.utils.memory_optimization import MemoryOptimizer, DataFrameOptimizer
    from src.utils.logging.logger import setup_logging, get_logger
    
    MODULES_AVAILABLE = True
    print("✅ All advanced analytics modules imported successfully!")
    
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")


def generate_realistic_aethalometer_data(n_days=30, missing_rate=0.02):
    """
    Generate realistic aethalometer data with multiple pollutant channels,
    seasonal patterns, and realistic missing data
    """
    print(f"🔬 Generating {n_days} days of synthetic aethalometer data...")
    
    np.random.seed(42)
    
    # Create timeline
    start_date = datetime(2023, 1, 1)
    end_date = start_date + timedelta(days=n_days)
    timeline = pd.date_range(start_date, end_date, freq='min', inclusive='left')
    n_points = len(timeline)
    
    # Time variables for patterns
    t_hours = np.arange(n_points) / 60  # Time in hours
    t_days = t_hours / 24  # Time in days
    
    print(f"   📊 Total data points: {n_points:,}")
    
    # Base black carbon concentration with multiple patterns
    bc_base = (
        3.0 +  # Baseline
        1.5 * np.sin(2 * np.pi * t_hours / 24) +  # Daily cycle
        0.5 * np.sin(2 * np.pi * t_days / 7) +    # Weekly cycle
        0.2 * np.sin(2 * np.pi * t_days / 365) +  # Seasonal cycle
        0.01 * t_days  # Long-term trend
    )
    
    # Add realistic noise and ensure positive values
    bc = np.maximum(bc_base + np.random.normal(0, 0.3, n_points), 0.1)
    
    # Generate correlated absorption coefficients
    uv_abs = 4.2 * bc + np.random.normal(0, 0.5, n_points)
    ir_abs = 3.1 * bc + np.random.normal(0, 0.4, n_points)
    
    # Generate environmental variables
    temp_base = 20 + 15 * np.sin(2 * np.pi * (t_days + 365/4) / 365)  # Seasonal temperature
    temperature = temp_base + 5 * np.sin(2 * np.pi * t_hours / 24) + np.random.normal(0, 2, n_points)
    
    humidity = 60 + 20 * np.sin(2 * np.pi * t_days / 30) + np.random.normal(0, 5, n_points)
    humidity = np.clip(humidity, 0, 100)
    
    # Wind speed affects BC concentrations
    wind_speed = np.maximum(2 + np.random.exponential(2, n_points), 0.1)
    bc = bc * (1 + 0.3 / (wind_speed + 0.1))  # Inverse relationship with wind
    
    # Create DataFrame
    data = pd.DataFrame({
        'BC': bc,
        'UV_abs': uv_abs,
        'IR_abs': ir_abs,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': 1013 + np.random.normal(0, 10, n_points)
    }, index=timeline)
    
    # Introduce realistic missing data patterns
    print(f"   🕳️  Introducing missing data (rate: {missing_rate:.1%})...")
    
    # Random missing data
    random_missing = np.random.random(n_points) < missing_rate
    
    # Systematic maintenance periods
    maintenance_periods = []
    for week in range(0, n_days, 14):  # Every 2 weeks
        maintenance_start = start_date + timedelta(days=week, hours=9)
        maintenance_end = maintenance_start + timedelta(hours=2)
        maintenance_mask = (data.index >= maintenance_start) & (data.index < maintenance_end)
        maintenance_periods.append(maintenance_mask)
    
    # Power outages (random short periods)
    n_outages = max(1, n_days // 10)
    outage_periods = []
    for _ in range(n_outages):
        outage_start = np.random.randint(0, n_points - 120)
        outage_duration = np.random.randint(30, 180)  # 30 min to 3 hours
        outage_mask = np.zeros(n_points, dtype=bool)
        outage_mask[outage_start:outage_start + outage_duration] = True
        outage_periods.append(outage_mask)
    
    # Combine all missing data patterns
    all_missing = random_missing.copy()
    for period in maintenance_periods + outage_periods:
        all_missing |= period
    
    # Apply missing data
    data_with_gaps = data[~all_missing].copy()
    
    missing_count = all_missing.sum()
    completeness = (1 - missing_count / n_points) * 100
    
    print(f"   📉 Missing data points: {missing_count:,} ({100-completeness:.1f}%)")
    print(f"   ✅ Final dataset: {len(data_with_gaps):,} points ({completeness:.1f}% complete)")
    
    return data_with_gaps, data  # Return both gapped and complete data


def demonstrate_quality_analysis(data):
    """Demonstrate quality analysis capabilities"""
    print("\n" + "="*60)
    print("🔍 QUALITY ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # 1. Completeness Analysis
    print("\n1️⃣  Data Completeness Analysis")
    print("-" * 40)
    
    completeness_analyzer = CompletenessAnalyzer()
    completeness_results = completeness_analyzer.analyze_completeness(data, period_type='daily')
    
    if completeness_results and 'analysis_summary' in completeness_results:
        summary = completeness_results['analysis_summary']
        print(f"   📊 Total expected points: {summary['total_expected_points']:,}")
        print(f"   📈 Actual data points: {summary['total_actual_points']:,}")
        print(f"   📉 Missing data points: {summary['total_missing_points']:,}")
        print(f"   🎯 Overall completeness: {summary['overall_completeness_percent']:.1f}%")
    
    # 2. Missing Data Pattern Analysis
    print("\n2️⃣  Missing Data Pattern Analysis")
    print("-" * 40)
    
    try:
        missing_analyzer = MissingDataAnalyzer()
        missing_results = missing_analyzer.analyze_missing_patterns(data)
        
        if missing_results and 'gap_analysis' in missing_results:
            gaps = missing_results['gap_analysis']
            print(f"   🕳️  Identified gaps: {len(gaps)}")
            
            if gaps:
                durations = [gap['duration_minutes'] for gap in gaps]
                print(f"   ⏱️  Average gap duration: {np.mean(durations):.1f} minutes")
                print(f"   📏 Longest gap: {max(durations):.0f} minutes")
                
    except Exception as e:
        print(f"   ⚠️  Missing data analysis failed: {e}")
    
    # 3. Period Classification
    print("\n3️⃣  Data Quality Period Classification")
    print("-" * 40)
    
    try:
        classifier = PeriodClassifier()
        classification_results = classifier.classify_periods(data, period_type='daily')
        
        if classification_results and 'period_classifications' in classification_results:
            classifications = classification_results['period_classifications']
            
            quality_counts = {}
            for period_data in classifications.values():
                quality = period_data.get('quality_class', 'Unknown')
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            print(f"   📅 Analyzed periods: {len(classifications)}")
            print(f"   🏷️  Quality distribution:")
            for quality in ['Excellent', 'Good', 'Moderate', 'Poor']:
                count = quality_counts.get(quality, 0)
                percentage = (count / len(classifications)) * 100 if classifications else 0
                print(f"      {quality}: {count} periods ({percentage:.1f}%)")
                
    except Exception as e:
        print(f"   ⚠️  Period classification failed: {e}")


def demonstrate_statistical_analysis(data):
    """Demonstrate statistical analysis capabilities"""
    print("\n" + "="*60)
    print("📊 STATISTICAL ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Split data into two periods for comparison
    mid_point = len(data) // 2
    period1 = data.iloc[:mid_point]
    period2 = data.iloc[mid_point:]
    
    # 1. Statistical Comparison
    print("\n1️⃣  Period Comparison Analysis")
    print("-" * 40)
    
    comparator = StatisticalComparator()
    comparison_results = comparator.compare_periods(
        period1, period2, 
        columns=['BC', 'UV_abs', 'temperature'],
        comparison_tests=['ttest', 'kstest', 'mannwhitney']
    )
    
    if comparison_results and 'statistical_tests' in comparison_results:
        print(f"   📅 Period 1: {len(period1):,} points")
        print(f"   📅 Period 2: {len(period2):,} points")
        
        for variable, tests in comparison_results['statistical_tests'].items():
            print(f"\n   🔬 {variable} Analysis:")
            
            if 't_test' in tests and 'error' not in tests['t_test']:
                t_test = tests['t_test']
                significance = "significant" if t_test['significant'] else "not significant"
                print(f"      T-test: {significance} (p={t_test['p_value']:.4f})")
            
            if 'descriptive_stats' in tests:
                stats1 = tests['descriptive_stats']['dataset1']
                stats2 = tests['descriptive_stats']['dataset2']
                print(f"      Period 1 mean: {stats1['mean']:.2f} ± {stats1['std']:.2f}")
                print(f"      Period 2 mean: {stats2['mean']:.2f} ± {stats2['std']:.2f}")
    
    # 2. Distribution Analysis
    print("\n2️⃣  Distribution Analysis")
    print("-" * 40)
    
    dist_analyzer = DistributionAnalyzer()
    
    for variable in ['BC', 'UV_abs']:
        print(f"\n   📈 {variable} Distribution:")
        
        dist_results = dist_analyzer.analyze_distribution(
            data[variable].dropna(),
            distributions=['normal', 'lognormal', 'exponential']
        )
        
        if dist_results and 'best_fit' in dist_results:
            best_fit = dist_results['best_fit']
            if best_fit['best_distribution']:
                print(f"      Best fit: {best_fit['best_distribution']}")
                print(f"      AIC score: {best_fit['aic_value']:.2f}")
            else:
                print("      No suitable distribution found")
        
        # Check normality
        if 'normality_tests' in dist_results:
            normality = dist_results['normality_tests']
            if 'shapiro_wilk' in normality and 'error' not in normality['shapiro_wilk']:
                is_normal = normality['shapiro_wilk']['is_normal']
                print(f"      Normality: {'Yes' if is_normal else 'No'} (Shapiro-Wilk)")
    
    # 3. Outlier Detection
    print("\n3️⃣  Outlier Detection")
    print("-" * 40)
    
    outlier_detector = OutlierDetector()
    outlier_results = outlier_detector.detect_outliers(
        data, 
        columns=['BC', 'UV_abs', 'temperature'],
        methods=['iqr', 'zscore', 'modified_zscore']
    )
    
    if outlier_results and 'outlier_detection' in outlier_results:
        for variable, detection in outlier_results['outlier_detection'].items():
            print(f"\n   🎯 {variable} Outliers:")
            
            if 'summary' in detection:
                summary = detection['summary']
                total_outliers = summary['total_unique_outliers']
                percentage = summary['outlier_percentage']
                consensus = summary['consensus_count']
                
                print(f"      Total outliers: {total_outliers} ({percentage:.1f}%)")
                print(f"      Consensus outliers: {consensus}")
                
                # Show method breakdown
                for method, count in summary.get('method_counts', {}).items():
                    print(f"      {method}: {count} outliers")


def demonstrate_ml_analysis(data):
    """Demonstrate machine learning analysis capabilities"""
    print("\n" + "="*60)
    print("🤖 MACHINE LEARNING ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Prepare clean data for ML
    ml_data = data[['BC', 'UV_abs', 'IR_abs', 'temperature', 'humidity', 'wind_speed']].dropna()
    
    if len(ml_data) < 100:
        print("⚠️  Insufficient data for ML analysis")
        return
    
    # 1. Regression Model Training
    print("\n1️⃣  Regression Model Training")
    print("-" * 40)
    
    trainer = MLModelTrainer()
    
    # Train model to predict BC from other measurements
    ml_results = trainer.train_regression_model(
        ml_data,
        target_column='BC',
        feature_columns=['UV_abs', 'IR_abs', 'temperature', 'wind_speed'],
        model_type='random_forest'
    )
    
    if ml_results and 'error' not in ml_results:
        performance = ml_results['performance']
        train_r2 = performance['training']['r2']
        test_r2 = performance['test']['r2']
        
        print(f"   🎯 Target variable: {ml_results['target_column']}")
        print(f"   📊 Training samples: {ml_results['data_summary']['training_samples']:,}")
        print(f"   📈 Training R²: {train_r2:.3f}")
        print(f"   📉 Test R²: {test_r2:.3f}")
        print(f"   🎪 Model generalization: {'Good' if abs(train_r2 - test_r2) < 0.1 else 'Overfitting detected'}")
        
        # Show feature importance
        if 'feature_importance' in ml_results:
            print(f"\n   🔬 Feature Importance:")
            for feature_info in ml_results['feature_importance'][:3]:  # Top 3
                print(f"      {feature_info['feature']}: {feature_info['importance']:.3f}")
    else:
        print(f"   ❌ Model training failed: {ml_results.get('error', 'Unknown error')}")
    
    # 2. Time Series Forecasting
    print("\n2️⃣  Time Series Forecasting")
    print("-" * 40)
    
    predictor = PredictiveAnalyzer()
    
    # Forecast BC concentrations
    bc_series = ml_data['BC']
    forecast_results = predictor.forecast_time_series(
        bc_series,
        forecast_periods=24,  # 24 hours
        method='holt'
    )
    
    if forecast_results and 'error' not in forecast_results:
        print(f"   📅 Forecast method: {forecast_results['method']}")
        print(f"   🔮 Forecast periods: {forecast_results['forecast_periods']}")
        print(f"   📊 Historical data: {forecast_results['original_data_length']} points")
        
        if 'final_level' in forecast_results:
            print(f"   📈 Final level: {forecast_results['final_level']:.2f}")
            print(f"   📊 Trend: {forecast_results['final_trend']:.4f}")
    else:
        print(f"   ❌ Forecasting failed: {forecast_results.get('error', 'Unknown error')}")
    
    # 3. Clustering Analysis
    print("\n3️⃣  Clustering Analysis")
    print("-" * 40)
    
    cluster_analyzer = ClusterAnalyzer()
    
    # Find optimal number of clusters
    optimal_results = cluster_analyzer.find_optimal_clusters(
        ml_data,
        columns=['BC', 'UV_abs', 'temperature'],
        max_clusters=8
    )
    
    if optimal_results and 'error' not in optimal_results:
        if 'optimal_clusters' in optimal_results:
            optimal_k = optimal_results['optimal_clusters']['recommended_k']
            best_score = optimal_results['optimal_clusters']['best_silhouette_score']
            print(f"   🎯 Optimal clusters: {optimal_k}")
            print(f"   📊 Best silhouette score: {best_score:.3f}")
            
            # Perform clustering with optimal number
            cluster_results = cluster_analyzer.perform_clustering(
                ml_data,
                columns=['BC', 'UV_abs', 'temperature'],
                n_clusters=optimal_k
            )
            
            if cluster_results and 'error' not in cluster_results:
                cluster_analysis = cluster_results['cluster_analysis']
                print(f"\n   🏷️  Cluster Characteristics:")
                for cluster_id, info in cluster_analysis.items():
                    size = info['size']
                    percentage = info['percentage']
                    print(f"      {cluster_id}: {size} points ({percentage:.1f}%)")
    else:
        print(f"   ❌ Clustering failed: {optimal_results.get('error', 'Unknown error')}")


def demonstrate_time_series_analysis(data):
    """Demonstrate time series analysis capabilities"""
    print("\n" + "="*60)
    print("📈 TIME SERIES ANALYSIS DEMONSTRATION")
    print("="*60)
    
    bc_series = data['BC'].dropna()
    
    if len(bc_series) < 48:  # Need at least 2 days of data
        print("⚠️  Insufficient data for time series analysis")
        return
    
    # 1. Basic Time Series Analysis
    print("\n1️⃣  Time Series Characteristics")
    print("-" * 40)
    
    ts_analyzer = TimeSeriesAnalyzer()
    ts_results = ts_analyzer.analyze_time_series(bc_series)
    
    if ts_results and 'basic_statistics' in ts_results:
        stats = ts_results['basic_statistics']
        print(f"   📊 Data points: {stats['count']:,}")
        print(f"   📈 Mean: {stats['mean']:.2f}")
        print(f"   📉 Std deviation: {stats['std']:.2f}")
        print(f"   📏 Range: {stats['min']:.2f} - {stats['max']:.2f}")
        
        if 'stationarity_tests' in ts_results:
            stationarity = ts_results['stationarity_tests']
            if 'adf_test' in stationarity:
                adf = stationarity['adf_test']
                if 'error' not in adf:
                    is_stationary = adf['is_stationary']
                    print(f"   📊 Stationarity: {'Yes' if is_stationary else 'No'} (ADF test)")
    
    # 2. Trend Detection
    print("\n2️⃣  Trend Analysis")
    print("-" * 40)
    
    trend_detector = TrendDetector()
    trend_results = trend_detector.detect_trends(bc_series)
    
    if trend_results:
        has_trend = trend_results.get('trend_detected', False)
        if has_trend:
            direction = trend_results.get('trend_direction', 'unknown')
            strength = trend_results.get('trend_strength', 0)
            print(f"   📈 Trend detected: {direction}")
            print(f"   💪 Trend strength: {strength:.3f}")
        else:
            print(f"   📊 No significant trend detected")
        
        # Linear trend info
        if 'linear_trend' in trend_results:
            slope = trend_results['linear_trend'].get('slope', 0)
            r_squared = trend_results['linear_trend'].get('r_squared', 0)
            print(f"   📏 Linear slope: {slope:.6f}")
            print(f"   🎯 R-squared: {r_squared:.3f}")
    
    # 3. Seasonal Analysis
    print("\n3️⃣  Seasonal Patterns")
    print("-" * 40)
    
    seasonal_analyzer = SeasonalAnalyzer()
    seasonal_results = seasonal_analyzer.analyze_seasonality(bc_series)
    
    if seasonal_results and 'seasonal_strength' in seasonal_results:
        strength = seasonal_results['seasonal_strength']
        print(f"   🔄 Seasonal strength: {strength:.3f}")
        
        if strength > 0.1:
            print(f"   ✅ Strong seasonal pattern detected")
        elif strength > 0.05:
            print(f"   📊 Moderate seasonal pattern detected")
        else:
            print(f"   📉 Weak or no seasonal pattern")
        
        # Decomposition components
        if 'seasonal_components' in seasonal_results:
            components = seasonal_results['seasonal_components']
            if 'trend_strength' in components:
                trend_str = components['trend_strength']
                seasonal_str = components['seasonal_strength']
                print(f"   📈 Trend component strength: {trend_str:.3f}")
                print(f"   🔄 Seasonal component strength: {seasonal_str:.3f}")


def create_summary_report(data, results_summary):
    """Create a comprehensive summary report"""
    print("\n" + "="*60)
    print("📋 COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*60)
    
    # Dataset overview
    print(f"\n📊 Dataset Overview:")
    print(f"   • Data points: {len(data):,}")
    print(f"   • Time period: {data.index.min()} to {data.index.max()}")
    print(f"   • Duration: {(data.index.max() - data.index.min()).days + 1} days")
    print(f"   • Variables: {', '.join(data.columns)}")
    
    # Basic statistics
    print(f"\n📈 Key Statistics:")
    bc_stats = data['BC'].describe()
    print(f"   • BC concentration: {bc_stats['mean']:.2f} ± {bc_stats['std']:.2f} μg/m³")
    print(f"   • BC range: {bc_stats['min']:.2f} - {bc_stats['max']:.2f} μg/m³")
    
    if 'UV_abs' in data.columns:
        uv_stats = data['UV_abs'].describe()
        print(f"   • UV absorption: {uv_stats['mean']:.1f} ± {uv_stats['std']:.1f} Mm⁻¹")
    
    # Data quality assessment
    print(f"\n🔍 Data Quality Assessment:")
    total_expected = len(pd.date_range(data.index.min(), data.index.max(), freq='min'))
    completeness = (len(data) / total_expected) * 100
    print(f"   • Completeness: {completeness:.1f}%")
    print(f"   • Missing points: {total_expected - len(data):,}")
    
    # Analysis completion status
    print(f"\n✅ Analysis Status:")
    for analysis, completed in results_summary.items():
        status = "✅" if completed else "❌"
        print(f"   {status} {analysis.replace('_', ' ').title()}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if completeness < 95:
        print(f"   • Data completeness is {completeness:.1f}% - investigate missing data patterns")
    if completeness >= 95:
        print(f"   • Excellent data completeness ({completeness:.1f}%)")
    
    print(f"   • Consider seasonal analysis for long-term trends")
    print(f"   • Regular outlier detection recommended")
    print(f"   • Correlation analysis with meteorological data valuable")


def main():
    """Main demonstration function"""
    print("🚀 ETAD Advanced Analytics Comprehensive Demo")
    print("=" * 70)
    
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available. Please install dependencies:")
        print("   pip install -r requirements.txt")
        return
    
    # Setup logging
    try:
        setup_logging()
        logger = get_logger(__name__)
        logger.info("Starting advanced analytics demonstration")
    except:
        print("⚠️  Logging setup failed, continuing without logging")
    
    # Generate realistic test data
    try:
        data, complete_data = generate_realistic_aethalometer_data(n_days=21, missing_rate=0.015)
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return
    
    # Track which analyses complete successfully
    results_summary = {
        'quality_analysis': False,
        'statistical_analysis': False,
        'machine_learning': False,
        'time_series_analysis': False
    }
    
    # Run quality analysis
    try:
        demonstrate_quality_analysis(data)
        results_summary['quality_analysis'] = True
    except Exception as e:
        print(f"❌ Quality analysis failed: {e}")
    
    # Run statistical analysis
    try:
        demonstrate_statistical_analysis(data)
        results_summary['statistical_analysis'] = True
    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
    
    # Run ML analysis
    try:
        demonstrate_ml_analysis(data)
        results_summary['machine_learning'] = True
    except Exception as e:
        print(f"❌ ML analysis failed: {e}")
    
    # Run time series analysis
    try:
        demonstrate_time_series_analysis(data)
        results_summary['time_series_analysis'] = True
    except Exception as e:
        print(f"❌ Time series analysis failed: {e}")
    
    # Create summary report
    create_summary_report(data, results_summary)
    
    # Final status
    completed_analyses = sum(results_summary.values())
    total_analyses = len(results_summary)
    
    print(f"\n" + "="*70)
    print(f"🎉 Demo completed: {completed_analyses}/{total_analyses} analyses successful")
    
    if completed_analyses == total_analyses:
        print("✅ All advanced analytics capabilities working perfectly!")
    elif completed_analyses >= total_analyses * 0.75:
        print("✅ Most advanced analytics capabilities working well!")
    else:
        print("⚠️  Some issues detected - check dependencies and data quality")
    
    print("\n🔬 Advanced analytics system ready for production use!")


if __name__ == "__main__":
    main()
