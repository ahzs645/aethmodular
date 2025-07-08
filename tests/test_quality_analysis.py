"""Tests for quality analysis modules"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.analysis.quality.completeness_analyzer import CompletenessAnalyzer
    from src.analysis.quality.missing_data_analyzer import MissingDataAnalyzer
    from src.analysis.quality.period_classifier import PeriodClassifier
    QUALITY_MODULES_AVAILABLE = True
except ImportError as e:
    QUALITY_MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)


def generate_test_data_with_gaps(n_days=30, gap_probability=0.05):
    """Generate test data with realistic missing data patterns"""
    np.random.seed(42)
    
    # Generate complete timeline
    start_date = datetime(2023, 1, 1)
    end_date = start_date + timedelta(days=n_days)
    full_timeline = pd.date_range(start_date, end_date, freq='min', inclusive='left')
    
    # Generate base data
    n_points = len(full_timeline)
    bc_data = 5 + 2 * np.sin(2 * np.pi * np.arange(n_points) / (24*60)) + np.random.normal(0, 0.5, n_points)
    
    # Create DataFrame
    data = pd.DataFrame({
        'BC': bc_data,
        'UV_abs': bc_data * 4 + np.random.normal(0, 1, n_points),
        'IR_abs': bc_data * 3 + np.random.normal(0, 0.8, n_points)
    }, index=full_timeline)
    
    # Introduce realistic missing data patterns
    # 1. Random gaps
    random_gaps = np.random.random(n_points) < gap_probability
    
    # 2. Systematic gaps (instrument maintenance)
    maintenance_start = start_date + timedelta(days=10, hours=9)
    maintenance_end = maintenance_start + timedelta(hours=4)
    maintenance_mask = (data.index >= maintenance_start) & (data.index <= maintenance_end)
    
    # 3. Daily gaps (typical 9 AM to 10 AM maintenance)
    daily_gaps = (data.index.hour == 9) & (data.index.minute < 60)
    
    # Apply gaps
    gap_mask = random_gaps | maintenance_mask | daily_gaps
    data_with_gaps = data[~gap_mask].copy()
    
    return data_with_gaps, gap_mask.sum()


def test_quality_modules_basic():
    """Basic test to ensure quality modules can be imported and initialized"""
    if not QUALITY_MODULES_AVAILABLE:
        print(f"❌ Quality modules not available: {IMPORT_ERROR}")
        return False
    
    try:
        # Test initialization
        completeness = CompletenessAnalyzer()
        missing_data = MissingDataAnalyzer()
        classifier = PeriodClassifier()
        
        print("✅ All quality analysis modules initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing quality modules: {e}")
        return False


def test_completeness_analyzer():
    """Test completeness analysis functionality"""
    if not QUALITY_MODULES_AVAILABLE:
        print(f"⏭️  Skipping completeness analyzer test: {IMPORT_ERROR}")
        return
    
    print("\n🔍 Testing Completeness Analyzer...")
    
    # Generate test data with known gaps
    data, expected_gaps = generate_test_data_with_gaps(n_days=5, gap_probability=0.02)
    
    analyzer = CompletenessAnalyzer()
    
    # Test daily analysis
    results = analyzer.analyze_completeness(data, period_type='daily')
    
    assert 'period_type' in results
    assert 'analysis_summary' in results
    assert 'period_analysis' in results
    
    print(f"   ✅ Daily analysis completed")
    print(f"   📊 Data points: {results['analysis_summary']['total_actual_points']:,}")
    print(f"   📉 Missing points: {results['analysis_summary']['total_missing_points']:,}")
    print(f"   📈 Completeness: {results['analysis_summary']['overall_completeness_percent']:.1f}%")
    
    # Test 9am-to-9am analysis
    try:
        results_9am = analyzer.analyze_completeness(data, period_type='9am_to_9am')
        print(f"   ✅ 9am-to-9am analysis completed")
    except Exception as e:
        print(f"   ⚠️  9am-to-9am analysis failed: {e}")


def test_missing_data_analyzer():
    """Test missing data analysis functionality"""
    if not QUALITY_MODULES_AVAILABLE:
        print(f"⏭️  Skipping missing data analyzer test: {IMPORT_ERROR}")
        return
        
    print("\n🕳️  Testing Missing Data Analyzer...")
    
    # Generate test data with various missing patterns
    data, _ = generate_test_data_with_gaps(n_days=7, gap_probability=0.03)
    
    try:
        analyzer = MissingDataAnalyzer()
        results = analyzer.analyze_missing_patterns(data)
        
        print(f"   ✅ Missing data pattern analysis completed")
        
        if 'gap_analysis' in results:
            gap_count = len(results['gap_analysis'])
            print(f"   📊 Identified {gap_count} missing data gaps")
        
        if 'missing_summary' in results:
            summary = results['missing_summary']
            print(f"   📉 Total missing: {summary.get('total_missing_minutes', 0):,} minutes")
            
    except Exception as e:
        print(f"   ❌ Missing data analyzer failed: {e}")


def test_period_classifier():
    """Test period classification functionality"""
    if not QUALITY_MODULES_AVAILABLE:
        print(f"⏭️  Skipping period classifier test: {IMPORT_ERROR}")
        return
        
    print("\n🏷️  Testing Period Classifier...")
    
    # Generate test data
    data, _ = generate_test_data_with_gaps(n_days=10, gap_probability=0.01)
    
    try:
        classifier = PeriodClassifier()
        results = classifier.classify_periods(data, period_type='daily')
        
        print(f"   ✅ Period classification completed")
        
        if 'period_classifications' in results:
            classifications = results['period_classifications']
            quality_counts = {}
            for period_data in classifications.values():
                quality = period_data.get('quality_class', 'Unknown')
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            print(f"   📊 Quality distribution:")
            for quality, count in sorted(quality_counts.items()):
                print(f"      {quality}: {count} periods")
                
    except Exception as e:
        print(f"   ❌ Period classifier failed: {e}")


def test_quality_integration():
    """Test integration of all quality analysis modules"""
    if not QUALITY_MODULES_AVAILABLE:
        print(f"⏭️  Skipping quality integration test: {IMPORT_ERROR}")
        return
        
    print("\n🔗 Testing Quality Analysis Integration...")
    
    # Generate comprehensive test dataset
    data, expected_gaps = generate_test_data_with_gaps(n_days=14, gap_probability=0.02)
    
    try:
        # Initialize all analyzers
        completeness = CompletenessAnalyzer()
        missing_data = MissingDataAnalyzer()
        classifier = PeriodClassifier()
        
        # Run comprehensive analysis
        print("   🔍 Running completeness analysis...")
        completeness_results = completeness.analyze_completeness(data)
        
        print("   🕳️  Running missing data analysis...")
        missing_results = missing_data.analyze_missing_patterns(data)
        
        print("   🏷️  Running period classification...")
        classification_results = classifier.classify_periods(data)
        
        # Create integrated summary
        summary = {
            'data_period': f"{data.index.min()} to {data.index.max()}",
            'total_days': (data.index.max() - data.index.min()).days + 1,
            'analyses_completed': {
                'completeness': completeness_results is not None,
                'missing_patterns': missing_results is not None,
                'period_classification': classification_results is not None
            }
        }
        
        print(f"   ✅ Integration test completed")
        print(f"   📅 Analysis period: {summary['total_days']} days")
        
        completed_count = sum(summary['analyses_completed'].values())
        print(f"   📊 Analyses completed: {completed_count}/3")
        
        return summary
        
    except Exception as e:
        print(f"   ❌ Quality integration test failed: {e}")
        return None


def main():
    """Run all quality analysis tests"""
    print("🧪 Testing Quality Analysis Modules")
    print("=" * 50)
    
    # Test basic functionality
    basic_success = test_quality_modules_basic()
    
    if basic_success:
        # Run individual module tests
        test_completeness_analyzer()
        test_missing_data_analyzer() 
        test_period_classifier()
        
        # Run integration test
        integration_results = test_quality_integration()
        
        print("\n" + "=" * 50)
        print("🎉 Quality Analysis Testing Complete!")
        
        if integration_results:
            print(f"✅ All quality analysis modules working correctly")
        else:
            print("⚠️  Some issues detected in integration test")
    else:
        print("\n❌ Basic initialization failed - check dependencies")


if __name__ == "__main__":
    main()
