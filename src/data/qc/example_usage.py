"""
Example Usage of the Quality Control (QC) Module

This script demonstrates how to use the modular QC system for analyzing
aethalometer data quality and mapping filter samples to quality periods.

Author: AethModular Team
Created: 2025-01-12
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import the QC modules
from src.data.qc import (
    MissingDataAnalyzer,
    QualityClassifier, 
    SeasonalPatternAnalyzer,
    FilterSampleMapper,
    QualityVisualizer,
    QualityReportGenerator,
    quick_quality_check
)

# Import convenience functions
from src.data.qc.reports import create_comprehensive_report, quick_assessment


def example_basic_quality_analysis():
    """Example of basic quality analysis using individual modules."""
    print("🔍 Example: Basic Quality Analysis")
    print("=" * 50)
    
    # Load your aethalometer data (replace with actual path)
    # df = pd.read_csv('your_aethalometer_data.csv', index_col='datetime', parse_dates=True)
    
    # For demo, create sample data
    date_range = pd.date_range('2023-01-01', '2023-12-31', freq='min')
    # Simulate some missing data
    missing_indices = np.random.choice(len(date_range), size=int(len(date_range) * 0.05), replace=False)
    available_indices = np.setdiff1d(np.arange(len(date_range)), missing_indices)
    
    df = pd.DataFrame({
        'BC1': np.random.normal(1000, 200, len(available_indices)),
        'BC2': np.random.normal(800, 150, len(available_indices))
    }, index=date_range[available_indices])
    
    print(f"Sample dataset: {len(df)} points from {df.index.min()} to {df.index.max()}")
    
    # 1. Analyze missing data patterns
    print("\n📊 Step 1: Analyzing missing data patterns...")
    missing_analyzer = MissingDataAnalyzer()
    missing_analysis = missing_analyzer.analyze_missing_patterns(df)
    
    # Print basic statistics
    timeline = missing_analysis['timeline']
    print(f"  - Data completeness: {100 - timeline['missing_percentage']:.1f}%")
    print(f"  - Missing points: {timeline['missing_points']:,}")
    print(f"  - Full missing days: {missing_analysis['daily_patterns']['n_full_missing_days']}")
    print(f"  - Partial missing days: {missing_analysis['daily_patterns']['n_partial_missing_days']}")
    
    # 2. Classify quality periods
    print("\n🏷️  Step 2: Classifying quality periods...")
    quality_classifier = QualityClassifier()
    quality_series = quality_classifier.classify_9am_to_9am_periods(missing_analysis)
    
    quality_summary = quality_classifier.get_quality_summary(quality_series)
    print(f"  - Total periods: {quality_summary['total_periods']}")
    print(f"  - High-quality periods: {quality_summary['high_quality_count']} ({quality_summary['usable_percentage']:.1f}%)")
    
    # 3. Analyze seasonal patterns
    print("\n🌍 Step 3: Analyzing seasonal patterns...")
    seasonal_analyzer = SeasonalPatternAnalyzer()
    seasonal_analysis = seasonal_analyzer.analyze_seasonal_missing_patterns(missing_analysis)
    
    print(f"  - Seasons analyzed: {', '.join(seasonal_analysis['seasons'])}")
    print(f"  - Years covered: {', '.join(map(str, seasonal_analysis['years']))}")
    
    # 4. Visualize results
    print("\n📈 Step 4: Creating visualizations...")
    visualizer = QualityVisualizer()
    
    # Plot missing patterns
    visualizer.plot_missing_patterns(missing_analysis)
    
    # Plot quality distribution
    visualizer.plot_quality_distribution(quality_series, "Quality Distribution (9am-to-9am periods)")
    
    # Plot seasonal patterns
    visualizer.plot_seasonal_patterns(seasonal_analysis)
    
    print("✅ Basic quality analysis complete!")
    return {
        'missing_analysis': missing_analysis,
        'quality_series': quality_series,
        'seasonal_analysis': seasonal_analysis
    }


def example_filter_mapping():
    """Example of mapping filter samples to quality periods."""
    print("\n🔬 Example: Filter Sample Mapping")
    print("=" * 50)
    
    # This example assumes you have a database with filter samples
    # db_path = '/path/to/your/filter_database.db'
    
    # For demo, create sample filter data
    filter_dates = pd.date_range('2023-01-15', '2023-12-15', freq='14D')  # Bi-weekly samples
    etad_data = pd.DataFrame({
        'filter_id': range(len(filter_dates)),
        'SampleDate': filter_dates,
        'Fabs': np.random.normal(50, 10, len(filter_dates)),
        'Site': 'ETAD'
    })
    
    ftir_data = etad_data.copy()
    ftir_data.rename(columns={'SampleDate': 'date'}, inplace=True)
    ftir_data['EC_FTIR'] = np.random.normal(5, 1, len(filter_dates))
    ftir_data['OC_FTIR'] = np.random.normal(15, 3, len(filter_dates))
    
    print(f"Sample filter data: {len(etad_data)} samples")
    
    # First run basic analysis to get quality series
    print("Running basic quality analysis first...")
    basic_results = example_basic_quality_analysis()
    quality_series = basic_results['quality_series']
    
    # Map filter samples to quality periods
    print("\n🔗 Mapping filter samples to quality periods...")
    filter_mapper = FilterSampleMapper()
    overlap_results = filter_mapper.map_to_quality_periods(quality_series, etad_data, ftir_data)
    
    # Print mapping results
    total_filters = len(overlap_results['filter_periods'])
    high_quality_overlaps = len(overlap_results['overlap_periods'])
    
    print(f"  - Total filter periods: {total_filters}")
    print(f"  - High-quality overlaps: {high_quality_overlaps} ({high_quality_overlaps/total_filters*100:.1f}%)")
    print(f"  - Excellent overlaps: {len(overlap_results['excellent_overlaps'])}")
    print(f"  - Good overlaps: {len(overlap_results['good_overlaps'])}")
    
    # Export usable periods
    print("\n💾 Exporting quality-filtered data...")
    export_df = filter_mapper.export_quality_filtered_periods(output_path='outputs/filter_quality_demo.csv')
    
    usable_periods = export_df[export_df['usable_for_comparison']].copy()
    print(f"  - Exported {len(export_df)} total periods")
    print(f"  - {len(usable_periods)} periods usable for comparison")
    
    # Visualize overlap results
    visualizer = QualityVisualizer()
    visualizer.plot_filter_overlap_summary(overlap_results)
    
    print("✅ Filter mapping complete!")
    return overlap_results


def example_comprehensive_report():
    """Example of generating a comprehensive quality report."""
    print("\n📋 Example: Comprehensive Quality Report")
    print("=" * 50)
    
    # Create sample data (same as in basic example)
    date_range = pd.date_range('2023-01-01', '2023-12-31', freq='min')
    missing_indices = np.random.choice(len(date_range), size=int(len(date_range) * 0.05), replace=False)
    available_indices = np.setdiff1d(np.arange(len(date_range)), missing_indices)
    
    df = pd.DataFrame({
        'BC1': np.random.normal(1000, 200, len(available_indices)),
        'BC2': np.random.normal(800, 150, len(available_indices))
    }, index=date_range[available_indices])
    
    print(f"Generating comprehensive report for {len(df)} data points...")
    
    # Generate comprehensive report (without database for demo)
    report_results = create_comprehensive_report(
        df=df,
        db_path=None,  # Set to your database path if available
        site_code='DEMO',
        output_dir='outputs/demo_quality_assessment',
        period_type='9am_to_9am'
    )
    
    # Print summary statistics
    summary_stats = report_results['summary_statistics']
    
    print("\n📊 Summary Statistics:")
    print(f"  Data Coverage:")
    print(f"    - Completeness: {summary_stats['data_coverage']['data_completeness_percentage']:.1f}%")
    print(f"    - Missing: {summary_stats['data_coverage']['missing_percentage']:.1f}%")
    
    print(f"  Quality Assessment:")
    print(f"    - Total periods: {summary_stats['quality_assessment']['total_classified_periods']}")
    print(f"    - High-quality: {summary_stats['quality_assessment']['high_quality_periods']} ({summary_stats['quality_assessment']['usable_percentage']:.1f}%)")
    
    print(f"  Temporal Patterns:")
    print(f"    - Full missing days: {summary_stats['temporal_patterns']['full_missing_days']}")
    print(f"    - Partial missing days: {summary_stats['temporal_patterns']['partial_missing_days']}")
    
    print("✅ Comprehensive report generated!")
    print(f"📁 Check 'outputs/demo_quality_assessment/' for exported files")
    
    return report_results


def example_quick_assessment():
    """Example of quick quality assessment."""
    print("\n⚡ Example: Quick Quality Assessment")
    print("=" * 50)
    
    # Create sample data
    date_range = pd.date_range('2023-01-01', '2023-03-31', freq='min')
    missing_indices = np.random.choice(len(date_range), size=int(len(date_range) * 0.03), replace=False)
    available_indices = np.setdiff1d(np.arange(len(date_range)), missing_indices)
    
    df = pd.DataFrame({
        'BC1': np.random.normal(1000, 200, len(available_indices))
    }, index=date_range[available_indices])
    
    # Quick assessment using convenience function
    print("Running quick quality check...")
    quick_quality_check(df)
    
    # More detailed quick assessment
    print("\nRunning detailed quick assessment...")
    quick_summary = quick_assessment(df)
    
    print("✅ Quick assessment complete!")
    return quick_summary


def main():
    """Run all examples."""
    print("🔧 AethModular Quality Control (QC) Module Examples")
    print("=" * 60)
    
    # Ensure output directory exists
    Path('outputs').mkdir(exist_ok=True)
    
    try:
        # Run examples
        print("\n1. Basic Quality Analysis")
        basic_results = example_basic_quality_analysis()
        
        print("\n2. Filter Sample Mapping") 
        filter_results = example_filter_mapping()
        
        print("\n3. Comprehensive Report")
        comprehensive_results = example_comprehensive_report()
        
        print("\n4. Quick Assessment")
        quick_results = example_quick_assessment()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("- Replace sample data with your actual aethalometer data")
        print("- Set up your filter sample database path")
        print("- Customize quality thresholds as needed")
        print("- Use individual modules for specific analysis needs")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure all dependencies are installed and data paths are correct.")


if __name__ == "__main__":
    main()
