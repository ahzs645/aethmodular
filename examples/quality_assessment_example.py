"""
Example usage of the Data Quality Assessment module.

This script demonstrates how to use the quality assessment tools 
to analyze aethalometer data and filter sample compatibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.quality_assessment import (
    DataQualityAssessment,
    QualityVisualization, 
    create_comprehensive_quality_report,
    quick_quality_check
)


def ethiopian_season_mapping(month):
    """Maps month number to Ethiopian season name."""
    if month in [10, 11, 12, 1, 2]:
        return 'Dry Season'
    elif month in [3, 4, 5]:
        return 'Belg Rainy Season'
    else:  # months 6–9
        return 'Kiremt Rainy Season'


def load_sample_aethalometer_data(csv_path):
    """
    Load and prepare aethalometer data for quality assessment.
    
    Parameters:
    -----------
    csv_path : str
        Path to the aethalometer CSV file
        
    Returns:
    --------
    pd.DataFrame
        Prepared DataFrame with datetime index
    """
    print(f"Loading aethalometer data from: {csv_path}")
    
    # Load the data
    df = pd.read_csv(
        csv_path,
        parse_dates=[['Date local (yyyy/MM/dd)', 'Time local (hh:mm:ss)']],
        infer_datetime_format=True
    )
    
    # Rename and set index
    df.rename(columns={'Date local (yyyy/MM/dd)_Time local (hh:mm:ss)': 'datetime_local'}, inplace=True)
    df['datetime_local'] = pd.to_datetime(df['datetime_local'])
    df.set_index('datetime_local', inplace=True)
    
    # Clean index
    df.index = df.index.floor('min')  # Ensure clean 1-min resolution
    df.sort_index(inplace=True)
    
    print(f"Data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
    return df


def example_basic_quality_check():
    """Example of basic quality checking."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Quality Check")
    print("="*60)
    
    # Create sample data with some missing periods
    date_range = pd.date_range('2023-01-01', '2023-01-31', freq='min')
    
    # Simulate missing data (remove some random periods)
    np.random.seed(42)
    missing_indices = np.random.choice(len(date_range), size=1000, replace=False)
    available_indices = np.setdiff1d(range(len(date_range)), missing_indices)
    
    # Create DataFrame
    df = pd.DataFrame(
        {'BC1': np.random.normal(100, 20, len(available_indices))},
        index=date_range[available_indices]
    )
    
    print("Created sample data with missing periods...")
    quick_quality_check(df)


def example_comprehensive_assessment():
    """Example of comprehensive quality assessment."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Comprehensive Quality Assessment")
    print("="*60)
    
    # Create more realistic sample data
    date_range = pd.date_range('2022-01-01', '2022-12-31', freq='min')
    
    # Simulate realistic missing patterns
    np.random.seed(42)
    
    # Full day outages (equipment maintenance)
    full_day_outages = pd.date_range('2022-06-15', '2022-06-17', freq='D')
    full_day_missing = []
    for day in full_day_outages:
        day_minutes = pd.date_range(day, day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), freq='min')
        full_day_missing.extend(day_minutes)
    
    # Random hourly outages (power issues)
    hourly_outages = np.random.choice(len(date_range), size=5000, replace=False)
    
    # Combine missing periods
    all_missing = set(full_day_missing + list(date_range[hourly_outages]))
    available_times = [t for t in date_range if t not in all_missing]
    
    # Create DataFrame with multiple BC columns
    df = pd.DataFrame({
        'BC1': np.random.normal(100, 20, len(available_times)),
        'BC2': np.random.normal(80, 15, len(available_times)),
        'BC3': np.random.normal(120, 25, len(available_times)),
    }, index=available_times)
    
    print(f"Created realistic sample data: {len(df)} records")
    
    # Initialize quality assessment
    qa = DataQualityAssessment(season_mapping_func=ethiopian_season_mapping)
    
    # Run missing data analysis
    print("\n1. Analyzing missing data patterns...")
    missing_analysis = qa.analyze_missing_data(df)
    
    # Classify quality periods
    print("\n2. Classifying data quality periods...")
    quality_daily = qa.classify_data_quality(missing_analysis, period_type='daily')
    quality_9to9 = qa.classify_data_quality(missing_analysis, period_type='9am_to_9am')
    
    print(f"Daily quality periods classified: {len(quality_daily)}")
    print(f"9am-to-9am quality periods classified: {len(quality_9to9)}")
    
    # Analyze seasonal patterns
    print("\n3. Analyzing seasonal patterns...")
    seasonal_analysis = qa.analyze_seasonal_patterns(missing_analysis)
    
    # Create visualizations
    print("\n4. Generating visualizations...")
    viz = QualityVisualization()
    
    # Plot missing patterns
    print("   - Missing data patterns")
    viz.plot_missing_patterns(missing_analysis)
    
    # Plot quality distributions
    print("   - Quality distributions")
    viz.plot_quality_distribution(quality_daily, "Daily Quality Distribution")
    viz.plot_quality_distribution(quality_9to9, "9am-to-9am Quality Distribution")
    
    # Plot seasonal patterns
    print("   - Seasonal patterns")
    viz.plot_seasonal_patterns(seasonal_analysis)
    
    return df, qa, missing_analysis, quality_9to9


def example_filter_sample_integration():
    """Example of integrating with filter sample data."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Filter Sample Integration")
    print("="*60)
    
    # Get data from previous example
    df, qa, missing_analysis, quality_series = example_comprehensive_assessment()
    
    # Create mock filter sample data
    print("Creating mock filter sample data...")
    
    # Sample dates (9am start times)
    sample_dates = pd.date_range('2022-01-15 09:00', '2022-12-15 09:00', freq='30D')
    
    # Mock ETAD data
    etad_data = pd.DataFrame({
        'filter_id': [f'ETAD_{i:03d}' for i in range(len(sample_dates))],
        'SampleDate': sample_dates + pd.Timedelta(days=1),  # Collection date (end of sampling)
        'Fabs': np.random.normal(15, 3, len(sample_dates)),
        'Site': 'ETAD'
    })
    
    # Mock FTIR data (same samples)
    ftir_data = pd.DataFrame({
        'filter_id': etad_data['filter_id'],
        'date': etad_data['SampleDate'],
        'EC_FTIR': np.random.normal(2.5, 0.5, len(sample_dates)),
        'OC_FTIR': np.random.normal(8.0, 1.5, len(sample_dates))
    })
    
    print(f"Created {len(etad_data)} mock filter samples")
    
    # Map filter samples to quality periods
    print("\nMapping filter samples to aethalometer quality...")
    overlap_results = qa.map_filter_samples_to_quality(quality_series, etad_data, ftir_data)
    
    # Export quality-filtered data
    print("\nExporting quality-filtered data...")
    output_path = 'outputs/example_quality_filtered_samples.csv'
    export_df = qa.export_quality_filtered_data(
        overlap_results, 
        output_path=output_path,
        include_all_quality=True
    )
    
    # Create visualization
    print("\nGenerating filter overlap visualization...")
    viz = QualityVisualization()
    viz.plot_filter_overlap_summary(overlap_results)
    
    return overlap_results, export_df


def example_full_workflow_with_real_data():
    """
    Example using the convenience function for full workflow.
    
    This shows how to use the module with real data files.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Full Workflow (Configure for Real Data)")
    print("="*60)
    
    # Example paths - update these for your actual data
    aethalometer_csv = "/path/to/your/aethalometer_data.csv"
    filter_db_path = "/path/to/your/filter_database.db"
    
    print("This example shows how to use the comprehensive workflow function.")
    print("Update the file paths below to use with your actual data:")
    print(f"  Aethalometer CSV: {aethalometer_csv}")
    print(f"  Filter Database: {filter_db_path}")
    
    # Example of how to use with real data (commented out)
    """
    # Load your aethalometer data
    df = load_sample_aethalometer_data(aethalometer_csv)
    
    # Run comprehensive assessment
    results = create_comprehensive_quality_report(
        df=df,
        db_path=filter_db_path,
        site_code='ETAD',
        output_dir='outputs/quality_assessment',
        period_type='9am_to_9am'
    )
    
    # Access results
    quality_series = results['quality_series']
    export_df = results['export_df']
    summary_stats = results['summary_stats']
    
    print(f"Analysis complete! Results saved to outputs/quality_assessment/")
    print(f"Usable filter periods: {summary_stats['usable_filter_periods']}")
    """
    
    print("\nUncomment the code above and update paths to run with real data.")


def main():
    """Run all examples."""
    print("Data Quality Assessment Module - Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_quality_check()
        example_comprehensive_assessment()
        example_filter_sample_integration()
        example_full_workflow_with_real_data()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Update file paths in example_full_workflow_with_real_data()")
        print("2. Run with your actual aethalometer and filter sample data")
        print("3. Use the exported CSV files for further analysis")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all required packages are installed:")
        print("  pip install pandas matplotlib seaborn numpy")


if __name__ == "__main__":
    main()
