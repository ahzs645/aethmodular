"""
Comprehensive example demonstrating all the enhanced modular features

This example shows how to use:
1. Aethalometer smoothening methods (ONA, CMA, DEMA)
2. 9AM-to-9AM period processing
3. Enhanced MAC calculation methods (all 4 methods)
4. Seasonal analysis for Ethiopian climate
5. Visualization capabilities

Run this after installing dependencies:
pip install pandas numpy scipy scikit-learn matplotlib seaborn
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import our modular components
from analysis.aethalometer.smoothening import ONASmoothing, CMASmoothing, DEMASmoothing
from analysis.aethalometer.period_processor import NineAMPeriodProcessor
from analysis.ftir.enhanced_mac_analyzer import EnhancedMACAnalyzer
from analysis.seasonal.ethiopian_seasons import EthiopianSeasonAnalyzer
from visualization.time_series import TimeSeriesPlotter

def create_sample_aethalometer_data(n_days=30):
    """Create realistic sample aethalometer data for demonstration"""
    
    # Create timestamp range (1-minute resolution)
    start_date = datetime(2023, 1, 1)
    timestamps = pd.date_range(start_date, periods=n_days*24*60, freq='1min')
    
    # Create realistic BC data with diurnal patterns and noise
    hours = np.array([t.hour + t.minute/60 for t in timestamps])
    
    # Base diurnal pattern (higher during traffic hours)
    diurnal_pattern = (
        2.0 +  # baseline
        3.0 * np.exp(-(hours - 8)**2 / 8) +  # morning peak
        2.5 * np.exp(-(hours - 18)**2 / 12) +  # evening peak
        1.0 * np.sin(2 * np.pi * hours / 24)  # general daily cycle
    )
    
    # Add seasonal pattern
    day_of_year = np.array([t.timetuple().tm_yday for t in timestamps])
    seasonal_pattern = 1.5 * np.sin(2 * np.pi * day_of_year / 365) + 2.0
    
    # Combine patterns and add noise
    bc_base = diurnal_pattern * seasonal_pattern
    noise = np.random.normal(0, 0.3, len(timestamps))  # measurement noise
    bc_values = bc_base + noise
    
    # Add some negative values (instrument artifacts)
    negative_mask = np.random.random(len(bc_values)) < 0.02
    bc_values[negative_mask] = np.random.normal(-0.5, 0.2, negative_mask.sum())
    
    # Create ATN values (correlated with BC but with different noise)
    atn_values = np.cumsum(bc_values * 0.1 + np.random.normal(0, 0.05, len(bc_values)))
    
    # Add some missing data
    missing_mask = np.random.random(len(timestamps)) < 0.05  # 5% missing
    bc_values[missing_mask] = np.nan
    atn_values[missing_mask] = np.nan
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'IR BCc': bc_values,  # Using IR wavelength as example
        'IR ATN1': atn_values,
        'UV BCc': bc_values * 0.9 + np.random.normal(0, 0.1, len(bc_values)),
        'UV ATN1': atn_values * 0.95 + np.random.normal(0, 0.02, len(atn_values))
    })
    
    return data

def create_sample_ftir_data(n_samples=200):
    """Create realistic sample FTIR data for MAC analysis"""
    
    # Create realistic FTIR data
    # EC values (μg/m³)
    ec_values = np.random.lognormal(mean=1.5, sigma=0.8, size=n_samples)
    ec_values = np.clip(ec_values, 0.1, 20)  # Realistic range
    
    # Fabs values - correlated with EC with realistic MAC
    true_mac = 10.0  # True MAC value (m²/g)
    fabs_values = true_mac * ec_values + np.random.normal(0, 2, n_samples)
    fabs_values = np.clip(fabs_values, 0, None)  # No negative absorption
    
    # Add some outliers
    outlier_mask = np.random.random(n_samples) < 0.05
    fabs_values[outlier_mask] *= np.random.uniform(2, 5, outlier_mask.sum())
    
    # Create timestamps spanning different seasons
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
    
    data = pd.DataFrame({
        'timestamp': dates,
        'ec_ftir': ec_values,
        'fabs': fabs_values,
        'oc_ftir': ec_values * np.random.uniform(1.5, 4.0, n_samples),  # OC for additional analysis
        'sample_id': [f'ETAD_{i:03d}' for i in range(n_samples)]
    })
    
    return data.sort_values('timestamp').reset_index(drop=True)

def demonstrate_smoothening_methods():
    """Demonstrate aethalometer smoothening methods"""
    print("=" * 60)
    print("1. AETHALOMETER SMOOTHENING METHODS DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    aeth_data = create_sample_aethalometer_data(n_days=7)  # 1 week of data
    print(f"Created sample aethalometer data: {len(aeth_data)} measurements")
    print(f"Data range: {aeth_data['timestamp'].min()} to {aeth_data['timestamp'].max()}")
    
    # Initialize smoothening methods
    ona = ONASmoothing(delta_atn_threshold=0.05)
    cma = CMASmoothing(window_size=15)
    dema = DEMASmoothing(alpha=0.2)
    
    # Apply each method
    wavelength = 'IR'
    print(f"\nApplying smoothening methods to {wavelength} wavelength...")
    
    results = {}
    
    # ONA Smoothening
    print("\nONA Smoothening:")
    ona_results = ona.analyze(aeth_data, wavelength)
    results['ONA'] = ona_results
    print(f"  - Valid samples: {ona_results['sample_info']['valid_samples']}")
    print(f"  - Data completeness: {ona_results['sample_info']['data_completeness']:.1f}%")
    print(f"  - Noise reduction: {ona_results['improvement_metrics']['noise_reduction_percent']:.1f}%")
    print(f"  - Negative reduction: {ona_results['improvement_metrics']['negative_reduction']} values")
    
    # CMA Smoothening  
    print("\nCMA Smoothening:")
    cma_results = cma.analyze(aeth_data, wavelength)
    results['CMA'] = cma_results
    print(f"  - Valid samples: {cma_results['sample_info']['valid_samples']}")
    print(f"  - Data completeness: {cma_results['sample_info']['data_completeness']:.1f}%")
    print(f"  - Noise reduction: {cma_results['improvement_metrics']['noise_reduction_percent']:.1f}%")
    print(f"  - Negative reduction: {cma_results['improvement_metrics']['negative_reduction']} values")
    
    # DEMA Smoothening
    print("\nDEMA Smoothening:")
    dema_results = dema.analyze(aeth_data, wavelength)
    results['DEMA'] = dema_results
    print(f"  - Valid samples: {dema_results['sample_info']['valid_samples']}")
    print(f"  - Data completeness: {dema_results['sample_info']['data_completeness']:.1f}%")
    print(f"  - Noise reduction: {dema_results['improvement_metrics']['noise_reduction_percent']:.1f}%")
    print(f"  - Lag metric: {dema_results['improvement_metrics']['lag_metric']:.2f}")
    
    # Compare methods
    print("\nSmoothing Method Comparison:")
    print("-" * 50)
    print("Method  | Noise Reduction | Correlation | Negatives Removed")
    print("-" * 50)
    for method, result in results.items():
        metrics = result['improvement_metrics']
        print(f"{method:7} | {metrics['noise_reduction_percent']:13.1f}% | "
              f"{metrics['correlation_with_original']:10.3f} | "
              f"{metrics['negative_reduction']:14d}")
    
    return results, aeth_data

def demonstrate_period_processing():
    """Demonstrate 9AM-to-9AM period processing"""
    print("\n\n" + "=" * 60)
    print("2. 9AM-TO-9AM PERIOD PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # Create longer dataset for period analysis
    aeth_data = create_sample_aethalometer_data(n_days=15)  # 2 weeks
    print(f"Created sample data: {len(aeth_data)} measurements over 15 days")
    
    # Initialize period processor
    processor = NineAMPeriodProcessor()
    
    # Process periods
    print("\nProcessing 9AM-to-9AM periods...")
    period_results = processor.analyze(aeth_data, date_column='timestamp')
    
    # Display results
    print(f"\nPeriod Processing Results:")
    print(f"- Total periods identified: {period_results['processing_info']['total_periods']}")
    print(f"- Date range: {period_results['processing_info']['date_range']['start']} to "
          f"{period_results['processing_info']['date_range']['end']}")
    
    # Quality summary
    summary = period_results['summary_statistics']
    print(f"\nQuality Distribution:")
    print(f"- Excellent periods: {summary['quality_distribution']['excellent']['count']} "
          f"({summary['quality_distribution']['excellent']['percentage']:.1f}%)")
    print(f"- Good periods: {summary['quality_distribution']['good']['count']} "
          f"({summary['quality_distribution']['good']['percentage']:.1f}%)")
    print(f"- Poor periods: {summary['quality_distribution']['poor']['count']} "
          f"({summary['quality_distribution']['poor']['percentage']:.1f}%)")
    
    print(f"\nData Completeness Statistics:")
    comp_stats = summary['completeness_statistics']
    print(f"- Mean completeness: {comp_stats['mean_completeness']:.1f}%")
    print(f"- Range: {comp_stats['min_completeness']:.1f}% - {comp_stats['max_completeness']:.1f}%")
    
    # Show individual periods
    print(f"\nIndividual Period Details:")
    print("-" * 70)
    print("Period | Date       | Quality   | Completeness | Missing Minutes")
    print("-" * 70)
    for classification in period_results['period_classifications'][:5]:  # Show first 5
        print(f"{classification['period_id']:6d} | {classification['date_label']} | "
              f"{classification['quality']:9s} | "
              f"{classification['data_completeness']['completeness_percentage']:11.1f}% | "
              f"{classification['data_completeness']['missing_minutes']:13.0f}")
    
    if len(period_results['period_classifications']) > 5:
        print(f"... and {len(period_results['period_classifications']) - 5} more periods")
    
    return period_results

def demonstrate_enhanced_mac_analysis():
    """Demonstrate enhanced MAC calculation methods"""
    print("\n\n" + "=" * 60)
    print("3. ENHANCED MAC CALCULATION METHODS DEMONSTRATION")
    print("=" * 60)
    
    # Create sample FTIR data
    ftir_data = create_sample_ftir_data(n_samples=150)
    print(f"Created sample FTIR data: {len(ftir_data)} samples")
    print(f"EC range: {ftir_data['ec_ftir'].min():.2f} - {ftir_data['ec_ftir'].max():.2f} μg/m³")
    print(f"Fabs range: {ftir_data['fabs'].min():.1f} - {ftir_data['fabs'].max():.1f} Mm⁻¹")
    
    # Initialize enhanced MAC analyzer
    mac_analyzer = EnhancedMACAnalyzer()
    
    # Perform analysis
    print("\nPerforming enhanced MAC analysis with all 4 methods...")
    mac_results = mac_analyzer.analyze(ftir_data)
    
    # Display results
    print(f"\nSample Information:")
    info = mac_results['sample_info']
    print(f"- Valid samples: {info['valid_samples']} / {info['total_samples']} "
          f"({info['data_completeness']:.1f}%)")
    
    print(f"\nMAC Calculation Results:")
    print("-" * 80)
    print("Method | Description                        | MAC Value | R² / Std  | Notes")
    print("-" * 80)
    
    for method_key, method_name in mac_results['mac_methods'].items():
        result = mac_results['mac_results'][method_key]
        
        if 'error' in result:
            print(f"{method_key} | {method_name:34s} | Error: {result['error']}")
            continue
        
        mac_val = result['mac_value']
        
        if 'r_squared' in result:
            metric = f"R²={result['r_squared']:.3f}"
        elif 'mac_std' in result:
            metric = f"σ={result['mac_std']:.2f}"
        else:
            metric = "N/A"
        
        # Add notes about method characteristics
        notes = ""
        if method_key == 'method_3' and 'intercept' in result:
            notes = f"int={result['intercept']:.2f}"
        elif method_key == 'method_4':
            notes = "zero int"
        
        print(f"{method_key} | {method_name:34s} | {mac_val:8.2f}  | {metric:9s} | {notes}")
    
    # Physical constraint analysis
    print(f"\nPhysical Constraint Analysis:")
    constraints = mac_results['physical_constraints']
    overall = constraints['overall_assessment']
    print(f"- Overall constraint compliance: {'PASS' if overall['passes_constraint'] else 'FAIL'}")
    print(f"- Total negative predictions: {overall['total_negative_predictions']}")
    print(f"- Recommended method: {overall['recommended_for_constraint']}")
    
    # Method comparison and recommendations
    print(f"\nMethod Comparison & Recommendations:")
    comparison = mac_results['method_comparison']
    
    if 'mac_statistics' in comparison:
        stats = comparison['mac_statistics']
        print(f"- MAC value range: {stats['min']:.2f} - {stats['max']:.2f} m²/g")
        print(f"- Coefficient of variation: {stats['coefficient_of_variation']:.1f}%")
    
    if 'recommendations' in comparison:
        rec = comparison['recommendations']
        print(f"- Best overall method: {rec['best_overall']}")
        print(f"- Rationale: {rec['rationale']}")
    
    return mac_results

def demonstrate_seasonal_analysis():
    """Demonstrate Ethiopian seasonal analysis"""
    print("\n\n" + "=" * 60)
    print("4. ETHIOPIAN SEASONAL ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data spanning multiple seasons
    ftir_data = create_sample_ftir_data(n_samples=300)
    print(f"Created sample data: {len(ftir_data)} samples across seasons")
    
    # Initialize seasonal analyzer
    seasonal_analyzer = EthiopianSeasonAnalyzer()
    
    # Perform seasonal analysis
    print("\nPerforming Ethiopian seasonal analysis...")
    seasonal_results = seasonal_analyzer.analyze(
        ftir_data, 
        date_column='timestamp',
        target_columns=['fabs', 'ec_ftir', 'oc_ftir']
    )
    
    # Display results
    print(f"\nSeasonal Analysis Results:")
    data_info = seasonal_results['data_info']
    print(f"- Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}")
    print(f"- Years covered: {', '.join(map(str, data_info['years_covered']))}")
    
    # Seasonal statistics
    print(f"\nSeasonal Statistics Summary:")
    print("-" * 90)
    print("Season                    | Samples | Fabs Mean | EC Mean | OC Mean | Notes")
    print("-" * 90)
    
    for season_key, season_data in seasonal_results['seasonal_statistics'].items():
        if 'error' in season_data:
            print(f"{season_data['name']:25s} | Error: {season_data['error']}")
            continue
        
        stats = season_data['statistics']
        fabs_mean = stats.get('fabs', {}).get('mean', 0)
        ec_mean = stats.get('ec_ftir', {}).get('mean', 0)
        oc_mean = stats.get('oc_ftir', {}).get('mean', 0)
        
        print(f"{season_data['name']:25s} | {season_data['sample_count']:7d} | "
              f"{fabs_mean:8.1f}  | {ec_mean:7.2f}  | {oc_mean:7.1f}  | "
              f"{', '.join(map(str, season_data['months']))}")
    
    # Seasonal MAC analysis
    if 'mac_analysis' in seasonal_results and 'error' not in seasonal_results['mac_analysis']:
        print(f"\nSeasonal MAC Analysis:")
        mac_analysis = seasonal_results['mac_analysis']
        
        print("-" * 75)
        print("Season                    | MAC (ind) | MAC (ratio) | MAC (reg) | R²")
        print("-" * 75)
        
        for season_key, mac_data in mac_analysis.items():
            if season_key == 'seasonal_comparison' or 'error' in mac_data:
                continue
            
            ind_mac = mac_data['individual_mac']['mean']
            ratio_mac = mac_data['ratio_mac']
            reg_mac = mac_data['regression_mac']['slope']
            r_squared = mac_data['regression_mac']['r_squared']
            
            print(f"{mac_data['name']:25s} | {ind_mac:8.2f}  | {ratio_mac:10.2f}  | "
                  f"{reg_mac:8.2f}  | {r_squared:.3f}")
        
        # Seasonal comparison
        if 'seasonal_comparison' in mac_analysis:
            comp = mac_analysis['seasonal_comparison']
            if 'individual_mac_comparison' in comp:
                stats = comp['individual_mac_comparison']['statistics']
                print(f"\nSeasonal MAC Variability:")
                print(f"- MAC coefficient of variation: {stats['cv_percent']:.1f}%")
                print(f"- MAC range: {stats['min']:.2f} - {stats['max']:.2f} m²/g")
    
    # Climate insights
    if 'climate_analytics' in seasonal_results:
        analytics = seasonal_results['climate_analytics']
        if 'climate_insights' in analytics:
            insights = analytics['climate_insights']
            print(f"\nClimate-Specific Insights:")
            for i, insight in enumerate(insights[:3], 1):  # Show first 3 insights
                print(f"{i}. {insight}")
    
    return seasonal_results

def demonstrate_visualization():
    """Demonstrate visualization capabilities"""
    print("\n\n" + "=" * 60)
    print("5. VISUALIZATION CAPABILITIES DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    aeth_data = create_sample_aethalometer_data(n_days=7)
    
    # Initialize plotter
    plotter = TimeSeriesPlotter()
    
    print("Creating visualization examples...")
    print("(Note: In a real environment, these would display as plots)")
    
    # Demonstrate diurnal patterns
    print("\n1. Diurnal Pattern Analysis:")
    print("   - Would show 24-hour patterns in BC concentrations")
    print("   - Missing data analysis by hour of day")
    print("   - Statistical summaries with error bars")
    
    # Try to create the plot (will work if matplotlib is available)
    try:
        fig = plotter.plot_diurnal_patterns(
            aeth_data, 
            date_column='timestamp',
            value_columns=['IR BCc'],
            missing_data_analysis=True
        )
        print("   ✓ Diurnal pattern plot created successfully")
        plt.close(fig)  # Close to prevent display in non-interactive mode
    except Exception as e:
        print(f"   ⚠ Plot creation skipped: {str(e)[:50]}...")
    
    print("\n2. Weekly Heatmap Analysis:")
    print("   - Would show day-of-week × hour patterns")
    print("   - Missing data visualization")
    print("   - Color-coded intensity maps")
    
    print("\n3. Seasonal Heatmap Analysis:")
    print("   - Would show month × year patterns")
    print("   - Long-term trend visualization")
    print("   - Data coverage assessment")
    
    print("\n4. Smoothening Comparison Plots:")
    print("   - Would show original vs smoothed data")
    print("   - Side-by-side method comparison")
    print("   - Improvement metrics visualization")
    
    return True

def main():
    """Main demonstration function"""
    print("ENHANCED MODULAR AETHALOMETER & FTIR ANALYSIS SYSTEM")
    print("Comprehensive Demonstration of New Features")
    print("=" * 80)
    
    try:
        # 1. Smoothening methods
        smoothening_results, aeth_data = demonstrate_smoothening_methods()
        
        # 2. Period processing
        period_results = demonstrate_period_processing()
        
        # 3. Enhanced MAC analysis
        mac_results = demonstrate_enhanced_mac_analysis()
        
        # 4. Seasonal analysis
        seasonal_results = demonstrate_seasonal_analysis()
        
        # 5. Visualization
        viz_success = demonstrate_visualization()
        
        # Summary
        print("\n\n" + "=" * 60)
        print("DEMONSTRATION SUMMARY")
        print("=" * 60)
        print("✓ Aethalometer smoothening methods (ONA, CMA, DEMA)")
        print("✓ 9AM-to-9AM period processing with quality classification")
        print("✓ Enhanced MAC calculation (4 different methods)")
        print("✓ Ethiopian seasonal analysis with climate insights")
        print("✓ Comprehensive visualization capabilities")
        
        print("\nAll enhanced features have been successfully demonstrated!")
        print("Your modular system now includes:")
        print("- Advanced data processing algorithms")
        print("- Quality assessment and classification")
        print("- Multiple analysis approaches for robustness")
        print("- Climate-specific seasonal patterns")
        print("- Rich visualization options")
        
        print("\nNext steps:")
        print("1. Integrate with your actual data sources")
        print("2. Customize parameters for your specific use case")
        print("3. Add machine learning components as needed")
        print("4. Implement chemical interference analysis")
        print("5. Develop automated reporting features")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("This might be due to missing dependencies.")
        print("Install required packages: pip install pandas numpy scipy scikit-learn matplotlib seaborn")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Demonstration completed successfully!")
    else:
        print("\n❌ Demonstration encountered issues.")
