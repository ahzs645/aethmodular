"""
Complete workflow example: PKL loading -> JPL format -> BC Analysis
Demonstrates the full pipeline for aethalometer data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loaders.aethalometer import AethalometerPKLLoader, load_aethalometer_data
from analysis.bc.black_carbon_analyzer import BlackCarbonAnalyzer, MultiWavelengthBCAnalyzer
from utils.file_io import save_results_to_json, ensure_output_directory
from config.plotting import setup_plotting_style

def run_complete_aethalometer_analysis(
    pkl_file_path: str,
    site_filter: Optional[str] = None,
    output_dir: str = "aeth_analysis_outputs"
) -> Dict[str, Any]:
    """
    Complete aethalometer analysis workflow
    
    Parameters:
    -----------
    pkl_file_path : str
        Path to aethalometer .pkl file
    site_filter : str, optional
        Filter analysis to specific site
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    Dict[str, Any]
        Complete analysis results
    """
    
    print("=" * 60)
    print("AETHALOMETER DATA ANALYSIS WORKFLOW")
    print("=" * 60)
    print()
    
    # Setup output directory
    output_path = ensure_output_directory(output_dir)
    print(f"Output directory: {output_path}")
    print()
    
    try:
        # Step 1: Load and examine the data
        print("STEP 1: Loading and examining data")
        print("-" * 40)
        
        loader = AethalometerPKLLoader(pkl_file_path, format_type="auto")
        
        # Get data summary
        data_summary = loader.get_data_summary(site_filter=site_filter)
        print("Data Summary:")
        for key, value in data_summary.items():
            if key not in ['columns', 'bc_data_availability']:
                print(f"  {key}: {value}")
        
        if 'bc_data_availability' in data_summary:
            print(f"  BC data availability:")
            for col, count in data_summary['bc_data_availability'].items():
                print(f"    {col}: {count} samples")
        print()
        
        # Step 2: Load data in JPL format
        print("STEP 2: Loading data in JPL format")
        print("-" * 40)
        
        df = loader.load(site_filter=site_filter, convert_to_jpl=True)
        print(f"Loaded {len(df)} samples")
        
        # Show available BC columns
        bc_columns = [col for col in df.columns if '.BCc' in col]
        print(f"Available BC measurements: {bc_columns}")
        
        if 'datetime_local' in df.columns:
            print(f"Time range: {df['datetime_local'].min()} to {df['datetime_local'].max()}")
        print()
        
        # Step 3: Basic Black Carbon Analysis
        print("STEP 3: Basic Black Carbon Analysis")
        print("-" * 40)
        
        bc_analyzer = BlackCarbonAnalyzer()
        bc_results = bc_analyzer.analyze(df, time_resolution='hourly', include_trends=True)
        
        # Display key results
        print("Basic Statistics (μg/m³):")
        for bc_type, stats in bc_results['basic_statistics'].items():
            print(f"  {bc_type}:")
            print(f"    Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
            print(f"    Median: {stats['median']:.2f}")
            print(f"    Range: {stats['min']:.2f} - {stats['max']:.2f}")
        print()
        
        # Source apportionment results
        if bc_results['source_apportionment']['available']:
            sa_stats = bc_results['source_apportionment']['statistics']
            print("Source Apportionment:")
            print(f"  Biomass fraction: {sa_stats['mean_biomass_fraction']:.2%}")
            print(f"  Fossil fraction: {sa_stats['mean_fossil_fraction']:.2%}")
            print(f"  Biomass BC: {sa_stats['biomass_contribution']['mean_ug_m3']:.2f} μg/m³")
            print(f"  Fossil BC: {sa_stats['fossil_contribution']['mean_ug_m3']:.2f} μg/m³")
        else:
            print("Source apportionment: Not available (missing Biomass.BCc or Fossil.fuel.BCc)")
        print()
        
        # Trends
        if 'trends' in bc_results:
            print("Trends (per year):")
            for bc_type, trend in bc_results['trends'].items():
                if 'error' not in trend:
                    direction = "↗" if trend['trend_direction'] == 'increasing' else "↘"
                    print(f"  {bc_type}: {direction} {trend['relative_trend_percent_per_year']:.1f}%/year")
        print()
        
        # Step 4: Multi-wavelength Analysis (if available)
        print("STEP 4: Multi-wavelength Analysis")
        print("-" * 40)
        
        multi_wl_analyzer = MultiWavelengthBCAnalyzer()
        multi_wl_results = multi_wl_analyzer.analyze(df)
        
        if 'error' not in multi_wl_results:
            print(f"Available wavelengths: {list(multi_wl_results['available_wavelengths'].keys())}")
            
            spectral_analysis = multi_wl_results['spectral_analysis']
            if spectral_analysis:
                print("Absorption Angstrom Exponent (AAE) results:")
                for pair_name, aae_data in spectral_analysis.items():
                    if 'error' not in aae_data:
                        wl1, wl2 = aae_data['wavelength_pair']
                        print(f"  {wl1}nm - {wl2}nm: AAE = {aae_data['aae_mean']:.2f} ± {aae_data['aae_std']:.2f}")
        else:
            print(f"Multi-wavelength analysis: {multi_wl_results['error']}")
        print()
        
        # Step 5: Create visualizations
        print("STEP 5: Creating visualizations")
        print("-" * 40)
        
        plots_created = create_analysis_plots(df, bc_results, output_path)
        print(f"Created {plots_created} plots in {output_path}")
        print()
        
        # Step 6: Save results
        print("STEP 6: Saving results")
        print("-" * 40)
        
        # Combine all results
        complete_results = {
            'data_summary': data_summary,
            'bc_analysis': bc_results,
            'multi_wavelength_analysis': multi_wl_results,
            'analysis_metadata': {
                'input_file': str(pkl_file_path),
                'site_filter': site_filter,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'total_samples_analyzed': len(df)
            }
        }
        
        # Save to JSON
        results_file = output_path / "aethalometer_analysis_results.json"
        save_results_to_json(complete_results, results_file)
        print(f"Results saved to: {results_file}")
        
        # Save processed data
        data_file = output_path / "processed_aethalometer_data.csv"
        df.to_csv(data_file, index=False)
        print(f"Processed data saved to: {data_file}")
        print()
        
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        
        return complete_results
        
    except Exception as e:
        print(f"ERROR in analysis workflow: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def create_analysis_plots(df: pd.DataFrame, 
                         bc_results: Dict[str, Any], 
                         output_path: Path) -> int:
    """
    Create analysis plots
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed data
    bc_results : Dict[str, Any]
        BC analysis results
    output_path : Path
        Output directory
        
    Returns:
    --------
    int
        Number of plots created
    """
    
    plots_created = 0
    
    try:
        # Set up plotting style
        setup_plotting_style()
        
        # Plot 1: Time series of BC concentrations
        if 'datetime_local' in df.columns:
            bc_columns = [col for col in df.columns if '.BCc' in col]
            if bc_columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for col in bc_columns[:4]:  # Limit to first 4 columns
                    if col in df.columns:
                        valid_data = df[df[col].notna()]
                        ax.plot(valid_data['datetime_local'], valid_data[col], 
                               label=col.replace('.BCc', ''), alpha=0.7)
                
                ax.set_xlabel('Date')
                ax.set_ylabel('BC Concentration (μg/m³)')
                ax.set_title('Black Carbon Time Series')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plot_file = output_path / "bc_timeseries.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plots_created += 1
        
        # Plot 2: BC concentration distributions
        bc_columns = [col for col in df.columns if '.BCc' in col]
        if bc_columns:
            n_cols = min(len(bc_columns), 4)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            for i, col in enumerate(bc_columns[:4]):
                if col in df.columns:
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        axes[i].hist(valid_data, bins=50, alpha=0.7, color=f'C{i}')
                        axes[i].set_xlabel('BC Concentration (μg/m³)')
                        axes[i].set_ylabel('Frequency')
                        axes[i].set_title(f'{col.replace(".BCc", "")} Distribution')
                        axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(bc_columns), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_file = output_path / "bc_distributions.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plots_created += 1
        
        # Plot 3: Source apportionment pie chart
        if (bc_results['source_apportionment']['available'] and 
            'Biomass.BCc' in df.columns and 'Fossil.fuel.BCc' in df.columns):
            
            sa_stats = bc_results['source_apportionment']['statistics']
            biomass_mean = sa_stats['biomass_contribution']['mean_ug_m3']
            fossil_mean = sa_stats['fossil_contribution']['mean_ug_m3']
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            sizes = [biomass_mean, fossil_mean]
            labels = ['Biomass BC', 'Fossil Fuel BC']
            colors = ['#ff9999', '#66b3ff']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            
            ax.set_title('Source Apportionment of Black Carbon')
            
            # Add concentration values
            for i, (label, size) in enumerate(zip(labels, sizes)):
                texts[i].set_text(f'{label}\n({size:.2f} μg/m³)')
            
            plt.tight_layout()
            plot_file = output_path / "source_apportionment.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plots_created += 1
        
        # Plot 4: Hourly patterns (if available)
        if ('datetime_local' in df.columns and 
            'temporal_patterns' in bc_results and 
            bc_results['temporal_patterns']['resolution'] == 'hourly'):
            
            patterns = bc_results['temporal_patterns']['patterns']
            bc_col = None
            
            # Find a BC column with hourly cycle data
            for col, pattern_data in patterns.items():
                if 'hourly_cycle' in pattern_data:
                    bc_col = col
                    break
            
            if bc_col:
                # Calculate hourly means from the data
                df_hourly = df.copy()
                df_hourly['hour'] = df_hourly['datetime_local'].dt.hour
                hourly_means = df_hourly.groupby('hour')[bc_col].mean()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.plot(hourly_means.index, hourly_means.values, 'o-', linewidth=2, markersize=6)
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('BC Concentration (μg/m³)')
                ax.set_title(f'Hourly Pattern - {bc_col.replace(".BCc", "")}')
                ax.set_xticks(range(0, 24, 2))
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_file = output_path / "hourly_pattern.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plots_created += 1
        
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    return plots_created

def example_batch_analysis():
    """
    Example of running analysis on multiple files
    """
    
    print("BATCH ANALYSIS EXAMPLE")
    print("=" * 40)
    
    # Example file list (adjust paths to your actual files)
    file_list = [
        {"path": "df_cleaned_Jacros_hourly.pkl", "site": "Jacros"},
        {"path": "df_cleaned_Central_hourly.pkl", "site": "Central"},
        # Add more files as needed
    ]
    
    batch_results = {}
    
    for file_info in file_list:
        file_path = file_info["path"]
        site_name = file_info["site"]
        
        if Path(file_path).exists():
            print(f"\nAnalyzing {site_name} data from {file_path}...")
            
            try:
                # Run analysis
                results = run_complete_aethalometer_analysis(
                    pkl_file_path=file_path,
                    site_filter=None,  # No additional filtering since file is site-specific
                    output_dir=f"aeth_analysis_{site_name}"
                )
                
                batch_results[site_name] = results
                print(f"✓ Analysis completed for {site_name}")
                
            except Exception as e:
                print(f"✗ Analysis failed for {site_name}: {e}")
                batch_results[site_name] = {"error": str(e)}
        else:
            print(f"✗ File not found: {file_path}")
    
    # Summary of batch results
    print("\nBATCH ANALYSIS SUMMARY")
    print("=" * 40)
    
    for site_name, results in batch_results.items():
        if "error" not in results:
            total_samples = results.get('analysis_metadata', {}).get('total_samples_analyzed', 'Unknown')
            print(f"✓ {site_name}: {total_samples} samples analyzed")
        else:
            print(f"✗ {site_name}: Failed - {results['error']}")

if __name__ == "__main__":
    # Example 1: Single file analysis
    print("Running single file analysis example...")
    
    # Adjust this path to your actual file
    test_file = "df_cleaned_Jacros_hourly.pkl"
    
    if Path(test_file).exists():
        results = run_complete_aethalometer_analysis(
            pkl_file_path=test_file,
            output_dir="example_analysis_output"
        )
    else:
        print(f"Test file {test_file} not found. Please update the path.")
        print("Creating a synthetic example instead...")
        
        # Create synthetic data for demonstration
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        synthetic_data = pd.DataFrame({
            'datetime_local': dates,
            'IR.BCc': np.random.lognormal(2, 0.5, 1000),
            'Biomass.BCc': np.random.lognormal(1.5, 0.6, 1000),
            'Fossil.fuel.BCc': np.random.lognormal(1.8, 0.4, 1000),
            'site': ['SyntheticSite'] * 1000
        })
        
        # Save synthetic data
        synthetic_file = "synthetic_aethalometer_data.pkl"
        synthetic_data.to_pickle(synthetic_file)
        
        # Run analysis on synthetic data
        results = run_complete_aethalometer_analysis(
            pkl_file_path=synthetic_file,
            output_dir="synthetic_analysis_output"
        )
        
        # Clean up
        Path(synthetic_file).unlink()
    
    print("\n" + "="*60)
    print("Example completed! Check the output directory for results and plots.")