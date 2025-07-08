"""Basic usage example of the modular system"""

import pandas as pd
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.ftir.fabs_ec_analyzer import FabsECAnalyzer
from data.loaders.database import FTIRHIPSLoader
from config.plotting import setup_plotting_style
from utils.file_io import save_results_to_json, ensure_output_directory

def run_basic_analysis(db_path: str = "data/spartan_ftir_hips.db", 
                      site_code: str = "ETAD",
                      output_dir: str = "outputs"):
    """
    Example of how to use the new modular system
    
    Parameters:
    -----------
    db_path : str
        Path to database file
    site_code : str
        Site code to analyze
    output_dir : str
        Output directory for results
    """
    
    # Setup
    setup_plotting_style()
    output_path = ensure_output_directory(output_dir)
    
    print("=== FTIR ANALYSIS USING MODULAR SYSTEM ===")
    print(f"Database: {db_path}")
    print(f"Site: {site_code}")
    print(f"Output: {output_path}")
    print()
    
    try:
        # Load data
        print("Loading data...")
        loader = FTIRHIPSLoader(db_path)
        data = loader.load(site_code)
        
        if data.empty:
            print("No data loaded. Please check database path and site code.")
            return
        
        print(f"Loaded {len(data)} samples")
        print(f"Date range: {data['sample_date'].min()} to {data['sample_date'].max()}")
        print()
        
        # Get data summary
        print("Data summary:")
        summary = loader.get_data_summary(site_code)
        if summary:
            for key, value in summary[0].items():
                print(f"  {key}: {value}")
        print()
        
        # Run analysis
        print("Running Fabs-EC analysis...")
        analyzer = FabsECAnalyzer()
        results = analyzer.analyze(data)
        
        # Display results
        print("=== ANALYSIS RESULTS ===")
        print(f"Total samples: {results['sample_info']['total_samples']}")
        print(f"Valid samples: {results['sample_info']['valid_samples']}")
        print(f"Data coverage: {results['sample_info']['data_coverage']:.2%}")
        print()
        
        print("MAC Statistics:")
        mac_stats = results['mac_statistics']
        print(f"  Mean: {mac_stats.get('mac_mean', 'N/A'):.2f}")
        print(f"  Std: {mac_stats.get('mac_std', 'N/A'):.2f}")
        print(f"  Median: {mac_stats.get('mac_median', 'N/A'):.2f}")
        print(f"  Min: {mac_stats.get('mac_min', 'N/A'):.2f}")
        print(f"  Max: {mac_stats.get('mac_max', 'N/A'):.2f}")
        print()
        
        print("Fabs-EC Correlation:")
        corr = results['correlations']
        print(f"  Pearson r: {corr['pearson_r']:.3f}")
        print(f"  P-value: {corr['pearson_p']:.3e}")
        print(f"  N samples: {corr['n_samples']}")
        print()
        
        # Save results
        output_file = output_path / f"fabs_ec_analysis_{site_code}.json"
        save_results_to_json(results, str(output_file))
        print(f"Results saved to: {output_file}")
        
        # Get MAC values for further analysis
        mac_values = analyzer.get_mac_values(data)
        print(f"\\nMAC values calculated for {len(mac_values)} samples")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_sample_data_demo():
    """
    Create sample data for demonstration when database is not available
    """
    print("=== SAMPLE DATA DEMONSTRATION ===")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'fabs': [10.5, 15.2, 20.8, 25.1, 30.7, 18.3, 22.9, 27.4],
        'ec_ftir': [2.1, 3.0, 4.2, 5.1, 6.3, 3.7, 4.6, 5.5],
        'oc_ftir': [8.4, 12.2, 16.6, 20.0, 24.4, 14.6, 18.3, 21.9],
        'sample_date': pd.date_range('2023-01-01', periods=8, freq='D')
    })
    
    print("Sample data created:")
    print(sample_data)
    print()
    
    # Run analysis
    analyzer = FabsECAnalyzer()
    results = analyzer.analyze(sample_data)
    
    # Display results
    print("=== ANALYSIS RESULTS ===")
    print(f"Valid samples: {results['sample_info']['valid_samples']}")
    print(f"MAC mean: {results['mac_statistics']['mac_mean']:.2f}")
    print(f"MAC std: {results['mac_statistics']['mac_std']:.2f}")
    print(f"Correlation: {results['correlations']['pearson_r']:.3f}")
    print(f"P-value: {results['correlations']['pearson_p']:.3e}")
    
    return results

if __name__ == "__main__":
    # Try to run with actual database, fall back to sample data
    try:
        results = run_basic_analysis()
        if results is None:
            print("\\nFalling back to sample data demonstration...")
            create_sample_data_demo()
    except Exception as e:
        print(f"Error with database analysis: {e}")
        print("\\nRunning sample data demonstration...")
        create_sample_data_demo()
