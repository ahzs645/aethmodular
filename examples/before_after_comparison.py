"""
Before and After: Migration Example

This script demonstrates the difference between the old monolithic approach
and the new modular approach using the same data and analysis.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import new modular components
from analysis.ftir.fabs_ec_analyzer import FabsECAnalyzer
from analysis.ftir.oc_ec_analyzer import OCECAnalyzer
from data.loaders.database import FTIRHIPSLoader
from utils.file_io import save_results_to_json

def create_sample_data():
    """Create consistent sample data for comparison"""
    np.random.seed(42)  # For reproducible results
    
    n_samples = 50
    
    # Create realistic FTIR data
    ec_base = np.random.exponential(3, n_samples)
    oc_base = ec_base * np.random.normal(4, 1, n_samples)  # OC typically higher than EC
    fabs_base = ec_base * np.random.normal(8, 2, n_samples)  # Fabs related to EC
    
    # Add some noise and ensure positive values
    data = pd.DataFrame({
        'ec_ftir': np.maximum(ec_base + np.random.normal(0, 0.1, n_samples), 0.1),
        'oc_ftir': np.maximum(oc_base + np.random.normal(0, 0.5, n_samples), 0.1),
        'fabs': np.maximum(fabs_base + np.random.normal(0, 1, n_samples), 1.0),
        'volume_m3': np.random.normal(24, 2, n_samples),
        'sample_date': pd.date_range('2023-01-01', periods=n_samples, freq='D')
    })
    
    # Add some missing values to test robustness
    data.loc[data.sample(5).index, 'ec_ftir'] = np.nan
    data.loc[data.sample(3).index, 'oc_ftir'] = np.nan
    data.loc[data.sample(4).index, 'fabs'] = np.nan
    
    return data

def old_monolithic_approach(data):
    """
    This simulates the old approach: large, monolithic functions
    """
    print("=== OLD MONOLITHIC APPROACH ===")
    
    # Simulate old analyze_fabs_ec_relationship function
    def analyze_fabs_ec_relationship_old(df):
        """Old-style monolithic function (simplified version)"""
        results = {}
        
        # Data validation (embedded in function)
        if 'fabs' not in df.columns or 'ec_ftir' not in df.columns:
            print("Required columns missing")
            return {}
        
        # Remove invalid values (logic embedded)
        valid_mask = (df['fabs'] > 0) & (df['ec_ftir'] > 0) & \
                    df['fabs'].notna() & df['ec_ftir'].notna()
        
        if valid_mask.sum() < 3:
            print("Insufficient data")
            return {}
        
        clean_data = df[valid_mask]
        
        # Calculate MAC (embedded logic)
        mac = clean_data['fabs'] / clean_data['ec_ftir']
        
        # Statistics (embedded calculations)
        results['mac_mean'] = mac.mean()
        results['mac_std'] = mac.std()
        results['mac_median'] = mac.median()
        results['mac_min'] = mac.min()
        results['mac_max'] = mac.max()
        results['n_samples'] = len(clean_data)
        
        # Correlation (embedded)
        from scipy.stats import pearsonr
        try:
            r, p = pearsonr(clean_data['fabs'], clean_data['ec_ftir'])
            results['correlation'] = r
            results['p_value'] = p
        except:
            results['correlation'] = np.nan
            results['p_value'] = np.nan
        
        return results
    
    # Run old analysis
    old_results = analyze_fabs_ec_relationship_old(data)
    
    print(f"Valid samples: {old_results.get('n_samples', 0)}")
    print(f"MAC mean: {old_results.get('mac_mean', 'N/A'):.3f}")
    print(f"MAC std: {old_results.get('mac_std', 'N/A'):.3f}")
    print(f"Correlation: {old_results.get('correlation', 'N/A'):.3f}")
    print(f"P-value: {old_results.get('p_value', 'N/A'):.3e}")
    
    return old_results

def new_modular_approach(data):
    """
    This demonstrates the new modular approach
    """
    print("\\n=== NEW MODULAR APPROACH ===")
    
    # Use the new modular system
    analyzer = FabsECAnalyzer()
    results = analyzer.analyze(data)
    
    # Extract results (now with rich structure)
    sample_info = results['sample_info']
    mac_stats = results['mac_statistics']
    correlations = results['correlations']
    
    print(f"Valid samples: {sample_info['valid_samples']}")
    print(f"Data coverage: {sample_info['data_coverage']:.1%}")
    print(f"MAC mean: {mac_stats['mac_mean']:.3f}")
    print(f"MAC std: {mac_stats['mac_std']:.3f}")
    print(f"MAC median: {mac_stats['mac_median']:.3f}")
    print(f"MAC CV: {mac_stats.get('mac_cv', 'N/A'):.3f}")
    print(f"Correlation: {correlations['pearson_r']:.3f}")
    print(f"P-value: {correlations['pearson_p']:.3e}")
    print(f"Confidence: {correlations['x_mean']:.2f} ± {correlations['x_std']:.2f}")
    
    return results

def demonstrate_oc_ec_analysis(data):
    """
    Demonstrate OC-EC analysis with new modular approach
    """
    print("\\n=== OC-EC ANALYSIS (NEW MODULAR) ===")
    
    analyzer = OCECAnalyzer()
    results = analyzer.analyze(data)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    # Rich, structured results
    sample_info = results['sample_info']
    correlations = results['correlations']
    oc_stats = results['statistics']['oc']
    ec_stats = results['statistics']['ec']
    ratio_stats = results['statistics']['oc_ec_ratio']
    
    print(f"Valid samples: {sample_info['valid_samples']}")
    print(f"OC mean: {oc_stats['oc_mean']:.2f} ± {oc_stats['oc_std']:.2f}")
    print(f"EC mean: {ec_stats['ec_mean']:.2f} ± {ec_stats['ec_std']:.2f}")
    print(f"OC/EC ratio: {ratio_stats['oc_ec_ratio_mean']:.2f} ± {ratio_stats['oc_ec_ratio_std']:.2f}")
    print(f"OC-EC correlation: {correlations['pearson_r']:.3f} (p={correlations['pearson_p']:.3e})")
    
    # Demonstrate additional methods
    correlation_summary = analyzer.get_correlation_summary()
    print(f"Significant correlation: {correlation_summary.get('significant', False)}")
    
    return results

def compare_approaches():
    """
    Main comparison function
    """
    print("BEFORE AND AFTER: MONOLITHIC vs MODULAR APPROACH")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    print(f"Sample data created: {len(data)} samples")
    print(f"Date range: {data['sample_date'].min()} to {data['sample_date'].max()}")
    
    # Run both approaches
    old_results = old_monolithic_approach(data)
    new_results = new_modular_approach(data)
    
    # Compare results
    print("\\n=== COMPARISON ===")
    if old_results and new_results:
        old_mac_mean = old_results.get('mac_mean', 0)
        new_mac_mean = new_results['mac_statistics']['mac_mean']
        
        old_corr = old_results.get('correlation', 0)
        new_corr = new_results['correlations']['pearson_r']
        
        print(f"MAC mean - Old: {old_mac_mean:.3f}, New: {new_mac_mean:.3f}")
        print(f"Correlation - Old: {old_corr:.3f}, New: {new_corr:.3f}")
        print(f"Results match: {abs(old_mac_mean - new_mac_mean) < 0.001}")
    
    # Demonstrate OC-EC analysis
    demonstrate_oc_ec_analysis(data)
    
    print("\\n=== ADVANTAGES OF NEW APPROACH ===")
    print("✓ Modular components can be tested independently")
    print("✓ Rich, structured output format")
    print("✓ Consistent error handling")
    print("✓ Reusable statistical functions")
    print("✓ Clear separation of concerns")
    print("✓ Easy to extend with new analyses")
    print("✓ Better documentation and type hints")
    
    # Save results
    try:
        save_results_to_json(new_results, "outputs/comparison_results.json")
        print("\\n✓ Results saved to outputs/comparison_results.json")
    except Exception as e:
        print(f"Note: Could not save results: {e}")

if __name__ == "__main__":
    compare_approaches()
