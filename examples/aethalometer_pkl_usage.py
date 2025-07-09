"""
Example usage of the new Aethalometer PKL Loader
Demonstrates loading .pkl files and converting between standard and JPL formats
"""

import pandas as pd
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loaders.aethalometer import AethalometerPKLLoader, load_aethalometer_data
from analysis.bc.source_apportionment import SourceApportionmentAnalyzer
from config.plotting import setup_plotting_style
from utils.file_io import ensure_output_directory

def example_load_pkl_data():
    """
    Example of loading aethalometer data from .pkl files
    """
    
    print("=== AETHALOMETER PKL LOADER EXAMPLE ===")
    print()
    
    # Example 1: Load data in auto-detect mode
    pkl_path = "df_cleaned_Jacros_hourly.pkl"  # Your actual file path
    
    try:
        loader = AethalometerPKLLoader(pkl_path, format_type="auto")
        
        # Get data summary first
        print("Data Summary:")
        summary = loader.get_data_summary()
        for key, value in summary.items():
            if key != 'columns':  # Skip columns for brevity
                print(f"  {key}: {value}")
        print()
        
        # Load data in standard format
        print("Loading data in standard format...")
        df_standard = loader.load(convert_to_jpl=False)
        print(f"Loaded {len(df_standard)} rows")
        print(f"Columns: {list(df_standard.columns)[:10]}...")  # Show first 10 columns
        print()
        
        # Load data in JPL format
        print("Loading data in JPL format...")
        df_jpl = loader.load(convert_to_jpl=True)
        print(f"Loaded {len(df_jpl)} rows")
        print(f"Columns: {list(df_jpl.columns)[:10]}...")  # Show first 10 columns
        print()
        
        # Show column mapping example
        print("Column mapping example:")
        standard_cols = ['IR BCc', 'Biomass BCc', 'Fossil fuel BCc']
        jpl_cols = ['IR.BCc', 'Biomass.BCc', 'Fossil.fuel.BCc']
        
        for std_col, jpl_col in zip(standard_cols, jpl_cols):
            if std_col in df_standard.columns and jpl_col in df_jpl.columns:
                print(f"  {std_col} -> {jpl_col}")
        print()
        
        return df_standard, df_jpl
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def example_site_filtering():
    """
    Example of filtering data by site
    """
    
    print("=== SITE FILTERING EXAMPLE ===")
    print()
    
    pkl_path = "master_addis_file.pkl"  # File with multiple sites
    
    try:
        loader = AethalometerPKLLoader(pkl_path)
        
        # Get available sites
        sites = loader.get_available_sites()
        print(f"Available sites: {sites}")
        print()
        
        # Load data for specific site
        if len(sites) > 0 and sites[0] != "No site column found":
            site_name = sites[0]
            print(f"Loading data for site: {site_name}")
            
            df_site = loader.load(site_filter=site_name, convert_to_jpl=True)
            print(f"Loaded {len(df_site)} rows for {site_name}")
            
            # Show data summary for this site
            site_summary = loader.get_data_summary(site_filter=site_name)
            print(f"Site summary: {site_summary}")
            print()
            
            return df_site
        else:
            print("No sites found or site column missing")
            return None
            
    except Exception as e:
        print(f"Error with site filtering: {e}")
        return None

def example_format_conversion():
    """
    Example of converting between formats
    """
    
    print("=== FORMAT CONVERSION EXAMPLE ===")
    print()
    
    # Create sample data in standard format
    sample_data = pd.DataFrame({
        'datetime_local': pd.date_range('2023-01-01', periods=100, freq='H'),
        'IR BCc': [10.5 + i*0.1 for i in range(100)],
        'Blue BCc': [8.2 + i*0.08 for i in range(100)],
        'Biomass BCc': [6.1 + i*0.06 for i in range(100)],
        'Fossil fuel BCc': [4.4 + i*0.04 for i in range(100)],
        'AAE calculated': [1.1 + i*0.01 for i in range(100)],
        'Flow total (mL/min)': [100] * 100,
        'site': ['TestSite'] * 100
    })
    
    # Save as pkl
    test_pkl_path = "test_aethalometer_data.pkl"
    sample_data.to_pickle(test_pkl_path)
    
    # Load and convert to JPL format
    loader = AethalometerPKLLoader(test_pkl_path)
    
    print("Original format (standard):")
    df_original = loader.load(convert_to_jpl=False)
    print(f"Columns: {list(df_original.columns)}")
    print()
    
    print("Converted to JPL format:")
    df_jpl = loader.load(convert_to_jpl=True)
    print(f"Columns: {list(df_jpl.columns)}")
    print()
    
    # Show specific conversions
    print("Conversion examples:")
    conversions = [
        ('IR BCc', 'IR.BCc'),
        ('Biomass BCc', 'Biomass.BCc'),
        ('Fossil fuel BCc', 'Fossil.fuel.BCc'),
        ('Flow total (mL/min)', 'Flow.total.mL.min')
    ]
    
    for std_col, jpl_col in conversions:
        if std_col in df_original.columns and jpl_col in df_jpl.columns:
            print(f"  {std_col} -> {jpl_col}")
            # Verify data is the same
            if df_original[std_col].equals(df_jpl[jpl_col]):
                print(f"    ✓ Data preserved")
            else:
                print(f"    ✗ Data mismatch!")
    
    # Clean up
    Path(test_pkl_path).unlink()
    
    return df_original, df_jpl

def example_integration_with_analysis():
    """
    Example of using the loaded data with analysis modules
    """
    
    print("=== INTEGRATION WITH ANALYSIS EXAMPLE ===")
    print()
    
    try:
        # Load aethalometer data
        pkl_path = "df_cleaned_Jacros_hourly.pkl"
        df = load_aethalometer_data(pkl_path, output_format='jpl')
        
        if df is None or len(df) == 0:
            print("No data loaded for analysis")
            return
        
        print(f"Loaded {len(df)} samples for analysis")
        
        # Check for required columns for source apportionment
        required_cols = ['IR.BCc', 'Biomass.BCc', 'Fossil.fuel.BCc']
        available_cols = [col for col in required_cols if col in df.columns]
        
        print(f"Available analysis columns: {available_cols}")
        
        if len(available_cols) >= 2:
            print("Sufficient data for source apportionment analysis")
            
            # Example analysis (pseudo-code, would need actual analyzer)
            # analyzer = SourceApportionmentAnalyzer()
            # results = analyzer.analyze(df)
            # print(f"Analysis results: {results}")
            
        else:
            print("Insufficient columns for source apportionment analysis")
        
        # Show data statistics
        print("\nData Statistics:")
        for col in available_cols:
            if col in df.columns:
                print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
        
    except Exception as e:
        print(f"Error in analysis integration: {e}")

def example_convenience_function():
    """
    Example using the convenience function
    """
    
    print("=== CONVENIENCE FUNCTION EXAMPLE ===")
    print()
    
    # Example file paths (adjust to your actual files)
    test_files = [
        "df_cleaned_Jacros_hourly.pkl",
        "df_cleaned_Central_hourly.pkl",
        "master_addis_file.csv"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            try:
                print(f"Loading {file_path}...")
                
                # Load in JPL format
                df = load_aethalometer_data(file_path, output_format='jpl')
                
                print(f"  Loaded {len(df)} rows")
                print(f"  Date range: {df['datetime_local'].min()} to {df['datetime_local'].max()}")
                
                # Show BC columns
                bc_cols = [col for col in df.columns if '.BCc' in col]
                print(f"  BC columns: {bc_cols[:5]}")  # First 5
                print()
                
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                print()
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    setup_plotting_style()
    
    # Run examples
    print("Running Aethalometer PKL Loader Examples...")
    print("=" * 50)
    print()
    
    # Basic loading
    df_std, df_jpl = example_load_pkl_data()
    
    # Site filtering
    df_site = example_site_filtering()
    
    # Format conversion
    df_orig, df_converted = example_format_conversion()
    
    # Integration with analysis
    example_integration_with_analysis()
    
    # Convenience function
    example_convenience_function()
    
    print("Examples completed!")