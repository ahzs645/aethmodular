#!/usr/bin/env python3
"""
PKL Data Cleaning Example

This example demonstrates how to use the PKL data cleaning pipeline
with configurable data directory paths.

The example shows both class-based and function-based approaches,
and demonstrates how to configure the data directory path.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Main example function."""
    print("=== PKL Data Cleaning Example ===")
    print(f"Timestamp: {datetime.now()}")
    
    # =============================================================================
    # CONFIGURE DATA DIRECTORY
    # =============================================================================
    
    # Option 1: Use relative path (default for backwards compatibility)
    data_directory = "../JPL_aeth/"
    
    # Option 2: Use absolute path (recommended for production)
    # data_directory = "/path/to/your/pkl/data/"
    
    # Option 3: Use environment variable
    # data_directory = os.getenv('PKL_DATA_PATH', '../JPL_aeth/')
    
    # Option 4: Command line argument
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
        print(f"📁 Using data directory from command line: {data_directory}")
    else:
        print(f"📁 Using default data directory: {data_directory}")
    
    # Check if directory exists
    if not os.path.exists(data_directory):
        print(f"⚠️  Warning: Directory '{data_directory}' does not exist!")
        print("   Please update the data_directory variable or provide a valid path.")
        return
    
    try:
        # Import PKL cleaning functionality
        from data.qc import PKLDataCleaner, load_and_clean_pkl_data
        
        print("\n=== Method 1: Using PKLDataCleaner Class (Recommended) ===")
        
        # Create cleaner instance with custom data directory
        cleaner = PKLDataCleaner(
            data_directory=data_directory,
            wavelengths_to_filter=['IR', 'Blue']
        )
        
        print(f"🔧 PKL cleaner initialized")
        print(f"   Data directory: {cleaner.data_directory}")
        print(f"   Wavelengths: {cleaner.wls_to_filter}")
        
        # Load and clean data using instance method
        print("\n📊 Loading and cleaning data...")
        df_cleaned = cleaner.load_and_clean_data(
            verbose=True,
            summary=True
        )
        
        print(f"\n✅ Data cleaning completed!")
        print(f"   Shape: {df_cleaned.shape}")
        print(f"   Date range: {df_cleaned['datetime_local'].min()} to {df_cleaned['datetime_local'].max()}")
        
        # Example of using individual cleaning methods
        print("\n=== Individual Cleaning Steps Example ===")
        
        # Start with a sample for demonstration
        df_sample = df_cleaned.head(1000).copy()
        print(f"📋 Working with sample of {len(df_sample)} rows")
        
        # Apply individual steps
        df_step1 = cleaner.clean_by_status(df_sample)
        df_step2 = cleaner.clean_optical_saturation(df_step1)
        df_step3 = cleaner.clean_flow_range(df_step2)
        
        print(f"   Final sample size: {len(df_step3)} rows")
        
        print("\n=== Method 2: Using Standalone Function ===")
        
        # Alternative: use standalone function
        df_cleaned_v2 = load_and_clean_pkl_data(
            directory_path=data_directory,
            verbose=False
        )
        
        print(f"✅ Standalone function completed: {df_cleaned_v2.shape}")
        
        # Basic quality assessment
        print("\n=== Data Quality Assessment ===")
        print(f"📊 Total data points: {len(df_cleaned):,}")
        print(f"📅 Time span: {(df_cleaned['datetime_local'].max() - df_cleaned['datetime_local'].min()).days} days")
        
        if 'Serial number' in df_cleaned.columns:
            instruments = df_cleaned['Serial number'].unique()
            print(f"📱 Unique instruments: {len(instruments)}")
            for inst in instruments[:3]:  # Show first 3
                count = (df_cleaned['Serial number'] == inst).sum()
                print(f"   {inst}: {count:,} points")
        
        # Check for missing values
        missing_counts = df_cleaned.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if len(missing_counts) > 0:
            print("⚠️  Missing values found:")
            for col, count in missing_counts.head(3).items():
                print(f"   {col}: {count:,} ({count/len(df_cleaned)*100:.2f}%)")
        else:
            print("✅ No missing values")
        
        print(f"\n🎉 PKL cleaning example completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the correct directory")
        print("   and that the src/data/qc modules are available")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("   Check that your data directory contains valid PKL files")
        print("   and that the external calibration module is available")

def usage():
    """Print usage information."""
    print("Usage:")
    print("  python pkl_cleaning_example.py [data_directory]")
    print("")
    print("Examples:")
    print("  python pkl_cleaning_example.py")
    print("  python pkl_cleaning_example.py ../my_pkl_data/")
    print("  python pkl_cleaning_example.py /opt/data/aethalometer/")
    print("")
    print("Environment variable:")
    print("  export PKL_DATA_PATH=/path/to/data")
    print("  python pkl_cleaning_example.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        usage()
    else:
        main()
