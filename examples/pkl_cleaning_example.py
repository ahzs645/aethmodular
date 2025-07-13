#!/usr/bin/env python3
"""
Example usage of PKL Data Cleaning Pipeline

This script demonstrates how to use the integrated PKL data cleaning 
functionality within the aethmodular framework.
"""

import pandas as pd
import sys
import os

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from src.data.qc import PKLDataCleaner, load_and_clean_pkl_data, table_removed_datapoints_by_month

def example_basic_usage():
    """Basic usage example of PKL data cleaning."""
    print("=== Basic PKL Data Cleaning Example ===")
    
    # Option 1: Use the convenience function to load and clean data in one step
    try:
        df_cleaned = load_and_clean_pkl_data(
            directory_path="../JPL_aeth/",  # Adjust path as needed
            verbose=True,
            summary=True
        )
        print(f"Loaded and cleaned data shape: {df_cleaned.shape}")
        print(f"Columns: {list(df_cleaned.columns)}")
        
    except FileNotFoundError:
        print("Data directory not found. Please adjust the directory_path parameter.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
        
    return df_cleaned

def example_step_by_step_cleaning():
    """Step-by-step cleaning example with custom parameters."""
    print("\n=== Step-by-Step PKL Data Cleaning Example ===")
    
    # Note: This example assumes you have already loaded your data
    # In practice, you would load your data here first
    print("This example shows the step-by-step cleaning process.")
    print("First, load your data using the calibration module.")
    
    # Initialize the cleaner
    cleaner = PKLDataCleaner(wavelengths_to_filter=['IR', 'Blue'])
    
    # Example of what the cleaning steps would look like:
    steps = [
        "1. Status cleaning using external calibration module",
        "2. Remove extreme BCc values",
        "3. Clean flow range violations",
        "4. Clean abnormal flow ratios",
        "5. Remove leak indicators",
        "6. Clean BCc denominator issues",
        "7. Temperature change filtering",
        "8. Roughness calculation and filtering"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print("\n   Use cleaner.clean_pipeline(df) to run all steps at once")
    print("   Or call individual methods for custom control:")
    print("   - cleaner.clean_by_status(df)")
    print("   - cleaner.clean_extreme_bcc(df)")
    print("   - cleaner.clean_flow_range(df)")
    print("   - etc.")

def example_custom_parameters():
    """Example with custom cleaning parameters."""
    print("\n=== Custom Parameters Example ===")
    
    # Initialize cleaner with custom wavelengths
    cleaner = PKLDataCleaner(wavelengths_to_filter=['IR', 'Blue', 'Green'])
    
    print("Custom parameters you can adjust:")
    print("- Flow threshold: cleaner.clean_flow_range(df, flow_threshold=0.15)")
    print("- Temperature thresholds: cleaner.clean_temperature_change(df)")
    print("- Roughness Z-score: cleaner.flag_high_roughness_periods(df, z_threshold=3)")
    print("- BCc bounds: cleaner.clean_bcc_ratio(df, lower_bound=0.1, upper_bound=8)")

def example_quality_report():
    """Example of generating quality reports."""
    print("\n=== Quality Report Example ===")
    
    # This would be used with actual data
    print("After cleaning, generate quality reports:")
    print("- table_removed_datapoints_by_month(original_df, cleaned_df)")
    print("- Compare data loss by time period")
    print("- Analyze cleaning effectiveness")

def main():
    """Run all examples."""
    print("PKL Data Cleaning Pipeline Examples")
    print("=" * 50)
    
    # Run basic example
    cleaned_df = example_basic_usage()
    
    # Run other examples
    example_step_by_step_cleaning()
    example_custom_parameters()
    example_quality_report()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    
    if cleaned_df is not None:
        print(f"\nCleaned data preview:")
        print(cleaned_df.head())
        print(f"\nData shape: {cleaned_df.shape}")
        print(f"Date range: {cleaned_df['datetime_local'].min()} to {cleaned_df['datetime_local'].max()}")

if __name__ == "__main__":
    main()
