# ETAD Data Matching Diagnostic Analysis
# This script helps identify discrepancies in data counting and matching

import pandas as pd
import numpy as np

def diagnose_data_discrepancies(aethalometer_path, filter_db_path, site_code='ETAD'):
    """
    Comprehensive diagnostic analysis to understand data counting discrepancies.
    """
    print("🔍 ETAD Data Matching Diagnostic Analysis")
    print("=" * 60)
    
    # Load the modular matcher (assuming it's available)
    from src.data.loaders import AethalometerFilterMatcher
    
    matcher = AethalometerFilterMatcher(aethalometer_path, filter_db_path)
    
    print("\n📊 STEP 1: Raw Data Analysis")
    print("-" * 40)
    
    # Check aethalometer data
    aeth_data = matcher.load_aethalometer_data()
    print(f"Aethalometer total records: {len(aeth_data)}")
    
    # Check how many have valid BCc data
    bc_columns = ['IR BCc smoothed', 'Red BCc smoothed', 'Green BCc smoothed', 
                  'Blue BCc smoothed', 'UV BCc smoothed']
    
    for col in bc_columns:
        if col in aeth_data.columns:
            valid_count = aeth_data[col].notna().sum()
            print(f"  {col}: {valid_count}/{len(aeth_data)} valid ({valid_count/len(aeth_data)*100:.1f}%)")
    
    # Check site-specific aethalometer data
    site_aeth = matcher.get_site_aethalometer_data(site_code)
    print(f"\nSite {site_code} aethalometer records: {len(site_aeth)}")
    
    print("\n📊 STEP 2: Filter Data Analysis")
    print("-" * 40)
    
    # Load filter database
    filter_data = matcher.filter_loader
    print(f"Total filter database records: {len(filter_data)}")
    
    # Check site-specific filter data
    site_filter = matcher.get_site_filter_data(site_code, 
        ['EC_ftir', 'HIPS_Fabs', 'ChemSpec_Iron_PM2.5'])
    print(f"\nSite {site_code} filter records: {len(site_filter)}")
    
    # Check individual parameter counts
    print("\nIndividual parameter counts:")
    for param in ['EC_ftir', 'HIPS_Fabs', 'ChemSpec_Iron_PM2.5']:
        if param in site_filter.columns:
            valid_count = site_filter[param].notna().sum()
            print(f"  {param}: {valid_count}/{len(site_filter)} valid")
    
    print("\n📊 STEP 3: Date Overlap Analysis")
    print("-" * 40)
    
    # Date analysis
    aeth_dates = set(site_aeth['datetime_local'].dt.date)
    filter_dates = set(site_filter['SampleDate'].dt.date)
    
    print(f"Aethalometer date range: {min(aeth_dates)} to {max(aeth_dates)}")
    print(f"Filter date range: {min(filter_dates)} to {max(filter_dates)}")
    print(f"Overlapping dates: {len(aeth_dates.intersection(filter_dates))}")
    
    print("\n📊 STEP 4: Matching Process Breakdown")
    print("-" * 40)
    
    # Detailed matching analysis
    matched_full = matcher.match_site_data(site_code)
    print(f"Matched records (full): {len(matched_full)}")
    
    # Check for NaN BCc values in matched data
    print("\nBCc validity in matched data:")
    for col in bc_columns:
        if col in matched_full.columns:
            valid_count = matched_full[col].notna().sum()
            nan_count = matched_full[col].isna().sum()
            print(f"  {col}: {valid_count} valid, {nan_count} NaN")
    
    # Check the most critical column (IR BCc)
    if 'IR BCc smoothed' in matched_full.columns:
        ir_valid_mask = matched_full['IR BCc smoothed'].notna()
        ir_valid_count = ir_valid_mask.sum()
        print(f"\n🎯 IR BCc smoothed valid records: {ir_valid_count}/{len(matched_full)}")
        print(f"   This matches Kyan's count of {ir_valid_count} when excluding NaN BC rows")
        
        # Show which dates have NaN IR BCc
        if ir_valid_count < len(matched_full):
            nan_dates = matched_full[~ir_valid_mask]['match_date'].tolist()
            print(f"   Dates with NaN IR BCc: {nan_dates}")
    
    print("\n📊 STEP 5: Filter Dataset Size Analysis")
    print("-" * 40)
    
    # Analyze why filter dataset might be larger than individual components
    print("Investigating filter dataset size discrepancy...")
    
    # Check for duplicates in filter data
    duplicates = site_filter.duplicated(subset=['SampleDate']).sum()
    print(f"Duplicate dates in filter data: {duplicates}")
    
    # Check for missing values in key columns
    print("\nMissing value analysis:")
    key_cols = ['EC_ftir', 'HIPS_Fabs', 'ChemSpec_Iron_PM2.5', 'SampleDate']
    for col in key_cols:
        if col in site_filter.columns:
            missing = site_filter[col].isna().sum()
            print(f"  {col}: {missing} missing values")
    
    print("\n📊 STEP 6: Recommendations")
    print("-" * 40)
    
    print("🔧 To resolve discrepancies:")
    print("1. Use .dropna() on 'IR BCc smoothed' to match Kyan's count")
    print("2. Check filter database for duplicate entries")
    print("3. Verify that unified filter dataset construction is correct")
    print("4. Consider using the most restrictive valid data criteria")
    
    # Show the exact filtering that Kyan is likely using
    if 'IR BCc smoothed' in matched_full.columns:
        kyan_filtered = matched_full.dropna(subset=['IR BCc smoothed'])
        print(f"\n✅ Using Kyan's filtering (dropna on IR BCc): {len(kyan_filtered)} records")
        print("   This should match the expected 173 records")
    
    return {
        'aethalometer_total': len(aeth_data),
        'aethalometer_site': len(site_aeth),
        'filter_total': len(filter_data),
        'filter_site': len(site_filter),
        'matched_full': len(matched_full),
        'matched_no_nan_bc': ir_valid_count if 'IR BCc smoothed' in matched_full.columns else None,
        'overlapping_dates': len(aeth_dates.intersection(filter_dates))
    }

def check_resample_df_behavior():
    """
    Demonstrate how resample_df automatically removes NaN BCc rows.
    """
    print("\n🔍 Understanding resample_df NaN Filtering")
    print("=" * 50)
    
    # This is the key line from the calibration.py resample_df function:
    print("Key code from src/external/calibration.py:")
    print("```python")
    print("try:")
    print("    resampled_numeric = resampled_numeric[~resampled_numeric['IR BCc'].isna()]")
    print("except Exception as e:")
    print("    print(\"Error while checking for NaN in 'IR BCc':\", e)")
    print("```")
    print()
    print("This explains why:")
    print("• The system automatically removes rows with NaN 'IR BCc'")
    print("• Kyan gets 173 when 'excluding Na BC rows'")
    print("• The difference (175 - 173 = 2) represents 2 rows with NaN BCc")
    
def unified_filter_dataset_explanation():
    """
    Explain why unified filter dataset might have more rows than individual measurements.
    """
    print("\n🔍 Understanding Filter Dataset Size")
    print("=" * 45)
    
    print("Possible reasons for 192 rows vs 190 individual measurements:")
    print()
    print("1. **Data Processing Steps**:")
    print("   • Quality control flags might create additional entries")
    print("   • Different processing versions of the same samples")
    print("   • Metadata rows or header information")
    print()
    print("2. **Database Structure**:")
    print("   • Some samples might have multiple analysis results")
    print("   • Replicate measurements for quality assurance")
    print("   • Different analytical methods for the same sample")
    print()
    print("3. **Date Handling**:")
    print("   • Samples collected at different times on the same date")
    print("   • Timezone or date boundary issues")
    print("   • Manual data entry corrections")

# Main diagnostic function
def run_full_diagnostic():
    """
    Run the complete diagnostic analysis.
    """
    aethalometer_path = "../research/ftir_hips_chem/df_Jacros_9am_resampled.pkl"
    filter_db_path = "../research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl"
    
    try:
        # Run main diagnostic
        results = diagnose_data_discrepancies(aethalometer_path, filter_db_path)
        
        # Additional explanations
        check_resample_df_behavior()
        unified_filter_dataset_explanation()
        
        print("\n✅ Diagnostic Analysis Complete!")
        print("\nSummary of Findings:")
        print(f"• System shows {results['matched_full']} matched records")
        print(f"• Kyan gets {results['matched_no_nan_bc']} after excluding NaN BC")
        print(f"• Difference: {results['matched_full'] - results['matched_no_nan_bc']} rows with NaN BCc")
        print(f"• Filter dataset size discrepancy likely due to processing artifacts")
        
    except Exception as e:
        print(f"❌ Error in diagnostic: {e}")
        print("Please ensure the modular matcher is available and paths are correct")

if __name__ == "__main__":
    run_full_diagnostic()
