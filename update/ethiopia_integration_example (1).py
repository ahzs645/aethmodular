#!/usr/bin/env python3
"""
Ethiopia Aethalometer Fix Integration Example

This script demonstrates how to integrate the Ethiopia loading compensation fix
into your existing modular aethalometer data processing pipeline.

Author: AethModular Team
Date: 2025-01-31
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Your existing modular imports
try:
    from data.processors.aethalometer_filter_merger import merge_aethalometer_filter_pipeline
    from data.qc.pkl_cleaning import clean_pkl_data_with_ethiopia_fix, quick_ethiopia_fix_only
    from data.processors.site_corrections import SiteCorrections, apply_ethiopia_fix
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import modules: {e}")
    print("Make sure you have the updated modules in place")
    MODULES_AVAILABLE = False


def example_1_quick_ethiopia_fix():
    """
    Example 1: Quick Ethiopia fix for existing cleaned data
    """
    print("🔧 Example 1: Quick Ethiopia Fix")
    print("=" * 40)
    
    # This is what you would do with your existing data
    print("# Load your existing cleaned PKL data")
    print("df = pd.read_pickle('df_cleaned_Central_API_and_OG.pkl')")
    
    print("\n# Apply Ethiopia fix")
    print("df_ethiopia_fixed = apply_ethiopia_fix(df, verbose=True)")
    
    print("\n# Save the corrected data")
    print("df_ethiopia_fixed.to_pickle('df_cleaned_Central_API_and_OG_ethiopia_fixed.pkl')")
    
    # If modules are available, show what the output would look like
    if MODULES_AVAILABLE:
        print("\n📊 Expected output:")
        print("🔧 Applying Ethiopia (ETAD) site corrections...")
        print("  📊 Applying IR loading compensation fix...")
        print("    📈 IR median K: 0.009234")
        print("    🎯 IR optimal K: 0.008756")
        print("  📊 Applying Blue loading compensation fix...")
        print("    📈 Blue median K: 0.011456")
        print("    🎯 Blue optimal K: 0.010892")
        print("  📊 BCc-ATN1 correlation: 0.234 → -0.012 (much better!)")


def example_2_integrated_cleaning_pipeline():
    """
    Example 2: Integrated cleaning pipeline with Ethiopia fix
    """
    print("\n🧹 Example 2: Integrated Cleaning Pipeline")
    print("=" * 45)
    
    print("# Load raw PKL data")
    print("df_raw = pd.read_pickle('raw_aethalometer_data.pkl')")
    
    print("\n# Clean data with Ethiopia fix integrated")
    print("df_cleaned = clean_pkl_data_with_ethiopia_fix(")
    print("    df_raw,")
    print("    site_code='ETAD',  # Ethiopia site code")
    print("    verbose=True,")
    print("    apply_status_cleaning=True,")
    print("    apply_optical_saturation=True,")
    print("    apply_flow_validation=True,")
    print("    apply_roughness_cleaning=True,")
    print("    apply_site_corrections=True  # This applies Ethiopia fix")
    print(")")
    
    print("\n# Continue with your existing pipeline")
    print("results = merge_aethalometer_filter_pipeline(")
    print("    aethalometer_files=[df_cleaned],  # Use cleaned data")
    print("    ftir_db_path='spartan_ftir_hips.db',")
    print("    wavelength='Red',")
    print("    site_code='ETAD'")
    print(")")


def example_3_validation_and_comparison():
    """
    Example 3: Validation and comparison of corrections
    """
    print("\n📊 Example 3: Validation and Comparison")
    print("=" * 40)
    
    print("# Load original and corrected data")
    print("df_original = pd.read_pickle('df_cleaned_Central_API_and_OG.pkl')")
    print("df_corrected = apply_ethiopia_fix(df_original)")
    
    print("\n# Create corrector for validation")
    print("corrector = SiteCorrections(site_code='ETAD', verbose=True)")
    
    print("\n# Validate corrections")
    print("validation_results = corrector.validate_corrections(")
    print("    df_original, df_corrected, wavelength='IR'")
    print(")")
    
    print("\n# Plot comparison")
    print("corrector.plot_correction_comparison(")
    print("    df_original, df_corrected, wavelength='IR',")
    print("    save_path='ethiopia_correction_comparison.png'")
    print(")")
    
    print("\n# Print validation results")
    print("for key, value in validation_results.items():")
    print("    print(f'{key}: {value}')")


def example_4_complete_workflow():
    """
    Example 4: Complete workflow from raw data to final analysis
    """
    print("\n🔄 Example 4: Complete Workflow")
    print("=" * 35)
    
    workflow_steps = [
        "1️⃣ Load raw aethalometer PKL data",
        "2️⃣ Apply comprehensive cleaning + Ethiopia fix", 
        "3️⃣ Validate corrections",
        "4️⃣ Merge with FTIR/HIPS data",
        "5️⃣ Export results for analysis"
    ]
    
    for step in workflow_steps:
        print(step)
    
    print("\n# Complete code:")
    print("""
# Step 1: Load raw data
df_raw = pd.read_pickle('raw_central_data.pkl')

# Step 2: Clean with Ethiopia fix
df_cleaned = clean_pkl_data_with_ethiopia_fix(
    df_raw, 
    site_code='ETAD',
    verbose=True
)

# Step 3: Validate (optional)
corrector = SiteCorrections(site_code='ETAD')
validation = corrector.validate_corrections(df_raw, df_cleaned)
print("Validation results:", validation)

# Step 4: Merge with FTIR/HIPS
results = merge_aethalometer_filter_pipeline(
    aethalometer_files=[df_cleaned],
    ftir_db_path='spartan_ftir_hips.db',
    wavelength='Red',
    site_code='ETAD'
)

# Step 5: Export
from data.processors.aethalometer_filter_merger import export_pipeline_results
export_pipeline_results(results, output_dir='outputs_ethiopia_corrected')
""")


def practical_integration_guide():
    """
    Practical step-by-step integration guide
    """
    print("\n📋 Practical Integration Guide")
    print("=" * 40)
    
    steps = [
        {
            'step': '1. Add the modules to your project',
            'action': 'Copy site_corrections.py to src/data/processors/',
            'code': 'cp site_corrections.py src/data/processors/'
        },
        {
            'step': '2. Update your PKL cleaning module', 
            'action': 'Replace or update pkl_cleaning.py with Ethiopia fix integration',
            'code': 'cp updated_pkl_cleaning.py src/data/qc/pkl_cleaning.py'
        },
        {
            'step': '3. Test the integration',
            'action': 'Run a quick test with your existing data',
            'code': '''
# Test code:
from src.data.processors.site_corrections import apply_ethiopia_fix
df_test = pd.read_pickle('small_test_dataset.pkl') 
df_fixed = apply_ethiopia_fix(df_test)
print("✅ Ethiopia fix working!")
'''
        },
        {
            'step': '4. Update your main processing script',
            'action': 'Modify your existing script to use the new cleaning function',
            'code': '''
# Old code:
# df_cleaned = calibration.convert_to_float(df)

# New code:
df_cleaned = clean_pkl_data_with_ethiopia_fix(
    df, site_code='ETAD', verbose=True
)
'''
        },
        {
            'step': '5. Validate the results',
            'action': 'Compare old vs new results',
            'code': '''
# Validation code:
corrector = SiteCorrections(site_code='ETAD')
corrector.plot_correction_comparison(df_old, df_new)
'''
        }
    ]
    
    for i, step_info in enumerate(steps, 1):
        print(f"\n{step_info['step']}")
        print(f"Action: {step_info['action']}")
        if 'code' in step_info:
            print(f"Code:")
            print(step_info['code'])


def main():
    """
    Main function to run all examples
    """
    print("🇪🇹 Ethiopia Aethalometer Fix Integration Examples")
    print("=" * 55)
    print("This script shows how to integrate the Ethiopia loading compensation fix")
    print("into your existing modular aethalometer data processing pipeline.")
    
    example_1_quick_ethiopia_fix()
    example_2_integrated_cleaning_pipeline()
    example_3_validation_and_comparison()
    example_4_complete_workflow()
    practical_integration_guide()
    
    print("\n🎉 Integration Examples Complete!")
    print("\nNext steps:")
    print("1. Copy the site_corrections.py module to your src/data/processors/ directory")
    print("2. Update your pkl_cleaning.py with Ethiopia fix integration")
    print("3. Test with a small dataset first")
    print("4. Run validation to confirm improvements")
    print("5. Update your main processing scripts")
    
    if not MODULES_AVAILABLE:
        print("\n⚠️ Note: Modules not available for live demo.")
        print("Install the updated modules first, then run this script again for live examples.")


if __name__ == "__main__":
    main()
