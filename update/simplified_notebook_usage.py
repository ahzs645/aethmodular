# %%
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), 'src'))

# Import the new enhanced PKL processing module
from data.qc.enhanced_pkl_processing import process_pkl_data_enhanced, EnhancedPKLProcessor
from config.notebook_config import NotebookConfig
from notebook_utils.pkl_cleaning_integration import create_enhanced_setup

# Your existing configuration
config = NotebookConfig(
    site_code='ETAD',
    wavelength='Red',
    quality_threshold=10,
    output_format='jpl',
    min_samples_for_analysis=30,
    confidence_level=0.95,
    outlier_threshold=3.0,
    figure_size=(12, 8),
    font_size=10,
    dpi=300
)

# Set your data paths (same as before)
base_data_path = "/Users/ahzs645/Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University/Research/Grad/UC Davis Ann/NASA MAIA/Data"

config.aethalometer_files = {
    'pkl_data': os.path.join(
        base_data_path,
        "Aethelometry Data/Kyan Data/Mergedcleaned and uncleaned MA350 data20250707030704",
        "df_uncleaned_Jacros_API_and_OG.pkl"
    ),
    'csv_data': os.path.join(
        base_data_path,
        "Aethelometry Data/Raw",
        "Jacros_MA350_1-min_2022-2024_Cleaned.csv"
    )
}

config.ftir_db_path = os.path.join(
    base_data_path,
    "EC-HIPS-Aeth Comparison/Data/Original Data/Combined Database",
    "spartan_ftir_hips.db"
)

# Create enhanced setup
setup = create_enhanced_setup(config)

# %%
print("📁 Loading datasets...")
datasets = setup.load_all_data()

# Get PKL data
pkl_data_original = setup.get_dataset('pkl_data')

# Quick fix for datetime_local issue (same as before)
if 'datetime_local' not in pkl_data_original.columns:
    if pkl_data_original.index.name == 'datetime_local':
        print("✅ Converting datetime_local from index to column...")
        pkl_data_original = pkl_data_original.reset_index()
    elif hasattr(pkl_data_original.index, 'tz'):
        print("✅ Creating datetime_local column from datetime index...")
        pkl_data_original['datetime_local'] = pkl_data_original.index
        pkl_data_original = pkl_data_original.reset_index(drop=True)

print(f"📊 PKL data ready: {pkl_data_original.shape}")
print(f"📅 Date range: {pkl_data_original['datetime_local'].min()} to {pkl_data_original['datetime_local'].max()}")

# %%
# SIMPLIFIED PROCESSING: Replace all the complex pipeline with one function call!

# Option 1: Simple one-liner (with export)
pkl_data_cleaned = process_pkl_data_enhanced(
    pkl_data_original,
    wavelengths_to_filter=['IR', 'Blue'],  # Focus on IR and Blue
    export_path='pkl_data_cleaned_enhanced',  # Will create .csv and .pkl files
    verbose=True  # Show detailed progress
)

print("\n🎉 Processing Complete!")
print(f"📊 Final shape: {pkl_data_cleaned.shape}")
print("🚀 Ready for further analysis!")

# %%
# Option 2: More control with the class (if you need to customize further)

# Create processor with custom settings
processor = EnhancedPKLProcessor(
    wavelengths_to_filter=['IR', 'Blue'],
    verbose=True,
    # You can pass additional PKLDataCleaner arguments here
    quality_threshold=10  # Example custom parameter
)

# Run the full pipeline
pkl_data_cleaned_v2 = processor.process_pkl_data(
    pkl_data_original,
    export_path='pkl_data_cleaned_v2'
)

# Or run individual steps if needed
# df_preprocessed = processor.comprehensive_preprocessing(pkl_data_original)
# df_smoothed = processor.apply_dema_smoothing(df_preprocessed)
# df_cleaned = processor.cleaner.clean_pipeline(df_smoothed, skip_preprocessing=True)

# %%
# Verification and comparison
print("\n📊 Final Verification:")
print("=" * 50)

# Check that we have all the key columns
key_columns = ['datetime_local', 'IR ATN1', 'IR BCc', 'Blue ATN1', 'Blue BCc', 'Flow total (mL/min)']
for col in key_columns:
    if col in pkl_data_cleaned.columns:
        print(f"✅ {col}")
    else:
        print(f"❌ {col}")

# Check smoothed columns
smoothed_cols = [col for col in pkl_data_cleaned.columns if 'smoothed' in col]
print(f"\n📈 Smoothed columns ({len(smoothed_cols)}):")
for col in smoothed_cols[:10]:  # Show first 10
    print(f"  • {col}")

# Summary statistics
print(f"\n📊 Summary Statistics:")
print(f"Original rows: {len(pkl_data_original):,}")
print(f"Final rows: {len(pkl_data_cleaned):,}")
print(f"Columns: {pkl_data_cleaned.shape[1]}")
print(f"Date range: {pkl_data_cleaned['datetime_local'].min()} to {pkl_data_cleaned['datetime_local'].max()}")

# Memory usage
memory_mb = pkl_data_cleaned.memory_usage(deep=True).sum() / 1024 / 1024
print(f"Memory usage: {memory_mb:.1f} MB")

print("\n✅ Enhanced PKL processing complete and verified!")

# %%
# Optional: Quick quality check using the modular QC tools
from data.qc import quick_quality_check

# Set datetime as index for quality check
pkl_for_qc = pkl_data_cleaned.set_index('datetime_local')
quick_quality_check(pkl_for_qc)