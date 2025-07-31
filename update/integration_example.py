# Integration Example - Replace your existing processing cells with this

# %%
# Cell 1: Setup and Configuration (Keep your existing setup)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), 'src'))

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

# Add timezone to config for proper 9am-to-9am handling
config.timezone = 'Africa/Addis_Ababa'

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

print("✅ Configuration and setup complete!")

# %%
# Cell 2: Import and Run Dual Dataset Processing
from dual_dataset_processor import DualDatasetProcessor, run_dual_dataset_processing

# 🎛️ Configuration: Toggle Ethiopia fix here
APPLY_ETHIOPIA_FIX = True  # Set to True to enable Ethiopia pneumatic pump fix

print(f"🚀 DUAL-DATASET PROCESSING {'WITH' if APPLY_ETHIOPIA_FIX else 'WITHOUT'} Ethiopia Fix")
print("=" * 70)

# Run the dual dataset processing
datasets = run_dual_dataset_processing(
    config=config,
    setup=setup,
    ethiopia_fix=APPLY_ETHIOPIA_FIX
)

# Access your datasets
high_resolution_data = datasets['high_resolution']
ftir_matched_data = datasets['ftir_matched']
raw_data = datasets['raw_data']
cleaned_data = datasets['cleaned_data']
ftir_data = datasets['ftir_data']

print(f"\n🎉 Processing Complete!")
print(f"📈 High-resolution data: {high_resolution_data.shape}")
print(f"🔗 FTIR-matched data: