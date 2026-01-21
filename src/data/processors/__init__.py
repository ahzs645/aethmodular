"""
Data processors package

This package includes:
- calibration: Aethalometer calibration utilities
- ftir_merger: FTIR data merger for aethalometer analysis
- aethalometer_filter_merger: Complete pipeline for merging aethalometer with FTIR/HIPS data
- dual_dataset_pipeline: Dual dataset processor for high-resolution and FTIR-matched data
"""

# Import existing processors
try:
    from .calibration import AethalometerCalibrator
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    AethalometerCalibrator = None

try:
    from .ftir_merger import FTIRMerger
    FTIR_MERGER_AVAILABLE = True
except ImportError:
    FTIR_MERGER_AVAILABLE = False
    FTIRMerger = None

# Import the new merger pipeline
try:
    from .aethalometer_filter_merger import (
        merge_aethalometer_filter_pipeline,
        load_ftir_hips_data,
        merge_aethalometer_filter_data,
        export_pipeline_results,
        identify_excellent_periods,
        extract_aethalometer_stats,
        map_ethiopian_seasons
    )
    MERGER_PIPELINE_AVAILABLE = True
except ImportError:
    MERGER_PIPELINE_AVAILABLE = False
    merge_aethalometer_filter_pipeline = None
    load_ftir_hips_data = None
    merge_aethalometer_filter_data = None
    export_pipeline_results = None
    identify_excellent_periods = None
    extract_aethalometer_stats = None
    map_ethiopian_seasons = None

# Import dual dataset pipeline
try:
    from .dual_dataset_pipeline import DualDatasetProcessor, run_dual_dataset_processing
    DUAL_DATASET_AVAILABLE = True
except ImportError:
    DUAL_DATASET_AVAILABLE = False
    DualDatasetProcessor = None
    run_dual_dataset_processing = None

# Define what's available for import
__all__ = []

# Add available imports to __all__
if CALIBRATION_AVAILABLE:
    __all__.append('AethalometerCalibrator')

if FTIR_MERGER_AVAILABLE:
    __all__.append('FTIRMerger')

if MERGER_PIPELINE_AVAILABLE:
    __all__.extend([
        'merge_aethalometer_filter_pipeline',
        'load_ftir_hips_data',
        'merge_aethalometer_filter_data',
        'export_pipeline_results',
        'identify_excellent_periods',
        'extract_aethalometer_stats',
        'map_ethiopian_seasons'
    ])

if DUAL_DATASET_AVAILABLE:
    __all__.extend([
        'DualDatasetProcessor',
        'run_dual_dataset_processing'
    ])

# Print availability status when imported (optional - remove if you don't want this)
if __name__ != '__main__':
    available_processors = []
    if CALIBRATION_AVAILABLE:
        available_processors.append('AethalometerCalibrator')
    if FTIR_MERGER_AVAILABLE:
        available_processors.append('FTIRMerger')
    if MERGER_PIPELINE_AVAILABLE:
        available_processors.append('MergerPipeline')
    if DUAL_DATASET_AVAILABLE:
        available_processors.append('DualDatasetProcessor')
    
    # Uncomment the line below if you want to see what's loaded
    # print(f"📦 Data processors loaded: {', '.join(available_processors)}")