"""
Multi-site aethalometer analysis scripts.

This package provides reusable modules for analyzing aethalometer and filter data
from multiple global sites (Beijing, Delhi, JPL, Addis Ababa).

Modules:
--------
config : Central configuration (paths, site definitions, parameters)
outliers : Outlier registry and exclusion functions
data_matching : Data loading and matching functions
plotting : Reusable plotting functions

Quick Start:
------------
    # In a notebook, add scripts folder to path
    import sys
    sys.path.insert(0, '../scripts')

    # Or if running from FTIR_HIPS_Chem directory:
    sys.path.insert(0, './scripts')

    # Import what you need
    from config import SITES, PROCESSED_SITES_DIR, FILTER_DATA_PATH
    from outliers import EXCLUDED_SAMPLES, apply_exclusion_flags, get_clean_data
    from data_matching import load_aethalometer_data, load_filter_data, match_aeth_filter_data
    from plotting import plot_crossplot, plot_before_after_comparison

    # Load data
    aethalometer_data = load_aethalometer_data()
    filter_data = load_filter_data()

    # Match and apply exclusions for a site
    matched = match_aeth_filter_data('Beijing', aethalometer_data['Beijing'],
                                      filter_data, SITES['Beijing']['code'])
    matched = apply_exclusion_flags(matched, 'Beijing')
    clean = get_clean_data(matched)

Example Notebook Cell:
----------------------
    # Setup cell for notebooks
    import sys
    sys.path.insert(0, '../scripts')

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from config import SITES, PROCESSED_SITES_DIR, FILTER_DATA_PATH, MAC_VALUE
    from outliers import (
        EXCLUDED_SAMPLES, MANUAL_OUTLIERS,
        apply_exclusion_flags, apply_threshold_flags,
        get_clean_data, print_exclusion_summary
    )
    from data_matching import (
        load_aethalometer_data, load_filter_data,
        match_aeth_filter_data, match_all_parameters,
        match_with_smooth_raw_info
    )
    from plotting import PlotConfig, crossplots, timeseries, distributions, comparisons
    from plotting.utils import calculate_regression_stats

    # Configure matplotlib — use default (white background) like
    # Analysis_Tasks_Jan2025.ipynb. Do NOT call plt.style.use('seaborn-v0_8-darkgrid');
    # that gives a grey axes background which doesn't print/publish well.
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10

    print("Modules loaded successfully!")
"""

# For convenient imports when using: from scripts import *
try:
    from config import (
        SITES, PROCESSED_SITES_DIR, FILTER_DATA_PATH,
        AERONET_DATA_DIR, WEATHER_DATA_DIR,
        MAC_VALUE, FLOW_FIX_PERIODS, MIN_EC_THRESHOLD,
        SMOOTH_RAW_THRESHOLDS, DEFAULT_BC_WAVELENGTH,
        FILTER_CATEGORIES, CROSS_COMPARISONS,
        ETAD_FACTOR_CONTRIBUTIONS_PATH, ETAD_FILTER_ID_PATH,
    )
    from outliers import (
        EXCLUDED_SAMPLES, MANUAL_OUTLIERS,
        apply_exclusion_flags, apply_threshold_flags,
        get_clean_data, get_excluded_data, get_outlier_data,
        print_exclusion_summary, identify_outlier_dates,
    )
    from data_matching import (
        load_aethalometer_data, load_filter_data,
        match_aeth_filter_data, match_all_parameters,
        match_with_smooth_raw_info, match_hips_with_smooth_raw,
        add_flow_period_column, add_base_filter_id, match_by_filter_id,
        pivot_filter_by_id, get_site_code, get_site_color, print_data_summary,
    )
    from etad_factors import (
        load_etad_factor_contributions, load_etad_filter_ids,
        load_etad_factors_with_filter_ids, match_etad_factors,
        ETAD_PMF_SOURCE_NAMES, ETAD_FACTOR_RENAME,
    )
    from plotting import (
        PlotConfig, apply_default_style, crossplots, timeseries,
        distributions, comparisons, calculate_regression_stats,
    )
except ImportError:
    from .config import (
        SITES, PROCESSED_SITES_DIR, FILTER_DATA_PATH,
        AERONET_DATA_DIR, WEATHER_DATA_DIR,
        MAC_VALUE, FLOW_FIX_PERIODS, MIN_EC_THRESHOLD,
        SMOOTH_RAW_THRESHOLDS, DEFAULT_BC_WAVELENGTH,
        FILTER_CATEGORIES, CROSS_COMPARISONS,
        ETAD_FACTOR_CONTRIBUTIONS_PATH, ETAD_FILTER_ID_PATH,
    )
    from .outliers import (
        EXCLUDED_SAMPLES, MANUAL_OUTLIERS,
        apply_exclusion_flags, apply_threshold_flags,
        get_clean_data, get_excluded_data, get_outlier_data,
        print_exclusion_summary, identify_outlier_dates,
    )
    from .data_matching import (
        load_aethalometer_data, load_filter_data,
        match_aeth_filter_data, match_all_parameters,
        match_with_smooth_raw_info, match_hips_with_smooth_raw,
        add_flow_period_column, add_base_filter_id, match_by_filter_id,
        pivot_filter_by_id, get_site_code, get_site_color, print_data_summary,
    )
    from .etad_factors import (
        load_etad_factor_contributions, load_etad_filter_ids,
        load_etad_factors_with_filter_ids, match_etad_factors,
        ETAD_PMF_SOURCE_NAMES, ETAD_FACTOR_RENAME,
    )
    from .plotting import (
        PlotConfig, apply_default_style, crossplots, timeseries,
        distributions, comparisons, calculate_regression_stats,
    )

__all__ = [
    # Config
    'SITES', 'PROCESSED_SITES_DIR', 'FILTER_DATA_PATH',
    'AERONET_DATA_DIR', 'WEATHER_DATA_DIR',
    'MAC_VALUE', 'FLOW_FIX_PERIODS', 'MIN_EC_THRESHOLD',
    'SMOOTH_RAW_THRESHOLDS', 'DEFAULT_BC_WAVELENGTH',
    'FILTER_CATEGORIES', 'CROSS_COMPARISONS',
    'ETAD_FACTOR_CONTRIBUTIONS_PATH', 'ETAD_FILTER_ID_PATH',
    # Outliers
    'EXCLUDED_SAMPLES', 'MANUAL_OUTLIERS',
    'apply_exclusion_flags', 'apply_threshold_flags',
    'get_clean_data', 'get_excluded_data', 'get_outlier_data',
    'print_exclusion_summary', 'identify_outlier_dates',
    # Data matching
    'load_aethalometer_data', 'load_filter_data',
    'match_aeth_filter_data', 'match_all_parameters',
    'match_with_smooth_raw_info', 'match_hips_with_smooth_raw',
    'add_flow_period_column', 'add_base_filter_id', 'match_by_filter_id',
    'pivot_filter_by_id', 'get_site_code', 'get_site_color', 'print_data_summary',
    'load_etad_factor_contributions', 'load_etad_filter_ids',
    'load_etad_factors_with_filter_ids', 'match_etad_factors',
    'ETAD_PMF_SOURCE_NAMES', 'ETAD_FACTOR_RENAME',
    # Plotting
    'PlotConfig', 'apply_default_style', 'crossplots', 'timeseries',
    'distributions', 'comparisons', 'calculate_regression_stats',
]
