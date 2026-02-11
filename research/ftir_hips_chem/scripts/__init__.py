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
    from plotting import (
        plot_crossplot, plot_before_after_comparison,
        calculate_regression_stats, create_tiled_threshold_plots,
        plot_smooth_raw_distribution
    )

    # Configure matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10

    print("Modules loaded successfully!")
"""

# For convenient imports when using: from scripts import *
from config import (
    SITES, PROCESSED_SITES_DIR, FILTER_DATA_PATH,
    MAC_VALUE, FLOW_FIX_PERIODS, MIN_EC_THRESHOLD,
    SMOOTH_RAW_THRESHOLDS, DEFAULT_BC_WAVELENGTH,
    FILTER_CATEGORIES, CROSS_COMPARISONS,
    ETAD_FACTOR_CONTRIBUTIONS_PATH
)

from outliers import (
    EXCLUDED_SAMPLES, MANUAL_OUTLIERS,
    apply_exclusion_flags, apply_threshold_flags,
    get_clean_data, get_excluded_data, get_outlier_data,
    print_exclusion_summary, identify_outlier_dates
)

from data_matching import (
    load_aethalometer_data, load_filter_data,
    match_aeth_filter_data, match_all_parameters,
    match_with_smooth_raw_info, add_flow_period_column,
    get_site_code, get_site_color, print_data_summary,
    load_etad_factor_contributions, match_etad_factors,
    ETAD_PMF_SOURCE_NAMES, ETAD_FACTOR_RENAME
)

from plotting import (
    calculate_regression_stats,
    plot_crossplot, plot_before_after_comparison,
    plot_crossplot_iron_gradient,
    create_tiled_threshold_plots, plot_smooth_raw_distribution,
    plot_bc_timeseries, plot_multiwavelength_bc,
    print_comparison_table
)

__all__ = [
    # Config
    'SITES', 'PROCESSED_SITES_DIR', 'FILTER_DATA_PATH',
    'MAC_VALUE', 'FLOW_FIX_PERIODS', 'MIN_EC_THRESHOLD',
    'SMOOTH_RAW_THRESHOLDS', 'DEFAULT_BC_WAVELENGTH',
    'FILTER_CATEGORIES', 'CROSS_COMPARISONS',
    'ETAD_FACTOR_CONTRIBUTIONS_PATH',
    # Outliers
    'EXCLUDED_SAMPLES', 'MANUAL_OUTLIERS',
    'apply_exclusion_flags', 'apply_threshold_flags',
    'get_clean_data', 'get_excluded_data', 'get_outlier_data',
    'print_exclusion_summary', 'identify_outlier_dates',
    # Data matching
    'load_aethalometer_data', 'load_filter_data',
    'match_aeth_filter_data', 'match_all_parameters',
    'match_with_smooth_raw_info', 'add_flow_period_column',
    'get_site_code', 'get_site_color', 'print_data_summary',
    'load_etad_factor_contributions', 'match_etad_factors',
    'ETAD_PMF_SOURCE_NAMES', 'ETAD_FACTOR_RENAME',
    # Plotting
    'calculate_regression_stats',
    'plot_crossplot', 'plot_before_after_comparison',
    'plot_crossplot_iron_gradient',
    'create_tiled_threshold_plots', 'plot_smooth_raw_distribution',
    'plot_bc_timeseries', 'plot_multiwavelength_bc',
    'print_comparison_table',
]
