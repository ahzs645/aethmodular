"""
Flow period classification and analysis functions.

Consolidates flow period logic that was duplicated across:
- FlowFix_BeforeAfter_Analysis.ipynb
- Multi_Site_Analysis_Modular.ipynb

Usage:
    from flow_periods import classify_flow_period, add_flow_period, FLOW_FIX_DATES

    # Classify a single date
    period = classify_flow_period(date, 'Beijing')

    # Add flow_period column to DataFrame
    df = add_flow_period(df, 'JPL')

    # Check if site has data in both periods
    if has_before_after_data('JPL'):
        ...
"""

import pandas as pd
import numpy as np


# =============================================================================
# FLOW FIX DATE CONFIGURATIONS
# =============================================================================

# These dates define the before/after periods for each site
# Updated from FlowFix_BeforeAfter_Analysis.ipynb

FLOW_FIX_DATES = {
    'Beijing': {
        'before_end': '2022-07-31',
        'after_start': '2023-09-01',
        'description': 'NO BEFORE DATA - Filter sampling started Sep 2023 (degraded period only)',
        'has_before_data': False,
        'flow_ratio_note': 'Ratio ~1.8-2.2 in available data period'
    },
    'Delhi': {
        'before_end': '2023-12-31',
        'after_start': '2024-02-01',
        'description': 'NO BEFORE DATA - Filter sampling started Feb 2024 (degraded period only)',
        'has_before_data': False,
        'flow_ratio_note': 'Ratio ~2.5-3.2 in available data period'
    },
    'JPL': {
        'before_end': '2022-09-30',
        'after_start': '2023-05-01',
        'description': 'Has data in both periods - suitable for before/after analysis',
        'has_before_data': True,
        'flow_ratio_note': 'Good flow ratio throughout'
    },
    'Addis_Ababa': {
        'before_end': None,
        'after_start': None,
        'description': 'No flow fix periods defined',
        'has_before_data': False,
        'flow_ratio_note': 'Consistently low flow ratio'
    }
}


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_flow_period(date, site_name):
    """
    Classify a date as 'before', 'after', or 'gap' based on flow fix periods.

    Parameters:
    -----------
    date : datetime-like
    site_name : str

    Returns:
    --------
    str: 'before', 'after', 'gap', or 'no_fix'
    """
    dates = FLOW_FIX_DATES.get(site_name, {})
    before_end = dates.get('before_end')
    after_start = dates.get('after_start')

    if before_end is None and after_start is None:
        return 'no_fix'

    date = pd.to_datetime(date)
    before_end_dt = pd.to_datetime(before_end) if before_end else None
    after_start_dt = pd.to_datetime(after_start) if after_start else None

    if before_end_dt and date <= before_end_dt:
        return 'before'
    elif after_start_dt and date >= after_start_dt:
        return 'after'
    else:
        return 'gap'


def add_flow_period(df, site_name, date_col='date'):
    """
    Add 'flow_period' column to DataFrame.

    Parameters:
    -----------
    df : DataFrame
    site_name : str
    date_col : str
        Column name containing dates (default 'date', also tries 'day_9am')

    Returns:
    --------
    DataFrame with 'flow_period' column added
    """
    df = df.copy()

    # Try to find the date column
    if date_col not in df.columns:
        if 'day_9am' in df.columns:
            date_col = 'day_9am'
        elif 'SampleDate' in df.columns:
            date_col = 'SampleDate'
        else:
            raise ValueError(f"Date column '{date_col}' not found. "
                           f"Available: {df.columns.tolist()}")

    df['flow_period'] = df[date_col].apply(lambda d: classify_flow_period(d, site_name))
    return df


def get_period_data(df, period, site_name=None, date_col='date'):
    """
    Filter DataFrame to specific flow period.

    Parameters:
    -----------
    df : DataFrame
    period : str
        'before', 'after', 'gap', or 'all'
    site_name : str (optional)
        If provided and flow_period not in df, will add it
    date_col : str

    Returns:
    --------
    DataFrame filtered to period
    """
    if period == 'all':
        return df

    if 'flow_period' not in df.columns:
        if site_name is None:
            raise ValueError("flow_period column not found and site_name not provided")
        df = add_flow_period(df, site_name, date_col)

    return df[df['flow_period'] == period].copy()


# =============================================================================
# DATA AVAILABILITY CHECKS
# =============================================================================

def has_before_after_data(site_name):
    """
    Check if a site has filter data in both before and after periods.

    Parameters:
    -----------
    site_name : str

    Returns:
    --------
    bool
    """
    return FLOW_FIX_DATES.get(site_name, {}).get('has_before_data', False)


def get_sites_with_before_after():
    """
    Get list of sites that have data in both before and after periods.

    Returns:
    --------
    list of site names
    """
    return [site for site, config in FLOW_FIX_DATES.items()
            if config.get('has_before_data', False)]


def print_flow_period_summary():
    """Print summary of flow period data availability."""
    print("Flow Fix Analysis - Data Availability:")
    print("=" * 80)

    for site, dates in FLOW_FIX_DATES.items():
        if dates.get('before_end') is None and dates.get('after_start') is None:
            continue

        print(f"\n{site}:")
        print(f"  Status: {dates.get('description', 'Unknown')}")
        print(f"  Flow ratio: {dates.get('flow_ratio_note', 'Unknown')}")

        if dates.get('has_before_data'):
            print(f"  ✓ SUITABLE for before/after analysis")
            print(f"    Before: <= {dates.get('before_end')}")
            print(f"    After:  >= {dates.get('after_start')}")
        else:
            print(f"  ✗ NOT SUITABLE - no filter data in 'before' period")

    print("\n" + "=" * 80)
    suitable = get_sites_with_before_after()
    print(f"Sites suitable for before/after comparison: {suitable}")
    print("=" * 80)


# =============================================================================
# PERIOD STATISTICS
# =============================================================================

def calculate_period_stats(df, site_name, x_col, y_col, date_col='date'):
    """
    Calculate regression statistics for before and after periods.

    Parameters:
    -----------
    df : DataFrame
    site_name : str
    x_col, y_col : str
        Column names for x and y data
    date_col : str

    Returns:
    --------
    dict: {'before': stats, 'after': stats}
    """
    from plotting.utils import calculate_regression_stats

    if 'flow_period' not in df.columns:
        df = add_flow_period(df, site_name, date_col)

    results = {}

    for period in ['before', 'after']:
        period_data = df[df['flow_period'] == period]

        if len(period_data) >= 3:
            x_data = period_data[x_col].dropna().values
            y_data = period_data[y_col].dropna().values

            # Align arrays
            mask = (~np.isnan(period_data[x_col].values)) & (~np.isnan(period_data[y_col].values))
            x_data = period_data[x_col].values[mask]
            y_data = period_data[y_col].values[mask]

            if len(x_data) >= 3:
                stats = calculate_regression_stats(x_data, y_data)
                results[period] = stats
            else:
                results[period] = None
        else:
            results[period] = None

    return results


def compare_periods(before_stats, after_stats):
    """
    Compare statistics between before and after periods.

    Parameters:
    -----------
    before_stats, after_stats : dict
        Output from calculate_regression_stats

    Returns:
    --------
    dict with comparison metrics
    """
    if before_stats is None or after_stats is None:
        return None

    return {
        'r2_before': before_stats['r_squared'],
        'r2_after': after_stats['r_squared'],
        'r2_change': after_stats['r_squared'] - before_stats['r_squared'],
        'slope_before': before_stats['slope'],
        'slope_after': after_stats['slope'],
        'slope_change': after_stats['slope'] - before_stats['slope'],
        'n_before': before_stats['n'],
        'n_after': after_stats['n'],
        # How much closer to 1.0 did the slope get?
        'slope_improvement': abs(before_stats['slope'] - 1) - abs(after_stats['slope'] - 1)
    }


def print_period_comparison(site_name, comparison):
    """Print formatted comparison of before/after periods."""
    if comparison is None:
        print(f"{site_name}: Insufficient data for comparison")
        return

    print(f"\n{site_name}:")
    print(f"  {'Metric':<15s} {'Before':>12s} {'After':>12s} {'Change':>12s}")
    print(f"  {'-'*55}")
    print(f"  {'n':<15s} {comparison['n_before']:>12d} {comparison['n_after']:>12d} "
          f"{comparison['n_after']-comparison['n_before']:>+12d}")
    print(f"  {'R²':<15s} {comparison['r2_before']:>12.3f} {comparison['r2_after']:>12.3f} "
          f"{comparison['r2_change']:>+12.3f}")
    print(f"  {'Slope':<15s} {comparison['slope_before']:>12.3f} {comparison['slope_after']:>12.3f} "
          f"{comparison['slope_change']:>+12.3f}")

    if comparison['r2_change'] > 0.05:
        print(f"\n  ✓ R² IMPROVED by {comparison['r2_change']:.3f}")
    elif comparison['r2_change'] < -0.05:
        print(f"\n  ✗ R² DECREASED by {abs(comparison['r2_change']):.3f}")
    else:
        print(f"\n  ~ R² relatively unchanged ({comparison['r2_change']:+.3f})")

    if comparison['slope_improvement'] > 0.05:
        print(f"  ✓ Slope moved closer to 1.0")
    elif comparison['slope_improvement'] < -0.05:
        print(f"  ✗ Slope moved further from 1.0")
