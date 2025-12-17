"""
Centralized outlier registry and exclusion functions.
All manual outlier definitions live here for consistency across analyses.

Usage:
    from outliers import EXCLUDED_SAMPLES, apply_exclusion_flags, get_clean_data

    # Apply exclusions to matched data
    matched_df = apply_exclusion_flags(matched_df, site_name)

    # Get only clean (non-excluded) data
    clean_df = get_clean_data(matched_df)
"""

import pandas as pd
import numpy as np

# =============================================================================
# EXCLUDED SAMPLES REGISTRY
# =============================================================================
#
# Add outliers here as you identify them.
# Each entry should have: date, reason, and optionally filter_id/approx values
#
# To find exact dates, run identify_outlier_dates() with your criteria
#

EXCLUDED_SAMPLES = {
    'Beijing': [
        {
            'date': '2022-05-15',  # TODO: Update with actual date from your data
            'filter_id': None,
            'aeth_bc_approx': 4700,
            'reason': 'Extreme aethalometer BC outlier - affects slope significantly'
        },
    ],
    'Delhi': [
        {
            'date': '2023-01-10',  # TODO: Update with actual dates
            'filter_id': None,
            'aeth_bc_approx': 10000,
            'reason': 'High aethalometer (~10000) with low FTIR EC (~2000) - measurement issue'
        },
        {
            'date': '2023-01-15',
            'filter_id': None,
            'aeth_bc_approx': 9500,
            'reason': 'High aethalometer (~9500) with low FTIR EC (~2500) - measurement issue'
        },
    ],
    'JPL': [
        {
            'date': '2022-06-20',  # TODO: Update with actual date
            'filter_id': None,
            'aeth_bc_approx': 1800,
            'reason': 'Pre-flow-fix period - aethalometer BC > 1700 threshold'
        },
        {
            'date': '2022-07-05',
            'filter_id': None,
            'filter_ec_approx': 1200,
            'reason': 'FTIR EC > 1000 threshold - likely contamination'
        },
    ],
    'Addis_Ababa': []
}

# =============================================================================
# MANUAL OUTLIER THRESHOLDS (for automatic flagging)
# =============================================================================
#
# These thresholds define automatic outlier detection criteria per site.
# Used when you don't have specific dates but want to flag by value.
#

MANUAL_OUTLIERS = {
    'Beijing': {
        'description': 'Remove 1 point with extremely high aethalometer BC (far right)',
        'remove_criteria': [
            {'type': 'high_aeth', 'aeth_bc_min': 4000}
        ]
    },
    'Delhi': {
        'description': 'Remove 2 points with high aethalometer but low FTIR EC',
        'remove_criteria': [
            {'type': 'high_aeth_low_ec', 'aeth_bc_min': 8000, 'filter_ec_max': 3000}
        ]
    },
    'JPL': {
        'description': 'Remove point(s) with FTIR EC > 1000 OR aethalometer > 1700',
        'remove_criteria': [
            {'type': 'high_either', 'aeth_bc_min': 1700, 'filter_ec_min': 1000}
        ]
    },
    'Addis_Ababa': {
        'description': 'No manual outliers identified yet',
        'remove_criteria': []
    }
}


# =============================================================================
# EXCLUSION FUNCTIONS
# =============================================================================

def apply_exclusion_flags(matched_df, site_name, date_tolerance_days=1):
    """
    Apply exclusion flags to matched data based on the EXCLUDED_SAMPLES registry.

    Parameters:
    -----------
    matched_df : DataFrame with 'date' column
    site_name : str
    date_tolerance_days : int - tolerance for date matching (default +/-1 day)

    Returns:
    --------
    DataFrame with added columns:
    - 'is_excluded': Boolean flag
    - 'exclusion_reason': String explaining why (empty if not excluded)
    """
    matched_df = matched_df.copy()
    matched_df['is_excluded'] = False
    matched_df['exclusion_reason'] = ''

    exclusions = EXCLUDED_SAMPLES.get(site_name, [])

    if len(exclusions) == 0:
        return matched_df

    for exclusion in exclusions:
        excl_date = pd.to_datetime(exclusion['date'])
        excl_filter_id = exclusion.get('filter_id')
        excl_reason = exclusion.get('reason', 'Manual exclusion')

        # Match by date (with tolerance)
        date_mask = (
            (matched_df['date'] >= excl_date - pd.Timedelta(days=date_tolerance_days)) &
            (matched_df['date'] <= excl_date + pd.Timedelta(days=date_tolerance_days))
        )

        # Optionally also match by filter_id if provided
        if excl_filter_id and 'filter_id' in matched_df.columns:
            date_mask = date_mask & (matched_df['filter_id'] == excl_filter_id)

        # Apply exclusion
        matched_df.loc[date_mask, 'is_excluded'] = True

        # Append reason (in case multiple exclusion criteria match)
        for idx in matched_df[date_mask].index:
            existing_reason = matched_df.loc[idx, 'exclusion_reason']
            if existing_reason:
                matched_df.loc[idx, 'exclusion_reason'] = f"{existing_reason}; {excl_reason}"
            else:
                matched_df.loc[idx, 'exclusion_reason'] = excl_reason

    return matched_df


def apply_threshold_flags(matched_df, site_name):
    """
    Apply outlier flags based on MANUAL_OUTLIERS thresholds.

    Parameters:
    -----------
    matched_df : DataFrame with 'aeth_bc' and 'filter_ec' columns
    site_name : str

    Returns:
    --------
    DataFrame with 'is_outlier' and 'outlier_reason' columns added
    """
    matched_df = matched_df.copy()
    matched_df['is_outlier'] = False
    matched_df['outlier_reason'] = ''

    config = MANUAL_OUTLIERS.get(site_name, {'remove_criteria': []})
    criteria_list = config.get('remove_criteria', [])

    for criteria in criteria_list:
        ctype = criteria.get('type', '')

        if ctype == 'high_aeth':
            threshold = criteria.get('aeth_bc_min', np.inf)
            mask = matched_df['aeth_bc'] > threshold
            matched_df.loc[mask, 'is_outlier'] = True
            matched_df.loc[mask, 'outlier_reason'] += f'high_aeth(>{threshold}); '

        elif ctype == 'low_aeth':
            threshold = criteria.get('aeth_bc_max', 0)
            mask = matched_df['aeth_bc'] < threshold
            matched_df.loc[mask, 'is_outlier'] = True
            matched_df.loc[mask, 'outlier_reason'] += f'low_aeth(<{threshold}); '

        elif ctype == 'high_aeth_low_ec':
            aeth_min = criteria.get('aeth_bc_min', np.inf)
            ec_max = criteria.get('filter_ec_max', 0)
            mask = (matched_df['aeth_bc'] > aeth_min) & (matched_df['filter_ec'] < ec_max)
            matched_df.loc[mask, 'is_outlier'] = True
            matched_df.loc[mask, 'outlier_reason'] += 'high_aeth_low_ec; '

        elif ctype == 'low_aeth_high_ec':
            aeth_max = criteria.get('aeth_bc_max', 0)
            ec_min = criteria.get('filter_ec_min', np.inf)
            mask = (matched_df['aeth_bc'] < aeth_max) & (matched_df['filter_ec'] > ec_min)
            matched_df.loc[mask, 'is_outlier'] = True
            matched_df.loc[mask, 'outlier_reason'] += 'low_aeth_high_ec; '

        elif ctype == 'high_both':
            aeth_min = criteria.get('aeth_bc_min', np.inf)
            ec_min = criteria.get('filter_ec_min', np.inf)
            mask = (matched_df['aeth_bc'] > aeth_min) & (matched_df['filter_ec'] > ec_min)
            matched_df.loc[mask, 'is_outlier'] = True
            matched_df.loc[mask, 'outlier_reason'] += f'high_both; '

        elif ctype == 'high_either':
            aeth_min = criteria.get('aeth_bc_min', np.inf)
            ec_min = criteria.get('filter_ec_min', np.inf)
            mask = (matched_df['aeth_bc'] > aeth_min) | (matched_df['filter_ec'] > ec_min)
            matched_df.loc[mask, 'is_outlier'] = True
            matched_df.loc[mask, 'outlier_reason'] += f'high_either; '

    return matched_df


def get_clean_data(matched_df):
    """
    Return only non-excluded data points.

    Checks for both 'is_excluded' (date-based) and 'is_outlier' (threshold-based) columns.
    """
    df = matched_df.copy()

    if 'is_excluded' in df.columns:
        df = df[~df['is_excluded']]

    if 'is_outlier' in df.columns:
        df = df[~df['is_outlier']]

    return df


def get_excluded_data(matched_df):
    """Return only excluded data points."""
    if 'is_excluded' not in matched_df.columns:
        return pd.DataFrame()
    return matched_df[matched_df['is_excluded']].copy()


def get_outlier_data(matched_df):
    """Return only outlier data points (threshold-based)."""
    if 'is_outlier' not in matched_df.columns:
        return pd.DataFrame()
    return matched_df[matched_df['is_outlier']].copy()


def print_exclusion_summary(matched_df, site_name):
    """Print summary of excluded vs retained data."""
    n_total = len(matched_df)

    n_excluded = matched_df['is_excluded'].sum() if 'is_excluded' in matched_df.columns else 0
    n_outlier = matched_df['is_outlier'].sum() if 'is_outlier' in matched_df.columns else 0

    # Combined exclusions (either flag)
    if 'is_excluded' in matched_df.columns and 'is_outlier' in matched_df.columns:
        n_any = (matched_df['is_excluded'] | matched_df['is_outlier']).sum()
    elif 'is_excluded' in matched_df.columns:
        n_any = n_excluded
    elif 'is_outlier' in matched_df.columns:
        n_any = n_outlier
    else:
        n_any = 0

    print(f"\n{site_name} Exclusion Summary:")
    print(f"  Total: {n_total}")
    print(f"  Date-based exclusions: {n_excluded}")
    print(f"  Threshold-based outliers: {n_outlier}")
    print(f"  Combined excluded: {n_any}")
    print(f"  Retained: {n_total - n_any}")

    # Show excluded points
    if n_excluded > 0 and 'is_excluded' in matched_df.columns:
        excluded = matched_df[matched_df['is_excluded']]
        print(f"\n  Date-based excluded points:")
        for _, row in excluded.iterrows():
            date_str = str(row['date'].date()) if pd.notna(row['date']) else 'N/A'
            reason = row.get('exclusion_reason', 'N/A')[:50]
            print(f"    {date_str}: {reason}")


def identify_outlier_dates(site_name, matched_df, criteria):
    """
    Helper to find exact dates of outliers based on criteria.
    Use this to populate the EXCLUDED_SAMPLES registry with the correct dates.

    Parameters:
    -----------
    site_name : str
    matched_df : DataFrame
    criteria : dict with keys like 'aeth_bc_min', 'filter_ec_max', etc.

    Returns:
    --------
    List of dicts ready to paste into EXCLUDED_SAMPLES
    """
    outliers = []

    aeth_col = 'aeth_bc' if 'aeth_bc' in matched_df.columns else 'ir_bcc'
    ec_col = 'filter_ec' if 'filter_ec' in matched_df.columns else 'ftir_ec'

    for _, row in matched_df.iterrows():
        is_outlier = False
        reasons = []

        aeth_val = row.get(aeth_col, np.nan)
        ec_val = row.get(ec_col, np.nan)

        # Check each criterion
        if 'aeth_bc_min' in criteria and pd.notna(aeth_val):
            if aeth_val > criteria['aeth_bc_min']:
                is_outlier = True
                reasons.append(f"Aeth BC ({aeth_val:.0f}) > {criteria['aeth_bc_min']}")

        if 'filter_ec_min' in criteria and pd.notna(ec_val):
            if ec_val > criteria['filter_ec_min']:
                is_outlier = True
                reasons.append(f"Filter EC ({ec_val:.0f}) > {criteria['filter_ec_min']}")

        if 'filter_ec_max' in criteria and pd.notna(ec_val):
            if ec_val < criteria['filter_ec_max']:
                if 'aeth_bc_min' in criteria and pd.notna(aeth_val) and aeth_val > criteria['aeth_bc_min']:
                    is_outlier = True
                    reasons.append(f"High aeth ({aeth_val:.0f}) with low EC ({ec_val:.0f})")

        if is_outlier:
            outliers.append({
                'date': str(row['date'].date()),
                'filter_id': row.get('filter_id'),
                'aeth_bc_approx': aeth_val,
                'filter_ec_approx': ec_val,
                'reason': '; '.join(reasons)
            })

    # Print in format ready to copy-paste
    print(f"\n# {site_name} outliers to add to EXCLUDED_SAMPLES:")
    print(f"'{site_name}': [")
    for o in outliers:
        print(f"    {{")
        print(f"        'date': '{o['date']}',")
        print(f"        'filter_id': {repr(o['filter_id'])},")
        print(f"        'aeth_bc_approx': {o['aeth_bc_approx']:.0f},")
        print(f"        'reason': '{o['reason']}'")
        print(f"    }},")
    print(f"],")

    return outliers
