"""
Functions for loading and matching aethalometer and filter data.

Supports matching by:
- Date (±1 day tolerance)
- FilterId (ensures same physical filter across measurements)

Usage:
    from data_matching import load_aethalometer_data, load_filter_data, match_aeth_filter_data

    # Load data
    aethalometer_data = load_aethalometer_data()
    filter_data = load_filter_data()

    # Match for a single site (by date)
    matched = match_aeth_filter_data('Beijing', aethalometer_data['Beijing'],
                                      filter_data, 'CHTS')

    # Match by FilterId (ensures same physical filter)
    matched = match_by_filter_id(filter_data, site_code='CHTS',
                                  params=['EC_ftir', 'HIPS_Fabs'])
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from config import (
    SITES, PROCESSED_SITES_DIR, FILTER_DATA_PATH,
    MIN_EC_THRESHOLD, MAC_VALUE, FLOW_FIX_PERIODS,
    ETAD_FACTOR_CONTRIBUTIONS_PATH
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_aethalometer_data(sites_dir=None, sites_config=None):
    """
    Load all aethalometer datasets.

    Parameters:
    -----------
    sites_dir : Path (optional, defaults to PROCESSED_SITES_DIR)
    sites_config : dict (optional, defaults to SITES from config)

    Returns:
    --------
    dict of DataFrames keyed by site name
    """
    if sites_dir is None:
        sites_dir = PROCESSED_SITES_DIR
    if sites_config is None:
        sites_config = SITES

    aethalometer_data = {}

    for site_name, config in sites_config.items():
        file_path = sites_dir / config['file']

        if file_path.exists():
            with open(file_path, 'rb') as f:
                df = pickle.load(f)

            # Ensure day_9am is datetime
            df['day_9am'] = pd.to_datetime(df['day_9am'])

            aethalometer_data[site_name] = df
            print(f"Loaded {site_name}: {len(df)} records, "
                  f"{df['day_9am'].min().date()} to {df['day_9am'].max().date()}")
        else:
            print(f"File not found: {file_path}")

    print(f"\nTotal sites loaded: {len(aethalometer_data)}")
    return aethalometer_data


def load_filter_data(filter_path=None):
    """
    Load unified filter dataset.

    Parameters:
    -----------
    filter_path : Path (optional, defaults to FILTER_DATA_PATH)

    Returns:
    --------
    DataFrame with filter measurements
    """
    if filter_path is None:
        filter_path = FILTER_DATA_PATH

    with open(filter_path, 'rb') as f:
        filter_data = pickle.load(f)

    filter_data['SampleDate'] = pd.to_datetime(filter_data['SampleDate'])

    print(f"Filter dataset loaded: {len(filter_data)} measurements")
    print(f"Sites: {filter_data['Site'].unique()}")
    print(f"Date range: {filter_data['SampleDate'].min().date()} to "
          f"{filter_data['SampleDate'].max().date()}")

    return filter_data


# =============================================================================
# FILTER ID PROCESSING
# =============================================================================

def add_base_filter_id(filter_data):
    """
    Add 'base_filter_id' column by stripping the -N suffix from FilterId.

    FTIR/HIPS data uses format like 'ETAD-0017-1'
    ChemSpec data uses format like 'ETAD-0017'

    Stripping the suffix allows matching across data sources.

    Parameters:
    -----------
    filter_data : DataFrame with FilterId column

    Returns:
    --------
    DataFrame with 'base_filter_id' column added
    """
    df = filter_data.copy()
    df['base_filter_id'] = df['FilterId'].str.replace(r'-\d+$', '', regex=True)
    return df


def match_by_filter_id(filter_data, site_code, params, min_concentration=None):
    """
    Match filter measurements by FilterId (same physical filter).

    This ensures that when comparing FTIR EC vs HIPS, you're comparing
    measurements from the exact same physical filter.

    Parameters:
    -----------
    filter_data : DataFrame
        Unified filter dataset (will add base_filter_id if missing)
    site_code : str
        Site code (e.g., 'CHTS', 'ETAD')
    params : list of str
        Parameter names to include (e.g., ['EC_ftir', 'HIPS_Fabs'])
    min_concentration : float (optional)
        Minimum concentration to include

    Returns:
    --------
    DataFrame with one row per filter, columns for each parameter
    """
    # Add base_filter_id if not present
    if 'base_filter_id' not in filter_data.columns:
        filter_data = add_base_filter_id(filter_data)

    # Filter to site
    site_data = filter_data[filter_data['Site'] == site_code].copy()

    if len(site_data) == 0:
        print(f"No data for site {site_code}")
        return None

    # Get unique base filter IDs
    base_ids = site_data['base_filter_id'].unique()

    matched_records = []

    for base_id in base_ids:
        filter_subset = site_data[site_data['base_filter_id'] == base_id]

        record = {
            'base_filter_id': base_id,
            'date': filter_subset['SampleDate'].iloc[0] if 'SampleDate' in filter_subset.columns else None
        }

        # Get each parameter
        for param in params:
            param_data = filter_subset[filter_subset['Parameter'] == param]

            if len(param_data) > 0:
                conc = param_data['Concentration'].iloc[0]

                # Apply minimum threshold
                if min_concentration is not None and conc < min_concentration:
                    continue

                # Standardize column names
                col_name = _param_to_column_name(param)
                record[col_name] = conc

        # Only keep if we have at least 2 parameters
        param_count = sum(1 for k in record.keys() if k not in ['base_filter_id', 'date'])
        if param_count >= 2:
            matched_records.append(record)

    if len(matched_records) == 0:
        return None

    return pd.DataFrame(matched_records)


def _param_to_column_name(param):
    """Convert parameter name to standardized column name."""
    mapping = {
        'EC_ftir': 'ftir_ec',
        'OC_ftir': 'ftir_oc',
        'HIPS_Fabs': 'hips_fabs',
        'ChemSpec_EC_PM2.5': 'chemspec_ec',
        'ChemSpec_OC_PM2.5': 'chemspec_oc',
        'ChemSpec_Iron_PM2.5': 'iron',
        'ChemSpec_BC_PM2.5': 'chemspec_bc',
    }
    return mapping.get(param, param.lower().replace(' ', '_'))


def pivot_filter_by_id(filter_data, site_code, params=None):
    """
    Pivot filter data so each row is one filter with all its measurements.

    Parameters:
    -----------
    filter_data : DataFrame
    site_code : str
    params : list (optional)
        Parameters to include. If None, includes all.

    Returns:
    --------
    DataFrame with columns: base_filter_id, date, param1, param2, ...
    """
    if 'base_filter_id' not in filter_data.columns:
        filter_data = add_base_filter_id(filter_data)

    site_data = filter_data[filter_data['Site'] == site_code].copy()

    if params is not None:
        site_data = site_data[site_data['Parameter'].isin(params)]

    # Pivot
    pivoted = site_data.pivot_table(
        index=['base_filter_id', 'SampleDate'],
        columns='Parameter',
        values='Concentration',
        aggfunc='first'
    ).reset_index()

    # Rename columns
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={'SampleDate': 'date'})

    # Rename parameter columns
    rename_map = {col: _param_to_column_name(col) for col in pivoted.columns
                  if col not in ['base_filter_id', 'date']}
    pivoted = pivoted.rename(columns=rename_map)

    return pivoted


# =============================================================================
# BASIC DATA MATCHING (BY DATE)
# =============================================================================

def match_aeth_filter_data(site_name, df_aeth, filter_data, site_code,
                            min_ec=None, date_tolerance_days=1):
    """
    Match aethalometer and filter data by date.

    Parameters:
    -----------
    site_name : str
    df_aeth : DataFrame with aethalometer data
    filter_data : DataFrame with all filter data
    site_code : str (e.g., 'CHTS', 'INDH')
    min_ec : float - minimum EC value to include (ug/m3), defaults to MIN_EC_THRESHOLD
    date_tolerance_days : int - date matching tolerance

    Returns:
    --------
    DataFrame with matched pairs:
    - date: sample date
    - aeth_bc: aethalometer BC (ng/m3)
    - filter_ec: FTIR EC (ng/m3)
    - filter_id: filter identifier
    """
    if min_ec is None:
        min_ec = MIN_EC_THRESHOLD

    # Get FTIR EC data for this site
    site_filters = filter_data[
        (filter_data['Site'] == site_code) &
        (filter_data['Parameter'] == 'EC_ftir')
    ].copy()

    # Filter out values below MDL
    site_filters = site_filters[site_filters['Concentration'] >= min_ec].copy()

    if len(site_filters) == 0:
        print(f"  {site_name}: No FTIR EC data available")
        return None

    # Match by date
    matched_records = []

    for _, filter_row in site_filters.iterrows():
        filter_date = filter_row['SampleDate']

        # Find aethalometer data within tolerance
        date_match = df_aeth[
            (df_aeth['day_9am'] >= filter_date - pd.Timedelta(days=date_tolerance_days)) &
            (df_aeth['day_9am'] <= filter_date + pd.Timedelta(days=date_tolerance_days))
        ]

        if len(date_match) > 0 and 'IR BCc' in date_match.columns:
            if date_match['IR BCc'].notna().any():
                aeth_bc = date_match['IR BCc'].mean()  # ng/m3
                filter_ec = filter_row['Concentration'] * 1000  # ug/m3 to ng/m3

                matched_records.append({
                    'date': filter_date,
                    'aeth_bc': aeth_bc,
                    'filter_ec': filter_ec,
                    'filter_id': filter_row.get('FilterId', 'unknown')
                })

    if len(matched_records) == 0:
        print(f"  {site_name}: No matched data")
        return None

    return pd.DataFrame(matched_records)


# =============================================================================
# MULTI-PARAMETER MATCHING
# =============================================================================

def match_all_parameters(site_name, site_code, df_aeth, filter_data):
    """
    Match all parameters (HIPS, FTIR EC, Iron, Aethalometer) by date.

    Returns DataFrame with columns:
    - date
    - ir_bcc (ug/m3)
    - hips_fabs (ug/m3 equivalent, divided by MAC)
    - ftir_ec (ug/m3)
    - iron (ug/m3)
    """
    def get_filter_param(param_name):
        param_data = filter_data[
            (filter_data['Site'] == site_code) &
            (filter_data['Parameter'] == param_name)
        ].copy()
        return param_data if len(param_data) > 0 else None

    hips_data = get_filter_param('HIPS_Fabs')
    ec_data = get_filter_param('EC_ftir')
    iron_data = get_filter_param('ChemSpec_Iron_PM2.5')

    # Get all unique filter dates
    all_dates = set()
    for data in [hips_data, ec_data, iron_data]:
        if data is not None:
            all_dates.update(data['SampleDate'].tolist())

    if len(all_dates) == 0:
        return None

    matched_records = []

    for filter_date in sorted(all_dates):
        record = {'date': filter_date}

        # Match aethalometer
        if df_aeth is not None and 'day_9am' in df_aeth.columns:
            date_match = df_aeth[
                (df_aeth['day_9am'] >= filter_date - pd.Timedelta(days=1)) &
                (df_aeth['day_9am'] <= filter_date + pd.Timedelta(days=1))
            ]
            if len(date_match) > 0 and 'IR BCc' in date_match.columns:
                if date_match['IR BCc'].notna().any():
                    record['ir_bcc'] = date_match['IR BCc'].mean() / 1000  # ng to ug

        # Match filter parameters
        for data, col_name, conversion in [
            (hips_data, 'hips_fabs', lambda x: x / MAC_VALUE),  # Mm^-1 to ug/m3
            (ec_data, 'ftir_ec', lambda x: x),  # Already ug/m3
            (iron_data, 'iron', lambda x: x)  # Already ug/m3
        ]:
            if data is not None:
                match = data[
                    (data['SampleDate'] >= filter_date - pd.Timedelta(days=1)) &
                    (data['SampleDate'] <= filter_date + pd.Timedelta(days=1))
                ]
                if len(match) > 0 and match['Concentration'].notna().any():
                    record[col_name] = conversion(match['Concentration'].mean())

        # Only keep if at least 2 parameters
        param_count = sum(1 for k in ['ir_bcc', 'hips_fabs', 'ftir_ec', 'iron'] if k in record)
        if param_count >= 2:
            matched_records.append(record)

    return pd.DataFrame(matched_records) if matched_records else None


# =============================================================================
# SMOOTH/RAW DIFFERENCE MATCHING
# =============================================================================

def match_with_smooth_raw_info(site_name, df_aeth, filter_data, site_code,
                                wavelength='IR', min_ec=None):
    """
    Match aethalometer and filter data, including smooth/raw difference info.

    Returns DataFrame with:
    - date
    - aeth_bc (using raw BC, ng/m3)
    - aeth_bc_smooth (ng/m3)
    - filter_ec (ng/m3)
    - smooth_raw_pct: % difference
    - smooth_raw_abs_pct: absolute % difference
    """
    if min_ec is None:
        min_ec = MIN_EC_THRESHOLD

    raw_col = f'{wavelength} BCc'
    smooth_col = f'{wavelength} BCc smoothed'

    # Check if columns exist
    if raw_col not in df_aeth.columns:
        print(f"  {site_name}: {raw_col} not found")
        return None

    has_smooth = smooth_col in df_aeth.columns

    # Get FTIR EC data for this site
    site_filters = filter_data[
        (filter_data['Site'] == site_code) &
        (filter_data['Parameter'] == 'EC_ftir')
    ].copy()
    site_filters = site_filters[site_filters['Concentration'] >= min_ec].copy()

    if len(site_filters) == 0:
        return None

    matched_records = []

    for _, filter_row in site_filters.iterrows():
        filter_date = filter_row['SampleDate']

        date_match = df_aeth[
            (df_aeth['day_9am'] >= filter_date - pd.Timedelta(days=1)) &
            (df_aeth['day_9am'] <= filter_date + pd.Timedelta(days=1))
        ]

        if len(date_match) > 0:
            bc_raw = date_match[raw_col].mean()
            bc_smooth = date_match[smooth_col].mean() if has_smooth else np.nan

            if pd.notna(bc_raw):
                filter_ec = filter_row['Concentration'] * 1000  # ug to ng

                # Calculate % difference
                if bc_raw != 0 and pd.notna(bc_smooth):
                    pct_diff = ((bc_smooth - bc_raw) / bc_raw) * 100
                else:
                    pct_diff = np.nan

                matched_records.append({
                    'date': filter_date,
                    'aeth_bc': bc_raw,
                    'aeth_bc_smooth': bc_smooth,
                    'filter_ec': filter_ec,
                    'smooth_raw_pct': pct_diff,
                    'smooth_raw_abs_pct': abs(pct_diff) if pd.notna(pct_diff) else np.nan,
                    'filter_id': filter_row.get('FilterId', 'unknown')
                })

    return pd.DataFrame(matched_records) if matched_records else None


def match_hips_with_smooth_raw(site_name, df_aeth, filter_data, site_code,
                                wavelength='IR'):
    """
    Match HIPS data with aethalometer, including smooth/raw BC info.

    This was previously duplicated in HIPS_Aeth_SmoothRaw_Analysis.ipynb
    and Multi_Site_Analysis_Modular.ipynb.

    Parameters:
    -----------
    site_name : str
    df_aeth : DataFrame with aethalometer data
    filter_data : DataFrame with filter data
    site_code : str
    wavelength : str

    Returns:
    --------
    DataFrame with columns:
    - date
    - hips_fabs (ug/m³, already divided by MAC)
    - ir_bcc (ug/m³)
    - ir_bcc_smooth (ug/m³)
    - smooth_raw_pct
    - smooth_raw_abs_pct
    - filter_id
    """
    raw_col = f'{wavelength} BCc'
    smooth_col = f'{wavelength} BCc smoothed'

    if raw_col not in df_aeth.columns:
        print(f"  {site_name}: {raw_col} not found")
        return None

    has_smooth = smooth_col in df_aeth.columns

    # Get HIPS data
    site_hips = filter_data[
        (filter_data['Site'] == site_code) &
        (filter_data['Parameter'] == 'HIPS_Fabs')
    ].copy()

    if len(site_hips) == 0:
        print(f"  {site_name}: No HIPS data")
        return None

    matched_records = []

    for _, hips_row in site_hips.iterrows():
        hips_date = hips_row['SampleDate']

        # Match aethalometer by date
        date_match = df_aeth[
            (df_aeth['day_9am'] >= hips_date - pd.Timedelta(days=1)) &
            (df_aeth['day_9am'] <= hips_date + pd.Timedelta(days=1))
        ]

        if len(date_match) > 0 and date_match[raw_col].notna().any():
            bc_raw = date_match[raw_col].mean() / 1000  # ng to ug
            bc_smooth = date_match[smooth_col].mean() / 1000 if has_smooth else np.nan

            # HIPS Fabs / MAC gives BC equivalent in ug/m³
            hips_fabs = hips_row['Concentration'] / MAC_VALUE

            # Calculate smooth/raw % difference
            if bc_raw != 0 and pd.notna(bc_smooth):
                pct_diff = ((bc_smooth - bc_raw) / bc_raw) * 100
            else:
                pct_diff = np.nan

            matched_records.append({
                'date': hips_date,
                'hips_fabs': hips_fabs,
                'ir_bcc': bc_raw,
                'ir_bcc_smooth': bc_smooth,
                'smooth_raw_pct': pct_diff,
                'smooth_raw_abs_pct': abs(pct_diff) if pd.notna(pct_diff) else np.nan,
                'filter_id': hips_row.get('FilterId', 'unknown')
            })

    if len(matched_records) == 0:
        print(f"  {site_name}: No matched HIPS data")
        return None

    return pd.DataFrame(matched_records)


# =============================================================================
# FLOW PERIOD CLASSIFICATION
# =============================================================================

def add_flow_period_column(df, site_name):
    """
    Add a column indicating whether each row is from 'before' or 'after' the flow fix.

    Parameters:
    -----------
    df : DataFrame with 'day_9am' column
    site_name : str

    Returns:
    --------
    DataFrame with 'flow_period' column added
    """
    df = df.copy()

    config = FLOW_FIX_PERIODS.get(site_name, {})
    before_end = config.get('before_end')
    after_start = config.get('after_start')

    if before_end is None and after_start is None:
        df['flow_period'] = 'all_data'
    else:
        before_end = pd.to_datetime(before_end) if before_end else None
        after_start = pd.to_datetime(after_start) if after_start else None

        def classify_period(date):
            if before_end and date <= before_end:
                return 'before_fix'
            elif after_start and date >= after_start:
                return 'after_fix'
            else:
                return 'gap_period'

        df['flow_period'] = df['day_9am'].apply(classify_period)

    return df


# =============================================================================
# ETAD FACTOR CONTRIBUTIONS MATCHING
# =============================================================================

ETAD_PMF_SOURCE_NAMES = {
    '1': 'Sea Salt Mixed',
    '2': 'Wood Burning',
    '3': 'Charcoal',
    '4': 'Polluted Marine',
    '5': 'Fossil Fuel Combustion',
}

# Column rename maps derived from source names
_GF_RENAME = {f'GF{i}': f'GF{i} ({name})'
              for i, name in ETAD_PMF_SOURCE_NAMES.items()}
_KF_RENAME = {f'K_F{i}(ug/m3)': f'K_F{i} {name} (ug/m3)'
              for i, name in ETAD_PMF_SOURCE_NAMES.items()}
ETAD_FACTOR_RENAME = {**_GF_RENAME, **_KF_RENAME}


def load_etad_factor_contributions(csv_path=None):
    """
    Load ETAD (Ethiopia/Addis Ababa) PMF factor contributions CSV.

    Columns are renamed to include the source fraction:
      GF1 -> GF1 (Sea Salt Mixed)          (daily contribution fraction)
      K_F1(ug/m3) -> K_F1 Sea Salt Mixed (ug/m3)  (daily concentration)

    Source fractions (from Naveed's PMF analysis):
      F1: Sea Salt Mixed
      F2: Wood Burning
      F3: Charcoal
      F4: Polluted Marine
      F5: Fossil Fuel Combustion

    Parameters:
    -----------
    csv_path : Path (optional, defaults to ETAD_FACTOR_CONTRIBUTIONS_PATH)

    Returns:
    --------
    DataFrame with parsed dates and renamed factor contribution columns
    """
    if csv_path is None:
        csv_path = ETAD_FACTOR_CONTRIBUTIONS_PATH

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['oldDate'], format='%m/%d/%Y')
    df = df.drop(columns=['oldDate'])

    # Rename to descriptive source names
    df = df.rename(columns=ETAD_FACTOR_RENAME)

    print(f"ETAD factor contributions loaded: {len(df)} records")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    return df


def match_etad_factors(target_df, target_date_col='date',
                       factor_csv_path=None, date_tolerance_days=1,
                       factor_cols=None):
    """
    Match a target DataFrame to ETAD factor contributions by date.

    Parameters:
    -----------
    target_df : DataFrame
        DataFrame to match against (must have a date column)
    target_date_col : str
        Name of the date column in target_df (default: 'date')
    factor_csv_path : Path (optional)
        Path to factor contributions CSV. Defaults to config path.
    date_tolerance_days : int
        Matching tolerance in days (default: 1)
    factor_cols : list of str (optional)
        Specific factor columns to include. If None, includes all
        GF and K_F columns.

    Returns:
    --------
    DataFrame with target data merged with matched factor contributions
    """
    factors_df = load_etad_factor_contributions(factor_csv_path)

    if factor_cols is not None:
        keep_cols = ['date'] + [c for c in factor_cols if c in factors_df.columns]
        factors_df = factors_df[keep_cols]

    target = target_df.copy()
    target[target_date_col] = pd.to_datetime(target[target_date_col])

    tolerance = pd.Timedelta(days=date_tolerance_days)

    matched_records = []

    for _, row in target.iterrows():
        t_date = row[target_date_col]

        date_match = factors_df[
            (factors_df['date'] >= t_date - tolerance) &
            (factors_df['date'] <= t_date + tolerance)
        ]

        if len(date_match) > 0:
            # Use closest date if multiple matches
            closest_idx = (date_match['date'] - t_date).abs().idxmin()
            factor_row = date_match.loc[closest_idx]

            record = row.to_dict()
            for col in factors_df.columns:
                if col != 'date':
                    record[col] = factor_row[col]
            record['factor_date'] = factor_row['date']
            matched_records.append(record)

    if len(matched_records) == 0:
        print("No date matches found between target data and ETAD factors")
        return None

    result = pd.DataFrame(matched_records)
    print(f"Matched {len(result)}/{len(target)} records to ETAD factor contributions")

    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_site_code(site_name):
    """Get the site code for a site name."""
    return SITES.get(site_name, {}).get('code')


def get_site_color(site_name):
    """Get the plot color for a site name."""
    return SITES.get(site_name, {}).get('color', '#333333')


def print_data_summary(aethalometer_data, filter_data):
    """Print a summary of loaded data."""
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)

    for site_name, df in aethalometer_data.items():
        print(f"\n{site_name}:")
        print(f"  Records: {len(df)}")
        print(f"  Date range: {df['day_9am'].min().date()} to {df['day_9am'].max().date()}")

        if 'IR BCc' in df.columns:
            valid_bc = df['IR BCc'].notna().sum()
            print(f"  Days with IR BC: {valid_bc} ({100*valid_bc/len(df):.1f}%)")

    print(f"\nFilter data: {len(filter_data)} measurements")
    print(f"  Sites: {sorted(filter_data['Site'].unique())}")
    print(f"  Parameters: {filter_data['Parameter'].nunique()}")
