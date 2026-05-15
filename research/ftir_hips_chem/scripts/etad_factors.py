"""
ETAD/Addis Ababa PMF factor loaders and match helpers.

These functions are split out from data_matching.py because the ETAD PMF
workflow is site-specific, while data_matching.py handles generic aeth/filter
matching across all sites.
"""

import pandas as pd

try:
    from config import ETAD_FACTOR_CONTRIBUTIONS_PATH, ETAD_FILTER_ID_PATH
except ImportError:  # Support importing as research.ftir_hips_chem.scripts.*
    from .config import ETAD_FACTOR_CONTRIBUTIONS_PATH, ETAD_FILTER_ID_PATH


ETAD_PMF_SOURCE_NAMES = {
    '1': 'Sea Salt Mixed',
    '2': 'Wood Burning',
    '3': 'Charcoal',
    '4': 'Polluted Marine',
    '5': 'Fossil Fuel Combustion',
}

_GF_RENAME = {f'GF{i}': f'GF{i} ({name})'
              for i, name in ETAD_PMF_SOURCE_NAMES.items()}
_KF_RENAME = {f'K_F{i}(ug/m3)': f'K_F{i} {name} (ug/m3)'
              for i, name in ETAD_PMF_SOURCE_NAMES.items()}
ETAD_FACTOR_RENAME = {**_GF_RENAME, **_KF_RENAME}


def load_etad_factor_contributions(csv_path=None):
    """
    Load ETAD (Ethiopia/Addis Ababa) PMF factor contributions CSV.

    Columns are renamed to include the source fraction:
      GF1 -> GF1 (Sea Salt Mixed)
      K_F1(ug/m3) -> K_F1 Sea Salt Mixed (ug/m3)
    """
    if csv_path is None:
        csv_path = ETAD_FACTOR_CONTRIBUTIONS_PATH

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['oldDate'], format='%m/%d/%Y')
    df = df.drop(columns=['oldDate'])
    df = df.rename(columns=ETAD_FACTOR_RENAME)

    print(f"ETAD factor contributions loaded: {len(df)} records")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    return df


def load_etad_filter_ids(csv_path=None):
    """
    Load the ETAD Filter ID mapping CSV.

    Returns a DataFrame with columns including FilterId, date, and
    base_filter_id, where base_filter_id strips the -N suffix.
    """
    if csv_path is None:
        csv_path = ETAD_FILTER_ID_PATH

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['oldDate'])
    df = df.drop(columns=['oldDate'])
    df['base_filter_id'] = df['FilterId'].str.replace(r'-\d+$', '', regex=True)

    print(f"ETAD Filter IDs loaded: {len(df)} filters")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    return df


def load_etad_factors_with_filter_ids(factor_csv_path=None, filter_id_csv_path=None):
    """
    Load ETAD factor contributions merged with Filter IDs via oldDate.

    Join chain:
        Factor Contributions.oldDate -> Filter ID.oldDate -> FilterId -> base_filter_id
    """
    factors_df = load_etad_factor_contributions(factor_csv_path)
    filter_ids_df = load_etad_filter_ids(filter_id_csv_path)

    merged = pd.merge(
        factors_df,
        filter_ids_df[['date', 'FilterId', 'base_filter_id', 'Barcode', 'LotId']],
        on='date',
        how='inner',
    )

    unmatched = len(factors_df) - len(merged)
    print(f"Merged: {len(merged)} records ({unmatched} factor rows had no matching FilterId)")

    return merged


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
        Name of the date column in target_df
    factor_csv_path : Path, optional
        Path to factor contributions CSV. Defaults to config path.
    date_tolerance_days : int
        Matching tolerance in days
    factor_cols : list[str], optional
        Specific factor columns to include. If None, includes all columns.
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

