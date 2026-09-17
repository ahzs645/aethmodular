"""
ETAD/Addis Ababa PMF factor loaders and match helpers.

These functions are split out from data_matching.py because the ETAD PMF
workflow is site-specific, while data_matching.py handles generic aeth/filter
matching across all sites.
"""

import numpy as np
import pandas as pd

try:
    from config import (
        BASE_FILTER_ID_PATTERN, BASE_FILTER_ID_REPL,
        ETAD_FACTOR_CONTRIBUTIONS_PATH, ETAD_FILTER_ID_PATH,
    )
except ImportError:  # Support importing as research.ftir_hips_chem.scripts.*
    from .config import (
        BASE_FILTER_ID_PATTERN, BASE_FILTER_ID_REPL,
        ETAD_FACTOR_CONTRIBUTIONS_PATH, ETAD_FILTER_ID_PATH,
    )


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
    df['base_filter_id'] = df['FilterId'].str.replace(
        BASE_FILTER_ID_PATTERN, BASE_FILTER_ID_REPL, regex=True
    )

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



# =============================================================================
# GF FRACTION NORMALIZATION
# =============================================================================

# Maps the renamed GF columns to the short fraction names the notebooks use.
FACTOR_TO_FRAC = {
    'GF1 (Sea Salt Mixed)':          'sea_salt_frac',
    'GF2 (Wood Burning)':            'wood_frac',
    'GF3 (Charcoal)':                'charcoal_frac',
    'GF4 (Polluted Marine)':         'polluted_marine_frac',
    'GF5 (Fossil Fuel Combustion)':  'fossil_fuel_frac',
}

GF_FRACTION_COLUMNS = list(FACTOR_TO_FRAC.values())


def normalize_gf_fractions(factors_df, frac_cols=None, rename=True):
    """Convert raw GF columns into relative source contributions.

    **This step is mandatory and easy to forget.** The raw ``GF1``-``GF5`` values
    in ``ETAD Factor Contributions .csv`` are PM2.5 *mass* fractions that sum to
    roughly 0.03-0.46 per row -- they are not relative source contributions.
    Used unnormalized, ``dominant_fraction`` tops out near 0.24 and no sample
    ever crosses a 30 % threshold; after normalization the mean is about 46 %.
    See the "Data join quirks" section of AGENTS.md.

    Each fraction column is divided by that row's sum across all fraction
    columns, so the columns sum to 1.0 per row.

    Parameters
    ----------
    factors_df : DataFrame
        Output of ``load_etad_factor_contributions`` or
        ``load_etad_factors_with_filter_ids``.
    frac_cols : list of str, optional
        Fraction columns to normalize. Defaults to the five GF fractions.
    rename : bool
        Rename ``GFn (Source)`` columns to short ``*_frac`` names first
        (the form the notebooks use). Set False if already renamed.

    Returns
    -------
    DataFrame
        A copy with the fraction columns normalized. Rows whose fractions sum to
        0 yield NaN rather than inf.
    """
    df = factors_df.copy()

    if rename:
        df = df.rename(columns=FACTOR_TO_FRAC)

    if frac_cols is None:
        frac_cols = [c for c in GF_FRACTION_COLUMNS if c in df.columns]

    missing = [c for c in frac_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"fraction columns not found: {missing}. "
            f"Available: {sorted(df.columns)}"
        )
    if not frac_cols:
        # Fail loudly rather than returning the frame untouched: a silent no-op
        # here looks exactly like a successful normalization to the caller, and
        # every downstream threshold comparison would then be wrong.
        raise KeyError(
            "no GF fraction columns found to normalize. Expected some of "
            f"{GF_FRACTION_COLUMNS} (or pass frac_cols explicitly). "
            f"Available: {sorted(df.columns)}"
        )

    frac_sum = df[frac_cols].sum(axis=1)
    # Guard the degenerate all-zero row: division would give inf/-inf, which
    # then silently wins an idxmax() for dominant source. np.nan (not pd.NA)
    # keeps the columns float64 -- pd.NA would upcast them to object.
    frac_sum = frac_sum.replace(0, np.nan)

    for col in frac_cols:
        df[col] = df[col] / frac_sum

    return df


def add_dominant_source(df, frac_cols=None, suffix='_frac'):
    """Add ``dominant_source`` / ``dominant_fraction`` from normalized fractions.

    Call ``normalize_gf_fractions`` first -- on raw GF values the fraction is a
    share of PM2.5 mass, not of the source mix, and every threshold comparison
    is wrong.
    """
    out = df.copy()
    if frac_cols is None:
        frac_cols = [c for c in GF_FRACTION_COLUMNS if c in out.columns]
    if not frac_cols:
        raise KeyError("no fraction columns found; pass frac_cols explicitly")

    fractions = out[frac_cols]
    # idxmax on an all-NA row emits a FutureWarning today and raises ValueError
    # in a later pandas. Compute only on rows that have at least one value.
    usable = fractions.notna().any(axis=1)

    out['dominant_source'] = pd.Series(None, index=out.index, dtype='object')
    out['dominant_fraction'] = pd.Series(float('nan'), index=out.index, dtype='float64')

    if usable.any():
        winners = fractions.loc[usable].idxmax(axis=1)
        out.loc[usable, 'dominant_source'] = winners.str.replace(suffix, '', regex=False)
        out.loc[usable, 'dominant_fraction'] = fractions.loc[usable].max(axis=1)

    return out


def attach_factors_by_date(df, factors=None, cols=None, normalize=True,
                           dominant=True, tolerance_days=0) -> pd.DataFrame:
    """Attach ETAD PMF factors to a frame using its calendar dates.

    A datetime index is used when present; otherwise ``df`` must contain a
    ``date`` column. Datetimes are normalized to midnight before matching and
    timezone information is removed only when it is present. With a positive
    ``tolerance_days``, the nearest factor date within that tolerance is used.

    Parameters
    ----------
    df : DataFrame
        Target data with a DatetimeIndex or a ``date`` column.
    factors : DataFrame, optional
        Factor data containing a ``date`` column. When omitted, factors are
        loaded with ``load_etad_factors_with_filter_ids``.
    cols : sequence of str, optional
        Factor columns to attach. Defaults to the normalized GF fraction
        columns. Dominance columns are always included when ``dominant=True``.
    normalize : bool
        Normalize raw GF mass fractions to relative source contributions.
    dominant : bool
        Add and attach ``dominant_source`` and ``dominant_fraction``.
    tolerance_days : int or float
        Maximum distance in days for nearest-date matching. Zero requires an
        exact calendar-date match.

    Returns
    -------
    DataFrame
        A copy of ``df`` with the requested factor columns attached.
    """
    if tolerance_days < 0:
        raise ValueError("tolerance_days must be non-negative")

    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        target_dates = out.index.normalize()
        if target_dates.tz is not None:
            target_dates = target_dates.tz_localize(None)
    elif 'date' in out.columns:
        target_dates = pd.DatetimeIndex(pd.to_datetime(out['date'])).normalize()
        if target_dates.tz is not None:
            target_dates = target_dates.tz_localize(None)
    else:
        raise TypeError("df must have a DatetimeIndex or a 'date' column")

    factor_data = (
        load_etad_factors_with_filter_ids() if factors is None else factors.copy()
    )
    if 'date' not in factor_data.columns:
        raise KeyError("factors must contain a 'date' column")

    if normalize:
        factor_data = normalize_gf_fractions(factor_data)
    else:
        # Keep the public output names consistent when callers explicitly opt
        # out of normalization but pass the loader's raw GF column names.
        factor_data = factor_data.rename(columns=FACTOR_TO_FRAC)

    if dominant:
        factor_data = add_dominant_source(factor_data)

    attach_cols = list(GF_FRACTION_COLUMNS if cols is None else cols)
    if dominant:
        for col in ('dominant_source', 'dominant_fraction'):
            if col not in attach_cols:
                attach_cols.append(col)

    missing = [col for col in attach_cols if col not in factor_data.columns]
    if missing:
        raise KeyError(
            f"factor columns not found: {missing}. "
            f"Available: {sorted(factor_data.columns)}"
        )

    factor_dates = pd.DatetimeIndex(pd.to_datetime(factor_data['date'])).normalize()
    if factor_dates.tz is not None:
        factor_dates = factor_dates.tz_localize(None)

    lookup = factor_data[attach_cols].copy()
    lookup.insert(0, '_factor_date', factor_dates)
    lookup = (
        lookup.dropna(subset=['_factor_date'])
        .drop_duplicates(subset='_factor_date')
        .sort_values('_factor_date')
    )

    row_positions = pd.RangeIndex(len(out))
    matched_values = lookup[attach_cols].iloc[0:0].reindex(row_positions)
    valid_targets = pd.DataFrame({
        '_target_date': target_dates,
        '_row_position': np.arange(len(out)),
    }).dropna(subset=['_target_date'])

    if not valid_targets.empty and not lookup.empty:
        matches = pd.merge_asof(
            valid_targets.sort_values('_target_date'),
            lookup,
            left_on='_target_date',
            right_on='_factor_date',
            direction='nearest',
            tolerance=pd.Timedelta(days=tolerance_days),
        )
        matched_values = (
            matches.set_index('_row_position')[attach_cols]
            .reindex(row_positions)
        )

    for col in attach_cols:
        out[col] = matched_values[col].to_numpy()

    return out
