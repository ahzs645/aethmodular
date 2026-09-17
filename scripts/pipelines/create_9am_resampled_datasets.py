"""
Create 9am-to-9am resampled aethalometer datasets matched to filter availability

This script processes high-resolution aethalometer data from Beijing, Delhi, JPL, and ETAD,
resampling to daily 9am-9am averages that align with filter sampling periods.
"""

import argparse
import os
import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd

import sys
# scripts/ is not an installed package and the CLI runs this file by path, so
# put scripts/ on sys.path to make `common` importable. See scripts/common/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO_ROOT, data_root  # noqa: E402

# Share the slot arithmetic with the research active-interval matcher.
sys.path.insert(0, str(REPO_ROOT / 'research/ftir_hips_chem/scripts'))
from observation_coverage import distinct_slot_means  # noqa: E402

DATA_ROOT = data_root()

# Configuration

FILTER_DATA_PATH = DATA_ROOT / "Filter Data" / "unified_filter_dataset.pkl"
OUTPUT_DIR = DATA_ROOT / "processed_sites"

# Site configurations
SITES = {
    'CHTS': {
        'name': 'Beijing',
        'aethalometer_path': os.environ.get("AETH_BEIJING_PKL", ""),
        'device_id': 'WF0010',
        'timezone': 'Asia/Shanghai',
        'resample_hour': 9  # 9 AM local time
    },
    'INDH': {
        'name': 'Delhi',
        'aethalometer_path': os.environ.get("AETH_DELHI_PKL", ""),
        'device_id': 'MA350-0216',
        'timezone': 'Asia/Kolkata',
        'resample_hour': 9
    },
    'USPA': {
        'name': 'JPL',
        'aethalometer_path': os.environ.get("AETH_JPL_PKL", ""),
        'device_id': 'MA350-0229',
        'timezone': 'America/Los_Angeles',
        'resample_hour': 9
    },
    'ETAD': {
        'name': 'Addis_Ababa',
        'aethalometer_path': str(DATA_ROOT / "processed_sites" / "df_Jacros_9am_resampled.pkl"),
        'device_id': 'MA350-0238',
        'timezone': 'Africa/Addis_Ababa',
        'resample_hour': 9
    }
}

def load_filter_dates(filter_path, site_code):
    """Load filter sample dates for a specific site"""
    print(f"  Loading filter dates for {site_code}...")

    with open(filter_path, 'rb') as f:
        filters = pickle.load(f)

    # Filter for this site
    site_filters = filters[filters['Site'] == site_code].copy()

    # Convert dates to datetime
    site_filters['SampleDate'] = pd.to_datetime(site_filters['SampleDate'])

    # Get unique filter dates
    filter_dates = site_filters['SampleDate'].unique()
    filter_dates = pd.Series(filter_dates).sort_values().reset_index(drop=True)

    print(f"    Found {len(filter_dates)} unique filter dates")
    print(f"    Date range: {filter_dates.min()} to {filter_dates.max()}")

    return filter_dates

def resample_to_9am_daily(df, timezone, resample_hour=9, observed_col=None):
    """
    Resample high-resolution data to daily 9am-9am averages

    Use local calendar intervals [start, end), labelled by their end. Coverage
    counts distinct valid minutes per channel, with the actual interval length
    as denominator (including 23/25-hour days). It measures input availability;
    upstream interpolation cannot be identified unless observed_col is supplied.

    Args:
        df: DataFrame with datetime_local as index or column
        timezone: Timezone string (e.g., 'Asia/Shanghai')
        resample_hour: Hour to use as daily boundary (default: 9 for 9 AM)
        observed_col: Optional boolean column identifying observed input rows.
            False/missing rows contribute to neither averages nor coverage.

    Returns:
        DataFrame with daily 9am-9am averages plus data completeness metrics
    """
    print(f"  Resampling to daily {resample_hour}am-{resample_hour}am averages...")

    if not isinstance(resample_hour, int) or not 0 <= resample_hour <= 23:
        raise ValueError("resample_hour must be an integer from 0 to 23")
    if not df.columns.is_unique:
        raise ValueError("Duplicate input columns must be resolved before resampling")
    df = df.copy()
    if 'datetime_local' in df.columns:
        df = df.set_index('datetime_local')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.empty or df.index.hasnans:
        raise ValueError("Resampling requires nonempty data with valid timestamps")
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)
    else:
        df.index = df.index.tz_convert(timezone)
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps must be resolved before resampling")
    df = df.sort_index()
    if observed_col is not None:
        observed = df[observed_col]
        if not pd.api.types.is_bool_dtype(observed.dtype):
            raise ValueError("observed_col must contain boolean flags")
    numeric = df.select_dtypes(include=['number']).replace([float('inf'), -float('inf')], float('nan'))
    if observed_col is not None:
        numeric = numeric.where(observed.fillna(False), axis=0)

    # Equal weight per available minute; multiple sub-minute records must not
    # inflate coverage or overweight the busier minutes. UTC flooring avoids
    # ambiguity in the repeated local hour at the autumn DST transition.
    minute_data = distinct_slot_means(numeric)
    minute_data.index = minute_data.index.tz_convert(timezone)
    local_days = (minute_data.index.tz_localize(None) - pd.Timedelta(hours=resample_hour)).normalize()
    calendar_days = pd.date_range(local_days.min(), local_days.max(), freq='D')
    starts = (calendar_days + pd.Timedelta(hours=resample_hour)).tz_localize(timezone)
    ends = (calendar_days + pd.Timedelta(days=1, hours=resample_hour)).tz_localize(timezone)
    expected_minutes = (ends - starts).total_seconds() / 60
    df_resampled = minute_data.groupby(local_days).mean().reindex(calendar_days)
    bc_cols = [col for col in numeric if col.endswith(' BCc') or ' BCc smoothed' in col]
    counts = minute_data[bc_cols].groupby(local_days).count().reindex(calendar_days, fill_value=0)
    for col in bc_cols:
        df_resampled[f'{col} valid_minutes'] = counts[col]
        df_resampled[f'{col} coverage_pct'] = counts[col] / expected_minutes * 100
    # Keep compatibility columns, but explicitly tie them to IR, not whichever
    # channel happens to be first in the source file.
    df_resampled['minutes_with_data'] = counts['IR BCc'] if 'IR BCc' in counts else float('nan')
    df_resampled['data_completeness_pct'] = df_resampled['minutes_with_data'] / expected_minutes * 100
    df_resampled['expected_minutes'] = expected_minutes
    df_resampled['interval_start'] = starts
    df_resampled['interval_end'] = ends
    df_resampled['datetime_local'] = ends
    df_resampled['day_9am'] = ends.date
    df_resampled = df_resampled.reset_index(drop=True)
    df_resampled.attrs.update({
        'resampling_version': 2,
        'interval_closed': 'left',
        'timestamp_label': 'interval_end',
        'coverage_channel': 'IR BCc',
        'coverage_basis': 'observed_flag' if observed_col else 'non_null_input_minutes',
        'observed_col': observed_col,
        'upstream_interpolation_verified': observed_col is not None,
    })
    avg_completeness = df_resampled['data_completeness_pct'].mean()
    print(f"    Resampled to {len(df_resampled)} daily records")
    print(f"    Average IR minute availability: {avg_completeness:.1f}%")

    return df_resampled

def filter_by_filter_dates(df_resampled, filter_dates, tolerance_days=1):
    """
    Keep only dates where filters are available (within tolerance)

    Args:
        df_resampled: Resampled daily aethalometer data
        filter_dates: Series of filter sample dates
        tolerance_days: Days tolerance for matching (default: 1)

    Returns:
        Filtered DataFrame
    """
    print(f"  Filtering to dates with filter availability (tolerance: ±{tolerance_days} days)...")

    # Convert filter dates to datetime if not already
    filter_dates = pd.to_datetime(filter_dates)

    # Create a column for the date portion
    df_resampled['date_only'] = pd.to_datetime(df_resampled['day_9am'])

    # For each row, check if there's a filter within tolerance
    matched_rows = []
    for idx, row in df_resampled.iterrows():
        row_date = row['date_only']

        # Check if any filter is within tolerance
        date_diffs = (filter_dates - row_date).abs()
        if date_diffs.min() <= timedelta(days=tolerance_days):
            matched_rows.append(idx)

    df_filtered = df_resampled.loc[matched_rows].copy()
    df_filtered = df_filtered.drop(columns=['date_only'])

    print(f"    Kept {len(df_filtered)} days matching filter dates")

    return df_filtered

def select_key_columns(df, site_code):
    """Select key columns for the final dataset"""

    # Essential columns to keep
    essential = ['datetime_local', 'day_9am', 'interval_start', 'interval_end',
                 'expected_minutes', 'data_completeness_pct', 'minutes_with_data']

    # Device info
    device_cols = ['Serial number', 'device_type', 'Firmware version']

    # Black carbon measurements (all wavelengths, raw and smoothed)
    bc_cols = [col for col in df.columns if any([
        'BCc' in col and 'smoothed' not in col,  # Raw BCc
        'BCc smoothed' in col,  # Smoothed BCc
        'BC1' in col and 'smoothed' in col,  # Smoothed BC1
        'BC2' in col and 'smoothed' in col,  # Smoothed BC2
        'ATN' in col and ('Blue' in col or 'IR' in col or 'UV' in col or 'Red' in col or 'Green' in col),
        'AAE' in col,  # Absorption Angstrom Exponent
        'BB (%)' in col,  # Biomass burning percentage
        'Biomass BCc' in col,  # Source apportionment
        'Fossil fuel BCc' in col,
        'Delta-C' in col
    ])]

    # Environmental sensors
    env_cols = [col for col in df.columns if any([
        'temp' in col.lower() and 'delta' not in col.lower() and 'rolling' not in col.lower(),
        'RH' in col and 'delta' not in col and 'rolling' not in col,
        'pressure' in col.lower(),
        'Flow' in col and 'ratio' not in col.lower(),
        'Accel' in col
    ])]

    # Particulate matter
    pm_cols = [col for col in df.columns if any([
        col.startswith('opc.bins.'),
        col.startswith('opc.pms.'),
        col.startswith('particulate.')
    ])]

    # CO2 if available
    co2_cols = [col for col in df.columns if 'co2' in col.lower()]

    # Quality flags
    quality_cols = [col for col in df.columns if any([
        'high_rough_period' in col,
        'roughness' in col.lower(),
        'Status' in col,
        'test' in col.lower()
    ])]

    # Combine all
    all_cols = essential + device_cols + bc_cols + env_cols + pm_cols + co2_cols + quality_cols

    # Get only columns that exist in df
    selected = [col for col in all_cols if col in df.columns]

    # Remove duplicates while preserving order
    selected = list(dict.fromkeys(selected))

    return df[selected]

def process_site(site_code, config, filter_path, output_dir):
    """Process one site's data"""
    print(f"\n{'='*80}")
    print(f"Processing {config['name']} ({site_code})")
    print(f"{'='*80}")

    # Load filter dates
    filter_dates = load_filter_dates(filter_path, site_code)

    # Load aethalometer data
    print(f"  Loading aethalometer data...")
    with open(config['aethalometer_path'], 'rb') as f:
        df_aeth = pickle.load(f)
    print(f"    Loaded {len(df_aeth)} records")

    # Check if already resampled (ETAD case)
    if 'day_9am' in df_aeth.columns and len(df_aeth) < 2000:
        print(f"  Data appears to be already resampled (daily format)")
        df_resampled = df_aeth.copy()

        # Ensure datetime_local exists
        if 'datetime_local' not in df_resampled.columns:
            if isinstance(df_resampled.index, pd.DatetimeIndex):
                df_resampled['datetime_local'] = df_resampled.index
            else:
                print("    ERROR: Cannot find datetime column")
                return None
    else:
        # Resample to 9am-9am daily averages
        df_resampled = resample_to_9am_daily(df_aeth, config['timezone'], config['resample_hour'])

    # Filter by filter availability
    df_final = filter_by_filter_dates(df_resampled, filter_dates, tolerance_days=1)

    # Select key columns
    print(f"  Selecting key columns...")
    df_final = select_key_columns(df_final, site_code)
    print(f"    Selected {len(df_final.columns)} columns")

    # Add site metadata
    df_final['Site_Code'] = site_code
    df_final['Site_Name'] = config['name']
    df_final['Device_ID'] = config['device_id']

    # Save to pickle
    output_path = Path(output_dir) / f"df_{config['name']}_9am_resampled.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(df_final, f)

    # Print summary
    print(f"\n  Summary:")
    print(f"    Site: {site_code} ({config['name']})")
    print(f"    Device: {config['device_id']}")
    print(f"    Records: {len(df_final)}")
    print(f"    Columns: {len(df_final.columns)}")
    print(f"    Date range: {df_final['day_9am'].min()} to {df_final['day_9am'].max()}")
    print(f"    Memory: {df_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"    Output: {output_path}")

    return df_final


def resolve_site_codes(values):
    """Resolve site codes or names supplied on the command line."""
    if not values:
        return list(SITES)

    aliases = {}
    for code, config in SITES.items():
        aliases[code.casefold()] = code
        aliases[config['name'].casefold()] = code
        aliases[config['name'].replace('_', ' ').casefold()] = code

    resolved = []
    for value in values:
        code = aliases.get(value.casefold())
        if code is None:
            available = ", ".join(
                f"{site_code} ({config['name']})" for site_code, config in SITES.items()
            )
            raise ValueError(f"Unknown site {value!r}. Available sites: {available}")
        if code not in resolved:
            resolved.append(code)
    return resolved


def main(site_values=None):
    """Main processing function"""
    print("\n" + "="*80)
    print("AETHALOMETER DATA RESAMPLING FOR FILTER MATCHING")
    print("="*80)
    print("\nThis script will:")
    print("1. Load filter dates for each site")
    print("2. Resample high-res aethalometer data to daily 9am-9am averages")
    print("3. Keep only dates where filters are available")
    print("4. Save processed data for each site")

    try:
        site_codes = resolve_site_codes(site_values)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    results = {}
    failures = []

    for site_code in site_codes:
        config = SITES[site_code]
        try:
            df = process_site(site_code, config, FILTER_DATA_PATH, OUTPUT_DIR)
            results[site_code] = df
        except Exception as e:
            failures.append(site_code)
            print(f"\n  ERROR processing {site_code}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}\n")

    print(f"{'Site':<15} {'Records':<10} {'Date Range':<40}")
    print("-"*80)
    for site_code, df in results.items():
        if df is not None:
            date_range = f"{df['day_9am'].min()} to {df['day_9am'].max()}"
            print(f"{site_code:<15} {len(df):<10} {date_range:<40}")

    print(f"\nOutput directory: {OUTPUT_DIR}")

    if failures:
        print(f"Failed sites: {', '.join(failures)}")
        return 1
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="Site code or name to process; repeat for multiple sites (default: all).",
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="List configured site codes and exit.",
    )
    cli_args = parser.parse_args()
    if cli_args.list_sites:
        for site_code, site_config in SITES.items():
            print(f"{site_code}\t{site_config['name']}")
        raise SystemExit(0)
    raise SystemExit(main(cli_args.sites))
