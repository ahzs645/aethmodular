"""
Create 9am-to-9am resampled aethalometer datasets matched to filter availability

This script processes high-resolution aethalometer data from Beijing, Delhi, JPL, and ETAD,
resampling to daily 9am-9am averages that align with filter sampling periods.
"""

import pandas as pd
import pickle
from pathlib import Path
from datetime import timedelta
import os

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AETHMODULAR_DATA_ROOT", REPO_ROOT / "research" / "ftir_hips_chem"))

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
        'aethalometer_path': str(DATA_ROOT / "df_Jacros_9am_resampled.pkl"),
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

def resample_to_9am_daily(df, timezone, resample_hour=9):
    """
    Resample high-resolution data to daily 9am-9am averages

    Also tracks data completeness for each day to ensure quality control.

    Args:
        df: DataFrame with datetime_local as index or column
        timezone: Timezone string (e.g., 'Asia/Shanghai')
        resample_hour: Hour to use as daily boundary (default: 9 for 9 AM)

    Returns:
        DataFrame with daily 9am-9am averages plus data completeness metrics
    """
    print(f"  Resampling to daily {resample_hour}am-{resample_hour}am averages...")

    # Ensure datetime_local is the index
    if 'datetime_local' in df.columns:
        df = df.set_index('datetime_local')

    # Ensure index is datetime with timezone
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)

    # Shift by (24 - resample_hour) hours so that resampling starts at resample_hour
    # For 9 AM: shift by 15 hours, so 9am today becomes midnight, and midnight tomorrow becomes 9am tomorrow
    shift_hours = 24 - resample_hour
    df_shifted = df.copy()
    df_shifted.index = df_shifted.index - timedelta(hours=shift_hours)

    # Now resample to daily (midnight to midnight in shifted time = 9am to 9am in real time)
    # Use 'D' for calendar day
    numeric_cols = df_shifted.select_dtypes(include=['number']).columns

    # Calculate mean
    df_resampled = df_shifted[numeric_cols].resample('D').mean()

    # Also calculate data completeness for key BC columns
    bc_cols = [col for col in numeric_cols if 'BCc' in col and 'smoothed' not in col.lower() and 'ng/m^3' not in col]

    if len(bc_cols) > 0:
        # Count non-null values per day for BC
        bc_counts = df_shifted[bc_cols].resample('D').count()
        # Expected records per day (1440 minutes)
        total_counts = df_shifted[bc_cols].resample('D').size()

        # Calculate completeness percentage for first BC column as representative
        if len(bc_cols) > 0:
            df_resampled['data_completeness_pct'] = (bc_counts.iloc[:, 0] / 1440 * 100).fillna(0)
            df_resampled['minutes_with_data'] = bc_counts.iloc[:, 0].fillna(0)
    else:
        df_resampled['data_completeness_pct'] = 0
        df_resampled['minutes_with_data'] = 0

    # Shift index back to represent the END of the 9am-9am period
    df_resampled.index = df_resampled.index + timedelta(hours=shift_hours)

    # Create a date column for the 9am day
    df_resampled['day_9am'] = df_resampled.index.date

    # Reset index to make datetime_local a column
    df_resampled = df_resampled.reset_index()
    df_resampled = df_resampled.rename(columns={'index': 'datetime_local'})

    # Report completeness stats
    avg_completeness = df_resampled['data_completeness_pct'].mean()
    print(f"    Resampled to {len(df_resampled)} daily records")
    print(f"    Average data completeness: {avg_completeness:.1f}% of 1440 min/day")

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
    essential = ['datetime_local', 'day_9am', 'data_completeness_pct', 'minutes_with_data']

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

def main():
    """Main processing function"""
    print("\n" + "="*80)
    print("AETHALOMETER DATA RESAMPLING FOR FILTER MATCHING")
    print("="*80)
    print("\nThis script will:")
    print("1. Load filter dates for each site")
    print("2. Resample high-res aethalometer data to daily 9am-9am averages")
    print("3. Keep only dates where filters are available")
    print("4. Save processed data for each site")

    results = {}

    for site_code, config in SITES.items():
        try:
            df = process_site(site_code, config, FILTER_DATA_PATH, OUTPUT_DIR)
            results[site_code] = df
        except Exception as e:
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

if __name__ == "__main__":
    main()
