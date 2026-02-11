import pandas as pd
import pickle
from pathlib import Path

# Use relative paths
filter_path = "FTIR_HIPS_Chem/Filter Data/unified_filter_dataset.pkl"
with open(filter_path, 'rb') as f:
    filters = pickle.load(f)

filters['SampleDate'] = pd.to_datetime(filters['SampleDate'])

# Site configurations
sites = {
    'ETAD': {'name': 'Addis_Ababa', 'file': 'df_Addis_Ababa_9am_resampled.pkl'}
}

base_dir = Path("FTIR_HIPS_Chem/processed_sites")

all_stats = []

for site_code, config in sites.items():
    # Load aethalometer data
    aeth_path = base_dir / config['file']
    with open(aeth_path, 'rb') as f:
        df_aeth = pickle.load(f)

    if 'day_9am' in df_aeth.columns:
        df_aeth['day_9am'] = pd.to_datetime(df_aeth['day_9am'])
    elif isinstance(df_aeth.index, pd.DatetimeIndex):
        df_aeth['day_9am'] = df_aeth.index.normalize()

    # Get filter data for this site
    site_filters = filters[filters['Site'] == site_code].copy()

    # Count unique filter dates (all parameters)
    total_filter_dates = site_filters['SampleDate'].nunique()
    date_range_filters = (site_filters['SampleDate'].min(), site_filters['SampleDate'].max())

    # Count filter EC measurements specifically
    ec_filters = site_filters[site_filters['Parameter'] == 'ChemSpec_EC_PM2.5'].copy()
    ec_filters_valid = ec_filters[ec_filters['Concentration'] >= 0.5].copy()
    unique_ec_dates = ec_filters_valid['SampleDate'].nunique()

    # Count aethalometer dates
    total_aeth_dates = len(df_aeth)
    date_range_aeth = (df_aeth['day_9am'].min(), df_aeth['day_9am'].max())

    # Find BC column
    bc_cols = [col for col in df_aeth.columns if 'BCc' in col]
    bc_col = bc_cols[0] if bc_cols else None

    if bc_col:
        total_aeth_with_bc = df_aeth[bc_col].notna().sum()
    else:
        total_aeth_with_bc = 0

    print(f"ETAD (Addis Ababa) Stats:")
    print(f"Filter Data:")
    print(f"  Total unique filter dates: {total_filter_dates}")
    print(f"  Filter date range: {date_range_filters[0]} to {date_range_filters[1]}")
    print(f"  Valid EC filter dates (>= 0.5): {unique_ec_dates}")
    
    print(f"Aethalometer Data (daily):")
    print(f"  Total days: {total_aeth_dates}")
    print(f"  Days with BC: {total_aeth_with_bc}")
    print(f"  Aethalometer date range: {date_range_aeth[0]} to {date_range_aeth[1]}")
