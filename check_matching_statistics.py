"""
Analyze filter vs aethalometer matching statistics for all sites

This script shows:
- How many filter dates exist (total and valid EC only)
- How many aethalometer dates have BC data
- How many successfully match between the two datasets
"""

import pandas as pd
import pickle
from pathlib import Path

# Load filter data
filter_path = "/Users/ahmadjalil/Github/aethmodular/FTIR_HIPS_Chem/Filter Data/unified_filter_dataset.pkl"
with open(filter_path, 'rb') as f:
    filters = pickle.load(f)

filters['SampleDate'] = pd.to_datetime(filters['SampleDate'])

# Site configurations
sites = {
    'CHTS': {'name': 'Beijing', 'file': 'df_Beijing_9am_resampled.pkl'},
    'INDH': {'name': 'Delhi', 'file': 'df_Delhi_9am_resampled.pkl'},
    'USPA': {'name': 'JPL', 'file': 'df_JPL_9am_resampled.pkl'},
    'ETAD': {'name': 'Addis_Ababa', 'file': 'df_Addis_Ababa_9am_resampled.pkl'}
}

base_dir = Path("/Users/ahmadjalil/Github/aethmodular/FTIR_HIPS_Chem/processed_sites")

print("\n" + "="*100)
print("FILTER vs AETHALOMETER MATCHING STATISTICS")
print("="*100)

all_stats = []

for site_code, config in sites.items():
    print(f"\n{config['name']} ({site_code})")
    print("-"*100)

    # Load aethalometer data
    aeth_path = base_dir / config['file']
    with open(aeth_path, 'rb') as f:
        df_aeth = pickle.load(f)

    df_aeth['day_9am'] = pd.to_datetime(df_aeth['day_9am'])

    # Get filter data for this site
    site_filters = filters[filters['Site'] == site_code].copy()

    # Count unique filter dates (all parameters)
    total_filter_dates = site_filters['SampleDate'].nunique()

    # Count filter EC measurements specifically
    ec_filters = site_filters[site_filters['Parameter'] == 'ChemSpec_EC_PM2.5'].copy()

    # Filter out blanks/MDL (< 0.5 µg/m³)
    ec_filters_valid = ec_filters[ec_filters['Concentration'] >= 0.5].copy()

    total_ec_filters = len(ec_filters)
    valid_ec_filters = len(ec_filters_valid)
    unique_ec_dates = ec_filters_valid['SampleDate'].nunique()

    # Count aethalometer dates
    total_aeth_dates = len(df_aeth)

    # Count how many have valid BC data
    if 'IR BCc' in df_aeth.columns:
        bc_col = 'IR BCc'
    elif 'IR BCc smoothed (ng/m^3)' in df_aeth.columns:
        bc_col = 'IR BCc smoothed (ng/m^3)'
    else:
        bc_col = None

    if bc_col:
        aeth_with_bc = df_aeth[df_aeth[bc_col].notna()]
        total_aeth_with_bc = len(aeth_with_bc)
    else:
        aeth_with_bc = df_aeth
        total_aeth_with_bc = 0

    # Match filter dates with aethalometer dates (±1 day)
    matched_pairs = []

    for _, filter_row in ec_filters_valid.iterrows():
        filter_date = filter_row['SampleDate']

        # Find aethalometer data within ±1 day
        date_diff = (df_aeth['day_9am'] - filter_date).abs()
        matches = df_aeth[date_diff <= pd.Timedelta(days=1)]

        if len(matches) > 0:
            # Get closest match
            closest_idx = date_diff.idxmin()
            aeth_row = df_aeth.loc[closest_idx]

            # Check if both have valid data
            if bc_col and pd.notna(aeth_row[bc_col]):
                matched_pairs.append({
                    'filter_date': filter_date,
                    'aeth_date': aeth_row['day_9am'],
                    'filter_ec': filter_row['Concentration'],
                    'aeth_bc': aeth_row[bc_col]
                })

    total_matches = len(matched_pairs)

    # Calculate percentages
    filter_match_pct = (total_matches / unique_ec_dates * 100) if unique_ec_dates > 0 else 0
    aeth_match_pct = (total_matches / total_aeth_with_bc * 100) if total_aeth_with_bc > 0 else 0

    print(f"\nFILTER DATA:")
    print(f"  Total filter dates (all parameters):        {total_filter_dates:4d}")
    print(f"  Total EC filter measurements:                {total_ec_filters:4d}")
    print(f"  Valid EC filters (≥ 0.5 µg/m³):              {valid_ec_filters:4d}")
    print(f"  Unique EC filter dates (valid):              {unique_ec_dates:4d}")
    print(f"  Blanks/MDL removed:                          {total_ec_filters - valid_ec_filters:4d} ({(total_ec_filters - valid_ec_filters)/total_ec_filters*100:.1f}%)")

    print(f"\nAETHALOMETER DATA:")
    print(f"  Total aethalometer dates (filter-matched):  {total_aeth_dates:4d}")
    print(f"  Dates with valid BC data:                   {total_aeth_with_bc:4d} ({total_aeth_with_bc/total_aeth_dates*100:.1f}%)")

    print(f"\nMATCHING RESULTS:")
    print(f"  Successfully matched pairs:                  {total_matches:4d}")
    print(f"  Match rate (of valid EC filters):            {filter_match_pct:.1f}%")
    print(f"  Match rate (of aeth dates with BC):          {aeth_match_pct:.1f}%")

    # Store for summary table
    all_stats.append({
        'Site': config['name'],
        'Code': site_code,
        'EC Filters': unique_ec_dates,
        'Aeth Days': total_aeth_with_bc,
        'Matched': total_matches,
        'Filter Match %': filter_match_pct,
        'Aeth Match %': aeth_match_pct
    })

# Summary table
print("\n\n" + "="*100)
print("SUMMARY TABLE")
print("="*100)

summary_df = pd.DataFrame(all_stats)
print(f"\n{'Site':<15} {'Code':<6} {'EC Filters':<12} {'Aeth Days':<12} {'Matched':<10} {'Filter %':<10} {'Aeth %':<10}")
print("-"*100)

for _, row in summary_df.iterrows():
    print(f"{row['Site']:<15} {row['Code']:<6} {row['EC Filters']:<12} {row['Aeth Days']:<12} {row['Matched']:<10} {row['Filter Match %']:<10.1f} {row['Aeth Match %']:<10.1f}")

print("\n" + "="*100)
print("\nNOTES:")
print("- 'EC Filters': Unique dates with valid ChemSpec EC measurements (≥ 0.5 µg/m³)")
print("- 'Aeth Days': Days with valid BC measurements from aethalometer")
print("- 'Matched': Successfully paired measurements (within ±1 day)")
print("- 'Filter %': Percentage of EC filter dates that found matching aethalometer data")
print("- 'Aeth %': Percentage of aethalometer days that found matching filter data")
print("="*100)
