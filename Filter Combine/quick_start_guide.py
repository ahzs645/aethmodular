#!/usr/bin/env python3
"""
QUICK START GUIDE - Filter Database Workflow
End-to-end helper to build the unified filter dataset and explore it.
"""

import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from main import FilterDataIntegrator  # noqa: E402  (import after sys.path tweak)

BASE_DIR = THIS_DIR


def _resolve_path(path_like):
    """Resolve paths relative to this directory."""
    candidate = Path(path_like)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate


def step_1_create_database():
    """Step 1: Create the unified database from source files."""
    print("🔧 STEP 1: Creating Unified Database")
    print("=" * 50)

    integrator = FilterDataIntegrator()

    # Load FTIR data (required)
    ftir_path = _resolve_path("Four_Sites_FTIR_data.v2.csv")
    integrator.load_ftir_data(ftir_path)

    # Load HIPS data if available
    hips_path = _resolve_path("Four_Sites_HIPS_data.csv")
    if hips_path.exists():
        integrator.load_hips_data(hips_path)
    else:
        print(f"⚠️ HIPS data not found: {hips_path}")

    # Load ChemSpec files when present
    chem_filenames = [
        "FilterBased_ChemSpecPM25_CHTS.csv",
        "FilterBased_ChemSpecPM25_INDH.csv",
        "FilterBased_ChemSpecPM25_ETAD.csv",
        "FilterBased_ChemSpecPM25_USPA.csv",
    ]
    chem_files = []
    for filename in chem_filenames:
        candidate = _resolve_path(filename)
        if candidate.exists():
            chem_files.append(candidate)
        else:
            print(f"⚠️ ChemSpec file missing: {candidate}")
    if chem_files:
        integrator.load_chem_spec_data(chem_files)
    else:
        print("⚠️ No ChemSpec files loaded; unified dataset will only include FTIR/HIPS records")

    integrator.create_unified_dataset()
    csv_path, pkl_path = integrator.save_dataset()

    print("✅ Database created successfully!")
    print(f"   CSV file: {csv_path}")
    print(f"   Pickle database: {pkl_path}")

    return integrator, csv_path, pkl_path


def step_2_load_and_query(pkl_path: Path):
    """Step 2: Load database from pickle and perform basic queries."""
    print("\n🔍 STEP 2: Loading Database and Basic Queries")
    print("=" * 50)

    df = pd.read_pickle(pkl_path)
    df = df.copy()
    df['SampleDate'] = pd.to_datetime(df['SampleDate'], errors='coerce')

    sites = sorted(df['Site'].dropna().unique())
    print(f"Available sites: {', '.join(sites)}")

    sources = sorted(df['DataSource'].dropna().unique())
    print(f"Data sources: {', '.join(sources)}")

    print("\nFilters per site:")
    filters_per_site = (
        df.dropna(subset=['FilterId'])
        .groupby('Site')['FilterId']
        .nunique()
        .reindex(sites, fill_value=0)
    )
    for site, count in filters_per_site.items():
        print(f"  {site}: {count} filters")

    return df


def step_3_analyze_specific_filter(df: pd.DataFrame):
    """Step 3: Analyze a specific filter."""
    print("\n🔬 STEP 3: Analyzing a Specific Filter")
    print("=" * 50)

    filters = df['FilterId'].dropna().unique()
    if len(filters) == 0:
        print("No filters available in the dataset.")
        return

    example_filter = filters[0]
    print(f"Analyzing filter: {example_filter}")

    filter_data = df[df['FilterId'] == example_filter]
    print(f"Total measurements for this filter: {len(filter_data)}")
    print(f"Data sources: {', '.join(sorted(filter_data['DataSource'].unique()))}")

    ftir_data = filter_data[filter_data['DataSource'] == 'FTIR']
    if not ftir_data.empty:
        print("\n📊 FTIR Chemical Data:")
        for _, row in ftir_data.iterrows():
            print(
                f"  {row['Parameter']}: {row['Concentration']} {row['Concentration_Units']}"
            )

    hips_data = filter_data[filter_data['DataSource'] == 'HIPS']
    if not hips_data.empty:
        print("\n📊 HIPS Optical Data:")
        for _, row in hips_data.iterrows():
            print(
                f"  {row['Parameter']}: {row['Concentration']} {row['Concentration_Units']}"
            )

    chem_data = filter_data[filter_data['DataSource'] == 'ChemSpec']
    if not chem_data.empty:
        print("\n📊 ChemSpec Data (first 5):")
        for _, row in chem_data.head(5).iterrows():
            print(
                f"  {row['Parameter']}: {row['Concentration']} {row['Concentration_Units']}"
            )
        if len(chem_data) > 5:
            print(f"  ... and {len(chem_data) - 5} more chemical species")


def step_4_compare_across_sites(df: pd.DataFrame, parameter: str = 'EC_ftir'):
    """Step 4: Compare a parameter across all sites."""
    print("\n📊 STEP 4: Comparing Parameters Across Sites")
    print("=" * 50)

    subset = df[(df['Parameter'] == parameter) & df['Concentration'].notna()]
    if subset.empty:
        print(f"No data found for {parameter}.")
        return

    stats = subset.groupby('Site')['Concentration'].agg(['count', 'mean', 'std', 'min', 'max'])
    stats = stats.round(3)
    print(f"{parameter} statistics:")
    print(stats)


def step_5_search_and_filter(df: pd.DataFrame):
    """Step 5: Demonstrate search and filter capabilities."""
    print("\n🔍 STEP 5: Advanced Search and Filtering")
    print("=" * 50)

    chts_ftir = (
        df[(df['Site'] == 'CHTS') & (df['DataSource'] == 'FTIR')]['FilterId']
        .dropna()
        .nunique()
    )
    print(f"CHTS filters with FTIR: {chts_ftir} found")

    sample_dates = pd.to_datetime(df['SampleDate'], errors='coerce')
    filters_2024 = (
        df[sample_dates.dt.year == 2024]['FilterId']
        .dropna()
        .unique()
    )
    print(f"Filters from 2024: {len(filters_2024)} found")

    ftir_params = sorted(df[df['DataSource'] == 'FTIR']['Parameter'].dropna().unique())
    if ftir_params:
        print(f"FTIR parameters: {', '.join(ftir_params)}")

    chem_params = sorted(df[df['DataSource'] == 'ChemSpec']['Parameter'].dropna().unique())
    if chem_params:
        display_count = min(len(chem_params), 10)
        preview = ", ".join(chem_params[:display_count])
        print(f"ChemSpec parameters (first {display_count}): {preview}")
        print(f"Total ChemSpec parameters: {len(chem_params)}")


def complete_workflow():
    """Run the complete workflow from creation to analysis."""
    print("🚀 COMPLETE FILTER DATABASE WORKFLOW")
    print("=" * 70)

    default_csv = _resolve_path('unified_filter_dataset.csv')
    default_pkl = _resolve_path('unified_filter_dataset.pkl')

    if default_csv.exists() and default_pkl.exists():
        print("Database found. Skipping creation step.")
        csv_path, pkl_path = default_csv, default_pkl
    else:
        _, csv_path, pkl_path = step_1_create_database()

    df = step_2_load_and_query(pkl_path)
    step_3_analyze_specific_filter(df)
    step_4_compare_across_sites(df)
    step_5_search_and_filter(df)

    print("\n✅ WORKFLOW COMPLETE!")
    print("=" * 50)
    print("💡 Next steps:")
    print("   1. Change the example filter in step 3 to inspect other IDs")
    print("   2. Update the parameter name in step 4 for different metrics")
    print("   3. Extend step 5 with your custom filtering logic")
    print("   4. Use the saved CSV/PKL in notebooks or other scripts")


if __name__ == "__main__":
    complete_workflow()
