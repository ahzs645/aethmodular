"""
Aethalometer-FTIR/HIPS Data Merger Module

This module provides functions to merge aethalometer data (PKL or CSV) with FTIR/HIPS 
filter measurements using 9am-9am period alignment and quality assessment.

Usage:
    from merger_functions import merge_aethalometer_filter_pipeline
    
    merged_results = merge_aethalometer_filter_pipeline(
        aethalometer_files=['data.pkl', 'data.csv'],
        ftir_db_path='database.db',
        wavelength='Red',
        site_code='ETAD'
    )
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import warnings

# Import your existing loaders (adjust imports based on your project structure)
try:
    from data.loaders.aethalometer import load_aethalometer_data
    from data.loaders.database import FTIRHIPSLoader
except ImportError:
    print("Warning: Could not import loaders. Ensure correct path to src modules.")


def merge_aethalometer_filter_pipeline(
    aethalometer_files: Union[str, List[str]], 
    ftir_db_path: str,
    wavelength: str = "Red",
    quality_threshold: int = 10,
    site_code: str = 'ETAD',
    output_format: str = 'jpl'
) -> Dict[str, pd.DataFrame]:
    """
    Complete pipeline to merge aethalometer data with FTIR/HIPS filter measurements.
    
    Parameters:
    -----------
    aethalometer_files : str or list
        Path(s) to aethalometer data files (.pkl or .csv)
    ftir_db_path : str
        Path to SQLite database with FTIR/HIPS data
    wavelength : str, default 'Red'
        Wavelength to analyze ('Red', 'Blue', 'Green', 'UV', 'IR')
    quality_threshold : int, default 10
        Maximum missing minutes per 24h period for "excellent" quality
    site_code : str, default 'ETAD'
        Site code for database filtering
    output_format : str, default 'jpl'
        Output format for aethalometer data ('jpl' or 'standard')
        
    Returns:
    --------
    dict
        Dictionary with keys:
        - 'merged_datasets': Dict of merged DataFrames by file
        - 'ftir_data': Original FTIR/HIPS DataFrame
        - 'aethalometer_summaries': Summary info for each file
        - 'merge_statistics': Summary statistics
    """
    
    print("🚀 Starting Aethalometer-FTIR/HIPS Merger Pipeline")
    print("=" * 60)
    
    # Ensure aethalometer_files is a list
    if isinstance(aethalometer_files, str):
        aethalometer_files = [aethalometer_files]
    
    # 1. Load FTIR/HIPS data
    print(f"📊 Loading FTIR/HIPS data from database...")
    ftir_data = load_ftir_hips_data(ftir_db_path, site_code)
    
    if ftir_data is None or len(ftir_data) == 0:
        raise ValueError("Failed to load FTIR/HIPS data or no data available")
    
    # 2. Load and process aethalometer files
    print(f"\n📁 Loading {len(aethalometer_files)} aethalometer file(s)...")
    aethalometer_datasets = {}
    aethalometer_summaries = {}
    
    for file_path in aethalometer_files:
        if not Path(file_path).exists():
            print(f"⚠️ File not found: {file_path}")
            continue
            
        dataset_name = f"{Path(file_path).stem}_{Path(file_path).suffix[1:]}"
        
        try:
            df, summary = load_and_process_aethalometer(file_path, output_format)
            if df is not None:
                aethalometer_datasets[dataset_name] = df
                aethalometer_summaries[dataset_name] = summary
                print(f"✅ Loaded {dataset_name}: {len(df)} rows")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    if not aethalometer_datasets:
        raise ValueError("No aethalometer datasets loaded successfully")
    
    # 3. Perform merging for each dataset
    print(f"\n🔗 Performing time-matched merging for wavelength: {wavelength}")
    merged_datasets = {}
    merge_stats = {}
    
    for dataset_name, aeth_df in aethalometer_datasets.items():
        print(f"\n📊 Processing {dataset_name}...")
        
        try:
            merged_df, stats = merge_aethalometer_filter_data(
                aethalometer_df=aeth_df,
                filter_df=ftir_data,
                wavelength=wavelength,
                quality_threshold=quality_threshold,
                dataset_name=dataset_name
            )
            
            if len(merged_df) > 0:
                merged_datasets[dataset_name] = merged_df
                merge_stats[dataset_name] = stats
                print(f"✅ {dataset_name}: {len(merged_df)} merged periods")
            else:
                print(f"⚠️ {dataset_name}: No overlapping periods found")
                
        except Exception as e:
            print(f"❌ Error merging {dataset_name}: {e}")
    
    # 4. Create summary statistics
    pipeline_stats = create_pipeline_statistics(
        merged_datasets, aethalometer_summaries, ftir_data, merge_stats
    )
    
    print(f"\n🎯 Pipeline completed: {len(merged_datasets)} datasets merged")
    
    return {
        'merged_datasets': merged_datasets,
        'ftir_data': ftir_data,
        'aethalometer_summaries': aethalometer_summaries,
        'merge_statistics': pipeline_stats
    }


def load_ftir_hips_data(db_path: str, site_code: str = 'ETAD') -> pd.DataFrame:
    """
    Load FTIR and HIPS data from SQLite database.
    
    Parameters:
    -----------
    db_path : str
        Path to SQLite database
    site_code : str
        Site code to filter data
        
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with FTIR and HIPS measurements
    """
    
    try:
        loader = FTIRHIPSLoader(db_path)
        df = loader.load(site_code)
        
        print(f"✅ Loaded {len(df)} filter samples for site {site_code}")
        print(f"📅 Date range: {df['sample_date'].min()} to {df['sample_date'].max()}")
        
        return df
        
    except ImportError:
        # Fallback direct SQLite loading if loader not available
        print("📝 Using fallback SQLite loading...")
        return load_ftir_hips_fallback(db_path, site_code)
        
    except Exception as e:
        print(f"❌ Error loading FTIR/HIPS data: {e}")
        return None


def load_ftir_hips_fallback(db_path: str, site_code: str) -> pd.DataFrame:
    """
    Fallback function to load FTIR/HIPS data directly from SQLite.
    """
    
    try:
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT 
            f.filter_id, f.sample_date, f.site_code,
            m.ec_ftir, m.oc_ftir, m.fabs
        FROM filters f
        JOIN ftir_measurements m ON f.filter_id = m.filter_id
        WHERE f.site_code = ?
        ORDER BY f.sample_date
        """
        
        df = pd.read_sql_query(query, conn, params=[site_code])
        conn.close()
        
        # Convert date column
        df['sample_date'] = pd.to_datetime(df['sample_date'])
        
        return df
        
    except Exception as e:
        print(f"❌ Fallback loading failed: {e}")
        return None


def load_and_process_aethalometer(file_path: str, output_format: str = 'jpl') -> tuple:
    """
    Load and process aethalometer data with timezone handling.
    
    Parameters:
    -----------
    file_path : str
        Path to aethalometer file
    output_format : str
        Output format ('jpl' or 'standard')
        
    Returns:
    --------
    tuple
        (DataFrame, summary_dict) or (None, None) if failed
    """
    
    try:
        # Use existing loader if available
        df = load_aethalometer_data(
            file_path, 
            output_format=output_format,
            set_datetime_index=True
        )
        
        # Handle timezone conversion for CSV files
        if Path(file_path).suffix.lower() == '.csv':
            df = process_csv_timezone(df)
        
        # Generate summary
        summary = {
            'file_name': Path(file_path).name,
            'file_type': Path(file_path).suffix,
            'shape': df.shape,
            'bc_columns': [col for col in df.columns if '.BCc' in col or ('BC' in col.upper() and 'BCc' not in col)],
            'time_range': (df.index.min(), df.index.max()) if hasattr(df.index, 'min') else None,
            'missing_data_pct': (df.isnull().sum().sum() / df.size) * 100
        }
        
        return df, summary
        
    except ImportError:
        # Fallback loading if loaders not available
        print("📝 Using fallback loading...")
        return load_aethalometer_fallback(file_path, output_format)
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None, None


def load_aethalometer_fallback(file_path: str, output_format: str) -> tuple:
    """
    Fallback function to load aethalometer data without custom loaders.
    """
    
    try:
        file_path = Path(file_path)
        
        if file_path.suffix == '.pkl':
            import pickle
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            df = process_csv_timezone(df)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Basic summary
        summary = {
            'file_name': file_path.name,
            'file_type': file_path.suffix,
            'shape': df.shape,
            'bc_columns': [col for col in df.columns if 'BC' in col.upper()],
            'time_range': None,
            'missing_data_pct': (df.isnull().sum().sum() / df.size) * 100
        }
        
        return df, summary
        
    except Exception as e:
        print(f"❌ Fallback loading failed: {e}")
        return None, None


def process_csv_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process timezone conversion for CSV data (UTC to Africa/Addis_Ababa).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with potential timezone issues
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with proper timezone handling
    """
    
    # Check if we have Time (UTC) column that needs conversion
    if 'Time (UTC)' in df.columns:
        try:
            print("🌍 Converting timezone from UTC to Africa/Addis_Ababa...")
            
            # Convert to datetime and set timezone
            df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], utc=True)
            df['Time (Local)'] = df['Time (UTC)'].dt.tz_convert('Africa/Addis_Ababa')
            df.set_index('Time (Local)', inplace=True)
            
            print("✅ Timezone conversion completed")
            
        except Exception as e:
            print(f"⚠️ Timezone conversion failed: {e}")
    
    # Try to set any datetime column as index if not already set
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        datetime_cols = ['datetime_local', 'datetime_utc', 'Date', 'date']
        for col in datetime_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    print(f"✅ Set {col} as datetime index")
                    break
                except Exception as e:
                    print(f"⚠️ Could not set {col} as index: {e}")
    
    return df


def merge_aethalometer_filter_data(
    aethalometer_df: pd.DataFrame, 
    filter_df: pd.DataFrame,
    wavelength: str = "Red", 
    quality_threshold: int = 10, 
    dataset_name: str = "aethalometer"
) -> tuple:
    """
    Merge Aethalometer and filter sample data with 9am-9am period alignment.
    
    Parameters:
    -----------
    aethalometer_df : pd.DataFrame
        Aethalometer data with datetime index and BC columns
    filter_df : pd.DataFrame
        Filter sample data from database
    wavelength : str
        Wavelength to process ('Red', 'Blue', 'Green', 'UV', 'IR')
    quality_threshold : int
        Maximum missing minutes allowed per 24h period for "excellent" quality
    dataset_name : str
        Name of the dataset for identification
        
    Returns:
    --------
    tuple
        (merged_df, merge_statistics)
    """
    
    # 1. Identify excellent quality periods
    excellent_periods = identify_excellent_periods(aethalometer_df, quality_threshold)
    
    # 2. Find overlapping periods and merge
    merged_df = find_overlaps_and_merge(
        aethalometer_df, filter_df, excellent_periods, wavelength, dataset_name
    )
    
    # 3. Calculate merge statistics
    merge_stats = {
        'total_aethalometer_data_points': len(aethalometer_df),
        'total_filter_samples': len(filter_df),
        'excellent_periods_found': len(excellent_periods),
        'overlapping_periods': len(merged_df),
        'merge_success_rate': len(merged_df) / len(filter_df) * 100 if len(filter_df) > 0 else 0,
        'data_coverage_rate': len(merged_df) / len(excellent_periods) * 100 if len(excellent_periods) > 0 else 0
    }
    
    return merged_df, merge_stats


def identify_excellent_periods(aethalometer_df: pd.DataFrame, quality_threshold: int = 10) -> pd.DataFrame:
    """
    Identify excellent quality 24-hour periods (9am-to-9am) based on data completeness.
    
    Parameters:
    -----------
    aethalometer_df : pd.DataFrame
        Aethalometer data with datetime index
    quality_threshold : int
        Maximum missing minutes per 24h period to be considered "excellent"
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: start_time, end_time, missing_minutes
    """
    
    # Create expected timeline (1-minute resolution)
    start_time = aethalometer_df.index.min()
    end_time = aethalometer_df.index.max()
    expected_idx = pd.date_range(start_time, end_time, freq='min')
    
    # Find missing timestamps
    actual_idx = aethalometer_df.index.unique().sort_values()
    missing_idx = expected_idx.difference(actual_idx)
    
    # Map each missing timestamp to its corresponding 9am-to-9am period
    nine_am_periods = missing_idx.map(lambda ts: 
        ts.normalize() + pd.Timedelta(hours=9) if ts.hour >= 9 
        else ts.normalize() + pd.Timedelta(hours=9) - pd.Timedelta(days=1)
    )
    
    # Count missing minutes per 9am-to-9am period
    missing_per_period = pd.Series(1, index=nine_am_periods).groupby(level=0).count()
    
    # Identify excellent periods (≤ quality_threshold missing minutes)
    excellent_periods_idx = missing_per_period[missing_per_period <= quality_threshold].index
    
    # Create DataFrame with excellent periods
    excellent_periods = pd.DataFrame({
        'start_time': excellent_periods_idx,
        'end_time': excellent_periods_idx + pd.Timedelta(days=1),
        'missing_minutes': missing_per_period[excellent_periods_idx]
    })
    
    return excellent_periods


def find_overlaps_and_merge(
    aethalometer_df: pd.DataFrame, 
    filter_df: pd.DataFrame, 
    excellent_periods: pd.DataFrame, 
    wavelength: str, 
    dataset_name: str
) -> pd.DataFrame:
    """
    Find overlapping periods and create merged dataset.
    
    Parameters:
    -----------
    aethalometer_df : pd.DataFrame
        Aethalometer data
    filter_df : pd.DataFrame
        Filter sample data
    excellent_periods : pd.DataFrame
        Excellent quality periods
    wavelength : str
        Wavelength for BC column
    dataset_name : str
        Dataset identifier
        
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    
    # Find the correct BC column
    bc_column = f"{wavelength}.BCc"
    
    if bc_column not in aethalometer_df.columns:
        # Try alternative column naming
        alt_columns = [col for col in aethalometer_df.columns 
                      if wavelength.lower() in col.lower() and 'bc' in col.lower()]
        if alt_columns:
            bc_column = alt_columns[0]
            print(f"📝 Using alternative BC column: {bc_column}")
        else:
            raise ValueError(f"No BC column found for wavelength '{wavelength}' in {dataset_name}")
    
    # Convert filter sample dates to corresponding 9am-to-9am measurement periods
    filter_measurement_periods = pd.DatetimeIndex([
        d.normalize() + pd.Timedelta(hours=9) - pd.Timedelta(days=1)
        for d in filter_df['sample_date']
    ])
    
    # Find overlap between filter measurement periods and excellent periods
    excellent_starts = excellent_periods['start_time']
    overlap_periods = pd.DatetimeIndex(filter_measurement_periods).intersection(excellent_starts)
    
    if len(overlap_periods) == 0:
        print("⚠️ Warning: No overlapping periods found")
        return pd.DataFrame()
    
    # Create merged dataset
    merged_data = []
    
    for period_start in overlap_periods:
        period_end = period_start + pd.Timedelta(days=1)
        
        # Find the corresponding filter sample
        collection_date = period_start + pd.Timedelta(days=1)
        
        # Find matching filter sample
        filter_matches = filter_df[
            filter_df['sample_date'].dt.date == collection_date.date()
        ]
        
        if len(filter_matches) == 0:
            continue
        
        filter_data = filter_matches.iloc[0]  # Take first match if multiple
        
        # Extract Aethalometer data for this period
        aeth_stats = extract_aethalometer_stats(aethalometer_df, period_start, period_end, bc_column)
        
        if aeth_stats is None:
            continue
        
        # Combine filter and Aethalometer data
        row_data = {
            'dataset_source': dataset_name,
            'period_start': period_start,
            'period_end': period_end,
            'collection_date': collection_date,
            'filter_id': filter_data['filter_id'],
            'EC_FTIR': filter_data['ec_ftir'],
            'OC_FTIR': filter_data['oc_ftir'],
            'Fabs': filter_data['fabs'],
            'site': filter_data['site_code'],
            'wavelength': wavelength
        }
        
        # Add Aethalometer statistics with 'aeth_' prefix
        for key, value in aeth_stats.items():
            row_data[f'aeth_{key}'] = value
        
        merged_data.append(row_data)
    
    # Convert to DataFrame
    merged_df = pd.DataFrame(merged_data)
    
    # Add derived variables if we have data
    if len(merged_df) > 0:
        # Mass Absorption Cross-section (MAC)
        merged_df['MAC'] = merged_df['Fabs'] / merged_df['EC_FTIR']
        
        # Add season information (Ethiopian seasons)
        merged_df['month'] = merged_df['collection_date'].dt.month
        merged_df['season'] = merged_df['month'].apply(map_ethiopian_seasons)
        
        # Add date information
        merged_df['date'] = merged_df['collection_date'].dt.date
        
    return merged_df


def extract_aethalometer_stats(
    aethalometer_df: pd.DataFrame, 
    period_start: pd.Timestamp, 
    period_end: pd.Timestamp, 
    bc_column: str
) -> Optional[Dict[str, float]]:
    """
    Extract statistics for Aethalometer data within a specific period.
    
    Parameters:
    -----------
    aethalometer_df : pd.DataFrame
        Aethalometer data
    period_start, period_end : pd.Timestamp
        Start and end of the period
    bc_column : str
        Name of the BC column to analyze
        
    Returns:
    --------
    dict or None
        Dictionary with statistics, or None if no valid data
    """
    
    try:
        # Extract data for the period
        period_data = aethalometer_df.loc[period_start:period_end, bc_column].dropna()
        
        if len(period_data) == 0:
            return None
        
        # Calculate statistics
        stats = {
            'count': len(period_data),
            'mean': period_data.mean(),
            'median': period_data.median(),
            'std': period_data.std(),
            'min': period_data.min(),
            'max': period_data.max(),
            'q25': period_data.quantile(0.25),
            'q75': period_data.quantile(0.75),
            'negative_count': (period_data < 0).sum(),
            'negative_pct': (period_data < 0).mean() * 100,
            'data_coverage_pct': (len(period_data) / 1440) * 100  # 1440 minutes in 24h
        }
        
        return stats
        
    except Exception as e:
        print(f"⚠️ Error extracting stats for period {period_start} to {period_end}: {e}")
        return None


def map_ethiopian_seasons(month: int) -> str:
    """
    Map month number to Ethiopian season name.
    
    Parameters:
    -----------
    month : int
        Month number (1-12)
        
    Returns:
    --------
    str
        Ethiopian season name
    """
    if month in [10, 11, 12, 1, 2]:
        return 'Dry Season'
    elif month in [3, 4, 5]:
        return 'Belg Rainy Season'
    else:  # months 6-9
        return 'Kiremt Rainy Season'


def create_pipeline_statistics(
    merged_datasets: Dict[str, pd.DataFrame],
    aethalometer_summaries: Dict[str, Dict],
    ftir_data: pd.DataFrame,
    merge_stats: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Create comprehensive pipeline statistics.
    
    Parameters:
    -----------
    merged_datasets : dict
        Dictionary of merged DataFrames
    aethalometer_summaries : dict
        Summary information for aethalometer datasets
    ftir_data : pd.DataFrame
        Original FTIR data
    merge_stats : dict
        Merge statistics for each dataset
        
    Returns:
    --------
    dict
        Comprehensive pipeline statistics
    """
    
    total_merged_periods = sum(len(df) for df in merged_datasets.values())
    total_aethalometer_points = sum(summary['shape'][0] for summary in aethalometer_summaries.values())
    
    pipeline_stats = {
        'input_files': len(aethalometer_summaries),
        'total_aethalometer_data_points': total_aethalometer_points,
        'total_filter_samples': len(ftir_data),
        'successful_merges': len(merged_datasets),
        'total_merged_periods': total_merged_periods,
        'overall_success_rate': total_merged_periods / len(ftir_data) * 100 if len(ftir_data) > 0 else 0,
        'datasets_processed': list(merged_datasets.keys()),
        'individual_merge_stats': merge_stats,
        'ftir_date_range': (ftir_data['sample_date'].min(), ftir_data['sample_date'].max())
    }
    
    return pipeline_stats


def export_pipeline_results(
    pipeline_results: Dict[str, Any], 
    output_dir: str = "outputs",
    include_individual: bool = True,
    include_combined: bool = True
) -> None:
    """
    Export pipeline results to files.
    
    Parameters:
    -----------
    pipeline_results : dict
        Results from merge_aethalometer_filter_pipeline
    output_dir : str
        Output directory
    include_individual : bool
        Whether to export individual merged datasets
    include_combined : bool
        Whether to export combined dataset
    """
    
    import os
    from pathlib import Path
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 Exporting results to: {output_dir}")
    
    merged_datasets = pipeline_results['merged_datasets']
    
    if include_individual:
        # Export individual datasets
        for dataset_name, merged_df in merged_datasets.items():
            csv_path = Path(output_dir) / f'merged_{dataset_name}_ftir_aeth.csv'
            merged_df.to_csv(csv_path, index=False)
            print(f"✅ Exported {dataset_name}: {csv_path}")
    
    if include_combined and len(merged_datasets) > 1:
        # Export combined dataset
        combined_df = pd.concat(merged_datasets.values(), ignore_index=True)
        combined_path = Path(output_dir) / 'merged_combined_ftir_aeth.csv'
        combined_df.to_csv(combined_path, index=False)
        print(f"✅ Exported combined dataset: {combined_path}")
    
    # Export statistics
    import json
    stats_path = Path(output_dir) / 'merge_statistics.json'
    
    # Convert datetime objects to strings for JSON serialization
    stats = pipeline_results['merge_statistics'].copy()
    if 'ftir_date_range' in stats:
        stats['ftir_date_range'] = [str(d) for d in stats['ftir_date_range']]
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"✅ Exported statistics: {stats_path}")
    
    print("📤 Export completed!")


# Example usage function
def example_usage():
    """
    Example of how to use the merger pipeline.
    """
    
    # Define your file paths
    aethalometer_files = [
        "/path/to/your/data.pkl",
        "/path/to/your/data.csv"
    ]
    
    ftir_db_path = "/path/to/your/spartan_ftir_hips.db"
    
    # Run the pipeline
    results = merge_aethalometer_filter_pipeline(
        aethalometer_files=aethalometer_files,
        ftir_db_path=ftir_db_path,
        wavelength="Red",
        quality_threshold=10,
        site_code='ETAD'
    )
    
    # Export results
    export_pipeline_results(results, output_dir="outputs")
    
    # Access individual components
    merged_datasets = results['merged_datasets']
    ftir_data = results['ftir_data']
    statistics = results['merge_statistics']
    
    print(f"Successfully merged {len(merged_datasets)} datasets")
    print(f"Total merged periods: {statistics['total_merged_periods']}")
    
    return results


if __name__ == "__main__":
    print("Aethalometer-FTIR/HIPS Merger Module")
    print("Use merge_aethalometer_filter_pipeline() for complete processing")
    print("See example_usage() for implementation details")
