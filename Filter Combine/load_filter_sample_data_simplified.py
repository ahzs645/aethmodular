"""
Simplified load_filter_sample_data function using your cleaned data approach.

Based on your dual_dataset_ftir_csv_pipeline.ipynb pattern:
1. Use your already processed/cleaned data
2. Load HIPS data from database (when available)
3. Match them efficiently using temporal alignment
4. No need to rerun processing - just match what you have!
"""

import pandas as pd
import sqlite3
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple

def load_filter_sample_data(
    cleaned_pkl_path: str,
    db_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    site_code: str = 'ETAD',
    temporal_window_hours: int = 2,
    use_9am_alignment: bool = True,
    output_format: str = 'both'  # 'daily', 'minutely', 'both'
) -> Dict:
    """
    Load filter sample data using your cleaned data approach.
    
    This function follows your dual_dataset_ftir_csv_pipeline.ipynb pattern:
    - Uses your already processed/cleaned PKL data
    - Loads HIPS/FTIR from database or CSV
    - Matches them efficiently without reprocessing
    
    Parameters:
    -----------
    cleaned_pkl_path : str
        Path to your processed PKL file (e.g., 'pkl_data_cleaned_ethiopia.pkl')
    db_path : str, optional
        Path to SQLite database with HIPS data
    csv_path : str, optional
        Path to FTIR CSV file (alternative to database)
    site_code : str, default 'ETAD'
        Site code for filtering
    temporal_window_hours : int, default 2
        Time window for temporal matching (±hours)
    use_9am_alignment : bool, default True
        Whether to use 9am-to-9am daily alignment
    output_format : str, default 'both'
        Output format: 'daily', 'minutely', or 'both'
    
    Returns:
    --------
    dict: Dictionary containing:
        - 'daily_matched': Daily 9am-to-9am matched data (if requested)
        - 'minutely_matched': High-resolution matched data (if requested) 
        - 'hips_data': HIPS reference data
        - 'metadata': Processing information
    """
    
    print("🔄 Simplified Load Filter Sample Data (Using Cleaned Data)")
    print("=" * 70)
    print("✅ Following your dual_dataset_ftir_csv_pipeline.ipynb approach")
    print("✅ No reprocessing needed - using your cleaned data!")
    
    result = {
        'daily_matched': None,
        'minutely_matched': None,
        'hips_data': None,
        'metadata': {
            'source_files': {
                'cleaned_pkl': cleaned_pkl_path,
                'database': db_path,
                'csv': csv_path
            },
            'site_code': site_code,
            'temporal_window_hours': temporal_window_hours,
            'processing_notes': []
        }
    }
    
    # === Step 1: Load your cleaned data ===
    print(f"\n📊 Step 1: Loading your cleaned aethalometer data...")
    
    if not os.path.exists(cleaned_pkl_path):
        print(f"❌ Cleaned PKL file not found: {cleaned_pkl_path}")
        return result
    
    try:
        cleaned_data = pd.read_pickle(cleaned_pkl_path)
        print(f"✅ Loaded cleaned data: {cleaned_data.shape}")
        
        # Ensure datetime column
        if 'datetime_local' in cleaned_data.columns:
            cleaned_data['datetime_local'] = pd.to_datetime(cleaned_data['datetime_local'])
        elif hasattr(cleaned_data.index, 'to_pydatetime'):
            cleaned_data['datetime_local'] = cleaned_data.index
        else:
            print("❌ No suitable datetime column found in cleaned data")
            return result
            
        print(f"📅 Cleaned data range: {cleaned_data['datetime_local'].min()} to {cleaned_data['datetime_local'].max()}")
        
        # Check for Ethiopia corrections (like in your notebook)
        bc_corrected = [col for col in cleaned_data.columns if 'BCc' in col and 'corrected' in col]
        bc_original = [col for col in cleaned_data.columns if 'BCc' in col and 'corrected' not in col and 'manual' not in col and 'optimized' not in col]
        
        print(f"🔧 Ethiopia-corrected BC columns: {len(bc_corrected)} found")
        print(f"📊 Original BC columns: {len(bc_original)} found")
        
        result['metadata']['processing_notes'].append(f'Loaded cleaned data with {len(bc_corrected)} corrected BC columns')
        
    except Exception as e:
        print(f"❌ Error loading cleaned data: {e}")
        return result
    
    # === Step 2: Load HIPS/FTIR reference data ===
    print(f"\n🔬 Step 2: Loading HIPS/FTIR reference data...")
    
    hips_data = None
    
    # Try database first
    if db_path and os.path.exists(db_path):
        try:
            print(f"📊 Loading from database: {os.path.basename(db_path)}")
            hips_data = load_hips_from_database(db_path, site_code)
            if hips_data is not None:
                print(f"✅ Database: {len(hips_data)} HIPS samples loaded")
                result['metadata']['processing_notes'].append('HIPS data loaded from database')
        except Exception as e:
            print(f"⚠️ Database loading failed: {e}")
    
    # Try CSV if database failed or not available
    if hips_data is None and csv_path and os.path.exists(csv_path):
        try:
            print(f"📄 Loading from CSV: {os.path.basename(csv_path)}")
            hips_data = load_hips_from_csv(csv_path, site_code)
            if hips_data is not None:
                print(f"✅ CSV: {len(hips_data)} FTIR samples loaded")
                result['metadata']['processing_notes'].append('FTIR data loaded from CSV')
        except Exception as e:
            print(f"⚠️ CSV loading failed: {e}")
    
    if hips_data is None:
        print(f"❌ No HIPS/FTIR data could be loaded")
        print(f"💡 Provide either db_path or csv_path to load reference data")
        return result
    
    result['hips_data'] = hips_data
    
    # === Step 3: Early filtering (like your optimized pipeline) ===
    print(f"\n⚡ Step 3: Early filtering to HIPS periods (optimized approach)...")
    
    # Get HIPS date range for filtering
    hips_start = hips_data['sample_date'].min()
    hips_end = hips_data['sample_date'].max()
    
    print(f"🔬 HIPS period: {hips_start.date()} to {hips_end.date()}")
    
    # Filter cleaned data to HIPS periods only
    original_size = len(cleaned_data)
    mask = (cleaned_data['datetime_local'] >= hips_start) & (cleaned_data['datetime_local'] <= hips_end)
    filtered_data = cleaned_data.loc[mask].copy()
    
    efficiency = (1 - len(filtered_data) / original_size) * 100
    print(f"⚡ Filtered: {original_size:,} -> {len(filtered_data):,} rows")
    print(f"🚀 Processing efficiency: {efficiency:.1f}% data reduction")
    
    result['metadata']['processing_notes'].append(f'Early filtering: {efficiency:.1f}% data reduction')
    
    # === Step 4: Create matched datasets ===
    print(f"\n🔗 Step 4: Creating matched datasets...")
    
    if output_format in ['daily', 'both']:
        print(f"📊 Creating daily matched dataset...")
        daily_matched = create_daily_matched_dataset(
            filtered_data, hips_data, 
            temporal_window_hours, use_9am_alignment
        )
        
        if daily_matched is not None:
            result['daily_matched'] = daily_matched
            print(f"✅ Daily matched: {len(daily_matched)} samples")
        else:
            print(f"❌ Daily matching failed")
    
    if output_format in ['minutely', 'both']:
        print(f"📈 Creating minutely matched dataset...")
        minutely_matched = create_minutely_matched_dataset(
            filtered_data, hips_data, temporal_window_hours
        )
        
        if minutely_matched is not None:
            result['minutely_matched'] = minutely_matched
            print(f"✅ Minutely matched: {len(minutely_matched)} rows across {minutely_matched['hips_period'].nunique()} HIPS periods")
        else:
            print(f"❌ Minutely matching failed")
    
    # === Step 5: Summary ===
    print(f"\n📋 Final Summary:")
    print(f"🏢 Site: {site_code}")
    print(f"📁 Source: {os.path.basename(cleaned_pkl_path)}")
    
    if result['daily_matched'] is not None:
        print(f"📊 Daily matched samples: {len(result['daily_matched'])}")
        
    if result['minutely_matched'] is not None:
        periods = result['minutely_matched']['hips_period'].nunique()
        print(f"📈 Minutely data: {len(result['minutely_matched']):,} rows across {periods} periods")
    
    print(f"🔬 HIPS reference: {len(result['hips_data'])} samples")
    print(f"⏱️ Temporal window: ±{temporal_window_hours} hours")
    
    if result['metadata']['processing_notes']:
        print(f"📝 Notes: {'; '.join(result['metadata']['processing_notes'])}")
    
    print(f"\n🎉 Matching complete! Ready for analysis.")
    
    return result


def load_hips_from_database(db_path: str, site_code: str) -> Optional[pd.DataFrame]:
    """Load HIPS data from SQLite database (original functionality)"""
    
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT f.filter_id, 
           f.sample_date, 
           m.ec_ftir,
           m.oc_ftir,
           m.fabs AS hips_fabs,
           f.site_code
    FROM filters f
    JOIN ftir_sample_measurements m USING(filter_id)
    WHERE f.site_code = ?
    ORDER BY f.sample_date;
    """
    
    data = pd.read_sql_query(query, conn, params=(site_code,))
    conn.close()
    
    # Clean and standardize
    data['sample_date'] = pd.to_datetime(data['sample_date'])
    data = data.dropna(subset=['sample_date'])
    
    if len(data) == 0:
        return None
        
    return data


def load_hips_from_csv(csv_path: str, site_code: str) -> Optional[pd.DataFrame]:
    """Load FTIR data from CSV (like your dual_dataset pipeline)"""
    
    df = pd.read_csv(csv_path)
    
    # Filter by site
    site_data = df[df['Site'] == site_code].copy()
    
    if len(site_data) == 0:
        return None
    
    # Convert date and pivot
    site_data['SampleDate'] = pd.to_datetime(site_data['SampleDate'])
    
    pivot_data = site_data.pivot_table(
        index='SampleDate',
        columns='Parameter',
        values='Concentration_ug_m3',
        aggfunc='mean'
    ).reset_index()
    
    # Rename to match database format
    pivot_data.rename(columns={
        'SampleDate': 'sample_date',
        'EC_ftir': 'ec_ftir',
        'OC_ftir': 'oc_ftir'
    }, inplace=True)
    
    pivot_data['site_code'] = site_code
    
    return pivot_data


def create_daily_matched_dataset(
    aeth_data: pd.DataFrame, 
    hips_data: pd.DataFrame,
    window_hours: int,
    use_9am_alignment: bool
) -> Optional[pd.DataFrame]:
    """Create daily matched dataset with 9am-to-9am alignment"""
    
    if use_9am_alignment:
        # Apply 9am-to-9am resampling (like your notebook)
        daily_aeth = resample_9am_to_9am(aeth_data)
    else:
        # Simple daily average
        daily_aeth = aeth_data.set_index('datetime_local').resample('D').mean()
    
    if len(daily_aeth) == 0:
        return None
    
    # Prepare HIPS data for merging
    hips_for_merge = hips_data.copy()
    
    if use_9am_alignment:
        # Set HIPS timestamps to 9am (no timezone)
        hips_for_merge['merge_time'] = pd.to_datetime(hips_for_merge['sample_date'].dt.date) + pd.Timedelta(hours=9)
    else:
        hips_for_merge['merge_time'] = pd.to_datetime(hips_for_merge['sample_date'].dt.date)
    
    # Ensure timezone compatibility
    hips_for_merge['merge_time'] = pd.to_datetime(hips_for_merge['merge_time']).dt.tz_localize(None)
    
    # Temporal merge
    merged = pd.merge_asof(
        hips_for_merge.sort_values('merge_time'),
        daily_aeth.sort_index().reset_index(),
        left_on='merge_time',
        right_on='datetime_local',
        tolerance=pd.Timedelta(hours=window_hours),
        direction='nearest'
    )
    
    # Remove unmatched rows
    merged = merged.dropna(subset=['datetime_local'])
    
    if len(merged) == 0:
        return None
    
    # Set proper index
    merged.set_index('datetime_local', inplace=True)
    
    return merged


def create_minutely_matched_dataset(
    aeth_data: pd.DataFrame,
    hips_data: pd.DataFrame, 
    window_hours: int
) -> Optional[pd.DataFrame]:
    """Create minutely matched dataset (high-resolution within HIPS periods)"""
    
    # Create HIPS periods (like your optimized pipeline)
    hips_periods = []
    
    for idx, hips_row in hips_data.iterrows():
        sample_date = hips_row['sample_date']
        
        # Define period around HIPS sample (±window)
        period_start = sample_date - pd.Timedelta(hours=window_hours)
        period_end = sample_date + pd.Timedelta(hours=window_hours)
        
        # Find aethalometer data in this period
        mask = (aeth_data['datetime_local'] >= period_start) & (aeth_data['datetime_local'] <= period_end)
        period_data = aeth_data.loc[mask].copy()
        
        if len(period_data) > 0:
            # Add HIPS reference info
            period_data['hips_period'] = f"HIPS_{idx:03d}_{sample_date.strftime('%Y%m%d')}"
            period_data['hips_sample_date'] = sample_date
            
            # Add HIPS measurements to each row
            for col in ['ec_ftir', 'oc_ftir', 'hips_fabs']:
                if col in hips_row:
                    period_data[col] = hips_row[col]
            
            hips_periods.append(period_data)
    
    if not hips_periods:
        return None
    
    # Combine all periods
    combined = pd.concat(hips_periods, ignore_index=True)
    
    return combined


def resample_9am_to_9am(df: pd.DataFrame, min_hours: int = 4) -> pd.DataFrame:
    """9am-to-9am resampling (from your notebook)"""
    
    df_work = df.copy()
    df_work = df_work.set_index('datetime_local')
    
    # Localize timezone if needed
    if df_work.index.tz is None:
        df_work.index = df_work.index.tz_localize('Africa/Addis_Ababa')
    
    # Shift back by 9 hours
    df_shifted = df_work.copy()
    df_shifted.index = df_shifted.index - pd.Timedelta(hours=9)
    
    # Resample to daily
    numeric_cols = df_shifted.select_dtypes(include=[np.number]).columns
    daily_means = df_shifted[numeric_cols].resample('D').mean()
    daily_counts = df_shifted[numeric_cols].resample('D').count()
    
    # Filter insufficient data
    for col in numeric_cols:
        insufficient_data = daily_counts[col] < min_hours
        daily_means.loc[insufficient_data, col] = np.nan
    
    # Shift forward by 9 hours and remove timezone for merge compatibility
    daily_means.index = daily_means.index + pd.Timedelta(hours=9)
    daily_means.index = daily_means.index.tz_localize(None)  # Remove timezone
    
    return daily_means


# Example usage function
def load_etad_with_cleaned_data():
    """
    Example of how to use this function with your specific files
    """
    
    print("🎯 Example: Load ETAD data using cleaned PKL + HIPS matching")
    print("=" * 70)
    
    # Your file paths
    cleaned_pkl = '/Users/ahzs645/Github/aethmodular-clean/notebooks/pkl_data_cleaned_ethiopia.pkl'
    hips_db = 'path/to/your/spartan_ftir_hips.db'  # Update this
    ftir_csv = '/Users/ahzs645/Github/aethmodular-clean/Four_Sites_FTIR_data.v2.csv'
    
    # Load using cleaned data approach
    result = load_filter_sample_data(
        cleaned_pkl_path=cleaned_pkl,
        db_path=hips_db if os.path.exists(hips_db) else None,
        csv_path=ftir_csv,
        site_code='ETAD',
        temporal_window_hours=2,
        use_9am_alignment=True,
        output_format='both'  # Get both daily and minutely data
    )
    
    if result['daily_matched'] is not None:
        print(f"\n✅ SUCCESS! Daily matched data ready for analysis")
        
        # Show what's available (like your notebook workflow)
        daily_data = result['daily_matched']
        
        # Check for Ethiopia corrections
        bc_corrected = [col for col in daily_data.columns if 'BCc' in col and 'corrected' in col]
        ftir_cols = [col for col in daily_data.columns if any(x in col.lower() for x in ['ftir', 'ec_', 'oc_'])]
        
        print(f"📊 Available for analysis:")
        print(f"   ✅ Ethiopia-corrected BC: {bc_corrected[:3]}")
        print(f"   🧪 FTIR columns: {ftir_cols}")
        print(f"   📈 Ready for BC vs FTIR correlation analysis!")
        
        return result
    else:
        print(f"\n❌ No matched data created - check file paths")
        return None


if __name__ == "__main__":
    # Run example
    result = load_etad_with_cleaned_data()
    
    if result:
        print(f"\n🎉 Function working perfectly!")
        print(f"📋 Use result['daily_matched'] for daily analysis")
        print(f"📋 Use result['minutely_matched'] for high-resolution analysis")
        print(f"📋 Use result['hips_data'] for reference measurements")