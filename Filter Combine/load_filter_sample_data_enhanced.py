"""
Enhanced load_filter_sample_data function that works with both database HIPS data 
and existing PKL files, based on your Untitled-1.ipynb usage patterns.

This function combines:
1. Database HIPS/FTIR data loading (original functionality)
2. Pre-processed PKL file loading (your current approach)
3. Temporal merging capabilities
4. Unit conversions and data standardization
"""

import pandas as pd
import sqlite3
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple

def load_filter_sample_data(
    db_path: Optional[str] = None,
    pkl_files: Optional[Dict[str, str]] = None,
    site_code: str = 'ETAD',
    merge_strategy: str = 'pkl_first',
    temporal_window_hours: int = 2,
    apply_9am_resampling: bool = True,
    timezone: str = 'Africa/Addis_Ababa'
) -> Dict:
    """
    Enhanced function to load ETAD (HIPS) and FTIR data from multiple sources.
    
    Based on the usage pattern in Untitled-1.ipynb, this function provides:
    1. Loading from existing processed PKL files (preferred method)
    2. Loading from SQLite database (fallback/supplementary)
    3. Temporal merging with configurable window
    4. 9am-to-9am daily resampling (optional)
    5. Unit conversions and standardization
    
    Parameters:
    -----------
    db_path : str, optional
        Path to SQLite database (spartan_ftir_hips.db)
    pkl_files : dict, optional
        Dictionary with paths to PKL files:
        {
            'aethalometer': 'pkl_data_cleaned_ethiopia.pkl',  # High-frequency aethalometer data
            'merged': 'aethalometer_ftir_merged_etad_9am.pkl'  # Already merged data
        }
    site_code : str, default 'ETAD'
        Site code for database queries
    merge_strategy : str, default 'pkl_first'
        - 'pkl_first': Use PKL files as primary source
        - 'database_first': Use database as primary source
        - 'pkl_only': Only use PKL files
        - 'database_only': Only use database
        - 'temporal_merge': Merge database HIPS with PKL aethalometer
    temporal_window_hours : int, default 2
        Time window for temporal matching (±hours)
    apply_9am_resampling : bool, default True
        Whether to apply 9am-to-9am daily resampling
    timezone : str, default 'Africa/Addis_Ababa'
        Timezone for data processing
    
    Returns:
    --------
    dict: Dictionary containing:
        - 'data': Main DataFrame with all available data
        - 'hips_data': DataFrame with HIPS measurements
        - 'ftir_data': DataFrame with FTIR measurements  
        - 'aethalometer_data': DataFrame with aethalometer measurements
        - 'metadata': Processing information and statistics
    """
    
    print("🔄 Enhanced Load Filter Sample Data (PKL + Database)")
    print("=" * 60)
    
    result = {
        'data': None,
        'hips_data': None,
        'ftir_data': None,
        'aethalometer_data': None,
        'metadata': {
            'sources_used': [],
            'merge_strategy': merge_strategy,
            'site_code': site_code,
            'total_samples': 0,
            'date_range': None,
            'processing_notes': [],
            'columns_available': {}
        }
    }
    
    # === 1. Load PKL Files ===
    pkl_aethalometer = None
    pkl_merged = None
    
    if pkl_files and merge_strategy != 'database_only':
        print(f"📊 Loading PKL files...")
        
        # Load aethalometer-only PKL
        if 'aethalometer' in pkl_files and os.path.exists(pkl_files['aethalometer']):
            try:
                print(f"   Loading aethalometer PKL: {os.path.basename(pkl_files['aethalometer'])}")
                pkl_aethalometer = pd.read_pickle(pkl_files['aethalometer'])
                
                # Standardize datetime column
                datetime_col = None
                if 'datetime_local' in pkl_aethalometer.columns:
                    datetime_col = 'datetime_local'
                elif hasattr(pkl_aethalometer.index, 'to_pydatetime'):
                    pkl_aethalometer['datetime_local'] = pkl_aethalometer.index
                    datetime_col = 'datetime_local'
                
                if datetime_col:
                    pkl_aethalometer['datetime_local'] = pd.to_datetime(pkl_aethalometer['datetime_local'])
                    
                    # Apply 9am resampling if requested
                    if apply_9am_resampling:
                        pkl_aethalometer = resample_9am_to_9am(
                            pkl_aethalometer, 
                            datetime_col='datetime_local',
                            timezone=timezone
                        )
                        result['metadata']['processing_notes'].append('Applied 9am-to-9am resampling to aethalometer data')
                
                print(f"   ✅ Aethalometer PKL: {pkl_aethalometer.shape}")
                result['metadata']['sources_used'].append('aethalometer_pkl')
                
                # Identify available BC and ATN columns
                bc_cols = [col for col in pkl_aethalometer.columns if 'BC' in col]
                atn_cols = [col for col in pkl_aethalometer.columns if 'ATN' in col]
                result['metadata']['columns_available']['bc_columns'] = bc_cols[:10]  # First 10
                result['metadata']['columns_available']['atn_columns'] = atn_cols[:10]
                
            except Exception as e:
                print(f"   ❌ Error loading aethalometer PKL: {e}")
                pkl_aethalometer = None
        
        # Load merged PKL (already contains FTIR data)
        if 'merged' in pkl_files and os.path.exists(pkl_files['merged']):
            try:
                print(f"   Loading merged PKL: {os.path.basename(pkl_files['merged'])}")
                pkl_merged = pd.read_pickle(pkl_files['merged'])
                
                # This file likely already has proper datetime indexing and FTIR data
                if hasattr(pkl_merged.index, 'to_pydatetime'):
                    pkl_merged['datetime_local'] = pkl_merged.index
                elif 'datetime_local' not in pkl_merged.columns and 'SampleDate' in pkl_merged.columns:
                    pkl_merged['datetime_local'] = pd.to_datetime(pkl_merged['SampleDate'])
                
                print(f"   ✅ Merged PKL: {pkl_merged.shape}")
                result['metadata']['sources_used'].append('merged_pkl')
                
                # Identify FTIR columns
                ftir_cols = [col for col in pkl_merged.columns if any(x in col.lower() for x in ['ftir', 'ec_', 'oc_'])]
                result['metadata']['columns_available']['ftir_columns'] = ftir_cols
                
            except Exception as e:
                print(f"   ❌ Error loading merged PKL: {e}")
                pkl_merged = None
    
    # === 2. Load Database ===
    db_data = None
    
    if db_path and os.path.exists(db_path) and merge_strategy != 'pkl_only':
        try:
            print(f"📊 Loading database: {os.path.basename(db_path)}")
            conn = sqlite3.connect(db_path)
            
            # Original HIPS/FTIR query
            query = f"""
            SELECT f.filter_id, 
                   f.sample_date AS SampleDate, 
                   m.ec_ftir AS EC_FTIR,
                   m.oc_ftir AS OC_FTIR,
                   m.fabs AS Fabs,
                   f.site_code AS Site
            FROM filters f
            JOIN ftir_sample_measurements m USING(filter_id)
            WHERE f.site_code = ?
            ORDER BY f.sample_date;
            """
            
            db_data = pd.read_sql_query(query, conn, params=(site_code,))
            conn.close()
            
            # Convert dates and clean
            db_data['SampleDate'] = pd.to_datetime(db_data['SampleDate'])
            db_data = db_data.dropna(subset=['SampleDate'])
            
            # Standardize column names to match PKL format
            if 'EC_FTIR' in db_data.columns:
                db_data['EC_ftir'] = db_data['EC_FTIR']
            if 'OC_FTIR' in db_data.columns:
                db_data['OC_ftir'] = db_data['OC_FTIR']
            
            print(f"   ✅ Database: {len(db_data)} samples ({site_code})")
            if len(db_data) > 0:
                print(f"   📅 Date range: {db_data['SampleDate'].min()} to {db_data['SampleDate'].max()}")
                result['metadata']['sources_used'].append('database')
                
        except Exception as e:
            print(f"   ❌ Database error: {e}")
            db_data = None
    
    # === 3. Apply Merge Strategy ===
    print(f"\n🔀 Applying merge strategy: {merge_strategy}")
    
    if merge_strategy == 'pkl_only' or (merge_strategy == 'pkl_first' and pkl_merged is not None):
        # Use merged PKL as primary data source
        if pkl_merged is not None:
            result['data'] = pkl_merged.copy()
            result['metadata']['total_samples'] = len(pkl_merged)
            
            # Extract component datasets
            if 'datetime_local' in pkl_merged.columns:
                result['metadata']['date_range'] = (
                    pkl_merged['datetime_local'].min(),
                    pkl_merged['datetime_local'].max()
                )
            
            # Extract FTIR data
            if any('ftir' in col.lower() for col in pkl_merged.columns):
                ftir_cols = ['datetime_local'] + [col for col in pkl_merged.columns 
                                                 if any(x in col.lower() for x in ['ftir', 'ec_', 'oc_'])]
                result['ftir_data'] = pkl_merged[ftir_cols].copy()
            
            # Extract HIPS data (if Fabs column exists)
            if 'Fabs' in pkl_merged.columns:
                result['hips_data'] = pkl_merged[['datetime_local', 'Fabs']].copy()
            
            print(f"   ✅ Using merged PKL as primary data source")
            
        elif pkl_aethalometer is not None and db_data is not None:
            # Temporal merge of PKL aethalometer with database HIPS/FTIR
            result['data'] = temporal_merge_datasets(
                pkl_aethalometer, 
                db_data, 
                window_hours=temporal_window_hours,
                timezone=timezone
            )
            result['metadata']['processing_notes'].append(f'Temporal merge: PKL + database')
            
    elif merge_strategy == 'database_only' or (merge_strategy == 'database_first' and db_data is not None):
        if db_data is not None:
            result['data'] = db_data.copy()
            result['hips_data'] = db_data[['SampleDate', 'Fabs', 'Site']].copy()
            result['ftir_data'] = db_data[['SampleDate', 'EC_ftir', 'OC_ftir', 'Site']].copy()
            result['metadata']['total_samples'] = len(db_data)
            result['metadata']['date_range'] = (db_data['SampleDate'].min(), db_data['SampleDate'].max())
            print(f"   ✅ Using database as primary data source")
    
    elif merge_strategy == 'temporal_merge' and pkl_aethalometer is not None and db_data is not None:
        # Explicit temporal merge
        result['data'] = temporal_merge_datasets(
            pkl_aethalometer, 
            db_data, 
            window_hours=temporal_window_hours,
            timezone=timezone
        )
        result['hips_data'] = db_data[['SampleDate', 'Fabs', 'Site']].copy()
        result['ftir_data'] = db_data[['SampleDate', 'EC_ftir', 'OC_ftir', 'Site']].copy()
        result['metadata']['processing_notes'].append(f'Temporal merge with ±{temporal_window_hours}h window')
        print(f"   ✅ Performed temporal merge")
    
    # === 4. Final Processing and Summary ===
    if result['data'] is not None:
        result['metadata']['total_samples'] = len(result['data'])
        
        # Categorize available columns
        all_cols = list(result['data'].columns)
        result['metadata']['columns_available'].update({
            'total_columns': len(all_cols),
            'bc_corrected': [col for col in all_cols if 'BCc' in col and 'corrected' in col],
            'bc_original': [col for col in all_cols if 'BCc' in col and 'corrected' not in col],
            'atn_columns': [col for col in all_cols if 'ATN' in col],
            'ftir_columns': [col for col in all_cols if any(x in col.lower() for x in ['ftir', 'ec_', 'oc_'])],
            'datetime_columns': [col for col in all_cols if any(x in col.lower() for x in ['time', 'date'])]
        })
        
        print(f"\n📋 Final Results Summary:")
        print(f"   🏢 Site: {site_code}")
        print(f"   📊 Total samples: {result['metadata']['total_samples']:,}")
        print(f"   📁 Sources: {', '.join(result['metadata']['sources_used'])}")
        print(f"   🔧 Strategy: {merge_strategy}")
        
        if result['metadata']['date_range']:
            start_date, end_date = result['metadata']['date_range']
            duration = (end_date - start_date).days
            print(f"   📅 Date range: {start_date.date()} to {end_date.date()} ({duration} days)")
        
        # Show key column availability
        key_cols = result['metadata']['columns_available']
        if key_cols.get('bc_corrected'):
            print(f"   ✅ Ethiopia-corrected BC: {len(key_cols['bc_corrected'])} columns")
        if key_cols.get('ftir_columns'):
            print(f"   🧪 FTIR data: {len(key_cols['ftir_columns'])} columns")
        if key_cols.get('atn_columns'):
            print(f"   📊 ATN data: {len(key_cols['atn_columns'])} columns")
            
        if result['metadata']['processing_notes']:
            print(f"   📝 Notes: {'; '.join(result['metadata']['processing_notes'])}")
            
    else:
        print(f"\n❌ No data loaded - check file paths and merge strategy")
        
    return result


def temporal_merge_datasets(
    aethalometer_df: pd.DataFrame, 
    hips_df: pd.DataFrame,
    window_hours: int = 2,
    timezone: str = 'Africa/Addis_Ababa'
) -> pd.DataFrame:
    """
    Merge aethalometer and HIPS datasets using temporal matching.
    
    Based on the approach shown in Untitled-1.ipynb cell "create_merged_pkl_hips_dataset"
    """
    print(f"🔗 Temporal merge: aethalometer ({len(aethalometer_df)} rows) + HIPS ({len(hips_df)} rows)")
    
    # Prepare aethalometer data
    aeth_work = aethalometer_df.copy()
    if 'datetime_local' in aeth_work.columns:
        aeth_work['datetime_local'] = pd.to_datetime(aeth_work['datetime_local'])
        if aeth_work['datetime_local'].dt.tz is None:
            aeth_work['datetime_local'] = aeth_work['datetime_local'].dt.tz_localize(timezone)
    
    # Prepare HIPS data
    hips_work = hips_df.copy()
    if 'SampleDate' in hips_work.columns:
        hips_work['SampleDate'] = pd.to_datetime(hips_work['SampleDate'])
        if hips_work['SampleDate'].dt.tz is None:
            hips_work['SampleDate'] = hips_work['SampleDate'].dt.tz_localize(timezone)
    
    # Sort both datasets
    aeth_sorted = aeth_work.sort_values('datetime_local')
    hips_sorted = hips_work.sort_values('SampleDate')
    
    # Perform temporal merge using merge_asof
    merged = pd.merge_asof(
        hips_sorted,
        aeth_sorted,
        left_on='SampleDate',
        right_on='datetime_local',
        tolerance=pd.Timedelta(hours=window_hours),
        direction='nearest'
    )
    
    # Remove rows where no match was found
    merged = merged.dropna(subset=['datetime_local'])
    
    print(f"   ✅ Merged: {len(merged)} matched pairs (success rate: {len(merged)/len(hips_df)*100:.1f}%)")
    
    return merged


def resample_9am_to_9am(
    df: pd.DataFrame, 
    datetime_col: str = 'datetime_local', 
    timezone: str = 'Africa/Addis_Ababa', 
    min_hours: int = 4
) -> pd.DataFrame:
    """
    Resample data from 9am to 9am next day (from Untitled-1.ipynb)
    """
    df_work = df.copy()
    
    # Ensure datetime column is datetime type
    df_work[datetime_col] = pd.to_datetime(df_work[datetime_col])
    
    # Set as index
    df_work = df_work.set_index(datetime_col)
    
    # Localize timezone if needed
    if df_work.index.tz is None:
        df_work.index = df_work.index.tz_localize(timezone)
    
    # Shift time back by 9 hours so 9am becomes start of day
    df_shifted = df_work.copy()
    df_shifted.index = df_shifted.index - pd.Timedelta(hours=9)
    
    # Get numeric columns only
    numeric_cols = df_shifted.select_dtypes(include=[np.number]).columns
    
    # Resample to daily, calculating mean and count
    daily_means = df_shifted[numeric_cols].resample('D').mean()
    daily_counts = df_shifted[numeric_cols].resample('D').count()
    
    # Filter out days with insufficient data
    for col in numeric_cols:
        insufficient_data = daily_counts[col] < min_hours
        daily_means.loc[insufficient_data, col] = np.nan
    
    # Shift index forward by 9 hours to get 9am timestamps
    daily_means.index = daily_means.index + pd.Timedelta(hours=9)
    daily_means.index.name = 'datetime_local'
    
    return daily_means


# Example usage function based on your files
def load_etad_data_example():
    """
    Example usage based on your specific file structure
    """
    
    # Database path (from your config)
    db_path = "/path/to/your/spartan_ftir_hips.db"  # Update this
    
    # PKL files (from your notebooks directory)
    pkl_files = {
        'aethalometer': '/Users/ahzs645/Github/aethmodular-clean/notebooks/pkl_data_cleaned_ethiopia.pkl',
        'merged': '/Users/ahzs645/Github/aethmodular-clean/notebooks/aethalometer_ftir_merged_etad_9am.pkl'
    }
    
    print("🎯 Example: Loading ETAD data with multiple strategies")
    print("=" * 60)
    
    # Strategy 1: Use merged PKL (fastest, most complete)
    print("\n1️⃣ Strategy: pkl_first (using merged PKL)")
    result1 = load_filter_sample_data(
        db_path=db_path,
        pkl_files=pkl_files,
        site_code='ETAD',
        merge_strategy='pkl_first',
    )
    
    if result1['data'] is not None:
        print(f"   ✅ Success: {len(result1['data'])} samples loaded")
        
        # Show available BC methods (similar to your notebook analysis)
        all_cols = list(result1['data'].columns)
        bc_corrected = [col for col in all_cols if 'BCc' in col and 'corrected' in col]
        bc_original = [col for col in all_cols if 'BCc' in col and 'corrected' not in col and 'manual' not in col]
        ftir_cols = [col for col in all_cols if any(x in col.lower() for x in ['ftir', 'ec_'])]
        
        print(f"   📊 Ethiopia-corrected BC: {bc_corrected[:3]}")
        print(f"   🔧 Original BC: {bc_original[:3]}")
        print(f"   🧪 FTIR columns: {ftir_cols}")
        
        # This data is ready for your analysis workflow from Untitled-1.ipynb
        return result1
    
    # Strategy 2: Temporal merge (if you want to combine fresh aethalometer data with database HIPS)
    print("\n2️⃣ Strategy: temporal_merge (database HIPS + PKL aethalometer)")
    result2 = load_filter_sample_data(
        db_path=db_path,
        pkl_files={'aethalometer': pkl_files['aethalometer']},  # Only aethalometer PKL
        site_code='ETAD',
        merge_strategy='temporal_merge',
        temporal_window_hours=2
    )
    
    if result2['data'] is not None:
        print(f"   ✅ Success: {len(result2['data'])} samples loaded")
        return result2
    
    print("\n❌ No successful data loading - check file paths")
    return None


if __name__ == "__main__":
    # Run example
    result = load_etad_data_example()
    
    if result:
        print(f"\n🎉 Data loaded successfully!")
        print(f"📊 Use result['data'] for your analysis workflow")
        print(f"📋 Available metadata: {list(result['metadata'].keys())}")