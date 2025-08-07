"""
Enhanced function to load HIPS/FTIR data from multiple sources:
1. SQLite database (original)
2. Pre-processed PKL files (with or without FTIR data already merged)
3. Combination of both sources
"""

import pandas as pd
import sqlite3
import os
import numpy as np
from pathlib import Path

def load_filter_sample_data(db_path=None, pkl_files=None, merge_strategy='database_first'):
    """
    Enhanced function to load ETAD (HIPS) and FTIR data from multiple sources.
    
    Parameters:
    -----------
    db_path : str, optional
        Path to the SQLite database containing HIPS/FTIR data
    pkl_files : dict or list, optional
        Dictionary with keys 'merged' and/or 'aethalometer' pointing to PKL files, or
        List of PKL file paths
    merge_strategy : str, default 'database_first'
        Strategy for combining data:
        - 'database_first': Use database as primary, supplement with PKL
        - 'pkl_first': Use PKL as primary, supplement with database
        - 'database_only': Only use database
        - 'pkl_only': Only use PKL files
        - 'merge_temporal': Merge database HIPS with PKL aethalometer data temporally
    
    Returns:
    --------
    dict: Dictionary containing:
        - 'hips_data': DataFrame with HIPS measurements (Fabs)
        - 'ftir_data': DataFrame with FTIR measurements (EC_FTIR, OC_FTIR)
        - 'merged_data': DataFrame with temporally matched data (if available)
        - 'metadata': Dictionary with information about data sources and processing
    """
    
    print("🔄 Enhanced HIPS/FTIR Data Loader")
    print("=" * 50)
    
    result = {
        'hips_data': None,
        'ftir_data': None, 
        'merged_data': None,
        'metadata': {
            'sources_used': [],
            'merge_strategy': merge_strategy,
            'total_samples': 0,
            'date_range': None,
            'processing_notes': []
        }
    }
    
    # === Load from Database ===
    db_data = None
    if db_path and os.path.exists(db_path):
        try:
            print(f"📊 Loading from database: {os.path.basename(db_path)}")
            conn = sqlite3.connect(db_path)
            
            # Load HIPS/FTIR data for the ETAD site
            query = """
            SELECT f.filter_id, 
                   f.sample_date AS SampleDate, 
                   m.ec_ftir AS EC_FTIR,
                   m.oc_ftir AS OC_FTIR,
                   m.fabs AS Fabs,
                   f.site_code AS Site
            FROM filters f
            JOIN ftir_sample_measurements m USING(filter_id)
            WHERE f.site_code = 'ETAD'
            ORDER BY f.sample_date;
            """
            
            db_data = pd.read_sql_query(query, conn)
            db_data['SampleDate'] = pd.to_datetime(db_data['SampleDate'])
            conn.close()
            
            # Remove rows with null dates
            db_data = db_data.dropna(subset=['SampleDate'])
            
            valid_samples = len(db_data)
            print(f"✅ Database: {valid_samples} samples loaded")
            if valid_samples > 0:
                print(f"   📅 Date range: {db_data['SampleDate'].min()} to {db_data['SampleDate'].max()}")
                print(f"   📊 Available: EC_FTIR, OC_FTIR, Fabs (HIPS)")
                result['metadata']['sources_used'].append('database')
                
        except Exception as e:
            print(f"❌ Database loading error: {e}")
            db_data = None
    elif db_path:
        print(f"⚠️ Database not found: {db_path}")
    
    # === Load from PKL files ===
    pkl_merged_data = None
    pkl_aethalometer_data = None
    
    if pkl_files:
        # Handle different input formats
        if isinstance(pkl_files, dict):
            merged_pkl_path = pkl_files.get('merged')
            aethalometer_pkl_path = pkl_files.get('aethalometer')
        elif isinstance(pkl_files, list):
            # Try to identify which is which based on size/content
            merged_pkl_path = None
            aethalometer_pkl_path = None
            for pkl_path in pkl_files:
                if os.path.exists(pkl_path):
                    # Quick check for FTIR columns
                    try:
                        df_sample = pd.read_pickle(pkl_path)
                        if any(col in df_sample.columns for col in ['EC_ftir', 'EC_FTIR', 'OC_ftir', 'OC_FTIR']):
                            merged_pkl_path = pkl_path
                        else:
                            aethalometer_pkl_path = pkl_path
                    except:
                        continue
        else:
            # Single file path
            if os.path.exists(pkl_files):
                # Try to determine if it has FTIR data
                try:
                    df_sample = pd.read_pickle(pkl_files)
                    if any(col in df_sample.columns for col in ['EC_ftir', 'EC_FTIR', 'OC_ftir', 'OC_FTIR']):
                        merged_pkl_path = pkl_files
                    else:
                        aethalometer_pkl_path = pkl_files
                except:
                    aethalometer_pkl_path = pkl_files
        
        # Load merged PKL (already contains FTIR data)
        if merged_pkl_path and os.path.exists(merged_pkl_path):
            try:
                print(f"📊 Loading merged PKL: {os.path.basename(merged_pkl_path)}")
                pkl_merged_data = pd.read_pickle(merged_pkl_path)
                
                # Standardize column names
                column_mapping = {
                    'EC_ftir': 'EC_FTIR',
                    'OC_ftir': 'OC_FTIR',
                    'datetime_local': 'SampleDate'
                }
                
                for old_col, new_col in column_mapping.items():
                    if old_col in pkl_merged_data.columns and new_col not in pkl_merged_data.columns:
                        pkl_merged_data = pkl_merged_data.rename(columns={old_col: new_col})
                
                # Ensure SampleDate is datetime
                if 'SampleDate' not in pkl_merged_data.columns:
                    # Use index if it's datetime
                    if hasattr(pkl_merged_data.index, 'to_pydatetime'):
                        pkl_merged_data['SampleDate'] = pkl_merged_data.index
                    else:
                        print("⚠️ No SampleDate column found in merged PKL")
                
                if 'SampleDate' in pkl_merged_data.columns:
                    pkl_merged_data['SampleDate'] = pd.to_datetime(pkl_merged_data['SampleDate'])
                    pkl_merged_data = pkl_merged_data.dropna(subset=['SampleDate'])
                
                print(f"✅ Merged PKL: {len(pkl_merged_data)} samples loaded")
                if len(pkl_merged_data) > 0 and 'SampleDate' in pkl_merged_data.columns:
                    print(f"   📅 Date range: {pkl_merged_data['SampleDate'].min()} to {pkl_merged_data['SampleDate'].max()}")
                
                # Check available columns
                ftir_cols = [col for col in pkl_merged_data.columns if col in ['EC_FTIR', 'OC_FTIR']]
                hips_cols = [col for col in pkl_merged_data.columns if 'Fabs' in col or 'fabs' in col]
                bc_cols = [col for col in pkl_merged_data.columns if 'BC1' in col]
                
                print(f"   📊 Available: {', '.join(ftir_cols + hips_cols + bc_cols[:3])}")
                result['metadata']['sources_used'].append('merged_pkl')
                
            except Exception as e:
                print(f"❌ Merged PKL loading error: {e}")
                pkl_merged_data = None
        
        # Load aethalometer-only PKL
        if aethalometer_pkl_path and os.path.exists(aethalometer_pkl_path):
            try:
                print(f"📊 Loading aethalometer PKL: {os.path.basename(aethalometer_pkl_path)}")
                pkl_aethalometer_data = pd.read_pickle(aethalometer_pkl_path)
                
                # Standardize datetime column
                if 'datetime_local' in pkl_aethalometer_data.columns:
                    pkl_aethalometer_data['SampleDate'] = pd.to_datetime(pkl_aethalometer_data['datetime_local'])
                elif hasattr(pkl_aethalometer_data.index, 'to_pydatetime'):
                    pkl_aethalometer_data['SampleDate'] = pkl_aethalometer_data.index
                
                if 'SampleDate' in pkl_aethalometer_data.columns:
                    pkl_aethalometer_data = pkl_aethalometer_data.dropna(subset=['SampleDate'])
                
                print(f"✅ Aethalometer PKL: {len(pkl_aethalometer_data)} samples loaded")
                if len(pkl_aethalometer_data) > 0 and 'SampleDate' in pkl_aethalometer_data.columns:
                    print(f"   📅 Date range: {pkl_aethalometer_data['SampleDate'].min()} to {pkl_aethalometer_data['SampleDate'].max()}")
                
                # Check BC columns
                bc_cols = [col for col in pkl_aethalometer_data.columns if 'BC1' in col]
                print(f"   📊 Available BC columns: {len(bc_cols)} ({bc_cols[:3]})")
                result['metadata']['sources_used'].append('aethalometer_pkl')
                
            except Exception as e:
                print(f"❌ Aethalometer PKL loading error: {e}")
                pkl_aethalometer_data = None
    
    # === Apply Merge Strategy ===
    print(f"\n🔀 Applying merge strategy: {merge_strategy}")
    
    if merge_strategy == 'database_only':
        if db_data is not None:
            result['hips_data'] = db_data[['filter_id', 'SampleDate', 'Fabs', 'Site']].copy()
            result['ftir_data'] = db_data[['filter_id', 'SampleDate', 'EC_FTIR', 'OC_FTIR', 'Site']].copy()
            result['merged_data'] = db_data.copy()
            result['metadata']['total_samples'] = len(db_data)
            result['metadata']['date_range'] = (db_data['SampleDate'].min(), db_data['SampleDate'].max())
            
    elif merge_strategy == 'pkl_only':
        if pkl_merged_data is not None:
            # Use merged PKL data
            result['merged_data'] = pkl_merged_data.copy()
            
            # Extract HIPS and FTIR components
            if 'Fabs' in pkl_merged_data.columns:
                hips_cols = ['SampleDate', 'Fabs']
                if 'filter_id' in pkl_merged_data.columns:
                    hips_cols.insert(1, 'filter_id')
                result['hips_data'] = pkl_merged_data[hips_cols].copy()
            
            ftir_cols = ['SampleDate'] + [col for col in pkl_merged_data.columns if col in ['EC_FTIR', 'OC_FTIR']]
            if len(ftir_cols) > 1:
                result['ftir_data'] = pkl_merged_data[ftir_cols].copy()
            
            result['metadata']['total_samples'] = len(pkl_merged_data)
            if 'SampleDate' in pkl_merged_data.columns:
                result['metadata']['date_range'] = (pkl_merged_data['SampleDate'].min(), pkl_merged_data['SampleDate'].max())
    
    elif merge_strategy == 'merge_temporal' and db_data is not None and pkl_aethalometer_data is not None:
        # Temporal merge of database HIPS with PKL aethalometer
        print("🔗 Performing temporal merge...")
        
        # Use pandas merge_asof for temporal matching
        db_sorted = db_data.sort_values('SampleDate')
        pkl_sorted = pkl_aethalometer_data.sort_values('SampleDate')
        
        # Merge HIPS data to aethalometer data (finding closest aethalometer measurement for each HIPS sample)
        merged = pd.merge_asof(
            db_sorted,
            pkl_sorted,
            on='SampleDate',
            tolerance=pd.Timedelta(hours=2),  # 2-hour window
            direction='nearest'
        )
        
        # Remove rows where no match was found
        merged = merged.dropna(subset=['SampleDate'])
        
        print(f"✅ Temporal merge: {len(merged)} matched pairs from {len(db_data)} HIPS samples")
        
        result['merged_data'] = merged
        result['hips_data'] = db_data[['filter_id', 'SampleDate', 'Fabs', 'Site']].copy()
        result['ftir_data'] = db_data[['filter_id', 'SampleDate', 'EC_FTIR', 'OC_FTIR', 'Site']].copy()
        result['metadata']['total_samples'] = len(merged)
        result['metadata']['date_range'] = (merged['SampleDate'].min(), merged['SampleDate'].max())
        result['metadata']['processing_notes'].append(f"Temporal merge: {len(merged)}/{len(db_data)} HIPS samples matched")
    
    else:
        # Default: use best available data
        if pkl_merged_data is not None:
            # Use merged PKL as primary
            result['merged_data'] = pkl_merged_data.copy()
            
            if 'Fabs' in pkl_merged_data.columns:
                result['hips_data'] = pkl_merged_data[['SampleDate', 'Fabs']].copy()
            
            ftir_cols = ['SampleDate'] + [col for col in pkl_merged_data.columns if col in ['EC_FTIR', 'OC_FTIR']]
            if len(ftir_cols) > 1:
                result['ftir_data'] = pkl_merged_data[ftir_cols].copy()
            
            result['metadata']['total_samples'] = len(pkl_merged_data)
            if 'SampleDate' in pkl_merged_data.columns:
                result['metadata']['date_range'] = (pkl_merged_data['SampleDate'].min(), pkl_merged_data['SampleDate'].max())
                
        elif db_data is not None:
            # Fallback to database
            result['hips_data'] = db_data[['filter_id', 'SampleDate', 'Fabs', 'Site']].copy()
            result['ftir_data'] = db_data[['filter_id', 'SampleDate', 'EC_FTIR', 'OC_FTIR', 'Site']].copy()
            result['merged_data'] = db_data.copy()
            result['metadata']['total_samples'] = len(db_data)
            result['metadata']['date_range'] = (db_data['SampleDate'].min(), db_data['SampleDate'].max())
    
    # === Final Summary ===
    print(f"\n📋 Loading Summary:")
    print(f"   Sources used: {', '.join(result['metadata']['sources_used'])}")
    print(f"   Strategy: {merge_strategy}")
    print(f"   Total samples: {result['metadata']['total_samples']}")
    
    if result['metadata']['date_range']:
        start_date, end_date = result['metadata']['date_range']
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Duration: {(end_date - start_date).days} days")
    
    # Check what data is available
    available_data = []
    if result['hips_data'] is not None:
        available_data.append(f"HIPS ({len(result['hips_data'])} samples)")
    if result['ftir_data'] is not None:
        available_data.append(f"FTIR ({len(result['ftir_data'])} samples)")
    if result['merged_data'] is not None:
        available_data.append(f"Merged ({len(result['merged_data'])} samples)")
    
    print(f"   Available: {', '.join(available_data)}")
    
    if result['metadata']['processing_notes']:
        print(f"   Notes: {'; '.join(result['metadata']['processing_notes'])}")
    
    return result


# Example usage functions
def load_old_hips_data_example():
    """Example of how to use the enhanced function with your specific files."""
    
    # Path to your database
    db_path = "/path/to/your/spartan_ftir_hips.db"
    
    # Path to your PKL files  
    pkl_files = {
        'merged': "/Users/ahzs645/Github/aethmodular-clean/notebooks/aethalometer_ftir_merged_etad_9am.pkl",
        'aethalometer': "/Users/ahzs645/Github/aethmodular-clean/notebooks/pkl_data_cleaned_ethiopia.pkl"
    }
    
    # Load with different strategies
    strategies = ['pkl_only', 'merge_temporal', 'database_first']
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"TESTING STRATEGY: {strategy}")
        print(f"{'='*60}")
        
        result = load_filter_sample_data(
            db_path=db_path,
            pkl_files=pkl_files,
            merge_strategy=strategy
        )
        
        # Use the result
        if result['merged_data'] is not None:
            print(f"✅ Strategy {strategy} successful!")
            print(f"   Merged data columns: {list(result['merged_data'].columns)[:10]}...")
        else:
            print(f"❌ Strategy {strategy} failed")


if __name__ == "__main__":
    # Run example
    load_old_hips_data_example()