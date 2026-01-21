"""
Aethalometer-Filter Data Matching Module

This module provides functionality to load and match aethalometer data (9AM-9AM daily averages)
with filter-based measurements for any site. It handles the temporal alignment between 
continuous aethalometer measurements and 24-hour filter collection periods.

Usage:
    from src.data.loaders.aethalometer_filter_matcher import AethalometerFilterMatcher
    
    matcher = AethalometerFilterMatcher(
        aethalometer_path="path/to/df_Jacros_9am_resampled.pkl",
        filter_db_path="path/to/unified_filter_dataset.pkl"
    )
    
    matched_data = matcher.match_site_data('ETAD')
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings

class AethalometerFilterMatcher:
    """
    A class to handle loading and matching of aethalometer and filter data.
    
    This class provides methods to:
    - Load aethalometer data (9AM-9AM resampled)
    - Load filter data for any site
    - Match data by date for correlation analysis
    - Extract specific parameters for analysis
    """
    
    def __init__(self, aethalometer_path: str, filter_db_path: str):
        """
        Initialize the matcher with data file paths.
        
        Args:
            aethalometer_path: Path to the aethalometer pickle file (9AM resampled)
            filter_db_path: Path to the unified filter database pickle file
        """
        self.aethalometer_path = aethalometer_path
        self.filter_db_path = filter_db_path
        self.aethalometer_data = None
        self.filter_loader = None
        self._setup_filter_loader()
        
    def _setup_filter_loader(self):
        """Setup the filter data loader."""
        try:
            # Try to import the data loader module
            # Add the FTIR_HIPS_Chem directory to path if needed
            filter_dir = os.path.dirname(self.filter_db_path)
            parent_dir = os.path.dirname(filter_dir)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            
            from data_loader_module import load_filter_database
            
            if os.path.exists(self.filter_db_path):
                self.filter_loader = load_filter_database(self.filter_db_path)
                print(f"✅ Filter database loaded from: {self.filter_db_path}")
            else:
                raise FileNotFoundError(f"Filter database not found: {self.filter_db_path}")
                
        except ImportError as e:
            raise ImportError(f"Could not import data_loader_module: {e}")
    
    def load_aethalometer_data(self) -> pd.DataFrame:
        """
        Load aethalometer data from pickle file.
        
        Returns:
            DataFrame with aethalometer data
        """
        if self.aethalometer_data is None:
            if os.path.exists(self.aethalometer_path):
                self.aethalometer_data = pd.read_pickle(self.aethalometer_path)
                print(f"✅ Aethalometer data loaded from: {self.aethalometer_path}")
                print(f"   Dataset shape: {self.aethalometer_data.shape}")
                print(f"   Date range: {self.aethalometer_data['datetime_local'].min()} to {self.aethalometer_data['datetime_local'].max()}")
                
                # Show available sites if Site column exists
                if 'Site' in self.aethalometer_data.columns:
                    sites = self.aethalometer_data['Site'].unique()
                    print(f"   Available sites: {sites}")
                else:
                    print("   No 'Site' column found - assuming single site data")
            else:
                raise FileNotFoundError(f"Aethalometer data not found: {self.aethalometer_path}")
        
        return self.aethalometer_data
    
    def get_available_sites(self) -> List[str]:
        """
        Get list of available sites in filter database.
        
        Returns:
            List of site codes
        """
        if self.filter_loader is None:
            return []
        return self.filter_loader.get_available_sites()
    
    def get_site_filter_data(self, site_code: str, parameters: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get filter data for a specific site.
        
        Args:
            site_code: Site code (e.g., 'ETAD', 'CHTS', 'INDH', 'USPA')
            parameters: List of parameters to extract. If None, extracts common parameters.
        
        Returns:
            DataFrame with filter data for the site
        """
        if self.filter_loader is None:
            raise ValueError("Filter loader not available")
        
        # Default parameters if none specified
        if parameters is None:
            parameters = ['EC_ftir', 'HIPS_Fabs', 'ChemSpec_Iron_PM2.5']
        
        print(f"📂 Extracting {site_code} filter measurements...")
        
        filter_datasets = []
        
        for param in parameters:
            try:
                param_data = self.filter_loader.get_parameter_data(site_code, param)
                
                if len(param_data) > 0:
                    df = param_data[['SampleDate', 'Concentration']].copy()
                    # Clean parameter name for column
                    clean_param = param.replace('ChemSpec_', '').replace('_PM2.5', '_ChemSpec')
                    df = df.rename(columns={'Concentration': clean_param})
                    df['SampleDate'] = pd.to_datetime(df['SampleDate'])
                    filter_datasets.append((clean_param, df))
                    print(f"   ✅ {clean_param}: {len(param_data)} measurements")
                    print(f"      Date range: {df['SampleDate'].min().date()} to {df['SampleDate'].max().date()}")
                else:
                    print(f"   ❌ {param}: No data found")
                    
            except Exception as e:
                print(f"   ⚠️  {param}: Error loading data - {e}")
                continue
        
        # Merge all filter datasets
        if len(filter_datasets) > 0:
            combined_df = filter_datasets[0][1].copy()
            
            for param_name, param_df in filter_datasets[1:]:
                combined_df = pd.merge(combined_df, param_df, on='SampleDate', how='outer')
            
            print(f"\n✅ Combined filter dataset: {combined_df.shape}")
            print(f"   Date range: {combined_df['SampleDate'].min().date()} to {combined_df['SampleDate'].max().date()}")
            
            # Show completeness
            print(f"   Data completeness:")
            for col in combined_df.columns:
                if col != 'SampleDate':
                    count = combined_df[col].notna().sum()
                    total = len(combined_df)
                    pct = (count / total) * 100
                    print(f"     {col}: {count}/{total} ({pct:.1f}%)")
            
            return combined_df
        else:
            print("❌ No filter data available")
            return pd.DataFrame()
    
    def get_site_aethalometer_data(self, site_code: Optional[str] = None) -> pd.DataFrame:
        """
        Get aethalometer data for a specific site.
        
        Args:
            site_code: Site code. If None and no Site column exists, returns all data.
        
        Returns:
            DataFrame with aethalometer data for the site
        """
        aeth_data = self.load_aethalometer_data()
        
        if site_code is None:
            print("📍 No site code specified - returning all aethalometer data")
            return aeth_data.copy()
        
        if 'Site' in aeth_data.columns:
            site_mask = aeth_data['Site'].str.upper() == site_code.upper()
            site_data = aeth_data[site_mask].copy()
            
            if len(site_data) > 0:
                print(f"✅ Found {len(site_data)} aethalometer records for {site_code}")
                print(f"   Date range: {site_data['datetime_local'].min()} to {site_data['datetime_local'].max()}")
                return site_data
            else:
                print(f"❌ No aethalometer data found for site: {site_code}")
                return pd.DataFrame()
        else:
            print(f"📍 No 'Site' column - assuming all data is for {site_code or 'target site'}")
            return aeth_data.copy()
    
    def match_site_data(
        self, 
        site_code: str, 
        filter_parameters: Optional[List[str]] = None,
        aethalometer_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Match aethalometer and filter data for a specific site.
        
        Args:
            site_code: Site code (e.g., 'ETAD', 'CHTS', 'INDH', 'USPA')
            filter_parameters: List of filter parameters to include
            aethalometer_columns: List of aethalometer columns to include
        
        Returns:
            DataFrame with matched data
        """
        print(f"🔗 Matching aethalometer and filter data for {site_code}...")
        
        # Get site-specific data
        aeth_data = self.get_site_aethalometer_data(site_code)
        filter_data = self.get_site_filter_data(site_code, filter_parameters)
        
        if aeth_data.empty or filter_data.empty:
            print("❌ Cannot perform matching - missing data for one or both sources")
            return pd.DataFrame()
        
        # Prepare data for matching
        aeth_match = aeth_data.copy()
        filter_match = filter_data.copy()
        
        # Convert to date for matching
        aeth_match['match_date'] = aeth_match['datetime_local'].dt.date
        filter_match['match_date'] = filter_match['SampleDate'].dt.date
        
        print(f"📅 Aethalometer data: {len(aeth_match)} daily averages")
        print(f"   Date range: {aeth_match['match_date'].min()} to {aeth_match['match_date'].max()}")
        
        print(f"📅 Filter data: {len(filter_match)} samples")
        print(f"   Date range: {filter_match['match_date'].min()} to {filter_match['match_date'].max()}")
        
        # Find overlapping dates
        aeth_dates = set(aeth_match['match_date'])
        filter_dates = set(filter_match['match_date'])
        overlapping_dates = aeth_dates.intersection(filter_dates)
        
        print(f"\n🎯 Overlapping dates: {len(overlapping_dates)}")
        
        if len(overlapping_dates) == 0:
            print("❌ No overlapping dates found")
            return pd.DataFrame()
        
        print(f"   Date range: {min(overlapping_dates)} to {max(overlapping_dates)}")
        
        # Perform merge
        merged_data = pd.merge(
            aeth_match,
            filter_match,
            on='match_date',
            how='inner',
            suffixes=('_aeth', '_filter')
        )
        
        print(f"\n✅ Successfully merged {len(merged_data)} matching records!")
        
        # Select columns if specified
        if aethalometer_columns:
            # Keep specified aethalometer columns plus essential ones
            keep_cols = ['match_date', 'datetime_local'] + aethalometer_columns
            aeth_cols = [col for col in keep_cols if col in merged_data.columns]
        else:
            # Default: keep BC columns
            bc_cols = [col for col in merged_data.columns if 'BC' in col and 'smoothed' in col]
            aeth_cols = ['match_date', 'datetime_local'] + bc_cols[:6]  # First 6 BC columns
        
        # Filter columns
        filter_cols = [col for col in merged_data.columns if col not in aeth_cols and 'filter' not in col]
        filter_cols = [col for col in filter_cols if col not in ['SampleDate', 'match_date']]
        
        # Create final dataset
        final_cols = aeth_cols + filter_cols
        available_cols = [col for col in final_cols if col in merged_data.columns]
        
        result = merged_data[available_cols].copy()
        
        # Summary statistics
        print(f"\n📊 Matched dataset summary:")
        print(f"   Total records: {len(result)}")
        print(f"   Date range: {result['match_date'].min()} to {result['match_date'].max()}")
        
        print(f"\n   Available measurements:")
        for col in result.columns:
            if col not in ['match_date', 'datetime_local']:
                count = result[col].notna().sum() if result[col].dtype in ['float64', 'int64'] else len(result)
                print(f"     {col}: {count} values")
        
        return result
    
    def get_correlation_summary(self, matched_data: pd.DataFrame) -> Dict:
        """
        Generate correlation summary for matched data.
        
        Args:
            matched_data: DataFrame from match_site_data()
        
        Returns:
            Dictionary with correlation statistics
        """
        if matched_data.empty:
            return {}
        
        # Get numeric columns
        numeric_cols = matched_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = matched_data[numeric_cols].corr()
        
        # Key correlation pairs
        key_pairs = [
            ('IR BCc smoothed', 'EC_ftir', 'IR BC vs EC'),
            ('Blue BCc smoothed', 'HIPS_Fabs', 'Blue BC vs Fabs'),
            ('UV BCc smoothed', 'Iron_ChemSpec', 'UV BC vs Iron'),
            ('EC_ftir', 'HIPS_Fabs', 'EC vs Fabs')
        ]
        
        correlations = {}
        for col1, col2, label in key_pairs:
            if col1 in matched_data.columns and col2 in matched_data.columns:
                mask = matched_data[[col1, col2]].notna().all(axis=1)
                if mask.sum() > 2:
                    corr = matched_data.loc[mask, col1].corr(matched_data.loc[mask, col2])
                    r2 = corr**2
                    n = mask.sum()
                    correlations[label] = {
                        'r': corr,
                        'r2': r2,
                        'n': n,
                        'columns': (col1, col2)
                    }
        
        return {
            'correlations': correlations,
            'correlation_matrix': corr_matrix,
            'numeric_columns': numeric_cols,
            'total_records': len(matched_data)
        }

def quick_match(aethalometer_path: str, filter_db_path: str, site_code: str) -> pd.DataFrame:
    """
    Quick utility function to match data for a site.
    
    Args:
        aethalometer_path: Path to aethalometer pickle file
        filter_db_path: Path to filter database pickle file
        site_code: Site code to match
    
    Returns:
        Matched DataFrame
    """
    matcher = AethalometerFilterMatcher(aethalometer_path, filter_db_path)
    return matcher.match_site_data(site_code)