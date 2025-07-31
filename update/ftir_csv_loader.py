"""
FTIR CSV Data Loader Module

This module loads FTIR data from CSV files and formats it for merging with aethalometer data.
Compatible with the existing pipeline structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import warnings

class FTIRCSVLoader:
    """Load and process FTIR data from CSV files"""
    
    def __init__(self, csv_path: str):
        """
        Initialize FTIR CSV loader
        
        Args:
            csv_path: Path to the FTIR CSV file
        """
        self.csv_path = Path(csv_path)
        self._validate_file()
        
    def _validate_file(self):
        """Validate that the CSV file exists and has expected structure"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"FTIR CSV file not found: {self.csv_path}")
            
    def load_site_data(self, 
                      site_code: str, 
                      parameters: Optional[List[str]] = None,
                      date_range: Optional[tuple] = None) -> pd.DataFrame:
        """
        Load FTIR data for a specific site
        
        Args:
            site_code: Site identifier (e.g., 'ETAD', 'JPL', etc.)
            parameters: List of parameters to include (e.g., ['EC_ftir', 'OC_ftir'])
            date_range: Tuple of (start_date, end_date) for filtering
            
        Returns:
            DataFrame with datetime index and parameter columns
        """
        
        print(f"📊 Loading FTIR data for site {site_code}...")
        
        # Load the full CSV
        df = pd.read_csv(self.csv_path)
        
        # Filter by site
        site_data = df[df['Site'] == site_code].copy()
        
        if len(site_data) == 0:
            available_sites = df['Site'].unique()
            raise ValueError(f"No data found for site '{site_code}'. Available sites: {list(available_sites)}")
        
        # Convert sample date to datetime
        site_data['SampleDate'] = pd.to_datetime(site_data['SampleDate'])
        
        # Apply date range filter if provided
        if date_range:
            start_date, end_date = date_range
            site_data = site_data[
                (site_data['SampleDate'] >= start_date) & 
                (site_data['SampleDate'] <= end_date)
            ]
        
        # Filter parameters if specified
        if parameters:
            site_data = site_data[site_data['Parameter'].isin(parameters)]
        
        # Pivot to get parameters as columns
        pivot_data = site_data.pivot_table(
            index=['SampleDate', 'FilterId'],
            columns='Parameter',
            values='Concentration_ug_m3',
            aggfunc='first'  # Handle any duplicates
        ).reset_index()
        
        # Flatten column names
        pivot_data.columns.name = None
        
        # Group by SampleDate and take mean if multiple filters per date
        if 'FilterId' in pivot_data.columns:
            daily_data = pivot_data.groupby('SampleDate').agg({
                col: 'mean' for col in pivot_data.columns if col not in ['SampleDate', 'FilterId']
            }).reset_index()
        else:
            daily_data = pivot_data
        
        # Set datetime index
        daily_data.set_index('SampleDate', inplace=True)
        daily_data.index.name = 'datetime_local'
        
        print(f"✅ Loaded {len(daily_data)} FTIR measurements")
        print(f"📅 Date range: {daily_data.index.min()} to {daily_data.index.max()}")
        print(f"🧪 Parameters: {list(daily_data.columns)}")
        
        return daily_data
    
    def get_available_sites(self) -> List[str]:
        """Get list of available sites in the CSV"""
        df = pd.read_csv(self.csv_path)
        return sorted(df['Site'].unique())
    
    def get_available_parameters(self, site_code: Optional[str] = None) -> List[str]:
        """Get list of available parameters, optionally filtered by site"""
        df = pd.read_csv(self.csv_path)
        
        if site_code:
            df = df[df['Site'] == site_code]
            
        return sorted(df['Parameter'].unique())
    
    def get_site_info(self, site_code: str) -> Dict[str, Any]:
        """Get metadata information for a site"""
        df = pd.read_csv(self.csv_path)
        site_data = df[df['Site'] == site_code]
        
        if len(site_data) == 0:
            raise ValueError(f"Site '{site_code}' not found")
        
        # Get unique site metadata
        site_info = {
            'site_code': site_code,
            'latitude': site_data['Latitude'].iloc[0],
            'longitude': site_data['Longitude'].iloc[0],
            'sample_count': len(site_data),
            'date_range': (
                pd.to_datetime(site_data['SampleDate']).min(),
                pd.to_datetime(site_data['SampleDate']).max()
            ),
            'parameters': sorted(site_data['Parameter'].unique()),
            'filter_types': sorted(site_data['FilterType'].unique())
        }
        
        return site_info


def create_ftir_merger_for_csv(csv_path: str, 
                              site_code: str,
                              timezone: str,
                              reference_time: str = '09:00') -> 'FTIRCSVMerger':
    """
    Factory function to create FTIR merger for CSV data
    
    Args:
        csv_path: Path to FTIR CSV file
        site_code: Site identifier
        timezone: Local timezone (e.g., 'Africa/Addis_Ababa', 'America/Los_Angeles')
        reference_time: Time to assign to daily measurements (default: '09:00')
        
    Returns:
        FTIRCSVMerger instance
    """
    return FTIRCSVMerger(csv_path, site_code, timezone, reference_time)


class FTIRCSVMerger:
    """Merge aethalometer data with FTIR data from CSV files"""
    
    def __init__(self, 
                 csv_path: str, 
                 site_code: str,
                 timezone: str,
                 reference_time: str = '09:00'):
        """
        Initialize CSV-based FTIR merger
        
        Args:
            csv_path: Path to FTIR CSV file
            site_code: Site identifier 
            timezone: Local timezone string
            reference_time: Time to assign to daily measurements
        """
        self.loader = FTIRCSVLoader(csv_path)
        self.site_code = site_code
        self.timezone = timezone
        self.reference_time = reference_time
        
    def merge_with_aethalometer(self, 
                               aethalometer_daily: pd.DataFrame,
                               parameters: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Merge FTIR data with daily-averaged aethalometer data
        
        Args:
            aethalometer_daily: Daily averaged aethalometer data with datetime_local index
            parameters: List of FTIR parameters to include
            
        Returns:
            Merged DataFrame with both aethalometer and FTIR data
        """
        
        # Load FTIR data for the site
        ftir_data = self.loader.load_site_data(
            site_code=self.site_code,
            parameters=parameters
        )
        
        # Convert FTIR timestamps to match aethalometer format
        # Assign the reference time (e.g., 9 AM) to each sample date
        ftir_data = ftir_data.copy()
        
        # Parse reference time
        hour, minute = map(int, self.reference_time.split(':'))
        
        # Create datetime with reference time and localize
        ftir_timestamps = pd.to_datetime(ftir_data.index.date).normalize() + pd.Timedelta(hours=hour, minutes=minute)
        ftir_timestamps = ftir_timestamps.tz_localize(self.timezone)
        
        ftir_data.index = ftir_timestamps
        ftir_data.index.name = 'datetime_local'
        
        # Ensure aethalometer data has proper datetime index
        if not isinstance(aethalometer_daily.index, pd.DatetimeIndex):
            if 'datetime_local' in aethalometer_daily.columns:
                aethalometer_daily = aethalometer_daily.set_index('datetime_local')
            else:
                raise ValueError("Aethalometer data must have datetime_local index or column")
        
        # Merge on datetime_local
        merged = pd.merge(
            aethalometer_daily,
            ftir_data,
            left_index=True,
            right_index=True,
            how='inner'  # Only keep dates with both aethalometer and FTIR data
        )
        
        print(f"🔗 Merged data: {len(merged)} matching days")
        print(f"📊 Aethalometer points: {len(aethalometer_daily)}")
        print(f"🧪 FTIR points: {len(ftir_data)}")
        
        return merged


# Example usage functions (similar to the notebook code)
def load_ftir_for_site(csv_path: str, site_code: str) -> pd.DataFrame:
    """
    Convenience function to load FTIR data for a site
    
    Args:
        csv_path: Path to FTIR CSV file
        site_code: Site identifier
        
    Returns:
        FTIR DataFrame for the site
    """
    loader = FTIRCSVLoader(csv_path)
    return loader.load_site_data(site_code)


def merge_aethalometer_ftir_csv(aethalometer_daily: pd.DataFrame,
                               csv_path: str,
                               site_code: str,
                               timezone: str) -> pd.DataFrame:
    """
    One-line function to merge aethalometer and FTIR data from CSV
    
    Args:
        aethalometer_daily: Daily averaged aethalometer data
        csv_path: Path to FTIR CSV file
        site_code: Site identifier
        timezone: Local timezone string
        
    Returns:
        Merged DataFrame
    """
    merger = create_ftir_merger_for_csv(csv_path, site_code, timezone)
    return merger.merge_with_aethalometer(aethalometer_daily)


# Site configuration mapping (based on the notebook examples)
SITE_CONFIGS = {
    'ETAD': {  # Jacros/AddisAbaba
        'timezone': 'Africa/Addis_Ababa',
        'site_names': ['AddisAbaba_Jacros', 'ETAD']  # Alternative names to try
    },
    'JPL': {
        'timezone': 'America/Los_Angeles', 
        'site_names': ['JPL', 'Pasadena']
    },
    'INDH': {  # New Delhi
        'timezone': 'Asia/Kolkata',
        'site_names': ['NewDelhi', 'INDH']
    },
    'Beijing': {
        'timezone': 'Asia/Shanghai',
        'site_names': ['Beijing']
    }
}


def auto_merge_with_csv(aethalometer_daily: pd.DataFrame,
                       csv_path: str,
                       site_code: str) -> pd.DataFrame:
    """
    Automatically merge with site configuration lookup
    
    Args:
        aethalometer_daily: Daily averaged aethalometer data
        csv_path: Path to FTIR CSV file  
        site_code: Site identifier
        
    Returns:
        Merged DataFrame
    """
    
    if site_code not in SITE_CONFIGS:
        raise ValueError(f"Unknown site code '{site_code}'. Available: {list(SITE_CONFIGS.keys())}")
    
    config = SITE_CONFIGS[site_code]
    loader = FTIRCSVLoader(csv_path)
    
    # Try different site name variations
    ftir_site_name = None
    for name in config['site_names']:
        try:
            test_data = loader.load_site_data(name)
            ftir_site_name = name
            print(f"✅ Found FTIR data for site name: {name}")
            break
        except ValueError:
            continue
    
    if ftir_site_name is None:
        available = loader.get_available_sites()
        raise ValueError(f"Could not find FTIR data for site {site_code}. Available sites: {available}")
    
    # Create merger and merge
    merger = create_ftir_merger_for_csv(
        csv_path, 
        ftir_site_name, 
        config['timezone']
    )
    
    return merger.merge_with_aethalometer(aethalometer_daily)
