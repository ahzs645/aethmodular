# ========================================================================
# PKL Data Cleaning Pipeline - Enhanced Version
# ========================================================================
"""
PKL data cleaning pipeline for aethalometer data processing.

This module provides comprehensive data cleaning functions specifically designed
for PKL format aethalometer data, including status cleaning, optical saturation
removal, flow validation, and roughness-based quality control.
"""

import sys
import os
import importlib.util
import pandas as pd
from itertools import groupby
from operator import itemgetter
from typing import Dict, List, Union, Optional, Tuple

# Optional statsmodels import - will be imported when needed
try:
    from statsmodels import robust
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Import external calibration module
def _import_calibration():
    """Import the external calibration module."""
    # Get the path to the external calibration module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    external_dir = os.path.join(current_dir, '..', '..', 'external')
    module_path = os.path.join(external_dir, 'calibration.py')
    
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Calibration module not found at {module_path}")
    
    module_name = "calibration"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import calibration module
calibration = _import_calibration()

# Default wavelengths to filter
DEFAULT_WAVELENGTHS = ['IR', 'Blue']

class PKLDataCleaner:
    """
    Main class for PKL data cleaning operations.
    
    This class provides a comprehensive suite of cleaning methods for
    aethalometer data in PKL format.
    """
    
    def __init__(self, wavelengths_to_filter=None, verbose=True):
        """
        Initialize the PKL data cleaner.
        
        Args:
            wavelengths_to_filter (list): List of wavelengths to process.
                                        Defaults to ['IR', 'Blue'].
            verbose (bool): Whether to print cleaning progress.
        """
        self.wls_to_filter = wavelengths_to_filter or DEFAULT_WAVELENGTHS
        self.verbose = verbose
    
    def diagnose_data_structure(self, df):
        """
        Diagnose the data structure before cleaning.
        
        Args:
            df (pd.DataFrame): DataFrame to diagnose
        """
        if not self.verbose:
            return
            
        print("🔍 Data Structure Diagnosis:")
        print("-" * 30)
        print(f"DataFrame shape: {df.shape}")
        print(f"Date range: {df['datetime_local'].min()} to {df['datetime_local'].max()}")
        
        # Check for BC columns
        bc_columns = []
        bc_smoothed_columns = []
        atn_columns = []
        flow_columns = []
        
        for col in df.columns:
            if ' BC' in col and 'smoothed' not in col:
                bc_columns.append(col)
            elif ' BC' in col and 'smoothed' in col:
                bc_smoothed_columns.append(col)
            elif ' ATN' in col:
                atn_columns.append(col)
            elif 'Flow' in col:
                flow_columns.append(col)
        
        print(f"BC columns: {len(bc_columns)} (e.g., {bc_columns[:3] if bc_columns else 'None'})")
        print(f"BC smoothed columns: {len(bc_smoothed_columns)} (e.g., {bc_smoothed_columns[:3] if bc_smoothed_columns else 'None'})")
        print(f"ATN columns: {len(atn_columns)} (e.g., {atn_columns[:3] if atn_columns else 'None'})")
        print(f"Flow columns: {len(flow_columns)} (e.g., {flow_columns[:3] if flow_columns else 'None'})")
        
        # Check specifically for wavelengths we want to filter
        print(f"\nTargeted wavelengths: {self.wls_to_filter}")
        for wl in self.wls_to_filter:
            bc_col = f'{wl} BCc'
            bc_smoothed_col = f'{wl} BCc smoothed'
            atn_col = f'{wl} ATN1'
            
            status = []
            if bc_col in df.columns:
                status.append("✅ BC")
            else:
                status.append("❌ BC")
                
            if bc_smoothed_col in df.columns:
                status.append("✅ BC smoothed")
            else:
                status.append("❌ BC smoothed")
                
            if atn_col in df.columns:
                status.append("✅ ATN")
            else:
                status.append("❌ ATN")
            
            print(f"  {wl}: {' | '.join(status)}")
        
        print("-" * 30)
    
    def report_removal(self, df_before, df_after, label):
        """
        Report the number and percentage of rows removed during cleaning.
        
        Args:
            df_before (pd.DataFrame): DataFrame before cleaning
            df_after (pd.DataFrame): DataFrame after cleaning
            label (str): Description of the cleaning step
            
        Returns:
            pd.DataFrame: The cleaned DataFrame
        """
        n_removed = len(df_before) - len(df_after)
        if self.verbose:
            print(f"{label}: Removed {n_removed} rows ({n_removed / len(df_before) * 100:.2f}%)")
        return df_after
    
    def clean_by_status(self, df):
        """Clean data based on status using external calibration module."""
        df_cleaned = calibration.clean_data(df)
        return self.report_removal(df, df_cleaned, "Status cleaning")
    
    def clean_optical_saturation(self, df):
        """Remove optically saturated measurements."""
        df_cleaned = df.loc[((df['IR ATN1'] < 2**10) | (df['IR ATN2'] < 2**10))]
        return self.report_removal(df, df_cleaned, "Optical saturation cleaning")
    
    def clean_extreme_bcc(self, df):
        """Remove extreme BCc values."""
        df_cleaned = df.copy()
        removed_count = 0
        
        for wl in self.wls_to_filter:
            smoothed_col = f'{wl} BCc smoothed'
            atn_col = f'{wl} ATN1'
            
            if smoothed_col in df_cleaned.columns and atn_col in df_cleaned.columns:
                before_count = len(df_cleaned)
                df_cleaned = df_cleaned[~((df_cleaned[smoothed_col] <= -15000) & 
                                        (df_cleaned[atn_col] >= 3))]
                removed_count += before_count - len(df_cleaned)
            elif self.verbose:
                print(f"⚠️ Missing columns for {wl} extreme BCc cleaning: {smoothed_col} or {atn_col}")
        
        return self.report_removal(df, df_cleaned, "Extreme BCc cleaning")
    
    def clean_flow_range(self, df, flow_threshold=0.1, setpoint=100):
        """Clean data based on flow range violations."""
        df_cleaned = df.loc[
            (df['Flow total (mL/min)'] <= setpoint * (1 + flow_threshold)) &
            (df['Flow total (mL/min)'] >= setpoint * (1 - flow_threshold))
        ]
        return self.report_removal(df, df_cleaned, "Flow range cleaning")
    
    def clean_flow_ratio(self, df, lower=1.05, upper=5):
        """Remove abnormal flow ratios."""
        df_cleaned = df[(df['ratio_flow'] >= lower) & (df['ratio_flow'] <= upper)]
        return self.report_removal(df, df_cleaned, "Abnormal flow ratio")
    
    def clean_leak_ratio(self, df, lower_bound=0.1, upper_bound=5):
        """Clean based on leak ratio indicators."""
        df_cleaned = df.copy()
        
        for wl in self.wls_to_filter:
            delta_atn1_col = f'delta {wl} ATN1 rolling mean'
            delta_atn2_col = f'delta {wl} ATN2 rolling mean'
            flow1_col = 'Flow1 (mL/min)'
            flow2_col = 'Flow2 (mL/min)'
            
            required_cols = [delta_atn1_col, delta_atn2_col, flow1_col, flow2_col]
            
            if all(col in df_cleaned.columns for col in required_cols):
                df_cleaned['ratio_dATN_flow'] = (
                    (df_cleaned[delta_atn1_col] / df_cleaned[flow1_col]) /
                    (df_cleaned[delta_atn2_col] / df_cleaned[flow2_col])
                )
                df_cleaned = df_cleaned.loc[
                    (df_cleaned['ratio_dATN_flow'] > lower_bound) & 
                    (df_cleaned['ratio_dATN_flow'] < upper_bound)
                ]
            elif self.verbose:
                missing_cols = [col for col in required_cols if col not in df_cleaned.columns]
                print(f"⚠️ Missing columns for {wl} leak ratio cleaning: {missing_cols}")
        
        return self.report_removal(df, df_cleaned, "Leak ratio cleaning")
    
    def clean_bcc_denominator(self, df, threshold=0.075, threshold_IR=0.1):
        """Clean based on BCc denominator values."""
        df_cleaned = df.copy()
        for wl in self.wls_to_filter:
            current_threshold = threshold_IR if wl == "IR" else threshold
            df_cleaned[f'{wl} BCc denominator'] = 1 - df_cleaned[f'{wl} K'] * df_cleaned[f'{wl} ATN1']
            df_cleaned = df_cleaned[abs(df_cleaned[f'{wl} BCc denominator']) > current_threshold]
        return self.report_removal(df, df_cleaned, "BCc denominator cleaning")
    
    def clean_bcc_ratio(self, df, lower_bound=0.2, upper_bound=5):
        """Clean based on BCc ratio values."""
        df_cleaned = df.copy()
        for wl in self.wls_to_filter:
            df_cleaned[f'{wl} BCc/BC1 smoothed'] = (
                df_cleaned[f'{wl} BCc smoothed'] / df_cleaned[f'{wl} BC1 smoothed']
            )
            df_cleaned[f'{wl} BCc/BC2 smoothed'] = (
                df_cleaned[f'{wl} BCc smoothed'] / df_cleaned[f'{wl} BC2 smoothed']
            )
            df_cleaned[f'{wl} BC1/BC2 smoothed'] = (
                df_cleaned[f'{wl} BC1 smoothed'] / df_cleaned[f'{wl} BC2 smoothed']
            )
            df_cleaned = df_cleaned.loc[
                df_cleaned[f'{wl} BCc/BC1 smoothed'].between(lower_bound, upper_bound) |
                df_cleaned[f'{wl} BCc/BC2 smoothed'].between(lower_bound, upper_bound)
            ]
        return self.report_removal(df, df_cleaned, "BCc ratio cleaning")
    
    def clean_temperature_change(self, df):
        """Clean based on temperature change criteria."""
        df_cleaned = df[abs(df['delta Sample temp (C)']) <= 0.5].copy()
        if self.verbose:
            print("Sharp change", df.shape[0] - df_cleaned.shape[0])
        
        df_cleaned['delta_temp_std'] = df_cleaned['delta Sample temp (C)'].rolling(window=7).std()
        df_cleaned_std = df_cleaned[df_cleaned['delta_temp_std'] < 0.1]
        if self.verbose:
            print("noise", df_cleaned.shape[0] - df_cleaned_std.shape[0])
        
        return self.report_removal(df, df_cleaned_std, "Temperature change cleaning")
    
    def add_roughness_columns(self, df):
        """Add roughness calculation columns."""
        wavelengths = self.wls_to_filter
        spots = [1, 2]
        df_cleaned = df.copy()
        
        for wl in wavelengths:
            for spot in spots:
                delta_col = f'delta {wl} ATN{spot}'
                mean_col = f'{delta_col} rolling mean'
                roughness_col = f'{wl} ATN{spot}_roughness'
                
                if delta_col in df_cleaned.columns and mean_col in df_cleaned.columns:
                    df_cleaned[roughness_col] = (
                        (df_cleaned[delta_col] - df_cleaned[mean_col]).abs()
                        .rolling(60, center=True, min_periods=30)
                        .mean()
                    )
        return df_cleaned
    
    def flag_high_roughness_periods(self, df, z_threshold=2, min_len=10, min_frac_high=2/3):
        """
        Flag periods where roughness is unusually high.
        
        Args:
            df (pd.DataFrame): DataFrame with roughness columns
            z_threshold (float): Z-score threshold for high roughness
            min_len (int): Minimum length of period to flag
            min_frac_high (float): Minimum fraction of high roughness points in period
            
        Returns:
            tuple: (cleaned_df, std_values)
        """
        wavelengths = self.wls_to_filter
        spots = [1, 2]
        df = df.copy()
        df['high_rough_period'] = False
        stds = []
        
        for wl in wavelengths:
            for spot in spots:
                col = f'{wl} ATN{spot}_roughness'
                if col not in df.columns:
                    if self.verbose:
                        print(f"Column {col} not found in DataFrame, skipping.")
                    continue
                
                rough_values = df[col].dropna()
                mean = rough_values.mean()
                std = rough_values.std()
                stds.append(std)
                is_high = df[col] > (mean + z_threshold * std)
                
                # Identify consecutive groups of data
                group_id = (is_high != is_high.shift()).cumsum()
                groups = df.groupby(group_id)
                
                for _, group in groups:
                    idx = group.index
                    if len(group) >= min_len:
                        frac_high = is_high.loc[idx].mean()
                        if frac_high > min_frac_high:
                            df.loc[idx, 'high_rough_period'] = True
                
                if self.verbose:
                    print(f"{col}: threshold={mean + z_threshold * std:.4f}, "
                          f"high periods flagged: {df['high_rough_period'].sum()} rows so far")
        
        df_cleaned = df.loc[df['high_rough_period'] == False].reset_index(drop=True)
        return df_cleaned, stds
    
    def dema_bc_and_atn(self, dataframe, DEMA_span_min=15, wl='IR'):
        """
        Apply Double Exponential Moving Average (DEMA) smoothing.
        
        Args:
            dataframe (pd.DataFrame): Input DataFrame
            DEMA_span_min (int): Span for DEMA calculation in minutes
            wl (str): Wavelength to process
            
        Returns:
            pd.DataFrame: DataFrame with DEMA smoothed columns
        """
        # Check if required columns exist
        required_vars = ['BC1', 'BC2', 'BCc']
        available_vars = []
        
        for vari in required_vars:
            varname = wl + ' ' + vari
            if varname in dataframe.columns:
                available_vars.append(vari)
            elif self.verbose:
                print(f"⚠️ Column {varname} not found, skipping")
        
        if not available_vars:
            if self.verbose:
                print(f"⚠️ No BC columns found for {wl}, skipping DEMA smoothing")
            return dataframe
        
        try:
            df_tp_list = calibration.create_df_list_atlevel_tapeposition(dataframe)
            df_interim_list = []
            
            for dfi in df_tp_list:
                if len(dfi) > 0:
                    for vari in available_vars:
                        varname = wl + ' ' + vari
                        if varname in dfi.columns:
                            if self.verbose and wl in self.wls_to_filter:  # Only print for filtered wavelengths
                                print(f"  Processing {varname}")
                            ema = dfi[varname].ewm(span=DEMA_span_min, adjust=False).mean()
                            ema_of_ema = ema.ewm(span=DEMA_span_min, adjust=False).mean()
                            dfi[varname + ' smoothed'] = 2 * ema - ema_of_ema
                    
                    df_interim_list.append(dfi)
            
            if df_interim_list:
                df_out = pd.concat(df_interim_list).reset_index(drop=True)
                return df_out
            else:
                if self.verbose:
                    print(f"⚠️ No data processed for {wl}")
                return dataframe
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error in DEMA smoothing for {wl}: {e}")
            return dataframe
    
    def preprocess_data(self, df):
        """
        Apply preprocessing steps before cleaning (DEMA smoothing, etc.).
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Preprocessed data ready for cleaning
        """
        df_processed = df.copy()
        
        # Ensure datetime_local is a column
        if 'datetime_local' not in df_processed.columns:
            if df_processed.index.name == 'datetime_local' or hasattr(df_processed.index, 'tz'):
                df_processed = df_processed.reset_index()
                if self.verbose:
                    print("✅ Converted datetime_local from index to column")
        
        # Apply DEMA smoothing for all wavelengths
        if self.verbose:
            print("🔄 Applying DEMA smoothing...")
        
        for wl in ['Blue', 'Green', 'Red', 'UV', 'IR']:
            try:
                df_processed = self.dema_bc_and_atn(df_processed, DEMA_span_min=10, wl=wl)
                if self.verbose and wl in self.wls_to_filter:
                    print(f"✅ DEMA smoothing applied for {wl}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ DEMA smoothing failed for {wl}: {e}")
        
        return df_processed
    
    def clean_pipeline(self, df, skip_preprocessing=False):
        """
        Run the complete cleaning pipeline.
        
        Args:
            df (pd.DataFrame): Raw aethalometer data
            skip_preprocessing (bool): If True, assumes data is already preprocessed
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if self.verbose:
            print("Starting PKL data cleaning pipeline...")
            print("=" * 50)
        
        # Apply preprocessing unless skipped
        if not skip_preprocessing:
            df_cleaned = self.preprocess_data(df)
            # Diagnose data structure after preprocessing
            self.diagnose_data_structure(df_cleaned)
        else:
            df_cleaned = df.copy()
            # Ensure datetime_local is a column
            if 'datetime_local' not in df_cleaned.columns:
                if df_cleaned.index.name == 'datetime_local' or hasattr(df_cleaned.index, 'tz'):
                    df_cleaned = df_cleaned.reset_index()
                    if self.verbose:
                        print("✅ Converted datetime_local from index to column")
            # Diagnose data structure
            self.diagnose_data_structure(df_cleaned)
        
        if self.verbose:
            print("\n🧹 Starting cleaning steps...")
        
        # Step 1: Status cleaning
        df_cleaned = self.clean_by_status(df_cleaned)
        
        # Step 2: Extreme BCc cleaning
        df_cleaned = self.clean_extreme_bcc(df_cleaned)
        
        # Step 3: Flow range cleaning
        df_cleaned = self.clean_flow_range(df_cleaned)
        
        # Step 4: Flow ratio cleaning
        df_cleaned = self.clean_flow_ratio(df_cleaned)
        
        # Step 5: Leak ratio cleaning
        df_cleaned = self.clean_leak_ratio(df_cleaned)
        
        # Step 6: BCc denominator cleaning
        df_cleaned = self.clean_bcc_denominator(df_cleaned)
        
        # Step 7: Temperature change cleaning
        df_cleaned = self.clean_temperature_change(df_cleaned)
        
        # Step 8: Roughness-based cleaning
        df_cleaned = self.add_roughness_columns(df_cleaned)
        df_cleaned, stds = self.flag_high_roughness_periods(df_cleaned)
        
        if self.verbose:
            print("=" * 50)
            print(f"Cleaning complete! Final data shape: {df_cleaned.shape}")
        
        return df_cleaned


def load_and_clean_pkl_data(directory_path="../JPL_aeth/", 
                           wavelengths_to_filter=None,
                           verbose=True, 
                           summary=True, 
                           **kwargs):
    """
    Load PKL data from directory and apply comprehensive cleaning pipeline.
    
    Args:
        directory_path (str): Path to data directory
        wavelengths_to_filter (list): Wavelengths to process
        verbose (bool): Whether to print progress
        summary (bool): Whether to print summary
        **kwargs: Additional parameters for calibration.readall_BCdata_from_dir
        
    Returns:
        pd.DataFrame: Cleaned aethalometer data
    """
    # Default parameters for data loading
    default_params = {
        'sep': ',',
        'mult_folders_in_dir': True,
        'verbose': False,
        'summary': False,
        'AE51_devices_only': False,
        'file_number_printout': True,
        'output_pax_averaged_by_minute': True,
        'PAX_correction': False,
        'inter_device_corr': True,
        'assign_testid_from_startdate': True,
        'assign_testid_from_filename': False,
        'num_mins_datetime_round': 1,
        'group_Session_Ids_together': False,
        'datetime_fixme_dict': {},
        'assign_unit_category_from_dirname': False,
        'test_batch_indication': False,
        'allow_ALx_files': True,
        'files_to_exclude': [],
        'output_first_datapoint_of_each_file_separately': False,
        'create_session_ids': False,
        'process_api_formatted_files': True,
    }
    
    # Update with user-provided parameters
    default_params.update(kwargs)
    
    if verbose:
        print("Loading PKL data from directory...")
    
    # Load data using external calibration module
    df = calibration.readall_BCdata_from_dir(
        directory_path=directory_path,
        **default_params
    )
    
    if verbose:
        print('Dropping duplicates...')
    
    # Drop MA duplicates based on Serial number and Datum ID
    df_ma = df.loc[df['Serial number'].str.contains("MA")].drop_duplicates(
        subset=['Serial number', 'Datum ID'], keep='first'
    )
    
    # Drop duplicates in other instruments based on Serial number and datetime_local
    df_others = df.loc[(df['Serial number'].str.contains("MA") == False)].drop_duplicates(
        subset=['Serial number', 'datetime_local'], keep='first'
    )
    
    # Concatenate the two duplicates-dropped dataframes
    df = pd.concat([df_ma, df_others], ignore_index=True)
    
    # Sort by serial number and datetime
    df = df.sort_values(by=['Serial number', 'datetime_local']).reset_index(drop=True)
    
    # Convert to datetime object with timezone (if not already done by modular loader)
    if 'datetime_local' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['datetime_local']):
            df['datetime_local'] = pd.to_datetime(
                df['datetime_local'], utc=True
            ).dt.tz_convert('Africa/Addis_Ababa')
    elif df.index.name == 'datetime_local':
        # If datetime_local is in index, reset it to column
        df = df.reset_index()
        if not pd.api.types.is_datetime64_any_dtype(df['datetime_local']):
            df['datetime_local'] = pd.to_datetime(
                df['datetime_local'], utc=True
            ).dt.tz_convert('Africa/Addis_Ababa')
    
    # Convert data types and add deltas
    df = calibration.convert_to_float(df)
    df = calibration.add_deltas(df)
    
    # Add session ID manually if using API data
    position_change = df['Tape position'] != df['Tape position'].shift()
    df['Session ID'] = position_change.cumsum()
    
    # Apply preprocessing and cleaning with cleaner
    cleaner = PKLDataCleaner(wavelengths_to_filter=wavelengths_to_filter, verbose=verbose)
    
    # Apply preprocessing (DEMA smoothing, etc.)
    df = cleaner.preprocess_data(df)
    
    # Set serial number and filter by year
    df['Serial number'] = "MA350-0238"
    df = df.loc[df['datetime_local'].dt.year >= 2022]
    
    # Apply cleaning pipeline (skip preprocessing since we already did it)
    df_cleaned = cleaner.clean_pipeline(df, skip_preprocessing=True)
    
    if summary:
        # Get datetime column for summary
        if 'datetime_local' in df_cleaned.columns:
            datetime_col = df_cleaned['datetime_local']
        elif df_cleaned.index.name == 'datetime_local':
            datetime_col = df_cleaned.index
        else:
            datetime_col = None
            
        print("\nPKL Data Loading and Cleaning Summary:")
        print("=" * 50)
        print(f"Final cleaned data shape: {df_cleaned.shape}")
        
        if datetime_col is not None:
            print(f"Date range: {datetime_col.min()} to {datetime_col.max()}")
        
        if 'Serial number' in df_cleaned.columns:
            print(f"Serial numbers: {df_cleaned['Serial number'].unique()}")
        else:
            print("Serial number information not available")
    
    return df_cleaned


def table_removed_datapoints_by_month(df, df_cleaned):
    """
    Calculate the number of removed datapoints by month.
    
    Args:
        df (pd.DataFrame): Original DataFrame
        df_cleaned (pd.DataFrame): Cleaned DataFrame
        
    Returns:
        pd.DataFrame: Monthly removal statistics
    """
    # Helper function to get datetime column
    def get_datetime_column(dataframe):
        if 'datetime_local' in dataframe.columns:
            return dataframe['datetime_local']
        elif dataframe.index.name == 'datetime_local':
            return dataframe.index
        elif hasattr(dataframe.index, 'tz'):  # DatetimeIndex
            return dataframe.index
        else:
            raise ValueError("Cannot find datetime column in DataFrame")
    
    # Get datetime columns
    df_datetime = get_datetime_column(df)
    df_cleaned_datetime = get_datetime_column(df_cleaned)
    
    # Ensure datetime type
    df_datetime = pd.to_datetime(df_datetime)
    df_cleaned_datetime = pd.to_datetime(df_cleaned_datetime)
    
    # Create temporary DataFrames with month columns
    df_temp = df.copy()
    df_cleaned_temp = df_cleaned.copy()
    
    df_temp['month'] = df_datetime.dt.to_period('M')
    df_cleaned_temp['month'] = df_cleaned_datetime.dt.to_period('M')
    
    # Count entries per month
    original_counts = df_temp['month'].value_counts().sort_index()
    cleaned_counts = df_cleaned_temp['month'].value_counts().sort_index()
    
    # Calculate the difference
    removed_counts = original_counts - cleaned_counts
    
    # Convert to DataFrame for readability
    removed_df = removed_counts.reset_index()
    removed_df.columns = ['month', 'removed_datapoints']
    return removed_df


# Enhanced processing integration
def load_and_clean_pkl_data_enhanced(pkl_file_path: str, 
                                   wavelengths_to_filter: Optional[List[str]] = None,
                                   export_path: Optional[str] = None,
                                   verbose: bool = True,
                                   **kwargs) -> pd.DataFrame:
    """
    Enhanced PKL data loading and cleaning with comprehensive preprocessing.
    
    This function combines the working notebook pipeline with the modular structure:
    1. Loads PKL data
    2. Applies comprehensive preprocessing (datetime, columns, types, sessions, deltas)
    3. Applies DEMA smoothing
    4. Runs quality control cleaning
    5. Optionally exports results
    
    Args:
        pkl_file_path (str): Path to PKL file
        wavelengths_to_filter (List[str]): Wavelengths to focus on (default: ['IR', 'Blue'])
        export_path (str, optional): Base path for export (without extension)
        verbose (bool): Enable verbose output
        **kwargs: Additional arguments for PKLDataCleaner
        
    Returns:
        pd.DataFrame: Fully processed and cleaned data
        
    Example:
        # Simple usage
        df_cleaned = load_and_clean_pkl_data_enhanced(
            'path/to/data.pkl',
            wavelengths_to_filter=['IR', 'Blue'],
            export_path='cleaned_data',
            verbose=True
        )
    """
    from .enhanced_pkl_processing import EnhancedPKLProcessor
    
    if verbose:
        print(f"📁 Loading PKL data from: {pkl_file_path}")
    
    # Load the data
    try:
        df_raw = pd.read_pickle(pkl_file_path)
        if verbose:
            print(f"✅ Loaded {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    except Exception as e:
        raise FileNotFoundError(f"Could not load PKL file: {e}")
    
    # Process with enhanced pipeline
    processor = EnhancedPKLProcessor(
        wavelengths_to_filter=wavelengths_to_filter or ['IR', 'Blue'],
        verbose=verbose,
        **kwargs
    )
    
    df_cleaned = processor.process_pkl_data(df_raw, export_path=export_path)
    
    return df_cleaned


def create_enhanced_pkl_cleaner(wavelengths_to_filter: Optional[List[str]] = None,
                               verbose: bool = True,
                               **kwargs) -> 'EnhancedPKLProcessor':
    """
    Factory function to create an enhanced PKL processor.
    
    Args:
        wavelengths_to_filter (List[str]): Wavelengths to focus on
        verbose (bool): Enable verbose output
        **kwargs: Additional arguments for PKLDataCleaner
        
    Returns:
        EnhancedPKLProcessor: Configured processor instance
    """
    from .enhanced_pkl_processing import EnhancedPKLProcessor
    
    return EnhancedPKLProcessor(
        wavelengths_to_filter=wavelengths_to_filter or ['IR', 'Blue'],
        verbose=verbose,
        **kwargs
    )


def compare_cleaning_methods(df: pd.DataFrame, 
                           wavelengths: Optional[List[str]] = None,
                           verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Compare the original PKL cleaning method with the enhanced method.
    
    Args:
        df (pd.DataFrame): Raw PKL data
        wavelengths (List[str]): Wavelengths to process
        verbose (bool): Show comparison details
        
    Returns:
        Dict[str, pd.DataFrame]: Results from both methods
    """
    wavelengths = wavelengths or ['IR', 'Blue']
    
    if verbose:
        print("🔄 Comparing PKL cleaning methods...")
        print("=" * 60)
    
    results = {}
    
    # Method 1: Original PKL cleaning
    if verbose:
        print("\n📊 Method 1: Original PKL Cleaning")
        print("-" * 40)
    
    try:
        original_cleaner = PKLDataCleaner(wavelengths_to_filter=wavelengths, verbose=verbose)
        df_original = original_cleaner.clean_pipeline(df)
        results['original'] = df_original
        
        if verbose:
            print(f"✅ Original method: {df_original.shape}")
    except Exception as e:
        if verbose:
            print(f"❌ Original method failed: {e}")
        results['original'] = None
    
    # Method 2: Enhanced PKL cleaning
    if verbose:
        print("\n📊 Method 2: Enhanced PKL Cleaning")
        print("-" * 40)
    
    try:
        from .enhanced_pkl_processing import EnhancedPKLProcessor
        enhanced_processor = EnhancedPKLProcessor(wavelengths_to_filter=wavelengths, verbose=verbose)
        df_enhanced = enhanced_processor.process_pkl_data(df)
        results['enhanced'] = df_enhanced
        
        if verbose:
            print(f"✅ Enhanced method: {df_enhanced.shape}")
    except Exception as e:
        if verbose:
            print(f"❌ Enhanced method failed: {e}")
        results['enhanced'] = None
    
    # Comparison summary
    if verbose and all(v is not None for v in results.values()):
        print("\n📊 Comparison Summary:")
        print("=" * 60)
        original_size = len(results['original'])
        enhanced_size = len(results['enhanced'])
        
        print(f"Original method:  {original_size:,} rows")
        print(f"Enhanced method:  {enhanced_size:,} rows")
        print(f"Difference:       {enhanced_size - original_size:,} rows")
        
        if original_size > 0:
            pct_diff = ((enhanced_size - original_size) / original_size) * 100
            print(f"Percentage diff:  {pct_diff:+.2f}%")
        
        # Column comparison
        orig_cols = set(results['original'].columns)
        enh_cols = set(results['enhanced'].columns)
        
        new_cols = enh_cols - orig_cols
        removed_cols = orig_cols - enh_cols
        
        if new_cols:
            print(f"New columns:      {len(new_cols)} (e.g., {list(new_cols)[:3]})")
        if removed_cols:
            print(f"Removed columns:  {len(removed_cols)} (e.g., {list(removed_cols)[:3]})")
    
    return results