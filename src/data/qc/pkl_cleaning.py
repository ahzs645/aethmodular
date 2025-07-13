# ========================================================================
# PKL Data Cleaning Pipeline
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
    
    def __init__(self, data_directory=None, wavelengths_to_filter=None):
        """
        Initialize the PKL data cleaner.
        
        Args:
            data_directory (str): Path to the directory containing PKL data files.
                                Defaults to "../JPL_aeth/" if not specified.
            wavelengths_to_filter (list): List of wavelengths to process.
                                        Defaults to ['IR', 'Blue'].
        """
        self.data_directory = data_directory or "../JPL_aeth/"
        self.wls_to_filter = wavelengths_to_filter or DEFAULT_WAVELENGTHS
    
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
        print(f"{label}: Removed {n_removed} rows ({n_removed / len(df_before) * 100:.2f}%)")
        return df_after

    def clean_by_status(self, df):
        """
        Clean data based on instrument status using external calibration module.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_cleaned = calibration.clean_data(df)
        return self.report_removal(df, df_cleaned, "Status cleaning")

    def clean_optical_saturation(self, df):
        """
        Remove data points with optical saturation.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_cleaned = df.loc[((df['IR ATN1'] < 2**10) | (df['IR ATN2'] < 2**10))]
        return self.report_removal(df, df_cleaned, "Optical saturation cleaning")

    def clean_extreme_bcc(self, df):
        """
        Remove extreme black carbon corrected (BCc) values.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_cleaned = df.copy()
        for wl in self.wls_to_filter:
            df_cleaned = df_cleaned[~((df_cleaned[f'{wl} BCc smoothed'] <= -15000) & 
                                    (df_cleaned[f'{wl} ATN1'] >= 3))]
        return self.report_removal(df, df_cleaned, "Extreme BCc cleaning")

    def clean_flow_range(self, df, flow_threshold=0.1, setpoint=100):
        """
        Clean data based on flow rate range.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            flow_threshold (float): Relative threshold for flow validation
            setpoint (float): Target flow rate setpoint
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_cleaned = df.loc[
            (df['Flow total (mL/min)'] <= setpoint * (1 + flow_threshold)) &
            (df['Flow total (mL/min)'] >= setpoint * (1 - flow_threshold))
        ]
        return self.report_removal(df, df_cleaned, "Flow range cleaning")

    def clean_flow_ratio(self, df, lower=1.05, upper=5):
        """
        Clean data based on flow ratio between channels.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            lower (float): Lower bound for flow ratio
            upper (float): Upper bound for flow ratio
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_cleaned = df[(df['ratio_flow'] >= lower) & (df['ratio_flow'] <= upper)]
        return self.report_removal(df, df_cleaned, "Abnormal flow ratio")

    def clean_leak_ratio(self, df, lower_bound=0.1, upper_bound=5):
        """
        Clean data based on leak detection through ATN/flow ratios.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            lower_bound (float): Lower bound for leak ratio
            upper_bound (float): Upper bound for leak ratio
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_cleaned = df.copy()
        for wl in self.wls_to_filter:
            df_cleaned['ratio_dATN_flow'] = (
                (df_cleaned[f'delta {wl} ATN1 rolling mean'] / df_cleaned['Flow1 (mL/min)']) /
                (df_cleaned[f'delta {wl} ATN2 rolling mean'] / df_cleaned['Flow2 (mL/min)'])
            )
            df_cleaned = df_cleaned.loc[
                (df_cleaned['ratio_dATN_flow'] > lower_bound) & 
                (df_cleaned['ratio_dATN_flow'] < upper_bound)
            ]
        return self.report_removal(df, df_cleaned, "Leak ratio cleaning")

    def clean_bcc_denominator(self, df, threshold=0.075, threshold_IR=0.1):
        """
        Clean data based on BCc denominator values.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            threshold (float): General threshold for BCc denominator
            threshold_IR (float): Specific threshold for IR wavelength
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_cleaned = df.copy()
        for wl in self.wls_to_filter:
            current_threshold = threshold_IR if wl == "IR" else threshold
            df_cleaned[f'{wl} BCc denominator'] = 1 - df_cleaned[f'{wl} K'] * df_cleaned[f'{wl} ATN1']
            df_cleaned = df_cleaned[abs(df_cleaned[f'{wl} BCc denominator']) > current_threshold]
        return self.report_removal(df, df_cleaned, "BCc denominator cleaning")

    def clean_bcc_ratio(self, df, lower_bound=0.2, upper_bound=5):
        """
        Clean data based on BCc ratio validation.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            lower_bound (float): Lower bound for BCc ratios
            upper_bound (float): Upper bound for BCc ratios
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_cleaned = df.copy()
        for wl in self.wls_to_filter:
            df_cleaned[f'{wl} BCc/BC1 smoothed'] = (df_cleaned[f'{wl} BCc smoothed'] / 
                                                   df_cleaned[f'{wl} BC1 smoothed'])
            df_cleaned[f'{wl} BCc/BC2 smoothed'] = (df_cleaned[f'{wl} BCc smoothed'] / 
                                                   df_cleaned[f'{wl} BC2 smoothed'])
            df_cleaned[f'{wl} BC1/BC2 smoothed'] = (df_cleaned[f'{wl} BC1 smoothed'] / 
                                                   df_cleaned[f'{wl} BC2 smoothed'])
            df_cleaned = df_cleaned.loc[
                df_cleaned[f'{wl} BCc/BC1 smoothed'].between(lower_bound, upper_bound) |
                df_cleaned[f'{wl} BCc/BC2 smoothed'].between(lower_bound, upper_bound)
            ]
        return self.report_removal(df, df_cleaned, "BCc ratio cleaning")

    def clean_temperature_change(self, df):
        """
        Clean data based on temperature change patterns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_cleaned = df[abs(df['delta Sample temp (C)']) <= 0.5].copy()
        print("Sharp change", df.shape[0] - df_cleaned.shape[0])
        
        df_cleaned['delta_temp_std'] = df_cleaned['delta Sample temp (C)'].rolling(window=7).std()
        df_cleaned_std = df_cleaned[df_cleaned['delta_temp_std'] < 0.1]
        print("noise", df_cleaned.shape[0] - df_cleaned_std.shape[0])
        
        return self.report_removal(df, df_cleaned_std, "Temperature change cleaning")

    def add_roughness_columns(self, df):
        """
        Add roughness calculation columns for ATN measurements.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with added roughness columns
        """
        wavelengths = self.wls_to_filter
        spots = [1, 2]
        df_cleaned = df.copy()

        for wl in wavelengths:
            for spot in spots:
                delta_col = f'delta {wl} ATN{spot}'
                mean_col = f'{delta_col} rolling mean'
                roughness_col = f'{wl} ATN{spot}_roughness'

                df_cleaned[roughness_col] = (
                    (df_cleaned[delta_col] - df_cleaned[mean_col]).abs()
                    .rolling(60, center=True, min_periods=30)
                    .mean()
                )
        return df_cleaned

    def clean_roughness(self, df, stds, z_threshold=3):
        """
        Clean data based on roughness threshold using provided standard deviations.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            stds (list): List of standard deviations for each roughness column
            z_threshold (float): Z-score threshold for roughness filtering
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        wavelengths = self.wls_to_filter
        spots = [1, 2]
        high_rough_mask = pd.Series(False, index=df.index)
        i = 0
        
        for wl in wavelengths:
            for spot in spots:
                rough_col = f'{wl} ATN{spot}_roughness'
                if rough_col not in df.columns:
                    print(f"Column {rough_col} not found in DataFrame, skipping.")
                    continue

                rough_values = df[rough_col].dropna()
                mean = rough_values.mean()
                std = stds[i]
                i += 1

                is_high = df[rough_col] > (mean + z_threshold * std)
                high_rough_mask = high_rough_mask | is_high

                print(f"{rough_col}: mean={mean:.4f}, std={std:.4f}, "
                      f"high rough threshold={mean + z_threshold * std:.4f}, count={is_high.sum()}")

        df_high_rough = df[high_rough_mask].reset_index(drop=True)
        print(f"Total rows with HIGH roughness points: {len(df_high_rough)}")
        return df[~high_rough_mask].reset_index(drop=True)

    def flag_high_roughness_periods(self, df, z_threshold=2, min_len=10, min_frac_high=2/3):
        """
        Flag periods where roughness is unusually high in at least one roughness column.
        
        A period is defined as a sequence of consecutive rows where:
        - len(period) >= min_len
        - more than min_frac_high of rows have roughness > threshold

        Args:
            df (pd.DataFrame): Input DataFrame
            z_threshold (float): Z-score threshold for roughness
            min_len (int): Minimum length of high roughness period
            min_frac_high (float): Minimum fraction of high roughness points in period
        
        Returns:
            tuple: (cleaned DataFrame, list of standard deviations)
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
                            
                print(f"{col}: threshold={mean + z_threshold * std:.4f}, "
                      f"high periods flagged: {df['high_rough_period'].sum()} rows so far")
        
        df = df.loc[df['high_rough_period'] == False].reset_index(drop=True)
        return df, stds

    def dema_bc_and_atn(self, dataframe, DEMA_span_min=15, wl='IR'):
        """
        Apply Double Exponential Moving Average (DEMA) smoothing to BC and ATN data.
        
        Args:
            dataframe (pd.DataFrame): Input DataFrame
            DEMA_span_min (int): Span for DEMA calculation in minutes
            wl (str): Wavelength to process
        
        Returns:
            pd.DataFrame: DataFrame with DEMA smoothed columns
        """
        df_tp_list = calibration.create_df_list_atlevel_tapeposition(dataframe)

        df_interim_list = []
        for dfi in df_tp_list:
            if len(dfi) > 0:            
                for vari in ['BC1', 'BC2', 'BCc']:
                    varname = wl + ' ' + vari
                    print(varname)
                    ema = dfi[varname].ewm(span=DEMA_span_min, adjust=False).mean()
                    ema_of_ema = ema.ewm(span=DEMA_span_min, adjust=False).mean()
                    dfi[varname + ' smoothed'] = 2 * ema - ema_of_ema

                df_interim_list.append(dfi)

        df_out = pd.concat(df_interim_list).reset_index(drop=True)
        return df_out

    def clean_pipeline(self, df):
        """
        Execute the complete PKL data cleaning pipeline.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: Fully cleaned DataFrame
        """
        print("Starting PKL data cleaning pipeline...")
        
        # Status cleaning
        df_cleaned = self.clean_by_status(df)
        
        # Extreme BCc cleaning
        df_cleaned = self.clean_extreme_bcc(df_cleaned)
        
        # Flow range cleaning
        df_cleaned = self.clean_flow_range(df_cleaned)
        
        # Flow ratio cleaning
        df_cleaned = self.clean_flow_ratio(df_cleaned)
        
        # Leak ratio cleaning
        df_cleaned = self.clean_leak_ratio(df_cleaned)
        
        # BCc denominator cleaning
        df_cleaned = self.clean_bcc_denominator(df_cleaned)
        
        # Temperature change cleaning
        df_cleaned = self.clean_temperature_change(df_cleaned)
        
        # Roughness-based cleaning
        df_cleaned = self.add_roughness_columns(df_cleaned)
        df_cleaned, stds = self.flag_high_roughness_periods(
            df_cleaned, z_threshold=2, min_len=10, min_frac_high=2/3
        )
        
        print("PKL data cleaning pipeline completed.")
        return df_cleaned
    
    def load_and_clean_data(self, **kwargs):
        """
        Load and clean PKL data using the instance's configured data directory.
        
        Args:
            **kwargs: Additional arguments for calibration.readall_BCdata_from_dir
        
        Returns:
            pd.DataFrame: Cleaned DataFrame ready for analysis
        """
        return load_and_clean_pkl_data(directory_path=self.data_directory, **kwargs)


def table_removed_datapoints_by_month(df, df_cleaned_status):
    """
    Calculate the number of removed datapoints by month from the original 
    DataFrame and the cleaned DataFrame.
    
    Args:
        df (pd.DataFrame): The original DataFrame with datetime_local column
        df_cleaned_status (pd.DataFrame): The cleaned DataFrame with datetime_local column
    
    Returns:
        pd.DataFrame: A DataFrame with two columns: 'month' and 'removed_datapoints'
    """
    # Ensure datetime_local is datetime type
    df['datetime_local'] = pd.to_datetime(df['datetime_local'])
    df_cleaned_status['datetime_local'] = pd.to_datetime(df_cleaned_status['datetime_local'])

    # Add a 'month' column to both DataFrames
    df['month'] = df['datetime_local'].dt.to_period('M')
    df_cleaned_status['month'] = df_cleaned_status['datetime_local'].dt.to_period('M')

    # Count entries per month
    original_counts = df['month'].value_counts().sort_index()
    cleaned_counts = df_cleaned_status['month'].value_counts().sort_index()

    # Calculate the difference
    removed_counts = original_counts - cleaned_counts

    # Convert to DataFrame for readability
    removed_df = removed_counts.reset_index()
    removed_df.columns = ['month', 'removed_datapoints']
    return removed_df


def load_and_clean_pkl_data(directory_path=None, **kwargs):
    """
    Complete workflow to load and clean PKL data.
    
    Args:
        directory_path (str): Path to the data directory. 
                            Defaults to "../JPL_aeth/" if not specified.
        **kwargs: Additional arguments for calibration.readall_BCdata_from_dir
    
    Returns:
        pd.DataFrame: Cleaned DataFrame ready for analysis
    """
    if directory_path is None:
        directory_path = "../JPL_aeth/"
    
    print(f"Loading PKL data from: {directory_path}")
    
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
    
    # Load data using external calibration module
    df = calibration.readall_BCdata_from_dir(
        directory_path=directory_path,
        **default_params
    )
    
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
    
    # Convert to datetime object with timezone
    df['datetime_local'] = pd.to_datetime(
        df['datetime_local'], utc=True
    ).dt.tz_convert('Africa/Addis_Ababa')
    
    # Convert data types and add deltas
    df = calibration.convert_to_float(df)
    df = calibration.add_deltas(df)
    
    # Add session ID manually if using API data
    position_change = df['Tape position'] != df['Tape position'].shift()
    df['Session ID'] = position_change.cumsum()
    
    # Apply DEMA smoothing for all wavelengths
    cleaner = PKLDataCleaner()
    for wl in ['Blue', 'Green', 'Red', 'UV', 'IR']:
        df = cleaner.dema_bc_and_atn(df, DEMA_span_min=10, wl=wl)
    
    # Set serial number and filter by year
    df['Serial number'] = "MA350-0238"
    df = df.loc[df['datetime_local'].dt.year >= 2022]
    
    # Apply cleaning pipeline
    df_cleaned = cleaner.clean_pipeline(df)
    
    print("PKL data loading and cleaning completed.")
    return df_cleaned
