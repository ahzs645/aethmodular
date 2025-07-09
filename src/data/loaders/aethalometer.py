"""
Aethalometer data loader for .pkl files with JPL repository format support
"""

import pandas as pd
import pickle
from typing import Optional, Dict, Any, List
from pathlib import Path
from ...core.base import BaseLoader
from ...core.exceptions import DataValidationError

class AethalometerPKLLoader(BaseLoader):
    """
    Loader for aethalometer data stored in .pkl files
    Supports both standard aethalometer format and JPL repository format
    """
    
    # Column mapping from standard aethalometer format to JPL format
    COLUMN_MAPPING = {
        # Black carbon concentrations
        'IR BCc': 'IR.BCc',
        'Blue BCc': 'Blue.BCc', 
        'Green BCc': 'Green.BCc',
        'Red BCc': 'Red.BCc',
        'UV BCc': 'UV.BCc',
        
        # Source apportioned BC
        'Biomass BCc': 'Biomass.BCc',
        'Fossil fuel BCc': 'Fossil.fuel.BCc',
        
        # Optical properties
        'AAE calculated': 'AAE.calculated',
        'BB percent': 'BB.percent',
        'Delta-C': 'Delta.C',
        'AAE biomass': 'AAE.biomass',
        'AAE fossil fuel': 'AAE.fossil.fuel',
        
        # Absorption coefficients
        'b_abs_1': 'b.abs.1',
        'b_abs_2': 'b.abs.2',
        
        # Flow and operational parameters
        'Flow total (mL/min)': 'Flow.total.mL.min',
        'Timebase (s)': 'Timebase.s',
        
        # Attenuation values
        'IR ATN1': 'IR.ATN1',
        'IR ATN2': 'IR.ATN2',
        'Blue ATN1': 'Blue.ATN1',
        'Blue ATN2': 'Blue.ATN2',
        'Green ATN1': 'Green.ATN1',
        'Green ATN2': 'Green.ATN2',
        'Red ATN1': 'Red.ATN1',
        'Red ATN2': 'Red.ATN2',
        'UV ATN1': 'UV.ATN1',
        'UV ATN2': 'UV.ATN2',
    }
    
    # Reverse mapping for JPL to standard format
    REVERSE_COLUMN_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}
    
    def __init__(self, pkl_path: str, format_type: str = "auto"):
        """
        Initialize the loader
        
        Parameters:
        -----------
        pkl_path : str
            Path to .pkl file containing aethalometer data
        format_type : str
            Format type: 'standard', 'jpl', or 'auto' (default)
            'auto' will attempt to detect the format
        """
        self.pkl_path = Path(pkl_path)
        if not self.pkl_path.exists():
            raise DataValidationError(f"PKL file not found: {pkl_path}")
        
        self.format_type = format_type.lower()
        if self.format_type not in ['standard', 'jpl', 'auto']:
            raise DataValidationError("format_type must be 'standard', 'jpl', or 'auto'")
    
    def load(self, site_filter: Optional[str] = None, 
             convert_to_jpl: bool = False) -> pd.DataFrame:
        """
        Load aethalometer data from pkl file
        
        Parameters:
        -----------
        site_filter : str, optional
            Filter data by site code/name (if 'site' column exists)
        convert_to_jpl : bool
            If True, convert column names to JPL format
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with aethalometer measurements
        """
        try:
            # Load the pickle file
            with open(self.pkl_path, 'rb') as f:
                df = pickle.load(f)
            
            if not isinstance(df, pd.DataFrame):
                raise DataValidationError("PKL file does not contain a pandas DataFrame")
            
            # Detect format if auto
            if self.format_type == 'auto':
                detected_format = self._detect_format(df)
                print(f"Detected format: {detected_format}")
            else:
                detected_format = self.format_type
            
            # Filter by site if requested
            if site_filter:
                df = self._filter_by_site(df, site_filter)
            
            # Convert datetime column if it exists
            df = self._process_datetime(df)
            
            # Convert column names if requested
            if convert_to_jpl and detected_format == 'standard':
                df = self._convert_to_jpl_format(df)
            elif not convert_to_jpl and detected_format == 'jpl':
                df = self._convert_to_standard_format(df)
            
            # Validate essential columns exist
            self._validate_essential_columns(df, 'jpl' if convert_to_jpl else detected_format)
            
            return df
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            raise DataValidationError(f"Error loading PKL file: {e}")
    
    def _detect_format(self, df: pd.DataFrame) -> str:
        """
        Detect whether the DataFrame uses standard or JPL format
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        str
            'standard' or 'jpl'
        """
        jpl_indicators = ['IR.BCc', 'Biomass.BCc', 'Fossil.fuel.BCc']
        standard_indicators = ['IR BCc', 'Biomass BCc', 'Fossil fuel BCc']
        
        jpl_score = sum(1 for col in jpl_indicators if col in df.columns)
        standard_score = sum(1 for col in standard_indicators if col in df.columns)
        
        if jpl_score > standard_score:
            return 'jpl'
        elif standard_score > 0:
            return 'standard'
        else:
            # Default to standard if no clear indicators
            return 'standard'
    
    def _filter_by_site(self, df: pd.DataFrame, site_filter: str) -> pd.DataFrame:
        """Filter DataFrame by site"""
        if 'site' in df.columns:
            return df[df['site'] == site_filter].copy()
        elif 'Site' in df.columns:
            return df[df['Site'] == site_filter].copy()
        else:
            print(f"Warning: No 'site' column found. Available columns: {list(df.columns)}")
            return df
    
    def _process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process datetime columns"""
        datetime_cols = ['datetime_local', 'datetime_utc', 'Date', 'date']
        
        for col in datetime_cols:
            if col in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
        
        return df
    
    def _convert_to_jpl_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert standard aethalometer column names to JPL format"""
        df_converted = df.copy()
        
        # Rename columns using mapping
        rename_dict = {}
        for standard_col, jpl_col in self.COLUMN_MAPPING.items():
            if standard_col in df_converted.columns:
                rename_dict[standard_col] = jpl_col
        
        if rename_dict:
            df_converted = df_converted.rename(columns=rename_dict)
            print(f"Converted {len(rename_dict)} columns to JPL format")
        
        return df_converted
    
    def _convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert JPL column names to standard aethalometer format"""
        df_converted = df.copy()
        
        # Rename columns using reverse mapping
        rename_dict = {}
        for jpl_col, standard_col in self.REVERSE_COLUMN_MAPPING.items():
            if jpl_col in df_converted.columns:
                rename_dict[jpl_col] = standard_col
        
        if rename_dict:
            df_converted = df_converted.rename(columns=rename_dict)
            print(f"Converted {len(rename_dict)} columns to standard format")
        
        return df_converted
    
    def _validate_essential_columns(self, df: pd.DataFrame, format_type: str):
        """Validate that essential columns exist"""
        if format_type == 'jpl':
            required_cols = ['IR.BCc']
            recommended_cols = ['datetime_local', 'Biomass.BCc', 'Fossil.fuel.BCc']
        else:
            required_cols = ['IR BCc']  
            recommended_cols = ['datetime_local', 'Biomass BCc', 'Fossil fuel BCc']
        
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise DataValidationError(f"Missing required columns: {missing_required}")
        
        missing_recommended = [col for col in recommended_cols if col not in df.columns]
        if missing_recommended:
            print(f"Warning: Missing recommended columns: {missing_recommended}")
    
    def get_available_sites(self) -> List[str]:
        """
        Get list of available sites in the data
        
        Returns:
        --------
        List[str]
            List of available site codes/names
        """
        try:
            with open(self.pkl_path, 'rb') as f:
                df = pickle.load(f)
            
            if 'site' in df.columns:
                return df['site'].unique().tolist()
            elif 'Site' in df.columns:
                return df['Site'].unique().tolist()
            else:
                return ["No site column found"]
                
        except Exception as e:
            return [f"Error reading file: {e}"]
    
    def get_data_summary(self, site_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of available data
        
        Parameters:
        -----------
        site_filter : str, optional
            Site code to summarize (if None, summarize all data)
            
        Returns:
        --------
        Dict[str, Any]
            Summary statistics
        """
        try:
            df = self.load(site_filter=site_filter)
            
            # Detect format
            format_type = self._detect_format(df)
            
            # Get datetime range
            datetime_col = None
            for col in ['datetime_local', 'datetime_utc', 'Date', 'date']:
                if col in df.columns:
                    datetime_col = col
                    break
            
            summary = {
                'total_samples': len(df),
                'format_type': format_type,
                'columns': list(df.columns),
                'file_path': str(self.pkl_path)
            }
            
            if datetime_col:
                summary.update({
                    'earliest_date': df[datetime_col].min(),
                    'latest_date': df[datetime_col].max(),
                    'datetime_column': datetime_col
                })
            
            # Count non-null values for key BC columns
            bc_cols = [col for col in df.columns if 'BCc' in col or 'BC1' in col]
            if bc_cols:
                summary['bc_data_availability'] = {
                    col: df[col].notna().sum() for col in bc_cols[:5]  # Limit to first 5
                }
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}


class AethalometerCSVLoader(BaseLoader):
    """
    Loader for aethalometer data from CSV files (for comparison/migration)
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise DataValidationError(f"CSV file not found: {csv_path}")
    
    def load(self, **kwargs) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(self.csv_path)
            
            # Convert datetime if exists
            if 'datetime_local' in df.columns:
                df['datetime_local'] = pd.to_datetime(df['datetime_local'])
            
            return df
            
        except Exception as e:
            raise DataValidationError(f"Error loading CSV file: {e}")


# Example usage function
def load_aethalometer_data(file_path: str, 
                          site_filter: Optional[str] = None,
                          output_format: str = 'jpl') -> pd.DataFrame:
    """
    Convenience function to load aethalometer data
    
    Parameters:
    -----------
    file_path : str
        Path to .pkl or .csv file
    site_filter : str, optional
        Filter by site name
    output_format : str
        Output format: 'jpl' or 'standard'
        
    Returns:
    --------
    pd.DataFrame
        Loaded and formatted data
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.pkl':
        loader = AethalometerPKLLoader(file_path)
        return loader.load(site_filter=site_filter, 
                          convert_to_jpl=(output_format == 'jpl'))
    elif file_path.suffix == '.csv':
        loader = AethalometerCSVLoader(file_path)
        return loader.load()
    else:
        raise DataValidationError(f"Unsupported file format: {file_path.suffix}")