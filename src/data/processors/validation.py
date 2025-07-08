"""Data validation utilities"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from ...core.exceptions import DataValidationError, InsufficientDataError
from ...config.settings import MIN_SAMPLES_FOR_ANALYSIS

def validate_columns_exist(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that required columns exist in dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    required_columns : List[str]
        List of required column names
        
    Raises:
    -------
    DataValidationError
        If any required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        available = list(df.columns)
        raise DataValidationError(
            f"Missing required columns: {missing}. "
            f"Available columns: {available}"
        )

def get_valid_data_mask(df: pd.DataFrame, columns: List[str], 
                       min_value: float = 0, 
                       allow_zero: bool = False) -> pd.Series:
    """
    Create boolean mask for valid data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to validate
    min_value : float, default 0
        Minimum allowed value
    allow_zero : bool, default False
        Whether to allow zero values
        
    Returns:
    --------
    pd.Series
        Boolean mask for valid data
    """
    mask = pd.Series(True, index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
            
        # Check for non-null values
        mask &= df[col].notna()
        
        # Check for valid range
        if allow_zero:
            mask &= (df[col] >= min_value)
        else:
            mask &= (df[col] > min_value)
    
    return mask

def validate_sample_size(mask: pd.Series, 
                        min_samples: Optional[int] = None,
                        operation_name: str = "analysis") -> None:
    """
    Validate minimum sample size
    
    Parameters:
    -----------
    mask : pd.Series
        Boolean mask of valid samples
    min_samples : int, optional
        Minimum required samples (default from config)
    operation_name : str
        Name of operation for error message
        
    Raises:
    -------
    InsufficientDataError
        If insufficient valid samples
    """
    if min_samples is None:
        min_samples = MIN_SAMPLES_FOR_ANALYSIS
    
    valid_count = mask.sum()
    if valid_count < min_samples:
        raise InsufficientDataError(
            f"Insufficient valid data for {operation_name}: "
            f"{valid_count} < {min_samples} required"
        )

def check_data_range(series: pd.Series, 
                    valid_range: Tuple[float, float],
                    column_name: str) -> Dict[str, int]:
    """
    Check if data falls within expected range
    
    Parameters:
    -----------
    series : pd.Series
        Data to check
    valid_range : Tuple[float, float]
        (min_value, max_value) tuple
    column_name : str
        Column name for reporting
        
    Returns:
    --------
    Dict[str, int]
        Statistics about range validation
    """
    min_val, max_val = valid_range
    
    below_min = (series < min_val).sum()
    above_max = (series > max_val).sum()
    in_range = ((series >= min_val) & (series <= max_val)).sum()
    
    return {
        'total_samples': len(series),
        'in_range': in_range,
        'below_min': below_min,
        'above_max': above_max,
        'column_name': column_name,
        'valid_range': valid_range
    }
