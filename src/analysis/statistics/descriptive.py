"""Descriptive statistics functions"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def calculate_basic_statistics(data: pd.Series, 
                             name: Optional[str] = None,
                             include_advanced: bool = False) -> Dict[str, float]:
    """
    Calculate comprehensive descriptive statistics
    
    Parameters:
    -----------
    data : pd.Series
        Input data series
    name : str, optional
        Prefix for statistic names
    include_advanced : bool, default False
        Whether to include skewness and kurtosis
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of statistics
    """
    if len(data) == 0:
        return {}
    
    # Remove NaN values for calculations
    clean_data = data.dropna()
    
    if len(clean_data) == 0:
        return {}
    
    prefix = f"{name}_" if name else ""
    
    stats = {
        f'{prefix}count': len(clean_data),
        f'{prefix}mean': clean_data.mean(),
        f'{prefix}std': clean_data.std(),
        f'{prefix}min': clean_data.min(),
        f'{prefix}q25': clean_data.quantile(0.25),
        f'{prefix}median': clean_data.median(),
        f'{prefix}q75': clean_data.quantile(0.75),
        f'{prefix}max': clean_data.max(),
    }
    
    # Add coefficient of variation if mean > 0
    if clean_data.mean() > 0:
        stats[f'{prefix}cv'] = clean_data.std() / clean_data.mean()
    
    # Add advanced statistics if requested
    if include_advanced:
        stats[f'{prefix}skewness'] = clean_data.skew()
        stats[f'{prefix}kurtosis'] = clean_data.kurtosis()
    
    return stats

def calculate_summary_by_group(df: pd.DataFrame, 
                              value_col: str,
                              group_col: str) -> pd.DataFrame:
    """
    Calculate summary statistics by groups
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    value_col : str
        Column to summarize
    group_col : str
        Column to group by
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics by group
    """
    return df.groupby(group_col)[value_col].agg([
        'count', 'mean', 'std', 'min', 'median', 'max'
    ]).round(3)
