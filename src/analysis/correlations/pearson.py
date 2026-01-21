"""Pearson correlation analysis"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, Optional
from ...core.exceptions import AnalysisError

def calculate_pearson_correlation(x: pd.Series, y: pd.Series,
                                min_samples: int = 3) -> Dict[str, float]:
    """
    Calculate Pearson correlation with validation
    
    Parameters:
    -----------
    x, y : pd.Series
        Input data series
    min_samples : int, default 3
        Minimum samples required
        
    Returns:
    --------
    Dict[str, float]
        Correlation results
        
    Raises:
    -------
    AnalysisError
        If calculation fails
    """
    # Align series and remove NaN
    aligned_data = pd.DataFrame({'x': x, 'y': y}).dropna()
    
    if len(aligned_data) < min_samples:
        raise AnalysisError(
            f"Insufficient valid data pairs: {len(aligned_data)} < {min_samples}"
        )
    
    try:
        correlation, p_value = pearsonr(aligned_data['x'], aligned_data['y'])
        
        return {
            'pearson_r': correlation,
            'pearson_p': p_value,
            'n_samples': len(aligned_data),
            'x_mean': aligned_data['x'].mean(),
            'y_mean': aligned_data['y'].mean(),
            'x_std': aligned_data['x'].std(),
            'y_std': aligned_data['y'].std(),
        }
        
    except Exception as e:
        raise AnalysisError(f"Error calculating Pearson correlation: {e}")
