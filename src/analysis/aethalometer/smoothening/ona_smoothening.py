"""Optimized Noise-reduction Algorithm (ONA) for Aethalometer data"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_smoothening import BaseSmoothing
from ....data.processors.validation import get_valid_data_mask


class ONASmoothing(BaseSmoothing):
    """
    Optimized Noise-reduction Algorithm (ONA) for Aethalometer data
    
    Based on Hagler et al. (2011) - adaptively time-averages BC data 
    based on incremental light attenuation (ΔATN).
    """
    
    def __init__(self, delta_atn_threshold: float = 0.05):
        super().__init__("ONASmoothing")
        self.delta_atn_threshold = delta_atn_threshold
    
    def analyze(self, data: pd.DataFrame, wavelength: str = 'IR') -> Dict[str, Any]:
        """
        Apply ONA smoothening to aethalometer data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Aethalometer data with timestamp, BC, and ATN columns
        wavelength : str
            Wavelength to process ('UV', 'Blue', 'Green', 'Red', 'IR')
            
        Returns:
        --------
        Dict[str, Any]
            Results including smoothed BC values and metadata
        """
        self.validate_wavelength_data(data, wavelength)
        
        bc_col = f"{wavelength} BCc"
        atn_col = f"{wavelength} ATN1"
        
        # Get clean data
        valid_mask = get_valid_data_mask(data, [bc_col, atn_col])
        clean_data = data[valid_mask].copy().reset_index(drop=True)
        
        if len(clean_data) < 2:
            raise ValueError(f"Insufficient data for ONA smoothening: {len(clean_data)} points")
        
        # Apply ONA algorithm
        smoothed_bc = self._apply_ona_algorithm(
            clean_data[bc_col].values, 
            clean_data[atn_col].values
        )
        
        # Calculate statistics
        original_bc = clean_data[bc_col].values
        improvement_metrics = self._calculate_improvement_metrics(original_bc, smoothed_bc)
        
        results = {
            'wavelength': wavelength,
            'algorithm': 'ONA',
            'parameters': {'delta_atn_threshold': self.delta_atn_threshold},
            'sample_info': {
                'total_samples': len(data),
                'valid_samples': len(clean_data),
                'data_completeness': len(clean_data) / len(data) * 100
            },
            'smoothed_data': {
                'timestamps': clean_data.index if 'timestamp' not in clean_data.columns else clean_data['timestamp'],
                'original_bc': original_bc,
                'smoothed_bc': smoothed_bc,
                'bc_column': bc_col
            },
            'improvement_metrics': improvement_metrics
        }
        
        return results
    
    def _apply_ona_algorithm(self, bc_values: np.ndarray, atn_values: np.ndarray) -> np.ndarray:
        """
        Apply ONA algorithm with adaptive time averaging
        
        The algorithm adaptively adjusts the averaging window based on
        the rate of change in light attenuation (ΔATN).
        """
        n = len(bc_values)
        smoothed_bc = np.full(n, np.nan)
        
        for i in range(n):
            # Calculate ΔATN for adaptive window
            if i == 0:
                delta_atn = 0
            else:
                delta_atn = abs(atn_values[i] - atn_values[i-1])
            
            # Determine averaging window based on ΔATN
            if delta_atn >= self.delta_atn_threshold:
                # High variability - use smaller window (favor temporal resolution)
                window_size = max(1, min(5, i + 1))
            else:
                # Low variability - use larger window (favor noise reduction)
                window_size = max(1, min(15, i + 1))
            
            # Calculate weighted average
            start_idx = max(0, i - window_size + 1)
            window_values = bc_values[start_idx:i+1]
            
            # Weight by inverse of time distance (recent points get higher weight)
            weights = np.exp(-0.1 * np.arange(len(window_values))[::-1])
            smoothed_bc[i] = np.average(window_values, weights=weights)
        
        return smoothed_bc
