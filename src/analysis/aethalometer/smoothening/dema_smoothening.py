"""Double Exponentially Weighted Moving Average (DEMA) smoothening for Aethalometer data"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_smoothening import BaseSmoothing
from ....data.processors.validation import get_valid_data_mask


class DEMASmoothing(BaseSmoothing):
    """
    Double Exponentially Weighted Moving Average (DEMA) smoothening
    
    Reduces noise-induced artifacts while limiting lag, especially 
    useful for source apportionment calculations.
    """
    
    def __init__(self, alpha: float = 0.2):
        super().__init__("DEMASmoothing")
        self.alpha = alpha  # Smoothing factor
    
    def analyze(self, data: pd.DataFrame, wavelength: str = 'IR') -> Dict[str, Any]:
        """Apply DEMA smoothening"""
        self.validate_wavelength_data(data, wavelength)
        
        bc_col = f"{wavelength} BCc"
        atn_col = f"{wavelength} ATN1"
        
        # Get clean data
        valid_mask = get_valid_data_mask(data, [bc_col, atn_col])
        clean_data = data[valid_mask].copy().reset_index(drop=True)
        
        if len(clean_data) < 3:
            raise ValueError(f"Insufficient data for DEMA smoothening: {len(clean_data)} points")
        
        # Apply DEMA algorithm
        smoothed_bc = self._apply_dema_algorithm(clean_data[bc_col].values)
        
        # Calculate statistics
        original_bc = clean_data[bc_col].values
        improvement_metrics = self._calculate_improvement_metrics(original_bc, smoothed_bc)
        
        results = {
            'wavelength': wavelength,
            'algorithm': 'DEMA',
            'parameters': {'alpha': self.alpha},
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
    
    def _apply_dema_algorithm(self, bc_values: np.ndarray) -> np.ndarray:
        """
        Apply Double Exponentially Weighted Moving Average
        
        DEMA = 2 * EMA1 - EMA2
        where EMA2 is the EMA of EMA1
        """
        n = len(bc_values)
        
        # Initialize arrays
        ema1 = np.full(n, np.nan)
        ema2 = np.full(n, np.nan)
        dema = np.full(n, np.nan)
        
        # Find first valid value
        first_valid_idx = None
        for i in range(n):
            if not np.isnan(bc_values[i]):
                first_valid_idx = i
                break
        
        if first_valid_idx is None:
            return dema  # All NaN
        
        # Initialize with first valid value
        ema1[first_valid_idx] = bc_values[first_valid_idx]
        ema2[first_valid_idx] = bc_values[first_valid_idx]
        dema[first_valid_idx] = bc_values[first_valid_idx]
        
        # Calculate DEMA
        for i in range(first_valid_idx + 1, n):
            if not np.isnan(bc_values[i]):
                # First EMA
                ema1[i] = self.alpha * bc_values[i] + (1 - self.alpha) * ema1[i-1]
                
                # Second EMA (EMA of EMA)
                ema2[i] = self.alpha * ema1[i] + (1 - self.alpha) * ema2[i-1]
                
                # DEMA calculation
                dema[i] = 2 * ema1[i] - ema2[i]
            else:
                # Carry forward previous values for missing data
                ema1[i] = ema1[i-1]
                ema2[i] = ema2[i-1]
                dema[i] = dema[i-1]
        
        return dema
    
    def _calculate_improvement_metrics(self, original: np.ndarray, smoothed: np.ndarray) -> Dict[str, float]:
        """Calculate improvement metrics for DEMA with lag analysis"""
        # Get base metrics
        base_metrics = super()._calculate_improvement_metrics(original, smoothed)
        
        if 'error' in base_metrics:
            return base_metrics
        
        # Calculate lag (specific to DEMA)
        valid_mask = ~(np.isnan(original) | np.isnan(smoothed))
        orig_clean = original[valid_mask]
        smooth_clean = smoothed[valid_mask]
        
        lag_metric = self._calculate_lag_metric(orig_clean, smooth_clean)
        base_metrics['lag_metric'] = round(lag_metric, 3)
        
        return base_metrics
    
    def _calculate_lag_metric(self, original: np.ndarray, smoothed: np.ndarray) -> float:
        """Calculate lag between original and smoothed data"""
        # Cross-correlation to find optimal lag
        if len(original) < 10:
            return 0.0
        
        # Normalize data
        orig_norm = (original - np.mean(original)) / np.std(original)
        smooth_norm = (smoothed - np.mean(smoothed)) / np.std(smoothed)
        
        # Calculate cross-correlation
        correlation = np.correlate(orig_norm, smooth_norm, mode='full')
        lags = np.arange(-len(smooth_norm) + 1, len(original))
        
        # Find lag with maximum correlation
        max_corr_idx = np.argmax(correlation)
        optimal_lag = lags[max_corr_idx]
        
        return float(abs(optimal_lag))
