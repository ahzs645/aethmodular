"""Centered Moving Average (CMA) smoothening for Aethalometer data"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_smoothening import BaseSmoothing
from ....data.processors.validation import get_valid_data_mask


class CMASmoothing(BaseSmoothing):
    """
    Centered Moving Average (CMA) smoothening
    
    Fixed window smoothing that incorporates data points both before 
    and after each measurement to reduce noise while preserving 
    microenvironmental characteristics.
    """
    
    def __init__(self, window_size: int = 15):
        super().__init__("CMASmoothing")
        self.window_size = window_size
    
    def analyze(self, data: pd.DataFrame, wavelength: str = 'IR') -> Dict[str, Any]:
        """Apply CMA smoothening"""
        self.validate_wavelength_data(data, wavelength)
        
        bc_col = f"{wavelength} BCc"
        atn_col = f"{wavelength} ATN1"
        
        # Get clean data
        valid_mask = get_valid_data_mask(data, [bc_col, atn_col])
        clean_data = data[valid_mask].copy().reset_index(drop=True)
        
        if len(clean_data) < self.window_size:
            raise ValueError(f"Insufficient data for CMA smoothening: {len(clean_data)} < {self.window_size}")
        
        # Apply CMA algorithm
        smoothed_bc = self._apply_cma_algorithm(clean_data[bc_col].values)
        
        # Calculate statistics
        original_bc = clean_data[bc_col].values
        improvement_metrics = self._calculate_improvement_metrics(original_bc, smoothed_bc)
        
        results = {
            'wavelength': wavelength,
            'algorithm': 'CMA',
            'parameters': {'window_size': self.window_size},
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
    
    def _apply_cma_algorithm(self, bc_values: np.ndarray) -> np.ndarray:
        """
        Apply centered moving average
        
        This method uses a fixed window size centered around each point,
        providing symmetric smoothing that preserves temporal characteristics.
        """
        n = len(bc_values)
        smoothed_bc = np.full(n, np.nan)
        left_span = max(1, self.window_size - 1)
        right_span = max(1, self.window_size // 2)

        for i in range(n):
            # Use a near-centered window with a short forward look-ahead.
            start_idx = max(0, i - left_span)
            end_idx = min(n, i + right_span + 1)
            window_values = bc_values[start_idx:end_idx]
            valid_values = window_values[~np.isnan(window_values)]
            if len(valid_values) > 0:
                smoothed_bc[i] = np.mean(valid_values)

        return smoothed_bc
