"""Aethalometer data smoothening algorithms"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from ...core.base import BaseAnalyzer
from ...data.processors.validation import validate_columns_exist, get_valid_data_mask

class BaseSmoothing(BaseAnalyzer):
    """Base class for smoothening algorithms"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.wavelengths = ['UV', 'Blue', 'Green', 'Red', 'IR']
        self.bc_columns = [f"{wl} BCc" for wl in self.wavelengths]
        self.atn_columns = [f"{wl} ATN1" for wl in self.wavelengths]
    
    def validate_wavelength_data(self, data: pd.DataFrame, wavelength: str) -> None:
        """Validate that required columns exist for a wavelength"""
        bc_col = f"{wavelength} BCc"
        atn_col = f"{wavelength} ATN1"
        required = [bc_col, atn_col]
        validate_columns_exist(data, required)


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
                # High variability - use smaller window
                window_size = max(1, min(5, i + 1))
            else:
                # Low variability - use larger window
                window_size = max(1, min(15, i + 1))
            
            # Calculate weighted average
            start_idx = max(0, i - window_size + 1)
            window_values = bc_values[start_idx:i+1]
            
            # Weight by inverse of time distance (recent points get higher weight)
            weights = np.exp(-0.1 * np.arange(len(window_values))[::-1])
            smoothed_bc[i] = np.average(window_values, weights=weights)
        
        return smoothed_bc
    
    def _calculate_improvement_metrics(self, original: np.ndarray, smoothed: np.ndarray) -> Dict[str, float]:
        """Calculate metrics showing improvement from smoothening"""
        # Remove NaN values for calculations
        valid_mask = ~(np.isnan(original) | np.isnan(smoothed))
        orig_clean = original[valid_mask]
        smooth_clean = smoothed[valid_mask]
        
        if len(orig_clean) < 2:
            return {'error': 'Insufficient valid data for metrics'}
        
        # Calculate noise reduction
        orig_std = np.std(orig_clean)
        smooth_std = np.std(smooth_clean)
        noise_reduction = (1 - smooth_std / orig_std) * 100 if orig_std > 0 else 0
        
        # Calculate correlation with original
        correlation = np.corrcoef(orig_clean, smooth_clean)[0, 1] if len(orig_clean) > 1 else 1.0
        
        # Count negative value reduction
        orig_negatives = np.sum(orig_clean < 0)
        smooth_negatives = np.sum(smooth_clean < 0)
        negative_reduction = orig_negatives - smooth_negatives
        
        return {
            'noise_reduction_percent': round(noise_reduction, 2),
            'correlation_with_original': round(correlation, 3),
            'original_std': round(orig_std, 3),
            'smoothed_std': round(smooth_std, 3),
            'original_negatives': int(orig_negatives),
            'smoothed_negatives': int(smooth_negatives),
            'negative_reduction': int(negative_reduction)
        }


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
        """Apply centered moving average"""
        n = len(bc_values)
        smoothed_bc = np.full(n, np.nan)
        half_window = self.window_size // 2
        
        for i in range(n):
            # Define window boundaries
            start_idx = max(0, i - half_window)
            end_idx = min(n, i + half_window + 1)
            
            # Extract window values
            window_values = bc_values[start_idx:end_idx]
            
            # Calculate centered moving average
            valid_values = window_values[~np.isnan(window_values)]
            if len(valid_values) > 0:
                smoothed_bc[i] = np.mean(valid_values)
        
        return smoothed_bc
    
    def _calculate_improvement_metrics(self, original: np.ndarray, smoothed: np.ndarray) -> Dict[str, float]:
        """Calculate improvement metrics for CMA"""
        # Same as ONA metrics
        valid_mask = ~(np.isnan(original) | np.isnan(smoothed))
        orig_clean = original[valid_mask]
        smooth_clean = smoothed[valid_mask]
        
        if len(orig_clean) < 2:
            return {'error': 'Insufficient valid data for metrics'}
        
        orig_std = np.std(orig_clean)
        smooth_std = np.std(smooth_clean)
        noise_reduction = (1 - smooth_std / orig_std) * 100 if orig_std > 0 else 0
        correlation = np.corrcoef(orig_clean, smooth_clean)[0, 1] if len(orig_clean) > 1 else 1.0
        
        orig_negatives = np.sum(orig_clean < 0)
        smooth_negatives = np.sum(smooth_clean < 0)
        negative_reduction = orig_negatives - smooth_negatives
        
        return {
            'noise_reduction_percent': round(noise_reduction, 2),
            'correlation_with_original': round(correlation, 3),
            'original_std': round(orig_std, 3),
            'smoothed_std': round(smooth_std, 3),
            'original_negatives': int(orig_negatives),
            'smoothed_negatives': int(smooth_negatives),
            'negative_reduction': int(negative_reduction)
        }


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
        """Apply Double Exponentially Weighted Moving Average"""
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
        """Calculate improvement metrics for DEMA"""
        valid_mask = ~(np.isnan(original) | np.isnan(smoothed))
        orig_clean = original[valid_mask]
        smooth_clean = smoothed[valid_mask]
        
        if len(orig_clean) < 2:
            return {'error': 'Insufficient valid data for metrics'}
        
        orig_std = np.std(orig_clean)
        smooth_std = np.std(smooth_clean)
        noise_reduction = (1 - smooth_std / orig_std) * 100 if orig_std > 0 else 0
        correlation = np.corrcoef(orig_clean, smooth_clean)[0, 1] if len(orig_clean) > 1 else 1.0
        
        orig_negatives = np.sum(orig_clean < 0)
        smooth_negatives = np.sum(smooth_clean < 0)
        negative_reduction = orig_negatives - smooth_negatives
        
        # Calculate lag (specific to DEMA)
        lag_metric = self._calculate_lag_metric(orig_clean, smooth_clean)
        
        return {
            'noise_reduction_percent': round(noise_reduction, 2),
            'correlation_with_original': round(correlation, 3),
            'original_std': round(orig_std, 3),
            'smoothed_std': round(smooth_std, 3),
            'original_negatives': int(orig_negatives),
            'smoothed_negatives': int(smooth_negatives),
            'negative_reduction': int(negative_reduction),
            'lag_metric': round(lag_metric, 3)
        }
    
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
        lags = np.arange(-len(smooth_norm) + 1, len(orig_norm))
        
        # Find lag with maximum correlation
        max_corr_idx = np.argmax(correlation)
        optimal_lag = lags[max_corr_idx]
        
        return float(abs(optimal_lag))
