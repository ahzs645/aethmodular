"""Base smoothening class for all aethalometer smoothing algorithms"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ....core.base import BaseAnalyzer
from ....data.processors.validation import validate_columns_exist, get_valid_data_mask


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
    
    def analyze(self, data: pd.DataFrame, wavelength: str = 'IR') -> Dict[str, Any]:
        """
        Abstract method for applying smoothening algorithm
        
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
        raise NotImplementedError("Subclasses must implement analyze method")
