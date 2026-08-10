"""Aethalometer calibration and preprocessing"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.core.base import BaseAnalyzer


class AethalometerCalibrator(BaseAnalyzer):
    """Calibration and preprocessing for aethalometer data"""
    
    def __init__(self, 
                 flow_rate_correction: bool = True,
                 loading_effect_correction: bool = True,
                 cross_sensitivity_correction: bool = True):
        """
        Initialize calibrator with correction options
        
        Args:
            flow_rate_correction: Apply flow rate correction
            loading_effect_correction: Apply filter loading effect correction
            cross_sensitivity_correction: Apply multiple scattering correction
        """
        self.flow_rate_correction = flow_rate_correction
        self.loading_effect_correction = loading_effect_correction
        self.cross_sensitivity_correction = cross_sensitivity_correction
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply calibration corrections to aethalometer data
        
        Args:
            data: Raw aethalometer DataFrame
            
        Returns:
            Dict containing calibrated data and correction factors
        """
        calibrated_data = data.copy()
        corrections_applied = []
        
        if self.flow_rate_correction:
            calibrated_data = self._apply_flow_rate_correction(calibrated_data)
            corrections_applied.append("flow_rate")
            
        if self.loading_effect_correction:
            calibrated_data = self._apply_loading_correction(calibrated_data)
            corrections_applied.append("loading_effect")
            
        if self.cross_sensitivity_correction:
            calibrated_data = self._apply_cross_sensitivity_correction(calibrated_data)
            corrections_applied.append("cross_sensitivity")
        
        return {
            'calibrated_data': calibrated_data,
            'corrections_applied': corrections_applied,
            'original_shape': data.shape,
            'calibrated_shape': calibrated_data.shape
        }
    
    def _apply_flow_rate_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply flow rate correction"""
        # Check if flow rate column exists
        flow_columns = [col for col in data.columns if 'flow' in col.lower()]
        
        if flow_columns:
            flow_col = flow_columns[0]  # Use first flow column found
            # Apply standard flow rate correction (adjust as needed)
            standard_flow = 5.0  # mL/min (typical standard)
            correction_factor = data[flow_col] / standard_flow
            
            # Apply correction to BC columns
            bc_columns = [col for col in data.columns if 'BC' in col and 'c' in col]
            for col in bc_columns:
                if col in data.columns:
                    data[col] = data[col] * correction_factor
        
        return data
    
    def _apply_loading_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply filter loading effect correction"""
        # Find attenuation columns
        atn_columns = [col for col in data.columns if 'ATN' in col]
        
        if atn_columns:
            # Apply Weingartner et al. (2003) loading correction
            # This is a simplified version - adjust parameters as needed
            for atn_col in atn_columns:
                if atn_col in data.columns:
                    # Loading parameter (typical value)
                    loading_param = 0.85
                    
                    # Apply correction: BC_corrected = BC_raw / (1 - loading_param * ATN/100)
                    wavelength = self._extract_wavelength_from_column(atn_col)
                    bc_col = f'{wavelength} BCc' if wavelength else None
                    
                    if bc_col and bc_col in data.columns:
                        correction_factor = 1 / (1 - loading_param * data[atn_col] / 100)
                        data[bc_col] = data[bc_col] * correction_factor
        
        return data
    
    def _apply_cross_sensitivity_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply multiple scattering and cross-sensitivity correction"""
        # Apply Schmid et al. (2006) correction
        # This is a simplified version - adjust as needed
        
        # Multiple scattering enhancement factor
        enhancement_factor = 2.14  # Typical value for aethalometer
        
        # Find and correct BC columns
        bc_columns = [col for col in data.columns if 'BC' in col and 'c' in col]
        for col in bc_columns:
            if col in data.columns:
                data[col] = data[col] / enhancement_factor
        
        return data
    
    def _extract_wavelength_from_column(self, column_name: str) -> Optional[str]:
        """Extract wavelength identifier from column name"""
        wavelengths = ['IR', 'Blue', 'Green', 'Red', 'UV']
        for wl in wavelengths:
            if wl in column_name:
                return wl
        return None
    
    def get_calibration_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of applied calibrations"""
        corrections = results.get('corrections_applied', [])
        original_shape = results.get('original_shape', (0, 0))
        
        summary = f"""
Calibration Summary:
==================
Original data shape: {original_shape}
Corrections applied: {', '.join(corrections) if corrections else 'None'}

Applied corrections:
- Flow rate correction: {'✓' if 'flow_rate' in corrections else '✗'}
- Loading effect correction: {'✓' if 'loading_effect' in corrections else '✗'}
- Cross-sensitivity correction: {'✓' if 'cross_sensitivity' in corrections else '✗'}
        """
        return summary.strip()
