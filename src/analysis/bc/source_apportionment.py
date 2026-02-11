"""
Source Apportionment Analyzer for Aethalometer Data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from src.core.base import BaseAnalyzer
from src.core.exceptions import DataValidationError

class SourceApportionmentAnalyzer(BaseAnalyzer):
    """
    Analyzer for source apportionment of black carbon using aethalometer data
    
    This analyzer estimates contributions from different sources (biomass burning, 
    fossil fuel combustion) based on multi-wavelength black carbon measurements.
    """
    
    def __init__(self):
        """Initialize the source apportionment analyzer"""
        try:
            super().__init__("SourceApportionmentAnalyzer")
        except TypeError:
            # Fallback if BaseAnalyzer doesn't require name parameter
            super().__init__()
        
        # Common wavelength-dependent columns found in aethalometer data
        self.bc_columns = [
            'UV BCc', 'IR BCc', 'Blue BCc', 'Green BCc', 'Red BCc',  # JPL format
            'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7',  # Raw format
            'BC_370', 'BC_470', 'BC_520', 'BC_590', 'BC_660', 'BC_880', 'BC_950'  # Wavelength format
        ]
        
        # Standard Angstrom exponents for different sources
        self.aae_biomass = 2.0  # Typical AAE for biomass burning
        self.aae_fossil = 1.0   # Typical AAE for fossil fuel
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform source apportionment analysis
        
        Args:
            data: DataFrame with aethalometer measurements
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        try:
            # Find available BC columns
            available_bc_cols = [col for col in self.bc_columns if col in data.columns]
            
            if len(available_bc_cols) < 2:
                results['error'] = f"Insufficient BC columns for source apportionment. Need at least 2, found {len(available_bc_cols)}"
                return results
            
            results['available_columns'] = available_bc_cols
            results['total_measurements'] = len(data)
            
            # Basic source apportionment using two-wavelength method
            if len(available_bc_cols) >= 2:
                bc1_col = available_bc_cols[0]  # Shorter wavelength (usually UV or Blue)
                bc2_col = available_bc_cols[-1]  # Longer wavelength (usually IR or Red)
                
                # Calculate Angstrom exponent
                aae_calculated = self._calculate_aae(data, bc1_col, bc2_col)
                
                # Estimate source contributions
                source_contrib = self._estimate_source_contributions(data, bc1_col, bc2_col, aae_calculated)
                
                results['aae_statistics'] = {
                    'mean': float(aae_calculated.mean()) if not aae_calculated.empty else 0,
                    'std': float(aae_calculated.std()) if not aae_calculated.empty else 0,
                    'median': float(aae_calculated.median()) if not aae_calculated.empty else 0,
                    'min': float(aae_calculated.min()) if not aae_calculated.empty else 0,
                    'max': float(aae_calculated.max()) if not aae_calculated.empty else 0
                }
                
                results['source_contributions'] = source_contrib
                results['wavelength_pair'] = {'bc1': bc1_col, 'bc2': bc2_col}
            
            # Advanced analysis if more wavelengths available
            if len(available_bc_cols) >= 3:
                multi_wavelength_results = self._multi_wavelength_analysis(data, available_bc_cols)
                results['multi_wavelength'] = multi_wavelength_results
            
            # Summary statistics
            results['summary'] = self._generate_summary(results)
            
        except Exception as e:
            results['error'] = f"Analysis failed: {str(e)}"
        
        return results
    
    def _calculate_aae(self, data: pd.DataFrame, bc1_col: str, bc2_col: str) -> pd.Series:
        """Calculate Angstrom Absorption Exponent"""
        
        # Assume standard wavelengths if not specified in column names
        wavelength_map = {
            'UV BCc': 370, 'Blue BCc': 470, 'Green BCc': 520, 
            'Red BCc': 660, 'IR BCc': 880,
            'BC1': 370, 'BC2': 470, 'BC3': 520, 'BC4': 590, 
            'BC5': 660, 'BC6': 880, 'BC7': 950
        }
        
        lambda1 = wavelength_map.get(bc1_col, 470)  # Default to blue
        lambda2 = wavelength_map.get(bc2_col, 880)  # Default to IR
        
        # Calculate AAE: AAE = -ln(BC1/BC2) / ln(λ1/λ2)
        bc_ratio = data[bc1_col] / data[bc2_col]
        wavelength_ratio = lambda1 / lambda2
        
        # Avoid division by zero and negative values
        valid_mask = (data[bc1_col] > 0) & (data[bc2_col] > 0) & (bc_ratio > 0)
        
        aae = pd.Series(index=data.index, dtype=float)
        aae[valid_mask] = -np.log(bc_ratio[valid_mask]) / np.log(wavelength_ratio)
        
        return aae
    
    def _estimate_source_contributions(self, data: pd.DataFrame, bc1_col: str, bc2_col: str, aae: pd.Series) -> Dict[str, Any]:
        """Estimate biomass and fossil fuel contributions"""
        
        # Use the longer wavelength (usually IR) as reference for total BC
        bc_total = data[bc2_col]
        
        # Simple two-component model
        # f_biomass = (AAE - AAE_fossil) / (AAE_biomass - AAE_fossil)
        f_biomass = (aae - self.aae_fossil) / (self.aae_biomass - self.aae_fossil)
        f_biomass = np.clip(f_biomass, 0, 1)  # Constrain between 0 and 1
        f_fossil = 1 - f_biomass
        
        bc_biomass = bc_total * f_biomass
        bc_fossil = bc_total * f_fossil
        
        # Calculate statistics
        valid_mask = ~(np.isnan(f_biomass) | np.isnan(f_fossil))
        
        return {
            'biomass_fraction': {
                'mean': float(f_biomass[valid_mask].mean()) if valid_mask.any() else 0,
                'std': float(f_biomass[valid_mask].std()) if valid_mask.any() else 0,
                'median': float(f_biomass[valid_mask].median()) if valid_mask.any() else 0
            },
            'fossil_fraction': {
                'mean': float(f_fossil[valid_mask].mean()) if valid_mask.any() else 0,
                'std': float(f_fossil[valid_mask].std()) if valid_mask.any() else 0,
                'median': float(f_fossil[valid_mask].median()) if valid_mask.any() else 0
            },
            'biomass_bc': {
                'mean': float(bc_biomass[valid_mask].mean()) if valid_mask.any() else 0,
                'std': float(bc_biomass[valid_mask].std()) if valid_mask.any() else 0,
                'total': float(bc_biomass[valid_mask].sum()) if valid_mask.any() else 0
            },
            'fossil_bc': {
                'mean': float(bc_fossil[valid_mask].mean()) if valid_mask.any() else 0,
                'std': float(bc_fossil[valid_mask].std()) if valid_mask.any() else 0,
                'total': float(bc_fossil[valid_mask].sum()) if valid_mask.any() else 0
            },
            'valid_measurements': int(valid_mask.sum())
        }
    
    def _multi_wavelength_analysis(self, data: pd.DataFrame, bc_cols: List[str]) -> Dict[str, Any]:
        """Perform multi-wavelength analysis for better source apportionment"""
        
        results = {}
        
        # Calculate AAE for all wavelength pairs
        aae_pairs = []
        for i in range(len(bc_cols)):
            for j in range(i+1, len(bc_cols)):
                col1, col2 = bc_cols[i], bc_cols[j]
                aae_pair = self._calculate_aae(data, col1, col2)
                aae_pairs.append({
                    'wavelength_pair': f"{col1}-{col2}",
                    'aae_mean': float(aae_pair.mean()) if not aae_pair.empty else 0,
                    'aae_std': float(aae_pair.std()) if not aae_pair.empty else 0
                })
        
        results['aae_all_pairs'] = aae_pairs
        
        # Spectral analysis
        bc_means = [data[col].mean() for col in bc_cols]
        results['spectral_profile'] = {
            'columns': bc_cols,
            'mean_concentrations': bc_means,
            'concentration_ratios': [bc_means[i]/bc_means[-1] if bc_means[-1] > 0 else 0 
                                   for i in range(len(bc_means))]
        }
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of the source apportionment analysis"""
        
        if 'error' in results:
            return f"Analysis failed: {results['error']}"
        
        summary_parts = []
        
        if 'source_contributions' in results:
            contrib = results['source_contributions']
            biomass_pct = contrib['biomass_fraction']['mean'] * 100
            fossil_pct = contrib['fossil_fraction']['mean'] * 100
            
            summary_parts.append(f"Source contributions: {biomass_pct:.1f}% biomass, {fossil_pct:.1f}% fossil fuel")
        
        if 'aae_statistics' in results:
            aae_mean = results['aae_statistics']['mean']
            summary_parts.append(f"Average AAE: {aae_mean:.2f}")
        
        if 'available_columns' in results:
            n_cols = len(results['available_columns'])
            summary_parts.append(f"Analysis based on {n_cols} wavelength channels")
        
        return "; ".join(summary_parts) if summary_parts else "Source apportionment analysis completed"

# For backwards compatibility
class SourceApportionmentAnalyser(SourceApportionmentAnalyzer):
    """British spelling alias"""
    pass
