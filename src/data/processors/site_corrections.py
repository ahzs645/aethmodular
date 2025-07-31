# ========================================================================
# Site-Specific Corrections Module
# ========================================================================
"""
Site-specific corrections for aethalometer data processing.

This module provides corrections for known site-specific issues,
particularly the Ethiopia (Addis Ababa) pneumatic pump loading compensation fix.
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple
import warnings


class SiteCorrections:
    """
    Main class for applying site-specific corrections to aethalometer data.
    """
    
    def __init__(self, site_code: str = None, verbose: bool = True):
        """
        Initialize site corrections.
        
        Args:
            site_code (str): Site code to determine which corrections to apply
            verbose (bool): Whether to print correction progress
        """
        self.site_code = site_code
        self.verbose = verbose
        self.corrections_applied = []
    
    def apply_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all relevant corrections for the specified site.
        
        Args:
            df (pd.DataFrame): Input aethalometer DataFrame
            
        Returns:
            pd.DataFrame: Corrected DataFrame
        """
        df_corrected = df.copy()
        
        # Site-specific corrections
        if self.site_code in ['ETAD', 'Ethiopia', 'Addis_Ababa']:
            if self.verbose:
                print("🔧 Applying Ethiopia (ETAD) site corrections...")
            df_corrected = self.apply_ethiopia_corrections(df_corrected)
        
        # Add other site-specific corrections here as needed
        # elif self.site_code == 'OTHER_SITE':
        #     df_corrected = self.apply_other_site_corrections(df_corrected)
        
        return df_corrected
    
    def apply_ethiopia_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Ethiopia-specific corrections (pneumatic pump loading compensation fix).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Corrected DataFrame with Ethiopia fixes
        """
        df_corrected = df.copy()
        
        # Apply loading compensation fix for each wavelength
        wavelengths_to_fix = ['IR', 'Blue', 'Red', 'Green', 'UV']
        
        for wl in wavelengths_to_fix:
            if f'{wl} BC1' in df_corrected.columns:
                if self.verbose:
                    print(f"  📊 Applying {wl} loading compensation fix...")
                
                df_corrected = self._apply_manual_loading_compensation_ethiopia(
                    df_corrected, 
                    wavelength=wl, 
                    optimize_k=True
                )
                
                self.corrections_applied.append(f'Ethiopia_{wl}_loading_compensation')
        
        return df_corrected
    
    def _apply_manual_loading_compensation_ethiopia(self, 
                                                 df: pd.DataFrame, 
                                                 wavelength: str = 'IR', 
                                                 optimize_k: bool = True) -> pd.DataFrame:
        """
        Apply manual loading compensation for Ethiopian aethalometer data with pneumatic issues.
        
        This is the core Ethiopia fix based on Kyan's presentation.
        """
        df = df.copy()
        wl = wavelength
        
        # Check if required columns exist
        required_cols = [f'{wl} BC1', f'{wl} ATN1', f'{wl} K']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            if self.verbose:
                print(f"    ⚠️ Skipping {wl}: Missing columns {missing_cols}")
            return df
        
        # Step 1: Calculate median K for manual correction
        mask_atn = df[f'{wl} ATN1'] > 3
        if mask_atn.sum() == 0:
            if self.verbose:
                print(f"    ⚠️ No data with ATN1 > 3 for {wl}")
            return df
        
        median_k = df[mask_atn][f'{wl} K'].median()
        
        if self.verbose:
            print(f"    📈 {wl} median K: {median_k:.6f}")
        
        # Step 2: Create corrected denominator and BCc
        df[f'{wl} denominator_manual'] = 1 - df[f'{wl} ATN1'] * median_k
        
        # Apply manual loading compensation
        df[f'{wl} BCc_manual'] = df[f'{wl} BC1'].where(
            df[f'{wl} ATN1'] <= 3,
            df[f'{wl} BC1'] / df[f'{wl} denominator_manual']
        )
        
        # Step 3: Optional K optimization
        if optimize_k:
            try:
                optimal_k = self._optimize_k_value(df, wavelength)
                if self.verbose:
                    print(f"    🎯 {wl} optimal K: {optimal_k:.6f}")
                
                # Apply optimized correction
                df[f'{wl} denominator_optimized'] = 1 - df[f'{wl} ATN1'] * optimal_k
                df[f'{wl} BCc_optimized'] = df[f'{wl} BC1'].where(
                    df[f'{wl} ATN1'] <= 3,
                    df[f'{wl} BC1'] / df[f'{wl} denominator_optimized']
                )
                
                # Use optimized as default corrected BCc
                df[f'{wl} BCc_corrected'] = df[f'{wl} BCc_optimized']
                
            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️ K optimization failed for {wl}: {e}")
                    print(f"    🔄 Using median K correction")
                df[f'{wl} BCc_corrected'] = df[f'{wl} BCc_manual']
        else:
            df[f'{wl} BCc_corrected'] = df[f'{wl} BCc_manual']
        
        return df
    
    def _optimize_k_value(self, 
                         df: pd.DataFrame, 
                         wavelength: str = 'IR', 
                         k_range: Tuple[float, float] = (0.005, 0.015), 
                         tolerance: float = 4e-4) -> float:
        """
        Optimize K value to minimize correlation between BCc and ATN1.
        Uses golden section search.
        """
        
        def objective_function(k):
            """Calculate absolute correlation between BCc and ATN1 for given k"""
            df_temp = df.copy()
            wl = wavelength
            
            # Apply k correction
            df_temp[f'{wl} denominator_temp'] = 1 - df_temp[f'{wl} ATN1'] * k
            df_temp[f'{wl} BCc_temp'] = df_temp[f'{wl} BC1'].where(
                df_temp[f'{wl} ATN1'] <= 3,
                df_temp[f'{wl} BC1'] / df_temp[f'{wl} denominator_temp']
            )
            
            # Calculate correlation
            mask = df_temp[f'{wl} BCc_temp'].notna() & df_temp[f'{wl} ATN1'].notna()
            if mask.sum() < 10:  # Not enough data
                return 1.0
                
            corr = abs(df_temp.loc[mask, f'{wl} BCc_temp'].corr(df_temp.loc[mask, f'{wl} ATN1']))
            return corr if not pd.isna(corr) else 1.0
        
        # Golden section search
        def golden_section_search(f, a, b, tol=tolerance, max_iter=100):
            gr = (np.sqrt(5) + 1) / 2  # golden ratio
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            
            for _ in range(max_iter):
                if abs(b - a) < tol:
                    break
                if f(c) < f(d):
                    b = d
                else:
                    a = c
                c = b - (b - a) / gr
                d = a + (b - a) / gr
            
            return (b + a) / 2
        
        return golden_section_search(objective_function, k_range[0], k_range[1])
    
    def validate_corrections(self, 
                           df_original: pd.DataFrame, 
                           df_corrected: pd.DataFrame,
                           wavelength: str = 'IR') -> Dict:
        """
        Validate the corrections by comparing key metrics.
        
        Args:
            df_original (pd.DataFrame): Original DataFrame
            df_corrected (pd.DataFrame): Corrected DataFrame
            wavelength (str): Wavelength to validate
            
        Returns:
            Dict: Validation statistics
        """
        wl = wavelength
        validation_results = {}
        
        # Check correlation with ATN1
        for df_name, df in [('original', df_original), ('corrected', df_corrected)]:
            bc_col = f'{wl} BCc' if df_name == 'original' else f'{wl} BCc_corrected'
            
            if bc_col in df.columns:
                mask = df[bc_col].notna() & df[f'{wl} ATN1'].notna()
                if mask.sum() > 10:
                    corr = df.loc[mask, bc_col].corr(df.loc[mask, f'{wl} ATN1'])
                    validation_results[f'{df_name}_atn_correlation'] = corr
        
        # Check denominator distribution
        for suffix in ['', '_manual', '_optimized']:
            col = f'{wl} denominator{suffix}'
            if col in df_corrected.columns:
                denom_stats = df_corrected[col].describe()
                validation_results[f'denominator{suffix}_stats'] = {
                    'min': denom_stats['min'],
                    'median': denom_stats['50%'],
                    'max': denom_stats['max'],
                    'std': denom_stats['std']
                }
        
        return validation_results
    
    def plot_correction_comparison(self, 
                                 df_original: pd.DataFrame, 
                                 df_corrected: pd.DataFrame,
                                 wavelength: str = 'IR', 
                                 save_path: Optional[str] = None):
        """
        Plot comparison of original vs corrected data.
        """
        wl = wavelength
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{wl} Wavelength: Ethiopia Loading Compensation Correction', fontsize=16)
        
        # 1. K distribution comparison
        if f'{wl} K' in df_original.columns:
            mask_atn = df_original[f'{wl} ATN1'] > 3
            k_data = df_original[mask_atn][f'{wl} K']
            
            axes[0,0].hist(k_data, bins=50, alpha=0.7, label='Original K')
            axes[0,0].axvline(k_data.median(), color='red', linestyle='--', label='Median K')
            axes[0,0].set_xlabel('K value')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].set_title(f'{wl} K Distribution')
            axes[0,0].legend()
        
        # 2. BCc time series comparison
        if 'datetime_local' in df_original.index.names or 'datetime_local' in df_original.columns:
            df_plot_orig = df_original.resample('1H').mean() if hasattr(df_original.index, 'freq') else df_original
            df_plot_corr = df_corrected.resample('1H').mean() if hasattr(df_corrected.index, 'freq') else df_corrected
            
            if f'{wl} BCc' in df_plot_orig.columns:
                axes[0,1].plot(df_plot_orig.index, df_plot_orig[f'{wl} BCc'], 
                              label='Original BCc', alpha=0.7, linewidth=0.8)
            
            if f'{wl} BCc_corrected' in df_plot_corr.columns:
                axes[0,1].plot(df_plot_corr.index, df_plot_corr[f'{wl} BCc_corrected'], 
                              label='Corrected BCc', alpha=0.7, linewidth=0.8)
            
            axes[0,1].set_ylabel('BC (ng/m³)')
            axes[0,1].set_title(f'{wl} BCc Time Series Comparison')
            axes[0,1].legend()
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Correlation with ATN1
        for i, (df_name, df, bc_col) in enumerate([
            ('Original', df_original, f'{wl} BCc'),
            ('Corrected', df_corrected, f'{wl} BCc_corrected')
        ]):
            if bc_col in df.columns and f'{wl} ATN1' in df.columns:
                mask = df[bc_col].notna() & df[f'{wl} ATN1'].notna()
                if mask.sum() > 10:
                    corr = df.loc[mask, bc_col].corr(df.loc[mask, f'{wl} ATN1'])
                    
                    axes[1,i].scatter(df.loc[mask, f'{wl} ATN1'], df.loc[mask, bc_col], 
                                    alpha=0.5, s=1)
                    axes[1,i].set_xlabel(f'{wl} ATN1')
                    axes[1,i].set_ylabel('BCc')
                    axes[1,i].set_title(f'{df_name} BCc vs ATN1 (r={corr:.3f})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def apply_ethiopia_fix(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to apply Ethiopia fix to aethalometer data.
    
    Args:
        df (pd.DataFrame): Input aethalometer DataFrame
        verbose (bool): Whether to print progress
        
    Returns:
        pd.DataFrame: DataFrame with Ethiopia corrections applied
    """
    corrector = SiteCorrections(site_code='ETAD', verbose=verbose)
    return corrector.apply_corrections(df)


# Example usage function
def example_ethiopia_correction():
    """Example of how to use the Ethiopia corrections."""
    
    # This would be your actual data loading
    print("📁 Load your aethalometer data:")
    print("df = pd.read_pickle('your_data.pkl')")
    print("# or")
    print("df = pd.read_csv('your_data.csv')")
    
    print("\n🔧 Apply Ethiopia corrections:")
    print("df_corrected = apply_ethiopia_fix(df)")
    
    print("\n📊 Validate corrections:")
    print("corrector = SiteCorrections(site_code='ETAD')")
    print("validation = corrector.validate_corrections(df, df_corrected)")
    print("corrector.plot_correction_comparison(df, df_corrected)")


if __name__ == "__main__":
    print("Site-Specific Corrections Module")
    print("Use apply_ethiopia_fix() for quick Ethiopia corrections")
    print("See example_ethiopia_correction() for usage details")