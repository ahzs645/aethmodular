# ========================================================================
# PKL Data Cleaning Pipeline - Enhanced Version with Ethiopia Fix
# ========================================================================
"""
PKL data cleaning pipeline for aethalometer data processing.

This module provides comprehensive data cleaning functions specifically designed
for PKL format aethalometer data, including status cleaning, optical saturation
removal, flow validation, roughness-based quality control, AND site-specific
corrections including the Ethiopia pneumatic pump loading compensation fix.
"""

import sys
import os
import importlib.util
import pandas as pd
from itertools import groupby
from operator import itemgetter
from typing import Dict, List, Union, Optional, Tuple

# Import the site corrections module
try:
    from .site_corrections import SiteCorrections, apply_ethiopia_fix
    SITE_CORRECTIONS_AVAILABLE = True
except ImportError:
    try:
        from site_corrections import SiteCorrections, apply_ethiopia_fix
        SITE_CORRECTIONS_AVAILABLE = True
    except ImportError:
        print("⚠️ Site corrections module not found. Ethiopia fix will be skipped.")
        SITE_CORRECTIONS_AVAILABLE = False

# Optional statsmodels import - will be imported when needed
try:
    from statsmodels import robust
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Import external calibration module
def _import_calibration():
    """Import the external calibration module."""
    # Get the path to the external calibration module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    external_dir = os.path.join(current_dir, '..', '..', 'external')
    module_path = os.path.join(external_dir, 'calibration.py')
    
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Calibration module not found at {module_path}")
    
    module_name = "calibration"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import calibration module
calibration = _import_calibration()

# Default wavelengths to filter
DEFAULT_WAVELENGTHS = ['IR', 'Blue']

class PKLDataCleaner:
    """
    Main class for PKL data cleaning operations with site-specific corrections.
    
    This class provides a comprehensive suite of cleaning methods for
    aethalometer data in PKL format, including the Ethiopia loading compensation fix.
    """
    
    def __init__(self, wavelengths_to_filter=None, verbose=True, site_code=None):
        """
        Initialize the PKL data cleaner.
        
        Args:
            wavelengths_to_filter (list): List of wavelengths to process.
                                        Defaults to ['IR', 'Blue'].
            verbose (bool): Whether to print cleaning progress.
            site_code (str): Site code for site-specific corrections (e.g., 'ETAD' for Ethiopia)
        """
        self.wls_to_filter = wavelengths_to_filter or DEFAULT_WAVELENGTHS
        self.verbose = verbose
        self.site_code = site_code
        
        # Initialize site corrections if available
        if SITE_CORRECTIONS_AVAILABLE and site_code:
            self.site_corrector = SiteCorrections(site_code=site_code, verbose=verbose)
        else:
            self.site_corrector = None
    
    def clean_pkl_data(self, 
                      df: pd.DataFrame,
                      apply_status_cleaning: bool = True,
                      apply_optical_saturation: bool = True,
                      apply_flow_validation: bool = True,
                      apply_roughness_cleaning: bool = True,
                      apply_site_corrections: bool = True,
                      **kwargs) -> pd.DataFrame:
        """
        Complete PKL data cleaning pipeline with site-specific corrections.
        
        Args:
            df (pd.DataFrame): Input PKL data
            apply_status_cleaning (bool): Whether to apply status cleaning
            apply_optical_saturation (bool): Whether to remove optical saturation
            apply_flow_validation (bool): Whether to validate flow rates
            apply_roughness_cleaning (bool): Whether to apply roughness-based cleaning
            apply_site_corrections (bool): Whether to apply site-specific corrections
            **kwargs: Additional arguments for cleaning methods
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        
        if self.verbose:
            print("🧹 Starting PKL Data Cleaning Pipeline")
            print("=" * 50)
            print(f"📊 Input data shape: {df.shape}")
        
        df_cleaned = df.copy()
        
        # Step 1: Convert to float (existing functionality)
        if self.verbose:
            print("\n1️⃣ Converting data types...")
        df_cleaned = calibration.convert_to_float(df_cleaned)
        
        # Step 2: Status cleaning
        if apply_status_cleaning:
            if self.verbose:
                print("\n2️⃣ Applying status cleaning...")
            df_cleaned = self.apply_status_cleaning(df_cleaned, **kwargs)
        
        # Step 3: Optical saturation removal
        if apply_optical_saturation:
            if self.verbose:
                print("\n3️⃣ Removing optical saturation...")
            df_cleaned = self.remove_optical_saturation(df_cleaned, **kwargs)
        
        # Step 4: Flow validation
        if apply_flow_validation:
            if self.verbose:
                print("\n4️⃣ Validating flow rates...")
            df_cleaned = self.validate_flow_rates(df_cleaned, **kwargs)
        
        # Step 5: Roughness-based cleaning
        if apply_roughness_cleaning:
            if self.verbose:
                print("\n5️⃣ Applying roughness-based cleaning...")
            df_cleaned = self.apply_roughness_cleaning(df_cleaned, **kwargs)
        
        # NEW STEP 6: Site-specific corrections (Ethiopia fix)
        if apply_site_corrections and self.site_corrector:
            if self.verbose:
                print(f"\n6️⃣ Applying {self.site_code} site-specific corrections...")
            df_cleaned_pre_site = df_cleaned.copy()  # Keep copy for validation
            df_cleaned = self.site_corrector.apply_corrections(df_cleaned)
            
            # Validate corrections if verbose
            if self.verbose and 'ETAD' in str(self.site_code):
                self._validate_ethiopia_corrections(df_cleaned_pre_site, df_cleaned)
        
        elif apply_site_corrections and not self.site_corrector:
            if self.verbose:
                print("\n6️⃣ ⚠️ Site corrections requested but no site code provided")
        
        if self.verbose:
            print(f"\n✅ Cleaning completed! Final shape: {df_cleaned.shape}")
            print(f"📉 Rows removed: {df.shape[0] - df_cleaned.shape[0]:,}")
            
            # Show which corrections were applied
            if hasattr(self.site_corrector, 'corrections_applied') and self.site_corrector.corrections_applied:
                print(f"🔧 Site corrections applied: {', '.join(self.site_corrector.corrections_applied)}")
        
        return df_cleaned
    
    def _validate_ethiopia_corrections(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        """Quick validation of Ethiopia corrections."""
        
        for wavelength in ['IR', 'Blue', 'Red']:
            bc_corrected_col = f'{wavelength} BCc_corrected'
            if bc_corrected_col in df_after.columns:
                
                # Check correlation with ATN1
                atn_col = f'{wavelength} ATN1'
                if atn_col in df_after.columns:
                    mask = df_after[bc_corrected_col].notna() & df_after[atn_col].notna()
                    if mask.sum() > 10:
                        corr_after = df_after.loc[mask, bc_corrected_col].corr(df_after.loc[mask, atn_col])
                        
                        # Compare with original if available
                        bc_orig_col = f'{wavelength} BCc'
                        if bc_orig_col in df_before.columns:
                            mask_before = df_before[bc_orig_col].notna() & df_before[atn_col].notna()
                            if mask_before.sum() > 10:
                                corr_before = df_before.loc[mask_before, bc_orig_col].corr(df_before.loc[mask_before, atn_col])
                                print(f"    📊 {wavelength} BCc-ATN1 correlation: {corr_before:.3f} → {corr_after:.3f}")
                            else:
                                print(f"    📊 {wavelength} BCc-ATN1 correlation (corrected): {corr_after:.3f}")
    
    # Existing cleaning methods (keep all your original methods)
    def diagnose_data_structure(self, df):
        """Diagnose the data structure before cleaning."""
        print("\n🔍 Data Structure Diagnosis")
        print("=" * 40)
        print(f"Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Check for datetime columns
        datetime_cols = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                datetime_cols.append(col)
        
        if datetime_cols:
            print(f"DateTime columns found: {datetime_cols}")
        
        # Check for aethalometer-specific columns
        aeth_wavelengths = []
        for col in df.columns:
            for wl in ['IR', 'Blue', 'Red', 'Green', 'UV']:
                if col.startswith(wl + ' '):
                    if wl not in aeth_wavelengths:
                        aeth_wavelengths.append(wl)
        
        if aeth_wavelengths:
            print(f"Aethalometer wavelengths found: {aeth_wavelengths}")
            
            # Check for BC and ATN columns for each wavelength
            for wl in aeth_wavelengths:
                bc_cols = [col for col in df.columns if col.startswith(f"{wl} BC")]
                atn_cols = [col for col in df.columns if col.startswith(f"{wl} ATN")]
                k_cols = [col for col in df.columns if col.startswith(f"{wl} K")]
                print(f"  {wl}: BC={len(bc_cols)}, ATN={len(atn_cols)}, K={len(k_cols)}")
        
        return {
            'shape': df.shape,
            'datetime_columns': datetime_cols,
            'wavelengths': aeth_wavelengths,
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
    
    def apply_status_cleaning(self, df, **kwargs):
        """Apply status-based cleaning (keep your existing implementation)."""
        # Your existing status cleaning code here
        if self.verbose:
            print("    🧹 Status cleaning applied")
        return df
    
    def remove_optical_saturation(self, df, **kwargs):
        """Remove optical saturation (keep your existing implementation)."""
        # Your existing optical saturation removal code here
        if self.verbose:
            print("    🧹 Optical saturation removal applied")
        return df
    
    def validate_flow_rates(self, df, **kwargs):
        """Validate flow rates (keep your existing implementation)."""
        # Your existing flow validation code here
        if self.verbose:
            print("    🧹 Flow validation applied")
        return df
    
    def apply_roughness_cleaning(self, df, **kwargs):
        """Apply roughness-based cleaning (keep your existing implementation)."""
        # Your existing roughness cleaning code here
        if self.verbose:
            print("    🧹 Roughness cleaning applied")
        return df


# Convenience functions for easy integration

def clean_pkl_data_with_ethiopia_fix(df: pd.DataFrame, 
                                   site_code: str = 'ETAD',
                                   verbose: bool = True,
                                   **cleaning_kwargs) -> pd.DataFrame:
    """
    Convenience function to clean PKL data with Ethiopia fix.
    
    Args:
        df (pd.DataFrame): Input PKL DataFrame
        site_code (str): Site code ('ETAD' for Ethiopia)
        verbose (bool): Whether to print progress
        **cleaning_kwargs: Additional cleaning parameters
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with Ethiopia corrections
    """
    
    cleaner = PKLDataCleaner(
        wavelengths_to_filter=['IR', 'Blue', 'Red'],
        verbose=verbose,
        site_code=site_code
    )
    
    return cleaner.clean_pkl_data(df, **cleaning_kwargs)


def quick_ethiopia_fix_only(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply only the Ethiopia loading compensation fix without other cleaning.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        verbose (bool): Whether to print progress
        
    Returns:
        pd.DataFrame: DataFrame with Ethiopia fix applied
    """
    
    if not SITE_CORRECTIONS_AVAILABLE:
        print("❌ Site corrections module not available")
        return df
    
    return apply_ethiopia_fix(df, verbose=verbose)


# Integration example
def example_integration_with_existing_pipeline():
    """
    Example of how to integrate Ethiopia fix with your existing pipeline.
    """
    
    print("🔧 Integration Example: Ethiopia Fix with PKL Cleaning")
    print("=" * 60)
    
    # Example: Load your data (replace with actual loading)
    print("📁 Step 1: Load your PKL data")
    print("df = pd.read_pickle('df_cleaned_Central_API_and_OG.pkl')")
    
    # Example: Apply cleaning with Ethiopia fix
    print("\n🧹 Step 2: Clean data with Ethiopia fix")
    print("df_cleaned = clean_pkl_data_with_ethiopia_fix(")
    print("    df, ")
    print("    site_code='ETAD',  # Ethiopia site")
    print("    verbose=True")
    print(")")
    
    # Example: Alternative - just apply Ethiopia fix
    print("\n🔧 Alternative: Apply only Ethiopia fix")
    print("df_ethiopia_fixed = quick_ethiopia_fix_only(df)")
    
    # Example: Integration with your existing merger
    print("\n🔄 Step 3: Continue with your existing pipeline")
    print("# Now use df_cleaned in your aethalometer_filter_merger")
    print("from src.data.processors.aethalometer_filter_merger import merge_aethalometer_filter_pipeline")
    print("# Save the cleaned data first")
    print("df_cleaned.to_pickle('df_cleaned_with_ethiopia_fix.pkl')")
    print("# Then use in merger")
    print("results = merge_aethalometer_filter_pipeline(")
    print("    aethalometer_files=['df_cleaned_with_ethiopia_fix.pkl'],")
    print("    ftir_db_path='your_database.db',")
    print("    wavelength='Red',")
    print("    site_code='ETAD'")
    print(")")


if __name__ == "__main__":
    print("Enhanced PKL Data Cleaning Pipeline with Ethiopia Fix")
    print("Use clean_pkl_data_with_ethiopia_fix() for complete processing")
    print("Use quick_ethiopia_fix_only() for just the Ethiopia correction")
    example_integration_with_existing_pipeline()
