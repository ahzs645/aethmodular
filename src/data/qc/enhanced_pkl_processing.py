# Enhanced PKL Data Processing Module
# src/data/qc/enhanced_pkl_processing.py

import pandas as pd
import numpy as np
import os
import sys
from typing import List, Optional, Dict, Tuple
from .pkl_cleaning import PKLDataCleaner

# Try to import calibration module
try:
    import calibration
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False
    print("⚠️ Calibration module not found, using fallback methods")

class EnhancedPKLProcessor:
    """
    Enhanced PKL data processor that combines comprehensive preprocessing,
    DEMA smoothing, and quality control cleaning into a modular pipeline.
    """
    
    def __init__(self, wavelengths_to_filter=None, verbose=True, apply_ethiopia_fix=False, site_code=None, **kwargs):
        """
        Initialize the enhanced PKL processor.
        
        Args:
            wavelengths_to_filter (List[str]): Wavelengths to focus on (e.g., ['IR', 'Blue'])
            verbose (bool): Enable verbose output
            apply_ethiopia_fix (bool): Whether to apply Ethiopia pneumatic pump loading compensation fix
            site_code (str): Site code for site-specific corrections (e.g., 'ETAD' for Ethiopia)
            **kwargs: Additional arguments passed to PKLDataCleaner
        """
        self.wavelengths_to_filter = wavelengths_to_filter or ['IR', 'Blue']
        self.verbose = verbose
        self.apply_ethiopia_fix = apply_ethiopia_fix
        self.site_code = site_code or ('ETAD' if apply_ethiopia_fix else None)
        self.cleaner = PKLDataCleaner(
            wavelengths_to_filter=wavelengths_to_filter, 
            verbose=verbose, 
            **kwargs
        )
    
    def comprehensive_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive preprocessing pipeline including datetime handling,
        column renaming, data type conversion, session ID, and delta calculations.
        
        Args:
            df (pd.DataFrame): Raw PKL data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.verbose:
            print("🔧 Comprehensive Preprocessing Pipeline")
            print("=" * 60)
        
        df_processed = df.copy()
        original_size = len(df_processed)
        
        # Step 1: Fix datetime column
        if self.verbose:
            print("Step 1: Processing datetime...")
        
        if 'datetime_local' in df_processed.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_processed['datetime_local']):
                df_processed['datetime_local'] = pd.to_datetime(
                    df_processed['datetime_local'], utc=True
                ).dt.tz_convert('Africa/Addis_Ababa')
                if self.verbose:
                    print("✅ Converted datetime_local to proper timezone")
        elif df_processed.index.name == 'datetime_local':
            df_processed['datetime_local'] = df_processed.index
            df_processed = df_processed.reset_index(drop=True)
            if self.verbose:
                print("✅ Converted datetime_local from index to column")
        elif hasattr(df_processed.index, 'tz'):
            df_processed['datetime_local'] = df_processed.index
            df_processed = df_processed.reset_index(drop=True)
            if self.verbose:
                print("✅ Created datetime_local column from datetime index")
        
        # Step 2: Column renaming
        if self.verbose:
            print("\nStep 2: Fixing column names...")
        
        column_mapping = {}
        
        # Map BC columns (handle both BC1->BCc conversion and dot notation)
        for wl in ['IR', 'Blue', 'Green', 'Red', 'UV']:
            # First priority: use .BCc if available
            if f'{wl}.BCc' in df_processed.columns:
                column_mapping[f'{wl}.BCc'] = f'{wl} BCc'
            # Second priority: rename BC1 to BCc
            elif f'{wl} BC1' in df_processed.columns:
                df_processed = df_processed.rename(columns={f'{wl} BC1': f'{wl} BCc'})
                if self.verbose:
                    print(f"  Renamed {wl} BC1 -> {wl} BCc")
        
        # Map ATN columns (dots to spaces)
        for wl in ['IR', 'Blue', 'Green', 'Red', 'UV']:
            for spot in [1, 2]:
                if f'{wl}.ATN{spot}' in df_processed.columns:
                    column_mapping[f'{wl}.ATN{spot}'] = f'{wl} ATN{spot}'
        
        # Map flow columns
        if 'Flow.total.mL.min' in df_processed.columns:
            column_mapping['Flow.total.mL.min'] = 'Flow total (mL/min)'
        
        # Apply column renaming
        if column_mapping:
            df_processed = df_processed.rename(columns=column_mapping)
            if self.verbose:
                print(f"✅ Renamed {len(column_mapping)} columns")
        
        # Step 3: Data type conversion
        if self.verbose:
            print("\nStep 3: Converting data types...")
        
        if HAS_CALIBRATION:
            df_processed = calibration.convert_to_float(df_processed)
            if self.verbose:
                print("✅ Applied calibration.convert_to_float()")
        else:
            # Manual data type conversion
            numeric_cols = []
            for col in df_processed.columns:
                if any(x in col for x in ['ATN', 'BC', 'Flow', 'temp', 'Temp']):
                    if df_processed[col].dtype == 'object':
                        try:
                            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                            numeric_cols.append(col)
                        except:
                            pass
            if self.verbose:
                print(f"✅ Converted {len(numeric_cols)} columns to numeric")
        
        # Step 4: Add Session ID
        if self.verbose:
            print("\nStep 4: Adding Session ID...")
        
        if 'Session ID' not in df_processed.columns and 'Tape position' in df_processed.columns:
            position_change = df_processed['Tape position'] != df_processed['Tape position'].shift()
            df_processed['Session ID'] = position_change.cumsum()
            if self.verbose:
                print("✅ Added Session ID based on tape position changes")
        
        # Step 5: Add delta calculations
        if self.verbose:
            print("\nStep 5: Adding delta calculations...")
        
        if HAS_CALIBRATION:
            df_processed = calibration.add_deltas(df_processed)
            if self.verbose:
                print("✅ Applied calibration.add_deltas()")
        else:
            # Manual delta calculation for critical columns
            if self.verbose:
                print("⚠️ Manual delta calculation (limited functionality)")
            
            attn_cols = [col for col in df_processed.columns if 'ATN' in col and col.count(' ') == 1]
            for col in attn_cols:
                try:
                    if 'Serial number' in df_processed.columns and 'Session ID' in df_processed.columns:
                        df_processed[f'delta {col}'] = (
                            df_processed.groupby(['Serial number', 'Session ID'])[col].diff()
                        )
                    else:
                        df_processed[f'delta {col}'] = df_processed[col].diff()
                except:
                    pass
            
            if self.verbose:
                print(f"✅ Added basic delta calculations for {len(attn_cols)} ATN columns")
        
        # Step 6: Set serial number and filter by year
        if self.verbose:
            print("\nStep 6: Final adjustments...")
        
        df_processed['Serial number'] = "MA350-0238"
        
        if 'datetime_local' in df_processed.columns:
            df_processed = df_processed.loc[df_processed['datetime_local'].dt.year >= 2022]
            if self.verbose:
                print(f"✅ Filtered to 2022+: {original_size:,} -> {len(df_processed):,} rows")
        
        return df_processed
    
    def apply_dema_smoothing(self, df: pd.DataFrame, wavelengths: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply DEMA (Double Exponential Moving Average) smoothing to BC columns.
        
        Args:
            df (pd.DataFrame): Preprocessed data
            wavelengths (List[str]): Wavelengths to smooth (defaults to self.wavelengths_to_filter)
            
        Returns:
            pd.DataFrame: Data with DEMA smoothed columns
        """
        if wavelengths is None:
            wavelengths = self.wavelengths_to_filter
        
        if self.verbose:
            print("🔄 Applying DEMA Smoothing...")
            print("=" * 40)
        
        df_smoothed = df.copy()
        
        for wl in wavelengths:
            if self.verbose:
                print(f"\nProcessing {wl} wavelength...")
            
            # Check what BC columns we have
            bc_cols = [col for col in df_smoothed.columns 
                      if wl in col and 'BC' in col and 'smoothed' not in col]
            
            if self.verbose:
                print(f"  Available BC columns: {bc_cols}")
            
            if not bc_cols:
                if self.verbose:
                    print(f"  ⚠️ No BC columns found for {wl}")
                continue
            
            # Process each BC column
            for bc_col in bc_cols:
                try:
                    # Group by measurement sessions for proper smoothing
                    groupby_cols = ['Serial number']
                    if 'Session ID' in df_smoothed.columns:
                        groupby_cols.append('Session ID')
                    if 'Tape position' in df_smoothed.columns:
                        groupby_cols.append('Tape position')
                    
                    smoothed_values = []
                    
                    for group_keys, group in df_smoothed.groupby(groupby_cols):
                        if len(group) > 2:  # Need at least 3 points for smoothing
                            values = group[bc_col].dropna()
                            if len(values) > 1:
                                # Apply DEMA algorithm
                                span = min(10, len(values) // 2)  # Adaptive span
                                if span < 2:
                                    span = 2
                                
                                # First EMA
                                ema1 = values.ewm(span=span, adjust=False).mean()
                                # Second EMA (EMA of EMA)
                                ema2 = ema1.ewm(span=span, adjust=False).mean()
                                # DEMA = 2*EMA1 - EMA2
                                dema = 2 * ema1 - ema2
                                
                                # Store results with original indices
                                for idx, val in dema.items():
                                    smoothed_values.append((idx, val))
                            else:
                                # Not enough data, use original values
                                for idx, val in values.items():
                                    smoothed_values.append((idx, val))
                        else:
                            # Very small group, use original values
                            values = group[bc_col].dropna()
                            for idx, val in values.items():
                                smoothed_values.append((idx, val))
                    
                    # Create smoothed column
                    smoothed_col = f'{bc_col} smoothed'
                    df_smoothed[smoothed_col] = np.nan
                    
                    for idx, val in smoothed_values:
                        df_smoothed.loc[idx, smoothed_col] = val
                    
                    if self.verbose:
                        print(f"  ✅ Created {smoothed_col}")
                    
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠️ Failed to smooth {bc_col}: {e}")
        
        return df_smoothed
    
    def apply_site_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply site-specific corrections if enabled.
        
        Args:
            df (pd.DataFrame): Data to correct
            
        Returns:
            pd.DataFrame: Data with site corrections applied
        """
        if not self.apply_ethiopia_fix:
            return df
        
        try:
            # Try multiple import paths for site corrections
            corrector = None
            
            try:
                from ..processors.site_corrections import SiteCorrections
                corrector = SiteCorrections(site_code=self.site_code, verbose=self.verbose)
            except ImportError:
                try:
                    from data.processors.site_corrections import SiteCorrections
                    corrector = SiteCorrections(site_code=self.site_code, verbose=self.verbose)
                except ImportError:
                    import sys
                    sys.path.append('src')
                    from data.processors.site_corrections import SiteCorrections
                    corrector = SiteCorrections(site_code=self.site_code, verbose=self.verbose)
            
            if corrector is None:
                raise ImportError("Could not import SiteCorrections")
            
            if self.verbose:
                print("\n🔧 Applying Site-Specific Corrections...")
                print("=" * 50)
            
            df_corrected = corrector.apply_corrections(df)
            
            if self.verbose:
                corrections_applied = getattr(corrector, 'corrections_applied', [])
                if corrections_applied:
                    print(f"✅ Applied corrections: {', '.join(corrections_applied)}")
                else:
                    print("⚠️ No corrections were applied - check site code or data")
                print("✅ Site corrections step completed!")
            
            return df_corrected
            
        except ImportError as e:
            if self.verbose:
                print(f"\n⚠️ Site corrections module not found: {e}")
                print("⚠️ Make sure site_corrections.py exists in src/data/processors/")
                print("Continuing without site corrections...")
            return df
        except Exception as e:
            if self.verbose:
                print(f"\n⚠️ Error applying site corrections: {e}")
                print("Continuing without site corrections...")
            return df
    
    def process_pkl_data(self, df: pd.DataFrame, export_path: Optional[str] = None) -> pd.DataFrame:
        """
        Complete PKL data processing pipeline: preprocessing → smoothing → cleaning.
        
        Args:
            df (pd.DataFrame): Raw PKL data
            export_path (str, optional): Path to export cleaned data (without extension)
            
        Returns:
            pd.DataFrame: Fully processed and cleaned data
        """
        if self.verbose:
            print("🚀 Enhanced PKL Data Processing Pipeline")
            print("=" * 60)
        
        original_size = len(df)
        
        # Step 1: Comprehensive preprocessing
        df_preprocessed = self.comprehensive_preprocessing(df)
        
        # Step 2: DEMA smoothing
        df_smoothed = self.apply_dema_smoothing(df_preprocessed)
        
        # Step 3: Site-specific corrections (e.g., Ethiopia fix)
        df_corrected = self.apply_site_corrections(df_smoothed)
        
        # Step 4: Quality control cleaning
        if self.verbose:
            print("\n🧹 Final Cleaning Pipeline")
            print("=" * 60)
        
        df_cleaned = self.cleaner.clean_pipeline(df_corrected, skip_preprocessing=True)
        
        # Summary
        if self.verbose:
            print("\n📊 Processing Results Summary:")
            print("=" * 60)
            print(f"Original data points: {original_size:,}")
            print(f"After preprocessing: {len(df_preprocessed):,}")
            print(f"After smoothing: {len(df_smoothed):,}")
            print(f"After site corrections: {len(df_corrected):,}")
            print(f"Final cleaned: {len(df_cleaned):,}")
            
            total_removed = original_size - len(df_cleaned)
            removal_pct = (total_removed / original_size * 100)
            print(f"Total removed: {total_removed:,} ({removal_pct:.2f}%)")
            
            print("\n✅ PKL data processing completed successfully!")
            
            # Final verification
            print(f"\n📊 Final data verification:")
            print(f"Shape: {df_cleaned.shape}")
            if 'datetime_local' in df_cleaned.columns:
                print(f"Date range: {df_cleaned['datetime_local'].min()} to {df_cleaned['datetime_local'].max()}")
            
            # Check for key columns
            key_cols = ['IR ATN1', 'IR BCc', 'Blue ATN1', 'Blue BCc', 'Flow total (mL/min)']
            for col in key_cols:
                status = "✅" if col in df_cleaned.columns else "❌"
                print(f"  {status} {col}")
            
            # Check smoothed columns
            smoothed_cols = [col for col in df_cleaned.columns if 'smoothed' in col]
            print(f"  ✅ Smoothed columns: {len(smoothed_cols)}")
        
        # Export if requested
        if export_path:
            self.export_cleaned_data(df_cleaned, export_path)
        
        return df_cleaned
    
    def export_cleaned_data(self, df: pd.DataFrame, base_path: str) -> Dict[str, str]:
        """
        Export cleaned data to pickle format only.
        
        Args:
            df (pd.DataFrame): Cleaned data
            base_path (str): Base path for export (without extension)
            
        Returns:
            Dict[str, str]: Path of exported file
        """
        output_pkl = f'{base_path}.pkl'
        
        df.to_pickle(output_pkl)
        
        if self.verbose:
            print(f"\n💾 Cleaned data exported:")
            print(f"  📦 Pickle: {output_pkl}")
        
        return {
            'pickle': output_pkl
        }

# Convenience function for easy use
def process_pkl_data_enhanced(df: pd.DataFrame, 
                             wavelengths_to_filter: Optional[List[str]] = None,
                             export_path: Optional[str] = None,
                             verbose: bool = True,
                             apply_ethiopia_fix: bool = False,
                             site_code: Optional[str] = None,
                             **kwargs) -> pd.DataFrame:
    """
    Convenience function for enhanced PKL data processing.
    
    Args:
        df (pd.DataFrame): Raw PKL data
        wavelengths_to_filter (List[str]): Wavelengths to focus on
        export_path (str, optional): Path to export cleaned data
        verbose (bool): Enable verbose output
        apply_ethiopia_fix (bool): Whether to apply Ethiopia pneumatic pump loading compensation fix
        site_code (str): Site code for site-specific corrections (e.g., 'ETAD' for Ethiopia)
        **kwargs: Additional arguments for PKLDataCleaner
        
    Returns:
        pd.DataFrame: Fully processed and cleaned data
    """
    processor = EnhancedPKLProcessor(
        wavelengths_to_filter=wavelengths_to_filter or ['IR', 'Blue'],
        verbose=verbose,
        apply_ethiopia_fix=apply_ethiopia_fix,
        site_code=site_code,
        **kwargs
    )
    
    return processor.process_pkl_data(df, export_path=export_path)

# Integration with existing setup
def create_enhanced_pkl_setup(config, **kwargs):
    """
    Create an enhanced PKL processor using notebook configuration.
    
    Args:
        config: NotebookConfig instance
        **kwargs: Additional arguments for EnhancedPKLProcessor
        
    Returns:
        EnhancedPKLProcessor: Configured processor
    """
    # Extract wavelengths from config if available
    wavelengths = getattr(config, 'wavelengths_to_filter', ['IR', 'Blue'])
    if hasattr(config, 'wavelength') and config.wavelength:
        if isinstance(config.wavelength, str):
            wavelengths = [config.wavelength]
        elif isinstance(config.wavelength, list):
            wavelengths = config.wavelength
    
    return EnhancedPKLProcessor(
        wavelengths_to_filter=wavelengths,
        verbose=True,
        **kwargs
    )