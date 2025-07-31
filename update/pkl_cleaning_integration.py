# Addition to src/data/qc/pkl_cleaning.py
# Add these imports and functions at the end of your existing pkl_cleaning.py file

# Enhanced processing integration
def load_and_clean_pkl_data_enhanced(pkl_file_path: str, 
                                   wavelengths_to_filter: Optional[List[str]] = None,
                                   export_path: Optional[str] = None,
                                   verbose: bool = True,
                                   **kwargs) -> pd.DataFrame:
    """
    Enhanced PKL data loading and cleaning with comprehensive preprocessing.
    
    This function combines the working notebook pipeline with the modular structure:
    1. Loads PKL data
    2. Applies comprehensive preprocessing (datetime, columns, types, sessions, deltas)
    3. Applies DEMA smoothing
    4. Runs quality control cleaning
    5. Optionally exports results
    
    Args:
        pkl_file_path (str): Path to PKL file
        wavelengths_to_filter (List[str]): Wavelengths to focus on (default: ['IR', 'Blue'])
        export_path (str, optional): Base path for export (without extension)
        verbose (bool): Enable verbose output
        **kwargs: Additional arguments for PKLDataCleaner
        
    Returns:
        pd.DataFrame: Fully processed and cleaned data
        
    Example:
        # Simple usage
        df_cleaned = load_and_clean_pkl_data_enhanced(
            'path/to/data.pkl',
            wavelengths_to_filter=['IR', 'Blue'],
            export_path='cleaned_data',
            verbose=True
        )
    """
    from .enhanced_pkl_processing import EnhancedPKLProcessor
    import pandas as pd
    
    if verbose:
        print(f"📁 Loading PKL data from: {pkl_file_path}")
    
    # Load the data
    try:
        df_raw = pd.read_pickle(pkl_file_path)
        if verbose:
            print(f"✅ Loaded {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    except Exception as e:
        raise FileNotFoundError(f"Could not load PKL file: {e}")
    
    # Process with enhanced pipeline
    processor = EnhancedPKLProcessor(
        wavelengths_to_filter=wavelengths_to_filter or ['IR', 'Blue'],
        verbose=verbose,
        **kwargs
    )
    
    df_cleaned = processor.process_pkl_data(df_raw, export_path=export_path)
    
    return df_cleaned


def create_enhanced_pkl_cleaner(wavelengths_to_filter: Optional[List[str]] = None,
                               verbose: bool = True,
                               **kwargs) -> 'EnhancedPKLProcessor':
    """
    Factory function to create an enhanced PKL processor.
    
    Args:
        wavelengths_to_filter (List[str]): Wavelengths to focus on
        verbose (bool): Enable verbose output
        **kwargs: Additional arguments for PKLDataCleaner
        
    Returns:
        EnhancedPKLProcessor: Configured processor instance
    """
    from .enhanced_pkl_processing import EnhancedPKLProcessor
    
    return EnhancedPKLProcessor(
        wavelengths_to_filter=wavelengths_to_filter or ['IR', 'Blue'],
        verbose=verbose,
        **kwargs
    )


def compare_cleaning_methods(df: pd.DataFrame, 
                           wavelengths: Optional[List[str]] = None,
                           verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Compare the original PKL cleaning method with the enhanced method.
    
    Args:
        df (pd.DataFrame): Raw PKL data
        wavelengths (List[str]): Wavelengths to process
        verbose (bool): Show comparison details
        
    Returns:
        Dict[str, pd.DataFrame]: Results from both methods
    """
    wavelengths = wavelengths or ['IR', 'Blue']
    
    if verbose:
        print("🔄 Comparing PKL cleaning methods...")
        print("=" * 60)
    
    results = {}
    
    # Method 1: Original PKL cleaning
    if verbose:
        print("\n📊 Method 1: Original PKL Cleaning")
        print("-" * 40)
    
    try:
        original_cleaner = PKLDataCleaner(wavelengths_to_filter=wavelengths, verbose=verbose)
        df_original = original_cleaner.clean_pipeline(df)
        results['original'] = df_original
        
        if verbose:
            print(f"✅ Original method: {df_original.shape}")
    except Exception as e:
        if verbose:
            print(f"❌ Original method failed: {e}")
        results['original'] = None
    
    # Method 2: Enhanced PKL cleaning
    if verbose:
        print("\n📊 Method 2: Enhanced PKL Cleaning")
        print("-" * 40)
    
    try:
        from .enhanced_pkl_processing import EnhancedPKLProcessor
        enhanced_processor = EnhancedPKLProcessor(wavelengths_to_filter=wavelengths, verbose=verbose)
        df_enhanced = enhanced_processor.process_pkl_data(df)
        results['enhanced'] = df_enhanced
        
        if verbose:
            print(f"✅ Enhanced method: {df_enhanced.shape}")
    except Exception as e:
        if verbose:
            print(f"❌ Enhanced method failed: {e}")
        results['enhanced'] = None
    
    # Comparison summary
    if verbose and all(v is not None for v in results.values()):
        print("\n📊 Comparison Summary:")
        print("=" * 60)
        original_size = len(results['original'])
        enhanced_size = len(results['enhanced'])
        
        print(f"Original method:  {original_size:,} rows")
        print(f"Enhanced method:  {enhanced_size:,} rows")
        print(f"Difference:       {enhanced_size - original_size:,} rows")
        
        if original_size > 0:
            pct_diff = ((enhanced_size - original_size) / original_size) * 100
            print(f"Percentage diff:  {pct_diff:+.2f}%")
        
        # Column comparison
        orig_cols = set(results['original'].columns)
        enh_cols = set(results['enhanced'].columns)
        
        new_cols = enh_cols - orig_cols
        removed_cols = orig_cols - enh_cols
        
        if new_cols:
            print(f"New columns:      {len(new_cols)} (e.g., {list(new_cols)[:3]})")
        if removed_cols:
            print(f"Removed columns:  {len(removed_cols)} (e.g., {list(removed_cols)[:3]})")
    
    return results


# Update the __all__ list at the bottom of pkl_cleaning.py to include:
__all__.extend([
    'load_and_clean_pkl_data_enhanced',
    'create_enhanced_pkl_cleaner', 
    'compare_cleaning_methods'
])