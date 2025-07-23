# src/notebook_utils/pkl_cleaning_integration.py
"""
Integration module to add PKL cleaning functionality to the NotebookSetup class.

This module extends the existing NotebookSetup class with PKL data cleaning capabilities.
"""

from typing import Dict, Optional, Any
import pandas as pd

from data.qc.pkl_cleaning import PKLDataCleaner, load_and_clean_pkl_data


class PKLCleaningMixin:
    """
    Mixin class to add PKL cleaning functionality to NotebookSetup.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pkl_cleaner = None
        self.cleaning_results = {}
    
    def initialize_pkl_cleaner(self, wavelengths_to_filter=None, verbose=True):
        """
        Initialize the PKL data cleaner.
        
        Args:
            wavelengths_to_filter (list): Wavelengths to process
            verbose (bool): Whether to show cleaning progress
        """
        self.pkl_cleaner = PKLDataCleaner(
            wavelengths_to_filter=wavelengths_to_filter,
            verbose=verbose
        )
        return self.pkl_cleaner
    
    def clean_pkl_dataset(self, dataset_name='pkl_data', 
                         wavelengths_to_filter=None,
                         store_as=None,
                         run_quality_assessment=True):
        """
        Clean a PKL dataset and optionally store it.
        
        Args:
            dataset_name (str): Name of the dataset to clean
            wavelengths_to_filter (list): Wavelengths to process
            store_as (str): Name to store cleaned dataset as
            run_quality_assessment (bool): Whether to run quality assessment
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(self.datasets.keys())}")
        
        # Initialize cleaner if not already done
        if self.pkl_cleaner is None:
            self.initialize_pkl_cleaner(wavelengths_to_filter=wavelengths_to_filter)
        
        # Get original dataset
        original_data = self.datasets[dataset_name]
        
        print(f"🧹 Cleaning PKL dataset: {dataset_name}")
        print(f"Original shape: {original_data.shape}")
        
        # Apply cleaning pipeline
        cleaned_data = self.pkl_cleaner.clean_pipeline(original_data.copy())
        
        # Store results
        storage_name = store_as or f"{dataset_name}_cleaned"
        self.datasets[storage_name] = cleaned_data
        
        # Store cleaning statistics
        self.cleaning_results[storage_name] = {
            'original_size': len(original_data),
            'cleaned_size': len(cleaned_data),
            'removed_count': len(original_data) - len(cleaned_data),
            'removal_percentage': ((len(original_data) - len(cleaned_data)) / len(original_data)) * 100,
            'wavelengths_processed': self.pkl_cleaner.wls_to_filter
        }
        
        print(f"✅ Cleaned dataset stored as: {storage_name}")
        print(f"Final shape: {cleaned_data.shape}")
        print(f"Removed: {self.cleaning_results[storage_name]['removed_count']:,} points "
              f"({self.cleaning_results[storage_name]['removal_percentage']:.2f}%)")
        
        # Run quality assessment on cleaned data if requested
        if run_quality_assessment:
            print(f"🔍 Running quality assessment on cleaned data...")
            # This would integrate with your existing quality assessment system
            # You may need to modify this based on your actual quality assessment implementation
            
        return cleaned_data
    
    def get_cleaning_summary(self):
        """
        Get a summary of all cleaning operations performed.
        
        Returns:
            dict: Summary of cleaning results
        """
        if not self.cleaning_results:
            print("No cleaning operations have been performed yet.")
            return {}
        
        print("📊 PKL Cleaning Summary")
        print("=" * 50)
        
        for dataset_name, results in self.cleaning_results.items():
            print(f"\nDataset: {dataset_name}")
            print(f"  Original size: {results['original_size']:,}")
            print(f"  Cleaned size: {results['cleaned_size']:,}")
            print(f"  Removed: {results['removed_count']:,} ({results['removal_percentage']:.2f}%)")
            print(f"  Wavelengths: {results['wavelengths_processed']}")
        
        return self.cleaning_results
    
    def load_and_clean_direct(self, directory_path, 
                             wavelengths_to_filter=None,
                             store_as='direct_loaded_cleaned',
                             **kwargs):
        """
        Load PKL data directly from directory and clean it.
        
        Args:
            directory_path (str): Path to data directory
            wavelengths_to_filter (list): Wavelengths to process
            store_as (str): Name to store the dataset as
            **kwargs: Additional parameters for loading
            
        Returns:
            pd.DataFrame: Loaded and cleaned data
        """
        print(f"📁 Loading and cleaning PKL data from: {directory_path}")
        
        cleaned_data = load_and_clean_pkl_data(
            directory_path=directory_path,
            wavelengths_to_filter=wavelengths_to_filter,
            verbose=True,
            summary=True,
            **kwargs
        )
        
        # Store in datasets
        self.datasets[store_as] = cleaned_data
        
        print(f"✅ Data loaded and cleaned, stored as: {store_as}")
        return cleaned_data


# Enhanced NotebookSetup class that includes PKL cleaning
from notebook_utils.setup import NotebookSetup as OriginalNotebookSetup

class EnhancedNotebookSetup(PKLCleaningMixin, OriginalNotebookSetup):
    """
    Enhanced NotebookSetup class with PKL cleaning capabilities.
    
    This class combines the original NotebookSetup functionality
    with PKL data cleaning capabilities.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        print("🧹 Enhanced setup with PKL cleaning capabilities loaded")
    
    def print_enhanced_summary(self):
        """Print an enhanced summary including cleaning information."""
        # Call original summary
        self.print_summary()
        
        # Add cleaning summary
        if self.cleaning_results:
            print("\n" + "="*60)
            print("🧹 PKL CLEANING RESULTS")
            print("="*60)
            self.get_cleaning_summary()


# Convenience function to create enhanced setup
def create_enhanced_setup(config=None):
    """
    Create an enhanced NotebookSetup with PKL cleaning capabilities.
    
    Args:
        config: NotebookConfig object
        
    Returns:
        EnhancedNotebookSetup: Enhanced setup object
    """
    return EnhancedNotebookSetup(config)