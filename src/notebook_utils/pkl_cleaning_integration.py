# src/notebook_utils/pkl_cleaning_integration.py
"""
Integration module to add PKL cleaning functionality to the NotebookSetup class.

This module extends the existing NotebookSetup class with PKL data cleaning capabilities.
"""

from typing import Dict, Optional, Any, List
import pandas as pd
import os

from data.qc.pkl_cleaning import PKLDataCleaner, load_and_clean_pkl_data
from data.loaders.ftir_csv_loader import (
    FTIRCSVLoader, 
    FTIRCSVMerger,
    auto_merge_with_csv,
    SITE_CONFIGS
)


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


class FTIRCSVMixin:
    """
    Mixin class to add FTIR CSV functionality to NotebookSetup.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ftir_loader = None
        self.ftir_merger = None
        self.ftir_results = {}
    
    def setup_ftir_csv(self, csv_path: str, verbose: bool = True):
        """
        Initialize FTIR CSV loader.
        
        Args:
            csv_path (str): Path to FTIR CSV file
            verbose (bool): Whether to show initialization info
        """
        self.ftir_loader = FTIRCSVLoader(csv_path)
        
        if verbose:
            print(f"🧪 FTIR CSV loader initialized")
            print(f"📁 CSV file: {csv_path}")
            print(f"🗺️ Available sites: {self.ftir_loader.get_available_sites()}")
            print(f"🧪 Available parameters: {self.ftir_loader.get_available_parameters()}")
        
        return self.ftir_loader
    
    def load_ftir_data(self, site_code: str, 
                      parameters: Optional[List[str]] = None,
                      date_range: Optional[tuple] = None,
                      store_as: Optional[str] = None) -> pd.DataFrame:
        """
        Load FTIR data for a specific site.
        
        Args:
            site_code (str): Site identifier (e.g., 'ETAD', 'JPL')
            parameters (List[str]): Parameters to include (e.g., ['EC_ftir', 'OC_ftir'])
            date_range (tuple): (start_date, end_date) for filtering
            store_as (str): Name to store dataset as
            
        Returns:
            pd.DataFrame: FTIR data
        """
        if self.ftir_loader is None:
            raise ValueError("FTIR loader not initialized. Call setup_ftir_csv() first.")
        
        ftir_data = self.ftir_loader.load_site_data(
            site_code=site_code,
            parameters=parameters,
            date_range=date_range
        )
        
        # Store in datasets if requested
        storage_name = store_as or f"ftir_{site_code.lower()}"
        self.datasets[storage_name] = ftir_data
        
        print(f"✅ FTIR data stored as: {storage_name}")
        return ftir_data
    
    def merge_with_ftir(self, aethalometer_dataset: str,
                       site_code: str,
                       csv_path: Optional[str] = None,
                       store_as: Optional[str] = None,
                       auto_merge: bool = True) -> pd.DataFrame:
        """
        Merge aethalometer data with FTIR data.
        
        Args:
            aethalometer_dataset (str): Name of aethalometer dataset
            site_code (str): Site identifier
            csv_path (str): Path to FTIR CSV (if not using existing loader)
            store_as (str): Name to store merged dataset
            auto_merge (bool): Use automatic site configuration
            
        Returns:
            pd.DataFrame: Merged dataset
        """
        if aethalometer_dataset not in self.datasets:
            raise ValueError(f"Dataset '{aethalometer_dataset}' not found")
        
        aethalometer_data = self.datasets[aethalometer_dataset]
        
        if auto_merge:
            # Use automatic merging with site configurations
            ftir_csv_path = csv_path or getattr(self.config, 'ftir_csv_path', None)
            if not ftir_csv_path:
                raise ValueError("FTIR CSV path not provided and not found in configuration")
            
            merged_data = auto_merge_with_csv(
                aethalometer_daily=aethalometer_data,
                csv_path=ftir_csv_path,
                site_code=site_code
            )
        else:
            # Manual merging (requires FTIR loader to be set up)
            if self.ftir_loader is None:
                raise ValueError("FTIR loader not initialized for manual merging")
            
            if site_code not in SITE_CONFIGS:
                raise ValueError(f"Unknown site code '{site_code}'. Available: {list(SITE_CONFIGS.keys())}")
            
            config = SITE_CONFIGS[site_code]
            merger = FTIRCSVMerger(
                csv_path=csv_path or self.ftir_loader.csv_path,
                site_code=site_code,
                timezone=config['timezone']
            )
            
            merged_data = merger.merge_with_aethalometer(aethalometer_data)
        
        # Convert units (ng/m³ to µg/m³) if needed
        if 'EC_ftir' in merged_data.columns:
            merged_data['EC_ftir'] = merged_data['EC_ftir'] * 1000
            print("🔄 Converted EC_ftir units: ng/m³ → µg/m³")
        
        # Store merged data
        storage_name = store_as or f"{aethalometer_dataset}_ftir_merged"
        self.datasets[storage_name] = merged_data
        
        # Store merge statistics
        self.ftir_results[storage_name] = {
            'aethalometer_points': len(aethalometer_data),
            'merged_points': len(merged_data),
            'site_code': site_code,
            'ftir_parameters': [col for col in merged_data.columns if 'ftir' in col.lower()]
        }
        
        print(f"✅ Merged data stored as: {storage_name}")
        print(f"📊 Merge success: {len(merged_data)} matching days")
        
        return merged_data
    
    def get_ftir_summary(self):
        """Get summary of FTIR operations."""
        if not self.ftir_results:
            print("No FTIR merge operations have been performed yet.")
            return {}
        
        print("🧪 FTIR Merge Summary")
        print("=" * 50)
        
        for dataset_name, results in self.ftir_results.items():
            print(f"\nDataset: {dataset_name}")
            print(f"  Site: {results['site_code']}")
            print(f"  Aethalometer points: {results['aethalometer_points']:,}")
            print(f"  Merged points: {results['merged_points']:,}")
            print(f"  Success rate: {(results['merged_points']/results['aethalometer_points']*100):.1f}%")
            print(f"  FTIR parameters: {results['ftir_parameters']}")
        
        return self.ftir_results
    
    def explore_ftir_data(self, csv_path: Optional[str] = None):
        """
        Explore available FTIR data in the CSV file.
        
        Args:
            csv_path (str): Path to FTIR CSV file
        """
        if csv_path:
            loader = FTIRCSVLoader(csv_path)
        elif self.ftir_loader:
            loader = self.ftir_loader
        else:
            raise ValueError("No FTIR CSV path provided and no loader initialized")
        
        print("🧪 FTIR Data Exploration")
        print("=" * 40)
        
        sites = loader.get_available_sites()
        print(f"📍 Available sites ({len(sites)}): {', '.join(sites)}")
        
        parameters = loader.get_available_parameters()
        print(f"🧪 Available parameters ({len(parameters)}): {', '.join(parameters)}")
        
        print(f"\n📊 Site Details:")
        for site in sites:
            try:
                site_info = loader.get_site_info(site)
                print(f"  {site}:")
                print(f"    📍 Location: {site_info['latitude']:.3f}, {site_info['longitude']:.3f}")
                print(f"    📅 Date range: {site_info['date_range'][0].strftime('%Y-%m-%d')} to {site_info['date_range'][1].strftime('%Y-%m-%d')}")
                print(f"    📊 Samples: {site_info['sample_count']}")
                print(f"    🧪 Parameters: {len(site_info['parameters'])}")
            except Exception as e:
                print(f"  {site}: Error - {e}")


# Enhanced NotebookSetup class that includes PKL cleaning and FTIR CSV
from notebook_utils.setup import NotebookSetup as OriginalNotebookSetup

class EnhancedNotebookSetup(FTIRCSVMixin, PKLCleaningMixin, OriginalNotebookSetup):
    """
    Enhanced NotebookSetup class with PKL cleaning and FTIR CSV capabilities.
    
    This class combines the original NotebookSetup functionality
    with PKL data cleaning and FTIR CSV loading capabilities.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        print("🧹 Enhanced setup with PKL cleaning and FTIR CSV capabilities loaded")
    
    def print_enhanced_summary(self):
        """Print an enhanced summary including cleaning and FTIR information."""
        # Call original summary
        self.print_summary()
        
        # Add cleaning summary
        if self.cleaning_results:
            print("\n" + "="*60)
            print("🧹 PKL CLEANING RESULTS")
            print("="*60)
            self.get_cleaning_summary()
            
        # Add FTIR summary
        if self.ftir_results:
            print("\n" + "="*60)
            print("🧪 FTIR MERGE RESULTS")
            print("="*60)
            self.get_ftir_summary()
    
    def quick_ftir_setup(self, csv_path: str):
        """
        Quick setup for FTIR CSV functionality.
        
        Args:
            csv_path (str): Path to FTIR CSV file
        """
        self.setup_ftir_csv(csv_path)
        self.explore_ftir_data()


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