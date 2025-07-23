# src/notebook_utils/setup.py
"""
Simplified notebook setup utility
Replaces all the complex setup code in notebooks
"""

import sys
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our enhanced modules - use absolute imports since we're adding src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.notebook_config import NotebookConfig, ConfigurationManager, get_default_etad_config, get_validated_config
from data.loaders.enhanced_notebook_loader import EnhancedNotebookLoader
from analysis.quality.data_quality_assessment import MultiDatasetQualityAssessor

class NotebookSetup:
    """
    One-stop setup class for notebooks
    """
    
    def __init__(self, config: Optional[NotebookConfig] = None):
        """
        Initialize notebook setup
        
        Parameters:
        -----------
        config : NotebookConfig, optional
            Configuration object. If None, uses default ETAD config.
        """
        
        self.config = config or get_default_etad_config()
        self.loader = None
        self.datasets = {}
        self.quality_results = {}
        
        # Setup plotting
        self._setup_plotting()
        
        print("🚀 Aethalometer-FTIR/HIPS Pipeline with Simplified Setup")
        print("=" * 60)
        
        # Validate and print configuration
        validated_config = get_validated_config(self.config)
        self.config = validated_config
    
    def _setup_plotting(self):
        """Setup plotting style"""
        
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = self.config.figure_size
        plt.rcParams['font.size'] = self.config.font_size
        
        try:
            # Try to import and use modular plotting setup
            from config.plotting import setup_plotting_style
            setup_plotting_style()
            print("✅ Advanced plotting style configured")
        except ImportError:
            print("✅ Basic plotting style configured")
    
    def initialize_loader(self) -> EnhancedNotebookLoader:
        """
        Initialize the data loader
        
        Returns:
        --------
        EnhancedNotebookLoader
            Configured data loader
        """
        
        self.loader = EnhancedNotebookLoader(self.config)
        return self.loader
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all configured datasets
        
        Returns:
        --------
        dict
            Dictionary of loaded datasets
        """
        
        if self.loader is None:
            self.initialize_loader()
        
        print("\n" + "="*60)
        print("📁 LOADING DATASETS")
        print("="*60)
        
        self.datasets = self.loader.load_all_datasets()
        
        print(f"\n📊 LOADING SUMMARY")
        print("="*60)
        print(f"✅ Successfully loaded {len(self.datasets)} datasets")
        
        for name, df in self.datasets.items():
            print(f"   - {name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        print("="*60)
        
        return self.datasets
    
    def assess_data_quality(self) -> Dict[str, Any]:
        """
        Assess data quality for all loaded datasets
        
        Returns:
        --------
        dict
            Quality assessment results
        """
        
        if not self.datasets:
            print("⚠️ No datasets loaded. Run load_all_data() first.")
            return {}
        
        assessor = MultiDatasetQualityAssessor(self.config.quality_threshold)
        self.quality_results = assessor.assess_all_datasets(self.datasets)
        
        return self.quality_results
    
    def get_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Get a specific dataset
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
            
        Returns:
        --------
        pd.DataFrame or None
            Requested dataset
        """
        
        return self.datasets.get(dataset_name)
    
    def get_bc_data_for_wavelength(self, 
                                 dataset_name: str, 
                                 wavelength: Optional[str] = None) -> Optional[pd.Series]:
        """
        Get BC data for specific wavelength
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
        wavelength : str, optional
            Wavelength (if None, uses config wavelength)
            
        Returns:
        --------
        pd.Series or None
            BC data for the wavelength
        """
        
        wavelength = wavelength or self.config.wavelength
        df = self.get_dataset(dataset_name)
        
        if df is None:
            print(f"⚠️ Dataset {dataset_name} not found")
            return None
        
        # Look for wavelength-specific BC columns
        bc_cols = [col for col in df.columns if wavelength in col and 'BC' in col]
        
        if not bc_cols:
            print(f"⚠️ No {wavelength} BC columns found in {dataset_name}")
            print(f"📊 Available BC columns: {[col for col in df.columns if 'BC' in col]}")
            return None
        
        # Use the first matching column
        bc_col = bc_cols[0]
        print(f"📊 Using BC column: {bc_col}")
        
        return df[bc_col]
    
    def get_ftir_data(self) -> Optional[pd.DataFrame]:
        """
        Get FTIR/HIPS data
        
        Returns:
        --------
        pd.DataFrame or None
            FTIR/HIPS data
        """
        
        return self.get_dataset('ftir_hips')
    
    def get_excellent_periods(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Get excellent quality periods for a dataset
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
            
        Returns:
        --------
        pd.DataFrame or None
            Excellent periods DataFrame
        """
        
        if not self.quality_results:
            print("⚠️ No quality assessment results. Run assess_data_quality() first.")
            return None
        
        if dataset_name in self.quality_results:
            return self.quality_results[dataset_name].excellent_periods_df
        else:
            print(f"⚠️ No quality results for {dataset_name}")
            return None
    
    def print_summary(self):
        """Print comprehensive summary of loaded data and quality"""
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE DATA SUMMARY")
        print("="*80)
        
        # Configuration summary
        print(f"\n🔧 Configuration:")
        print(f"   Site: {self.config.site_code}")
        print(f"   Wavelength: {self.config.wavelength}")
        print(f"   Output format: {self.config.output_format}")
        print(f"   Quality threshold: {self.config.quality_threshold} minutes")
        
        # Data summary
        print(f"\n📁 Loaded datasets: {len(self.datasets)}")
        for name, df in self.datasets.items():
            print(f"   - {name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            if isinstance(df.index, pd.DatetimeIndex):
                print(f"     📅 Time range: {df.index.min()} to {df.index.max()}")
        
        # Quality summary
        if self.quality_results:
            print(f"\n🔍 Quality assessment:")
            for name, result in self.quality_results.items():
                print(f"   - {name}: {result.excellent_periods}/{result.total_periods} excellent periods ({result.excellent_percentage:.1f}%)")
        
        # BC columns for configured wavelength
        print(f"\n🧮 {self.config.wavelength} BC columns:")
        for name, df in self.datasets.items():
            if 'ftir' not in name.lower():
                bc_cols = [col for col in df.columns if self.config.wavelength in col and 'BC' in col]
                print(f"   - {name}: {bc_cols}")
        
        print("="*80)

# Convenience functions for direct use in notebooks
def quick_setup(custom_config: Optional[NotebookConfig] = None) -> NotebookSetup:
    """
    Quick setup function for notebooks
    
    Parameters:
    -----------
    custom_config : NotebookConfig, optional
        Custom configuration
        
    Returns:
    --------
    NotebookSetup
        Configured setup object
    """
    
    return NotebookSetup(custom_config)

def load_etad_data(quality_assessment: bool = True) -> Tuple[NotebookSetup, Dict[str, pd.DataFrame]]:
    """
    One-line function to load ETAD data with quality assessment
    
    Parameters:
    -----------
    quality_assessment : bool
        Whether to run quality assessment
        
    Returns:
    --------
    tuple
        (setup_object, datasets_dict)
    """
    
    # Create setup
    setup = quick_setup()
    
    # Load data
    datasets = setup.load_all_data()
    
    # Assess quality if requested
    if quality_assessment:
        setup.assess_data_quality()
    
    # Print summary
    setup.print_summary()
    
    return setup, datasets

def create_custom_config(site_code: str,
                        aethalometer_files: Dict[str, str],
                        ftir_db_path: str,
                        wavelength: str = 'Red',
                        **kwargs) -> NotebookConfig:
    """
    Create custom configuration for non-ETAD sites
    
    Parameters:
    -----------
    site_code : str
        Site identifier
    aethalometer_files : dict
        Dictionary of aethalometer file paths
    ftir_db_path : str
        Path to FTIR database
    wavelength : str
        Wavelength to analyze
    **kwargs
        Additional configuration parameters
        
    Returns:
    --------
    NotebookConfig
        Custom configuration
    """
    
    return ConfigurationManager.create_custom_config(
        site_code=site_code,
        aethalometer_files=aethalometer_files,
        ftir_db_path=ftir_db_path,
        wavelength=wavelength,
        **kwargs
    )

# Example usage for notebooks:
"""
# Simple one-line setup for ETAD data:
setup, datasets = load_etad_data()

# Manual setup with more control:
setup = quick_setup()
datasets = setup.load_all_data()
quality_results = setup.assess_data_quality()

# Get specific data:
pkl_data = setup.get_dataset('pkl_data')
red_bc = setup.get_bc_data_for_wavelength('pkl_data', 'Red')
ftir_data = setup.get_ftir_data()
excellent_periods = setup.get_excellent_periods('pkl_data')

# Custom configuration:
custom_config = create_custom_config(
    site_code='MYSITE',
    aethalometer_files={'data': '/path/to/data.pkl'},
    ftir_db_path='/path/to/database.db',
    wavelength='Blue'
)
setup = quick_setup(custom_config)
"""