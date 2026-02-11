#!/usr/bin/env python3
"""
Generic Data Loader Module - Load and query filter data from pickle database
Clean, reusable interface for any parameter or site analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class FilterDataLoader:
    """
    Generic data loader class for unified filter dataset
    Provides clean, flexible interface for any analysis needs
    """
    
    def __init__(self, pkl_path: str = 'unified_filter_dataset.pkl'):
        """Initialize and load the complete dataset"""
        self.pkl_path = pkl_path
        self.data = None
        self.metadata = None
        self._loaded = False
        
        # Load the data
        self.load_all_data()
        
    def load_all_data(self):
        """Load the complete unified dataset from pickle"""
        pkl_file = Path(self.pkl_path)
        
        if not pkl_file.exists():
            raise FileNotFoundError(f"Database file not found: {self.pkl_path}")
        
        print(f"Loading complete filter dataset from {self.pkl_path}...")
        
        # Load main dataset
        self.data = pd.read_pickle(self.pkl_path)
        
        # Load metadata if available
        metadata_path = self.pkl_path.replace('.pkl', '_metadata.pkl')
        if Path(metadata_path).exists():
            self.metadata = pd.read_pickle(metadata_path)
        
        # Convert SampleDate to datetime for time series analysis
        self.data['SampleDate'] = pd.to_datetime(self.data['SampleDate'], errors='coerce')
        
        self._loaded = True
        
        # Display loading summary
        self._display_load_summary()
        
    def _display_load_summary(self):
        """Display summary of loaded data"""
        print(f"Dataset loaded successfully!")
        print(f"   Total measurements: {len(self.data):,}")
        print(f"   Unique filters: {self.data['FilterId'].nunique():,}")
        print(f"   Sites: {', '.join(sorted(self.data['Site'].unique()))}")
        print(f"   Date range: {self.data['SampleDate'].min().date()} to {self.data['SampleDate'].max().date()}")
        print(f"   Data sources: {', '.join(sorted(self.data['DataSource'].unique()))}")
        
    def get_site_data(self, site: str) -> pd.DataFrame:
        """Get all data for a specific site"""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_all_data() first.")
            
        return self.data[self.data['Site'] == site].copy()
    
    def get_parameter_data(self, site: str, parameter: str) -> pd.DataFrame:
        """
        Get time series data for a specific parameter at a specific site
        
        Args:
            site: Site code (e.g., 'ETAD', 'CHTS', 'INDH', 'USPA')
            parameter: Parameter name (e.g., 'EC_ftir', 'HIPS_Fabs', 'ChemSpec_Iron_PM2.5')
            
        Returns:
            DataFrame with columns: SampleDate, Concentration, Concentration_Units, FilterId, MDL, Uncertainty
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_all_data() first.")
        
        # Filter for site and parameter
        filtered_data = self.data[
            (self.data['Site'] == site) & 
            (self.data['Parameter'] == parameter)
        ].copy()
        
        if len(filtered_data) == 0:
            return pd.DataFrame()
        
        # Sort by date and select relevant columns
        result = filtered_data[['SampleDate', 'Concentration', 'Concentration_Units', 'FilterId', 'MDL', 'Uncertainty']].copy()
        result = result.sort_values('SampleDate').reset_index(drop=True)
        
        # Remove any rows with null dates or concentrations
        result = result.dropna(subset=['SampleDate', 'Concentration'])
        
        return result
    
    def list_available_parameters(self, site: str, data_source: Optional[str] = None) -> List[str]:
        """List all available parameters for a site"""
        site_data = self.get_site_data(site)
        
        if data_source:
            site_data = site_data[site_data['DataSource'] == data_source]
        
        return sorted(site_data['Parameter'].unique())
    
    def search_parameters(self, site: str, search_term: str) -> List[str]:
        """Search for parameters containing a specific term"""
        available_params = self.list_available_parameters(site)
        matching = [p for p in available_params if search_term.lower() in p.lower()]
        return matching
    
    def get_site_summary(self, site: str) -> Dict:
        """Get comprehensive summary for a site"""
        site_data = self.get_site_data(site)
        
        if len(site_data) == 0:
            return {}
        
        summary = {
            'site': site,
            'total_measurements': len(site_data),
            'unique_filters': site_data['FilterId'].nunique(),
            'date_range': {
                'start': site_data['SampleDate'].min(),
                'end': site_data['SampleDate'].max()
            },
            'data_sources': {},
            'parameters_by_source': {}
        }
        
        # Summary by data source
        for source in site_data['DataSource'].unique():
            source_data = site_data[site_data['DataSource'] == source]
            summary['data_sources'][source] = {
                'measurements': len(source_data),
                'filters': source_data['FilterId'].nunique(),
                'parameters': source_data['Parameter'].nunique()
            }
            summary['parameters_by_source'][source] = sorted(source_data['Parameter'].unique())
        
        return summary
    
    def get_available_sites(self) -> List[str]:
        """Get list of all available sites"""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_all_data() first.")
        return sorted(self.data['Site'].unique())
    
    def get_multiple_parameters(self, site: str, parameters: List[str]) -> Dict[str, pd.DataFrame]:
        """Get multiple parameters for a site"""
        results = {}
        
        for param in parameters:
            results[param] = self.get_parameter_data(site, param)
        
        return results

# Convenience function for quick loading
def load_filter_database(pkl_path: str = 'unified_filter_dataset.pkl'):
    """Load the complete filter database"""
    return FilterDataLoader(pkl_path=pkl_path)

if __name__ == "__main__":
    # Test the loader
    print("Testing generic data loader...")
    
    try:
        # Load complete dataset
        loader = load_filter_database()
        
        # Test basic functionality
        sites = loader.get_available_sites()
        print(f"Available sites: {sites}")
        
        if sites:
            test_site = sites[0]
            summary = loader.get_site_summary(test_site)
            print(f"{test_site} summary: {summary['total_measurements']} measurements")
            
            # Test parameter search
            ec_params = loader.search_parameters(test_site, 'EC')
            print(f"EC-like parameters: {ec_params}")
        
        print("Generic data loader test successful!")
        
    except Exception as e:
        print(f"Error testing data loader: {e}")