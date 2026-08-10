"""FTIR Fabs-EC relationship analyzer"""

import pandas as pd
from typing import Dict, Any
from ...core.base import BaseAnalyzer
from ...data.processors.validation import (
    validate_columns_exist, get_valid_data_mask, validate_sample_size
)
from ...analysis.statistics.descriptive import calculate_basic_statistics
from ...analysis.correlations.pearson import calculate_pearson_correlation

class FabsECAnalyzer(BaseAnalyzer):
    """
    Analyzer for Fabs-EC relationship and MAC calculation
    """
    
    def __init__(self):
        super().__init__("FabsECAnalyzer")
        self.required_columns = ['fabs', 'ec_ftir']
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform complete Fabs-EC analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with fabs and ec_ftir columns
            
        Returns:
        --------
        Dict[str, Any]
            Complete analysis results
        """
        # Validation
        validate_columns_exist(data, self.required_columns)
        valid_mask = get_valid_data_mask(data, self.required_columns)
        validate_sample_size(valid_mask, operation_name="Fabs-EC analysis")
        
        # Extract clean data
        clean_data = data[valid_mask].copy()
        fabs = clean_data['fabs']
        ec = clean_data['ec_ftir']
        
        # Calculate MAC
        mac_values = fabs / ec
        
        # Build results
        results = {
            'sample_info': {
                'total_samples': len(data),
                'valid_samples': len(clean_data),
                'data_coverage': len(clean_data) / len(data)
            },
            'mac_statistics': calculate_basic_statistics(mac_values, 'mac'),
            'correlations': calculate_pearson_correlation(fabs, ec),
            'data_ranges': {
                'fabs': calculate_basic_statistics(fabs, 'fabs'),
                'ec': calculate_basic_statistics(ec, 'ec')
            }
        }
        
        self.results = results
        return results
    
    def get_mac_values(self, data: pd.DataFrame) -> pd.Series:
        """
        Get MAC values for external use
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.Series
            MAC values
        """
        validate_columns_exist(data, self.required_columns)
        valid_mask = get_valid_data_mask(data, self.required_columns)
        clean_data = data[valid_mask].copy()
        
        return clean_data['fabs'] / clean_data['ec_ftir']
