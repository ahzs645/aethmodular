"""
OC-EC Relationship Analyzer

This module demonstrates how to migrate from the old monolithic function
to the new modular structure.
"""

import pandas as pd
from typing import Dict, Any
from ...core.base import BaseAnalyzer
from ...data.processors.validation import (
    validate_columns_exist, get_valid_data_mask, validate_sample_size
)
from ...analysis.statistics.descriptive import calculate_basic_statistics
from ...analysis.correlations.pearson import calculate_pearson_correlation
from ...core.exceptions import InsufficientDataError

class OCECAnalyzer(BaseAnalyzer):
    """
    Analyzer for OC-EC relationship
    
    This replaces the old `analyze_oc_ec_relationship` function with
    a modular, testable approach.
    """
    
    def __init__(self):
        super().__init__("OCECAnalyzer")
        self.required_columns = ['oc_ftir', 'ec_ftir']
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze OC-EC relationship
        
        This breaks down the original large function into smaller,
        focused components.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with oc_ftir and ec_ftir columns
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results
        """
        # Step 1: Validation (extracted to separate module)
        validate_columns_exist(data, self.required_columns)
        valid_mask = get_valid_data_mask(data, self.required_columns)
        
        try:
            validate_sample_size(valid_mask, operation_name="OC-EC analysis")
        except InsufficientDataError as e:
            return {
                'error': str(e),
                'sample_info': {
                    'total_samples': len(data),
                    'valid_samples': 0
                }
            }
        
        # Step 2: Extract clean data
        clean_data = data[valid_mask].copy()
        oc = clean_data['oc_ftir']
        ec = clean_data['ec_ftir']
        
        # Step 3: Calculate basic statistics (using reusable module)
        oc_stats = calculate_basic_statistics(oc, 'oc')
        ec_stats = calculate_basic_statistics(ec, 'ec')
        
        # Step 4: Calculate OC/EC ratio statistics
        oc_ec_ratio = oc / ec
        ratio_stats = calculate_basic_statistics(oc_ec_ratio, 'oc_ec_ratio')
        
        # Step 5: Calculate correlations (using reusable module)
        try:
            correlations = calculate_pearson_correlation(oc, ec)
        except Exception as e:
            correlations = {'error': str(e)}
        
        # Step 6: Build comprehensive results
        results = {
            'sample_info': {
                'total_samples': len(data),
                'valid_samples': len(clean_data),
                'data_coverage': len(clean_data) / len(data)
            },
            'correlations': correlations,
            'statistics': {
                'oc': oc_stats,
                'ec': ec_stats,
                'oc_ec_ratio': ratio_stats
            }
        }
        
        self.results = results
        return results
    
    def get_oc_ec_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Get OC/EC ratio values for external use
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.Series
            OC/EC ratio values
        """
        validate_columns_exist(data, self.required_columns)
        valid_mask = get_valid_data_mask(data, self.required_columns)
        clean_data = data[valid_mask].copy()
        
        return clean_data['oc_ftir'] / clean_data['ec_ftir']
    
    def get_correlation_summary(self) -> Dict[str, float]:
        """
        Get simplified correlation summary
        
        Returns:
        --------
        Dict[str, float]
            Correlation summary
        """
        if not self.results or 'correlations' not in self.results:
            return {}
        
        corr = self.results['correlations']
        if 'error' in corr:
            return {'error': corr['error']}
        
        return {
            'pearson_r': corr['pearson_r'],
            'pearson_p': corr['pearson_p'],
            'n_samples': corr['n_samples'],
            'significant': corr['pearson_p'] < 0.05
        }
