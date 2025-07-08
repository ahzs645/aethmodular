"""Adaptive smoothening that automatically selects the best method"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .smoothening_comparison import SmoothingComparison
from .smoothening_factory import SmoothingFactory


class AdaptiveSmoothing:
    """
    Adaptive smoothening that automatically selects the best method
    
    Analyzes data characteristics and selects the most appropriate 
    smoothening method based on performance criteria.
    """
    
    def __init__(self, selection_criterion: str = 'overall', 
                 test_sample_size: int = 1000):
        """
        Initialize adaptive smoothening
        
        Parameters:
        -----------
        selection_criterion : str
            Criterion for method selection ('overall', 'noise_reduction', 'correlation')
        test_sample_size : int
            Number of samples to use for method testing (for performance)
        """
        self.selection_criterion = selection_criterion
        self.test_sample_size = test_sample_size
        self.selected_method = None
        self.comparison_results = None
    
    def analyze(self, data: pd.DataFrame, wavelength: str = 'IR') -> Dict[str, Any]:
        """
        Automatically select and apply the best smoothening method
        
        Parameters:
        -----------
        data : pd.DataFrame
            Aethalometer data
        wavelength : str
            Wavelength to analyze
            
        Returns:
        --------
        Dict[str, Any]
            Results from the selected method plus selection metadata
        """
        print(f"🔍 Analyzing data characteristics for adaptive smoothening...")
        
        # Use a sample for method selection if data is large
        test_data = self._get_test_sample(data)
        
        # Compare all methods on test data
        comparator = SmoothingComparison()
        comparison_results = comparator.compare_methods(test_data, wavelength)
        self.comparison_results = comparison_results
        
        # Select best method
        best_method = comparator.get_best_method(self.selection_criterion)
        
        if best_method is None:
            raise ValueError("Failed to select an appropriate smoothening method")
        
        self.selected_method = best_method
        print(f"✅ Selected method: {best_method} (criterion: {self.selection_criterion})")
        
        # Apply selected method to full data
        smoother = SmoothingFactory.create_smoother(best_method)
        results = smoother.analyze(data, wavelength)
        
        # Add adaptive selection metadata
        results['adaptive_selection'] = {
            'selected_method': best_method,
            'selection_criterion': self.selection_criterion,
            'comparison_results': comparison_results['comparison_summary'],
            'data_characteristics': self._analyze_data_characteristics(data, wavelength)
        }
        
        return results
    
    def _get_test_sample(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get a representative sample for method testing"""
        if len(data) <= self.test_sample_size:
            return data
        
        # Take evenly spaced samples to maintain temporal structure
        step = len(data) // self.test_sample_size
        indices = np.arange(0, len(data), step)[:self.test_sample_size]
        return data.iloc[indices].copy()
    
    def _analyze_data_characteristics(self, data: pd.DataFrame, wavelength: str) -> Dict[str, Any]:
        """Analyze data characteristics for method selection"""
        bc_col = f"{wavelength} BCc"
        
        if bc_col not in data.columns:
            return {'error': f'Column {bc_col} not found'}
        
        bc_values = data[bc_col].dropna()
        
        if len(bc_values) < 10:
            return {'error': 'Insufficient valid data'}
        
        # Calculate characteristics
        characteristics = {
            'sample_size': len(bc_values),
            'mean_value': float(bc_values.mean()),
            'std_value': float(bc_values.std()),
            'coefficient_of_variation': float(bc_values.std() / bc_values.mean()) if bc_values.mean() != 0 else np.inf,
            'negative_values': int((bc_values < 0).sum()),
            'zero_values': int((bc_values == 0).sum()),
            'data_range': float(bc_values.max() - bc_values.min()),
            'missing_percentage': float((data[bc_col].isna().sum() / len(data)) * 100)
        }
        
        # Classify data characteristics
        characteristics['noise_level'] = self._classify_noise_level(characteristics['coefficient_of_variation'])
        characteristics['variability'] = self._classify_variability(characteristics['std_value'])
        characteristics['data_quality'] = self._classify_data_quality(characteristics['missing_percentage'])
        
        return characteristics
    
    def _classify_noise_level(self, cv: float) -> str:
        """Classify noise level based on coefficient of variation"""
        if cv < 0.2:
            return 'low'
        elif cv < 0.5:
            return 'medium'
        else:
            return 'high'
    
    def _classify_variability(self, std: float) -> str:
        """Classify variability based on standard deviation"""
        if std < 1.0:
            return 'low'
        elif std < 5.0:
            return 'medium'
        else:
            return 'high'
    
    def _classify_data_quality(self, missing_pct: float) -> str:
        """Classify data quality based on missing percentage"""
        if missing_pct < 5:
            return 'excellent'
        elif missing_pct < 15:
            return 'good'
        elif missing_pct < 30:
            return 'moderate'
        else:
            return 'poor'
    
    def get_selection_rationale(self) -> str:
        """Get explanation for method selection"""
        if not self.selected_method or not self.comparison_results:
            return "No method selected yet"
        
        summary = self.comparison_results['comparison_summary']
        
        rationale = f"Selected {self.selected_method} based on {self.selection_criterion} performance.\n"
        
        if self.selection_criterion == 'overall':
            if summary['performance_ranking']:
                top_method = summary['performance_ranking'][0]
                rationale += f"Achieved highest overall score: {top_method['score']}"
        elif self.selection_criterion == 'noise_reduction':
            if summary['best_noise_reduction']:
                best = summary['best_noise_reduction']
                rationale += f"Achieved best noise reduction: {best['value']:.1f}%"
        elif self.selection_criterion == 'correlation':
            if summary['best_correlation']:
                best = summary['best_correlation']
                rationale += f"Achieved highest correlation: {best['value']:.3f}"
        
        return rationale
