"""Test suite for smoothening algorithms"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.analysis.aethalometer.smoothening import (
    ONASmoothing, CMASmoothing, DEMASmoothing,
    SmoothingFactory, SmoothingComparison, AdaptiveSmoothing
)


class TestONASmoothing:
    """Test ONA (Optimized Noise-reduction Algorithm) smoothening"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.ona = ONASmoothing(delta_atn_threshold=0.05)
    
    def test_init(self):
        """Test ONA initialization"""
        assert self.ona.delta_atn_threshold == 0.05
        assert self.ona.name == "ONASmoothing"
    
    def test_validate_wavelength_data(self, sample_aethalometer_data):
        """Test wavelength data validation"""
        # Should not raise error for valid data
        self.ona.validate_wavelength_data(sample_aethalometer_data, 'IR')
        
        # Should raise error for missing wavelength
        with pytest.raises(ValueError, match="Missing required columns"):
            self.ona.validate_wavelength_data(sample_aethalometer_data, 'NonExistent')
    
    def test_analyze_basic(self, sample_aethalometer_data):
        """Test basic ONA analysis"""
        result = self.ona.analyze(sample_aethalometer_data, 'IR')
        
        # Check result structure
        assert 'wavelength' in result
        assert 'algorithm' in result
        assert 'smoothed_data' in result
        assert 'improvement_metrics' in result
        assert result['algorithm'] == 'ONA'
        assert result['wavelength'] == 'IR'
        
        # Check smoothed data
        smoothed_data = result['smoothed_data']
        assert 'original_bc' in smoothed_data
        assert 'smoothed_bc' in smoothed_data
        assert len(smoothed_data['original_bc']) == len(smoothed_data['smoothed_bc'])
    
    def test_analyze_insufficient_data(self):
        """Test ONA with insufficient data"""
        data = pd.DataFrame({
            'IR BCc': [1.0],  # Only one point
            'IR ATN1': [50.0]
        })
        
        with pytest.raises(ValueError, match="Insufficient data for ONA smoothening"):
            self.ona.analyze(data, 'IR')
    
    def test_ona_algorithm_basic(self):
        """Test the core ONA algorithm"""
        bc_values = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
        atn_values = np.array([50.0, 50.1, 50.05, 50.2, 50.15])
        
        smoothed = self.ona._apply_ona_algorithm(bc_values, atn_values)
        
        assert len(smoothed) == len(bc_values)
        assert not np.any(np.isnan(smoothed))
        assert np.all(smoothed >= 0)  # BC should not be negative after smoothing
    
    def test_improvement_metrics(self):
        """Test improvement metrics calculation"""
        original = np.array([1.0, 2.0, 1.5, 3.0, 2.5, -0.1])  # Include one negative
        smoothed = np.array([1.1, 1.9, 1.6, 2.8, 2.4, 0.1])   # Smoothed version
        
        metrics = self.ona._calculate_improvement_metrics(original, smoothed)
        
        assert 'noise_reduction_percent' in metrics
        assert 'correlation_with_original' in metrics
        assert 'negative_reduction' in metrics
        assert isinstance(metrics['noise_reduction_percent'], float)
        assert 0 <= metrics['correlation_with_original'] <= 1


class TestCMASmoothing:
    """Test CMA (Centered Moving Average) smoothening"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cma = CMASmoothing(window_size=15)
    
    def test_init(self):
        """Test CMA initialization"""
        assert self.cma.window_size == 15
        assert self.cma.name == "CMASmoothing"
    
    def test_analyze_basic(self, sample_aethalometer_data):
        """Test basic CMA analysis"""
        result = self.cma.analyze(sample_aethalometer_data, 'IR')
        
        assert result['algorithm'] == 'CMA'
        assert result['parameters']['window_size'] == 15
        
        # Check that smoothing reduces noise
        metrics = result['improvement_metrics']
        assert metrics['noise_reduction_percent'] >= 0
    
    def test_insufficient_data(self):
        """Test CMA with insufficient data"""
        # Create data with fewer points than window size
        data = pd.DataFrame({
            'IR BCc': np.random.random(10),  # Less than window_size=15
            'IR ATN1': np.random.random(10)
        })
        
        with pytest.raises(ValueError, match="Insufficient data for CMA smoothening"):
            self.cma.analyze(data, 'IR')
    
    def test_cma_algorithm(self):
        """Test CMA algorithm with known values"""
        bc_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cma_small = CMASmoothing(window_size=3)
        
        smoothed = cma_small._apply_cma_algorithm(bc_values)
        
        # Check that middle values are averaged properly
        # For point index 2 (value 3.0), should average [1,2,3,4] = 2.5
        assert abs(smoothed[2] - 2.5) < 0.1
        assert len(smoothed) == len(bc_values)


class TestDEMASmoothing:
    """Test DEMA (Double Exponentially Weighted Moving Average) smoothening"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.dema = DEMASmoothing(alpha=0.2)
    
    def test_init(self):
        """Test DEMA initialization"""
        assert self.dema.alpha == 0.2
        assert self.dema.name == "DEMASmoothing"
    
    def test_analyze_basic(self, sample_aethalometer_data):
        """Test basic DEMA analysis"""
        result = self.dema.analyze(sample_aethalometer_data, 'IR')
        
        assert result['algorithm'] == 'DEMA'
        assert result['parameters']['alpha'] == 0.2
        
        # DEMA should include lag metric
        metrics = result['improvement_metrics']
        assert 'lag_metric' in metrics
        assert isinstance(metrics['lag_metric'], float)
    
    def test_dema_algorithm(self):
        """Test DEMA algorithm with known sequence"""
        bc_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        smoothed = self.dema._apply_dema_algorithm(bc_values)
        
        assert len(smoothed) == len(bc_values)
        assert smoothed[0] == bc_values[0]  # First value should be unchanged
        assert not np.any(np.isnan(smoothed))
    
    def test_lag_calculation(self):
        """Test lag metric calculation"""
        # Create data with known lag
        original = np.sin(np.linspace(0, 4*np.pi, 100))
        smoothed = np.roll(original, 2)  # Shift by 2 positions
        
        lag = self.dema._calculate_lag_metric(original, smoothed)
        
        # Should detect the 2-position lag
        assert abs(lag - 2) <= 1  # Allow for some tolerance


class TestSmoothingFactory:
    """Test SmoothingFactory pattern"""
    
    def test_available_methods(self):
        """Test getting available methods"""
        methods = SmoothingFactory.get_available_methods()
        
        assert 'ONA' in methods
        assert 'CMA' in methods
        assert 'DEMA' in methods
        assert len(methods) == 3
    
    def test_create_smoother_ona(self):
        """Test creating ONA smoother"""
        smoother = SmoothingFactory.create_smoother('ONA', delta_atn_threshold=0.1)
        
        assert isinstance(smoother, ONASmoothing)
        assert smoother.delta_atn_threshold == 0.1
    
    def test_create_smoother_cma(self):
        """Test creating CMA smoother"""
        smoother = SmoothingFactory.create_smoother('CMA', window_size=20)
        
        assert isinstance(smoother, CMASmoothing)
        assert smoother.window_size == 20
    
    def test_create_smoother_dema(self):
        """Test creating DEMA smoother"""
        smoother = SmoothingFactory.create_smoother('DEMA', alpha=0.3)
        
        assert isinstance(smoother, DEMASmoothing)
        assert smoother.alpha == 0.3
    
    def test_create_smoother_invalid(self):
        """Test creating invalid smoother"""
        with pytest.raises(ValueError, match="Unknown smoothening method"):
            SmoothingFactory.create_smoother('INVALID')
    
    def test_get_method_info(self):
        """Test getting method information"""
        info = SmoothingFactory.get_method_info()
        
        assert 'ONA' in info
        assert 'CMA' in info
        assert 'DEMA' in info
        
        # Check ONA info structure
        ona_info = info['ONA']
        assert 'description' in ona_info
        assert 'parameters' in ona_info
        assert 'best_for' in ona_info
    
    def test_recommend_method(self):
        """Test method recommendation"""
        # Test high variability
        rec = SmoothingFactory.recommend_method({'variability': 'high'})
        assert rec == 'ONA'
        
        # Test real-time requirement
        rec = SmoothingFactory.recommend_method({'real_time_required': True})
        assert rec == 'DEMA'
        
        # Test high noise
        rec = SmoothingFactory.recommend_method({'noise_level': 'high'})
        assert rec == 'CMA'


class TestSmoothingComparison:
    """Test SmoothingComparison utility"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.comparator = SmoothingComparison(['ONA', 'CMA'])
    
    def test_init(self):
        """Test comparison initialization"""
        assert self.comparator.methods == ['ONA', 'CMA']
        assert self.comparator.results == {}
    
    def test_compare_methods(self, sample_aethalometer_data):
        """Test method comparison"""
        results = self.comparator.compare_methods(
            sample_aethalometer_data, 
            wavelength='IR'
        )
        
        assert 'individual_results' in results
        assert 'comparison_summary' in results
        assert 'ONA' in results['individual_results']
        assert 'CMA' in results['individual_results']
        
        # Check summary structure
        summary = results['comparison_summary']
        assert 'performance_ranking' in summary
        assert 'metrics_comparison' in summary
    
    def test_get_best_method(self, sample_aethalometer_data):
        """Test getting best method"""
        self.comparator.compare_methods(sample_aethalometer_data, 'IR')
        
        best_overall = self.comparator.get_best_method('overall')
        best_noise = self.comparator.get_best_method('noise_reduction')
        
        assert best_overall in ['ONA', 'CMA']
        assert best_noise in ['ONA', 'CMA']


class TestAdaptiveSmoothing:
    """Test AdaptiveSmoothing algorithm"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.adaptive = AdaptiveSmoothing(selection_criterion='overall')
    
    def test_init(self):
        """Test adaptive smoothing initialization"""
        assert self.adaptive.selection_criterion == 'overall'
        assert self.adaptive.test_sample_size == 1000
        assert self.adaptive.selected_method is None
    
    @patch('builtins.print')  # Mock print to avoid output during tests
    def test_analyze(self, mock_print, sample_aethalometer_data):
        """Test adaptive analysis"""
        result = self.adaptive.analyze(sample_aethalometer_data, 'IR')
        
        assert 'adaptive_selection' in result
        assert self.adaptive.selected_method is not None
        
        # Check adaptive selection metadata
        selection_info = result['adaptive_selection']
        assert 'selected_method' in selection_info
        assert 'selection_criterion' in selection_info
        assert 'data_characteristics' in selection_info
    
    def test_data_characteristics_analysis(self, sample_aethalometer_data):
        """Test data characteristics analysis"""
        characteristics = self.adaptive._analyze_data_characteristics(
            sample_aethalometer_data, 'IR'
        )
        
        assert 'sample_size' in characteristics
        assert 'noise_level' in characteristics
        assert 'variability' in characteristics
        assert 'data_quality' in characteristics
        assert characteristics['sample_size'] > 0
    
    def test_get_test_sample(self, sample_aethalometer_data):
        """Test test sample extraction"""
        # Test with large dataset
        large_data = pd.concat([sample_aethalometer_data] * 5)  # Make it larger
        self.adaptive.test_sample_size = 100
        
        sample = self.adaptive._get_test_sample(large_data)
        
        assert len(sample) <= 100
        assert len(sample) <= len(large_data)


# Test fixtures
@pytest.fixture
def sample_aethalometer_data():
    """Generate sample aethalometer data for testing"""
    n_points = 1000
    timestamps = pd.date_range('2022-01-01', periods=n_points, freq='min')
    
    # Generate realistic BC data with some noise
    np.random.seed(42)  # For reproducible tests
    
    # Base signal with some trends
    base_signal = 2.0 + 0.5 * np.sin(np.linspace(0, 4*np.pi, n_points))
    noise = np.random.normal(0, 0.1, n_points)
    bc_values = base_signal + noise
    bc_values = np.maximum(bc_values, 0)  # Ensure non-negative
    
    # ATN values (monotonically increasing with some noise)
    atn_values = 50 + np.cumsum(np.random.normal(0.01, 0.01, n_points))
    
    return pd.DataFrame({
        'datetime_local': timestamps,
        'IR BCc': bc_values,
        'IR ATN1': atn_values,
        'UV BCc': bc_values * 1.1,  # Correlated but different
        'UV ATN1': atn_values * 0.9
    })


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
