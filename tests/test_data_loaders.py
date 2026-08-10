"""Test suite for data loaders"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.data.loaders.database import DatabaseLoader


class TestDatabaseLoader:
    """Test DatabaseLoader functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.loader = DatabaseLoader("test.db")
    
    def test_init(self):
        """Test DatabaseLoader initialization"""
        assert self.loader.db_path == "test.db"
        assert hasattr(self.loader, 'connection')
    
    @patch('sqlite3.connect')
    def test_connect(self, mock_connect):
        """Test database connection"""
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        
        self.loader.connect()
        
        mock_connect.assert_called_once_with("test.db")
        assert self.loader.connection == mock_connection
    
    @patch('pandas.read_sql_query')
    def test_load_etad_data(self, mock_read_sql):
        """Test ETAD data loading"""
        # Mock data
        mock_data = pd.DataFrame({
            'sample_id': [1, 2, 3],
            'bc_hips': [1.5, 2.0, 1.8],
            'oc_hips': [3.5, 4.0, 3.8]
        })
        mock_read_sql.return_value = mock_data
        
        # Mock connection
        self.loader.connection = MagicMock()
        
        result = self.loader.load_etad_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        mock_read_sql.assert_called_once()
    
    def test_load_etad_data_no_connection(self):
        """Test ETAD data loading without connection"""
        self.loader.connection = None
        
        with pytest.raises(ValueError, match="Database connection not established"):
            self.loader.load_etad_data()
    
    @patch('pandas.read_sql_query')
    def test_load_ftir_data(self, mock_read_sql):
        """Test FTIR data loading"""
        mock_data = pd.DataFrame({
            'sample_id': [1, 2, 3],
            'fabs_IR': [10.5, 12.0, 11.8],
            'date_collected': ['2022-01-01', '2022-01-02', '2022-01-03']
        })
        mock_read_sql.return_value = mock_data
        
        self.loader.connection = MagicMock()
        
        result = self.loader.load_ftir_data()
        
        assert isinstance(result, pd.DataFrame)
        assert 'fabs_IR' in result.columns
        mock_read_sql.assert_called_once()
    
    @patch('pandas.read_sql_query')
    def test_load_aethalometer_data(self, mock_read_sql):
        """Test Aethalometer data loading"""
        mock_data = pd.DataFrame({
            'datetime_local': pd.date_range('2022-01-01', periods=100, freq='min'),
            'IR BCc': np.random.normal(2.0, 0.5, 100),
            'IR ATN1': np.random.normal(50, 10, 100)
        })
        mock_read_sql.return_value = mock_data
        
        self.loader.connection = MagicMock()
        
        result = self.loader.load_aethalometer_data()
        
        assert isinstance(result, pd.DataFrame)
        assert 'IR BCc' in result.columns
        assert len(result) == 100
    
    def test_close_connection(self):
        """Test connection closing"""
        mock_connection = MagicMock()
        self.loader.connection = mock_connection
        
        self.loader.close()
        
        mock_connection.close.assert_called_once()
        assert self.loader.connection is None


class TestDataLoaderIntegration:
    """Integration tests for data loaders"""
    
    def test_load_and_validate_real_structure(self):
        """Test loading data with realistic structure"""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            # This would be a more comprehensive test with actual database setup
            # For now, we test the interface
            loader = DatabaseLoader(db_path)
            
            # Test that the loader initializes correctly
            assert loader.db_path == db_path
            assert loader.connection is None
            
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestDataValidation:
    """Test data validation utilities"""
    
    def test_validate_required_columns(self):
        """Test column validation"""
        from src.data.processors.validation import validate_columns_exist
        
        # Test data with required columns
        data = pd.DataFrame({
            'IR BCc': [1, 2, 3],
            'IR ATN1': [10, 20, 30],
            'other_col': [100, 200, 300]
        })
        
        # Should not raise error
        validate_columns_exist(data, ['IR BCc', 'IR ATN1'])
        
        # Should raise error for missing column
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_columns_exist(data, ['IR BCc', 'missing_col'])
    
    def test_get_valid_data_mask(self):
        """Test valid data mask generation"""
        from src.data.processors.validation import get_valid_data_mask
        
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [10, np.nan, 30, 40]
        })
        
        mask = get_valid_data_mask(data, ['col1', 'col2'])
        expected = np.array([True, False, False, True])
        
        np.testing.assert_array_equal(mask, expected)


# Test fixtures for sample data
@pytest.fixture
def sample_aethalometer_data():
    """Generate sample aethalometer data for testing"""
    n_points = 1000
    timestamps = pd.date_range('2022-01-01', periods=n_points, freq='min')
    
    # Simulate realistic BC data with noise
    bc_clean = np.random.lognormal(mean=0.5, sigma=0.3, size=n_points)
    bc_noisy = bc_clean + np.random.normal(0, bc_clean * 0.1)  # 10% noise
    
    # Simulate ATN data
    atn_values = 50 + np.cumsum(np.random.normal(0, 0.1, n_points))
    
    return pd.DataFrame({
        'datetime_local': timestamps,
        'IR BCc': bc_noisy,
        'IR ATN1': atn_values,
        'UV BCc': bc_noisy * 1.2,  # Different wavelength
        'UV ATN1': atn_values * 0.8
    })


@pytest.fixture  
def sample_ftir_data():
    """Generate sample FTIR data for testing"""
    n_samples = 50
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    
    return pd.DataFrame({
        'sample_id': range(1, n_samples + 1),
        'date_collected': dates,
        'fabs_IR': np.random.normal(15, 5, n_samples),
        'oc_concentration': np.random.normal(8, 2, n_samples),
        'ec_concentration': np.random.normal(3, 1, n_samples)
    })


if __name__ == "__main__":
    pytest.main([__file__])
