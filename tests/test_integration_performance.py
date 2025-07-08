"""Integration tests for performance monitoring and parallel processing"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from src.core.monitoring import PerformanceMonitor, ErrorHandler, SystemMonitor
from src.core.parallel_processing import ParallelProcessor, PipelineProcessor, ParallelProcessingConfig
from src.utils.memory_optimization import MemoryOptimizer, BatchProcessor, CacheManager
from src.utils.logging.logger import ETADLogger

class TestPerformanceMonitoring:
    """Test performance monitoring functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.logger = ETADLogger("TestLogger")
        self.monitor = PerformanceMonitor(self.logger)
        self.error_handler = ErrorHandler(self.logger)
    
    def test_performance_monitor_decorator(self):
        """Test performance monitoring decorator"""
        
        @self.monitor.monitor()
        def sample_function(data_size: int = 1000):
            # Simulate some work
            data = np.random.random(data_size)
            return np.sum(data)
        
        result = sample_function(5000)
        
        # Check that metrics were recorded
        assert len(self.monitor.metrics_history) == 1
        metrics = self.monitor.metrics_history[0]
        
        assert metrics.function_name == "sample_function"
        assert metrics.execution_time > 0
        assert metrics.data_size == 5000
        assert metrics.throughput > 0
        assert isinstance(result, (int, float))
    
    def test_error_handler_retry(self):
        """Test error handler with retry logic"""
        
        call_count = 0
        
        @self.error_handler.handle_with_retry(max_retries=2, backoff_factor=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 times
                raise ValueError("Simulated error")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3
        assert len(self.error_handler.error_history) == 2  # 2 failures recorded
    
    def test_error_handler_graceful_degradation(self):
        """Test graceful degradation on errors"""
        
        @self.error_handler.graceful_degradation(default_value="default")
        def always_failing_function():
            raise RuntimeError("This always fails")
        
        result = always_failing_function()
        assert result == "default"
        assert len(self.error_handler.error_history) == 1
    
    def test_system_monitor_context(self):
        """Test system resource monitoring"""
        resource_data = []
        
        with self.monitor.__class__(self.logger).__class__(self.logger) as monitor:
            # Mock the system monitor for testing
            with patch('src.core.monitoring.SystemMonitor') as mock_monitor:
                mock_monitor.return_value.monitor_resources.return_value.__enter__.return_value = resource_data
                mock_monitor.return_value.monitor_resources.return_value.__exit__.return_value = None
                
                # Simulate some work
                time.sleep(0.1)
        
        # Test passes if no exceptions were raised

class TestParallelProcessing:
    """Test parallel processing functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.logger = ETADLogger("TestLogger")
        self.config = ParallelProcessingConfig(max_workers=2, chunk_size=100)
        self.processor = ParallelProcessor(self.config, self.logger)
    
    def test_parallel_dataframe_processing(self):
        """Test parallel DataFrame processing"""
        
        # Create test DataFrame
        df = pd.DataFrame({
            'A': np.random.random(500),
            'B': np.random.random(500),
            'C': np.random.random(500)
        })
        
        def multiply_by_two(chunk_df):
            return chunk_df * 2
        
        result = self.processor.parallel_apply(df, multiply_by_two)
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert np.allclose(result['A'].values, df['A'].values * 2)
    
    def test_parallel_list_processing(self):
        """Test parallel list processing"""
        
        data = list(range(1000))
        
        def square_number(x):
            return x ** 2
        
        result = self.processor.parallel_apply(data, square_number)
        
        # Verify results
        assert len(result) == len(data)
        assert result[0] == 0
        assert result[10] == 100
        assert result[999] == 999 ** 2
    
    def test_pipeline_processing(self):
        """Test pipeline processing"""
        
        pipeline = PipelineProcessor(self.config, self.logger)
        
        # Add pipeline steps
        pipeline.add_step("multiply", lambda df: df * 2, parallel=False)
        pipeline.add_step("add_column", lambda df: df.assign(sum_col=df.sum(axis=1)), parallel=False)
        pipeline.add_step("filter", lambda df: df[df['sum_col'] > df['sum_col'].median()], parallel=False)
        
        # Create test data
        df = pd.DataFrame({
            'A': np.random.random(200),
            'B': np.random.random(200)
        })
        
        result = pipeline.process(df)
        
        # Verify pipeline execution
        assert isinstance(result, pd.DataFrame)
        assert 'sum_col' in result.columns
        assert len(result) <= len(df)  # Some rows filtered out
        assert all(result['A'] == df.loc[result.index, 'A'] * 2)  # Multiplication applied

class TestMemoryOptimization:
    """Test memory optimization functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.logger = ETADLogger("TestLogger")
        self.optimizer = MemoryOptimizer(self.logger)
        self.batch_processor = BatchProcessor(batch_size=100, logger=self.logger)
    
    def test_memory_optimization_decorator(self):
        """Test memory optimization decorator"""
        
        @self.optimizer.optimize_memory_usage()
        def memory_intensive_function():
            # Create large arrays and delete them
            large_data = np.random.random((1000, 1000))
            result = np.sum(large_data)
            del large_data
            return result
        
        result = memory_intensive_function()
        assert isinstance(result, (int, float))
    
    def test_dataframe_memory_reduction(self):
        """Test DataFrame memory reduction"""
        
        # Create DataFrame with inefficient data types
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000).astype('int64'),
            'float_col': np.random.random(1000).astype('float64'),
            'category_col': np.random.choice(['A', 'B', 'C'], 1000).astype('object')
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = self.optimizer.reduce_dataframe_memory(df, verbose=False)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Memory should be reduced
        assert optimized_memory <= original_memory
        assert optimized_df['category_col'].dtype.name == 'category'
        assert optimized_df['int_col'].dtype in [np.int8, np.int16, np.int32]
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        
        # Create large DataFrame
        df = pd.DataFrame({
            'A': np.random.random(1000),
            'B': np.random.random(1000)
        })
        
        def add_columns(batch_df):
            return batch_df.assign(
                C=batch_df['A'] + batch_df['B'],
                D=batch_df['A'] * batch_df['B']
            )
        
        result = self.batch_processor.process_dataframe_batches(
            df, add_columns, batch_size=200, show_progress=False
        )
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert 'C' in result.columns and 'D' in result.columns
        assert np.allclose(result['C'], df['A'] + df['B'])

class TestCacheManager:
    """Test cache management functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.logger = ETADLogger("TestLogger")
        # Use temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(cache_dir=self.temp_dir, logger=self.logger)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        self.cache_manager.clear_cache()
    
    def test_cache_and_load_dataframe(self):
        """Test caching and loading DataFrames"""
        
        # Create test DataFrame
        df = pd.DataFrame({
            'A': np.random.random(100),
            'B': np.random.random(100)
        })
        
        # Cache DataFrame
        cache_path = self.cache_manager.cache_dataframe(df, 'test_df', compress=True)
        assert Path(cache_path).exists()
        
        # Load from cache
        loaded_df = self.cache_manager.load_cached_dataframe('test_df')
        
        # Verify loaded data
        assert loaded_df is not None
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(df)
        pd.testing.assert_frame_equal(loaded_df, df)
    
    def test_cache_miss(self):
        """Test cache miss scenario"""
        
        # Try to load non-existent cache
        result = self.cache_manager.load_cached_dataframe('non_existent')
        assert result is None
    
    def test_temporary_cache_context(self):
        """Test temporary cache context manager"""
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        with self.cache_manager.temporary_cache('temp_test') as cache_key:
            # Cache some data
            self.cache_manager.cache_dataframe(df, cache_key)
            
            # Verify it exists
            loaded = self.cache_manager.load_cached_dataframe(cache_key)
            assert loaded is not None
        
        # After context, cache should be cleaned up
        loaded = self.cache_manager.load_cached_dataframe('temp_test')
        assert loaded is None

class TestIntegration:
    """Integration tests combining multiple components"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.logger = ETADLogger("TestLogger")
        
    def test_monitored_parallel_processing(self):
        """Test parallel processing with performance monitoring"""
        
        # Setup components
        monitor = PerformanceMonitor(self.logger)
        config = ParallelProcessingConfig(max_workers=2, chunk_size=50)
        processor = ParallelProcessor(config, self.logger)
        
        @monitor.monitor()
        def process_data_parallel(df):
            def transform_chunk(chunk):
                return chunk ** 2
            
            return processor.parallel_apply(df, transform_chunk)
        
        # Create test data
        df = pd.DataFrame({
            'A': np.random.random(500),
            'B': np.random.random(500)
        })
        
        result = process_data_parallel(df)
        
        # Verify monitoring occurred
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0].function_name == "process_data_parallel"
        
        # Verify processing results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
    
    def test_error_handling_with_batch_processing(self):
        """Test error handling in batch processing"""
        
        error_handler = ErrorHandler(self.logger)
        batch_processor = BatchProcessor(batch_size=100, logger=self.logger)
        
        # Function that fails on certain conditions
        @error_handler.graceful_degradation(default_value=pd.DataFrame())
        def sometimes_failing_process(df):
            if len(df) > 150:  # Fail on large batches
                raise ValueError("Batch too large")
            return df * 2
        
        # Create data that will cause some batches to fail
        df = pd.DataFrame({
            'A': np.random.random(300)
        })
        
        result = batch_processor.process_dataframe_batches(
            df, sometimes_failing_process, batch_size=200, show_progress=False
        )
        
        # Should get some results despite errors
        assert isinstance(result, pd.DataFrame)
        # Check error history
        assert len(error_handler.error_history) > 0

def create_sample_aethalometer_data(n_points: int = 1000) -> pd.DataFrame:
    """Create sample aethalometer data for testing"""
    timestamps = pd.date_range('2023-01-01', periods=n_points, freq='1min')
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'Blue BCc': np.random.lognormal(0, 0.5, n_points),
        'Blue ATN1': np.random.uniform(0, 100, n_points),
        'IR BCc': np.random.lognormal(0, 0.5, n_points),
        'IR ATN1': np.random.uniform(0, 100, n_points),
        'Timebase (s)': [60] * n_points
    })
    
    # Add some realistic noise and patterns
    data['Blue BCc'] = np.maximum(0, data['Blue BCc'] + np.random.normal(0, 0.1, n_points))
    data['IR BCc'] = np.maximum(0, data['IR BCc'] + np.random.normal(0, 0.1, n_points))
    
    return data

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
