"""Memory optimization and batch processing utilities for ETAD analysis"""

import gc
import sys
import warnings
from typing import Iterator, List, Dict, Any, Optional, Callable, Union, Tuple
from functools import wraps
from contextlib import contextmanager
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from datetime import datetime

from ..utils.logging.logger import ETADLogger

@dataclass
class MemoryProfile:
    """Memory usage profile for analysis"""
    initial_memory: float  # MB
    peak_memory: float     # MB
    final_memory: float    # MB
    memory_saved: float    # MB
    gc_collections: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'initial_memory_mb': self.initial_memory,
            'peak_memory_mb': self.peak_memory,
            'final_memory_mb': self.final_memory,
            'memory_saved_mb': self.memory_saved,
            'gc_collections': self.gc_collections,
            'timestamp': self.timestamp.isoformat()
        }

class MemoryOptimizer:
    """Optimize memory usage for large dataset analysis"""
    
    def __init__(self, logger: Optional[ETADLogger] = None):
        self.logger = logger or ETADLogger("MemoryOptimizer")
        self._gc_threshold = (700, 10, 10)  # More aggressive GC
        self._original_threshold = gc.get_threshold()
        
    def optimize_memory_usage(self, aggressive: bool = False):
        """
        Decorator to optimize memory usage for functions
        
        Parameters:
        -----------
        aggressive : bool
            Use aggressive memory optimization strategies
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_memory_optimization(
                    func, args, kwargs, aggressive
                )
            return wrapper
        return decorator
    
    def _execute_with_memory_optimization(
        self, 
        func: Callable, 
        args: tuple, 
        kwargs: dict,
        aggressive: bool = False
    ) -> Any:
        """Execute function with memory optimization"""
        
        # Get initial memory state
        initial_memory = self._get_memory_usage()
        gc_collections_before = sum(gc.get_stats()[i]['collections'] for i in range(3))
        
        if aggressive:
            # Set aggressive GC thresholds
            gc.set_threshold(*self._gc_threshold)
            
        try:
            with self._memory_monitor() as monitor:
                # Force garbage collection before execution
                gc.collect()
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Force garbage collection after execution
                gc.collect()
                
            final_memory = self._get_memory_usage()
            gc_collections_after = sum(gc.get_stats()[i]['collections'] for i in range(3))
            peak_memory = monitor.peak_memory
            
            # Create memory profile
            profile = MemoryProfile(
                initial_memory=initial_memory,
                peak_memory=peak_memory,
                final_memory=final_memory,
                memory_saved=max(0, peak_memory - final_memory),
                gc_collections=gc_collections_after - gc_collections_before,
                timestamp=datetime.now()
            )
            
            self.logger.info(
                f"Memory optimization for {func.__name__}",
                extra={
                    'memory_profile': profile.to_dict(),
                    'memory_reduction': profile.memory_saved > 0
                }
            )
            
            return result
            
        finally:
            if aggressive:
                # Restore original GC thresholds
                gc.set_threshold(*self._original_threshold)
    
    @contextmanager
    def _memory_monitor(self):
        """Monitor peak memory usage during execution"""
        
        class MemoryMonitor:
            def __init__(self):
                self.peak_memory = 0.0
                self.monitoring = True
                
            def update(self, memory_mb: float):
                if self.monitoring:
                    self.peak_memory = max(self.peak_memory, memory_mb)
        
        monitor = MemoryMonitor()
        monitor.peak_memory = self._get_memory_usage()
        
        # Start monitoring in a separate thread
        def monitor_memory():
            while monitor.monitoring:
                try:
                    current_memory = self._get_memory_usage()
                    monitor.update(current_memory)
                    threading.Event().wait(0.1)  # Check every 100ms
                except:
                    break
        
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            yield monitor
        finally:
            monitor.monitoring = False
            monitor_thread.join(timeout=1.0)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback to basic memory tracking
            return sys.getsizeof(gc.get_objects()) / 1024 / 1024
    
    def reduce_dataframe_memory(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Reduce DataFrame memory usage by optimizing data types
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to optimize
        verbose : bool
            Print memory reduction details
            
        Returns:
        --------
        pd.DataFrame
            Memory-optimized DataFrame
        """
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize object columns (potential categories)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        memory_reduction = initial_memory - final_memory
        
        if verbose:
            self.logger.info(
                f"DataFrame memory optimized: {initial_memory:.2f}MB → {final_memory:.2f}MB "
                f"(saved {memory_reduction:.2f}MB, {memory_reduction/initial_memory*100:.1f}%)"
            )
        
        return df
    
    @contextmanager
    def temporary_memory_optimization(self):
        """Context manager for temporary aggressive memory optimization"""
        original_threshold = gc.get_threshold()
        
        try:
            # Set aggressive GC
            gc.set_threshold(*self._gc_threshold)
            gc.collect()
            
            self.logger.debug("Temporary aggressive memory optimization enabled")
            yield
            
        finally:
            # Restore original settings
            gc.set_threshold(*original_threshold)
            gc.collect()
            self.logger.debug("Memory optimization restored to normal")

class BatchProcessor:
    """Process large datasets in batches to manage memory"""
    
    def __init__(self, 
                 batch_size: int = 10000, 
                 logger: Optional[ETADLogger] = None,
                 memory_optimizer: Optional[MemoryOptimizer] = None):
        self.batch_size = batch_size
        self.logger = logger or ETADLogger("BatchProcessor")
        self.memory_optimizer = memory_optimizer or MemoryOptimizer(logger)
        
    def process_dataframe_batches(
        self,
        df: pd.DataFrame,
        process_func: Callable[[pd.DataFrame], pd.DataFrame],
        batch_size: Optional[int] = None,
        preserve_index: bool = True,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Process DataFrame in batches
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to process
        process_func : Callable
            Function to apply to each batch
        batch_size : int, optional
            Override default batch size
        preserve_index : bool
            Whether to preserve original index
        show_progress : bool
            Show processing progress
            
        Returns:
        --------
        pd.DataFrame
            Processed DataFrame
        """
        batch_size = batch_size or self.batch_size
        total_rows = len(df)
        batches = list(range(0, total_rows, batch_size))
        
        if show_progress:
            self.logger.info(f"Processing {total_rows} rows in {len(batches)} batches of {batch_size}")
        
        results = []
        
        for i, start_idx in enumerate(batches):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = df.iloc[start_idx:end_idx]
            
            try:
                with self.memory_optimizer.temporary_memory_optimization():
                    processed_batch = process_func(batch)
                    results.append(processed_batch)
                
                if show_progress and (i + 1) % 10 == 0:
                    self.logger.info(f"Processed batch {i + 1}/{len(batches)} "
                                   f"({(i + 1)/len(batches)*100:.1f}%)")
                    
            except Exception as e:
                self.logger.error(f"Error processing batch {i + 1}: {e}")
                raise
        
        # Combine results
        result_df = pd.concat(results, ignore_index=not preserve_index)
        
        if show_progress:
            self.logger.info(f"Batch processing complete: {len(result_df)} rows processed")
        
        return result_df
    
    def process_file_batches(
        self,
        file_path: Union[str, Path],
        process_func: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: Optional[int] = None,
        file_format: str = 'csv',
        **read_kwargs
    ) -> Union[pd.DataFrame, str]:
        """
        Process file in batches to manage memory
        
        Parameters:
        -----------
        file_path : str or Path
            Input file path
        process_func : Callable
            Function to apply to each batch
        output_path : str or Path, optional
            Output file path (if None, returns DataFrame)
        batch_size : int, optional
            Override default batch size
        file_format : str
            File format ('csv', 'parquet', 'excel')
        **read_kwargs
            Additional arguments for file reading
            
        Returns:
        --------
        DataFrame or str
            Processed data or output file path
        """
        batch_size = batch_size or self.batch_size
        file_path = Path(file_path)
        
        if file_format == 'csv':
            reader = pd.read_csv(file_path, chunksize=batch_size, **read_kwargs)
        elif file_format == 'parquet':
            # For parquet, we need to read and split manually
            df = pd.read_parquet(file_path, **read_kwargs)
            return self.process_dataframe_batches(df, process_func, batch_size)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        results = []
        batch_count = 0
        
        self.logger.info(f"Starting batch processing of {file_path}")
        
        for batch in reader:
            batch_count += 1
            
            try:
                with self.memory_optimizer.temporary_memory_optimization():
                    processed_batch = process_func(batch)
                    results.append(processed_batch)
                
                if batch_count % 10 == 0:
                    self.logger.info(f"Processed {batch_count} batches")
                    
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_count}: {e}")
                raise
        
        # Combine results
        final_result = pd.concat(results, ignore_index=True)
        
        if output_path:
            output_path = Path(output_path)
            if output_path.suffix.lower() == '.csv':
                final_result.to_csv(output_path, index=False)
            elif output_path.suffix.lower() == '.parquet':
                final_result.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported output format: {output_path.suffix}")
            
            self.logger.info(f"Batch processing complete: saved to {output_path}")
            return str(output_path)
        
        self.logger.info(f"Batch processing complete: {len(final_result)} rows")
        return final_result
    
    def chunk_array(self, array: np.ndarray, chunk_size: Optional[int] = None) -> Iterator[np.ndarray]:
        """
        Split array into chunks
        
        Parameters:
        -----------
        array : np.ndarray
            Array to chunk
        chunk_size : int, optional
            Size of each chunk
            
        Yields:
        -------
        np.ndarray
            Array chunks
        """
        chunk_size = chunk_size or self.batch_size
        
        for i in range(0, len(array), chunk_size):
            yield array[i:i + chunk_size]

class CacheManager:
    """Manage temporary caching for large analysis operations"""
    
    def __init__(self, 
                 cache_dir: Optional[Union[str, Path]] = None,
                 max_cache_size: int = 1000,  # MB
                 logger: Optional[ETADLogger] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "etad_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.logger = logger or ETADLogger("CacheManager")
        self._cache_registry: Dict[str, Dict[str, Any]] = {}
        
    def cache_dataframe(self, 
                       df: pd.DataFrame, 
                       key: str,
                       compress: bool = True) -> str:
        """
        Cache DataFrame to disk
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to cache
        key : str
            Cache key
        compress : bool
            Use compression
            
        Returns:
        --------
        str
            Cache file path
        """
        cache_file = self.cache_dir / f"{key}.{'parquet' if compress else 'pkl'}"
        
        try:
            if compress:
                df.to_parquet(cache_file, compression='snappy')
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
            
            # Register in cache
            self._cache_registry[key] = {
                'file_path': str(cache_file),
                'timestamp': datetime.now(),
                'size_mb': cache_file.stat().st_size / 1024 / 1024,
                'compressed': compress
            }
            
            self.logger.debug(f"Cached DataFrame '{key}' to {cache_file}")
            
            # Check cache size
            self._cleanup_if_needed()
            
            return str(cache_file)
            
        except Exception as e:
            self.logger.error(f"Error caching DataFrame '{key}': {e}")
            raise
    
    def load_cached_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """
        Load cached DataFrame
        
        Parameters:
        -----------
        key : str
            Cache key
            
        Returns:
        --------
        pd.DataFrame or None
            Cached DataFrame or None if not found
        """
        if key not in self._cache_registry:
            return None
        
        cache_info = self._cache_registry[key]
        cache_file = Path(cache_info['file_path'])
        
        if not cache_file.exists():
            del self._cache_registry[key]
            return None
        
        try:
            if cache_info['compressed']:
                df = pd.read_parquet(cache_file)
            else:
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
            
            self.logger.debug(f"Loaded cached DataFrame '{key}' from {cache_file}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading cached DataFrame '{key}': {e}")
            return None
    
    def _cleanup_if_needed(self):
        """Clean up cache if it exceeds size limit"""
        total_size = sum(info['size_mb'] for info in self._cache_registry.values())
        
        if total_size > self.max_cache_size:
            # Remove oldest files first
            sorted_items = sorted(
                self._cache_registry.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            removed_size = 0
            removed_count = 0
            
            for key, info in sorted_items:
                if total_size - removed_size <= self.max_cache_size * 0.8:
                    break
                
                cache_file = Path(info['file_path'])
                if cache_file.exists():
                    cache_file.unlink()
                
                removed_size += info['size_mb']
                removed_count += 1
                del self._cache_registry[key]
            
            if removed_count > 0:
                self.logger.info(f"Cache cleanup: removed {removed_count} files, "
                               f"freed {removed_size:.1f}MB")
    
    def clear_cache(self):
        """Clear all cached data"""
        removed_count = 0
        
        for key, info in list(self._cache_registry.items()):
            cache_file = Path(info['file_path'])
            if cache_file.exists():
                cache_file.unlink()
                removed_count += 1
            del self._cache_registry[key]
        
        self.logger.info(f"Cache cleared: removed {removed_count} files")
    
    @contextmanager
    def temporary_cache(self, key: str):
        """Context manager for temporary caching"""
        try:
            yield key
        finally:
            if key in self._cache_registry:
                cache_file = Path(self._cache_registry[key]['file_path'])
                if cache_file.exists():
                    cache_file.unlink()
                del self._cache_registry[key]

# Global instances
memory_optimizer = MemoryOptimizer()
batch_processor = BatchProcessor()
cache_manager = CacheManager()

# Convenience decorators and functions
optimize_memory = memory_optimizer.optimize_memory_usage
reduce_memory = memory_optimizer.reduce_dataframe_memory
process_in_batches = batch_processor.process_dataframe_batches
