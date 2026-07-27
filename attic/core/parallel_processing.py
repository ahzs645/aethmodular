"""Parallel processing utilities for ETAD analysis"""

import logging
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Union, Iterator, Tuple
import time
from functools import partial, wraps
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import queue
import pickle
import tempfile

try:
    from src.utils.logging.logger import ETADLogger
except ImportError:  # pragma: no cover - compatibility fallback
    ETADLogger = None
from ..utils.memory_optimization import MemoryOptimizer, CacheManager


def _resolve_logger(
    logger: Optional[Union[logging.Logger, "ETADLogger"]],
    default_name: str,
) -> logging.Logger:
    if isinstance(logger, logging.Logger):
        return logger
    if logger is None and ETADLogger is not None:
        return ETADLogger(default_name).get_logger()
    if ETADLogger is not None and isinstance(logger, ETADLogger):
        return logger.get_logger()
    if logger is not None and hasattr(logger, "get_logger"):
        candidate = logger.get_logger()
        if isinstance(candidate, logging.Logger):
            return candidate
    logging.basicConfig()
    return logging.getLogger(default_name)

@dataclass
class ParallelProcessingConfig:
    """Configuration for parallel processing"""
    max_workers: Optional[int] = None
    use_threads: bool = True  # Default to threads for notebook/test compatibility.
    chunk_size: int = 1000
    timeout: Optional[float] = None
    memory_limit_mb: float = 1000
    enable_caching: bool = True
    cache_intermediate: bool = False
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = mp.cpu_count() - 1 if not self.use_threads else mp.cpu_count() * 2

class ParallelProcessor:
    """Parallel processing manager for ETAD analysis"""
    
    def __init__(self, 
                 config: Optional[ParallelProcessingConfig] = None,
                 logger: Optional[Union[logging.Logger, "ETADLogger"]] = None):
        self.config = config or ParallelProcessingConfig()
        self.logger = _resolve_logger(logger, "ParallelProcessor")
        self.memory_optimizer = MemoryOptimizer(logger)
        self.cache_manager = CacheManager(logger=logger) if self.config.enable_caching else None
        
    def parallel_apply(self,
                      data: Union[pd.DataFrame, List[Any], np.ndarray],
                      func: Callable,
                      func_args: Optional[tuple] = None,
                      func_kwargs: Optional[dict] = None,
                      progress_callback: Optional[Callable[[int, int], None]] = None) -> Union[pd.DataFrame, List[Any]]:
        """
        Apply function in parallel to data
        
        Parameters:
        -----------
        data : DataFrame, List, or ndarray
            Data to process
        func : Callable
            Function to apply
        func_args : tuple, optional
            Additional arguments for function
        func_kwargs : dict, optional
            Additional keyword arguments for function
        progress_callback : Callable, optional
            Callback for progress updates (current, total)
            
        Returns:
        --------
        Processed data
        """
        func_args = func_args or ()
        func_kwargs = func_kwargs or {}
        
        if isinstance(data, pd.DataFrame):
            return self._parallel_apply_dataframe(data, func, func_args, func_kwargs, progress_callback)
        elif isinstance(data, (list, np.ndarray)):
            return self._parallel_apply_iterable(data, func, func_args, func_kwargs, progress_callback)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _parallel_apply_dataframe(self,
                                 df: pd.DataFrame,
                                 func: Callable,
                                 func_args: tuple,
                                 func_kwargs: dict,
                                 progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """Apply function to DataFrame chunks in parallel"""
        
        # Split DataFrame into chunks
        chunks = self._split_dataframe(df, self.config.chunk_size)
        total_chunks = len(chunks)
        
        self.logger.info(f"Processing DataFrame with {len(df)} rows in {total_chunks} chunks "
                        f"using {self.config.max_workers} workers")
        
        # Process chunks in parallel
        executor_class = ThreadPoolExecutor if self.config.use_threads else ProcessPoolExecutor
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for i, chunk in enumerate(chunks):
                future = executor.submit(self._safe_apply_func, func, chunk, func_args, func_kwargs)
                future_to_idx[future] = i
            
            # Collect results
            results = [None] * total_chunks
            completed = 0
            
            for future in as_completed(future_to_idx, timeout=self.config.timeout):
                idx = future_to_idx[future]
                
                try:
                    results[idx] = future.result()
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total_chunks)
                    
                    if completed % max(1, total_chunks // 10) == 0:
                        self.logger.info(f"Completed {completed}/{total_chunks} chunks "
                                       f"({completed/total_chunks*100:.1f}%)")
                        
                except Exception as e:
                    self.logger.error(f"Error processing chunk {idx}: {e}")
                    raise
        
        # Combine results
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            raise ValueError("No valid results from parallel processing")
        
        combined_result = pd.concat(valid_results, ignore_index=True)
        self.logger.info(f"Parallel processing complete: {len(combined_result)} rows")
        
        return combined_result
    
    def _parallel_apply_iterable(self,
                                data: Union[List[Any], np.ndarray],
                                func: Callable,
                                func_args: tuple,
                                func_kwargs: dict,
                                progress_callback: Optional[Callable] = None) -> List[Any]:
        """Apply function to iterable items in parallel"""
        
        data_list = list(data) if not isinstance(data, list) else data
        total_items = len(data_list)
        
        self.logger.info(f"Processing {total_items} items using {self.config.max_workers} workers")
        
        executor_class = ThreadPoolExecutor if self.config.use_threads else ProcessPoolExecutor
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for i, item in enumerate(data_list):
                future = executor.submit(self._safe_apply_func, func, item, func_args, func_kwargs)
                future_to_idx[future] = i
            
            # Collect results
            results = [None] * total_items
            completed = 0
            
            for future in as_completed(future_to_idx, timeout=self.config.timeout):
                idx = future_to_idx[future]
                
                try:
                    results[idx] = future.result()
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total_items)
                    
                    if completed % max(1, total_items // 20) == 0:
                        self.logger.info(f"Completed {completed}/{total_items} items "
                                       f"({completed/total_items*100:.1f}%)")
                        
                except Exception as e:
                    self.logger.error(f"Error processing item {idx}: {e}")
                    raise
        
        self.logger.info(f"Parallel processing complete: {completed} items processed")
        return results
    
    def _safe_apply_func(self, func: Callable, data: Any, args: tuple, kwargs: dict) -> Any:
        """Safely apply function with error handling and memory optimization"""
        try:
            with self.memory_optimizer.temporary_memory_optimization():
                if args or kwargs:
                    return func(data, *args, **kwargs)
                else:
                    return func(data)
        except Exception as e:
            self.logger.error(f"Error in parallel function execution: {e}")
            raise
    
    def _split_dataframe(self, df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """Split DataFrame into chunks"""
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            chunks.append(chunk)
        return chunks
    
    def parallel_file_processing(self,
                                file_paths: List[Union[str, Path]],
                                process_func: Callable[[str], Any],
                                combine_func: Optional[Callable[[List[Any]], Any]] = None,
                                output_path: Optional[Union[str, Path]] = None) -> Any:
        """
        Process multiple files in parallel
        
        Parameters:
        -----------
        file_paths : List of file paths
        process_func : Function to process each file
        combine_func : Function to combine results (default: concatenate DataFrames)
        output_path : Optional output path for combined results
        
        Returns:
        --------
        Combined results or output path
        """
        file_paths = [Path(fp) for fp in file_paths]
        total_files = len(file_paths)
        
        self.logger.info(f"Processing {total_files} files in parallel")
        
        def default_combine(results):
            # Default combination for DataFrames
            valid_results = [r for r in results if r is not None]
            if valid_results and isinstance(valid_results[0], pd.DataFrame):
                return pd.concat(valid_results, ignore_index=True)
            return valid_results
        
        combine_func = combine_func or default_combine
        
        # Process files in parallel
        results = self.parallel_apply(
            data=[str(fp) for fp in file_paths],
            func=process_func,
            progress_callback=lambda c, t: self.logger.info(f"Processed {c}/{t} files")
        )
        
        # Combine results
        combined_result = combine_func(results)
        
        if output_path and isinstance(combined_result, pd.DataFrame):
            output_path = Path(output_path)
            combined_result.to_csv(output_path, index=False)
            self.logger.info(f"Combined results saved to {output_path}")
            return str(output_path)
        
        return combined_result

class AsyncProcessor:
    """Asynchronous processing for I/O-bound operations"""
    
    def __init__(
        self,
        max_workers: int = 10,
        logger: Optional[Union[logging.Logger, "ETADLogger"]] = None,
    ):
        self.max_workers = max_workers
        self.logger = _resolve_logger(logger, "AsyncProcessor")
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.shutdown_event = threading.Event()
        
    def start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {self.max_workers} async workers")
    
    def stop_workers(self):
        """Stop worker threads"""
        self.shutdown_event.set()
        
        # Signal all workers to stop
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        self.shutdown_event.clear()
        self.logger.info("Stopped async workers")
    
    def _worker_loop(self, worker_id: int):
        """Worker thread main loop"""
        while not self.shutdown_event.is_set():
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                task_id, func, args, kwargs = task
                
                try:
                    result = func(*args, **kwargs)
                    self.result_queue.put((task_id, 'success', result))
                except Exception as e:
                    self.result_queue.put((task_id, 'error', str(e)))
                finally:
                    self.task_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs):
        """Submit a task for async processing"""
        self.task_queue.put((task_id, func, args, kwargs))
    
    def get_results(self, timeout: Optional[float] = None) -> List[Tuple[str, str, Any]]:
        """Get all available results"""
        results = []
        end_time = time.time() + (timeout or 0)
        
        while True:
            try:
                remaining_time = end_time - time.time() if timeout else 1.0
                if timeout and remaining_time <= 0:
                    break
                
                result = self.result_queue.get(timeout=max(0.1, remaining_time))
                results.append(result)
                
                if self.result_queue.empty():
                    break
                    
            except queue.Empty:
                break
        
        return results

class PipelineProcessor:
    """Process data through a pipeline of parallel operations"""
    
    def __init__(self, 
                 config: Optional[ParallelProcessingConfig] = None,
                 logger: Optional[Union[logging.Logger, "ETADLogger"]] = None):
        self.config = config or ParallelProcessingConfig()
        self.logger = _resolve_logger(logger, "PipelineProcessor")
        self.parallel_processor = ParallelProcessor(config, logger)
        self.pipeline_steps: List[Dict[str, Any]] = []
        
    def add_step(self, 
                 name: str,
                 func: Callable,
                 parallel: bool = True,
                 cache_result: bool = False,
                 **kwargs):
        """
        Add a processing step to the pipeline
        
        Parameters:
        -----------
        name : str
            Step name
        func : Callable
            Processing function
        parallel : bool
            Whether to run in parallel
        cache_result : bool
            Whether to cache the result
        **kwargs
            Additional arguments for the function
        """
        step = {
            'name': name,
            'func': func,
            'parallel': parallel,
            'cache_result': cache_result,
            'kwargs': kwargs
        }
        self.pipeline_steps.append(step)
        self.logger.info(f"Added pipeline step: {name}")
    
    def process(self, data: Any, progress_callback: Optional[Callable] = None) -> Any:
        """
        Process data through the pipeline
        
        Parameters:
        -----------
        data : Any
            Input data
        progress_callback : Callable, optional
            Progress callback (step, total_steps)
            
        Returns:
        --------
        Any
            Final processed data
        """
        if not self.pipeline_steps:
            raise ValueError("No pipeline steps defined")
        
        total_steps = len(self.pipeline_steps)
        current_data = data
        
        self.logger.info(f"Starting pipeline with {total_steps} steps")
        
        for i, step in enumerate(self.pipeline_steps):
            step_name = step['name']
            start_time = time.time()
            
            self.logger.info(f"Executing step {i + 1}/{total_steps}: {step_name}")
            
            try:
                # Check cache first
                cache_key = f"pipeline_{step_name}_{hash(str(current_data))}"
                if step['cache_result'] and self.parallel_processor.cache_manager:
                    cached_result = self.parallel_processor.cache_manager.load_cached_dataframe(cache_key)
                    if cached_result is not None:
                        self.logger.info(f"Using cached result for step: {step_name}")
                        current_data = cached_result
                        continue
                
                # Execute step
                if step['parallel'] and hasattr(current_data, '__len__') and len(current_data) > self.config.chunk_size:
                    current_data = self.parallel_processor.parallel_apply(
                        current_data, 
                        step['func'],
                        func_kwargs=step['kwargs']
                    )
                else:
                    current_data = step['func'](current_data, **step['kwargs'])
                
                # Cache result if requested
                if step['cache_result'] and self.parallel_processor.cache_manager and isinstance(current_data, pd.DataFrame):
                    self.parallel_processor.cache_manager.cache_dataframe(current_data, cache_key)
                
                execution_time = time.time() - start_time
                self.logger.info(f"Step {step_name} completed in {execution_time:.2f}s")
                
                if progress_callback:
                    progress_callback(i + 1, total_steps)
                    
            except Exception as e:
                self.logger.error(f"Error in pipeline step {step_name}: {e}")
                raise
        
        self.logger.info("Pipeline processing complete")
        return current_data
    
    def clear_pipeline(self):
        """Clear all pipeline steps"""
        self.pipeline_steps.clear()
        self.logger.info("Pipeline cleared")

def parallelize(func: Optional[Callable] = None, 
               max_workers: Optional[int] = None,
               use_threads: bool = False,
               chunk_size: int = 1000):
    """
    Decorator to automatically parallelize function execution
    
    Parameters:
    -----------
    func : Callable, optional
        Function to parallelize
    max_workers : int, optional
        Maximum number of workers
    use_threads : bool
        Use threads instead of processes
    chunk_size : int
        Chunk size for data splitting
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Check if first argument is suitable for parallelization
            if args and hasattr(args[0], '__len__') and len(args[0]) > chunk_size:
                config = ParallelProcessingConfig(
                    max_workers=max_workers,
                    use_threads=use_threads,
                    chunk_size=chunk_size
                )
                processor = ParallelProcessor(config)
                
                # Extract data and other arguments
                data = args[0]
                other_args = args[1:]
                
                return processor.parallel_apply(data, f, other_args, kwargs)
            else:
                # Execute normally if data is too small
                return f(*args, **kwargs)
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

# Global instances
default_processor = ParallelProcessor()
default_pipeline = PipelineProcessor()

# Convenience functions
def parallel_apply(data, func, **kwargs):
    """Apply function in parallel to data"""
    return default_processor.parallel_apply(data, func, **kwargs)

def process_files_parallel(file_paths, process_func, **kwargs):
    """Process multiple files in parallel"""
    return default_processor.parallel_file_processing(file_paths, process_func, **kwargs)
