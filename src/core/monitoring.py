"""Performance monitoring and error handling utilities for ETAD analysis"""

import time
import psutil
import functools
import traceback
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.logging.logger import ETADLogger

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    execution_time: float = 0.0
    memory_usage: float = 0.0  # MB
    peak_memory: float = 0.0   # MB
    cpu_usage: float = 0.0     # %
    data_size: int = 0         # Number of records processed
    throughput: float = 0.0    # Records per second
    timestamp: datetime = field(default_factory=datetime.now)
    function_name: str = ""
    module_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'peak_memory': self.peak_memory,
            'cpu_usage': self.cpu_usage,
            'data_size': self.data_size,
            'throughput': self.throughput,
            'timestamp': self.timestamp.isoformat(),
            'function_name': self.function_name,
            'module_name': self.module_name
        }

class PerformanceMonitor:
    """Monitor performance metrics for ETAD analysis functions"""
    
    def __init__(self, logger: Optional[ETADLogger] = None):
        self.logger = logger or ETADLogger("PerformanceMonitor")
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_monitors: Dict[str, Dict[str, Any]] = {}
        
    def monitor(self, include_memory: bool = True, include_cpu: bool = True):
        """
        Decorator to monitor function performance
        
        Parameters:
        -----------
        include_memory : bool
            Whether to monitor memory usage
        include_cpu : bool
            Whether to monitor CPU usage
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._monitor_execution(
                    func, args, kwargs, include_memory, include_cpu
                )
            return wrapper
        return decorator
    
    def _monitor_execution(
        self, 
        func: Callable, 
        args: tuple, 
        kwargs: dict,
        include_memory: bool = True,
        include_cpu: bool = True
    ) -> Any:
        """Execute function with performance monitoring"""
        
        process = psutil.Process()
        start_time = time.time()
        
        # Initial measurements
        if include_memory:
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        if include_cpu:
            initial_cpu = process.cpu_percent()
        
        peak_memory = initial_memory if include_memory else 0.0
        data_size = 0
        
        # Try to estimate data size from arguments
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                data_size += len(arg)
            elif isinstance(arg, np.ndarray):
                data_size += arg.size
            elif isinstance(arg, (list, tuple)):
                data_size += len(arg)
        
        try:
            # Execute function
            with self._memory_tracker(process) as memory_tracker:
                result = func(*args, **kwargs)
            
            # Final measurements
            end_time = time.time()
            execution_time = end_time - start_time
            
            if include_memory:
                final_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = memory_tracker.peak_memory if memory_tracker else final_memory
            else:
                final_memory = 0.0
                
            if include_cpu:
                # Get average CPU usage during execution
                cpu_usage = process.cpu_percent()
            else:
                cpu_usage = 0.0
            
            # Calculate throughput
            throughput = data_size / execution_time if execution_time > 0 else 0.0
            
            # Create metrics
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=final_memory,
                peak_memory=peak_memory,
                cpu_usage=cpu_usage,
                data_size=data_size,
                throughput=throughput,
                function_name=func.__name__,
                module_name=func.__module__
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Log performance
            self.logger.info(
                f"Performance metrics for {func.__name__}",
                extra={
                    'performance_metrics': metrics.to_dict(),
                    'execution_time': execution_time,
                    'memory_peak_mb': peak_memory,
                    'throughput_per_sec': throughput
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                f"Function {func.__name__} failed after {execution_time:.2f}s",
                extra={
                    'function_name': func.__name__,
                    'module_name': func.__module__,
                    'execution_time': execution_time,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
            )
            raise
    
    @contextmanager
    def _memory_tracker(self, process):
        """Context manager to track peak memory usage"""
        class MemoryTracker:
            def __init__(self):
                self.peak_memory = 0.0
                
        tracker = MemoryTracker()
        tracker.peak_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            yield tracker
        finally:
            current_memory = process.memory_info().rss / 1024 / 1024
            tracker.peak_memory = max(tracker.peak_memory, current_memory)
    
    def get_metrics_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        metrics = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        execution_times = [m.execution_time for m in metrics]
        memory_usage = [m.memory_usage for m in metrics]
        throughputs = [m.throughput for m in metrics if m.throughput > 0]
        
        return {
            'total_functions_monitored': len(metrics),
            'execution_time': {
                'mean': np.mean(execution_times),
                'median': np.median(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times),
                'std': np.std(execution_times)
            },
            'memory_usage_mb': {
                'mean': np.mean(memory_usage),
                'median': np.median(memory_usage),
                'min': np.min(memory_usage),
                'max': np.max(memory_usage),
                'std': np.std(memory_usage)
            },
            'throughput_per_sec': {
                'mean': np.mean(throughputs) if throughputs else 0.0,
                'median': np.median(throughputs) if throughputs else 0.0,
                'min': np.min(throughputs) if throughputs else 0.0,
                'max': np.max(throughputs) if throughputs else 0.0
            },
            'time_range': {
                'start': metrics[0].timestamp.isoformat(),
                'end': metrics[-1].timestamp.isoformat()
            }
        }
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to CSV file"""
        if not self.metrics_history:
            self.logger.warning("No metrics to export")
            return
        
        df = pd.DataFrame([m.to_dict() for m in self.metrics_history])
        df.to_csv(filepath, index=False)
        self.logger.info(f"Exported {len(self.metrics_history)} metrics to {filepath}")
    
    def clear_metrics(self) -> None:
        """Clear metrics history"""
        count = len(self.metrics_history)
        self.metrics_history.clear()
        self.logger.info(f"Cleared {count} metrics from history")

class ErrorHandler:
    """Centralized error handling for ETAD analysis"""
    
    def __init__(self, logger: Optional[ETADLogger] = None):
        self.logger = logger or ETADLogger("ErrorHandler")
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        
    def handle_with_retry(
        self, 
        max_retries: int = 3, 
        backoff_factor: float = 1.0,
        exceptions: Tuple[Exception, ...] = (Exception,)
    ):
        """
        Decorator to handle errors with retry logic
        
        Parameters:
        -----------
        max_retries : int
            Maximum number of retry attempts
        backoff_factor : float
            Factor to increase delay between retries
        exceptions : tuple
            Tuple of exception types to catch and retry
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_retry(
                    func, args, kwargs, max_retries, backoff_factor, exceptions
                )
            return wrapper
        return decorator
    
    def _execute_with_retry(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        max_retries: int,
        backoff_factor: float,
        exceptions: Tuple[Exception, ...]
    ) -> Any:
        """Execute function with retry logic"""
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except exceptions as e:
                last_exception = e
                error_key = f"{func.__name__}:{type(e).__name__}"
                self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
                
                error_info = {
                    'timestamp': datetime.now().isoformat(),
                    'function_name': func.__name__,
                    'module_name': func.__module__,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'attempt': attempt + 1,
                    'max_retries': max_retries,
                    'traceback': traceback.format_exc()
                }
                
                self.error_history.append(error_info)
                
                if attempt < max_retries:
                    delay = backoff_factor * (2 ** attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}, "
                        f"retrying in {delay:.1f}s",
                        extra=error_info
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"All {max_retries + 1} attempts failed for {func.__name__}",
                        extra=error_info
                    )
        
        # Re-raise the last exception if all retries failed
        raise last_exception
    
    def graceful_degradation(self, default_value: Any = None, log_error: bool = True):
        """
        Decorator for graceful degradation - return default value on error
        
        Parameters:
        -----------
        default_value : Any
            Value to return if function fails
        log_error : bool
            Whether to log the error
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if log_error:
                        error_info = {
                            'function_name': func.__name__,
                            'module_name': func.__module__,
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                            'default_value': str(default_value),
                            'traceback': traceback.format_exc()
                        }
                        
                        self.logger.warning(
                            f"Function {func.__name__} failed, returning default value",
                            extra=error_info
                        )
                        
                        self.error_history.append({
                            **error_info,
                            'timestamp': datetime.now().isoformat(),
                            'handled_with_default': True
                        })
                    
                    return default_value
            return wrapper
        return decorator
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered"""
        if not self.error_history:
            return {"message": "No errors recorded"}
        
        error_types = {}
        function_errors = {}
        
        for error in self.error_history:
            error_type = error['error_type']
            function_name = error['function_name']
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            function_errors[function_name] = function_errors.get(function_name, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'unique_error_types': len(error_types),
            'error_counts_by_type': error_types,
            'error_counts_by_function': function_errors,
            'most_recent_error': self.error_history[-1] if self.error_history else None,
            'time_range': {
                'first_error': self.error_history[0]['timestamp'],
                'last_error': self.error_history[-1]['timestamp']
            } if self.error_history else None
        }

class SystemMonitor:
    """Monitor system resources during analysis"""
    
    def __init__(self, logger: Optional[ETADLogger] = None):
        self.logger = logger or ETADLogger("SystemMonitor")
        self.monitoring = False
        
    @contextmanager
    def monitor_resources(self, interval: float = 1.0, log_interval: int = 60):
        """
        Context manager to monitor system resources
        
        Parameters:
        -----------
        interval : float
            Monitoring interval in seconds
        log_interval : int
            Logging interval in seconds
        """
        import threading
        
        self.monitoring = True
        resource_data = []
        
        def monitor_loop():
            last_log_time = time.time()
            
            while self.monitoring:
                try:
                    # Get system metrics
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    current_time = time.time()
                    resource_data.append({
                        'timestamp': current_time,
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_available_gb': memory.available / (1024**3),
                        'disk_percent': disk.percent,
                        'disk_free_gb': disk.free / (1024**3)
                    })
                    
                    # Log periodically
                    if current_time - last_log_time >= log_interval:
                        recent_data = resource_data[-min(10, len(resource_data)):]
                        avg_cpu = np.mean([d['cpu_percent'] for d in recent_data])
                        avg_memory = np.mean([d['memory_percent'] for d in recent_data])
                        
                        self.logger.info(
                            f"System resources - CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%",
                            extra={
                                'avg_cpu_percent': avg_cpu,
                                'avg_memory_percent': avg_memory,
                                'current_memory_available_gb': memory.available / (1024**3),
                                'current_disk_free_gb': disk.free / (1024**3)
                            }
                        )
                        last_log_time = current_time
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in resource monitoring: {e}")
                    break
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        try:
            yield resource_data
        finally:
            self.monitoring = False
            monitor_thread.join(timeout=2.0)
            
            if resource_data:
                # Log final summary
                cpu_values = [d['cpu_percent'] for d in resource_data]
                memory_values = [d['memory_percent'] for d in resource_data]
                
                self.logger.info(
                    "Resource monitoring summary",
                    extra={
                        'monitoring_duration': len(resource_data) * interval,
                        'avg_cpu_percent': np.mean(cpu_values),
                        'max_cpu_percent': np.max(cpu_values),
                        'avg_memory_percent': np.mean(memory_values),
                        'max_memory_percent': np.max(memory_values),
                        'samples_collected': len(resource_data)
                    }
                )

# Global instances for easy access
performance_monitor = PerformanceMonitor()
error_handler = ErrorHandler()
system_monitor = SystemMonitor()

# Convenience decorators
monitor_performance = performance_monitor.monitor
handle_with_retry = error_handler.handle_with_retry
graceful_degradation = error_handler.graceful_degradation
