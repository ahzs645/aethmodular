"""Core package for base classes and utilities"""

from .base import BaseAnalyzer
from .exceptions import AnalysisError, DataValidationError, InsufficientDataError
from .monitoring import (
    PerformanceMonitor, ErrorHandler, SystemMonitor,
    performance_monitor, error_handler, system_monitor,
    monitor_performance, handle_with_retry, graceful_degradation
)
from .parallel_processing import (
    ParallelProcessor, AsyncProcessor, PipelineProcessor,
    ParallelProcessingConfig, parallelize, parallel_apply, process_files_parallel
)

__all__ = [
    # Base classes
    'BaseAnalyzer',
    
    # Exceptions
    'AnalysisError', 'DataValidationError', 'InsufficientDataError',
    
    # Monitoring
    'PerformanceMonitor', 'ErrorHandler', 'SystemMonitor',
    'performance_monitor', 'error_handler', 'system_monitor',
    'monitor_performance', 'handle_with_retry', 'graceful_degradation',
    
    # Parallel processing
    'ParallelProcessor', 'AsyncProcessor', 'PipelineProcessor',
    'ParallelProcessingConfig', 'parallelize', 'parallel_apply', 'process_files_parallel'
]
