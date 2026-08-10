"""Comprehensive logging configuration for ETAD analysis modules"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class ETADLogger:
    """
    Centralized logging system for ETAD analysis modules
    
    Provides structured logging with multiple handlers, formatters,
    and analysis-specific context.
    """
    
    def __init__(self, 
                 name: str = 'etad',
                 log_level: str = 'INFO',
                 log_dir: Optional[str] = None,
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True,
                 max_log_size: int = 10 * 1024 * 1024,  # 10 MB
                 backup_count: int = 5):
        """
        Initialize ETAD logger
        
        Parameters:
        -----------
        name : str
            Logger name
        log_level : str
            Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_dir : str, optional
            Directory for log files. If None, uses './logs'
        enable_file_logging : bool
            Enable logging to files
        enable_console_logging : bool
            Enable logging to console
        max_log_size : int
            Maximum log file size in bytes
        backup_count : int
            Number of backup log files to keep
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path('./logs')
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.max_log_size = max_log_size
        self.backup_count = backup_count
        
        # Create log directory
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = self._setup_logger()
        
        # Analysis context for structured logging
        self.analysis_context = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with handlers and formatters"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_formatter = self._get_console_formatter()
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # File handlers
        if self.enable_file_logging:
            # Main log file (rotating)
            main_log_file = self.log_dir / f'{self.name}.log'
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=self.max_log_size,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_formatter = self._get_file_formatter()
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Error log file (errors only)
            error_log_file = self.log_dir / f'{self.name}_errors.log'
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=self.max_log_size,
                backupCount=self.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            logger.addHandler(error_handler)
            
            # Analysis log file (structured analysis events)
            analysis_log_file = self.log_dir / f'{self.name}_analysis.log'
            analysis_handler = logging.handlers.RotatingFileHandler(
                analysis_log_file,
                maxBytes=self.max_log_size,
                backupCount=self.backup_count
            )
            analysis_handler.setLevel(logging.INFO)
            analysis_formatter = self._get_analysis_formatter()
            analysis_handler.setFormatter(analysis_formatter)
            analysis_handler.addFilter(AnalysisLogFilter())
            logger.addHandler(analysis_handler)
        
        return logger
    
    def _get_console_formatter(self) -> logging.Formatter:
        """Get formatter for console output"""
        return logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def _get_file_formatter(self) -> logging.Formatter:
        """Get formatter for file output"""
        return logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(module)s.%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _get_analysis_formatter(self) -> logging.Formatter:
        """Get formatter for analysis events (JSON format)"""
        return AnalysisJSONFormatter()
    
    def set_analysis_context(self, **context):
        """Set analysis context for structured logging"""
        self.analysis_context.update(context)
    
    def clear_analysis_context(self):
        """Clear analysis context"""
        self.analysis_context.clear()
    
    def log_analysis_start(self, analysis_type: str, **kwargs):
        """Log start of analysis with context"""
        context = {
            'event_type': 'analysis_start',
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            **self.analysis_context,
            **kwargs
        }
        self.logger.info("ANALYSIS_EVENT", extra={'analysis_data': context})
    
    def log_analysis_end(self, analysis_type: str, success: bool = True, **kwargs):
        """Log end of analysis with results"""
        context = {
            'event_type': 'analysis_end',
            'analysis_type': analysis_type,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            **self.analysis_context,
            **kwargs
        }
        self.logger.info("ANALYSIS_EVENT", extra={'analysis_data': context})
    
    def log_data_processing(self, operation: str, **kwargs):
        """Log data processing events"""
        context = {
            'event_type': 'data_processing',
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            **self.analysis_context,
            **kwargs
        }
        self.logger.info("ANALYSIS_EVENT", extra={'analysis_data': context})
    
    def log_quality_assessment(self, quality_level: str, **kwargs):
        """Log quality assessment results"""
        context = {
            'event_type': 'quality_assessment',
            'quality_level': quality_level,
            'timestamp': datetime.now().isoformat(),
            **self.analysis_context,
            **kwargs
        }
        self.logger.info("ANALYSIS_EVENT", extra={'analysis_data': context})
    
    def log_performance_metric(self, metric_name: str, value: float, **kwargs):
        """Log performance metrics"""
        context = {
            'event_type': 'performance_metric',
            'metric_name': metric_name,
            'value': value,
            'timestamp': datetime.now().isoformat(),
            **self.analysis_context,
            **kwargs
        }
        self.logger.info("ANALYSIS_EVENT", extra={'analysis_data': context})
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance"""
        return self.logger


class AnalysisLogFilter(logging.Filter):
    """Filter for analysis-specific log events"""
    
    def filter(self, record):
        """Filter records that contain analysis data"""
        return hasattr(record, 'analysis_data')


class AnalysisJSONFormatter(logging.Formatter):
    """JSON formatter for structured analysis logging"""
    
    def format(self, record):
        """Format record as JSON"""
        if hasattr(record, 'analysis_data'):
            return json.dumps(record.analysis_data, indent=None, separators=(',', ':'))
        else:
            # Fallback to standard formatting
            return super().format(record)


class ContextualLogger:
    """
    Context manager for analysis logging with automatic context management
    """
    
    def __init__(self, logger: ETADLogger, analysis_type: str, **context):
        self.logger = logger
        self.analysis_type = analysis_type
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.set_analysis_context(**self.context)
        self.logger.log_analysis_start(self.analysis_type, **self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        success = exc_type is None
        
        end_context = {
            'duration_seconds': duration,
            'error_type': exc_type.__name__ if exc_type else None,
            'error_message': str(exc_val) if exc_val else None
        }
        
        self.logger.log_analysis_end(
            self.analysis_type, 
            success=success, 
            **end_context
        )
        
        self.logger.clear_analysis_context()


# Global logger instances
_loggers: Dict[str, ETADLogger] = {}


def get_logger(name: str = 'etad', **kwargs) -> ETADLogger:
    """
    Get or create a logger instance
    
    Parameters:
    -----------
    name : str
        Logger name
    **kwargs
        Logger configuration parameters
        
    Returns:
    --------
    ETADLogger
        Logger instance
    """
    if name not in _loggers:
        _loggers[name] = ETADLogger(name=name, **kwargs)
    return _loggers[name]


def configure_logging(config: Dict[str, Any]):
    """
    Configure logging from configuration dictionary
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Logging configuration
    """
    logger_name = config.get('name', 'etad')
    
    # Clear existing logger if reconfiguring
    if logger_name in _loggers:
        del _loggers[logger_name]
    
    # Create new logger with configuration
    _loggers[logger_name] = ETADLogger(**config)


def log_function_call(logger_name: str = 'etad'):
    """
    Decorator for automatic function call logging
    
    Parameters:
    -----------
    logger_name : str
        Name of logger to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            
            # Log function entry
            logger.get_logger().debug(
                f"Entering {func.__module__}.{func.__name__} with args={args}, kwargs={kwargs}"
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful exit
                logger.get_logger().debug(
                    f"Exiting {func.__module__}.{func.__name__} successfully"
                )
                
                return result
                
            except Exception as e:
                # Log error exit
                logger.get_logger().error(
                    f"Error in {func.__module__}.{func.__name__}: {str(e)}"
                )
                raise
        
        return wrapper
    return decorator


# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    'name': 'etad',
    'log_level': 'INFO',
    'log_dir': './logs',
    'enable_file_logging': True,
    'enable_console_logging': True,
    'max_log_size': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5
}


# Module-specific logger configurations
MODULE_CONFIGS = {
    'smoothing': {
        'name': 'etad.smoothing',
        'log_level': 'INFO'
    },
    'quality': {
        'name': 'etad.quality',
        'log_level': 'INFO'
    },
    'analysis': {
        'name': 'etad.analysis',
        'log_level': 'INFO'
    },
    'data_loading': {
        'name': 'etad.data',
        'log_level': 'DEBUG'  # More verbose for data issues
    }
}


def setup_module_logging():
    """Setup logging for all modules"""
    for module, config in MODULE_CONFIGS.items():
        full_config = {**DEFAULT_LOGGING_CONFIG, **config}
        get_logger(**full_config)
