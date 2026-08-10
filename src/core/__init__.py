"""Core package for base classes and utilities"""

from .base import BaseAnalyzer
from .exceptions import AnalysisError, DataValidationError, InsufficientDataError

# monitoring / parallel_processing moved to attic/core/ on 2026-07-26 (no
# consumer outside the test suite). See attic/README.md.

__all__ = [
    # Base classes
    'BaseAnalyzer',

    # Exceptions
    'AnalysisError', 'DataValidationError', 'InsufficientDataError',
]
