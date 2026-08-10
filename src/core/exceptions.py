"""Custom exceptions"""

class AnalysisError(ValueError):
    """Base exception for analysis errors"""
    pass

class DataValidationError(AnalysisError):
    """Exception for data validation errors"""
    pass

class InsufficientDataError(AnalysisError):
    """Exception for insufficient data"""
    pass
