"""
Data package for loading and processing

This package includes:
- loaders: Data loading utilities for various formats
- qc: Modular quality control and assessment tools
- processors: Data processing and transformation utilities
"""

# Import main QC classes for easy access
try:
    from .qc import (
        MissingDataAnalyzer,
        QualityClassifier,
        SeasonalPatternAnalyzer,
        FilterSampleMapper,
        QualityVisualizer,
        QualityReportGenerator,
        quick_quality_check
    )
    
    __all__ = [
        'MissingDataAnalyzer',
        'QualityClassifier',
        'SeasonalPatternAnalyzer', 
        'FilterSampleMapper',
        'QualityVisualizer',
        'QualityReportGenerator',
        'quick_quality_check'
    ]
except ImportError:
    # Gracefully handle missing dependencies
    __all__ = []
