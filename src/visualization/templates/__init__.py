"""
AethModular Visualization Templates Package

This package provides a flexible template system for creating consistent visualizations
across different aethalometer analysis workflows.
"""

try:
    from .base_template import BaseVisualizationTemplate
    from .time_series_templates import (
        TimeSeriesTemplate, 
        SmootheningComparisonTemplate, 
        DiurnalPatternTemplate
    )
    from .heatmap_templates import (
        WeeklyHeatmapTemplate, 
        SeasonalHeatmapTemplate
    )
    from .scientific_templates import (
        MACAnalysisTemplate, 
        CorrelationAnalysisTemplate,
        ScatterPlotTemplate
    )
    from .factory import VisualizationTemplateFactory
    
    __all__ = [
        'BaseVisualizationTemplate',
        'TimeSeriesTemplate',
        'SmootheningComparisonTemplate', 
        'DiurnalPatternTemplate',
        'WeeklyHeatmapTemplate',
        'SeasonalHeatmapTemplate',
        'MACAnalysisTemplate',
        'CorrelationAnalysisTemplate',
        'ScatterPlotTemplate',
        'VisualizationTemplateFactory'
    ]

except ImportError as e:
    # Handle case where required dependencies are not available
    print(f"Warning: Some visualization templates may not be available due to missing dependencies: {e}")
    
    # Provide minimal exports
    __all__ = []
