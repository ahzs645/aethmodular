"""Visualization modules for FTIR and Aethalometer analysis"""

from .time_series import TimeSeriesPlotter

# Import template system if available
try:
    from .templates import VisualizationTemplateFactory
    from .templates.factory import create_plot
    __all__ = ['TimeSeriesPlotter', 'VisualizationTemplateFactory', 'create_plot']
except ImportError:
    # Templates not available due to missing dependencies
    __all__ = ['TimeSeriesPlotter']
