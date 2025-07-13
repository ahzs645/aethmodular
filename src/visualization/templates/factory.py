"""
Visualization Template Factory

This module provides a factory pattern for creating and managing
visualization templates in the AethModular system.
"""

from typing import Dict, Optional, List, Type
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

class VisualizationTemplateFactory:
    """Factory for creating visualization templates"""
    
    _templates: Dict[str, Type[BaseVisualizationTemplate]] = {
        # Time series templates
        'time_series': TimeSeriesTemplate,
        'smoothening_comparison': SmootheningComparisonTemplate,
        'diurnal_patterns': DiurnalPatternTemplate,
        
        # Heatmap templates
        'weekly_heatmap': WeeklyHeatmapTemplate,
        'seasonal_heatmap': SeasonalHeatmapTemplate,
        
        # Scientific analysis templates
        'mac_analysis': MACAnalysisTemplate,
        'correlation_analysis': CorrelationAnalysisTemplate,
        'scatter_plot': ScatterPlotTemplate,
    }
    
    @classmethod
    def create_template(cls, template_type: str, config: Optional[Dict] = None) -> BaseVisualizationTemplate:
        """
        Create a visualization template of the specified type
        
        Args:
            template_type: Type of template to create
            config: Optional configuration dictionary
            
        Returns:
            BaseVisualizationTemplate: The created template instance
            
        Raises:
            ValueError: If template_type is not recognized
        """
        if template_type not in cls._templates:
            available = ', '.join(cls._templates.keys())
            raise ValueError(f"Unknown template type '{template_type}'. Available: {available}")
        
        template_class = cls._templates[template_type]
        return template_class(template_type, config)
    
    @classmethod
    def register_template(cls, template_type: str, template_class: Type[BaseVisualizationTemplate]):
        """
        Register a new template type
        
        Args:
            template_type: Name for the template type
            template_class: Class that implements the template
        """
        if not issubclass(template_class, BaseVisualizationTemplate):
            raise ValueError("template_class must inherit from BaseVisualizationTemplate")
        
        cls._templates[template_type] = template_class
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """
        List all available template types
        
        Returns:
            List[str]: List of template type names
        """
        return list(cls._templates.keys())
    
    @classmethod
    def get_template_info(cls, template_type: str) -> Dict:
        """
        Get information about a specific template type
        
        Args:
            template_type: Type of template to get info for
            
        Returns:
            Dict: Information about the template
        """
        if template_type not in cls._templates:
            raise ValueError(f"Unknown template type '{template_type}'")
        
        template_class = cls._templates[template_type]
        return {
            'name': template_type,
            'class': template_class.__name__,
            'module': template_class.__module__,
            'docstring': template_class.__doc__,
            'required_params': getattr(template_class, 'REQUIRED_PARAMS', []),
            'optional_params': getattr(template_class, 'OPTIONAL_PARAMS', [])
        }
    
    @classmethod
    def list_templates_by_category(cls) -> Dict[str, List[str]]:
        """
        List templates organized by category
        
        Returns:
            Dict[str, List[str]]: Templates organized by category
        """
        categories = {
            'Time Series': ['time_series', 'smoothening_comparison', 'diurnal_patterns'],
            'Heatmaps': ['weekly_heatmap', 'seasonal_heatmap'],
            'Scientific Analysis': ['mac_analysis', 'correlation_analysis', 'scatter_plot']
        }
        
        # Filter to only include available templates
        available_categories = {}
        for category, templates in categories.items():
            available_templates = [t for t in templates if t in cls._templates]
            if available_templates:
                available_categories[category] = available_templates
        
        return available_categories


# Convenience function for quick template creation
def create_plot(template_type: str, config: Optional[Dict] = None, **kwargs):
    """
    Convenience function to create a plot using a template
    
    Args:
        template_type: Type of template to use
        config: Optional configuration
        **kwargs: Parameters to pass to the template's create_plot method
        
    Returns:
        matplotlib.Figure: The created plot
    """
    template = VisualizationTemplateFactory.create_template(template_type, config)
    return template.create_plot(**kwargs)
