"""
Configuration utilities for visualization templates

This module provides utilities for loading and managing configuration
files for the visualization template system.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Manager for loading and accessing template configurations"""
    
    def __init__(self):
        self.config_dir = Path(__file__).parent / 'config'
        self._plot_styles = None
        self._color_schemes = None
        self._default_parameters = None
    
    @property
    def plot_styles(self) -> Dict[str, Any]:
        """Load and cache plot styles configuration"""
        if self._plot_styles is None:
            self._plot_styles = self._load_config('plot_styles.json')
        return self._plot_styles
    
    @property
    def color_schemes(self) -> Dict[str, Any]:
        """Load and cache color schemes configuration"""
        if self._color_schemes is None:
            self._color_schemes = self._load_config('color_schemes.json')
        return self._color_schemes
    
    @property
    def default_parameters(self) -> Dict[str, Any]:
        """Load and cache default parameters configuration"""
        if self._default_parameters is None:
            self._default_parameters = self._load_config('default_parameters.json')
        return self._default_parameters
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load a configuration file"""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Configuration file {filename} not found. Using empty config.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Error parsing {filename}: {e}. Using empty config.")
            return {}
    
    def get_style_config(self, style_name: str = 'default') -> Dict[str, Any]:
        """Get style configuration by name"""
        styles = self.plot_styles
        if style_name in styles:
            return styles[style_name]
        elif 'default' in styles:
            print(f"Warning: Style '{style_name}' not found. Using 'default'.")
            return styles['default']
        else:
            print("Warning: No styles found. Using minimal defaults.")
            return {
                'figsize': [12, 8],
                'style': 'whitegrid',
                'color_palette': 'Set1',
                'font_size': 12,
                'save_format': 'png',
                'dpi': 300
            }
    
    def get_color_scheme(self, scheme_name: str) -> Dict[str, str]:
        """Get color scheme by name"""
        schemes = self.color_schemes
        if scheme_name in schemes:
            return schemes[scheme_name]
        else:
            print(f"Warning: Color scheme '{scheme_name}' not found. Using default colors.")
            return {}
    
    def get_template_defaults(self, template_name: str) -> Dict[str, Any]:
        """Get default parameters for a template"""
        defaults = self.default_parameters
        if template_name in defaults:
            return defaults[template_name]
        else:
            print(f"Warning: No defaults found for template '{template_name}'.")
            return {}
    
    def merge_configs(self, user_config: Optional[Dict] = None, 
                     style_name: str = 'default',
                     template_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Merge multiple configuration sources
        
        Args:
            user_config: User-provided configuration
            style_name: Name of style configuration to use
            template_name: Name of template to get defaults for
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        # Start with style config
        config = self.get_style_config(style_name).copy()
        
        # Add template defaults if specified
        if template_name:
            template_defaults = self.get_template_defaults(template_name)
            config.update(template_defaults)
        
        # Apply user overrides
        if user_config:
            config.update(user_config)
        
        return config

# Global config manager instance
config_manager = ConfigManager()

def load_config(style_name: str = 'default', 
               template_name: Optional[str] = None,
               user_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to load merged configuration
    
    Args:
        style_name: Name of style configuration
        template_name: Name of template for defaults
        user_config: User-provided overrides
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    return config_manager.merge_configs(user_config, style_name, template_name)
