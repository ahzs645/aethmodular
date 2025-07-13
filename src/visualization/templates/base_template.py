"""
Base Template System for AethModular Visualizations

This module provides the abstract base class and shared functionality
for all visualization templates in the AethModular system.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from pathlib import Path
import json

class BaseVisualizationTemplate(ABC):
    """
    Abstract base class for all visualization templates
    Provides consistent interface and shared functionality
    """
    
    def __init__(self, template_name: str, config: Optional[Dict] = None):
        self.template_name = template_name
        self.config = config or self._load_default_config()
        self.setup_styling()
    
    def _load_default_config(self) -> Dict:
        """Load default configuration for the template"""
        return {
            'figsize': (12, 8),
            'style': 'whitegrid',
            'color_palette': 'Set1',
            'font_size': 12,
            'save_format': 'png',
            'dpi': 300
        }
    
    def setup_styling(self):
        """Configure matplotlib/seaborn styling"""
        sns.set_style(self.config.get('style', 'whitegrid'))
        plt.rcParams['figure.figsize'] = self.config.get('figsize', (12, 8))
        plt.rcParams['font.size'] = self.config.get('font_size', 12)
        plt.rcParams['axes.labelsize'] = self.config.get('font_size', 12) + 2
        plt.rcParams['axes.titlesize'] = self.config.get('font_size', 12) + 4
    
    @abstractmethod
    def validate_parameters(self, **kwargs) -> bool:
        """Validate required parameters for the template"""
        pass
    
    @abstractmethod
    def create_plot(self, **kwargs) -> plt.Figure:
        """Create the visualization - must be implemented by subclasses"""
        pass
    
    def save_plot(self, fig: plt.Figure, path: str, **kwargs):
        """Save plot with consistent formatting"""
        save_kwargs = {
            'dpi': self.config.get('dpi', 300),
            'bbox_inches': 'tight',
            'format': self.config.get('save_format', 'png')
        }
        save_kwargs.update(kwargs)
        fig.savefig(path, **save_kwargs)
    
    def _auto_detect_bc_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect BC columns for plotting"""
        bc_columns = [col for col in data.columns if 'BC' in str(col) and 'c' in str(col)]
        return bc_columns[:5]  # Limit to first 5 columns
    
    def _ensure_datetime_index(self, data: pd.DataFrame, date_column: Optional[str] = None) -> pd.DataFrame:
        """Ensure data has datetime index or convert specified column"""
        if date_column and date_column in data.columns:
            data = data.copy()
            if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
                data[date_column] = pd.to_datetime(data[date_column])
            return data
        elif pd.api.types.is_datetime64_any_dtype(data.index):
            return data
        else:
            raise ValueError("Data must have datetime index or specify valid date_column")
    
    def _get_color_palette(self, n_colors: int) -> List:
        """Get color palette for plotting"""
        palette_name = self.config.get('color_palette', 'Set1')
        if palette_name == 'Set1':
            return plt.cm.Set1(np.linspace(0, 1, max(n_colors, 3)))
        else:
            return sns.color_palette(palette_name, n_colors)
