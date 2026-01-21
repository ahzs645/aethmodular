"""
Modular plotting package for multi-site aethalometer analysis.

Usage:
    from plotting import PlotConfig, crossplots, timeseries, distributions

    # Set defaults once at top of notebook
    PlotConfig.set(
        sites='all',           # 'all', ['Beijing', 'JPL'], or 'Beijing'
        layout='individual',   # 'individual', 'grid', or 'combined'
        figsize=(10, 8),
        show_stats=True,
        show_1to1=True
    )

    # All subsequent plots use those defaults
    crossplots.bc_vs_ec(matched_data)
    timeseries.bc(aethalometer_data)

    # Override for specific plot
    crossplots.bc_vs_ec(data, sites=['Beijing'], layout='grid')
"""

import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SITES


class PlotConfig:
    """
    Global configuration for plotting defaults.
    Set once at the top of a notebook, used by all plotting functions.
    """

    # Default settings
    _defaults = {
        'sites': 'all',              # 'all', list of site names, or single site name
        'layout': 'individual',      # 'individual', 'grid', 'combined'
        'figsize': (10, 8),          # Default figure size
        'figsize_grid': (14, 12),    # Figure size for grid layout
        'show_stats': True,          # Show R², slope, n text box
        'show_1to1': True,           # Show 1:1 reference line
        'show_legend': True,         # Show legend
        'equal_axes': True,          # Lock axes to 1:1 aspect ratio
        'grid_alpha': 0.3,           # Grid transparency
        'marker_size': 80,           # Scatter marker size
        'line_width': 1.5,           # Line width for plots
        'font_size': 11,             # Base font size
        'title_size': 13,            # Title font size
        'dpi': 100,                  # Figure DPI
    }

    # Current settings (copy of defaults, can be modified)
    _current = _defaults.copy()

    @classmethod
    def set(cls, **kwargs):
        """
        Set plotting defaults.

        Parameters:
        -----------
        sites : str or list
            'all' for all sites, list like ['Beijing', 'JPL'], or single 'Beijing'
        layout : str
            'individual' - one figure per site
            'grid' - all sites in a grid (e.g., 2x2)
            'combined' - all sites overlaid on one axes
        figsize : tuple
            Default figure size (width, height)
        show_stats : bool
            Show statistics text box
        show_1to1 : bool
            Show 1:1 reference line
        equal_axes : bool
            Lock x and y axes to same scale

        Example:
        --------
        PlotConfig.set(sites='all', layout='individual', figsize=(12, 8))
        """
        for key, value in kwargs.items():
            if key in cls._defaults:
                cls._current[key] = value
            else:
                raise ValueError(f"Unknown config option: {key}. "
                               f"Valid options: {list(cls._defaults.keys())}")

    @classmethod
    def get(cls, key):
        """Get a config value."""
        return cls._current.get(key, cls._defaults.get(key))

    @classmethod
    def get_all(cls):
        """Get all current config values."""
        return cls._current.copy()

    @classmethod
    def reset(cls):
        """Reset all settings to defaults."""
        cls._current = cls._defaults.copy()

    @classmethod
    def show(cls):
        """Print current configuration."""
        print("Current PlotConfig settings:")
        print("-" * 40)
        for key, value in cls._current.items():
            print(f"  {key}: {value}")

    @classmethod
    def get_sites_list(cls, sites_override=None):
        """
        Get list of site names to plot.

        Parameters:
        -----------
        sites_override : str or list (optional)
            Override the default sites setting

        Returns:
        --------
        list of site names
        """
        sites = sites_override if sites_override is not None else cls._current['sites']

        if sites == 'all':
            return list(SITES.keys())
        elif isinstance(sites, str):
            return [sites]
        elif isinstance(sites, list):
            return sites
        else:
            raise ValueError(f"Invalid sites value: {sites}")

    @classmethod
    def get_site_color(cls, site_name):
        """Get the configured color for a site."""
        return SITES.get(site_name, {}).get('color', '#333333')

    @classmethod
    def get_site_config(cls, site_name):
        """Get full config dict for a site."""
        return SITES.get(site_name, {})


def resolve_sites(sites=None):
    """
    Resolve sites parameter to a list of site names.
    Helper function for plotting functions.

    Parameters:
    -----------
    sites : str, list, or None
        If None, uses PlotConfig default

    Returns:
    --------
    list of site names
    """
    return PlotConfig.get_sites_list(sites)


def resolve_layout(layout=None):
    """
    Resolve layout parameter.
    Helper function for plotting functions.

    Parameters:
    -----------
    layout : str or None
        If None, uses PlotConfig default

    Returns:
    --------
    str: 'individual', 'grid', or 'combined'
    """
    if layout is None:
        layout = PlotConfig.get('layout')

    valid_layouts = ['individual', 'grid', 'combined']
    if layout not in valid_layouts:
        raise ValueError(f"Invalid layout: {layout}. Must be one of {valid_layouts}")

    return layout


# Import submodules for convenient access
from . import utils
from . import crossplots
from . import timeseries
from . import distributions
from . import comparisons

# Import commonly used utility functions for convenience
from .utils import calculate_regression_stats

__all__ = [
    'PlotConfig',
    'resolve_sites',
    'resolve_layout',
    'utils',
    'crossplots',
    'timeseries',
    'distributions',
    'comparisons',
    'calculate_regression_stats',
]
