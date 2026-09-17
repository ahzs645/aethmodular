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

import matplotlib.pyplot as _plt

try:
    from config import SITES, FIGURE_DPI, SAVEFIG_DPI
except ImportError:  # Support importing as research.ftir_hips_chem.scripts.plotting
    from ..config import SITES, FIGURE_DPI, SAVEFIG_DPI


def apply_default_style():
    """
    Apply the library's default plot style: white background, subtle grid.

    This matches the clean look used in Analysis_Tasks_Jan2025.ipynb — pure
    matplotlib defaults for axes/figure facecolor so plots print and publish
    cleanly. We explicitly reset rather than relying on the user not having
    called ``plt.style.use('seaborn-v0_8-darkgrid')`` first.

    Users can override after import:

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')   # or whatever you want

    Or re-apply at any point:

        from plotting import apply_default_style
        apply_default_style()

    Also sets figure/savefig DPI from ``config.SAVEFIG_DPI`` and
    ``config.FIGURE_DPI``, so saved-figure resolution has one knob instead of a
    per-notebook literal. An explicit ``savefig(dpi=...)`` still wins.
    """
    _plt.rcParams.update({
        'figure.dpi':       FIGURE_DPI,
        'savefig.dpi':      SAVEFIG_DPI,
        'axes.facecolor':   'white',
        'figure.facecolor': 'white',
        'savefig.facecolor':'white',
        'axes.edgecolor':   '#333333',
        'axes.grid':         True,
        'grid.color':       '#CCCCCC',
        'grid.alpha':        0.5,
        'grid.linestyle':   '-',
        'grid.linewidth':    0.5,
    })


# Apply on import so any downstream plotting inherits white backgrounds
apply_default_style()


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


def resolve_layout(layout=None, supported=None):
    """
    Resolve layout parameter.
    Helper function for plotting functions.

    Parameters:
    -----------
    layout : str or None
        If None, uses PlotConfig default
    supported : iterable of str, optional
        Layouts this particular plot implements. When the resolved layout is
        not among them, warn and fall back to 'individual' rather than letting
        the caller's if/elif chain fall through.

        Several plots only implement a subset (e.g. no 'combined'), and their
        branch chains had no else -- so setting `PlotConfig.set(layout='combined')`
        globally made them silently draw nothing and return None. A visible
        warning plus a working figure beats a blank cell.

    Returns:
    --------
    str: 'individual', 'grid', or 'combined'
    """
    if layout is None:
        layout = PlotConfig.get('layout')

    valid_layouts = ['individual', 'grid', 'combined']
    if layout not in valid_layouts:
        raise ValueError(f"Invalid layout: {layout}. Must be one of {valid_layouts}")

    if supported is not None and layout not in supported:
        import warnings
        warnings.warn(
            f"layout={layout!r} is not implemented by this plot "
            f"(supports {sorted(supported)}); falling back to 'individual'.",
            stacklevel=2,
        )
        return 'individual'

    return layout


# Import submodules for convenient access
from . import utils
from . import crossplots
from . import timeseries
from . import distributions
from . import comparisons
from . import overlays

# Import commonly used utility functions for convenience
from .utils import calculate_regression_stats, deming, deming_lambda

__all__ = [
    'PlotConfig',
    'apply_default_style',
    'resolve_sites',
    'resolve_layout',
    'utils',
    'crossplots',
    'timeseries',
    'distributions',
    'comparisons',
    'overlays',
    'calculate_regression_stats',
    'deming',
    'deming_lambda',
]
