"""
Utility functions for plotting: regression stats, axis helpers, grid layouts.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil


def calculate_regression_stats(x, y):
    """
    Calculate linear regression statistics.

    Parameters:
    -----------
    x, y : array-like data

    Returns:
    --------
    dict with n, slope, intercept, r_squared, correlation (or None if insufficient data)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return None

    coefficients = np.polyfit(x, y, 1)
    slope, intercept = coefficients
    correlation = np.corrcoef(x, y)[0, 1]

    return {
        'n': len(x),
        'slope': slope,
        'intercept': intercept,
        'r_squared': correlation ** 2,
        'correlation': correlation
    }


def format_equation(slope, intercept):
    """Format regression equation as string."""
    sign = '+' if intercept >= 0 else '-'
    return f"y = {slope:.3f}x {sign} {abs(intercept):.2f}"


def format_stats_text(stats, show_equation=True, extra_text=None):
    """
    Format statistics for display in text box.

    Parameters:
    -----------
    stats : dict from calculate_regression_stats
    show_equation : bool
    extra_text : str (optional additional text)

    Returns:
    --------
    str formatted text
    """
    if stats is None:
        return "Insufficient data"

    text = f"n = {stats['n']}\nR² = {stats['r_squared']:.3f}"

    if show_equation:
        text += f"\n{format_equation(stats['slope'], stats['intercept'])}"

    if extra_text:
        text += f"\n{extra_text}"

    return text


def get_clean_data(x_data, y_data, outlier_mask=None):
    """
    Extract clean (non-NaN, non-outlier) data.

    Parameters:
    -----------
    x_data, y_data : arrays
    outlier_mask : boolean array (True = outlier)

    Returns:
    --------
    tuple: (x_clean, y_clean, x_outliers, y_outliers)
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    valid_mask = (~np.isnan(x_data)) & (~np.isnan(y_data))

    if outlier_mask is not None:
        outlier_mask = np.asarray(outlier_mask)
        clean_mask = valid_mask & ~outlier_mask
        outlier_plot_mask = valid_mask & outlier_mask
    else:
        clean_mask = valid_mask
        outlier_plot_mask = np.zeros(len(x_data), dtype=bool)

    return (
        x_data[clean_mask],
        y_data[clean_mask],
        x_data[outlier_plot_mask],
        y_data[outlier_plot_mask]
    )


def calculate_axis_limits(data_arrays, padding=0.1, start_zero=True):
    """
    Calculate axis limits from multiple data arrays.

    Parameters:
    -----------
    data_arrays : list of arrays
    padding : float (fraction to add as padding)
    start_zero : bool (if True, min is 0)

    Returns:
    --------
    tuple: (min_val, max_val)
    """
    all_vals = []
    for arr in data_arrays:
        arr = np.asarray(arr)
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            all_vals.extend(valid)

    if len(all_vals) == 0:
        return (0, 100)

    min_val = 0 if start_zero else min(all_vals) * (1 - padding)
    max_val = max(all_vals) * (1 + padding)

    return (min_val, max_val)


def create_grid_layout(n_plots, max_cols=2):
    """
    Create figure and axes for grid layout.

    Parameters:
    -----------
    n_plots : int (number of subplots needed)
    max_cols : int (maximum columns)

    Returns:
    --------
    tuple: (fig, axes_list)
    """
    n_cols = min(n_plots, max_cols)
    n_rows = ceil(n_plots / n_cols)

    from . import PlotConfig
    figsize = PlotConfig.get('figsize_grid')

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten axes array for easy iteration
    if n_plots == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten().tolist()

    # Hide unused subplots
    for idx in range(n_plots, len(axes_list)):
        axes_list[idx].set_visible(False)

    return fig, axes_list[:n_plots]


def create_individual_figure():
    """Create a single figure with one axes."""
    from . import PlotConfig
    figsize = PlotConfig.get('figsize')
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def create_combined_figure():
    """Create a single figure for combined/overlay plot."""
    from . import PlotConfig
    figsize = PlotConfig.get('figsize')
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def add_stats_textbox(ax, stats, position='upper_left', box_color='white'):
    """
    Add statistics text box to axes.

    Parameters:
    -----------
    ax : matplotlib axes
    stats : dict from calculate_regression_stats
    position : str ('upper_left', 'upper_right', 'lower_left', 'lower_right')
    box_color : str or color
    """
    from . import PlotConfig

    if not PlotConfig.get('show_stats') or stats is None:
        return

    text = format_stats_text(stats)

    positions = {
        'upper_left': (0.05, 0.95, 'top', 'left'),
        'upper_right': (0.95, 0.95, 'top', 'right'),
        'lower_left': (0.05, 0.05, 'bottom', 'left'),
        'lower_right': (0.95, 0.05, 'bottom', 'right'),
    }

    x, y, va, ha = positions.get(position, positions['upper_left'])

    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=PlotConfig.get('font_size') - 1,
            verticalalignment=va, horizontalalignment=ha,
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9))


def add_regression_line(ax, stats, x_range, color='green', label='Best fit'):
    """Add regression line to axes."""
    if stats is None:
        return

    x_line = np.array(x_range)
    y_line = stats['slope'] * x_line + stats['intercept']
    ax.plot(x_line, y_line, color=color, linestyle='-',
            linewidth=2, alpha=0.8, label=label)


def add_one_to_one_line(ax, max_val, label='1:1 line'):
    """Add 1:1 reference line to axes."""
    from . import PlotConfig

    if not PlotConfig.get('show_1to1'):
        return

    ax.plot([0, max_val], [0, max_val], 'k--',
            alpha=0.5, linewidth=1.5, label=label)


def setup_equal_axes(ax, max_val):
    """Setup equal aspect ratio axes starting from 0."""
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal', adjustable='box')


def style_axes(ax, xlabel, ylabel, title=None, show_legend=True):
    """Apply consistent styling to axes."""
    from . import PlotConfig

    ax.set_xlabel(xlabel, fontsize=PlotConfig.get('font_size'))
    ax.set_ylabel(ylabel, fontsize=PlotConfig.get('font_size'))

    if title:
        ax.set_title(title, fontsize=PlotConfig.get('title_size'), fontweight='bold')

    if show_legend and PlotConfig.get('show_legend'):
        ax.legend(loc='lower right', fontsize=PlotConfig.get('font_size') - 2)

    ax.grid(True, alpha=PlotConfig.get('grid_alpha'))


def print_stats_table(results_dict, title="Comparison"):
    """
    Print a formatted table of statistics.

    Parameters:
    -----------
    results_dict : dict
        {site_name: stats_dict} or {site_name: {threshold: stats_dict}}
    title : str
    """
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)

    for site_name, site_results in results_dict.items():
        if site_results is None:
            print(f"\n{site_name}: No data")
            continue

        if isinstance(site_results, dict) and 'n' in site_results:
            # Single result
            print(f"\n{site_name}:")
            print(f"  n = {site_results['n']}")
            print(f"  R² = {site_results['r_squared']:.3f}")
            print(f"  Slope = {site_results['slope']:.3f}")
        else:
            # Multiple thresholds/periods
            print(f"\n{site_name}:")
            print(f"{'Key':<15s} {'n':>8s} {'R²':>10s} {'Slope':>10s}")
            print("-" * 45)
            for key, stats in site_results.items():
                if stats:
                    print(f"{str(key):<15s} {stats['n']:>8d} "
                          f"{stats['r_squared']:>10.3f} {stats['slope']:>10.3f}")
                else:
                    print(f"{str(key):<15s} {'--':>8s} {'--':>10s} {'--':>10s}")
