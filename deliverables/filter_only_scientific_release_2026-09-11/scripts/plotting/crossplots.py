"""
Cross-plot (scatter) plotting functions.

Usage:
    from plotting import crossplots, PlotConfig

    PlotConfig.set(sites='all', layout='individual')

    # Basic crossplot
    crossplots.scatter(matched_data, x_col='aeth_bc', y_col='filter_ec')

    # Preset crossplots
    crossplots.bc_vs_ec(matched_data)
    crossplots.hips_vs_ftir(matched_data)
    crossplots.hips_vs_aeth(matched_data)
"""

import numpy as np
import matplotlib.pyplot as plt

from . import PlotConfig, resolve_sites, resolve_layout
from .utils import (
    calculate_regression_stats, get_clean_data, calculate_axis_limits,
    create_grid_layout, create_individual_figure, create_combined_figure,
    add_stats_textbox, add_regression_line, add_one_to_one_line,
    setup_equal_axes, style_axes
)


def scatter(data, x_col, y_col, sites=None, layout=None,
            xlabel=None, ylabel=None, title=None,
            outlier_col=None, color_col=None, cmap='plasma',
            equal_axes=None, show_1to1=None):
    """
    Create scatter cross-plots for one or more sites.

    Parameters:
    -----------
    data : dict or DataFrame
        If dict: {site_name: DataFrame}
        If DataFrame: single site data
    x_col, y_col : str
        Column names for x and y data
    sites : str, list, or None
        Sites to plot (None uses PlotConfig default)
    layout : str or None
        'individual', 'grid', 'combined' (None uses default)
    xlabel, ylabel : str (optional)
        Axis labels (auto-generated if None)
    title : str (optional)
        Plot title
    outlier_col : str (optional)
        Column name for outlier boolean mask
    color_col : str (optional)
        Column name for color gradient (e.g., iron concentration)
    cmap : str
        Colormap for color gradient
    equal_axes : bool (optional)
        Override PlotConfig.equal_axes
    show_1to1 : bool (optional)
        Override PlotConfig.show_1to1

    Returns:
    --------
    dict: {site_name: stats_dict}
    """
    sites_list = resolve_sites(sites)
    layout = resolve_layout(layout)

    if equal_axes is None:
        equal_axes = PlotConfig.get('equal_axes')
    if show_1to1 is None:
        show_1to1 = PlotConfig.get('show_1to1')

    # Handle single DataFrame input
    if not isinstance(data, dict):
        data = {sites_list[0]: data}

    # Auto-generate labels
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col

    results = {}

    if layout == 'individual':
        results = _scatter_individual(
            data, sites_list, x_col, y_col,
            xlabel, ylabel, title, outlier_col, color_col, cmap,
            equal_axes, show_1to1
        )
    elif layout == 'grid':
        results = _scatter_grid(
            data, sites_list, x_col, y_col,
            xlabel, ylabel, title, outlier_col, color_col, cmap,
            equal_axes, show_1to1
        )
    elif layout == 'combined':
        results = _scatter_combined(
            data, sites_list, x_col, y_col,
            xlabel, ylabel, title, outlier_col,
            equal_axes, show_1to1
        )

    return results


def _scatter_individual(data, sites_list, x_col, y_col,
                        xlabel, ylabel, title, outlier_col, color_col, cmap,
                        equal_axes, show_1to1):
    """Create individual figures for each site."""
    results = {}

    for site_name in sites_list:
        if site_name not in data:
            print(f"{site_name}: No data")
            continue

        df = data[site_name]
        if x_col not in df.columns or y_col not in df.columns:
            print(f"{site_name}: Missing columns {x_col} or {y_col}")
            continue

        fig, ax = create_individual_figure()
        color = PlotConfig.get_site_color(site_name)

        stats = _plot_single_scatter(
            ax, df, x_col, y_col, color,
            outlier_col, color_col, cmap, equal_axes, show_1to1
        )

        site_title = f"{title} - {site_name}" if title else site_name
        style_axes(ax, xlabel, ylabel, site_title)

        plt.tight_layout()
        plt.show()

        results[site_name] = stats

    return results


def _scatter_grid(data, sites_list, x_col, y_col,
                  xlabel, ylabel, title, outlier_col, color_col, cmap,
                  equal_axes, show_1to1):
    """Create grid layout with all sites."""
    # Filter to sites that have data
    valid_sites = [s for s in sites_list if s in data]
    if len(valid_sites) == 0:
        print("No valid sites with data")
        return {}

    fig, axes = create_grid_layout(len(valid_sites))
    results = {}

    # Calculate common axis limits
    all_x, all_y = [], []
    for site_name in valid_sites:
        df = data[site_name]
        if x_col in df.columns:
            all_x.extend(df[x_col].dropna().values)
        if y_col in df.columns:
            all_y.extend(df[y_col].dropna().values)

    _, max_val = calculate_axis_limits([all_x, all_y])

    for idx, site_name in enumerate(valid_sites):
        ax = axes[idx]
        df = data[site_name]
        color = PlotConfig.get_site_color(site_name)

        stats = _plot_single_scatter(
            ax, df, x_col, y_col, color,
            outlier_col, color_col, cmap, equal_axes, show_1to1,
            fixed_max=max_val
        )

        style_axes(ax, xlabel, ylabel, site_name)
        results[site_name] = stats

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.show()

    return results


def _scatter_combined(data, sites_list, x_col, y_col,
                      xlabel, ylabel, title, outlier_col,
                      equal_axes, show_1to1):
    """Create combined plot with all sites overlaid."""
    fig, ax = create_combined_figure()
    results = {}

    all_x, all_y = [], []

    for site_name in sites_list:
        if site_name not in data:
            continue

        df = data[site_name]
        if x_col not in df.columns or y_col not in df.columns:
            continue

        color = PlotConfig.get_site_color(site_name)
        x_data = df[x_col].values
        y_data = df[y_col].values

        outlier_mask = df[outlier_col].values if outlier_col and outlier_col in df.columns else None
        x_clean, y_clean, _, _ = get_clean_data(x_data, y_data, outlier_mask)

        if len(x_clean) < 3:
            continue

        all_x.extend(x_clean)
        all_y.extend(y_clean)

        # Plot points
        ax.scatter(x_clean, y_clean, color=color, alpha=0.6,
                   s=PlotConfig.get('marker_size'),
                   edgecolors='black', linewidth=0.5,
                   label=f"{site_name} (n={len(x_clean)})")

        # Calculate stats
        stats = calculate_regression_stats(x_clean, y_clean)
        results[site_name] = stats

    # Set axis limits
    if equal_axes and len(all_x) > 0:
        _, max_val = calculate_axis_limits([all_x, all_y])
        setup_equal_axes(ax, max_val)
        if show_1to1:
            add_one_to_one_line(ax, max_val)

    style_axes(ax, xlabel, ylabel, title)

    plt.tight_layout()
    plt.show()

    return results


def _plot_single_scatter(ax, df, x_col, y_col, color,
                         outlier_col, color_col, cmap, equal_axes, show_1to1,
                         fixed_max=None):
    """Plot scatter on a single axes."""
    x_data = df[x_col].values
    y_data = df[y_col].values

    outlier_mask = df[outlier_col].values if outlier_col and outlier_col in df.columns else None
    x_clean, y_clean, x_outliers, y_outliers = get_clean_data(x_data, y_data, outlier_mask)

    if len(x_clean) < 3:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        return None

    # Determine if using color gradient
    if color_col and color_col in df.columns:
        # Color gradient plot
        color_data = df[color_col].values
        valid_mask = (~np.isnan(x_data)) & (~np.isnan(y_data)) & (~np.isnan(color_data))
        if outlier_mask is not None:
            valid_mask = valid_mask & ~outlier_mask

        scatter = ax.scatter(x_data[valid_mask], y_data[valid_mask],
                             c=color_data[valid_mask], cmap=cmap,
                             alpha=0.7, s=PlotConfig.get('marker_size'),
                             edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, shrink=0.8, label=color_col)
    else:
        # Single color plot
        ax.scatter(x_clean, y_clean, color=color, alpha=0.6,
                   s=PlotConfig.get('marker_size'),
                   edgecolors='black', linewidth=1,
                   label=f'Data (n={len(x_clean)})')

    # Plot outliers
    if len(x_outliers) > 0:
        ax.scatter(x_outliers, y_outliers, color='red', alpha=0.9,
                   s=150, marker='X', linewidths=2,
                   label=f'Excluded (n={len(x_outliers)})')

    # Calculate regression
    stats = calculate_regression_stats(x_clean, y_clean)

    # Set axis limits
    if fixed_max:
        max_val = fixed_max
    else:
        _, max_val = calculate_axis_limits([x_clean, y_clean, x_outliers, y_outliers])

    if equal_axes:
        setup_equal_axes(ax, max_val)

    # Add regression line
    if stats:
        add_regression_line(ax, stats, [0, max_val])
        add_stats_textbox(ax, stats)

    # Add 1:1 line
    if show_1to1 and equal_axes:
        add_one_to_one_line(ax, max_val)

    return stats


# =============================================================================
# PRESET CROSSPLOTS
# =============================================================================

def bc_vs_ec(data, sites=None, layout=None, **kwargs):
    """
    Plot Aethalometer BC vs Filter EC.

    Parameters:
    -----------
    data : dict of DataFrames
        Must have 'aeth_bc' and 'filter_ec' columns
    sites, layout : see scatter()
    """
    return scatter(
        data, x_col='aeth_bc', y_col='filter_ec',
        xlabel='Aethalometer BC (ng/m³)', ylabel='Filter EC (ng/m³)',
        title='Aethalometer BC vs Filter EC',
        sites=sites, layout=layout, **kwargs
    )


def hips_vs_ftir(data, sites=None, layout=None, **kwargs):
    """
    Plot HIPS Fabs vs FTIR EC.

    Parameters:
    -----------
    data : dict of DataFrames
        Must have 'hips_fabs' and 'ftir_ec' columns (in µg/m³)
    """
    return scatter(
        data, x_col='hips_fabs', y_col='ftir_ec',
        xlabel='HIPS Fabs / MAC (µg/m³)', ylabel='FTIR EC (µg/m³)',
        title='HIPS vs FTIR EC',
        sites=sites, layout=layout, **kwargs
    )


def hips_vs_aeth(data, sites=None, layout=None, **kwargs):
    """
    Plot HIPS Fabs vs Aethalometer BC.

    Parameters:
    -----------
    data : dict of DataFrames
        Must have 'hips_fabs' and 'ir_bcc' columns (in µg/m³)
    """
    return scatter(
        data, x_col='ir_bcc', y_col='hips_fabs',
        xlabel='Aethalometer IR BCc (µg/m³)', ylabel='HIPS Fabs / MAC (µg/m³)',
        title='HIPS vs Aethalometer',
        sites=sites, layout=layout, **kwargs
    )


def with_iron_gradient(data, x_col, y_col, sites=None, layout=None,
                       xlabel=None, ylabel=None, title=None, **kwargs):
    """
    Plot crossplot with iron concentration as color gradient.

    Parameters:
    -----------
    data : dict of DataFrames
        Must have specified x_col, y_col, and 'iron' column
    """
    return scatter(
        data, x_col=x_col, y_col=y_col,
        xlabel=xlabel, ylabel=ylabel, title=title,
        color_col='iron', cmap='plasma',
        sites=sites, layout=layout, **kwargs
    )
