"""
Comparison plotting functions: before/after, threshold analysis, outlier removal.

Usage:
    from plotting import comparisons, PlotConfig

    PlotConfig.set(sites='all', layout='grid')

    # Before/after outlier removal
    comparisons.before_after_outliers(matched_data)

    # Threshold analysis
    comparisons.threshold_analysis(matched_data, thresholds=[1, 2.5, 4, 5])

    # Flow period comparison (before/after flow fix)
    comparisons.flow_periods(matched_data)
"""

import numpy as np
import matplotlib.pyplot as plt

from . import PlotConfig, resolve_sites, resolve_layout
from .utils import (
    calculate_regression_stats, get_clean_data, calculate_axis_limits,
    create_grid_layout, create_individual_figure,
    add_stats_textbox, add_regression_line, add_one_to_one_line,
    setup_equal_axes, style_axes, format_stats_text
)


def before_after_outliers(data, x_col='aeth_bc', y_col='filter_ec',
                          outlier_col='is_excluded', sites=None, layout=None,
                          xlabel=None, ylabel=None):
    """
    Create side-by-side plots showing data before and after outlier removal.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame}
    x_col, y_col : str
        Column names for x and y data
    outlier_col : str
        Column name for outlier boolean mask
    sites : str, list, or None
    layout : str or None
        Only 'individual' is supported for this plot type
    xlabel, ylabel : str (optional)

    Returns:
    --------
    dict: {site_name: {'all': stats, 'clean': stats}}
    """
    sites_list = resolve_sites(sites)

    if xlabel is None:
        xlabel = f'{x_col}'
    if ylabel is None:
        ylabel = f'{y_col}'

    results = {}

    for site_name in sites_list:
        if site_name not in data:
            continue

        df = data[site_name]
        if x_col not in df.columns or y_col not in df.columns:
            continue

        color = PlotConfig.get_site_color(site_name)
        x_data = df[x_col].values
        y_data = df[y_col].values

        outlier_mask = df[outlier_col].values if outlier_col in df.columns else None

        # Calculate shared axis limits
        _, max_val = calculate_axis_limits([x_data, y_data])

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: All data
        ax1 = axes[0]
        stats_all = _plot_comparison_panel(
            ax1, x_data, y_data, None, color, max_val,
            xlabel, ylabel, 'Before: All Data'
        )

        # Highlight outliers on left plot
        if outlier_mask is not None and outlier_mask.any():
            ax1.scatter(x_data[outlier_mask], y_data[outlier_mask],
                        color='red', alpha=0.9, s=200, marker='X', linewidths=3,
                        label=f'Outliers (n={outlier_mask.sum()})')
            ax1.legend(loc='lower right', fontsize=8)

        # Right: Outliers removed
        ax2 = axes[1]
        stats_clean = _plot_comparison_panel(
            ax2, x_data, y_data, outlier_mask, color, max_val,
            xlabel, ylabel, 'After: Outliers Removed'
        )

        fig.suptitle(f'{site_name}: Outlier Removal Impact',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        results[site_name] = {'all': stats_all, 'clean': stats_clean}

        # Print improvement
        if stats_all and stats_clean:
            r2_change = stats_clean['r_squared'] - stats_all['r_squared']
            print(f"{site_name}: R² changed by {r2_change:+.3f} "
                  f"({stats_all['r_squared']:.3f} → {stats_clean['r_squared']:.3f})")

    return results


def _plot_comparison_panel(ax, x_data, y_data, outlier_mask, color, max_val,
                           xlabel, ylabel, title):
    """Plot a single panel for before/after comparison."""
    x_clean, y_clean, x_out, y_out = get_clean_data(x_data, y_data, outlier_mask)

    if len(x_clean) < 3:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        style_axes(ax, xlabel, ylabel, title, show_legend=False)
        return None

    # Plot data
    ax.scatter(x_clean, y_clean, color=color, alpha=0.6,
               s=PlotConfig.get('marker_size'),
               edgecolors='black', linewidth=1,
               label=f'Data (n={len(x_clean)})')

    # Calculate regression
    stats = calculate_regression_stats(x_clean, y_clean)

    if stats:
        add_regression_line(ax, stats, [0, max_val])
        add_stats_textbox(ax, stats)

    setup_equal_axes(ax, max_val)
    add_one_to_one_line(ax, max_val)
    style_axes(ax, xlabel, ylabel, title)

    return stats


def threshold_analysis(data, x_col='aeth_bc', y_col='filter_ec',
                       threshold_col='smooth_raw_abs_pct',
                       thresholds=None, sites=None,
                       xlabel=None, ylabel=None):
    """
    Create tiled plots showing effect of different thresholds.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame}
    x_col, y_col : str
        Column names
    threshold_col : str
        Column to use for thresholding
    thresholds : list
        Threshold values (default [1, 2.5, 4, 5])
    sites : str, list, or None
    xlabel, ylabel : str (optional)

    Returns:
    --------
    dict: {site_name: {threshold: stats}}
    """
    sites_list = resolve_sites(sites)

    if thresholds is None:
        thresholds = [1, 2.5, 4, 5]

    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col

    results = {}

    for site_name in sites_list:
        if site_name not in data:
            continue

        df = data[site_name]
        if x_col not in df.columns or y_col not in df.columns:
            continue
        if threshold_col not in df.columns:
            print(f"{site_name}: {threshold_col} not found")
            continue

        color = PlotConfig.get_site_color(site_name)

        # Calculate shared axis limits
        x_data = df[x_col].dropna().values
        y_data = df[y_col].dropna().values
        _, max_val = calculate_axis_limits([x_data, y_data])

        # Create 2x2 grid for 4 thresholds
        n_thresholds = len(thresholds)
        n_cols = min(n_thresholds, 2)
        n_rows = (n_thresholds + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
        if n_thresholds == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        site_results = {}

        for idx, threshold in enumerate(thresholds):
            ax = axes[idx]

            # Filter by threshold
            below = df[df[threshold_col] <= threshold]
            above = df[df[threshold_col] > threshold]
            nan_vals = df[df[threshold_col].isna()]

            n_kept = len(below)
            n_removed = len(above)

            # Plot kept points
            if len(below) > 0:
                ax.scatter(below[x_col], below[y_col],
                           color=color, alpha=0.6, s=80,
                           edgecolors='black', linewidth=1,
                           label=f'≤{threshold}% (n={n_kept})')

            # Plot removed points
            if len(above) > 0:
                ax.scatter(above[x_col], above[y_col],
                           color='gray', alpha=0.5, s=120, marker='X',
                           linewidths=2, label=f'>{threshold}% (n={n_removed})')

            # Calculate regression on kept
            if n_kept >= 3:
                stats = calculate_regression_stats(
                    below[x_col].values, below[y_col].values
                )
                site_results[threshold] = stats

                if stats:
                    add_regression_line(ax, stats, [0, max_val])

                    # Stats box
                    text = f"Threshold: ≤{threshold}%\n"
                    text += f"n = {stats['n']}\nR² = {stats['r_squared']:.3f}"
                    ax.text(0.05, 0.95, text, transform=ax.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
            else:
                site_results[threshold] = None
                ax.text(0.5, 0.5, f'Insufficient data\nat ≤{threshold}%',
                        ha='center', va='center', transform=ax.transAxes)

            setup_equal_axes(ax, max_val)
            add_one_to_one_line(ax, max_val)
            style_axes(ax, xlabel, ylabel, f'Threshold: ≤{threshold}%')

        # Hide unused subplots
        for idx in range(len(thresholds), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'{site_name}: Threshold Analysis',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

        results[site_name] = site_results

    return results


def flow_periods(data, x_col='aeth_bc', y_col='filter_ec',
                 period_col='flow_period', sites=None,
                 xlabel=None, ylabel=None):
    """
    Compare before/after flow fix periods.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame} with 'flow_period' column
    x_col, y_col : str
        Column names
    period_col : str
        Column with period labels ('before', 'after', 'gap')
    sites : str, list, or None
    xlabel, ylabel : str (optional)

    Returns:
    --------
    dict: {site_name: {'before': stats, 'after': stats}}
    """
    sites_list = resolve_sites(sites)

    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col

    results = {}

    period_colors = {
        'before': '#E74C3C',
        'after': '#2ECC71',
        'gap': '#95A5A6'
    }

    for site_name in sites_list:
        if site_name not in data:
            continue

        df = data[site_name]
        if x_col not in df.columns or y_col not in df.columns:
            continue
        if period_col not in df.columns:
            print(f"{site_name}: {period_col} not found")
            continue

        # Check if we have before AND after data
        periods_present = df[period_col].unique()
        has_before = 'before' in periods_present
        has_after = 'after' in periods_present

        if not (has_before and has_after):
            print(f"{site_name}: Only has {list(periods_present)} - skipping")
            continue

        # Calculate shared axis limits
        x_data = df[x_col].dropna().values
        y_data = df[y_col].dropna().values
        _, max_val = calculate_axis_limits([x_data, y_data])

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        site_results = {}

        for idx, period in enumerate(['before', 'after']):
            ax = axes[idx]
            period_data = df[df[period_col] == period]

            if len(period_data) < 3:
                ax.text(0.5, 0.5, f'Insufficient data\n(n={len(period_data)})',
                        ha='center', va='center', transform=ax.transAxes)
                style_axes(ax, xlabel, ylabel, f'{period.upper()}')
                site_results[period] = None
                continue

            x_period = period_data[x_col].values
            y_period = period_data[y_col].values
            x_clean, y_clean, _, _ = get_clean_data(x_period, y_period)

            # Plot data
            ax.scatter(x_clean, y_clean, color=period_colors[period],
                       alpha=0.6, s=PlotConfig.get('marker_size'),
                       edgecolors='black', linewidth=0.5,
                       label=f'Data (n={len(x_clean)})')

            # Calculate regression
            stats = calculate_regression_stats(x_clean, y_clean)
            site_results[period] = stats

            if stats:
                add_regression_line(ax, stats, [0, max_val])

                # Stats box with period label
                text = f"{period.upper()}\n"
                text += f"n = {stats['n']}\nR² = {stats['r_squared']:.3f}\n"
                text += f"Slope = {stats['slope']:.3f}"
                ax.text(0.05, 0.95, text, transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round',
                                  facecolor=period_colors[period], alpha=0.3))

            setup_equal_axes(ax, max_val)
            add_one_to_one_line(ax, max_val)

            period_label = 'BEFORE' if period == 'before' else 'AFTER'
            style_axes(ax, xlabel, ylabel, period_label)

        fig.suptitle(f'{site_name}: Before/After Flow Fix Comparison',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

        results[site_name] = site_results

        # Print summary
        before = site_results.get('before')
        after = site_results.get('after')
        if before and after:
            r2_change = after['r_squared'] - before['r_squared']
            slope_change = after['slope'] - before['slope']
            print(f"{site_name}: R² {before['r_squared']:.3f} → {after['r_squared']:.3f} ({r2_change:+.3f})")
            print(f"           Slope {before['slope']:.3f} → {after['slope']:.3f} ({slope_change:+.3f})")

    return results


def summary_bars(results, metric='r_squared', sites=None):
    """
    Create bar chart comparing metrics across sites/periods.

    Parameters:
    -----------
    results : dict
        Output from flow_periods or threshold_analysis
    metric : str
        'r_squared', 'slope', 'n'
    sites : list (optional)
        Sites to include
    """
    if sites is None:
        sites = list(results.keys())

    # Check structure
    sample_result = list(results.values())[0]
    if sample_result is None:
        print("No valid results")
        return

    # Determine if this is period comparison or threshold comparison
    keys = list(sample_result.keys())

    fig, ax = create_individual_figure()

    x = np.arange(len(sites))
    width = 0.35

    # Get values for each key
    for idx, key in enumerate(keys[:2]):  # Max 2 for side-by-side
        values = []
        for site in sites:
            if site in results and results[site] and key in results[site]:
                stats = results[site][key]
                if stats:
                    values.append(stats.get(metric, 0))
                else:
                    values.append(0)
            else:
                values.append(0)

        offset = (idx - 0.5) * width
        color = '#E74C3C' if idx == 0 else '#2ECC71'
        bars = ax.bar(x + offset, values, width, label=str(key), color=color, alpha=0.7)

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sites, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    if metric == 'r_squared':
        ax.set_ylim(0, 1)
    elif metric == 'slope':
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ideal')
        ax.set_ylim(0, 1.5)

    ax.set_title(f'{metric.replace("_", " ").title()} Comparison',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
