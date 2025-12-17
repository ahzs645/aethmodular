"""
Reusable plotting functions for cross-plots, time series, and distributions.

Usage:
    from plotting import plot_crossplot, plot_before_after_comparison, calculate_regression_stats

    # Create a cross-plot
    fig, ax = plt.subplots()
    stats = plot_crossplot(ax, x_data, y_data, 'X Label', 'Y Label', color='blue')
"""

import numpy as np
import matplotlib.pyplot as plt
from config import MAC_VALUE


# =============================================================================
# REGRESSION STATISTICS
# =============================================================================

def calculate_regression_stats(x, y):
    """
    Calculate linear regression statistics.

    Parameters:
    -----------
    x, y : array-like data

    Returns:
    --------
    dict with n, slope, intercept, r_squared (or None if insufficient data)
    """
    # Convert to numpy arrays and remove NaN
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
        'r_squared': correlation ** 2
    }


# =============================================================================
# CROSS-PLOTS
# =============================================================================

def plot_crossplot(ax, x_data, y_data, x_label, y_label,
                   color='blue', outlier_mask=None, equal_axes=True,
                   show_stats=True, show_1to1=True, show_mac=False):
    """
    Create a scatter cross-plot with regression line.

    Parameters:
    -----------
    ax : matplotlib axis
    x_data, y_data : arrays
    x_label, y_label : str
    color : str - color for data points
    outlier_mask : boolean array (True = outlier, shown as red X)
    equal_axes : bool - lock axes to 1:1
    show_stats : bool - show R^2, slope, n text box
    show_1to1 : bool - show 1:1 reference line
    show_mac : bool - show MAC value in stats box

    Returns:
    --------
    dict with regression stats (or None)
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    # Handle outliers
    valid_mask = (~np.isnan(x_data)) & (~np.isnan(y_data))

    if outlier_mask is not None:
        outlier_mask = np.asarray(outlier_mask)
        clean_mask = valid_mask & ~outlier_mask
        outlier_plot_mask = valid_mask & outlier_mask
    else:
        clean_mask = valid_mask
        outlier_plot_mask = np.zeros(len(x_data), dtype=bool)

    x_clean = x_data[clean_mask]
    y_clean = y_data[clean_mask]
    x_outliers = x_data[outlier_plot_mask]
    y_outliers = y_data[outlier_plot_mask]

    if len(x_clean) < 3:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        return None

    # Plot clean data
    ax.scatter(x_clean, y_clean, color=color, alpha=0.6, s=80,
               edgecolors='black', linewidth=1, label=f'Data (n={len(x_clean)})')

    # Plot outliers
    if len(x_outliers) > 0:
        ax.scatter(x_outliers, y_outliers, color='red', alpha=0.9, s=200,
                   marker='X', linewidths=3, label=f'Excluded (n={len(x_outliers)})')

    # Calculate regression
    stats = calculate_regression_stats(x_clean, y_clean)

    if stats:
        # Set axis limits
        if equal_axes:
            all_vals = np.concatenate([x_clean, y_clean])
            if len(x_outliers) > 0:
                all_vals = np.concatenate([all_vals, x_outliers, y_outliers])
            max_val = all_vals.max() * 1.1
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_aspect('equal', adjustable='box')
            x_line = np.array([0, max_val])
        else:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            x_line = np.array([0, x_clean.max() * 1.1])

        # Plot regression line
        y_line = stats['slope'] * x_line + stats['intercept']
        ax.plot(x_line, y_line, 'g-', linewidth=2, alpha=0.8, label='Best fit')

        # Plot 1:1 line
        if show_1to1 and equal_axes:
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5,
                    linewidth=1.5, label='1:1 line')

        # Stats text box
        if show_stats:
            sign = '+' if stats['intercept'] >= 0 else '-'
            eq = f"y = {stats['slope']:.3f}x {sign} {abs(stats['intercept']):.2f}"
            text = f"n = {stats['n']}\nR^2 = {stats['r_squared']:.3f}\n{eq}"
            if show_mac:
                text += f"\n(MAC = {MAC_VALUE} m^2/g)"
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    return stats


def plot_before_after_comparison(matched_df, site_name, site_color,
                                  x_col='aeth_bc', y_col='filter_ec',
                                  outlier_col='is_excluded',
                                  x_label=None, y_label=None):
    """
    Create side-by-side plots showing data before and after outlier removal.

    Parameters:
    -----------
    matched_df : DataFrame
    site_name : str
    site_color : str
    x_col, y_col : column names for x and y data
    outlier_col : column name for outlier flag
    x_label, y_label : axis labels (auto-generated if None)

    Returns:
    --------
    tuple: (fig, stats_all, stats_clean)
    """
    if x_label is None:
        x_label = f'{x_col} (ng/m^3)'
    if y_label is None:
        y_label = f'{y_col} (ng/m^3)'

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    x_data = matched_df[x_col].values
    y_data = matched_df[y_col].values

    if outlier_col in matched_df.columns:
        outlier_mask = matched_df[outlier_col].values
    else:
        outlier_mask = None

    # Calculate axis limits (same for both plots)
    all_vals = np.concatenate([
        x_data[~np.isnan(x_data)],
        y_data[~np.isnan(y_data)]
    ])
    max_val = all_vals.max() * 1.1 if len(all_vals) > 0 else 100

    # Left: All data with outliers highlighted
    ax1 = axes[0]
    stats_all = _plot_single_crossplot(
        ax1, x_data, y_data, x_label, y_label,
        site_color, max_val, outlier_mask=None, title='Before: All Data'
    )

    # Highlight outliers on left plot
    if outlier_mask is not None and outlier_mask.any():
        ax1.scatter(x_data[outlier_mask], y_data[outlier_mask],
                    facecolors='none', edgecolors='red', s=200, linewidths=3)

    # Right: Outliers removed
    ax2 = axes[1]
    stats_clean = _plot_single_crossplot(
        ax2, x_data, y_data, x_label, y_label,
        site_color, max_val, outlier_mask=outlier_mask, title='After: Outliers Removed'
    )

    fig.suptitle(f'{site_name}: Outlier Removal Impact', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, stats_all, stats_clean


def _plot_single_crossplot(ax, x_data, y_data, x_label, y_label,
                            color, max_val, outlier_mask=None, title=''):
    """Helper function for before/after comparison plots."""
    valid_mask = (~np.isnan(x_data)) & (~np.isnan(y_data))

    if outlier_mask is not None:
        plot_mask = valid_mask & ~outlier_mask
        removed_mask = valid_mask & outlier_mask
    else:
        plot_mask = valid_mask
        removed_mask = np.zeros(len(x_data), dtype=bool)

    x_plot = x_data[plot_mask]
    y_plot = y_data[plot_mask]

    # Plot retained points
    ax.scatter(x_plot, y_plot, color=color, alpha=0.6, s=80,
               edgecolors='black', linewidth=1, label=f'Data (n={len(x_plot)})')

    # Plot removed points
    if removed_mask.any():
        ax.scatter(x_data[removed_mask], y_data[removed_mask],
                   color='red', alpha=0.9, s=200, marker='X',
                   linewidths=3, label=f'Removed (n={removed_mask.sum()})')

    # Calculate regression on plotted data
    stats = calculate_regression_stats(x_plot, y_plot)

    if stats:
        # Regression line
        x_line = np.array([0, max_val])
        y_line = stats['slope'] * x_line + stats['intercept']
        ax.plot(x_line, y_line, 'g-', linewidth=2, alpha=0.8, label='Best fit')

        # Stats text
        sign = '+' if stats['intercept'] >= 0 else '-'
        eq = f"y = {stats['slope']:.3f}x {sign} {abs(stats['intercept']):.2f}"
        text = f"n = {stats['n']}\nR^2 = {stats['r_squared']:.3f}\n{eq}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Set axes
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal', adjustable='box')
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1.5, label='1:1 line')

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    return stats


# =============================================================================
# TILED THRESHOLD PLOTS
# =============================================================================

def create_tiled_threshold_plots(matched_df, site_name, site_color,
                                  thresholds=[1, 2.5, 4, 5],
                                  threshold_col='smooth_raw_abs_pct'):
    """
    Create a 2x2 tiled plot showing cross-plots at different smooth/raw thresholds.

    Parameters:
    -----------
    matched_df : DataFrame with aeth_bc, filter_ec, and threshold_col
    site_name : str
    site_color : str
    thresholds : list of % thresholds to test
    threshold_col : column to use for thresholding

    Returns:
    --------
    dict with stats for each threshold
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    # Calculate axis limits
    all_vals = np.concatenate([
        matched_df['aeth_bc'].dropna().values,
        matched_df['filter_ec'].dropna().values
    ])
    max_val = all_vals.max() * 1.1 if len(all_vals) > 0 else 100

    results = {}

    for idx, threshold in enumerate(thresholds):
        ax = axes[idx]

        # Separate by threshold
        below_threshold = matched_df[matched_df[threshold_col] <= threshold].copy()
        above_threshold = matched_df[matched_df[threshold_col] > threshold].copy()
        nan_threshold = matched_df[matched_df[threshold_col].isna()].copy()

        n_kept = len(below_threshold)
        n_removed = len(above_threshold)
        n_nan = len(nan_threshold)

        # Plot kept points
        if len(below_threshold) > 0:
            ax.scatter(below_threshold['aeth_bc'], below_threshold['filter_ec'],
                       color=site_color, alpha=0.6, s=80,
                       edgecolors='black', linewidth=1,
                       label=f'<={threshold}% diff (n={n_kept})')

        # Plot removed points
        if len(above_threshold) > 0:
            ax.scatter(above_threshold['aeth_bc'], above_threshold['filter_ec'],
                       color='gray', alpha=0.5, s=120, marker='X',
                       linewidths=2, label=f'>{threshold}% diff (n={n_removed})')

        # Plot NaN points
        if len(nan_threshold) > 0:
            ax.scatter(nan_threshold['aeth_bc'], nan_threshold['filter_ec'],
                       color='lightgray', alpha=0.4, s=40, marker='o',
                       label=f'No smooth data (n={n_nan})')

        # Calculate regression on kept points
        if len(below_threshold) >= 3:
            stats = calculate_regression_stats(
                below_threshold['aeth_bc'].values,
                below_threshold['filter_ec'].values
            )

            if stats:
                x_line = np.array([0, max_val])
                y_line = stats['slope'] * x_line + stats['intercept']
                ax.plot(x_line, y_line, 'g-', linewidth=2, alpha=0.8, label='Best fit')

                sign = '+' if stats['intercept'] >= 0 else '-'
                eq = f"y = {stats['slope']:.3f}x {sign} {abs(stats['intercept']):.2f}"
                stats_text = f"Threshold: <={threshold}%\nn = {stats['n']}\nR^2 = {stats['r_squared']:.3f}\n{eq}"

                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

                results[threshold] = {
                    'n_total': len(matched_df),
                    'n_kept': n_kept,
                    'n_removed': n_removed,
                    'pct_removed': (n_removed / len(matched_df)) * 100 if len(matched_df) > 0 else 0,
                    **stats
                }
        else:
            ax.text(0.5, 0.5, f'Insufficient data\nat <={threshold}% threshold\n(n={n_kept})',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            results[threshold] = None

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect('equal', adjustable='box')
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1.5, label='1:1 line')

        ax.set_xlabel('Aethalometer BC (ng/m^3)', fontsize=11)
        ax.set_ylabel('FTIR EC (ng/m^3)', fontsize=11)
        ax.set_title(f'Threshold: <={threshold}% Smooth-Raw Difference',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{site_name}: Effect of Smooth/Raw Difference Threshold',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig, results


# =============================================================================
# DISTRIBUTION PLOTS
# =============================================================================

def plot_smooth_raw_distribution(matched_df, site_name, site_color,
                                  thresholds=[1, 2.5, 4, 5],
                                  col='smooth_raw_abs_pct'):
    """
    Plot the distribution of smooth/raw % differences with threshold lines.

    Parameters:
    -----------
    matched_df : DataFrame
    site_name : str
    site_color : str
    thresholds : list of thresholds to mark
    col : column with % difference values

    Returns:
    --------
    fig
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    valid_diffs = matched_df[col].dropna()

    if len(valid_diffs) == 0:
        print(f"  {site_name}: No smooth/raw difference data available")
        return None

    # Left: Histogram
    ax1 = axes[0]
    bins = np.arange(0, valid_diffs.max() + 1, 0.5)
    ax1.hist(valid_diffs, bins=bins, color=site_color, alpha=0.7,
             edgecolor='black', linewidth=0.5)

    threshold_colors = ['green', 'blue', 'orange', 'red']
    for threshold, color in zip(thresholds, threshold_colors):
        ax1.axvline(x=threshold, color=color, linestyle='--', linewidth=2,
                    label=f'{threshold}%: {(valid_diffs <= threshold).sum()} kept')

    ax1.set_xlabel('Absolute % Difference (Smooth - Raw)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'{site_name}: Distribution', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Cumulative
    ax2 = axes[1]
    sorted_diffs = np.sort(valid_diffs)
    cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs) * 100

    ax2.plot(sorted_diffs, cumulative, color=site_color, linewidth=2)
    ax2.fill_between(sorted_diffs, cumulative, alpha=0.3, color=site_color)

    for threshold, color in zip(thresholds, threshold_colors):
        pct_below = (valid_diffs <= threshold).sum() / len(valid_diffs) * 100
        ax2.axvline(x=threshold, color=color, linestyle='--', linewidth=2)
        ax2.annotate(f'{threshold}%: {pct_below:.1f}% kept',
                     xy=(threshold, pct_below), xytext=(threshold + 0.5, pct_below - 5),
                     fontsize=9, color=color)

    ax2.set_xlabel('Absolute % Difference (Smooth - Raw)', fontsize=11)
    ax2.set_ylabel('Cumulative % of Data Points', fontsize=11)
    ax2.set_title(f'{site_name}: Cumulative Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


# =============================================================================
# TIME SERIES PLOTS
# =============================================================================

def plot_bc_timeseries(ax, site_name, df, config, wavelength='IR'):
    """Plot BC time series for a specific wavelength."""
    col_name = f'{wavelength} BCc'

    if col_name in df.columns:
        valid_data = df[df[col_name].notna()].copy()

        if len(valid_data) > 0:
            ax.plot(valid_data['day_9am'], valid_data[col_name],
                    color=config['color'], label=f"{site_name}",
                    alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{wavelength} BC (ng/m^3)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)


def plot_multiwavelength_bc(ax, site_name, df, config):
    """Plot BC for multiple wavelengths."""
    wavelengths = ['UV', 'Blue', 'Green', 'Red', 'IR']

    for wavelength in wavelengths:
        col_name = f'{wavelength} BCc'

        if col_name in df.columns:
            valid_data = df[df[col_name].notna()].copy()

            if len(valid_data) > 0:
                ax.plot(valid_data['day_9am'], valid_data[col_name],
                        label=f"{wavelength}", alpha=0.6, linewidth=1)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('BC (ng/m^3)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)


# =============================================================================
# COMPARISON TABLES
# =============================================================================

def print_comparison_table(results_dict, metric_name='R^2'):
    """
    Print a comparison table of statistics across sites/thresholds.

    Parameters:
    -----------
    results_dict : dict of dicts with stats
    metric_name : str for table header
    """
    print("\n" + "=" * 80)
    print(f"COMPARISON TABLE: {metric_name}")
    print("=" * 80)

    # Get all sites and thresholds
    sites = list(results_dict.keys())

    for site_name, site_results in results_dict.items():
        if site_results is None:
            print(f"\n{site_name}: No data")
            continue

        if isinstance(site_results, dict) and 'n' in site_results:
            # Single result
            print(f"\n{site_name}:")
            print(f"  n = {site_results['n']}")
            print(f"  R^2 = {site_results['r_squared']:.3f}")
            print(f"  Slope = {site_results['slope']:.3f}")
        else:
            # Multiple thresholds
            print(f"\n{site_name}:")
            print(f"{'Threshold':<15s} {'n':>8s} {'R^2':>10s} {'Slope':>10s}")
            print("-" * 45)
            for threshold, stats in site_results.items():
                if stats:
                    print(f"{threshold:<15} {stats['n']:>8d} "
                          f"{stats['r_squared']:>10.3f} {stats['slope']:>10.3f}")
                else:
                    print(f"{threshold:<15} {'--':>8s} {'--':>10s} {'--':>10s}")
