"""
Distribution plotting functions: histograms, boxplots, cumulative distributions.

Usage:
    from plotting import distributions, PlotConfig

    PlotConfig.set(sites='all', layout='grid')

    # BC distribution boxplot
    distributions.bc_boxplot(aethalometer_data)

    # Smooth/raw difference histogram
    distributions.smooth_raw_histogram(matched_data)

    # Multi-wavelength boxplot
    distributions.wavelength_boxplot(aethalometer_data)
"""

import numpy as np
import matplotlib.pyplot as plt

from . import PlotConfig, resolve_sites, resolve_layout
from .utils import create_grid_layout, create_individual_figure, style_axes


def bc_boxplot(data, wavelength='IR', sites=None, layout=None):
    """
    Create boxplot of BC distribution across sites.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame} with BC columns
    wavelength : str
        'IR', 'UV', etc.
    sites : str, list, or None
    layout : str or None
        'individual' - one boxplot per site (multi-wavelength)
        'grid' or 'combined' - all sites in one figure
    """
    sites_list = resolve_sites(sites)
    layout = resolve_layout(layout)

    col_name = f'{wavelength} BCc'

    if layout == 'individual':
        # Individual = show all wavelengths for each site
        _bc_boxplot_multiwavelength(data, sites_list)
    else:
        # Grid/combined = compare sites for one wavelength
        _bc_boxplot_sites(data, sites_list, col_name, wavelength)


def _bc_boxplot_sites(data, sites_list, col_name, wavelength):
    """Boxplot comparing BC across sites."""
    fig, ax = create_individual_figure()

    bc_data = []
    labels = []
    colors = []

    for site_name in sites_list:
        if site_name not in data:
            continue

        df = data[site_name]
        if col_name not in df.columns:
            continue

        valid_data = df[col_name].dropna()
        if len(valid_data) > 0:
            bc_data.append(valid_data.values)
            labels.append(site_name)
            colors.append(PlotConfig.get_site_color(site_name))

    if len(bc_data) == 0:
        print("No valid BC data")
        return

    bp = ax.boxplot(bc_data, labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    style_axes(ax, '', f'{wavelength} BC (ng/m³)',
               f'{wavelength} BC Distribution - All Sites', show_legend=False)

    plt.tight_layout()
    plt.show()


def _bc_boxplot_multiwavelength(data, sites_list):
    """Boxplot of all wavelengths for each site."""
    wavelengths = ['UV', 'Blue', 'Green', 'Red', 'IR']

    for site_name in sites_list:
        if site_name not in data:
            continue

        df = data[site_name]
        fig, ax = create_individual_figure()
        color = PlotConfig.get_site_color(site_name)

        bc_data = []
        labels = []

        for wavelength in wavelengths:
            col_name = f'{wavelength} BCc'
            if col_name not in df.columns:
                continue

            valid_data = df[col_name].dropna()
            if len(valid_data) > 0:
                bc_data.append(valid_data.values)
                labels.append(wavelength)

        if len(bc_data) == 0:
            continue

        bp = ax.boxplot(bc_data, labels=labels, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        style_axes(ax, 'Wavelength', 'BC (ng/m³)',
                   f'BC Distribution by Wavelength - {site_name}', show_legend=False)

        plt.tight_layout()
        plt.show()


def wavelength_boxplot(data, sites=None, layout=None):
    """
    Create boxplot comparing BC across wavelengths.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame}
    """
    # This is same as individual bc_boxplot
    bc_boxplot(data, sites=sites, layout='individual')


def smooth_raw_histogram(data, sites=None, layout=None,
                         col='smooth_raw_abs_pct', thresholds=None):
    """
    Plot histogram of smooth/raw % difference.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame} with smooth_raw_abs_pct column
    thresholds : list (optional)
        Threshold values to mark (default [1, 2.5, 4, 5])
    """
    sites_list = resolve_sites(sites)
    layout = resolve_layout(layout)

    if thresholds is None:
        thresholds = [1, 2.5, 4, 5]

    threshold_colors = ['green', 'blue', 'orange', 'red']

    if layout == 'individual':
        for site_name in sites_list:
            if site_name not in data:
                continue

            df = data[site_name]
            if col not in df.columns:
                continue

            valid_diffs = df[col].dropna()
            if len(valid_diffs) == 0:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            color = PlotConfig.get_site_color(site_name)

            # Histogram
            ax1 = axes[0]
            bins = np.arange(0, min(valid_diffs.max() + 1, 20), 0.5)
            ax1.hist(valid_diffs, bins=bins, color=color, alpha=0.7,
                     edgecolor='black', linewidth=0.5)

            for threshold, tc in zip(thresholds, threshold_colors):
                n_below = (valid_diffs <= threshold).sum()
                ax1.axvline(x=threshold, color=tc, linestyle='--', linewidth=2,
                            label=f'{threshold}%: {n_below} kept')

            style_axes(ax1, 'Absolute % Difference', 'Count',
                       f'{site_name}: Distribution')

            # Cumulative
            ax2 = axes[1]
            sorted_diffs = np.sort(valid_diffs)
            cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs) * 100

            ax2.plot(sorted_diffs, cumulative, color=color, linewidth=2)
            ax2.fill_between(sorted_diffs, cumulative, alpha=0.3, color=color)

            for threshold, tc in zip(thresholds, threshold_colors):
                pct_below = (valid_diffs <= threshold).sum() / len(valid_diffs) * 100
                ax2.axvline(x=threshold, color=tc, linestyle='--', linewidth=2)

            ax2.set_ylim(0, 105)
            style_axes(ax2, 'Absolute % Difference', 'Cumulative %',
                       f'{site_name}: Cumulative', show_legend=False)

            plt.tight_layout()
            plt.show()

    elif layout == 'grid':
        valid_sites = [s for s in sites_list if s in data and col in data[s].columns]
        if not valid_sites:
            return

        fig, axes = create_grid_layout(len(valid_sites))

        for idx, site_name in enumerate(valid_sites):
            ax = axes[idx]
            df = data[site_name]
            color = PlotConfig.get_site_color(site_name)

            valid_diffs = df[col].dropna()
            if len(valid_diffs) == 0:
                continue

            bins = np.arange(0, min(valid_diffs.max() + 1, 20), 0.5)
            ax.hist(valid_diffs, bins=bins, color=color, alpha=0.7,
                    edgecolor='black', linewidth=0.5)

            for threshold, tc in zip(thresholds, threshold_colors):
                n_below = (valid_diffs <= threshold).sum()
                pct_below = n_below / len(valid_diffs) * 100
                ax.axvline(x=threshold, color=tc, linestyle='--', linewidth=2,
                           label=f'{threshold}%: {pct_below:.0f}%')

            style_axes(ax, '% Diff', 'Count', site_name)

        fig.suptitle('Smooth/Raw Difference Distribution',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()


def uv_ir_ratio_histogram(data, sites=None, layout=None):
    """
    Plot histogram of UV/IR ratio.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame} with UV BCc and IR BCc columns
    """
    sites_list = resolve_sites(sites)
    layout = resolve_layout(layout)

    if layout == 'individual':
        for site_name in sites_list:
            if site_name not in data:
                continue

            df = data[site_name]
            if 'UV BCc' not in df.columns or 'IR BCc' not in df.columns:
                continue

            # Calculate ratio
            valid_mask = df['UV BCc'].notna() & df['IR BCc'].notna() & (df['IR BCc'] > 0)
            valid_df = df[valid_mask].copy()
            valid_df['UV_IR_ratio'] = valid_df['UV BCc'] / valid_df['IR BCc']

            if len(valid_df) == 0:
                continue

            fig, ax = create_individual_figure()
            color = PlotConfig.get_site_color(site_name)

            ax.hist(valid_df['UV_IR_ratio'], bins=30, color=color, alpha=0.7,
                    edgecolor='black', linewidth=0.5)
            ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2,
                       label='Ratio = 1')

            # Stats
            mean_ratio = valid_df['UV_IR_ratio'].mean()
            median_ratio = valid_df['UV_IR_ratio'].median()
            ax.axvline(x=mean_ratio, color='red', linestyle='-', linewidth=1.5,
                       label=f'Mean = {mean_ratio:.2f}')

            style_axes(ax, 'UV/IR Ratio', 'Count',
                       f'UV/IR Ratio Distribution - {site_name}')

            plt.tight_layout()
            plt.show()


def correlation_matrix(data, columns, sites=None, layout=None):
    """
    Plot correlation matrix heatmap.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame}
    columns : list
        Column names to include in correlation
    """
    import seaborn as sns

    sites_list = resolve_sites(sites)
    layout = resolve_layout(layout)

    if layout == 'individual':
        for site_name in sites_list:
            if site_name not in data:
                continue

            df = data[site_name]
            available_cols = [c for c in columns if c in df.columns]

            if len(available_cols) < 2:
                continue

            fig, ax = create_individual_figure()
            color = PlotConfig.get_site_color(site_name)

            corr_df = df[available_cols].dropna()
            if len(corr_df) < 3:
                continue

            corr_matrix = corr_df.corr()

            sns.heatmap(corr_matrix, ax=ax, annot=True, fmt='.3f',
                        cmap='RdYlBu_r', vmin=0.5, vmax=1.0,
                        square=True, linewidths=0.5,
                        cbar_kws={'shrink': 0.8, 'label': 'R'})

            ax.set_title(f'Correlation Matrix - {site_name} (n={len(corr_df)})',
                         fontsize=PlotConfig.get('title_size'), fontweight='bold')

            plt.tight_layout()
            plt.show()

    elif layout == 'grid':
        valid_sites = [s for s in sites_list if s in data]
        if not valid_sites:
            return

        fig, axes = create_grid_layout(len(valid_sites))

        for idx, site_name in enumerate(valid_sites):
            ax = axes[idx]
            df = data[site_name]
            available_cols = [c for c in columns if c in df.columns]

            if len(available_cols) < 2:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(site_name)
                continue

            corr_df = df[available_cols].dropna()
            if len(corr_df) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(site_name)
                continue

            corr_matrix = corr_df.corr()

            import seaborn as sns
            sns.heatmap(corr_matrix, ax=ax, annot=True, fmt='.2f',
                        cmap='RdYlBu_r', vmin=0.5, vmax=1.0,
                        square=True, linewidths=0.5, annot_kws={'size': 8},
                        cbar=False)

            ax.set_title(f'{site_name} (n={len(corr_df)})',
                         fontsize=PlotConfig.get('font_size'), fontweight='bold')

        fig.suptitle('Correlation Matrices', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
