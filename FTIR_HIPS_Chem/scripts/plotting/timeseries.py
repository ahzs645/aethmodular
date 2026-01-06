"""
Time series plotting functions.

Usage:
    from plotting import timeseries, PlotConfig

    PlotConfig.set(sites='all', layout='individual')

    # Single wavelength BC
    timeseries.bc(aethalometer_data, wavelength='IR')

    # Multi-wavelength BC
    timeseries.bc_multiwavelength(aethalometer_data)

    # Data completeness
    timeseries.data_completeness(aethalometer_data)

    # Filter vs aethalometer comparison
    timeseries.filter_vs_aeth(aethalometer_data, filter_data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import PlotConfig, resolve_sites, resolve_layout
from .utils import create_grid_layout, create_individual_figure, style_axes


def bc(data, wavelength='IR', sites=None, layout=None, title=None):
    """
    Plot BC time series for a specific wavelength.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame} with 'day_9am' and '{wavelength} BCc' columns
    wavelength : str
        'IR', 'UV', 'Blue', 'Green', 'Red'
    sites : str, list, or None
    layout : str or None
    title : str (optional)

    Returns:
    --------
    None
    """
    sites_list = resolve_sites(sites)
    layout = resolve_layout(layout)

    col_name = f'{wavelength} BCc'

    if layout == 'individual':
        _bc_individual(data, sites_list, col_name, wavelength, title)
    elif layout == 'grid':
        _bc_grid(data, sites_list, col_name, wavelength, title)
    elif layout == 'combined':
        _bc_combined(data, sites_list, col_name, wavelength, title)


def _bc_individual(data, sites_list, col_name, wavelength, title):
    """Plot individual BC time series."""
    for site_name in sites_list:
        if site_name not in data:
            continue

        df = data[site_name]
        if col_name not in df.columns:
            print(f"{site_name}: {col_name} not found")
            continue

        fig, ax = create_individual_figure()
        color = PlotConfig.get_site_color(site_name)

        valid_data = df[df[col_name].notna()].copy()
        if len(valid_data) > 0:
            ax.plot(valid_data['day_9am'], valid_data[col_name],
                    color=color, alpha=0.7,
                    linewidth=PlotConfig.get('line_width'))

        site_title = title or f'{wavelength} BC Time Series - {site_name}'
        style_axes(ax, 'Date', f'{wavelength} BC (ng/m³)', site_title, show_legend=False)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


def _bc_grid(data, sites_list, col_name, wavelength, title):
    """Plot BC time series in grid."""
    valid_sites = [s for s in sites_list if s in data and col_name in data[s].columns]
    if not valid_sites:
        print("No valid sites with data")
        return

    fig, axes = create_grid_layout(len(valid_sites))

    for idx, site_name in enumerate(valid_sites):
        ax = axes[idx]
        df = data[site_name]
        color = PlotConfig.get_site_color(site_name)

        valid_data = df[df[col_name].notna()].copy()
        if len(valid_data) > 0:
            ax.plot(valid_data['day_9am'], valid_data[col_name],
                    color=color, alpha=0.7,
                    linewidth=PlotConfig.get('line_width'))

        style_axes(ax, 'Date', f'{wavelength} BC (ng/m³)', site_name, show_legend=False)
        ax.tick_params(axis='x', rotation=45)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.show()


def _bc_combined(data, sites_list, col_name, wavelength, title):
    """Plot all sites on one axes."""
    fig, ax = create_individual_figure()

    for site_name in sites_list:
        if site_name not in data:
            continue

        df = data[site_name]
        if col_name not in df.columns:
            continue

        color = PlotConfig.get_site_color(site_name)
        valid_data = df[df[col_name].notna()].copy()

        if len(valid_data) > 0:
            ax.plot(valid_data['day_9am'], valid_data[col_name],
                    color=color, alpha=0.7,
                    linewidth=PlotConfig.get('line_width'),
                    label=site_name)

    plot_title = title or f'{wavelength} BC Time Series - All Sites'
    style_axes(ax, 'Date', f'{wavelength} BC (ng/m³)', plot_title)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def bc_multiwavelength(data, sites=None, layout=None):
    """
    Plot BC for multiple wavelengths.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame}
    sites : str, list, or None
    layout : str or None
    """
    sites_list = resolve_sites(sites)
    layout = resolve_layout(layout)

    wavelengths = ['UV', 'Blue', 'Green', 'Red', 'IR']
    wavelength_colors = {
        'UV': '#9B59B6',
        'Blue': '#3498DB',
        'Green': '#2ECC71',
        'Red': '#E74C3C',
        'IR': '#34495E'
    }

    if layout == 'combined':
        # For combined, show one site at a time with all wavelengths
        layout = 'individual'

    if layout == 'individual':
        for site_name in sites_list:
            if site_name not in data:
                continue

            df = data[site_name]
            fig, ax = create_individual_figure()

            for wavelength in wavelengths:
                col_name = f'{wavelength} BCc'
                if col_name not in df.columns:
                    continue

                valid_data = df[df[col_name].notna()].copy()
                if len(valid_data) > 0:
                    ax.plot(valid_data['day_9am'], valid_data[col_name],
                            color=wavelength_colors[wavelength],
                            alpha=0.6, linewidth=1,
                            label=wavelength)

            style_axes(ax, 'Date', 'BC (ng/m³)',
                       f'Multi-Wavelength BC - {site_name}')
            ax.tick_params(axis='x', rotation=45)

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

            for wavelength in wavelengths:
                col_name = f'{wavelength} BCc'
                if col_name not in df.columns:
                    continue

                valid_data = df[df[col_name].notna()].copy()
                if len(valid_data) > 0:
                    ax.plot(valid_data['day_9am'], valid_data[col_name],
                            color=wavelength_colors[wavelength],
                            alpha=0.6, linewidth=1,
                            label=wavelength)

            style_axes(ax, 'Date', 'BC (ng/m³)', site_name)
            ax.tick_params(axis='x', rotation=45)

        fig.suptitle('Multi-Wavelength BC Comparison',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()


def data_completeness(data, sites=None, layout=None):
    """
    Plot data completeness over time.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame} with 'day_9am' and optionally 'data_completeness_pct'
    """
    sites_list = resolve_sites(sites)
    layout = resolve_layout(layout)

    if layout == 'individual':
        for site_name in sites_list:
            if site_name not in data:
                continue

            df = data[site_name]
            fig, ax = create_individual_figure()
            color = PlotConfig.get_site_color(site_name)

            if 'data_completeness_pct' in df.columns:
                ax.plot(df['day_9am'], df['data_completeness_pct'],
                        color=color, alpha=0.7,
                        linewidth=PlotConfig.get('line_width'))
                ax.axhline(y=80, color='green', linestyle='--', alpha=0.5,
                           label='High quality (80%)')
                ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5,
                           label='Medium quality (50%)')
            elif 'IR BCc' in df.columns:
                # Show BC availability as binary
                completeness = (df['IR BCc'].notna()).astype(int) * 100
                ax.scatter(df['day_9am'], completeness, color=color, alpha=0.5, s=20)

            ax.set_ylim(-5, 105)
            style_axes(ax, 'Date', 'Data Completeness (%)',
                       f'Data Completeness - {site_name}')
            ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()

    elif layout == 'grid':
        valid_sites = [s for s in sites_list if s in data]
        fig, axes = create_grid_layout(len(valid_sites))

        for idx, site_name in enumerate(valid_sites):
            ax = axes[idx]
            df = data[site_name]
            color = PlotConfig.get_site_color(site_name)

            if 'data_completeness_pct' in df.columns:
                ax.plot(df['day_9am'], df['data_completeness_pct'],
                        color=color, alpha=0.7,
                        linewidth=PlotConfig.get('line_width'))
                ax.axhline(y=80, color='green', linestyle='--', alpha=0.5)
                ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5)

            ax.set_ylim(-5, 105)
            style_axes(ax, 'Date', 'Completeness (%)', site_name, show_legend=False)
            ax.tick_params(axis='x', rotation=45)

        fig.suptitle('Data Completeness Over Time',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()


def filter_vs_aeth(aeth_data, filter_data, sites=None, layout=None,
                   aeth_col='IR BCc', filter_param='ChemSpec_EC_PM2.5'):
    """
    Plot filter EC and aethalometer BC on the same time series.

    Parameters:
    -----------
    aeth_data : dict
        {site_name: DataFrame} with aethalometer data
    filter_data : DataFrame
        Unified filter dataset
    sites : str, list, or None
    layout : str or None
    aeth_col : str
        Aethalometer column name
    filter_param : str
        Filter parameter name
    """
    import sys
    sys.path.insert(0, str(PlotConfig.__module__).rsplit('.', 2)[0])
    from config import SITES

    sites_list = resolve_sites(sites)
    layout = resolve_layout(layout)

    if layout == 'individual':
        for site_name in sites_list:
            if site_name not in aeth_data:
                continue

            df_aeth = aeth_data[site_name]
            site_code = SITES[site_name]['code']
            color = PlotConfig.get_site_color(site_name)

            # Get filter data for this site
            site_filters = filter_data[
                (filter_data['Site'] == site_code) &
                (filter_data['Parameter'] == filter_param)
            ].copy()

            fig, ax = create_individual_figure()

            # Plot aethalometer
            valid_aeth = df_aeth[df_aeth[aeth_col].notna()].copy()
            if len(valid_aeth) > 0:
                ax.plot(valid_aeth['day_9am'], valid_aeth[aeth_col],
                        color=color, alpha=0.7, linewidth=2,
                        label=f'Aethalometer {aeth_col}')

            # Plot filter
            if len(site_filters) > 0:
                site_filters = site_filters.sort_values('SampleDate')
                # Convert to ng/m³ for comparison
                filter_conc = site_filters['Concentration'] * 1000
                ax.plot(site_filters['SampleDate'], filter_conc,
                        color=color, linestyle=':', linewidth=2.5,
                        alpha=0.8, label=f'Filter {filter_param}')

            style_axes(ax, 'Date', 'Concentration (ng/m³)',
                       f'Filter vs Aethalometer - {site_name}')
            ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()


def flow_ratio(data, sites=None, layout=None):
    """
    Plot flow ratio (Flow1/Flow2) over time.

    Parameters:
    -----------
    data : dict
        {site_name: DataFrame} with 'day_9am' and 'ratio_flow' columns
    """
    sites_list = resolve_sites(sites)
    layout = resolve_layout(layout)

    if layout == 'individual':
        for site_name in sites_list:
            if site_name not in data:
                continue

            df = data[site_name]
            if 'ratio_flow' not in df.columns:
                print(f"{site_name}: No ratio_flow column")
                continue

            fig, ax = create_individual_figure()
            color = PlotConfig.get_site_color(site_name)

            valid_data = df[df['ratio_flow'].notna()].copy()
            if len(valid_data) > 0:
                ax.scatter(valid_data['day_9am'], valid_data['ratio_flow'],
                           color=color, alpha=0.5, s=20)

                # Reference lines
                ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7,
                           label='Ideal (1.0)')
                ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7,
                           label='Degraded (2.0)')

            style_axes(ax, 'Date', 'Flow Ratio (Flow1/Flow2)',
                       f'Flow Ratio - {site_name}')
            ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()

    elif layout == 'grid':
        valid_sites = [s for s in sites_list if s in data and 'ratio_flow' in data[s].columns]
        if not valid_sites:
            return

        fig, axes = create_grid_layout(len(valid_sites))

        for idx, site_name in enumerate(valid_sites):
            ax = axes[idx]
            df = data[site_name]
            color = PlotConfig.get_site_color(site_name)

            valid_data = df[df['ratio_flow'].notna()].copy()
            if len(valid_data) > 0:
                ax.scatter(valid_data['day_9am'], valid_data['ratio_flow'],
                           color=color, alpha=0.5, s=15)
                ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
                ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7)

            style_axes(ax, 'Date', 'Flow Ratio', site_name, show_legend=False)
            ax.tick_params(axis='x', rotation=45)

        fig.suptitle('Flow Ratio Over Time', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
