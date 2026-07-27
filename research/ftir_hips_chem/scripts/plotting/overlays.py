"""Axes-level plotting primitives.

The rest of the ``plotting`` package owns its figures: you call
``crossplots.scatter(data, ...)`` and it creates, lays out, and shows a figure
per site. That is the right default for the standard analyses.

Some notebooks need the opposite: they build a shared multi-site figure
themselves and draw one series onto an axes they already hold. That was the only
capability ``plotting_legacy`` still provided, which is why it survived long
after AGENTS.md deprecated it.

These functions close that gap. They take an ``ax`` as the first argument, draw
onto it, and return the regression stats (or ``None``). **Styling matches the
retired ``plotting_legacy`` functions exactly** -- marker sizes, the ``R^2``
glyph, the 8-point legend -- so migrating a notebook off the legacy module does
not change its figures.

Usage::

    import matplotlib.pyplot as plt
    from plotting import overlays

    fig, ax = plt.subplots()
    stats = overlays.scatter_on_axes(ax, x, y, 'HIPS Fabs', 'FTIR EC')

One deliberate improvement over the legacy code: points are masked with
``np.isfinite`` rather than ``~np.isnan``, so +/-inf is dropped instead of
reaching ``np.polyfit`` (which raised ``LinAlgError``) or the axis-limit
computation (which raised ``ValueError: Axis limits cannot be NaN or Inf``).
On finite data the two masks are equivalent and output is unchanged.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from config import MAC_VALUE
except ImportError:  # Support importing as research.ftir_hips_chem.scripts.*
    from ..config import MAC_VALUE

from .utils import calculate_regression_stats


__all__ = ['scatter_on_axes', 'iron_gradient_on_axes', 'bc_timeseries_on_axes']


def scatter_on_axes(ax, x_data, y_data, x_label, y_label,
                    color='blue', outlier_mask=None, equal_axes=True,
                    show_stats=True, show_1to1=True, show_mac=False):
    """Draw a scatter cross-plot with regression onto an existing axes.

    Drop-in replacement for the retired ``plotting_legacy.plot_crossplot``.

    Parameters
    ----------
    ax : matplotlib axes
        Drawn onto in place.
    x_data, y_data : array-like
    x_label, y_label : str
    color : str
        Colour for the retained points.
    outlier_mask : array-like of bool or None
        True marks an excluded point, drawn as a red X.
    equal_axes : bool
        Lock both axes to a shared 0..max range with 1:1 aspect.
    show_stats : bool
        Overlay the n / R^2 / equation box.
    show_1to1 : bool
        Draw the 1:1 reference line (only when ``equal_axes``).
    show_mac : bool
        Append the MAC value to the stats box.

    Returns
    -------
    dict or None
        Regression stats, or None when fewer than 3 usable points.
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    # isfinite, not ~isnan: +/-inf must be dropped here too. Legacy masked with
    # ~isnan and passed inf into np.polyfit, raising LinAlgError; masking only
    # inside calculate_regression_stats is not enough, because the axis-limit
    # computation below would still see inf. Identical to ~isnan on finite data.
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)

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

    ax.scatter(x_clean, y_clean, color=color, alpha=0.6, s=80,
               edgecolors='black', linewidth=1, label=f'Data (n={len(x_clean)})')

    if len(x_outliers) > 0:
        ax.scatter(x_outliers, y_outliers, color='red', alpha=0.9, s=200,
                   marker='X', linewidths=3, label=f'Excluded (n={len(x_outliers)})')

    stats = calculate_regression_stats(x_clean, y_clean)

    if stats:
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

        y_line = stats['slope'] * x_line + stats['intercept']
        ax.plot(x_line, y_line, 'g-', linewidth=2, alpha=0.8, label='Best fit')

        if show_1to1 and equal_axes:
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5,
                    linewidth=1.5, label='1:1 line')

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


def bc_timeseries_on_axes(ax, site_name, df, config, wavelength='IR'):
    """Draw one site's BC time series onto an existing axes.

    Drop-in replacement for the retired ``plotting_legacy.plot_bc_timeseries``.
    Unlike ``timeseries.bc``, this labels the series with ``site_name`` so a
    caller can overlay several sites on one shared axes and get a legend.

    Parameters
    ----------
    ax : matplotlib axes
    site_name : str
        Used as the series label.
    df : pandas.DataFrame
        Must carry ``day_9am`` and ``'{wavelength} BCc'``.
    config : dict
        Site config; ``config['color']`` sets the line colour. Accepts the
        entries of ``config.SITES``.
    wavelength : str
        One of UV, Blue, Green, Red, IR.
    """
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


def iron_gradient_on_axes(ax, x_data, y_data, iron_data,
                          x_label, y_label, equal_axes=True,
                          outlier_mask=None, cmap='plasma'):
    """Scatter cross-plot coloured by iron concentration, drawn onto an axes.

    Drop-in replacement for the retired
    ``plotting_legacy.plot_crossplot_iron_gradient``.

    Note this is **not** equivalent to ``crossplots.with_iron_gradient``: that
    function computes its regression from x/y alone, while this one (like the
    legacy original) requires iron to be present as well, so rows with missing
    iron are excluded from the fit. On real data that changes n, slope and R^2.
    The stricter mask is kept here deliberately so migrated notebooks report the
    same numbers they always have.

    Returns
    -------
    tuple
        ``(stats dict or None, scatter artist or None)``
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    iron_data = np.asarray(iron_data)

    # Requires all three to be finite -- see the note above.
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(iron_data)

    if outlier_mask is not None:
        outlier_mask = np.asarray(outlier_mask)
        clean_mask = valid_mask & ~outlier_mask
        outlier_plot_mask = valid_mask & outlier_mask
    else:
        clean_mask = valid_mask
        outlier_plot_mask = np.zeros(len(x_data), dtype=bool)

    x_clean = x_data[clean_mask]
    y_clean = y_data[clean_mask]
    iron_clean = iron_data[clean_mask]

    if len(x_clean) < 3:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        return None, None

    scatter = ax.scatter(x_clean, y_clean, c=iron_clean, cmap=cmap,
                         alpha=0.7, s=100, edgecolors='black', linewidth=0.5,
                         label=f'Data (n={len(x_clean)})')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Iron (ng/m3)', fontsize=10)

    if outlier_plot_mask.any():
        ax.scatter(x_data[outlier_plot_mask], y_data[outlier_plot_mask],
                   color='red', alpha=0.9, s=200, marker='X', linewidths=3,
                   label=f'Excluded (n={outlier_plot_mask.sum()})')

    stats = calculate_regression_stats(x_clean, y_clean)

    if stats:
        if equal_axes:
            all_vals = np.concatenate([x_clean, y_clean])
            if outlier_plot_mask.any():
                all_vals = np.concatenate([all_vals,
                                           x_data[outlier_plot_mask],
                                           y_data[outlier_plot_mask]])
            max_val = all_vals.max() * 1.1
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_aspect('equal', adjustable='box')
            x_line = np.array([0, max_val])
        else:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            x_line = np.array([0, x_clean.max() * 1.1])
            max_val = x_line[1]

        y_line = stats['slope'] * x_line + stats['intercept']
        ax.plot(x_line, y_line, 'g-', linewidth=2, alpha=0.8, label='Best fit')

        if equal_axes:
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5,
                    linewidth=1.5, label='1:1 line')

        sign = '+' if stats['intercept'] >= 0 else '-'
        eq = f"y = {stats['slope']:.3f}x {sign} {abs(stats['intercept']):.2f}"
        text = f"n = {stats['n']}\nR^2 = {stats['r_squared']:.3f}\n{eq}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    return stats, scatter
