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

from .utils import (
    add_one_to_one_line,
    add_regression_line,
    add_stats_textbox,
    calculate_axis_limits,
    calculate_regression_stats,
    format_stats_text,
    get_clean_data,
    setup_equal_axes,
    style_axes,
)


__all__ = [
    'scatter_on_axes',
    'iron_gradient_on_axes',
    'bc_timeseries_on_axes',
    'crossplot_on_axes',
]


def crossplot_on_axes(ax, x, y, x_label, y_label, *, color=None,
                      one_to_one=True, stats_box=True, equal_axes=True,
                      outlier_mask=None, stats=None, fit_line=True,
                      errors_in_variables=None, deming_line=False) -> dict | None:
    """Draw a reusable cross-plot panel onto an existing axes.

    Unlike :func:`scatter_on_axes`, which preserves the retired
    ``plotting_legacy`` API, this function exposes the common panel controls
    used by the phase-3 notebooks and delegates plotting details to the shared
    helpers in :mod:`plotting.utils`.

    Parameters
    ----------
    ax : matplotlib axes
        Drawn onto in place.
    x, y : array-like
        Paired observations. Non-finite pairs are omitted.
    x_label, y_label : str
        Axis labels.
    color : matplotlib color or None
        Colour for retained observations. ``None`` uses the active colour
        cycle.
    one_to_one, stats_box, equal_axes, fit_line : bool
        Toggle the 1:1 line, statistics box, equal axes, and fitted line.
        The 1:1 line is only meaningful when ``equal_axes`` is true.
    outlier_mask : array-like of bool or None
        True marks an excluded observation. Excluded observations are drawn
        but omitted from the regression.
    stats : dict or None
        Optional precomputed regression metrics. Both the plotting-helper
        ``r_squared`` key and the ``R2`` key returned by
        ``pls_transfer.regression_metrics`` are accepted.
    errors_in_variables : bool or None
        Also compute a Deming (errors-in-variables) slope and show it in the
        statistics box. ``None`` (the default) means *auto*: enabled exactly
        when ``one_to_one and equal_axes``, because a 1:1 line asserts both axes
        measure the same quantity, so x carries error and plain OLS is biased
        shallow. Ignored when ``stats`` is supplied, since the caller's numbers
        are then authoritative. Set ``False`` to force OLS only.
    deming_line : bool
        Also draw the Deming fit as a dashed line. Off by default: the reported
        number is the useful part, and a third line crowds the panel.

    Returns
    -------
    dict or None
        The supplied or calculated regression statistics, or ``None`` when
        fewer than three usable, non-outlier pairs remain.
    """
    x_data = np.asarray(x, dtype=float)
    y_data = np.asarray(y, dtype=float)

    # get_clean_data is the canonical splitter. Convert infinities to NaN first
    # because that helper intentionally implements the historical NaN contract.
    finite_pairs = np.isfinite(x_data) & np.isfinite(y_data)
    x_usable = np.where(finite_pairs, x_data, np.nan)
    y_usable = np.where(finite_pairs, y_data, np.nan)
    x_clean, y_clean, x_outliers, y_outliers = get_clean_data(
        x_usable, y_usable, outlier_mask
    )

    if len(x_clean) < 3:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center')
        style_axes(ax, x_label, y_label, show_legend=False)
        return None

    ax.scatter(x_clean, y_clean, color=color, alpha=0.6, s=80,
               edgecolors='black', linewidth=1,
               label=f'Data (n={len(x_clean)})')

    if len(x_outliers) > 0:
        ax.scatter(x_outliers, y_outliers, color='red', alpha=0.9, s=200,
                   marker='X', linewidths=3,
                   label=f'Excluded (n={len(x_outliers)})')

    use_eiv = (one_to_one and equal_axes) if errors_in_variables is None \
        else bool(errors_in_variables)
    if stats is not None:
        result = stats
    else:
        result = calculate_regression_stats(
            x_clean, y_clean, errors_in_variables=use_eiv
        )

    # Precomputed phase-3 metrics use R2, while plotting helpers historically
    # consume r_squared. Keep the caller's dictionary untouched and returned
    # verbatim; this normalized copy exists only at the helper boundary.
    helper_stats = dict(result)
    if 'r_squared' not in helper_stats:
        if 'R2' in helper_stats:
            helper_stats['r_squared'] = helper_stats['R2']
        elif 'r2' in helper_stats:
            helper_stats['r_squared'] = helper_stats['r2']

    _, max_val = calculate_axis_limits(
        [x_clean, y_clean, x_outliers, y_outliers]
    )
    if max_val <= 0:
        max_val = 1.0

    if equal_axes:
        setup_equal_axes(ax, max_val)
    else:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    if fit_line:
        add_regression_line(ax, helper_stats, [0, max_val])

    if deming_line and np.isfinite(helper_stats.get('deming_slope', np.nan)):
        d_s = helper_stats['deming_slope']
        d_i = helper_stats['deming_intercept']
        ax.plot([0, max_val], [d_i, d_s * max_val + d_i],
                ls='--', lw=1.4, color='0.35',
                label=f"Deming (slope={d_s:.2f})")

    # The helper honors PlotConfig for figure-owning callers. Here the explicit
    # axes-level arguments are authoritative, so temporarily enable the helper
    # while preserving the caller's global configuration.
    from . import PlotConfig

    if one_to_one and equal_axes:
        configured = PlotConfig.get('show_1to1')
        try:
            if not configured:
                PlotConfig.set(show_1to1=True)
            add_one_to_one_line(ax, max_val)
        finally:
            if not configured:
                PlotConfig.set(show_1to1=False)

    if stats_box:
        configured = PlotConfig.get('show_stats')
        text_count = len(ax.texts)
        try:
            if not configured:
                PlotConfig.set(show_stats=True)
            add_stats_textbox(ax, helper_stats)
        finally:
            if not configured:
                PlotConfig.set(show_stats=False)

        # add_stats_textbox owns positioning and styling. Extend its helper-
        # formatted text with RMSE when supplied by the superset metrics.
        extras = []
        if 'RMSE' in helper_stats:
            extras.append(f"RMSE = {helper_stats['RMSE']:.2f}")
        if np.isfinite(helper_stats.get('deming_slope', np.nan)):
            # Report the errors-in-variables slope alongside OLS rather than
            # replacing it, so a reader can see how much of the slope was
            # regression dilution rather than signal.
            extras.append(f"Deming = {helper_stats['deming_slope']:.2f}")
        if len(ax.texts) > text_count and extras:
            text = format_stats_text(helper_stats)
            missing = [e for e in extras if e.split(' =')[0] not in text]
            if missing:
                text = format_stats_text(
                    helper_stats, extra_text="\n".join(missing)
                )
            ax.texts[-1].set_text(text)

    style_axes(ax, x_label, y_label)
    return result


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
