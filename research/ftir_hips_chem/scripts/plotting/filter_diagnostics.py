"""Publication-style descriptive figures using canonical site colors and overlays."""

from hashlib import sha256

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from config import SITES
from .overlays import crossplot_on_axes
from .utils import add_regression_line, style_axes, calculate_regression_stats

EC_LABEL = "FTIR-predicted EC (µg m⁻³)"
RATIO_LABEL = "HIPS / FTIR-predicted EC\n[(Mm⁻¹)/(µg m⁻³)]"
DISPLAY = {s: s.replace("_", " ") for s in SITES}


def _grid(title, subtitle):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=19, y=0.99)
    fig.text(0.5, 0.942, subtitle, ha="center", fontsize=11, color="0.3")
    fig.subplots_adjust(top=0.885, bottom=0.10, hspace=0.36, wspace=0.25)
    return fig, axes.ravel()


def _scatter(ax, frame, x, y, xlabel, ylabel, site, fit=False):
    result = crossplot_on_axes(
        ax,
        frame[x],
        frame[y],
        xlabel,
        ylabel,
        color=SITES[site]["color"],
        one_to_one=False,
        equal_axes=False,
        stats_box=False,
        fit_line=False,
    )
    for collection in ax.collections:
        collection.set_sizes([25])
        collection.set_linewidths(0.35)
        collection.set_alpha(0.72)
    xvalues = frame[x].to_numpy()
    yvalues = frame[y].to_numpy()
    xlo, xhi = np.nanmin(xvalues), np.nanmax(xvalues)
    ylo, yhi = np.nanmin(yvalues), np.nanmax(yvalues)
    xpad = max((xhi - xlo) * 0.07, 0.05)
    ypad = max((yhi - ylo) * 0.07, 0.05)
    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(min(0, ylo - ypad), yhi + ypad)
    if fit:
        add_regression_line(ax, result, [xlo, xhi])
        for line in ax.lines:
            line.set_label("Descriptive OLS")
        ax.text(
            0.03,
            0.97,
            f"n = {len(frame)}   R² = {result['r_squared']:.3f}\nOLS slope = {result['slope']:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.92),
        )
    ax.set_title(DISPLAY[site], fontsize=14)
    ax.legend().remove()
    return result


def relationship(points):
    fig, axes = _grid(
        "HIPS and FTIR-predicted EC: relationships differ by site",
        "545 eligible diagnostic pairs · unweighted descriptive OLS · site-specific axis ranges",
    )
    for site, ax in zip(SITES, axes):
        frame = points.loc[points.site.eq(site) & points.eligible_filter_diagnostic]
        _scatter(
            ax, frame, "ftir_ec_ugm3", "hips_fabs_Mm1", EC_LABEL, "HIPS (Mm⁻¹)", site, fit=True
        )
        low = frame.loc[~frame.eligible_ec_ratio_analysis]
        ax.scatter(
            low.ftir_ec_ugm3,
            low.hips_fabs_Mm1,
            s=70,
            facecolors="none",
            edgecolors="0.15",
            linewidths=1.2,
            label=f"Ratio-ineligible (n={len(low)})",
        )
        if len(low):
            ax.legend(loc="lower right", fontsize=9)
    fig.text(
        0.5,
        0.025,
        "Open rings retain ratio-ineligible predictions in the diagnostic population. No independent EC validation is implied.",
        ha="center",
        fontsize=10,
    )
    return fig


def ratio_distribution(points):
    fig, ax = plt.subplots(figsize=(12, 6.6))
    fig.suptitle("HIPS / FTIR-predicted-EC ratios vary across sites", fontsize=19, y=0.98)
    fig.text(
        0.5,
        0.915,
        "480 ratio-eligible pairs · boxes show median and IQR · every eligible point is retained",
        ha="center",
        fontsize=11,
        color="0.3",
    )
    labels = []
    for i, site in enumerate(SITES, 1):
        frame = points.loc[points.site.eq(site) & points.eligible_ec_ratio_analysis]
        box = ax.boxplot(
            [frame.ratio],
            positions=[i],
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
        )
        box["boxes"][0].set(facecolor=SITES[site]["color"], alpha=0.2)
        jitter = (
            np.array(
                [
                    int(sha256(v.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF - 0.5
                    for v in frame.point_id
                ]
            )
            * 0.3
        )
        ax.scatter(
            i + jitter, frame.ratio, s=18, color=SITES[site]["color"], alpha=0.6, edgecolors="none"
        )
        ax.text(
            i,
            frame.ratio.median() + 1.1,
            f"{frame.ratio.median():.2f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
        labels.append(f"{DISPLAY[site]}\nn = {len(frame)}")
    ax.set_xticks(range(1, 5), labels)
    ax.set_xlim(0.4, 4.6)
    ax.set_ylim(bottom=0)
    style_axes(ax, "", RATIO_LABEL, show_legend=False)
    fig.subplots_adjust(top=0.85, bottom=0.14, left=0.12)
    return fig


def ratio_denominator(points):
    fig, axes = _grid(
        "Check large ratios against their EC denominators",
        "480 eligible pairs · open rings: EC from 1× to <2× its reported MDL · no fit to a ratio sharing its denominator",
    )
    for site, ax in zip(SITES, axes):
        frame = points.loc[points.site.eq(site) & points.eligible_ec_ratio_analysis]
        _scatter(ax, frame, "ftir_ec_ugm3", "ratio", EC_LABEL, RATIO_LABEL, site)
        near = frame.loc[frame.near_mdl_1_to_2]
        ax.scatter(
            near.ftir_ec_ugm3,
            near.ratio,
            s=70,
            facecolors="none",
            edgecolors="0.15",
            linewidths=1,
            label=f"1–<2× MDL (n={len(near)})",
        )
        ax.legend(loc="upper right", fontsize=9)
    return fig


def ratio_dates(points):
    fig, axes = _grid(
        "Ratio patterns across reported sample dates",
        "Individual filters · dates are reported sample dates, not verified active intervals · site-specific date ranges",
    )
    for site, ax in zip(SITES, axes):
        frame = points.loc[points.site.eq(site) & points.eligible_ec_ratio_analysis].copy()
        frame["date_number"] = mdates.date2num(frame.date)
        _scatter(ax, frame, "date_number", "ratio", "Reported sample date", RATIO_LABEL, site)
        locator = mdates.MonthLocator(bymonth=[1, 7])
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.axhline(frame.ratio.median(), color="0.4", ls="--", lw=1)
    return fig


def denominator_sensitivity(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    fig.suptitle(
        "Stricter EC denominator rules change the retained populations", fontsize=18, y=0.98
    )
    fig.text(
        0.5,
        0.9,
        "Descriptive sensitivity only · baseline remains EC > 0 and EC ≥ its reported MDL",
        ha="center",
        fontsize=11,
        color="0.3",
    )
    for site in SITES:
        frame = summary.loc[summary.site.eq(site)]
        for ax, y in zip(axes, ["n", "ratio_median"]):
            ax.plot(
                frame.minimum_ec_mdl_multiple,
                frame[y],
                marker="o",
                color=SITES[site]["color"],
                label=DISPLAY[site],
            )
    for ax in axes:
        ax.set_xticks([1, 1.5, 2, 3, 5])
        ax.set_xlabel("Minimum EC / reported MDL")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Retained filters")
    axes[1].set_ylabel("Median HIPS / FTIR-predicted-EC ratio")
    axes[0].legend(fontsize=10)
    fig.subplots_adjust(top=0.80, bottom=0.15, wspace=0.25)
    return fig


def registry_context(points):
    site = "Delhi"
    frame = points.loc[points.site.eq(site) & points.has_hips_ftir_pair]
    fig, ax = plt.subplots(figsize=(10, 6))
    crossplot_on_axes(
        ax,
        frame.ftir_ec_ugm3,
        frame.hips_fabs_Mm1,
        EC_LABEL,
        "HIPS (Mm⁻¹)",
        color=SITES[site]["color"],
        one_to_one=False,
        equal_axes=False,
        stats_box=False,
        fit_line=False,
        outlier_mask=frame.is_excluded,
    )
    clean = frame.loc[frame.eligible_filter_diagnostic]
    fit = calculate_regression_stats(clean.ftir_ec_ugm3, clean.hips_fabs_Mm1)
    add_regression_line(ax, fit, [clean.ftir_ec_ugm3.min(), clean.ftir_ec_ugm3.max()])
    flagged = frame.loc[frame.is_excluded]
    for row in flagged.itertuples():
        ax.annotate(
            row.base_filter_id,
            (row.ftir_ec_ugm3, row.hips_fabs_Mm1),
            xytext=(-12, -30),
            textcoords="offset points",
            ha="right",
        )
    ax.set_xlim(frame.ftir_ec_ugm3.min() - 2, frame.ftir_ec_ugm3.max() * 1.1)
    ax.set_ylim(min(-2, frame.hips_fabs_Mm1.min() - 2), frame.hips_fabs_Mm1.max() * 1.1)
    ax.set_title("Delhi: the registry exclusion remains visible", fontsize=18)
    fig.text(
        0.5,
        0.02,
        "The red X is retained for audit and excluded from the 545-pair diagnostic cohort and its descriptive fits.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


FIGURES = [
    ("01_site_relationships", relationship),
    ("02_ratio_distributions", ratio_distribution),
    ("03_ratio_denominators", ratio_denominator),
    ("04_ratio_dates", ratio_dates),
    ("05_denominator_sensitivity", denominator_sensitivity),
    ("06_registry_context", registry_context),
]
