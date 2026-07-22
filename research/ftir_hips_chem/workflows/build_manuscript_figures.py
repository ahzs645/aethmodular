#!/usr/bin/env python3
"""Build reproducible, data-backed figures for the AETH Modular manuscript.

Chart contracts
---------------
Figures 1 and 2 describe the example repository assets without treating them as
final study estimates. Figure 1 shows observed dates, while Figure 2 reports the
record counts available for possible comparisons.

Figure 4 asks whether HIPS-derived BC and FTIR EC agree on the same physical
filters. It uses base FilterId matching, the configured MAC of 10 m2/g, and the
project's documented analytical exclusions.

Figure 3 asks how sensitive a preliminary aethalometer/FTIR comparison is to a
date-only join. It varies the tolerance from zero to three days and reports R2,
OLS slope, and retained sample size. This is a sensitivity analysis, not a
substitute for integration over verified physical filter windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESEARCH_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = RESEARCH_DIR / "scripts"
PLOTS_DIR = RESEARCH_DIR / "output" / "plots" / "manuscript"
TABLES_DIR = RESEARCH_DIR / "output" / "tables" / "manuscript"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import MAC_VALUE, MIN_EC_THRESHOLD, SITES  # noqa: E402
from data_matching import (  # noqa: E402
    load_aethalometer_data,
    load_filter_data,
    match_aeth_filter_data,
    match_by_filter_id,
)
from outliers import (  # noqa: E402
    apply_exclusion_flags,
    apply_threshold_flags,
    get_clean_data,
)
from plotting.utils import calculate_regression_stats  # noqa: E402


SITE_LABELS = {
    "Beijing": "Beijing",
    "Delhi": "Delhi",
    "JPL": "JPL/Pasadena",
    "Addis_Ababa": "Addis Ababa",
}

METHOD_COLORS = {
    "Aethalometer IR BCc": "#4C78A8",
    "FTIR EC filters": "#222222",
    "Same-filter HIPS–FTIR": "#9C6ADE",
}


def configure_style() -> None:
    """Set an explicit, journal-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save high-resolution raster and vector versions of a figure."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add a consistent panel letter just outside the plotting area."""
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def same_filter_data(filter_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prepare same-filter HIPS/FTIR pairs and documented exclusion flags."""
    prepared: dict[str, pd.DataFrame] = {}

    for site_name, site_config in SITES.items():
        matched = match_by_filter_id(
            filter_data,
            site_code=site_config["code"],
            params=["EC_ftir", "HIPS_Fabs"],
        )
        if matched is None:
            continue

        matched = matched.copy()
        matched["site"] = SITE_LABELS[site_name]
        matched["ftir_ec_ug_m3"] = pd.to_numeric(matched["ftir_ec"], errors="coerce")
        matched["hips_bc_mac10_ug_m3"] = (
            pd.to_numeric(matched["hips_fabs"], errors="coerce") / MAC_VALUE
        )
        matched["is_excluded"] = False
        matched["exclusion_reason"] = ""

        invalid = (
            matched["ftir_ec_ug_m3"].isna()
            | matched["hips_bc_mac10_ug_m3"].isna()
            | (matched["ftir_ec_ug_m3"] < MIN_EC_THRESHOLD)
            | (matched["hips_bc_mac10_ug_m3"] <= 0)
        )
        matched.loc[invalid, "is_excluded"] = True
        matched.loc[invalid, "exclusion_reason"] = "Below EC threshold or non-positive/missing pair"

        # These are the documented project exclusions used in the existing
        # cross-plots and COMPLETE_RESEARCH_SUMMARY.md.
        if site_name == "Delhi":
            extreme_ec = matched["ftir_ec_ug_m3"] > 20
            matched.loc[extreme_ec, "is_excluded"] = True
            matched.loc[extreme_ec, "exclusion_reason"] = "Documented extreme FTIR EC outlier (>20 µg/m³)"
        elif site_name == "JPL":
            jpl_threshold = matched["ftir_ec_ug_m3"] > 1
            matched.loc[jpl_threshold, "is_excluded"] = True
            matched.loc[jpl_threshold, "exclusion_reason"] = "Documented JPL FTIR EC threshold (>1 µg/m³)"

        prepared[site_name] = matched

    return prepared


def build_coverage_figure(
    aethalometer_data: dict[str, pd.DataFrame],
    filter_data: pd.DataFrame,
    filter_pairs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build separate figures for observed dates and supporting record counts."""
    timeline_fig, ax_timeline = plt.subplots(figsize=(8.8, 4.8))
    counts_fig, ax_counts = plt.subplots(figsize=(7.2, 4.8))
    site_names = list(SITES)
    y_positions = np.arange(len(site_names))[::-1]
    coverage_rows: list[dict[str, object]] = []

    for y_position, site_name in zip(y_positions, site_names):
        site_config = SITES[site_name]
        aeth = aethalometer_data[site_name].copy()
        aeth_dates = pd.to_datetime(aeth["day_9am"])
        aeth_valid = pd.to_numeric(aeth["IR BCc"], errors="coerce").notna()
        valid_aeth_dates = aeth_dates[aeth_valid]

        site_filters = filter_data[
            (filter_data["Site"] == site_config["code"])
            & (filter_data["Parameter"] == "EC_ftir")
        ].copy()
        site_filters["Concentration"] = pd.to_numeric(
            site_filters["Concentration"], errors="coerce"
        )
        site_filters = site_filters[site_filters["Concentration"] >= MIN_EC_THRESHOLD]
        filter_dates = pd.to_datetime(site_filters["SampleDate"])

        pairs = filter_pairs[site_name]
        pair_count = int((~pairs["is_excluded"]).sum())
        color = site_config["color"]

        ax_timeline.scatter(
            valid_aeth_dates,
            np.full(len(valid_aeth_dates), y_position + 0.10),
            s=11,
            marker="|",
            linewidths=0.9,
            color=color,
            alpha=0.75,
            rasterized=True,
        )
        ax_timeline.scatter(
            filter_dates,
            np.full(len(filter_dates), y_position - 0.12),
            s=18,
            marker="D",
            facecolors="white",
            edgecolors="#222222",
            linewidths=0.6,
            alpha=0.8,
            rasterized=True,
        )

        coverage_rows.append(
            {
                "site": SITE_LABELS[site_name],
                "valid_aeth_days": int(aeth_valid.sum()),
                "aeth_start": valid_aeth_dates.min().date(),
                "aeth_end": valid_aeth_dates.max().date(),
                "eligible_ftir_filters": int(len(site_filters)),
                "filter_start": filter_dates.min().date(),
                "filter_end": filter_dates.max().date(),
                "retained_same_filter_hips_ftir_pairs": pair_count,
            }
        )

    ax_timeline.set_yticks(y_positions, [SITE_LABELS[name] for name in site_names])
    ax_timeline.set_ylim(-0.65, len(site_names) - 0.35)
    ax_timeline.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_timeline.tick_params(axis="x", rotation=35)
    ax_timeline.set_xlabel("Local date")
    ax_timeline.grid(axis="y", visible=False)
    ax_timeline.scatter([], [], marker="|", color="#555555", label="Valid daily IR BCc")
    ax_timeline.scatter(
        [],
        [],
        marker="D",
        facecolors="white",
        edgecolors="#222222",
        label="Eligible FTIR EC filter",
    )
    ax_timeline.legend(loc="lower left", ncol=2, fontsize=9)

    coverage = pd.DataFrame(coverage_rows)
    count_columns = [
        ("valid_aeth_days", "Aethalometer IR BCc"),
        ("eligible_ftir_filters", "FTIR EC filters"),
        ("retained_same_filter_hips_ftir_pairs", "Same-filter HIPS–FTIR"),
    ]
    bar_height = 0.22
    offsets = [bar_height, 0, -bar_height]
    for (column, label), offset in zip(count_columns, offsets):
        values = coverage.set_index("site").loc[
            [SITE_LABELS[name] for name in site_names], column
        ]
        bars = ax_counts.barh(
            y_positions + offset,
            values,
            height=bar_height * 0.82,
            color=METHOD_COLORS[label],
            label=label,
            alpha=0.9,
        )
        ax_counts.bar_label(bars, padding=3, fontsize=8)

    ax_counts.set_yticks(y_positions, [SITE_LABELS[name] for name in site_names])
    ax_counts.set_xlabel("Number of observations")
    ax_counts.set_xlim(0, coverage["valid_aeth_days"].max() * 1.18)
    ax_counts.grid(axis="y", visible=False)
    # The upper-right corner is clear because Beijing has the shortest blue
    # bar; the lower-right corner would obscure Addis Ababa's longest bar.
    ax_counts.legend(loc="upper right", fontsize=8)
    timeline_fig.tight_layout()
    save_figure(timeline_fig, "figure_1_measurement_dates")
    counts_fig.tight_layout()
    save_figure(counts_fig, "figure_2_supporting_record_counts")
    return coverage


def build_same_filter_figure(
    filter_pairs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Figure 4: example same-filter HIPS BC versus FTIR EC."""
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.6))
    metrics_rows: list[dict[str, object]] = []
    points_rows: list[pd.DataFrame] = []

    for panel_label, ax, site_name in zip("ABCD", axes.flat, SITES):
        all_pairs = filter_pairs[site_name].copy()
        clean = all_pairs[~all_pairs["is_excluded"]].copy()
        stats = calculate_regression_stats(
            clean["ftir_ec_ug_m3"], clean["hips_bc_mac10_ug_m3"]
        )
        if stats is None:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            continue

        color = SITES[site_name]["color"]
        x = clean["ftir_ec_ug_m3"].to_numpy(dtype=float)
        y = clean["hips_bc_mac10_ug_m3"].to_numpy(dtype=float)
        max_value = max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.08

        ax.scatter(
            x,
            y,
            s=30,
            color=color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
            rasterized=True,
        )
        ax.plot(
            [0, max_value],
            [0, max_value],
            linestyle=(0, (4, 3)),
            color="#777777",
            linewidth=1.1,
            label="1:1",
        )
        x_line = np.linspace(0, max_value, 200)
        ax.plot(
            x_line,
            stats["slope"] * x_line + stats["intercept"],
            color=color,
            linewidth=2.2,
            label="OLS fit",
        )
        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("FTIR EC (µg/m³)")
        ax.set_ylabel("HIPS BC equivalent, MAC = 10 (µg/m³)")
        ax.set_title(SITE_LABELS[site_name], loc="left", fontweight="bold")
        ax.text(
            0.04,
            0.96,
            (
                f"n = {stats['n']}\n"
                f"R² = {stats['r_squared']:.3f}\n"
                f"y = {stats['slope']:.2f}x {stats['intercept']:+.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
        )
        ax.text(
            0.97,
            0.04,
            f"Excluded by documented rules: {int(all_pairs['is_excluded'].sum())}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#666666",
        )
        add_panel_label(ax, panel_label)

        metrics_rows.append(
            {
                "site": SITE_LABELS[site_name],
                "n": int(stats["n"]),
                "slope": float(stats["slope"]),
                "intercept": float(stats["intercept"]),
                "r_squared": float(stats["r_squared"]),
                "origin_slope": float(stats["origin_slope"]),
                "excluded_n": int(all_pairs["is_excluded"].sum()),
            }
        )
        points_rows.append(all_pairs)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.93, 0.975),
        ncol=2,
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(fig, "figure_4_same_filter_hips_ftir")
    return pd.DataFrame(metrics_rows), pd.concat(points_rows, ignore_index=True)


def build_date_tolerance_figure(
    aethalometer_data: dict[str, pd.DataFrame], filter_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Figure 3: sensitivity of date-only matching to join tolerance."""
    sensitivity_rows: list[dict[str, object]] = []
    point_rows: list[pd.DataFrame] = []

    for tolerance_days in range(4):
        for site_name, site_config in SITES.items():
            matched = match_aeth_filter_data(
                site_name,
                aethalometer_data[site_name],
                filter_data,
                site_config["code"],
                date_tolerance_days=tolerance_days,
            )
            if matched is None:
                continue

            matched = apply_exclusion_flags(
                matched, site_name, date_tolerance_days=tolerance_days
            )
            matched = apply_threshold_flags(matched, site_name)
            matched["site"] = SITE_LABELS[site_name]
            matched["tolerance_days"] = tolerance_days
            matched["is_flagged"] = matched["is_excluded"] | matched["is_outlier"]
            point_rows.append(matched)

            clean = get_clean_data(matched).dropna(subset=["aeth_bc", "filter_ec"])
            clean = clean[(clean["aeth_bc"] > 0) & (clean["filter_ec"] > 0)]
            stats = calculate_regression_stats(clean["aeth_bc"], clean["filter_ec"])
            if stats is None:
                continue

            sensitivity_rows.append(
                {
                    "site": SITE_LABELS[site_name],
                    "site_key": site_name,
                    "tolerance_days": tolerance_days,
                    "n": int(stats["n"]),
                    "slope": float(stats["slope"]),
                    "intercept_ng_m3": float(stats["intercept"]),
                    "r_squared": float(stats["r_squared"]),
                    "origin_slope": float(stats["origin_slope"]),
                    "flagged_n": int(matched["is_flagged"].sum()),
                }
            )

    sensitivity = pd.DataFrame(sensitivity_rows)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.7))
    panel_specs = [
        ("r_squared", "R²", (0, 1.02)),
        ("slope", "OLS slope: FTIR EC on IR BCc", (0, 1.12)),
        ("n", "Retained matched pairs", (0, sensitivity["n"].max() * 1.12)),
    ]

    for panel_label, ax, (metric, ylabel, ylim) in zip("ABC", axes, panel_specs):
        for site_name in SITES:
            site_data = sensitivity[sensitivity["site_key"] == site_name].sort_values(
                "tolerance_days"
            )
            ax.plot(
                site_data["tolerance_days"],
                site_data[metric],
                marker="o",
                markersize=5.5,
                linewidth=2,
                color=SITES[site_name]["color"],
                label=SITE_LABELS[site_name],
            )
        if metric == "slope":
            ax.axhline(1, color="#777777", linestyle=(0, (4, 3)), linewidth=1)
        ax.set_xticks(range(4))
        ax.set_xlabel("Date-match tolerance (± days)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="x", visible=False)
        add_panel_label(ax, panel_label)

    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "figure_3_date_matching_sensitivity")
    return sensitivity, pd.concat(point_rows, ignore_index=True)


def write_captions() -> None:
    """Write manuscript-ready captions with explicit provenance and caveats."""
    captions = f"""# Data-backed manuscript figure captions

## Figure 1. Observed measurement dates in the example repository assets

Illustrative diagnostic generated from the current repository example assets.
Observed temporal coverage for valid daily aethalometer infrared black-carbon
concentration (IR BCc) summaries and eligible FTIR elemental-carbon (EC) filter
measurements at Beijing, Delhi, JPL/Pasadena, and Addis Ababa. Eligible FTIR EC
measurements satisfy the configured minimum of {MIN_EC_THRESHOLD:g} µg/m³.
Dates and counts are workflow examples, not final study estimates.

## Figure 2. Records supporting comparisons in the example repository assets

Illustrative counts of valid aethalometer days, eligible FTIR EC filters, and
retained same-filter HIPS–FTIR pairs in the current repository example assets.
These counts demonstrate the reporting workflow and must be regenerated from
the verified study dataset before scientific interpretation.

## Figure 3. Sensitivity of example aethalometer–FTIR comparisons to date matching

Illustrative sensitivity of the aethalometer IR BCc versus FTIR EC relationship
to date-only matching tolerance (0 to ±3 days). Panels show the coefficient of
determination (R²), ordinary least-squares slope, and retained matched-pair count.
This example demonstrates an algorithmic sensitivity check; it is neither a
final result nor integration over verified physical filter windows.

## Figure 4. Example same-filter comparison of HIPS-derived BC and FTIR EC

Illustrative comparison generated from the current repository example assets.
HIPS-derived black-carbon equivalent (filter absorption divided by the
configured mass absorption cross-section, MAC = {MAC_VALUE:g} m²/g) versus FTIR
EC for measurements joined by base FilterId, ensuring that each pair represents
the same physical filter. Solid lines are ordinary least-squares fits; dashed
lines denote 1:1 agreement. Panel axes use site-specific ranges. Fits use the
repository's documented example thresholds and site-specific exclusions.
Values and fits are not final study estimates and must be regenerated after the
study dataset, physical sample windows, QC rules, and conversions are verified.

## Provenance

All figures are generated by `workflows/build_manuscript_figures.py` from
`processed_sites/df_*_9am_resampled.pkl` and
`Filter Data/unified_filter_dataset.pkl`. PNG and PDF files are written to
`output/plots/manuscript/`; point-level and summary CSVs are written to
`output/tables/manuscript/`.
"""
    (PLOTS_DIR / "figure_captions.md").write_text(captions, encoding="utf-8")


def main() -> None:
    """Generate all manuscript figures and their provenance tables."""
    configure_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    aethalometer_data = load_aethalometer_data()
    filter_data = load_filter_data()
    filter_pairs = same_filter_data(filter_data)

    coverage = build_coverage_figure(aethalometer_data, filter_data, filter_pairs)
    same_filter_metrics, same_filter_points = build_same_filter_figure(filter_pairs)
    date_sensitivity, date_points = build_date_tolerance_figure(
        aethalometer_data, filter_data
    )

    coverage.to_csv(TABLES_DIR / "measurement_coverage.csv", index=False)
    same_filter_metrics.to_csv(
        TABLES_DIR / "same_filter_hips_ftir_metrics.csv", index=False
    )
    same_filter_points.to_csv(
        TABLES_DIR / "same_filter_hips_ftir_points.csv", index=False
    )
    date_sensitivity.to_csv(
        TABLES_DIR / "date_matching_sensitivity_metrics.csv", index=False
    )
    date_points.to_csv(TABLES_DIR / "date_matching_sensitivity_points.csv", index=False)
    write_captions()

    print(f"Figures: {PLOTS_DIR}")
    print(f"Tables:  {TABLES_DIR}")


if __name__ == "__main__":
    main()
