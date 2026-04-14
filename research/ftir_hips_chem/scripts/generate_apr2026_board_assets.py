#!/usr/bin/env python3
"""Generate slide-ready figures for the April 2026 FTIR group talk draft.

This script focuses on the assets explicitly requested in the April 1 meeting:

1. Iron analysis across all four sites, not just Addis Ababa.
2. Seasonal analysis across all four sites.
3. Summary CSV tables that can be used to write slide bullets or speaker notes.

The script reuses the existing aethalometer/filter matching logic and outlier
definitions so the outputs stay consistent with the notebooks already in the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import SITES  # noqa: E402
from data_matching import load_aethalometer_data, load_filter_data, match_all_parameters  # noqa: E402
from outliers import apply_exclusion_flags, apply_threshold_flags  # noqa: E402
from plotting_legacy import calculate_regression_stats  # noqa: E402
from src.config.multi_site_seasons import SITE_SEASONS  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "research/ftir_hips_chem" / "output" / "group_talk_apr2026"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_LABELS = {
    "Beijing": "Beijing",
    "Delhi": "Delhi",
    "JPL": "JPL/Pasadena",
    "Addis_Ababa": "Addis Ababa",
}

COMPARISONS = {
    "hips_vs_ftir": {
        "x_col": "ftir_ec",
        "y_col": "hips_fabs",
        "x_label": "FTIR EC (µg/m³)",
        "y_label": "HIPS Fabs / MAC (µg/m³)",
        "title": "HIPS vs FTIR EC",
    },
    "hips_vs_aeth": {
        "x_col": "ir_bcc",
        "y_col": "hips_fabs",
        "x_label": "Aethalometer IR BCc (µg/m³)",
        "y_label": "HIPS Fabs / MAC (µg/m³)",
        "title": "HIPS vs Aethalometer IR BCc",
    },
}


def season_for_month(site_name: str, month: int) -> str | None:
    """Return the site-specific season label for a month."""
    for season_name, info in SITE_SEASONS[site_name].items():
        if month in info["months"]:
            return season_name
    return None


def prepare_matched_data() -> dict[str, pd.DataFrame]:
    """Load and clean matched multi-parameter data for each site."""
    aethalometer_data = load_aethalometer_data()
    filter_data = load_filter_data()

    prepared: dict[str, pd.DataFrame] = {}

    for site_name, config in SITES.items():
        df_aeth = aethalometer_data.get(site_name)
        if df_aeth is None:
            continue

        matched = match_all_parameters(site_name, config["code"], df_aeth, filter_data)
        if matched is None or len(matched) < 3:
            continue

        matched = matched.copy()
        matched["date"] = pd.to_datetime(matched["date"])
        matched["season"] = matched["date"].dt.month.map(lambda m: season_for_month(site_name, m))

        # Threshold outlier logic expects ng/m³ columns with these names.
        if "ir_bcc" in matched.columns:
            matched["aeth_bc"] = matched["ir_bcc"] * 1000
        if "ftir_ec" in matched.columns:
            matched["filter_ec"] = matched["ftir_ec"] * 1000

        matched = apply_exclusion_flags(matched, site_name)
        matched = apply_threshold_flags(matched, site_name)
        matched["is_flagged"] = matched.get("is_excluded", False) | matched.get("is_outlier", False)
        prepared[site_name] = matched

    return prepared


def annotate_panel(ax, site_name: str, lines: list[str]) -> None:
    """Add a compact annotation box to a panel."""
    text = "\n".join(lines)
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_title(SITE_LABELS[site_name], fontsize=13, fontweight="bold")


def set_equal_axes(ax, x: np.ndarray, y: np.ndarray) -> float:
    """Set equal axes with a 1:1 line."""
    max_val = max(np.nanmax(x), np.nanmax(y)) * 1.08
    ax.plot([0, max_val], [0, max_val], "--", color="gray", linewidth=1.1, alpha=0.6)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal", adjustable="box")
    return max_val


def save_iron_split_figure(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a 2x2 multi-site iron split figure for HIPS vs FTIR EC."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    summary_rows: list[dict[str, object]] = []

    for ax, site_name in zip(axes, SITES):
        df = data.get(site_name)
        if df is None:
            ax.set_visible(False)
            continue

        cfg = COMPARISONS["hips_vs_ftir"]
        valid = df.dropna(subset=[cfg["x_col"], cfg["y_col"], "iron"]).copy()
        valid = valid[~valid["is_flagged"]].copy()

        if len(valid) < 6:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(SITE_LABELS[site_name], fontsize=13, fontweight="bold")
            continue

        iron_threshold = valid["iron"].median()
        valid["iron_group"] = np.where(valid["iron"] <= iron_threshold, "Low iron", "High iron")

        group_colors = {"Low iron": "#4C78A8", "High iron": "#E45756"}
        annotation_lines = [f"Median iron = {iron_threshold:.2f} µg/m³", f"n = {len(valid)}"]

        for group_name in ["Low iron", "High iron"]:
            grp = valid[valid["iron_group"] == group_name]
            ax.scatter(
                grp[cfg["x_col"]],
                grp[cfg["y_col"]],
                label=f"{group_name} (n={len(grp)})",
                color=group_colors[group_name],
                alpha=0.75,
                s=50,
                edgecolors="black",
                linewidth=0.4,
            )
            stats = calculate_regression_stats(grp[cfg["x_col"]], grp[cfg["y_col"]])
            if stats is not None:
                x_line = np.linspace(0, grp[cfg["x_col"]].max() * 1.05, 100)
                y_line = stats["slope"] * x_line + stats["intercept"]
                ax.plot(x_line, y_line, color=group_colors[group_name], linewidth=2)
                annotation_lines.append(f"{group_name}: slope={stats['slope']:.2f}, R²={stats['r_squared']:.2f}")
                summary_rows.append(
                    {
                        "site": SITE_LABELS[site_name],
                        "comparison": cfg["title"],
                        "grouping": "iron_median_split",
                        "group": group_name,
                        "threshold_ug_m3": round(float(iron_threshold), 3),
                        "n": int(stats["n"]),
                        "slope": round(float(stats["slope"]), 4),
                        "intercept": round(float(stats["intercept"]), 4),
                        "r_squared": round(float(stats["r_squared"]), 4),
                    }
                )

        overall_stats = calculate_regression_stats(valid[cfg["x_col"]], valid[cfg["y_col"]])
        if overall_stats is not None:
            summary_rows.append(
                {
                    "site": SITE_LABELS[site_name],
                    "comparison": cfg["title"],
                    "grouping": "iron_median_split",
                    "group": "Overall",
                    "threshold_ug_m3": round(float(iron_threshold), 3),
                    "n": int(overall_stats["n"]),
                    "slope": round(float(overall_stats["slope"]), 4),
                    "intercept": round(float(overall_stats["intercept"]), 4),
                    "r_squared": round(float(overall_stats["r_squared"]), 4),
                }
            )

        set_equal_axes(ax, valid[cfg["x_col"]].to_numpy(), valid[cfg["y_col"]].to_numpy())
        ax.set_xlabel(cfg["x_label"])
        ax.set_ylabel(cfg["y_label"])
        ax.legend(fontsize=8, loc="lower right")
        annotate_panel(ax, site_name, annotation_lines)

    fig.suptitle(
        "Iron Median Split by Site: HIPS vs FTIR EC\nRequested for April 2026 group-talk board",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "iron_split_all_sites_hips_vs_ftir.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "iron_split_all_sites_hips_vs_ftir_summary.csv", index=False)
    return summary_df


def save_seasonal_split_figure(data: dict[str, pd.DataFrame], comparison_key: str) -> pd.DataFrame:
    """Create a 2x2 multi-site seasonal figure for a given comparison."""
    cfg = COMPARISONS[comparison_key]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    summary_rows: list[dict[str, object]] = []

    for ax, site_name in zip(axes, SITES):
        df = data.get(site_name)
        if df is None:
            ax.set_visible(False)
            continue

        valid = df.dropna(subset=[cfg["x_col"], cfg["y_col"], "season"]).copy()
        valid = valid[~valid["is_flagged"]].copy()
        if len(valid) < 6:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(SITE_LABELS[site_name], fontsize=13, fontweight="bold")
            continue

        season_cfg = SITE_SEASONS[site_name]
        annotation_lines = [f"Overall n = {len(valid)}"]

        for season_name, info in season_cfg.items():
            grp = valid[valid["season"] == season_name]
            if len(grp) < 3:
                continue

            ax.scatter(
                grp[cfg["x_col"]],
                grp[cfg["y_col"]],
                label=f"{season_name} (n={len(grp)})",
                color=info["color"],
                alpha=0.75,
                s=48,
                edgecolors="black",
                linewidth=0.35,
            )

            stats = calculate_regression_stats(grp[cfg["x_col"]], grp[cfg["y_col"]])
            if stats is not None:
                x_line = np.linspace(0, grp[cfg["x_col"]].max() * 1.05, 100)
                y_line = stats["slope"] * x_line + stats["intercept"]
                ax.plot(x_line, y_line, color=info["color"], linewidth=2)
                annotation_lines.append(f"{season_name}: slope={stats['slope']:.2f}, R²={stats['r_squared']:.2f}")
                summary_rows.append(
                    {
                        "site": SITE_LABELS[site_name],
                        "comparison": cfg["title"],
                        "grouping": "season",
                        "group": season_name,
                        "n": int(stats["n"]),
                        "slope": round(float(stats["slope"]), 4),
                        "intercept": round(float(stats["intercept"]), 4),
                        "r_squared": round(float(stats["r_squared"]), 4),
                    }
                )

        overall_stats = calculate_regression_stats(valid[cfg["x_col"]], valid[cfg["y_col"]])
        if overall_stats is not None:
            summary_rows.append(
                {
                    "site": SITE_LABELS[site_name],
                    "comparison": cfg["title"],
                    "grouping": "season",
                    "group": "Overall",
                    "n": int(overall_stats["n"]),
                    "slope": round(float(overall_stats["slope"]), 4),
                    "intercept": round(float(overall_stats["intercept"]), 4),
                    "r_squared": round(float(overall_stats["r_squared"]), 4),
                }
            )

        set_equal_axes(ax, valid[cfg["x_col"]].to_numpy(), valid[cfg["y_col"]].to_numpy())
        ax.set_xlabel(cfg["x_label"])
        ax.set_ylabel(cfg["y_label"])
        ax.legend(fontsize=7, loc="lower right")
        annotate_panel(ax, site_name, annotation_lines)

    fig.suptitle(
        f"Seasonal Split by Site: {cfg['title']}\nRequested for April 2026 group-talk board",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUTPUT_DIR / f"seasonal_split_all_sites_{comparison_key}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / f"seasonal_split_all_sites_{comparison_key}_summary.csv", index=False)
    return summary_df


def save_board_manifest(artifacts: list[dict[str, str]]) -> None:
    """Write a small CSV manifest mapping files to slide use."""
    pd.DataFrame(artifacts).to_csv(OUTPUT_DIR / "board_asset_manifest.csv", index=False)


def main() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
        }
    )

    data = prepare_matched_data()
    if not data:
        raise SystemExit("No matched site data available; nothing to plot.")

    iron_summary = save_iron_split_figure(data)
    seasonal_hips_ftir = save_seasonal_split_figure(data, "hips_vs_ftir")
    seasonal_hips_aeth = save_seasonal_split_figure(data, "hips_vs_aeth")

    artifacts = [
        {
            "asset": "iron_split_all_sites_hips_vs_ftir.png",
            "intended_slide": "All-site iron analysis",
            "notes": "Use this to show the high-vs-low iron regressions stay similar across sites.",
        },
        {
            "asset": "seasonal_split_all_sites_hips_vs_ftir.png",
            "intended_slide": "All-site seasonal rule-out (HIPS vs FTIR EC)",
            "notes": "Supports the claim that seasonality does not remove the Addis anomaly.",
        },
        {
            "asset": "seasonal_split_all_sites_hips_vs_aeth.png",
            "intended_slide": "Optional seasonal support for the core HIPS vs aeth figure",
            "notes": "Useful if the talk keeps the aethalometer-first framing.",
        },
        {
            "asset": "iron_split_all_sites_hips_vs_ftir_summary.csv",
            "intended_slide": "Speaker notes / bullet writing",
            "notes": f"{len(iron_summary)} rows of per-site iron split regressions.",
        },
        {
            "asset": "seasonal_split_all_sites_hips_vs_ftir_summary.csv",
            "intended_slide": "Speaker notes / bullet writing",
            "notes": f"{len(seasonal_hips_ftir)} rows of per-site seasonal regressions.",
        },
        {
            "asset": "seasonal_split_all_sites_hips_vs_aeth_summary.csv",
            "intended_slide": "Speaker notes / bullet writing",
            "notes": f"{len(seasonal_hips_aeth)} rows of per-site seasonal regressions.",
        },
    ]
    save_board_manifest(artifacts)

    print(f"Saved assets to: {OUTPUT_DIR}")
    for item in artifacts:
        print(f"- {item['asset']}: {item['intended_slide']}")


if __name__ == "__main__":
    main()
