"""Describe matched calibration transfer by Bega/dry, Belg, and Kiremt.

Uses already frozen source-calibration predictions. No optical observation or
Ethiopian season selects a preprocessing recipe, PLS component, or MAC. New
reporting uses belg_feb; dry_feb reproduces the historical calendar sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))

from calibration_seasonality import (  # noqa: E402
    CALENDARS, CLAIM_LIMIT, PRIMARY_CALENDAR, PUBLISHED_CONTEXT, UNDATED,
    complete_year_summary, coverage_summary, february_sensitivity, mac_scenario_ledger,
    season_ledger, seasonal_optical_summary, source_calendar_errors,
    with_published_context,
)
from config import MAC_VALUE, resolve_seasons  # noqa: E402
from plotting import apply_default_style  # noqa: E402
from portable_cohort_screen import PUBLIC_CHEMISTRY_PATH, load_public_chemistry  # noqa: E402


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(frame: pd.DataFrame, output: Path, name: str) -> None:
    frame.to_csv(output / name, index=False)


def make_figures(summary: pd.DataFrame, coverage: pd.DataFrame, source: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    apply_default_style()
    primary = summary.loc[
        summary.season_convention.eq(PRIMARY_CALENDAR)
        & summary.aggregation_sensitivity.eq("all_original_scans")
        & summary.cohort.eq("all_common_direct_pairs")
        & summary.scenario_mac_m2_g.eq(10) & summary.season.ne(UNDATED)
    ]
    names = {
        "raw_full": "Raw spectra: full grid", "raw_shared": "Raw spectra: shared window",
        "spline6_4": "Spline baseline (6/4)", "spline6_3": "Spline baseline (6/3)",
        "spline6_6": "Spline baseline (6/6)", "neutral_arpls": "arPLS: original 50-iteration run",
        "neutral_arpls_iter500": "arPLS: 500-iteration check",
        "removed_continuum": "Removed spline background",
        "raw_without_1600": "Raw without 1580–1650 cm⁻¹",
        "spline_without_1600": "Spline without 1580–1650 cm⁻¹",
        "spline_1600_only": "Spline: 1580–1650 cm⁻¹ only",
        "second_derivative": "Second derivative",
        PUBLISHED_CONTEXT: "Current published FTIR (context)",
    }
    for site, part in primary.groupby("SiteCode", sort=True):
        seasons = list(resolve_seasons(PRIMARY_CALENDAR))
        arms = [name for name in names if name in part.arm.unique()]
        arms.extend(name for name in part.arm.unique() if name not in arms)
        matrix = part.pivot(index="arm", columns="season", values="median_ftir_minus_optical_ugm3").reindex(index=arms, columns=seasons)
        counts = part.groupby("season").n_direct_pairs.first().reindex(seasons)
        limit = max(float(np.nanmax(np.abs(matrix))), .1)
        fig, ax = plt.subplots(figsize=(12, max(5.5, len(arms)*.44)))
        im = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
        ax.set_yticks(range(len(arms)), [names.get(arm, arm) for arm in arms])
        ax.set_xticks(range(3), [f"{season}\nn = {int(counts.loc[season])}" for season in seasons])
        for i in range(len(arms)):
            for j in range(3):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:+.2f}", ha="center", va="center", color="white" if abs(value) > .62*limit else "black")
        ax.set_title(f"{site}: median FTIR minus optical BC, by season\nSame direct-HIPS filter panel for every method; MAC 10 m²/g")
        fig.colorbar(im, ax=ax, label="Median difference (µg/m³)", shrink=.8)
        fig.text(.5, .01, "Bega Oct–Jan · Belg Feb–May · Kiremt Jun–Sep. Optical agreement does not establish EC accuracy.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, 1))
        for extension in ("png", "svg"):
            fig.savefig(output / f"seasonal_optical_disagreement_{site}.{extension}")
        plt.close(fig)
    # Coverage explains why a pooled mean or lot effect can be misleading.
    cov = coverage.loc[
        coverage.season_convention.eq(PRIMARY_CALENDAR)
        & coverage.aggregation_sensitivity.eq("all_original_scans")
        & coverage.ExternalFilterType.eq("PM2.5") & coverage.season.ne(UNDATED)
    ]
    first_arm = source.arm.iloc[0]
    months = source.loc[source.arm.eq(first_arm) & source.bin_type.eq("calendar_month")].set_index("bin_number").reindex(range(1, 13))
    for site, part in cov.groupby("SiteCode", sort=True):
        seasons = list(resolve_seasons(PRIMARY_CALENDAR))
        lots = sorted(part.lot_group.unique())
        pivot = part.pivot(index="lot_group", columns="season", values="n_physical_filters").reindex(index=lots, columns=seasons).fillna(0)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})
        im = axes[0].imshow(pivot, aspect="auto", cmap="Blues")
        axes[0].set_yticks(range(len(lots)), [f"Lot {lot}" for lot in lots])
        axes[0].set_xticks(range(3), ["Bega/dry", "Belg", "Kiremt"])
        for i in range(len(lots)):
            for j in range(3):
                value = int(pivot.iloc[i, j])
                axes[0].text(j, i, str(value), ha="center", va="center", color="white" if value > pivot.to_numpy().max()*.65 else "black")
        axes[0].set_title(f"{site}: dated PM2.5 filters by lot and season")
        fig.colorbar(im, ax=axes[0], label="Physical filters", shrink=.8)
        axes[1].bar(range(1, 13), months.n_filters.fillna(0), color="#4C566A")
        axes[1].set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
        axes[1].set_ylabel("Source calibration filters")
        axes[1].set_title("IMPROVE source support by calendar month\nThese are not Ethiopian rainy seasons")
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output / f"seasonal_sampling_support_{site}.{extension}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--additional-calibration-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research/ftir_hips_chem/output/tables/calibration_seasonality_20261103")
    parser.add_argument("--plot-dir", type=Path, default=ROOT / "research/ftir_hips_chem/output/plots/calibration_seasonality_20261103")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    target_parts, heldout_parts, input_paths = [], [], []
    for directory in [args.calibration_dir, *args.additional_calibration_dir]:
        target_path, heldout_path = directory / "target_optical_pairs.csv", directory / "heldout_predictions.csv"
        target_parts.append(pd.read_csv(target_path))
        heldout_parts.append(pd.read_csv(heldout_path))
        input_paths.extend((target_path, heldout_path))
    target = pd.concat(target_parts, ignore_index=True)
    heldout = pd.concat(heldout_parts, ignore_index=True)
    public = load_public_chemistry(ROOT)
    input_paths.append(ROOT / PUBLIC_CHEMISTRY_PATH)
    fabs = public[["id", "hips_fabs_Mm1"]].rename(columns={"id": "base_filter_id"})
    target = target.merge(fabs, on="base_filter_id", how="left", validate="many_to_one")
    direct = target.current_optical_source.eq("direct_hips_fabs") & target.current_optical_bc_ugm3.notna()
    np.testing.assert_allclose(target.loc[direct, "hips_fabs_Mm1"]/MAC_VALUE,
                               target.loc[direct, "current_optical_bc_ugm3"], rtol=1e-10, atol=1e-10)
    raw_arms = target.arm.unique().tolist()
    ledger = season_ledger(with_published_context(target))
    scenarios = mac_scenario_ledger(ledger)
    summary = seasonal_optical_summary(scenarios)
    coverage = coverage_summary(ledger)
    lot_coverage = coverage_summary(ledger, by_lot=True)
    source = source_calendar_errors(heldout)
    write(ledger, args.output_dir, "physical_season_eligibility_ledger.csv")
    write(coverage, args.output_dir, "season_coverage.csv")
    write(lot_coverage, args.output_dir, "season_lot_coverage.csv")
    write(coverage_summary(ledger, by_year=True), args.output_dir, "season_year_coverage.csv")
    write(summary, args.output_dir, "seasonal_optical_summary.csv")
    write(seasonal_optical_summary(scenarios, by_lot=True), args.output_dir, "season_lot_optical_summary.csv")
    write(seasonal_optical_summary(scenarios, by_year=True), args.output_dir, "season_year_optical_summary.csv")
    write(february_sensitivity(summary), args.output_dir, "february_calendar_sensitivity.csv")
    write(source, args.output_dir, "source_calendar_error_summary.csv")
    complete_years, year_audit = complete_year_summary(scenarios)
    write(complete_years, args.output_dir, "complete_calendar_year_optical_summary.csv")
    write(year_audit, args.output_dir, "complete_calendar_year_selection_ledger.csv")
    physical = ledger.loc[ledger.arm.eq(raw_arms[0]) & ledger.aggregation_sensitivity.eq("all_original_scans")]
    write(physical.loc[physical.calendar_month.eq(2)], args.output_dir, "february_moving_filter_ledger.csv")
    write(physical.loc[~physical.base_direct_eligible], args.output_dir, "noneligible_physical_filter_ledger.csv")
    month_support = physical.groupby([
        "season_convention", "SiteCode", "ExternalFilterType", "season", "lot_group", "month_block"
    ], dropna=False).agg(n_physical_filters=("base_filter_id", "size"), n_direct_pairs=("base_direct_eligible", "sum")).reset_index()
    write(month_support, args.output_dir, "season_lot_month_support.csv")
    # Calculate common observed month blocks rather than assuming lot overlap.
    overlap = []
    for (convention, site, season), part in month_support.loc[month_support.ExternalFilterType.eq("PM2.5") & month_support.season.ne(UNDATED)].groupby(["season_convention", "SiteCode", "season"]):
        by_lot = {lot: set(part.loc[part.lot_group.eq(lot), "month_block"].dropna()) for lot in ("248", "251")}
        common = sorted(by_lot["248"] & by_lot["251"])
        overlap.append({"season_convention": convention, "SiteCode": site, "season": season,
                        "lot248_month_blocks": len(by_lot["248"]), "lot251_month_blocks": len(by_lot["251"]),
                        "n_shared_month_blocks": len(common), "shared_month_blocks": ";".join(common)})
    write(pd.DataFrame(overlap), args.output_dir, "known_lot_common_month_support.csv")
    scenario_columns = ["season_convention", "arm", "prediction_product", "aggregation_sensitivity", "SiteCode", "base_filter_id", "native_date", "season", "lot_group", "common_arm_direct_eligible", "seasonal_pair_reason", "prior_problem_filter", "hips_fabs_Mm1", "prediction_ugm3", "scenario_mac_m2_g", "scenario_optical_bc_ugm3", "scenario_ftir_minus_optical_ugm3"]
    write(scenarios[scenario_columns], args.output_dir, "scenario_physical_filter_ledger.csv")
    make_figures(summary, lot_coverage, source, args.plot_dir)
    primary = physical.loc[physical.season_convention.eq(PRIMARY_CALENDAR)]
    info = {
        "schema_version": 1, "primary_calendar": PRIMARY_CALENDAR, "sensitivity_calendars": list(CALENDARS),
        "calendar_source": "docs/site-seasonality.md; canonical config.resolve_seasons()",
        "date_rule": "native sampling start date; end-date season boundary crossings are flagged",
        "n_target_physical_filters": len(primary), "n_target_PM25": int(primary.is_pm25.sum()),
        "n_direct_HIPS_pairs": int(primary.base_direct_eligible.sum()),
        "n_common_arm_direct_pairs": int(primary.common_arm_direct_eligible.sum()),
        "n_PM25_missing_dates": int((primary.is_pm25 & primary.date_missing).sum()),
        "n_February_direct_pairs": int((primary.calendar_month.eq(2) & primary.base_direct_eligible).sum()),
        "matched_arms": raw_arms, "context_arm": PUBLISHED_CONTEXT,
        "mac_scenarios_m2_g": [7.5, 10., 12.5, 15.], "HIPS_wavelength_nm": 633,
        "mac_note": "Illustrative scenarios, not fitted or independently validated MAC estimates. The common 7.5 literature value at550nm is not transferred as an estimate here.",
        "claim_limit": CLAIM_LIMIT,
        "limitations": [
            "Calendar bins do not establish observed rainfall or physical seasonal mechanisms.",
            "No new season-specific model is fit; source sites/months are unbalanced and Ethiopian target TOR is unavailable.",
            "Kiremt has no lot248 PM2.5 observations; season and lot contrasts are not interchangeable.",
            "All field blanks lack collection dates, so season-specific blank banks cannot be constructed from this metadata.",
            "Source-month scores reuse frozen site-held-out predictions; they are not new leave-season-out validation.",
            "Target optical eligibility preserves the existing current-pair cohort, requiring its published FTIR/optical availability.",
            "Published FTIR context was not recalibrated and is not another independent EC reference.",
            "The original neutral_arpls arm includes a50-iteration stopping limit; convergence findings belong in the calibration report.",
        ],
        "inputs": [{"name": path.name, "relative_context": path.parent.name, "sha256": sha(path), "bytes": path.stat().st_size} for path in input_paths],
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            Path(__file__), ROOT / "research/ftir_hips_chem/scripts/calibration_seasonality.py",
            ROOT / "research/ftir_hips_chem/scripts/config.py",
        )},
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(info, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: info[key] for key in ("primary_calendar", "n_target_physical_filters", "n_target_PM25", "n_direct_HIPS_pairs", "n_common_arm_direct_pairs", "n_PM25_missing_dates", "n_February_direct_pairs")}, indent=2))
    chosen = summary.loc[summary.season_convention.eq(PRIMARY_CALENDAR) & summary.aggregation_sensitivity.eq("all_original_scans") & summary.cohort.eq("all_common_direct_pairs") & summary.scenario_mac_m2_g.eq(10) & summary.season.ne(UNDATED)]
    print(chosen[["arm", "season", "n_direct_pairs", "median_ftir_minus_optical_ugm3"]].to_string(index=False))


if __name__ == "__main__":
    main()
