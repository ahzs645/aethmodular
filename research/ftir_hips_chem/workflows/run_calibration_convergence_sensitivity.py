"""Recalibrate a fixed 500-iteration neutral baseline on the matched source set.

Extends a completed ``run_existing_calibration_controls.py`` experiment without
changing its data, physical-filter inclusion, source-site folds, component rule,
or spectral window. All paths are CLI inputs and original artifacts remain
read-only. Target optical measurements never choose a baseline or calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pybaselines
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neutral_convergence import neutral_with_diagnostics  # noqa: E402
from run_existing_calibration_controls import (  # noqa: E402
    paired_site_bootstrap,
    score,
    source_validation,
    target_predictions,
)
from run_native_addis_baseline_audit import load_native  # noqa: E402

ARM = "neutral_arpls_iter500"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--ftir-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "research/ftir_hips_chem/output/tables/calibration_convergence_20261103")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.parent_run / "summary.json").read_text())
    if manifest["source_policy"] != "documented_qc" or manifest["source_filters"] != 905:
        raise ValueError("This extension requires the audited common905 source run")
    dependencies = {name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "pandas", "scikit-learn", "pybaselines")}
    if dependencies["pybaselines"] != "1.2.1":
        raise ValueError("Use pybaselines1.2.1 to isolate iteration-budget sensitivity")
    inputs = {item["name"]: item for item in manifest["sources"]}
    archive_path = args.ftir_dir / "apps/apps_data.npz"
    if sha(archive_path) != inputs[archive_path.name]["sha256"]:
        raise ValueError("Source spectra archive changed")
    for name in ("ETAD_FTIR_spectra.csv", "ETAD_metadata.csv", "etad_spectra_lotmap.csv",
                 "spectra_baselined_AIRSPEC.csv"):
        if sha(args.native_dir / name) != inputs[name]["sha256"]:
            raise ValueError(f"Native input changed: {name}")
    original = pd.read_csv(args.parent_run / "source_original906_with_flags.csv")
    primary = pd.read_csv(args.parent_run / "source_cohort.csv")
    if not original.AnalysisId.is_unique or not primary.AnalysisId.is_unique:
        raise ValueError("Duplicate source analysis identity")
    if not primary.FilterId.is_unique or len(primary) != 905 or len(original) != 906:
        raise ValueError("Source physical-filter population changed")
    source_rows = pd.Index(original.AnalysisId).get_indexer(primary.AnalysisId)
    if np.any(source_rows < 0) or not np.array_equal(source_rows, np.sort(source_rows)):
        raise ValueError("Primary source rows are not an ordered subset of original906")
    if original.iloc[source_rows].is_excluded.any() or primary.is_excluded.any():
        raise ValueError("A flagged source record entered the primary calibration")
    np.testing.assert_array_equal(original.iloc[source_rows].FilterId, primary.FilterId)
    np.testing.assert_allclose(original.iloc[source_rows].TOR_EC_loading_ug,
                               primary.TOR_EC_loading_ug, atol=0, rtol=0)
    cache_path = args.parent_run / "preprocessing_cache.npz"
    cache = np.load(cache_path, allow_pickle=False)
    if str(cache["signature"].item()) != manifest["signature"]:
        raise ValueError("Parent preprocessing cache signature changed")
    raw = cache["raw_full"]
    wn = np.load(archive_path, allow_pickle=False)["wn"].astype(float)
    scans, native_wn, native_raw, _, _ = load_native(args.native_dir)
    target_aligned = np.vstack([
        np.interp(wn[::-1], native_wn[::-1], row[::-1])[::-1] for row in native_raw
    ])
    if raw.shape != (len(original) + len(scans), len(wn)):
        raise ValueError("Parent source/target matrix changed shape")
    np.testing.assert_allclose(raw[len(original):], target_aligned, atol=0, rtol=0)
    row_ledger = pd.concat([
        pd.DataFrame({"population": "source", "analysis_id": original.AnalysisId,
                      "base_filter_id": original.FilterId.astype(str),
                      "in_primary_source": original.AnalysisId.isin(primary.AnalysisId)}),
        pd.DataFrame({"population": "target", "analysis_id": scans.SampleAnalysisId,
                      "base_filter_id": scans.base_filter_id, "in_primary_source": False}),
    ], ignore_index=True)
    row_ledger["row"] = np.arange(len(row_ledger))
    with threadpool_limits(limits=1):
        print(f"Replaying default and500-iteration corrections for {len(raw)} spectra", flush=True)
        grid, default, diag50 = neutral_with_diagnostics(wn, raw, max_iter=50)
        grid500, extended, diag500 = neutral_with_diagnostics(wn, raw, max_iter=500)
        np.testing.assert_array_equal(grid, grid500)
        retained = (grid > 1425) & (grid < 4000)
        default = default[:, retained][:, ::-1]
        extended = extended[:, retained][:, ::-1]
        np.testing.assert_allclose(default, cache["neutral_arpls"], atol=1e-12, rtol=0)
        if not np.isfinite(extended).all():
            raise ValueError("Extended correction returned nonfinite values; do not silently delete cases")
        diagnostics = pd.concat([diag50, diag500], ignore_index=True).merge(
            row_ledger, on="row", how="left", validate="many_to_one")
        diagnostics.to_csv(output / "preprocessing_diagnostics.csv", index=False)
        change = row_ledger.copy()
        change["max_abs_correction_change"] = np.max(np.abs(extended - default), axis=1)
        change["rms_correction_change"] = np.sqrt(np.mean((extended - default) ** 2, axis=1))
        change.to_csv(output / "preprocessing_changes.csv", index=False)
        np.savez_compressed(output / "preprocessing_cache.npz", wavenumbers=grid[retained][::-1],
                            neutral_arpls_iter500=extended)
        counts = diagnostics.groupby(["population", "max_iter", "in_primary_source"]).agg(
            n=("row", "size"), converged_n=("success", "sum"))
        print(counts.to_string(), flush=True)
        metrics, heldout, models = source_validation(
            {ARM: extended[source_rows]}, primary, output,
            max_components=max(manifest["inner_cv"]["candidates"]),
        )
        assigned = pd.read_csv(output / "source_cohort.csv")
        np.testing.assert_array_equal(assigned.AnalysisId, primary.AnalysisId)
        np.testing.assert_array_equal(assigned.outer_fold, primary.outer_fold)
        target_predictions({ARM: extended}, len(original), scans, models, output)
    comparison = pd.concat([pd.read_csv(args.parent_run / "heldout_predictions.csv"), heldout],
                           ignore_index=True)
    paired_site_bootstrap(comparison, output)
    strata = []
    for column in ("OC_EC_stratum", "EC_loading_stratum"):
        for value, part in heldout.groupby(column, observed=True):
            strata.append({"arm": ARM, "stratum_type": column, "stratum": str(value),
                           "n_sites": part.Site.nunique(),
                           **score(part.TOR_EC_loading_ug, part.prediction_loading_ug)})
    pd.DataFrame(strata).to_csv(output / "source_stratum_metrics.csv", index=False)
    comparison_metrics = pd.concat([pd.read_csv(args.parent_run / "metrics.csv"), metrics],
                                   ignore_index=True)
    comparison_metrics.to_csv(output / "all_arm_metrics.csv", index=False)
    blank_predictions = pd.read_csv(output / "target_predictions.csv")
    blank_predictions = blank_predictions.loc[
        blank_predictions.ExternalFilterType.eq("FB")
        & blank_predictions.aggregation_sensitivity.eq("all_original_scans")]
    if len(blank_predictions) != 37 or not blank_predictions.base_filter_id.is_unique:
        raise ValueError("Expected37 unique physical field-blank predictions")
    blank_predictions.to_csv(output / "target_blank_predictions.csv", index=False)
    source500 = diagnostics.loc[diagnostics.in_primary_source & diagnostics.max_iter.eq(500)]
    target500 = diagnostics.loc[diagnostics.population.eq("target") & diagnostics.max_iter.eq(500)]
    package_root = Path(pybaselines.__file__).parent
    summary = {
        "arm": ARM, "parent_run_signature": manifest["signature"],
        "source_policy": manifest["source_policy"], "source_filters": len(primary),
        "source_sites": primary.Site.nunique(), "target_scans": len(scans),
        "default50_replay_max_abs_error": float(np.max(np.abs(default - cache["neutral_arpls"]))),
        "source905_converged_at500": int(source500.success.sum()),
        "target319_converged_at500": int(target500.success.sum()),
        "baseline_recipe": {"lambda": 1e6, "max_iter": 500, "tol": 1e-3,
                            "num_knots": 100, "spline_degree": 3, "diff_order": 2,
                            "fit_grid": ">=700, excluding1100<wn<1300",
                            "prediction_grid": "1425<wn<4000"},
        "outer_cv": manifest["outer_cv"], "inner_cv": manifest["inner_cv"],
        "time_block": manifest["time_block"], "runtime_versions": dependencies,
        "parent_artifacts_sha256": {name: sha(args.parent_run / name) for name in (
            "summary.json", "preprocessing_cache.npz", "source_original906_with_flags.csv",
            "source_cohort.csv", "heldout_predictions.csv", "metrics.csv")},
        "source_inputs": manifest["sources"],
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            Path(__file__), ROOT / "research/ftir_hips_chem/scripts/neutral_convergence.py",
            ROOT / "research/ftir_hips_chem/workflows/run_existing_calibration_controls.py",
            ROOT / "research/ftir_hips_chem/scripts/pls_transfer.py",
            ROOT / "research/ftir_ec_phase3/scripts/neutral_baseline.py")},
        "pybaselines_source_sha256": {str(p.relative_to(package_root)): sha(p)
                                       for p in sorted(package_root.rglob("*.py"))},
        "scope": [
            "Numerical sensitivity; 500iterations fixed before source or optical scoring.",
            "Same905 physical source filters, source-site folds, PLS component rule and target scans as parent.",
            "All original source906 and target319 baseline statuses retained; finite nonconvergence is explicit.",
            "Site-heldout source TOR scores do not establish Ethiopian EC accuracy.",
            "Target optical comparisons and37blank predictions are descriptive; no new calibration labels.",
            "No upstream dataset, source exclusion, production model or public product changed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(metrics.loc[metrics.evaluation.isin(["site_grouped_pooled", "time_block_2022_11_01"])]
          .to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
