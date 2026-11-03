"""Audit the completed matched calibration: existing flags, blanks and support.

Consumes run_existing_calibration_controls.py outputs and the same native input
folder. It neither selects a different model nor changes the production EC data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_existing_calibration_controls import load_native, score, sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--ftir-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run_dir
    manifest = json.loads((folder / "summary.json").read_text())
    if manifest["source_policy"] != "documented_qc" or manifest["source_filters"] != 905:
        raise ValueError("This diagnostic requires the905-filter primary source-QC run")
    original = pd.read_csv(folder / "source_original906_with_flags.csv")
    source = pd.read_csv(folder / "source_cohort.csv")
    rows = original.reset_index().set_index("AnalysisId").loc[source.AnalysisId, "index"].to_numpy(int)
    models = pd.read_csv(folder / "final_source_models.csv")
    primary = pd.read_csv(folder / "heldout_predictions.csv")
    audit = pd.read_csv(folder / "historical906_all_scans/heldout_predictions.csv")
    sensitivity = []
    for arm, part in audit.groupby("arm", sort=False):
        refitted = primary.loc[primary.arm.eq(arm)]
        common = part.loc[part.AnalysisId.isin(source.AnalysisId)]
        old_fold = common.set_index("AnalysisId").loc[refitted.AnalysisId, "fold"].to_numpy()
        np.testing.assert_array_equal(old_fold, refitted.fold)
        for cohort, table in [("original906_including_known_offset", part),
                              ("same_original_models_scored_on905", common),
                              ("refitted905_documented_QC", refitted)]:
            sensitivity.append({"arm": arm, "cohort": cohort, "units": "ug/filter",
                                **score(table.TOR_EC_loading_ug, table.prediction_loading_ug)})
    pd.DataFrame(sensitivity).to_csv(folder / "source_flag_sensitivity.csv", index=False)
    target = pd.read_csv(folder / "target_predictions.csv")
    blank_rows = []
    blank = target.loc[target.ExternalFilterType.eq("FB")
                       & target.aggregation_sensitivity.eq("all_original_scans")]
    for arm, part in blank.groupby("arm", sort=False):
        if len(part) != 37 or part.MediaId.nunique() != 37:
            raise ValueError("Blank summary must contain37 independent physical field blanks per arm")
        values = part.prediction_loading_ug.to_numpy(float)
        blank_rows.append({"arm": arm, "n_physical_blanks": len(part), "units": "ug/filter",
                           "median": np.median(values), "mean": np.mean(values),
                           "q25": np.quantile(values, .25), "q75": np.quantile(values, .75),
                           "rmse_from_nominal_zero": np.sqrt(np.mean(values**2)),
                           "min": values.min(), "max": values.max(),
                           "negative_n": int((values < 0).sum()),
                           "scope": "out-of-calibration field-blank diagnostic; nominal zero is not a measured TOR label"})
    pd.DataFrame(blank_rows).to_csv(folder / "target_blank_predictions.csv", index=False)
    scans, _, _, _, _ = load_native(args.native_dir)
    groups = scans.groupby("MediaId", sort=True).indices
    cache = np.load(folder / "preprocessing_cache.npz", allow_pickle=False)
    applicability, support = [], []
    with threadpool_limits(limits=1):
        for item in models.itertuples(index=False):
            matrix = cache[item.arm]
            X = matrix[rows]
            model = PLSRegression(n_components=int(item.n_components), scale=False,
                                  max_iter=1000, tol=1e-10).fit(X, source.TOR_EC_loading_ug)
            scores = model.transform(X)
            variance = scores.var(axis=0, ddof=1)
            if np.any(variance <= 0):
                raise ValueError("PLS score support has a zero-variance component")
            source_t2 = np.sum(scores**2 / variance, axis=1)
            source_q = np.mean((X-model._x_mean-scores@model.x_loadings_.T)**2, axis=1)
            t95, q95 = np.quantile(source_t2, .95), np.quantile(source_q, .95)
            target_matrix = matrix[len(original):]
            for media, indices in groups.items():
                observed = target_matrix[indices].mean(axis=0)[None, :]
                target_score = model.transform(observed)
                t2 = float(np.sum(target_score**2 / variance))
                q = float(np.mean((observed-model._x_mean-target_score@model.x_loadings_.T)**2))
                row = scans.iloc[indices[0]]
                applicability.append({"arm": item.arm, "MediaId": media,
                    "base_filter_id": row.base_filter_id, "ExternalFilterType": row.ExternalFilterType,
                    "lot_group": row.lot_group, "n_components": item.n_components,
                    "source_t2_p95": t95, "source_q_p95": q95,
                    "target_t2": t2, "target_q": q,
                    "t2_over_source_p95": t2/t95, "q_over_source_p95": q/q95,
                    "outside_t2_p95": t2 > t95, "outside_q_p95": q > q95,
                    "outside_either_p95": t2 > t95 or q > q95})
    table = pd.DataFrame(applicability)
    table.to_csv(folder / "target_source_support.csv", index=False)
    for (arm, kind), part in table.groupby(["arm", "ExternalFilterType"], sort=False):
        support.append({"arm": arm, "ExternalFilterType": kind, "n": len(part),
                        "outside_t2_p95_n": int(part.outside_t2_p95.sum()),
                        "outside_q_p95_n": int(part.outside_q_p95.sum()),
                        "outside_either_p95_n": int(part.outside_either_p95.sum()),
                        "median_t2_over_source_p95": part.t2_over_source_p95.median(),
                        "median_q_over_source_p95": part.q_over_source_p95.median()})
    pd.DataFrame(support).to_csv(folder / "target_source_support_summary.csv", index=False)
    # The native blank controls found that finite arPLS output can still hit
    # its iteration limit. Record that numerical status for these exact source
    # and target inputs; do not silently call the fixed50-iteration arm converged.
    from pybaselines import Baseline
    from neutral_baseline import neutral_grid
    arrays_path = args.ftir_dir / "apps/apps_data.npz"
    expected = next(item["sha256"] for item in manifest["sources"] if item["name"] == "apps_data.npz")
    if sha(arrays_path) != expected:
        raise ValueError("The source archive changed after primary calibration")
    wn = np.load(arrays_path, allow_pickle=False)["wn"].astype(float)
    grid, indices = neutral_grid(wn)
    region = (grid > 1425) & (grid < 4000)
    fitter = Baseline(x_data=grid)
    convergence = []
    max_correction_error = 0.0
    for i, row in enumerate(cache["raw_full"]):
        z, info = fitter.pspline_arpls(row[indices], lam=1e6)
        history = np.asarray(info["tol_history"], float)
        correction = (row[indices]-z)[region][::-1]
        max_correction_error = max(max_correction_error,
            float(np.max(np.abs(correction-cache["neutral_arpls"][i]))))
        if i < len(original):
            record = {"population": "source", "analysis_id": int(original.iloc[i].AnalysisId),
                      "in_primary_source": not bool(original.iloc[i].is_excluded),
                      "base_filter_id": str(original.iloc[i].FilterId), "type": "source_PM"}
        else:
            scan = scans.iloc[i-len(original)]
            record = {"population": "target", "analysis_id": int(scan.SampleAnalysisId),
                      "in_primary_source": False, "base_filter_id": scan.base_filter_id,
                      "type": scan.ExternalFilterType}
        record.update({"iterations": len(history), "last_tolerance": float(history[-1]),
                       "converged": bool(np.isfinite(history[-1]) and history[-1] <= 1e-3)})
        convergence.append(record)
    if max_correction_error > 1e-12:
        raise ValueError("Convergence audit changed the neutral calibration preprocessing")
    convergence = pd.DataFrame(convergence)
    convergence.to_csv(folder / "neutral_default_convergence.csv", index=False)
    receipt = {"workflow": Path(__file__).name, "workflow_sha256": sha(Path(__file__)),
               "source_run_signature": manifest["signature"],
               "checks": ["Same outer site-fold membership verified for all905 common filters and all11 arms",
                          "37 unique physical field blanks per arm", "Original906 source rows remain in audit ledger"],
               "support_scope": "PLS score T2 and reconstruction Q relative to fitted905-source empirical95th percentiles; descriptive support, not a calibrated probability or targetEC validation",
               "blank_scope": "Field-blank loading predictions from models trained only on existing source TOR; no volume conversion or new blank labels added to training",
               "neutral_default_convergence": {
                   "source905_nonconverged": int((convergence.in_primary_source & ~convergence.converged).sum()),
                   "target319_nonconverged": int((convergence.population.eq("target") & ~convergence.converged).sum()),
                   "max_difference_from_primary_corrections": max_correction_error,
                   "scope": "Neutral calibration arm is the fixed repository default max_iter50; finite output is not proof of optimization convergence."}}
    (folder / "diagnostics_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(pd.DataFrame(blank_rows)[["arm", "n_physical_blanks", "median", "rmse_from_nominal_zero"]].to_string(index=False))
    print(pd.DataFrame(support).loc[lambda d: d.ExternalFilterType.eq("PM2.5")].to_string(index=False))


if __name__ == "__main__":
    main()
