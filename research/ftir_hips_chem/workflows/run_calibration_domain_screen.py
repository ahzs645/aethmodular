"""Project native raw Addis scans into one frozen historical EC PLS calibration.

This is an applicability diagnostic, not target EC validation or a replacement
EC product. Only the reviewed full-grid raw-absorbance exports are accepted.
No target optical or carbon labels enter fitting, thresholds, or projections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from outliers import apply_exclusion_flags
from pls_transfer import (
    load_current_pls_model, mahalanobis_distance_squared, project_scores,
    score_metric, spectral_q_residual,
)
from run_native_addis_baseline_audit import SOURCE_NAMES, load_native, safe_json


MODEL_ID = "historical_EC_906_filters_31_components_raw_2722"
RAW_REPRESENTATION = "raw_absorbance_full_grid"
# These are reviewed data-file hashes, not credentials. A changed calibration
# needs its representation/provenance reviewed before this workflow accepts it.
FROZEN_MODEL_HASHES = {
    "apps/apps_data.npz": "8b2bd5b70d2f00bbbf82f410cce37bb4e1ba3be318ba175525d2e65fbfeadd53",
    "notebook/extracted/EC_samples.csv": "603bf1e383c3844ce8f242e3a28374a3d8a416ceef57f7666e14fc8c857e4be4",
    "notebook/extracted/EC_stats_chosen.csv": "6d8fc58528b85444ef985a12fe4d0387874d16d04c661f613ad2d5c326701cbd",
    "notebook/extracted/EC_fitted_long.csv": "514f97ac1a2d9d9f01e366bd778246da51a7d221b972faff9f59f0656e16f761",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_model_hashes(actual, expected=FROZEN_MODEL_HASHES):
    if set(actual) != set(expected) or any(actual[name] != expected[name] for name in expected):
        raise ValueError("unreviewed calibration export: frozen model hashes differ")


def verify_raw_grid(source_wn, native_wn, source_X, native_X, *, representation=RAW_REPRESENTATION):
    """Accept exact channel identity at the exported float32 grid precision.

    No interpolation, channel truncation, reversal, normalization, derivatives,
    or baseline substitution is allowed. This checks the documented archived
    raw-absorbance contract; it cannot infer acquisition processing from shape.
    """
    if representation != RAW_REPRESENTATION:
        raise ValueError("only the reviewed raw-absorbance representation is accepted")
    source_wn, native_wn = np.asarray(source_wn, float), np.asarray(native_wn, float)
    if source_wn.shape != (2722,) or native_wn.shape != source_wn.shape:
        raise ValueError("both full grids must contain exactly 2722 channels")
    if not (np.isfinite(source_wn).all() and np.isfinite(native_wn).all()
            and (np.diff(source_wn) < 0).all() and (np.diff(native_wn) < 0).all()):
        raise ValueError("finite strictly descending source and native grids are required")
    if not np.array_equal(source_wn.astype(np.float32), native_wn.astype(np.float32)):
        raise ValueError("native grid is not identical at exported float32 precision")
    for values in (source_X, native_X):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != 2722 or not np.isfinite(values).all():
            raise ValueError("full-grid finite raw spectra are required")
    return {"grid_identity": "exact_after_float32_cast", "channels": 2722,
            "maximum_grid_rounding_difference_cm1": float(np.max(abs(source_wn-native_wn))),
            "interpolation": False, "normalization": False, "baseline_correction": False}


def aggregate_physical_targets(scans, raw):
    """Check identities first, then mean repeated raw scans per physical filter."""
    if scans.SampleAnalysisId.duplicated().any() or len(scans) != len(raw):
        raise ValueError("one aligned raw spectrum per unique scan is required")
    if (scans.groupby("MediaId").base_filter_id.nunique(dropna=False).max() != 1
            or scans.groupby("base_filter_id").MediaId.nunique(dropna=False).max() != 1):
        raise ValueError("physical MediaId and base_filter_id must be one-to-one")
    rows, matrix = [], []
    for _, indices in scans.groupby("MediaId", sort=True).indices.items():
        part = scans.iloc[indices]
        for column in ("base_filter_id", "ExternalFilterType", "lot_group", "native_date", "SampleVolume_m3"):
            if part[column].nunique(dropna=False) != 1:
                raise ValueError(f"conflicting repeated-filter metadata: {column}")
        row = part.iloc[0].drop(labels=["SampleAnalysisId"]).to_dict()
        row.update(n_scans=len(part), scan_ids="|".join(part.SampleAnalysisId.astype(str)))
        rows.append(row)
        matrix.append(np.asarray(raw)[indices].mean(axis=0))
    return pd.DataFrame(rows), np.vstack(matrix)


def domain_metrics(model, X):
    """Reuse the established PLS helpers and expose both distance conventions."""
    calibration_scores, inverse_ss = score_metric(model)
    if np.max(abs(calibration_scores.mean(axis=0))) > 1e-8:
        raise ValueError("training scores must be centered for the covariance scaling")
    n = len(calibration_scores)
    scores = project_scores(model, X)
    leverage = mahalanobis_distance_squared(scores, inverse_ss)
    return pd.DataFrame({
        "score_leverage": leverage,
        "score_mahalanobis_squared": (n-1)*leverage,
        "spectral_Q_sum_squared_absorbance": spectral_q_residual(model, X),
    })


def add_reference_flags(frame, source):
    """Descriptive training-reference percentiles, not validated error limits."""
    result = frame.copy()
    thresholds = {}
    pairs = [("D2", "score_mahalanobis_squared"), ("Q", "spectral_Q_sum_squared_absorbance")]
    for short, column in pairs:
        ref = source[column].to_numpy(float)
        result[f"{short}_training_percentile"] = np.searchsorted(np.sort(ref), result[column], side="right") / len(ref)
        for pct in (95, 99):
            threshold = float(np.percentile(ref, pct))
            thresholds[f"{short}_p{pct}"] = threshold
            result[f"{short}_above_training_p{pct}"] = result[column] > threshold
    for pct in (95, 99):
        result[f"either_above_training_p{pct}"] = result[f"D2_above_training_p{pct}"] | result[f"Q_above_training_p{pct}"]
    return result, thresholds


def summarize(frame, *, label):
    rows = []
    for kind, group in frame.groupby("ExternalFilterType", sort=True):
        row = {"unit": label, "filter_type": kind, "n": len(group),
               "canonical_exclusion_flags": int(group.is_excluded.sum())}
        for short, column in [("D2", "score_mahalanobis_squared"), ("Q", "spectral_Q_sum_squared_absorbance")]:
            row[f"median_{short}"] = float(group[column].median())
            row[f"max_{short}"] = float(group[column].max())
        for pct in (95, 99):
            for short in ("D2", "Q", "either"):
                key = f"{short}_above_training_p{pct}"
                row[key+"_n"] = int(group[key].sum())
                row[key+"_fraction"] = float(group[key].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run(ftir_dir, native_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    model_hashes = {name: digest(ftir_dir/name) for name in FROZEN_MODEL_HASHES}
    verify_model_hashes(model_hashes)
    native_receipts = {name: digest(native_dir/name) for name in SOURCE_NAMES}
    rmsep_path = ftir_dir / "notebook/extracted/EC_rmsep.csv"
    cv = pd.read_csv(rmsep_path) if rmsep_path.exists() else None
    cv_hash = digest(rmsep_path) if cv is not None else None
    provenance_script = ROOT / "research/ftir_hips_chem/workflows/build_pls_transfer_notebooks.py"
    provenance_text = provenance_script.read_text()
    if "raw 2722-point spectra share a grid" not in provenance_text:
        raise ValueError("archived raw-grid preprocessing contract is missing; review before projection")
    code_files = [Path(__file__), ROOT / "research/ftir_hips_chem/scripts/pls_transfer.py",
                  ROOT / "research/ftir_hips_chem/workflows/run_native_addis_baseline_audit.py", provenance_script]
    code_receipts = {str(p.relative_to(ROOT)): digest(p) for p in code_files}
    with threadpool_limits(limits=1):
        frozen = load_current_pls_model(ftir_dir, "EC")
        if (len(frozen.y), frozen.chosen_n_components, len(frozen.wavenumbers)) != (906, 31, 2722):
            raise ValueError("historical model dimensions differ from reviewed contract")
        if len(np.unique(frozen.analysis_ids)) != 906 or len(np.unique(frozen.filter_ids)) != 906:
            raise ValueError("calibration analyses and physical filters must be unique")
        scans, native_wn, raw, _, _ = load_native(native_dir)
        if set(frozen.sites) & set(scans.base_filter_id.str.split("-").str[0]):
            raise ValueError("a target site appears in the calibration source population")
        grid_receipt = verify_raw_grid(frozen.wavenumbers, native_wn, frozen.X, raw)
        physical, mean_raw = aggregate_physical_targets(scans, raw)
        source = domain_metrics(frozen.model, frozen.X)
        source["AnalysisId"], source["source_filter_id"], source["Site"] = frozen.analysis_ids, frozen.filter_ids, frozen.sites
        source["reference_response"] = frozen.y
        source["fitted_response"] = frozen.model.predict(frozen.X).ravel()
        source["training_residual"] = source.fitted_response-source.reference_response
        source["evaluation_role"] = "in_sample_calibration_fit"
        scan_values = domain_metrics(frozen.model, raw)
        physical_values = domain_metrics(frozen.model, mean_raw)
    source, thresholds = add_reference_flags(source, source)
    scan_values, _ = add_reference_flags(scan_values, source)
    physical_values, _ = add_reference_flags(physical_values, source)
    scans = pd.concat([scans.reset_index(drop=True), scan_values], axis=1)
    physical = pd.concat([physical.reset_index(drop=True), physical_values], axis=1)
    for frame in (scans, physical):
        frame["date"] = frame.native_date
    scans = apply_exclusion_flags(scans, "Addis_Ababa", date_tolerance_days=0)
    physical = apply_exclusion_flags(physical, "Addis_Ababa", date_tolerance_days=0)
    summaries = pd.concat([summarize(physical, label="physical_filter_mean"),
                           summarize(scans, label="scan")], ignore_index=True)
    scan_ranges = scans.groupby("base_filter_id").agg(
        scan_min_D2=("score_mahalanobis_squared", "min"),
        scan_max_D2=("score_mahalanobis_squared", "max"),
        scan_min_Q=("spectral_Q_sum_squared_absorbance", "min"),
        scan_max_Q=("spectral_Q_sum_squared_absorbance", "max"),
    ).reset_index()
    shortlist_parts = []
    for label, column in [("Q", "spectral_Q_sum_squared_absorbance"), ("D2", "score_mahalanobis_squared")]:
        selected = physical.loc[physical.ExternalFilterType.eq("PM2.5")].sort_values(column, ascending=False).head(5).copy()
        selected["ranking_metric"], selected["rank"] = label, np.arange(1, len(selected)+1)
        shortlist_parts.append(selected)
    shortlist = pd.concat(shortlist_parts, ignore_index=True).merge(
        scan_ranges, on="base_filter_id", how="left", validate="many_to_one")
    source.to_csv(output / "source_training_domain.csv", index=False)
    scans.to_csv(output / "native_scan_domain.csv", index=False)
    physical.to_csv(output / "native_physical_filter_domain.csv", index=False)
    summaries.to_csv(output / "domain_summary.csv", index=False)
    shortlist.to_csv(output / "native_pm25_domain_shortlist.csv", index=False)
    pd.DataFrame([thresholds]).to_csv(output / "training_reference_thresholds.csv", index=False)
    source_meta = pd.read_csv(ftir_dir / "notebook/extracted/EC_samples.csv")
    exported_cv = None
    if cv is not None:
        selected = cv.loc[cv.ncomp.eq(frozen.chosen_n_components)]
        if len(selected) != 1:
            raise ValueError("exported RMSEP table lacks a unique selected-component row")
        selected.assign(evaluation_role="historical_aggregate_CV_fold_scheme_not_verified").to_csv(
            output / "historical_exported_CV_summary.csv", index=False)
        exported_cv = {"sha256": cv_hash, **selected.iloc[0].to_dict(),
                       "warning": "Historical aggregate only; no per-filter held-out predictions or fold membership supplied"}
    residual = source.training_residual.to_numpy(float)
    manifest = {
        "model_id": MODEL_ID, "species": "EC", "representation": RAW_REPRESENTATION,
        "source_filters": len(source), "source_sites": len(np.unique(frozen.sites)),
        "source_date_start": str(pd.to_datetime(source_meta.SampleDate).min().date()),
        "source_date_end": str(pd.to_datetime(source_meta.SampleDate).max().date()),
        "components": frozen.chosen_n_components, "scale": False,
        "calibration_response_units": "original exported response; archived workflow treats this as ug/filter loading",
        "R_reconstruction_max_absolute_difference": frozen.r_prediction_max_abs_error,
        "R_reconstruction_median_absolute_difference": frozen.r_prediction_median_abs_error,
        "source_training_RMSE": float(np.sqrt(np.mean(residual**2))),
        "source_training_MAE": float(np.mean(abs(residual))), "source_training_bias": float(np.mean(residual)),
        "heldout_source_residuals_available": False, "historical_aggregate_cv": exported_cv,
        "distance_convention": "leverage=t'(T'T)^(-1)t; covariance-normalized D2=(n_train-1)*leverage",
        "covariance_multiplier": len(source)-1,
        "Q_convention": "sum of squared full-grid spectral reconstruction residuals; absorbance squared",
        "reference_rule": "Empirical in-sample source percentiles; not calibrated error probabilities or validated acceptance thresholds",
        "thresholds": thresholds, "grid_verification": grid_receipt,
        "native_scans": len(scans), "native_physical_filters": len(physical),
        "repeated_physical_filters": int(physical.n_scans.gt(1).sum()),
        "source_filter_grouping": "906 unique calibration physical filters; native repeats never enter source fit/reference",
        "target_aggregation": "All original scans retained; checked physical identities before arithmetic mean of raw spectra",
        "claims": "Applicability to a historical model only; no target EC accuracy, no equivalence to current public EC, no target EC concentrations generated",
        "receipt_snapshot": "start_of_run",
        "model_inputs": model_hashes, "native_inputs": native_receipts, "code": code_receipts,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    (output / "manifest.json").write_text(json.dumps(safe_json(manifest), indent=2)+"\n")
    print(json.dumps(safe_json({k:v for k,v in manifest.items() if k not in ("model_inputs","native_inputs","code")}), indent=2))
    print(summaries.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ftir-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/ftir_hips_chem/output/tables/calibration_domain_20261102")
    args = parser.parse_args()
    run(args.ftir_dir, args.native_dir, args.output)
