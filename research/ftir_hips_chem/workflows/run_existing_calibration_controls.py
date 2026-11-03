"""Matched EC calibration controls using existing IMPROVE and Addis spectra.

This deliberately reconstructs the 906-filter historical EC calibration cohort,
not the later rank-70 VIBES full pool. Every preprocessing arm is recalibrated
against the same thermal-optical EC loadings. No optical target is used for
fitting, preprocessing choices, component selection, or source cross-validation.

Inputs are supplied explicitly; raw source files and download URLs are not copied
into the repository. Optional pybaselines dependencies may be supplied on
PYTHONPATH. Run with ``uv run`` and the repository's scientific environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from airspec_baseline import airspec_baseline_matrix  # noqa: E402
from neutral_baseline import neutral_baseline_matrix  # noqa: E402
from outliers import get_clean_data  # noqa: E402
from pls_transfer import (  # noqa: E402
    component_cv_curve,
    load_current_pls_model,
    select_first_major_minimum,
)
from portable_cohort_screen import load_public_chemistry, optical_pair_ledger  # noqa: E402
from run_native_addis_baseline_audit import load_native  # noqa: E402

SEED = 20261103


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score(y, prediction) -> dict:
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    error = prediction - y
    variance = np.sum((y - np.mean(y)) ** 2)
    return {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "median_residual": float(np.median(error)),
        "pearson_r2": float(np.corrcoef(y, prediction)[0, 1] ** 2)
        if len(y) > 2 and np.std(y) > 0 and np.std(prediction) > 0 else np.nan,
        "predictive_r2": float(1 - np.sum(error**2) / variance)
        if variance > 0 else np.nan,
        "negative_predictions": int(np.sum(prediction < 0)),
    }


def write_table(frame, output, name):
    pd.DataFrame(frame).to_csv(output / name, index=False)


def source_cohort(folder):
    model = load_current_pls_model(folder, "EC")
    meta = pd.read_csv(folder / "notebook/extracted/EC_samples.csv")
    if meta.AnalysisId.duplicated().any() or len(np.unique(model.filter_ids)) != len(meta):
        raise ValueError("Source calibration must contain unique analyses and physical filters")
    np.testing.assert_array_equal(meta.AnalysisId, model.analysis_ids)
    meta["FilterId"] = model.filter_ids
    meta = meta.rename(columns={"measured": "TOR_EC_loading_ug"})
    oc = pd.read_csv(folder / "notebook/extracted/OC_samples.csv")
    meta = meta.merge(oc[["AnalysisId", "measured"]].rename(
        columns={"measured": "TOR_OC_loading_ug"}),
        on="AnalysisId", how="left", validate="one_to_one", sort=False)
    meta["OC_EC_ratio"] = meta.TOR_OC_loading_ug / meta.TOR_EC_loading_ug
    catalog = pd.read_csv(folder / "local_db/tables/ftir_catalog.csv")
    if catalog.AnalysisId.duplicated().any():
        raise ValueError("Source catalog analysis IDs must be unique")
    meta = meta.merge(catalog[["AnalysisId", "Site", "FilterId", "LotNumber", "Status", "AnalysisQcCode"]],
                      on="AnalysisId", how="left", suffixes=("", "_catalog"),
                      validate="one_to_one", sort=False)
    known = meta.Site_catalog.notna()
    if not (meta.loc[known, "Site"] == meta.loc[known, "Site_catalog"]).all():
        raise ValueError("Source catalog site mismatch")
    if not (meta.loc[known, "FilterId"] == meta.loc[known, "FilterId_catalog"]).all():
        raise ValueError("Source catalog filter mismatch")
    meta["catalog_missing"] = ~known
    meta["LotNumber"] = meta.LotNumber.fillna("unknown")
    meta["SampleDate"] = pd.to_datetime(meta.SampleDate).dt.strftime("%Y-%m-%d")
    meta["OC_EC_stratum"] = pd.cut(meta.OC_EC_ratio, [-np.inf, 5, 10, np.inf],
                                    labels=["below_5", "5_to_10", "above_10"], right=False)
    meta["EC_loading_stratum"] = pd.cut(meta.TOR_EC_loading_ug, [-np.inf, 10, 30, np.inf],
                                        labels=["below_10", "10_to_30", "above_30"], right=False)
    arrays = np.load(folder / "apps/apps_data.npz", allow_pickle=False)
    ledger = pd.DataFrame({"AnalysisId": arrays["EC_id"], "FilterId": arrays["EC_fid"],
                           "Site": arrays["EC_site"]})
    ledger["in_official_EC_calibration"] = ledger.AnalysisId.isin(meta.AnalysisId)
    ledger["selection_reason"] = np.where(ledger.in_official_EC_calibration,
        "Listed in historical EC_samples.csv", "Absent from historical EC_samples.csv; not newly excluded")
    return model, meta, ledger


def preprocessing(wn, raw, *, jobs, cache_path, cache_signature):
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        if str(cache["signature"].item()) == cache_signature:
            return {key: cache[key] for key in cache.files if key != "signature"}
    shared = (wn > 1425) & (wn < 4000)
    out = {"raw_full": raw.copy(), "raw_shared": raw[:, shared].copy()}
    for df2 in (4, 3, 6):
        print(f"Preprocessing spline df6/{df2}: {len(raw)} spectra", flush=True)
        baseline, corrected = airspec_baseline_matrix(wn, raw, df1=6, df2=df2, n_jobs=jobs)
        out[f"spline6_{df2}"] = corrected[:, shared]
        if df2 == 4:
            out["removed_continuum"] = baseline[:, shared]
    print(f"Preprocessing neutral arPLS: {len(raw)} spectra", flush=True)
    neutral_wn, neutral = neutral_baseline_matrix(wn, raw, lam=1e6)
    neutral_shared = (neutral_wn > 1425) & (neutral_wn < 4000)
    np.testing.assert_allclose(neutral_wn[neutral_shared][::-1], wn[shared], atol=0, rtol=0)
    out["neutral_arpls"] = neutral[:, neutral_shared][:, ::-1]
    shared_wn = wn[shared]
    band = (shared_wn >= 1580) & (shared_wn <= 1650)
    out["raw_without_1600"] = out["raw_shared"][:, ~band]
    out["spline_without_1600"] = out["spline6_4"][:, ~band]
    out["spline_1600_only"] = out["spline6_4"][:, band]
    # An exactly specified shape control; the common grid is uniform to float32
    # precision and no test label chooses the window or polynomial degree.
    out["second_derivative"] = savgol_filter(raw, 31, 3, deriv=2,
        delta=float(np.median(np.abs(np.diff(wn)))), axis=1)[:, shared]
    if any(not np.isfinite(values).all() for values in out.values()):
        raise ValueError("A preprocessing arm returned nonfinite analyzed values")
    np.savez_compressed(cache_path, signature=np.asarray(cache_signature), **out)
    return out


def fit_inner(X, y, groups, max_components):
    curve = component_cv_curve(X, y, range(1, max_components + 1), groups=groups,
                               n_splits=5, random_state=42)
    selected, curve = select_first_major_minimum(curve)
    model = PLSRegression(n_components=selected, scale=False, max_iter=1000, tol=1e-10).fit(X, y)
    return model, selected, curve


def source_validation(arms, meta, output, max_components):
    y = meta.TOR_EC_loading_ug.to_numpy(float)
    groups = meta.Site.to_numpy(str)
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(y, y, groups))
    fold_for_row = np.full(len(meta), -1)
    for fold, (train, test) in enumerate(splits):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("A source site leaked across an outer fold")
        fold_for_row[test] = fold
    source_ledger = meta.copy()
    source_ledger["outer_fold"] = fold_for_row
    write_table(source_ledger, output, "source_cohort.csv")
    metrics, predictions, curves, final_models, selected_rows = [], [], [], {}, []
    chronology = pd.to_datetime(meta.SampleDate).lt("2022-11-01").to_numpy()
    for name, X in arms.items():
        print(f"Validating {name}: {X.shape}", flush=True)
        oof = np.full(len(y), np.nan)
        for fold, (train, test) in enumerate(splits):
            model, k, curve = fit_inner(X[train], y[train], groups[train], max_components)
            prediction = model.predict(X[test]).ravel()
            oof[test] = prediction
            metrics.append({"arm": name, "evaluation": "site_grouped_outer_fold", "fold": fold,
                            "n_sites": len(np.unique(groups[test])), "n_components": k,
                            "channels": X.shape[1], "units": "ug/filter", **score(y[test], prediction)})
            curves.append(curve.assign(arm=name, evaluation="site_grouped", fold=fold))
        if not np.isfinite(oof).all():
            raise AssertionError("Every source filter must have exactly one outer prediction")
        metrics.append({"arm": name, "evaluation": "site_grouped_pooled", "fold": "all",
                        "n_sites": len(np.unique(groups)), "n_components": np.nan,
                        "channels": X.shape[1], "units": "ug/filter", **score(y, oof)})
        prediction_frame = meta.copy()
        prediction_frame["arm"] = name
        prediction_frame["fold"] = fold_for_row
        prediction_frame["prediction_loading_ug"] = oof
        prediction_frame["residual_loading_ug"] = oof - y
        predictions.append(prediction_frame)
        # Secondary time-block test shares the same predeclared split for all
        # arms; it does not hold sites out and is labelled separately.
        model, k, curve = fit_inner(X[chronology], y[chronology], groups[chronology], max_components)
        pred_chrono = model.predict(X[~chronology]).ravel()
        metrics.append({"arm": name, "evaluation": "time_block_2022_11_01", "fold": f"later_{int((~chronology).sum())}",
                        "n_sites": len(np.unique(groups[~chronology])), "n_components": k,
                        "channels": X.shape[1], "units": "ug/filter", **score(y[~chronology], pred_chrono)})
        curves.append(curve.assign(arm=name, evaluation="time_block", fold="train_before_2022_11_01"))
        part = meta.loc[~chronology].copy()
        part["arm"] = name
        part["evaluation"] = "time_block_2022_11_01"
        part["prediction_loading_ug"] = pred_chrono
        part["residual_loading_ug"] = pred_chrono - y[~chronology]
        write_table(part, output, f"time_block_predictions_{name}.csv")
        final, k, curve = fit_inner(X, y, groups, max_components)
        final_models[name] = final
        selected_rows.append({"arm": name, "n_components": k, "n_source_filters": len(y),
                              "n_source_sites": len(np.unique(groups)), "channels": X.shape[1]})
        curves.append(curve.assign(arm=name, evaluation="all_source_for_target", fold="all"))
        write_table(metrics, output, "metrics.csv")
        write_table(pd.concat(predictions, ignore_index=True), output, "heldout_predictions.csv")
        write_table(pd.concat(curves, ignore_index=True), output, "component_selection.csv")
        write_table(selected_rows, output, "final_source_models.csv")
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), final_models


def paired_site_bootstrap(predictions, output, repeats=2000):
    baseline = predictions.loc[predictions.arm.eq("spline6_4")].set_index("AnalysisId")
    sites = sorted(baseline.Site.unique())
    counts = baseline.groupby("Site").size().reindex(sites).to_numpy(float)
    base_sse = baseline.residual_loading_ug.pow(2).groupby(baseline.Site).sum().reindex(sites).to_numpy(float)
    base_sae = baseline.residual_loading_ug.abs().groupby(baseline.Site).sum().reindex(sites).to_numpy(float)
    rng = np.random.default_rng(SEED)
    weights = np.zeros((repeats, len(sites)))
    for i in range(repeats):
        weights[i] = np.bincount(rng.integers(0, len(sites), size=len(sites)), minlength=len(sites))
    denom = weights @ counts
    rows = []
    for arm, part in predictions.groupby("arm", sort=False):
        part = part.set_index("AnalysisId").loc[baseline.index]
        np.testing.assert_array_equal(part.Site, baseline.Site)
        sse = part.residual_loading_ug.pow(2).groupby(part.Site).sum().reindex(sites).to_numpy(float)
        sae = part.residual_loading_ug.abs().groupby(part.Site).sum().reindex(sites).to_numpy(float)
        rmse_delta = np.sqrt(weights @ sse / denom) - np.sqrt(weights @ base_sse / denom)
        mae_delta = weights @ (sae - base_sae) / denom
        rows.append({"arm": arm, "reference_arm": "spline6_4", "n_filters": len(part),
                     "n_sites": len(sites), "bootstrap_repeats": repeats,
                     "delta_rmse": np.sqrt(sse.sum()/counts.sum())-np.sqrt(base_sse.sum()/counts.sum()),
                     "delta_rmse_ci025": np.quantile(rmse_delta, .025),
                     "delta_rmse_ci975": np.quantile(rmse_delta, .975),
                     "delta_mae": (sae.sum()-base_sae.sum())/counts.sum(),
                     "delta_mae_ci025": np.quantile(mae_delta, .025),
                     "delta_mae_ci975": np.quantile(mae_delta, .975),
                     "units": "ug/filter"})
    write_table(rows, output, "paired_site_bootstrap.csv")


def target_predictions(arms, n_source, scans, models, output):
    predictions = []
    groups = scans.groupby("MediaId", sort=True).indices
    for arm, model in models.items():
        transformed = arms[arm][n_source:]
        for media_id, indices in groups.items():
            row = scans.iloc[indices[0]].to_dict()
            row.pop("_merge", None)
            row["arm"] = arm
            row["n_scans"] = len(indices)
            row["aggregation_sensitivity"] = "all_original_scans"
            row["prediction_loading_ug"] = float(model.predict(transformed[indices].mean(axis=0)[None, :]).ravel()[0])
            volume = row["SampleVolume_m3"]
            row["prediction_ugm3"] = row["prediction_loading_ug"] / volume if np.isfinite(volume) and volume > 0 else np.nan
            row["volume_valid"] = bool(np.isfinite(volume) and volume > 0)
            row["prior_problem_filter"] = row["base_filter_id"] in ("ETAD-0037", "ETAD-0139", "ETAD-0158")
            predictions.append(row)
            if row["base_filter_id"] == "ETAD-0139":
                retained = [i for i in indices if int(scans.iloc[i].SampleAnalysisId) == 1668]
                if len(retained) != 1:
                    raise ValueError("ETAD-0139 conditional scan1668 sensitivity is not identifiable")
                alternative = row.copy()
                alternative["aggregation_sensitivity"] = "conditional_scan1668_only"
                alternative["n_scans"] = 1
                alternative["prediction_loading_ug"] = float(model.predict(transformed[retained]).ravel()[0])
                alternative["prediction_ugm3"] = alternative["prediction_loading_ug"] / volume
                predictions.append(alternative)
    frame = pd.DataFrame(predictions)
    write_table(frame, output, "target_predictions.csv")
    chem = load_public_chemistry(ROOT)
    optical = optical_pair_ledger(chem)
    optical = optical.loc[optical.code.eq("ETAD"), [
        "id", "date", "pair_eligible", "ec_ftir_ugm3", "optical_bc_ugm3", "optical_source",
    ]].rename(columns={"id": "base_filter_id", "date": "current_date",
                       "pair_eligible": "current_measurement_pair_eligible",
                       "ec_ftir_ugm3": "current_ec_ftir_ugm3",
                       "optical_bc_ugm3": "current_optical_bc_ugm3",
                       "optical_source": "current_optical_source"})
    optical["current_date"] = pd.to_datetime(optical.current_date)
    # Only optical pairs already in the current public ledger are joined. The
    # response used for training remains the source thermal EC loading above.
    joined = frame.merge(optical, on="base_filter_id", how="left", validate="many_to_one", suffixes=("", "_current"))
    joined["current_pair_eligible"] = (joined.current_measurement_pair_eligible.eq(True)
        & joined.ExternalFilterType.eq("PM2.5") & joined.volume_valid
        & joined.native_date.eq(joined.current_date))
    joined["experiment_ftir_minus_hips_ugm3"] = (
        joined.prediction_ugm3 - joined.current_optical_bc_ugm3).where(joined.current_pair_eligible)
    write_table(joined, output, "target_optical_pairs.csv")
    rows = []
    for arm, part in joined.loc[joined.aggregation_sensitivity.eq("all_original_scans")].groupby("arm", sort=False):
        for cohort in ("current_all_available", "direct_hips_only", "current_without_prior_problem_filters"):
            eligible = part.current_pair_eligible & part.prediction_ugm3.notna()
            if cohort == "direct_hips_only":
                eligible &= part.current_optical_source.eq("direct_hips_fabs")
            elif cohort == "current_without_prior_problem_filters":
                eligible &= ~part.prior_problem_filter
            selected = part.loc[eligible]
            stats = score(selected.current_optical_bc_ugm3, selected.prediction_ugm3)
            rows.append({"arm": arm, "cohort": cohort, "units": "ug/m3",
                         "scope": "optical agreement diagnostic; not target EC accuracy",
                         "residual_sign": "experiment_FTIR_minus_HIPS", **stats})
    write_table(rows, output, "target_optical_summary.csv")
    return frame, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ftir-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "research/ftir_hips_chem/output/tables/calibration_existing_data_20261103")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--max-components", type=int, default=40)
    parser.add_argument("--source-policy", choices=("documented_qc", "all_original"), default="documented_qc",
                        help="Primary excludes the already documented TONT1 offset scan in every arm; all originals remain in a flagged ledger")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    source, meta, ledger = source_cohort(args.ftir_dir)
    # This is an existing source exclusion recorded on2026-10-20, not a
    # post-hoc rule discovered from the present method comparison. Apply the
    # same flagged source cohort to all preprocessing arms.
    meta["prior_documented_source_flag"] = (meta.AnalysisId.eq(164549)
        & meta.FilterId.eq(2008808) & meta.Site.eq("TONT1") & meta.SampleDate.eq("2022-11-19"))
    if int(meta.prior_documented_source_flag.sum()) != 1:
        raise ValueError("The historical TONT1 source exclusion does not identify exactly one scan")
    meta["is_excluded"] = meta.prior_documented_source_flag & (args.source_policy == "documented_qc")
    meta["exclusion_reason"] = np.where(meta.is_excluded,
        "Previously documented +3 absorbance offset: docs/spartan-ftir-ec-recreation.md,20Oct2026; held from all matched arms", "")
    write_table(meta, output, "source_original906_with_flags.csv")
    model_meta = get_clean_data(meta)
    source_rows = model_meta.index.to_numpy(int)
    model_meta = model_meta.reset_index(drop=True)
    ledger["prior_documented_source_flag"] = ledger.AnalysisId.eq(164549)
    ledger["in_this_matched_experiment"] = ledger.AnalysisId.isin(model_meta.AnalysisId)
    write_table(ledger, output, "archive_selection_ledger.csv")
    scans, native_wn, native_raw, _, _ = load_native(args.native_dir)
    if len(source.wavenumbers) != len(native_wn) or np.max(np.abs(source.wavenumbers-native_wn)) > .001:
        raise ValueError("Source/target native grids differ by more than float32 rounding")
    # Source NPZ uses float32 wavenumbers; align target to that exact grid before
    # applying every baseline. Maximum displacement is recorded below.
    target_raw = np.vstack([np.interp(source.wavenumbers[::-1], native_wn[::-1], row[::-1])[::-1]
                            for row in native_raw])
    combined = np.vstack([source.X, target_raw])
    input_files = [args.ftir_dir / "apps/apps_data.npz",
                   *[args.ftir_dir / "notebook/extracted" / name for name in (
                       "EC_samples.csv", "EC_stats_chosen.csv", "EC_fitted_long.csv", "OC_samples.csv")],
                   args.ftir_dir / "local_db/tables/ftir_catalog.csv",
                   *[args.native_dir / name for name in ("ETAD_FTIR_spectra.csv", "ETAD_metadata.csv",
                                                       "etad_spectra_lotmap.csv", "spectra_baselined_AIRSPEC.csv")]]
    sources = [{"name": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for p in input_files]
    signature = hashlib.sha256(json.dumps({"sources": sources, "workflow": sha(Path(__file__)),
        "airspec": sha(ROOT / "research/ftir_ec_phase3/scripts/airspec_baseline.py"),
        "neutral": sha(ROOT / "research/ftir_ec_phase3/scripts/neutral_baseline.py")}, sort_keys=True).encode()).hexdigest()
    with threadpool_limits(limits=1):
        all_arms = preprocessing(source.wavenumbers, combined, jobs=args.jobs,
                                 cache_path=output / "preprocessing_cache.npz", cache_signature=signature)
        source_arms = {name: values[source_rows] for name, values in all_arms.items()}
        metrics, heldout, models = source_validation(source_arms, model_meta, output, args.max_components)
        paired_site_bootstrap(heldout, output)
        target, optical = target_predictions(all_arms, len(meta), scans, models, output)
    strata = []
    for (arm, key), part in heldout.groupby(["arm", "OC_EC_stratum"], observed=True):
        strata.append({"arm": arm, "stratum_type": "OC_EC_ratio", "stratum": str(key),
                       "n_sites": part.Site.nunique(), **score(part.TOR_EC_loading_ug, part.prediction_loading_ug)})
    for (arm, key), part in heldout.groupby(["arm", "EC_loading_stratum"], observed=True):
        strata.append({"arm": arm, "stratum_type": "EC_loading_ug", "stratum": str(key),
                       "n_sites": part.Site.nunique(), **score(part.TOR_EC_loading_ug, part.prediction_loading_ug)})
    write_table(strata, output, "source_stratum_metrics.csv")
    summary = {
        "schema_version": 1, "run_date": "2026-11-03", "source_version": "historical_smoke906_EC_samples_export_2026_07_07",
        "source_policy": args.source_policy,
        "sources": sources, "workflow_sha256": sha(Path(__file__)), "signature": signature,
        "source_filters": len(model_meta), "official_source_filters": len(meta),
        "archive_filters": len(ledger), "source_sites": model_meta.Site.nunique(),
        "source_dates": [meta.SampleDate.min(), meta.SampleDate.max()],
        "source_lots": model_meta.LotNumber.value_counts(dropna=False).to_dict(),
        "source_R_reconstruction_max_abs_error_ug_filter": source.r_prediction_max_abs_error,
        "source_R_components": source.chosen_n_components,
        "target_scans": len(scans), "target_physical_filters": scans.MediaId.nunique(),
        "target_grid_max_displacement_cm1": float(np.max(np.abs(source.wavenumbers-native_wn))),
        "outer_cv": {"n_splits": 5, "group": "Site", "shuffle": True, "seed": SEED},
        "inner_cv": {"n_splits": 5, "group": "Site", "seed": 42,
                     "candidates": list(range(1, args.max_components+1)), "selection": "first local minimum within one SE of global minimum"},
        "time_block": {"cutoff": "2022-11-01",
                       "before_n": int(pd.to_datetime(model_meta.SampleDate).lt("2022-11-01").sum()),
                       "after_n": int(pd.to_datetime(model_meta.SampleDate).ge("2022-11-01").sum()),
                       "caveat": "Same predeclared dates for all arms; sites are not separated in this secondary check."},
        "neutral_arpls": {"lambda": 1e6, "pybaselines_version": "1.2.1", "grid": ">=700 excluding1100<wn<1300; model uses1425<wn<4000"},
        "limitations": [
            "Source906 is the historical smoke-selected calibration cohort, not the current full-pool VIBES rank70 calibration.",
            "Source catalog has887 lot251 records and19 unknown before the documented TONT1 flag; no source lot248 arm is identifiable here.",
            "Primary source policy flags the previously documented TONT1 offset scan in every arm; original906 results must be labelled separately.",
            "All metrics compare the same source folds, but preprocessing candidates are exploratory and not an untouched external validation.",
            "HIPS and Addis spectra do not select calibration components or minimize source errors.",
            "Addis experiment predictions diagnose transfer and optical agreement, not independent target EC accuracy.",
            "Original scans are retained; only0139 receives a separately labelled scan1668-only sensitivity.",
            "Negative spectra and negative predictions are retained. No calibrated production product is replaced.",
            "All-source fitted models and any training-domain projection are not held-out validation of the target domain.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(metrics.loc[metrics.evaluation.eq("site_grouped_pooled"),
                      ["arm", "n", "rmse", "mae", "bias", "pearson_r2", "predictive_r2"]].to_string(index=False))
    print(optical[["arm", "cohort", "n", "bias", "median_residual"]].to_string(index=False))


if __name__ == "__main__":
    main()
