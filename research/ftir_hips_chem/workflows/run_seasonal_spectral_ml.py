"""Whole-spectrum seasonal diagnostics using existing physical Addis filters.

Two distinct questions are evaluated: season classification from spectra and
prediction of the existing FTIR-minus-HIPS disagreement. Neither is an EC
accuracy validation or a chemical/source apportionment. Models and spectral
intervals are fixed before reading held-out scores. Generated inputs stay local.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import savgol_filter
from sklearn.dummy import DummyClassifier, DummyRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))

from config import MAC_VALUE  # noqa: E402
from seasonal_spectral_ml import (  # noqa: E402
    build_model, evaluate_folds, grouped_region_permutation,
    leave_region_out_refit, make_cycle_folds, make_feature_frame,
    season_labels, spectral_regions,
)

TABLES = ROOT / "research/ftir_hips_chem/output/tables"
SEED = 20261103
SCAN_QC_OMIT = ("ETAD-0037", "ETAD-0139", "ETAD-0158")
NAMED_REGIONS = {
    "500–1000 · lower fingerprint": (500, 1000),
    "1000–1300 · PTFE-rich region": (1000, 1300),
    "1300–1560 · lower mid-infrared": (1300, 1560),
    "1560–1700 · overlapping mid-infrared bands": (1560, 1700),
    "1700–1800 · carbonyl region": (1700, 1800),
    "1800–2250 · middle continuum": (1800, 2250),
    "2250–2400 · CO₂-prone region": (2250, 2400),
    "2400–2800 · lower stretching continuum": (2400, 2800),
    "2800–3000 · C–H stretching region": (2800, 3000),
    "3000–3500 · broad O–H/N–H region": (3000, 3500),
    "3500–4000 · upper continuum": (3500, 4000),
}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")) + "\n")


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def load_inputs(args):
    ledger = pd.read_csv(args.ledger)
    if ledger.id.duplicated().any():
        raise ValueError("diagnostic ledger must contain one row per physical filter")
    # Eligibility already carries the canonical exclusion and optical-source audit.
    cohort = ledger.loc[ledger.code.eq("ETAD") & ledger.spectral_pair_eligible.eq(True)
                        & ledger.optical_source.eq("direct_hips_fabs")].copy()
    cohort["date"] = pd.to_datetime(cohort.date)
    cohort = cohort.sort_values(["date", "id"]).reset_index(drop=True)
    required = ["optical_bc_ugm3", "pm25_ugm3", "volume_m3"]
    if not (np.isfinite(cohort[required]).all().all()
            and cohort[required].gt(0).all().all()):
        raise ValueError("positive, finite loading/volume metadata required; no silent row dropping")
    with np.load(args.arrays, allow_pickle=False) as arrays:
        ids = arrays["physical_filter_ids"].astype(str)
        if pd.Index(ids).duplicated().any():
            raise ValueError("native arrays duplicate a physical filter")
        order = pd.Index(ids).get_indexer(cohort.id)
        if (order < 0).any():
            raise ValueError("some eligible direct pairs lack native spectra")
        wn = arrays["wavenumbers"].copy()
        raw = arrays["physical_raw"][order].copy()
        corrected = arrays["physical_corrected__spline_default"][order].copy()
        shared = arrays["shared_mask"].copy()
    boundaries = pd.read_csv(args.boundaries)
    bounds = boundaries.loc[boundaries.method.eq("spline_default")
                            & boundaries.base_filter_id.isin(cohort.id)]
    if set(bounds.base_filter_id) != set(cohort.id):
        raise ValueError("every selected physical filter needs native support boundaries")
    max_extension = float(bounds.extension_last_wavenumber.max())
    if max_extension >= 1560:
        raise ValueError("fixed 1560 cm-1 cutoff no longer excludes all forced-zero tails")
    common = shared & (wn >= 1560) & (wn <= 4000)
    direct_fit = shared & (wn >= 1600) & (wn <= 4000)
    derivative = savgol_filter(raw, 15, 3, deriv=2,
                               delta=float(abs(np.median(np.diff(wn)))), axis=1)
    representations = {
        "raw_full": (raw, wn, "Raw · full measured spectrum"),
        "raw_common": (raw[:, common], wn[common], "Raw · 1560–4000 cm⁻¹"),
        "spline_common": (corrected[:, common], wn[common], "Spline · 1560–4000 cm⁻¹"),
        "removed_common": ((raw-corrected)[:, common], wn[common], "Removed spline component · 1560–4000 cm⁻¹"),
        "derivative_common": (derivative[:, common], wn[common], "Raw second derivative · 1560–4000 cm⁻¹"),
        "raw_fit_window": (raw[:, direct_fit], wn[direct_fit], "Raw · 1600–4000 cm⁻¹ sensitivity"),
        "spline_fit_window": (corrected[:, direct_fit], wn[direct_fit], "Spline · 1600–4000 cm⁻¹ sensitivity"),
    }
    # Independent per-filter preprocessing; no PCA, scaling or target fitting here.
    artifact_mask = ~(((wn >= 1000) & (wn <= 1300)) | ((wn >= 2250) & (wn <= 2400)))
    representations["raw_masked"] = (raw[:, artifact_mask], wn[artifact_mask],
                                      "Raw · PTFE-rich/CO₂-prone regions omitted")
    for values, _, _ in representations.values():
        if not np.isfinite(values).all():
            raise ValueError("a proposed representation contains unsupported channels")
    for col in required:
        cohort["log_" + col] = np.log(cohort[col])
    cohort["lot_group"] = cohort.lot_group.fillna("unknown").astype(str)
    cohort["season"] = season_labels(cohort.date, convention="belg_feb")
    cohort["cycle_end"] = cohort.date.dt.year + cohort.date.dt.month.ge(10).astype(int)
    weather = json.loads(args.weather.read_text())
    w = pd.DataFrame.from_dict(weather["rows"], orient="index", columns=weather["fields"])
    w = w.reindex(cohort.id)
    cohort["rh"] = w["RH (station, sample day)"].to_numpy()
    cohort["log_rain"] = np.log1p(w["Rain (station, sample day)"].to_numpy())
    support = {"lower_common_cutoff": 1560, "lower_direct_fit_sensitivity": 1600,
               "max_extension_endpoint": max_extension,
               "max_fitted_lower_anchor": float(bounds.segment2_lower_boundary.max()),
               "all_forced_zero_tails_excluded": True,
               "common_window_has_some_extrapolated_support": True}
    return cohort, representations, support, weather


def model_specs():
    specs = [{"id": "dummy", "label": "Training-only reference", "family": "dummy",
              "representation": "none", "covariates": "none"}]
    for family, label in (("linear", "Linear benchmark"), ("rf", "Random forest")):
        specs.append({"id": family + "_metadata", "label": label + " · loading and lot",
                      "family": family, "representation": "none", "covariates": "loading_lot"})
        for rep in ("raw_full", "raw_common", "spline_common", "removed_common", "derivative_common"):
            specs.append({"id": family + "_" + rep, "label": label + " · " + rep,
                          "family": family, "representation": rep, "covariates": "none"})
    for rep in ("raw_full", "raw_common", "spline_common", "removed_common", "derivative_common"):
        specs.append({"id": "pls_" + rep, "label": "Partial least squares · " + rep,
                      "family": "pls", "representation": rep, "covariates": "none",
                      "regression_only": True})
    for family, rep, cov, ident in (
        ("linear", "none", "loading_lot_season", "linear_metadata_season"),
        ("rf", "none", "loading_lot_season", "rf_metadata_season"),
        ("rf", "raw_full", "loading_lot", "rf_metadata_raw_full"),
        ("rf", "spline_common", "loading_lot", "rf_metadata_spline_common"),
        ("linear", "none", "loading_lot_weather", "linear_metadata_weather"),
        ("linear", "none", "loading_lot_weather_season", "linear_metadata_weather_season"),
        ("rf", "none", "loading_lot_weather", "rf_metadata_weather"),
        ("rf", "spline_common", "loading_lot_weather", "rf_metadata_weather_spline"),
    ):
        specs.append({"id": ident, "label": ident.replace("_", " "), "family": family,
                      "representation": rep, "covariates": cov, "regression_only": True})
    for rep in ("raw_fit_window", "spline_fit_window", "raw_masked"):
        specs.append({"id": "rf_" + rep, "label": "Random forest · " + rep,
                      "family": "rf", "representation": rep, "covariates": "none",
                      "sensitivity": True})
    for rep in ("raw_full", "spline_common"):
        specs.append({"id": "rf_" + rep + "_scan_qc", "label": "Random forest · " + rep,
                      "family": "rf", "representation": rep, "covariates": "none",
                      "sensitivity": True, "scan_qc": True,
                      "label_note": "Omit three filters flagged in the prior native-scan audit:0037,0139,0158; no new production exclusion"})
    return specs


def feature_input(cohort, representations, spec):
    rep = spec["representation"]
    spectra = np.empty((len(cohort), 0)) if rep == "none" else representations[rep][0]
    cov = spec["covariates"]
    numeric = ["log_optical_bc_ugm3", "log_pm25_ugm3", "log_volume_m3"] if cov != "none" else []
    categorical = ["lot_group"] if cov != "none" else []
    if "weather" in cov:
        numeric += ["rh", "log_rain"]
    if "season" in cov:
        categorical += ["season"]
    cols = numeric + categorical
    X, spectral_columns = make_feature_frame(spectra, metadata=cohort[cols] if cols else None,
                                             physical_ids=cohort.id.to_numpy())
    return X, spectral_columns, numeric, categorical


def save_checkpoint(out, scores, predictions, importance):
    pd.DataFrame(scores).to_csv(out / "model_scores.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "heldout_predictions.csv", index=False)
    pd.DataFrame(importance).to_csv(out / "region_importance.csv", index=False)


def run(args):
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    input_paths = [args.ledger, args.arrays, args.boundaries, args.weather,
                   Path(__file__), ROOT / "research/ftir_hips_chem/scripts/seasonal_spectral_ml.py",
                   ROOT / "research/ftir_hips_chem/scripts/config.py", ROOT / "uv.lock"]
    input_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in input_paths}
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cohort, representations, support, weather = load_inputs(args)
    cohort.to_csv(out / "physical_filter_cohort.csv", index=False)
    specs = model_specs()
    rep_labels = {key: value[2] for key, value in representations.items()}
    families = {"linear": "Linear benchmark", "rf": "Random forest", "pls": "Partial least squares"}
    covariate_labels = {
        "loading_lot": "HIPS, PM₂.₅, volume and lot",
        "loading_lot_season": "HIPS, PM₂.₅, volume, lot and season",
        "loading_lot_weather": "HIPS, PM₂.₅, volume, lot and weather",
        "loading_lot_weather_season": "HIPS, PM₂.₅, volume, lot, weather and season",
    }
    for spec in specs:
        if spec["representation"] != "none":
            spec["label"] = families[spec["family"]] + " · " + rep_labels[spec["representation"]]
            if spec["covariates"] != "none":
                spec["label"] += " + " + covariate_labels[spec["covariates"]]
        elif spec["family"] != "dummy":
            spec["label"] = families[spec["family"]] + " · " + covariate_labels[spec["covariates"]]
    scores, predictions, importance, confusion, fold_rows, component_scores = [], [], [], [], [], []
    detailed_permutations = []
    component_predictions = []
    participating_ids = set()
    fitted_cache = {}
    for calendar in ("belg_feb", "dry_feb"):
        active = cohort.copy()
        active["season"] = season_labels(active.date, convention=calendar)
        for spec in specs:
            # Calendar sensitivity is explicit and deliberately narrower than the primary screen.
            if calendar == "dry_feb" and spec["id"] not in {
                "dummy", "rf_raw_full", "rf_spline_common", "linear_metadata_season",
                "rf_metadata_raw_full", "rf_metadata_spline_common",
            }:
                continue
            use = active.copy()
            indices = np.arange(len(active))
            cohort_name = "direct232"
            if spec.get("scan_qc"):
                keep = ~active.id.isin(SCAN_QC_OMIT).to_numpy()
                indices = indices[keep]
                use = active.loc[keep].reset_index(drop=True)
                cohort_name = "scan_qc229"
            elif "weather" in spec["covariates"]:
                keep = np.isfinite(active[["rh", "log_rain"]]).all(axis=1).to_numpy()
                indices = indices[keep]
                use = active.loc[keep].reset_index(drop=True)
                cohort_name = "weather188"
            reps = {key: (values[indices], wn, label)
                    for key, (values, wn, label) in representations.items()}
            X, spectral_columns, numeric, categorical = feature_input(use, reps, spec)
            folds, inventory = make_cycle_folds(
                use.date.to_numpy(), use.id.to_numpy(), convention=calendar, mode="forward",
                train_coverage="all_seasons", test_coverage="all_seasons", min_train=50, min_test=20,
            )
            if spec["id"] == "dummy" and calendar == "belg_feb":
                inventory.to_csv(out / "cycle_coverage.csv", index=False)
            for fold in folds:
                if calendar == "belg_feb" and cohort_name == "direct232":
                    participating_ids.update(use.iloc[np.r_[fold.train, fold.test]].id)
                test_dates = use.iloc[fold.test].date
                seen = sorted(set(test_dates.dt.month.tolist()))
                item = {"id": fold.name, "label": f"{fold.test_cycle-1}–{fold.test_cycle}",
                        "n_train": len(fold.train), "n_test": len(fold.test),
                        "complete_test": fold.complete_test,
                        "missing_test_months": sorted(set(range(1, 13))-set(seen)),
                        "note": "All 12 months sampled in test cycle" if fold.complete_test else "Sparse later cycle; missing months retained",
                        "test_start": test_dates.min().date().isoformat(),
                        "test_end": test_dates.max().date().isoformat()}
                if item["id"] not in {r["id"] for r in fold_rows}:
                    fold_rows.append(item)
            for task, core_task in (("season", "classification"), ("disagreement", "regression")):
                if task == "season" and spec.get("regression_only"):
                    continue
                y = use.season.to_numpy() if task == "season" else use.optical_disagreement_ugm3.to_numpy()
                if spec["family"] == "dummy":
                    estimator = DummyClassifier(strategy="most_frequent") if task == "season" else DummyRegressor(strategy="mean")
                    X_eval = pd.DataFrame({"constant": np.zeros(len(use))}, index=use.id)
                else:
                    estimator = build_model(task=core_task, model=spec["family"], spectral_columns=spectral_columns,
                                            numeric_metadata=numeric, categorical_metadata=categorical,
                                            n_components=8, seed=SEED, n_jobs=args.jobs)
                    X_eval = X
                log(f"Fit {task} / {calendar} / {cohort_name} / {spec['id']}")
                result = evaluate_folds(X_eval, y, use.id.to_numpy(), folds, estimator, task=core_task)
                prefix = {"task": task, "calendar": calendar, "cohort": cohort_name, "model_id": spec["id"]}
                for row in result["metrics"].to_dict("records"):
                    row["fold_id"] = row.pop("fold")
                    scores.append({**prefix, **row})
                by_id = use.set_index("id")
                for row in result["predictions"].to_dict("records"):
                    identifier = row.pop("physical_id")
                    row["fold_id"] = row.pop("fold")
                    row["actual"] = row.pop("truth")
                    row["predicted"] = row.pop("prediction")
                    predictions.append({**prefix, **row, "id": identifier,
                                        "date": by_id.at[identifier, "date"].date().isoformat(),
                                        "season": by_id.at[identifier, "season"]})
                if task == "season":
                    p = result["predictions"]
                    for (fold_id, truth, prediction), count in p.groupby(["fold", "truth", "prediction"]).size().items():
                        confusion.append({**prefix, "fold_id": fold_id, "actual": truth,
                                          "predicted": prediction, "n": int(count)})
                if calendar == "belg_feb" and spec["id"] in {
                    "rf_raw_full", "rf_spline_common", "rf_removed_common", "rf_metadata_spline_common",
                }:
                    fitted_cache[(task, spec["id"])] = (result["fitted_folds"], X_eval, y, use, spectral_columns, reps)
                save_checkpoint(out, scores, predictions, importance)

    # Interpretation is prespecified: no interval is selected using a test score.
    for (task, model_id), (fitted, X, y, use, spectral_columns, reps) in fitted_cache.items():
        spec = next(s for s in specs if s["id"] == model_id)
        wn = reps[spec["representation"]][1]
        regions = spectral_regions(wn, spectral_columns, bin_width=100,
                                   named_regions=NAMED_REGIONS, include_whole=True)
        core_task = "classification" if task == "season" else "regression"
        for fold_id, fitted_record in fitted.items():
            estimator, fold = fitted_record["estimator"], fitted_record["fold"]
            prefix = {"task": task, "calendar": "belg_feb", "cohort": "direct232",
                      "model_id": model_id, "fold_id": fold_id}
            # Fine intervals only in the primary complete test cycle; broad regions remain in the sparse repeat.
            allowed = regions if fold.complete_test else {k: v for k, v in regions.items()
                                                         if k in NAMED_REGIONS or k == "whole_spectrum"}
            for scheme in (["across_filters", "within_season"] if task == "disagreement" else ["across_filters"]):
                log(f"Permute {task} / {model_id} / {fold_id} / {scheme}")
                strata = use.iloc[fold.test].season.to_numpy() if scheme == "within_season" else None
                detail = grouped_region_permutation(estimator, X.iloc[fold.test], y[fold.test], allowed,
                                                    task=core_task, repeats=args.permutations, seed=SEED,
                                                    strata=strata)
                for row in detail.to_dict("records"):
                    detailed_permutations.append({**prefix, "scheme": scheme, **row})
                for region_id, group in detail.groupby("region", sort=False):
                    cols = allowed[region_id]
                    channel_idx = [spectral_columns.index(c) for c in cols]
                    values = group.importance.to_numpy(float)
                    importance.append({**prefix, "kind": "permutation", "scheme": scheme,
                                       "region_id": region_id, "region_label": region_id,
                                       "low": float(wn[channel_idx].min()), "high": float(wn[channel_idx].max()),
                                       "n_channels": len(cols), "delta_mean": float(values.mean()),
                                       "delta_sd": float(values.std(ddof=1)) if len(values) > 1 else 0,
                                       "q05": float(np.quantile(values, .05)), "q95": float(np.quantile(values, .95)),
                                       "repeats": len(values), "baseline_loss": float(group.baseline_loss.iloc[0]),
                                       "metric": "balanced_error" if task == "season" else "rmse",
                                       "moved_fraction": float(group.moved_fraction.mean())})
            # Refitting changes the information available; it is not zeroing channels in the test set.
            wide = {key: value for key, value in regions.items() if key in NAMED_REGIONS}
            log(f"Refit without intervals {task} / {model_id} / {fold_id}")
            ablated = leave_region_out_refit(estimator, X.iloc[fold.train], y[fold.train],
                                            X.iloc[fold.test], y[fold.test], wide, task=core_task)
            for row in ablated.to_dict("records"):
                region_id = row["region"]
                channel_idx = [spectral_columns.index(c) for c in wide[region_id]]
                importance.append({**prefix, "kind": "ablation", "scheme": "across_filters",
                                   "region_id": region_id, "region_label": region_id,
                                   "low": float(wn[channel_idx].min()), "high": float(wn[channel_idx].max()),
                                   "n_channels": len(channel_idx), "delta_mean": row["importance"],
                                   "repeats": 1, "baseline_loss": row["baseline_loss"],
                                   "metric": "balanced_error" if task == "season" else "rmse"})
            save_checkpoint(out, scores, predictions, importance)

    # The disagreement contains an FTIR-derived component. Show this mathematical coupling separately.
    for model_id in ("rf_raw_full", "rf_spline_common"):
        _, X, _, use, spectral_columns, _ = fitted_cache[("disagreement", model_id)]
        folds, _ = make_cycle_folds(use.date.to_numpy(), use.id.to_numpy(), convention="belg_feb",
                                   mode="forward", train_coverage="all_seasons", test_coverage="all_seasons",
                                   min_train=50, min_test=20)
        for target in ("ec_ftir_ugm3", "optical_bc_ugm3"):
            estimator = build_model(task="regression", model="rf", spectral_columns=spectral_columns,
                                    n_components=8, seed=SEED, n_jobs=args.jobs)
            result = evaluate_folds(X, use[target].to_numpy(), use.id.to_numpy(), folds, estimator, task="regression")
            component_scores.extend({"target": target, "model_id": model_id, **row}
                                    for row in result["metrics"].to_dict("records"))
            component_predictions.extend({"target": target, "model_id": model_id, **row}
                                         for row in result["predictions"].to_dict("records"))

    pd.DataFrame(detailed_permutations).to_csv(out / "permutation_repeats.csv", index=False)
    pd.DataFrame(confusion).to_csv(out / "confusion.csv", index=False)
    pd.DataFrame(component_scores).to_csv(out / "component_prediction_checks.csv", index=False)
    pd.DataFrame(component_predictions).to_csv(out / "component_predictions.csv", index=False)
    curves = []
    for key, (matrix, wn, _) in representations.items():
        if key not in {"raw_full", "raw_common", "spline_common", "removed_common", "derivative_common"}:
            continue
        for season, group in cohort.groupby("season", sort=False):
            values = matrix[group.index.to_numpy()]
            curves.append({"representation": key, "season": season, "n": len(group),
                           "wavenumbers": wn, "median": np.median(values, axis=0),
                           "q25": np.quantile(values, .25, axis=0), "q75": np.quantile(values, .75, axis=0)})
    final_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in input_paths}
    if final_hashes != input_hashes:
        raise RuntimeError("Analysis inputs or executable source changed during the run; rerun from frozen files")
    heldout_ids = {row["id"] for row in predictions}
    receipt = {
        "inputs": input_hashes, "started_at": started_at, "inputs_unchanged_during_run": True,
        "source_revision": source_revision,
        "source_revision_note": "Base Git revision; exact executed worktree sources and dependencies are identified by the input SHA-256 hashes",
        "settings": {"seed": SEED, "rf_trees": 200, "rf_max_features": .33, "rf_min_leaf": 5,
                     "rf_max_depth": 8, "latent_components": 8, "permutations": args.permutations,
                     "model_selection": "Fixed settings; no tuning or region selection on held-out outcomes",
                     "season_primary": "belg_feb", "season_sensitivity": "dry_feb", "support": support,
                     "scan_qc_omit": list(SCAN_QC_OMIT), "scan_qc_scope": "RF performance sensitivity only; prior scan-audit flags",
                     "derivative": "Savitzky–Golay 15 native channels, polynomial3, second derivative per cm^-2",
                     "regional_interpretation": "Predeclared100cm-1 intervals, named wider intervals and whole spectrum",
                     "weather_source": weather["source"], "weather_join": weather["note"]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": time.monotonic()-started,
    }
    data = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": source_revision,
        "counts": {"physical_filters": len(cohort), "raw_channels": representations["raw_full"][0].shape[1],
                   "common_channels": representations["raw_common"][0].shape[1],
                   "model_participating_filters": len(participating_ids), "unique_heldout_filters": len(heldout_ids),
                   "descriptive_only_filters": len(cohort)-len(participating_ids),
                   "weather_filters": int(cohort[["rh", "log_rain"]].notna().all(axis=1).sum()),
                   "mac_m2_g": MAC_VALUE},
        "models": specs,
        "representations": [{"id": key, "label": label, "n_channels": len(wn),
                             "low": float(wn.min()), "high": float(wn.max())}
                            for key, (_, wn, label) in representations.items()],
        "folds": fold_rows, "scores": scores, "predictions": predictions,
        "importance": importance, "confusion": confusion, "seasonal_spectra": curves,
        "component_checks": component_scores, "provenance": receipt,
        "limitations": [
            "These are two different targets: season labels and FTIR-minus-optical disagreement. Neither validates Ethiopian EC accuracy.",
            "Disagreement contains FTIR-derived EC(X); spectral prediction can partly rediscover that term. Separate component scores and physical-filter predictions are retained.",
            "HIPS, PM2.5, volume and lot benchmarks include HIPS, a component of the disagreement target. They are HIPS-conditioned diagnostics, not an independent reference.",
            "One row is one physical filter. Original scans were averaged within each filter before any split; no repeat crosses a split.",
            "Scan-QC sensitivity omits0037,0139,0158 using prior native-audit flags, without changing production exclusions. It repeats RF performance, not all interval-importance tests.",
            "Both tasks start from232direct-HIPS Addis filters.218participate in model folds and127have held-out predictions;14latest Bega-only filters contribute only to descriptive spectra.",
            "Only the test cycle ending2024 has all12sampledmonths. The later30-filter cycle lacksSeptember; the earliest training cycle lacksOctober/November.",
            "Held-out importance measures model reliance. Intervals are correlated; joint permutation may create unusual cross-interval combinations.",
            "Permutation repeat spread is Monte Carlo variation, not a population confidence interval. Negative importance is retained.",
            "Within-season permutation preserves season membership only; it does not fully condition on loading or all correlated channels.",
            "Removing and refitting a band measures replaceable information. Small effects can reflect redundancy rather than no chemical contribution.",
            "Region names describe candidate spectral neighborhoods, not exclusive species assignments, sources, or mass fractions.",
            "Raw full and raw common windows are separate controls. The1560cutoff excludes padded spline zeros;1600sensitivity also excludes the short extrapolated tail.",
            "All signed values are retained. No positive-area normalization or clipping is used. Scaling, encoding, PCA and fitted models see training rows only.",
            "Weather is airport sample-day weather, not full collection-period or instrument-chamber humidity; later direct filters lack it.",
            "Bishoftu does not provide a three-season validation population in the available spectra.",
        ],
    }
    write_json(out / "seasonal_ml.json", data)
    write_json(out / "receipt.json", receipt)
    log(f"Complete: {len(scores)} scores, {len(predictions)} held-out predictions, {len(importance)} region results")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=TABLES / "baseline_review_20261102/filter_diagnostic_ledger.csv")
    parser.add_argument("--arrays", type=Path, default=TABLES / "baseline_existing_data_20261103/native_method_arrays.npz")
    parser.add_argument("--boundaries", type=Path, default=TABLES / "baseline_existing_data_20261103/boundary_sensitivity.csv")
    parser.add_argument("--weather", type=Path, default=ROOT / "gallery/app/public/data/research/addis_met/filters.json")
    parser.add_argument("--output", type=Path, default=TABLES / "seasonal_spectral_ml_20261103")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
