"""Controlled calibration-coverage experiments on independently measured source EC.

All repeated scans are reduced to one physical filter before a site split. The
outer test sites never influence sampling, preprocessing, component choice, or
loading-bin boundaries. Depletion/restoration arms have the same assay budget
and the same test filters. This is source-domain validation, not Addis accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from analog_diagnostics import paired_discrepancy_bootstrap
from outliers import get_clean_data
from seasonal_ml_checks import regression_scores
from seasonal_spectral_ml import build_model


SEED = 20261107
REFERENCE_KIND = "thermal_optical_ec"
DEFAULT_OPTIONS = {
    "ridge": ({"n_components": 8, "alpha": 1.}, {"n_components": 8, "alpha": 10.}),
    "pls": ({"n_components": 4}, {"n_components": 8}),
    "rf": ({"rf_params": {"min_samples_leaf": 2}}, {"rf_params": {"min_samples_leaf": 5}}),
}
LIMITS = [
    "Source thermal-optical EC performance does not establish Addis EC accuracy.",
    "The controlled depletion uses source reference labels; deployment selection must not use unknown reference EC.",
    "Site-bootstrap intervals condition on the fitted models and this prespecified split; they do not include training-selection uncertainty.",
    "Spectral coverage coordinates are descriptive, not validated error probabilities.",
]


@dataclass
class PhysicalSpectra:
    X: np.ndarray
    metadata: pd.DataFrame
    scan_ledger: pd.DataFrame


def _boolean_column(frame, name, default=False):
    values = frame[name] if name in frame else pd.Series(default, index=frame.index)
    if not values.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError(f"{name} must contain explicit booleans")
    return values.astype(bool)


def prepare_physical_spectra(
    X, metadata, *, physical_id_col="filter_id", site_col="site", target_col=None,
    stable_columns: Sequence[str] = (),
):
    """Mean repeated scans, retaining a complete, non-destructive eligibility ledger.

    ``physical_id_col`` must already be the authoritative physical identity, not
    an acquisition ID. Call the existing base-filter-ID resolver upstream for
    SPARTAN scan identifiers. A bad or nonfinite scan flags its entire physical
    filter; this routine never selects a replicate because it agrees with a target.
    """
    values = np.asarray(X, dtype=float)
    if not isinstance(metadata, pd.DataFrame) or metadata.columns.duplicated().any():
        raise ValueError("metadata must have distinct named columns")
    if values.ndim != 2 or not values.shape[1] or len(values) != len(metadata) or not len(values):
        raise ValueError("one nonempty spectrum is required per metadata row")
    needed = {physical_id_col, site_col, *stable_columns}
    if target_col is not None:
        needed.add(target_col)
    if not needed <= set(metadata):
        raise ValueError(f"missing spectral metadata columns: {sorted(needed-set(metadata))}")
    scans = metadata.copy(deep=True).reset_index(drop=True)
    for name in (physical_id_col, site_col):
        if scans[name].isna().any() or scans[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} must be complete")
        scans[name] = scans[name].astype(str).str.strip()
    if "analysis_id" in scans and scans.analysis_id.duplicated().any():
        raise ValueError("analysis_id must be unique; repeated scans need distinct acquisition IDs")
    scans["input_row"] = np.arange(len(scans))
    scans["is_excluded"] = _boolean_column(scans, "is_excluded")
    scans["is_outlier"] = _boolean_column(scans, "is_outlier")
    scans["is_blank"] = _boolean_column(scans, "is_blank")
    if "exclusion_reason" not in scans:
        scans["exclusion_reason"] = ""
    scans["exclusion_reason"] = scans.exclusion_reason.fillna("").astype(str)
    if (scans.is_excluded & scans.exclusion_reason.str.strip().eq("")).any():
        raise ValueError("every existing exclusion requires an exclusion_reason")
    if "outlier_reason" not in scans:
        scans["outlier_reason"] = ""
    scans["outlier_reason"] = scans.outlier_reason.fillna("").astype(str)
    if (scans.is_outlier & scans.outlier_reason.str.strip().eq("")).any():
        raise ValueError("every existing outlier flag requires an outlier_reason")
    scans["spectrum_finite"] = np.isfinite(values).all(axis=1)
    if target_col is not None:
        scans[target_col] = pd.to_numeric(scans[target_col], errors="raise")
    rows, means = [], []
    fixed = list(dict.fromkeys([site_col, "is_blank", *stable_columns]
                               + ([target_col] if target_col is not None else [])))
    for identity, indices in scans.groupby(physical_id_col, sort=True).indices.items():
        part = scans.iloc[indices]
        for name in fixed:
            if part[name].nunique(dropna=False) != 1:
                raise ValueError(f"conflicting repeated-filter metadata: {name} for {identity}")
        row = part.iloc[0].to_dict()
        reasons = sorted({r for r in [*part.exclusion_reason, *part.outlier_reason] if r.strip()})
        if not part.spectrum_finite.all():
            reasons.append("nonfinite_spectrum_in_physical_filter")
        if target_col is not None and not np.isfinite(part[target_col].to_numpy(float)).all():
            reasons.append("nonfinite_reference")
        if part.is_blank.iloc[0]:
            reasons.append("blank_reserved_for_QC")
        excluded, outlier = bool(part.is_excluded.any()), bool(part.is_outlier.any())
        row.update(n_scans=len(part), input_rows="|".join(map(str, indices)),
                   is_excluded=excluded, exclusion_reason=";".join(sorted(set(part.exclusion_reason)-{""})),
                   is_outlier=outlier, outlier_reason=";".join(sorted(set(part.outlier_reason)-{""})),
                   spectrum_finite=bool(part.spectrum_finite.all()),
                   experiment_eligible=not excluded and not reasons,
                   eligibility_reason=";".join(dict.fromkeys(reasons)))
        # A nonempty note on an otherwise accepted row is not an exclusion.
        row["experiment_eligible"] = bool(not excluded and not outlier and part.spectrum_finite.all()
            and not part.is_blank.iloc[0]
            and (target_col is None or np.isfinite(part[target_col].to_numpy(float)).all()))
        if row["experiment_eligible"]:
            row["eligibility_reason"] = ""
        rows.append(row)
        means.append(values[indices].mean(axis=0))
    physical = pd.DataFrame(rows).reset_index(drop=True)
    eligibility = physical.set_index(physical_id_col)[["experiment_eligible", "eligibility_reason"]]
    scans = scans.join(eligibility, on=physical_id_col, validate="many_to_one")
    return PhysicalSpectra(np.asarray(means), physical, scans)


def source_test_sites(metadata, *, site_col="site", fraction=.25, seed=SEED):
    """Choose and record complete test sites using site identities only."""
    if not 0 < fraction < 1 or metadata[site_col].isna().any():
        raise ValueError("a complete site column and 0 < fraction < 1 are required")
    sites = np.asarray(sorted(set(metadata[site_col].astype(str))))
    if len(sites) < 4:
        raise ValueError("at least four sites are needed for training and site-cluster evaluation")
    n = min(len(sites)-2, max(2, int(np.ceil(fraction*len(sites)))))
    return sorted(np.random.default_rng(seed).choice(sites, size=n, replace=False).tolist())


def spectral_coordinates(X, *, seed=SEED, n_components=8):
    """Training/pool-only mean-centered PCA scores, scaled by score SD."""
    values = np.asarray(X, float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite spectra required for coverage coordinates")
    k = min(int(n_components), len(values)-1, values.shape[1])
    if k < 1:
        raise ValueError("positive PCA component count required")
    with threadpool_limits(limits=1):
        model = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(values)
        coordinates = model.transform(values)
    scale = np.std(coordinates, axis=0)
    scale[scale <= 1e-12] = 1.
    coordinates /= scale
    return coordinates, {"n_components": k, "center_sha256": hashlib.sha256(
        np.asarray(model.mean_, dtype="<f8").tobytes()).hexdigest(),
        "basis": "mean-centered PCA, scores divided by population SD; no response labels"}


def calibration_scores(actual, predicted):
    """Existing prediction metrics plus explicit compression/negative diagnostics."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("one prediction per test physical filter is required")
    if not np.isfinite([actual, predicted]).all():
        raise ValueError("test observations and predictions must be finite; no silent complete-case change")
    span = float(np.diff(np.quantile(actual, [.05, .95]))[0])
    prediction_span = float(np.diff(np.quantile(predicted, [.05, .95]))[0])
    return {"n": len(actual), **regression_scores(actual, predicted),
            "median_bias": float(np.median(predicted-actual)),
            "prediction_p90_span": prediction_span, "reference_p90_span": span,
            "p90_span_ratio": prediction_span/span if span > 0 else np.nan,
            "negative_predictions": int((predicted < 0).sum())}


def fit_source_model(X, y, sites, physical_ids, *, family, inner_folds=3,
                     seed=SEED, options=None):
    """Select a pipeline inside training-site CV, then fit that training subset."""
    if family not in DEFAULT_OPTIONS:
        raise ValueError(f"unknown source model: {family}")
    X, y, sites, physical_ids = np.asarray(X, float), np.asarray(y, float), np.asarray(sites), np.asarray(physical_ids)
    if X.ndim != 2 or y.shape != (len(X),) or sites.shape != y.shape or physical_ids.shape != y.shape:
        raise ValueError("source model inputs are not aligned")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("source model inputs must be finite")
    if len(set(physical_ids)) != len(X):
        raise ValueError("fit requires one row per physical filter")
    if inner_folds < 2 or len(set(sites)) < 2 or len(X) < 4:
        raise ValueError("site-held-out tuning needs at least two training sites and four physical filters")
    n_folds = min(inner_folds, len(set(sites)))
    splits = list(GroupKFold(n_splits=n_folds).split(X, y, sites))
    if any(len(train) < 2 for train, _ in splits):
        raise ValueError("every inner training fold needs at least two physical filters")
    columns = [f"sp_{i:05d}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=columns)
    candidates = list(DEFAULT_OPTIONS[family] if options is None else options)
    if not candidates:
        raise ValueError("at least one model configuration is required")
    allowed = {"n_components", "alpha", "scale_spectra", "rf_params"}
    for option in candidates:
        if not isinstance(option, dict) or set(option)-allowed:
            raise ValueError("model options may only configure components, alpha, scaling and RF parameters")

    def estimator(option):
        return build_model(task="regression", model="linear" if family == "ridge" else family,
                           spectral_columns=columns, seed=seed, n_jobs=1, **option)

    folds = []
    for fold, (train, validation) in enumerate(splits):
        for role, indices in (("train", train), ("validation", validation)):
            folds.extend({"inner_fold": fold, "role": role, "filter_id": str(physical_ids[i]),
                          "site": str(sites[i])} for i in indices)
    tuning = []
    with threadpool_limits(limits=1):
        for candidate, option in enumerate(candidates):
            prediction = np.full(len(y), np.nan)
            error = ""
            try:
                for train, validation in splits:
                    model = estimator(option).fit(frame.iloc[train], y[train])
                    prediction[validation] = np.asarray(model.predict(frame.iloc[validation])).reshape(-1)
                score = calibration_scores(y, prediction)["rmse"]
            except (ValueError, np.linalg.LinAlgError) as exc:
                score, error = np.nan, str(exc)
            tuning.append({"candidate": candidate, "parameters": json.dumps(option, sort_keys=True),
                           "inner_rmse": score, "status": "failed" if error else "completed", "reason": error})
        valid = [r for r in tuning if np.isfinite(r["inner_rmse"])]
        if not valid:
            raise ValueError("no model candidate completed all inner-site folds")
        chosen = min(valid, key=lambda row: (row["inner_rmse"], row["candidate"]))["candidate"]
        fitted = estimator(candidates[chosen]).fit(frame, y)
    for row in tuning:
        row["selected"] = row["candidate"] == chosen
    return fitted, columns, pd.DataFrame(tuning), pd.DataFrame(folds)


def paired_source_contrast(actual, baseline, candidate, sites, *, repeats=200, seed=SEED):
    """Reuse the existing paired cluster bootstrap with source sites as blocks."""
    a, b = calibration_scores(actual, baseline), calibration_scores(actual, candidate)
    if len(set(sites)) < 2 or repeats == 0:
        return {**{f"delta_{k}": b[k]-a[k] for k in ("rmse", "mae", "bias")},
                "n_sites": len(set(sites)), "n_boot": 0, "interval_status": "not_estimable"}
    result = paired_discrepancy_bootstrap(actual, baseline, candidate, sites,
                                         n_boot=repeats, seed=seed)
    result["n_sites"] = result.pop("n_months")
    result["resampling_unit"] = "held_out_source_site"
    result["interval_status"] = "conditional_on_fitted_models"
    return result


def run_coverage_experiment(
    X, metadata, *, reference_kind, test_sites, target_col="ec_ug",
    physical_id_col="filter_id", site_col="site", budget=None,
    regions=("low", "high", "spectral"), retain_fractions=(0., .5, 1.),
    models=("ridge", "pls", "rf"), inner_folds=3, seed=SEED,
    model_options=None, bootstrap_repeats=200, response_unit="ug/filter",
):
    """Restore predeclared low/high/spectral coverage at an equal training budget.

    ``retain_fraction=1`` is the restored arm, with approximately the natural
    training-pool prevalence of that region. Depleted slots are replaced by
    non-region filters from a fixed randomized ordering. The full outer training
    pool supplies design thresholds; each predictive pipeline fits only its
    selected training filters, including during inner CV.
    """
    if reference_kind != REFERENCE_KIND:
        raise ValueError("coverage evaluation requires independently measured thermal_optical_ec")
    if response_unit != "ug/filter":
        raise ValueError("coverage evaluation is defined for EC loading in ug/filter")
    if "reference_kind" in metadata and not metadata.reference_kind.eq(reference_kind).all():
        raise ValueError("metadata reference_kind conflicts with the declared independent EC reference")
    regions, fractions, models = tuple(regions), tuple(retain_fractions), tuple(models)
    if not regions or len(set(regions)) != len(regions) or set(regions)-{"low", "high", "spectral"}:
        raise ValueError("regions must be distinct low, high or spectral choices")
    if (not fractions or len(set(fractions)) != len(fractions) or 1. not in fractions
            or any(not np.isfinite(f) or not 0 <= f <= 1 for f in fractions)):
        raise ValueError("distinct retain fractions in [0,1], including restored=1, are required")
    if not models or len(set(models)) != len(models) or set(models)-set(DEFAULT_OPTIONS):
        raise ValueError("choose distinct ridge, pls or rf model families")
    if model_options is not None and (not isinstance(model_options, dict) or set(model_options)-set(models)):
        raise ValueError("model_options must be keyed only by requested model families")
    if not isinstance(bootstrap_repeats, int) or bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be a nonnegative integer")
    if isinstance(test_sites, str) or not test_sites or len(set(test_sites)) != len(test_sites):
        raise ValueError("test_sites must be a nonempty list of distinct site identities")
    cohort = prepare_physical_spectra(X, metadata, physical_id_col=physical_id_col,
                                     site_col=site_col, target_col=target_col)
    meta = cohort.metadata.copy()
    test_sites = sorted(map(str, test_sites))
    if not set(test_sites) <= set(meta[site_col]):
        raise ValueError("some requested test sites are absent from the input")
    meta["role"] = np.where(meta[site_col].isin(test_sites), "test", "training_pool")
    eligible = meta.experiment_eligible.to_numpy(bool)
    train = np.flatnonzero(eligible & meta.role.eq("training_pool").to_numpy())
    test = np.flatnonzero(eligible & meta.role.eq("test").to_numpy())
    parameters = dict(reference_kind=reference_kind, target_col=target_col, response_unit=response_unit,
                      test_sites=test_sites, budget=budget, regions=list(regions),
                      retain_fractions=list(fractions), models=list(models), inner_folds=inner_folds,
                      seed=seed, bootstrap_repeats=bootstrap_repeats, model_options=model_options)
    result = {"summary": {"status": "insufficient_data", "parameters": parameters,
              "limitations": LIMITS, "n_training_pool": len(train), "n_test": len(test)},
              "tables": {"physical_ledger": meta, "scan_ledger": cohort.scan_ledger}}
    if len(train) < 6 or not len(test) or meta.iloc[train][site_col].nunique() < 2:
        result["summary"]["reason"] = "need at least six eligible training filters, two training sites and a nonempty fixed test set"
        return result
    y = meta[target_col].to_numpy(float)
    ids, sites = meta[physical_id_col].to_numpy(str), meta[site_col].to_numpy(str)
    low, high = np.quantile(y[train], [.2, .8])
    coordinates, descriptor = spectral_coordinates(cohort.X[train], seed=seed)
    radius = np.linalg.norm(coordinates, axis=1)
    radius_cut = float(np.quantile(radius, .8))
    masks = {"low": y[train] <= low, "high": y[train] >= high, "spectral": radius >= radius_cut}
    if any(not masks[r].any() or masks[r].all() for r in regions):
        result["summary"]["reason"] = "a requested coverage region has no distinct region/complement; thresholds may be tied"
        return result
    maximum = min(int((~masks[r]).sum()) for r in regions)
    if budget is None:
        budget = min(200, int(.7*len(train)), maximum)
    if not isinstance(budget, int) or budget < 6:
        raise ValueError("budget must be an integer of at least six physical filters")
    if budget > maximum:
        raise ValueError(f"budget exceeds the common fully-depleted capacity ({maximum}); reduce it")
    parameters["budget"] = budget
    result["summary"]["spectral_descriptor"] = descriptor
    result["tables"]["thresholds"] = pd.DataFrame([
        {"variable": target_col, "boundary": "training_q20", "value": low},
        {"variable": target_col, "boundary": "training_q80", "value": high},
        {"variable": "spectral_score_radius", "boundary": "training_q80", "value": radius_cut},
    ])
    test_bins = np.where(y[test] <= low, "low", np.where(y[test] >= high, "high", "middle"))
    selections, predictions, metrics, tunings, inner_members, run_status, contrasts = [], [], [], [], [], [], []
    for region in regions:
        rng = np.random.default_rng(np.random.SeedSequence([seed, ("low", "high", "spectral").index(region)]))
        inside = rng.permutation(train[masks[region]])
        outside = rng.permutation(train[~masks[region]])
        restored_n = min(len(inside), max(1, int(round(budget*float(masks[region].mean())))))
        family_predictions = {}
        for fraction in sorted(fractions):
            region_n = int(round(restored_n*fraction))
            chosen = np.sort(np.r_[inside[:region_n], outside[:budget-region_n]])
            chosen_set = set(chosen)
            arm = f"{region}_retain_{fraction:g}"
            for i in train:
                selected = i in chosen_set
                selections.append({"arm": arm, "region": region, "retain_fraction": fraction,
                    "filter_id": ids[i], "site": sites[i], "selected": selected,
                    "in_region": bool(masks[region][np.searchsorted(train, i)]),
                    "selection_reason": "selected_at_fixed_budget" if selected else "not_selected_by_prespecified_coverage_design"})
            for family in models:
                key = dict(arm=arm, region=region, retain_fraction=fraction, model=family)
                try:
                    fitted, columns, tuning, folds = fit_source_model(
                        cohort.X[chosen], y[chosen], sites[chosen], ids[chosen], family=family,
                        inner_folds=inner_folds, seed=seed,
                        options=(model_options or {}).get(family))
                    with threadpool_limits(limits=1):
                        pred = np.asarray(fitted.predict(pd.DataFrame(cohort.X[test], columns=columns))).reshape(-1)
                    calibration_scores(y[test], pred)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    run_status.append({**key, "status": "failed", "reason": str(exc), "n_train": len(chosen)})
                    continue
                run_status.append({**key, "status": "completed", "reason": "", "n_train": len(chosen),
                                   "n_region": region_n, "n_train_sites": len(set(sites[chosen]))})
                tunings.append(tuning.assign(**key))
                inner_members.append(folds.assign(**key))
                family_predictions[family, fraction] = pred
                for j, i in enumerate(test):
                    predictions.append({**key, "filter_id": ids[i], "site": sites[i],
                        "actual": y[i], "prediction": pred[j], "loading_bin": test_bins[j],
                        "squared_error": (pred[j]-y[i])**2, "response_unit": "ug/filter"})
                for label in ("all", "low", "middle", "high"):
                    selected = np.ones(len(test), bool) if label == "all" else test_bins == label
                    if selected.any():
                        metrics.append({**key, "loading_bin": label, "n_train": len(chosen),
                                        **calibration_scores(y[test][selected], pred[selected])})
        for family in models:
            restored = family_predictions.get((family, 1.))
            if restored is None:
                continue
            for fraction in sorted(fractions):
                depleted = family_predictions.get((family, fraction))
                if fraction == 1. or depleted is None:
                    continue
                contrasts.append({"region": region, "model": family, "retain_fraction": fraction,
                    "comparison": "depleted_minus_restored", **paired_source_contrast(
                        y[test], restored, depleted, sites[test], repeats=bootstrap_repeats, seed=seed)})
    result["tables"].update(training_selection=pd.DataFrame(selections), predictions=pd.DataFrame(predictions),
        metrics=pd.DataFrame(metrics), tuning=pd.concat(tunings, ignore_index=True) if tunings else pd.DataFrame(),
        inner_membership=pd.concat(inner_members, ignore_index=True) if inner_members else pd.DataFrame(),
        runs=pd.DataFrame(run_status), paired_contrasts=pd.DataFrame(contrasts))
    completed = sum(r["status"] == "completed" for r in run_status)
    result["summary"].update(status="completed" if completed == len(run_status) else "failed",
        completed_runs=completed, attempted_runs=len(run_status), fixed_test_filter_ids=ids[test].tolist(),
        test_membership_sha256=hashlib.sha256(json.dumps(ids[test].tolist()).encode()).hexdigest(),
        thresholds_fit_on="outer training pool only", reference_scope="source_domain_only")
    return result


def load_historical_source(folder, *, source_policy="documented_qc"):
    """Adapt the verified historical export while retaining all 906 filters.

    The shared loader reconstructs and verifies the archived R model. Its fitted
    model is discarded here: all experiment fits and choices start fresh inside
    the new split. The already documented TONT1 scan flag follows the November 3
    matched-source control and remains visible in the returned metadata.
    """
    from pls_transfer import load_current_pls_model

    if source_policy not in {"documented_qc", "original906"}:
        raise ValueError("unknown historical source policy")
    folder = Path(folder)
    with threadpool_limits(limits=1):
        historical = load_current_pls_model(folder, "EC")
    samples = pd.read_csv(folder / "notebook/extracted/EC_samples.csv")
    if len(samples) != 906 or len(set(historical.filter_ids)) != 906:
        raise ValueError("the historical source adapter requires the reviewed 906 physical filters")
    np.testing.assert_array_equal(samples.AnalysisId, historical.analysis_ids)
    meta = pd.DataFrame({"analysis_id": historical.analysis_ids.astype(str),
                         "filter_id": historical.filter_ids.astype(str), "site": historical.sites,
                         "sample_date": pd.to_datetime(samples.SampleDate).dt.strftime("%Y-%m-%d"),
                         "ec_ug": historical.y, "is_excluded": False, "exclusion_reason": "",
                         "reference_kind": REFERENCE_KIND, "is_blank": False})
    policy = (meta.analysis_id.eq("164549") & meta.filter_id.eq("2008808") & meta.site.eq("TONT1")
              & meta.sample_date.eq("2022-11-19"))
    if policy.sum() != 1:
        raise ValueError("historical TONT1 exclusion must identify exactly one physical filter")
    if source_policy == "documented_qc":
        meta.loc[policy, "is_excluded"] = True
        meta.loc[policy, "exclusion_reason"] = "previously_documented_TONT1_offset_scan"
    meta["source_policy"] = source_policy
    # Exercise the same canonical flag interpretation used by the other source
    # controls, retaining the complete original metadata in this adapter.
    if len(get_clean_data(meta)) != (905 if source_policy == "documented_qc" else 906):
        raise ValueError("historical clean-source population differs from the stated policy")
    return historical.X, meta, historical.wavenumbers
