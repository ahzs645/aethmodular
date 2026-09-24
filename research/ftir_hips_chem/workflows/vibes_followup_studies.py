"""Frozen common-test and training-site routing follow-up for AIRSpec/VIBES.

The original corrected arrays and outer site split are read-only. Selection
masks alter analog membership, never the 2,002-channel PLS fitting grid.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
from seasonal_analogs import mean_correlation_scores, spectral_region_mask

SOURCE = AREA / "output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"
OUTPUT = AREA / "output/tables/vibes_followup_notebook"
METHODS = ("AIRSpec", "VIBES")
REGIONS = ((1425, 1800), (1800, 2500), (2500, 3000), (3000, 4001))
MASKS = (
    ("analog_full", None, None),
    ("analog_exclude_1800_2500", 1800, None),
    ("analog_exclude_1800_2500_above_3600", 1800, 3600),
    ("analog_exclude_1800_2500_above_3500", 1800, 3500),
)
EXTRA_FROZEN_HASHES = {
    "RUN_MANIFEST.json": "1824bf4ff4b5dd4194a8f13f6e2bbf004f981167a1c41a2afe553a1145e48f44",
    "calibration_scores.csv": "a6ac2d47fbc29c3388fd58b2cfda85c4fbbd29b69278866b7da00565b1757e7f",
    "addis_predictions.csv": "c2cd8f34ee0b4f0fc37b345a4f1afa74e1d7e65225def1d091b57d264849178a",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_inputs(source: Path = SOURCE) -> dict:
    source = Path(source).resolve()
    manifest = json.loads((source / "RUN_MANIFEST.json").read_text())
    if manifest["n_failed"] or manifest["n_cases"] != 12808:
        raise ValueError("Expected complete frozen full-profile run")
    paths = ["RUN_MANIFEST.json", "case_audit.csv", "heldout_predictions.csv",
             "addis_predictions.csv", "calibration_scores.csv", "wn.npy",
             "corrected_AIRSpec.npy", "corrected_VIBES.npy"]
    hashes = {name: sha(source / name) for name in paths}
    freeze = json.loads((ROOT / "docs/openresearch-retrospective/experiments/vibes-inputs.json").read_text())
    if manifest["signature"] != freeze["source_signature"]:
        raise ValueError("Completed run signature differs from the frozen audit")
    for item in freeze["inputs"]:
        name = Path(item["path"]).name
        if name in hashes and item["role"] == "frozen saved Colab input" and hashes[name] != item["sha256"]:
            raise ValueError(f"Frozen saved Colab input changed: {name}")
    for name, expected in EXTRA_FROZEN_HASHES.items():
        if hashes[name] != expected:
            raise ValueError(f"Frozen completed-run artifact changed: {name}")
    cases = pd.read_csv(source / "case_audit.csv")
    if len(cases) != manifest["n_cases"] or not cases.sample_id.is_unique:
        raise ValueError("Case ledger does not match run manifest")
    if not cases.paired_valid.all():
        raise ValueError("Expected paired, finite completed corrections")
    arrays = {method: np.load(source / f"corrected_{method}.npy", mmap_mode="r")
              for method in METHODS}
    wn = np.load(source / "wn.npy")
    if len(wn) != 2002 or any(arr.shape != (len(cases), len(wn)) for arr in arrays.values()):
        raise ValueError("Saved case/feature alignment changed")
    cal = cases.kind.eq("calibration")
    train = np.flatnonzero(cal & cases.split.eq("train"))
    test = np.flatnonzero(cal & cases.split.eq("test"))
    targets = np.flatnonzero(cases.kind.eq("target"))
    if len(train) != 10066 or len(test) != 2327 or len(targets) != 253:
        raise ValueError("Frozen population counts changed")
    if set(cases.Site.iloc[train]) & set(cases.Site.iloc[test]):
        raise ValueError("Outer test sites leaked into training")
    scores = pd.read_csv(source / "calibration_scores.csv")
    saved = pd.read_csv(source / "heldout_predictions.csv")
    if saved.duplicated(["sample_id", "cohort", "method"]).any():
        raise ValueError("Duplicate frozen held-out prediction")
    return dict(source=source, manifest=manifest, hashes=hashes, cases=cases,
                arrays=arrays, wn=wn, train=train, test=test, targets=targets,
                scores=scores, saved=saved)


def fit_predict(X: np.ndarray, y: np.ndarray, train: np.ndarray,
                predict: np.ndarray, k: int) -> np.ndarray:
    with threadpool_limits(limits=1):
        model = PLSRegression(n_components=k, scale=False, max_iter=1000,
                              tol=1e-10).fit(X[train], y[train])
        return model.predict(X[predict]).ravel()


def score(y: np.ndarray, prediction: np.ndarray) -> dict:
    return {"n": len(y), "RMSE": float(np.sqrt(np.mean((prediction-y)**2))),
            "MAE": float(np.mean(np.abs(prediction-y))),
            "bias": float(np.mean(prediction-y)),
            "predictive_R2": float(r2_score(y, prediction))}


def paired_site_interval(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray,
                         sites: np.ndarray, seed: int = 20260922,
                         repeats: int = 2000) -> tuple[float, float, float]:
    frame = pd.DataFrame({"Site": sites, "candidate_sq": (candidate-y)**2,
                          "reference_sq": (reference-y)**2})
    grouped = frame.groupby("Site", sort=True).agg(
        candidate_sq=("candidate_sq", "sum"),
        reference_sq=("reference_sq", "sum"), n=("candidate_sq", "size"),
    ).to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(repeats, len(grouped)))
    totals = grouped[draws].sum(axis=1)
    delta = np.sqrt(totals[:, 0]/totals[:, 2])-np.sqrt(totals[:, 1]/totals[:, 2])
    point = np.sqrt(np.mean((candidate-y)**2))-np.sqrt(np.mean((reference-y)**2))
    return float(point), *map(float, np.quantile(delta, [.025, .975]))


def common_test(inputs: dict, output: Path = OUTPUT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c, X, wn = inputs["cases"], inputs["arrays"], inputs["wn"]
    train, test, targets = inputs["train"], inputs["test"], inputs["targets"]
    y = c.y.to_numpy(float)
    test_ids = c.sample_id.iloc[test].to_numpy()
    test_sites = c.Site.iloc[test].to_numpy()
    frozen = inputs["saved"]
    k = {method: int(inputs["scores"].query("cohort == 'full_pool' and method == @method").k.iloc[0])
         for method in METHODS}
    rows, membership = [], []
    # All alternatives are scored on the exact same 2,327 outer-site filters.
    for cohort in ("full_pool", "locked800"):
        fitting = train if cohort == "full_pool" else train[c.locked800.iloc[train].to_numpy(bool)]
        for method in METHODS:
            components = int(inputs["scores"].query("cohort == @cohort and method == @method").k.iloc[0])
            prediction = fit_predict(X[method], y, fitting, test, components)
            # Check the refit against the original file on its original test subset.
            prior = frozen.query("cohort == @cohort and method == @method").set_index("sample_id")
            overlap = pd.Index(test_ids).isin(prior.index)
            old = prior.loc[test_ids[overlap], "prediction"].to_numpy(float)
            if not np.allclose(prediction[overlap], old, rtol=0, atol=1e-8):
                raise ValueError(f"Frozen {cohort}/{method} model did not reproduce")
            rows.extend(dict(sample_id=sid, Site=site, y=truth, model=cohort,
                             method=method, prediction=float(pred))
                        for sid, site, truth, pred in zip(test_ids, test_sites, y[test], prediction))
    # Selection uses *AIRSpec* corrected training and Addis spectra for both
    # methods, so the two methods receive exactly the same selected filter IDs.
    selected_sets = {}
    for name, low, upper in MASKS:
        mask = spectral_region_mask(wn, co2_low=low, upper=upper)
        similarities = mean_correlation_scores(X["AIRSpec"][train],
                                               X["AIRSpec"][targets], mask)
        order = np.lexsort((c.sample_id.iloc[train].to_numpy(), -similarities))
        chosen = train[order[:500]]
        selected_sets[name] = set(c.sample_id.iloc[chosen])
        for rank, row in enumerate(chosen, 1):
            membership.append(dict(model=name, rank=rank, sample_id=c.sample_id.iloc[row],
                                   Site=c.Site.iloc[row], selection_score=float(similarities[order[rank-1]]),
                                   matching_channels=int(mask.sum()), pls_channels=len(wn)))
        if c.Site.iloc[chosen].nunique() < 5:
            raise ValueError(f"{name} has too few training sites for transfer")
        for method in METHODS:
            prediction = fit_predict(X[method], y, chosen, test, k[method])
            rows.extend(dict(sample_id=sid, Site=site, y=truth, model=name,
                             method=method, prediction=float(pred))
                        for sid, site, truth, pred in zip(test_ids, test_sites, y[test], prediction))
    predictions = pd.DataFrame(rows)
    if predictions.groupby(["model", "method"]).sample_id.nunique().ne(len(test)).any():
        raise ValueError("Candidate models do not cover the common test")
    base = predictions.query("model == 'full_pool' and method == 'AIRSpec'").prediction.to_numpy()
    metrics = []
    for (model, method), part in predictions.groupby(["model", "method"], sort=False):
        pred = part.prediction.to_numpy(float)
        point, low, high = paired_site_interval(y[test], pred, base, test_sites)
        metrics.append(dict(model=model, method=method, n_train=(len(train) if model=="full_pool" else
                            int(c.locked800.iloc[train].sum()) if model=="locked800" else 500),
                            n_train_sites=(c.Site.iloc[train].nunique() if model=="full_pool" else
                                           c.Site.iloc[train[c.locked800.iloc[train].to_numpy(bool)]].nunique()
                                           if model=="locked800" else
                                           len({m["Site"] for m in membership if m["model"]==model})),
                            components=k[method] if model.startswith("analog_") else
                            int(inputs["scores"].query("cohort == @model and method == @method").k.iloc[0]),
                            **score(y[test], pred), delta_RMSE_vs_full_AIRSpec=point,
                            delta_RMSE_CI_low=low, delta_RMSE_CI_high=high))
    metrics = pd.DataFrame(metrics)
    selected = pd.DataFrame(membership)
    restricted = predictions.loc[predictions.model.eq("locked800")].copy()
    restricted["in_historical_membership"] = c.set_index("sample_id").loc[
        restricted.sample_id, "locked800"].to_numpy(bool)
    restricted_summary = pd.DataFrame([
        dict(method=method, in_historical_membership=member, **score(part.y.to_numpy(), part.prediction.to_numpy()))
        for (method, member), part in restricted.groupby(["method", "in_historical_membership"])
    ])
    selected.to_csv(output / "analog_membership.csv", index=False)
    predictions.to_csv(output / "common_test_predictions.csv", index=False)
    metrics.to_csv(output / "common_test_metrics.csv", index=False)
    restricted_summary.to_csv(output / "restricted_membership_sensitivity.csv", index=False)
    (output / "selection_overlap.json").write_text(json.dumps({
        name: {"replaced_vs_unmasked": 500-len(ids & selected_sets["analog_full"]),
               "n_training_sites": int(selected.loc[selected.model.eq(name), "Site"].nunique())}
        for name, ids in selected_sets.items()}, indent=2) + "\n")
    return metrics, predictions, selected


def spectral_features(inputs: dict) -> np.ndarray:
    wn, arrays = inputs["wn"], inputs["arrays"]
    cols = []
    for method in METHODS:
        for low, high in REGIONS:
            band = np.asarray(arrays[method][:, (wn >= low) & (wn < high)], float)
            cols.extend((band.mean(axis=1), band.std(axis=1)))
    features = np.column_stack(cols)
    if features.shape != (len(inputs["cases"]), 16) or not np.isfinite(features).all():
        raise ValueError("Nonfinite or misaligned spectral routing features")
    return features


def oof_predictions(inputs: dict, indices: np.ndarray, folds: int) -> dict[str, np.ndarray]:
    c, X = inputs["cases"], inputs["arrays"]
    y = c.y.to_numpy(float)
    groups = c.Site.iloc[indices].to_numpy()
    result = {method: np.full(len(indices), np.nan) for method in METHODS}
    k = {method: int(inputs["scores"].query("cohort == 'full_pool' and method == @method").k.iloc[0])
         for method in METHODS}
    for fit_pos, hold_pos in GroupKFold(n_splits=folds).split(indices, groups=groups):
        fit, hold = indices[fit_pos], indices[hold_pos]
        if set(c.Site.iloc[fit]) & set(c.Site.iloc[hold]):
            raise ValueError("Grouped calibration fold leaked sites")
        for method in METHODS:
            result[method][hold_pos] = fit_predict(X[method], y, fit, hold, k[method])
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError("Incomplete out-of-fold base predictions")
    return result


def train_router(features: np.ndarray, y: np.ndarray, air: np.ndarray,
                 vibes: np.ndarray):
    winner = (np.abs(vibes-y) < np.abs(air-y)).astype(int)
    if np.unique(winner).size != 2:
        raise ValueError("Routing training has a single winning method")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features, winner)
    return model


def routing_study(inputs: dict, output: Path = OUTPUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c, X = inputs["cases"], inputs["arrays"]
    y, train, test = c.y.to_numpy(float), inputs["train"], inputs["test"]
    features = spectral_features(inputs)
    k = {method: int(inputs["scores"].query("cohort == 'full_pool' and method == @method").k.iloc[0])
         for method in METHODS}
    nested = []
    groups = c.Site.iloc[train].to_numpy()
    for outer, (fit_pos, hold_pos) in enumerate(GroupKFold(n_splits=5).split(train, groups=groups), 1):
        fit, hold = train[fit_pos], train[hold_pos]
        inner = oof_predictions(inputs, fit, folds=4)
        router = train_router(features[fit], y[fit], inner["AIRSpec"], inner["VIBES"])
        predictions = {method: fit_predict(X[method], y, fit, hold, k[method])
                       for method in METHODS}
        choice = router.predict(features[hold])
        routed = np.where(choice, predictions["VIBES"], predictions["AIRSpec"])
        nested.extend(dict(sample_id=c.sample_id.iloc[row], Site=c.Site.iloc[row],
                           y=y[row], fold=outer, choice="VIBES" if ch else "AIRSpec",
                           AIRSpec=predictions["AIRSpec"][j],
                           VIBES=predictions["VIBES"][j], routed=routed[j])
                      for j, (row, ch) in enumerate(zip(hold, choice)))
    nested = pd.DataFrame(nested)
    if len(nested) != len(train) or nested.sample_id.nunique() != len(train):
        raise ValueError("Nested training-site evaluation coverage changed")
    # Final router sees only training-site spectra and training-site OOF labels.
    inner = oof_predictions(inputs, train, folds=5)
    router = train_router(features[train], y[train], inner["AIRSpec"], inner["VIBES"])
    frozen = inputs["saved"].query("cohort == 'full_pool'")
    test_ids = c.sample_id.iloc[test].to_numpy()
    test_base = {method: frozen.query("method == @method").set_index("sample_id")
                 .loc[test_ids, "prediction"].to_numpy(float) for method in METHODS}
    choice = router.predict(features[test])
    routed = np.where(choice, test_base["VIBES"], test_base["AIRSpec"])
    external = pd.DataFrame(dict(sample_id=test_ids, Site=c.Site.iloc[test].to_numpy(),
                                 y=y[test], choice=np.where(choice, "VIBES", "AIRSpec"),
                                 AIRSpec=test_base["AIRSpec"], VIBES=test_base["VIBES"],
                                 routed=routed))
    metrics = []
    for population, frame in (("training_site_nested", nested), ("prior_inspected_test", external)):
        ref = frame.AIRSpec.to_numpy(float)
        truth = frame.y.to_numpy(float)
        for method in ("AIRSpec", "VIBES", "routed"):
            pred = frame[method].to_numpy(float)
            point, low, high = paired_site_interval(truth, pred, ref, frame.Site.to_numpy())
            metrics.append(dict(population=population, method=method,
                                n_sites=frame.Site.nunique(),
                                selected_VIBES=int(frame.choice.eq("VIBES").sum()) if method=="routed" else np.nan,
                                **score(truth, pred), delta_RMSE_vs_AIRSpec=point,
                                delta_RMSE_CI_low=low, delta_RMSE_CI_high=high))
    metrics = pd.DataFrame(metrics)
    nested.to_csv(output / "routing_nested_training_predictions.csv", index=False)
    external.to_csv(output / "routing_prior_test_predictions.csv", index=False)
    metrics.to_csv(output / "routing_metrics.csv", index=False)
    diagnostic = nested.assign(VIBES_abs_error=(nested.VIBES-nested.y).abs())
    diagnostic.nlargest(10, "VIBES_abs_error").to_csv(output / "routing_largest_VIBES_errors.csv", index=False)
    return metrics, external


def addis_gate(inputs: dict, output: Path = OUTPUT) -> dict:
    """Check for an authoritative independent TOR crosswalk; never use proxies."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = os.environ.get("AETH_ADDIS_VALIDATION_CSV")
    frozen = pd.read_csv(inputs["source"] / "addis_predictions.csv")
    full = frozen.loc[frozen.cohort.eq("full_pool")]
    if len(full) != 506 or full.groupby("method").sample_id.nunique().ne(253).any():
        raise ValueError("Frozen Addis prediction coverage changed")
    receipt = {"status": "reference_unavailable", "n_addis_targets": 253,
               "n_independent_thermal_matches": 0,
               "addis_prediction_sha256": sha(inputs["source"] / "addis_predictions.csv"),
               "reference_path": path,
               "reason": "No identity-confirmed independent Addis thermal EC reference supplied."}
    if path:
        reference_path = Path(path).resolve()
        needed = {"sample_id", "ptfe_filter_id", "quartz_filter_id", "TOR_EC_loading_ug",
                  "reference_method", "reference_source", "authoritative_match",
                  "sampling_equivalent"}
        ref = pd.read_csv(reference_path)
        if not needed.issubset(ref) or ref.sample_id.duplicated().any():
            raise ValueError("Independent reference requires unique identities and declared provenance")
        if (ref[["sample_id", "ptfe_filter_id", "quartz_filter_id", "reference_source"]].isna().any().any()
                or ref.reference_source.astype(str).str.strip().eq("").any()
                or ref.ptfe_filter_id.eq(ref.quartz_filter_id).any()):
            raise ValueError("Missing provenance or indistinct PTFE/quartz physical IDs")
        if not ref.reference_method.eq("TOR").all():
            raise ValueError("Only independently sourced TOR rows qualify")
        if not ref.authoritative_match.eq(True).all() or not ref.sampling_equivalent.eq(True).all():
            raise ValueError("Identity or sampling equivalence is unconfirmed")
        if not np.isfinite(ref.TOR_EC_loading_ug).all():
            raise ValueError("Nonfinite TOR values require an explicit lab handling decision")
        if set(ref.sample_id) - set(full.sample_id):
            raise ValueError("Reference sample ID missing from frozen predictions")
        a = full.loc[full.method.eq("AIRSpec")].set_index("sample_id")
        if any(a.loc[ref.sample_id, "filter_id"].to_numpy() != ref.ptfe_filter_id.to_numpy()):
            raise ValueError("Physical PTFE IDs do not match frozen predictions")
        receipt.update(status="reference_ready_for_separate_locked_analysis",
                       n_independent_thermal_matches=len(ref),
                       reference_sha256=sha(reference_path),
                       reason="Authoritative reference passed schema/identity gate; outcome analysis must use the frozen protocol.")
    (output / "addis_readiness.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def write_manifest(inputs: dict, output: Path = OUTPUT) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_manifest.json").write_text(json.dumps({
        "source_signature": inputs["manifest"]["signature"],
        "source_sha256": inputs["hashes"],
        "design": "Fixed original IMPROVE outer site split; 2,327 shared test filters; "
                  "AIRSpec-based 500-filter analog selection; masks only selection; "
                  "fixed full-pool component counts; spectral-only nested training-site router; "
                  "prior-inspected test exploratory; Addis thermal accuracy gated.",
        "candidate_masks": [name for name, _, _ in MASKS],
    }, indent=2) + "\n")
