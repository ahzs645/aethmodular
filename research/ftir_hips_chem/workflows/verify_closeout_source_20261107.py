"""Verify saved closeout fits and refit only fold 0 under artificial label changes.

This acceptance step runs after the continuum collection, baseline/EPO fits and
fixed-k source scorer exist. It does not select settings or rerun other folds.
All inputs stay read-only; --output names one new JSON acceptance receipt.

The perturbed-label metrics are an artificial integrity check, not scientific
performance estimates. Reconstructed target predictions are the separate saved
all-source models; this verification does not validate absolute target EC.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path[:0] = [str(AREA / "scripts"), str(AREA / "workflows"), str(ROOT)]
from aethmodular_cli import env  # noqa: E402
import candidate_metrics as cm  # noqa: E402
import continuum_anchor as ca  # noqa: E402
import epo_validation as ev  # noqa: E402
import run_intercept_fixes_20261103 as ix  # noqa: E402
import run_transfer_diagnosis_20261102 as td  # noqa: E402
import spectral_provenance as sp  # noqa: E402

PROTOCOL = "closeout_source_artifact_acceptance_v1"
PRIMARY_MODELS = ("raw", "spline", "vibes", "vibes_deflate", "vibes_anchorhp")
GROUPS = ("improve", "addis", "network")
ATOL = RTOL = 1e-10  # Fixed numerical acceptance bounds; not tuned on results.
NO_EXPECTED_HASH = object()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def new_output(path):
    if Path(path).exists() or Path(path).is_symlink():
        raise FileExistsError(f"acceptance output exists: {env.display_path(path)}; choose a new --output")


def recorded_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


class Inputs:
    """Bind every inspected input to its original bytes and recheck at the end."""

    def __init__(self):
        self.hashes = {}

    def add(self, path, expected=NO_EXPECTED_HASH):
        path = Path(path).resolve()
        if path not in self.hashes:
            self.hashes[path] = cm.sha256_file(path)
        actual = self.hashes[path]
        if expected is not NO_EXPECTED_HASH:
            require(isinstance(expected, str) and actual == expected,
                    f"input hash differs: {env.display_path(path)}")
        return actual

    def read_json(self, path):
        self.add(path)
        return json.loads(Path(path).read_text())

    def manifest(self, directory):
        directory = Path(directory)
        manifest = self.read_json(directory / "MANIFEST.json")
        files = manifest.get("files")
        require(isinstance(files, dict) and bool(files), "frozen manifest has no files")
        for name, entry in files.items():
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts,
                    f"unsafe manifest entry: {name}")
            self.add(directory / relative, entry.get("sha256") if isinstance(entry, dict) else entry)
        return manifest

    def recorded(self, mapping):
        require(isinstance(mapping, dict) and bool(mapping), "fit receipt lacks recorded input hashes")
        for path, digest in mapping.items():
            self.add(recorded_path(path), digest)

    def unchanged(self):
        for path, expected in self.hashes.items():
            require(cm.sha256_file(path) == expected,
                    f"input changed during acceptance; no receipt published: {env.display_path(path)}")

    def display(self):
        return {env.display_path(path): digest for path, digest in sorted(self.hashes.items())}


def exact(name, actual, expected):
    require(np.array_equal(np.asarray(actual), np.asarray(expected)), f"{name}: identity/order differs")


def numerical(name, actual, expected):
    """Report exact equality when achieved and the actual error under fixed bounds."""
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    require(actual.shape == expected.shape, f"{name}: shape differs")
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), f"{name}: infinite values")
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), f"{name}: missing-value pattern differs")
    finite = np.isfinite(actual)
    delta = np.abs(actual[finite] - expected[finite])
    allowed = ATOL + RTOL * np.abs(expected[finite])
    max_abs = float(delta.max()) if delta.size else 0.0
    max_fraction = float((delta / allowed).max()) if delta.size else 0.0
    require(max_fraction <= 1.0,
            f"{name}: numerical reconstruction differs (max absolute difference {max_abs:.12g}; "
            f"maximum fraction of fixed tolerance {max_fraction:.12g})")
    return dict(exact_equal=bool(np.array_equal(actual, expected, equal_nan=True)),
                max_absolute_difference=max_abs, max_fraction_of_tolerance=max_fraction,
                n_finite=int(finite.sum()), n_missing=int(np.isnan(actual).sum()))


def restored_model(archive, prefix, values, *, epo):
    X = np.asarray(values, float)
    xm, ym, coefficients = (archive[f"{prefix}|{name}"] for name in ("xm", "ym", "B"))
    require(coefficients.ndim == 2 and xm.shape == (X.shape[1],) and ym.shape == (),
            f"{prefix}: saved PLS dimensions differ")
    require(coefficients.shape[0] == X.shape[1], f"{prefix}: saved PLS channel count differs")
    require(all(np.isfinite(value).all() for value in (xm, ym, coefficients)), f"{prefix}: nonfinite fitted parameters")
    valid = np.isfinite(X).all(axis=1)
    transformed = X[valid]
    if epo:
        basis = archive[f"{prefix}|basis"]
        require(basis.ndim == 2 and basis.shape[0] == X.shape[1] and np.isfinite(basis).all(),
                f"{prefix}: saved EPO basis dimensions/values differ")
        numerical(f"{prefix} orthonormal basis", basis.T @ basis, np.eye(basis.shape[1]))
        transformed = transformed - (transformed @ basis) @ basis.T
    result = np.full((len(X), coefficients.shape[1]), np.nan)
    result[valid] = (transformed - xm) @ coefficients + ym
    return result


def model_lineage(archive, prefix, receipt, X, y, chemistry, ids, sites, train, test, qc_mask):
    selected = ev.training_rows(X, y, train, qc_mask=qc_mask)
    projection_rows = selected[np.isfinite(chemistry[selected]).all(axis=1)]
    expected = {
        "training_filter_ids": ids[selected], "projection_training_filter_ids": ids[projection_rows],
        "test_filter_ids": ids[test], "training_sites": sorted(set(sites[selected])),
        "test_sites": sorted(set(sites[test])),
    }
    for name, values in expected.items():
        exact(f"{prefix} {name}", receipt.get(name), values)
    exact(f"{prefix} archived training IDs", archive[f"{prefix}|training_ids"], ids[selected])
    exact(f"{prefix} archived projection IDs", archive[f"{prefix}|projection_training_ids"], ids[projection_rows])
    require(not np.intersect1d(selected, test).size and not np.intersect1d(projection_rows, test).size,
            f"{prefix}: training/test physical-filter overlap")
    require(not set(sites[selected]) & set(sites[test]), f"{prefix}: training/test site overlap")
    require(qc_mask[selected].all() and qc_mask[projection_rows].all(), f"{prefix}: QC-excluded training record")
    for name, actual in {
        "n_train": len(selected), "n_projection_train": len(projection_rows), "n_test": len(test),
        "projection_rank": archive[f"{prefix}|basis"].shape[1],
        "k_max": archive[f"{prefix}|B"].shape[1], "chemistry_design_rank": len(ev.CHEMISTRY_COLUMNS),
        "training_spectra_sha256": ev.array_sha256(X[selected]),
        "training_tor_sha256": ev.array_sha256(y[selected]),
        "projection_chemistry_sha256": ev.array_sha256(chemistry[projection_rows]),
        "basis_sha256": ev.array_sha256(archive[f"{prefix}|basis"]),
    }.items():
        require(receipt.get(name) == actual, f"{prefix}: saved {name} differs")
    return dict(n_train=len(selected), n_projection_train=len(projection_rows), n_test=len(test),
                n_training_sites=len(set(sites[selected])), n_test_sites=len(set(sites[test])),
                training_test_filter_overlap=0, training_test_site_overlap=0, qc_excluded_training_rows=0,
                training_filter_ids_sha256=cm.ids_sha256(ids[selected]),
                projection_training_filter_ids_sha256=cm.ids_sha256(ids[projection_rows]),
                test_filter_ids_sha256=cm.ids_sha256(ids[test]), recorded_training_hashes_match=True)


def compare_table(name, actual, expected):
    require(len(actual) == len(expected), f"{name}: row count differs")
    differences = []
    for column in expected:
        if column == "evaluation_status":
            continue  # The scorer explicitly promotes the verified EPO row.
        require(column in actual, f"{name}: missing column {column}")
        if column in ("filter_id", "site", "model"):
            exact(f"{name}.{column}", actual[column].astype(str), expected[column].astype(str))
        elif pd.api.types.is_numeric_dtype(expected[column]):
            differences.append(numerical(f"{name}.{column}", actual[column], expected[column]))
        else:
            exact(f"{name}.{column}", actual[column].fillna(""), expected[column].fillna(""))
    return dict(rows=len(expected), checked_columns=len(expected.columns) - ("evaluation_status" in expected),
                max_absolute_difference=max((v["max_absolute_difference"] for v in differences), default=0.0))


def score_checks(inputs, directory, receipt, baseline, epo, im, folds):
    k = receipt.get("population", {}).get("k")
    require(isinstance(k, int) and not isinstance(k, bool) and k > 0, "source-score receipt has no fixed positive k")
    predictions = {}
    for model in (*PRIMARY_MODELS, "epo"):
        values = (epo if model == "epo" else baseline)[f"{model}|improve_oof"]
        require(values.ndim == 2 and values.shape[0] == len(im) and values.shape[1] >= k,
                f"{model}: prediction archive does not contain the declared fixed-k population")
        predictions[model] = values[:, k - 1]
    expected, population, ledger = cm.paired_candidate_metrics(im, predictions, k=k)
    for name, value in population.items():
        require(receipt["population"].get(name) == value, f"source-score population differs: {name}")
    ledger["fold_id"] = folds
    actual = pd.read_csv(directory / "baseline_scoreboard.csv").set_index("model")
    require(actual.index.is_unique and set(actual.index) == set(predictions), "source-score candidates differ")
    expected = expected.set_index("model")
    checks = {
        "primary_metrics": compare_table("primary scores", actual.loc[expected.index], expected),
        "primary_ledger": compare_table("primary ledger", pd.read_csv(directory / "baseline_scoreboard_filters.csv", dtype={"filter_id": str, "site": str}), ledger),
    }
    require(actual.loc["epo", "evaluation_status"] == "corrected", "EPO score is not marked corrected")
    control = receipt.get("matched_resolution_control")
    if control is not None:
        controls = {"anchorhp_bin2_base": epo["base|improve_oof"][:, k - 1], "epo_bin2": predictions["epo"]}
        expected, control_population, control_ledger = cm.paired_candidate_metrics(im, controls, k=k)
        for name, value in control_population.items():
            if name != "common_filter_ids":
                require(control.get("population", {}).get(name) == value, f"matched-resolution population differs: {name}")
        control_ledger["fold_id"] = folds
        actual = pd.read_csv(directory / cm.EPO_RESOLUTION_CONTROL_FILES[0]).set_index("model")
        expected = expected.set_index("model")
        require(actual.index.is_unique and set(actual.index) == set(controls), "matched-resolution candidates differ")
        checks["matched_resolution_metrics"] = compare_table("matched-resolution scores", actual.loc[expected.index], expected)
        checks["matched_resolution_ledger"] = compare_table("matched-resolution ledger", pd.read_csv(directory / cm.EPO_RESOLUTION_CONTROL_FILES[1], dtype={"filter_id": str, "site": str}), control_ledger)
    for name, entry in receipt["outputs"].items():
        inputs.add(directory / name, entry["sha256"])
    return k, checks


def artificial_metrics(y, prediction, qc_mask):
    eligible = qc_mask & np.isfinite(y) & (y < 150) & np.isfinite(prediction)
    residual = prediction[eligible] - y[eligible]
    require(len(residual) > 0, "artificial integrity metric has no eligible records")
    return dict(n=int(eligible.sum()), median_absolute_error=float(np.median(np.abs(residual))),
                mean_absolute_error=float(np.mean(np.abs(residual))),
                rmse=float(np.sqrt(np.mean(residual ** 2))), mean_signed_bias=float(np.mean(residual)))


def perturb_fold_zero(archive, receipt, X, y, chemistry, ids, sites, splits, qc_mask, k):
    train, heldout = splits[0]
    spectrum_hash = ev.array_sha256(X)
    qc_hash = hashlib.sha256(qc_mask.astype(np.uint8).tobytes()).hexdigest()
    require(len(heldout) >= 5, "fold 0 needs at least five held-out rows for the declared perturbation")
    changed_y, changed_chemistry = y.copy(), chemistry.copy()
    pattern = np.array([0.5, 149.5, 150.0, 10000.0, np.nan])
    changed_y[heldout] = np.resize(pattern, len(heldout))
    changed_chemistry[heldout, 0] = changed_y[heldout]
    for column in range(1, len(ev.CHEMISTRY_COLUMNS)):
        changed_chemistry[heldout, column] = (column + 1) * 1000.0 + np.arange(len(heldout))
    changed_chemistry[heldout[1], 2] = np.nan
    require(np.array_equal(y[train], changed_y[train], equal_nan=True)
            and np.array_equal(chemistry[train], changed_chemistry[train], equal_nan=True),
            "perturbation changed training labels or chemistry")
    expected_splits = ev.site_folds(ids, sites)
    for original, current in zip(splits, expected_splits):
        exact("unchanged split training rows", original[0], current[0])
        exact("unchanged split held-out rows", original[1], current[1])
    kmax = archive["epo_fold_0|B"].shape[1]
    # This is the acceptance script's only model-fitting call.
    fitted = ev.fit_training_epo(X, changed_y, changed_chemistry, train, kmax, qc_mask=qc_mask)
    values = ev.model_arrays("epo_fold_0", fitted, ids)
    comparisons = {}
    for suffix in ("basis", "signatures", "chemistry_mean", "spectrum_mean", "xm", "ym", "B"):
        key = f"epo_fold_0|{suffix}"
        comparisons[suffix] = numerical(f"perturbed fold 0 {suffix}", values[key], archive[key])
    for suffix in ("training_ids", "projection_training_ids"):
        exact(f"perturbed fold 0 {suffix}", values[f"epo_fold_0|{suffix}"], archive[f"epo_fold_0|{suffix}"])
    require(fitted.projection.rank == receipt["projection_rank"]
            and fitted.projection.chemistry_rank == receipt["chemistry_design_rank"], "perturbed projection rank differs")
    comparisons["rank_tolerance"] = numerical("perturbed rank tolerance", fitted.projection.tolerance, receipt["rank_tolerance"])
    original_predictions = archive["epo|improve_oof"][heldout]
    changed_predictions = fitted.predict(X[heldout])
    comparisons["heldout_predictions"] = numerical("perturbed held-out predictions", changed_predictions, original_predictions)
    require(ev.array_sha256(X) == spectrum_hash, "source spectra changed during perturbation refit")
    require(hashlib.sha256(qc_mask.astype(np.uint8).tobytes()).hexdigest() == qc_hash, "fixed QC mask changed during perturbation refit")
    require(np.array_equal(y[train], changed_y[train], equal_nan=True)
            and np.array_equal(chemistry[train], changed_chemistry[train], equal_nan=True),
            "training labels or chemistry changed during perturbation refit")
    for original, current in zip(splits, expected_splits):
        exact("preserved split training rows", original[0], current[0])
        exact("preserved split held-out rows", original[1], current[1])
    before = artificial_metrics(y[heldout], original_predictions[:, k - 1], qc_mask[heldout])
    after = artificial_metrics(changed_y[heldout], changed_predictions[:, k - 1], qc_mask[heldout])
    require(before != after, "artificial label perturbation did not change the evaluation metric")
    return dict(
        fold_id=0, refitted_folds=[0], refit_calls=1, k_max=kmax, n_heldout=len(heldout),
        fixed_qc_mask_sha256=qc_hash, source_spectra_sha256=spectrum_hash,
        heldout_filter_ids_sha256=cm.ids_sha256(ids[heldout]), training_arrays_exactly_unchanged=True,
        spectra_unchanged=True, splits_exactly_unchanged=True,
        tor_replacement_cycle=[0.5, 149.5, 150.0, 10000.0, "NaN"],
        nuisance_replacement="held-out chemistry columns j=1..4 become (j+1)*1000 + held-out row offset; one other_PM becomes NaN",
        comparisons=comparisons,
        artificial_metric_check=dict(
            scope="Artificial held-out-label integrity test only; these are not scientific performance estimates",
            k=k, units="micrograms per filter",
            rule="QC-pass held-out records with finite reference below 150 and finite prediction",
            original_labels=before, perturbed_labels=after, metrics_changed=True),
    )


def verify(bundle, continuum_output, baseline_output, epo_output, scores_output, output):
    bundle, continuum_output, baseline_output, epo_output, scores_output, output = map(
        Path, (bundle, continuum_output, baseline_output, epo_output, scores_output, output))
    new_output(output)
    inputs = Inputs()
    inputs.manifest(bundle)
    continuum_manifest = inputs.manifest(continuum_output / "bundle")
    require(continuum_manifest.get("schema_version") == 2 and continuum_manifest.get("protocol") == "vibes_continuum_regeneration_v2",
            "acceptance requires the versioned continuum regeneration manifest and collection")
    require(continuum_manifest.get("content_hash") == hashlib.sha256(json.dumps(continuum_manifest["files"], sort_keys=True).encode()).hexdigest(),
            "continuum manifest content hash differs")
    source = continuum_manifest.get("source", {})
    require({"MANIFEST.json", "wn_full.npy", *[f"{group}_{kind}" for group in GROUPS for kind in ("raw.npy", "meta.csv")]}.issubset(source.get("decision_files", {})),
            "continuum source identity lacks the decision bundle inputs")
    for name, digest in source["decision_files"].items():
        inputs.add(bundle / name, digest)
    inputs.add(continuum_output / "bundle/background_model.npz", source.get("background_model_sha256"))
    im, ad, nw = td.load(bundle)
    metadata = dict(zip(GROUPS, (im, ad, nw)))
    for group, frame in metadata.items():
        cm.validate_physical_ids(frame.filter_id, label=group)
    wn = np.load(bundle / "wn_full.npy", allow_pickle=False)
    for name, digest in sp.verify_corrected_spectra(continuum_output, metadata, wn, ("deflate", "anchorhp")).items():
        inputs.add(continuum_output / name, digest)
    collection = inputs.read_json(continuum_output / "collection_receipt.json")
    require(collection.get("status") == "complete" and collection.get("failed_fits") == 0,
            "this primary closeout acceptance requires a complete collection with no failed fits")
    matrices = {}
    for group in GROUPS:
        matrices[group], binned_grid = ix.bin2(np.load(continuum_output / f"corrected_anchorhp_{group}.npy", mmap_mode="r", allow_pickle=False), wn)
        require(np.isfinite(matrices[group]).all(), f"{group}: nonfinite primary corrected spectra")
        matrices[group].flags.writeable = False
    X = matrices["improve"]
    y, ids, sites = im.tor_ug.to_numpy(float), im.filter_id.astype(str).to_numpy(), im.site.astype(str).to_numpy()
    chemistry = ev.chemistry_matrix(im, pd.read_csv(bundle / "improve_mass.csv"))
    qc_mask, qc_policy = cm.recorded_source_qc(im)
    require(qc_policy["flag_present"], "closeout acceptance requires the decision bundle's recorded raw_qc_ok flags")
    for values in (y, chemistry, qc_mask):
        values.flags.writeable = False
    baseline_receipt = inputs.read_json(baseline_output / "fit_receipt.json")
    epo_receipt = inputs.read_json(epo_output / "fit_receipt.json")
    require(epo_receipt.get("epo_protocol") == ev.EPO_PROTOCOL, "EPO fit protocol is not training-only site CV")
    tor_domain = np.isfinite(y) & (y < 150)
    for name, receipt in (("baseline", baseline_receipt), ("EPO", epo_receipt)):
        require(receipt.get("source_qc_policy") == qc_policy, f"{name}: recorded QC policy differs")
        for key, value in {"source_domain_rows_before_qc": int(tor_domain.sum()),
                           "source_domain_qc_excluded": int((tor_domain & ~qc_mask).sum()),
                           "fit_rows": int((tor_domain & qc_mask).sum()),
                           "source_qc_excluded_filter_ids_sha256": cm.ids_sha256(ids[~qc_mask])}.items():
            require(receipt.get(key) == value, f"{name}: recorded QC calibration count/identity differs: {key}")
    inputs.recorded(baseline_receipt.get("spectral_inputs"))
    inputs.recorded(epo_receipt.get("inputs"))
    require(inputs.add(recorded_path(baseline_receipt["vibes_model"])) == source["background_model_sha256"], "baseline and continuum background model identities differ")
    baseline_path, epo_path = baseline_output / "predictions.npz", epo_output / "predictions.npz"
    inputs.add(baseline_path)
    inputs.add(epo_path)
    inputs.add(baseline_output / "coefficients.npz", baseline_receipt.get("coefficients_sha256"))
    inputs.add(epo_output / "fold_membership.csv", epo_receipt.get("fold_membership_sha256"))
    score_receipt = inputs.read_json(scores_output / "baseline_scoreboard_receipt.json")
    require(cm.verified_score_receipt(scores_output).get("status") == "corrected", "source-score outputs/receipt are not verified")
    for name, entry in score_receipt["outputs"].items():
        inputs.add(scores_output / name, entry["sha256"])
    require(score_receipt.get("provenance", {}).get("epo_protocol") == ev.EPO_PROTOCOL, "source-score EPO protocol differs")
    bindings = {
        "epo_predictions_sha256": epo_path, "epo_fit_receipt_sha256": epo_output / "fit_receipt.json",
        "baseline_predictions_sha256": baseline_path, "baseline_fit_receipt_sha256": baseline_output / "fit_receipt.json",
        "improve_metadata_sha256": bundle / "improve_meta.csv", "wavenumber_grid_sha256": bundle / "wn_full.npy",
        "decision_manifest_sha256": bundle / "MANIFEST.json", "scorer_source_sha256": AREA / "workflows/run_hips_response_20261103.py",
    }
    for key, path in bindings.items():
        require(score_receipt["provenance"].get(key) == inputs.add(path), f"source-score provenance differs: {key}")
    code = [Path(__file__), Path(td.__file__), Path(ix.__file__), Path(ev.__file__), Path(ca.__file__), Path(cm.__file__), Path(sp.__file__), bindings["scorer_source_sha256"]]
    code_by_name = {path.name: path for path in code}
    for name, receipt in (("baseline", baseline_receipt), ("EPO", epo_receipt)):
        require(bool(receipt.get("source_code")), f"{name}: fit source-code hashes are missing")
        for filename, digest in receipt["source_code"].items():
            require(filename in code_by_name, f"{name}: unrecognized fitting source {filename}")
            inputs.add(code_by_name[filename], digest)
    source_code = {env.display_path(path): inputs.add(path) for path in code}
    for name in ("uv.lock", "pyproject.toml"):
        inputs.add(ROOT / name)
    with np.load(baseline_path, allow_pickle=False) as baseline, np.load(epo_path, allow_pickle=False) as epo, np.load(baseline_output / "coefficients.npz", allow_pickle=False) as coefficients:
        folds = cm.validate_archive_identity(baseline, baseline_receipt, im, bundle / "improve_meta.csv", baseline_path, label="baseline")
        exact("baseline/EPO frozen folds", cm.validate_archive_identity(epo, epo_receipt, im, bundle / "improve_meta.csv", epo_path, label="EPO"), folds)
        exact("pair-averaged spectral grid", epo["wn_bin2"], binned_grid)
        splits = ev.site_folds(ids, sites)
        for fold, (_, heldout) in enumerate(splits):
            exact(f"fold {fold} frozen held-out rows", np.flatnonzero(folds == fold), heldout)
        membership = pd.read_csv(epo_output / "fold_membership.csv", dtype={"filter_id": str, "site": str})
        for column, values in {"filter_id": ids, "site": sites, "fold_id": folds, "raw_qc_ok": qc_mask,
                               "source_domain_before_qc": tor_domain, "source_domain": tor_domain & qc_mask,
                               "spectra_available": np.isfinite(X).all(axis=1)}.items():
            exact(f"EPO fold-membership {column}", membership[column], values)
        require(len(epo_receipt.get("epo", {}).get("folds", [])) == 5, "EPO fit receipt must record all five outer folds")
        fold_checks = []
        for fold, (train, heldout) in enumerate(splits):
            saved = epo_receipt["epo"]["folds"][fold]
            require(saved.get("fold_id") == fold and saved.get("role") == "outer_training_fold", "EPO fold roles/order differ")
            prefix = f"epo_fold_{fold}"
            checks = model_lineage(epo, prefix, saved, X, y, chemistry, ids, sites, train, heldout, qc_mask)
            checks.update(fold_id=fold, oof_reconstruction=numerical(f"fold {fold} OOF predictions", restored_model(epo, prefix, X[heldout], epo=True), epo["epo|improve_oof"][heldout]))
            fold_checks.append(checks)
        final = epo_receipt["epo"]["final_model"]
        require(final.get("role") == "all_source_target_model" and final.get("fold_id") is None, "EPO final model role differs")
        final_checks = model_lineage(epo, "epo", final, X, y, chemistry, ids, sites, np.arange(len(ids)), np.array([], int), qc_mask)
        final_checks["targets"] = {group: numerical(f"EPO final {group}", restored_model(epo, "epo", matrices[group], epo=True), epo[f"epo|{group}"]) for group in ("addis", "network")}
        require(set(baseline_receipt.get("models", [])) == set(PRIMARY_MODELS), "acceptance is limited to the declared five primary baseline candidates")
        baseline_targets = {}
        for model in PRIMARY_MODELS:
            variant = td.VC_VARIANTS.get(model)
            baseline_targets[model] = {}
            for group in ("addis", "network"):
                path = continuum_output / f"corrected_{variant}_{group}.npy" if variant else bundle / f"{group}_{model}.npy"
                inputs.add(path)
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                columns = coefficients[f"{model}|cols"]
                require(columns.dtype == np.bool_ and columns.shape == (values.shape[1],), f"{model}: saved baseline channel selection differs")
                baseline_targets[model][group] = numerical(f"{model} final {group}", restored_model(coefficients, model, values[:, columns], epo=False), baseline[f"{model}|{group}"])
        k, scores = score_checks(inputs, scores_output, score_receipt, baseline, epo, im, folds)
        perturbation = perturb_fold_zero(epo, epo_receipt["epo"]["folds"][0], X, y, chemistry, ids, sites, splits, qc_mask, k)
    inputs.unchanged()
    tor_domain = np.isfinite(y) & (y < 150)
    result = dict(
        schema_version=1, protocol=PROTOCOL, status="passed", created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        paths={name: env.display_path(path) for name, path in dict(bundle=bundle, continuum_output=continuum_output, baseline_output=baseline_output, epo_output=epo_output, scores_output=scores_output).items()},
        source_code=source_code, input_hashes=inputs.display(), on_disk_input_hashes_preserved=True,
        tolerances=dict(atol=ATOL, rtol=RTOL, rule="abs(actual - saved) <= atol + rtol * abs(saved); exact equality is reported separately"),
        runtime=dict(python=platform.python_version(), packages={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
                     threads={name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}),
        protocols=dict(epo=ev.EPO_PROTOCOL, scores=cm.SCORE_PROTOCOL, source_qc=qc_policy["protocol"]),
        population=dict(n_saved=len(ids), n_saved_sites=len(set(sites)), n_common_sites=score_receipt["population"]["n_sites"],
                        n_tor_domain_before_qc=int(tor_domain.sum()),
                        n_qc_excluded=int((~qc_mask).sum()), n_tor_domain_qc_excluded=int((tor_domain & ~qc_mask).sum()),
                        n_eligible=int((tor_domain & qc_mask).sum()), n_common=score_receipt["population"]["n_common"],
                        filter_ids_sha256=cm.ids_sha256(ids), fold_id_sha256=ev.array_sha256(folds), source_qc_policy=qc_policy),
        collection=dict(expected_fits=collection["expected_fits"], collected_fits=collection["collected_fits"], failed_fits=collection["failed_fits"], manifest_and_output_hashes_verified=True),
        same_frozen_site_folds=True, epo_folds=fold_checks, epo_all_source=final_checks,
        baseline_all_source_targets=baseline_targets, source_scores=scores, heldout_perturbation=perturbation,
        limits=["Only EPO fold 0 is refitted; the other EPO folds are reconstructed from saved transforms/coefficients.",
                "Baseline OOF identities, folds, QC policy and score inputs are checked; baseline fold coefficients were not saved for independent reconstruction.",
                "Perturbed-label metrics are an artificial integrity test, not scientific performance estimates.",
                "Target predictions use separate all-source models; this receipt does not establish absolute Addis EC accuracy or independent model-selection validation."],
    )
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    new_output(output)
    with tempfile.TemporaryDirectory(prefix="acceptance-stage-", dir=output.parent) as directory:
        staged = Path(directory) / "receipt.json"
        staged.write_text(encoded)
        os.link(staged, output)  # Publish complete bytes once; never replace another receipt.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "continuum-output", "baseline-output", "epo-output", "scores-output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON acceptance receipt; existing files are never overwritten")
    args = parser.parse_args()
    result = verify(args.bundle, args.continuum_output, args.baseline_output, args.epo_output, args.scores_output, args.output)
    print(f"Acceptance passed: {result['population']['n_common']} common filters; five EPO folds reconstructed; only fold 0 refitted")
    print(env.display_path(args.output))


if __name__ == "__main__":
    main()
