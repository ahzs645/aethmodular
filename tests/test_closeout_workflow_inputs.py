"""Versioned scientific reruns must use declared inputs and preserve prior fits."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research/ftir_hips_chem/scripts"), str(ROOT / "research/ftir_hips_chem/workflows")]
import run_intercept_fixes_20261103 as epo_workflow  # noqa: E402
import run_transfer_diagnosis_20261102 as baseline  # noqa: E402
from candidate_metrics import sha256_file, validate_archive_identity, verified_score_receipt  # noqa: E402
import test_epo_validation as epo_cases  # noqa: E402


PRIMARY_MODELS = ("raw", "spline", "vibes", "vibes_deflate", "vibes_anchorhp")
frozen_inputs = epo_cases.frozen_inputs


@pytest.fixture
def baseline_inputs(frozen_inputs, tmp_path):
    bundle, corrected, frames, matrices, wn = frozen_inputs
    # The real decision grid stores the original float64 model grid as float32.
    # This is a recorded dtype conversion, not a shift or interpolation.
    grid = wn.astype(np.float32)
    np.save(bundle / "wn_full.npy", grid)
    np.save(bundle / "wn_spline.npy", grid)
    np.save(corrected / "bundle/wn.npy", grid)
    for group, values in matrices.items():
        values = np.repeat(values, 2, axis=1)
        for prep, factor in (("raw", 1.0), ("spline", .9), ("vibes", .8)):
            np.save(bundle / f"{group}_{prep}.npy", values * factor)
        np.save(corrected / f"corrected_deflate_{group}.npy", values * .95)
    epo_cases.write_manifest(bundle, [p.name for p in bundle.iterdir() if p.suffix in (".npy", ".csv")])
    epo_cases.write_manifest(corrected / "bundle", ("rows.csv", "wn.npy"), plain_hashes=True)
    background = tmp_path / "background_model.npz"
    rng = np.random.default_rng(42)
    np.savez(background, wn=wn, mean=np.zeros_like(wn), components=rng.normal(size=(len(wn), 4)))
    return bundle, corrected, background, frames


@pytest.mark.parametrize(("kind", "filename"), [
    ("baseline", "predictions.npz"), ("baseline", "coefficients.npz"),
    ("baseline", "fit_receipt.json"), ("baseline", "anchored_network.npy"),
    ("epo", "predictions.npz"), ("epo", "fit_receipt.json"), ("epo", "fold_membership.csv"),
])
def test_existing_complete_or_partial_fit_is_rejected_before_input_loading(tmp_path, monkeypatch, kind, filename):
    destination = tmp_path / "saved"
    destination.mkdir()
    sentinel = destination / filename
    sentinel.write_bytes(b"preserved prior result")

    def forbidden(*args, **kwargs):
        raise AssertionError("existing outputs must be detected before inputs are loaded")

    monkeypatch.setattr(baseline, "load", forbidden)
    fit = baseline.fit if kind == "baseline" else epo_workflow.fit
    with pytest.raises(FileExistsError, match="select a new --output"):
        fit(tmp_path / "missing_bundle", output=destination)
    assert sentinel.read_bytes() == b"preserved prior result"
    assert list(destination.iterdir()) == [sentinel]


@pytest.mark.parametrize("kind", ["baseline", "epo"])
def test_explicit_missing_continuum_variant_fails_before_other_inputs(tmp_path, monkeypatch, kind):
    def forbidden(*args, **kwargs):
        raise AssertionError("a requested missing variant must not be silently omitted")

    monkeypatch.setattr(baseline, "load", forbidden)
    fit = baseline.fit if kind == "baseline" else epo_workflow.fit
    models = ("raw", "vibes_deflate") if kind == "baseline" else ("base", "epo")
    with pytest.raises(FileNotFoundError, match="requested continuum variant"):
        fit(tmp_path / "missing_bundle", models=models, output=tmp_path / "new_fit",
            continuum_output=tmp_path / "missing_continuum")
    assert not (tmp_path / "new_fit").exists()


def test_primary_baseline_and_epo_outputs_integrate_using_explicit_inputs(baseline_inputs, tmp_path, monkeypatch):
    import run_hips_response_20261103 as response

    bundle, corrected, background, frames = baseline_inputs
    monkeypatch.setattr(baseline, "VC", tmp_path / "unavailable_default_continuum")
    monkeypatch.setattr(baseline, "VIBES_MODEL", tmp_path / "unavailable_default_model.npz")

    def forbidden(*args, **kwargs):
        raise AssertionError("the primary candidate set must not compute optional anchored/balanced models")

    monkeypatch.setattr(baseline.ca, "AnchoredBackground", forbidden)
    monkeypatch.setattr(baseline.pd, "qcut", forbidden)
    base_output = baseline.fit(bundle, models=PRIMARY_MODELS, output=tmp_path / "source_cv_v2", kmax=3,
                               continuum_output=corrected, vibes_model=background)
    receipt = json.loads((base_output / "fit_receipt.json").read_text())
    assert receipt["models"] == list(PRIMARY_MODELS)
    assert receipt["k_max"] == 3 and receipt["balanced_weights"] is None
    assert receipt["coefficients_sha256"] == sha256_file(base_output / "coefficients.npz")
    assert receipt["source_code"]["run_transfer_diagnosis_20261102.py"] == sha256_file(Path(baseline.__file__))
    assert receipt["spectral_inputs"][baseline.env.display_path(background)] == sha256_file(background)
    assert not list(base_output.glob("anchored_*.npy"))
    with np.load(base_output / "predictions.npz", allow_pickle=False) as archive:
        assert baseline.prediction_models(archive) == list(PRIMARY_MODELS)
        folds = validate_archive_identity(archive, receipt, frames["improve"], bundle / "improve_meta.csv",
                                           base_output / "predictions.npz")
        assert set(folds) == set(range(5))
        assert all(archive[f"{model}|improve_oof"].shape == (90, 3) for model in PRIMARY_MODELS)
        with np.load(base_output / "coefficients.npz", allow_pickle=False) as coefficients:
            for model in PRIMARY_MODELS:
                variant = baseline.VC_VARIANTS.get(model)
                values = np.load(corrected / f"corrected_{variant}_addis.npy" if variant else bundle / f"addis_{model}.npy")
                restored = (values[:, coefficients[f"{model}|cols"]] - coefficients[f"{model}|xm"]) @ coefficients[f"{model}|B"] + coefficients[f"{model}|ym"]
                np.testing.assert_allclose(restored, archive[f"{model}|addis"], rtol=1e-12, atol=1e-12)

    epo_output = epo_workflow.fit(bundle, models=("base", "epo"), output=tmp_path / "epo_site_cv_v2", kmax=3,
                                  continuum_output=corrected)
    epo_receipt = json.loads((epo_output / "fit_receipt.json").read_text())
    assert epo_receipt["continuum_output"] == baseline.env.display_path(corrected)
    scores_output = tmp_path / "source_scores_v2"
    response.source_scores(bundle, baseline_output=base_output, epo_output=epo_output, scores_output=scores_output, k=3)
    scores = verified_score_receipt(scores_output)
    assert scores["status"] == "corrected" and scores["population"]["n_common"] == 90


def test_omitted_baseline_models_preserve_legacy_candidates_when_continuum_absent(baseline_inputs, tmp_path):
    bundle, _, background, _ = baseline_inputs
    destination = baseline.fit(bundle, output=tmp_path / "legacy_candidates", kmax=2,
                               continuum_output=tmp_path / "absent", vibes_model=background)
    receipt = json.loads((destination / "fit_receipt.json").read_text())
    assert receipt["models"] == list(baseline.BASE_MODELS)
    assert "historical full-source" in receipt["balanced_weights"]
    assert "not training-only" in receipt["balanced_weights"]
    assert len(list(destination.glob("anchored_*.npy"))) == 3


def test_recorded_raw_qc_excludes_training_but_retains_identity_and_predictions(baseline_inputs, tmp_path, monkeypatch):
    bundle, corrected, background, _ = baseline_inputs
    metadata = baseline.pd.read_csv(bundle / "improve_meta.csv")
    metadata["raw_qc_ok"] = True
    metadata.loc[0, "raw_qc_ok"] = False
    metadata.loc[0, "ec_ug"] = 149.5  # In the TOR domain; exclusion must come from QC.
    metadata.loc[0, "tor_ec_ugm3"] = 149.5 / metadata.loc[0, "volume_m3"]
    metadata.to_csv(bundle / "improve_meta.csv", index=False)
    epo_cases.write_manifest(bundle, [p.name for p in bundle.iterdir() if p.suffix in (".npy", ".csv")])
    fit_pls, training_labels = baseline.ca.pls_fit, []

    def record_training(X, y, *args, **kwargs):
        training_labels.append(np.asarray(y).copy())
        return fit_pls(X, y, *args, **kwargs)

    monkeypatch.setattr(baseline.ca, "pls_fit", record_training)
    output = baseline.fit(bundle, models=("raw",), output=tmp_path / "qc_fit", kmax=3,
                          continuum_output=corrected, vibes_model=background)
    assert len(training_labels) == 6
    assert all(not np.any(values == 149.5) for values in training_labels)
    receipt = json.loads((output / "fit_receipt.json").read_text())
    assert receipt["source_domain_rows_before_qc"] == 90
    assert receipt["source_domain_qc_excluded"] == 1 and receipt["fit_rows"] == 89
    assert receipt["source_qc_excluded_filter_ids"] == [metadata.loc[0, "filter_id"]]
    with np.load(output / "predictions.npz", allow_pickle=False) as archive:
        assert archive["improve_filter_ids"].shape == (90,)
        assert archive["improve_raw_qc_ok"].dtype == np.bool_
        assert not archive["improve_raw_qc_ok"][0] and archive["improve_raw_qc_ok"][1:].all()
        assert np.isfinite(archive["raw|improve_oof"]).all()
        folds = validate_archive_identity(archive, receipt, metadata, bundle / "improve_meta.csv", output / "predictions.npz")
        for fold, (_, heldout) in enumerate(epo_workflow.ev.site_folds(metadata.filter_id, metadata.site)):
            np.testing.assert_array_equal(np.flatnonzero(folds == fold), heldout)


def test_background_grid_shift_cannot_pass_as_dtype_rounding(baseline_inputs, tmp_path):
    bundle, corrected, background, _ = baseline_inputs
    with np.load(background, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["wn"] = values["wn"] + .001
    np.savez(background, **values)
    destination = tmp_path / "shifted_grid"
    with pytest.raises(ValueError, match="background grid differs"):
        baseline.fit(bundle, models=("raw",), output=destination, kmax=3,
                     continuum_output=corrected, vibes_model=background)
    assert not destination.exists()


def test_table_model_discovery_accepts_legacy_and_identity_stamped_archives(tmp_path):
    for metadata in ({}, {"improve_filter_ids": np.array(["A", "B"]),
                         "improve_site_ids": np.array(["ONE", "TWO"]), "improve_fold_id": np.array([0, 1])}):
        path = tmp_path / "predictions.npz"
        np.savez(path, **metadata, **{"raw|improve_oof": np.ones((2, 3)), "raw|addis": np.ones((1, 3)),
                                     "vibes|improve_oof": np.ones((2, 3))})
        with np.load(path, allow_pickle=False) as archive:
            assert baseline.prediction_models(archive) == ["raw", "vibes"]


def test_epo_rechecks_every_provenance_file_returned_by_the_verifier(frozen_inputs, tmp_path, monkeypatch):
    bundle, corrected, _, _, _ = frozen_inputs
    receipt = corrected / "collection_receipt.json"
    receipt.write_text("original collection receipt")
    verify = epo_workflow.verify_corrected_spectra

    def with_receipt(*args, **kwargs):
        return {**verify(*args, **kwargs), "collection_receipt.json": sha256_file(receipt)}

    fit_pls = epo_workflow.ca.pls_fit

    def changed_during_fit(*args, **kwargs):
        result = fit_pls(*args, **kwargs)
        receipt.write_text("changed while fitting")
        return result

    monkeypatch.setattr(epo_workflow, "verify_corrected_spectra", with_receipt)
    monkeypatch.setattr(epo_workflow.ca, "pls_fit", changed_during_fit)
    output = tmp_path / "changed_inputs"
    with pytest.raises(RuntimeError, match="input changed during fitting"):
        epo_workflow.fit(bundle, models=("base",), output=output, kmax=3, continuum_output=corrected)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["baseline", "epo"])
def test_outputs_created_by_another_run_during_fitting_are_preserved(baseline_inputs, tmp_path, monkeypatch, kind):
    bundle, corrected, background, _ = baseline_inputs
    destination = tmp_path / "concurrent"
    fit_pls = baseline.ca.pls_fit

    def competing_output(*args, **kwargs):
        result = fit_pls(*args, **kwargs)
        destination.mkdir(exist_ok=True)
        (destination / "predictions.npz").write_bytes(b"other run's result")
        return result

    monkeypatch.setattr(baseline.ca, "pls_fit", competing_output)
    fit = baseline.fit if kind == "baseline" else epo_workflow.fit
    kwargs = {"models": ("raw",), "vibes_model": background} if kind == "baseline" else {"models": ("base",)}
    with pytest.raises(FileExistsError, match="select a new --output"):
        fit(bundle, output=destination, continuum_output=corrected, kmax=3, **kwargs)
    assert (destination / "predictions.npz").read_bytes() == b"other run's result"
    assert len(list(destination.iterdir())) == 1


def test_baseline_cli_forwards_selected_inputs_and_candidate_limits(tmp_path, monkeypatch):
    received = {}

    def capture(bundle, **kwargs):
        received.update(bundle=bundle, **kwargs)

    for name in ("OUT", "VC", "VIBES_MODEL"):
        monkeypatch.setattr(baseline, name, getattr(baseline, name))
    monkeypatch.setattr(baseline, "fit", capture)
    monkeypatch.setattr(sys, "argv", ["baseline", "--bundle", str(tmp_path / "bundle"), "--output", str(tmp_path / "fit"),
                                      "--continuum-output", str(tmp_path / "continuum"), "--vibes-model", str(tmp_path / "model.npz"),
                                      "--models", *PRIMARY_MODELS, "--k-max", "18", "--stages", "fit"])
    baseline.main()
    assert received == dict(bundle=tmp_path / "bundle", output=tmp_path / "fit", continuum_output=tmp_path / "continuum",
                            vibes_model=tmp_path / "model.npz", models=list(PRIMARY_MODELS), kmax=18)


def test_epo_cli_forwards_explicit_continuum_directory(tmp_path, monkeypatch):
    received = {}

    def capture(bundle, **kwargs):
        received.update(bundle=bundle, **kwargs)

    monkeypatch.setattr(epo_workflow, "OUT", epo_workflow.OUT)
    monkeypatch.setattr(baseline, "VC", baseline.VC)
    monkeypatch.setattr(epo_workflow, "fit", capture)
    monkeypatch.setattr(sys, "argv", ["epo", "--bundle", str(tmp_path / "bundle"), "--output", str(tmp_path / "fit"),
                                      "--continuum-output", str(tmp_path / "continuum"), "--models", "base", "epo",
                                      "--k-max", "18", "--stages", "fit"])
    epo_workflow.main()
    assert received == dict(bundle=tmp_path / "bundle", output=tmp_path / "fit", continuum_output=tmp_path / "continuum",
                            models=["base", "epo"], kmax=18)
