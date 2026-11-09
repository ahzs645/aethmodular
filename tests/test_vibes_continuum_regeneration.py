"""Regeneration must preserve identity and never promote partial/mixed batches."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / "research/ftir_hips_chem"
sys.path.insert(0, str(AREA / "scripts"))
from spectral_provenance import verify_corrected_spectra  # noqa: E402


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("continuum_regeneration_under_test", AREA / "workflows/vibes_continuum_20261102.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_bundle(tmp_path, runner, sizes=(3, 2, 3)):
    source = tmp_path / "decision"
    source.mkdir()
    rng = np.random.default_rng(64)
    model_grid = np.linspace(3998.4321, 500.1234, 64)
    np.save(source / "wn_full.npy", model_grid.astype(np.float32))
    for group, count in zip(("improve", "addis", "network"), sizes):
        np.save(source / f"{group}_raw.npy", rng.normal(0.01, 0.001, (count, 64)).astype(np.float32))
        pd.DataFrame({"filter_id": [f"{group}-{i:04d}" for i in range(count)]}).to_csv(source / f"{group}_meta.csv", index=False)
    files = {p.name: {"sha256": runner._sha(p)} for p in source.iterdir()}
    (source / "MANIFEST.json").write_text(json.dumps({"files": files}))
    model = tmp_path / "background.npz"
    np.savez(model, wn=model_grid, mean=np.full(64, 0.005), components=rng.normal(0, 0.001, (64, 4)))
    return source, model


def prepare(tmp_path, runner, sizes=(3, 2, 3)):
    source, model = source_bundle(tmp_path, runner, sizes)
    runner.OUT = tmp_path / "versioned_run"
    runner.prepare(SimpleNamespace(source=source, background_model=model))
    return source, model, runner.OUT / "bundle"


def checkpoint(runner, bundle, *, first=0, directory=None, failed=None):
    manifest = runner._verify_manifest(bundle)
    identity = runner._run_identity(bundle, manifest)
    jobs = runner._jobs(manifest["n_rows"])[first:first + runner.BATCH]
    baseline = np.full((len(jobs), 64), 0.001, np.float32)
    diag = [{"variant": variant, "row": row, "success": (variant, row) != failed, "seconds": 0.1}
            for variant, row in jobs]
    for i, job in enumerate(jobs):
        if job == failed:
            baseline[i] = np.nan
    directory = directory or runner.OUT / "results/local/run/checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"batch_{first:06d}.npz"
    np.savez_compressed(path, first=first, signature=np.array(manifest["content_hash"]),
                        variant=np.array([v for v, _ in jobs]), row=np.array([r for _, r in jobs]),
                        baseline=baseline, diagnostics=np.array(json.dumps(diag)),
                        run_identity=np.array(json.dumps(identity, sort_keys=True)))
    return path


def rewrite_checkpoint(path, **changes):
    with np.load(path, allow_pickle=False) as saved:
        values = {name: saved[name] for name in saved.files}
    values.update(changes)
    np.savez_compressed(path, **values)


def collect(runner, *, allow_failures=False):
    runner.collect(SimpleNamespace(results=None, allow_failures=allow_failures))


def metadata(bundle):
    rows = pd.read_csv(bundle / "rows.csv")
    return {group: part[["filter_id"]].reset_index(drop=True) for group, part in rows.groupby("group", sort=False)}


def test_prepare_checks_source_and_records_storage_grid_identity_without_replacing(tmp_path, runner):
    source, model, bundle = prepare(tmp_path, runner)
    manifest = runner._verify_manifest(bundle)
    assert manifest["schema_version"] == 2
    assert manifest["source"]["background_model_sha256"] == runner._sha(model)
    assert manifest["source"]["source_grid_dtype"] == "float32"
    assert manifest["source"]["background_grid_dtype"] == "float64"
    assert manifest["source"]["grid_identity"] == "exact after source-storage cast; no interpolation"
    assert "code/scripts/vibes_baseline.py" in manifest["files"]
    assert "code/vibes_continuum.py" in manifest["files"]
    assert "environment.json" in manifest["files"]
    assert manifest["solver"] == {"adapter": "vibes_baseline_matrix", "loss": "PB", "tau": 0.1, "maxiter": 10000, "retry_failed": True}
    before = runner._sha(bundle / "MANIFEST.json")
    with pytest.raises(FileExistsError, match="existing run"):
        runner.prepare(SimpleNamespace(source=source, background_model=model))
    assert runner._sha(bundle / "MANIFEST.json") == before


def test_prepare_rejects_changed_raw_input_before_writing(tmp_path, runner):
    source, model = source_bundle(tmp_path, runner)
    runner.OUT = tmp_path / "versioned_run"
    raw = np.load(source / "improve_raw.npy")
    np.save(source / "improve_raw.npy", raw + 0.1)
    with pytest.raises(ValueError, match="source bundle file differs"):
        runner.prepare(SimpleNamespace(source=source, background_model=model))
    assert not runner.OUT.exists()


def test_prepare_rejects_a_close_but_different_background_grid(tmp_path, runner):
    source, model = source_bundle(tmp_path, runner)
    runner.OUT = tmp_path / "versioned_run"
    with np.load(model) as saved:
        fields = {key: saved[key] for key in saved.files}
    fields["wn"][5] += 0.002  # passes the former loose allclose at this wavenumber
    np.savez(model, **fields)
    with pytest.raises(ValueError, match="storage cast"):
        runner.prepare(SimpleNamespace(source=source, background_model=model))


def test_solver_dispatch_keeps_original_settings_and_records_failures(runner):
    calls = []

    def fit(wn, values, background, **kwargs):
        calls.append(kwargs)
        return np.ones_like(values), values - 1, pd.DataFrame([{"success": True}])

    baseline, diag = runner._fit_spectrum("anchorhp", 3, np.ones(64), np.arange(64), None, fit)
    assert calls == [{"sample_ids": ["anchorhp:3"], "tau": 0.1, "loss": "PB", "maxiter": 10000, "retry_failed": True}]
    assert np.isfinite(baseline).all() and diag["success"] is True

    def fail(*args, **kwargs):
        raise RuntimeError("solver failure fixture")

    baseline, diag = runner._fit_spectrum("deflate", 2, np.ones(64), np.arange(64), None, fail)
    assert np.isnan(baseline).all() and diag["success"] is False
    assert "solver failure fixture" in diag["error"]


@pytest.mark.parametrize("defect", ["signature", "membership", "identity", "success_values", "failure_type"])
def test_collection_rejects_mixed_or_malformed_checkpoints(tmp_path, runner, defect):
    _, _, bundle = prepare(tmp_path, runner)
    path = checkpoint(runner, bundle)
    with np.load(path, allow_pickle=False) as saved:
        values = {name: saved[name] for name in saved.files}
    if defect == "signature":
        values["signature"] = np.array("0" * 64)
    elif defect == "membership":
        values["row"][0] = 1
    elif defect == "identity":
        identity = json.loads(str(values["run_identity"]))
        identity["solver"]["maxiter"] = 9999
        values["run_identity"] = np.array(json.dumps(identity))
    elif defect == "success_values":
        values["baseline"][0, 0] = np.nan
    else:
        diag = json.loads(str(values["diagnostics"]))
        diag[0]["success"] = "False"
        values["diagnostics"] = np.array(json.dumps(diag))
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError):
        collect(runner)
    assert not list(runner.OUT.glob("corrected_*.npy"))
    assert not (runner.OUT / "collection_receipt.json").exists()


def test_collection_rejects_missing_and_duplicate_jobs(tmp_path, runner):
    _, _, bundle = prepare(tmp_path, runner, sizes=(3, 2, 56))
    path = checkpoint(runner, bundle)
    with pytest.raises(ValueError, match="2 missing"):
        collect(runner)
    other = runner.OUT / "results/another/run/checkpoints"
    other.mkdir(parents=True)
    shutil.copyfile(path, other / path.name)
    with pytest.raises(ValueError, match="duplicate checkpoint job"):
        collect(runner)
    assert not (runner.OUT / "collection_receipt.json").exists()


def test_collection_commits_verified_outputs_once_and_binds_downstream_hashes(tmp_path, runner):
    _, _, bundle = prepare(tmp_path, runner)
    checkpoint(runner, bundle)
    collect(runner)
    receipt = json.loads((runner.OUT / "collection_receipt.json").read_text())
    assert receipt["expected_fits"] == receipt["collected_fits"] == 16
    assert receipt["failed_fits"] == 0 and receipt["status"] == "complete"
    np.testing.assert_allclose(np.load(runner.OUT / "corrected_deflate_improve.npy"),
                               np.load(bundle / "raw.npy")[:3] - np.float32(0.001), rtol=0, atol=0)
    hashes = verify_corrected_spectra(runner.OUT, metadata(bundle), np.load(bundle / "wn.npy"), runner.VARIANTS)
    assert "collection_receipt.json" in hashes and "fit_diagnostics.csv" in hashes
    with pytest.raises(FileExistsError, match="overwrite"):
        collect(runner)
    path = runner.OUT / "corrected_anchorhp_addis.npy"
    np.save(path, np.load(path) + 0.0001)
    with pytest.raises(ValueError, match="collection receipt"):
        verify_corrected_spectra(runner.OUT, metadata(bundle), np.load(bundle / "wn.npy"), ("anchorhp",))


def test_failed_fits_require_explicit_retention_and_remain_nan(tmp_path, runner):
    _, _, bundle = prepare(tmp_path, runner)
    checkpoint(runner, bundle, failed=("anchorhp", 0))
    with pytest.raises(ValueError, match="1 fits failed"):
        collect(runner)
    failure = json.loads((runner.OUT / "collection_failure.json").read_text())
    assert failure["failed_fits"] == 1 and failure["by_variant"]["anchorhp"]["failed"] == 1
    assert not list(runner.OUT.glob("corrected_*.npy"))
    collect(runner, allow_failures=True)
    assert np.isnan(np.load(runner.OUT / "corrected_anchorhp_improve.npy")[0]).all()
    receipt = json.loads((runner.OUT / "collection_receipt.json").read_text())
    assert receipt["status"] == "complete_with_failures" and receipt["failed_fits"] == 1


def test_new_regenerated_bundle_cannot_be_used_without_collection_receipt(tmp_path, runner):
    _, _, bundle = prepare(tmp_path, runner)
    with pytest.raises(ValueError, match="completed collection receipt"):
        verify_corrected_spectra(runner.OUT, metadata(bundle), np.load(bundle / "wn.npy"), ("anchorhp",))


def test_local_run_resumes_verified_batches_without_refitting(tmp_path, runner, monkeypatch):
    _, _, bundle = prepare(tmp_path, runner)
    calls = []

    def fit(variant, row, y, wn, background, adapter):
        calls.append((variant, row))
        return np.zeros(len(wn), np.float32), {"variant": variant, "row": row, "success": True, "seconds": 0.1}

    monkeypatch.setattr(runner, "_fit_spectrum", fit)
    args = SimpleNamespace(rows="", workers=1)
    runner.local(args)
    assert calls == runner._jobs(8)
    calls.clear()
    runner.local(args)
    assert calls == []
    status = json.loads((runner.OUT / "results/local/run/execution_status.json").read_text())
    assert status["resumed"] == 16 and status["failed"] == 0 and status["full_bundle"] is True
    path = runner.OUT / "results/local/run/checkpoints/batch_000000.npz"
    rewrite_checkpoint(path, signature=np.array("1" * 64))
    with pytest.raises(ValueError, match="bundle signature"):
        runner.local(args)
    assert calls == []
