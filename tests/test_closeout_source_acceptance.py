"""Saved-fit acceptance checks use artifacts, not just a success-status label."""
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research/ftir_hips_chem/scripts"), str(ROOT / "research/ftir_hips_chem/workflows")]
import test_closeout_workflow_inputs as caller_cases  # noqa: E402
import run_hips_response_20261103 as scorer  # noqa: E402
import vibes_continuum_20261102 as continuum  # noqa: E402
import verify_closeout_source_20261107 as acceptance  # noqa: E402

frozen_inputs = caller_cases.frozen_inputs
baseline_inputs = caller_cases.baseline_inputs


@pytest.fixture
def saved_fits(baseline_inputs, tmp_path, monkeypatch):
    bundle, legacy, background, _ = baseline_inputs
    metadata = pd.read_csv(bundle / "improve_meta.csv")
    # Like the real IMPROVE export, these IDs contain only digits. CSV parsers
    # must not make a string/integer distinction look like a changed identity.
    metadata["filter_id"] = np.arange(3000000, 3000000 + len(metadata)).astype(str)
    mass = pd.read_csv(bundle / "improve_mass.csv")
    mass["filter_id"] = metadata.filter_id
    mass.to_csv(bundle / "improve_mass.csv", index=False)
    metadata["raw_qc_ok"] = True
    metadata.loc[0, "raw_qc_ok"] = False
    metadata.to_csv(bundle / "improve_meta.csv", index=False)
    caller_cases.epo_cases.write_manifest(bundle, [p.name for p in bundle.iterdir() if p.suffix in (".npy", ".csv")])
    corrected = tmp_path / "regenerated"
    monkeypatch.setattr(continuum, "OUT", corrected)
    continuum.prepare(SimpleNamespace(source=bundle, background_model=background))
    manifest = json.loads((corrected / "bundle/MANIFEST.json").read_text())
    outputs = {}
    # Synthetic corrected arrays exercise the saved collection boundary without
    # running any VIBES optimization. All subsequent PLS/EPO fits are real calls.
    for variant in continuum.VARIANTS:
        for group in acceptance.GROUPS:
            name = f"corrected_{variant}_{group}.npy"
            shutil.copyfile(legacy / name, corrected / name)
            outputs[name] = {"sha256": acceptance.cm.sha256_file(corrected / name)}
    diagnostics = pd.DataFrame([dict(variant=variant, row=row, success=True) for row in range(manifest["n_rows"]) for variant in continuum.VARIANTS])
    diagnostics.to_csv(corrected / "fit_diagnostics.csv", index=False)
    outputs["fit_diagnostics.csv"] = {"sha256": acceptance.cm.sha256_file(corrected / "fit_diagnostics.csv")}
    (corrected / "collection_receipt.json").write_text(json.dumps(dict(
        schema_version=2, protocol=continuum.PROTOCOL, status="complete", solver=continuum.SOLVER,
        bundle_manifest_sha256=acceptance.cm.sha256_file(corrected / "bundle/MANIFEST.json"),
        bundle_content_hash=manifest["content_hash"], expected_fits=len(diagnostics), collected_fits=len(diagnostics),
        failed_fits=0, outputs=outputs,
    )))
    baseline = caller_cases.baseline.fit(bundle, models=acceptance.PRIMARY_MODELS, output=tmp_path / "baseline_fit", kmax=3,
                                          continuum_output=corrected, vibes_model=background)
    epo = caller_cases.epo_workflow.fit(bundle, models=("base", "epo"), output=tmp_path / "epo_fit", kmax=3,
                                        continuum_output=corrected)
    scores = tmp_path / "source_scores"
    scorer.source_scores(bundle, baseline_output=baseline, epo_output=epo, scores_output=scores, k=3)
    return dict(bundle=bundle, continuum_output=corrected, baseline_output=baseline, epo_output=epo, scores_output=scores)


def test_acceptance_reconstructs_all_folds_and_targets_but_refits_only_fold_zero(saved_fits, tmp_path, monkeypatch):
    fit = acceptance.ev.fit_training_epo
    calls = []

    def one_refit(X, y, chemistry, rows, kmax, *, qc_mask=None):
        calls.append(dict(rows=np.asarray(rows).copy(), kmax=kmax, mask=qc_mask.copy()))
        return fit(X, y, chemistry, rows, kmax, qc_mask=qc_mask)

    monkeypatch.setattr(acceptance.ev, "fit_training_epo", one_refit)
    output = tmp_path / "acceptance.json"
    result = acceptance.verify(**saved_fits, output=output)
    assert result["status"] == "passed" and result["protocol"] == acceptance.PROTOCOL
    assert json.loads(output.read_text()) == result
    assert len(calls) == 1 and calls[0]["kmax"] == 3
    assert not calls[0]["mask"][0]
    assert result["population"]["n_saved"] == 90
    assert result["population"]["n_tor_domain_before_qc"] == 90
    assert result["population"]["n_eligible"] == result["population"]["n_common"] == 89
    assert result["population"]["n_tor_domain_qc_excluded"] == 1
    assert len(result["epo_folds"]) == 5
    assert all(row["training_test_filter_overlap"] == row["training_test_site_overlap"] == row["qc_excluded_training_rows"] == 0 for row in result["epo_folds"])
    assert set(result["epo_all_source"]["targets"]) == {"addis", "network"}
    assert set(result["baseline_all_source_targets"]) == set(acceptance.PRIMARY_MODELS)
    assert result["on_disk_input_hashes_preserved"] is True
    assert "matched_resolution_metrics" in result["source_scores"]
    perturbation = result["heldout_perturbation"]
    assert perturbation["refitted_folds"] == [0] and perturbation["refit_calls"] == 1
    assert perturbation["training_arrays_exactly_unchanged"] is True
    assert perturbation["artificial_metric_check"]["metrics_changed"] is True
    assert "not scientific performance" in perturbation["artificial_metric_check"]["scope"]
    assert perturbation["comparisons"]["heldout_predictions"]["max_absolute_difference"] <= 1e-10
    assert '"training_filter_ids":' not in output.read_text()
    assert '"test_filter_ids":' not in output.read_text()


def test_rebound_hashes_cannot_hide_wrong_saved_fold_coefficients(saved_fits, tmp_path, monkeypatch):
    path = saved_fits["epo_output"] / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["epo_fold_1|B"][0, 0] += 10
    np.savez_compressed(path, **values)
    fit_path = saved_fits["epo_output"] / "fit_receipt.json"
    fit_receipt = json.loads(fit_path.read_text())
    fit_receipt["source_identity"]["predictions_sha256"] = acceptance.cm.sha256_file(path)
    fit_path.write_text(json.dumps(fit_receipt))
    score_path = saved_fits["scores_output"] / "baseline_scoreboard_receipt.json"
    score_receipt = json.loads(score_path.read_text())
    score_receipt["provenance"]["epo_predictions_sha256"] = acceptance.cm.sha256_file(path)
    score_receipt["provenance"]["epo_fit_receipt_sha256"] = acceptance.cm.sha256_file(fit_path)
    score_path.write_text(json.dumps(score_receipt))

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid saved reconstructions must fail before the acceptance refit")

    monkeypatch.setattr(acceptance.ev, "fit_training_epo", forbidden)
    output = tmp_path / "invalid-acceptance.json"
    with pytest.raises(ValueError, match="fold 1 OOF predictions: numerical reconstruction differs"):
        acceptance.verify(**saved_fits, output=output)
    assert not output.exists()


def test_input_changed_during_refit_cannot_receive_acceptance(saved_fits, tmp_path, monkeypatch):
    fit = acceptance.ev.fit_training_epo

    def changed_input(*args, **kwargs):
        result = fit(*args, **kwargs)
        with (saved_fits["bundle"] / "improve_mass.csv").open("a") as stream:
            stream.write("\n")
        return result

    monkeypatch.setattr(acceptance.ev, "fit_training_epo", changed_input)
    output = tmp_path / "changed-input.json"
    with pytest.raises(ValueError, match="input changed during acceptance"):
        acceptance.verify(**saved_fits, output=output)
    assert not output.exists()


def test_acceptance_never_overwrites_a_saved_receipt(tmp_path):
    output = tmp_path / "acceptance.json"
    output.write_text("preserved receipt")
    with pytest.raises(FileExistsError, match="acceptance output exists"):
        acceptance.verify(*([tmp_path / "missing"] * 5), output)
    assert output.read_text() == "preserved receipt"
