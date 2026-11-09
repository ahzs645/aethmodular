"""The saved binned baseline is a separate, integrity-bound EPO control."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research/ftir_hips_chem/scripts"), str(ROOT / "research/ftir_hips_chem/workflows")]
import run_hips_response_20261103 as response  # noqa: E402
from candidate_metrics import (  # noqa: E402
    EPO_RESOLUTION_CONTROL_FILES, EPO_RESOLUTION_CONTROL_PROTOCOL, ids_sha256, sha256_file,
    verified_score_receipt,
)

PRIMARY_MODELS = ("raw", "spline", "vibes", "vibes_deflate", "vibes_anchorhp", "epo")
OUTPUT_NAMES = ("baseline_scoreboard.csv", "baseline_scoreboard_filters.csv", *EPO_RESOLUTION_CONTROL_FILES,
                "baseline_scoreboard_receipt.json")


@pytest.fixture
def score_inputs(tmp_path, monkeypatch):
    # Saved arrays with deliberately different full-grid and binned baseline
    # errors isolate scoring and publication from the separately tested fitter.
    bundle, baseline, epo = (tmp_path / name for name in ("bundle", "baseline", "epo"))
    for directory in (bundle, baseline, epo):
        directory.mkdir()
    reference = np.arange(1., 51.)
    metadata = pd.DataFrame(dict(filter_id=[f"F{i:03d}" for i in range(50)],
                                 site=np.repeat([f"SITE{i}" for i in range(10)], 5),
                                 tor_ug=reference, ocec=2.))
    metadata.loc[0, "tor_ug"] = np.nan
    metadata.loc[1, "tor_ug"] = 150.
    metadata.to_csv(bundle / "improve_meta.csv", index=False)
    grid = np.linspace(4000, 500, 2722, dtype=np.float32)
    np.save(bundle / "wn_full.npy", grid)
    (bundle / "MANIFEST.json").write_text(json.dumps({"files": {
        name: {"sha256": sha256_file(bundle / name)} for name in ("improve_meta.csv", "wn_full.npy")}}))
    identities = {"improve_filter_ids": metadata.filter_id.to_numpy().astype(str),
                  "improve_site_ids": metadata.site.to_numpy().astype(str),
                  "improve_fold_id": np.repeat(np.arange(5), 10)}
    for directory, offsets in ((baseline, dict(raw=5., spline=4., vibes=3., vibes_deflate=2.5, vibes_anchorhp=10.)),
                               (epo, dict(base=2., epo=.5))):
        arrays = dict(identities)
        for model, offset in offsets.items():
            arrays[f"{model}|improve_oof"] = np.repeat((reference + 99)[:, None], 18, axis=1)
            arrays[f"{model}|improve_oof"][:, 17] = reference + offset
        if directory == epo:
            arrays["wn_bin2"] = grid.reshape(-1, 2).mean(axis=1)
        np.savez(directory / "predictions.npz", **arrays)
        receipt = dict(source_identity=dict(protocol="physical_filter_order_and_site_folds_v1",
                                            filter_ids_sha256=ids_sha256(metadata.filter_id),
                                            improve_metadata_sha256=sha256_file(bundle / "improve_meta.csv"),
                                            predictions_sha256=sha256_file(directory / "predictions.npz")))
        if directory == epo:
            receipt["epo_protocol"] = "epo_train_only_site_cv_v2"
        (directory / "fit_receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(response.td, "load", lambda _: (metadata.copy(), None, None))
    return bundle, baseline, epo


def score(inputs, destination):
    bundle, baseline, epo = inputs
    return response.source_scores(bundle, baseline_output=baseline, epo_output=epo, scores_output=destination)


def rewrite_archive(directory, change):
    path = directory / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    change(arrays)
    np.savez(path, **arrays)
    receipt_path = directory / "fit_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["source_identity"]["predictions_sha256"] = sha256_file(path)
    receipt_path.write_text(json.dumps(receipt))


def test_binned_control_cannot_replace_the_native_grid_primary_candidate(score_inputs, tmp_path):
    output = score(score_inputs, tmp_path / "scores")
    primary = pd.read_csv(output / "baseline_scoreboard.csv").set_index("model")
    control = pd.read_csv(output / EPO_RESOLUTION_CONTROL_FILES[0]).set_index("model")
    assert set(primary.index) == set(PRIMARY_MODELS) and len(primary) == 6
    assert set(control.index) == {"anchorhp_bin2_base", "epo_bin2"}
    assert primary.loc["vibes_anchorhp", "improve_median_absolute_error"] == 10.
    assert control.loc["anchorhp_bin2_base", "improve_median_absolute_error"] == 2.
    assert primary.loc["epo", "improve_median_absolute_error"] == control.loc["epo_bin2", "improve_median_absolute_error"] == .5
    assert control.k.eq(18).all() and control.spectral_channels.eq(1361).all()
    assert control.comparison_scope.eq("supplementary_matched_resolution_control").all()
    receipt = verified_score_receipt(output)
    assert receipt["status"] == "corrected"
    supplementary = receipt["matched_resolution_control"]
    assert supplementary["protocol"] == EPO_RESOLUTION_CONTROL_PROTOCOL
    assert supplementary["primary_comparison_member"] is False
    assert supplementary["resolution"]["primary_anchorhp_model"] == "vibes_anchorhp"
    assert supplementary["resolution"]["primary_anchorhp_channels"] == 2722
    assert supplementary["resolution"]["control_channels"] == 1361
    assert supplementary["same_common_filters_as_primary"] is True
    assert supplementary["population"]["n_common"] == 48
    assert "common_filter_ids" not in supplementary["population"]
    assert supplementary["population"]["common_filter_ids_sha256"] == receipt["population"]["common_filter_ids_sha256"]
    assert {row["model"] for row in supplementary["summary"]} == set(control.index)
    for filename in EPO_RESOLUTION_CONTROL_FILES:
        assert supplementary["outputs"][filename] == receipt["outputs"][filename] == {"sha256": sha256_file(output / filename)}
    ledger = pd.read_csv(output / EPO_RESOLUTION_CONTROL_FILES[1])
    assert len(ledger) == 50 and ledger.filter_id.is_unique
    assert set(ledger.fold_id) == set(range(5))


def test_equal_population_counts_cannot_hide_different_control_filter_identities(score_inputs, tmp_path):
    _, baseline, epo = score_inputs
    rewrite_archive(baseline, lambda a: a["raw|improve_oof"].__setitem__((5, 17), np.nan))
    rewrite_archive(epo, lambda a: a["base|improve_oof"].__setitem__((6, 17), np.nan))
    output = score(score_inputs, tmp_path / "scores")
    receipt = verified_score_receipt(output)
    control = receipt["matched_resolution_control"]
    assert receipt["population"]["n_common"] == control["population"]["n_common"] == 47
    assert control["same_common_filters_as_primary"] is False
    assert control["primary_common_filter_ids_sha256"] != control["population"]["common_filter_ids_sha256"]
    primary_ids = set(pd.read_csv(output / "baseline_scoreboard_filters.csv").query("common").filter_id)
    control_ids = set(pd.read_csv(output / EPO_RESOLUTION_CONTROL_FILES[1]).query("common").filter_id)
    assert primary_ids - control_ids == {"F006"}
    assert control_ids - primary_ids == {"F005"}


@pytest.mark.parametrize("filename", OUTPUT_NAMES)
def test_existing_source_score_artifact_is_preserved_before_loading_inputs(tmp_path, monkeypatch, filename):
    output = tmp_path / "scores"
    output.mkdir()
    sentinel = output / filename
    sentinel.write_bytes(b"previous scientific result")
    monkeypatch.setattr(response.td, "load", lambda _: pytest.fail("must not load inputs before the no-overwrite guard"))
    with pytest.raises(FileExistsError, match="select a new --output"):
        response.source_scores(tmp_path / "missing", scores_output=output)
    assert sentinel.read_bytes() == b"previous scientific result"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize("moment", ["metrics", "serialization"])
def test_inputs_changed_during_scoring_cannot_receive_a_corrected_receipt(score_inputs, tmp_path, monkeypatch, moment):
    bundle, _, _ = score_inputs
    output = tmp_path / "scores"

    def change_input():
        with (bundle / "improve_meta.csv").open("a") as stream:
            stream.write("\n")

    if moment == "metrics":
        original = response.paired_candidate_metrics

        def mutated_metrics(*args, **kwargs):
            result = original(*args, **kwargs)
            change_input()
            return result

        monkeypatch.setattr(response, "paired_candidate_metrics", mutated_metrics)
    else:
        original = pd.DataFrame.to_csv

        def mutated_write(frame, path, *args, **kwargs):
            result = original(frame, path, *args, **kwargs)
            if Path(path).name == EPO_RESOLUTION_CONTROL_FILES[1]:
                change_input()
            return result

        monkeypatch.setattr(pd.DataFrame, "to_csv", mutated_write)
    with pytest.raises(RuntimeError, match="input changed during scoring"):
        score(score_inputs, output)
    assert not any((output / name).exists() for name in OUTPUT_NAMES)


def test_concurrent_score_writer_cannot_be_overwritten_after_the_last_guard(score_inputs, tmp_path, monkeypatch):
    output = tmp_path / "scores"
    original = response.td.require_new_fit_outputs
    calls = 0

    def concurrent_output(directory, names):
        nonlocal calls
        original(directory, names)
        calls += 1
        if calls == 2:
            (directory / "baseline_scoreboard.csv").write_bytes(b"concurrent saved result")

    monkeypatch.setattr(response.td, "require_new_fit_outputs", concurrent_output)
    with pytest.raises(FileExistsError):
        score(score_inputs, output)
    assert (output / "baseline_scoreboard.csv").read_bytes() == b"concurrent saved result"
    assert not (output / "baseline_scoreboard_receipt.json").exists()


@pytest.mark.parametrize("problem", ["missing_base", "wrong_grid"])
def test_control_requires_its_saved_baseline_and_exact_pair_averaged_grid(score_inputs, tmp_path, problem):
    _, _, epo = score_inputs
    if problem == "missing_base":
        rewrite_archive(epo, lambda a: a.pop("base|improve_oof"))
    else:
        rewrite_archive(epo, lambda a: a.__setitem__("wn_bin2", a["wn_bin2"] + .1))
    with pytest.raises(ValueError, match="missing saved|pair-averaged source grid"):
        score(score_inputs, tmp_path / "scores")


@pytest.mark.parametrize("problem", ["changed_csv", "missing_file_entry", "wrong_protocol", "unsafe_output"])
def test_control_integrity_is_required_by_the_shared_receipt_reader(score_inputs, tmp_path, problem):
    output = score(score_inputs, tmp_path / "scores")
    path = output / "baseline_scoreboard_receipt.json"
    receipt = json.loads(path.read_text())
    if problem == "changed_csv":
        with (output / EPO_RESOLUTION_CONTROL_FILES[0]).open("a") as stream:
            stream.write("tampered\n")
    elif problem == "missing_file_entry":
        receipt["outputs"].pop(EPO_RESOLUTION_CONTROL_FILES[1])
    elif problem == "wrong_protocol":
        receipt["matched_resolution_control"]["protocol"] = "unverified_control"
    else:
        receipt["outputs"]["../outside.csv"] = {"sha256": "a" * 64}
    path.write_text(json.dumps(receipt))
    status = verified_score_receipt(output)
    assert status["status"] == "legacy_unverified"
    assert status["reason"].startswith("Source scores withheld:")
