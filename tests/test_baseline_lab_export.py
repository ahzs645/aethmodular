"""Data-boundary checks for the graph-focused baseline investigation."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gallery/data"))
import export_baseline_lab as baseline_export  # noqa: E402
from export_baseline_lab import (  # noqa: E402
    aggregate_curves, append_native_records, calibration_tables, control_tables, curve,
)


def test_signed_spectra_and_uncertainty_keep_the_same_observed_grid():
    row = curve("test", "Test", "spline", "corrected", [1700, 1617, 1500],
                [-.1, np.nan, -.2], lo=[-.2, np.nan, -.3], hi=[0, np.nan, -.1])
    assert row["wn"] == [1700, 1617, 1500]
    assert row["y"] == [-.1, None, -.2]
    assert row["lo"] == [-.2, None, -.3]
    assert row["hi"] == [0, None, -.1]
    with pytest.raises(ValueError, match="uncertainty"):
        curve("bad", "Bad", "spline", "corrected", [1700, 1617], [1, 2], lo=[1])


def test_native_only_filters_do_not_acquire_an_optical_reference():
    native = pd.DataFrame([
        {"base_filter_id": "ETAD-0001", "ExternalFilterType": "PM2.5", "lot_group": "251"},
        {"base_filter_id": "ETAD-0002", "ExternalFilterType": "FB", "lot_group": "unknown"},
    ])
    rows = append_native_records([{"id": "ETAD-0001", "spectral_pair_eligible": True}], native)
    assert len(rows) == 2
    blank = rows[1]
    assert blank["spectral_pair_eligible"] is False
    assert blank["ec_ftir_ugm3"] is None
    assert blank["optical_bc_ugm3"] is None
    assert blank["lot_group"] == "unknown"


def test_injection_parent_identity_and_validation_scheme_are_not_pooled(tmp_path):
    rows = []
    for scheme, recovery in [("grouped_5fold", .9), ("known_lot_transfer", .2)]:
        for identifier in ["ETAD-0001", "ETAD-0002"]:
            rows.append({"scheme": scheme, "method": "spline_default", "control": "broad",
                         "amplitude": .01, "base_filter_id": identifier,
                         "area_recovery_fraction": recovery})
    path = tmp_path / "injection_recovery.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    _, _, injections, _, _ = control_tables(tmp_path)
    assert len(injections) == 2
    assert sorted(r["recovery_fraction"] for r in injections) == [.2, .9]
    assert all(r["n_parents"] == 2 and r["method"] == "spline" for r in injections)
    pd.DataFrame([*rows, rows[0]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one observation per physical parent"):
        control_tables(tmp_path)


def test_source_qc_sensitivity_is_not_a_second_score_in_the_primary_population(tmp_path):
    row = {"arm": "raw_full", "evaluation": "site_grouped_pooled", "fold": "all",
           "n": 905, "rmse": 16.9, "mae": 10, "predictive_r2": .8}
    pd.DataFrame([row]).to_csv(tmp_path / "metrics.csv", index=False)
    historical = tmp_path / "historical906_all_scans"
    historical.mkdir()
    pd.DataFrame([{**row, "n": 906, "rmse": 63.4}]).to_csv(historical / "metrics.csv", index=False)
    methods, scores, _ = calibration_tables(tmp_path)
    assert len(methods) == 1
    assert len({r["cohort"] for r in scores}) == 2
    assert [r["n"] for r in scores] == [905, 906]
    assert all(r["units"] == "µg/filter" and r["calibration_matched"] for r in scores)


def test_duplicate_repeat_scans_cannot_weight_the_physical_filter_twice(tmp_path):
    row = {"method": "spline_default", "base_filter_id": "ETAD-0139", "filter_type": "PM2.5",
           "n_pairs": 1, "relative_pair_rms": 1.9, "raw_pair_rms": .2,
           "corrected_pair_rms": .01}
    pd.DataFrame([row, row]).to_csv(tmp_path / "replicate_filter_metrics.csv", index=False)
    with pytest.raises(ValueError, match="aggregated to physical filters"):
        control_tables(tmp_path)


def test_legacy_aggregate_without_support_cannot_present_padding_as_observed_zero(tmp_path):
    rows = []
    for view, value in [("raw", -.1), ("saved_spline", 0.)]:
        for wn in [1600, 1500]:
            rows.append({"type": "FB", "view": view, "wavenumber": wn,
                         "median_absorbance": value, "q25_absorbance": value,
                         "q75_absorbance": value, "physical_filters": 2})
    pd.DataFrame(rows).to_csv(tmp_path / "signed_median_spectra.csv", index=False)
    curves = aggregate_curves(tmp_path)
    assert len(curves) == 1
    assert curves[0]["kind"] == "raw"
    assert curves[0]["y"] == [-.1, -.1]


@pytest.fixture
def baseline_export_args(tmp_path, monkeypatch):
    """A tiny complete export fixture with no private spectral inputs."""
    monkeypatch.setattr(baseline_export, "ROOT", tmp_path)
    monkeypatch.setattr(baseline_export, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(baseline_export, "export_packed_spectra", lambda *_: ["ETAD-0001"])
    monkeypatch.setattr(baseline_export, "export_native_spectra", lambda *_: [])
    monkeypatch.setattr(baseline_export.subprocess, "check_output", lambda *_args, **_kwargs: "fixture-revision\n")
    portable = tmp_path / "portable"
    portable.mkdir()
    pd.DataFrame([{
        "id": "ETAD-0001", "code": "ETAD", "date": "2024-01-15",
        "lot_group": "251", "spectral_pair_eligible": True, "is_excluded": False,
        "optical_source": "direct_hips_fabs", "optical_bc_ugm3": 2.,
        "pm25_ugm3": 20., "volume_m3": 10., "optical_disagreement_ugm3": -.5,
    }]).to_csv(portable / "filter_diagnostic_ledger.csv", index=False)
    map_path = tmp_path / "gallery/app/public/data/spartan_map.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text('{"sites": []}\n')
    return SimpleNamespace(
        output=tmp_path / "export", portable_dir=portable,
        native_audit_dir=tmp_path / "missing-native", controls_dir=tmp_path / "missing-controls",
        calibration_dir=tmp_path / "missing-calibration", native_dir=None,
    )


@pytest.mark.parametrize("seasonal_present", [False, True], ids=["absent", "present"])
def test_optional_seasonal_export_preserves_downloads_and_input_hashes(
    tmp_path, baseline_export_args, seasonal_present,
):
    args = baseline_export_args
    source_files = {}
    if seasonal_present:
        args.seasonal_ml_dir = tmp_path / "tables/seasonal-fixture"
        args.seasonal_ml_dir.mkdir(parents=True)
        source_files = {
            "seasonal_ml.json": b'{"schema_version":1,"scores":[{"bias":-0.125}]}\n',
            "model_scores.csv": b"model_id,bias\nrf_raw_full,-0.125\n",
        }
        for name, content in source_files.items():
            (args.seasonal_ml_dir / name).write_bytes(content)
    # Omitting the argument also checks compatibility with older exporter callers.
    data = baseline_export.build_export(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    downloads = [item for item in data["downloads"] if item["label"].startswith("seasonal_ml · ")]
    assert len(data["filters"]) == 1
    assert len(downloads) == len(source_files)
    if seasonal_present:
        assert (args.output / "seasonal_ml.json").read_bytes() == source_files["seasonal_ml.json"]
        assert {item["path"] for item in downloads} == {
            "research/baseline_lab/tables/seasonal_ml/" + name for name in source_files
        }
        for name, content in source_files.items():
            assert (args.output / "tables/seasonal_ml" / name).read_bytes() == content
            source_key = (args.seasonal_ml_dir / name).relative_to(tmp_path).as_posix()
            assert manifest["inputs"][source_key] == hashlib.sha256(content).hexdigest()
    else:
        assert not (args.output / "seasonal_ml.json").exists()
        assert not (args.output / "tables/seasonal_ml").exists()
        assert not any("seasonal" in path for path in manifest["inputs"])


def seasonal_merge_payloads():
    hashes = {role: {"filename": role + ".csv", "sha256": "a" * 64}
              for role in ("main_workflow", "core", "calendar_config", "ledger", "arrays", "boundaries", "weather")}
    identity = {"task": "disagreement", "calendar": "belg_feb", "cohort": "direct232",
                "model_id": "rf_raw_full", "fold_id": "forward_2024"}
    importance = {**identity, "kind": "permutation", "scheme": "within_season",
                  "region_id": "whole_spectrum", "delta_mean": -.0123456789012345,
                  "metric": "rmse"}
    main = {
        "schema_version": 1, "models": [{"id": "rf_raw_full"}], "folds": [{"id": "forward_2024"}],
        "scores": [{**identity, "n_train": 91, "n_test": 97, "rmse": .987654321012345}],
        "importance": [importance],
        "provenance": {"inputs": {"input/" + r["filename"]: r["sha256"] for r in hashes.values()}},
    }
    addon = {
        "schema_version": 1,
        "importance": [{**importance, "season": "Bega", "n_test": 37}],
        "performance": [{**identity, "season": "Bega", "n_train": 91, "n_test": 37, "rmse": .8}],
        "provenance": {"hashes_start": hashes, "hashes_end": json.loads(json.dumps(hashes)),
                       "source_and_inputs_unchanged": True},
        "limitations": ["Pooled training; season-specific held-out evaluation."],
    }
    return main, addon


def test_seasonal_addon_merges_served_view_but_preserves_original_downloads(tmp_path, baseline_export_args):
    args = baseline_export_args
    main, addon = seasonal_merge_payloads()
    args.seasonal_ml_dir = tmp_path / "tables/seasonal-main"
    args.seasonal_ml_by_season_dir = tmp_path / "tables/seasonal-addon"
    args.seasonal_ml_dir.mkdir(parents=True)
    args.seasonal_ml_by_season_dir.mkdir()
    main_bytes, addon_bytes = (json.dumps(value, indent=2).encode() for value in (main, addon))
    main_path = args.seasonal_ml_dir / "seasonal_ml.json"
    addon_path = args.seasonal_ml_by_season_dir / "by_season.json"
    main_path.write_bytes(main_bytes)
    addon_path.write_bytes(addon_bytes)
    data = baseline_export.build_export(args)
    served = json.loads((args.output / "seasonal_ml.json").read_text())
    assert served["scores"] == main["scores"]  # Preserve full precision and original population.
    assert served["importance"][0] == main["importance"][0]
    assert len(served["importance"]) == 2
    seasonal = served["importance"][1]
    assert seasonal["n_train"] == 91 and seasonal["n_test"] == 37
    assert seasonal["training_scope"] == "pooled" and "37 held-out" in seasonal["evaluation_note"]
    assert served["seasonal_region_scores"][0]["n_test"] == 37
    assert served["provenance"]["seasonal_region_analysis"]["hashes_start"] == addon["provenance"]["hashes_start"]
    assert main_path.read_bytes() == main_bytes and addon_path.read_bytes() == addon_bytes
    assert (args.output / "tables/seasonal_ml/seasonal_ml.json").read_bytes() == main_bytes
    assert (args.output / "tables/seasonal_ml_by_season/by_season.json").read_bytes() == addon_bytes
    assert "research/baseline_lab/tables/seasonal_ml_by_season/by_season.json" in {
        row["path"] for row in data["downloads"]
    }
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["inputs"][addon_path.relative_to(tmp_path).as_posix()] == hashlib.sha256(addon_bytes).hexdigest()


@pytest.mark.parametrize("problem, message", [
    ("hash", "SHA-256 mismatch"), ("model", "unknown model or fold"),
    ("fold", "unknown model or fold"), ("duplicate", "duplicate seasonal importance"),
    ("changed_during_run", "changed during analysis"),
])
def test_seasonal_addon_rejects_incompatible_or_duplicate_results(problem, message):
    main, addon = seasonal_merge_payloads()
    if problem == "hash":
        for key in ("hashes_start", "hashes_end"):
            addon["provenance"][key]["arrays"]["sha256"] = "b" * 64
    elif problem == "model":
        addon["importance"][0]["model_id"] = "unknown"
    elif problem == "fold":
        addon["importance"][0]["fold_id"] = "unknown"
    elif problem == "duplicate":
        addon["importance"].append(dict(addon["importance"][0]))
    else:
        addon["provenance"]["source_and_inputs_unchanged"] = False
    with pytest.raises(ValueError, match=message):
        baseline_export.merge_seasonal_region_analysis(main, addon)


def test_missing_main_removes_only_owned_seasonal_outputs(tmp_path, baseline_export_args):
    args = baseline_export_args
    args.seasonal_ml_dir = tmp_path / "tables/missing-main"
    args.seasonal_ml_by_season_dir = tmp_path / "tables/orphan-addon"
    args.seasonal_ml_by_season_dir.mkdir(parents=True)
    (args.seasonal_ml_by_season_dir / "by_season.json").write_text('{"orphan": true}\n')
    for name in ("seasonal_ml", "seasonal_ml_by_season", "spline_transition", "portable"):
        folder = args.output / "tables" / name
        folder.mkdir(parents=True)
        (folder / "sentinel.txt").write_text("old export")
    (args.output / "seasonal_ml.json").write_text('{"stale": true}\n')
    data = baseline_export.build_export(args)
    assert not (args.output / "seasonal_ml.json").exists()
    assert not (args.output / "tables/seasonal_ml").exists()
    assert not (args.output / "tables/seasonal_ml_by_season").exists()
    assert not (args.output / "tables/spline_transition").exists()
    assert (args.output / "tables/portable/sentinel.txt").read_text() == "old export"
    assert (args.seasonal_ml_by_season_dir / "by_season.json").is_file()
    assert not any(row["label"].startswith("seasonal_ml") for row in data["downloads"])


def transition_merge_payloads():
    main, seasonal = seasonal_merge_payloads()
    main["models"][0]["representation"] = "raw_common"
    main["representations"] = [{"id": "raw_common", "n_channels": 10}]
    main["scores"][0].update(n_train=3, n_test=2)
    main["predictions"] = [
        {**main["scores"][0], "id": identifier, "actual": 1., "predicted": .8}
        for identifier in ("ETAD-0001", "ETAD-0002")
    ]
    seasonal["importance"][0]["n_test"] = 1
    seasonal["performance"][0].update(n_train=3, n_test=1)
    new_id = "rf_raw_common_transitions_masked"
    transition = {
        "schema_version": 1,
        "models": [{"id": new_id, "representation": "raw_common_transitions_masked"}],
        "representations": [{"id": "raw_common_transitions_masked", "n_channels": 5}],
        "scores": [{**main["scores"][0], "model_id": new_id}],
        "predictions": [{**row, "model_id": new_id} for row in main["predictions"]],
        "provenance": json.loads(json.dumps(seasonal["provenance"])),
        "limitations": ["Fixed transition mask; baseline unchanged."],
    }
    return main, seasonal, transition


def test_transition_append_and_pmf_downloads_keep_original_model_inputs_separate(tmp_path, baseline_export_args):
    args = baseline_export_args
    main, seasonal, transition = transition_merge_payloads()
    originals = {}
    for attribute, prefix, filename, value in (
        ("seasonal_ml_dir", "seasonal_ml", "seasonal_ml.json", main),
        ("seasonal_ml_by_season_dir", "seasonal_ml_by_season", "by_season.json", seasonal),
        ("spline_transition_dir", "spline_transition", "transition_sensitivity.json", transition),
        ("seasonal_pmf_dir", "seasonal_pmf", "summary.json", {"descriptive_only": True}),
    ):
        folder = tmp_path / "inputs" / prefix
        folder.mkdir(parents=True)
        setattr(args, attribute, folder)
        content = json.dumps(value, indent=2).encode()
        (folder / filename).write_bytes(content)
        originals[(prefix, filename)] = content
    pmf_csv = b"factor,rho,n\nfactor_1,-0.5,2\n"
    (args.seasonal_pmf_dir / "correlations.csv").write_bytes(pmf_csv)
    originals[("seasonal_pmf", "correlations.csv")] = pmf_csv
    data = baseline_export.build_export(args)
    served = json.loads((args.output / "seasonal_ml.json").read_text())
    for field in ("models", "representations", "scores", "predictions"):
        assert served[field] == main[field] + transition[field]
    assert len(served["seasonal_region_scores"]) == 1
    assert "spline_transition_analysis" in served["provenance"]
    assert "seasonal_region_analysis" in served["provenance"]
    assert "pmf" not in json.dumps(served).lower()
    download_paths = {row["path"] for row in data["downloads"]}
    manifest = json.loads((args.output / "manifest.json").read_text())
    for (prefix, filename), content in originals.items():
        assert (tmp_path / "inputs" / prefix / filename).read_bytes() == content
        assert (args.output / "tables" / prefix / filename).read_bytes() == content
        assert f"research/baseline_lab/tables/{prefix}/{filename}" in download_paths
        assert manifest["inputs"][f"inputs/{prefix}/{filename}"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("problem, message", [
    ("fold", "unknown model or fold"), ("hash", "SHA-256 mismatch"),
    ("changed_during_run", "changed during analysis"), ("model_collision", "without collisions"),
    ("counts", "sample counts"), ("physical_id", "held-out physical filter"),
])
def test_transition_rejects_changed_inputs_or_population(problem, message):
    main, _, transition = transition_merge_payloads()
    if problem == "fold":
        transition["scores"][0]["fold_id"] = "unknown"
    elif problem == "hash":
        for key in ("hashes_start", "hashes_end"):
            transition["provenance"][key]["arrays"]["sha256"] = "b" * 64
    elif problem == "changed_during_run":
        transition["provenance"]["source_and_inputs_unchanged"] = False
    elif problem == "model_collision":
        transition["models"][0]["id"] = main["models"][0]["id"]
    elif problem == "counts":
        transition["scores"][0]["n_test"] = 1
    else:
        transition["predictions"][0]["id"] = "ETAD-9999"
    with pytest.raises(ValueError, match=message):
        baseline_export.merge_spline_transition_analysis(main, transition)
