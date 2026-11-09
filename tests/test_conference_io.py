"""The suite must preserve identity, failure status, and reproducibility records."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from conference_experiments.io import load_dataset, new_output_directory, write_json, write_result
from conference_experiments.runner import run_manifest, validate_manifest


def test_column_projection_preserves_all_rows_and_hashes_the_complete_input(tmp_path):
    frame = pd.DataFrame({"time": [1, 2, 3], "needed": [4, 5, 6], "unused": ["a", "b", "c"]})
    frame.to_csv(tmp_path / "minutes.csv", index=False)
    selected, narrow_receipts = load_dataset({"path": "minutes.csv", "columns": ["time", "needed"]}, tmp_path)
    _, full_receipts = load_dataset({"path": "minutes.csv"}, tmp_path)
    pd.testing.assert_frame_equal(selected, frame[["time", "needed"]])
    assert narrow_receipts == full_receipts
    with pytest.raises(ValueError, match="columns"):
        load_dataset({"path": "minutes.csv", "columns": ["time", "time"]}, tmp_path)


def test_spectral_alignment_uses_ids_and_rejects_a_different_population(tmp_path):
    np.savez(tmp_path / "spectra.npz", X=np.array([[10., 11.], [20., 21.]]), analysis_ids=np.array(["a", "b"]))
    pd.DataFrame({"analysis_id": ["b", "a"], "filter_id": ["fb", "fa"], "ec": [2., 1.]}).to_csv(tmp_path / "meta.csv", index=False)
    spec = {"kind": "spectra", "path": "spectra.npz", "metadata": "meta.csv"}
    (matrix, meta), receipts = load_dataset(spec, tmp_path)
    assert meta.filter_id.tolist() == ["fa", "fb"]
    assert meta.ec.tolist() == [1., 2.]
    assert matrix[:, 0].tolist() == [10., 20.]
    assert all(len(r["sha256"]) == 64 for r in receipts)
    pd.DataFrame({"analysis_id": ["a", "c"]}).to_csv(tmp_path / "meta.csv", index=False)
    with pytest.raises(ValueError, match="match exactly"):
        load_dataset(spec, tmp_path)


def test_pickle_arrays_and_duplicate_row_ids_are_not_accepted(tmp_path):
    np.savez(tmp_path / "spectra.npz", X=np.ones((2, 2)), analysis_ids=np.array(["a", "b"], dtype=object))
    pd.DataFrame({"analysis_id": ["a", "b"]}).to_csv(tmp_path / "meta.csv", index=False)
    spec = {"kind": "spectra", "path": "spectra.npz", "metadata": "meta.csv"}
    with pytest.raises(ValueError, match="Object arrays"):
        load_dataset(spec, tmp_path)
    pd.DataFrame({"analysis_id": ["a", "a"]}).to_csv(tmp_path / "meta.csv", index=False)
    with pytest.raises(ValueError, match="unique"):
        load_dataset(spec, tmp_path)


def test_reports_cannot_overwrite_or_escape_output_directory(tmp_path):
    output = new_output_directory(tmp_path / "run")
    write_json(output / "receipt.json", {"undefined": np.nan, "count": np.int64(3)})
    assert json.loads((output / "receipt.json").read_text()) == {"undefined": None, "count": 3}
    with pytest.raises(FileExistsError):
        new_output_directory(output)
    with pytest.raises(ValueError):
        write_result(output, "../outside", "test", {"summary": {}, "tables": {}})
    with pytest.raises(ValueError):
        write_result(output, "invalid", "test", {"summary": {"status": "looks_good"}})


def test_failure_and_missing_evidence_are_retained_as_different_states(tmp_path, monkeypatch):
    from conference_experiments import runner
    pd.DataFrame({"physical_id": ["a"]}).to_csv(tmp_path / "input.csv", index=False)
    manifest = {"schema_version": 1, "data_scope": "research",
                "datasets": {"available": {"path": "input.csv"}, "missing": {"path": "missing.csv"}},
                "experiments": [
                    {"id": "bad_fit", "kind": "measurement_plan", "inputs": {"data": "available"}},
                    {"id": "missing_data", "kind": "measurement_plan", "inputs": {"data": "missing"}},
                ]}
    def fail(*_):
        raise ArithmeticError("model failed")
    monkeypatch.setattr(runner, "dispatch", fail)
    write_json(tmp_path / "manifest.json", manifest)
    result = run_manifest(tmp_path / "manifest.json", tmp_path / "run")
    assert result["status"] == "failed"
    assert [r["status"] for r in result["experiments"]] == ["failed", "blocked"]
    assert result["experiments"][0]["summary"]["error_type"] == "ArithmeticError"
    assert (tmp_path / "run/report.md").is_file()
    assert result["inputs"]["available"]["files"][0]["path"] == "input.csv"


def test_manifest_disallows_forward_dependencies(tmp_path):
    manifest = {"schema_version": 1, "data_scope": "synthetic", "datasets": {},
                "experiments": [{"id": "first", "kind": "measurement_plan", "inputs": {
                    "data": {"experiment": "later", "table": "selected"}}}]}
    with pytest.raises(ValueError, match="earlier"):
        validate_manifest(manifest)
