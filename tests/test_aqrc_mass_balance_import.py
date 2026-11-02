"""Verify the data handoff rejects bad archives and preserves raw observations."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "research/ftir_hips_chem/workflows/import_aqrc_mass_balance_export.py"
spec = importlib.util.spec_from_file_location("mass_balance_import", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def manifest(folder):
    rows = []
    for path in sorted(folder.iterdir()):
        if path.name == "manifest.csv":
            continue
        rows.append({"File": path.name, "Bytes": path.stat().st_size,
                     "CsvRows": len(pd.read_csv(path)) if path.suffix == ".csv" else "",
                     "SHA256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(folder / "manifest.csv", index=False)


@pytest.fixture
def export(tmp_path):
    folder = tmp_path / "export"
    folder.mkdir()
    pd.DataFrame([{"FilterId": "ETAD-0001", "Site": "Addis Ababa", "Date": "2023-01-01"},
                  {"FilterId": "INDH-0009", "Site": "Delhi", "Date": "2022-06-28"}]).to_csv(folder / "requested_filters.csv", index=False)
    catalog = [
        {"RequestedFilterId": "ETAD-0001", "ExternalFilterId": "ETAD-0001-12", "MediaId": "7", "SamplingStartDate": "2023-01-01T09:00:00", "MassCollectedOnFilter": "-3", "SampleVolume_m3": "0"},
        {"RequestedFilterId": "INDH-0009", "ExternalFilterId": "", "MediaId": "", "SamplingStartDate": "", "MassCollectedOnFilter": "", "SampleVolume_m3": ""},
    ]
    common = ["RequestedFilterId", "ExternalFilterId", "MediaId"]
    for name in module.REQUIRED:
        if name == "requested_filter_coverage":
            frame = pd.DataFrame(catalog)
        elif name == "hips_readings":
            frame = pd.DataFrame([{**dict(zip(common, ["ETAD-0001", "ETAD-0001-12", "7"])), "Transmittance": "0", "Reflectance": "0.4"}])
        elif name in {"qc_codes", "visible_result_objects"}:
            frame = pd.DataFrame(columns=["context"])
        elif name in {"hips_analyses", "ftir_analyses"}:
            frame = pd.DataFrame(columns=[*common, "AnalysisQcCode"])
        elif name == "media_comments":
            frame = pd.DataFrame(columns=[*common, "Entry", "CommentId"])
        else:
            frame = pd.DataFrame(columns=common)
        frame.to_csv(folder / f"{name}.csv", index=False)
    pd.DataFrame([{"Dataset": name, "Status": "ok", "CsvRows": len(pd.read_csv(folder / f"{name}.csv")), "File": name + ".csv"} for name in sorted(module.REQUIRED)]).to_csv(folder / "exports.csv", index=False)
    pd.DataFrame(columns=["ExceptionType"]).to_csv(folder / "errors.csv", index=False)
    source = ROOT / "research/ftir_ec_phase3/scripts/export_spartan_mass_balance_followup.ps1"
    receipt = {"Status": "complete", "Errors": 0, "SuccessfulExports": 9, "RequestedFilters": 2,
               "SourceSHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "RequestSHA256": hashlib.sha256((folder / "requested_filters.csv").read_bytes()).hexdigest()}
    (folder / "receipt.json").write_text(json.dumps(receipt))
    manifest(folder)
    return folder


def test_preserves_missing_registry_negative_mass_and_zero_volume(export, tmp_path):
    result = module.import_export(export, tmp_path / "raw", tmp_path / "tables")
    assert result["registered_physical_filters"] == 1
    assert result["requested_filters"] == 2
    coverage = pd.read_csv(tmp_path / "tables/aqrc_physical_filter_coverage.csv", keep_default_na=False)
    assert coverage.iloc[0].SourceMassValues == "-3"
    assert coverage.iloc[0].SourceVolumeValues == "0"
    assert coverage.iloc[1].RegistryStatus == "not_registered"
    assert coverage.OfficialChemicalReconstruction.eq("not_exported").all()
    assert (tmp_path / "raw/export/hips_readings.csv").read_bytes() == (export / "hips_readings.csv").read_bytes()


def test_rejects_changed_bytes(export):
    (export / "hips_readings.csv").write_text("altered")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.verify_export(export)


def test_valid_analysis_does_not_erase_source_sample_rejection(export, tmp_path):
    identity = {"RequestedFilterId": "ETAD-0001", "ExternalFilterId": "ETAD-0001-12", "MediaId": "7"}
    pd.DataFrame([{**identity, "AnalysisQcCode": "1"}, {**identity, "AnalysisQcCode": "0"}]).to_csv(export / "hips_analyses.csv", index=False)
    pd.DataFrame([{**identity, "CommentId": "123", "Entry": "INVALID -- cartridge affected by water infiltration"},
                  {**identity, "CommentId": "124", "Entry": "gridded"}]).to_csv(export / "media_comments.csv", index=False)
    statuses = pd.read_csv(export / "exports.csv")
    statuses.loc[statuses.Dataset.isin(["hips_analyses", "media_comments"]), "CsvRows"] = 2
    statuses.to_csv(export / "exports.csv", index=False)
    manifest(export)
    result = module.import_export(export, tmp_path / "raw", tmp_path / "tables")
    coverage = pd.read_csv(tmp_path / "tables/aqrc_physical_filter_coverage.csv", keep_default_na=False)
    assert result["source_invalidation_documented_filters"] == ["ETAD-0001"]
    assert json.loads(coverage.iloc[0].hips_analyses_qc_counts) == {"0": 1, "1": 1}
    assert coverage.iloc[0].SourceInvalidationDocumented
    assert coverage.iloc[0].SourceInvalidationCommentIds == "123"
    assert coverage.iloc[0].RegistryStatus == "matched"
    assert not coverage.iloc[1].SourceInvalidationDocumented


def test_rejects_failed_export_even_with_valid_manifest(export):
    path = export / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["Status"] = "partial_with_errors"
    path.write_text(json.dumps(receipt))
    manifest(export)
    with pytest.raises(ValueError, match="Partial or failed"):
        module.verify_export(export)


def test_rejects_wrong_physical_identity(export):
    path = export / "requested_filter_coverage.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.loc[0, "ExternalFilterId"] = "ETAD-0002-12"
    frame.to_csv(path, index=False)
    manifest(export)
    with pytest.raises(ValueError, match="External identity"):
        module.verify_export(export)


def test_rejects_archive_path_traversal(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as stream:
        stream.writestr("../receipt.json", "{}")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        module.safe_extract(path, tmp_path / "extract")
