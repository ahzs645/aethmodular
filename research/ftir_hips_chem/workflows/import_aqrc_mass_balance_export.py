"""Verify a Davis export, retain its raw files, and build physical-filter coverage.

This importer never turns raw optical readings or mass metadata into chemistry.
It uses the canonical physical-ID helper and the existing AQRC manifest verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))
from data_matching import base_filter_id
from audit_spartan_network_coverage import verify_reference_folder
from aethmodular_cli.env import display_path

REQUIRED = {"requested_filter_coverage", "hips_analyses", "hips_readings", "hips_flags",
            "ftir_analyses", "media_comments", "filter_integrity", "qc_codes", "visible_result_objects"}


def safe_extract(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as stream:
        entries = stream.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        if sum(entry.file_size for entry in entries) > 512 * 1024 * 1024:
            raise ValueError("Export exceeds the scoped import budget")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename or ":" in entry.filename:
                raise ValueError("Unsafe archive path")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Archive contains a symbolic link")
        stream.extractall(destination)
    manifests = list(destination.rglob("manifest.csv"))
    if len(manifests) != 1:
        raise ValueError("Expected exactly one export manifest")
    return manifests[0].parent


def verify_export(folder: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    verified = verify_reference_folder(folder)
    if verified["status"] != "complete" or verified["errors"] != 0:
        raise ValueError("Partial or failed export remains unresolved")
    exports = pd.read_csv(folder / "exports.csv", dtype=str, keep_default_na=False)
    if set(exports.Dataset) != REQUIRED or exports.Dataset.duplicated().any() or not exports.Status.eq("ok").all():
        raise ValueError("Missing, duplicate or unsuccessful datasets")
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8-sig"))
    if receipt["RequestSHA256"].lower() != hashlib.sha256((folder / "requested_filters.csv").read_bytes()).hexdigest():
        raise ValueError("Request hash does not match the receipt")
    expected = ROOT / "research/ftir_ec_phase3/scripts/export_spartan_mass_balance_followup.ps1"
    if receipt["SourceSHA256"].lower() != hashlib.sha256(expected.read_bytes()).hexdigest():
        raise ValueError("Unexpected exporter source version")
    frames = {name: pd.read_csv(folder / f"{name}.csv", dtype=str, keep_default_na=False) for name in REQUIRED}
    if len(pd.read_csv(folder / "errors.csv")):
        raise ValueError("Export contains recorded errors")
    for entry in exports.itertuples():
        if entry.File != entry.Dataset + ".csv" or int(entry.CsvRows) != len(frames[entry.Dataset]):
            raise ValueError("Export status disagrees with its CSV")
    requests = pd.read_csv(folder / "requested_filters.csv", dtype=str, keep_default_na=False)
    if requests.FilterId.duplicated().any() or len(requests) != int(receipt["RequestedFilters"]):
        raise ValueError("Request cohort is inconsistent")
    requested = set(requests.FilterId)
    coverage = frames["requested_filter_coverage"]
    if set(coverage.RequestedFilterId) != requested:
        raise ValueError("Registry query lost or added requested filters")
    matched = coverage[coverage.ExternalFilterId.ne("")]
    if not matched.ExternalFilterId.map(base_filter_id).eq(matched.RequestedFilterId).all():
        raise ValueError("External identity does not map to the requested physical filter")
    valid_media = set(zip(matched.RequestedFilterId, matched.MediaId))
    for name, frame in frames.items():
        if "RequestedFilterId" not in frame:
            continue
        if not set(frame.RequestedFilterId).issubset(requested):
            raise ValueError(f"Out-of-cohort observations in {name}")
        if name != "requested_filter_coverage" and not set(zip(frame.RequestedFilterId, frame.MediaId)).issubset(valid_media):
            raise ValueError(f"Unmatched media identities in {name}")
    frames["requested_filters"] = requests
    return verified, frames


def import_export(source: Path, destination: Path, tables: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="aqrc-mass-balance-") as temporary:
        folder = safe_extract(source, Path(temporary)) if source.is_file() else source
        verified, frames = verify_export(folder)
        receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8-sig"))
        registry = frames["requested_filter_coverage"]
        requests = frames["requested_filters"]
        records = []
        for request in requests.to_dict("records"):
            fid = request["FilterId"]
            catalog = registry[registry.RequestedFilterId.eq(fid) & registry.ExternalFilterId.ne("")]
            record = {"FilterId": fid, "Site": request["Site"], "ExpectedDate": request["Date"],
                      "RegistryStatus": "matched" if len(catalog) else "not_registered",
                      "MediaRows": len(catalog), "MediaIds": ";".join(catalog.MediaId),
                      "SourceMassValues": ";".join(catalog.MassCollectedOnFilter),
                      "SourceVolumeValues": ";".join(catalog.SampleVolume_m3),
                      "SamplingDateConflict": any(value[:10] != request["Date"] for value in catalog.SamplingStartDate if value),
                      "OfficialChemicalReconstruction": "not_exported"}
            for name in ("hips_analyses", "hips_readings", "hips_flags", "ftir_analyses", "media_comments", "filter_integrity"):
                record[name + "_rows"] = int(frames[name].RequestedFilterId.eq(fid).sum())
            for name in ("hips_analyses", "ftir_analyses"):
                analyses = frames[name].loc[frames[name].RequestedFilterId.eq(fid)]
                record[name + "_qc_counts"] = json.dumps(
                    analyses.AnalysisQcCode.value_counts().sort_index().to_dict(), sort_keys=True
                )
            comments = frames["media_comments"].loc[frames["media_comments"].RequestedFilterId.eq(fid)]
            # A literal source label is evidence about the physical sample.
            # Analysis QC=1 does not override an independently rejected sample.
            invalid = comments.loc[comments.Entry.str.strip().str.startswith("INVALID --")]
            record["SourceInvalidationDocumented"] = not invalid.empty
            record["SourceInvalidationCommentIds"] = ";".join(invalid.CommentId)
            record["SourceInvalidationComments"] = "\n".join(invalid.Entry)
            records.append(record)
        target = destination / (source.stem if source.is_file() else source.name)
        if target.exists():
            verify_export(target)
            if (target / "manifest.csv").read_bytes() != (folder / "manifest.csv").read_bytes():
                raise FileExistsError("Refusing to overwrite a different raw export")
        else:
            shutil.copytree(folder, target)
        tables.mkdir(parents=True, exist_ok=True)
        coverage = pd.DataFrame(records)
        coverage.to_csv(tables / "aqrc_physical_filter_coverage.csv", index=False)
        result = {"imported": datetime.now(timezone.utc).isoformat(), "raw_export": display_path(target),
                  "verified": verified, "requested_filters": len(coverage),
                  "registered_physical_filters": int(coverage.RegistryStatus.eq("matched").sum()),
                  "registered_by_site": coverage.loc[coverage.RegistryStatus.eq("matched"), "Site"].value_counts().to_dict(),
                  "sampling_date_conflicts": int(coverage.SamplingDateConflict.sum()),
                  "source_invalidation_documented_filters": coverage.loc[coverage.SourceInvalidationDocumented, "FilterId"].tolist(),
                  "datasets": {name: len(frame) for name, frame in frames.items()},
                  "request_sha256": receipt["RequestSHA256"],
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
                  "measurement_limit": "Raw native observations and metadata retained with QC; no chemical mass balance was fabricated and no current gallery value was replaced."}
        (tables / "aqrc_import_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=ROOT / "data/aqrc/spartan_mass_balance")
    parser.add_argument("--tables", type=Path, default=ROOT / "research/ftir_hips_chem/output/tables/spartan_mass_balance_audit")
    args = parser.parse_args()
    print(json.dumps(import_export(args.source, args.destination, args.tables), indent=2))
