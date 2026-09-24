"""Finish the large FTIR part of the audited Davis Data ingest.

Large CSVs become typed ZSTD Parquet partitions; RDS and SQLite bytes are
preserved in numbered ZIP chunks. The receipt is resumable and records the
original and part hashes. No Drive source is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import duckdb

from ingest_davis_drive import MAX_BYTES, request, upload

CSV_GROUPS = {
    "FTIR local spectra": {
        "name": "AETH FTIR local spectral matrix 248-251",
        "description": "Typed Parquet partitions of the 248-251 wide FTIR spectra CSV. source_row gives the original row order; original CSV SHA-256 and conversion are in source_manifest.",
        "paths": ["FTIR/local_db/spectra_248_251.csv"], "partition_rows": 1800,
    },
    "FTIR local scans": {
        "name": "AETH FTIR local scan payloads 248-251",
        "description": "Parquet partitions of FTIR scan metadata and base64 payloads. source_row gives the original row order; original CSV SHA-256 is in source_manifest.",
        "paths": ["FTIR/local_db/tables/scans_248_251.csv"], "partition_rows": 5000,
    },
    "FTIR large results": {
        "name": "AETH FTIR large results and July spectra",
        "description": "ZSTD Parquet conversions of XRF and TOR result exports and July 2026 FTIR spectra. Each original CSV remains on Drive; source hashes, row counts and conversion are in source_manifest.",
        "paths": ["FTIR/local_db/tables/results_xrf.csv", "FTIR/local_db/tables/results_tor.csv", "FTIR/runs/2026-07-07/ftir-spectra-2026-07-07.csv"],
        "partition_rows": None,
    },
}
ARCHIVE_NAME = "AETH FTIR model and SQLite archive"
CHUNK_BYTES = 40 * 1024 * 1024


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for piece in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def q(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def dataset(name: str, description: str) -> dict:
    matches = [d for d in request("GET", "/datasets")["datasets"] if d["name"] == name]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous dataset: {name}")
    return matches[0] if matches else request("POST", "/datasets", body={"name": name, "description": description})["dataset"]


def attached(dataset_id: str) -> dict:
    return {f["originalName"]: f for f in request("GET", f"/datasets/{dataset_id}")["dataset"]["sourceFiles"]}


def upload_missing(dataset_id: str, path: Path, name: str, files: dict) -> str:
    if name not in files:
        result = upload(dataset_id, path, name)
        if name not in result.get("uploaded", []):
            raise RuntimeError(f"Upload rejected: {name}: {result}")
        files.update({f["originalName"]: f for f in result["dataset"]["sourceFiles"]})
        print(f"uploaded {name}", flush=True)
    return files[name]["id"]


def write_manifest(dataset_id: str, rows: list[dict], files: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="zoer-large-manifest-") as temporary:
        path = Path(temporary) / "source_manifest.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        if "source_manifest.jsonl" in files:
            request("DELETE", f"/datasets/{dataset_id}/files/{files['source_manifest.jsonl']['id']}")
            del files["source_manifest.jsonl"]
        upload_missing(dataset_id, path, path.name, files)


def build(dataset_id: str) -> dict:
    result = request("POST", f"/datasets/{dataset_id}/rebuild")["dataset"]
    if result["status"] != "ready":
        raise RuntimeError(f"Build failed: {dataset_id}: {result.get('errorMessage')}")
    return result


def convert_csv(source: Path, output: Path) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as db:
        db.execute("SET threads=4")
        db.execute("SET memory_limit='3GB'")
        if not output.exists():
            db.execute(f"COPY (SELECT * FROM read_csv_auto({q(source)}, sample_size=-1, header=true)) TO {q(output)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        rows = db.execute(f"SELECT count(*) FROM read_parquet({q(output)})").fetchone()[0]
        source_rows = db.execute(f"SELECT count(*) FROM read_csv_auto({q(source)}, sample_size=-1, header=true)").fetchone()[0]
        if rows != source_rows:
            raise RuntimeError(f"Row count changed in conversion: {source}")
    return rows, output.stat().st_size


def parquet_parts(path: Path, stage: Path, rows: int, partition_rows: int | None) -> list[Path]:
    if path.stat().st_size <= MAX_BYTES:
        return [path]
    if not partition_rows:
        raise RuntimeError(f"Need a partition size for {path}")
    output: list[Path] = []
    with duckdb.connect() as db:
        db.execute("SET threads=4")
        db.execute("SET memory_limit='3GB'")
        for start in range(0, rows, partition_rows):
            index = start // partition_rows + 1
            part = stage / f"{path.stem}__part{index:03d}.parquet"
            if not part.exists():
                db.execute(f"COPY (SELECT row_number() OVER () + {start} AS source_row, * FROM (SELECT * FROM read_parquet({q(path)}) LIMIT {partition_rows} OFFSET {start})) TO {q(part)} (FORMAT PARQUET, COMPRESSION ZSTD)")
            if part.stat().st_size > MAX_BYTES:
                raise RuntimeError(f"Partition still exceeds upload limit: {part}")
            output.append(part)
        total = sum(db.execute(f"SELECT count(*) FROM read_parquet({q(p)})").fetchone()[0] for p in output)
        if total != rows:
            raise RuntimeError(f"Partition row count mismatch for {path}: {total} != {rows}")
    return output


def ingest_csv_group(key: str, config: dict, root: Path, audit: dict, stage: Path, receipt: dict) -> None:
    record = dataset(config["name"], config["description"])
    dataset_id = record["id"]
    files = attached(dataset_id)
    current = receipt["datasets"].setdefault(key, {"id": dataset_id, "files": {}})
    if current["id"] != dataset_id:
        raise RuntimeError("Dataset ID changed")
    manifest = []
    for relative in config["paths"]:
        source = root / relative
        audited = audit[relative]
        if source.stat().st_size != audited["bytes"] or sha(source) != audited["sha256"]:
            raise RuntimeError(f"Source changed since audit: {relative}")
        stem = source.stem + "__" + hashlib.sha256(relative.encode()).hexdigest()[:8]
        output = stage / (stem + ".parquet")
        rows, _ = convert_csv(source, output)
        parts = parquet_parts(output, stage, rows, config["partition_rows"])
        file_ids = [upload_missing(dataset_id, part, part.name, files) for part in parts]
        entry = {"path": relative, "sha256": audited["sha256"], "bytes": audited["bytes"],
                 "transformation": "DuckDB read_csv_auto(sample_size=-1) -> ZSTD Parquet",
                 "rows": rows, "partition_rows": config["partition_rows"],
                 "parts": [{"name": part.name, "sha256": sha(part), "bytes": part.stat().st_size, "file_id": fid}
                           for part, fid in zip(parts, file_ids)]}
        manifest.append(entry)
        current["files"][relative] = entry
    write_manifest(dataset_id, manifest, files)
    built = build(dataset_id)
    current["status"] = built["status"]
    print(json.dumps({"group": key, "dataset_id": dataset_id, "tables": built.get("tableCount"), "sources": len(manifest)}), flush=True)


def ingest_archive(root: Path, audit: dict, receipt: dict) -> None:
    entries = [(p, row) for p, row in audit.items() if p.startswith("FTIR/") and Path(p).suffix.lower() in {".rds", ".sqlite"}]
    record = dataset(ARCHIVE_NAME, "Exact RDS model exports and the offline IMPROVE SQLite mirror, held as numbered ZIP parts. source_manifest records original and part hashes. Download parts in order and concatenate their .bin members to reconstruct a source; scientific SQLite tables are represented separately in queryable FTIR exports.")
    dataset_id = record["id"]
    files = attached(dataset_id)
    current = receipt["datasets"].setdefault("FTIR binary archive", {"id": dataset_id, "files": {}})
    if current["id"] != dataset_id:
        raise RuntimeError("Dataset ID changed")
    manifest = []
    for relative, audited in entries:
        source = root / relative
        if source.stat().st_size != audited["bytes"] or sha(source) != audited["sha256"]:
            raise RuntimeError(f"Source changed since audit: {relative}")
        prefix = source.stem[:54] + "__" + hashlib.sha256(relative.encode()).hexdigest()[:10]
        parts = []
        with source.open("rb") as handle:
            index = 0
            while piece := handle.read(CHUNK_BYTES):
                index += 1
                name = f"{prefix}__part{index:03d}.zip"
                with tempfile.TemporaryDirectory(prefix="zoer-model-piece-") as temporary:
                    path = Path(temporary) / name
                    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                        archive.writestr(f"part{index:03d}.bin", piece)
                    if path.stat().st_size > MAX_BYTES:
                        raise RuntimeError(f"Archive part exceeds upload limit: {name}")
                    file_id = upload_missing(dataset_id, path, name, files)
                    parts.append({"name": name, "file_id": file_id, "sha256": hashlib.sha256(piece).hexdigest(), "bytes": len(piece)})
        entry = {"path": relative, "sha256": audited["sha256"], "bytes": audited["bytes"],
                 "storage": "ordered ZIP chunks; concatenate .bin members in ascending part order", "parts": parts}
        manifest.append(entry)
        current["files"][relative] = entry
        print(f"archived {relative}: {len(parts)} parts", flush=True)
    write_manifest(dataset_id, manifest, files)
    built = build(dataset_id)
    current["status"] = built["status"]
    print(json.dumps({"group": "FTIR binary archive", "dataset_id": dataset_id, "sources": len(entries), "parts": sum(len(e["parts"]) for e in manifest)}), flush=True)


def finalize_ftir(receipt: dict) -> None:
    current = receipt["datasets"]["FTIR"]
    dataset_id = current["id"]
    files = attached(dataset_id)
    external_groups = ["FTIR local spectra", "FTIR local scans", "FTIR large results", "FTIR binary archive"]
    for relative, entry in current["files"].items():
        if entry["status"] != "pending-large":
            continue
        found = [(key, receipt["datasets"][key]) for key in external_groups
                 if relative in receipt["datasets"][key]["files"]]
        if len(found) != 1:
            raise RuntimeError(f"Missing or ambiguous large FTIR source: {relative}")
        key, target = found[0]
        hosted = request("GET", f"/datasets/{target['id']}")["dataset"]
        if hosted["status"] != "ready":
            raise RuntimeError(f"Referenced large dataset is not ready: {key}")
        entry["status"] = "hosted-archive" if key == "FTIR binary archive" else "hosted-parquet"
        entry["dataset_id"] = target["id"]
        entry["source_manifest"] = f"/api/datasets/{target['id']}"
    pending = [path for path, item in current["files"].items() if item["status"] == "pending-large"]
    if pending or len(current["files"]) != 67:
        raise RuntimeError(f"FTIR coverage incomplete: {pending}")
    manifest = [{"path": path, **entry} for path, entry in current["files"].items()]
    write_manifest(dataset_id, manifest, files)
    request("PATCH", f"/datasets/{dataset_id}", body={"description":
        "FTIR small source tables, calibration exports, app outputs and original app captures. All 67 audited FTIR files are covered across this dataset and the linked large spectra, scans, results and binary archive datasets; source_manifest points to their locations."})
    built = build(dataset_id)
    current["status"] = built["status"]
    print(json.dumps({"group": "FTIR", "dataset_id": dataset_id, "covered_sources": len(current["files"]), "pending": 0}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=[*CSV_GROUPS, "FTIR binary archive", "finalize"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stage", type=Path, default=Path("/tmp/davis_parquet_stage"))
    args = parser.parse_args()
    args.stage.mkdir(parents=True, exist_ok=True)
    audit = {row["path"]: row for row in json.loads(args.audit.read_text())["files"]}
    receipt = json.loads(args.receipt.read_text())
    if args.mode == "finalize":
        finalize_ftir(receipt)
    elif args.mode == "FTIR binary archive":
        ingest_archive(args.root, audit, receipt)
    else:
        ingest_csv_group(args.mode, CSV_GROUPS[args.mode], args.root, audit, args.stage, receipt)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
