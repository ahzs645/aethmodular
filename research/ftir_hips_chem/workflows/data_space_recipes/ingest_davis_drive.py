"""Copy an audited Davis Data source family into an existing Zoer instance.

The audit JSON supplies relative paths, byte counts and SHA-256 hashes. The
script never changes Drive files. Run one family at a time, and keep the receipt
outside Git. Large files are reported as pending until a separate, checked
transformation or archive path is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import tempfile
import zipfile
from pathlib import Path

from zoer_env import zoer_api

FAMILIES = {
    "DAVIS": ("AETH DAVIS raw AQRC pulls", "Raw AQRC ETAD FTIR, SPARTAN FTIR/HIPS, Adama TOR and CSU AMOD pulls. Distinct from processed AETH tables."),
    "EC-HIPS-Aeth Comparison": ("AETH EC-HIPS lab and portal history", "Original lab deliverables, combined database and historical ETAD/USPA portal releases. Keep dates and revisions distinct from newer hosted public exports."),
    "Han": ("AETH Han 2007 reference", "Reference paper and two distinct supporting workbooks; paper is retained as a downloadable archive."),
    "Purple Air Data": ("AETH PurpleAir historical downloads", "Sensor 93783 overlapping exports are separate revisions with different coverage and columns; includes sensor 93733 and request metadata."),
    "Weather Data": ("AETH Jacros BAM and Meteostat source", "Raw year-by-year Jacros BAM, station metadata and local Meteostat raw/cleaned versions. These are not interchangeable with the hosted AETH Meteostat master."),
    "FTIR": ("AETH FTIR source archive", "Offline IMPROVE mirror, calibration sessions, model exports and application tables. This family may be incomplete until large sources are transformed and uploaded."),
}
MAX_BYTES = 49 * 1024 * 1024


def request(method: str, path: str, *, body: dict | None = None) -> dict:
    cmd = ["curl", "-fsS", "--retry", "2", "--max-time", "600", "-X", method,
           "-H", "Content-Type: application/json"]
    if body is not None:
        cmd += ["--data-binary", json.dumps(body)]
    cmd += [zoer_api() + path]
    return json.loads(subprocess.check_output(cmd, text=True))


def upload(dataset_id: str, path: Path, name: str) -> dict:
    cmd = ["curl", "-fsS", "--retry", "2", "--max-time", "600", "-X", "POST",
           "-F", f"files=@{path};filename={name}", zoer_api() + f"/datasets/{dataset_id}/files"]
    return json.loads(subprocess.check_output(cmd, text=True))


def unique_name(relative: str) -> str:
    path = Path(relative)
    prefix = hashlib.sha256(relative.encode()).hexdigest()[:10]
    return f"{path.stem[:88]}__{prefix}{path.suffix.lower()}"


def sqlite_derivatives(source: Path, temp: Path, stem: str) -> list[Path]:
    import duckdb

    output: list[Path] = []
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as sqlite:
        tables = [row[0] for row in sqlite.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
    with duckdb.connect() as duck:
        # The lab DB mixes SQLite storage classes within declared INTEGER
        # columns. Read losslessly as text; the original DB remains attached
        # as an archive for typed downstream reconstruction.
        duck.execute("SET sqlite_all_varchar=true")
        escaped = str(source).replace("'", "''")
        duck.execute(f"ATTACH '{escaped}' AS source (TYPE sqlite, READ_ONLY)")
        for table in tables:
            safe = "".join(c if c.isalnum() or c == "_" else "_" for c in table)
            path = temp / f"{stem}__{safe}.parquet"
            quoted = table.replace('"', '""')
            target = str(path).replace("'", "''")
            duck.execute(f"COPY (SELECT * FROM source.\"{quoted}\") TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            output.append(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    rows = [row for row in audit["files"] if row["path"].startswith(args.family + "/")]
    rows = [row for row in rows if not row.get("hosted_matches")]
    name, base_description = FAMILIES[args.family]
    existing = request("GET", "/datasets")["datasets"]
    matches = [item for item in existing if item["name"] == name]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous existing dataset name: {name}")
    dataset = matches[0] if matches else request("POST", "/datasets", body={"name": name, "description": base_description})["dataset"]
    dataset_id = dataset["id"]
    receipt = json.loads(args.receipt.read_text()) if args.receipt.exists() else {"datasets": {}}
    receipt["datasets"].setdefault(args.family, {"id": dataset_id, "files": {}})
    current = receipt["datasets"][args.family]
    if current["id"] != dataset_id:
        raise RuntimeError("Receipt points to a different hosted dataset")
    attached = {f["originalName"]: f for f in request("GET", f"/datasets/{dataset_id}")["dataset"]["sourceFiles"]}
    for row in rows:
        relative = row["path"]
        source = args.root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != row["bytes"]:
            raise RuntimeError(f"Size changed since audit: {relative}")
        upload_name = unique_name(relative)
        if row["bytes"] > MAX_BYTES:
            previous = current["files"].get(relative, {})
            if previous.get("status") not in {"hosted-parquet", "hosted-archive"}:
                current["files"][relative] = {"status": "pending-large", "sha256": row["sha256"], "bytes": row["bytes"]}
            continue
        with tempfile.TemporaryDirectory(prefix="zoer-davis-") as temp:
            temp_dir = Path(temp)
            derivatives: list[Path] = []
            if source.suffix.lower() in {".pdf", ".paw", ".rds", ".db", ".sqlite", ".zip"}:
                upload_name += ".zip"
                archive = temp_dir / upload_name
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                    # Zoer scans supported archive members during a build.
                    # Keep the exact binary as a downloadable member without
                    # trying to mount SQLite from NFS or parse a reference PDF.
                    z.write(source, arcname=source.name + ".bin")
                payload = archive
                if source.suffix.lower() in {".db", ".sqlite"}:
                    derivatives = sqlite_derivatives(source, temp_dir, Path(unique_name(relative)).stem)
            else:
                payload = source
            if any(p.stat().st_size > MAX_BYTES for p in [payload, *derivatives]):
                previous = current["files"].get(relative, {})
                if previous.get("status") not in {"hosted-parquet", "hosted-archive"}:
                    current["files"][relative] = {"status": "pending-large", "sha256": row["sha256"], "bytes": row["bytes"]}
                continue
            old_name = unique_name(relative)
            if payload != source and old_name in attached:
                request("DELETE", f"/datasets/{dataset_id}/files/{attached[old_name]['id']}")
                del attached[old_name]
            for item, name_to_upload in [(payload, upload_name), *[(p, p.name) for p in derivatives]]:
                if name_to_upload not in attached:
                    result = upload(dataset_id, item, name_to_upload)
                    if name_to_upload not in result.get("uploaded", []):
                        raise RuntimeError(f"Upload rejected: {relative}: {result}")
                    attached = {f["originalName"]: f for f in result["dataset"]["sourceFiles"]}
                    print(f"uploaded {relative} as {name_to_upload}", flush=True)
            current["files"][relative] = {"status": "hosted", "upload_name": upload_name,
                                          "file_id": attached[upload_name]["id"], "sha256": row["sha256"], "bytes": row["bytes"],
                                          "derived_files": [attached[p.name]["id"] for p in derivatives]}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="zoer-davis-manifest-") as temp:
        manifest_path = Path(temp) / "source_manifest.jsonl"
        manifest_path.write_text("".join(json.dumps({"path": row["path"], **current["files"][row["path"]]}) + "\n" for row in rows))
        if "source_manifest.jsonl" in attached:
            request("DELETE", f"/datasets/{dataset_id}/files/{attached['source_manifest.jsonl']['id']}")
        result = upload(dataset_id, manifest_path, "source_manifest.jsonl")
        if "source_manifest.jsonl" not in result.get("uploaded", []):
            raise RuntimeError(f"Manifest upload rejected: {result}")
    pending = sum(item["status"] == "pending-large" for item in current["files"].values())
    description = base_description + (f" {pending} audited large source files pending ingestion." if pending else " All audited source files are covered here or in linked transformed/archive datasets.")
    request("PATCH", f"/datasets/{dataset_id}", body={"description": description})
    built = request("POST", f"/datasets/{dataset_id}/rebuild")["dataset"]
    if built["status"] != "ready":
        raise RuntimeError(f"Build did not reach ready: {built['status']}: {built.get('errorMessage')}")
    current["table_count"] = built.get("tableCount")
    current["status"] = built["status"]
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"family": args.family, "dataset_id": dataset_id, "files": len(rows), "pending": pending,
                      "tables": built.get("tableCount"), "rows": built.get("rowCount")}), flush=True)


if __name__ == "__main__":
    main()
