"""Host the new IMPROVE Aerosol archive without confusing source and derived rows.

The repository's generated copy is used for reliable reads; matching file
sizes are checked against the Google Drive folder. Original bytes are kept as
opaque, downloadable ZIP members, and the two CSV analysis tables become typed
Parquet. Every uploaded artifact is read back from Zoer before success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import duckdb

from ingest_davis_drive import MAX_BYTES, request, upload
from zoer_env import davis_data_root, zoer_api

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parents[1] / "output/tables/improve_tor_fractions"
DEFAULT_DRIVE = davis_data_root() / "improve_tor_fractions"
RECEIPT = HERE / "improve_tor_hosted.receipt.json"
DATASET_NAME = "IMPROVE Aerosol TOR fractions 2019–2026"
DESCRIPTION = (
    "Validated 2019–2025 and January 2026 IMPROVE Aerosol exports, plus separately "
    "flagged preliminary February–April 2026 records. Two queryable Parquet tables "
    "contain thermal fractions and the FTIR-pool site/date match. Raw FED exports "
    "and original CSVs remain downloadable with source hashes in source_manifest. "
    "These overlap the existing IMPROVE portal chemistry and FTIR calibration pool; "
    "site/date/POC is the IMPROVE sample key, not a SPARTAN FilterId."
)
DERIVED = {
    "improve_tor_fractions_2019_2026_all_sites.csv": "improve_tor_fractions_all_sites.parquet",
    "improve_tor_fractions_pool_site_dates.csv": "improve_tor_fractions_pool_matches.parquet",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def hosted_sha256(dataset_id: str, file_id: str) -> str:
    digest = hashlib.sha256()
    with urlopen(f"{zoer_api()}/datasets/{dataset_id}/files/{file_id}/download", timeout=120) as response:
        for piece in iter(lambda: response.read(4 * 1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def ensure_uploaded(dataset_id: str, path: Path, name: str, attached: dict) -> str:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"Upload is over the Zoer limit: {name}")
    if name not in attached:
        result = upload(dataset_id, path, name)
        if name not in result.get("uploaded", []):
            raise RuntimeError(f"Upload rejected: {name}: {result}")
        attached.update({f["originalName"]: f for f in result["dataset"]["sourceFiles"]})
        print(f"uploaded {name}", flush=True)
    file_id = attached[name]["id"]
    if hosted_sha256(dataset_id, file_id) != sha256(path):
        raise RuntimeError(f"Hosted download differs from upload: {name}")
    return file_id


def make_opaque(source: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(source, arcname=source.name + ".bin")
    if output.stat().st_size > MAX_BYTES:
        raise ValueError(f"Opaque archive still exceeds upload limit: {source}")
    with zipfile.ZipFile(output) as archive:
        if len(archive.namelist()) != 1:
            raise RuntimeError(f"Unexpected archive member count: {output}")
        digest = hashlib.sha256(archive.read(archive.namelist()[0])).hexdigest()
    if digest != sha256(source):
        raise RuntimeError(f"Opaque archive changed source bytes: {source}")


def opaque_parts(source: Path, stage: Path, base_name: str) -> list[tuple[Path, str, str]]:
    """Return ZIP path, member, and SHA-256 for exact source byte parts."""
    if source.stat().st_size <= MAX_BYTES:
        path = stage / base_name
        make_opaque(source, path)
        return [(path, source.name + ".bin", sha256(source))]
    parts = []
    with source.open("rb") as stream:
        for index, piece in enumerate(iter(lambda: stream.read(40 * 1024 * 1024), b""), start=1):
            name = base_name.removesuffix(".zip") + f"__part{index:03d}.zip"
            path = stage / name
            member = f"part{index:03d}.bin"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr(member, piece)
            if path.stat().st_size > MAX_BYTES:
                raise RuntimeError(f"Archive part exceeds upload limit: {name}")
            parts.append((path, member, hashlib.sha256(piece).hexdigest()))
    digest = hashlib.sha256()
    for path, member, _ in parts:
        with zipfile.ZipFile(path) as archive:
            digest.update(archive.read(member))
    if digest.hexdigest() != sha256(source):
        raise RuntimeError(f"Archive parts changed source bytes: {source}")
    return parts


def make_parquet(source: Path, output: Path) -> dict:
    with duckdb.connect() as db:
        db.execute("SET threads=4")
        db.execute("SET memory_limit='3GB'")
        db.execute(f"COPY (SELECT * FROM read_csv_auto({quote(source)}, sample_size=-1, header=true)) TO {quote(output)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        rows = db.execute(f"SELECT count(*) FROM read_csv_auto({quote(source)}, sample_size=-1, header=true)").fetchone()[0]
        output_rows = db.execute(f"SELECT count(*) FROM read_parquet({quote(output)})").fetchone()[0]
        keys = ("validation, source_file, SiteCode, POC, SampleDate" if "all_sites" in source.name
                else "Site, SampleDate")
        duplicate_keys = db.execute(f"SELECT count(*) FROM (SELECT {keys}, count(*) n FROM read_parquet({quote(output)}) GROUP BY {keys} HAVING n > 1)").fetchone()[0]
        if rows != output_rows or duplicate_keys:
            raise RuntimeError(f"Conversion grain failed for {source.name}: {rows}, {output_rows}, {duplicate_keys}")
    return {"rows": rows, "sha256": sha256(output), "bytes": output.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--drive", type=Path, default=DEFAULT_DRIVE)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    source_root = args.source.expanduser().resolve()
    drive_root = args.drive.expanduser()
    source_paths = sorted(p for p in source_root.rglob("*") if p.is_file())
    if len(source_paths) != 14:
        raise RuntimeError(f"Expected 14 source files, found {len(source_paths)}")
    archive = json.loads((source_root / "archive_manifest.json").read_text())
    expected = {f"raw/{entry['file']}": entry for entry in archive["files"]}
    for path in source_paths:
        relative = path.relative_to(source_root).as_posix()
        drive_path = drive_root / relative
        if not drive_path.is_file() or drive_path.stat().st_size != path.stat().st_size:
            raise RuntimeError(f"Drive mirror is missing or differs in size: {relative}")
        if relative in expected and (sha256(path) != expected[relative]["sha256"] or path.stat().st_size != expected[relative]["bytes"]):
            raise RuntimeError(f"Raw source differs from archive manifest: {relative}")
    matches = [item for item in request("GET", "/datasets")["datasets"] if item["name"] == DATASET_NAME]
    if len(matches) > 1:
        raise RuntimeError("Ambiguous hosted dataset name")
    dataset = matches[0] if matches else request("POST", "/datasets", body={"name": DATASET_NAME, "description": DESCRIPTION})["dataset"]
    dataset_id = dataset["id"]
    attached = {f["originalName"]: f for f in request("GET", f"/datasets/{dataset_id}")["dataset"]["sourceFiles"]}
    entries = []
    with tempfile.TemporaryDirectory(prefix="zoer-improve-tor-") as temp:
        stage = Path(temp)
        for source in source_paths:
            relative = source.relative_to(source_root).as_posix()
            source_hash = sha256(source)
            opaque_name = source.name + "__" + hashlib.sha256(relative.encode()).hexdigest()[:10] + ".zip"
            if source.stat().st_size > MAX_BYTES and opaque_name in attached:
                request("DELETE", f"/datasets/{dataset_id}/files/{attached[opaque_name]['id']}")
                del attached[opaque_name]
            parts = []
            for opaque_path, member, member_sha in opaque_parts(source, stage, opaque_name):
                opaque_id = ensure_uploaded(dataset_id, opaque_path, opaque_path.name, attached)
                parts.append({"name": opaque_path.name, "fileId": opaque_id,
                              "sha256": sha256(opaque_path), "bytes": opaque_path.stat().st_size,
                              "member": member, "memberSha256": member_sha})
            entry = {"path": f"Davis Data/improve_tor_fractions/{relative}", "sha256": source_hash,
                     "bytes": source.stat().st_size, "sourceUrl": expected.get(relative, {}).get("source"),
                     "validation": expected.get(relative, {}).get("validation"),
                     "downloads": parts,
                     "storage": "concatenate numbered ZIP .bin members" if len(parts) > 1 else "single ZIP .bin member"}
            if source.name in DERIVED:
                parquet = stage / DERIVED[source.name]
                converted = make_parquet(source, parquet)
                converted["name"] = parquet.name
                converted["fileId"] = ensure_uploaded(dataset_id, parquet, parquet.name, attached)
                entry["queryableParquet"] = converted
            entries.append(entry)
        source_manifest = stage / "source_manifest.jsonl"
        source_manifest.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries))
        if source_manifest.name in attached:
            request("DELETE", f"/datasets/{dataset_id}/files/{attached[source_manifest.name]['id']}")
            del attached[source_manifest.name]
        manifest_id = ensure_uploaded(dataset_id, source_manifest, source_manifest.name, attached)
    request("PATCH", f"/datasets/{dataset_id}", body={"description": DESCRIPTION})
    built = request("POST", f"/datasets/{dataset_id}/rebuild")["dataset"]
    if built["status"] != "ready":
        raise RuntimeError(f"Dataset build failed: {built.get('errorMessage')}")
    table_rows = {table["name"]: table["rowCount"] for table in built["lastBuildSummary"]["tables"]}
    if table_rows.get("improve_tor_fractions_all_sites") != archive["fraction_samples"] or table_rows.get("improve_tor_fractions_pool_matches") != 13031 or table_rows.get("source_manifest") != 14:
        raise RuntimeError(f"Unexpected hosted table rows: {table_rows}")
    receipt = {"datasetId": dataset_id, "datasetName": DATASET_NAME, "sourceRoot": "Davis Data/improve_tor_fractions",
               "sourceFiles": entries, "sourceManifestFileId": manifest_id,
               "tables": table_rows, "status": built["status"],
               "driveVerification": "all 14 path names and byte sizes matched; direct Drive byte reads stalled; repository copy hashes were checked against archive_manifest.json"}
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    readback = request("GET", f"/datasets/{dataset_id}")["dataset"]
    if readback["status"] != "ready" or readback["tableCount"] != len(table_rows):
        raise RuntimeError("Dataset readback did not match build")
    print(json.dumps({"datasetId": dataset_id, "sourceFiles": len(entries), "tables": table_rows,
                      "receipt": str(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
