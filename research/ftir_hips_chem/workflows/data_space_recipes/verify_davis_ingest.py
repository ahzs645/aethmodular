"""Verify hosted Davis Data bytes and derived Parquet against the ingest receipt."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

from ingest_davis_drive import request
from zoer_env import zoer_api

SMALL = ["DAVIS", "EC-HIPS-Aeth Comparison", "Han", "Purple Air Data", "Weather Data", "FTIR"]
LARGE = ["FTIR local spectra", "FTIR local scans", "FTIR large results"]


def download(dataset_id: str, file_id: str) -> bytes:
    with urlopen(f"{zoer_api()}/datasets/{dataset_id}/files/{file_id}/download", timeout=180) as response:
        return response.read()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--mode", choices=["small", "parquet", "archive", "all"], default="all")
    args = parser.parse_args()
    records = json.loads(args.receipt.read_text())["datasets"]
    checked = 0
    if args.mode in {"small", "all"}:
        for key in SMALL:
            entry = records[key]
            dataset_id = entry["id"]
            hosted = request("GET", f"/datasets/{dataset_id}")["dataset"]
            if hosted["status"] != "ready":
                raise RuntimeError(f"Dataset not ready: {key}")
            for relative, source in entry["files"].items():
                if source["status"] != "hosted":
                    continue
                data = download(dataset_id, source["file_id"])
                if source["upload_name"].endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(data)) as archive:
                        members = archive.namelist()
                        if len(members) != 1 or not members[0].endswith(".bin"):
                            raise RuntimeError(f"Unexpected archive layout: {relative}")
                        data = archive.read(members[0])
                if digest(data) != source["sha256"] or len(data) != source["bytes"]:
                    raise RuntimeError(f"Hosted source differs: {relative}")
                checked += 1
            print(f"verified {key}", flush=True)
    if args.mode in {"parquet", "all"}:
        for key in LARGE:
            entry = records[key]
            dataset_id = entry["id"]
            hosted = request("GET", f"/datasets/{dataset_id}")["dataset"]
            if hosted["status"] != "ready":
                raise RuntimeError(f"Dataset not ready: {key}")
            table_rows = {row["sourceFileId"]: row["rowCount"] for row in hosted["lastBuildSummary"]["tables"]}
            for relative, source in entry["files"].items():
                total = 0
                for part in source["parts"]:
                    data = download(dataset_id, part["file_id"])
                    if digest(data) != part["sha256"] or len(data) != part["bytes"]:
                        raise RuntimeError(f"Hosted Parquet differs: {relative}/{part['name']}")
                    total += table_rows[part["file_id"]]
                    checked += 1
                if total != source["rows"]:
                    raise RuntimeError(f"Hosted Parquet row mismatch: {relative}: {total} != {source['rows']}")
            print(f"verified {key}", flush=True)
    if args.mode in {"archive", "all"}:
        entry = records["FTIR binary archive"]
        dataset_id = entry["id"]
        if request("GET", f"/datasets/{dataset_id}")["dataset"]["status"] != "ready":
            raise RuntimeError("Archive dataset not ready")
        for relative, source in entry["files"].items():
            full_hash = hashlib.sha256()
            total = 0
            for part in source["parts"]:
                downloaded = download(dataset_id, part["file_id"])
                with zipfile.ZipFile(io.BytesIO(downloaded)) as archive:
                    members = archive.namelist()
                    if len(members) != 1 or not members[0].endswith(".bin"):
                        raise RuntimeError(f"Unexpected archive layout: {relative}/{part['name']}")
                    data = archive.read(members[0])
                if digest(data) != part["sha256"] or len(data) != part["bytes"]:
                    raise RuntimeError(f"Archive part differs: {relative}/{part['name']}")
                full_hash.update(data)
                total += len(data)
                checked += 1
            if full_hash.hexdigest() != source["sha256"] or total != source["bytes"]:
                raise RuntimeError(f"Reconstructed archive differs: {relative}")
            print(f"verified {relative}", flush=True)
    print(json.dumps({"checked_hosted_files_or_parts": checked, "mode": args.mode}), flush=True)


if __name__ == "__main__":
    main()
