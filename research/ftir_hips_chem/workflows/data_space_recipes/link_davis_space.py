"""Link verified Davis Data datasets to the existing AETH Data Space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingest_davis_drive import request

PROJECT = "a264ff30-5603-4617-a7ff-a2ca22835972"
SPACE = "519a43d9-a84f-4f45-9884-e72b9b747e2f"
RECORD = f"/projects/{PROJECT}/database/collections/data_spaces/records/{SPACE}"
MANIFEST = f"/projects/{PROJECT}/database/data-spaces/{SPACE}/manifest"
NEW_GROUPS = ["DAVIS", "EC-HIPS-Aeth Comparison", "Han", "Purple Air Data", "Weather Data",
              "FTIR", "FTIR large results", "FTIR local scans", "FTIR local spectra",
              "FTIR binary archive", "Davis source notes"]
NOTE = (" Davis Data Drive coverage (2026-09-23): all 136 audited scientific/reference files are "
        "accounted for: 3 exact SPARTAN files, 2 transformed IMPROVE workbooks, and 131 DAVIS, "
        "EC/HIPS, FTIR, Han, PurpleAir and Weather files in linked source families. FTIR CSVs over "
        "the upload limit are typed Parquet partitions; RDS models and the SQLite mirror are "
        "checksum-verified downloadable parts. The notes dataset includes archive documentation. "
        "Historical releases and derived measurements overlap; do not sum rows as independent observations.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())["datasets"]
    ids = [receipt[key]["id"] for key in NEW_GROUPS]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate dataset IDs in receipt")
    for key, dataset_id in zip(NEW_GROUPS, ids):
        item = request("GET", f"/datasets/{dataset_id}")["dataset"]
        if item["status"] != "ready":
            raise RuntimeError(f"Dataset not ready: {key}")
    record = request("GET", RECORD)
    data = record["data"]
    if data["name"] != "AETH research" or data["version"] != 1:
        raise RuntimeError("Unexpected Data Space; refusing to change it")
    resources = list(data["resources"])
    known = {(r["kind"], r["id"]) for r in resources}
    for dataset_id in ids:
        if ("dataset", dataset_id) not in known:
            resources.append({"kind": "dataset", "id": dataset_id})
    description = data["description"]
    if "Davis Data Drive coverage (2026-09-23):" not in description:
        description += NOTE
    updated = {**data, "resources": resources, "description": description}
    if len(description) > 2000 or len(resources) > 100:
        raise RuntimeError("Data Space schema limit would be exceeded")
    if updated != data:
        request("PUT", RECORD, body={"data": updated, "expectedRevision": record["revision"]})
    saved = request("GET", RECORD)
    if saved["data"] != updated:
        raise RuntimeError("Saved Data Space failed readback")
    manifest = request("GET", MANIFEST)
    print(json.dumps({"space_id": SPACE, "revision": saved["revision"],
                      "resources": len(resources), "new_datasets": len(ids),
                      "manifest_keys": list(manifest)}))


if __name__ == "__main__":
    main()
