"""Replace the AETH Data Space's linked catalog index and verify readback."""

from __future__ import annotations

import json
from pathlib import Path

from ingest_davis_drive import request, upload

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[3] / "research/ftir_hips_chem/output/tables/zoer_sources"
DATASET_ID = "561b587f-2d8d-4d9a-9839-078c811a3b19"
PATHS = [OUTPUT / "catalog_datasets.jsonl", OUTPUT / "catalog_files.jsonl",
         OUTPUT / "catalog_lineage.jsonl", HERE / "zoer_hosted_catalog.manifest.json"]


def main() -> None:
    manifest = json.loads(PATHS[-1].read_text())
    if (len(manifest["datasets"]), len(manifest["files"]), len(manifest["lineage"])) != (31, 494, 24):
        raise RuntimeError("Unexpected catalog size")
    for path in PATHS:
        item = request("GET", f"/datasets/{DATASET_ID}")["dataset"]
        old = [f for f in item["sourceFiles"] if f["originalName"] == path.name]
        if len(old) > 1:
            raise RuntimeError(f"Duplicate catalog source: {path.name}")
        if old:
            request("DELETE", f"/datasets/{DATASET_ID}/files/{old[0]['id']}")
        result = upload(DATASET_ID, path, path.name)
        if path.name not in result.get("uploaded", []):
            raise RuntimeError(f"Catalog upload failed: {path.name}")
        print(f"uploaded {path.name}", flush=True)
    request("PATCH", f"/datasets/{DATASET_ID}", body={"description":
        "Consolidated source index for AETH research: 31 linked research datasets, 494 hosted source or archive files and 24 lineage relationships. catalog_datasets, catalog_files and catalog_lineage expose dataset IDs, source paths, hashes when available, table names and row counts. The linked detailed JSON includes IMPROVE/SPARTAN source provenance, the Davis Data Drive audit, and a tested method-aware filter sampling-time map. Raw, revised and derived observations can overlap; the index does not merge scientific rows."})
    built = request("POST", f"/datasets/{DATASET_ID}/rebuild")["dataset"]
    if built["status"] != "ready":
        raise RuntimeError(f"Catalog rebuild failed: {built.get('errorMessage')}")
    names = {item["originalName"] for item in built["sourceFiles"]}
    if names != {path.name for path in PATHS}:
        raise RuntimeError(f"Catalog source list mismatch: {names}")
    table_rows = {item["name"]: item["rowCount"] for item in built["lastBuildSummary"]["tables"]}
    if (table_rows.get("catalog_datasets"), table_rows.get("catalog_files"), table_rows.get("catalog_lineage")) != (31, 494, 24):
        raise RuntimeError(f"Catalog row count mismatch: {table_rows}")
    print(json.dumps({"dataset_id": DATASET_ID, "status": built["status"], "rows": table_rows}), flush=True)


if __name__ == "__main__":
    main()
