"""Consolidate local recipe receipts and the live Zoer inventory into one manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUTPUT = REPO / "research/ftir_hips_chem/output/tables/zoer_sources"
BASE_URL = "https://zoer.example.org/api/datasets"
UNIFIED_ID = "caa04b1f-7f86-4c34-b059-a5a8d04c7501"
IMPROVE_ID = "9c6f0a21-c0c6-4955-9bc1-b40077ca9622"


def load(path: Path) -> dict | list:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: str) -> str:
    candidate = Path(path)
    return str(candidate.relative_to(REPO) if candidate.is_absolute() else candidate)


def add_file(target: dict, dataset_id: str, name: str, path: str, checksum: str, input_path: str | None = None) -> None:
    if name in target[dataset_id]["files"]:
        raise ValueError(f"Duplicate hosted source name: {dataset_id}/{name}")
    target[dataset_id]["files"][name] = {
        "repositoryPath": relative(path),
        "sha256": checksum,
        "inputPath": relative(input_path or path),
    }


def main() -> None:
    core = load(OUTPUT / "hosted_ingest_receipt.json")
    raw = load(OUTPUT / "spartan_raw/hosted_ingest_receipt.json")
    raw_provenance = load(OUTPUT / "spartan_raw/spartan_raw.provenance.json")
    chem_provenance = load(OUTPUT / "four_site_chemspec.provenance.json")
    unified_provenance = load(REPO / "research/ftir_hips_chem/output/tables/unified_filter_dataset.provenance.json")
    improve_provenance = load(OUTPUT / "improve_valid_cleaned.provenance.json")

    entries: dict[str, dict] = {}
    recipe_info = {
        UNIFIED_ID: "unified_filter_parquet.manifest.json",
        IMPROVE_ID: "improve_clean_parquet.manifest.json",
    }
    for item in core:
        recipe = "four_site_chemspec_parquet.manifest.json" if item["name"] == "AETH four-site ChemSpec sources" else None
        entries[item["id"]] = {"files": {}, "recipe": recipe}
        if recipe:
            recipe_info[item["id"]] = recipe
        for source in item["sources"]:
            add_file(entries, item["id"], Path(source["path"]).name, source["path"], source["sha256"])
    for item in raw:
        entries[item["id"]] = {"files": {}, "recipe": "spartan_raw_parquet.manifest.json"}
        recipe_info[item["id"]] = "spartan_raw_parquet.manifest.json"
        for file in raw_provenance["groups"][item["group"]]:
            add_file(entries, item["id"], Path(file["output"]).name, file["output"], file["outputSha256"], file["source"])
    for file in chem_provenance["files"]:
        item = next(value for value in core if value["name"] == "AETH four-site ChemSpec sources")
        entries[item["id"]]["files"][Path(file["output"]).name]["inputPath"] = file["source"]
    entries[UNIFIED_ID] = {"files": {}, "recipe": recipe_info[UNIFIED_ID]}
    add_file(entries, UNIFIED_ID, Path(unified_provenance["output"]["path"]).name,
             unified_provenance["output"]["path"], unified_provenance["output"]["sha256"],
             unified_provenance["source"]["path"])
    entries[IMPROVE_ID] = {"files": {}, "recipe": recipe_info[IMPROVE_ID]}
    add_file(entries, IMPROVE_ID, Path(improve_provenance["output"]["path"]).name,
             "research/ftir_hips_chem/" + improve_provenance["output"]["path"],
             improve_provenance["output"]["sha256"],
             "research/ftir_hips_chem/" + improve_provenance["source"]["path"])

    with urlopen(BASE_URL, timeout=30) as response:
        hosted = {item["id"]: item for item in json.load(response)["datasets"]}
    if len(entries) != 15:
        raise ValueError(f"Expected 15 hosted research datasets, found {len(entries)}")
    datasets = []
    files = []
    for dataset_id, entry in entries.items():
        item = hosted[dataset_id]
        if item["status"] != "ready":
            raise ValueError(f"Dataset is not ready: {item['name']}")
        tables = {table["sourceName"]: table for table in item["lastBuildSummary"]["tables"]}
        source_names = {source["originalName"] for source in item["sourceFiles"]}
        if source_names != set(entry["files"]) or source_names != set(tables):
            raise ValueError(f"Hosted source/manifest mismatch: {item['name']}")
        recipe = entry["recipe"]
        datasets.append({
            "datasetId": dataset_id, "name": item["name"], "status": item["status"],
            "tableCount": item["tableCount"], "rowCount": item["rowEstimate"],
            "fileCount": len(source_names), "recipeManifestPath": relative(str(HERE / recipe)) if recipe else None,
            "recipeManifestSha256": sha256(HERE / recipe) if recipe else None,
        })
        for name, file in entry["files"].items():
            path = REPO / file["repositoryPath"]
            if not path.is_file() or sha256(path) != file["sha256"]:
                raise ValueError(f"Local source hash changed: {file['repositoryPath']}")
            files.append({
                "datasetId": dataset_id, "datasetName": item["name"], "hostedName": name,
                "tableName": tables[name]["name"], "rowCount": tables[name]["rowCount"],
                **file,
            })
    if len(files) != 244 or sum(item["rowCount"] for item in datasets) != 1_430_977:
        raise ValueError("Unexpected hosted catalog totals")
    manifest = {
        "schemaVersion": 1, "sourceRepository": "ahzs645/aethmodular",
        "zoerDatasetsUrl": "https://zoer.example.org/#/datasets",
        "ownerProjectId": "a264ff30-5603-4617-a7ff-a2ca22835972",
        "dataSpaceIds": ["519a43d9-a84f-4f45-9884-e72b9b747e2f", "79bad9e5-d1fc-4b3c-b246-b142f80d91c0"],
        "datasets": sorted(datasets, key=lambda item: item["name"]),
        "files": sorted(files, key=lambda item: (item["datasetName"], item["hostedName"])),
        "note": "Rows overlap across public source products and derived analysis tables; these are not unique samples.",
    }
    if "/Users/" in json.dumps(manifest):
        raise ValueError("Local absolute path leaked into the hosted catalog")
    (HERE / "zoer_hosted_catalog.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "catalog_datasets.jsonl").write_text("".join(json.dumps(item) + "\n" for item in manifest["datasets"]))
    (OUTPUT / "catalog_files.jsonl").write_text("".join(json.dumps(item) + "\n" for item in manifest["files"]))
    print(json.dumps({"datasets": len(datasets), "files": len(files), "tables": sum(item["tableCount"] for item in datasets), "rows": sum(item["rowCount"] for item in datasets)}))


if __name__ == "__main__":
    main()
