"""Consolidate local recipe receipts and the live Zoer inventory into one manifest."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from openpyxl import load_workbook

from zoer_env import zoer_api, zoer_url


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUTPUT = REPO / "research/ftir_hips_chem/output/tables/zoer_sources"
UNIFIED_ID = "caa04b1f-7f86-4c34-b059-a5a8d04c7501"
IMPROVE_ID = "9c6f0a21-c0c6-4955-9bc1-b40077ca9622"
KYAN_HOURLY_ID = "d76ecec7-6caf-44e2-a551-92586b68223a"
IMPROVE_RAW_ID = "768e3bb0-d2c8-45ea-8b7f-a50fad328cf9"
SPARTAN_HIPS_ID = "c5a0dd7b-1068-41d3-b023-309998a70823"
SPARTAN_SITES_ID = "5576fd5b-f39b-41bb-919f-434b77bfc8f3"
SPARTAN_CHEMSPEC_PM25_ID = "7955ed6c-3b53-46e1-8443-caa80ae64cdf"
METEOSTAT_ID = "9b49715d-2a06-4dec-be1a-b6219ed03188"
IMPROVE_TOR_ID = "bf0bc7b0-6a0b-4948-b6cc-9428f4572e87"


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


def add_hosted_source(target: dict, dataset_id: str, name: str, checksum: str, input_path: str) -> None:
    if name in target[dataset_id]["files"]:
        raise ValueError(f"Duplicate hosted source name: {dataset_id}/{name}")
    target[dataset_id]["files"][name] = {
        "sha256": checksum,
        "inputPath": input_path,
        "sourceLocation": "Davis Data/Spartan",
    }


def hosted_sha256(dataset_id: str, file_id: str) -> str:
    digest = hashlib.sha256()
    with urlopen(f"{zoer_api()}/datasets/{dataset_id}/files/{file_id}/download", timeout=30) as response:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hosted_site_codes(file_id: str) -> set[str]:
    url = f"{zoer_api()}/datasets/{SPARTAN_SITES_ID}/files/{file_id}/download"
    with urlopen(url, timeout=30) as response:
        workbook = load_workbook(BytesIO(response.read()), read_only=True, data_only=True)
    rows = workbook.active.values
    if next(rows)[0] != "SiteCode":
        raise ValueError("Unexpected SPARTAN site lookup header")
    return {str(row[0]).strip() for row in rows if row[0]}


def main() -> None:
    core = load(OUTPUT / "hosted_ingest_receipt.json")
    raw = load(OUTPUT / "spartan_raw/hosted_ingest_receipt.json")
    raw_provenance = load(OUTPUT / "spartan_raw/spartan_raw.provenance.json")
    chem_provenance = load(OUTPUT / "four_site_chemspec.provenance.json")
    unified_provenance = load(REPO / "research/ftir_hips_chem/output/tables/unified_filter_dataset.provenance.json")
    improve_provenance = load(OUTPUT / "improve_valid_cleaned.provenance.json")
    kyan_provenance = load(REPO / "research/ftir_hips_chem/output/tables/kyan_aethalometry/kyan_corrected_hourly_v1.provenance.json")
    improve_raw_provenance = load(OUTPUT / "improve_raw/improve_raw.provenance.json")

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
    entries[KYAN_HOURLY_ID] = {"files": {}, "recipe": "kyan_hourly_parquet.manifest.json"}
    kyan_name = Path(kyan_provenance["output"]).name
    add_file(entries, KYAN_HOURLY_ID, kyan_name, kyan_provenance["output"], kyan_provenance["outputSha256"])
    entries[KYAN_HOURLY_ID]["files"][kyan_name]["inputPath"] = "Aethalometry Data/Kyan Data/MAIA data/"
    entries[KYAN_HOURLY_ID]["files"][kyan_name]["inputFiles"] = kyan_provenance["sources"]
    entries[IMPROVE_RAW_ID] = {"files": {}, "recipe": "improve_raw_parquet.manifest.json"}
    for output in improve_raw_provenance["outputs"]:
        source_kind = output["source"]
        input_name = (
            "both IMPROVE exports" if source_kind == "both"
            else improve_raw_provenance["sources"][source_kind]["fileName"]
        )
        add_file(entries, IMPROVE_RAW_ID, output["fileName"],
                 str(OUTPUT / "improve_raw" / output["fileName"]), output["sha256"],
                 f"Davis Data/Improve/{input_name}:{output['sheet']}")
    receipt_path = OUTPUT / "improve_raw/improve_raw.provenance.json"
    add_file(entries, IMPROVE_RAW_ID, receipt_path.name, str(receipt_path), sha256(receipt_path),
             "Davis Data/Improve/IMPROVE_chem-mass-fabs_2026-04-22.xlsx + IMPROVE_laser-optics-635nm_2026-04-22.xlsx")
    entries[SPARTAN_HIPS_ID] = {"files": {}, "recipe": None}
    add_hosted_source(entries, SPARTAN_HIPS_ID, "SPARTAN_HIPS_Batch1-51.v2.csv",
                      "4276eaa23b9be587a0c240d7faba53526858443c2b45110481f5c2e02eccf4fb",
                      "Davis Data/Spartan/SPARTAN_HIPS_Batch1-51.v2.csv")
    add_hosted_source(entries, SPARTAN_HIPS_ID, "SPARTAN_HIPS_Batch1-51.csv",
                      "360c5ccd111959d3b00e6165cc8ad244badfadcd4fc783fe41785db4fdc4e94b",
                      "Davis Data/Spartan/SPARTAN_HIPS_Batch1-51.csv")
    entries[SPARTAN_SITES_ID] = {"files": {}, "recipe": None}
    add_hosted_source(entries, SPARTAN_SITES_ID, "SPARTAN_Site_quick_lookup.xlsx",
                      "abd7426da8b43e79bcdc47ed956af8216041d7d105a864e7757fbdcf3ad36892",
                      "Davis Data/Spartan/SPARTAN_Site_quick_lookup.xlsx")

    with urlopen(f"{zoer_api()}/datasets", timeout=30) as response:
        hosted = {item["id"]: item for item in json.load(response)["datasets"]}
    if len(entries) != 19:
        raise ValueError(f"Expected 19 hosted research datasets, found {len(entries)}")
    datasets = []
    files = []
    for dataset_id, entry in entries.items():
        item = hosted[dataset_id]
        if item["status"] != "ready":
            raise ValueError(f"Dataset is not ready: {item['name']}")
        sources = {source["originalName"]: source for source in item["sourceFiles"]}
        source_names = set(sources)
        tables_by_source = {}
        for table in item["lastBuildSummary"]["tables"]:
            tables_by_source.setdefault(table["sourceFileId"], []).append(table)
        if source_names != set(entry["files"]) or any(source["id"] not in tables_by_source for source in sources.values()):
            raise ValueError(f"Hosted source/manifest mismatch: {item['name']}")
        recipe = entry["recipe"]
        datasets.append({
            "datasetId": dataset_id, "name": item["name"], "status": item["status"],
            "tableCount": item["tableCount"], "rowCount": item["rowEstimate"],
            "fileCount": len(source_names), "recipeManifestPath": relative(str(HERE / recipe)) if recipe else None,
            "recipeManifestSha256": sha256(HERE / recipe) if recipe else None,
        })
        for name, file in entry["files"].items():
            if "repositoryPath" in file:
                path = REPO / file["repositoryPath"]
                if not path.is_file() or sha256(path) != file["sha256"]:
                    raise ValueError(f"Local source hash changed: {file['repositoryPath']}")
            elif hosted_sha256(dataset_id, sources[name]["id"]) != file["sha256"]:
                raise ValueError(f"Hosted source hash changed: {dataset_id}/{name}")
            source_tables = tables_by_source[sources[name]["id"]]
            files.append({
                "datasetId": dataset_id, "datasetName": item["name"], "hostedName": name,
                "tableName": source_tables[0]["name"], "rowCount": sum(table["rowCount"] for table in source_tables),
                **file,
            })
    if len(files) != 260 or sum(item["rowCount"] for item in datasets) != 2_341_431:
        raise ValueError("Unexpected hosted catalog totals")
    lookup_file = hosted[SPARTAN_SITES_ID]["sourceFiles"]
    if len(lookup_file) != 1 or lookup_file[0]["originalName"] != "SPARTAN_Site_quick_lookup.xlsx":
        raise ValueError("Unexpected hosted SPARTAN site lookup")
    lookup_codes = hosted_site_codes(lookup_file[0]["id"])
    if len(lookup_codes) != 37:
        raise ValueError(f"Expected 37 SPARTAN lookup codes, found {len(lookup_codes)}")
    lineage = [
        {"sourceDatasetId": IMPROVE_RAW_ID, "targetDatasetId": IMPROVE_ID,
         "relationship": "filtered-and-joined-derivative", "joinKey": "SiteCode, POC, Date, AuxID",
         "note": "The cleaned optics table uses the same portal exports; its rows are not additional observations."},
        {"sourceDatasetId": SPARTAN_HIPS_ID, "targetDatasetId": SPARTAN_CHEMSPEC_PM25_ID,
         "relationship": "filter-to-sampling-window", "joinKey": "HIPS Site + base FilterId = ChemSpec Site_Code + Filter_ID",
         "sharedSampleFilterIds": 2541,
         "note": "2,542 of 3,963 HIPS filters match one local start/end window in public PM2.5 ChemSpec; 2,541 also have a HIPS SampleDate. One HIPS date disagrees, one public end precedes its start, and one matched HIPS date is missing. Use local hours as local time, not UTC. HIPS Fabs and public BC can be overlapping products, not independent measurements."},
        {"sourceDatasetId": SPARTAN_SITES_ID, "targetDatasetId": SPARTAN_HIPS_ID,
         "relationship": "site-reference", "joinKey": "SiteCode to Site",
         "note": "The lookup provides names and coordinates, not new filter observations."},
    ]
    for item in raw:
        site_codes = {Path(file["output"]).stem.rsplit("_", 1)[-1]
                      for file in raw_provenance["groups"][item["group"]]}
        shared = lookup_codes & site_codes
        missing = sorted(site_codes - lookup_codes)
        note = f"{len(shared)} of {len(site_codes)} hosted site tables match lookup codes."
        if missing:
            note += f" Unmatched hosted codes: {', '.join(missing)}."
        note += " The lookup adds names and coordinates, not observations."
        lineage.append({
            "sourceDatasetId": SPARTAN_SITES_ID, "targetDatasetId": item["id"],
            "relationship": "site-reference", "joinKey": "SiteCode to site table suffix",
            "sharedSiteCodes": len(shared), "hostedSiteCodes": len(site_codes),
            "unmatchedHostedSiteCodes": missing, "note": note,
        })
    # The original 19-dataset checks above remain a stable baseline. Extend
    # the current catalog with the separately audited Davis Drive families.
    davis = load(HERE / "davis_drive_hosted.receipt.json")["datasets"]
    new_groups = ["DAVIS", "EC-HIPS-Aeth Comparison", "Han", "Purple Air Data", "Weather Data",
                  "FTIR", "FTIR large results", "FTIR local scans", "FTIR local spectra",
                  "FTIR binary archive", "Davis source notes"]
    for group in new_groups:
        record = davis[group]
        item = hosted[record["id"]]
        if item["status"] != "ready":
            raise ValueError(f"Davis dataset is not ready: {group}")
        recipe = ("ingest_davis_notes.py" if group == "Davis source notes"
                  else "ingest_davis_large.py" if group.startswith("FTIR ") and group != "FTIR"
                  else "ingest_davis_drive.py")
        datasets.append({
            "datasetId": item["id"], "name": item["name"], "status": item["status"],
            "tableCount": item["tableCount"], "rowCount": item["rowEstimate"],
            "fileCount": len(item["sourceFiles"]),
            "recipeManifestPath": relative(str(HERE / recipe)),
            "recipeManifestSha256": sha256(HERE / recipe),
        })
        provenance = {}
        for input_path, source in record["files"].items():
            if source.get("status") in {"hosted-parquet", "hosted-archive"}:
                continue  # Attached to one of the large FTIR datasets below.
            if source.get("upload_name"):
                provenance[source["upload_name"]] = (input_path, source, None)
            for file_id in source.get("derived_files", []):
                provenance[file_id] = (input_path, source, None)
            for part in source.get("parts", []):
                provenance[part["file_id"]] = (input_path, source, part)
        tables_by_source = {}
        for table in item["lastBuildSummary"]["tables"]:
            tables_by_source.setdefault(table["sourceFileId"], []).append(table)
        for source_file in item["sourceFiles"]:
            source = provenance.get(source_file["id"]) or provenance.get(source_file["originalName"])
            source_tables = tables_by_source.get(source_file["id"], [])
            row = {
                "datasetId": item["id"], "datasetName": item["name"],
                "hostedName": source_file["originalName"],
                "tableName": source_tables[0]["name"] if source_tables else None,
                "rowCount": sum(table["rowCount"] for table in source_tables),
                "sourceFileId": source_file["id"],
                "inputPath": "Davis Data/" + source[0] if source else "Davis Data indexed metadata",
                "sourceSha256": source[1].get("sha256") if source else None,
                "partSha256": source[2].get("sha256") if source and source[2] else None,
                "sha256": (source[1].get("sha256") if source and source[1].get("upload_name") == source_file["originalName"]
                           and not source_file["originalName"].endswith(".zip") else None),
            }
            files.append(row)
    def group_id(key: str) -> str:
        return davis[key]["id"]
    lineage.extend([
        {"sourceDatasetId": group_id("DAVIS"), "targetDatasetId": SPARTAN_HIPS_ID,
         "relationship": "raw-filter-sampling-window", "joinKey": "DAVIS ExternalFilterId = HIPS FilterId",
         "note": "The five DAVIS filter metadata tables contain 1,048 unique filter IDs and 895 non-null start/end dates; 952 full IDs match processed HIPS. HIPS SampleDate agrees where matched. The 6,876 raw T/R rows are repeated instrument measurements, not 6,876 filters."},
        {"sourceDatasetId": group_id("EC-HIPS-Aeth Comparison"), "targetDatasetId": SPARTAN_CHEMSPEC_PM25_ID,
         "relationship": "overlapping-portal-release", "joinKey": "Filter_ID, Parameter_Code, Method_Code",
         "note": "Older ETAD/USPA releases share keys with the newer public export but contain changed values and old-only keys. Preserve release dates."},
        {"sourceDatasetId": group_id("Weather Data"), "targetDatasetId": METEOSTAT_ID,
         "relationship": "overlapping-weather-timestamps", "joinKey": "time",
         "note": "Local Meteostat and hosted AETH master overlap in timestamps but many values differ; retain both versions."},
        {"sourceDatasetId": group_id("FTIR binary archive"), "targetDatasetId": group_id("FTIR"),
         "relationship": "sqlite-export-overlap", "joinKey": "table names",
         "note": "The offline SQLite mirror has eight tables with row counts matching CSV exports; cell-level equality was not established. RDS model runs are distinct artifacts."},
    ])
    for target in ["FTIR large results", "FTIR local scans", "FTIR local spectra"]:
        lineage.append({"sourceDatasetId": group_id("FTIR"), "targetDatasetId": group_id(target),
                        "relationship": "large-source-companion",
                        "note": "Large FTIR CSVs are typed Parquet companions to the smaller source files; source_manifest records original hashes and row counts."})
    tor_receipt = load(HERE / "improve_tor_hosted.receipt.json")
    if tor_receipt["datasetId"] != IMPROVE_TOR_ID or len(tor_receipt["sourceFiles"]) != 14:
        raise ValueError("Unexpected IMPROVE TOR receipt")
    tor_mounted_check = load(HERE / "improve_tor_drive_verification.json")
    tor_cloud_check = load(HERE / "improve_tor_connector_verification.json")
    if tor_mounted_check["matched"] != 9 or tor_mounted_check["total"] != 14 or tor_cloud_check["matched"] != 5 or tor_cloud_check["total"] != 5:
        raise ValueError("IMPROVE TOR follow-up source verification is incomplete")
    tor_hosted = hosted[IMPROVE_TOR_ID]
    if tor_hosted["status"] != "ready" or tor_hosted["tableCount"] != 3:
        raise ValueError("IMPROVE TOR archive is not ready")
    datasets.append({
        "datasetId": IMPROVE_TOR_ID, "name": tor_hosted["name"], "status": tor_hosted["status"],
        "tableCount": tor_hosted["tableCount"], "rowCount": tor_hosted["rowEstimate"],
        "fileCount": len(tor_hosted["sourceFiles"]),
        "recipeManifestPath": relative(str(HERE / "ingest_improve_tor_archive.py")),
        "recipeManifestSha256": sha256(HERE / "ingest_improve_tor_archive.py"),
        "sourceVerification": "9 raw mounted Drive files and 5 top-level authenticated Drive downloads matched ingested bytes by SHA-256; raw folder cloud sync unverified",
    })
    tor_source_by_id = {item["id"]: item for item in tor_hosted["sourceFiles"]}
    tor_tables_by_source = {}
    for table in tor_hosted["lastBuildSummary"]["tables"]:
        tor_tables_by_source.setdefault(table["sourceFileId"], []).append(table)
    for entry in tor_receipt["sourceFiles"]:
        for part in [*entry["downloads"], *([entry["queryableParquet"]] if "queryableParquet" in entry else [])]:
            item = tor_source_by_id[part["fileId"]]
            if item["originalName"] != part["name"] or hosted_sha256(IMPROVE_TOR_ID, item["id"]) != part["sha256"]:
                raise ValueError(f"IMPROVE TOR hosted source changed: {part['name']}")
            tables = tor_tables_by_source.get(item["id"], [])
            files.append({
                "datasetId": IMPROVE_TOR_ID, "datasetName": tor_hosted["name"],
                "hostedName": item["originalName"], "sourceFileId": item["id"],
                "tableName": tables[0]["name"] if tables else None,
                "rowCount": sum(table["rowCount"] for table in tables),
                "inputPath": entry["path"], "sourceSha256": entry["sha256"],
                "sha256": part["sha256"], "validation": entry.get("validation"),
            })
    tor_manifest_file = tor_source_by_id[tor_receipt["sourceManifestFileId"]]
    files.append({
        "datasetId": IMPROVE_TOR_ID, "datasetName": tor_hosted["name"],
        "hostedName": tor_manifest_file["originalName"], "sourceFileId": tor_manifest_file["id"],
        "tableName": "source_manifest", "rowCount": 14,
        "inputPath": "Davis Data/improve_tor_fractions/source_manifest.jsonl",
        "sha256": hosted_sha256(IMPROVE_TOR_ID, tor_manifest_file["id"]),
    })
    lineage.extend([
        {"sourceDatasetId": SPARTAN_HIPS_ID, "targetDatasetId": "9f73e74d-31f3-44da-bd3f-8ba8513405a0",
         "relationship": "same-full-filter-id-HIPS-FTIR", "joinKey": "HIPS FilterId = FTIR FilterId",
         "sharedFilterIds": 750,
         "note": "All 750 distinct four-site FTIR full FilterIds occur in HIPS v2, with the same site and no disagreement between known sample dates. These are linked measurements on the same identified sampled filter; HIPS and FTIR values are different analytical outputs, not 750 extra samples."},
        {"sourceDatasetId": "9f73e74d-31f3-44da-bd3f-8ba8513405a0", "targetDatasetId": "f82f3a0f-8366-445a-9cfc-c4f60b1c77c6",
         "relationship": "filter-family-method-aware", "joinKey": "FTIR Site + base FilterId = ChemSpec Site_Code + Filter_ID; inspect Method_Code and Collection_Description",
         "sharedBaseFilterIds": 548,
         "note": "548 of 750 FTIR full filter IDs match a four-site ChemSpec base ID. Of these, 500 have FTIR methods 217/218 and 542 have HIPS method 221, all labeled stretched Teflon with the same local start date as FTIR. The same base IDs also link 523 ion-chromatography entries on Teflon and 86 on nylon. ChemSpec omits the full filter suffix, so verify method and medium; the entire table is neither one physical-filter assay nor uniformly non-destructive."},
        {"sourceDatasetId": "9f73e74d-31f3-44da-bd3f-8ba8513405a0", "targetDatasetId": UNIFIED_ID,
         "relationship": "same-FTIR-observations", "joinKey": "FilterId + Parameter",
         "note": "All 5,250 four-site FTIR rows match the unified table on FilterId + Parameter. Sample dates agree; 11 concentration differences are below 1e-16 and reflect text/float precision. Count these once."},
        {"sourceDatasetId": "f82f3a0f-8366-445a-9cfc-c4f60b1c77c6", "targetDatasetId": SPARTAN_CHEMSPEC_PM25_ID,
         "relationship": "overlapping-ChemSpec-export", "joinKey": "Site_Code + Filter_ID + Parameter_Code + Method_Code; inspect Value and release",
         "note": "At the four sites, 27,471 rows are exact shared rows, while 5,302 are four-site-only and 10,693 public-only. The four-field key repeats for 1,284 four-site measurements, so it is not unique; compare complete records or retain release identity."},
        {"sourceDatasetId": IMPROVE_TOR_ID, "targetDatasetId": IMPROVE_RAW_ID,
         "relationship": "overlapping-IMPROVE-release", "joinKey": "SiteCode + POC + SampleDate (portal sample_date)",
         "note": "130,643 sample keys occur in both. Validated 2019–2024 EC/OC values agree after treating -999 as missing; 2,911 real EC and OC values in 2025 differ between the newer Aerosol export and the April portal workbook. Keep release and validation status; do not silently coalesce values."},
    ])
    if len(datasets) != 31 or len(files) != 494 or len(lineage) != 24:
        raise ValueError(f"Unexpected extended catalog totals: {len(datasets)} datasets, {len(files)} files")
    manifest = {
        "schemaVersion": 1, "sourceRepository": "ahzs645/aethmodular",
        "zoerDatasetsUrl": f"{zoer_url()}/#/datasets",
        "ownerProjectId": "a264ff30-5603-4617-a7ff-a2ca22835972",
        "dataSpaceIds": ["519a43d9-a84f-4f45-9884-e72b9b747e2f", "79bad9e5-d1fc-4b3c-b246-b142f80d91c0"],
        "datasets": sorted(datasets, key=lambda item: item["name"]),
        "files": sorted(files, key=lambda item: (item["datasetName"], item["hostedName"])),
        "lineage": lineage,
        "sourceWorkbooks": improve_raw_provenance["sources"],
        "note": "Rows overlap across public source products, local revisions and derived analysis tables; these are not unique samples. Some archive files have no queryable table; their manifest records reconstruction details.",
    }
    if "/Users/" in json.dumps(manifest):
        raise ValueError("Local absolute path leaked into the hosted catalog")
    (HERE / "zoer_hosted_catalog.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "catalog_datasets.jsonl").write_text("".join(json.dumps(item) + "\n" for item in manifest["datasets"]))
    (OUTPUT / "catalog_files.jsonl").write_text("".join(json.dumps(item) + "\n" for item in manifest["files"]))
    (OUTPUT / "catalog_lineage.jsonl").write_text("".join(json.dumps(item) + "\n" for item in lineage))
    print(json.dumps({"datasets": len(datasets), "files": len(files), "tables": sum(item["tableCount"] for item in datasets), "rows": sum(item["rowCount"] for item in datasets)}))


if __name__ == "__main__":
    main()
