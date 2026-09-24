"""Link the verified IMPROVE TOR archive to the existing AETH Data Space."""

from __future__ import annotations

import json

from ingest_davis_drive import request

DATASET_ID = "bf0bc7b0-6a0b-4948-b6cc-9428f4572e87"
PROJECT = "a264ff30-5603-4617-a7ff-a2ca22835972"
SPACE = "519a43d9-a84f-4f45-9884-e72b9b747e2f"
RECORD = f"/projects/{PROJECT}/database/collections/data_spaces/records/{SPACE}"
MANIFEST = f"/projects/{PROJECT}/database/data-spaces/{SPACE}/manifest"
NOTE = (" IMPROVE Aerosol TOR fractions (2019–2026) now include validated and separately "
        "flagged preliminary records, plus exact raw FED downloads. Join IMPROVE samples by "
        "SiteCode + POC + SampleDate; 2025 values can differ from the earlier portal workbook. "
        "The FTIR-pool match is derived from those samples, not extra observations.")


def main() -> None:
    dataset = request("GET", f"/datasets/{DATASET_ID}")["dataset"]
    if dataset["status"] != "ready" or dataset["tableCount"] != 3:
        raise RuntimeError("IMPROVE TOR dataset is not ready")
    record = request("GET", RECORD)
    data = record["data"]
    if data["name"] != "AETH research" or data["version"] != 1:
        raise RuntimeError("Unexpected Data Space")
    resources = list(data["resources"])
    if not any(ref["kind"] == "dataset" and ref["id"] == DATASET_ID for ref in resources):
        resources.append({"kind": "dataset", "id": DATASET_ID})
    description = data["description"]
    if "IMPROVE Aerosol TOR fractions (2019–2026)" not in description:
        description += NOTE
    updated = {**data, "resources": resources, "description": description}
    if len(description) > 2000 or len(resources) > 100:
        raise RuntimeError("Data Space limits exceeded")
    if updated != data:
        request("PUT", RECORD, body={"data": updated, "expectedRevision": record["revision"]})
    saved = request("GET", RECORD)
    if saved["data"] != updated:
        raise RuntimeError("Data Space readback mismatch")
    manifest = request("GET", MANIFEST)
    if not any(item.get("id") == DATASET_ID and item.get("status") == "ready" for item in manifest["resources"]):
        raise RuntimeError("New dataset missing from resolved Space manifest")
    print(json.dumps({"spaceId": SPACE, "revision": saved["revision"], "resources": len(resources),
                      "datasetId": DATASET_ID, "status": "ready"}))


if __name__ == "__main__":
    main()
