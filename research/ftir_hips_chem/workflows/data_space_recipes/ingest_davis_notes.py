"""Index Davis Data archive notes and an authenticated Google Doc export."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from ingest_davis_drive import request, upload

NAME = "AETH Davis Data source notes"
DOC_URL = "https://docs.google.com/document/d/1XrBYoxyZHabHt5E0YFMStOjElg6vKJHPvF_hCgs9SCA/edit"
FOLDERS = ["DAVIS", "EC-HIPS-Aeth Comparison", "FTIR", "Han", "Improve", "Purple Air Data", "Spartan", "Weather Data"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--doc-text", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.root / name for name in ["README.md", "GOOGLE-DRIVE-NOTES.md", "CLEANUP-LOG.md"]]
    paths += sorted(path for folder in FOLDERS for path in (args.root / folder).rglob("*.md"))
    rows = []
    for path in paths:
        data = path.read_bytes()
        rows.append({"path": str(path.relative_to(args.root)), "sourceType": "markdown",
                     "sourceUrl": None, "sha256": hashlib.sha256(data).hexdigest(),
                     "bytes": len(data), "text": data.decode("utf-8-sig")})
    doc_bytes = args.doc_text.read_bytes()
    rows.append({"path": "Notes about Data.gdoc", "sourceType": "google-doc-exported-text",
                 "sourceUrl": DOC_URL, "sha256": hashlib.sha256(doc_bytes).hexdigest(),
                 "bytes": len(doc_bytes), "text": doc_bytes.decode("utf-8-sig")})
    matches = [item for item in request("GET", "/datasets")["datasets"] if item["name"] == NAME]
    if len(matches) > 1:
        raise RuntimeError("Ambiguous notes dataset")
    item = matches[0] if matches else request("POST", "/datasets", body={
        "name": NAME,
        "description": "Research archive README and MANIFEST notes plus text fetched from the linked Google Doc. The notes describe provenance, data-use conditions, duplicate revisions and interpretation; they are not additional observations.",
    })["dataset"]
    dataset_id = item["id"]
    attached = {f["originalName"]: f for f in request("GET", f"/datasets/{dataset_id}")["dataset"]["sourceFiles"]}
    with tempfile.TemporaryDirectory(prefix="zoer-davis-notes-") as temporary:
        file = Path(temporary) / "davis_source_notes.jsonl"
        file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        if file.name in attached:
            request("DELETE", f"/datasets/{dataset_id}/files/{attached[file.name]['id']}")
        result = upload(dataset_id, file, file.name)
        if file.name not in result.get("uploaded", []):
            raise RuntimeError(f"Notes upload rejected: {result}")
    built = request("POST", f"/datasets/{dataset_id}/rebuild")["dataset"]
    if built["status"] != "ready":
        raise RuntimeError(f"Notes build failed: {built.get('errorMessage')}")
    receipt = json.loads(args.receipt.read_text())
    receipt["datasets"]["Davis source notes"] = {"id": dataset_id, "status": "ready", "files": {
        row["path"]: {"sha256": row["sha256"], "bytes": row["bytes"], "sourceType": row["sourceType"]}
        for row in rows}}
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"dataset_id": dataset_id, "notes": len(rows), "tables": built.get("tableCount")}))


if __name__ == "__main__":
    main()
