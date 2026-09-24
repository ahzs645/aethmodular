"""Add the audited filter-identity caveat to the saved AETH Data Space."""

from __future__ import annotations

import json

from ingest_davis_drive import request

PROJECT = "a264ff30-5603-4617-a7ff-a2ca22835972"
SPACE = "519a43d9-a84f-4f45-9884-e72b9b747e2f"
RECORD = f"/projects/{PROJECT}/database/collections/data_spaces/records/{SPACE}"
OLD_NOTE = (" Shared filters: all 750 four-site FTIR full FilterIds match HIPS v2; "
            "548 match a shortened four-site ChemSpec base ID. Check ChemSpec Method_Code "
            "and Collection_Description before calling records the same physical filter: "
            "ChemSpec includes nylon and other analytical media as well as FTIR/HIPS.")
NOTE = (" Shared filters: all 750 four-site FTIR full IDs match HIPS v2. "
        "ChemSpec base IDs link 500 FTIR-method and 542 HIPS-method entries on "
        "stretched Teflon with matching local start dates. Other ChemSpec rows "
        "use ion chromatography or nylon; check method and medium before "
        "claiming the same physical filter or non-destructive analysis.")


def main() -> None:
    record = request("GET", RECORD)
    data = record["data"]
    if data["name"] != "AETH research" or data["version"] != 1:
        raise RuntimeError("Unexpected Data Space")
    description = data["description"]
    if OLD_NOTE in description:
        description = description.replace(OLD_NOTE, NOTE)
    elif NOTE not in description:
        description += NOTE
    if len(description) > 2000:
        raise RuntimeError("Data Space description limit exceeded")
    updated = {**data, "description": description}
    if updated != data:
        request("PUT", RECORD, body={"data": updated, "expectedRevision": record["revision"]})
    saved = request("GET", RECORD)
    if saved["data"] != updated:
        raise RuntimeError("Data Space readback mismatch")
    print(json.dumps({"spaceId": SPACE, "revision": saved["revision"],
                      "descriptionLength": len(description)}))


if __name__ == "__main__":
    main()
