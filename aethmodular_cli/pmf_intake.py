"""Auditable future-batch eligibility and frozen historical membership.

Input metadata CSV must include:
filter_id,site,particle_fraction,analysis_id,analysis_status,date,volume_m3,lot
Optional: exclusion_reason,blank_count,quality_flag.
This module does not manufacture uncertainties or derive spectra from raw scans.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def evaluate(records, historical_ids=(), minimum_blanks=2):
    """Return one auditable disposition per physical filter, with no fixed cohort count."""
    historical = set(historical_ids)
    selected = {}
    for row in records:
        key = (row["site"], row["filter_id"])
        if not key[0] or not key[1]:
            raise ValueError("site and filter_id are required")
        selected.setdefault(key, []).append(row)
    results = []
    for (site, filter_id), candidates in sorted(selected.items()):
        # An explicit retained scan is required when there are multiple analyses.
        retained = [r for r in candidates if r["analysis_status"].strip().lower() == "retained"]
        if len(retained) != 1:
            results.append({"site": site, "filter_id": filter_id,
                            "status": "held", "reason": "expected exactly one retained analysis",
                            "historical_102": filter_id in historical})
            continue
        row = retained[0]
        reason = ""
        status = "eligible"
        if row["particle_fraction"].strip().upper() != "PM2.5":
            status, reason = "excluded", "not PM2.5"
        elif row.get("exclusion_reason", "").strip():
            status, reason = "excluded", row["exclusion_reason"].strip()
        elif not row.get("date", "").strip() or not row.get("volume_m3", "").strip():
            status, reason = "held", "missing sampling date or volume"
        else:
            try:
                if float(row["volume_m3"]) <= 0:
                    status, reason = "held", "invalid sampled volume"
            except ValueError:
                status, reason = "held", "invalid sampled volume"
        if status == "eligible" and not row.get("lot", "").strip():
            status, reason = "held", "unknown filter lot"
        if status == "eligible":
            try:
                count = int(row.get("blank_count", ""))
            except ValueError:
                count = 0
            if count < minimum_blanks:
                status, reason = "held", "insufficient matching blanks"
        results.append({
            "site": site, "filter_id": filter_id, "analysis_id": row["analysis_id"],
            "status": status, "reason": reason, "historical_102": filter_id in historical,
        })
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--historical-ids", type=Path,
                        help="CSV with filter_id column, explicit historical allowlist")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-blanks", type=int, default=2)
    args = parser.parse_args(argv)
    if args.minimum_blanks < 2:
        parser.error("minimum-blanks must be >= 2")
    with args.metadata.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"filter_id", "site", "particle_fraction", "analysis_id",
                    "analysis_status", "date", "volume_m3", "lot"}
        if not required.issubset(reader.fieldnames or []):
            parser.error(f"missing metadata columns: {sorted(required - set(reader.fieldnames or []))}")
        records = list(reader)
    historical = set()
    if args.historical_ids:
        with args.historical_ids.open(newline="", encoding="utf-8-sig") as stream:
            historical = {r["filter_id"] for r in csv.DictReader(stream)}
    results = evaluate(records, historical, args.minimum_blanks)
    if args.output.exists():
        parser.error("output exists; do not overwrite an existing audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "site", "filter_id", "analysis_id", "status", "reason", "historical_102"])
        writer.writeheader()
        writer.writerows(results)
    counts = {status: sum(r["status"] == status for r in results)
              for status in ("eligible", "held", "excluded")}
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
