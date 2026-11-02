"""Audit saved lab deliveries against the historical 141-filter request.

Read-only with respect to source workbooks, the database and gallery assets.
Run after ``uv run aeth doctor``. Downloads and live SQL responses are retained
separately in the gitignored evidence directory; this workflow never fetches or
executes their contents. Workbook row numbers are one-based Excel rows.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from aethmodular_cli.env import display_path

EVIDENCE = ROOT / "data/aqrc/spartan_mass_balance/further_probe_20261102"
OUT = ROOT / "research/ftir_hips_chem/output/tables/spartan_mass_balance_audit"
FIELDS = ("volume", "aCOH", "aCH", "naCO", "COOH", "OM", "FTIR_OC", "FTIR_EC")
FG = ("aCOH", "aCH", "naCO", "COOH", "OM")


def numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and pd.notna(value)


def audit() -> dict:
    requests = list(csv.DictReader((OUT / "aqrc_mass_balance_pull/requested_filters.csv").open()))
    requested = {row["FilterId"]: row for row in requests}
    assert len(requested) == len(requests) == 141
    sources = json.loads((EVIDENCE / "drive_deliveries.json").read_text())
    matches, sheets = [], []
    for source in sources:
        path = EVIDENCE / source["local"]
        with path.open("rb") as handle:
            workbook = openpyxl.load_workbook(handle, read_only=True, data_only=True)
            for sheet in workbook:
                values = list(sheet.values)
                header = list(values[1]) if len(values) > 1 else []
                # An ID-only scratch tab is not a measurement table.
                measurement = "sample" in header and "date" in header
                sheets.append({"source": source["name"], "sheet": sheet.title,
                               "rows": sheet.max_row, "columns": sheet.max_column,
                               "measurement_table": measurement})
                if not measurement:
                    continue
                for number, row in enumerate(values[2:], 3):
                    record = dict(zip(header, row))
                    found = re.search(r"((?:CHTS|ETAD|INDH|USPA))[_-](\d{4})(?=[_\-.]|$)", str(record.get("sample")))
                    if not found:
                        continue
                    fid = f"{found[1]}-{found[2]}"
                    if fid not in requested:
                        continue
                    date = record["date"]
                    date = date.date().isoformat() if isinstance(date, datetime) else str(date)
                    matches.append({"source": source["name"], "sheet": sheet.title,
                                    "row": number, "FilterId": fid,
                                    "request_date": requested[fid]["Date"], "source_date": date,
                                    "date_match": date == requested[fid]["Date"],
                                    "sample": record["sample"],
                                    **{key: record.get(key) for key in FIELDS},
                                    "numeric_FG_OM": all(numeric(record.get(key)) for key in FG),
                                    "numeric_OC_EC": all(numeric(record.get(key)) for key in ("FTIR_OC", "FTIR_EC"))})
            workbook.close()
    frame = pd.DataFrame(matches)
    frame.to_csv(OUT / "lab_delivery_matched_rows.csv", index=False)
    qualified = frame[frame.date_match & frame.numeric_FG_OM]
    # Preserve all versions in the detailed table. This set counts identities,
    # not a selected or harmonized concentration across conflicting versions.
    qualified_ids = set(qualified.FilterId)
    unified = pd.read_pickle(ROOT / "research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl")
    unified["base_id"] = unified.FilterId.str.extract(r"^([A-Z]{4}-\d{4})(?:-\d+)?$")
    existing = set(unified.loc[unified.Parameter.eq("OM"), "base_id"])
    missing_om = sorted(qualified_ids - existing)
    frame[frame.FilterId.isin(missing_om)].to_csv(OUT / "lab_delivery_additional_organic_candidates.csv", index=False)
    conflicts = frame[~frame.date_match]
    conflicts.to_csv(OUT / "lab_delivery_date_conflicts.csv", index=False)
    original = frame[frame.source.eq("batch4_original")].set_index("FilterId")
    v2 = frame[frame.source.eq("batch4_v2")].set_index("FilterId")
    assert original.index.is_unique and v2.index.is_unique
    shared = original.index.intersection(v2.index)
    changes = []
    for fid in shared:
        for key in FIELDS:
            before, after = original.loc[fid, key], v2.loc[fid, key]
            if before != after and not (pd.isna(before) and pd.isna(after)):
                changes.append({"FilterId": fid, "field": key, "original": before, "v2": after})
    pd.DataFrame(changes).to_csv(OUT / "lab_delivery_batch4_version_differences.csv", index=False)
    query_names = ("improve_spartan_barcodes", "cohort_integrity", "delhi_delivery_na_volume",
                   "networks_qc", "networks_paths", "networks_types", "networks_integrity_issues")
    queries = {}
    for name in query_names:
        response = json.loads((EVIDENCE / (name + ".json")).read_text())
        assert response["ok"] and not response["hasMore"]
        assert not response.get("byteBudgetReached") and not response.get("truncatedCellCount")
        queries[name] = {key: response.get(key) for key in
                         ("database", "rowCount", "limit", "hasMore", "byteBudgetReached", "truncatedCellCount")}
    delhi = json.loads((EVIDENCE / "delhi_delivery_na_volume.json").read_text())["rows"]
    na_ids = set(frame.loc[frame.source.eq("batch4_v2") & ~frame.numeric_FG_OM, "FilterId"])
    assert {row["FilterId"] for row in delhi} == na_ids
    null_volumes = [row["FilterId"] for row in delhi if row["SampleVolume_m3"] is None and row["MediaSampleVolume"] is None]
    publisher = json.loads((EVIDENCE / "publisher_recheck.json").read_text())
    summary = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_directory": display_path(EVIDENCE), "requested_filters": len(requested),
        "workbooks": sources, "sheets": sheets,
        "by_source": [{"source": name, "matched_ids": int(group.FilterId.nunique()),
                       "matching_date_rows": int(group.date_match.sum()),
                       "numeric_FG_OM_rows": int(group.numeric_FG_OM.sum())}
                      for name, group in frame.groupby("source")],
        "matching_date_numeric_FG_OM_ids": len(qualified_ids),
        "qualified_by_site": dict(Counter(fid.split("-")[0] for fid in qualified_ids)),
        "already_have_OM_in_unified": len(qualified_ids & existing),
        "additional_organic_candidate_ids": missing_om,
        "date_conflict_ids": sorted(set(conflicts.FilterId)),
        "batch4_changes_by_field": dict(Counter(row["field"] for row in changes)),
        "batch4_na_concentration_ids": sorted(na_ids),
        "native_sql_null_volume_ids": null_volumes, "live_queries": queries,
        "publisher_files_checked": len(publisher),
        "publisher_all_match_cache_and_alternate_prefix": all(
            row["matches_existing_cached_source"] and row["by_site_matches_by_product"] for row in publisher),
        "additional_compatible_reconstructions_recovered": 0,
        "interpretation": "Versioned organic results and metadata gaps, not final reconstruction or release statuses. No data imported or exclusions changed.",
    }
    (OUT / "further_probe_summary.json").write_text(json.dumps(summary, indent=2))
    manifest = [{"path": display_path(path), "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(EVIDENCE.rglob("*")) if path.is_file() and path.name != "manifest.json"]
    (EVIDENCE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {key: summary[key] for key in ("requested_filters", "matching_date_numeric_FG_OM_ids",
                                        "already_have_OM_in_unified", "additional_organic_candidate_ids",
                                        "date_conflict_ids", "batch4_changes_by_field",
                                        "publisher_all_match_cache_and_alternate_prefix")}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
