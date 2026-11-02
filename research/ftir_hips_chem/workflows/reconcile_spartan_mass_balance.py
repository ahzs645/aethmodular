"""Compare complete release snapshots without substituting one residual for another.

Gallery inputs already carry central exclusion flags; this workflow audits the
previously identified missing cohort and preserves every source value and flag.
Outputs are separate audit tables. No public gallery asset is edited.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from common.spartan_io import RAW_DIR, build_datetime, read_spartan_csv
from data_paths import maia_data_root

OUT = ROOT / "research/ftir_hips_chem/output/tables/spartan_mass_balance_audit"
REFERENCE_URL = "https://www.spartan-network.org/_files/ugd/6321a4_219864c4f1454c58a956339bc58b9120.xlsx?dn=Sampling_Parameters_Methods_2.3.xlsx"


def long_table(path: Path) -> pd.DataFrame:
    frame = read_spartan_csv(path)
    frame.columns = frame.columns.str.strip()
    frame["date"] = build_datetime(frame).reindex(frame.index)
    frame["value_ugm3"] = pd.to_numeric(frame.Value, errors="coerce")
    nano = frame.Units.str.strip().str.startswith("Nano", na=False)
    frame.loc[nano, "value_ugm3"] /= 1000
    return frame


def wide_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pivot_table(index="Filter_ID", columns="Parameter_Name", values="value_ugm3", aggfunc="first")


def delta_summary(values: pd.Series) -> dict:
    valid = values.dropna()
    return {"n": int(len(valid)), "median_signed": float(valid.median()) if len(valid) else None,
            "median_absolute": float(valid.abs().median()) if len(valid) else None,
            "p95_absolute": float(valid.abs().quantile(.95)) if len(valid) else None,
            "max_absolute": float(valid.abs().max()) if len(valid) else None,
            "within_0_02": int(valid.abs().le(.02 + 1e-9).sum())}


def reconcile() -> dict:
    drive = maia_data_root()
    missing = {r["id"]: r for r in json.loads((OUT / "audit.json").read_text())["records"]}
    summary, comparisons, history, source_hashes = [], [], [], {}
    for code in ("CHTS", "ETAD", "INDH", "USPA"):
        old_path = drive / "Combine csv files/Chemical Speciation/Rec2.5" / f"FilterBased_ReconstrPM25_{code}.csv"
        current_path = RAW_DIR / "FilterBased/ReconstrPM25" / f"FilterBased_ReconstrPM25_{code}.csv"
        old_long, current_long = long_table(old_path), long_table(current_path)
        old, current = wide_table(old_long), wide_table(current_long)
        old_dates, current_dates = (d.groupby("Filter_ID").date.first() for d in (old_long, current_long))
        ids = old.index.intersection(current.index)
        same_date = old_dates.reindex(ids).eq(current_dates.reindex(ids))
        valid_ids = ids[same_date]
        o, n = old.reindex(valid_ids), current.reindex(valid_ids)
        for residual in ("Residual Matter Only", "Residual Matter with OC", "Residual Matter with OM"):
            diff = o["Residual Matter"] - n[residual]
            summary.append({"site": code, "comparison": "old Residual Matter minus " + residual, **delta_summary(diff)})
            comparisons.extend({"site": code, "id": fid, "date": str(old_dates[fid].date()),
                                "old_residual": float(o.loc[fid, "Residual Matter"]),
                                "current_parameter": residual, "current_residual": float(n.loc[fid, residual]),
                                "difference": float(value)} for fid, value in diff.dropna().items())
        identity = n["Residual Matter Only"] - n["Residual Matter with OC"] - n["OC PM2.5"].fillna(0)
        summary.append({"site": code, "comparison": "current Only minus current with OC minus current OC", **delta_summary(identity)})
        for name in ("Filter PM2.5 mass", "BC PM2.5", "OC PM2.5", "Fine Soil", "Sea Salt", "Trace Element Oxides"):
            summary.append({"site": code, "comparison": "old minus current input: " + name, **delta_summary(o[name] - n[name])})
        source_hashes[str(old_path.relative_to(drive))] = hashlib.sha256(old_path.read_bytes()).hexdigest()
        source_hashes[str(current_path.relative_to(ROOT))] = hashlib.sha256(current_path.read_bytes()).hexdigest()
        for title, frame in (("archive_v3_0", old_long),):
            for fid in sorted(set(missing).intersection(old.index)):
                if pd.isna(old.loc[fid].get("Residual Matter")) or pd.isna(old.loc[fid].get("Filter PM2.5 mass")):
                    continue
                rows = frame[frame.Filter_ID.eq(fid)].copy()
                rows["release_source"] = title
                rows["source_relative_path"] = str(old_path.relative_to(drive))
                rows["current_release_status"] = "absent_or_no_compatible_balance"
                history.append(rows)
    # The earlier v2.4 archive contributes two further physical JPL filters.
    earlier_path = drive / "EC-HIPS-Aeth Comparison/Data/SPARTAN portal downloads/USPA/FilterBased_ReconstrPM25_USPA.csv"
    earlier = long_table(earlier_path)
    wide = wide_table(earlier)
    for fid in sorted(set(missing).intersection(wide.index)):
        if pd.notna(wide.loc[fid].get("Residual Matter")) and pd.notna(wide.loc[fid].get("Filter PM2.5 mass")):
            rows = earlier[earlier.Filter_ID.eq(fid)].copy()
            rows["release_source"] = "archive_v2_4"
            rows["source_relative_path"] = str(earlier_path.relative_to(drive))
            rows["current_release_status"] = "absent_or_no_compatible_balance"
            history.append(rows)
    source_hashes[str(earlier_path.relative_to(drive))] = hashlib.sha256(earlier_path.read_bytes()).hexdigest()
    archival = pd.concat(history, ignore_index=True)
    archival.to_csv(OUT / "archived_missing_filter_components.csv", index=False)
    # Review releases independently; absent values remain absent. Old releases
    # do not expose the same wet/dry component parameter set.
    review = []
    for (fid, release), rows in archival.groupby(["Filter_ID", "release_source"]):
        values = wide_table(rows).loc[fid]
        flags = sorted({str(flag).strip() for cell in rows.Flag.dropna()
                        for flag in str(cell).split(";") if str(flag).strip()})
        review.append({"FilterId": fid, "date": str(rows.date.iloc[0].date()),
                       "release_source": release, "pm_ugm3": values.get("Filter PM2.5 mass"),
                       "legacy_residual_ugm3": values.get("Residual Matter"),
                       "separate_OC_reported": pd.notna(values.get("OC PM2.5")),
                       "source_flags": ";".join(flags),
                       "reported_parameters": ";".join(sorted(values.dropna().index)),
                       "source_relative_path": rows.source_relative_path.iloc[0],
                       "current_stack_status": "pending_release_definition_reconciliation"})
    pd.DataFrame(review).to_csv(OUT / "jpl_historical_balance_review.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "release_comparison_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "release_comparison_pairs.csv", index=False)
    flag_rows = archival[archival.Parameter_Name.eq("Residual Matter")]
    flags = Counter(code.strip() for cell in flag_rows.Flag.dropna() for code in str(cell).split(";") if code.strip())
    result = {"checked": datetime.now(timezone.utc).isoformat(), "units": "ug/m3 after explicit nanogram conversion",
              "historical_physical_filters": int(archival.Filter_ID.nunique()),
              "historical_versions": {str(k): int(v) for k, v in archival.groupby("release_source").Filter_ID.nunique().items()},
              "residual_row_flags_by_version": dict(flags),
              "flag_definitions": {"SGP": "Screen or Grid Pattern", "VIH": "Visible Inhomogeneous Deposition", "SUS": "Suspect or Compromised Data"},
              "method_reference": REFERENCE_URL, "source_sha256": source_hashes, "comparisons": summary,
              "decision": "Keep archived components as versioned historical observations. Do not rename legacy Residual Matter or subtract OC to populate the current twelve-component stack.",
              "reason": "Current Only - with OC equals OC within reporting precision, but no universal old-to-current residual mapping holds. Soil, sea salt, trace oxides and some carbon values were revised while measured PM mass is generally unchanged.",
              "claim_limit": "Historical presence is established; the publisher's reason for subsequent absence is unresolved. SGP and VIH are morphology flags, not the SUS flag. No additional scientific exclusion is inferred."}
    (OUT / "reconciliation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = reconcile()
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_sha256", "comparisons")}, indent=2))
