"""Index downloaded FED IMPROVE Aerosol files and extract thermal fractions.

The yearly ZIPs are full, validated IMPROVE Aerosol exports (all sites and
parameters). The two 2026 text reports are raw FED Query Wizard exports with
all available sites, parameters, and output fields. Keep preliminary results
separate from validated results; never treat them as interchangeable.

Run from the repository root with ``uv run python``. Outputs live under the
git-ignored ``output/tables/improve_tor_fractions`` directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output/tables/improve_tor_fractions"
RAW = OUT / "raw"
FRACTIONS = {
    "ECf": "EC_TOR",
    "OCf": "OC_TOR",
    "EC1f": "EC1",
    "EC2f": "EC2",
    "EC3f": "EC3",
    "OC1f": "OC1",
    "OC2f": "OC2",
    "OC3f": "OC3",
    "OC4f": "OC4",
    "OPf": "OPTR",
    "OPTf": "OPTT",
}
QUERY_FILES = {
    "IMPAER_2026_Jan_validated_FED_query_20260923.txt": "validated",
    "IMPAERPRE_2026_Feb-Apr_FED_query_20260923.txt": "preliminary",
}
QUERY_OUTPUT_URLS = {
    "IMPAER_2026_Jan_validated_FED_query_20260923.txt": (
        "https://views.cira.colostate.edu/fed/Temp/DataFiles/"
        "ahzs645_20260923_121028_PMswt.txt"
    ),
    "IMPAERPRE_2026_Feb-Apr_FED_query_20260923.txt": (
        "https://views.cira.colostate.edu/fed/Temp/DataFiles/"
        "ahzs645_20260923_120904_NtvL0.txt"
    ),
}


def report_chunks(path: Path):
    """Yield data rows from a FED report after its metadata sections."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line in stream:
            if line.startswith("Dataset,SiteCode,POC,Date,ParamCode,"):
                header = next(csv.reader([line]))
                break
        else:
            raise ValueError(f"No data header in {path}")
        yield from pd.read_csv(stream, names=header, chunksize=250_000, low_memory=False)


def archive_chunks(path: Path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Damaged ZIP: {path}")
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected one data member in {path}: {names}")
        with archive.open(names[0]) as stream:
            yield from pd.read_csv(stream, sep="|", chunksize=250_000, low_memory=False)


def scan(path: Path, validation: str, *, report: bool) -> tuple[dict, pd.DataFrame]:
    rows = 0
    sites: set[str] = set()
    parameters: set[str] = set()
    first = "9999-99-99"
    last = "0000-00-00"
    fraction_parts = []
    for chunk in report_chunks(path) if report else archive_chunks(path):
        date_col = "Date" if report else "FactDate"
        value_col = "Val" if report else "FactValue"
        units_col = "Unit" if report else "Units"
        rows += len(chunk)
        sites.update(chunk["SiteCode"].dropna().astype(str).unique())
        parameters.update(chunk["ParamCode"].dropna().astype(str).unique())
        dates = pd.to_datetime(chunk[date_col], errors="raise")
        first = min(first, dates.min().strftime("%Y-%m-%d"))
        last = max(last, dates.max().strftime("%Y-%m-%d"))
        fraction = chunk.loc[
            chunk["ParamCode"].isin(FRACTIONS),
            ["SiteCode", "POC", "ParamCode", date_col, value_col, "Status", units_col],
        ].copy()
        if not fraction.empty:
            fraction["SampleDate"] = pd.to_datetime(fraction.pop(date_col)).dt.strftime("%Y-%m-%d")
            fraction["Value_ugm3"] = pd.to_numeric(fraction.pop(value_col), errors="coerce").replace(-999, np.nan)
            if set(fraction[units_col].dropna()) != {"ug/m^3"}:
                raise ValueError(f"Unexpected fraction units in {path}")
            fraction_parts.append(fraction.drop(columns=units_col))
    fractions = pd.concat(fraction_parts, ignore_index=True)
    fractions["validation"] = validation
    fractions["source_file"] = path.name
    if fractions.duplicated(["SiteCode", "POC", "SampleDate", "ParamCode"]).any():
        raise ValueError(f"Duplicate fraction keys in {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "file": path.name,
        "source": (
            f"https://vibe.cira.colostate.edu/data/export/IMPAER/{path.name}"
            if not report else QUERY_OUTPUT_URLS[path.name]
        ),
        "validation": validation,
        "date_min": first,
        "date_max": last,
        "rows": rows,
        "sites": len(sites),
        "parameters": len(parameters),
        "fraction_rows": len(fractions),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }
    return manifest, fractions


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = [(RAW / f"IMPAER_{year}.txt.zip", "validated", False)
             for year in range(2019, 2026)]
    files += [(RAW / name, status, True) for name, status in QUERY_FILES.items()]
    missing = [str(path) for path, _, _ in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing download(s): " + ", ".join(missing))

    manifest = []
    fraction_parts = []
    for path, validation, report in files:
        entry, fraction = scan(path, validation, report=report)
        print(f"{path.name}: {entry['rows']:,} rows; {entry['date_min']}–{entry['date_max']}")
        manifest.append(entry)
        fraction_parts.append(fraction)

    long = pd.concat(fraction_parts, ignore_index=True)
    index = ["validation", "source_file", "SiteCode", "POC", "SampleDate"]
    values = long.pivot(index=index, columns="ParamCode", values="Value_ugm3")
    statuses = long.pivot(index=index, columns="ParamCode", values="Status")
    values = values.rename(columns={k: f"{v}_ugm3" for k, v in FRACTIONS.items()})
    statuses = statuses.rename(columns={k: f"status_{v}" for k, v in FRACTIONS.items()})
    wide = values.join(statuses).reset_index()
    numeric = [f"{name}_ugm3" for name in FRACTIONS.values()]
    wide["fractions_complete"] = wide[numeric].notna().all(axis=1)
    wide["EC_closure_error_ugm3"] = (
        wide["EC_TOR_ugm3"] - wide["EC1_ugm3"] - wide["EC2_ugm3"]
        - wide["EC3_ugm3"] + wide["OPTR_ugm3"]
    )
    wide["OC_closure_error_ugm3"] = (
        wide["OC_TOR_ugm3"] - wide["OC1_ugm3"] - wide["OC2_ugm3"]
        - wide["OC3_ugm3"] - wide["OC4_ugm3"] - wide["OPTR_ugm3"]
    )
    wide = wide.sort_values(["SampleDate", "SiteCode", "POC", "validation"])
    table = OUT / "improve_tor_fractions_2019_2026_all_sites.csv"
    wide.to_csv(table, index=False)
    keys = ["SiteCode", "POC", "SampleDate"]
    overlaps = wide.groupby(keys)["validation"].nunique().gt(1).sum()
    summary = {
        "retrieved_on": "2026-09-23",
        "source_catalog": "https://views.cira.colostate.edu/fed/DataFiles/",
        "query_wizard": "https://views.cira.colostate.edu/fed/QueryWizard/",
        "files": manifest,
        "fraction_output": table.name,
        "fraction_samples": len(wide),
        "complete_fraction_samples": int(wide["fractions_complete"].sum()),
        "overlapping_validated_preliminary_keys": int(overlaps),
        "notes": [
            "Yearly ZIPs contain all published IMPROVE Aerosol parameters, not only carbon fractions.",
            "The 2026 Query Wizard reports contain all selected sites, parameters, fields, and embedded metadata.",
            "The preliminary dataset's metadata lists January–April 2026, but its current full-year query returned only February–April; the validated dataset supplied January.",
            "Preliminary data are unvalidated; blank corrections are estimated and uncertainty/MDL are not calculated.",
            "-999 source values are missing in the extracted fraction table; raw files retain original values and flags.",
            "SiteCode/date/POC identifies an IMPROVE sample; it is not a SPARTAN filter identifier.",
        ],
    }
    (OUT / "archive_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(wide):,} fraction samples; {int(wide['fractions_complete'].sum()):,} complete")
    print(table)


if __name__ == "__main__":
    main()
