"""Refresh the gallery's validated IMPROVE population from FED.

Completed years use the published IMPAER yearly ZIPs. Export the current year
from FED Query Wizard (IMPROVE Aerosol, all sites/parameters/fields, normalized
pipe-delimited text, data and metadata) and pass its saved file with --query.
Preliminary exports are never inputs to this workflow. Run with ``uv run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/ftir_hips_chem/output/tables/improve_validated_archive"
RAW = OUT / "raw"
CATALOG = "https://views.cira.colostate.edu/fed/DataFiles/"
QUERY = "https://views.cira.colostate.edu/fed/QueryWizard/?dsidse=10001"


def build_gallery(files, receipt):
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
    from improve_io import clean_improve, improve_clean_path, read_improve_integrated_export
    from outliers import get_clean_data
    from aethmodular_cli.env import display_path

    flagged_dir = OUT / "flagged_samples"
    flagged_dir.mkdir(exist_ok=True)
    frames = []
    sites = pd.DataFrame()
    coverage_notes = []
    for entry in files:
        wide, metadata = read_improve_integrated_export(RAW / entry["file"])
        wide.to_parquet(flagged_dir / (entry["file"].split(".")[0] + ".parquet"), index=False)
        if not metadata.empty:
            sites = metadata
        entry.update(samples=len(wide), sites=int(wide.SiteCode.nunique()),
                     date_min=wide.Date.min().strftime("%Y-%m-%d"),
                     date_max=wide.Date.max().strftime("%Y-%m-%d"),
                     retained_samples=int((~wide.is_excluded).sum()))
        absorption_end = wide.loc[wide.fAbs_Val.notna(), "Date"].max()
        entry["absorption_date_max"] = absorption_end.strftime("%Y-%m-%d") if pd.notna(absorption_end) else None
        if pd.notna(absorption_end) and (wide.Date.max() - absorption_end).days > 60:
            coverage_notes.append(f"FED's current {absorption_end.year} absorption records end on "
                                  f"{absorption_end:%Y-%m-%d}; later samples cannot enter the paired EC/absorption population.")
        # The full flagged values/methods/QA remain in Parquet and raw downloads.
        # Keep concentrations, flows, flags and uncertainties in the CSV cache.
        columns = [c for c in wide if not c.endswith(("_Method", "_Unit", "_Reported"))]
        frames.append(get_clean_data(wide[columns]).copy())
        print(f"{entry['file']}: {entry['samples']:,} samples, {entry['retained_samples']:,} retained",
              flush=True)
    if sites.empty:
        raise ValueError("The validated Query Wizard report must include Sites metadata")
    samples = pd.concat(frames, ignore_index=True)
    keys = ["SiteCode", "POC", "Date"]
    if samples.duplicated(keys).any():
        raise ValueError("Overlapping validated inputs: duplicate site/date/POC samples")
    clean = clean_improve(samples, metadata={"Sites": sites})
    if clean.validation.ne("validated").any() or clean.Dataset.ne("IMPAER").any():
        raise ValueError("Preliminary records reached the validated gallery cache")
    unlocated = sorted(clean.loc[clean[["Latitude", "Longitude"]].isna().any(axis=1),
                                "SiteCode"].unique().tolist())
    if clean.duplicated(keys).any() or len(clean) != len(samples):
        raise ValueError("Unexpected loss or duplication when deriving cleaned IMPROVE columns")
    clean = clean.sort_values(keys).reset_index(drop=True)
    cache = improve_clean_path()
    audit = {"previous": None}
    if cache.exists():
        with cache.open("rb") as stream:
            previous_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        old = pd.read_csv(cache, usecols=keys + ["ECf_Val", "fAbs_Val"], parse_dates=["Date"])
        backup = OUT / f"previous_cleaned_{previous_sha[:12]}.csv"
        if not backup.exists():
            shutil.copy2(cache, backup)
        comparison = old.merge(clean[keys + ["ECf_Val", "fAbs_Val"]], on=keys,
                               how="outer", suffixes=("_old", "_new"), indicator=True,
                               validate="one_to_one")
        common = comparison.loc[comparison._merge.eq("both")]
        audit["previous"] = {"samples": len(old), "sites": int(old.SiteCode.nunique()),
                             "date_max": old.Date.max().strftime("%Y-%m-%d"),
                             "sha256": previous_sha, "backup": display_path(backup),
                             "common_samples": len(common),
                             "new_samples": int(comparison._merge.eq("right_only").sum()),
                             "removed_samples": int(comparison._merge.eq("left_only").sum()),
                             "changed_ec": int((~np.isclose(common.ECf_Val_old, common.ECf_Val_new,
                                                            rtol=0, atol=1e-9)).sum()),
                             "changed_absorption": int((~np.isclose(common.fAbs_Val_old, common.fAbs_Val_new,
                                                                    rtol=0, atol=1e-9)).sum())}
        comparison.loc[comparison._merge.ne("both")].to_csv(OUT / "sample_changes.csv", index=False)
    temp = cache.with_suffix(".csv.part")
    clean.to_csv(temp, index=False)
    with temp.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt.update(cache=display_path(cache), cache_sha256=digest,
                   samples=len(clean), sites=int(clean.SiteCode.nunique()),
                   date_min=clean.Date.min().strftime("%Y-%m-%d"),
                   date_max=clean.Date.max().strftime("%Y-%m-%d"),
                   unlocated_sites=unlocated,
                   coverage_notes=coverage_notes,
                   month_counts=clean.Date.dt.strftime("%Y-%m").value_counts().sort_index().to_dict(),
                   notes=["Validated IMPAER only; IMPAERPRE is stored separately and never read.",
                          "Preserves the gallery's 2003-onward scope.",
                          "Sample identity is SiteCode/POC/Date; sampler module is retained per parameter.",
                          "Only dated samples with source-valid, positive EC and absorption enter this comparison population.",
                          "FED V* qualified values are retained with their original flags; invalid/missing values are unavailable for statistics.",
                          "Flagged Parquet samples preserve excluded records and original reported values.",
                          "Older archives may lack flow/duration or module codes; unavailable volumes remain missing.",
                          "Carbon mass uses module C volume; PM2.5 mass uses module A volume."],
                   audit=audit)
    # All validation has passed before replacing the existing cache.
    temp.replace(cache)
    serialized = json.dumps(receipt, indent=2) + "\n"
    cache.with_suffix(".manifest.json").write_text(serialized)
    (OUT / "archive_manifest.json").write_text(serialized)
    print(json.dumps({k: receipt[k] for k in ["samples", "sites", "date_min", "date_max", "audit"]},
                     indent=2), flush=True)


def download_year(year: int, *, refresh=False) -> dict:
    name = f"IMPAER_{year}.txt.zip"
    url = f"https://vibe.cira.colostate.edu/data/export/IMPAER/{name}"
    path = RAW / name
    if refresh or not path.exists():
        temp = path.with_suffix(".zip.part")
        with urllib.request.urlopen(url, timeout=180) as response, temp.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        with zipfile.ZipFile(temp) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Damaged archive: {name}")
        temp.replace(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"{name}: {path.stat().st_size / 1e6:.1f} MB", flush=True)
    return {"file": name, "source": url, "validation": "validated",
            "bytes": path.stat().st_size, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2003,
                        help="Preserve the gallery's existing 2003-onward scope")
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--query", type=Path, help="Saved validated current-year FED report")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--refresh-downloads", action="store_true",
                        help="Re-download completed years to pick up any FED redelivery")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(lambda year: download_year(year, refresh=args.refresh_downloads),
                              range(args.start_year, args.end_year + 1)))
    receipt = {"retrieved_on": datetime.now(timezone.utc).isoformat(),
               "source_catalog": CATALOG, "query_wizard": QUERY,
               "validation": "validated", "files": files}
    (OUT / "download_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not args.download_only:
        query = args.query
        if query is None:
            candidates = list(RAW.glob(f"IMPAER_{args.end_year + 1}_validated_FED_query_*.txt"))
            if len(candidates) != 1:
                raise SystemExit("Pass --query with exactly one validated current-year FED report")
            query = candidates[0]
        name = f"IMPAER_{args.end_year + 1}_validated_FED_query_{datetime.now(timezone.utc):%Y%m%d}.txt"
        destination = RAW / name
        if query.resolve() != destination.resolve():
            shutil.copy2(query, destination)
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({"file": name, "source": QUERY, "validation": "validated",
                      "bytes": destination.stat().st_size, "sha256": digest,
                      "selection": "All sites, parameters and fields; current year, all months; normalized pipe-delimited text; data and metadata"})
        build_gallery(files, receipt)


if __name__ == "__main__":
    main()
