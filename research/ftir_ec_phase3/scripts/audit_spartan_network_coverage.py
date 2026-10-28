"""Audit SPARTAN method coverage without changing inputs or gallery data.

Public products, a lab HIPS delivery and optional Davis native coverage stay
separate. Presence means a finite reported value, not a QC or measurement claim.
Run with uv run; output must be a fresh directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from common.spartan_io import (  # noqa: E402
    RAW_DIR, build_datetime, hips_blank_mask, normalize_fid, read_spartan_csv,
)
from aethmodular_cli.env import display_path  # noqa: E402

PUBLIC_FIELDS = ["ec_ftir", "oc_ftir", "om_ftir", "public_hips_bc", "hips_ssr_estimate",
                 "ssr_equivalent_bc", "mass", "ions", "elements"]
NATIVE_FIELDS = {"FtirAnalyses": "native_ftir_analyses", "FtirAnalysesWithPairedScans": "native_paired_scans",
                 "HipsAnalyses": "native_hips_analyses", "HipsSampleReadings": "native_hips_sample_readings",
                 "HipsSampleAnalyses": "native_hips_sample_analyses", "HipsSampleReadingsWithTR": "native_hips_readings_with_tr"}


def finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").map(np.isfinite).fillna(False)


def classify_public(raw: pd.DataFrame) -> pd.Series:
    name = raw.Parameter_Name.fillna("").str.strip()
    method = pd.to_numeric(raw.Method_Code, errors="coerce")
    description = raw.Analysis_Description.fillna("").str.strip().str.upper()
    category = pd.Series("unclassified", index=raw.index)
    for field, parameter in [("ec_ftir", "EC PM2.5"), ("oc_ftir", "OC PM2.5"), ("om_ftir", "OM PM2.5")]:
        category.loc[name.eq(parameter) & method.isin([217, 218]) & description.eq("FTIR")] = field
    category.loc[name.eq("BC PM2.5") & method.isin([219, 221]) & description.eq("HIPS")] = "public_hips_bc"
    category.loc[name.eq("BC PM2.5") & method.eq(220) & description.str.startswith("ESTIMATED FROM THE HIPS-SSR")] = "hips_ssr_estimate"
    category.loc[name.eq("Equivalent BC PM2.5") & method.isin([212, 213, 214, 215, 216]) & description.str.startswith("SMOKE STAIN REFLECTOMETER")] = "ssr_equivalent_bc"
    category.loc[name.eq("Filter PM2.5 mass") & description.str.startswith("GRAVIMETRIC")] = "mass"
    category.loc[name.str.contains(" Ion PM2.5", regex=False) & description.str.startswith("ION CHROMATOGRAPHY")] = "ions"
    category.loc[description.str.startswith(("ED-XRF", "ICP-MS"))] = "elements"
    return category


def _catalog(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep physical IDs unique; report date conflicts rather than hiding them."""
    if frame.groupby("filter_id").site.nunique().gt(1).any():
        raise ValueError("A physical filter ID maps to multiple sites")
    return frame.groupby("filter_id").agg(site=("site", "first"), date=("date", "min"),
                                           date_conflict=("date", lambda x: x.nunique() > 1))


def load_public(public_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    paths = sorted(public_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError("No public ChemSpecPM25 CSV files were found")
    raw = pd.concat([read_spartan_csv(path) for path in paths], ignore_index=True)
    required = {"Site_Code", "Filter_ID", "Parameter_Name", "Method_Code", "Analysis_Description", "Units", "Value"}
    if not required.issubset(raw):
        raise ValueError("Public source is missing required identity/method columns")
    if raw.Filter_ID.isna().any() or raw.Site_Code.isna().any():
        raise ValueError("Public rows without filter/site identity need investigation")
    raw["filter_id"] = normalize_fid(raw.Filter_ID)
    raw["site"] = raw.Site_Code.astype(str).str.strip()
    raw["date"] = build_datetime(raw).reindex(raw.index)
    raw["finite"] = finite(raw.Value)
    raw["category"] = classify_public(raw)
    result = _catalog(raw)
    result["public_filter"] = True
    for field in PUBLIC_FIELDS:
        result[field] = result.index.isin(raw.loc[raw.finite & raw.category.eq(field), "filter_id"])
    result["public_flagged_rows"] = raw.assign(flagged=raw.Flag.fillna("").astype(str).str.strip().ne("")).groupby("filter_id").flagged.sum() if "Flag" in raw else 0
    return result, raw, paths


def load_lab_hips(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    required = {"FilterId", "Site", "SampleDate", "FilterType", "Fabs"}
    if not required.issubset(raw):
        raise ValueError("Lab HIPS source is missing required columns")
    raw["SampleDate"] = pd.to_datetime(raw.SampleDate, errors="coerce")
    raw["is_blank"] = hips_blank_mask(raw)
    eligible = raw.loc[raw.FilterType.eq("PM2.5") & ~raw.is_blank].copy()
    if eligible.FilterId.isna().any() or eligible.Site.isna().any():
        raise ValueError("Lab HIPS rows without identity need investigation")
    eligible["filter_id"] = normalize_fid(eligible.FilterId)
    eligible["site"] = eligible.Site.astype(str).str.strip()
    eligible["date"] = eligible.SampleDate
    result = _catalog(eligible)
    result["lab_hips_catalog_filter"] = True
    result["lab_hips_fabs"] = result.index.isin(eligible.loc[finite(eligible.Fabs), "filter_id"])
    return result, {"raw_rows": len(raw), "eligible_pm25_rows": len(eligible),
                    "blank_or_undated_rows": int(raw.is_blank.sum()),
                    "repeated_physical_keys": int(eligible.filter_id.duplicated().sum())}


def load_native(folder: Path) -> tuple[pd.DataFrame, list[Path]]:
    path = folder / "spartan_native_filter_coverage.csv"
    exports = pd.read_csv(folder / "exports.csv", keep_default_na=False)
    status = exports.loc[exports.Dataset.eq("spartan_native_filter_coverage"), "Status"]
    if len(status) != 1 or status.iloc[0] != "ok":
        raise ValueError("Native coverage was not successfully exported; absence is unknown")
    manifest = pd.read_csv(folder / "manifest.csv", keep_default_na=False)
    entry = manifest.loc[manifest.File.eq(path.name)]
    if len(entry) != 1 or hashlib.sha256(path.read_bytes()).hexdigest().upper() != entry.iloc[0].SHA256.upper():
        raise ValueError("Native coverage hash differs from its manifest")
    raw = pd.read_csv(path)
    if len(raw) != int(entry.iloc[0].CsvRows):
        raise ValueError("Native coverage row count differs from its manifest")
    if raw.MediaId.duplicated().any():
        raise ValueError("Native coverage repeats a MediaId; counts would multiply")
    eligible = raw.loc[raw.ExternalFilterType.eq("PM2.5")].copy()
    if eligible.ExternalFilterId.isna().any() or eligible.SiteCode.isna().any():
        raise ValueError("Native PM2.5 media without physical/site identity need investigation")
    eligible["filter_id"] = normalize_fid(eligible.ExternalFilterId)
    eligible["site"] = eligible.SiteCode.astype(str).str.strip()
    eligible["date"] = pd.to_datetime(eligible.SamplingStartDate, errors="coerce")
    result = _catalog(eligible)
    result["native_filter"] = True
    result["native_media_rows"] = eligible.groupby("filter_id").MediaId.size()
    for source, dest in NATIVE_FIELDS.items():
        eligible[source] = pd.to_numeric(eligible[source], errors="raise")
        if eligible[source].isna().any() or eligible[source].lt(0).any():
            raise ValueError("Native counts must be available and nonnegative")
        result[dest] = eligible.groupby("filter_id")[source].sum()
    return result, [path, folder / "exports.csv", folder / "manifest.csv"]


def merge_catalog(result: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    overlap = result.index.intersection(other.index)
    if result.loc[overlap, "site"].ne(other.loc[overlap, "site"]).any():
        raise ValueError("Source catalogs disagree about a filter's site")
    merged = result.join(other, how="outer", rsuffix="_other")
    conflicts = merged.date.notna() & merged.date_other.notna() & merged.date.dt.normalize().ne(merged.date_other.dt.normalize())
    for name in ["site", "date", "date_conflict"]:
        if name == "date_conflict":
            merged[name] = merged[name].astype("boolean").fillna(False) | merged[name + "_other"].astype("boolean").fillna(False) | conflicts
        else:
            merged[name] = merged[name].combine_first(merged[name + "_other"])
        merged = merged.drop(columns=name + "_other")
    return merged


def _iso(value) -> str:
    return "" if pd.isna(value) else pd.Timestamp(value).isoformat()


def verify_reference_folder(folder: Path) -> dict:
    """Verify copied outputs before using them; never execute archive contents."""
    manifest = pd.read_csv(folder / "manifest.csv", keep_default_na=False)
    if manifest.File.duplicated().any():
        raise ValueError("Reference manifest has duplicate filenames")
    for entry in manifest.itertuples():
        if Path(entry.File).name != entry.File or any(char in entry.File for char in '/\\:') or entry.File in ('.', '..'):
            raise ValueError("Reference manifest has an unsafe filename")
        path = folder / entry.File
        if hashlib.sha256(path.read_bytes()).hexdigest().upper() != entry.SHA256.upper():
            raise ValueError(f"Reference manifest hash mismatch: {entry.File}")
        if entry.CsvRows != '' and len(pd.read_csv(path)) != int(entry.CsvRows):
            raise ValueError(f"Reference CSV count mismatch: {entry.File}")
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8-sig"))
    return {"status": receipt["Status"], "successful_exports": receipt["SuccessfulExports"],
            "errors": receipt["Errors"], "verified_files": len(manifest)}


def crosscheck_fg_request(request_path: Path, folder: Path, public: pd.DataFrame,
                          known_fg_ids: set[str] | None) -> pd.DataFrame:
    request = pd.read_csv(request_path, dtype=str, keep_default_na=False)
    if not {"filter_id", "shipment", "site", "media_id"}.issubset(request):
        raise ValueError("FG request lacks filter/shipment/site/media identity")
    request["filter_id"] = normalize_fid(request.filter_id)
    if request.filter_id.duplicated().any():
        raise ValueError("FG request repeats physical filter IDs")
    exports = pd.read_csv(folder / "exports.csv", keep_default_na=False)
    if not exports.loc[exports.Dataset.eq("spartan_filter_batches"), "Status"].eq("ok").any():
        raise ValueError("The reference batch map was not successfully exported")
    metadata = pd.read_csv(folder / "spartan_filter_batches.csv", dtype=str, keep_default_na=False)
    metadata = metadata.loc[metadata.ExternalFilterType.eq("PM2.5")].copy()
    metadata["filter_id"] = normalize_fid(metadata.ExternalFilterId)
    groups = {key: data for key, data in metadata.groupby("filter_id")}
    public = public.set_index("filter_id")
    records = []
    for row in request.to_dict("records"):
        hits = groups.get(row["filter_id"], metadata.iloc[:0])
        values = dict(row, registered_in_reference=not hits.empty, reference_media_matches=len(hits),
                      public_om_reported=bool(row["filter_id"] in public.index and public.loc[row["filter_id"], "om_ftir"]),
                      known_lab_fg_present=pd.NA if known_fg_ids is None else row["filter_id"] in known_fg_ids,
                      requested_media_id_matches=pd.NA if not row["media_id"] else bool(hits.MediaId.eq(row["media_id"]).any()),
                      requested_shipment_matches=pd.NA if hits.empty else bool(hits.BatchLabel.eq(row["shipment"]).any()))
        records.append(values)
    return pd.DataFrame(records)


def known_functional_groups() -> tuple[Path, set[str]] | None:
    # Reuse the maintained batch-2/3 delivery parser; do not select batch 4 by
    # filename or infer official groups from OM or from OC/EC calibrations.
    sys.path.insert(0, str(ROOT))
    from gallery.data.export_spartan_reference import FG_FIELDS, load_functional_groups
    try:
        path, frame = load_functional_groups()
    except (FileNotFoundError, RuntimeError):
        return None
    present = pd.DataFrame({field: finite(frame[field]) for field in FG_FIELDS}).any(axis=1)
    return path, set(frame.index[present])


def audit(public_dir: Path, out: Path, *, hips_file: Path | None = None,
          native_folder: Path | None = None, as_of: str | None = None,
          reference_folder: Path | None = None, fg_request: Path | None = None) -> dict:
    result, raw, paths = load_public(public_dir)
    public_sites = set(result.site)
    sources = list(paths)
    lab_info = None
    reference_info = verify_reference_folder(reference_folder) if reference_folder is not None else None
    if native_folder is None and reference_folder is not None and (reference_folder / "spartan_native_filter_coverage.csv").exists():
        native_folder = reference_folder
    if hips_file is not None:
        lab, lab_info = load_lab_hips(hips_file)
        result = merge_catalog(result, lab)
        sources.append(hips_file)
    if native_folder is not None:
        native, native_paths = load_native(native_folder)
        result = merge_catalog(result, native)
        sources.extend(native_paths)
    bool_fields = [*PUBLIC_FIELDS, "public_filter", "lab_hips_catalog_filter", "lab_hips_fabs", "native_filter"]
    for name in bool_fields:
        result[name] = result.get(name, pd.Series(False, index=result.index)).astype("boolean").fillna(False).astype(bool)
    for name in ["native_media_rows", *NATIVE_FIELDS.values()]:
        result[name] = result.get(name, pd.Series(index=result.index, dtype="Int64"))
        if native_folder is not None:
            result[name] = result[name].fillna(0).astype("Int64")
    if hips_file is None:
        result["lab_hips_fabs"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        result["lab_hips_catalog_filter"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    if native_folder is None:
        result["native_filter"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["public_site_file"] = result.site.isin(public_sites)
    result["direct_hips_observed"] = result.public_hips_bc | result.lab_hips_fabs.fillna(False)
    result["ssr_in_legacy_hips_group"] = (result.ssr_equivalent_bc | result.hips_ssr_estimate) & ~result.direct_hips_observed
    result["missing_public_ec"] = result.public_filter & ~result.ec_ftir
    result["missing_public_hips_bc"] = result.public_filter & ~result.public_hips_bc
    cutoff = pd.Timestamp(as_of or datetime.now(timezone.utc).date().isoformat())
    result["future_date"] = result.date.dt.normalize().gt(cutoff)
    result["year"] = result.date.dt.year.astype("Int64").astype("string").fillna("unknown")
    result = result.reset_index().sort_values(["site", "filter_id"])
    counts = ["public_filter", *PUBLIC_FIELDS, "lab_hips_catalog_filter", "lab_hips_fabs", "native_filter",
              "direct_hips_observed", "ssr_in_legacy_hips_group", "date_conflict", "future_date"]

    def rollup(keys: list[str]) -> pd.DataFrame:
        group = result.groupby(keys, dropna=False, sort=True)
        data = group[counts].sum(min_count=1)
        data.insert(0, "filters_in_union", group.size())
        data["first_date"] = group.date.min().map(_iso)
        data["last_date"] = group.date.max().map(_iso)
        data["public_source_available"] = group.public_site_file.any()
        data["lab_source_available"] = hips_file is not None
        data["native_source_available"] = native_folder is not None
        return data.reset_index()

    sites = rollup(["site"])
    years = rollup(["site", "year"])
    method_keys = ["site", "Parameter_Name", "Method_Code", "Analysis_Description", "Units", "category"]
    methods = raw.groupby(method_keys, dropna=False).agg(
        source_rows=("filter_id", "size"), finite_rows=("finite", "sum"),
        physical_filters=("filter_id", "nunique"), first_date=("date", "min"), last_date=("date", "max"),
    ).reset_index()
    valid_ids = raw.loc[raw.finite].groupby(method_keys, dropna=False).filter_id.nunique().rename("filters_with_values")
    methods = methods.merge(valid_ids.reset_index(), on=method_keys, how="left")
    methods["filters_with_values"] = methods.filters_with_values.fillna(0).astype(int)
    methods["first_date"] = methods.first_date.map(_iso)
    methods["last_date"] = methods.last_date.map(_iso)
    first_dates = {field: _iso(raw.loc[raw.finite & raw.category.eq(field), "date"].min()) for field in PUBLIC_FIELDS}
    sites["all_dates_before_first_public_ec"] = sites.last_date.map(lambda v: bool(v and first_dates["ec_ftir"] and v < first_dates["ec_ftir"]))
    sites["suggested_ec_message"] = np.where(~sites.public_source_available, "No public ChemSpec file in these inputs; measurement status unknown",
        np.where(sites.ec_ftir.eq(0), "No FTIR EC values reported in the inspected public snapshot", "FTIR EC values available on some or all public filters"))
    sites["optical_note"] = np.where(sites.ssr_in_legacy_hips_group.gt(0), "SSR/curve estimates must not count as direct HIPS", "Keep direct HIPS and SSR/curve estimates separate")
    fg_check = None
    if fg_request is not None:
        if reference_folder is None:
            raise ValueError("--fg-request requires --reference-folder for the shipment crosscheck")
        known = known_functional_groups()
        fg_check = crosscheck_fg_request(fg_request, reference_folder, result, None if known is None else known[1])
        sources.extend([fg_request, reference_folder / "spartan_filter_batches.csv"])
        if known is not None:
            sources.append(known[0])
    if reference_folder is not None:
        sources.extend([reference_folder / "receipt.json", reference_folder / "manifest.csv"])
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "as_of": cutoff.date().isoformat(),
        "sites": len(sites), "filters_in_union": len(result), "public_filters": int(result.public_filter.sum()),
        "lab_source_available": hips_file is not None, "native_source_available": native_folder is not None,
        "first_observed_public_dates": first_dates, "lab_hips": lab_info,
        "reference_export": reference_info,
        "fg_request": None if fg_check is None else {
            "requested_filters": len(fg_check), "registered_in_reference": int(fg_check.registered_in_reference.sum()),
            "public_om_reported": int(fg_check.public_om_reported.sum()),
            "known_lab_fg_present": None if fg_check.known_lab_fg_present.isna().all() else int(fg_check.known_lab_fg_present.sum()),
            "note": "This crosscheck locates identities and compares the known batch-2/3 delivery; it does not obtain missing group concentrations or identify a corrected batch-4 delivery.",
        },
        "unclassified_rows": int(raw.category.eq("unclassified").sum()),
        "invalid_public_date_rows": int(raw.date.isna().sum()), "future_filter_dates": int(result.future_date.sum()),
        "note": "Missing means absent from the inspected source snapshot, not never measured. Earliest dates are observed records, not instrument introduction dates. Finite zero/negative/flagged values count as reported; no QC exclusion is applied. Unknown optional sources remain unknown. Raw native readings are not final products. Date conflicts and future dates are retained and flagged.",
        "sources": [{"path": display_path(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources],
    }
    out.mkdir(parents=True, exist_ok=False)
    result.to_csv(out / "filter_coverage.csv", index=False)
    sites.to_csv(out / "site_coverage.csv", index=False)
    years.to_csv(out / "site_year_coverage.csv", index=False)
    methods.to_csv(out / "public_method_coverage.csv", index=False)
    raw.loc[raw.category.eq("unclassified"), ["site", "filter_id", "Parameter_Name", "Method_Code", "Analysis_Description", "Units", "Value"]].to_csv(out / "unclassified_public_rows.csv", index=False)
    if fg_check is not None:
        fg_check.to_csv(out / "fg_request_crosscheck.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    columns = ["site", "filters_in_union", "public_filter", "ec_ftir", "oc_ftir", "public_hips_bc", "lab_hips_fabs", "hips_ssr_estimate", "ssr_equivalent_bc"]
    lines = ["# SPARTAN source coverage", "", summary["note"], "", "Counts are distinct physical filter IDs, using the union of supplied catalogs.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in sites[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join("unknown" if pd.isna(v) else str(v) for v in row) + " |")
    (out / "coverage_report.md").write_text("\n".join(lines) + "\n")
    entries = []
    for path in sorted(out.iterdir()):
        csv_rows = len(pd.read_csv(path)) if path.suffix == ".csv" else None
        entries.append({"File": path.name, "Bytes": path.stat().st_size, "CsvRows": csv_rows,
                        "SHA256": hashlib.sha256(path.read_bytes()).hexdigest().upper()})
    pd.DataFrame(entries).to_csv(out / "manifest.csv", index=False)
    return {key: value for key, value in summary.items() if key != "sources"}


def default_hips_file() -> Path | None:
    sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
    from data_paths import maia_data_root
    try:
        path = maia_data_root() / "Spartan" / "SPARTAN_HIPS_Batch1-51.v2.csv"
    except (FileNotFoundError, RuntimeError):
        return None
    return path if path.is_file() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, default=RAW_DIR / "FilterBased/ChemSpecPM25")
    parser.add_argument("--hips-file", type=Path, help="lab HIPS delivery; otherwise discover via configured data paths")
    parser.add_argument("--public-only", action="store_true", help="leave lab coverage unknown")
    parser.add_argument("--native-folder", type=Path, help="verified Davis reference/coverage export folder")
    parser.add_argument("--reference-folder", type=Path, help="existing Davis reference export; verify its full manifest")
    parser.add_argument("--fg-request", type=Path, help="crosscheck requested FG filter IDs and shipments against the reference export")
    parser.add_argument("--as-of", help="date used only to flag future records (YYYY-MM-DD)")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    hips = None if args.public_only else (args.hips_file or default_hips_file())
    print(json.dumps(audit(args.public_dir, args.out, hips_file=hips, native_folder=args.native_folder, as_of=args.as_of,
                           reference_folder=args.reference_folder, fg_request=args.fg_request), indent=2))
