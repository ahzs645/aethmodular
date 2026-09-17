#!/usr/bin/env python
"""Build evidence catalogs and active-period matches, without calibration fits."""

from dataclasses import asdict, fields
import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))

from active_interval_matching import (
    CoveragePolicy,
    StreamContract,
    eligibility,
    empty_intervals,
    summarize_active_segments,
    validate_intervals,
)
from config import SITES
from data_paths import aethalometry_dir, maia_data_root
from instrument_provenance import (
    apply_processing_evidence,
    processing_records,
    stage_instrument_csv,
)
from interval_evidence import (
    file_hash,
    recover_lab_date_evidence,
    schedule_catalog,
    trace_ec_provenance,
)


def fingerprint(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def verified_audit_input(audit_dir, name, manifest):
    path = (audit_dir / name).resolve()
    recorded = next(r for r in manifest["outputs"] if Path(r["path"]).name == name)
    if file_hash(path) != recorded["sha256"]:
        raise ValueError(f"Audit artifact changed: {path}; rebuild the audit first")
    return pd.read_parquet(path)


def build_matches(catalog, intervals, processing, policy):
    """Retain every physical filter × channel, with unresolved means left null."""
    rows = []
    segment_frames = []
    source_cache = {}
    interval_groups = {k: g for k, g in intervals.groupby(["site", "base_filter_id"])}
    for parent in catalog.to_dict("records"):
        periods = interval_groups.get((parent["site"], parent["base_filter_id"]), empty_intervals())
        for stream in processing.loc[processing.site.eq(parent["site"])].to_dict("records"):
            contract = StreamContract(**{f.name: stream[f.name] for f in fields(StreamContract)})
            summary = {
                k: np.nan
                for k in [
                    "expected_slots",
                    "input_slots",
                    "observed_slots",
                    "valid_observed_slots",
                    "input_availability_fraction",
                    "observed_coverage_fraction",
                    "valid_observed_coverage_fraction",
                    "minimum_segment_valid_observed_coverage",
                    "mean_input_native",
                    "mean_valid_observed_native",
                    "mean_input_ebc_ugm3",
                    "mean_valid_observed_ebc_ugm3",
                    "absorption_Mm1",
                ]
            }
            summary["observation_provenance_status"] = "unresolved"
            status = (
                "active_schedule_unverified" if periods.empty else "timestamped_input_unavailable"
            )
            if len(periods) and stream.get("staged_file"):
                path = stream["staged_file"]
                if path not in source_cache:
                    if file_hash(path) != stream["staged_file_hash"]:
                        raise ValueError("Staged observations changed since provenance inventory")
                    frame = pd.read_parquet(path).set_index("timestamp_utc")
                    if not {"source_row", "source_file_hash"} <= set(frame):
                        raise ValueError("Observation table requires source row identities")
                    if (
                        frame.source_row.isna().any()
                        or (frame.source_row < 0).any()
                        or (frame.source_row % 1 != 0).any()
                    ):
                        raise ValueError("Observation source rows must be nonnegative integers")
                    source_cache[path] = frame
                frame = source_cache[path]
                if not frame.source_file_hash.eq(stream["source_file_hash"]).all():
                    raise ValueError("Observation rows do not match the declared source stream")
                if frame.index.hasnans or frame.index.has_duplicates:
                    status = "timestamp_identity_requires_source_resolution"
                else:
                    segments, summary = summarize_active_segments(
                        frame,
                        periods,
                        stream["staged_channel"],
                        contract,
                        observed_column=stream["observed_column"],
                        valid_column=stream["valid_column"],
                    )
                    segments["site"] = parent["site"]
                    segments["base_filter_id"] = parent["base_filter_id"]
                    segments["channel"] = stream["channel"]
                    segments["stream_id"] = stream["stream_id"]
                    segment_frames.append(segments)
                    status = "active_intervals_summarized"
            ftir_resolved = (
                not parent["ftir_ec_ugm3_conflict"] and parent["ftir_ec_ugm3_units_valid"]
            )
            decisions = eligibility(
                parent,
                summary,
                contract,
                policy,
                schedule_verified=bool(len(periods)),
                ec_parameter_resolved=bool(ftir_resolved),
            )
            rows.append(
                {
                    **parent,
                    **summary,
                    **decisions,
                    "channel": stream["channel"],
                    "stream_id": stream["stream_id"],
                    "instrument_source_file_hash": stream["source_file_hash"],
                    "instrument_staged_file_hash": stream["staged_file_hash"],
                    "original_channel": stream["original_channel"],
                    "native_units": contract.units if contract.units_verified else "unverified",
                    "configured_wavelength_nm": contract.wavelength_nm,
                    "processing_history_verified": contract.processing_history_verified,
                    "optical_conversion_verified": contract.optical_conversion_verified,
                    "wavelength_treatment_verified": contract.wavelength_treatment_verified,
                    "matching_status": status,
                    "active_interval_ids": json.dumps(periods.interval_id.tolist()),
                    "coverage_basis": "declared_acquisition_slots_within_verified_active_periods",
                    "availability_grid_verified": contract.cadence_verified
                    and contract.timestamp_role_verified,
                    "eligible_chemspec_ec_ratio_analysis": False,
                    "chemspec_ec_ratio_ineligibility_reason": "authoritative_chemspec_ec_parameter_unresolved",
                }
            )
    segments = (
        pd.concat(segment_frames, ignore_index=True)
        if segment_frames
        else pd.DataFrame(
            {
                "site": pd.Series(dtype="string"),
                "base_filter_id": pd.Series(dtype="string"),
                "channel": pd.Series(dtype="string"),
                "stream_id": pd.Series(dtype="string"),
                "interval_id": pd.Series(dtype="string"),
                "expected_slots": pd.Series(dtype="Int64"),
                "input_slots": pd.Series(dtype="Int64"),
                "observed_slots": pd.Series(dtype="Float64"),
                "valid_observed_slots": pd.Series(dtype="Float64"),
                "input_availability_fraction": pd.Series(dtype="float64"),
                "observed_coverage_fraction": pd.Series(dtype="float64"),
                "valid_observed_coverage_fraction": pd.Series(dtype="float64"),
                "raw_source_links": pd.Series(dtype="string"),
                "active_start_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "active_end_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            }
        )
    )
    return pd.DataFrame(rows), segments


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = args.audit.resolve()
    manifest = json.loads((audit / "manifest.json").read_text())
    # Reusing the audit's flags is valid only for the same identity/unit rules
    # and exclusion registry. Do not advertise current rules over stale flags.
    for name in ["config.py", "data_matching.py", "outliers.py"]:
        recorded = next(r for r in manifest["code"] if Path(r["path"]).name == name)
        if file_hash(AREA / "scripts" / name) != recorded["sha256"]:
            raise ValueError(f"Audit rules changed ({name}); regenerate the audit first")
    names = [
        "matched_sample_candidates.parquet",
        "filter_measurements.parquet",
        "portal_measurement_rows.parquet",
    ]
    catalog, measurements, portal = [verified_audit_input(audit, n, manifest) for n in names]
    unified = next(
        r for r in manifest["inputs"] if Path(r["path"]).name == "unified_filter_dataset.pkl"
    )
    if file_hash(unified["path"]) != unified["sha256"]:
        raise ValueError("Unified source changed since audit; regenerate the audit before matching")
    inputs = [
        fingerprint(audit / "manifest.json"),
        *[fingerprint(audit / n) for n in names],
        unified,
    ]
    intervals = (
        empty_intervals()
        if args.sampling_intervals is None
        else pd.read_parquet(args.sampling_intervals)
    )
    intervals = validate_intervals(intervals, catalog)
    if args.sampling_intervals:
        inputs.append(fingerprint(args.sampling_intervals))
    for source in intervals.source_file.unique():
        inputs.append(fingerprint(source))
    lab_paths = [
        maia_data_root() / "DAVIS" / "SPARTAN FTIR pulls" / site / f"{site}_filters.csv"
        for site in ["CHTS", "INDH", "USPA"]
    ]
    recovered = []
    unavailable = []
    for path in lab_paths:
        if path.exists():
            recovered.append(path)
        else:
            unavailable.append({"path": str(path), "status": "not_found_at_inspected_location"})
    lab = recover_lab_date_evidence(recovered)
    inputs.extend(fingerprint(p) for p in recovered)
    parent = schedule_catalog(catalog, lab, intervals)
    ec, ec_rows = trace_ec_provenance(parent, measurements, unified["sha256"], portal)
    ec["source_file"] = unified["path"]
    ec_rows["source_file"] = unified["path"]
    staged = []
    for site, filename in [
        ("Addis_Ababa", "Jacros_MA350_1-min_2022-2024_Cleaned.csv"),
        ("JPL", "Pasadena_MA350_1-min_2023-2024_Cleaned.csv"),
    ]:
        path = aethalometry_dir() / "Raw" / filename
        if path.exists():
            print(f"Staging/verifying timestamped source: {site}", flush=True)
            source = stage_instrument_csv(path, out / "observations", site)
            staged.append(source)
            inputs.append(
                {
                    "path": source["source_file"],
                    "sha256": source["source_file_hash"],
                    "bytes": source["source_bytes"],
                }
            )
        else:
            unavailable.append({"path": str(path), "status": "not_found_at_inspected_location"})
    processing = processing_records(staged, SITES)
    if args.processing_evidence:
        processing = apply_processing_evidence(processing, args.processing_evidence)
        inputs.append(fingerprint(args.processing_evidence))
        for path in processing.provenance_evidence_file.dropna().unique():
            inputs.append(fingerprint(path))
    policy = CoveragePolicy(
        args.minimum_coverage,
        args.minimum_segment_coverage,
        f"predeclared_total_{args.minimum_coverage:g}_each_segment_{args.minimum_segment_coverage:g}",
    )
    matches, segments = build_matches(parent, intervals, processing, policy)
    quarter_rows = []
    for segment in segments.to_dict("records"):
        for quarter in json.loads(segment["quarter_coverage_json"]):
            quarter_rows.append(
                {
                    **{
                        key: segment[key]
                        for key in ["site", "base_filter_id", "channel", "stream_id", "interval_id"]
                    },
                    **quarter,
                }
            )
    quarter_coverage = (
        pd.DataFrame(quarter_rows)
        if quarter_rows
        else pd.DataFrame(
            {
                key: pd.Series(dtype=dtype)
                for key, dtype in {
                    "site": "string",
                    "base_filter_id": "string",
                    "channel": "string",
                    "stream_id": "string",
                    "interval_id": "string",
                    "quarter": "Int64",
                    "expected_slots": "Int64",
                    "valid_observed_coverage_fraction": "float64",
                    "diagnostic_only": "boolean",
                }.items()
            }
        )
    )
    filter_flags = matches.drop_duplicates(["site", "base_filter_id"])
    summary = []
    for site in SITES:
        c = parent.loc[parent.site.eq(site)]
        f = filter_flags.loc[filter_flags.site.eq(site)]
        site_lab = lab.loc[lab.site.eq(site)]
        summary.append(
            dict(
                site=site,
                physical_filters=len(c),
                eligible_filter_diagnostic=int(f.eligible_filter_diagnostic.sum()),
                eligible_ftir_ec_ratio_analysis=int(f.eligible_ec_ratio_analysis.sum()),
                verified_schedules=int(c.sampling_schedule_verified.sum()),
                lab_source_rows=len(site_lab),
                lab_ids_in_parent=int(site_lab.base_filter_id.isin(c.base_filter_id).sum()),
                eligible_interval_ebc_filters_any_channel=int(
                    matches.loc[matches.site.eq(site)]
                    .groupby("base_filter_id")
                    .eligible_interval_ebc_comparison.any()
                    .sum()
                ),
                eligible_absorption_filters_any_channel=int(
                    matches.loc[matches.site.eq(site)]
                    .groupby("base_filter_id")
                    .eligible_absorption_comparison.any()
                    .sum()
                ),
                unresolved_chemspec_ec_groups=int((ec.site.eq(site) & ~ec.resolved).sum()),
            )
        )
    summary = pd.DataFrame(summary)
    tables = {
        "sampling_intervals": intervals,
        "filter_schedule_catalog": parent,
        "lab_date_evidence": lab,
        "ec_provenance_resolution": ec,
        "ec_provenance_source_rows": ec_rows,
        "instrument_processing_provenance": processing,
        "matched_active_intervals": matches,
        "active_segment_coverage": segments,
        "active_quarter_coverage": quarter_coverage,
        "selection_summary": summary,
    }
    for name, table in tables.items():
        table.to_parquet(out / f"{name}.parquet", index=False)
    inventory = {
        "staged_sources": staged,
        "unavailable_inputs": unavailable,
        "search_scope": [
            "scoped DAVIS lab exports",
            "local SPARTAN portal downloads",
            "Aethalometry Data/Raw timestamped CSVs",
        ],
        "schedule_result": "No sampler on/off logs recovered from inspected sources"
        if intervals.empty
        else "Supplied interval evidence validated",
        "portal_access_note": "CHTS and INDH direct portal downloads timed out during this investigation; this is not evidence of absence",
        "unstaged_sites": ["Beijing", "Delhi"],
        "unstaged_reason": "Processed manual-BCc pickles located; original observation and processing history not established",
    }
    (out / "source_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    report = f"""# Active-interval evidence build

This build preserves {len(parent):,} physical filter identities and {len(matches):,} filter/channel decisions.
It contains **{len(intervals):,} verified active intervals**. Unresolved collection envelopes never become assumed intervals.
No calibration, absorption regression, uncertainty weighting or performance-tuned selection was performed.

{summary.to_markdown(index=False)}

## Sampling evidence

{len(lab):,} lab export rows retain their original date strings, source paths, SHA-256 hashes and zero-based data-row numbers.
Midnight date fields are not interpreted as verified collection clocks or assigned a timezone.
Schedule status counts: {json.dumps(parent.schedule_status.value_counts().to_dict(), sort_keys=True)}.
The original 317 positive HIPS collection envelopes, 347 date candidates and 545 filter-diagnostic pairs remain distinct cohorts.
The 430 hour-consistent envelopes remain candidates; that agreement does not establish pump operation.

## EC provenance

All {len(ec_rows):,} ChemSpec EC source rows are retained in {len(ec):,} physical-filter groups.
There are {int(ec.resolution_category.eq("unresolved_discrepancy").sum()):,} conflicting groups.
In {int(ec.one_value_matches_reported_mdl_rounding.sum()):,} groups, a value matches the reported MDL rounded to two decimals.
This supports a possible MDL-to-concentration flattening error, but does not establish the original parameter roles.
No authoritative ChemSpec EC value was selected. Original parameter codes, method definitions, reference conditions and export versions remain unresolved.
FTIR EC remains explicitly a prediction, and ChemSpec EC is not treated as an independent reference.

## Instrument and coverage provenance

{sum(s["source_rows"] for s in staged):,} timestamped CSV rows were staged, with source row links and original BC1/BC2/BCc fields.
Neither datum IDs nor the word “Cleaned” proves unmodified observations. Observation flags remain unknown.
Instrument processing history, export-specific unit evidence, acquisition timestamp role/grid origin and optical coefficients require evidence.
Configured wavelengths are retained from the canonical MA350 configuration; wavelength treatment is unverified.
No HIPS MAC constant is used to reconstruct aethalometer absorption.

The configurable screening policy is **{policy.minimum_valid_observed_fraction:.0%} total valid observed slots and
{policy.minimum_segment_valid_observed_fraction:.0%} in every active segment**, declared before inspecting matched performance.
This is an analyst screening choice, not an established instrument standard. Expected nominal instants are in [start, end), on the declared UTC cadence grid.
Exact timestamps are clipped before slot aggregation; repeated sub-cadence rows cannot increase slot counts or slot weight.
The pooled mean weights observed slots, not segment means. Duplicate acquisition timestamps require explicit resolution.
Unknown observed coverage is null; nonmissing input availability is separate and its grid verification is explicit.

## Eligibility and uncertainties

Filter diagnostics, interval eBC, absorption, FTIR-denominator ratios, ChemSpec-denominator ratios and independent EC validation have separate flags.
FTIR ratio denominators must be resolved, finite, positive and at or above their own reported MDL, with no MDL conflict.
This low-denominator rule is explicit; it does not establish an uncertainty distribution for the MDL.
HIPS uncertainty/MDL source rows are retained in the parent catalog. Their semantics remain unresolved, and AIRSpec RMSE is not assigned to all FTIR products.

## Remaining evidence

Source-backed sampler on/off records (or a documented continuous-operation record tied to each filter),
original EC parameter-role definitions, and stream-specific observation/correction/conversion records are needed to admit primary comparisons.
The source inventory records what was inspected and distinguishes access gaps from missing scientific data.
Supply reviewed intervals with --sampling-intervals and reviewed stream records with --processing-evidence, then rebuild.
Hashes establish reproducibility and content identity; they do not replace review of the evidence's scientific meaning.
"""
    (out / "selection_report.md").write_text(report)
    outputs = [fingerprint(p) for p in sorted(out.glob("*.parquet"))]
    outputs.extend(fingerprint(out / n) for n in ["selection_report.md", "source_inventory.json"])
    for s in staged:
        outputs.extend(
            [
                fingerprint(s["staged_file"]),
                fingerprint(Path(s["staged_file"]).with_suffix(".json")),
            ]
        )
    code = [
        Path(__file__),
        *[
            AREA / "scripts" / n
            for n in [
                "active_interval_matching.py",
                "observation_coverage.py",
                "interval_evidence.py",
                "instrument_provenance.py",
                "config.py",
                "data_matching.py",
                "outliers.py",
            ]
        ],
    ]
    result = dict(
        status="EVIDENCE_CATALOG_WITH_EXPLICIT_UNRESOLVED_MATCHES"
        if intervals.empty
        else "EVIDENCE_GATED_INTERVAL_BUILD",
        inputs=inputs,
        outputs=outputs,
        code=[fingerprint(p) for p in code],
        coverage_policy=asdict(policy),
        python=platform.python_version(),
        pandas=pd.__version__,
        numpy=np.__version__,
        source_row_convention="zero-based data row; headers excluded",
    )
    (out / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote {out}/selection_report.md")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AREA / "output/tables/matched_sample_audit")
    parser.add_argument(
        "--output", type=Path, default=AREA / "output/tables/active_interval_matches"
    )
    parser.add_argument("--sampling-intervals", type=Path)
    parser.add_argument("--processing-evidence", type=Path)
    parser.add_argument("--minimum-coverage", type=float, default=0.75)
    parser.add_argument("--minimum-segment-coverage", type=float, default=0.50)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
