"""Sampling and EC evidence catalogs without implicit scientific adjudication."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import SITES
    from data_matching import add_base_filter_id
except ImportError:
    from .config import SITES
    from .data_matching import add_base_filter_id


def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recover_lab_date_evidence(paths):
    """Keep lab dates as dates. Midnight date fields do not establish UTC clocks."""
    frames = []
    for path in paths:
        path = Path(path)
        source = pd.read_csv(path)
        required = {"ExternalFilterId", "SamplingStartDate", "SamplingEndDate"}
        if not required <= set(source):
            raise ValueError(f"Unrecognized lab filter metadata: {path}")
        source.insert(0, "source_row", np.arange(len(source)))
        source = add_base_filter_id(source.rename(columns={"ExternalFilterId": "FilterId"}))
        source["source_file"] = str(path.resolve())
        source["source_file_hash"] = file_hash(path)
        source["site_code"] = source.base_filter_id.str.split("-").str[0]
        source["site"] = source.site_code.map({v["code"]: k for k, v in SITES.items()})
        source["schedule_evidence_type"] = "lab_export_date_fields"
        source["clock_time_verified"] = False
        source["collection_mode_verified"] = False
        source["time_interpretation"] = "date_fields_only; midnight_and_timezone_not_adjudicated"
        frames.append(source)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["site", "base_filter_id", "source_file", "source_file_hash", "source_row"]
        )
    )


def schedule_catalog(catalog, lab_evidence, verified_intervals):
    out = catalog.copy()
    lab_groups = {key: g for key, g in lab_evidence.groupby(["site", "base_filter_id"])}
    interval_groups = {key: g for key, g in verified_intervals.groupby(["site", "base_filter_id"])}
    statuses = []
    reasons = []
    links = []
    counts = []
    for row in out.itertuples():
        key = (row.site, row.base_filter_id)
        current = interval_groups.get(key)
        lab = lab_groups.get(key)
        status = str(getattr(row, "portal_timing_status", ""))
        if current is not None:
            statuses.append(current.collection_mode.iloc[0])
            counts.append(len(current))
            reasons.append(
                "Documented active intervals supplied with source hashes and local clocks"
            )
        else:
            counts.append(0)
            if "invalid" in status or "conflicting" in status:
                statuses.append("timing_contradictory")
                reasons.append("Reported envelope or sampled hours require source reconciliation")
            elif "consistent_with_continuous_window" in status:
                statuses.append("bounds_consistent_mode_unverified")
                reasons.append("Sampled-hour consistency does not document continuous operation")
            elif bool(getattr(row, "portal_window_available", False)):
                statuses.append("active_periods_unknown")
                reasons.append(
                    "A collection envelope is available but individual active periods are unknown"
                )
            elif lab is not None:
                statuses.append("date_bounds_only_mode_unverified")
                reasons.append(
                    "Lab sampling dates recovered; clock time, timezone interpretation and mode remain unverified"
                )
            else:
                statuses.append("missing_schedule_evidence")
                reasons.append(
                    "No active schedule or dated lab bounds recovered from inspected inputs"
                )
        link = (
            []
            if lab is None
            else lab[
                [
                    "source_file",
                    "source_file_hash",
                    "source_row",
                    "SamplingStartDate",
                    "SamplingEndDate",
                ]
            ].to_dict("records")
        )
        links.append(json.dumps(link, sort_keys=True, default=str))
    out["schedule_status"] = statuses
    out["schedule_resolution_reason"] = reasons
    out["verified_active_interval_count"] = counts
    out["sampling_schedule_verified"] = out.verified_active_interval_count.gt(0)
    out["lab_date_evidence_links"] = links
    out["hips_uncertainty_semantics"] = "unresolved; do not assume standard deviation"
    out["hips_mdl_semantics"] = "retained as reported; interpretation unverified"
    out["ftir_uncertainty_semantics"] = (
        "product-specific uncertainty not established by AIRSpec RMSE"
    )
    return out


def trace_ec_provenance(catalog, measurements, source_hash, available_portal_rows=None):
    """Retain every EC row, distinguish numerical clues from source adjudication.

    A value matching a rounded MDL is a traceable clue, not enough evidence to
    rewrite its original parameter definition or choose an authoritative EC.
    """
    ec = measurements.loc[measurements.Parameter.eq("ChemSpec_EC_PM2.5")].copy()
    if ec.empty:
        return pd.DataFrame(), ec
    lookup = catalog.set_index(["site_code", "base_filter_id"])
    ec["source_file_hash"] = source_hash
    ec["matches_rounded_reported_mdl"] = np.isclose(
        ec.Concentration, ec.MDL.round(2), atol=1e-10, rtol=0
    )
    ec["rounded_mdl_candidate_role"] = np.where(
        ec.matches_rounded_reported_mdl, "possible_mdl_in_concentration_field", "competing_value"
    )
    ec["adjudication_status"] = "original_parameter_code_and_definition_unrecovered"
    results = []
    for (site_code, filter_id), g in ec.groupby(["Site", "base_filter_id"], sort=True):
        parent = lookup.loc[(site_code, filter_id)]
        finite = g.loc[np.isfinite(g.Concentration)]
        vals = sorted(finite.Concentration.unique().tolist())
        mdl_rows = g.loc[g.matches_rounded_reported_mdl]
        mapped = (
            pd.DataFrame()
            if available_portal_rows is None
            else available_portal_rows.loc[
                available_portal_rows.Site_Code.eq(site_code)
                & available_portal_rows.base_filter_id.eq(filter_id)
                & available_portal_rows.Parameter_Name.str.strip().eq("EC PM2.5")
            ]
        )
        ftir = parent.ftir_ec_ugm3
        ftir_match = bool(
            np.isfinite(ftir)
            and np.isclose(g.Concentration, round(ftir, 2), atol=1e-10, rtol=0).any()
        )
        if len(vals) > 1:
            category = "unresolved_discrepancy"
            reason = (
                "Original parameter codes/roles are unavailable; competing values remain separate"
            )
        elif len(g) > 1:
            category = "same_numeric_value_repeated_definition_unverified"
            reason = "Numerical equality alone does not establish identical measurement provenance"
        else:
            category = "single_record_definition_unverified"
            reason = (
                "One value is present, but its original EC parameter definition is not recovered"
            )
        results.append(
            dict(
                site=parent.site,
                site_code=site_code,
                base_filter_id=filter_id,
                original_parameter="ChemSpec_EC_PM2.5",
                resolution_category=category,
                resolved=False,
                selected_ec_ugm3=np.nan,
                selected_source_row=pd.NA,
                original_values=json.dumps(vals),
                source_rows=json.dumps([int(v) for v in g.source_row]),
                source_file_hash=source_hash,
                source_method_labels=json.dumps(sorted(g.CalibrationSetId.dropna().unique())),
                original_units=json.dumps(sorted(g.Concentration_Units.dropna().unique())),
                original_parameter_codes=json.dumps(sorted(mapped.Parameter_Code.unique().tolist()))
                if len(mapped)
                else "[]",
                authoritative_method_definition=None,
                reference_conditions=None,
                source_export_version=None,
                numerical_basis="original Concentration field; interpretation unresolved",
                reported_mdl_values=json.dumps(sorted(g.MDL.dropna().unique().tolist())),
                candidate_mdl_source_rows=json.dumps([int(v) for v in mdl_rows.source_row]),
                one_value_matches_reported_mdl_rounding=bool(len(mdl_rows)),
                a_value_matches_rounded_ftir_prediction=ftir_match,
                candidate_interpretation="possible_MDL_flattening_into_concentration"
                if len(mdl_rows)
                else None,
                resolution_reason=reason,
                independent_ec_reference=False,
                ftir_prediction_retained_ugm3=ftir,
                ftir_prediction_source_rows=parent.ftir_ec_ugm3_source_rows,
                volume_m3=parent.volume_m3,
                volume_conflict=bool(parent.volume_conflict),
            )
        )
    return pd.DataFrame(results), ec
