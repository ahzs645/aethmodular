"""Executable readiness ledgers for independent follow-up measurements.

Plans are requests for evidence, never fabricated laboratory observations.
Optical absorption, refractory BC and thermal-optical EC retain distinct names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIREMENTS = {
    "offline_ams": {
        "quantity": "organic_composition",
        "sample_requirement": "compatible archived material with verified storage and remaining mass",
        "evidence": ("substrate_compatibility", "remaining_material", "storage_history",
                     "extraction_blank", "recovery_standard", "independent_method", "sample_identity"),
    },
    "thermal_ec": {
        "quantity": "thermal_optical_ec",
        "sample_requirement": "suitable quartz or laboratory-approved substrate; matched co-sampling if needed",
        "evidence": ("substrate_compatibility", "laboratory_protocol", "blank_correction", "detection_limit",
                     "measurement_uncertainty", "sample_identity", "sampling_equivalence"),
    },
    "photoacoustic": {
        "quantity": "absorption_Mm1",
        "sample_requirement": "co-located airborne sampling with independent absorption calibration",
        "evidence": ("calibration_traceability", "measurement_uncertainty", "inlet_losses",
                     "size_fraction", "wavelength_alignment", "clock_alignment", "active_intervals"),
    },
    "sp2": {
        "quantity": "refractory_bc_ugm3",
        "sample_requirement": "suitable airborne sampling and instrument feasibility review",
        "evidence": ("facility_feasibility", "mass_calibration", "size_detection_range",
                     "inlet_losses", "size_fraction", "clock_alignment", "active_intervals"),
    },
}


def measurement_plan(candidates: pd.DataFrame, *, physical_id_col="physical_id",
                     methods=("offline_ams", "thermal_ec", "photoacoustic", "sp2"),
                     evidence: pd.DataFrame | None = None) -> dict:
    """Prepare per-candidate requirements and optionally assess supplied evidence.

    Evidence columns: physical_id, method, requirement, verified, evidence.
    Only a literal boolean true (or CSV 'true') plus a citation completes a check.
    This assesses protocol readiness, not assay validity or measured accuracy.
    """
    if physical_id_col not in candidates:
        raise ValueError(f"Missing candidate identity: {physical_id_col}")
    identifiers = candidates[physical_id_col].astype(str).str.strip()
    if candidates.empty or candidates[physical_id_col].isna().any() or identifiers.eq("").any() or identifiers.duplicated().any():
        raise ValueError("Candidates require unique physical-filter identities")
    if not methods or any(m not in REQUIREMENTS for m in methods):
        raise ValueError("Unknown or empty measurement methods")
    if len(set(methods)) != len(methods):
        raise ValueError("Measurement methods must be unique")
    supplied = {}
    if evidence is not None:
        columns = {"physical_id", "method", "requirement", "verified", "evidence"}
        if not columns.issubset(evidence):
            raise ValueError(f"Evidence needs columns {sorted(columns)}")
        if evidence.duplicated(["physical_id", "method", "requirement"]).any():
            raise ValueError("Duplicate evidence checks are ambiguous")
        for row in evidence.itertuples(index=False):
            supplied[(str(row.physical_id), row.method, row.requirement)] = row
    rows, plans = [], []
    for _, candidate in candidates.iterrows():
        identifier = str(candidate[physical_id_col]).strip()
        for method in methods:
            spec = REQUIREMENTS[method]
            missing = []
            for requirement in spec["evidence"]:
                row = supplied.get((identifier, method, requirement))
                verified = row is not None and _true(row.verified) and _text(row.evidence)
                if not verified:
                    missing.append(requirement)
                rows.append({"physical_id": identifier, "method": method,
                             "requirement": requirement, "verified": bool(verified),
                             "evidence": _text(row.evidence) if row is not None else ""})
            plans.append({"physical_id": identifier, "method": method,
                          "panel_role": candidate.get("panel_role", candidate.get("panel", "unspecified")),
                          "quantity": spec["quantity"], "sample_requirement": spec["sample_requirement"],
                          "status": "protocol_ready" if not missing else "needs_evidence",
                          "missing_evidence": "|".join(missing),
                          "independent_addis_ec_observed": False})
    return {"summary": {"status": "completed", "n_candidates": len(candidates),
                        "n_protocols": len(plans), "n_ready": sum(p["status"] == "protocol_ready" for p in plans),
                        "limitations": ["A completed plan contains no new measurements.",
                                        "Offline AMS is organic-composition evidence, not independent EC.",
                                        "SP2 refractory BC and thermal-optical EC are distinct quantities.",
                                        "Archived PTFE filters are not automatically suitable for thermal EC or SP2."]},
            "tables": {"measurement_plan": pd.DataFrame(plans), "evidence_requirements": pd.DataFrame(rows)}}


def _true(value):
    return value is True or isinstance(value, np.bool_) and bool(value) or isinstance(value, str) and value.lower() == "true"


def _text(value):
    return str(value).strip() if pd.notna(value) else ""


PAIR_RULES = {
    "ftir_thermal_ec": ("ec_ftir", "thermal_ec", "ug/m3", "ug/m3", ("TOR", "TOT")),
    "ma350_photoacoustic": ("b_atn", "b_abs", "Mm^-1", "Mm^-1", ("photoacoustic",)),
    "absorption_sp2": ("b_abs", "rbc", "Mm^-1", "ug/m3", ("SP2",)),
}


def assess_colocation_pairs(pairs: pd.DataFrame, *, minimum_overlap=0.9) -> dict:
    """Flag candidate pairs using timing, quantity, calibration and inlet evidence.

    The overlap fraction must pass for both collection intervals. Exact wavelength
    agreement is required for the optical comparison; interpolation needs a
    separate reviewed input product and is not inferred by this function.
    """
    if not 0 < minimum_overlap <= 1:
        raise ValueError("minimum_overlap must be in (0, 1]")
    required = {"pair_id", "comparison", "left_quantity", "right_quantity", "left_unit", "right_unit",
                "right_method", "left_start_utc", "left_end_utc", "right_start_utc", "right_end_utc",
                "left_size_fraction", "right_size_fraction", "sampling_verified", "calibration_verified",
                "identity_evidence", "sampling_evidence", "calibration_evidence"}
    if not required.issubset(pairs):
        raise ValueError(f"Missing pair columns: {sorted(required - set(pairs))}")
    identifiers = pairs.pair_id.astype(str).str.strip()
    if pairs.pair_id.isna().any() or identifiers.eq("").any() or identifiers.duplicated().any():
        raise ValueError("Pair identities must be unique and nonmissing")
    rows = []
    for record in pairs.to_dict("records"):
        reasons = []
        rule = PAIR_RULES.get(record["comparison"])
        if rule is None:
            reasons.append("unsupported_comparison")
        else:
            for column, expected in zip(("left_quantity", "right_quantity", "left_unit", "right_unit"), rule[:4]):
                if record[column] != expected:
                    reasons.append(f"{column}_mismatch")
            if record["right_method"] not in rule[4]:
                reasons.append("reference_method_not_independent_for_question")
        for check in ("sampling", "calibration"):
            if not _true(record[f"{check}_verified"]) or not _text(record[f"{check}_evidence"]):
                reasons.append(f"{check}_unverified")
        if not _text(record["identity_evidence"]):
            reasons.append("identity_unverified")
        left_size = _text(record["left_size_fraction"])
        if not left_size or left_size.lower() in {"unknown", "unspecified"} or left_size != _text(record["right_size_fraction"]):
            reasons.append("size_fraction_unmatched")
        overlaps = [np.nan, np.nan]
        try:
            stamps = [pd.Timestamp(record[key]) for key in
                      ("left_start_utc", "left_end_utc", "right_start_utc", "right_end_utc")]
            if any(pd.isna(t) or t.tzinfo is None for t in stamps):
                raise ValueError("timezone-aware intervals required")
            a, b, c, d = [t.tz_convert("UTC") for t in stamps]
            if b <= a or d <= c:
                raise ValueError("nonpositive interval")
            overlap = max(0.0, (min(b, d) - max(a, c)).total_seconds())
            overlaps = [overlap / (b - a).total_seconds(), overlap / (d - c).total_seconds()]
            if min(overlaps) < minimum_overlap:
                reasons.append("insufficient_shared_sampling")
        except (ValueError, TypeError):
            reasons.append("invalid_or_unzoned_intervals")
        if record["comparison"] == "ma350_photoacoustic":
            try:
                left_wl, right_wl = float(record["left_wavelength_nm"]), float(record["right_wavelength_nm"])
                if not np.isfinite([left_wl, right_wl]).all() or left_wl <= 0 or left_wl != right_wl:
                    reasons.append("wavelength_not_exactly_matched")
            except (KeyError, ValueError, TypeError):
                reasons.append("wavelength_evidence_missing")
        elif record["comparison"] == "absorption_sp2":
            try:
                wavelength = float(record["left_wavelength_nm"])
                if not np.isfinite(wavelength) or wavelength <= 0:
                    reasons.append("absorption_wavelength_invalid")
            except (KeyError, ValueError, TypeError):
                reasons.append("absorption_wavelength_missing")
        rows.append({**record, "left_overlap_fraction": overlaps[0], "right_overlap_fraction": overlaps[1],
                     "eligible": not reasons, "exclusion_reason": "|".join(reasons)})
    ledger = pd.DataFrame(rows)
    eligible = int(ledger.eligible.sum()) if len(ledger) else 0
    return {"summary": {"status": "completed" if eligible else "insufficient_data", "n_candidates": len(pairs),
                        "n_eligible": eligible, "minimum_overlap": minimum_overlap,
                        "limitations": ["Eligibility is a protocol check, not method agreement or accuracy.",
                                        "MA350 attenuation and photoacoustic absorption retain their distinct quantities."]},
            "tables": {"colocation_eligibility": ledger}}
