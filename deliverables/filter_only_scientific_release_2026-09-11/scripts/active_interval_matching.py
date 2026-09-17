"""Evidence-gated filter matching over documented active sampling periods.

No schedule is inferred from a date, volume, elapsed envelope, or sampled-hour
total. Means are weighted by distinct acquisition slots across all segments.
Eligibility decisions retain individual reasons and never depend on a global
dropna/clean-data intersection.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from observation_coverage import (
        boolean_slot_flags,
        cadence_ns,
        distinct_slot_means,
        expected_slots,
        utc_index,
    )
except ImportError:
    from .observation_coverage import (
        boolean_slot_flags,
        cadence_ns,
        distinct_slot_means,
        expected_slots,
        utc_index,
    )


INTERVAL_COLUMNS = {
    "site": "string",
    "base_filter_id": "string",
    "interval_id": "string",
    "active_start_utc": "datetime64[ns, UTC]",
    "active_end_utc": "datetime64[ns, UTC]",
    "original_local_start": "string",
    "original_local_end": "string",
    "timezone": "string",
    "collection_mode": "string",
    "schedule_evidence_type": "string",
    "source_file": "string",
    "source_file_hash": "string",
    "source_row": "Int64",
    "resolution_reason": "string",
}
MODES = {"documented_continuous", "documented_intermittent"}


def optical_conversion_available(contract):
    return (
        contract.optical_conversion_verified
        and contract.wavelength_treatment_verified
        and bool(contract.absorption_conversion_evidence)
        and np.isfinite(contract.wavelength_nm)
        and contract.wavelength_nm > 0
        and np.isfinite(contract.absorption_coefficient_m2_g)
        and contract.absorption_coefficient_m2_g > 0
    )


def empty_intervals():
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in INTERVAL_COLUMNS.items()})


def validate_intervals(intervals, catalog, verify_source_files=True):
    """Fail closed on missing evidence, overlapping periods, or invented local clocks."""
    if intervals.empty:
        return empty_intervals()
    missing = set(INTERVAL_COLUMNS) - set(intervals)
    if missing:
        raise ValueError(f"Missing interval evidence columns: {sorted(missing)}")
    out = intervals.copy()
    for field in ["active_start_utc", "active_end_utc"]:
        # utc=True alone would silently accept naive clocks. Reject them first.
        if any(pd.isna(v) or pd.Timestamp(v).tzinfo is None for v in out[field]):
            raise ValueError("Active interval UTC fields must include a timezone")
        out[field] = pd.to_datetime(out[field], utc=True)
    if out.interval_id.isna().any() or out.interval_id.duplicated().any():
        raise ValueError("Interval IDs must be present and unique")
    if not out.collection_mode.isin(MODES).all():
        raise ValueError("Only documented collection modes may create active intervals")
    for field in [
        "site",
        "base_filter_id",
        "interval_id",
        "timezone",
        "original_local_start",
        "original_local_end",
        "schedule_evidence_type",
        "source_file",
        "source_file_hash",
        "resolution_reason",
    ]:
        if out[field].isna().any() or out[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"Interval requires {field}")
    if not out.source_file_hash.str.fullmatch("[0-9a-f]{64}").all():
        raise ValueError("Interval requires a SHA-256 source hash")
    if out.source_row.isna().any() or (out.source_row < 0).any() or (out.source_row % 1 != 0).any():
        raise ValueError("Interval source row must be a nonnegative integer")
    identities = set(zip(catalog.site, catalog.base_filter_id))
    hashes = {}
    for row in out.itertuples():
        ZoneInfo(row.timezone)  # Validate the declared zone even for offset-bearing clocks.
        if (row.site, row.base_filter_id) not in identities:
            raise ValueError("Interval identity does not occur in the parent filter catalog")
        if row.active_end_utc <= row.active_start_utc:
            raise ValueError("Active interval must have positive duration")
        for local_field, utc_field in [
            ("original_local_start", "active_start_utc"),
            ("original_local_end", "active_end_utc"),
        ]:
            stamp = pd.Timestamp(getattr(row, local_field))
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize(row.timezone, ambiguous="raise", nonexistent="raise")
            elif stamp.utcoffset() != stamp.tz_convert(row.timezone).utcoffset():
                raise ValueError("Original local clock offset disagrees with the declared timezone")
            if stamp.tz_convert("UTC") != getattr(row, utc_field):
                raise ValueError("Original local clock does not match the active UTC bound")
        if verify_source_files:
            path = Path(row.source_file)
            if path not in hashes:
                digest = sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[path] = digest.hexdigest()
            if hashes[path] != row.source_file_hash:
                raise ValueError(f"Interval source hash changed: {path}")
    out = out.sort_values(["site", "base_filter_id", "active_start_utc"]).reset_index(drop=True)
    for _, group in out.groupby(["site", "base_filter_id"]):
        if group.collection_mode.nunique() != 1:
            raise ValueError("Collection mode must agree within a filter")
        if len(group) > 1 and group.collection_mode.iloc[0] == "documented_continuous":
            raise ValueError("Documented continuous collection requires one interval")
        if (
            group.active_start_utc.iloc[1:].reset_index(drop=True)
            < group.active_end_utc.iloc[:-1].reset_index(drop=True)
        ).any():
            raise ValueError("Overlapping active intervals would double-count coverage")
    return out


@dataclass(frozen=True)
class StreamContract:
    stream_id: str
    cadence_seconds: float = 60
    slot_origin_utc: str = "1970-01-01T00:00:00Z"
    cadence_verified: bool = False
    timestamp_role_verified: bool = False
    observation_provenance_verified: bool = False
    processing_history_verified: bool = False
    units: str = "ng/m3"
    units_verified: bool = False
    quality_rule: str = "finite_numeric_values"
    wavelength_nm: float = np.nan
    optical_conversion_verified: bool = False
    wavelength_treatment_verified: bool = False
    absorption_coefficient_m2_g: float = np.nan
    absorption_conversion_evidence: str = ""
    evidence_scope: str = "unreviewed"
    scope_start_utc: str = ""
    scope_end_utc: str = ""
    scope_session_ids_json: str = "[]"
    scope_justification: str = ""

    def __post_init__(self):
        cadence_ns(self.cadence_seconds)
        origin = pd.Timestamp(self.slot_origin_utc)
        if pd.isna(origin) or origin.tzinfo is None:
            raise ValueError("Stream acquisition origin must be timezone-aware")
        for field in [
            "cadence_verified",
            "timestamp_role_verified",
            "observation_provenance_verified",
            "processing_history_verified",
            "units_verified",
            "optical_conversion_verified",
            "wavelength_treatment_verified",
        ]:
            if not isinstance(getattr(self, field), (bool, np.bool_)):
                raise ValueError(f"{field} must be an explicit boolean")
        if self.units_verified and self.units not in {"ng/m3", "ug/m3"}:
            raise ValueError("Verified eBC units must be ng/m3 or ug/m3")
        if not isinstance(self.quality_rule, str) or not self.quality_rule.strip():
            raise ValueError("A stream contract must declare its validity rule")
        if self.evidence_scope not in {"unreviewed", "full_export", "bounded_period"}:
            raise ValueError("Evidence scope must be unreviewed, full_export or bounded_period")
        sessions = json.loads(self.scope_session_ids_json)
        if not isinstance(sessions, list) or any(not isinstance(s, str) for s in sessions):
            raise ValueError("Scoped session IDs must be a JSON list of strings")
        if self.evidence_scope != "unreviewed" and not self.scope_justification.strip():
            raise ValueError("Reviewed evidence requires an explicit scope justification")
        if self.evidence_scope == "bounded_period":
            start, end = map(pd.Timestamp, [self.scope_start_utc, self.scope_end_utc])
            if (
                pd.isna(start)
                or pd.isna(end)
                or start.tzinfo is None
                or end.tzinfo is None
                or end <= start
            ):
                raise ValueError("Bounded evidence requires positive timezone-aware UTC bounds")


def evidence_covers_segment(contract, interval, frame):
    """A reviewed assertion only applies within its documented time/session scope."""
    if contract.evidence_scope == "unreviewed":
        return False
    if contract.evidence_scope == "bounded_period" and not (
        pd.Timestamp(contract.scope_start_utc) <= interval.active_start_utc
        and interval.active_end_utc <= pd.Timestamp(contract.scope_end_utc)
    ):
        return False
    sessions = json.loads(contract.scope_session_ids_json)
    if sessions and (
        "Session ID" not in frame
        or frame["Session ID"].isna().any()
        or not frame["Session ID"].astype(str).isin(sessions).all()
    ):
        return False
    return True


@dataclass(frozen=True)
class CoveragePolicy:
    """Declared screening choice, never selected from calibration performance."""

    minimum_valid_observed_fraction: float = 0.75
    minimum_segment_valid_observed_fraction: float = 0.50
    name: str = "initial_75pct_total_50pct_each_segment"

    def __post_init__(self):
        for value in [
            self.minimum_valid_observed_fraction,
            self.minimum_segment_valid_observed_fraction,
        ]:
            if not 0 <= value <= 1:
                raise ValueError("Coverage fractions must be between zero and one")


def summarize_active_segments(
    observations, intervals, value_column, contract, observed_column=None, valid_column=None
):
    """Return per-segment coverage and a slot-weighted filter summary.

    observation_provenance_verified must come from the stream's evidence record.
    Neither non-null values nor a clean-looking file name establishes it. Flags
    can be channel specific by invoking this function with the channel's columns.
    """
    if intervals.empty:
        return pd.DataFrame(), {}
    if intervals[["site", "base_filter_id"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Summarize one physical filter at a time")
    ordered = intervals.sort_values("active_start_utc")
    if (
        ordered.active_start_utc.iloc[1:].reset_index(drop=True)
        < ordered.active_end_utc.iloc[:-1].reset_index(drop=True)
    ).any():
        raise ValueError("Overlapping active intervals would double-count coverage")
    frame = observations.copy()
    frame.index = utc_index(frame.index)
    if frame.index.has_duplicates:
        raise ValueError("Duplicate acquisition timestamps require explicit resolution")
    if value_column not in frame:
        raise ValueError(f"Missing channel: {value_column}")
    values = pd.to_numeric(frame[value_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    obs = (
        pd.Series(pd.NA, index=frame.index, dtype="boolean")
        if observed_column is None
        else frame[observed_column]
    )
    valid = (
        pd.Series(True, index=frame.index, dtype="boolean")
        if valid_column is None
        else frame[valid_column]
    )
    if not pd.api.types.is_bool_dtype(obs.dtype) or not pd.api.types.is_bool_dtype(valid.dtype):
        raise ValueError("Observed/valid flags must be boolean, including nullable booleans")
    rows = []
    all_input, all_valid = [], []
    for r in intervals.itertuples():
        inside = (frame.index >= r.active_start_utc) & (frame.index < r.active_end_utc)
        segment = values.loc[inside].to_frame(value_column)
        segment_obs, segment_valid = obs.loc[inside], valid.loc[inside]
        scope_known = evidence_covers_segment(contract, r, frame.loc[inside])
        expected = expected_slots(
            r.active_start_utc, r.active_end_utc, contract.cadence_seconds, contract.slot_origin_utc
        )
        if len(expected) == 0:
            raise ValueError("No expected acquisition slots inside the active segment")
        input_slots = distinct_slot_means(
            segment, cadence_seconds=contract.cadence_seconds, origin_utc=contract.slot_origin_utc
        ).reindex(expected)
        observed_slots, unknown_obs = boolean_slot_flags(
            segment_obs, segment.index, contract.cadence_seconds, contract.slot_origin_utc
        )
        _, unknown_valid = boolean_slot_flags(
            segment_valid, segment.index, contract.cadence_seconds, contract.slot_origin_utc
        )
        observed_count = int(observed_slots.reindex(expected, fill_value=False).sum())
        valid_values = segment.where((segment_obs & segment_valid).fillna(False), axis=0)
        valid_slots = distinct_slot_means(
            valid_values,
            cadence_seconds=contract.cadence_seconds,
            origin_utc=contract.slot_origin_utc,
        ).reindex(expected)
        input_count = int(input_slots[value_column].notna().sum())
        valid_count = int(valid_slots[value_column].notna().sum())
        observation_known = (
            contract.observation_provenance_verified
            and scope_known
            and contract.cadence_verified
            and contract.timestamp_role_verified
            and not unknown_obs.any()
        )
        validity_known = observation_known and not unknown_valid.any()
        quarters = []
        for quarter in range(4):
            quarter_start = (
                r.active_start_utc + (r.active_end_utc - r.active_start_utc) * quarter / 4
            )
            quarter_end = (
                r.active_start_utc + (r.active_end_utc - r.active_start_utc) * (quarter + 1) / 4
            )
            quarter_slots = expected[(expected >= quarter_start) & (expected < quarter_end)]
            n = len(quarter_slots)
            n_input = int(input_slots.loc[quarter_slots, value_column].notna().sum())
            n_valid = int(valid_slots.loc[quarter_slots, value_column].notna().sum())
            quarters.append(
                dict(
                    quarter=quarter + 1,
                    start_utc=quarter_start.isoformat(),
                    end_utc=quarter_end.isoformat(),
                    expected_slots=n,
                    input_slots=n_input,
                    input_availability_fraction=n_input / n if n else None,
                    valid_observed_slots=n_valid if validity_known else None,
                    valid_observed_coverage_fraction=n_valid / n if n and validity_known else None,
                    diagnostic_only=True,
                )
            )
        # Source identities may be present even for non-observed/interpolated
        # inputs. Preserve links separately from their inclusion in the mean.
        links = (
            [
                {"source_file_hash": str(h), "source_rows": [int(v) for v in g.source_row]}
                for h, g in frame.loc[inside].groupby("source_file_hash", sort=True)
            ]
            if {"source_row", "source_file_hash"} <= set(frame)
            else []
        )
        rows.append(
            dict(
                interval_id=r.interval_id,
                expected_slots=len(expected),
                input_slots=input_count,
                observed_slots=observed_count if observation_known else np.nan,
                valid_observed_slots=valid_count if validity_known else np.nan,
                input_availability_fraction=input_count / len(expected),
                observed_coverage_fraction=observed_count / len(expected)
                if observation_known
                else np.nan,
                valid_observed_coverage_fraction=valid_count / len(expected)
                if validity_known
                else np.nan,
                unknown_observation_slots=int(unknown_obs.sum()),
                unknown_validity_slots=int(unknown_valid.sum()),
                observation_provenance_status="verified" if observation_known else "unresolved",
                processing_evidence_scope_status="verified"
                if scope_known
                else "unreviewed_or_outside_scope",
                quarter_coverage_json=json.dumps(quarters, sort_keys=True),
                raw_source_links=json.dumps(links, sort_keys=True),
                active_start_utc=r.active_start_utc,
                active_end_utc=r.active_end_utc,
            )
        )
        all_input.append(input_slots[value_column])
        all_valid.append(valid_slots[value_column] if validity_known else pd.Series(dtype=float))
    segments = pd.DataFrame(rows)
    expected_n = int(segments.expected_slots.sum())
    known_obs = segments.observed_slots.notna().all()
    known_valid = segments.valid_observed_slots.notna().all()
    summary = dict(
        expected_slots=expected_n,
        input_slots=int(segments.input_slots.sum()),
        observed_slots=float(segments.observed_slots.sum()) if known_obs else np.nan,
        valid_observed_slots=float(segments.valid_observed_slots.sum()) if known_valid else np.nan,
        input_availability_fraction=float(segments.input_slots.sum() / expected_n),
        observed_coverage_fraction=float(segments.observed_slots.sum() / expected_n)
        if known_obs
        else np.nan,
        valid_observed_coverage_fraction=float(segments.valid_observed_slots.sum() / expected_n)
        if known_valid
        else np.nan,
        minimum_segment_valid_observed_coverage=float(
            segments.valid_observed_coverage_fraction.min()
        )
        if known_valid
        else np.nan,
        mean_input_native=float(pd.concat(all_input).mean()),
        mean_valid_observed_native=float(pd.concat(all_valid).mean()) if known_valid else np.nan,
        observation_provenance_status="verified" if known_obs else "unresolved",
        processing_evidence_scope_status="verified"
        if segments.processing_evidence_scope_status.eq("verified").all()
        else "unreviewed_or_outside_scope",
    )
    if (
        contract.units_verified
        and contract.units in {"ng/m3", "ug/m3"}
        and summary["processing_evidence_scope_status"] == "verified"
    ):
        factor = 0.001 if contract.units == "ng/m3" else 1.0
        summary["mean_input_ebc_ugm3"] = summary["mean_input_native"] * factor
        summary["mean_valid_observed_ebc_ugm3"] = summary["mean_valid_observed_native"] * factor
    else:
        summary["mean_input_ebc_ugm3"] = summary["mean_valid_observed_ebc_ugm3"] = np.nan
    summary["absorption_Mm1"] = np.nan
    if optical_conversion_available(contract):
        summary["absorption_Mm1"] = (
            summary["mean_valid_observed_ebc_ugm3"] * contract.absorption_coefficient_m2_g
        )
    return segments, summary


def eligibility(
    filter_row,
    summary,
    contract,
    policy=CoveragePolicy(),
    schedule_verified=False,
    ec_parameter_resolved=False,
):
    """Question-specific gates; ambiguous ChemSpec EC does not gate HIPS/eBC."""
    reasons = []
    if not schedule_verified:
        reasons.append("active_schedule_unverified")
    if bool(filter_row.get("is_excluded", False)):
        reasons.append("registered_exclusion")
    if not np.isfinite(filter_row.get("hips_fabs_Mm1", np.nan)):
        reasons.append("hips_unavailable")
    if bool(filter_row.get("filter_type_conflict", True)):
        reasons.append("filter_type_conflict")
    if filter_row.get("filter_type") != "PM2.5":
        reasons.append("not_confirmed_pm25_sample")
    if bool(filter_row.get("hips_fabs_Mm1_conflict", True)):
        reasons.append("hips_value_conflict")
    if not bool(filter_row.get("hips_fabs_Mm1_units_valid", False)):
        reasons.append("hips_units_unverified")
    if not contract.observation_provenance_verified:
        reasons.append("observation_history_unverified")
    if not contract.processing_history_verified:
        reasons.append("bcc_processing_history_unverified")
    if not contract.units_verified:
        reasons.append("ebc_units_unverified")
    if summary.get("processing_evidence_scope_status") != "verified":
        reasons.append("processing_evidence_scope_unverified_or_outside_period")
    if not contract.cadence_verified or not contract.timestamp_role_verified:
        reasons.append("acquisition_grid_unverified")
    cov = summary.get("valid_observed_coverage_fraction", np.nan)
    seg = summary.get("minimum_segment_valid_observed_coverage", np.nan)
    if not np.isfinite(cov):
        reasons.append("observed_coverage_unknown")
    elif cov < policy.minimum_valid_observed_fraction:
        reasons.append("total_coverage_below_policy")
    if not np.isfinite(seg):
        reasons.append("segment_coverage_unknown")
    elif seg < policy.minimum_segment_valid_observed_fraction:
        reasons.append("segment_coverage_below_policy")
    if not np.isfinite(summary.get("mean_valid_observed_ebc_ugm3", np.nan)):
        reasons.append("valid_observed_ebc_mean_unavailable")
    interval_eligible = not reasons
    optical = optical_conversion_available(contract)
    filter_ok = bool(filter_row.get("eligible_hips_ftir_diagnostic", False))
    ec = filter_row.get("ftir_ec_ugm3", np.nan)
    mdl = filter_row.get("ftir_ec_mdl_ugm3", np.nan)
    ratio_reasons = []
    if not filter_ok:
        ratio_reasons.append("filter_diagnostic_ineligible")
    if not ec_parameter_resolved:
        ratio_reasons.append("ftir_ec_identity_unresolved")
    if not np.isfinite(ec) or ec <= 0:
        ratio_reasons.append("ftir_ec_denominator_not_positive")
    if not np.isfinite(mdl) or mdl < 0 or bool(filter_row.get("ftir_ec_mdl_conflict", True)):
        ratio_reasons.append("ftir_ec_mdl_unresolved")
    elif np.isfinite(ec) and ec < mdl:
        ratio_reasons.append("ftir_ec_below_reported_mdl")
    ratio = not ratio_reasons
    independent = (
        filter_ok
        and bool(filter_row.get("independent_ec_reference_available", False))
        and bool(filter_row.get("independent_ec_reference_evidence", ""))
    )
    return dict(
        eligible_filter_diagnostic=filter_ok,
        eligible_interval_ebc_comparison=interval_eligible,
        eligible_absorption_comparison=interval_eligible and optical,
        eligible_ec_ratio_analysis=bool(ratio),
        ec_ratio_denominator="ftir_ec_ugm3",
        ec_ratio_rule="positive_and_at_or_above_reported_ftir_mdl; requires_resolved_ftir_identity",
        eligible_independent_ec_validation=independent,
        independent_ec_ineligibility_reason=""
        if independent
        else "documented_independent_ec_reference_unavailable",
        ec_ratio_ineligibility_reason="; ".join(ratio_reasons),
        interval_ineligibility_reason="; ".join(reasons),
        absorption_ineligibility_reason="; ".join(
            reasons + ([] if optical else ["optical_conversion_or_wavelength_unverified"])
        ),
        coverage_policy=policy.name,
    )
