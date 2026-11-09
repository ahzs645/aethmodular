"""Auditable hourly comparison with a named published study, without refitting FTIR.

The caller supplies deployment identities and upstream QC provenance. This module
does not infer raw-instrument QC or silently treat an Addis record as Central.
"""

from __future__ import annotations

import hashlib
import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.bc.source_model import two_source_absorption


PROFILE_PATH = Path(__file__).resolve().parents[1] / "workflows/profiles/shlipak_2026.json"
REQUIRED_COLUMNS = {"deployment_id", "timestamp", "bc_ug_m3", "qc_pass"}
OPTICAL_COLUMNS = {"absorption_short_mm1", "absorption_long_mm1"}


def load_profile() -> dict:
    return json.loads(PROFILE_PATH.read_text())


def _hour(value, label: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid timestamp") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{label}: a timestamp with an explicit UTC offset is required")
    stamp = stamp.tz_convert("UTC")
    if stamp != stamp.floor("h"):
        raise ValueError(f"{label}: expected an exact hour-start timestamp, not minute data")
    return stamp


def _required_text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: a nonempty explicit value is required")
    return value.strip()


def _number(value) -> float | None:
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _summary(values: pd.Series, expected: int) -> dict:
    valid = values.dropna()
    return {
        "valid_hours": len(valid),
        "expected_hours": expected,
        "coverage_percent": 100 * len(valid) / expected if expected else None,
        "mean_ug_m3": _number(valid.mean()),
        "sd_ug_m3": _number(valid.std(ddof=1)),
        "max_ug_m3": _number(valid.max()),
    }


def compare_hourly(hourly: pd.DataFrame, metadata: dict, profile: dict | None = None) -> dict:
    """Compare QC-eligible, unique hours with explicit deployment denominators.

    Naive timestamps, duplicates and malformed QC flags fail closed. Missing,
    nonfinite or negative BC remains counted as unavailable, never as zero.
    Optical fractions have a separate eligibility mask and reference wavelength;
    they are not reported as chemical biomass mass or as the paper's BC fraction.
    """
    profile = load_profile() if profile is None else profile
    months = [month for values in profile["seasons"].values() for month in values]
    if sorted(months) != list(range(1, 13)):
        raise ValueError("Profile seasons must assign every month exactly once")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if metadata.get("schema_version") != 1:
        raise ValueError("metadata.schema_version must be 1")
    for key, required in (("timezone", profile["timezone"]), ("bc_unit", "ug/m3"),
                          ("timestamp_basis", "hour_start_with_offset")):
        if metadata.get(key) != required:
            raise ValueError(f"metadata.{key} must be {required!r}")
    qc = metadata.get("qc", {})
    if not isinstance(qc, dict):
        raise ValueError("metadata.qc must be an object with description and reference")
    for key in ("description", "reference"):
        _required_text(qc.get(key), f"metadata.qc.{key}")
    for key in ("instrument_model", "inlet"):
        _required_text(metadata.get(key), f"metadata.{key}")
    deployments = metadata.get("deployments")
    if not isinstance(deployments, dict) or not deployments:
        raise ValueError("metadata.deployments must identify at least one of central or jacros")
    if set(deployments) - {"central", "jacros"}:
        raise ValueError("Use distinct central and jacros deployment IDs; do not map a generic ETAD label implicitly")
    identities = []
    for name, deployment in deployments.items():
        if not isinstance(deployment, dict):
            raise ValueError(f"deployments.{name} must be a JSON object")
        serial = _required_text(deployment.get("serial"), f"{name}.serial")
        begin = _hour(deployment.get("start"), f"{name}.start")
        finish = _hour(deployment.get("end_exclusive"), f"{name}.end_exclusive")
        if begin >= finish:
            raise ValueError(f"{name}: deployment start must precede end_exclusive")
        normalized = "".join(c for c in serial.casefold() if c.isalnum())
        for previous, previous_serial, previous_start, previous_end in identities:
            if normalized == previous_serial and max(begin, previous_start) < min(finish, previous_end):
                raise ValueError(f"{previous} and {name} assign the same instrument serial to overlapping deployments")
        identities.append((name, normalized, begin, finish))
    if not hourly.columns.is_unique:
        raise ValueError("Hourly input contains duplicate column names")
    missing = REQUIRED_COLUMNS - set(hourly.columns)
    if missing:
        raise ValueError(f"Missing hourly columns: {', '.join(sorted(missing))}")
    data = hourly.copy()
    if data.deployment_id.isna().any() or not set(data.deployment_id).issubset(deployments):
        raise ValueError("Every row must name a declared deployment_id")
    data["utc"] = pd.to_datetime([_hour(t, "hourly.timestamp") for t in data.timestamp], utc=True)
    if data.duplicated(["deployment_id", "utc"]).any():
        raise ValueError("Duplicate deployment/hour observations: reconcile them upstream before comparison")
    qc_values = data.qc_pass.astype(str).str.strip().str.lower()
    if not qc_values.isin(["true", "false", "1", "0"]).all():
        raise ValueError("qc_pass must contain true/false or 1/0 for every row")
    data["qc"] = qc_values.isin(["true", "1"])
    bc = pd.to_numeric(data.bc_ug_m3, errors="coerce")
    data["bc"] = bc.where(data.qc & np.isfinite(bc) & (bc >= 0))
    data["month"] = data.utc.dt.tz_convert(profile["timezone"]).dt.month
    season_by_month = {m: name for name, months in profile["seasons"].items() for m in months}
    data["season"] = data.month.map(season_by_month)
    start = _hour(profile["start_local"], "profile.start_local")
    end = _hour(profile["end_local_exclusive"], "profile.end_local_exclusive")
    study_hours = int((end - start) / pd.Timedelta(hours=1))
    if study_hours <= 0:
        raise ValueError("Profile start must precede end_local_exclusive")

    optical = set(data.columns) & OPTICAL_COLUMNS
    source_metadata = metadata.get("source_model")
    if optical and optical != OPTICAL_COLUMNS:
        raise ValueError("Provide both short- and long-wavelength absorption columns")
    if optical:
        if not isinstance(source_metadata, dict) or source_metadata.get("signal_basis") != "absorption":
            raise ValueError("Optical columns require source_model.signal_basis='absorption'; BCc is not absorption")
        _required_text(source_metadata.get("correction_reference"), "source_model.correction_reference")
        allowed = {"signal_basis", "short_wavelength_nm", "long_wavelength_nm", "correction_reference"}
        if set(source_metadata) - allowed:
            raise ValueError("source_model declares optical inputs only; end members and clipping are fixed by this workflow")
        if any(source_metadata.get(key) is None for key in ("short_wavelength_nm", "long_wavelength_nm")):
            raise ValueError("Declare both source-model wavelengths; the paper pair is not inferred")
        short = pd.to_numeric(data.absorption_short_mm1, errors="coerce")
        long = pd.to_numeric(data.absorption_long_mm1, errors="coerce")
        result = two_source_absorption(
            short, long,
            wavelength_short_nm=source_metadata["short_wavelength_nm"],
            wavelength_long_nm=source_metadata["long_wavelength_nm"],
            alpha_fossil=profile["source_model"]["alpha_fossil"],
            alpha_biomass=profile["source_model"]["alpha_biomass"],
        )
        data["source_valid"] = data.qc & result.valid
        data["bb_fraction"] = np.where(data.source_valid, result.biomass_fraction, np.nan)
        data["bb_fraction_unbounded"] = np.where(data.source_valid, result.biomass_fraction_unbounded, np.nan)
        data["clipped_low"] = data.source_valid & result.clipped_low
        data["clipped_high"] = data.source_valid & result.clipped_high
        data["long_absorption"] = long
        data["source_invalid_reason"] = result.invalid_reason

    report = {
        "schema_version": 1,
        "profile_id": profile["id"],
        "citation": profile["citation"],
        "scope": profile["scope"],
        "paper_reproduction_complete": False,
        "independent_validation": False,
        "unresolved_for_exact_reproduction": profile["unresolved_for_exact_reproduction"],
        "study_calendar_hours": study_hours,
        "timezone": profile["timezone"],
        "season_calendar": profile["season_calendar"],
        "festival_exclusions_applied": False,
        "input_rows": len(data),
        "upstream_qc": qc,
        "declared_protocol": {key: metadata[key] for key in ("instrument_model", "inlet")},
        "protocol_differences": {key: {"declared": metadata[key], "paper": profile[key]}
                                 for key in ("instrument_model", "inlet") if metadata[key] != profile[key]},
        "deployments": {},
    }
    matched = []
    for deployment_id, deployment in deployments.items():
        serial = _required_text(deployment.get("serial"), f"{deployment_id}.serial")
        deployed_start = _hour(deployment.get("start"), f"{deployment_id}.start")
        deployed_end = _hour(deployment.get("end_exclusive"), f"{deployment_id}.end_exclusive")
        if deployed_start >= deployed_end:
            raise ValueError(f"{deployment_id}: deployment start must precede end_exclusive")
        lo, hi = max(start, deployed_start), min(end, deployed_end)
        if lo >= hi:
            raise ValueError(f"{deployment_id}: deployment does not overlap the study")
        expected_index = pd.date_range(lo, hi, freq="h", inclusive="left")
        rows = data.loc[data.deployment_id.eq(deployment_id)]
        within = rows.utc.ge(lo) & rows.utc.lt(hi)
        selected = rows.loc[within]
        summary = _summary(selected.bc, len(expected_index))
        published = profile["reported"][deployment_id]
        expected_season = pd.Series(expected_index.tz_convert(profile["timezone"]).month).map(season_by_month)
        summary.update({
            "serial": serial,
            "start_utc": lo.isoformat(), "end_utc_exclusive": hi.isoformat(),
            "input_hours": len(selected),
            "outside_comparison_window_rows": int((~within).sum()),
            "qc_rejected_hours": int((~selected.qc).sum()),
            "invalid_bc_after_qc_hours": int((selected.qc & selected.bc.isna()).sum()),
            "study_calendar_coverage_percent": 100 * selected.bc.notna().sum().item() / study_hours,
            "reported_paper": published,
            "mean_difference_from_paper_ug_m3": (
                summary["mean_ug_m3"] - published["bc_mean_ug_m3"] if summary["mean_ug_m3"] is not None else None
            ),
            "seasonal": {season: _summary(selected.loc[selected.season.eq(season), "bc"],
                                          int(expected_season.eq(season).sum()))
                         for season in profile["seasons"]},
        })
        if optical:
            valid = selected.loc[selected.source_valid]
            invalid = selected.loc[selected.qc & ~selected.source_valid]
            summary["optical_source_model"] = {
                "signal_basis": "absorption",
                "short_wavelength_nm": float(source_metadata["short_wavelength_nm"]),
                "long_wavelength_nm": float(source_metadata["long_wavelength_nm"]),
                "correction_reference": source_metadata["correction_reference"],
                "alpha_fossil": profile["source_model"]["alpha_fossil"],
                "alpha_biomass": profile["source_model"]["alpha_biomass"],
                "clip": True,
                "quantity": "fraction_of_long_wavelength_absorption",
                "valid_hours": len(valid),
                "invalid_after_qc_hours": len(invalid),
                "invalid_reasons": {str(reason): int(count) for reason, count
                                    in invalid.source_invalid_reason.value_counts().items()},
                "coverage_percent": 100 * len(valid) / len(expected_index),
                "clipped_low_hours": int(valid.clipped_low.sum()),
                "clipped_high_hours": int(valid.clipped_high.sum()),
                "mean_fraction": _number(valid.bb_fraction.mean()),
                "mean_unbounded_fraction": _number(valid.bb_fraction_unbounded.mean()),
                "absorption_weighted_fraction": (
                    _number(np.average(valid.bb_fraction, weights=valid.long_absorption)) if len(valid) else None
                ),
                "comparable_to_reported_biomass_bc_percent": False,
            }
        report["deployments"][deployment_id] = summary
        matched.append(selected.set_index("utc").bc.rename(deployment_id))
    common = pd.concat(matched, axis=1).dropna()
    report["common_valid_hours"] = len(common) if len(matched) > 1 else None
    report["common_hour_means_ug_m3"] = (
        {name: _number(common[name].mean()) for name in common} if len(matched) > 1 else None
    )
    report["notes"] = [
        "Coverage uses unique absolute hours inside the declared deployment intersected with the study; calendar coverage is separate.",
        "Paper completeness and source-fraction denominators remain unresolved; differences are descriptive, without pass/fail thresholds.",
        "Seasonal summaries include festival dates and are not the paper's event-excluded climatologies.",
        "The paper's MA350 inlet has no size cut; check the declared inlet before comparing with size-selected PM2.5. Optical BC is not independent thermal EC validation.",
    ]
    return report


def compare_files(hourly_path: Path, metadata_path: Path) -> dict:
    snapshots = {name: (path, path.read_bytes()) for name, path in
                 (("hourly", hourly_path), ("metadata", metadata_path), ("profile", PROFILE_PATH))}
    header = next(csv.reader(io.StringIO(snapshots["hourly"][1].decode("utf-8-sig"))), [])
    if len(header) != len(set(header)):
        raise ValueError("Hourly CSV contains duplicate column names")
    report = compare_hourly(pd.read_csv(io.BytesIO(snapshots["hourly"][1])),
                            json.loads(snapshots["metadata"][1]), json.loads(snapshots["profile"][1]))
    report["inputs"] = {name: {"file": path.name, "sha256": hashlib.sha256(body).hexdigest()}
                        for name, (path, body) in snapshots.items()}
    return report
