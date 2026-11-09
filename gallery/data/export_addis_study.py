"""Build the Addis study page from the saved, identified Jacros hourly export.

Run with ``uv run python gallery/data/export_addis_study.py``. This needs only
the public overview snapshot and the named paper profile, not private raw data.
It does not rerun instrument QC or reconstruct source fractions. All statistics
use available finite hours, with both full-calendar and snapshot denominators.
Festival windows remain included in the seasonal and diurnal summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/ftir_hips_chem/scripts/config.py"
sys.path.insert(0, str(CONFIG.parent))
from config import SITES, resolve_seasons  # noqa: E402

DATA = ROOT / "gallery/app/public/data/aethalometer"
PROFILE = ROOT / "research/ftir_hips_chem/workflows/profiles/shlipak_2026.json"
SOURCE_REVIEW = ROOT / "research/ftir_hips_chem/workflows/profiles/ec_bc_jpl_review.json"
OUT = DATA / "addis-study.json"
SITE = "ETAD"
SITE_NAME = "Addis_Ababa"
ZONE = SITES[SITE_NAME]["timezone"]
UPSTREAM_RECORD = "Jacros_MA350_1-min_2022-2024_Cleaned.csv"
HOUR = pd.Timedelta(hours=1)


def _local_hour(value, label: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid hour with a UTC offset") from exc
    if pd.isna(stamp) or stamp.tzinfo is None or stamp != stamp.floor("h"):
        raise ValueError(f"{label} must be an exact hour with a UTC offset")
    return stamp.tz_convert(ZONE)


def decode_snapshot(snapshot: dict, metadata: dict) -> pd.Series:
    """Decode one implicit hourly grid, retaining finite values without a cutoff.

    A generic ETAD file is not sufficient identity evidence: the companion index
    must explicitly name the known Jacros minute record. NaN/infinity are missing;
    finite negative/zero values are retained, matching the upstream finite-value
    policy. Numeric strings and booleans fail as malformed observations.
    """
    if snapshot.get("site") != SITE or snapshot.get("unit") != "ng/m3":
        raise ValueError("Expected the ETAD IR snapshot explicitly stored in ng/m3")
    site = metadata.get("sites", {}).get(SITE, {})
    if (
        site.get("site") != SITE_NAME
        or site.get("resolution") != "minute"
        or Path(site.get("hourly_file", "")).name != UPSTREAM_RECORD
    ):
        raise ValueError("The overview index must identify the Jacros minute record")
    values = snapshot.get("IR")
    if not isinstance(values, list) or not values:
        raise ValueError("IR must be a nonempty list of hourly numbers or nulls")
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)))
           for value in values):
        raise ValueError("IR observations must be numeric or null")
    offset = snapshot.get("offset")
    if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
        raise ValueError("The snapshot must declare a finite local-standard UTC offset")
    try:
        first = pd.Timestamp(snapshot.get("first"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Snapshot first must be a local-standard hour") from exc
    if pd.isna(first) or first.tzinfo is not None or first != first.floor("h"):
        raise ValueError("Snapshot first must be a naive local-standard hour")
    try:
        first = first.tz_localize(timezone(timedelta(hours=offset))).tz_convert(ZONE)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid snapshot UTC offset") from exc
    index = pd.date_range(first, periods=len(values), freq="h")
    if any(stamp.utcoffset().total_seconds() / 3600 != offset for stamp in (index[0], index[-1])):
        raise ValueError("Snapshot local-standard offset does not match Addis Ababa")
    if (
        site.get("offset") != offset
        or site.get("hours") != len(index)
        or site.get("first") != snapshot["first"]
        or site.get("last") != index[-1].strftime("%Y-%m-%d %H:%M")
    ):
        raise ValueError("Snapshot and companion index disagree on the hourly grid")
    numbers = np.array([np.nan if value is None else value for value in values], dtype=float)
    numbers[~np.isfinite(numbers)] = np.nan
    if site.get("valid_hours") != int(np.isfinite(numbers).sum()):
        raise ValueError("Snapshot and companion index disagree on finite-hour support")
    # The sole concentration conversion: the saved overview uses ng/m3 integers.
    return pd.Series(numbers / 1000.0, index=index, name="bc_ug_m3")


def _number(value) -> float | None:
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _stats(frame: pd.DataFrame) -> dict:
    values = frame.bc_ug_m3.dropna()
    expected, snapshot, valid = len(frame), int(frame.source_present.sum()), len(values)
    return {
        "expected_hours": expected,
        "snapshot_hours": snapshot,
        "valid_hours": valid,
        "coverage_percent": 100 * valid / expected if expected else None,
        "snapshot_coverage_percent": 100 * valid / snapshot if snapshot else None,
        "mean_bc_ug_m3": _number(values.mean()),
        "median_bc_ug_m3": _number(values.median()),
        "sd_bc_ug_m3": _number(values.std(ddof=1)),
        "max_bc_ug_m3": _number(values.max()),
        "q25_bc_ug_m3": _number(values.quantile(0.25)),
        "q75_bc_ug_m3": _number(values.quantile(0.75)),
    }


def _timed_stats(frame: pd.DataFrame) -> dict:
    return {
        "start_local": frame.index[0].isoformat(),
        "end_local_exclusive": (frame.index[-1] + HOUR).isoformat(),
        **_stats(frame),
    }


def _calendars(frame: pd.DataFrame, profile: dict) -> dict:
    calendars = {}
    for calendar_id, label in (
        ("belg_feb", "February in Belg (paper calendar)"),
        ("dry_feb", "February in the dry season (legacy calendar)"),
    ):
        definitions = resolve_seasons(calendar_id)
        if calendar_id == profile["season_calendar"]:
            if list(profile["seasons"].values()) != [s["months"] for s in definitions.values()]:
                raise ValueError("The paper profile and named season calendar disagree")
        seasonal, diurnal = [], []
        for name, definition in definitions.items():
            season_id = name.split()[0].lower()
            if season_id == "dry":
                season_id = "bega"
            subset = frame.loc[frame.index.month.isin(definition["months"])]
            seasonal.append({
                "id": season_id, "label": name,
                "months": definition["months"], "color": definition["color"],
                **_stats(subset),
            })
            for hour in range(24):
                diurnal.append({
                    "season_id": season_id, "hour": hour,
                    **_stats(subset.loc[subset.index.hour == hour]),
                })
        calendars[calendar_id] = {
            "id": calendar_id, "label": label,
            "festival_exclusions_applied": False,
            "seasons": seasonal, "diurnal": diurnal,
        }
    return calendars


def _events(frame: pd.DataFrame, profile: dict) -> list[dict]:
    events = []
    for name, dates in profile["event_dates_local"].items():
        if name == "Meskel":
            start_hour = 12
            rule = (
                "Noon on the festival date to noon the next day (24 hours), the display window "
                "in Shlipak et al. Figure 6; not an inferred climatology exclusion."
            )
        elif name == "Hidar Sitaten":
            start_hour = 0
            rule = (
                "Local calendar day, midnight to midnight (24 hours), also used by the archived "
                "JPL Figures.ipynb Hidar display. Its relation to the final paper's exact "
                "climatology exclusion remains unverified."
            )
        else:
            raise ValueError(f"No explicit display window declared for event {name!r}")
        for day in dates:
            start = pd.Timestamp(day).tz_localize(ZONE) + start_hour * HOUR
            index = pd.date_range(start, periods=24, freq="h")
            subset = frame.reindex(index)
            subset["source_present"] = subset.source_present.fillna(False).astype(bool)
            stats = _timed_stats(subset)
            events.append({
                "id": f"{name.lower().replace(' ', '-')}-{start.year}",
                "label": name, "date_local": day, "window_rule": rule,
                **stats,
                "available_hours": stats["valid_hours"],
                "incomplete": stats["valid_hours"] < stats["expected_hours"],
                "unavailable": stats["valid_hours"] == 0,
                "trace": [
                    {"elapsed_hour": hour, "local": stamp.isoformat(),
                     "bc_ug_m3": _number(row.bc_ug_m3), "source_present": bool(row.source_present)}
                    for hour, (stamp, row) in enumerate(subset.iterrows())
                ],
            })
    return events


def build_study(snapshot: dict, metadata: dict, profile: dict, *, generated: str | None = None) -> dict:
    """Return a complete study grid and summaries without inventing missing hours."""
    if profile.get("timezone") != ZONE or profile.get("season_calendar") != "belg_feb":
        raise ValueError("Expected the named Addis profile and its belg_feb calendar")
    source = decode_snapshot(snapshot, metadata)
    start = _local_hour(profile["start_local"], "profile.start_local")
    end = _local_hour(profile["end_local_exclusive"], "profile.end_local_exclusive")
    if start >= end:
        raise ValueError("Study start must precede its exclusive end")
    index = pd.date_range(start, end, freq="h", inclusive="left")
    frame = pd.DataFrame({
        "bc_ug_m3": source.reindex(index),
        "source_present": index.isin(source.index),
    }, index=index)
    valid_index = frame.index[frame.bc_ug_m3.notna()]
    summary = _timed_stats(frame)
    coverage = {
        **_stats(frame),
        "outside_snapshot_hours": len(frame) - int(frame.source_present.sum()),
        "missing_within_snapshot_hours": int(frame.source_present.sum()) - len(valid_index),
        "first_valid_local": valid_index[0].isoformat() if len(valid_index) else None,
        "last_valid_local": valid_index[-1].isoformat() if len(valid_index) else None,
    }
    return {
        "schema_version": 1,
        "generated": generated or datetime.now(timezone.utc).isoformat(),
        "paper": profile,
        "paper_reproduction_complete": False,
        "independent_validation": False,
        "deployment": {
            "id": "jacros", "label": "Jacros", "site_code": SITE, "serial": "MA350-0238",
            "instrument_model": "MA350", "timezone": ZONE, "channel": "IR", "unit": "ug/m3",
        },
        "period": {"start_local": start.isoformat(), "end_local_exclusive": end.isoformat(), "timezone": ZONE},
        "source": {
            "kind": "saved_public_hourly_export",
            "generated": metadata["generated"],
            "upstream_record": metadata["sites"][SITE]["hourly_file"],
            "timestamp_basis": "Local-standard hour starts, explicit UTC+03:00; Addis has no daylight-saving shift",
            "source_unit": "ng/m3", "output_unit": "ug/m3",
            "min_valid_ir_minutes": metadata["rules"]["min_minutes"],
            "finite_handling": (
                "Inherited overview rule: each hour needs at least the declared count of finite IR minutes; "
                "no 3-SD upper screen. This export retains every finite saved hourly value, including zero "
                "or negative values; null and nonfinite values are unavailable. No raw-minute QC is rerun."
            ),
            "additional_value_screen": False,
            "first_local": source.index[0].isoformat(),
            "last_local": source.index[-1].isoformat(),
            "end_local_exclusive": (source.index[-1] + HOUR).isoformat(),
        },
        "coverage": coverage, "summary": summary,
        "daily": [_timed_stats(group) for _, group in frame.groupby(frame.index.date, sort=True)],
        "monthly": [_timed_stats(group) for _, group in frame.groupby(frame.index.strftime("%Y-%m"), sort=True)],
        "calendars": _calendars(frame, profile),
        "events": _events(frame, profile),
        "capabilities": {
            "central": {"available": False, "reason": "No identified Central observations are present in this public snapshot."},
            "source_fractions": {
                "available": False,
                "reason": "This snapshot has IR concentration only; paired, corrected absorption and its coverage are required for defensible source fractions.",
            },
        },
        "limitations": [
            "Saved derived observations cover only part of the paper period; no private raw-minute data or full paper QC has been rerun.",
            "All means describe available finite hours, with no new daily or event completeness gate. Missing hours are not zero, and sparse bins can be unrepresentative.",
            "Full-study calendar coverage and coverage within the saved grid have different denominators. Neither reproduces the paper's reported 93.9% deployment completeness.",
            "All festival hours remain in the monthly, seasonal and diurnal summaries; the paper excludes festivals in some climatologies, with exact exclusion windows unresolved.",
            "Meskel uses the paper's Figure 6 noon-to-noon display window. Hidar Sitaten uses the archived JPL notebook's midnight-to-midnight display window, not a verified final-paper climatology exclusion. Event means are available-hour means, not causal effects.",
            "ETAD and ETAD-JACROS identify overlapping records from the same Jacros MA350-0238, not two independent sites. Central is not reconstructed from either alias.",
            "Published BC statistics and source estimates are separate benchmarks; overlapping Jacros observations cannot provide independent validation.",
            "Saved concentrations were rounded to integer ng/m3 upstream and are converted to ug/m3 once. SD uses the sample definition (n-1); quartiles describe the available hourly distribution.",
        ],
    }


def _recorded_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def build_source_review(profile: dict) -> dict:
    """Keep attributed historical summaries separate from an analytical response.

    The response is a constant-k illustration, never a correction to saved BC.
    Negative or singular denominators are explicitly unavailable. No fit or
    uncertainty interval is reconstructed from deck images or notebook outputs.
    """
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported source-material review profile")
    candidates = profile.get("loading_candidates", [])
    if not candidates or len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("Loading candidates need unique identities")
    series = []
    for candidate in candidates:
        k = candidate.get("k")
        if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k <= 0:
            raise ValueError("Loading candidate k must be finite and positive")
        points = []
        for atn in range(61):
            denominator = 1 - k * atn
            valid = denominator > 0
            points.append({"atn1": atn, "multiplier": 1 / denominator if valid else None,
                           "denominator_valid": valid})
        series.append({"candidate_id": candidate["id"], "singular_atn1": 1 / k, "points": points})
    return {
        **profile,
        "applied_to_observations": False,
        "loading_sensitivity": {
            "basis": "analytical_constant_k",
            "equation": "BCc / BC1 = 1 / (1 - k * ATN1)",
            "atn_min": 0, "atn_max": 60,
            "scope": "Illustrative ATN interval in instrument units; not an observed range or a QC cutoff.",
            "series": series,
        },
    }


def export_study(data_dir: Path = DATA, profile_path: Path = PROFILE, output: Path = OUT,
                 source_review_path: Path = SOURCE_REVIEW) -> dict:
    """Read and hash the exact input bytes, then write strict JSON."""
    input_path = data_dir / "overview/hourly/ETAD.json"
    metadata_path = data_dir / "overview/index.json"
    inputs = [("input", input_path), ("metadata", metadata_path), ("profile", profile_path),
              ("source_review", source_review_path), ("config", CONFIG), ("exporter", Path(__file__))]
    raw = {key: path.read_bytes() for key, path in inputs}
    result = build_study(*(json.loads(raw[key]) for key in ("input", "metadata", "profile")))
    result["source_review"] = build_source_review(json.loads(raw["source_review"]))
    for key, path in inputs:
        result["source"][f"{key}_file"] = _recorded_path(path)
        result["source"][f"{key}_sha256"] = hashlib.sha256(raw[key]).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--source-review", type=Path, default=SOURCE_REVIEW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    result = export_study(args.data_dir, args.profile, args.out, args.source_review)
    coverage = result["coverage"]
    print(f"Wrote {_recorded_path(args.out)}: {coverage['valid_hours']:,}/{coverage['expected_hours']:,} "
          "study-calendar hours; identified Jacros IR observations only")


if __name__ == "__main__":
    main()
