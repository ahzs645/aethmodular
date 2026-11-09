"""Optical evolution within externally defined episodes, with whole-event tests.

Episode definitions must be independent of the optical response being tested.
A changing attenuation exponent is not labelled a source fraction, a measured
soot-mass change, or evidence of aging. This module tests temporal structure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

try:
    from config import WAVELENGTHS_NM
    from optics import aae
except ImportError:
    from ..config import WAVELENGTHS_NM
    from ..optics import aae

from .humidity import _cluster_mean, _connected_blocks, _forward_evaluate, _prepare_minutes, _slice, _supported


@dataclass(frozen=True)
class EpisodeConfig:
    time_col: str = "tu"
    rh_col: str = "Sample RH (%)"
    temp_col: str = "Sample temp (C)"
    session_col: str = "Session ID"
    tape_col: str = "Tape position"
    atn_col: str = "IR ATN1"
    flow1_col: str = "Flow1 (mL/min)"
    flow2_col: str = "Flow2 (mL/min)"
    reference_signal: str = "IR"
    short_signal: str = "UV"
    long_signal: str = "IR"
    cadence_minutes: float = 1.0
    max_gap_minutes: float = 2.0
    lag_tolerance_minutes: float = 0.25
    min_coverage: float = 0.8
    early_fraction: float = 0.25
    late_fraction: float = 0.25
    min_duration_minutes: float = 60.0
    min_matched_pairs: int = 8
    rh_match_pp: float = 2.0
    temp_match_c: float = 2.0
    atn_match: float = 5.0
    flow_ratio_match: float = 0.15
    total_flow_relative_match: float = 0.15
    reference_relative_match: float = 0.25
    model_stride_minutes: float = 5.0
    test_fraction: float = 0.25
    min_train_blocks: int = 3
    min_test_blocks: int = 2
    ridge_alpha: float = 10.0
    n_bootstrap: int = 300
    seed: int = 20261107


def _validate_config(config: EpisodeConfig) -> None:
    for name in ("cadence_minutes", "max_gap_minutes", "min_duration_minutes", "rh_match_pp",
                 "temp_match_c", "atn_match", "flow_ratio_match", "total_flow_relative_match", "reference_relative_match",
                 "model_stride_minutes", "ridge_alpha"):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0 < config.min_coverage <= 1 or not 0 < config.test_fraction < 1:
        raise ValueError("min_coverage must be in (0, 1] and test_fraction in (0, 1)")
    if not (0 < config.early_fraction < 1 and 0 < config.late_fraction < 1
            and config.early_fraction + config.late_fraction <= 1):
        raise ValueError("early/late fractions must define nonoverlapping positive windows")
    if config.max_gap_minutes < config.cadence_minutes:
        raise ValueError("max_gap_minutes must be at least cadence_minutes")
    if not 0 <= config.lag_tolerance_minutes < config.max_gap_minutes:
        raise ValueError("lag tolerance must be nonnegative and shorter than a gap")
    if min(config.min_train_blocks, config.min_test_blocks, config.min_matched_pairs, config.n_bootstrap) < 1:
        raise ValueError("sample, block, and bootstrap counts must be positive")
    if config.short_signal not in WAVELENGTHS_NM or config.long_signal not in WAVELENGTHS_NM:
        raise ValueError("episode signals must name canonical MA350 wavelength channels")
    if WAVELENGTHS_NM[config.short_signal] >= WAVELENGTHS_NM[config.long_signal]:
        raise ValueError("short_signal must have the shorter wavelength")


def _pair_phases(early: pd.DataFrame, late: pd.DataFrame, config: EpisodeConfig) -> pd.DataFrame:
    columns = ["_rh", "_temp", "_atn", "_flow_ratio", "_flow_total", "_reference", "_exponent"]
    first = early.loc[np.isfinite(early[columns]).all(axis=1) & early._reference.gt(0)]
    second = late.loc[np.isfinite(late[columns]).all(axis=1) & late._reference.gt(0)]
    available = set(second.index)
    records = []
    for _, row in first.iterrows():
        candidates = second.loc[second.index.isin(available)]
        distance = pd.DataFrame({
            "rh": (candidates._rh - row._rh).abs() / config.rh_match_pp,
            "temperature": (candidates._temp - row._temp).abs() / config.temp_match_c,
            "attenuation": (candidates._atn - row._atn).abs() / config.atn_match,
            "flow_ratio": (candidates._flow_ratio - row._flow_ratio).abs() / config.flow_ratio_match,
            "total_flow": (candidates._flow_total - row._flow_total).abs()
                          / (config.total_flow_relative_match * row._flow_total),
            "reference_signal": (candidates._reference - row._reference).abs()
                                / (config.reference_relative_match * row._reference),
        })
        valid = distance.le(1).all(axis=1)
        if not valid.any():
            continue
        chosen = distance.loc[valid].pow(2).sum(axis=1).idxmin()
        other = candidates.loc[chosen]
        available.remove(chosen)
        records.append({"early_utc": row._utc, "late_utc": other._utc,
                        "early_input_row": int(row._input_row), "late_input_row": int(other._input_row),
                        "early_exponent": row._exponent, "late_exponent": other._exponent,
                        "late_minus_early": other._exponent - row._exponent,
                        "rh_difference_pp": other._rh - row._rh,
                        "reference_ratio": other._reference / row._reference,
                        "total_flow_ratio": other._flow_total / row._flow_total,
                        "normalized_matching_distance": float(distance.loc[chosen].pow(2).sum())})
    return pd.DataFrame(records)


def run_episode_evolution(
    minutes: pd.DataFrame, episodes: pd.DataFrame | None, *,
    signal_columns: Mapping[str, str], signal_quantity: str, timezone: str,
    rh_provenance: str, config: EpisodeConfig | None = None,
) -> dict:
    """Evaluate early/late spectral evolution and its reproducibility in later events.

    Episode columns: ``event_id``, ``start_utc``, ``end_utc`` (exclusive),
    ``definition`` (exactly ``independent``), and nonempty ``evidence`` naming the
    external definition source. Supply flags with reasons for prior exclusions.
    All events sharing any local day remain in a single validation/bootstrap
    block. An episode spanning a tape, session, invalid-RH interval, or recording
    gap is retained in the audit and excluded from the contiguous-window test.
    """
    config = config or EpisodeConfig()
    _validate_config(config)
    if signal_quantity not in {"b_atn", "b_abs"}:
        raise ValueError("signal_quantity must be 'b_atn' or 'b_abs'; BCc is not absorption")
    if not isinstance(rh_provenance, str) or not rh_provenance.strip():
        raise ValueError("rh_provenance must describe the measured RH source/location")
    quantity = "attenuation_exponent" if signal_quantity == "b_atn" else "absorption_angstrom_exponent"
    summary = {
        "status": "completed", "parameters": asdict(config), "signal_quantity": signal_quantity,
        "outcome": quantity, "timestamp_basis": "UTC", "day_timezone": timezone,
        "rh_provenance": rh_provenance,
        "limitations": [
            "Episode definitions are supplied evidence; this function cannot verify their external provenance.",
            "Spectral evolution alone cannot separate aging, changing sources, mixing, and instrument response.",
            "No source fraction, constant soot mass, bleaching rate, or atmospheric correction is inferred.",
            "The model is a retrospective episode comparison, not a real-time forecast.",
            "Matched phase contrasts apply only to observed common support within the declared calipers.",
        ],
    }
    if episodes is None or episodes.empty:
        return {"summary": {**summary, "status": "blocked", "reason": "independent episode definitions are required"},
                "tables": {}}
    required = {"event_id", "start_utc", "end_utc", "definition", "evidence"}
    missing = sorted(required.difference(episodes.columns))
    if missing:
        raise ValueError(f"episode input lacks {missing}")
    events = episodes.copy().reset_index(drop=True)
    if events.event_id.isna().any() or events.event_id.astype(str).str.strip().eq("").any():
        raise ValueError("every episode needs a nonempty event_id")
    if events.event_id.duplicated().any():
        raise ValueError("episode identifiers must be unique")
    for column in ("start_utc", "end_utc"):
        events[column] = pd.to_datetime(events[column], utc=True, errors="coerce")
    events = events.sort_values("start_utc", kind="stable", na_position="last").reset_index(drop=True)
    valid_times = events.start_utc.notna() & events.end_utc.notna() & events.end_utc.gt(events.start_utc)
    timed = events.loc[valid_times]
    if len(timed) > 1 and (timed.start_utc.iloc[1:].to_numpy() < timed.end_utc.cummax().iloc[:-1].to_numpy()).any():
        raise ValueError("episode intervals must not overlap; repeated minutes would leak between events")
    for label in (config.short_signal, config.long_signal, config.reference_signal):
        if label not in signal_columns:
            raise ValueError(f"required canonical signal {label!r} is missing")
    frame = _prepare_minutes(minutes, signal_columns, timezone, config)
    frame["_exponent"] = aae(frame[f"_signal_{config.short_signal}"],
                              frame[f"_signal_{config.long_signal}"],
                              WAVELENGTHS_NM[config.short_signal], WAVELENGTHS_NM[config.long_signal])
    frame["_reference"] = frame[f"_signal_{config.reference_signal}"]
    frame["_log_reference"] = np.log(frame._reference.where(frame._reference > 0))
    eligible = frame.loc[frame._eligible].copy()
    audit, phase_rows, pair_tables, model_rows, day_sets, eligible_ids = [], [], [], [], [], []
    for i, event in events.iterrows():
        reasons = []
        if not valid_times.loc[i]:
            reasons.append("invalid_episode_interval")
        if str(event.definition).strip() != "independent" or pd.isna(event.evidence) or not str(event.evidence).strip():
            reasons.append("independent_definition_evidence_missing")
        if "is_excluded" in events:
            value = event.is_excluded
            if pd.notna(value) and value not in [True, False, 0, 1]:
                raise ValueError("episode is_excluded must contain boolean or 0/1 flags")
            if pd.notna(value) and bool(value):
                reasons.append("upstream_exclusion:" + str(event.get("exclusion_reason", "unspecified")))
        duration = (event.end_utc - event.start_utc).total_seconds() / 60 if valid_times.loc[i] else np.nan
        if np.isfinite(duration) and duration < config.min_duration_minutes:
            reasons.append("episode_too_short")
        window = _slice(eligible, event.start_utc, event.end_utc) if valid_times.loc[i] else eligible.iloc[:0]
        if window.empty:
            reasons.append("no_overlapping_minutes")
        else:
            if window._segment.nunique() != 1:
                reasons.append("episode_crosses_tape_session_gap_or_invalid_rows")
            if (window._utc.iloc[0] > event.start_utc + pd.Timedelta(minutes=config.lag_tolerance_minutes)
                    or window._utc.iloc[-1] < event.end_utc - pd.Timedelta(minutes=config.cadence_minutes + config.lag_tolerance_minutes)
                    or not _supported(window, duration, config)):
                reasons.append("insufficient_contiguous_coverage")
        record = {"event_id": event.event_id, "start_utc": event.start_utc, "end_utc": event.end_utc,
                  "definition": event.definition, "evidence": event.evidence, "duration_minutes": duration,
                  "n_minutes": len(window), "n_valid_exponent": int(window._exponent.notna().sum()),
                  "is_excluded": bool(reasons), "exclusion_reason": ";".join(reasons)}
        audit.append(record)
        if reasons:
            continue
        early = _slice(window, event.start_utc,
                       event.start_utc + pd.Timedelta(minutes=duration * config.early_fraction))
        late = _slice(window, event.end_utc - pd.Timedelta(minutes=duration * config.late_fraction), event.end_utc)
        for name, rows, fraction in (("early", early, config.early_fraction), ("late", late, config.late_fraction)):
            support = _supported(rows, duration * fraction, config, "_exponent")
            phase_rows.append({"event_id": event.event_id, "phase": name,
                               "n": len(rows), "n_valid_exponent": int(rows._exponent.notna().sum()),
                               "supported": bool(support), "mean_exponent": rows._exponent.mean() if support else np.nan,
                               "mean_reference": rows._reference.mean(), "mean_rh": rows._rh.mean(),
                               "mean_temperature": rows._temp.mean(), "mean_atn": rows._atn.mean(),
                               "mean_flow_ratio": rows._flow_ratio.mean(), "mean_total_flow": rows._flow_total.mean()})
        pairs = _pair_phases(early, late, config)
        record["n_matched_pairs"] = len(pairs)
        record["matching_status"] = "completed" if len(pairs) >= config.min_matched_pairs else "insufficient_common_support"
        if not pairs.empty:
            pair_tables.append(pairs.assign(event_id=event.event_id))
        sample = window.copy()
        sample["_elapsed_hours"] = (sample._utc - event.start_utc).dt.total_seconds() / 3600
        sample["event_id"] = event.event_id
        bucket = sample._utc.astype("int64") // int(pd.Timedelta(minutes=config.model_stride_minutes).value)
        sample = sample.loc[~bucket.duplicated()]
        model_rows.append(sample)
        day_sets.append(set(window._local_day.dropna().astype(str)))
        eligible_ids.append(event.event_id)
    audit_frame = pd.DataFrame(audit)
    tables = {
        "episodes": audit_frame,
        "phases": pd.DataFrame(phase_rows),
        "matched_pairs": pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame(),
        "minute_exclusions": frame.loc[~frame._eligible, ["_input_row", "_utc", "_reason"]].rename(
            columns={"_input_row": "input_row", "_utc": "time_utc", "_reason": "exclusion_reason"}),
    }
    summary.update(n_input_minutes=len(frame), n_input_episodes=len(events), n_eligible_episodes=len(eligible_ids),
                   n_excluded_episodes=len(events) - len(eligible_ids))
    if not eligible_ids:
        return {"summary": {**summary, "status": "insufficient_data", "reason": "no eligible independent contiguous episodes"},
                "tables": tables}
    blocks = dict(zip(eligible_ids, _connected_blocks(day_sets)))
    tables["episodes"]["day_block"] = tables["episodes"].event_id.map(blocks)
    paired = tables["matched_pairs"]
    effect_rows = []
    if not paired.empty:
        for event_id, rows in paired.groupby("event_id", sort=False):
            enough = len(rows) >= config.min_matched_pairs
            effect_rows.append({"event_id": event_id, "day_block": blocks[event_id], "n_pairs": len(rows),
                                "supported": enough,
                                "mean_late_minus_early": rows.late_minus_early.mean() if enough else np.nan})
    effects = pd.DataFrame(effect_rows)
    tables["event_effects"] = effects
    summary["matched_contrast"] = (_cluster_mean(effects.mean_late_minus_early, effects.day_block,
                                                   config.n_bootstrap, config.seed)
                                     if not effects.empty else {"n": 0, "n_blocks": 0})
    model = pd.concat(model_rows, ignore_index=True)
    model["_day_block"] = model.event_id.map(blocks)
    result = _forward_evaluate(
        model, target="_exponent",
        base_features=["_rh", "_temp", "_atn", "_flow_ratio", "_flow_total", "_log_reference", "_hour_sin", "_hour_cos"],
        extra_features=["_elapsed_hours"], group_col="_day_block", time_col="_utc",
        test_fraction=config.test_fraction, min_train_groups=config.min_train_blocks,
        min_test_groups=config.min_test_blocks, ridge_alpha=config.ridge_alpha,
        n_bootstrap=config.n_bootstrap, seed=config.seed,
    )
    summary["model"] = result["summary"]
    for name, table in result["tables"].items():
        table = table.rename(columns={"instantaneous": "state_only", "with_history": "with_elapsed_time",
                                      "sse_instantaneous": "sse_state_only", "sse_with_history": "sse_with_elapsed_time",
                                      "rmse_instantaneous": "rmse_state_only", "rmse_with_history": "rmse_with_elapsed_time"})
        if "model" in table:
            table["model"] = table.model.replace({"instantaneous": "state_only", "with_history": "with_elapsed_time"})
        tables[f"model_{name}"] = table
    return {"summary": summary, "tables": tables}
