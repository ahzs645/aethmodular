"""Prespecified, time-respecting tests of optical response after RH changes.

These are association/diagnostic tests, not an MA350 humidity correction. Input
timestamps are UTC; local time is used only for day blocks and matching. Every
lag and event window stays inside one tape/session/gap segment. No optical
outcome is used to select humidity events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from seasonal_ml_checks import regression_scores
    from active_interval_matching import StreamContract, summarize_active_segments, validate_intervals
except ImportError:
    from ..seasonal_ml_checks import regression_scores
    from ..active_interval_matching import StreamContract, summarize_active_segments, validate_intervals


@dataclass(frozen=True)
class HumidityConfig:
    time_col: str = "tu"
    rh_col: str = "Sample RH (%)"
    temp_col: str = "Sample temp (C)"
    session_col: str = "Session ID"
    tape_col: str = "Tape position"
    atn_col: str = "IR ATN1"
    flow1_col: str = "Flow1 (mL/min)"
    flow2_col: str = "Flow2 (mL/min)"
    reference_signal: str = "IR"
    cadence_minutes: float = 1.0
    max_gap_minutes: float = 2.0
    lag_tolerance_minutes: float = 0.25
    step_minutes: float = 10.0
    baseline_minutes: float = 10.0
    sustain_minutes: float = 5.0
    recovery_minutes: float = 60.0
    bin_minutes: float = 5.0
    rh_change_pp: float = 5.0
    stable_plateau_range_pp: float = 3.0
    stability_pp_per_min: float = 0.1
    min_coverage: float = 0.8
    min_event_separation_minutes: float = 90.0
    lag_minutes: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0)
    rh_match_pp: float = 2.0
    temp_match_c: float = 2.0
    atn_match: float = 5.0
    flow_ratio_match: float = 0.15
    total_flow_relative_match: float = 0.15
    reference_relative_match: float = 0.25
    hour_match: float = 2.0
    model_stride_minutes: float = 5.0
    test_fraction: float = 0.25
    min_train_days: int = 3
    min_test_days: int = 2
    ridge_alpha: float = 10.0
    n_bootstrap: int = 300
    seed: int = 20261107


def _validate_config(config: HumidityConfig) -> None:
    positive = (
        "cadence_minutes", "max_gap_minutes", "step_minutes", "baseline_minutes",
        "sustain_minutes", "recovery_minutes", "bin_minutes", "rh_change_pp",
        "stable_plateau_range_pp", "min_event_separation_minutes", "rh_match_pp",
        "temp_match_c", "atn_match", "flow_ratio_match", "total_flow_relative_match", "reference_relative_match",
        "hour_match", "model_stride_minutes", "ridge_alpha",
    )
    for name in positive:
        if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0 < config.min_coverage <= 1 or not 0 < config.test_fraction < 1:
        raise ValueError("min_coverage must be in (0, 1] and test_fraction in (0, 1)")
    if config.max_gap_minutes < config.cadence_minutes:
        raise ValueError("max_gap_minutes must be at least cadence_minutes")
    if not 0 <= config.lag_tolerance_minutes < config.max_gap_minutes:
        raise ValueError("lag tolerance must be nonnegative and shorter than a gap")
    if not config.lag_minutes or any(not np.isfinite(v) or v <= 0 for v in config.lag_minutes):
        raise ValueError("lag_minutes must contain positive, finite durations")
    if config.min_train_days < 1 or config.min_test_days < 1 or config.n_bootstrap < 1:
        raise ValueError("day counts and bootstrap count must be positive")
    if config.stability_pp_per_min < 0 or not np.isfinite(config.stability_pp_per_min):
        raise ValueError("stability_pp_per_min must be nonnegative and finite")


def reconstruct_ma350_attenuation(
    minutes: pd.DataFrame,
    *,
    cross_sections: Mapping[str, float],
    bc_unit: str = "ng/m3",
    include_spots: bool = True,
) -> tuple[pd.DataFrame, dict[str, str], dict]:
    """Reconstruct attenuation-scale signals with explicitly supplied firmware sigma.

    Pass the project's existing MA350 ``SIGMA`` mapping. BCc uses the compensated
    channel; BC1/BC2 retain their spot identities. No scattering ``C`` correction
    is applied, and the result must never be labelled independent absorption.
    """
    if bc_unit not in {"ng/m3", "ug/m3"}:
        raise ValueError("bc_unit must explicitly be 'ng/m3' or 'ug/m3'")
    out = minutes.copy()
    signals: dict[str, str] = {}
    conversion = 0.001 if bc_unit == "ng/m3" else 1.0
    sources = {}
    for channel, sigma in cross_sections.items():
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("attenuation cross-sections must be positive and finite")
        for kind in ("BCc", "BC1", "BC2") if include_spots else ("BCc",):
            source = f"{channel} {kind}"
            if source not in minutes:
                continue
            label = channel if kind == "BCc" else f"{channel}_spot{kind[-1]}"
            column = f"{label} b_ATN (Mm-1)"
            out[column] = pd.to_numeric(minutes[source], errors="coerce") * conversion * sigma
            signals[label] = column
            sources[label] = {"input_column": source, "sigma_m2_g": float(sigma)}
    if not signals:
        raise ValueError("no MA350 BCc/BC1/BC2 channels found")
    return out, signals, {
        "signal_quantity": "b_atn", "signal_unit": "Mm-1", "input_bc_unit": bc_unit,
        "scattering_correction_applied": False, "sources": sources,
        "interpretation": "Attenuation-scale instrument signals; not independent b_abs or EC.",
    }


def _prepare_minutes(
    minutes: pd.DataFrame, signal_columns: Mapping[str, str], timezone: str,
    config: HumidityConfig,
) -> pd.DataFrame:
    """Keep all input rows, flag unusable rows, and make contiguous UTC segments."""
    if not minutes.columns.is_unique:
        raise ValueError("minute input column names must be unique")
    required = {config.time_col, config.rh_col, config.session_col, config.tape_col}
    required.update(signal_columns.values())
    missing = sorted(required.difference(minutes.columns))
    if missing:
        raise ValueError(f"minute input lacks {missing}")
    if not signal_columns or len(set(signal_columns.values())) != len(signal_columns):
        raise ValueError("signal_columns must map distinct labels to distinct columns")
    if config.reference_signal not in signal_columns:
        raise ValueError(f"reference signal {config.reference_signal!r} is missing")
    out = minutes.copy().reset_index(drop=True)
    out["_input_row"] = np.arange(len(out))
    out["_utc"] = pd.to_datetime(out[config.time_col], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    out = out.sort_values("_utc", kind="stable", na_position="last").reset_index(drop=True)
    out["_rh"] = pd.to_numeric(out[config.rh_col], errors="coerce")
    out["_local_day"] = out._utc.dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
    local = out._utc.dt.tz_convert(timezone)
    out["_hour"] = local.dt.hour + local.dt.minute / 60
    out["_month"] = local.dt.month
    out["_hour_sin"] = np.sin(2 * np.pi * out._hour / 24)
    out["_hour_cos"] = np.cos(2 * np.pi * out._hour / 24)
    reasons = np.full(len(out), "", dtype=object)

    def flag(mask, reason):
        nonlocal reasons
        use = np.asarray(mask, bool)
        reasons[use] = np.where(reasons[use] == "", reason, reasons[use] + ";" + reason)

    flag(out._utc.isna(), "invalid_utc_timestamp")
    flag(out._utc.notna() & out._utc.duplicated(keep=False), "duplicate_utc_timestamp")
    flag(~np.isfinite(out._rh) | ~out._rh.between(0, 100), "invalid_rh")
    for name in (config.session_col, config.tape_col):
        flag(out[name].isna() | out[name].astype(str).str.strip().eq(""), f"missing_{name}")
    if "is_excluded" in out:
        excluded = out.is_excluded.fillna(False)
        if not pd.api.types.is_bool_dtype(excluded):
            if not excluded.isin([0, 1, True, False]).all():
                raise ValueError("is_excluded must contain boolean or 0/1 flags")
            excluded = excluded.astype(bool)
        flag(excluded, "upstream_exclusion")
        if "exclusion_reason" in out:
            details = out.exclusion_reason.fillna("").astype(str)
            for i in np.flatnonzero(excluded.to_numpy()):
                if details.iloc[i]:
                    reasons[i] += ":" + details.iloc[i]
    out["_reason"] = reasons
    out["_eligible"] = reasons == ""
    dt = out._utc.diff().dt.total_seconds() / 60
    change = (out[config.session_col].ne(out[config.session_col].shift())
              | out[config.tape_col].ne(out[config.tape_col].shift())
              | dt.gt(config.max_gap_minutes) | dt.le(0)
              | ~out._eligible | ~out._eligible.shift(fill_value=False))
    out["_segment"] = change.cumsum().astype(int)
    out["_rh_rate"] = (out._rh.diff() / dt).where(~change)
    for label, column in signal_columns.items():
        values = pd.to_numeric(out[column], errors="coerce")
        # A negative instrument excursion can be the humidity response of
        # interest. Keep finite amplitudes; positivity belongs at ratios/logs.
        out[f"_signal_{label}"] = values.where(np.isfinite(values))
    for source, target in ((config.temp_col, "_temp"), (config.atn_col, "_atn")):
        out[target] = (pd.to_numeric(out[source], errors="coerce")
                       if source in out else np.nan)
        out[target] = out[target].where(np.isfinite(out[target]))
    if config.flow1_col in out and config.flow2_col in out:
        f1 = pd.to_numeric(out[config.flow1_col], errors="coerce")
        f2 = pd.to_numeric(out[config.flow2_col], errors="coerce")
        out["_flow_ratio"] = (f1 / f2).where((f1 > 0) & (f2 > 0) & np.isfinite(f1 / f2))
        out["_flow_total"] = (f1 + f2).where((f1 > 0) & (f2 > 0) & np.isfinite(f1 + f2))
    else:
        out["_flow_ratio"] = np.nan
        out["_flow_total"] = np.nan
    return out


def _time_lag(frame: pd.DataFrame, values: pd.Series, minutes: float,
              tolerance_minutes: float) -> pd.Series:
    """Backward-only as-of lag within segments, measured in elapsed UTC time."""
    result = np.full(len(frame), np.nan)
    lag_ns = int(pd.Timedelta(minutes=minutes).value)
    tolerance_ns = int(pd.Timedelta(minutes=tolerance_minutes).value)
    numeric = np.asarray(values, float)
    for _, group in frame.loc[frame._eligible].groupby("_segment", sort=False):
        indices = group.index.to_numpy()
        stamps = group._utc.astype("int64").to_numpy()
        target = stamps - lag_ns
        pos = np.searchsorted(stamps, target, side="right") - 1
        valid = pos >= 0
        selected = np.clip(pos, 0, max(0, len(stamps) - 1))
        valid &= target - stamps[selected] <= tolerance_ns
        result[indices[valid]] = numeric[indices[selected[valid]]]
    return pd.Series(result, index=frame.index)


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    stamps = frame._utc.array.asi8
    lo, hi = np.searchsorted(stamps, [start.value, end.value])
    return frame.iloc[lo:hi]


def _supported(window: pd.DataFrame, duration: float, config: HumidityConfig,
               column: str = "_rh") -> bool:
    expected = duration / config.cadence_minutes
    observed = window.loc[window[column].notna(), "_utc"].astype("int64")
    # Dense bursts cannot stand in for a window's missing minutes.
    occupied = (observed // int(pd.Timedelta(minutes=config.cadence_minutes).value)).nunique()
    return occupied >= max(1, int(np.ceil(expected * config.min_coverage)))


def _median(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.median()) if len(values) else float("nan")


def _cluster_mean(values, blocks, n_bootstrap: int, seed: int) -> dict:
    data = pd.DataFrame({"value": values, "block": blocks}).dropna()
    day_means = data.groupby("block", sort=True).value.mean().to_numpy()
    if not len(day_means):
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0, "n_blocks": 0}
    ci = (np.nan, np.nan)
    if len(day_means) >= 2:
        rng = np.random.default_rng(seed)
        boot = rng.choice(day_means, (n_bootstrap, len(day_means)), replace=True).mean(axis=1)
        ci = np.quantile(boot, [0.025, 0.975])
    return {"mean": float(day_means.mean()), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "n": len(data), "n_blocks": len(day_means), "weighting": "equal block means"}


def _connected_blocks(day_sets: list[set[str]]) -> list[int]:
    """Link records sharing any local day so a reused day never counts twice."""
    parent = list(range(len(day_sets)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, days in enumerate(day_sets):
        for day in sorted(days):
            if day in seen:
                parent[root(i)] = root(seen[day])
            seen[day] = i
    labels = {}
    return [labels.setdefault(root(i), len(labels)) for i in range(len(day_sets))]


def _forward_evaluate(
    frame: pd.DataFrame, *, target: str, base_features: list[str], extra_features: list[str],
    group_col: str, time_col: str, test_fraction: float, min_train_groups: int,
    min_test_groups: int, ridge_alpha: float, n_bootstrap: int, seed: int,
) -> dict:
    """One fixed chronological holdout, shared complete cases, training-only scaling."""
    columns = [target, *base_features, *extra_features]
    finite = np.isfinite(frame[columns].to_numpy(float)).all(axis=1)
    available = frame.loc[finite & frame[group_col].notna()].copy()
    group_order = (available.groupby(group_col)[time_col].agg(["min", "max"])
                   .sort_values("min", kind="stable"))
    n_test = max(min_test_groups, int(np.ceil(len(group_order) * test_fraction)))
    n_train = len(group_order) - n_test
    summary = {"eligible_rows": len(available), "ineligible_rows": len(frame) - len(available),
               "n_groups": len(group_order), "split": "chronological complete groups",
               "base_features": base_features, "extra_features": extra_features}
    if n_train < min_train_groups:
        return {"summary": {**summary, "status": "insufficient_data",
                             "reason": "too few complete training/evaluation groups"}, "tables": {}}
    train_ids = set(group_order.index[:n_train])
    is_train = available[group_col].isin(train_ids)
    train, test = available.loc[is_train], available.loc[~is_train]
    if train[time_col].max() >= test[time_col].min():
        raise ValueError("chronological groups overlap in time; supply whole nonoverlapping blocks")
    if len(train) < max(10, 2 * len(columns)) or len(test) < 2:
        return {"summary": {**summary, "status": "insufficient_data", "reason": "too few model rows"},
                "tables": {}}
    predictions = test[[time_col, group_col, target]].copy().rename(columns={target: "observed"})
    metrics = []
    coefficients = []
    for name, features in (("instantaneous", base_features), ("with_history", base_features + extra_features)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        model.fit(train[features].to_numpy(float), train[target].to_numpy(float))
        predicted = model.predict(test[features].to_numpy(float))
        predictions[name] = predicted
        metrics.append({"model": name, "n_train": len(train), "n_test": len(test),
                        "n_train_groups": n_train, "n_test_groups": n_test,
                        **regression_scores(test[target], predicted)})
        scaler, reg = model.steps[0][1], model.steps[1][1]
        original_coef = reg.coef_ / scaler.scale_
        coefficients.extend({"model": name, "feature": feature, "coefficient": float(value)}
                            for feature, value in zip(features, original_coef))
        coefficients.append({"model": name, "feature": "intercept",
                             "coefficient": float(reg.intercept_ - original_coef @ scaler.mean_)})
    daily = []
    for group, rows in predictions.groupby(group_col, sort=True):
        a, p, h = (rows[name].to_numpy(float) for name in ("observed", "instantaneous", "with_history"))
        daily.append({"group": group, "n": len(rows), "sse_instantaneous": float(((p - a) ** 2).sum()),
                      "sse_with_history": float(((h - a) ** 2).sum()),
                      "rmse_instantaneous": float(np.sqrt(np.mean((p - a) ** 2))),
                      "rmse_with_history": float(np.sqrt(np.mean((h - a) ** 2)))})
    daily = pd.DataFrame(daily)
    rng = np.random.default_rng(seed)
    gain = metrics[0]["rmse"] - metrics[1]["rmse"]
    ci = (np.nan, np.nan)
    if len(daily) >= 2:
        sampled = rng.integers(len(daily), size=(n_bootstrap, len(daily)))
        n = daily.n.to_numpy()[sampled].sum(axis=1)
        first = daily.sse_instantaneous.to_numpy()[sampled].sum(axis=1)
        second = daily.sse_with_history.to_numpy()[sampled].sum(axis=1)
        ci = np.quantile(np.sqrt(first / n) - np.sqrt(second / n), [0.025, 0.975])
    folds = group_order.reset_index().rename(columns={"min": "first_utc", "max": "last_utc"})
    folds["partition"] = np.where(folds[group_col].isin(train_ids), "training", "evaluation")
    return {"summary": {**summary, "status": "completed", "n_train": len(train), "n_test": len(test),
                         "rmse_gain": float(gain), "rmse_gain_ci_low": float(ci[0]),
                         "rmse_gain_ci_high": float(ci[1]), "uncertainty_unit": group_col,
                         "test_start_utc": test[time_col].min().isoformat(),
                         "hyperparameters": {"ridge_alpha": ridge_alpha, "selection": "fixed before evaluation"}},
            "tables": {"metrics": pd.DataFrame(metrics), "predictions": predictions,
                       "group_errors": daily, "folds": folds, "coefficients": pd.DataFrame(coefficients)}}


def _humidity_events(frame: pd.DataFrame, config: HumidityConfig) -> pd.DataFrame:
    previous_rh = _time_lag(frame, frame._rh, config.step_minutes, config.lag_tolerance_minutes)
    delta = frame._rh - previous_rh
    direction = np.where(delta >= config.rh_change_pp, 1,
                         np.where(delta <= -config.rh_change_pp, -1, 0))
    direction[~frame._eligible.to_numpy()] = 0
    runs = (pd.Series(direction).ne(pd.Series(direction).shift())
            | frame._segment.ne(frame._segment.shift())).cumsum()
    records = []
    last_end = None
    eligible = frame.loc[frame._eligible]
    segments = {key: group for key, group in eligible.groupby("_segment", sort=False)}
    for _, candidates in frame.loc[direction != 0].groupby(runs, sort=False):
        last = None
        attempts = 0
        for i, row in candidates.iterrows():
            attempts += 1
            at = row._utc
            group = segments[row._segment]
            pre_start = at - pd.Timedelta(minutes=config.step_minutes + config.baseline_minutes)
            pre_end = at - pd.Timedelta(minutes=config.step_minutes)
            end = at + pd.Timedelta(minutes=config.recovery_minutes)
            reason = []
            if last_end is not None and pre_start < last_end:
                reason.append("overlaps_previous_event_window")
            pre = _slice(group, pre_start, pre_end)
            post = _slice(group, at, at + pd.Timedelta(minutes=config.sustain_minutes))
            window = _slice(group, pre_start, end)
            if (group._utc.iloc[0] > pre_start + pd.Timedelta(minutes=config.lag_tolerance_minutes)
                    or group._utc.iloc[-1] < end - pd.Timedelta(minutes=config.cadence_minutes + config.lag_tolerance_minutes)
                    or not _supported(window, config.step_minutes + config.baseline_minutes + config.recovery_minutes, config)):
                reason.append("incomplete_contiguous_event_window")
            if not _supported(pre, config.baseline_minutes, config) or not _supported(post, config.sustain_minutes, config):
                reason.append("insufficient_rh_baseline_or_sustain")
            before, after = _median(pre._rh), _median(post._rh)
            change = after - before
            if (not np.isfinite(change) or np.sign(change) != direction[i]
                    or abs(change) < config.rh_change_pp):
                reason.append("rh_change_not_sustained")
            if (pre._rh.max() - pre._rh.min() > config.stable_plateau_range_pp
                    or post._rh.max() - post._rh.min() > config.stable_plateau_range_pp):
                reason.append("rh_plateau_unstable")
            late = _slice(group, at + pd.Timedelta(minutes=config.recovery_minutes / 2), end)
            stable_late = late.loc[late._rh_rate.abs() <= config.stability_pp_per_min]
            late_support = _supported(stable_late, config.recovery_minutes / 2, config)
            last = {"event_id": f"RH-{len(records) + 1:06d}", "event_utc": at,
                    "window_start_utc": pre_start, "window_end_utc": end,
                    "segment": int(row._segment), "direction": "rising" if direction[i] > 0 else "falling",
                    "rh_before": before, "rh_after": after, "rh_change_pp": change,
                    "local_day": row._local_day, "hour": row._hour, "month": row._month,
                    "temp": _median(post._temp), "atn": _median(post._atn),
                    "flow_ratio": _median(post._flow_ratio),
                    "flow_total": _median(post._flow_total),
                    "late_stable_rh": _median(stable_late._rh) if late_support else np.nan,
                    "late_stable_temp": _median(stable_late._temp) if late_support else np.nan,
                    "late_stable_atn": _median(stable_late._atn) if late_support else np.nan,
                    "late_stable_flow_ratio": _median(stable_late._flow_ratio) if late_support else np.nan,
                    "late_stable_flow_total": _median(stable_late._flow_total) if late_support else np.nan,
                    "reference_before": _median(pre[f"_signal_{config.reference_signal}"]),
                    "is_excluded": bool(reason), "exclusion_reason": ";".join(reason)}
            if not reason:
                last_end = max(end, at + pd.Timedelta(minutes=config.min_event_separation_minutes))
                break
            if "overlaps_previous_event_window" in reason:
                break
        if last is not None:
            last["candidate_endpoints_checked"] = attempts
            records.append(last)
    return pd.DataFrame(records)


def _event_curves(frame: pd.DataFrame, events: pd.DataFrame, signals: Mapping[str, str],
                  config: HumidityConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves, effects = [], []
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    segments = {key: group for key, group in frame.loc[frame._eligible].groupby("_segment", sort=False)}
    for _, event in events.loc[~events.is_excluded].iterrows():
        group = segments[event.segment]
        at = event.event_utc
        baseline = _slice(group, event.window_start_utc, at - pd.Timedelta(minutes=config.step_minutes))
        post = _slice(group, at, event.window_end_utc)
        late = _slice(group, at + pd.Timedelta(minutes=config.recovery_minutes / 2), event.window_end_utc)
        stable_late = late.loc[late._rh_rate.abs() <= config.stability_pp_per_min]
        for label in signals:
            column = f"_signal_{label}"
            supported_baseline = _supported(baseline, config.baseline_minutes, config, column)
            before = baseline[column].mean() if supported_baseline else np.nan
            reason = "" if supported_baseline else "insufficient_signal_baseline"
            late_ok = _supported(late, config.recovery_minutes / 2, config, column)
            stable_ok = _supported(stable_late, config.recovery_minutes / 2, config, column)
            if not late_ok:
                reason += (";" if reason else "") + "insufficient_late_signal"
            effects.append({"event_id": event.event_id, "local_day": event.local_day,
                            "direction": event.direction, "signal": label, "baseline": before,
                            "post_mean_delta": post[column].mean() - before,
                            "late_mean_delta": late[column].mean() - before if late_ok else np.nan,
                            "late_stable_mean_delta": stable_late[column].mean() - before if stable_ok else np.nan,
                            "late_stable_n": int(stable_late[column].notna().sum()),
                            "late_stable_supported": bool(stable_ok and supported_baseline),
                            "is_excluded": bool(reason), "exclusion_reason": reason})
            edges = np.arange(-config.step_minutes - config.baseline_minutes,
                              config.recovery_minutes, config.bin_minutes)
            for minute in edges:
                duration = min(config.bin_minutes, config.recovery_minutes - minute)
                window = _slice(group, at + pd.Timedelta(minutes=float(minute)),
                                at + pd.Timedelta(minutes=float(minute + duration)))
                valid = _supported(window, duration, config, column) and supported_baseline
                curves.append({"event_id": event.event_id, "local_day": event.local_day,
                               "direction": event.direction, "signal": label,
                               "minutes_from_threshold": float(minute), "bin_minutes": duration,
                               "mean_signal": window[column].mean() if valid else np.nan,
                               "delta_from_baseline": window[column].mean() - before if valid else np.nan,
                               "mean_rh": window._rh.mean(), "n": int(window[column].notna().sum()),
                               "supported": bool(valid)})
    return pd.DataFrame(curves), pd.DataFrame(effects)


def _match_events(events: pd.DataFrame, effects: pd.DataFrame, config: HumidityConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty or effects.empty:
        return pd.DataFrame(), pd.DataFrame()
    accepted = events.loc[~events.is_excluded].copy()
    requirements = ["rh_after", "late_stable_rh", "late_stable_temp", "late_stable_atn",
                    "late_stable_flow_ratio", "late_stable_flow_total", "reference_before", "hour", "month"]
    accepted["match_exclusion_reason"] = ""
    complete = np.isfinite(accepted[requirements].to_numpy(float)).all(axis=1) & accepted.reference_before.gt(0)
    accepted.loc[~complete, "match_exclusion_reason"] = "missing_matching_covariates"
    fall = accepted.loc[complete & accepted.direction.eq("falling")]
    available = set(fall.index)
    pairs = []
    for i, rise in accepted.loc[complete & accepted.direction.eq("rising")].iterrows():
        candidates = fall.loc[fall.index.isin(available)].copy()
        hour_difference = (candidates.hour - rise.hour).abs()
        distance = pd.DataFrame({
            "rh": (candidates.rh_after - rise.rh_after).abs() / config.rh_match_pp,
            "late_rh": (candidates.late_stable_rh - rise.late_stable_rh).abs() / config.rh_match_pp,
            "temp": (candidates.late_stable_temp - rise.late_stable_temp).abs() / config.temp_match_c,
            "atn": (candidates.late_stable_atn - rise.late_stable_atn).abs() / config.atn_match,
            "flow": (candidates.late_stable_flow_ratio - rise.late_stable_flow_ratio).abs() / config.flow_ratio_match,
            "total_flow": (candidates.late_stable_flow_total - rise.late_stable_flow_total).abs()
                          / (config.total_flow_relative_match * rise.late_stable_flow_total),
            "reference": (candidates.reference_before - rise.reference_before).abs()
                         / (config.reference_relative_match * rise.reference_before),
            "hour": np.minimum(hour_difference, 24 - hour_difference) / config.hour_match,
        })
        qualified = distance.le(1).all(axis=1) & candidates.month.eq(rise.month)
        if not qualified.any():
            accepted.loc[i, "match_exclusion_reason"] = "no_opposite_direction_event_within_calipers"
            continue
        chosen = distance.loc[qualified].pow(2).sum(axis=1).idxmin()
        other = candidates.loc[chosen]
        available.remove(chosen)
        pairs.append({"pair_id": f"PAIR-{len(pairs) + 1:05d}", "rising_event": rise.event_id,
                      "falling_event": other.event_id, "rising_day": rise.local_day,
                      "falling_day": other.local_day, "rh_difference_pp": rise.rh_after - other.rh_after,
                      "late_stable_rh_difference_pp": rise.late_stable_rh - other.late_stable_rh,
                      "late_stable_total_flow_ratio": rise.late_stable_flow_total / other.late_stable_flow_total,
                      "normalized_matching_distance": float(distance.loc[chosen].pow(2).sum())})
    accepted.loc[list(available), "match_exclusion_reason"] = "unmatched_falling_event"
    if not pairs:
        return pd.DataFrame(), accepted[["event_id", "direction", "match_exclusion_reason"]]
    blocks = _connected_blocks([{p["rising_day"], p["falling_day"]} for p in pairs])
    rows = []
    for pair, block in zip(pairs, blocks):
        first = effects.loc[effects.event_id.eq(pair["rising_event"])].set_index("signal")
        second = effects.loc[effects.event_id.eq(pair["falling_event"])].set_index("signal")
        for signal in sorted(set(first.index) & set(second.index)):
            a, b = first.loc[signal], second.loc[signal]
            rows.append({**pair, "day_block": block, "signal": signal,
                         "rising_late_stable_delta": a.late_stable_mean_delta,
                         "falling_late_stable_delta": b.late_stable_mean_delta,
                         "rising_minus_falling": a.late_stable_mean_delta - b.late_stable_mean_delta})
    return pd.DataFrame(rows), accepted[["event_id", "direction", "match_exclusion_reason"]]


def _filter_window_sensitivity(
    frame: pd.DataFrame, events: pd.DataFrame, signals: Mapping[str, str], config: HumidityConfig,
    *, active_intervals: pd.DataFrame | None, filter_catalog: pd.DataFrame | None,
    stream_site: str | None, processing_evidence: Mapping | None,
) -> dict:
    """Observed-slot selection sensitivity on independently documented pump periods."""
    base = {"status": "blocked", "interpretation": "All observed versus outside-event selection sensitivity; not corrected bias."}
    if active_intervals is None or active_intervals.empty or filter_catalog is None:
        return {"summary": {**base, "reason": "verified active intervals and their parent filter catalog are required"}, "tables": {}}
    if not stream_site:
        return {"summary": {**base, "reason": "stream_site is required; site cannot be inferred from timestamps"}, "tables": {}}
    evidence = dict(processing_evidence or {})
    checks = ("cadence_verified", "timestamp_role_verified", "observation_provenance_verified", "processing_history_verified")
    missing = [name for name in checks if evidence.get(name) is not True]
    missing += [name for name in ("stream_id", "evidence", "observed_column", "valid_column", "slot_origin_utc")
                if not isinstance(evidence.get(name), str) or not evidence[name].strip()]
    if evidence.get("evidence_scope") not in {"full_export", "bounded_period"}:
        missing.append("reviewed_evidence_scope")
    if missing:
        return {"summary": {**base, "reason": "observation/processing evidence is incomplete",
                             "missing_evidence": sorted(missing)}, "tables": {}}
    observed_col, valid_col = evidence["observed_column"], evidence["valid_column"]
    if observed_col not in frame or valid_col not in frame:
        return {"summary": {**base, "reason": "declared per-minute observation/validity flags are missing"}, "tables": {}}
    if not pd.api.types.is_bool_dtype(frame[observed_col]) or not pd.api.types.is_bool_dtype(frame[valid_col]):
        raise ValueError("per-minute observed and valid flags must be boolean, including nullable booleans")
    intervals = validate_intervals(active_intervals, filter_catalog, verify_source_files=True)
    other_sites = int(intervals.site.ne(stream_site).sum())
    intervals = intervals.loc[intervals.site.eq(stream_site)].copy()
    if intervals.empty:
        return {"summary": {**base, "reason": "no verified intervals for the declared stream site",
                             "other_site_intervals": other_sites}, "tables": {}}
    # This contract is used only for native input units. In particular, do not
    # permit the eBC-to-absorption branch to reconvert an already converted signal.
    contract = StreamContract(
        stream_id=evidence["stream_id"], cadence_seconds=config.cadence_minutes * 60,
        slot_origin_utc=evidence["slot_origin_utc"], cadence_verified=True,
        timestamp_role_verified=True, observation_provenance_verified=True,
        processing_history_verified=True, units="native_optical_signal", units_verified=False,
        evidence_scope=evidence["evidence_scope"], scope_justification=evidence["evidence"],
        scope_start_utc=evidence.get("scope_start_utc", ""), scope_end_utc=evidence.get("scope_end_utc", ""),
        scope_session_ids_json=evidence.get("scope_session_ids_json", "[]"),
    )
    observed = frame.loc[frame._eligible].set_index("_utc", drop=False).copy()
    observed["_in_recovery"] = False
    if not events.empty:
        for event in events.loc[~events.is_excluded].itertuples():
            observed.loc[(observed.index >= event.event_utc) & (observed.index < event.window_end_utc), "_in_recovery"] = True
    observed["_outside_recovery_valid"] = observed[valid_col] & ~observed._in_recovery
    summaries, segment_tables = [], []
    catalog = filter_catalog.set_index(["site", "base_filter_id"])
    if catalog.index.has_duplicates:
        raise ValueError("parent filter catalog identities must be unique")
    for (site, identifier), parts in intervals.groupby(["site", "base_filter_id"], sort=True):
        registered = catalog.loc[(site, identifier)]
        for label in signals:
            segments, all_minutes = summarize_active_segments(
                observed, parts, f"_signal_{label}", contract, observed_column=observed_col, valid_column=valid_col)
            _, outside = summarize_active_segments(
                observed, parts, f"_signal_{label}", contract, observed_column=observed_col,
                valid_column="_outside_recovery_valid")
            reason = []
            registered_exclusion = registered.get("is_excluded", False)
            if pd.isna(registered_exclusion):
                reason.append("registered_filter_exclusion_unknown")
            elif not isinstance(registered_exclusion, (bool, np.bool_)):
                raise ValueError("registered filter is_excluded must be boolean or explicitly unknown")
            elif registered_exclusion:
                reason.append("registered_filter_exclusion:" + str(registered.get("exclusion_reason", "unspecified")))
            if all_minutes["processing_evidence_scope_status"] != "verified":
                reason.append("processing_evidence_outside_scope")
            coverage = all_minutes["valid_observed_coverage_fraction"]
            if (not np.isfinite(coverage) or coverage < config.min_coverage
                    or all_minutes["minimum_segment_valid_observed_coverage"] < config.min_coverage):
                reason.append("insufficient_valid_observed_coverage")
            n_all, n_out = all_minutes["valid_observed_slots"], outside["valid_observed_slots"]
            a, b = all_minutes["mean_valid_observed_native"], outside["mean_valid_observed_native"]
            summaries.append({"site": site, "base_filter_id": identifier, "signal": label,
                              "expected_slots": all_minutes["expected_slots"], "valid_observed_slots": n_all,
                              "outside_recovery_slots": n_out, "coverage": coverage,
                              "recovery_slot_fraction": (n_all - n_out) / n_all if n_all > 0 else np.nan,
                              "mean_all_observed": a if not reason else np.nan,
                              "mean_outside_recovery": b if not reason else np.nan,
                              "all_minus_outside_mean": a - b if not reason else np.nan,
                              "is_excluded": bool(reason), "exclusion_reason": ";".join(reason),
                              "quantity": "declared optical signal in Mm-1; not corrected EC"})
            segment_tables.append(segments.assign(site=site, base_filter_id=identifier, signal=label))
    ledger = pd.DataFrame(summaries)
    return {"summary": {**base, "status": "completed" if (~ledger.is_excluded).any() else "insufficient_data",
                         "stream_site": stream_site, "n_physical_filters": ledger.base_filter_id.nunique(),
                         "other_site_intervals": other_sites, "processing_evidence": evidence,
                         "interval_source_files_verified": True,
                         "minute_screen": "Analysis-eligible rows with valid RH, unique UTC, session and tape identity; declared observed and valid flags also required."},
            "tables": {"filter_window_sensitivity": ledger,
                       "filter_interval_coverage": pd.concat(segment_tables, ignore_index=True),
                       "verified_active_intervals": intervals}}


def run_humidity_memory(
    minutes: pd.DataFrame, *, signal_columns: Mapping[str, str], signal_quantity: str,
    timezone: str, rh_provenance: str, config: HumidityConfig | None = None,
    active_intervals: pd.DataFrame | None = None, filter_catalog: pd.DataFrame | None = None,
    stream_site: str | None = None, processing_evidence: Mapping | None = None,
) -> dict:
    """Return event curves, matched contrasts, blocked prediction scores and audits.

    ``signal_quantity`` must be ``b_atn`` or ``b_abs``. ``rh_provenance`` identifies
    where RH was measured (e.g. sample stream, chamber, or ambient station).
    Recorded chamber RH does not automatically establish sample/filter RH.
    The reported tests describe this declared measurement; no correction is fit.
    """
    config = config or HumidityConfig()
    _validate_config(config)
    if signal_quantity not in {"b_atn", "b_abs"}:
        raise ValueError("signal_quantity must be explicit 'b_atn' or 'b_abs'; BCc is not absorption")
    if not isinstance(rh_provenance, str) or not rh_provenance.strip():
        raise ValueError("rh_provenance must describe the measured RH source/location")
    frame = _prepare_minutes(minutes, signal_columns, timezone, config)
    summary = {
        "status": "completed", "n_input": len(frame), "n_eligible": int(frame._eligible.sum()),
        "n_excluded": int((~frame._eligible).sum()), "signal_quantity": signal_quantity,
        "signal_unit": "Mm-1", "timestamp_basis": "UTC", "day_timezone": timezone,
        "rh_provenance": rh_provenance, "parameters": asdict(config),
        "limitations": [
            "An association can reflect changing aerosol or air masses as well as instrument response.",
            "No humidity correction or MA200 correction coefficients are applied.",
            "Late stable means use the declared RH-rate threshold, not the gallery's complete clean mask.",
            "An attenuation-scale result is not independent absorption or a carbon mass reference.",
            "Event/paired intervals use day blocks; model intervals resample complete evaluation days.",
            "Spot-flow ratio and the sum of the two measured spot flows are controlled; unmeasured flow disturbances remain possible.",
        ],
    }
    exclusions = frame.loc[~frame._eligible, ["_input_row", "_utc", "_reason"]].rename(
        columns={"_input_row": "input_row", "_utc": "time_utc", "_reason": "exclusion_reason"})
    signal_audit = pd.DataFrame([
        {"signal": label, "input_column": column, "n_eligible_minutes": int(frame._eligible.sum()),
         "n_nonfinite": int((frame._eligible & frame[f"_signal_{label}"].isna()).sum()),
         "n_nonpositive_retained": int((frame._eligible & frame[f"_signal_{label}"].le(0)).sum())}
        for label, column in signal_columns.items()
    ])
    tables = {"exclusions": exclusions, "signal_support": signal_audit}
    if not frame._eligible.any():
        return {"summary": {**summary, "status": "insufficient_data", "reason": "no eligible minute rows"},
                "tables": tables}
    events = _humidity_events(frame, config)
    curves, effects = _event_curves(frame, events, signal_columns, config)
    pairs, match_audit = _match_events(events, effects, config)
    tables.update(events=events, event_curves=curves, event_effects=effects,
                  matched_pairs=pairs, matching_audit=match_audit)
    effect_summary = []
    if not effects.empty:
        for (signal, direction), group in effects.groupby(["signal", "direction"]):
            effect_summary.append({"signal": signal, "direction": direction,
                                   **_cluster_mean(group.late_stable_mean_delta, group.local_day,
                                                   config.n_bootstrap, config.seed)})
    tables["event_summary"] = pd.DataFrame(effect_summary)
    paired_summary = []
    if not pairs.empty:
        for signal, group in pairs.groupby("signal"):
            paired_summary.append({"signal": signal,
                                   **_cluster_mean(group.rising_minus_falling, group.day_block,
                                                   config.n_bootstrap, config.seed)})
    tables["matched_summary"] = pd.DataFrame(paired_summary)
    model = frame.loc[frame._eligible].copy()
    for lag in config.lag_minutes:
        feature = f"_rh_lag_{lag:g}min"
        frame[feature] = _time_lag(frame, frame._rh, lag, config.lag_tolerance_minutes)
        model[feature] = frame.loc[model.index, feature]
    history = [f"_rh_lag_{lag:g}min" for lag in config.lag_minutes]
    reference = _time_lag(frame, frame[f"_signal_{config.reference_signal}"],
                          max(config.lag_minutes), config.lag_tolerance_minutes)
    model["_reference_history"] = reference.loc[model.index]
    bucket = (model._utc.astype("int64") // int(pd.Timedelta(minutes=config.model_stride_minutes).value))
    model = model.loc[~pd.DataFrame({"segment": model._segment, "bucket": bucket}).duplicated()].copy()
    controls = ["_rh", "_temp", "_atn", "_flow_ratio", "_flow_total", "_reference_history", "_hour_sin", "_hour_cos"]
    models = {}
    for label in signal_columns:
        result = _forward_evaluate(
            model, target=f"_signal_{label}", base_features=controls, extra_features=history,
            group_col="_local_day", time_col="_utc", test_fraction=config.test_fraction,
            min_train_groups=config.min_train_days, min_test_groups=config.min_test_days,
            ridge_alpha=config.ridge_alpha, n_bootstrap=config.n_bootstrap, seed=config.seed,
        )
        models[label] = result["summary"]
        for name, table in result["tables"].items():
            key = f"model_{name}"
            table = table.assign(signal=label)
            tables[key] = pd.concat([tables[key], table], ignore_index=True) if key in tables else table
    summary["models"] = models
    summary["n_event_candidates"] = len(events)
    summary["n_events"] = int((~events.is_excluded).sum()) if len(events) else 0
    summary["n_matched_pairs"] = int(pairs.pair_id.nunique()) if len(pairs) else 0
    summary["n_segments"] = int(frame.loc[frame._eligible, "_segment"].nunique())
    summary["eligible_utc_bounds_inclusive"] = [frame.loc[frame._eligible, "_utc"].min().isoformat(),
                                                frame.loc[frame._eligible, "_utc"].max().isoformat()]
    summary["recorded_rh_range_pp"] = [float(frame.loc[frame._eligible, "_rh"].min()),
                                        float(frame.loc[frame._eligible, "_rh"].max())]
    step_change = frame._rh - _time_lag(frame, frame._rh, config.step_minutes, config.lag_tolerance_minutes)
    maximum_change = step_change.abs().max()
    summary["maximum_step_rh_change_pp"] = float(maximum_change) if np.isfinite(maximum_change) else None
    summary["event_stage_status"] = "completed" if summary["n_events"] else "insufficient_data"
    summary["amplitude_nonpositive_policy"] = "Finite nonpositive signals retained; matching requires a positive pre-event reference."
    summary["prior_reference_transform"] = "identity; finite negative instrument excursions retained"
    summary["event_matching_state"] = "Post-step and late stable RH; late stable temperature, ATN, spot-flow ratio and total spot flow; pre-event reference amplitude; event hour and calendar month."
    window_result = _filter_window_sensitivity(
        frame, events, signal_columns, config, active_intervals=active_intervals,
        filter_catalog=filter_catalog, stream_site=stream_site, processing_evidence=processing_evidence)
    summary["filter_window_stage"] = window_result["summary"]
    tables.update(window_result["tables"])
    if summary["n_events"] == 0 and not any(m["status"] == "completed" for m in models.values()):
        summary["status"] = "insufficient_data"
    return {"summary": summary, "tables": tables}
