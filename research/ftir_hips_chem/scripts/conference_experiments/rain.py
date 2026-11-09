"""Rain-event history with explicit daily or verified active-interval precision.

Missing precipitation is never interpreted as dry weather. Daily files supply
calendar-day and antecedent-day quantities only. They cannot establish rain
during an unverified filter collection period. Hourly totals are used only for
whole, documented bins; no uniform within-hour rainfall is invented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from active_interval_matching import validate_intervals
except ImportError:
    from ..active_interval_matching import validate_intervals

from .chemistry import (
    SEED, _base_filters, _evaluate_residual_models, _required, _result, _utc,
)


@dataclass(frozen=True)
class RainConfig:
    mode: str = "daily"
    antecedent_days: tuple[int, ...] = (1, 3, 7)
    wet_threshold_mm: float = 1.0
    event_dry_gap_hours: int = 24
    timezone: str = "Africa/Addis_Ababa"
    precipitation_column: str = "rain_mm"
    date_column: str = "date"

    def __post_init__(self):
        if self.mode not in {"daily", "hourly"}:
            raise ValueError("rain mode must be daily or hourly")
        if (not self.antecedent_days or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1 for v in self.antecedent_days)
                or len(set(self.antecedent_days)) != len(self.antecedent_days)):
            raise ValueError("antecedent_days must contain distinct positive integers")
        if not np.isfinite(self.wet_threshold_mm) or self.wet_threshold_mm <= 0:
            raise ValueError("wet_threshold_mm must be positive and prespecified")
        if not isinstance(self.event_dry_gap_hours, (int, np.integer)) or self.event_dry_gap_hours < 0:
            raise ValueError("event_dry_gap_hours must be a nonnegative integer")
        if self.mode == "daily" and self.event_dry_gap_hours % 24:
            raise ValueError("daily precipitation only supports whole-day event gaps")
        ZoneInfo(self.timezone)


def _weather(weather: pd.DataFrame, config: RainConfig) -> pd.DataFrame:
    column = config.precipitation_column
    required = [column, config.date_column] if config.mode == "daily" else [column, "interval_start_utc", "interval_end_utc"]
    _required(weather, required, "weather")
    if weather.empty:
        raise ValueError("weather must contain at least one precipitation record")
    values = pd.to_numeric(weather[column], errors="coerce").to_numpy(float)
    valid = np.isfinite(values) & (values >= 0)
    if config.mode == "daily":
        stamps = weather[config.date_column]
        if any(pd.isna(v) or pd.Timestamp(v).tzinfo is not None for v in stamps):
            raise ValueError("daily weather dates must be complete timezone-naive local calendar dates")
        index = pd.DatetimeIndex(pd.to_datetime(stamps))
        if (index != index.normalize()).any():
            raise ValueError("daily weather must identify calendar days, not ambiguous within-day timestamps")
        frequency = "D"
    else:
        index = _utc(weather.interval_start_utc, "weather interval_start_utc")
        ends = _utc(weather.interval_end_utc, "weather interval_end_utc")
        if not ((ends - index) == pd.Timedelta(hours=1)).all():
            raise ValueError("hourly weather requires explicit one-hour accumulation intervals")
        if (index != index.floor("h")).any():
            raise ValueError("hourly weather bins must start on UTC hour boundaries")
        frequency = "h"
    if index.duplicated().any():
        raise ValueError("duplicate precipitation bins cannot be silently aggregated")
    out = pd.DataFrame({"rain_mm": np.where(valid, values, np.nan),
                        "original_rain_mm": values,
                        "weather_reason": np.where(valid, "", "missing_nonfinite_or_negative_precipitation")}, index=index).sort_index()
    grid = pd.date_range(out.index.min(), out.index.max(), freq=frequency)
    out = out.reindex(grid)
    out["weather_reason"] = out.weather_reason.fillna("missing_precipitation_bin")
    out["valid_precipitation"] = out.rain_mm.notna()
    step_hours = 24 if config.mode == "daily" else 1
    maximum_gap = config.event_dry_gap_hours // step_hours
    previous_wet_position = None
    current_event = None
    dry_start = None
    event_ids, states = [], []
    for i, (stamp, row) in enumerate(out.iterrows()):
        if not row.valid_precipitation:
            previous_wet_position, current_event, dry_start = None, None, None
            event_ids.append("")
            states.append(f"missing:{stamp.isoformat()}")
        elif row.rain_mm >= config.wet_threshold_mm:
            if previous_wet_position is None or i - previous_wet_position > maximum_gap + 1:
                current_event = f"rain:{stamp.isoformat()}"
            previous_wet_position, dry_start = i, None
            event_ids.append(current_event)
            states.append(current_event)
        else:
            if dry_start is None:
                dry_start = stamp
            event_ids.append("")
            states.append(current_event or f"dry:{dry_start.isoformat()}")
    out["rain_event_id"] = event_ids
    out["rain_state_id"] = states
    out.index.name = "weather_date" if config.mode == "daily" else "interval_start_utc"
    return out


def _sum_bins(series: pd.Series, index: pd.DatetimeIndex) -> dict:
    values = series.reindex(index)
    finite = np.isfinite(values.to_numpy(float))
    coverage = float(finite.mean()) if len(values) else 0.0
    observed_sum = float(values.loc[finite].sum()) if finite.any() else np.nan
    return {"value": observed_sum if len(values) and finite.all() else np.nan,
            "coverage": coverage, "observed_sum": observed_sum,
            "expected_bins": len(values), "valid_bins": int(finite.sum())}


def _since_wet(weather, anchor, *, daily):
    step = pd.Timedelta(days=1) if daily else pd.Timedelta(hours=1)
    prior = weather.loc[weather.index + step <= anchor]
    wet = prior.rain_mm.ge(0) & prior.rain_event_id.ne("")
    if wet.any():
        last = prior.index[wet][-1]
        # Extend the check to the anchor, including bins absent beyond the file.
        after_index = pd.date_range(last + step, anchor, freq="D" if daily else "h", inclusive="left")
        after_index = after_index[after_index + step <= anchor]
        after = prior.reindex(after_index)
        if after.valid_precipitation.astype("boolean").fillna(False).all():
            # Hourly rainfall occurred somewhere in [last,last+1h): give bounds.
            distance = (anchor - last).total_seconds() / 3600
            return ((distance / 24 if daily else max(0.0, distance - 1)),
                    distance / 24 if daily else distance, False)
    known_dry = 0
    if prior.empty or prior.index[-1] + step != anchor:
        return (0.0, np.nan, True)
    for row in prior.iloc[::-1].itertuples():
        if not row.valid_precipitation or row.rain_event_id:
            break
        known_dry += 1
    return (float(known_dry), np.nan, True)


def _connected_blocks(rows):
    """Keep shared rain episodes together, including filters spanning episodes."""
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, row in enumerate(rows):
        keys = row.get("rain_event_ids", []) or [row["rain_state_id"]]
        for event in keys:
            if event in owner:
                parent[root(i)] = root(owner[event])
            else:
                owner[event] = i
    names = {}
    for i, row in enumerate(rows):
        key = root(i)
        names[key] = min(names.get(key, row["physical_id"]), row["physical_id"])
    return [f"rain_block:{names[root(i)]}" for i in range(len(rows))]


def _daily_features(filters, weather, config):
    rows = []
    for item in filters.itertuples():
        day = item.date
        row = {"physical_id": item.physical_id, "rain_precision": "local_calendar_day",
               "rain_during_collection_mm": np.nan,
               "collection_rain_status": "unavailable_from_daily_sample_date",
               "rain_event_ids": [], "rain_state_id": f"unobserved:{item.physical_id}"}
        if pd.isna(day):
            rows.append(row)
            continue
        for length in config.antecedent_days:
            index = pd.date_range(day - pd.Timedelta(days=int(length)), periods=int(length), freq="D")
            result = _sum_bins(weather.rain_mm, index)
            row.update({f"rain_before_{length}d_mm": result["value"],
                        f"rain_before_{length}d_coverage": result["coverage"],
                        f"rain_before_{length}d_observed_mm": result["observed_sum"]})
        sample = _sum_bins(weather.rain_mm, pd.DatetimeIndex([day]))
        row.update(sample_day_rain_mm=sample["value"], sample_day_rain_coverage=sample["coverage"])
        low, high, censored = _since_wet(weather, day, daily=True)
        row.update(time_since_rain_days=high, dry_history_lower_bound_days=low,
                   time_since_rain_censored=censored)
        support = weather.loc[(weather.index >= day - pd.Timedelta(days=max(config.antecedent_days))) & (weather.index <= day)]
        row["rain_event_ids"] = sorted(set(support.rain_event_id) - {""})
        if day in weather.index:
            row["rain_state_id"] = weather.at[day, "rain_state_id"]
        rows.append(row)
    return rows


def _hourly_features(filters, weather, intervals, config):
    rows = []
    for item in filters.itertuples():
        parts = intervals.loc[intervals.base_filter_id.eq(item.physical_id)]
        row = {"physical_id": item.physical_id, "rain_precision": "verified_active_intervals_hourly",
               "rain_event_ids": [], "rain_state_id": f"unobserved:{item.physical_id}",
               "collection_rain_status": "missing_verified_active_intervals"}
        if parts.empty:
            rows.append(row)
            continue
        anchor = parts.active_start_utc.min()
        row["window_start_utc"] = anchor
        row["window_end_utc"] = parts.active_end_utc.max()
        row["active_interval_count"] = len(parts)
        row["active_hours"] = float((parts.active_end_utc - parts.active_start_utc).dt.total_seconds().sum() / 3600)
        aligned = all(stamp == stamp.floor("h") for stamp in [*parts.active_start_utc, *parts.active_end_utc])
        active_bins = []
        for part in parts.itertuples():
            active_bins.extend(pd.date_range(part.active_start_utc.ceil("h"), part.active_end_utc.floor("h"), freq="h", inclusive="left"))
        active_bins = pd.DatetimeIndex(active_bins)
        result = _sum_bins(weather.rain_mm, active_bins)
        row["rain_during_collection_mm"] = result["value"] if aligned else np.nan
        row["rain_during_collection_observed_mm"] = result["observed_sum"]
        row["rain_during_collection_coverage"] = result["valid_bins"] / row["active_hours"] if row["active_hours"] else 0.0
        row["collection_rain_status"] = ("complete_hourly_bins" if aligned and np.isfinite(result["value"])
                                         else "partial_hour_bounds" if not aligned else "missing_hourly_precipitation")
        for length in config.antecedent_days:
            index = pd.date_range(anchor - pd.Timedelta(days=int(length)), anchor, freq="h", inclusive="left")
            result = _sum_bins(weather.rain_mm, index)
            row.update({f"rain_before_{length}d_mm": result["value"] if anchor == anchor.floor("h") else np.nan,
                        f"rain_before_{length}d_coverage": result["coverage"],
                        f"rain_before_{length}d_observed_mm": result["observed_sum"]})
        low, high, censored = _since_wet(weather, anchor, daily=False)
        row.update(time_since_rain_hours_lower=low, time_since_rain_hours_upper=high,
                   time_since_rain_censored=censored)
        antecedent = weather.loc[(weather.index >= anchor - pd.Timedelta(days=max(config.antecedent_days))) & (weather.index < anchor)]
        active = weather.reindex(active_bins)
        row["rain_event_ids"] = sorted((set(antecedent.rain_event_id.dropna()) | set(active.rain_event_id.dropna())) - {""})
        if anchor in weather.index:
            row["rain_state_id"] = weather.at[anchor, "rain_state_id"]
        rows.append(row)
    return rows


def build_rain_history(
    filters: pd.DataFrame, weather: pd.DataFrame, *, config: RainConfig | None = None,
    active_intervals: pd.DataFrame | None = None, site_name="Addis Ababa",
    calendar="belg_feb", verify_interval_sources=True,
) -> dict:
    """Build causally ordered rainfall features and retain support/evidence audit.

    Daily mode expects local ``date`` / ``rain_mm`` (names configurable). Hourly
    mode expects ``interval_start_utc``, ``interval_end_utc``, ``rain_mm`` and the
    existing evidence-backed active-interval schema. Daily collection rain stays
    unavailable, even if sample dates are conventionally labelled "24 h".
    """
    config = config or RainConfig()
    audit = _base_filters(filters, site_name=site_name, calendar=calendar, require_lot=False)
    met = _weather(weather, config)
    summary = {
        "experiment": "rain_history_features", "status": "completed", "n_input": len(filters),
        "parameters": asdict(config), "calendar": calendar,
        "limitations": [
            "Weather describes the supplied station; it is not instrument-chamber humidity.",
            "Missing precipitation bins and incomplete lookback periods never become zero rain.",
            "Rainfall and composition associations are exploratory and do not establish a removal mechanism.",
        ],
    }
    if config.mode == "daily":
        rows = _daily_features(audit, met, config)
        summary["limitations"].append("Calendar-day precipitation cannot establish rain during the actual active collection intervals.")
    else:
        if active_intervals is None or active_intervals.empty:
            summary.update(status="blocked", reason="verified active collection intervals required for hourly collection rain")
            return _result(summary, {"features": audit, "weather_audit": met.reset_index(), "episodes": pd.DataFrame()})
        catalog = audit.assign(site=site_name, base_filter_id=audit.physical_id)
        intervals = validate_intervals(active_intervals, catalog, verify_source_files=verify_interval_sources)
        rows = _hourly_features(audit, met, intervals, config)
        summary["interval_source_files_verified"] = bool(verify_interval_sources)
        summary["limitations"].append("Hourly rainfall timing is bounded by the accumulation bin; partial-hour totals are not estimated.")
    features = pd.DataFrame(rows)
    if len(features):
        features["rain_block_id"] = _connected_blocks(rows)
        expected = [f"rain_before_{length}d_mm" for length in config.antecedent_days]
        for column in [*expected, "sample_day_rain_mm" if config.mode == "daily" else "rain_during_collection_mm"]:
            if column not in features:
                features[column] = np.nan
        features["rain_history_complete"] = features[expected].notna().all(axis=1)
        duplicate_columns = (set(audit) & set(features)) - {"physical_id"}
        for column in duplicate_columns:
            given = audit.set_index("physical_id")[column]
            computed = features.set_index("physical_id")[column].reindex(given.index)
            both = given.notna() & computed.notna()
            if both.any() and not given.loc[both].equals(computed.loc[both]):
                raise ValueError(f"input metadata disagrees with validated rain history: {column}")
        audit = audit.drop(columns=list(duplicate_columns))
        features = audit.merge(features, on="physical_id", validate="one_to_one")
    else:
        features = audit
    episode_rows = []
    for key, group in met.loc[met.rain_event_id.ne("")].groupby("rain_event_id", sort=True):
        episode_rows.append({"rain_event_id": key, "first_wet_bin": group.index.min(), "last_wet_bin": group.index.max(),
                             "wet_bins": len(group), "wet_bin_total_mm": float(group.rain_mm.sum()),
                             "total_note": "sum of threshold-exceeding bins; event dry gaps excluded"})
    summary.update(n_complete_history=int(features.get("rain_history_complete", pd.Series(dtype=bool)).sum()),
                   n_weather_bins=len(met), n_valid_weather_bins=int(met.valid_precipitation.sum()),
                   n_rain_episodes=len(episode_rows), n_rain_blocks=int(features.get("rain_block_id", pd.Series(dtype=str)).nunique()))
    if not summary["n_complete_history"]:
        summary.update(status="insufficient_data", reason="no filters have complete antecedent precipitation history")
    return _result(summary, {"features": features, "weather_audit": met.reset_index(), "episodes": pd.DataFrame(episode_rows)})


def evaluate_rain_history(
    filters: pd.DataFrame, *, antecedent_columns=("rain_before_1d_mm", "rain_before_3d_mm", "rain_before_7d_mm"),
    during_column="sample_day_rain_mm", numeric_controls=("pm25_ugm3",),
    site_name="Addis Ababa", calendar="belg_feb", min_train=30, min_test=6,
    test_coverage="all_months", alpha=1.0, bootstrap_reps=300, seed=SEED,
) -> dict:
    """Forward residual comparisons, purging complete shared rain-event blocks.

    Input is the ``features`` table from ``build_rain_history``. The daily arm
    is labelled sample-day association, and never reported as active washout.
    All arms are evaluated on the same support, including the declared during
    or sample-day covariate. Missing precipitation is not imputed.
    """
    if not antecedent_columns or len(set(antecedent_columns)) != len(antecedent_columns):
        raise ValueError("prespecify distinct antecedent-rain columns")
    if during_column not in {"sample_day_rain_mm", "rain_during_collection_mm"}:
        raise ValueError("during_column must explicitly identify sample day or actual collection")
    if set(numeric_controls) & (set(antecedent_columns) | {during_column}):
        raise ValueError("rain terms cannot already be baseline controls")
    _required(filters, ["rain_precision", "rain_block_id", *antecedent_columns, during_column])
    precision = set(filters.rain_precision.dropna())
    wanted = "local_calendar_day" if during_column == "sample_day_rain_mm" else "verified_active_intervals_hourly"
    if precision != {wanted}:
        raise ValueError("rain covariate name disagrees with the declared temporal precision")
    audit = _base_filters(filters, site_name=site_name, calendar=calendar)
    arm = "plus_sample_day_rain" if during_column == "sample_day_rain_mm" else "plus_active_collection_rain"
    result = _evaluate_residual_models(
        audit, extra_arms={"plus_antecedent_rain": list(antecedent_columns), arm: [*antecedent_columns, during_column]},
        numeric_controls=numeric_controls, calendar=calendar, min_train=min_train,
        min_test=min_test, test_coverage=test_coverage, group_column="rain_block_id",
        alpha=alpha, bootstrap_reps=bootstrap_reps, seed=seed,
    )
    result["summary"].update(experiment="rain_history_residual", rain_precision=wanted)
    result["summary"]["limitations"].extend([
        "Rain episodes shared between training and test remove their training filters; the test period stays fixed.",
        "Complete antecedent weather and all prespecified covariates are required for paired evaluation.",
    ])
    if wanted == "local_calendar_day":
        result["summary"]["limitations"].append("The sample-day arm cannot establish rain during actual collection or a subdaily washout response.")
    return result
