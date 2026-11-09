"""Rain history must preserve timing, missingness and whole-event validation."""

from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"
sys.path.insert(0, str(SCRIPTS))

from config import MAC_VALUE
from conference_experiments.rain import RainConfig, build_rain_history, evaluate_rain_history


def daily_weather():
    dates = pd.date_range("2020-01-01", periods=14, freq="D")
    rain = np.zeros(len(dates))
    rain[8], rain[9], rain[11] = 3, 50, 4
    return pd.DataFrame({"date": dates, "rain_mm": rain})


def filters(dates=("2020-01-10",)):
    return pd.DataFrame({"physical_id": [f"ETAD-{7000+i}" for i in range(len(dates))], "date": pd.to_datetime(list(dates))})


def test_daily_antecedent_excludes_sample_day_and_future_rain():
    weather = daily_weather()
    result = build_rain_history(filters(), weather)
    row = result["tables"]["features"].iloc[0]
    assert row.rain_before_1d_mm == 3
    assert row.rain_before_3d_mm == 3
    assert row.sample_day_rain_mm == 50
    assert np.isnan(row.rain_during_collection_mm)
    assert row.collection_rain_status == "unavailable_from_daily_sample_date"
    assert row.time_since_rain_days == 1
    modified = weather.copy()
    modified.loc[modified.date >= "2020-01-10", "rain_mm"] = 9999
    other = build_rain_history(filters(), modified)["tables"]["features"].iloc[0]
    assert row.rain_before_7d_mm == other.rain_before_7d_mm
    assert row.time_since_rain_days == other.time_since_rain_days


def test_rain_baseline_rejects_response_components():
    frame = build_rain_history(filters(), daily_weather())["tables"]["features"]
    with pytest.raises(ValueError, match="HIPS component"):
        evaluate_rain_history(frame, numeric_controls=["hips_fabs_Mm1"])


def test_missing_and_negative_precipitation_never_become_dry_weather():
    weather = daily_weather()
    weather.loc[8, "rain_mm"] = np.nan
    weather.loc[7, "rain_mm"] = -1
    result = build_rain_history(filters(), weather)
    row = result["tables"]["features"].iloc[0]
    assert np.isnan(row.rain_before_1d_mm)
    assert np.isnan(row.rain_before_3d_mm)
    assert row.rain_before_3d_coverage == pytest.approx(1 / 3)
    assert row.sample_day_rain_mm == 50
    assert row.time_since_rain_censored
    audit = result["tables"]["weather_audit"]
    assert audit.weather_reason.str.contains("negative").sum() == 2
    assert not row.rain_history_complete


def test_rain_history_outside_weather_extent_is_censored():
    result = build_rain_history(filters(("2020-01-03", "2020-02-01")), daily_weather())
    rows = result["tables"]["features"]
    assert rows.rain_before_7d_mm.isna().all()
    assert rows.time_since_rain_days.isna().all()
    assert rows.time_since_rain_censored.all()
    assert rows.dry_history_lower_bound_days.iloc[1] == 0
    assert result["summary"]["status"] == "insufficient_data"


def test_duplicate_or_subdaily_dates_do_not_silently_define_daily_bins():
    weather = daily_weather()
    with pytest.raises(ValueError, match="duplicate precipitation"):
        build_rain_history(filters(), pd.concat([weather, weather.iloc[[0]]]))
    weather.loc[0, "date"] += pd.Timedelta(hours=3)
    with pytest.raises(ValueError, match="calendar days"):
        build_rain_history(filters(), weather)
    with pytest.raises(ValueError, match="whole-day"):
        RainConfig(event_dry_gap_hours=3)


def hourly_weather():
    starts = pd.date_range("2020-01-01", periods=72, freq="h", tz="UTC")
    return pd.DataFrame({"interval_start_utc": starts, "interval_end_utc": starts + pd.Timedelta(hours=1), "rain_mm": 1.})


def active_periods(tmp_path, bounds):
    evidence = tmp_path / "synthetic_pump_log.txt"
    evidence.write_text("Synthetic test only: pump state and clock verified.")
    rows = []
    for i, (start, end) in enumerate(bounds):
        rows.append({"site": "Addis Ababa", "base_filter_id": "ETAD-7000", "interval_id": f"part-{i}",
                     "active_start_utc": pd.Timestamp(start), "active_end_utc": pd.Timestamp(end),
                     "original_local_start": start, "original_local_end": end, "timezone": "UTC",
                     "collection_mode": "documented_continuous" if len(bounds) == 1 else "documented_intermittent",
                     "schedule_evidence_type": "synthetic pump log", "source_file": str(evidence),
                     "source_file_hash": sha256(evidence.read_bytes()).hexdigest(), "source_row": i,
                     "resolution_reason": "synthetic known active interval"})
    return pd.DataFrame(rows)


def test_hourly_rain_uses_active_intervals_and_ignores_inactive_gap(tmp_path):
    weather = hourly_weather()
    weather.loc[weather.interval_start_utc.eq(pd.Timestamp("2020-01-02T03:00Z")), "rain_mm"] = 1000.
    intervals = active_periods(tmp_path, [("2020-01-02T00:00Z", "2020-01-02T02:00Z"),
                                         ("2020-01-02T06:00Z", "2020-01-02T08:00Z")])
    result = build_rain_history(filters(("2020-01-02",)), weather, config=RainConfig(mode="hourly", antecedent_days=(1,)), active_intervals=intervals)
    row = result["tables"]["features"].iloc[0]
    assert row.rain_before_1d_mm == 24
    assert row.rain_during_collection_mm == 4
    assert row.rain_during_collection_coverage == 1
    assert row.active_hours == 4
    assert row.active_interval_count == 2
    assert result["summary"]["interval_source_files_verified"]


def test_hourly_bounds_are_half_open_and_partial_bins_are_not_prorated(tmp_path):
    weather = hourly_weather()
    weather.loc[weather.interval_start_utc.eq(pd.Timestamp("2020-01-02T02:00Z")), "rain_mm"] = 999
    intervals = active_periods(tmp_path, [("2020-01-02T00:00Z", "2020-01-02T02:00Z")])
    exact = build_rain_history(filters(("2020-01-02",)), weather, config=RainConfig(mode="hourly", antecedent_days=(1,)), active_intervals=intervals)
    assert exact["tables"]["features"].rain_during_collection_mm.iloc[0] == 2
    partial = active_periods(tmp_path, [("2020-01-02T00:30Z", "2020-01-02T02:00Z")])
    result = build_rain_history(filters(("2020-01-02",)), weather, config=RainConfig(mode="hourly", antecedent_days=(1,)), active_intervals=partial)
    row = result["tables"]["features"].iloc[0]
    assert np.isnan(row.rain_during_collection_mm)
    assert np.isnan(row.rain_before_1d_mm)
    assert row.collection_rain_status == "partial_hour_bounds"
    assert row.rain_during_collection_coverage == pytest.approx(2 / 3)


def test_hourly_collection_requires_evidence_and_missing_rain_is_not_zero(tmp_path):
    weather = hourly_weather()
    blocked = build_rain_history(filters(), weather, config=RainConfig(mode="hourly"))
    assert blocked["summary"]["status"] == "blocked"
    intervals = active_periods(tmp_path, [("2020-01-02T00:00Z", "2020-01-02T02:00Z")])
    weather.loc[weather.interval_start_utc.eq(pd.Timestamp("2020-01-02T01:00Z")), "rain_mm"] = np.nan
    result = build_rain_history(filters(("2020-01-02",)), weather, config=RainConfig(mode="hourly", antecedent_days=(1,)), active_intervals=intervals)
    row = result["tables"]["features"].iloc[0]
    assert np.isnan(row.rain_during_collection_mm)
    assert row.rain_during_collection_observed_mm == 1
    assert row.rain_during_collection_coverage == .5
    bad = intervals.copy()
    bad.active_start_utc = bad.active_start_utc.dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone"):
        build_rain_history(filters(("2020-01-02",)), weather, config=RainConfig(mode="hourly"), active_intervals=bad)


def test_shared_rain_episodes_are_assigned_one_validation_block():
    weather = daily_weather()
    result = build_rain_history(filters(("2020-01-10", "2020-01-12", "2020-01-14")), weather)
    rows = result["tables"]["features"]
    assert rows.rain_block_id.nunique() == 1
    assert all(rows.rain_event_ids.map(bool))


def residual_cohort():
    rng = np.random.default_rng(55)
    dates = [month + pd.Timedelta(days=day) for month in pd.date_range("2019-10-01", periods=24, freq="MS") for day in (4, 14, 24)]
    n = len(dates)
    ec = rng.uniform(1, 8, n)
    antecedent = rng.uniform(0, 10, n)
    during = rng.uniform(0, 5, n)
    disagreement = .2 + .1 * ec + .15 * antecedent + .2 * during
    return pd.DataFrame({"physical_id": [f"ETAD-{8000+i}" for i in range(n)], "date": dates,
                         "lot": "synthetic_lot", "ftir_ec_ugm3": ec,
                         "hips_fabs_Mm1": (ec - disagreement) * MAC_VALUE, "hips_source": "direct_HIPS",
                         "pm25_ugm3": ec * 3, "rain_before_3d_mm": antecedent,
                         "sample_day_rain_mm": during, "rain_precision": "local_calendar_day",
                         "rain_block_id": [f"episode-{i}" for i in range(n)]})


def test_rain_residual_keeps_episodes_out_of_training_and_pairs_model_support():
    frame = residual_cohort()
    frame.loc[[35, 36], "rain_block_id"] = "shared-boundary-storm"
    result = evaluate_rain_history(frame, antecedent_columns=("rain_before_3d_mm",), min_train=18, bootstrap_reps=20)
    assert result["summary"]["status"] == "completed"
    assert result["tables"]["folds"].n_train.iloc[0] == 35
    assert result["tables"]["folds"].purged_train_ids.iloc[0] == [frame.physical_id.iloc[35]]
    predictions = result["tables"]["predictions"]
    assert predictions.groupby("arm").physical_id.nunique().nunique() == 1
    metrics = result["tables"]["metrics"].set_index("arm")
    assert metrics.loc["plus_sample_day_rain", "rmse"] < .2 * metrics.loc["controlled_baseline", "rmse"]
    assert result["summary"]["rain_precision"] == "local_calendar_day"


def test_rain_precision_and_missing_history_are_enforced_in_model():
    frame = residual_cohort()
    frame["rain_during_collection_mm"] = frame.sample_day_rain_mm
    with pytest.raises(ValueError, match="temporal precision"):
        evaluate_rain_history(frame, antecedent_columns=("rain_before_3d_mm",), during_column="rain_during_collection_mm")
    frame.loc[36, "rain_before_3d_mm"] = np.nan
    result = evaluate_rain_history(frame, antecedent_columns=("rain_before_3d_mm",), min_train=18, bootstrap_reps=0)
    audit = result["tables"]["eligibility"]
    assert audit.loc[36, "is_excluded"]
    assert "rain_before_3d_mm" in audit.exclusion_reason.iloc[36]
    assert frame.physical_id.iloc[36] not in set(result["tables"]["predictions"].physical_id)
