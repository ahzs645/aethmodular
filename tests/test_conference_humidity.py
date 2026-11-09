"""Known humidity recoveries, elapsed-time support, and held-out-day safeguards."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"
sys.path.insert(0, str(SCRIPTS))
from conference_experiments.humidity import (
    HumidityConfig, _prepare_minutes, _time_lag, reconstruct_ma350_attenuation,
    run_humidity_memory,
)


SIGNALS = {"IR": "IR_atn", "UV": "UV_atn", "IR_spot1": "IR_spot1_atn"}


def recovery_fixture(days=8):
    rows = []
    for day in range(days):
        t = np.arange(600)
        sign = 1 if day % 2 == 0 else -1
        initial = 40 if sign == 1 else 80
        rh = np.full(600, float(initial))
        rh[150:400] += sign * 20
        response = np.zeros(600)
        response[150:] += sign * 3 * np.exp(-(t[150:] - 150) / 20)
        response[400:] -= sign * 3 * np.exp(-(t[400:] - 400) / 20)
        rows.append(pd.DataFrame({
            "tu": pd.date_range(pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day), periods=600, freq="min"),
            "Sample RH (%)": rh, "Sample temp (C)": 23.0,
            "Session ID": day, "Tape position": 1,
            "IR ATN1": 12.0, "Flow1 (mL/min)": 70.0, "Flow2 (mL/min)": 35.0,
            "IR_atn": 10 + response, "UV_atn": 25 + 2 * response,
            "IR_spot1_atn": 9 + 0.8 * response,
        }))
    return pd.concat(rows, ignore_index=True)


def run(frame, config=None):
    return run_humidity_memory(frame, signal_columns=SIGNALS, signal_quantity="b_atn",
                               timezone="UTC", rh_provenance="Synthetic sample-stream RH", config=config)


def test_known_recovery_persists_after_rh_stabilizes_and_history_predicts_held_out_days():
    result = run(recovery_fixture(), replace(HumidityConfig(), n_bootstrap=40))
    assert result["summary"]["n_events"] == 16
    effects = result["tables"]["event_effects"]
    ir = effects.loc[effects.signal.eq("IR")]
    assert ir.late_stable_supported.all()
    assert (ir.loc[ir.direction.eq("rising"), "late_stable_mean_delta"] > 0.2).all()
    assert (ir.loc[ir.direction.eq("falling"), "late_stable_mean_delta"] < -0.2).all()
    assert result["summary"]["models"]["IR"]["rmse_gain"] > 0.1
    scores = result["tables"]["model_metrics"]
    scores = scores.loc[scores.signal.eq("IR")].set_index("model")
    assert scores.loc["with_history", "rmse"] < scores.loc["instantaneous", "rmse"] / 2
    folds = result["tables"]["model_folds"]
    assert set(folds.loc[folds.partition.eq("training"), "_local_day"]).isdisjoint(
        folds.loc[folds.partition.eq("evaluation"), "_local_day"])
    assert folds.loc[folds.partition.eq("training"), "last_utc"].max() < folds.loc[
        folds.partition.eq("evaluation"), "first_utc"].min()
    assert "IR_spot1" in set(result["tables"]["event_curves"].signal)


def test_same_rh_matching_uses_opposite_direction_events_without_reuse():
    result = run(recovery_fixture(), replace(HumidityConfig(), n_bootstrap=30))
    pairs = result["tables"]["matched_pairs"].query("signal == 'IR'")
    assert len(pairs) == 4
    assert pairs.rising_event.is_unique and pairs.falling_event.is_unique
    assert np.allclose(pairs.rh_difference_pp, 0)
    assert (pairs.rising_minus_falling > 0.4).all()
    assert len(result["tables"]["matched_summary"]) == len(SIGNALS)
    assert result["tables"]["matched_summary"].n_blocks.min() == 4


def test_same_initial_plateau_does_not_match_different_late_rh_states():
    frame = recovery_fixture(days=2)
    # Both initial post-step plateaus are 60%; only the rising event then drifts.
    frame.loc[185:399, "Sample RH (%)"] = 70.0
    result = run(frame, replace(HumidityConfig(), n_bootstrap=20))
    events = result["tables"]["events"].query("not is_excluded")
    first = events.loc[events.rh_after.eq(60)]
    assert len(first) == 2
    assert set(first.late_stable_rh) == {60.0, 70.0}
    assert result["summary"]["n_matched_pairs"] == 0


def test_same_flow_ratio_with_doubled_total_flow_cannot_match_humidity_events():
    frame = recovery_fixture(days=2)
    frame.loc[600:, ["Flow1 (mL/min)", "Flow2 (mL/min)"]] *= 2
    result = run(frame, replace(HumidityConfig(), n_bootstrap=20))
    events = result["tables"]["events"].query("not is_excluded and rh_after == 60")
    assert events.flow_ratio.nunique() == 1
    assert set(events.late_stable_flow_total) == {105.0, 210.0}
    assert result["summary"]["n_matched_pairs"] == 0


def test_events_never_cross_tape_changes_or_recording_gaps():
    frame = recovery_fixture(days=2)
    frame.loc[frame.tu.between("2024-01-01 02:35Z", "2024-01-01 23:59Z"), "Tape position"] = 2
    frame = frame.loc[~frame.tu.between("2024-01-02 02:50Z", "2024-01-02 03:00Z")]
    result = run(frame, replace(HumidityConfig(), n_bootstrap=20))
    accepted = result["tables"]["events"].query("not is_excluded")
    assert len(accepted) == 2  # only the later return-to-baseline events
    rejected = result["tables"]["events"].query("is_excluded")
    assert rejected.exclusion_reason.str.contains("incomplete_contiguous_event_window").all()


def test_lags_are_elapsed_time_and_do_not_jump_a_session_or_missing_timestamp():
    frame = recovery_fixture(days=1).iloc[:25].copy()
    frame.loc[:, "Sample RH (%)"] = np.arange(25.0)
    frame = frame.drop(index=9).reset_index(drop=True)
    frame.loc[frame.tu >= pd.Timestamp("2024-01-01 00:20Z"), "Session ID"] = 2
    config = HumidityConfig()
    prepared = _prepare_minutes(frame, SIGNALS, "UTC", config)
    lag = _time_lag(prepared, prepared._rh, 5, config.lag_tolerance_minutes)
    at = prepared.set_index("_utc")
    lag.index = at.index
    assert lag.loc["2024-01-01 00:13Z"] == 8
    assert np.isnan(lag.loc["2024-01-01 00:14Z"])  # requested 00:09, not five rows ago
    assert lag.loc["2024-01-01 00:15Z"] == 10
    assert lag.loc["2024-01-01 00:20Z":"2024-01-01 00:24Z"].isna().all()


def test_invalid_rows_are_audited_and_negative_amplitude_remains_a_measured_excursion():
    frame = recovery_fixture(days=1)
    frame.loc[3, "Sample RH (%)"] = 140
    frame.loc[5, "IR_atn"] = -5
    frame.loc[7, "tu"] = pd.NaT
    frame["is_excluded"] = False
    frame["exclusion_reason"] = ""
    frame.loc[10, ["is_excluded", "exclusion_reason"]] = [True, "maintenance"]
    frame = pd.concat([frame, frame.iloc[[20]]], ignore_index=True)
    result = run(frame, replace(HumidityConfig(), n_bootstrap=20))
    excluded = result["tables"]["exclusions"]
    assert len(excluded) == 5
    assert excluded.exclusion_reason.str.contains("duplicate_utc_timestamp").sum() == 2
    assert excluded.exclusion_reason.str.contains("maintenance").any()
    assert result["tables"]["signal_support"].set_index("signal").loc["IR", "n_nonpositive_retained"] == 1
    assert frame.loc[5, "IR_atn"] == -5  # input unchanged


def test_negative_response_is_kept_in_humidity_event_amplitudes():
    frame = recovery_fixture(days=1)
    frame.loc[151, "IR_atn"] = -100.0
    result = run(frame, replace(HumidityConfig(), n_bootstrap=20))
    curve = result["tables"]["event_curves"].query(
        "event_id == 'RH-000001' and signal == 'IR' and minutes_from_threshold == 0")
    assert len(curve) == 1
    assert curve.iloc[0].mean_signal == pytest.approx(frame.loc[150:154, "IR_atn"].mean())
    assert curve.iloc[0].mean_signal < 0


def test_dense_burst_cannot_substitute_for_missing_window_coverage():
    from conference_experiments.humidity import _supported
    window = pd.DataFrame({"_utc": pd.date_range("2024-01-01", periods=100, freq="s", tz="UTC"),
                           "_rh": 50.0})
    assert not _supported(window, 10, HumidityConfig())


def test_microsecond_timestamp_storage_does_not_change_lag_units():
    frame = recovery_fixture(days=1).iloc[:20].copy()
    frame["tu"] = frame.tu.astype("datetime64[us, UTC]")
    frame["Sample RH (%)"] = np.arange(20.0)
    prepared = _prepare_minutes(frame, SIGNALS, "UTC", HumidityConfig())
    lag = _time_lag(prepared, prepared._rh, 5, 0.25)
    np.testing.assert_allclose(lag.iloc[5:], np.arange(15.0))


def test_future_optical_outcomes_cannot_change_fitted_coefficients():
    frame = recovery_fixture()
    config = replace(HumidityConfig(), n_bootstrap=20)
    original = run(frame, config)
    frame.loc[frame.tu >= pd.Timestamp("2024-01-07", tz="UTC"), "UV_atn"] += 100
    changed = run(frame, config)
    for result in (original, changed):
        assert result["summary"]["models"]["UV"]["test_start_utc"].startswith("2024-01-07")
    a = original["tables"]["model_coefficients"].query("signal == 'UV'").coefficient
    b = changed["tables"]["model_coefficients"].query("signal == 'UV'").coefficient
    np.testing.assert_allclose(a, b)


def test_raw_conversion_keeps_units_and_spot_provenance_explicit():
    raw = pd.DataFrame({"UV BCc": [1000, -1], "IR BCc": [2000, 0], "IR BC1": [1500, 100]})
    frame, signals, metadata = reconstruct_ma350_attenuation(
        raw, cross_sections={"UV": 24.069, "IR": 10.120}, bc_unit="ng/m3")
    np.testing.assert_allclose(frame[signals["UV"]], [24.069, -0.024069])
    np.testing.assert_allclose(frame[signals["IR_spot1"]], [15.18, 1.012])
    assert metadata["signal_quantity"] == "b_atn"
    assert metadata["scattering_correction_applied"] is False
    assert metadata["sources"]["IR_spot1"]["input_column"] == "IR BC1"


def test_required_provenance_and_boundary_metadata_cannot_be_silently_inferred():
    frame = recovery_fixture(days=1)
    with pytest.raises(ValueError, match="BCc"):
        run_humidity_memory(frame, signal_columns=SIGNALS, signal_quantity="BCc",
                            timezone="UTC", rh_provenance="chamber")
    with pytest.raises(ValueError, match="rh_provenance"):
        run_humidity_memory(frame, signal_columns=SIGNALS, signal_quantity="b_atn",
                            timezone="UTC", rh_provenance="")
    with pytest.raises(ValueError, match="Session ID"):
        run(frame.drop(columns="Session ID"))
    with pytest.raises(ValueError, match="lag_minutes"):
        run(frame, replace(HumidityConfig(), lag_minutes=(-5,)))


def test_missing_matching_covariates_preserve_event_curves_but_block_adjusted_scores():
    result = run(recovery_fixture().drop(columns="Sample temp (C)"), replace(HumidityConfig(), n_bootstrap=20))
    assert result["summary"]["n_events"] == 16
    assert result["tables"]["matching_audit"].match_exclusion_reason.eq("missing_matching_covariates").all()
    assert result["summary"]["models"]["IR"]["status"] == "insufficient_data"
    assert result["summary"]["n_matched_pairs"] == 0


def test_raw_manifest_dispatch_reconstructs_attenuation_and_rejects_absorption_relabelling():
    from conference_experiments.runner import dispatch
    raw = recovery_fixture(days=1).rename(columns={"IR_atn": "IR BCc", "UV_atn": "UV BCc"})
    options = {
        "raw_ma350": {"cross_sections": {"UV": 24.069, "IR": 10.120}, "bc_unit": "ng/m3", "include_spots": False},
        "timezone": "UTC", "rh_provenance": "Synthetic declared RH", "config": {"n_bootstrap": 20},
    }
    result = dispatch("humidity_memory", {"data": raw}, options)
    assert result["summary"]["signal_quantity"] == "b_atn"
    conversion = result["summary"]["raw_ma350_conversion"]
    assert conversion["sources"]["IR"]["sigma_m2_g"] == 10.120
    assert not conversion["scattering_correction_applied"]
    with pytest.raises(ValueError, match="not corrected b_abs"):
        dispatch("humidity_memory", {"data": raw}, {**options, "signal_quantity": "b_abs"})


def documented_interval(tmp_path):
    import hashlib
    source = tmp_path / "pump_log.csv"
    source.write_text("Synthetic pump log: active 00:00 to 05:00 UTC on 2024-01-01\n")
    row = {"site": "USPA", "base_filter_id": "USPA-0001", "interval_id": "active-1",
           "active_start_utc": "2024-01-01T00:00:00Z", "active_end_utc": "2024-01-01T05:00:00Z",
           "original_local_start": "2024-01-01T00:00:00Z", "original_local_end": "2024-01-01T05:00:00Z",
           "timezone": "UTC", "collection_mode": "documented_continuous",
           "schedule_evidence_type": "synthetic_pump_log", "source_file": str(source),
           "source_file_hash": hashlib.sha256(source.read_bytes()).hexdigest(), "source_row": 0,
           "resolution_reason": "Synthetic measured pump starts and stops"}
    catalog = pd.DataFrame({"site": ["USPA"], "base_filter_id": ["USPA-0001"]})
    evidence = {"stream_id": "synthetic-ma350", "evidence": "Explicit synthetic observed acquisition flags",
                "observed_column": "observed", "valid_column": "valid", "slot_origin_utc": "1970-01-01T00:00:00Z",
                "cadence_verified": True, "timestamp_role_verified": True,
                "observation_provenance_verified": True, "processing_history_verified": True,
                "evidence_scope": "full_export"}
    return pd.DataFrame([row]), catalog, evidence, source


def test_verified_filter_windows_quantify_selection_sensitivity_without_claiming_correction(tmp_path):
    frame = recovery_fixture(days=1).assign(observed=True, valid=True)
    intervals, catalog, evidence, _ = documented_interval(tmp_path)
    result = run_humidity_memory(
        frame, signal_columns=SIGNALS, signal_quantity="b_atn", timezone="UTC", rh_provenance="Synthetic sample stream",
        config=replace(HumidityConfig(), n_bootstrap=20), active_intervals=intervals,
        filter_catalog=catalog, stream_site="USPA", processing_evidence=evidence)
    stage = result["summary"]["filter_window_stage"]
    assert stage["status"] == "completed"
    assert "not corrected bias" in stage["interpretation"]
    row = result["tables"]["filter_window_sensitivity"].set_index("signal").loc["IR"]
    assert row.expected_slots == row.valid_observed_slots == 300
    assert row.recovery_slot_fraction == pytest.approx(0.2)
    assert row.mean_all_observed == pytest.approx(frame.iloc[:300].IR_atn.mean())
    expected_outside = pd.concat([frame.iloc[:150], frame.iloc[210:300]]).IR_atn.mean()
    assert row.mean_outside_recovery == pytest.approx(expected_outside)


def test_cleaned_filename_does_not_establish_observation_provenance(tmp_path):
    frame = recovery_fixture(days=1).assign(observed=True, valid=True)
    intervals, catalog, evidence, _ = documented_interval(tmp_path)
    evidence["observation_provenance_verified"] = False
    result = run_humidity_memory(
        frame, signal_columns=SIGNALS, signal_quantity="b_atn", timezone="UTC", rh_provenance="Sample RH field",
        config=replace(HumidityConfig(), n_bootstrap=20), active_intervals=intervals,
        filter_catalog=catalog, stream_site="USPA", processing_evidence=evidence)
    assert result["summary"]["filter_window_stage"]["status"] == "blocked"
    assert result["summary"]["n_events"] == 2  # diagnostic events remain available
    assert "filter_window_sensitivity" not in result["tables"]


def test_unknown_observed_flags_cannot_be_counted_as_actual_filter_coverage(tmp_path):
    frame = recovery_fixture(days=1).assign(observed=True, valid=True)
    frame["observed"] = frame.observed.astype("boolean")
    frame.loc[10, "observed"] = pd.NA
    intervals, catalog, evidence, _ = documented_interval(tmp_path)
    result = run_humidity_memory(
        frame, signal_columns=SIGNALS, signal_quantity="b_atn", timezone="UTC", rh_provenance="Sample RH field",
        config=replace(HumidityConfig(), n_bootstrap=20), active_intervals=intervals,
        filter_catalog=catalog, stream_site="USPA", processing_evidence=evidence)
    assert result["summary"]["filter_window_stage"]["status"] == "insufficient_data"
    assert result["tables"]["filter_window_sensitivity"].mean_all_observed.isna().all()


def test_active_interval_source_hash_is_verified(tmp_path):
    frame = recovery_fixture(days=1).assign(observed=True, valid=True)
    intervals, catalog, evidence, source = documented_interval(tmp_path)
    source.write_text("Changed pump log\n")
    with pytest.raises(ValueError, match="source hash changed"):
        run_humidity_memory(
            frame, signal_columns=SIGNALS, signal_quantity="b_atn", timezone="UTC", rh_provenance="Sample RH field",
            config=replace(HumidityConfig(), n_bootstrap=20), active_intervals=intervals,
            filter_catalog=catalog, stream_site="USPA", processing_evidence=evidence)


def test_unknown_registered_filter_exclusion_is_audited(tmp_path):
    frame = recovery_fixture(days=1).assign(observed=True, valid=True)
    intervals, catalog, evidence, _ = documented_interval(tmp_path)
    catalog["is_excluded"] = pd.Series([pd.NA], dtype="boolean")
    result = run_humidity_memory(
        frame, signal_columns=SIGNALS, signal_quantity="b_atn", timezone="UTC", rh_provenance="Sample RH field",
        config=replace(HumidityConfig(), n_bootstrap=20), active_intervals=intervals,
        filter_catalog=catalog, stream_site="USPA", processing_evidence=evidence)
    ledger = result["tables"]["filter_window_sensitivity"]
    assert ledger.exclusion_reason.eq("registered_filter_exclusion_unknown").all()
    assert ledger.mean_all_observed.isna().all()
