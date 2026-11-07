"""Independent episode definitions, known spectral evolution, and block leakage."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"
sys.path.insert(0, str(SCRIPTS))
from config import WAVELENGTHS_NM
from conference_experiments.episodes import EpisodeConfig, run_episode_evolution


SIGNALS = {"IR": "IR_atn", "UV": "UV_atn"}


def episode_fixture(days=8):
    frames, events = [], []
    hours = [0, 5, 10, 15, 3, 8, 13, 18]
    for day in range(days):
        start = pd.Timestamp("2024-02-01", tz="UTC") + pd.Timedelta(days=day, hours=hours[day % len(hours)])
        exponent = 1.8 - 0.002 * np.arange(180)
        frames.append(pd.DataFrame({
            "tu": pd.date_range(start, periods=180, freq="min"),
            "Sample RH (%)": 50.0, "Sample temp (C)": 23.0,
            "Session ID": day, "Tape position": 1, "IR ATN1": 15.0,
            "Flow1 (mL/min)": 70.0, "Flow2 (mL/min)": 35.0,
            "IR_atn": 10.0,
            "UV_atn": 10.0 * (WAVELENGTHS_NM["UV"] / WAVELENGTHS_NM["IR"]) ** -exponent,
        }))
        events.append({"event_id": f"E{day}", "start_utc": start,
                       "end_utc": start + pd.Timedelta(minutes=180),
                       "definition": "independent", "evidence": "Synthetic independently fixed chamber schedule"})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(events)


def run(frame, episodes, config=None):
    return run_episode_evolution(frame, episodes, signal_columns=SIGNALS, signal_quantity="b_atn",
                                 timezone="UTC", rh_provenance="Synthetic sample-stream RH", config=config)


def test_known_early_late_spectral_change_and_reproduction_on_whole_later_events():
    frame, events = episode_fixture()
    result = run(frame, events, replace(EpisodeConfig(), n_bootstrap=30))
    assert result["summary"]["n_eligible_episodes"] == 8
    assert result["summary"]["outcome"] == "attenuation_exponent"
    effects = result["tables"]["event_effects"]
    np.testing.assert_allclose(effects.mean_late_minus_early, -0.27, atol=1e-12)
    assert effects.supported.all()
    scores = result["tables"]["model_metrics"].set_index("model")
    assert scores.loc["with_elapsed_time", "rmse"] < scores.loc["state_only", "rmse"] / 10
    folds = result["tables"]["model_folds"]
    train_blocks = set(folds.loc[folds.partition.eq("training"), "_day_block"])
    test_blocks = set(folds.loc[folds.partition.eq("evaluation"), "_day_block"])
    assert train_blocks.isdisjoint(test_blocks)
    assert result["summary"]["matched_contrast"]["n_blocks"] == 8


def test_external_event_definitions_are_required_and_not_inferred_from_optical_peaks():
    frame, events = episode_fixture()
    assert run(frame, None)["summary"]["status"] == "blocked"
    events.loc[:, "definition"] = "detected_from_uv_ir_response"
    result = run(frame, events)
    assert result["summary"]["n_eligible_episodes"] == 0
    assert result["tables"]["episodes"].exclusion_reason.str.contains("independent_definition_evidence_missing").all()


def test_event_crossing_tape_change_or_gap_is_audited_not_silently_spliced():
    frame, events = episode_fixture()
    frame.loc[60:179, "Tape position"] = 2
    frame = frame.drop(index=range(220, 230))
    result = run(frame, events, replace(EpisodeConfig(), n_bootstrap=20))
    audit = result["tables"]["episodes"].set_index("event_id")
    assert audit.loc["E0", "is_excluded"]
    assert audit.loc["E1", "is_excluded"]
    assert "crosses_tape_session_gap" in audit.loc["E0", "exclusion_reason"]
    assert result["summary"]["n_eligible_episodes"] == 6


def test_overlapping_or_duplicate_episodes_cannot_reuse_minutes_across_validation():
    frame, events = episode_fixture()
    with pytest.raises(ValueError, match="unique"):
        run(frame, pd.concat([events, events.iloc[[0]]]))
    overlap = events.iloc[[0]].copy()
    overlap.event_id = "another-label-same-minutes"
    with pytest.raises(ValueError, match="overlap"):
        run(frame, pd.concat([events, overlap]))


def test_invalid_optical_values_are_missing_and_do_not_become_zero_exponents():
    frame, events = episode_fixture()
    frame.loc[0:5, "UV_atn"] = -1
    result = run(frame, events, replace(EpisodeConfig(), n_bootstrap=20))
    phases = result["tables"]["phases"]
    phase = phases.loc[phases.event_id.eq("E0") & phases.phase.eq("early")].iloc[0]
    assert phase.n_valid_exponent == 39
    assert phase.mean_exponent > 1.7
    assert result["tables"]["episodes"].set_index("event_id").loc["E0", "n_valid_exponent"] == 174


def test_matching_does_not_claim_support_when_rh_or_amplitude_regimes_do_not_overlap():
    frame, events = episode_fixture()
    for day in range(8):
        frame.loc[day * 180 + 135:day * 180 + 179, "Sample RH (%)"] = 80
    result = run(frame, events, replace(EpisodeConfig(), n_bootstrap=20))
    assert result["tables"]["matched_pairs"].empty
    assert result["tables"]["episodes"].matching_status.eq("insufficient_common_support").all()


def test_same_flow_ratio_with_doubled_total_flow_cannot_match_episode_phases():
    frame, events = episode_fixture()
    for day in range(8):
        frame.loc[day * 180 + 135:day * 180 + 179, ["Flow1 (mL/min)", "Flow2 (mL/min)"]] *= 2
    result = run(frame, events, replace(EpisodeConfig(), n_bootstrap=20))
    phases = result["tables"]["phases"]
    assert phases.mean_flow_ratio.nunique() == 1
    assert set(phases.mean_total_flow) == {105.0, 210.0}
    assert result["tables"]["matched_pairs"].empty
    assert result["tables"]["episodes"].matching_status.eq("insufficient_common_support").all()
    assert "_flow_total" in result["summary"]["model"]["base_features"]


def test_altering_evaluation_spectra_leaves_training_coefficients_unchanged():
    frame, events = episode_fixture()
    config = replace(EpisodeConfig(), n_bootstrap=20)
    original = run(frame, events, config)
    frame.loc[frame.tu >= events.iloc[6].start_utc, "UV_atn"] *= 2
    changed = run(frame, events, config)
    np.testing.assert_allclose(original["tables"]["model_coefficients"].coefficient,
                               changed["tables"]["model_coefficients"].coefficient)


def test_events_sharing_a_local_day_stay_in_one_independence_block():
    frame, events = episode_fixture()
    # Split the first externally scheduled event into two adjacent independent intervals.
    events.loc[0, "end_utc"] = events.loc[0, "start_utc"] + pd.Timedelta(minutes=90)
    extra = events.iloc[[0]].copy()
    extra.event_id = "E0-second"
    extra.start_utc = extra.end_utc
    extra.end_utc += pd.Timedelta(minutes=90)
    result = run(frame, pd.concat([events, extra]), replace(EpisodeConfig(), n_bootstrap=20))
    audit = result["tables"]["episodes"].set_index("event_id")
    assert audit.loc["E0", "day_block"] == audit.loc["E0-second", "day_block"]
    assert result["summary"]["matched_contrast"]["n_blocks"] == 8


def test_quantity_and_wavelength_order_are_explicit():
    frame, events = episode_fixture()
    with pytest.raises(ValueError, match="BCc"):
        run_episode_evolution(frame, events, signal_columns=SIGNALS, signal_quantity="BCc",
                               timezone="UTC", rh_provenance="sample stream")
    with pytest.raises(ValueError, match="shorter wavelength"):
        run(frame, events, replace(EpisodeConfig(), short_signal="IR", long_signal="UV"))
