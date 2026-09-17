"""Scientific gates introduced by evidence-backed active-interval matching."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"
sys.path.insert(0, str(SCRIPTS))
from active_interval_matching import (
    StreamContract,
    eligibility,
    empty_intervals,
    summarize_active_segments,
    validate_intervals,
)
from interval_evidence import schedule_catalog, trace_ec_provenance
from instrument_provenance import (
    CHANNEL_COLUMNS,
    stage_instrument_csv,
    apply_processing_evidence,
    processing_records,
)
from observation_coverage import expected_slots


def contract():
    return StreamContract(
        "synthetic:IR",
        cadence_verified=True,
        timestamp_role_verified=True,
        observation_provenance_verified=True,
        processing_history_verified=True,
        units_verified=True,
        wavelength_nm=880,
        evidence_scope="full_export",
        scope_justification="Synthetic evidence covers the complete fixture",
    )


def periods(bounds):
    return pd.DataFrame(
        [
            dict(
                site="JPL",
                base_filter_id="USPA-test",
                interval_id=f"part{i}",
                active_start_utc=pd.Timestamp(a),
                active_end_utc=pd.Timestamp(b),
            )
            for i, (a, b) in enumerate(bounds)
        ]
    )


def parent():
    return dict(
        site="JPL",
        base_filter_id="USPA-test",
        filter_type="PM2.5",
        filter_type_conflict=False,
        is_excluded=False,
        hips_fabs_Mm1=10.0,
        hips_fabs_Mm1_conflict=False,
        hips_fabs_Mm1_units_valid=True,
        eligible_hips_ftir_diagnostic=True,
        ftir_ec_ugm3=1.0,
        ftir_ec_mdl_ugm3=0.2,
        ftir_ec_mdl_conflict=False,
        chemspec_ec_ugm3_conflict=True,
        independent_ec_reference_available=False,
    )


def test_two_twelve_hour_periods_use_active_hours_and_ignore_envelope_values():
    intervals = periods(
        [("2024-01-01T00:00Z", "2024-01-01T12:00Z"), ("2024-01-08T12:00Z", "2024-01-09T00:00Z")]
    )
    index = pd.date_range("2024-01-01", periods=8 * 24 * 60, freq="min", tz="UTC")
    frame = pd.DataFrame({"IR": 1e9, "observed": True}, index=index)
    frame.loc["2024-01-01T00:00Z":"2024-01-01T11:59Z", "IR"] = 1000.0
    frame.loc["2024-01-08T12:00Z":"2024-01-08T17:59Z", "IR"] = 3000.0
    frame.loc["2024-01-08T18:00Z":, "observed"] = False  # interpolated input, not observed
    segments, summary = summarize_active_segments(frame, intervals, "IR", contract(), "observed")
    assert summary["expected_slots"] == 1440
    assert summary["valid_observed_slots"] == 1080
    assert summary["valid_observed_coverage_fraction"] == 0.75
    assert segments.valid_observed_coverage_fraction.tolist() == [1.0, 0.5]
    assert summary["mean_valid_observed_ebc_ugm3"] == pytest.approx(5 / 3)
    assert summary["input_availability_fraction"] == 1.0
    assert np.isnan(summary["absorption_Mm1"])


def test_unknown_observation_history_never_becomes_full_observed_coverage():
    intervals = periods([("2024-01-01T00:00Z", "2024-01-01T01:00Z")])
    frame = pd.DataFrame(
        {"IR": 1000.0}, index=pd.date_range("2024-01-01", periods=60, freq="min", tz="UTC")
    )
    _, s = summarize_active_segments(frame, intervals, "IR", contract())
    assert s["input_availability_fraction"] == 1
    assert np.isnan(s["observed_coverage_fraction"])
    assert np.isnan(s["mean_valid_observed_ebc_ugm3"])
    frame["observed"] = True
    _, s = summarize_active_segments(
        frame,
        intervals,
        "IR",
        replace(contract(), observation_provenance_verified=False),
        "observed",
    )
    assert np.isnan(s["observed_coverage_fraction"])


def test_channel_validity_and_unknown_flags_are_separate_from_observation():
    intervals = periods([("2024-01-01T00:00Z", "2024-01-01T00:04Z")])
    frame = pd.DataFrame(
        {
            "IR": [1, 2, np.nan, np.inf],
            "Blue": [1, 2, 3, 4],
            "observed": pd.array([True, True, True, False], dtype="boolean"),
        },
        index=pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC"),
    )
    _, ir = summarize_active_segments(frame, intervals, "IR", contract(), "observed")
    _, blue = summarize_active_segments(frame, intervals, "Blue", contract(), "observed")
    assert ir["observed_coverage_fraction"] == blue["observed_coverage_fraction"] == 0.75
    assert ir["valid_observed_coverage_fraction"] == 0.5
    assert blue["valid_observed_coverage_fraction"] == 0.75
    frame.loc[frame.index[0], "observed"] = pd.NA
    _, unknown = summarize_active_segments(frame, intervals, "IR", contract(), "observed")
    assert np.isnan(unknown["observed_coverage_fraction"])


def test_subminute_rows_cannot_inflate_slot_weight_and_end_is_excluded():
    intervals = periods([("2024-01-01T00:00Z", "2024-01-01T00:02Z")])
    frame = pd.DataFrame(
        {"IR": [0, 10, 20, 10000], "observed": True},
        index=pd.to_datetime(
            [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:30Z",
                "2024-01-01T00:01:00Z",
                "2024-01-01T00:02:00Z",
            ]
        ),
    )
    _, s = summarize_active_segments(frame, intervals, "IR", contract(), "observed")
    assert s["expected_slots"] == s["valid_observed_slots"] == 2
    assert s["mean_valid_observed_native"] == 12.5
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_active_segments(
            pd.concat([frame, frame.iloc[:1]]), intervals, "IR", contract(), "observed"
        )


def test_nominal_grid_boundary_and_dst_expected_instants():
    assert expected_slots("2024-01-01T00:00:30Z", "2024-01-01T00:02:00Z").tolist() == [
        pd.Timestamp("2024-01-01T00:01:00Z")
    ]
    assert (
        len(
            expected_slots(
                pd.Timestamp("2024-03-09 09:00", tz="America/Los_Angeles"),
                pd.Timestamp("2024-03-10 09:00", tz="America/Los_Angeles"),
            )
        )
        == 1380
    )


def test_consistent_envelope_stays_unverified_and_does_not_create_an_interval():
    c = pd.DataFrame(
        [
            {
                **parent(),
                "portal_timing_status": "reported_hours_consistent_with_continuous_window; observation_log_unverified",
                "portal_window_available": True,
            }
        ]
    )
    lab = pd.DataFrame(columns=["site", "base_filter_id"])
    result = schedule_catalog(c, lab, empty_intervals())
    assert result.schedule_status.iloc[0] == "bounds_consistent_mode_unverified"
    assert not result.sampling_schedule_verified.iloc[0]
    assert result.verified_active_interval_count.iloc[0] == 0


def test_unresolved_ec_blocks_only_ratios_using_that_parameter():
    s = {
        "valid_observed_coverage_fraction": 0.9,
        "minimum_segment_valid_observed_coverage": 0.8,
        "mean_valid_observed_ebc_ugm3": 1.0,
        "processing_evidence_scope_status": "verified",
    }
    decisions = eligibility(
        parent(), s, contract(), schedule_verified=True, ec_parameter_resolved=False
    )
    assert decisions["eligible_interval_ebc_comparison"]
    assert decisions["eligible_filter_diagnostic"]
    assert not decisions["eligible_ec_ratio_analysis"]
    assert not decisions["eligible_independent_ec_validation"]
    # A resolved FTIR denominator does not require adjudicating ChemSpec EC.
    resolved = eligibility(
        parent(), s, contract(), schedule_verified=True, ec_parameter_resolved=True
    )
    assert resolved["eligible_ec_ratio_analysis"]
    low = eligibility(
        {**parent(), "ftir_ec_ugm3": 0.1},
        s,
        contract(),
        schedule_verified=True,
        ec_parameter_resolved=True,
    )
    assert not low["eligible_ec_ratio_analysis"]


def test_optical_conversion_requires_positive_coefficient_and_evidence():
    s = {
        "valid_observed_coverage_fraction": 1.0,
        "minimum_segment_valid_observed_coverage": 1.0,
        "mean_valid_observed_ebc_ugm3": 1.0,
        "processing_evidence_scope_status": "verified",
    }
    c = replace(
        contract(),
        optical_conversion_verified=True,
        wavelength_treatment_verified=True,
        absorption_coefficient_m2_g=-1,
        absorption_conversion_evidence="synthetic evidence",
    )
    assert not eligibility(parent(), s, c, schedule_verified=True)["eligible_absorption_comparison"]
    c = replace(c, absorption_coefficient_m2_g=5.0)
    assert eligibility(parent(), s, c, schedule_verified=True)["eligible_absorption_comparison"]
    assert not eligibility(
        parent(), s, replace(c, absorption_conversion_evidence=""), schedule_verified=True
    )["eligible_absorption_comparison"]


def test_interval_source_hash_local_clock_and_overlap_are_enforced(tmp_path):
    source = tmp_path / "sampler.csv"
    source.write_text("synthetic fixture only\n")
    p = periods([("2024-01-01T00:00Z", "2024-01-01T12:00Z")])
    for k, v in dict(
        original_local_start="2024-01-01 00:00",
        original_local_end="2024-01-01 12:00",
        timezone="UTC",
        collection_mode="documented_intermittent",
        schedule_evidence_type="synthetic_log",
        source_file=str(source),
        source_file_hash=sha256(source.read_bytes()).hexdigest(),
        source_row=0,
        resolution_reason="Synthetic observed operation for test only",
    ).items():
        p[k] = v
    assert len(validate_intervals(p, pd.DataFrame([parent()]))) == 1
    overlap = pd.concat([p, p.assign(interval_id="overlap")], ignore_index=True)
    with pytest.raises(ValueError, match="Overlapping"):
        validate_intervals(overlap, pd.DataFrame([parent()]))
    bad = p.assign(original_local_end="2024-01-01 13:00")
    with pytest.raises(ValueError, match="local clock"):
        validate_intervals(bad, pd.DataFrame([parent()]))
    source.write_text("changed evidence\n")
    with pytest.raises(ValueError, match="hash changed"):
        validate_intervals(p, pd.DataFrame([parent()]))


def test_rounded_mdl_clue_never_selects_authoritative_ec():
    catalog = pd.DataFrame(
        [
            {
                **parent(),
                "site_code": "USPA",
                "ftir_ec_ugm3": 0.932718,
                "ftir_ec_ugm3_source_rows": "[2]",
                "volume_m3": 7.2,
                "volume_conflict": False,
            }
        ]
    )
    measurements = pd.DataFrame(
        [
            dict(
                Parameter="ChemSpec_EC_PM2.5",
                Site="USPA",
                base_filter_id="USPA-test",
                Concentration=v,
                MDL=0.06375,
                source_row=i,
                CalibrationSetId="ChemSpec_217",
                Concentration_Units="ug/m3",
            )
            for i, v in enumerate([0.93, 0.06])
        ]
    )
    groups, rows = trace_ec_provenance(catalog, measurements, "a" * 64)
    assert groups.resolution_category.iloc[0] == "unresolved_discrepancy"
    assert groups.one_value_matches_reported_mdl_rounding.iloc[0]
    assert json.loads(groups.candidate_mdl_source_rows.iloc[0]) == [1]
    assert not groups.resolved.iloc[0]
    assert np.isnan(groups.selected_ec_ugm3.iloc[0])
    assert len(rows) == 2


def test_staging_preserves_source_rows_and_unknown_observation_flags(tmp_path):
    raw = pd.DataFrame({c: ["1", "2", "NaN"] for c in CHANNEL_COLUMNS})
    raw["Time (UTC)"] = ["2024-01-01T00:00Z", "2024-01-01T00:01Z", "2024-01-01T00:01Z"]
    raw["Datum ID"] = ["0", "1", "1"]
    raw["Session ID"] = "1"
    raw["Serial number"] = "test"
    source = tmp_path / "instrument.csv"
    raw.to_csv(source, index=False)
    meta = stage_instrument_csv(source, tmp_path / "stage", "JPL", chunksize=2)
    frame = pd.read_parquet(meta["staged_file"])
    assert frame.source_row.tolist() == [0, 1, 2]
    assert frame["IR BCc"].tolist() == ["1", "2", "NaN"]
    assert frame.is_observed.isna().all()
    assert meta["duplicate_timestamp_rows"] == 1
    assert meta["duplicate_acquisition_ids"] == 1
    assert meta["finite_native_rows"]["IR"] == 2
    assert stage_instrument_csv(source, tmp_path / "stage", "JPL") == meta


def test_reviewed_schedule_and_observation_evidence_can_admit_a_defensible_subset(tmp_path):
    sys.path.insert(0, str(SCRIPTS.parent / "workflows"))
    from build_active_interval_matches import build_matches
    from active_interval_matching import CoveragePolicy

    raw = pd.DataFrame({c: ["1000", "2000"] for c in CHANNEL_COLUMNS})
    raw["Time (UTC)"] = ["2024-01-01T00:00Z", "2024-01-01T00:01Z"]
    source = tmp_path / "instrument.csv"
    raw.to_csv(source, index=False)
    meta = stage_instrument_csv(source, tmp_path / "stage", "JPL")
    records = processing_records([meta], ["JPL"])
    observed = pd.read_parquet(meta["staged_file"])
    observed["is_observed"] = True  # Synthetic acquisition log documents both rows.
    documented = tmp_path / "documented_observations.parquet"
    observed.to_parquet(documented, index=False)
    evidence = tmp_path / "review.md"
    evidence.write_text("Synthetic test evidence: observations, units and grid verified.")
    update = dict(
        stream_id="JPL:IR:timestamped_csv",
        source_file_hash=meta["source_file_hash"],
        provenance_evidence_file=str(evidence),
        provenance_evidence_hash=sha256(evidence.read_bytes()).hexdigest(),
        resolution_reason="Synthetic fixture only",
        staged_file=str(documented),
        staged_file_hash=sha256(documented.read_bytes()).hexdigest(),
        cadence_verified=True,
        timestamp_role_verified=True,
        observation_provenance_verified=True,
        processing_history_verified=True,
        units_verified=True,
        evidence_scope="full_export",
        scope_justification="Synthetic evidence covers this complete two-row export",
    )
    updates = tmp_path / "evidence.json"
    updates.write_text(json.dumps([update]))
    records = apply_processing_evidence(records, updates)
    c = pd.DataFrame(
        [{**parent(), "ftir_ec_ugm3_conflict": False, "ftir_ec_ugm3_units_valid": True}]
    )
    p = periods([("2024-01-01T00:00Z", "2024-01-01T00:02Z")])
    p["source_file"] = str(evidence)
    p["source_file_hash"] = update["provenance_evidence_hash"]
    p["source_row"] = 0
    p["original_local_start"] = "2024-01-01 00:00"
    p["original_local_end"] = "2024-01-01 00:02"
    p["timezone"] = "UTC"
    p["collection_mode"] = "documented_continuous"
    p["schedule_evidence_type"] = "synthetic_run_log"
    p["resolution_reason"] = "Synthetic continuous operation"
    p = validate_intervals(p, c)
    matches, segments = build_matches(c, p, records, CoveragePolicy())
    assert matches.eligible_interval_ebc_comparison.sum() == 1
    ir = matches.set_index("channel").loc["IR"]
    assert ir.mean_valid_observed_ebc_ugm3 == 1.5
    assert not ir.eligible_absorption_comparison
    assert ir.eligible_ec_ratio_analysis  # FTIR is separate from unresolved ChemSpec.
    assert segments.set_index("channel").loc["IR", "valid_observed_coverage_fraction"] == 1.0
    assert '"source_rows": [0, 1]' in segments.set_index("channel").loc["IR", "raw_source_links"]


def test_bounded_evidence_cannot_certify_other_periods_or_sessions():
    p = periods([("2024-01-01T00:00Z", "2024-01-01T00:02Z")])
    frame = pd.DataFrame(
        {"IR": 1000.0, "observed": True, "Session ID": "one"},
        index=pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC"),
    )
    c = replace(
        contract(),
        evidence_scope="bounded_period",
        scope_start_utc="2024-01-01T00:00Z",
        scope_end_utc="2024-01-01T00:02Z",
        scope_session_ids_json='["one"]',
        scope_justification="Evidence applies only to session one in this two-minute period",
    )
    _, inside = summarize_active_segments(frame, p, "IR", c, "observed")
    assert inside["valid_observed_coverage_fraction"] == 1
    wrong_session = frame.assign(**{"Session ID": "two"})
    _, outside = summarize_active_segments(wrong_session, p, "IR", c, "observed")
    assert outside["input_availability_fraction"] == 1
    assert np.isnan(outside["observed_coverage_fraction"])
    assert np.isnan(outside["mean_input_ebc_ugm3"])
    later = periods([("2024-01-02T00:00Z", "2024-01-02T00:02Z")])
    frame.index = frame.index + pd.Timedelta(days=1)
    _, outside = summarize_active_segments(frame, later, "IR", c, "observed")
    assert np.isnan(outside["valid_observed_coverage_fraction"])
    assert not eligibility(parent(), outside, c, schedule_verified=True)[
        "eligible_interval_ebc_comparison"
    ]


def test_affirmative_flags_without_reviewed_scope_do_not_certify_coverage():
    p = periods([("2024-01-01T00:00Z", "2024-01-01T00:02Z")])
    frame = pd.DataFrame(
        {"IR": 1000.0, "observed": True},
        index=pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC"),
    )
    _, summary = summarize_active_segments(
        frame, p, "IR", replace(contract(), evidence_scope="unreviewed"), "observed"
    )
    assert np.isnan(summary["observed_coverage_fraction"])
    with pytest.raises(ValueError, match="scope justification"):
        replace(contract(), scope_justification="")


def test_continuous_quarters_expose_concentrated_gap_without_new_exclusion():
    p = periods([("2024-01-01T00:00Z", "2024-01-01T04:00Z")])
    frame = pd.DataFrame(
        {"IR": 1000.0, "observed": True},
        index=pd.date_range("2024-01-01", periods=240, freq="min", tz="UTC"),
    )
    frame.loc[frame.index[180:], "observed"] = False
    segments, summary = summarize_active_segments(frame, p, "IR", contract(), "observed")
    quarters = json.loads(segments.quarter_coverage_json.iloc[0])
    assert [q["valid_observed_coverage_fraction"] for q in quarters] == [1, 1, 1, 0]
    assert all(q["diagnostic_only"] for q in quarters)
    assert summary["valid_observed_coverage_fraction"] == 0.75
    assert summary["minimum_segment_valid_observed_coverage"] == 0.75
    assert eligibility(parent(), summary, contract(), schedule_verified=True)[
        "eligible_interval_ebc_comparison"
    ]
