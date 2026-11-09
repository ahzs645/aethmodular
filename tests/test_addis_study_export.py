"""Scientific contracts for the public-snapshot Addis study export."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_addis_study", ROOT / "gallery/data/export_addis_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def profile(start="2022-04-01T00:00:00+03:00", end="2022-04-03T00:00:00+03:00", events=None):
    result = json.loads(study.PROFILE.read_bytes())
    result["start_local"], result["end_local_exclusive"] = start, end
    result["event_dates_local"] = {} if events is None else events
    return result


def snapshot(values, first="2022-04-01 00:00"):
    data = {"site": "ETAD", "first": first, "offset": 3.0, "unit": "ng/m3", "IR": values}
    last = pd.Timestamp(first) + pd.Timedelta(hours=len(values) - 1)
    metadata = {
        "generated": "2026-10-27T21:10:21+00:00", "rules": {"min_minutes": 30},
        "sites": {"ETAD": {
            "site": "Addis_Ababa", "resolution": "minute",
            "hourly_file": "Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv",
            "offset": 3.0, "first": first, "last": last.strftime("%Y-%m-%d %H:%M"),
            "hours": len(values),
            "valid_hours": int(sum(value is not None and np.isfinite(value) for value in values)),
        }},
    }
    return data, metadata


def build(values, *, first="2022-04-01 00:00", paper=None):
    return study.build_study(*snapshot(values, first), paper or profile(), generated="2026-11-09T00:00:00Z")


def seasons(result, calendar):
    return {row["id"]: row for row in result["calendars"][calendar]["seasons"]}


def test_units_finite_values_and_two_coverage_denominators():
    result = build([1000, None, 3000, 0, -1000, float("inf")], first="2022-04-01 12:00")
    coverage = result["coverage"]
    assert coverage["expected_hours"] == 48
    assert coverage["snapshot_hours"] == 6
    assert coverage["valid_hours"] == 4
    assert coverage["outside_snapshot_hours"] == 42
    assert coverage["missing_within_snapshot_hours"] == 2
    assert coverage["coverage_percent"] == pytest.approx(100 * 4 / 48)
    assert coverage["snapshot_coverage_percent"] == pytest.approx(100 * 4 / 6)
    assert result["summary"]["mean_bc_ug_m3"] == 0.75
    assert result["summary"]["max_bc_ug_m3"] == 3
    assert result["source"]["additional_value_screen"] is False
    assert result["source"]["min_valid_ir_minutes"] == 30
    assert coverage["first_valid_local"] == "2022-04-01T12:00:00+03:00"
    assert coverage["last_valid_local"] == "2022-04-01T16:00:00+03:00"
    assert result["daily"][0]["mean_bc_ug_m3"] == 0.75
    assert result["daily"][1]["mean_bc_ug_m3"] is None
    json.dumps(result, allow_nan=False)


def test_study_window_excludes_both_outside_values_and_uses_exclusive_end():
    result = build(
        [99000, 1000, 3000, 77000], first="2022-03-31 23:00",
        paper=profile(end="2022-04-01T02:00:00+03:00"),
    )
    assert result["summary"]["expected_hours"] == 2
    assert result["summary"]["snapshot_hours"] == 2
    assert result["summary"]["valid_hours"] == 2
    assert result["summary"]["mean_bc_ug_m3"] == 2
    assert result["summary"]["end_local_exclusive"] == "2022-04-01T02:00:00+03:00"
    assert result["source"]["first_local"] == "2022-03-31T23:00:00+03:00"


def test_daily_bins_and_diurnal_support_use_addis_local_hours():
    result = build([1000, 3000], first="2022-04-01 23:00")
    assert [row["mean_bc_ug_m3"] for row in result["daily"]] == [1, 3]
    assert result["daily"][1]["start_local"] == "2022-04-02T00:00:00+03:00"
    hourly = {row["hour"]: row for row in result["calendars"]["belg_feb"]["diurnal"] if row["season_id"] == "belg"}
    assert hourly[23]["mean_bc_ug_m3"] == 1
    assert hourly[0]["mean_bc_ug_m3"] == 3
    assert hourly[0]["valid_hours"] == 1
    assert hourly[0]["expected_hours"] == 2
    assert hourly[1]["mean_bc_ug_m3"] is None


def test_means_are_hour_weighted_and_sparse_daily_values_remain_explicit():
    result = build([1000] + [None] * 23 + [3000] * 24)
    assert [row["mean_bc_ug_m3"] for row in result["daily"]] == [1, 3]
    assert [row["valid_hours"] for row in result["daily"]] == [1, 24]
    assert result["summary"]["mean_bc_ug_m3"] == pytest.approx(73 / 25)
    assert result["monthly"][0]["mean_bc_ug_m3"] == pytest.approx(73 / 25)


def test_february_uses_named_calendar_and_season_specific_hour_support():
    paper = profile("2023-02-28T00:00:00+03:00", "2023-03-02T00:00:00+03:00")
    result = build([10000, 20000], first="2023-02-28 23:00", paper=paper)
    named = seasons(result, "belg_feb")
    legacy = seasons(result, "dry_feb")
    assert named["bega"]["mean_bc_ug_m3"] is None
    assert named["belg"]["mean_bc_ug_m3"] == 15
    assert named["belg"]["expected_hours"] == 48
    assert legacy["bega"]["mean_bc_ug_m3"] == 10
    assert legacy["belg"]["mean_bc_ug_m3"] == 20
    assert legacy["bega"]["expected_hours"] == legacy["belg"]["expected_hours"] == 24
    assert 2 in named["belg"]["months"] and 2 not in named["bega"]["months"]
    assert 2 in legacy["bega"]["months"] and 2 not in legacy["belg"]["months"]
    assert all(not row["festival_exclusions_applied"] for row in result["calendars"].values())


def test_complete_calendar_has_empty_months_and_leap_day_without_fabricated_data():
    paper = json.loads(study.PROFILE.read_bytes())
    result = build([1000], first="2022-04-12 13:00", paper=paper)
    assert result["coverage"]["expected_hours"] == 26304
    assert result["coverage"]["valid_hours"] == 1
    assert result["coverage"]["outside_snapshot_hours"] == 26303
    assert len(result["monthly"]) == 36
    assert len(result["daily"]) == 1096
    assert sum(row["expected_hours"] for row in result["daily"]) == 26304
    by_month = {row["start_local"][:7]: row for row in result["monthly"]}
    assert by_month["2024-02"]["expected_hours"] == 29 * 24
    assert by_month["2024-09"]["valid_hours"] == 0
    assert by_month["2024-09"]["mean_bc_ug_m3"] is None
    assert by_month["2024-09"]["snapshot_coverage_percent"] is None
    assert result["monthly"][-1]["end_local_exclusive"] == "2025-04-01T00:00:00+03:00"
    assert sum(row["expected_hours"] for row in seasons(result, "belg_feb").values()) == 26304
    assert all(event["unavailable"] for event in result["events"])
    assert result["paper"]["reported"]["jacros"]["bc_mean_ug_m3"] == 7.67
    assert result["summary"]["mean_bc_ug_m3"] == 1
    assert result["paper_reproduction_complete"] is False
    assert result["independent_validation"] is False
    assert result["capabilities"]["central"]["available"] is False
    assert result["capabilities"]["source_fractions"]["available"] is False


@pytest.mark.parametrize("name,day,start_hour,mean,qualifier", [
    ("Meskel", "2022-09-26", 12, 23.5, "Figure 6"),
    ("Hidar Sitaten", "2022-11-21", 0, 11.5, "archived"),
])
def test_event_windows_are_explicit_twenty_four_local_hours(name, day, start_hour, mean, qualifier):
    beginning = pd.Timestamp(day).tz_localize(study.ZONE)
    paper = profile(beginning.isoformat(), (beginning + pd.Timedelta(days=2)).isoformat(), {name: [day]})
    result = build([1000 * hour for hour in range(48)], first=f"{day} 00:00", paper=paper)
    event = result["events"][0]
    assert event["start_local"] == (beginning + pd.Timedelta(hours=start_hour)).isoformat()
    assert event["end_local_exclusive"] == (beginning + pd.Timedelta(hours=start_hour + 24)).isoformat()
    assert event["available_hours"] == event["expected_hours"] == 24
    assert event["mean_bc_ug_m3"] == mean
    assert event["incomplete"] is False
    assert event["unavailable"] is False
    assert qualifier in event["window_rule"]
    assert [row["elapsed_hour"] for row in event["trace"]] == list(range(24))
    assert event["trace"][0]["bc_ug_m3"] == start_hour
    assert result["summary"]["valid_hours"] == 48  # Festival hours remain in the climatology.


def test_sparse_event_has_available_hour_mean_without_eighteen_hour_gate():
    paper = json.loads(study.PROFILE.read_bytes())
    result = build([None] * 12 + [1000] * 17 + [None] * 7, first="2022-09-26 00:00", paper=paper)
    events = {event["id"]: event for event in result["events"]}
    event = events["meskel-2022"]
    assert event["available_hours"] == 17
    assert event["snapshot_hours"] == event["expected_hours"] == 24
    assert event["mean_bc_ug_m3"] == 1
    assert event["incomplete"] is True
    assert event["unavailable"] is False
    assert event["trace"][-1]["bc_ug_m3"] is None
    assert events["meskel-2024"]["available_hours"] == 0
    assert events["meskel-2024"]["mean_bc_ug_m3"] is None
    assert events["hidar-sitaten-2024"]["unavailable"] is True
    assert all(row["bc_ug_m3"] is None for row in events["meskel-2024"]["trace"])


@pytest.mark.parametrize("field,value,match", [
    ("site", "ETAD-JACROS", "ETAD"),
    ("unit", "ug/m3", "ng/m3"),
    ("offset", 2, "offset"),
    ("first", "2022-04-01 00:30", "hour"),
    ("first", "2022-04-01T00:00:00+03:00", "naive"),
    ("IR", [True], "numeric"),
    ("IR", ["1000"], "numeric"),
])
def test_ambiguous_identity_units_or_time_fail_closed(field, value, match):
    data, metadata = snapshot([1000])
    data[field] = value
    with pytest.raises(ValueError, match=match):
        study.build_study(data, metadata, profile())


@pytest.mark.parametrize("field,value", [
    ("hourly_file", "Raw/Central_MA350.csv"),
    ("hours", 2), ("valid_hours", 2), ("last", "2022-04-02 00:00"),
])
def test_companion_metadata_cannot_silently_reidentify_or_change_coverage(field, value):
    data, metadata = snapshot([1000])
    metadata["sites"]["ETAD"][field] = value
    with pytest.raises(ValueError):
        study.build_study(data, metadata, profile())


def test_exact_input_bytes_are_hashed_and_output_is_strict_json(tmp_path):
    data, metadata = snapshot([1000, None, 3000])
    directory = tmp_path / "data"
    hourly = directory / "overview/hourly/ETAD.json"
    hourly.parent.mkdir(parents=True)
    paths = {"input": hourly, "metadata": directory / "overview/index.json", "profile": tmp_path / "profile.json"}
    for key, payload in (("input", data), ("metadata", metadata), ("profile", profile())):
        paths[key].write_text(json.dumps(payload, indent=2) + "\n")
    before = copy.deepcopy(data)
    output = tmp_path / "addis-study.json"
    result = study.export_study(directory, paths["profile"], output)
    for key, path in {**paths, "source_review": study.SOURCE_REVIEW, "config": study.CONFIG, "exporter": Path(study.__file__)}.items():
        assert result["source"][f"{key}_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(output.read_text()) == result
    assert json.loads(hourly.read_text()) == before
    assert result["source"]["source_unit"] == "ng/m3"
    assert result["deployment"]["unit"] == "ug/m3"
    assert result["summary"]["mean_bc_ug_m3"] == 2
    assert result["source_review"]["applied_to_observations"] is False


def test_loading_illustration_is_separate_and_preserves_invalid_denominators():
    source = {"schema_version": 1, "loading_candidates": [{"id": "example", "k": 0.02}]}
    before = copy.deepcopy(source)
    review = study.build_source_review(source)
    assert source == before
    assert review["applied_to_observations"] is False
    response = review["loading_sensitivity"]["series"][0]
    assert response["singular_atn1"] == 50
    points = {row["atn1"]: row for row in response["points"]}
    assert points[0]["multiplier"] == 1
    assert points[25]["multiplier"] == 2
    assert points[49]["multiplier"] == pytest.approx(50)
    for atn in range(50, 61):
        assert points[atn]["multiplier"] is None
        assert points[atn]["denominator_valid"] is False
    json.dumps(review, allow_nan=False)


@pytest.mark.parametrize("candidate", [0, -0.1, True, "0.01", float("nan"), float("inf")])
def test_loading_illustration_rejects_invalid_candidate(candidate):
    with pytest.raises(ValueError, match="finite and positive"):
        study.build_source_review({"schema_version": 1, "loading_candidates": [{"id": "bad", "k": candidate}]})


def test_archived_sensitivity_keeps_distinct_settings_and_explicit_provenance():
    source = json.loads(study.SOURCE_REVIEW.read_bytes())
    review = study.build_source_review(source)
    assert review["historical_regressions"] == source["historical_regressions"]
    assert review["optical_sensitivity"] == source["optical_sensitivity"]
    rows = review["optical_sensitivity"]["rows"]
    assert len(rows) == 18
    assert len({(r["site"], r["alpha_fossil"], r["alpha_biomass"]) for r in rows}) == 18
    paper_settings = [r for r in rows if r["alpha_fossil"] == 1 and r["alpha_biomass"] == 2]
    pipeline_settings = [r for r in rows if r["alpha_fossil"] == 1 and r["alpha_biomass"] == 1.8]
    assert {r["site"] for r in paper_settings} == {"Central", "Jacros"}
    assert all(r["biomass_percent"] == 5.8 for r in paper_settings)
    assert all(r["biomass_percent"] == 7.7 for r in pipeline_settings)
    assert all(len(s["sha256"]) == 64 for s in review["sources"])
