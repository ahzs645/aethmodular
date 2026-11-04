"""Check spectral response identity, shared cohorts and response-free adjustment."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))

from spectral_response_screen import (  # noqa: E402
    TARGETS, average_physical_scans, build_response_ledger,
    nuisance_design, response_associations,
)


def response_fixture():
    rng = np.random.default_rng(2901)
    n = 96
    spectral = rng.normal(size=n)
    optical = rng.normal(size=n)
    dates = pd.date_range("2023-01-01", periods=n, freq="9D")
    frame = pd.DataFrame({
        "base_filter_id": [f"ETAD-{index:04d}" for index in range(n)],
        "date": dates.strftime("%Y-%m-%d"), "lot_group": np.where(np.arange(n) < 48, "248", "251"),
        "pm25_ugm3": rng.uniform(10, 30, n), "volume_m3": rng.uniform(6, 8, n),
        "indicator": spectral, "ec_ftir_ugm3": 5 + spectral + rng.normal(scale=0.01, size=n),
        "hips_fabs_Mm1": 30 + 10 * optical,
        "response_base_eligible": True, "response_base_reason": "eligible",
    })
    frame["ec_minus_hips_ugm3"] = frame.ec_ftir_ugm3 - frame.hips_fabs_Mm1 / 10
    return frame


def test_repeat_scans_average_after_correction_and_keep_provenance():
    scans = pd.DataFrame({
        "MediaId": [10, 10, 20], "SampleAnalysisId": [100, 101, 200],
        "base_filter_id": ["ETAD-0001", "ETAD-0001", "ETAD-0002"],
        "ExternalFilterType": ["PM2.5"] * 3, "lot_group": ["248"] * 3,
        "native_date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
    })
    raw = np.array([[1, 3], [3, 5], [9, 10]])
    corrected = np.array([[0.1, 0.3], [0.5, 0.7], [0.9, 1.0]])
    physical, average_raw, average_corrected = average_physical_scans(scans, raw, corrected)
    assert physical.n_scans.tolist() == [2, 1]
    assert physical.analysis_ids.tolist() == ["100;101", "200"]
    np.testing.assert_allclose(average_raw, [[2, 4], [9, 10]])
    np.testing.assert_allclose(average_corrected, [[0.3, 0.5], [0.9, 1.]])
    scans.loc[1, "lot_group"] = "251"
    with pytest.raises(ValueError, match="conflicting lot_group"):
        average_physical_scans(scans, raw, corrected)


def test_all_three_responses_use_exactly_same_complete_feature_cohort():
    frame = response_fixture()
    frame.loc[0, "indicator"] = np.nan
    frame.loc[1, "hips_fabs_Mm1"] = np.nan
    result, membership = response_associations(frame, ("indicator",))
    assert result.n_physical_filters.tolist() == [94, 94, 94]
    assert result.cohort_id_sha256.nunique() == 1
    assert membership.common_cohort_member.sum() == 94
    assert set(result.target) == set(TARGETS)


def test_optical_response_never_enters_FTIR_nuisance_design():
    frame = response_fixture()
    before, _ = response_associations(frame, ("indicator",))
    changed = frame.copy()
    changed["hips_fabs_Mm1"] = 100 * changed.indicator + 1
    changed["ec_minus_hips_ugm3"] = changed.ec_ftir_ugm3 - changed.hips_fabs_Mm1 / 10
    after, _ = response_associations(changed, ("indicator",))
    b = before.set_index("target").loc["ec_ftir_ugm3"]
    a = after.set_index("target").loc["ec_ftir_ugm3"]
    assert a.partial_rank_r == b.partial_rank_r
    assert a.design_columns == b.design_columns
    assert a.partial_rank_r > 0.99


def test_squared_statistics_and_direction():
    frame = response_fixture()
    frame["negative_indicator"] = -frame.indicator
    result, _ = response_associations(frame, ("indicator", "negative_indicator"))
    ec = result.loc[result.target.eq("ec_ftir_ugm3")].set_index("indicator")
    assert ec.at["indicator", "partial_rank_r"] > 0
    assert ec.at["negative_indicator", "partial_rank_r"] < 0
    np.testing.assert_allclose(result.partial_rank_r_squared, result.partial_rank_r**2)
    assert ec.at["indicator", "partial_rank_r_squared"] == pytest.approx(
        ec.at["negative_indicator", "partial_rank_r_squared"]
    )


def test_rank_deficient_control_design_is_reported():
    frame = response_fixture()
    frame["volume_m3"] = 2 * frame.pm25_ugm3
    _, info = nuisance_design(frame)
    assert info["design_rank_deficient"]
    assert info["design_rank"] < info["design_n_columns"]
    result, _ = response_associations(frame, ("indicator",))
    assert result.analysis_status.eq("estimable_with_collinear_controls").all()


def test_constant_feature_is_not_reported_as_zero_association():
    frame = response_fixture()
    frame["indicator"] = 1.
    result, _ = response_associations(frame, ("indicator",))
    assert result.partial_rank_r.isna().all()
    assert result.analysis_status.eq("degenerate_feature_or_target_after_adjustment").all()


def test_duplicate_physical_id_rejected_and_source_frame_preserved():
    frame = response_fixture()
    original = frame.copy(deep=True)
    response_associations(frame, ("indicator",))
    pd.testing.assert_frame_equal(frame, original)
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per physical filter"):
        response_associations(duplicate, ("indicator",))


def test_join_requires_direct_source_date_match_and_retains_exclusion_reason():
    ids = ["ETAD-0001", "ETAD-0002", "ETAD-0003", "ETAD-0004"]
    physical = pd.DataFrame({
        "base_filter_id": ids, "ExternalFilterType": ["PM2.5"] * 4,
        "native_date": pd.to_datetime(["2023-01-01"] * 4), "lot_group": ["248"] * 4,
    })
    chemistry = pd.DataFrame({
        "id": ids, "code": ["ETAD"] * 4,
        "date": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-02"],
        "is_excluded": [False, False, True, False],
        "exclusion_reason": ["", "", "known acquisition fault", ""],
        "ec_ftir_ugm3": [2.] * 4, "hips_fabs_Mm1": [30., np.nan, 30., 30.],
        "bc_public_hips_ugm3": [3.] * 4,
    })
    ledger = build_response_ledger(physical, chemistry).set_index("base_filter_id")
    assert ledger.response_base_eligible.tolist() == [True, False, False, False]
    assert ledger.at[ids[1], "response_base_reason"] == "not_direct_HIPS_absorption"
    assert ledger.at[ids[2], "exclusion_reason"] == "known acquisition fault"
    assert not ledger.at[ids[3], "exact_collection_date_match"]
