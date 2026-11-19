"""Safe failures of the existing, opt-in loading diagnostic; no archive code runs."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.processors.site_corrections import SiteCorrections


def loading_fixture(n=160):
    t = np.linspace(0, 1, n)
    atn = 4 + 36 * t
    truth = 1000 + 100 * np.cos(2 * np.pi * t)
    return pd.DataFrame(
        {
            "IR ATN1": atn,
            "IR BC1": truth * (1 - 0.012 * atn),
            "IR K": np.full(n, 0.012),
            "IR BCc": truth,
        }
    )


def test_existing_finite_loading_result_and_original_columns_are_preserved():
    data = loading_fixture()
    before = data.copy(deep=True)
    correction = SiteCorrections("ETAD", verbose=False)
    # Recorded from the existing algorithm before the safe-failure changes.
    assert correction._optimize_k_value(data) == pytest.approx(
        0.011909830056250525, abs=1e-15
    )
    out = correction.apply_corrections(data)
    np.testing.assert_allclose(out["IR BCc_manual"], data["IR BCc"])
    assert out["IR BCc_optimized"].iloc[0] == pytest.approx(1099.583405992159)
    assert out["IR BCc_optimized"].iloc[-1] == pytest.approx(1092.4227921753006)
    assert_frame_equal(out[data.columns], before)
    assert_frame_equal(data, before)
    assert correction.corrections_applied == ["Ethiopia_IR_loading_compensation"]


def test_insufficient_finite_pairs_raise_instead_of_selecting_a_bound():
    data = loading_fixture(5)
    correction = SiteCorrections(verbose=False)
    with pytest.raises(ValueError, match="at least 10 finite"):
        correction._optimize_k_value(data)
    data = loading_fixture(20)
    data.loc[:10, "IR BC1"] = np.inf
    with pytest.raises(ValueError, match="at least 10 finite"):
        correction._optimize_k_value(data)


@pytest.mark.parametrize("column,value", [("IR ATN1", 4.0), ("IR BC1", 100.0)])
def test_constant_input_has_no_reported_optimum(column, value):
    data = loading_fixture()
    data[column] = value
    with pytest.raises(ValueError, match="varying BC1 and ATN1"):
        SiteCorrections(verbose=False)._optimize_k_value(data)


def test_no_loaded_observations_or_a_flat_objective_cannot_identify_k():
    correction = SiteCorrections(verbose=False)
    data = loading_fixture()
    data["IR ATN1"] = np.linspace(0, 3, len(data))
    with pytest.raises(ValueError, match="ATN1 > 3"):
        correction._optimize_k_value(data)

    # Both inputs vary, but k acts only on zero-valued loaded observations.
    # The objective therefore stays constant over the whole valid domain.
    data = pd.DataFrame(
        {
            "IR ATN1": np.r_[np.linspace(0, 3, 12), np.linspace(4, 40, 12)],
            "IR BC1": np.r_[np.linspace(1, 10, 12), np.zeros(12)],
        }
    )
    with pytest.raises(ValueError, match="objective is constant"):
        correction._optimize_k_value(data)


@pytest.mark.parametrize(
    "bounds",
    [(np.nan, 0.015), (0.015, 0.005), (0.01, 0.01), (0.005, np.inf), (-1e308, 1e308), (0.01,)],
)
def test_invalid_search_intervals_fail_before_search(bounds):
    with pytest.raises(ValueError, match="bounds|interval"):
        SiteCorrections(verbose=False)._optimize_k_value(loading_fixture(), k_range=bounds)


@pytest.mark.parametrize("tolerance", [0, -1e-4, np.nan, np.inf, 0.01])
def test_invalid_search_tolerance_cannot_return_an_unevaluated_midpoint(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        SiteCorrections(verbose=False)._optimize_k_value(
            loading_fixture(), tolerance=tolerance
        )


def test_search_rejects_a_loading_pole_without_changing_its_population():
    data = loading_fixture()
    data["IR ATN1"] = np.linspace(70, 100, len(data))
    with pytest.raises(ValueError, match="nonpositive loading denominator"):
        SiteCorrections(verbose=False)._optimize_k_value(data)


def test_failed_optimization_uses_a_finite_declared_median_without_bogus_success(capsys):
    data = loading_fixture(5)
    correction = SiteCorrections("ETAD", verbose=True)
    out = correction.apply_corrections(data)
    output = capsys.readouterr().out
    assert "optimization failed" in output
    assert "Using median K correction" in output
    assert "optimal K:" not in output
    assert "IR BCc_optimized" not in out
    np.testing.assert_allclose(out["IR BCc_corrected"], out["IR BCc_manual"])
    assert correction.corrections_applied == ["Ethiopia_IR_loading_compensation"]


def test_manual_candidate_preserves_bypass_and_marks_unsafe_values_unavailable():
    data = pd.DataFrame(
        {
            "IR ATN1": [2.0, 4.0, 100.0, 120.0, np.nan, np.inf],
            "IR BC1": [-2.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "IR K": 0.01,
            "IR BCc": 7.0,
        }
    )
    correction = SiteCorrections("ETAD", verbose=False)
    out = correction._apply_manual_loading_compensation_ethiopia(data, optimize_k=False)
    assert out["IR BCc_corrected"].iloc[0] == -2.0
    assert out["IR BCc_corrected"].iloc[1] == pytest.approx(10 / 0.96)
    assert out["IR BCc_corrected"].iloc[2:].isna().all()
    assert np.isfinite(out["IR BCc_corrected"].dropna()).all()
    assert_frame_equal(out[data.columns], data)
    assert correction.corrections_applied == ["Ethiopia_IR_loading_compensation"]


@pytest.mark.parametrize("case", ["missing_columns", "nonfinite_k", "all_denominators_unsafe", "all_bc_missing"])
def test_no_usable_correction_is_not_reported_as_applied(case, capsys):
    if case == "missing_columns":
        # A pre-existing derived column also cannot establish this call applied anything.
        data = pd.DataFrame({"IR BC1": [1.0, 2.0], "IR BCc_corrected": [3.0, 4.0]})
    elif case == "nonfinite_k":
        data = pd.DataFrame({"IR BC1": [1.0, 2.0], "IR ATN1": [4.0, 5.0], "IR K": [np.nan, np.inf]})
    elif case == "all_denominators_unsafe":
        data = pd.DataFrame({"IR BC1": [1.0, 2.0], "IR ATN1": [100.0, 120.0], "IR K": 0.01})
    else:
        data = loading_fixture()
        data["IR BC1"] = np.nan
    correction = SiteCorrections("ETAD", verbose=True)
    out = correction.apply_corrections(data)
    assert correction.corrections_applied == []
    assert "optimal K:" not in capsys.readouterr().out
    assert_frame_equal(out, data)


def test_applied_report_describes_this_call_not_an_earlier_success():
    correction = SiteCorrections("ETAD", verbose=False)
    correction.apply_corrections(loading_fixture())
    assert correction.corrections_applied
    correction.apply_corrections(pd.DataFrame({"IR BC1": [1.0]}))
    assert correction.corrections_applied == []
    correction.apply_ethiopia_corrections(loading_fixture())
    assert correction.corrections_applied
    correction.apply_ethiopia_corrections(pd.DataFrame({"IR BC1": [1.0]}))
    assert correction.corrections_applied == []


def test_rerun_with_median_fallback_removes_stale_optimized_results():
    correction = SiteCorrections("ETAD", verbose=False)
    first = correction.apply_corrections(loading_fixture()).head(5)
    before = first.copy(deep=True)
    assert "IR BCc_optimized" in first
    second = correction.apply_corrections(first)
    assert "IR BCc_optimized" not in second
    assert "IR denominator_optimized" not in second
    np.testing.assert_allclose(second["IR BCc_corrected"], second["IR BCc_manual"])
    assert_frame_equal(first, before)
    assert_frame_equal(second[loading_fixture().columns], first[loading_fixture().columns])


def test_skipped_rerun_is_unchanged_and_reports_no_new_correction():
    correction = SiteCorrections("ETAD", verbose=False)
    first = correction.apply_corrections(loading_fixture())
    first["IR K"] = np.nan
    first["unrelated_result"] = 42.0
    before = first.copy(deep=True)
    second = correction.apply_corrections(first)
    assert correction.corrections_applied == []
    assert_frame_equal(second, before)
    assert_frame_equal(first, before)
