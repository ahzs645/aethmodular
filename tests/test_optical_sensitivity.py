"""Guard units, physical identities, source separation and denominator handling."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))

from optical_sensitivity import (  # noqa: E402
    add_mac_scenarios, direct_sensitivity_ledger, effective_ratio_summary,
    ethiopia_overlap_summary, fabs_to_equivalent_bc, scenario_summary,
)


def inputs():
    return pd.DataFrame({
        "id": ["ETAD-0001", "ETAD-0002", "ETAD-0003", "ETBI-0001", "ETBI-0002", "ETBI-0003"],
        "code": ["ETAD"] * 3 + ["ETBI"] * 3,
        "date": ["2025-11-01", "2025-11-05", "2025-11-08", "2025-11-05", "2025-11-06", "2025-11-09"],
        "ec_ftir_ugm3": [1., 0., 2., 0.5, 2., 0.05],
        "hips_fabs_Mm1": [10., 20., np.nan, 10., 20., 1.],
        "bc_public_hips_ugm3": [1., 2., 3., 1., 2., 0.1],
        "is_excluded": [False, False, False, False, True, False],
        "exclusion_reason": ["", "", "", "", "known acquisition fault", ""],
    })


def test_fabs_units_and_signed_values():
    np.testing.assert_allclose(fabs_to_equivalent_bc([10, 25, -5, np.inf], 10),
                               [1, 2.5, -0.5, np.nan], equal_nan=True)
    with pytest.raises(ValueError, match="MAC"):
        fabs_to_equivalent_bc([10], 0)


def test_direct_source_and_original_data_preserved():
    source = inputs()
    before = source.copy(deep=True)
    ledger = direct_sensitivity_ledger(source)
    scenarios = add_mac_scenarios(ledger, [10, 20])
    pd.testing.assert_frame_equal(source, before)
    assert len(ledger) == len(source)
    assert len(scenarios) == 2 * len(source)
    extension = scenarios.loc[scenarios.id.eq("ETAD-0003")]
    assert extension.optical_source.eq("public_hips_extension").all()
    assert extension.scenario_equivalent_bc_ugm3.isna().all()
    excluded = scenarios.loc[scenarios.id.eq("ETBI-0002")]
    assert excluded.scenario_equivalent_bc_ugm3.isna().all()
    assert excluded.exclusion_reason.eq("known acquisition fault").all()


def test_zero_ec_retained_for_difference_but_not_ratio():
    ledger = direct_sensitivity_ledger(inputs())
    row = ledger.set_index("id").loc["ETAD-0002"]
    assert row.direct_scenario_eligible
    assert not row.effective_ratio_eligible
    assert np.isnan(row.effective_fabs_per_ftir_ec_m2_g)
    scenarios = add_mac_scenarios(ledger, [10])
    assert scenarios.set_index("id").loc["ETAD-0002", "scenario_ec_minus_optical_ugm3"] == -2


def test_effective_ratio_thresholds_are_inclusive_and_audited():
    summary = effective_ratio_summary(direct_sensitivity_ledger(inputs()), [0.1, 0.5, 1.])
    b = summary.loc[summary.code.eq("ETBI")].set_index("ftir_ec_min_ugm3_inclusive")
    assert b.at[0.5, "n_ratios"] == 1
    assert b.at[0.5, "ratio_median_m2_g"] == 20
    assert b.at[1., "n_ratios"] == 0
    assert np.isnan(b.at[1., "ratio_median_m2_g"])
    assert b.at[0.1, "n_positive_ec_below_threshold"] == 1
    a = summary.loc[summary.code.eq("ETAD")]
    assert a.n_nonpositive_ec_denominator.eq(1).all()


def test_missing_ec_denominators_have_their_own_count():
    source = inputs()
    source.loc[0, "ec_ftir_ugm3"] = np.nan
    ledger = direct_sensitivity_ledger(source)
    assert not ledger.set_index("id").at["ETAD-0001", "direct_scenario_eligible"]
    summary = effective_ratio_summary(ledger, [0.5]).set_index("code")
    assert summary.at["ETAD", "n_direct_hips_missing_ec"] == 1
    assert summary.at["ETAD", "n_nonpositive_ec_denominator"] == 1


def test_summary_direction_and_counts():
    summary = scenario_summary(add_mac_scenarios(direct_sensitivity_ledger(inputs()), [10, 20]))
    a = summary.loc[summary.code.eq("ETAD")].set_index("scenario_mac_m2_g")
    assert a.at[10, "n_direct_pairs"] == 2
    assert a.at[10, "n_public_extension_pairs_not_used"] == 1
    assert a.at[10, "n_ftir_below_optical"] == 1
    assert a.at[10, "n_ftir_equal_optical"] == 1
    assert a.at[20, "n_ftir_above_optical"] == 1
    assert a.at[10, "median_ec_minus_optical_ugm3"] == -1


def test_duplicate_physical_filter_ids_fail():
    frame = inputs()
    frame.loc[1, "id"] = frame.loc[0, "id"]
    with pytest.raises(ValueError, match="duplicate physical-filter"):
        direct_sensitivity_ledger(frame)


def test_common_window_determined_from_eligible_direct_measurements():
    scenarios = add_mac_scenarios(direct_sensitivity_ledger(inputs()), [10])
    summary, info = ethiopia_overlap_summary(scenarios)
    assert info["start"] == "2025-11-05"
    assert info["end"] == "2025-11-05"
    assert summary.n_direct_pairs.tolist() == [1, 1]
    assert summary.cohort.eq("shared_ethiopia_collection_date_window").all()


def test_invalid_scenarios_fail_instead_of_silently_repeating():
    ledger = direct_sensitivity_ledger(inputs())
    for values in ([10, 10], [0], [np.nan], []):
        with pytest.raises(ValueError):
            add_mac_scenarios(ledger, values)
    with pytest.raises(ValueError):
        effective_ratio_summary(ledger, [-0.1])
