"""Prevent calendar ambiguity, repeated-filter leakage, and optical overclaims."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))

from calibration_seasonality import (  # noqa: E402
    PUBLISHED_CONTEXT, complete_year_summary, coverage_summary, february_sensitivity,
    mac_scenario_ledger, season_ledger, seasonal_optical_summary,
    source_calendar_errors, with_published_context,
)


def inputs():
    base = pd.DataFrame({
        "MediaId": [1, 2, 3, 4, 5], "SiteCode": ["ETAD"]*5,
        "base_filter_id": [f"ETAD-{i:04d}" for i in range(1, 6)],
        "native_date": ["2024-01-10", "2024-02-29", "2024-06-15", None, "2024-04-12"],
        "current_date": ["2024-01-10", "2024-02-29", "2024-06-15", None, "2024-04-12"],
        "SamplingEndDate": ["2024-01-11", "2024-03-01", "2024-06-16", None, "2024-04-13"],
        "ExternalFilterType": ["PM2.5"]*3+["FB", "PM2.5"],
        "lot_group": ["248", "251", "251", "248", "unknown"],
        "n_scans": [1, 2, 1, 1, 1], "SampleVolume_m3": [10., 10., 10., np.nan, 10.],
        "volume_valid": [True, True, True, False, True],
        "current_pair_eligible": [True, True, True, False, True],
        "current_optical_source": ["direct_hips_fabs"]*3+[None, "public_hips_extension"],
        "current_optical_bc_ugm3": [1., 2., 3., np.nan, 4.],
        "hips_fabs_Mm1": [10., 20., 30., np.nan, np.nan],
        "current_ec_ftir_ugm3": [1.5, 2.5, 2., np.nan, 3.],
        "prediction_ugm3": [1.5, -1., 2., np.nan, 3.],
        "prediction_loading_ug": [15., -10., 20., np.nan, 30.],
        "prior_problem_filter": [False, True, False, False, False],
        "aggregation_sensitivity": ["all_original_scans"]*5,
    })
    return pd.concat([base.assign(arm="raw"), base.assign(arm="spline")], ignore_index=True)


def test_february_conventions_and_missing_dates_are_explicit():
    frame = inputs()
    original = frame.copy(deep=True)
    ledger = season_ledger(frame)
    pd.testing.assert_frame_equal(frame, original)
    feb = ledger.loc[ledger.base_filter_id.eq("ETAD-0002")]
    assert feb.loc[feb.season_convention.eq("belg_feb"), "season"].eq("Belg (Feb-May, short rains)").all()
    assert feb.loc[feb.season_convention.eq("dry_feb"), "season"].eq("Dry (Oct-Feb)").all()
    assert not feb.loc[feb.season_convention.eq("belg_feb"), "spans_season_boundary"].any()
    assert feb.loc[feb.season_convention.eq("dry_feb"), "spans_season_boundary"].all()
    blank = ledger.loc[ledger.ExternalFilterType.eq("FB")]
    assert blank.season.eq("Undated").all()
    assert not blank.common_arm_direct_eligible.any()
    assert blank.seasonal_pair_reason.str.contains("missing_native_date").all()


def test_physical_filter_counts_do_not_multiply_by_arms_or_scans():
    coverage = coverage_summary(season_ledger(with_published_context(inputs())))
    first = coverage.loc[coverage.season_convention.eq("belg_feb")]
    assert first.n_physical_filters.sum() == 5
    assert first.n_original_scans.sum() == 6
    assert first.n_direct_pairs.sum() == 3
    assert first.n_common_arm_direct_pairs.sum() == 3


def test_incomplete_arm_remains_in_ledger_but_cannot_change_method_cohort():
    frame = inputs()
    frame.loc[(frame.arm == "spline") & (frame.base_filter_id == "ETAD-0001"), "prediction_ugm3"] = np.nan
    ledger = season_ledger(frame)
    jan = ledger.loc[ledger.base_filter_id.eq("ETAD-0001")]
    assert len(jan) == 4
    assert not jan.common_arm_direct_eligible.any()
    assert jan.loc[jan.arm.eq("raw"), "seasonal_pair_reason"].str.contains("not_complete_across_all_arms").all()
    assert ledger.loc[ledger.base_filter_id.eq("ETAD-0002"), "common_arm_direct_eligible"].all()


def test_duplicates_inconsistent_comparators_and_non_ethiopian_sites_fail():
    frame = inputs()
    with pytest.raises(ValueError, match="duplicate physical filters"):
        season_ledger(pd.concat([frame, frame.iloc[[0]]]))
    frame.loc[5, "current_optical_bc_ugm3"] = 999.
    with pytest.raises(ValueError, match="inconsistent physical metadata"):
        season_ledger(frame)
    frame = inputs().assign(SiteCode="ABCD1")
    with pytest.raises(ValueError, match="non-Ethiopian"):
        season_ledger(frame)


def test_mac_scenarios_preserve_signed_predictions_and_cannot_use_extension():
    ledger = season_ledger(inputs(), ["belg_feb"])
    scenarios = mac_scenario_ledger(ledger, [10, 20])
    negative = scenarios.loc[scenarios.base_filter_id.eq("ETAD-0002")]
    assert negative.prediction_ugm3.eq(-1).all()
    assert negative.loc[negative.scenario_mac_m2_g.eq(10), "scenario_ftir_minus_optical_ugm3"].eq(-3).all()
    assert negative.loc[negative.scenario_mac_m2_g.eq(20), "scenario_ftir_minus_optical_ugm3"].eq(-2).all()
    assert scenarios.loc[scenarios.base_filter_id.eq("ETAD-0005"), "scenario_optical_bc_ugm3"].isna().all()
    for macs in ([10, 10], [0], [np.nan], []):
        with pytest.raises(ValueError, match="MAC"):
            mac_scenario_ledger(ledger, macs)


def test_context_is_not_an_old_arm_prediction_or_residual():
    frame = inputs()
    frame["experiment_ftir_minus_hips_ugm3"] = 42.
    context = with_published_context(frame)
    published = context.loc[context.arm.eq(PUBLISHED_CONTEXT)].set_index("base_filter_id")
    assert len(published) == 5
    assert published.at["ETAD-0002", "prediction_ugm3"] == 2.5
    assert published.at["ETAD-0002", "experiment_ftir_minus_hips_ugm3"] == .5
    assert published.prediction_loading_ug.isna().all()
    with pytest.raises(ValueError, match="already present"):
        with_published_context(context)


def test_february_changes_reported_cohort_and_kiremt_stays_same():
    summary = seasonal_optical_summary(mac_scenario_ledger(season_ledger(inputs()), [10]))
    sensitivity = february_sensitivity(summary)
    primary = sensitivity.loc[(sensitivity.arm == "raw") & (sensitivity.cohort == "all_common_direct_pairs")].set_index("season_family")
    assert primary.at["Belg", "delta_n_belg_feb_minus_dry_feb"] == 1
    assert primary.at["Dry/Bega", "delta_n_belg_feb_minus_dry_feb"] == -1
    assert primary.at["Kiremt", "delta_n_belg_feb_minus_dry_feb"] == 0
    assert primary.at["Kiremt", "delta_median_disagreement_belg_feb_minus_dry_feb_ugm3"] == 0


def test_dates_and_volume_affect_eligibility_without_dropping_rows():
    frame = inputs()
    frame.loc[frame.base_filter_id.eq("ETAD-0001"), "current_date"] = "2024-01-11"
    frame.loc[frame.base_filter_id.eq("ETAD-0002"), "volume_valid"] = False
    ledger = season_ledger(frame)
    assert len(ledger) == 20
    invalid = ledger.loc[ledger.base_filter_id.isin(["ETAD-0001", "ETAD-0002"])]
    assert not invalid.common_arm_direct_eligible.any()


def test_source_calendar_errors_are_labelled_months_with_exact_filter_counts():
    frame = pd.DataFrame({
        "arm": ["raw"]*4, "AnalysisId": [1, 2, 3, 4], "Site": ["A", "B", "A", "B"],
        "SampleDate": ["2023-01-10", "2023-01-11", "2023-06-10", "2023-06-11"],
        "TOR_EC_loading_ug": [10., 20., 10., 20.], "prediction_loading_ug": [11., 21., 8., 18.],
    })
    result = source_calendar_errors(frame)
    jan = result.loc[result.bin_type.eq("calendar_month") & result.bin_number.eq(1)].iloc[0]
    assert jan.n_filters == 2
    assert jan.n_sites == 2
    assert jan.rmse_ug_filter == 1
    assert jan.mean_bias_ug_filter == 1
    assert "not Ethiopian seasons" in jan.scope
    with pytest.raises(ValueError, match="duplicate source"):
        source_calendar_errors(pd.concat([frame, frame.iloc[[0]]]))


def test_complete_year_selection_uses_coverage_without_prediction_values():
    rows = []
    base = inputs().iloc[0].to_dict()
    for year, months in ((2023, range(1, 13)), (2024, range(1, 12))):
        for month in months:
            row = base | {"MediaId": year*100+month, "base_filter_id": f"ETAD-{year*100+month}",
                          "native_date": f"{year}-{month:02d}-10", "current_date": f"{year}-{month:02d}-10",
                          "SamplingEndDate": f"{year}-{month:02d}-11", "prediction_ugm3": -999.}
            rows.extend([row | {"arm": arm} for arm in ("raw", "spline")])
    scenarios = mac_scenario_ledger(season_ledger(pd.DataFrame(rows)), [10])
    summary, audit = complete_year_summary(scenarios)
    chosen = audit.set_index("calendar_year")
    assert chosen.at[2023, "complete_twelve_month_coverage"]
    assert not chosen.at[2024, "complete_twelve_month_coverage"]
    assert summary.included_calendar_years.eq("2023").all()
    assert summary.median_prediction_ugm3.eq(-999.).all()
    one_panel = summary.loc[(summary.arm == "raw") & (summary.season_convention == "belg_feb") & (summary.cohort == "all_common_direct_pairs")]
    assert one_panel.n_direct_pairs.sum() == 12
