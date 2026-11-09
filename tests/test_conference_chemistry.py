"""Scientific contracts for independent chemistry and forward response tests."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"
sys.path.insert(0, str(SCRIPTS))

from config import MAC_VALUE
from conference_experiments.chemistry import (
    ChemicalMeasurement, adapt_unified_filters, evaluate_chemistry_prediction,
    evaluate_regime_interactions,
)
from conference_experiments.tracked_inputs import SOURCE_FILES, _method_evidence, prepare_tracked_inputs


def chemical(**changes):
    values = dict(column="sulfate", analyte="sulfate", method="ion_chromatography",
                  units="ug/m3", source_evidence="synthetic independent IC certificate",
                  independence_verified=True)
    return ChemicalMeasurement(**{**values, **changes})


def cohort():
    rng = np.random.default_rng(214)
    months = pd.date_range("2019-10-01", periods=24, freq="MS")
    dates = pd.DatetimeIndex([month + pd.Timedelta(days=day) for month in months for day in (4, 14, 24)])
    n = len(dates)
    ec = rng.uniform(0.5, 8, n)
    sulfate = rng.uniform(0.1, 1.9, n)
    pm = 4 + 3 * ec + rng.normal(0, 0.5, n)
    d = 0.2 + 0.15 * ec + 0.35 * (ec - 4) * (sulfate - 1) + rng.normal(0, 0.015, n)
    return pd.DataFrame({
        "physical_id": [f"ETAD-{5000+i}" for i in range(n)], "date": dates,
        "lot": np.where(np.arange(n) % 2, "synthetic_A", "synthetic_B"),
        "ftir_ec_ugm3": ec, "hips_fabs_Mm1": (ec - d) * MAC_VALUE,
        "hips_source": "direct_HIPS", "pm25_ugm3": pm, "sulfate": sulfate,
    })


def optical_cohort():
    frame = cohort()
    rng = np.random.default_rng(5)
    n = len(frame)
    frame["optical_ir"] = rng.uniform(1, 10, n)
    frame["shape_uv_ir"] = rng.uniform(1, 3, n)
    frame["within_window_iqr"] = rng.uniform(0.2, 2, n)
    frame["temperature"] = rng.uniform(10, 25, n)
    frame["sulfate"] = (0.4 * frame.optical_ir + 1.2 * frame.shape_uv_ir
                        + 0.8 * frame.within_window_iqr + rng.normal(0, .01, n))
    for name in ("sampling_schedule_verified", "observation_provenance_verified", "processing_history_verified"):
        frame[name] = True
    frame["valid_observed_coverage_fraction"] = 0.95
    frame["minimum_segment_valid_observed_coverage"] = 0.9
    frame["window_evidence_id"] = "synthetic acquisition and active-pump record"
    frame["window_start_utc"] = frame.date.dt.tz_localize("Africa/Addis_Ababa").dt.tz_convert("UTC")
    frame["window_end_utc"] = frame.window_start_utc + pd.Timedelta(hours=24)
    frame["optical_quantity"] = "b_ATN"
    frame["optical_units"] = "Mm-1"
    return frame


def test_independent_provenance_is_required_and_carbon_cannot_be_aliased():
    with pytest.raises(ValueError, match="independence_verified"):
        chemical(independence_verified=False)
    for args in ({"column": "ChemSpec_EC_PM2.5"}, {"column": "FTIR_factor3"}, {"analyte": "ec"}, {"method": "ftir"}):
        with pytest.raises(ValueError):
            chemical(**args)
    with pytest.raises(ValueError, match="units"):
        chemical(units="ug/filter")
    with pytest.raises(ValueError, match="source_evidence"):
        chemical(source_evidence="")


def test_interaction_detects_known_signal_on_same_forward_filters():
    result = evaluate_regime_interactions(cohort(), chemistry=chemical(), min_train=18, bootstrap_reps=20)
    assert result["summary"]["status"] == "completed"
    metrics = result["tables"]["metrics"].set_index("arm")
    assert metrics.loc["chemistry_slope_interaction", "rmse"] < 0.5 * metrics.loc["controlled_baseline", "rmse"]
    predictions = result["tables"]["predictions"]
    assert predictions.groupby("arm").physical_id.nunique().nunique() == 1
    assert predictions.physical_id.nunique() == 36
    assert predictions.date.min() >= pd.Timestamp("2020-10-01")
    differences = result["tables"]["paired_differences"]
    specific = differences.loc[(differences.baseline == "chemistry_main") & (differences.arm == "chemistry_slope_interaction")]
    assert len(specific) == 1 and specific.delta_rmse.iloc[0] < 0
    assert specific.n_blocks.iloc[0] == 36


def test_test_targets_do_not_change_training_parameters_or_predictions():
    frame = cohort()
    first = evaluate_regime_interactions(frame, chemistry=chemical(), min_train=18, bootstrap_reps=0)
    changed = frame.copy()
    changed.loc[changed.date >= "2020-10-01", "hips_fabs_Mm1"] += 500
    second = evaluate_regime_interactions(changed, chemistry=chemical(), min_train=18, bootstrap_reps=0)
    pd.testing.assert_frame_equal(first["tables"]["training_parameters"], second["tables"]["training_parameters"])
    np.testing.assert_allclose(first["tables"]["predictions"].prediction, second["tables"]["predictions"].prediction)
    assert not np.allclose(first["tables"]["predictions"].actual, second["tables"]["predictions"].actual)


def test_physical_repeats_are_rejected_and_missingness_is_audited():
    frame = cohort()
    repeat = frame.iloc[[0]].copy()
    repeat.physical_id += "-2"
    with pytest.raises(ValueError, match="duplicate physical"):
        evaluate_regime_interactions(pd.concat([frame, repeat]), chemistry=chemical())
    frame.loc[0, "sulfate"] = np.nan
    frame.loc[1, "hips_source"] = "HIPS_SSR_estimate"
    frame.loc[2, "lot"] = "unknown"
    frame.loc[3, "is_excluded"] = True
    frame.loc[3, "exclusion_reason"] = "synthetic prior QC"
    result = evaluate_regime_interactions(frame, chemistry=chemical(), min_train=18, bootstrap_reps=0)
    audit = result["tables"]["eligibility"]
    assert len(audit) == len(frame)
    assert audit.loc[:3, "is_excluded"].all()
    assert "chemical_ugm3" in audit.exclusion_reason.iloc[0]
    assert "direct_HIPS" in audit.exclusion_reason.iloc[1]
    assert "unknown_filter_lot" in audit.exclusion_reason.iloc[2]
    assert "synthetic prior QC" in audit.exclusion_reason.iloc[3]
    assert not set(frame.physical_id.iloc[:4]) & set(result["tables"]["predictions"].physical_id)


def test_nullable_missing_filter_id_is_not_converted_to_a_literal_name():
    frame = cohort()
    frame.loc[0, "physical_id"] = pd.NA
    with pytest.raises(ValueError, match="nonmissing physical"):
        evaluate_regime_interactions(frame, chemistry=chemical())
    source = long_filter()
    source.loc[0, "FilterId"] = pd.NA
    with pytest.raises(ValueError, match="missing physical IDs"):
        adapt_unified_filters(source, chemical_specs={})


def test_missing_lot_or_direct_hips_cannot_be_reported_as_controlled():
    frame = cohort().drop(columns="lot")
    result = evaluate_regime_interactions(frame, chemistry=chemical())
    assert result["summary"]["status"] == "insufficient_data"
    assert result["summary"]["n_eligible"] == 0


def test_chemistry_prediction_requires_active_windows_and_observed_support():
    frame = optical_cohort()
    result = evaluate_chemistry_prediction(frame, target=chemical(), baseline_numeric=["optical_ir", "temperature"],
                                           shape_columns=["shape_uv_ir"], temporal_columns=["within_window_iqr"],
                                           min_train=18, bootstrap_reps=20)
    assert result["summary"]["status"] == "completed"
    scores = result["tables"]["metrics"].set_index("arm")
    assert scores.loc["plus_optical_temporal", "rmse"] < scores.loc["routine_baseline", "rmse"] * .3
    no_evidence = frame.drop(columns=["sampling_schedule_verified", "window_evidence_id"])
    blocked = evaluate_chemistry_prediction(no_evidence, target=chemical(), baseline_numeric=["optical_ir"], shape_columns=["shape_uv_ir"])
    assert blocked["summary"]["status"] == "blocked"
    assert blocked["summary"]["n_eligible"] == 0
    assert blocked["tables"]["eligibility"].exclusion_reason.str.contains("sampling_schedule_verified").all()
    frame["valid_observed_coverage_fraction"] = np.nan
    missing_observed = evaluate_chemistry_prediction(frame, target=chemical(), baseline_numeric=["optical_ir"], shape_columns=["shape_uv_ir"])
    assert missing_observed["summary"]["status"] == "blocked"


def test_chemistry_prediction_rejects_fake_flags_mixed_quantities_and_target_features():
    frame = optical_cohort()
    frame["sampling_schedule_verified"] = "true"
    with pytest.raises(ValueError, match="explicit booleans"):
        evaluate_chemistry_prediction(frame, target=chemical(), baseline_numeric=["optical_ir"], shape_columns=["shape_uv_ir"])
    frame = optical_cohort()
    frame.loc[0, "optical_quantity"] = "b_abs"
    with pytest.raises(ValueError, match="do not pool"):
        evaluate_chemistry_prediction(frame, target=chemical(), baseline_numeric=["optical_ir"], shape_columns=["shape_uv_ir"])
    with pytest.raises(ValueError, match="target cannot"):
        evaluate_chemistry_prediction(frame, target=chemical(), baseline_numeric=["sulfate"], shape_columns=["shape_uv_ir"])


def test_overlapping_filter_windows_are_purged_before_fitting():
    frame = optical_cohort()
    last_train = frame.index[frame.date < "2020-10-01"][-1]
    first_test = frame.index[frame.date >= "2020-10-01"][0]
    frame.loc[last_train, "window_end_utc"] = frame.loc[first_test, "window_end_utc"]
    result = evaluate_chemistry_prediction(frame, target=chemical(), baseline_numeric=["optical_ir"], shape_columns=["shape_uv_ir"], min_train=18, bootstrap_reps=0)
    folds = result["tables"]["folds"]
    assert folds.n_train.iloc[0] == 35
    assert folds.purged_train_ids.iloc[0] == [frame.physical_id.iloc[last_train]]


def long_filter(physical_id="ETAD-5000", *, sulfate_units="ng/m3"):
    rows = []
    for parameter, value, units, source in [
        ("EC_ftir", 2., "ug/m3", "FTIR"), ("HIPS_Fabs", 30., "Mm-1", "HIPS"),
        ("ChemSpec_Sulfate_Ion_PM2.5", 1000., sulfate_units, "ChemSpec"),
        ("ChemSpec_Filter_PM2.5_mass", 20., "ug/m3", "ChemSpec"),
    ]:
        rows.append(dict(Site="ETAD", FilterId=physical_id, SampleDate="2020-06-01", LotId="251.0",
                         FilterType="PM2.5", Parameter=parameter, Concentration=value,
                         Concentration_Units=units, DataSource=source))
    return pd.DataFrame(rows)


def test_unified_adapter_normalizes_units_preserves_ids_and_catches_conflicting_repeats():
    source = long_filter()
    specs = {"ChemSpec_Sulfate_Ion_PM2.5": chemical()}
    adapted = adapt_unified_filters(source, chemical_specs=specs)
    assert adapted.physical_id.iloc[0] == "ETAD-5000"
    assert adapted.lot.iloc[0] == "251"
    assert adapted.sulfate.iloc[0] == pytest.approx(1.)
    assert adapted.hips_fabs_Mm1.iloc[0] == 30
    assert adapted.hips_bc_ugm3.iloc[0] == 30 / MAC_VALUE
    assert not adapted.is_excluded.iloc[0]
    conflicting = source.iloc[[0]].copy()
    conflicting.FilterId += "-2"
    conflicting.Concentration = 9.
    result = adapt_unified_filters(pd.concat([source, conflicting]), chemical_specs=specs)
    assert len(result) == 1 and result.is_excluded.iloc[0]
    assert np.isnan(result.ftir_ec_ugm3.iloc[0])
    assert "conflicting_repeat_values" in result.exclusion_reason.iloc[0]


def test_unified_adapter_never_substitutes_public_bc_or_accepts_carbon_aliases():
    source = long_filter()
    source.loc[source.Parameter.eq("HIPS_Fabs"), ["Parameter", "DataSource", "Concentration_Units"]] = ["ChemSpec_BC_PM2.5", "ChemSpec", "ug/m3"]
    result = adapt_unified_filters(source, chemical_specs={"ChemSpec_Sulfate_Ion_PM2.5": chemical()})
    assert result.hips_source.iloc[0] == "unverified"
    assert np.isnan(result.hips_fabs_Mm1.iloc[0])
    with pytest.raises(ValueError, match="parameter must agree"):
        adapt_unified_filters(source, chemical_specs={"ChemSpec_EC_PM2.5": chemical()})


def test_unified_adapter_invalid_units_are_flagged_and_retained():
    source = long_filter(sulfate_units="ug/filter")
    result = adapt_unified_filters(source, chemical_specs={"ChemSpec_Sulfate_Ion_PM2.5": chemical()})
    assert len(result) == 1 and result.is_excluded.iloc[0]
    assert np.isnan(result.sulfate.iloc[0])
    assert "invalid_units:sulfate" in result.exclusion_reason.iloc[0]


def test_areal_loading_uses_physical_volume_and_area_without_claiming_active_times():
    source = long_filter()
    source["Volume_m3"], source["DepositArea_cm2"] = 7., 3.5
    adapted = adapt_unified_filters(source, chemical_specs={})
    assert adapted.pm25_areal_ugcm2.iloc[0] == 40.
    assert "not active-interval verification" in adapted.loading_derivation.iloc[0]
    repeat = source.iloc[[0]].copy()
    repeat.Volume_m3 = 9.
    adapted = adapt_unified_filters(pd.concat([source, repeat]), chemical_specs={})
    assert np.isnan(adapted.pm25_areal_ugcm2.iloc[0])
    assert "collection_volume_m3" in adapted.loading_metadata_reason.iloc[0]


def method_table():
    return pd.DataFrame({"Site_Code": ["ETAD"], "Filter_ID": ["ETAD-5000"],
                         "Parameter_Name": ["Sulfate Ion PM2.5"], "Method_Code": [316],
                         "Analysis_Description": ["Ion Chromatography"],
                         "Units": ["Micrograms per cubic meter (ug/m3)"], "Value": [1.]})


def test_tracked_preparation_requires_reviewed_method_and_value_match(tmp_path):
    repo = tmp_path / "repo"
    for relative in SOURCE_FILES.values():
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
    source = long_filter()
    source["Volume_m3"], source["DepositArea_cm2"] = 7., 3.5
    source.to_csv(repo / SOURCE_FILES["unified_filters"], index=False)
    method_table().to_csv(repo / SOURCE_FILES["chemical_methods"], index=False)
    pd.DataFrame({"date": ["2020-06-01"], "prcp": [0.], "tavg": [15.], "wspd": [5.]}).to_csv(repo / SOURCE_FILES["daily_weather"], index=False)
    manifest = prepare_tracked_inputs(repo, tmp_path / "inputs")
    declaration = json.loads(manifest.read_text())
    data = pd.read_csv(manifest.parent / "chemical_filters.csv")
    assert data.chemical_method_value_verified.all()
    assert data.pm25_areal_ugcm2.iloc[0] == 40.
    assert data.sample_day_temperature_c.iloc[0] == 15.
    assert not any("verified" in column for column in data if column != "chemical_method_value_verified")
    assert "unavailable_reason" in declaration["datasets"]["verified_optical_filter_windows"]
    wrong = method_table()
    wrong.Value = 9.
    wrong.to_csv(repo / SOURCE_FILES["chemical_methods"], index=False)
    other = prepare_tracked_inputs(repo, tmp_path / "mismatch")
    rejected = pd.read_csv(other.parent / "chemical_filters.csv")
    assert rejected.sulfate_ugm3.isna().all()
    assert rejected.is_excluded.all()
    assert rejected.exclusion_reason.str.contains("not_verified_against_method_source").all()


@pytest.mark.parametrize("column,value", [("Method_Code", 316.5), ("Analysis_Description", "FTIR"), ("Units", "ug/filter")])
def test_tracked_method_change_requires_review(tmp_path, column, value):
    table = method_table()
    table[column] = value
    path = tmp_path / "methods.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError):
        _method_evidence(path)
