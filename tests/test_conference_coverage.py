"""Leakage, identity and equal-budget controls for calibration coverage tests."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from conference_experiments.coverage import (
    calibration_scores, fit_source_model, load_historical_source, paired_source_contrast,
    prepare_physical_spectra, run_coverage_experiment, source_test_sites,
)


FAST_OPTIONS = {
    "ridge": [{"n_components": 3, "alpha": .1}, {"n_components": 3, "alpha": 5.}],
    "pls": [{"n_components": 2}],
    "rf": [{"rf_params": {"n_estimators": 12, "min_samples_leaf": 2, "max_depth": 4}}],
}


def source_fixture():
    rng = np.random.default_rng(91)
    X = rng.normal(size=(96, 6))
    X[:, 0] += np.tile(np.linspace(0, 5, 12), 8)
    meta = pd.DataFrame({"filter_id": [f"F{i:03d}" for i in range(len(X))],
        "analysis_id": [f"A{i:03d}" for i in range(len(X))],
        "site": np.repeat([f"S{i}" for i in range(8)], 12),
        "ec_ug": 10+3*X[:, 0]+.5*X[:, 1], "is_excluded": False,
        "exclusion_reason": "", "season": np.tile(["A", "B", "C"], 32),
        "lot": np.tile(["L1", "L2"], 48)})
    return X, meta


def run(X, meta, **kwargs):
    options = dict(reference_kind="thermal_optical_ec", test_sites=["S6", "S7"],
        budget=30, regions=("low", "high", "spectral"), retain_fractions=(0., .5, 1.),
        models=("ridge",), inner_folds=3, seed=37,
        model_options={"ridge": FAST_OPTIONS["ridge"]}, bootstrap_repeats=20)
    options.update(kwargs)
    return run_coverage_experiment(X, meta, **options)


def test_repeated_scans_aggregate_once_and_flag_entire_physical_filter():
    X = np.array([[1., 3.], [8., 4.], [3., 5.], [4., np.nan]])
    meta = pd.DataFrame({"filter_id": ["F1", "F2", "F1", "F3"],
                         "site": ["S1", "S2", "S1", "S3"],
                         "ec_ug": [2., 6., 2., 5.], "is_excluded": [False, True, False, False],
                         "exclusion_reason": ["", "instrument_failure", "", ""]})
    result = prepare_physical_spectra(X, meta, target_col="ec_ug")
    assert result.metadata.filter_id.tolist() == ["F1", "F2", "F3"]
    assert result.metadata.n_scans.tolist() == [2, 1, 1]
    np.testing.assert_array_equal(result.X[0], [2., 4.])
    assert result.metadata.experiment_eligible.tolist() == [True, False, False]
    assert "instrument_failure" in result.metadata.eligibility_reason.iloc[1]
    assert "nonfinite_spectrum" in result.metadata.eligibility_reason.iloc[2]
    assert len(result.scan_ledger) == 4
    assert "experiment_eligible" not in meta
    damaged = X.copy()
    damaged[2, 0] = np.nan
    assert not prepare_physical_spectra(damaged, meta, target_col="ec_ug").metadata.experiment_eligible.iloc[0]


@pytest.mark.parametrize("column,value", [("site", "other_site"), ("ec_ug", 999.)])
def test_conflicting_repeat_identity_or_reference_fails_before_split(column, value):
    X, meta = source_fixture()
    repeat = meta.iloc[[0]].copy()
    repeat["analysis_id"] = "new_scan"
    repeat[column] = value
    with pytest.raises(ValueError, match="conflicting repeated-filter"):
        prepare_physical_spectra(np.vstack([X, X[0]]), pd.concat([meta, repeat]), target_col="ec_ug")


def test_existing_threshold_flags_and_missing_references_remain_in_ledger():
    X, meta = source_fixture()
    meta["is_outlier"], meta["outlier_reason"] = False, ""
    meta.loc[0, ["is_outlier", "outlier_reason"]] = [True, "recorded_threshold"]
    meta.loc[1, "ec_ug"] = np.nan
    result = prepare_physical_spectra(X, meta, target_col="ec_ug")
    assert len(result.metadata) == len(meta)
    assert not result.metadata.experiment_eligible.iloc[:2].any()
    assert "recorded_threshold" in result.metadata.eligibility_reason.iloc[0]
    assert "nonfinite_reference" in result.metadata.eligibility_reason.iloc[1]
    meta.loc[0, "outlier_reason"] = ""
    with pytest.raises(ValueError, match="outlier_reason"):
        prepare_physical_spectra(X, meta, target_col="ec_ug")


def test_equal_budgets_fixed_test_filters_and_disjoint_inner_sites():
    X, meta = source_fixture()
    result = run(X, meta, models=("ridge", "pls", "rf"), model_options=FAST_OPTIONS)
    assert result["summary"]["status"] == "completed"
    assert result["summary"]["completed_runs"] == 27
    tables = result["tables"]
    selected = tables["training_selection"].query("selected")
    assert selected.groupby("arm").size().eq(30).all()
    assert not selected.site.isin(["S6", "S7"]).any()
    for _, frame in selected.groupby("region"):
        retained = frame.groupby("retain_fraction").in_region.sum().sort_index()
        assert retained.iloc[0] == 0
        assert retained.is_monotonic_increasing
        assert retained.iloc[-1] > 0
    expected_ids = set(meta.loc[meta.site.isin(["S6", "S7"]), "filter_id"])
    for _, frame in tables["predictions"].groupby(["arm", "model"]):
        assert len(frame) == len(expected_ids)
        assert set(frame.filter_id) == expected_ids
    for _, frame in tables["inner_membership"].groupby(["arm", "model", "inner_fold"]):
        a, b = frame.query("role == 'train'"), frame.query("role == 'validation'")
        assert not set(a.site) & set(b.site)
        assert not set(a.filter_id) & set(b.filter_id)
        assert not set(frame.filter_id) & expected_ids
    assert tables["paired_contrasts"].resampling_unit.eq("held_out_source_site").all()


def test_test_labels_cannot_change_selection_thresholds_or_predictions():
    X, meta = source_fixture()
    initial = run(X, meta, regions=("high",), retain_fractions=(0., 1.))
    changed = meta.copy()
    changed.loc[changed.site.isin(["S6", "S7"]), "ec_ug"] += 10000
    rerun = run(X, changed, regions=("high",), retain_fractions=(0., 1.))
    for name in ("training_selection", "thresholds", "tuning", "inner_membership"):
        pd.testing.assert_frame_equal(initial["tables"][name], rerun["tables"][name])
    np.testing.assert_array_equal(initial["tables"]["predictions"].prediction,
                                  rerun["tables"]["predictions"].prediction)
    assert not np.allclose(initial["tables"]["metrics"].query("loading_bin == 'all'").rmse,
                           rerun["tables"]["metrics"].query("loading_bin == 'all'").rmse)


def test_test_spectra_cannot_change_training_coordinates_or_model_choices():
    X, meta = source_fixture()
    initial = run(X, meta, regions=("spectral",), retain_fractions=(0., 1.))
    shifted = X.copy()
    shifted[meta.site.isin(["S6", "S7"])] += 100000
    rerun = run(shifted, meta, regions=("spectral",), retain_fractions=(0., 1.))
    assert initial["summary"]["spectral_descriptor"] == rerun["summary"]["spectral_descriptor"]
    pd.testing.assert_frame_equal(initial["tables"]["thresholds"], rerun["tables"]["thresholds"])
    pd.testing.assert_frame_equal(initial["tables"]["training_selection"], rerun["tables"]["training_selection"])
    pd.testing.assert_frame_equal(initial["tables"]["tuning"], rerun["tables"]["tuning"])


def test_selection_is_stable_under_row_reordering_and_target_free_site_split():
    X, meta = source_fixture()
    initial = run(X, meta, regions=("low",), retain_fractions=(0., 1.))
    order = np.random.default_rng(22).permutation(len(X))
    rerun = run(X[order], meta.iloc[order], regions=("low",), retain_fractions=(0., 1.))
    pd.testing.assert_frame_equal(initial["tables"]["training_selection"], rerun["tables"]["training_selection"])
    np.testing.assert_allclose(initial["tables"]["predictions"].prediction, rerun["tables"]["predictions"].prediction)
    assert source_test_sites(meta) == source_test_sites(meta.iloc[order].assign(ec_ug=99999))


def test_fitted_preprocessor_is_trained_on_selected_rows_only():
    X, meta = source_fixture()
    model, _, _, _ = fit_source_model(X[:60], meta.ec_ug.iloc[:60], meta.site.iloc[:60],
        meta.filter_id.iloc[:60], family="ridge", options=[FAST_OPTIONS["ridge"][0]])
    mean = model.named_steps["features"].named_transformers_["spectrum"].named_steps["center"].mean_
    np.testing.assert_allclose(mean, X[:60].mean(axis=0))
    assert not np.allclose(mean, X.mean(axis=0))


def test_invalid_provenance_budget_flags_and_degenerate_regions_fail_honestly():
    X, meta = source_fixture()
    with pytest.raises(ValueError, match="thermal_optical_ec"):
        run(X, meta, reference_kind="FTIR_EC")
    with pytest.raises(ValueError, match="reference_kind conflicts"):
        run(X, meta.assign(reference_kind="FTIR_EC"))
    with pytest.raises(ValueError, match="ug/filter"):
        run(X, meta, response_unit="ug/m3")
    with pytest.raises(ValueError, match="capacity"):
        run(X, meta, budget=72)
    with pytest.raises(ValueError, match="including restored"):
        run(X, meta, retain_fractions=(0., .5))
    with pytest.raises(ValueError, match="explicit booleans"):
        run(X, meta.assign(is_excluded="False"))
    with pytest.raises(ValueError, match="distinct site"):
        run(X, meta, test_sites="S7")
    result = run(X, meta.assign(ec_ug=1.))
    assert result["summary"]["status"] == "insufficient_data"
    assert "distinct region" in result["summary"]["reason"]
    assert "predictions" not in result["tables"]


def test_metrics_do_not_clip_negative_predictions_or_confuse_correlation_with_accuracy():
    actual = np.array([1., 2., 3.])
    predicted = np.array([-1., 0., 1.])
    metrics = calibration_scores(actual, predicted)
    assert metrics["negative_predictions"] == 1
    assert metrics["bias"] == -2
    assert metrics["r2"] < 0
    assert metrics["p90_span_ratio"] == pytest.approx(1)
    with pytest.raises(ValueError, match="no silent"):
        calibration_scores(actual, [1, 2, np.nan])


def test_site_bootstrap_preserves_pairing_and_marks_one_site_unestimable():
    actual, baseline, candidate = np.arange(6.), np.arange(6.)+.2, np.arange(6.)+.8
    sites = ["S1"]*3+["S2"]*3
    first = paired_source_contrast(actual, baseline, candidate, sites, repeats=50)
    second = paired_source_contrast(actual, candidate, baseline, sites, repeats=50)
    assert first["delta_rmse"] == pytest.approx(-second["delta_rmse"])
    assert first["n_sites"] == 2
    limited = paired_source_contrast(actual, baseline, candidate, ["S1"]*6)
    assert limited["interval_status"] == "not_estimable"


def test_historical_source_adapter_preserves_originals_and_exact_documented_flag(tmp_path, monkeypatch):
    import pls_transfer

    analyses = np.r_[164549, np.arange(500000, 500905)]
    filters = np.r_[2008808, np.arange(3000000, 3000905)]
    sites = np.asarray(["TONT1"]+[f"S{i % 139}" for i in range(905)])
    model = SimpleNamespace(X=np.ones((906, 2)), y=np.arange(906.),
        wavenumbers=np.array([1000., 2000.]), analysis_ids=analyses, filter_ids=filters, sites=sites)
    monkeypatch.setattr(pls_transfer, "load_current_pls_model", lambda *args: model)
    extracted = tmp_path / "notebook/extracted"
    extracted.mkdir(parents=True)
    samples = pd.DataFrame({"AnalysisId": analyses, "SampleDate": ["2022-11-19"]*906})
    samples.to_csv(extracted / "EC_samples.csv", index=False)
    _, primary, _ = load_historical_source(tmp_path)
    assert len(primary) == 906
    assert primary.is_excluded.sum() == 1
    assert primary.loc[primary.is_excluded, "analysis_id"].tolist() == ["164549"]
    _, sensitivity, _ = load_historical_source(tmp_path, source_policy="original906")
    assert not sensitivity.is_excluded.any()
    # A changed acquisition identity cannot silently reuse a filter-only exclusion.
    model.analysis_ids[0] = 164550
    samples.loc[0, "AnalysisId"] = 164550
    samples.to_csv(extracted / "EC_samples.csv", index=False)
    with pytest.raises(ValueError, match="exactly one"):
        load_historical_source(tmp_path)


def test_fit_failure_is_failed_not_missing_data_and_retains_run_ledger(monkeypatch):
    from conference_experiments import coverage

    def fail(*args, **kwargs):
        raise ValueError("synthetic estimator convergence failure")

    monkeypatch.setattr(coverage, "fit_source_model", fail)
    X, meta = source_fixture()
    result = run(X, meta, regions=("high",), retain_fractions=(0., 1.))
    assert result["summary"]["status"] == "failed"
    assert result["summary"]["completed_runs"] == 0
    assert len(result["tables"]["runs"]) == 2
    assert result["tables"]["runs"].reason.eq("synthetic estimator convergence failure").all()
    assert len(result["tables"]["physical_ledger"]) == len(meta)


def test_source_preparation_freezes_split_records_hashes_and_uses_safe_ids(tmp_path, monkeypatch):
    from conference_experiments import preparation
    from conference_experiments.io import load_dataset, sha256

    source, destination = tmp_path / "native", tmp_path / "prepared"
    for name in preparation.NATIVE_SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    X, meta = source_fixture()
    meta["reference_kind"] = "thermal_optical_ec"
    monkeypatch.setattr(preparation, "load_historical_source",
        lambda folder, *, source_policy: (X, meta, np.arange(X.shape[1])))
    manifest_path = preparation.prepare_source_inputs(source, destination, budget=30)
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads((destination / "source_receipt.json").read_text())
    coverage, sampling = [e["parameters"] for e in manifest["experiments"]]
    assert coverage["test_sites"] == sampling["test_sites"] == source_test_sites(meta)
    assert coverage["model_options"] == preparation.SOURCE_MODEL_OPTIONS
    assert coverage["models"] == ["ridge", "pls", "rf"]
    assert coverage["bootstrap_repeats"] == 500
    assert sampling["seeds"] == [20261107, 20261108, 20261109]
    assert coverage["budget"] == sampling["budget"] == 30
    assert receipt["manifest_sha256"] == sha256(manifest_path)
    assert receipt["native_inputs"] == {n: sha256(source / n) for n in preparation.NATIVE_SOURCE_FILES}
    assert receipt["n_training_pool"] + receipt["n_test"] == len(meta)
    assert str(tmp_path) not in json.dumps(receipt) + json.dumps(manifest)
    (loaded, aligned), _ = load_dataset(manifest["datasets"]["source"], destination)
    np.testing.assert_array_equal(loaded, X)
    assert aligned.analysis_id.tolist() == meta.analysis_id.tolist()
    with np.load(destination / "source.npz", allow_pickle=False) as arrays:
        assert arrays["analysis_ids"].dtype.kind == "U"
    with pytest.raises(FileExistsError, match="not empty"):
        preparation.prepare_source_inputs(source, destination, budget=30)
