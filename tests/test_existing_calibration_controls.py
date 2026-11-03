"""Protect the physical-filter and method-agreement claims in the new audit."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pybaselines", reason="Optional neutral-baseline research dependency")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "existing_calibration_controls",
    ROOT / "research/ftir_hips_chem/workflows/run_existing_calibration_controls.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_perfect_correlation_does_not_become_predictive_accuracy():
    result = MODULE.score([1, 2, 3], [11, 12, 13])
    assert result["pearson_r2"] == pytest.approx(1)
    assert result["predictive_r2"] < 0
    assert result["bias"] == pytest.approx(10)


def test_target_averaging_flags_dates_and_missing_volume(monkeypatch, tmp_path):
    class AffineModel:
        def predict(self, values):
            return np.asarray(values)[:, 0] + 10

    scans = pd.DataFrame([
        {"MediaId": 1, "SampleAnalysisId": 1626, "base_filter_id": "ETAD-0139",
         "native_date": pd.Timestamp("2023-11-14"), "SampleVolume_m3": 2.0},
        {"MediaId": 1, "SampleAnalysisId": 1668, "base_filter_id": "ETAD-0139",
         "native_date": pd.Timestamp("2023-11-14"), "SampleVolume_m3": 2.0},
        {"MediaId": 2, "SampleAnalysisId": 406, "base_filter_id": "ETAD-0037",
         "native_date": pd.Timestamp("2023-01-09"), "SampleVolume_m3": np.nan},
        {"MediaId": 3, "SampleAnalysisId": 1995, "base_filter_id": "ETAD-0158",
         "native_date": pd.Timestamp("2023-12-20"), "SampleVolume_m3": 1.0},
        {"MediaId": 4, "SampleAnalysisId": 9999, "base_filter_id": "ETAD-9999",
         "native_date": pd.Timestamp("2024-01-01"), "SampleVolume_m3": 2.0},
    ])
    scans["ExternalFilterType"] = "PM2.5"
    optical = pd.DataFrame({
        "code": ["ETAD"] * 4,
        "id": ["ETAD-0139", "ETAD-0037", "ETAD-0158", "ETAD-9999"],
        "date": ["2023-11-14", "2023-01-09", "2023-12-21", "2024-01-01"],
        "pair_eligible": [True] * 4, "ec_ftir_ugm3": [4.0] * 4,
        "optical_bc_ugm3": [3.0, 4.0, 5.0, 7.0],
        "optical_source": ["direct_hips_fabs"] * 4,
    })
    monkeypatch.setattr(MODULE, "load_public_chemistry", lambda _: optical)
    monkeypatch.setattr(MODULE, "optical_pair_ledger", lambda frame: frame)
    # First row belongs to the source matrix, so it must never enter the target
    # mean. The last three target rows test missing volume, a date mismatch,
    # and a legitimate unflagged physical-filter pair.
    arms = {"trial": np.array([[1000.0], [-4.0], [2.0], [-5.0], [-14.0], [4.0]])}
    target, summary = MODULE.target_predictions(
        arms, 1, scans, {"trial": AffineModel()}, tmp_path,
    )
    original = target.loc[target.aggregation_sensitivity.eq("all_original_scans")]
    assert len(original) == 4
    primary = original.set_index("base_filter_id")
    assert primary.loc["ETAD-0139", "n_scans"] == 2
    assert primary.loc["ETAD-0139", "prediction_loading_ug"] == pytest.approx(9)
    assert primary.loc["ETAD-0139", "prediction_ugm3"] == pytest.approx(4.5)
    alternative = target.loc[target.aggregation_sensitivity.eq("conditional_scan1668_only")]
    assert len(alternative) == 1
    assert alternative.iloc[0].prediction_ugm3 == pytest.approx(6)
    assert np.isnan(primary.loc["ETAD-0037", "prediction_ugm3"])
    assert primary.loc["ETAD-0158", "prediction_loading_ug"] == pytest.approx(-4)
    joined = pd.read_csv(tmp_path / "target_optical_pairs.csv")
    assert not joined.loc[joined.base_filter_id.eq("ETAD-0158"), "current_pair_eligible"].any()
    assert not joined.loc[joined.base_filter_id.eq("ETAD-0037"), "current_pair_eligible"].any()
    assert summary.set_index("cohort").loc["current_all_available", "n"] == 2
    assert summary.set_index("cohort").loc["current_without_prior_problem_filters", "n"] == 1
