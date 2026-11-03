"""Data-boundary checks for the graph-focused baseline investigation."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gallery/data"))
from export_baseline_lab import (  # noqa: E402
    aggregate_curves, append_native_records, calibration_tables, control_tables, curve,
)


def test_signed_spectra_and_uncertainty_keep_the_same_observed_grid():
    row = curve("test", "Test", "spline", "corrected", [1700, 1617, 1500],
                [-.1, np.nan, -.2], lo=[-.2, np.nan, -.3], hi=[0, np.nan, -.1])
    assert row["wn"] == [1700, 1617, 1500]
    assert row["y"] == [-.1, None, -.2]
    assert row["lo"] == [-.2, None, -.3]
    assert row["hi"] == [0, None, -.1]
    with pytest.raises(ValueError, match="uncertainty"):
        curve("bad", "Bad", "spline", "corrected", [1700, 1617], [1, 2], lo=[1])


def test_native_only_filters_do_not_acquire_an_optical_reference():
    native = pd.DataFrame([
        {"base_filter_id": "ETAD-0001", "ExternalFilterType": "PM2.5", "lot_group": "251"},
        {"base_filter_id": "ETAD-0002", "ExternalFilterType": "FB", "lot_group": "unknown"},
    ])
    rows = append_native_records([{"id": "ETAD-0001", "spectral_pair_eligible": True}], native)
    assert len(rows) == 2
    blank = rows[1]
    assert blank["spectral_pair_eligible"] is False
    assert blank["ec_ftir_ugm3"] is None
    assert blank["optical_bc_ugm3"] is None
    assert blank["lot_group"] == "unknown"


def test_injection_parent_identity_and_validation_scheme_are_not_pooled(tmp_path):
    rows = []
    for scheme, recovery in [("grouped_5fold", .9), ("known_lot_transfer", .2)]:
        for identifier in ["ETAD-0001", "ETAD-0002"]:
            rows.append({"scheme": scheme, "method": "spline_default", "control": "broad",
                         "amplitude": .01, "base_filter_id": identifier,
                         "area_recovery_fraction": recovery})
    path = tmp_path / "injection_recovery.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    _, _, injections, _, _ = control_tables(tmp_path)
    assert len(injections) == 2
    assert sorted(r["recovery_fraction"] for r in injections) == [.2, .9]
    assert all(r["n_parents"] == 2 and r["method"] == "spline" for r in injections)
    pd.DataFrame([*rows, rows[0]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one observation per physical parent"):
        control_tables(tmp_path)


def test_source_qc_sensitivity_is_not_a_second_score_in_the_primary_population(tmp_path):
    row = {"arm": "raw_full", "evaluation": "site_grouped_pooled", "fold": "all",
           "n": 905, "rmse": 16.9, "mae": 10, "predictive_r2": .8}
    pd.DataFrame([row]).to_csv(tmp_path / "metrics.csv", index=False)
    historical = tmp_path / "historical906_all_scans"
    historical.mkdir()
    pd.DataFrame([{**row, "n": 906, "rmse": 63.4}]).to_csv(historical / "metrics.csv", index=False)
    methods, scores, _ = calibration_tables(tmp_path)
    assert len(methods) == 1
    assert len({r["cohort"] for r in scores}) == 2
    assert [r["n"] for r in scores] == [905, 906]
    assert all(r["units"] == "µg/filter" and r["calibration_matched"] for r in scores)


def test_duplicate_repeat_scans_cannot_weight_the_physical_filter_twice(tmp_path):
    row = {"method": "spline_default", "base_filter_id": "ETAD-0139", "filter_type": "PM2.5",
           "n_pairs": 1, "relative_pair_rms": 1.9, "raw_pair_rms": .2,
           "corrected_pair_rms": .01}
    pd.DataFrame([row, row]).to_csv(tmp_path / "replicate_filter_metrics.csv", index=False)
    with pytest.raises(ValueError, match="aggregated to physical filters"):
        control_tables(tmp_path)


def test_legacy_aggregate_without_support_cannot_present_padding_as_observed_zero(tmp_path):
    rows = []
    for view, value in [("raw", -.1), ("saved_spline", 0.)]:
        for wn in [1600, 1500]:
            rows.append({"type": "FB", "view": view, "wavenumber": wn,
                         "median_absorbance": value, "q25_absorbance": value,
                         "q75_absorbance": value, "physical_filters": 2})
    pd.DataFrame(rows).to_csv(tmp_path / "signed_median_spectra.csv", index=False)
    curves = aggregate_curves(tmp_path)
    assert len(curves) == 1
    assert curves[0]["kind"] == "raw"
    assert curves[0]["y"] == [-.1, -.1]
