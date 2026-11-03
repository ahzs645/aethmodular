"""Calibration-specific distances, grid gates, and physical-filter identities."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/workflows"))
from run_calibration_domain_screen import (
    aggregate_physical_targets, domain_metrics, verify_model_hashes, verify_raw_grid,
)


def test_pls_distance_matches_sample_covariance_normalization():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(80, 12))
    y = 2*X[:, 0]-X[:, 3]+rng.normal(scale=.1, size=len(X))
    model = PLSRegression(n_components=4, scale=False).fit(X, y)
    target = rng.normal(size=(7, 12))
    actual = domain_metrics(model, target)
    scores = model.transform(target)
    inverse_covariance = np.linalg.pinv(np.cov(model.x_scores_, rowvar=False, ddof=1))
    expected = np.einsum("ij,jk,ik->i", scores, inverse_covariance, scores)
    np.testing.assert_allclose(actual.score_mahalanobis_squared, expected, rtol=1e-10)
    np.testing.assert_allclose(actual.score_mahalanobis_squared, 79*actual.score_leverage)
    assert (actual.spectral_Q_sum_squared_absorbance >= 0).all()


def test_target_rows_cannot_change_source_metric():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(50, 10))
    model = PLSRegression(n_components=3, scale=False).fit(X, X[:, 0]+.3*X[:, 4])
    target = rng.normal(size=(1, 10))
    single = domain_metrics(model, target)
    many = domain_metrics(model, np.vstack([target, rng.normal(size=(20, 10))*100]))
    np.testing.assert_allclose(single.iloc[0], many.iloc[0])


def test_raw_grid_requires_channel_identity_at_export_precision():
    native = np.linspace(3998, 500, 2722)
    exported = native.astype(np.float32)
    X = np.zeros((2, 2722))
    receipt = verify_raw_grid(exported, native, X, X)
    assert receipt["grid_identity"] == "exact_after_float32_cast"
    changed = native.copy()
    changed[100] += .01
    with pytest.raises(ValueError, match="not identical"):
        verify_raw_grid(exported, changed, X, X)
    with pytest.raises(ValueError, match="raw-absorbance"):
        verify_raw_grid(exported, native, X, X, representation="spline_corrected")
    with pytest.raises(ValueError, match="2722"):
        verify_raw_grid(exported[:-1], native[:-1], X[:, :-1], X[:, :-1])


def test_changed_model_export_fails_closed():
    verify_model_hashes({"model": "a"}, expected={"model": "a"})
    with pytest.raises(ValueError, match="unreviewed calibration"):
        verify_model_hashes({"model": "b"}, expected={"model": "a"})


def target_fixture():
    return pd.DataFrame({
        "SampleAnalysisId": [11, 12, 13], "MediaId": [7, 8, 7],
        "base_filter_id": ["ETAD-0001", "ETAD-0002", "ETAD-0001"],
        "ExternalFilterType": ["PM2.5"]*3, "lot_group": ["251"]*3,
        "native_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01"]),
        "SampleVolume_m3": [30., 40., 30.],
    })


def test_physical_mean_follows_ids_despite_noncontiguous_repeats():
    scans = target_fixture()
    metadata, mean = aggregate_physical_targets(scans, np.array([[1., 2.], [9., 8.], [3., 4.]]))
    assert metadata.base_filter_id.tolist() == ["ETAD-0001", "ETAD-0002"]
    assert metadata.n_scans.tolist() == [2, 1]
    assert metadata.scan_ids.tolist() == ["11|13", "12"]
    np.testing.assert_array_equal(mean, [[2., 3.], [9., 8.]])


def test_ambiguous_physical_id_cannot_be_averaged():
    scans = target_fixture()
    scans.loc[2, "base_filter_id"] = "ETAD-0003"
    with pytest.raises(ValueError, match="one-to-one"):
        aggregate_physical_targets(scans, np.ones((3, 2)))
