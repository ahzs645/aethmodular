"""Synthetic-data tests for the shared FTIR spectra package."""

import numpy as np
import pytest

from research.ftir_hips_chem.scripts.spectra import bands, grid, preprocess


def test_snv_flat_row_uses_zero_sigma_guard():
    X = np.array([[4.0, 4.0, 4.0], [1.0, 2.0, 3.0]])

    result = preprocess.snv(X)

    assert np.isfinite(result).all()
    assert np.array_equal(result[0], np.zeros(3))
    assert np.isclose(result[1].mean(), 0.0)
    assert np.isclose(result[1].std(), 1.0)


def test_area_normalization_negative_rules_are_distinct():
    wn = np.array([0.0, 1.0, 2.0])
    X = np.array([[-1.0, 0.0, 1.0]])

    shifted = preprocess.normalize_area(X, wn, negatives="min-shift")
    clipped = preprocess.normalize_area(X, wn, negatives="clip")

    assert not np.allclose(shifted, clipped)
    assert np.isclose(np.trapezoid(shifted[0], wn), 1.0)
    assert np.isclose(np.trapezoid(clipped[0], wn), 1.0)
    assert np.all(clipped >= 0)


def test_area_normalization_rejects_unknown_negative_rule():
    with pytest.raises(ValueError, match="negatives"):
        preprocess.normalize_area(
            np.ones((1, 3)),
            np.arange(3),
            negatives="discard",
        )


@pytest.mark.parametrize("descending", [False, True])
def test_band_integral_is_positive_for_both_axis_directions(descending):
    wn = np.array([0.0, 1.0, 2.0])
    X = np.array([[2.0, 3.0, 4.0]])
    if descending:
        wn = wn[::-1]
        X = X[:, ::-1]

    result = bands.band_integral(X, wn, (0.0, 2.0))

    assert np.allclose(result, [6.0])


@pytest.mark.parametrize("descending", [False, True])
def test_min_baseline_band_integral_is_axis_direction_independent(descending):
    wn = np.array([0.0, 1.0, 2.0])
    X = np.array([[2.0, 3.0, 4.0]])
    if descending:
        wn = wn[::-1]
        X = X[:, ::-1]

    result = bands.band_integral(
        X,
        wn,
        (0.0, 2.0),
        baseline="min",
    )

    assert np.allclose(result, [2.0])


def test_resampling_round_trips_between_descending_and_ascending_grids():
    wn_descending = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    X_descending = np.vstack(
        [2.0 * wn_descending + 1.0, -wn_descending + 3.0]
    )
    wn_common = grid.common_grid(step=0.5, window=(0.0, 4.0))

    on_common = grid.resample(X_descending, wn_descending, wn_common)
    round_trip = grid.resample(on_common, wn_common, wn_descending)

    assert np.allclose(round_trip, X_descending)


def test_ascending_keeps_spectra_aligned():
    wn = np.array([3.0, 1.0, 2.0])
    X = np.array([[30.0, 10.0, 20.0]])

    sorted_wn, sorted_X = grid.ascending(wn, X)

    assert np.array_equal(sorted_wn, [1.0, 2.0, 3.0])
    assert np.array_equal(sorted_X, [[10.0, 20.0, 30.0]])


def test_exclude_accepts_named_mask_regions():
    wn = np.array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0])
    X = np.arange(5.0)[None, :]

    kept_wn, kept_X = grid.exclude(wn, X, regions=["ptfe_cf"])

    assert np.array_equal(kept_wn, [1000.0, 1400.0])
    assert np.array_equal(kept_X, [[0.0, 4.0]])


def test_band_mean_is_not_an_integral():
    wn = np.array([1000.0, 1002.0, 1004.0])
    X = np.array([[1.0, 2.0, 3.0]])

    assert np.allclose(bands.band_mean(X, wn, (1000.0, 1004.0)), [2.0])
    assert np.allclose(
        bands.band_integral(X, wn, (1000.0, 1004.0)),
        [8.0],
    )


def test_detrend_removes_linear_spectra():
    wn = np.linspace(1000.0, 1100.0, 11)
    X = np.vstack([0.2 * wn + 3.0, -0.5 * wn + 7.0])

    assert np.allclose(preprocess.detrend(X, wn), 0.0, atol=1e-12)


def test_vector_norm_leaves_zero_row_finite():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])

    result = preprocess.vector_norm(X)

    assert np.array_equal(result[0], [0.0, 0.0])
    assert np.isclose(np.linalg.norm(result[1]), 1.0)


def test_second_derivative_uses_run_char_11_defaults():
    coordinate = np.arange(21.0)
    X = (coordinate**2)[None, :]

    result = preprocess.second_derivative(X)

    assert np.allclose(result, 2.0, atol=1e-10)
