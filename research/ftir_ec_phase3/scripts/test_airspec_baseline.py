"""Minimal tests for the AIRSpec baseline port (run: python -m pytest or python directly)."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from airspec_baseline import (  # noqa: E402
    airspec_baseline_matrix,
    fit_spline_segm2ext,
    make_mask,
    smooth_spline_df,
)


def test_spline_reproduces_linear_signal_exactly():
    # Linear functions span the curvature-penalty null space: reproduced at any lambda.
    x = np.linspace(0.0, 1.0, 200)
    y = 0.3 + 0.5 * x
    fitted = smooth_spline_df(x, y, np.ones_like(x), df=6)
    assert np.max(np.abs(fitted - y)) < 1e-8


def test_spline_tracks_smooth_curvature_approximately():
    x = np.linspace(0.0, 1.0, 200)
    y = 0.3 + 0.5 * x + 0.2 * x**2
    for df, tol in ((6, 5e-3), (10, 1e-3), (20, 3e-4)):
        fitted = smooth_spline_df(x, y, np.ones_like(x), df=df)
        assert np.max(np.abs(fitted - y)) < tol, (df, np.max(np.abs(fitted - y)))


def test_zero_weight_points_are_interpolated_not_fit():
    x = np.linspace(0.0, 1.0, 300)
    w = np.ones_like(x)
    inside = (x > 0.4) & (x < 0.6)
    w[inside] = 0.0
    y = x.copy()
    y[inside] += 5.0  # a huge peak the spline must ignore
    fitted = smooth_spline_df(x, y, w, df=5)
    assert np.max(np.abs(fitted - x)) < 1e-6


def test_segm2ext_tail_passthrough():
    # Descending grid like segment 2 (2000 -> 1425).
    x = np.linspace(2000.0, 1425.0, 450)
    y = 0.001 + (x - 1425.0) * 1e-6 + 0.01 * np.exp(-((x - 1700.0) ** 2) / 400.0)
    result = fit_spline_segm2ext(x, y, df=4)
    imin_zone = make_mask(x, (1600.0, 1520.0))
    positions = np.flatnonzero(imin_zone)
    imin = positions[np.argmin(y[positions])]
    stop = imin + 21
    # Beyond min + n_ext the baseline is the spectrum itself: corrected == 0.
    assert np.allclose(result["absorbance"][stop:], 0.0)
    assert not np.allclose(result["absorbance"][:stop], 0.0)


def test_matrix_mean_stitch_and_region():
    x = np.linspace(3998.0, 500.0, 2722)
    rng = np.random.default_rng(0)
    Y = 0.002 + 1e-6 * (x[None, :] - 500.0) + 1e-4 * rng.standard_normal((2, x.size))
    baseline, corrected = airspec_baseline_matrix(x, Y, df1=6, df2=4)
    region = make_mask(x, (4000.0, 1820.0)) | make_mask(x, (2000.0, 1425.0))
    assert np.isnan(corrected[:, ~region]).all()
    assert np.isfinite(corrected[:, region]).all()
    # A near-linear spectrum should be almost fully removed by the baseline.
    assert np.nanmax(np.abs(corrected[:, region])) < 5e-4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
