import numpy as np

from neutral_baseline import neutral_baseline_matrix, neutral_grid


def test_neutral_grid_masks_ptfe_regions_and_sorts():
    wn = np.array([1400.0, 1200.0, 600.0, 1000.0, 1300.0, 1100.0])
    retained, indices = neutral_grid(wn)
    assert retained.tolist() == [1000.0, 1100.0, 1300.0, 1400.0]
    assert wn[indices].tolist() == retained.tolist()


def test_neutral_baseline_matrix_is_finite_and_removes_smooth_offset():
    wn = np.linspace(700, 1800, 300)
    baseline = 0.2 + 2e-7 * (wn - 1200) ** 2
    band = 0.08 * np.exp(-0.5 * ((wn - 1617) / 18) ** 2)
    spectra = np.vstack([baseline + band, baseline + 0.03 + 2 * band])
    retained, corrected = neutral_baseline_matrix(wn, spectra)
    assert corrected.shape == (2, len(retained))
    assert np.isfinite(corrected).all()
    peak = np.argmin(np.abs(retained - 1617))
    shoulder = np.argmin(np.abs(retained - 1500))
    assert corrected[0, peak] > corrected[0, shoulder] + 0.03
    assert corrected[1, peak] > corrected[0, peak]
