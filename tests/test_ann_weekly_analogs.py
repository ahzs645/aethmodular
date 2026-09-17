"""Scientific invariants for the Sept 3 analog and uncertainty follow-up."""

import numpy as np
import pandas as pd
import pytest
from research.ftir_hips_chem.scripts.seasonal_analogs import (
    spectral_region_mask,
    mean_correlation_scores,
    rank_unique_filters,
    seasonal_month_split,
)
from research.ftir_hips_chem.scripts.plotting.utils import deming_bootstrap


def test_region_endpoints_and_grid_order():
    wn = np.array([1426, 1799, 1800, 1849, 1850, 2499, 2500, 2501, 3500, 3501, 3600, 3601])
    mask = spectral_region_mask(wn, 1800, 3500)
    assert wn[mask].tolist() == [1426, 1799, 2501, 3500]
    assert np.array_equal(spectral_region_mask(wn[::-1], 1800, 3500), mask[::-1])
    assert spectral_region_mask(wn, 1850, None)[2]


def test_correlation_exclusion_ignores_instrumental_signal_and_preserves_sign():
    wn = np.array([1500, 1600, 1700, 1900, 2400, 2800, 3000])
    target = np.array([[1, 2, 3, 99, 123, 4, 2]], float)
    library = np.vstack([target[0], -target[0], target[0]])
    library[2, 3:5] = [-8000, 9999]
    scores = mean_correlation_scores(library, target, spectral_region_mask(wn))
    np.testing.assert_allclose(scores, [1, -1, 1], atol=1e-12)
    with pytest.raises(ValueError, match="constant"):
        mean_correlation_scores(np.ones((2, 7)), target, spectral_region_mask(wn))


def test_filter_replicates_do_not_inflate_cohort_and_ties_are_deterministic():
    scores = np.array([0.99, 0.98, 0.97, 0.98])
    ids = [1, 1, 2, 3]
    scans = [20, 21, 22, 19]
    assert rank_unique_filters(scores, ids, scans).tolist() == [0, 3, 2]
    assert rank_unique_filters(scores, ids, scans, [False, True, True, True]).tolist() == [3, 1, 2]


def test_split_keeps_months_and_filter_replicates_together():
    dates = pd.to_datetime(
        [
            "2022-01-01",
            "2022-01-01",
            "2023-01-01",
            "2024-01-01",
            "2022-07-01",
            "2023-07-01",
            "2024-07-01",
        ]
    )
    frame = pd.DataFrame({"date": dates, "season": ["Dry"] * 4 + ["Kiremt"] * 3})
    a = seasonal_month_split(frame)
    b = seasonal_month_split(frame)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("month_block").role.nunique().max() == 1
    assert a.groupby("season").role.nunique().eq(2).all()


def test_deming_interval_exact_line_and_axis_swap():
    x = np.linspace(1, 20, 40)
    y = 1.7 * x - 2
    a = deming_bootstrap(x, y, 3, n_boot=200)
    np.testing.assert_allclose([a["slope_ci_low"], a["slope_ci_high"]], [1.7, 1.7], atol=1e-12)
    np.testing.assert_allclose(
        [a["intercept_ci_low"], a["intercept_ci_high"]], [-2, -2], atol=1e-12
    )
    rng = np.random.default_rng(4)
    y = y + rng.normal(0, 1, 40)
    a = deming_bootstrap(x, y, 3, n_boot=300)
    b = deming_bootstrap(y, x, 1 / 3, n_boot=300)
    assert np.isclose(a["deming_slope"], 1 / b["deming_slope"])
    assert np.isclose(a["deming_intercept"], -b["deming_intercept"] / b["deming_slope"])
    assert a["slope_se"] > 0 and a["intercept_se"] > 0


def test_deming_month_block_resampling_and_invalid_inputs():
    rng = np.random.default_rng(3)
    x = np.linspace(1, 10, 48)
    groups = np.repeat(np.arange(12), 4)
    y = 1.2 * x + np.repeat(rng.normal(0, 2, 12), 4)
    a = deming_bootstrap(x, y, 3, groups=groups, n_boot=200)
    b = deming_bootstrap(x, y, 3, groups=groups, n_boot=200)
    assert a == b and a["n_blocks"] == 12 and a["bootstrap_unit"] == "group"
    with pytest.raises(ValueError, match="positive"):
        deming_bootstrap(x, y, 0)
    with pytest.raises(ValueError, match="three independent"):
        deming_bootstrap(x, y, 3, groups=np.zeros(48))
