"""Checks of statistical invariants for the added exploratory diagnostics."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from analog_diagnostics import (
    discrepancy_stats,
    month_resamples,
    paired_discrepancy_bootstrap,
    site_concentration,
    spectral_shape_projection,
)


def test_paired_identical_models_have_exact_zero_delta_intervals():
    x = np.arange(12, dtype=float)
    p = x * 0.8 + 1
    result = paired_discrepancy_bootstrap(x, p, p, np.repeat(["a", "b", "c"], 4), n_boot=50)
    for name, value in result.items():
        if name.startswith("delta_"):
            assert value == 0


def test_constant_offset_is_not_hidden_by_perfect_correlation():
    x = np.arange(10, dtype=float)
    result = discrepancy_stats(x, x + 3)
    assert np.isclose(result["r_squared"], 1)
    assert result["rmse"] == result["bias"] == result["mae"] == 3


def test_single_filter_month_retains_bias_without_inventing_correlation():
    result = discrepancy_stats([4], [2])
    assert result["bias"] == -2 and result["rmse"] == 2
    assert np.isnan(result["r_squared"])


def test_month_resampling_keeps_complete_clusters_and_is_reproducible():
    blocks = np.array(["a", "a", "b", "b", "b", "c"])
    draws = list(month_resamples(blocks, 20, 123))
    for indices, same in zip(draws, month_resamples(blocks, 20, 123)):
        assert np.array_equal(indices, same)
        counts = np.bincount(indices, minlength=len(blocks))
        for block in np.unique(blocks):
            assert len(np.unique(counts[blocks == block])) == 1


def test_equal_site_contributions_recover_distinct_site_count():
    m = pd.DataFrame({"FilterId": range(12), "Site": np.repeat(["A", "B", "C"], 4)})
    result, curve = site_concentration(m)
    assert np.isclose(result["effective_sites"], 3)
    assert np.isclose(curve.cumulative_fraction.iloc[-1], 1)


def test_target_changes_cannot_change_source_pca_basis():
    rng = np.random.default_rng(75)
    source = rng.normal(size=(24, 8))
    target = rng.normal(size=(3, 8))
    metadata = pd.DataFrame(
        {"AnalysisId": np.arange(24), "FilterId": np.arange(24), "calibration_eligible": True}
    )
    first = spectral_shape_projection(source, target, np.ones(8, bool), metadata, n_components=3)
    second = spectral_shape_projection(
        source, target[:, ::-1], np.ones(8, bool), metadata, n_components=3
    )
    assert np.array_equal(first["components"], second["components"])
    assert np.array_equal(first["source_mean"], second["source_mean"])
    assert np.array_equal(first["source_q"], second["source_q"])
    assert not np.allclose(first["target_scores"], second["target_scores"])
