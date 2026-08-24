import numpy as np

from optimizer_stability import (
    percentile_summary,
    selection_frequency,
    stratified_bootstrap_indices,
)


def test_stratified_bootstrap_preserves_group_counts_and_seed():
    groups = np.array(["dry", "dry", "belg", "belg", "belg", "kiremt"])
    first = stratified_bootstrap_indices(groups, 20, seed=7)
    second = stratified_bootstrap_indices(groups, 20, seed=7)
    assert np.array_equal(first, second)
    for draw in first:
        sampled = groups[draw]
        assert (sampled == "dry").sum() == 2
        assert (sampled == "belg").sum() == 3
        assert (sampled == "kiremt").sum() == 1


def test_selection_frequency_handles_nonfinite_scores():
    scores = np.array([[1.0, 2.0], [3.0, 1.0], [np.nan, 0.5], [np.nan, np.nan]])
    counts, pct = selection_frequency(scores)
    assert counts.tolist() == [1, 2]
    assert np.allclose(pct, [100 / 3, 200 / 3])


def test_percentile_summary_ignores_nonfinite_values():
    out = percentile_summary(np.array([1.0, 2.0, 3.0, np.nan]))
    assert out["median"] == 2.0
    assert out["lo"] < out["median"] < out["hi"]
