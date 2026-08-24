"""Bootstrap utilities for calibration-candidate stability reporting."""

from __future__ import annotations

import numpy as np


def stratified_bootstrap_indices(groups, n_boot: int, seed: int) -> np.ndarray:
    """Resample rows within every target stratum, preserving stratum sizes."""
    labels = np.asarray(groups, dtype=object)
    if labels.ndim != 1 or len(labels) < 3:
        raise ValueError("groups must be a one-dimensional array with at least 3 rows")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    levels = list(dict.fromkeys(labels.tolist()))
    positions = [np.flatnonzero(labels == level) for level in levels]
    draws = np.empty((n_boot, len(labels)), dtype=int)
    for draw in range(n_boot):
        draws[draw] = np.concatenate(
            [rng.choice(index, size=len(index), replace=True) for index in positions]
        )
    return draws


def selection_frequency(score_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return winner counts and percentages for draw-by-candidate scores."""
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("score_matrix must be a non-empty draw-by-candidate matrix")
    valid = np.isfinite(scores).any(axis=1)
    winners = np.full(scores.shape[0], -1, dtype=int)
    winners[valid] = np.nanargmin(
        np.where(np.isfinite(scores[valid]), scores[valid], np.inf), axis=1
    )
    counts = np.bincount(winners[valid], minlength=scores.shape[1])
    pct = 100.0 * counts / max(1, int(valid.sum()))
    return counts, pct


def percentile_summary(values: np.ndarray) -> dict[str, float | None]:
    """Median and central 95% percentile interval, ignoring non-finite draws."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"median": None, "lo": None, "hi": None}
    lo, median, hi = np.percentile(finite, [2.5, 50, 97.5])
    return {"median": float(median), "lo": float(lo), "hi": float(hi)}

