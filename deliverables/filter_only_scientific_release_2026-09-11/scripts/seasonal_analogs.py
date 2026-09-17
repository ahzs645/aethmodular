"""Whole-spectrum analog selection and an auditable retrospective target split.

The masks apply to correlation selection only. Calibration wavelength choices
are separate so a mask comparison does not change two things at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def spectral_region_mask(wavenumbers, co2_low=1800, upper=None):
    """Keep the available grid except inclusive [co2_low, 2500] and >upper.

    co2_low=None is the unmasked control. No interpolation spans the gap.
    """
    wn = np.asarray(wavenumbers, dtype=float)
    if wn.ndim != 1 or not np.isfinite(wn).all():
        raise ValueError("wavenumbers must be a finite one-dimensional array")
    if co2_low is not None and not 0 < co2_low < 2500:
        raise ValueError("co2_low must be positive and below 2500")
    keep = np.ones(wn.size, dtype=bool)
    if co2_low is not None:
        keep &= ~((wn >= co2_low) & (wn <= 2500))
    if upper is not None:
        keep &= wn <= upper
    if keep.sum() < 3:
        raise ValueError("at least three retained channels are required")
    return keep


def centered_unit_spectra(spectra):
    """Center each spectrum and give it unit Euclidean norm for Pearson r."""
    x = np.asarray(spectra, float)
    if x.ndim != 2 or x.shape[1] < 3 or not np.isfinite(x).all():
        raise ValueError("spectra must be finite with at least three channels")
    centered = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("constant spectra have undefined correlation")
    return centered / norm


def mean_correlation_scores(library, targets, mask):
    """Mean signed Pearson r against *every* target, equal weight per filter.

    Descending signed r preserves the distinction between similar and inverted
    spectra. Squaring before ranking would incorrectly reward anticorrelation.
    """
    if len(targets) == 0:
        raise ValueError("at least one target spectrum is required")
    lib = centered_unit_spectra(np.asarray(library)[:, mask])
    target = centered_unit_spectra(np.asarray(targets)[:, mask])
    return np.clip(lib @ target.mean(axis=0), -1, 1)


def rank_unique_filters(scores, filter_ids, analysis_ids, eligible=None):
    """Rank eligible scan rows, retaining the best scan per physical filter."""
    frame = pd.DataFrame(
        {
            "score": scores,
            "FilterId": filter_ids,
            "AnalysisId": analysis_ids,
            "position": np.arange(len(scores)),
        }
    )
    if eligible is not None:
        frame = frame.loc[np.asarray(eligible, bool)]
    frame = frame.loc[np.isfinite(frame.score)]
    return (
        frame.sort_values(["score", "AnalysisId"], ascending=[False, True])
        .drop_duplicates("FilterId")
        .position.to_numpy(int)
    )


def seasonal_month_split(frame, *, seed=20260910, test_fraction=1 / 3):
    """Assign whole calendar months within each season to selection or validation.

    This is a retrospective design manifest, not a claim of a pristine holdout.
    All rows from a month stay together. Requires >=2 months per season.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie strictly between zero and one")
    result = frame.copy()
    dates = pd.to_datetime(result["date"], errors="raise")
    if dates.isna().any() or result["season"].isna().any():
        raise ValueError("every row needs a date and season")
    result["month_block"] = dates.dt.to_period("M").astype(str)
    result["role"] = "selection"
    rng = np.random.default_rng(seed)
    for season, part in result.groupby("season", sort=True):
        blocks = sorted(part.month_block.unique())
        if len(blocks) < 2:
            raise ValueError(f"{season}: need at least two month blocks")
        n_test = min(len(blocks) - 1, max(1, round(len(blocks) * test_fraction)))
        chosen = rng.choice(blocks, n_test, replace=False)
        result.loc[part.index[part.month_block.isin(chosen)], "role"] = "validation"
    if (result.groupby("month_block").role.nunique() > 1).any():
        raise ValueError("a month spans inconsistent season labels")
    return result
