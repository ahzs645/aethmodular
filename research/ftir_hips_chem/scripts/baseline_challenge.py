"""Existing-data challenges for FTIR background correction.

These are signal-preservation experiments, not EC calibrations. Synthetic
additions do not supply EC truth. Blank references are fitted from independent
physical filters; additive models do not normalize sample loading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import expit


PROTECTED_WINDOWS = ((1500, 1800), (2200, 2450), (2800, 3050), (3100, 3600))


def validate_grid(wn, matrix):
    wn = np.asarray(wn, dtype=float)
    values = np.asarray(matrix, dtype=float)
    if wn.ndim != 1 or len(wn) < 5 or not np.isfinite(wn).all():
        raise ValueError("a finite spectral grid with at least five channels is required")
    if not (np.all(np.diff(wn) > 0) or np.all(np.diff(wn) < 0)):
        raise ValueError("spectral grid must be strictly monotonic")
    if values.ndim != 2 or values.shape[1] != len(wn) or not np.isfinite(values).all():
        raise ValueError("spectra must be a finite rows-by-wavenumbers matrix")
    return wn, values


def independent_blank_bank(values, filter_ids, lots, test_id, *, holdout="filter"):
    """Select a background bank without target-filter or known-lot leakage.

    Inputs must already have one row per physical blank. Unknown-lot blanks
    cannot establish independence from a known target lot, so are not used in
    the known-lot holdout. Unknown is never treated as a real filter lot.
    """
    values = np.asarray(values, dtype=float)
    ids, lots = np.asarray(filter_ids, dtype=str), np.asarray(lots, dtype=str)
    if values.ndim != 2 or len(values) != len(ids) or len(lots) != len(ids):
        raise ValueError("blank bank arrays have incompatible dimensions")
    if len(set(ids)) != len(ids):
        raise ValueError("one row per independent physical blank is required")
    target = np.flatnonzero(ids == str(test_id))
    if len(target) != 1:
        raise ValueError("test_id must identify exactly one physical blank")
    keep = ids != str(test_id)
    if holdout == "known_lot":
        target_lot = lots[target[0]]
        if target_lot in ("unknown", "", "nan", "<NA>"):
            raise ValueError("known-lot holdout requires a known target lot")
        keep &= (lots != target_lot) & ~np.isin(lots, ["unknown", "", "nan", "<NA>"])
    elif holdout != "filter":
        raise ValueError("holdout must be filter or known_lot")
    if keep.sum() < 4:
        raise ValueError("at least four independent training blanks are required")
    return values[keep], ids[keep], lots[keep]


@dataclass
class AdditiveBlankModel:
    """EMSC-inspired additive nuisance fit; no multiplicative normalization.

    The basis contains a constant, a linear trend, and training-blank PCs.
    Broad analyte absorption overlapping that span is not identifiable from
    background. The challenge measures this ambiguity instead of hiding it.
    """

    wavenumbers: np.ndarray
    mean: np.ndarray
    design: np.ndarray
    fit_mask: np.ndarray
    blank_ids: tuple[str, ...]
    rank: int
    condition_number: float

    def correct(self, spectra, *, sample_ids):
        _, values = validate_grid(self.wavenumbers, spectra)
        ids = tuple(map(str, sample_ids))
        if len(ids) != len(values):
            raise ValueError("one sample ID is required per spectrum")
        if set(ids) & set(self.blank_ids):
            raise ValueError("training blanks cannot also be evaluation samples")
        coefficients = np.linalg.lstsq(
            self.design[self.fit_mask],
            (values - self.mean)[:, self.fit_mask].T,
            rcond=None,
        )[0]
        baseline = self.mean + (self.design @ coefficients).T
        return values - baseline


def fit_additive_blank_model(wn, blanks, *, blank_ids, rank=3, protected=False):
    wn, values = validate_grid(wn, blanks)
    ids = tuple(map(str, blank_ids))
    if len(ids) != len(values) or len(ids) != len(set(ids)):
        raise ValueError("one unique physical blank ID is required per row")
    if len(values) < 4 or not isinstance(rank, int) or not 0 <= rank < len(values):
        raise ValueError("at least four blanks and rank below blank count are required")
    mean = values.mean(axis=0)
    _, singular, vectors = svd(values - mean, full_matrices=False, check_finite=False)
    available = int(np.sum(singular > max(singular[0], 1e-30) * 1e-10))
    used_rank = min(rank, available)
    x = 2 * (wn - wn.min()) / np.ptp(wn) - 1
    design = np.column_stack([np.ones(len(wn)), x, vectors[:used_rank].T])
    # Scaling columns improves numerical conditioning without changing span.
    design /= np.linalg.norm(design, axis=0)
    mask = np.ones(len(wn), dtype=bool)
    if protected:
        for low, high in PROTECTED_WINDOWS:
            mask &= ~((wn >= low) & (wn <= high))
    if mask.sum() <= design.shape[1] or np.linalg.matrix_rank(design[mask]) < design.shape[1]:
        raise ValueError("background basis is not identifiable on fitting channels")
    return AdditiveBlankModel(
        wn.copy(), mean, design, mask, ids, used_rank, float(np.linalg.cond(design[mask])),
    )


def whittaker_arpls(values, *, lam=1e6, max_iter=100, tol=1e-3):
    """Baek et al. (2015) arPLS on a uniformly spaced channel grid.

    DOI: 10.1039/C4AN01061B. This is the Whittaker second-difference method,
    not pybaselines' penalized-spline arPLS used by the existing neutral cache.
    Lambda depends on grid spacing; all native challenge comparisons fix it.
    Convergence is retained separately and never inferred from finite output.
    """
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) < 5 or not np.isfinite(y).all():
        raise ValueError("a finite spectrum with at least five channels is required")
    if not np.isfinite(lam) or lam <= 0 or max_iter < 1 or tol <= 0:
        raise ValueError("positive lambda, iteration budget and tolerance are required")
    n = len(y)
    difference = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
                       [0, 1, 2], shape=(n - 2, n), format="csc")
    penalty = float(lam) * difference.T @ difference
    weights = np.ones(n)
    relative_change = np.inf
    reason = "iteration_limit"
    converged = False
    baseline = np.full(n, np.nan)
    for iteration in range(1, int(max_iter) + 1):
        baseline = spsolve(penalty + diags(weights, format="csc"), weights * y)
        residual = y - baseline
        if not np.isfinite(baseline).all():
            reason = "nonfinite_solve"
            break
        negative = residual[residual < 0]
        scale = max(float(np.max(np.abs(y))), 1.0)
        if np.max(np.abs(residual)) < 1e-9 * scale:
            reason, converged, relative_change = "linear_input", True, 0.0
            break
        if len(negative) < 2 or np.std(negative, ddof=0) <= np.finfo(float).eps * scale:
            reason = "degenerate_negative_residuals"
            break
        mean, sd = float(np.mean(negative)), float(np.std(negative, ddof=0))
        new_weights = expit(-2 * (residual - (2 * sd - mean)) / sd)
        relative_change = float(np.linalg.norm(weights - new_weights) / np.linalg.norm(weights))
        if relative_change < tol:
            reason, converged = "weight_tolerance", True
            break
        weights = new_weights
    return y - baseline, {
        "converged": converged, "iterations": iteration,
        "relative_weight_change": relative_change, "reason": reason,
    }


def injected_shapes(wn):
    """Prespecified shape challenges; none is a unique species assignment."""
    wn = np.asarray(wn, dtype=float)
    definitions = {
        "1617_sigma8": (1617.0, 8.0),
        "1617_sigma40": (1617.0, 40.0),
        "1617_sigma120": (1617.0, 120.0),
        "ch_2920_sigma18": (2920.0, 18.0),
        "oh_3300_sigma150": (3300.0, 150.0),
    }
    shapes = {name: np.exp(-0.5 * ((wn - center) / sigma) ** 2)
              for name, (center, sigma) in definitions.items()}
    shapes["broad_linear_continuum"] = 0.2 + 0.8 * (wn - wn.min()) / np.ptp(wn)
    return shapes


def recovery_metrics(corrected_original, corrected_injected, injected_signal):
    """Measure C(blank+signal)-C(blank), rather than fitted peak appearance."""
    original, changed, signal = (np.asarray(x, dtype=float)
                                 for x in (corrected_original, corrected_injected, injected_signal))
    if original.shape != changed.shape or original.shape != signal.shape or signal.ndim != 1:
        raise ValueError("challenge arrays must share one-dimensional shape")
    if not all(np.isfinite(x).all() for x in (original, changed, signal)):
        raise ValueError("challenge arrays must be finite")
    energy = float(signal @ signal)
    if energy <= 0:
        raise ValueError("injected signal must be nonzero")
    delta = changed - original
    gain = float(delta @ signal / energy)
    return {
        "signal_gain": gain,
        "relative_recovery_error": float(np.linalg.norm(delta - signal) / np.sqrt(energy)),
        "relative_shape_distortion": float(np.linalg.norm(delta - gain * signal) / np.sqrt(energy)),
    }
