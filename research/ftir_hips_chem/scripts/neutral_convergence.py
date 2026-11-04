"""Auditable iteration-budget sensitivity for the existing neutral baseline.

The objective and spectral masking are identical to ``neutral_baseline``.
Increasing the iteration cap is a numerical sensitivity, not parameter tuning
against carbon references or optical agreement. Finite outputs and convergence
are recorded separately.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pybaselines import Baseline

from neutral_baseline import neutral_grid


def tolerance_status(history, *, tol=1e-3):
    """Report the library's stopping criterion without equating finite to fit."""
    history = np.asarray(history, dtype=float)
    if history.ndim != 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("one-dimensional history and positive finite tolerance required")
    last = float(history[-1]) if len(history) else np.nan
    return {
        "iterations": len(history),
        "last_tolerance": last,
        "converged": bool(np.isfinite(last) and last <= tol),
    }


def neutral_with_diagnostics(wavenumbers, spectra, *, max_iter=500, lam=1e6, tol=1e-3):
    """Return retained ascending grid, corrected matrix, and one status per row.

    Spline defaults are fixed explicitly to pybaselines 1.2.1: 100 knots,
    cubic basis, and second-order penalty. A library ``max_iter`` of 50 can
    produce 51 recorded solves; the returned history is authoritative.
    Nonconverged finite results are retained and marked, never quietly replaced.
    """
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("lam and tol must be positive and finite")
    grid, indices = neutral_grid(wavenumbers)
    matrix = np.asarray(spectra, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(wavenumbers) or not len(matrix):
        raise ValueError("nonempty rows-by-wavenumbers matrix required")
    retained = matrix[:, indices]
    if not np.isfinite(retained).all():
        raise ValueError("retained spectral channels must be finite")
    fitter = Baseline(x_data=grid)
    corrected = np.empty_like(retained)
    records = []
    for row_index, row in enumerate(retained):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            baseline, info = fitter.pspline_arpls(
                row, lam=lam, num_knots=100, spline_degree=3, diff_order=2,
                max_iter=max_iter, tol=tol,
            )
        corrected[row_index] = row - baseline
        status = tolerance_status(info["tol_history"], tol=tol)
        finite = bool(np.isfinite(corrected[row_index]).all())
        records.append({
            "row": row_index, "max_iter": max_iter, "lambda": lam,
            "tolerance": tol, **status, "finite_output": finite,
            "success": finite and status["converged"],
            "warnings": " | ".join(dict.fromkeys(str(w.message) for w in caught)),
        })
    return grid, corrected, pd.DataFrame(records)
