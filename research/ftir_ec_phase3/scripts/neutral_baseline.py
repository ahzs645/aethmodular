"""Neutral FTIR baseline correction used as an AIRSpec-independent check.

The method deliberately has no band-specific anchor window.  It masks the
PTFE-saturated regions, fits ``pybaselines`` penalized-spline arPLS, and returns
the corrected spectra on one deterministic ascending wavenumber grid.
"""

from __future__ import annotations

import numpy as np
from pybaselines import Baseline

DEFAULT_LAM = 1e6
MASK_WINDOWS = ((-np.inf, 700.0), (1100.0, 1300.0))

_WORKER_WN: np.ndarray | None = None
_WORKER_LAM: float = DEFAULT_LAM
_WORKER_FITTER: Baseline | None = None


def neutral_grid(wavenumbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending retained wavenumbers and their indices in the input."""
    wn = np.asarray(wavenumbers, dtype=float)
    if wn.ndim != 1 or not np.isfinite(wn).all():
        raise ValueError("wavenumbers must be a finite one-dimensional array")
    if len(np.unique(wn)) != len(wn):
        raise ValueError("wavenumbers must be unique")
    order = np.argsort(wn)
    sorted_wn = wn[order]
    keep = (sorted_wn >= 700.0) & ~(
        (sorted_wn > 1100.0) & (sorted_wn < 1300.0)
    )
    return sorted_wn[keep], order[keep]


def neutral_baseline_spectrum(
    spectrum: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    lam: float = DEFAULT_LAM,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct one spectrum and return ``(retained_wavenumbers, corrected)``."""
    wn, indices = neutral_grid(wavenumbers)
    values = np.asarray(spectrum, dtype=float)
    if values.shape != np.asarray(wavenumbers).shape:
        raise ValueError("spectrum and wavenumber arrays must have the same shape")
    y = values[indices]
    if not np.isfinite(y).all():
        raise ValueError("spectrum contains non-finite retained values")
    baseline, _ = Baseline(x_data=wn).pspline_arpls(y, lam=float(lam))
    return wn, y - baseline


def _init_worker(wavenumbers: np.ndarray, lam: float) -> None:
    global _WORKER_WN, _WORKER_LAM, _WORKER_FITTER
    _WORKER_WN = np.asarray(wavenumbers, dtype=float)
    _WORKER_LAM = float(lam)
    _WORKER_FITTER = Baseline(x_data=_WORKER_WN)


def _worker_correct(values: np.ndarray) -> np.ndarray:
    if _WORKER_FITTER is None:
        raise RuntimeError("neutral-baseline worker was not initialized")
    baseline, _ = _WORKER_FITTER.pspline_arpls(
        np.asarray(values, dtype=float), lam=_WORKER_LAM
    )
    return np.asarray(values, dtype=float) - baseline


def neutral_baseline_matrix(
    wavenumbers: np.ndarray,
    spectra: np.ndarray,
    *,
    lam: float = DEFAULT_LAM,
    pool=None,
    chunksize: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct a matrix; an optional multiprocessing pool accelerates rows."""
    wn, indices = neutral_grid(wavenumbers)
    matrix = np.asarray(spectra, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(wavenumbers):
        raise ValueError("spectra must be rows by the supplied wavenumber grid")
    retained = matrix[:, indices]
    if not np.isfinite(retained).all():
        raise ValueError("spectra contain non-finite retained values")
    if pool is None:
        fitter = Baseline(x_data=wn)
        out = np.empty_like(retained)
        for index, row in enumerate(retained):
            baseline, _ = fitter.pspline_arpls(row, lam=float(lam))
            out[index] = row - baseline
    else:
        # The caller must create its pool with initializer=_init_worker.  This
        # avoids serializing the 2.4k-channel grid with every spectrum.
        out = np.asarray(
            pool.map(_worker_correct, retained, chunksize=max(1, int(chunksize)))
        )
    return wn, out

