"""AIRSpec/APRLssb segmented smoothing-spline baseline correction.

This module ports the functions used by ``FitSplineKDT.R``.  The smoothing
spline is solved in the natural-cubic-spline value representation described
by Green & Silverman: the fit minimizes

    sum_i w_i (y_i - f_i)**2 + lambda * integral(f''(x)**2 dx)

and lambda is chosen so that the trace of the smoother is ``df``.  Points
with zero weight are omitted from the data term but are evaluated from the
resulting natural spline, matching ``smooth.spline(..., all.knots=TRUE)``.

Only NumPy and SciPy are required.  The core solve is pentadiagonal and the
trace uses a band-limited Takahashi inverse recursion, so no dense n-by-n
matrix is formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import multiprocessing as mp
from typing import Iterable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import cholesky_banded, solveh_banded
from scipy.optimize import brentq
from scipy.sparse import csc_matrix, diags


SEG1 = (4000.0, 1820.0)
SEG2 = (2000.0, 1425.0)


def make_mask(x: np.ndarray, interval: Iterable[float]) -> np.ndarray:
    """R ``MakeMask``: membership between endpoints, excluding both ends."""
    lo, hi = sorted(tuple(interval))
    x = np.asarray(x, dtype=float)
    return (x > lo) & (x < hi)


def _natural_spline_matrices(x: np.ndarray, w: np.ndarray):
    """Return Q, R diagonals and Q' W^-1 Q diagonals for sorted unique x."""
    n = x.size
    if n < 3:
        raise ValueError("at least three positive-weight unique x values required")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing")
    m = n - 2
    j = np.arange(m)
    rows = np.concatenate((j, j + 1, j + 2))
    cols = np.tile(j, 3)
    vals = np.concatenate(
        (1.0 / h[:-1], -(1.0 / h[:-1] + 1.0 / h[1:]), 1.0 / h[1:])
    )
    q = csc_matrix((vals, (rows, cols)), shape=(n, m))
    r0 = (h[:-1] + h[1:]) / 3.0
    r1 = h[1:-1] / 6.0
    g = (q.T @ diags(1.0 / w) @ q).tocsc()
    g0 = g.diagonal(0)
    g1 = g.diagonal(1)
    g2 = g.diagonal(2)
    return q, r0, r1, g0, g1, g2


def _lower_banded_matrix(
    diag0: np.ndarray, diag1: np.ndarray, diag2: np.ndarray
) -> np.ndarray:
    """Pack a symmetric pentadiagonal matrix for SciPy's lower-band format."""
    n = diag0.size
    ab = np.zeros((3, n), dtype=float)
    ab[0] = diag0
    ab[1, :-1] = diag1
    ab[2, :-2] = diag2
    return ab


def _selected_inverse_from_cholesky(cb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selected inverse diagonal/first off-diagonal from lower band Cholesky.

    ``cb`` is the lower-triangular banded Cholesky factor with bandwidth two.
    Takahashi's descending recursion computes precisely the inverse entries
    needed for ``trace(C^-1 R)`` in O(n * bandwidth**2).
    """
    n = cb.shape[1]
    # Store upper selected inverse entries: z[d, i] = Z[i, i+d].
    z = np.zeros((3, n), dtype=float)
    for i in range(n - 1, -1, -1):
        lii = cb[0, i]
        max_j = min(n - 1, i + 2)
        # Off-diagonal entries first; their recursion uses only later rows.
        for jj in range(max_j, i, -1):
            total = 0.0
            for k in range(i + 1, min(n, i + 3)):
                lki = cb[k - i, i]
                if k <= jj:
                    val = z[jj - k, k]
                else:
                    val = z[k - jj, jj]
                total += lki * val
            z[jj - i, i] = -total / lii
        total = 0.0
        for k in range(i + 1, min(n, i + 3)):
            total += cb[k - i, i] * z[k - i, i]
        z[0, i] = 1.0 / (lii * lii) - total / lii
    return z[0], z[1, :-1]


@dataclass
class _SplineSystem:
    x: np.ndarray
    w: np.ndarray
    df_target: float
    q: csc_matrix
    r0: np.ndarray
    r1: np.ndarray
    g0: np.ndarray
    g1: np.ndarray
    g2: np.ndarray
    lam: float
    system_ab: np.ndarray
    achieved_df: float

    @classmethod
    def build(cls, x: np.ndarray, w: np.ndarray, df: float) -> "_SplineSystem":
        q, r0, r1, g0, g1, g2 = _natural_spline_matrices(x, w)
        n = x.size
        if not 2.0 < df <= n:
            raise ValueError(f"df must be in (2, {n}], got {df}")

        def make_ab(lam: float) -> np.ndarray:
            return _lower_banded_matrix(
                r0 + lam * g0,
                r1 + lam * g1,
                lam * g2,
            )

        def effective_df(log_lam: float) -> float:
            lam = float(np.exp(log_lam))
            cb = cholesky_banded(make_ab(lam), lower=True, check_finite=False)
            inv0, inv1 = _selected_inverse_from_cholesky(cb)
            return 2.0 + np.dot(inv0, r0) + 2.0 * np.dot(inv1, r1)

        # Bracket on log(lambda); normalized x keeps practical roots moderate.
        lo, hi = -40.0, 40.0
        while effective_df(lo) < df:
            lo -= 20.0
        while effective_df(hi) > df:
            hi += 20.0
        log_lam = brentq(
            lambda value: effective_df(value) - df,
            lo,
            hi,
            xtol=2e-12,
            rtol=2e-14,
        )
        lam = float(np.exp(log_lam))
        ab = make_ab(lam)
        achieved = effective_df(log_lam)
        return cls(x, w, df, q, r0, r1, g0, g1, g2, lam, ab, achieved)

    def fit_values(self, y: np.ndarray) -> np.ndarray:
        rhs = np.asarray(self.q.T @ y).ravel()
        gamma = solveh_banded(
            self.system_ab, rhs, lower=True, check_finite=False,
            overwrite_b=False, overwrite_ab=False,
        )
        return y - self.lam * (np.asarray(self.q @ gamma).ravel() / self.w)


@lru_cache(maxsize=256)
def _cached_spline_system(
    x_bytes: bytes, w_bytes: bytes, n: int, df: float
) -> _SplineSystem:
    """Build/cache a system shared by spectra with the same grid and mask.

    AIRSpec repeatedly uses a small set of analyte bounds.  The smoothing
    parameter and banded normal matrix depend on x, weights, and df, but not
    on y, so retaining these systems in each worker avoids solving the same
    effective-df root problem thousands of times.
    """
    x = np.frombuffer(x_bytes, dtype=np.float64, count=n).copy()
    w = np.frombuffer(w_bytes, dtype=np.float64, count=n).copy()
    return _SplineSystem.build(x, w, df)


def _get_spline_system(x: np.ndarray, w: np.ndarray, df: float) -> _SplineSystem:
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    w64 = np.ascontiguousarray(w, dtype=np.float64)
    return _cached_spline_system(x64.tobytes(), w64.tobytes(), x64.size, float(df))


def _prepare_xyw(x, y, w):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or w.shape != x.shape:
        raise ValueError("x, y, and w must be one-dimensional arrays of equal length")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("x and y must be finite")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative")
    keep = w > 0
    if keep.sum() < 3:
        raise ValueError("at least three points must have positive weight")
    order = np.argsort(x[keep], kind="mergesort")
    xp = x[keep][order]
    yp = y[keep][order]
    wp = w[keep][order]
    if np.any(np.diff(xp) <= 0):
        # R smooth.spline groups x values within its tolerance. Spectral grids
        # used here are unique; explicit grouping handles exact duplicates.
        ux, inv = np.unique(xp, return_inverse=True)
        sw = np.bincount(inv, weights=wp)
        sy = np.bincount(inv, weights=wp * yp)
        xp, yp, wp = ux, sy / sw, sw
    # R rescales x to [0,1]. This is mathematically immaterial at fixed df,
    # but improves conditioning and reproduces its numerical regime.
    xmin = xp[0]
    xrange = xp[-1] - xmin
    xpn = (xp - xmin) / xrange
    xalln = (x - xmin) / xrange
    return xpn, yp, wp, xalln


def smooth_spline_df(x, y, w, df):
    """Fit R-compatible cubic smoothing spline at a requested effective df.

    Returns fitted values at *all* supplied x positions, including positions
    whose weight is zero.  Input order is preserved.
    """
    xpn, yp, wp, xalln = _prepare_xyw(x, y, w)
    system = _get_spline_system(xpn, wp, float(df))
    fitted_obs = system.fit_values(yp)
    return CubicSpline(xpn, fitted_obs, bc_type="natural")(xalln)


def find_min_pos(x, y, interval=(1600.0, 1520.0)) -> int:
    """Zero-based port of R ``FindMinPos`` (first minimum on a strict mask)."""
    mask = make_mask(np.asarray(x), interval)
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        raise ValueError("interval contains no x values")
    return int(positions[np.argmin(np.asarray(y)[positions])])


def _fit_spline(x, y, df, interval):
    weights = (~make_mask(np.asarray(x), interval)).astype(float)
    return smooth_spline_df(x, y, weights, df)


def find_bound(x, y, p, fixed=2220.0, init_bound=3710.0, dx=-10.0):
    """Port R ``FindBound``: lower bound while the trailing analyte sum < 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bound = float(init_bound)
    while True:
        baseline = _fit_spline(x, y, p, (fixed, bound))
        analyte = y - baseline
        trailing = make_mask(x, (bound, bound + dx))
        value = np.nansum(analyte[trailing])
        if not value < 0:
            return bound
        bound += dx


def fit_spline_segm1(
    x, y, df, fixed=2220.0, init_bound=3710.0, dx=-10.0
):
    """Port ``FitSplineSegm1``; return a dict mirroring the R list."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bound = find_bound(x, y, df, fixed, init_bound, dx)
    baseline = _fit_spline(x, y, df, (fixed, bound))
    return {
        "bounds": np.array([fixed, bound]),
        "param": float(df),
        "baseline": baseline,
        "absorbance": y - baseline,
    }


def fit_spline_segm2ext(
    x, y, df, fixed=1820.0, interval=(1600.0, 1520.0), n_ext=20
):
    """Port ``FitSplineSegm2ext``; x must be in descending order."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if fixed < max(interval) or np.diff(x[:2])[0] > 0:
        raise ValueError("check FitSplineSegm2ext definition")
    imin = find_min_pos(x, y, interval)
    bound = x[imin]
    stop = min(x.size, imin + int(n_ext) + 1)  # R head(index, imin+n.ext)
    # R positions are one-based: imin_R + n.ext elements == imin_zero+n.ext+1.
    baseline = np.empty_like(y)
    baseline[:stop] = _fit_spline(x[:stop], y[:stop], df, (fixed, bound))
    baseline[stop:] = y[stop:]
    return {
        "bounds": np.array([fixed, bound]),
        "param": float(df),
        "baseline": baseline,
        "absorbance": y - baseline,
    }


def _baseline_one(args):
    x, y, df1, df2 = args
    m1 = make_mask(x, SEG1)
    m2 = make_mask(x, SEG2)
    acc = np.zeros_like(x, dtype=float)
    count = np.zeros(x.size, dtype=np.int8)
    s1 = fit_spline_segm1(x[m1], y[m1], df1)
    s2 = fit_spline_segm2ext(x[m2], y[m2], df2)
    acc[m1] += s1["baseline"]
    count[m1] += 1
    acc[m2] += s2["baseline"]
    count[m2] += 1
    baseline = np.full_like(x, np.nan, dtype=float)
    analyte = count > 0
    baseline[analyte] = acc[analyte] / count[analyte]
    return baseline


def airspec_baseline_matrix(
    wavenumbers_desc, Y, df1=6, df2=4, *, n_jobs=None, chunksize=1,
    pool=None,
):
    """Apply the complete segmented AIRSpec baseline to a spectra matrix.

    Parameters
    ----------
    wavenumbers_desc : (n_wn,) array
        Descending spectral grid.
    Y : (n_spectra, n_wn) array
        Complete finite spectra.
    df1, df2 : float
        Effective df for segments 1 and 2.
    n_jobs : int or None
        Worker processes. ``None`` uses one process; values >1 use a fork
        pool where available.
    pool : multiprocessing.pool.Pool or None
        Optional reusable worker pool.  This is useful for chunked inputs and
        lets per-worker spline-system caches persist between chunks.  When
        supplied, ``n_jobs`` is ignored and the pool is not closed here.

    Returns
    -------
    baseline, corrected : ndarray
        Full-grid arrays with NaN outside the strict 1425--4000 union.
    """
    x = np.asarray(wavenumbers_desc, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if x.ndim != 1 or Y.ndim != 2 or Y.shape[1] != x.size:
        raise ValueError("Y must be 2-D with one column per wavenumber")
    if np.any(np.diff(x) >= 0):
        raise ValueError("wavenumbers_desc must be strictly descending")
    if np.any(~np.isfinite(Y)):
        raise ValueError("Y contains non-finite values")
    tasks = ((x, row, float(df1), float(df2)) for row in Y)
    workers = 1 if n_jobs is None else int(n_jobs)
    if pool is not None:
        rows = list(pool.imap(_baseline_one, tasks, chunksize=chunksize))
    elif workers <= 1:
        rows = [_baseline_one(task) for task in tasks]
    else:
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork" if "fork" in methods else methods[0])
        with ctx.Pool(processes=workers) as pool:
            rows = list(pool.imap(_baseline_one, tasks, chunksize=chunksize))
    baseline = np.vstack(rows) if rows else np.empty_like(Y)
    corrected = Y - baseline
    return baseline, corrected


__all__ = [
    "SEG1", "SEG2", "make_mask", "smooth_spline_df", "find_bound",
    "find_min_pos", "fit_spline_segm1", "fit_spline_segm2ext",
    "airspec_baseline_matrix",
]
