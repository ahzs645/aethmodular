"""Regression helpers: Deming/ODR and OLS statistics.

Canonical implementations consolidated from research/addis_fabs_ec_deming
(the closed-form Deming) and research/improve_hips_offset (the defensive OLS
stats that strip non-finite values). These previously existed as 8+ divergent
copies scattered across notebooks.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def deming_lambda(sigma_x: float, sigma_y: float) -> float:
    """Error-variance ratio lambda = Var(y-error) / Var(x-error).

    lambda = 1 is orthogonal / total least squares; lambda -> inf approaches
    OLS of y-on-x; lambda -> 0 approaches inverse OLS (x-on-y).
    """
    return (sigma_y / sigma_x) ** 2


def deming(x, y, lam: float = 1.0):
    """Closed-form Deming regression. Returns (slope, intercept).

    lam is the ratio Var(y-error)/Var(x-error); lam=1 is orthogonal regression.
    Non-finite pairs are dropped before fitting.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return np.nan, np.nan

    xbar, ybar = x.mean(), y.mean()
    sxx = np.sum((x - xbar) ** 2)
    syy = np.sum((y - ybar) ** 2)
    sxy = np.sum((x - xbar) * (y - ybar))
    if sxy == 0:
        return np.nan, np.nan

    slope = (
        syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)
    ) / (2 * sxy)
    intercept = ybar - slope * xbar
    return slope, intercept


def regression_stats(x, y, positive_only: bool = False) -> Mapping[str, float]:
    """OLS regression statistics with non-finite handling.

    Returns dict of n, slope, intercept, r2, origin_slope (slope of the
    through-origin fit). Infinities and NaNs are always dropped. Set
    positive_only=True to additionally keep only strictly-positive x and y --
    appropriate for fAbs/EC where non-positive values are unphysical, but off
    by default so the helper is safe for general use.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if positive_only:
        mask &= (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    n = int(x.size)
    if n < 3:
        return {
            "n": n,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "origin_slope": np.nan,
        }

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    origin_slope = np.sum(x * y) / np.sum(x ** 2)
    return {
        "n": n,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "origin_slope": float(origin_slope),
    }


# Assumed 1-sigma measurement uncertainties for the fAbs-vs-EC fit
# (from research/addis_fabs_ec_deming).
SIGMA_EC_DEFAULT = 0.2      # ug/m3
SIGMA_FABS_DEFAULT = 1.0    # Mm-1


def fit_fabs_ec(ec, fabs, sigma_ec: float = SIGMA_EC_DEFAULT,
                sigma_fabs: float = SIGMA_FABS_DEFAULT) -> Mapping[str, float]:
    """Fit fAbs (y) vs EC (x) with both OLS and Deming.

    Convenience wrapper for the recurring errors-in-variables fAbs-EC fit.
    Returns the OLS stats plus deming_slope / deming_intercept computed with
    lambda derived from the assumed measurement uncertainties.
    """
    stats = dict(regression_stats(ec, fabs, positive_only=True))
    lam = deming_lambda(sigma_ec, sigma_fabs)
    d_slope, d_intercept = deming(ec, fabs, lam)
    stats["deming_slope"] = d_slope
    stats["deming_intercept"] = d_intercept
    stats["lambda"] = lam
    return stats
