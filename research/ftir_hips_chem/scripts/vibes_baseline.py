"""Blank-trained VIBES adapter; upstream 1.0.0 is preserved in ../vendor/pyvibes.

Unlike a spline baseline, this requires independent interference examples.
Inputs and outputs retain the caller's wavenumber order. No clipping, rescaling,
interpolation, or blank subtraction is applied outside the upstream model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from cvxpy.error import SolverError
from scipy.optimize import minimize

VENDOR = Path(__file__).resolve().parents[1] / "vendor/pyvibes"
sys.path.insert(0, str(VENDOR / "src"))
from vibes.interference_models.pca import pca, loo_pca
from vibes.absorbance_estimators.vibes import VibeSpec
from vibes.absorbance_estimators.map import MAPEstimator


def _grid(wn):
    wn = np.asarray(wn, dtype=float)
    if wn.ndim != 1 or len(wn) < 3 or not np.isfinite(wn).all():
        raise ValueError("wavenumbers must be finite, one dimensional, length >= 3")
    if not (np.all(np.diff(wn) > 0) or np.all(np.diff(wn) < 0)):
        raise ValueError("wavenumbers must be unique and strictly monotonic")
    return wn


def _matrix(values, wn):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(wn) or not len(values):
        raise ValueError("spectra must be a nonempty rows-by-wavenumbers matrix")
    if not np.isfinite(values).all():
        raise ValueError("spectra contain non-finite values; flag before fitting")
    return values


@dataclass
class VibesBackground:
    wavenumbers: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    cv_errors: pd.DataFrame
    blank_ids: tuple[str, ...]
    fit_seconds: float


def fit_vibes_background(wavenumbers, blanks, *, blank_ids, max_components=None):
    """Select PCA rank using upstream blank-only leave-one-out one-SE rule.

    Replicate scans must be averaged by physical blank before this call.
    The rank cap is an explicit computational/model-complexity limit, not
    selected using AIRSpec agreement or test-sample outcomes.
    """
    start = time.perf_counter()
    wn = _grid(wavenumbers)
    blanks = _matrix(blanks, wn)
    ids = tuple(map(str, blank_ids))
    if len(ids) != len(blanks) or len(set(ids)) != len(ids):
        raise ValueError("supply one unique physical blank ID per row")
    if len(blanks) < 4:
        raise ValueError("at least four independent training blanks required")
    if max_components is None:
        max_components = len(blanks) - 2
    if not isinstance(max_components, int) or max_components < 1:
        raise ValueError("max_components must be a positive integer")
    rank = np.linalg.matrix_rank(blanks - blanks.mean(axis=0))
    cap = min(max_components, len(blanks) - 2, rank)
    if cap < 1:
        raise ValueError("blank library has no background variation")
    _, c, errors = loo_pca(blanks, max_ncomp=cap)
    mean, _, _, W = pca(blanks)
    W = W[:, :c]
    if np.any(np.sum(W**2, axis=1) == 0):
        raise ValueError("blank PCA has zero variance at a retained channel")
    cv = pd.DataFrame(
        {
            "components": np.arange(1, cap + 1),
            "mean_mse": errors.mean(axis=0),
            "se_mse": errors.std(axis=0) / np.sqrt(len(blanks)),
        }
    )
    return VibesBackground(wn.copy(), mean, W, cv, ids, time.perf_counter() - start)


def vibes_baseline_matrix(
    wavenumbers,
    spectra,
    background,
    *,
    sample_ids,
    tau=0.1,
    loss="PB",
    maxiter=10000,
    retry_failed=True,
    progress=None,
):
    """Return (baseline, corrected, diagnostics) using upstream ELBO + MAP.

    The upstream fit discards scipy's convergence result. Here its identical
    objective, gradient, bounds and MAP solver are called explicitly so each
    optimizer outcome and numerical warning can be audited. Failed fits remain
    in diagnostics with NaN output; there is no silent fallback to AIRSpec.
    tau=None estimates asymmetry by ELBO; the default fixes upstream's 0.1.
    On failure, one logged restart from the finite last iterate uses five times
    the iteration budget and maxls=50. Set retry_failed=False for strict upstream
    optimizer settings. The objective, bounds and MAP estimator are unchanged.
    """
    wn = _grid(wavenumbers)
    values = _matrix(spectra, wn)
    if not np.array_equal(wn, background.wavenumbers):
        raise ValueError("sample and blank wavenumber grids must match exactly")
    ids = tuple(map(str, sample_ids))
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError("supply one unique sample ID per spectrum")
    if set(ids) & set(background.blank_ids):
        raise ValueError("training blanks cannot also be evaluation samples")
    if loss not in ("PB", "ALS"):
        raise ValueError("loss must be PB or ALS")
    if tau is not None and not 0 < tau < 0.5:
        raise ValueError("tau must be None or between 0 and 0.5")
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    baseline = np.full_like(values, np.nan)
    records = []
    c = background.components.shape[1]
    for i, (sid, y) in enumerate(zip(ids, values)):
        start = time.perf_counter()
        rec = {
            "sample_id": sid,
            "success": False,
            "components": c,
            "loss": loss,
            "tau": np.nan,
            "sigma": np.nan,
            "elbo": np.nan,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                mod = VibeSpec(c=c, loss=loss, tau_init=0.1 if tau is None else tau)
                lo, hi = (1e-5, 0.5 + 1e-5) if tau is None else (tau, tau)
                fun, jac, init, bounds = mod.optim_prep(
                    y,
                    mod.tau_init,
                    mod.nu_init,
                    mod.d_init,
                    background.mean,
                    background.components,
                    lo,
                    hi,
                    1e-10,
                )
                opt = minimize(
                    fun,
                    init,
                    method="L-BFGS-B",
                    jac=jac,
                    bounds=bounds,
                    options={"maxiter": maxiter},
                )
                initial_iterations = int(opt.nit)
                rec.update(
                    initial_optimizer_success=bool(opt.success),
                    initial_message=str(opt.message),
                    retry_count=0,
                )
                if retry_failed and not opt.success and np.isfinite(opt.x).all():
                    opt = minimize(
                        fun,
                        opt.x,
                        method="L-BFGS-B",
                        jac=jac,
                        bounds=bounds,
                        options={"maxiter": 5 * maxiter, "maxls": 50, "maxfun": 20 * maxiter},
                    )
                    rec["retry_count"] = 1
                asym, nu, d = opt.x[0], opt.x[1 : 1 + c], opt.x[1 + c :]
                sigma = mod.comp_sigma_hat(y, asym, nu, d, background.mean, background.components)
                rec.update(
                    optimizer_success=bool(opt.success),
                    message=str(opt.message),
                    iterations=int(opt.nit) + (initial_iterations if rec["retry_count"] else 0),
                    tau=float(asym),
                    sigma=float(sigma),
                    elbo=float(-opt.fun),
                )
                if not opt.success or not np.isfinite([sigma, opt.fun]).all() or sigma <= 0:
                    raise RuntimeError("ELBO optimization failed: " + str(opt.message))
                z, latent = MAPEstimator(loss=loss, tau=asym, sigma=sigma).solve(
                    y, background.mean, background.components
                )
                if not np.isfinite(z).all() or not np.isfinite(latent).all():
                    raise RuntimeError("MAP solver returned non-finite output")
                baseline[i] = z
                rec["success"] = True
            except (ValueError, RuntimeError, ArithmeticError, SolverError) as exc:
                rec["message"] = str(exc)
            rec["warnings"] = " | ".join(dict.fromkeys(str(w.message) for w in caught))
        rec["seconds"] = time.perf_counter() - start
        records.append(rec)
        if progress is not None:
            progress(i + 1, len(values))
    return baseline, values - baseline, pd.DataFrame(records)
