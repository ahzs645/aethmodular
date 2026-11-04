"""Experimental batched solver for the *unchanged* VIBES pinball MAP objective.

This is an opt-in numerical experiment, not a replacement for ``vibes_baseline``.
It solves, independently for every spectrum, exactly the objective in the
vendored ``vibes.absorbance_estimators.map.map_pb``::

    min_x sum_j pinball_tau(y_j - mean_j - (W x)_j) + sigma/2 * ||x||²

``sigma`` must be strictly positive and must already have been estimated by the
original variational ELBO stage. This module does NOT accelerate or replace that
stage, select a blank bank/rank, or alter spectra. NumPy is the reference backend;
CuPy is optional and explicitly requested. Both use float64 throughout.

ADMM splits the residual ``r = y - mean - W x``. Its residual update is an
asymmetric soft threshold. A single eigendecomposition of W.T @ W makes all
latent updates diagonal, even for a nonorthogonal supplied basis. The PCA basis
used by VIBES is already orthogonal. All remaining large operations are shared
matrix multiplies, suitable for GPU batching. Each row has its own stopping
decision and (optionally adaptive) ADMM penalty.

The feasible dual has bounds ``tau - 1 <= a <= tau`` and objective
``a @ (y - mean) - ||W.T @ a||² / (2*sigma)``. Convergence requires a primal-dual
gap certificate as well as ADMM residual checks. Unconverged/nonfinite rows
return NaN spectra and latent coefficients; their diagnostics remain visible.
Numerical equivalence is testable; GPU performance is not implied by this file.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class BatchMAPResult:
    baseline: np.ndarray
    corrected: np.ndarray
    latent: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    primal_objective: np.ndarray
    dual_objective: np.ndarray
    duality_gap: np.ndarray
    gap_tolerance: np.ndarray
    primal_residual: np.ndarray
    dual_residual: np.ndarray
    numerical_failure: np.ndarray
    backend: str
    seconds: float

    @property
    def diagnostics(self):
        """Plain row dictionaries, suitable for JSON or a pandas DataFrame."""
        return [
            {
                "case_row": i,
                "success": bool(self.converged[i]),
                "status": "converged" if self.converged[i] else (
                    "numerical_failure" if self.numerical_failure[i] else "max_iterations"
                ),
                "iterations": int(self.iterations[i]),
                "primal_objective": float(self.primal_objective[i]),
                "dual_objective": float(self.dual_objective[i]),
                "duality_gap": float(self.duality_gap[i]),
                "gap_tolerance": float(self.gap_tolerance[i]),
                "primal_residual": float(self.primal_residual[i]),
                "dual_residual": float(self.dual_residual[i]),
                "backend": self.backend,
            }
            for i in range(len(self.converged))
        ]


def _row_parameter(value, rows, name):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim == 0:
        a = np.full(rows, float(a), dtype=np.float64)
    if a.shape != (rows,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite and scalar or length n_spectra")
    return a


def solve_pinball_map_batch(
    spectra,
    mean,
    components,
    *,
    sigma,
    tau=0.1,
    backend="numpy",
    rho=1.0,
    maxiter=10000,
    atol=1e-7,
    rtol=1e-7,
    check_every=10,
    adaptive_rho=True,
):
    """Solve independent positive-sigma PB MAP problems in a shared basis.

    Inputs are host arrays: spectra is rows by channels, mean is channels, and
    components is channels by latent components. ``tau``, ``sigma`` and ``rho``
    may each be scalar or one value per row. Batch members never share fitted
    coefficients, sigma, tau, convergence state, or residuals. ``tau`` permits
    the mathematical pinball range (0, 1); upstream VIBES fitting uses a narrower
    range. Returned arrays are NumPy float64 arrays regardless of backend.

    An unavailable CUDA/CuPy backend raises; there is no silent CPU fallback.
    Timing includes decomposition, transfers and final device synchronization.
    Callers must check ``converged``; nonconverged results are masked with NaN.
    """
    started = time.perf_counter()
    y = np.asarray(spectra, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    W = np.asarray(components, dtype=np.float64)
    if y.ndim != 2 or min(y.shape) < 1:
        raise ValueError("spectra must be a nonempty rows-by-channels matrix")
    rows, channels = y.shape
    if mu.shape != (channels,) or W.ndim != 2 or W.shape[0] != channels or W.shape[1] < 1:
        raise ValueError("mean/components must match the spectrum channels")
    if not all(np.isfinite(a).all() for a in (y, mu, W)):
        raise ValueError("spectra, mean and components must be finite")
    sigma_host = _row_parameter(sigma, rows, "sigma")
    tau_host = _row_parameter(tau, rows, "tau")
    rho_host = _row_parameter(rho, rows, "rho")
    if np.any(sigma_host <= 0):
        raise ValueError("sigma must be strictly positive; sigma=0 EBS is unsupported")
    if np.any((tau_host <= 0) | (tau_host >= 1)):
        raise ValueError("tau must lie strictly between zero and one")
    if np.any(rho_host <= 0):
        raise ValueError("rho must be strictly positive")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (maxiter, check_every)):
        raise ValueError("maxiter and check_every must be positive integers")
    if not np.isfinite([atol, rtol]).all() or atol <= 0 or rtol < 0:
        raise ValueError("atol must be positive and rtol nonnegative")
    if backend == "numpy":
        xp = np
        to_host = np.asarray
    elif backend == "cupy":
        try:
            import cupy as xp
        except ImportError as exc:
            raise RuntimeError("The requested GPU backend requires CuPy") from exc
        try:
            if xp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA device is available")
        except Exception as exc:
            raise RuntimeError("The requested CuPy backend has no usable CUDA device") from exc
        to_host = xp.asnumpy
    else:
        raise ValueError("backend must be 'numpy' or 'cupy'")

    # Orthogonal rotation preserves the ridge norm and every fitted spectrum.
    # Clamp only roundoff-negative eigenvalues of the positive semidefinite Gram.
    eigenvalues, rotation_host = np.linalg.eigh(W.T @ W)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    A = xp.asarray(W @ rotation_host, dtype=xp.float64)
    rotation = xp.asarray(rotation_host, dtype=xp.float64)
    eigenvalues = xp.asarray(eigenvalues, dtype=xp.float64)
    v = xp.asarray(y - mu, dtype=xp.float64)
    sig = xp.asarray(sigma_host, dtype=xp.float64)
    taus = xp.asarray(tau_host, dtype=xp.float64)
    penalties = xp.asarray(rho_host, dtype=xp.float64)
    rank = W.shape[1]
    coefficients = xp.zeros((rows, rank), dtype=xp.float64)
    fitted = xp.zeros_like(v)
    residual = v.copy()
    scaled_dual = xp.zeros_like(v)
    converged = xp.zeros(rows, dtype=bool)
    failed = xp.zeros(rows, dtype=bool)
    iterations = xp.full(rows, maxiter, dtype=xp.int64)
    active = ~(converged | failed)
    dual_residual = xp.full(rows, xp.nan, dtype=xp.float64)
    norm = lambda a: xp.sqrt(xp.sum(a * a, axis=1))

    for iteration in range(1, maxiter + 1):
        previous_residual = residual
        candidate = ((v - residual - scaled_dual) @ A) / (
            eigenvalues[None, :] + sig[:, None] / penalties[:, None]
        )
        coefficients = xp.where(active[:, None], candidate, coefficients)
        candidate_fitted = coefficients @ A.T
        fitted = xp.where(active[:, None], candidate_fitted, fitted)
        threshold_input = v - fitted - scaled_dual
        candidate_residual = xp.maximum(threshold_input - taus[:, None] / penalties[:, None], 0.0)
        candidate_residual += xp.minimum(
            threshold_input + (1.0 - taus[:, None]) / penalties[:, None], 0.0
        )
        residual = xp.where(active[:, None], candidate_residual, residual)
        scaled_dual = xp.where(active[:, None], scaled_dual + fitted + residual - v, scaled_dual)

        if iteration % check_every and iteration != maxiter:
            continue
        primal_residual = norm(fitted + residual - v)
        dual_residual = xp.where(active, penalties * norm((residual - previous_residual) @ A),
                                 dual_residual)
        actual_residual = v - fitted
        primal = xp.sum(xp.maximum(taus[:, None] * actual_residual,
                                   (taus[:, None] - 1.0) * actual_residual), axis=1)
        primal += 0.5 * sig * xp.sum(coefficients * coefficients, axis=1)
        dual_feasible = xp.clip(-penalties[:, None] * scaled_dual,
                                taus[:, None] - 1.0, taus[:, None])
        projected_dual = dual_feasible @ A
        dual = xp.sum(dual_feasible * v, axis=1)
        dual -= xp.sum(projected_dual * projected_dual, axis=1) / (2.0 * sig)
        gap = primal - dual
        gap_tolerance = atol + rtol * xp.maximum(1.0, xp.maximum(xp.abs(primal), xp.abs(dual)))
        primal_tolerance = np.sqrt(channels) * atol + rtol * xp.maximum(
            norm(v), xp.maximum(norm(fitted), norm(residual))
        )
        dual_tolerance = np.sqrt(rank) * atol + rtol * penalties * norm(scaled_dual @ A)
        finite = xp.isfinite(primal) & xp.isfinite(dual) & xp.isfinite(gap)
        finite &= xp.isfinite(primal_residual) & xp.isfinite(dual_residual)
        newly_failed = active & ~finite
        failed |= newly_failed
        # A materially negative gap signals a numerical/certificate failure.
        bad_certificate = active & finite & (gap < -gap_tolerance)
        failed |= bad_certificate
        newly_converged = active & finite & ~bad_certificate & (gap <= gap_tolerance)
        newly_converged &= (primal_residual <= primal_tolerance) & (dual_residual <= dual_tolerance)
        iterations = xp.where(newly_converged | newly_failed | bad_certificate, iteration, iterations)
        converged |= newly_converged
        active = ~(converged | failed)
        if not bool(to_host(xp.any(active))):
            break
        if adaptive_rho:
            factor = xp.where(primal_residual > 10.0 * dual_residual, 2.0,
                               xp.where(dual_residual > 10.0 * primal_residual, 0.5, 1.0))
            factor = xp.where(active, factor, 1.0)
            updated = xp.clip(penalties * factor, 1e-8, 1e8)
            scaled_dual *= (penalties / updated)[:, None]
            penalties = updated

    latent = coefficients @ rotation.T
    baseline = xp.asarray(mu)[None, :] + fitted
    corrected = v - fitted
    # Never let a caller accidentally use an uncertified candidate spectrum.
    baseline = xp.where(converged[:, None], baseline, xp.nan)
    corrected = xp.where(converged[:, None], corrected, xp.nan)
    latent = xp.where(converged[:, None], latent, xp.nan)
    arrays = [to_host(a) for a in (
        baseline, corrected, latent, converged, iterations, primal, dual, gap,
        gap_tolerance, primal_residual, dual_residual, failed,
    )]
    return BatchMAPResult(*arrays, backend=backend, seconds=time.perf_counter() - started)
