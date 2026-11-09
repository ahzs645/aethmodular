"""Independent uncertainty-weighted PMF with optional Huber robust fitting.

The observations may be signed; only G and H are constrained nonnegative.
This is a new Python implementation of documented objectives, not the ME-2
binary or an exact port of its proprietary optimization algorithm.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment, nnls
from threadpoolctl import threadpool_limits

HISTORY_COLUMNS = ("iteration", "objective", "Q_true", "Q_robust_excluded_4")


def validate_arrays(X, U):
    """Validate without changing, clipping, imputing or shifting observations."""
    x, u = np.asarray(X, dtype=np.float64), np.asarray(U, dtype=np.float64)
    if x.ndim != 2 or x.shape != u.shape or min(x.shape) < 1:
        raise ValueError("X and U must have identical nonempty two-dimensional shapes")
    if not np.isfinite(x).all():
        raise ValueError("X contains missing or nonfinite values")
    if not np.isfinite(u).all() or np.any(u <= 0):
        raise ValueError("Every U value must be finite and strictly positive")
    return x, u


def array_sha256(array):
    a = np.ascontiguousarray(array, dtype=np.float64)
    h = hashlib.sha256()
    h.update(json.dumps({"shape": a.shape, "dtype": a.dtype.str}, sort_keys=True).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _validate_loss(loss, delta):
    if loss not in ("quadratic", "huber"):
        raise ValueError("loss must be 'quadratic' or 'huber'")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")


def rho(residual, loss="quadratic", delta=4.0):
    """Per-cell loss: r², or twice Huber's usual half-squared convention."""
    _validate_loss(loss, delta)
    r = np.asarray(residual, dtype=np.float64)
    if not np.isfinite(r).all():
        raise ValueError("Scaled residuals must be finite")
    if loss == "quadratic":
        values = r * r
    else:
        absolute = np.abs(r)
        central = np.minimum(absolute, delta)
        values = central * central + 2.0 * delta * (absolute - central)
    if not np.isfinite(values).all():
        raise ValueError("Loss overflow; inspect the data and uncertainty scales")
    return values


def objective_from_residual(residual, loss="quadratic", delta=4.0):
    value = float(np.sum(rho(residual, loss, delta)))
    if not np.isfinite(value):
        raise ValueError("Objective overflow; inspect the data and uncertainty scales")
    return value


def robust_weights_from_residual(residual, loss="huber", delta=4.0):
    """IRLS multiplier min(1, delta/|r|); this does not modify supplied U."""
    _validate_loss(loss, delta)
    r = np.asarray(residual, dtype=np.float64)
    if not np.isfinite(r).all():
        raise ValueError("Scaled residuals must be finite")
    weights = np.ones_like(r)
    if loss == "huber":
        mask = np.abs(r) > delta
        weights[mask] = delta / np.abs(r[mask])
    return weights


def diagnostics(X, U, G, H, loss="quadratic", delta=4.0):
    """Keep the optimization objective separate from EPA-style Q diagnostics."""
    x, u = validate_arrays(X, U)
    residual = (x - np.asarray(G) @ np.asarray(H)) / u
    q_true = objective_from_residual(residual, "quadratic", delta)
    retained = np.abs(residual) <= 4.0
    q_excluded = float(np.sum(residual[retained] ** 2))
    q_huber = objective_from_residual(residual, "huber", delta)
    lower_residual = np.minimum(x, 0.0) / u
    return {
        "objective": q_true if loss == "quadratic" else q_huber,
        "Q_true": q_true,
        "Q_robust_excluded_4": q_excluded,
        "Q_huber": q_huber,
        "huber_delta": float(delta),
        "Q_robust_retained_cells": int(retained.sum()),
        "Q_robust_excluded_cells": int((~retained).sum()),
        "Q_robust_excluded_fraction": float((~retained).mean()),
        "fraction_abs_scaled_residual_gt3": float((np.abs(residual) > 3).mean()),
        "huber_downweighted_fraction": float((np.abs(residual) > delta).mean()),
        "weighted_rmse_true": float(np.sqrt(q_true / x.size)),
        "negative_observation_Q_lower_bound": objective_from_residual(lower_residual, "quadratic", delta),
        "negative_observation_objective_lower_bound": objective_from_residual(lower_residual, loss, delta),
        "fraction_negative_observations": float((x < 0).mean()),
        "n_observed_cells": int(x.size),
    }


def _g_block(a, weights, h, g, sweeps=12):
    """Minimize nonnegative sample coordinates for a fixed weighted quadratic."""
    p = h.shape[0]
    gram = ((h[:, None, :] * h[None, :, :]).reshape(p * p, -1) @ weights.T).reshape(p, p, -1)
    rhs = (weights * a) @ h.T
    for _ in range(sweeps):
        for k in range(p):
            den = gram[k, k]
            num = rhs[:, k] - np.sum(gram[k].T * g, axis=1)
            step = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            g[:, k] = np.maximum(0.0, g[:, k] + step)
    return g


def _h_block(a, weights, g, h, sweeps=12):
    """Minimize nonnegative feature coordinates for a fixed weighted quadratic."""
    p = g.shape[1]
    gram = ((g.T[:, None, :] * g.T[None, :, :]).reshape(p * p, -1) @ weights).reshape(p, p, -1)
    rhs = g.T @ (weights * a)
    for _ in range(sweeps):
        for k in range(p):
            den = gram[k, k]
            num = rhs[k] - np.sum(gram[k] * h, axis=0)
            step = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            h[k] = np.maximum(0.0, h[k] + step)
    return h


def _stationarity(a, weights, g, h):
    """Relative projected gradient in the explicitly conditioned coordinates."""
    residual = (g @ h - a) * weights
    dg, dh = residual @ h.T, g.T @ residual
    pg = np.where((g > 0) | (dg < 0), dg, 0.0)
    ph = np.where((h > 0) | (dh < 0), dh, 0.0)
    scale = np.linalg.norm((weights * a) @ h.T) ** 2 + np.linalg.norm(g.T @ (weights * a)) ** 2
    norm = np.sqrt((np.sum(pg * pg) + np.sum(ph * ph)) / max(scale, np.finfo(float).tiny))
    return float(norm)


def normalize_factors(G, H, sort_factors=True):
    """Set each nonzero H-row sum to one, preserving the fitted product."""
    g, h = np.array(G, dtype=float, copy=True), np.array(H, dtype=float, copy=True)
    if g.ndim != 2 or h.ndim != 2 or g.shape[1] != h.shape[0]:
        raise ValueError("Factor dimensions are incompatible")
    if not np.isfinite(g).all() or not np.isfinite(h).all() or np.any(g < 0) or np.any(h < 0):
        raise ValueError("Factors must be finite and nonnegative")
    sums = h.sum(axis=1)
    active = sums > 0
    h[active] /= sums[active, None]
    g[:, active] *= sums[None, active]
    g[:, ~active] = 0.0
    order = np.argsort(-g.sum(axis=0), kind="stable") if sort_factors else np.arange(h.shape[0])
    return g[:, order], h[order], order


def fit(X, U, n_factors=5, seed=42, *, loss="quadratic", delta=4.0,
        max_iter=2000, tol=1e-7, inner_sweeps=12, check_every=10,
        stationarity_tol=1e-5, initial_G=None, initial_H=None,
        sort_factors=True, threads=1):
    """Fit weighted nonnegative factors to a possibly signed observation matrix.

    Huber fitting uses a majorization/IRLS weight update every outer iteration.
    Both weighted NNLS blocks decrease that surrogate, hence the Huber objective.
    A stationary objective plateau is a local numerical stopping condition and
    is not proof of a global optimum, unique factors, or valid physical sources.
    """
    started = time.perf_counter()
    x, u = validate_arrays(X, U)
    _validate_loss(loss, delta)
    n, m = x.shape
    if isinstance(n_factors, bool) or int(n_factors) != n_factors or not 1 <= n_factors <= min(n, m):
        raise ValueError("Invalid factor count")
    n_factors = int(n_factors)
    for name, value in (("max_iter", max_iter), ("inner_sweeps", inner_sweeps), ("check_every", check_every), ("threads", threads)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(tol) or tol <= 0 or not np.isfinite(stationarity_tol) or stationarity_tol <= 0:
        raise ValueError("Stopping tolerances must be finite and positive")
    if (initial_G is None) != (initial_H is None):
        raise ValueError("Supply both initial_G and initial_H, or neither")
    config = {"n_factors": n_factors, "seed": int(seed), "loss": loss, "delta": float(delta),
              "max_iter": int(max_iter), "tol": float(tol), "inner_sweeps": int(inner_sweeps),
              "check_every": int(check_every), "stationarity_tol": float(stationarity_tol),
              "sort_factors": bool(sort_factors), "threads": int(threads),
              "initialization": "independent_positive_random" if initial_G is None else "provided_factors"}
    column_scale = np.median(u, axis=0)
    row_scale = np.median(u / column_scale, axis=1)
    a = x / row_scale[:, None] / column_scale
    b = u / row_scale[:, None] / column_scale
    base_weights = 1.0 / (b * b)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(base_weights).all() or np.any(base_weights <= 0):
        raise ValueError("Numerical overflow/underflow in conditioning; inspect the input scales")
    rng = np.random.default_rng(seed)
    if initial_G is None:
        # The positive part chooses initialization magnitudes only. The original
        # signed data a, not this initialization helper, is fitted throughout.
        positive_mean = np.mean(np.maximum(a, 0.0), axis=0)
        initial_scale = max(float(np.mean(positive_mean)), np.finfo(float).eps)
        h = rng.uniform(0.2, 1.8, (n_factors, m)) * np.maximum(positive_mean, initial_scale * .01)[None, :] / n_factors
        g = rng.uniform(0.2, 1.8, (n, n_factors))
    else:
        g, h = np.array(initial_G, dtype=float, copy=True), np.array(initial_H, dtype=float, copy=True)
        if g.shape != (n, n_factors) or h.shape != (n_factors, m):
            raise ValueError("Initial factor dimensions do not match X and n_factors")
        if not np.isfinite(g).all() or not np.isfinite(h).all() or np.any(g < 0) or np.any(h < 0):
            raise ValueError("Initial factors must be finite and nonnegative")
        if np.any((g.sum(axis=0) == 0) & (h.sum(axis=1) == 0)) and np.any(x > 0):
            raise ValueError("A factor with both initial G and H zero cannot revive; supply a nonzero initialization")
        config["initial_G_sha256"], config["initial_H_sha256"] = array_sha256(g), array_sha256(h)
        g /= row_scale[:, None]
        h /= column_scale[None, :]
    residual = (a - g @ h) / b
    current = objective_from_residual(residual, loss, delta)
    def history_row(iteration, r, objective):
        q = objective_from_residual(r, "quadratic", delta)
        q4 = float(np.sum(r[np.abs(r) <= 4.0] ** 2))
        return (iteration, objective, q, q4)
    history = [history_row(0, residual, current)]
    last_check = current
    stable_checks = 0
    objective_plateau = False
    converged = False
    stationarity = float("inf")
    max_increase = 0.0
    with threadpool_limits(limits=threads):
        for iteration in range(1, max_iter + 1):
            effective_weights = base_weights * robust_weights_from_residual(residual, loss, delta)
            if not np.isfinite(effective_weights).all() or np.any(effective_weights <= 0):
                raise FloatingPointError("Effective weights overflowed/underflowed; no weight floor was applied")
            _g_block(a, effective_weights, h, g, inner_sweeps)
            _h_block(a, effective_weights, g, h, inner_sweeps)
            norms = np.linalg.norm(h, axis=1)
            norms[norms == 0] = 1.0
            h /= norms[:, None]
            g *= norms[None, :]
            if not np.isfinite(g).all() or not np.isfinite(h).all():
                raise FloatingPointError("Nonfinite factors encountered")
            residual = (a - g @ h) / b
            updated = objective_from_residual(residual, loss, delta)
            relative_increase = (updated - current) / max(1.0, abs(current))
            max_increase = max(max_increase, relative_increase)
            if relative_increase > 1e-10:
                raise RuntimeError(f"Objective increased beyond numerical tolerance at iteration {iteration}")
            current = updated
            if iteration % check_every == 0 or iteration == max_iter:
                relative_drop = (last_check - current) / max(abs(last_check), np.finfo(float).tiny)
                stable_checks = stable_checks + 1 if -1e-10 <= relative_drop < tol else 0
                objective_plateau = stable_checks >= 3
                effective_weights = base_weights * robust_weights_from_residual(residual, loss, delta)
                stationarity = _stationarity(a, effective_weights, g, h)
                history.append(history_row(iteration, residual, current))
                last_check = current
                if objective_plateau and stationarity <= stationarity_tol:
                    converged = True
                    break
    G, H, order = normalize_factors(g * row_scale[:, None], h * column_scale[None, :], sort_factors)
    metrics = diagnostics(x, u, G, H, loss, delta)
    metrics.update({"iterations": int(iteration), "converged": converged,
                    "stop_reason": "stationary_objective_plateau" if converged else "iteration_cap",
                    "objective_plateau": objective_plateau,
                    "relative_projected_gradient_conditioned": stationarity,
                    "stationarity_tolerance_met": bool(stationarity <= stationarity_tol),
                    "maximum_objective_relative_increase": max_increase,
                    "objective_increases_above_tolerance": 0,
                    "collapsed_factors": int(((G.sum(axis=0) == 0) | (H.sum(axis=1) == 0)).sum()),
                    "elapsed_seconds": time.perf_counter() - started,
                    "data_shape": [n, m], "n_factors": n_factors, "seed": int(seed), "loss": loss})
    if not np.isclose(metrics["objective"], history[-1][1], rtol=1e-10, atol=1e-9):
        raise RuntimeError("Final normalization did not preserve the objective")
    provenance = {"method": "alternating_weighted_NNLS_with_optional_Huber_IRLS",
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                  "X_sha256": array_sha256(x), "U_sha256": array_sha256(u),
                  "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                  "history_columns": list(HISTORY_COLUMNS),
                  "factor_order": order.tolist(),
                  "profile_normalization": "H rows sum to one, with paired G rescaling preserving GH",
                  "original_observations_modified": False, "source_mass_calibrated": False,
                  "native_EPA_PMF_or_ME2_executed": False,
                  "global_optimum_or_unique_sources_established": False}
    return {"G": G, "H": H, "history": np.asarray(history), "metrics": metrics, "config": config, "provenance": provenance}


def profile_match(H_reference, H_other, feature_weights=None):
    a, b = np.asarray(H_reference, dtype=float), np.asarray(H_other, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Profile arrays must be finite and share feature coordinates")
    if feature_weights is not None:
        w = np.asarray(feature_weights, dtype=float)
        if w.shape != (a.shape[1],) or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("Feature weights must be finite, nonnegative and match the channels")
        a, b = a * w, b * w
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    denom = an[:, None] * bn[None, :]
    similarity = np.divide(a @ b.T, denom, out=np.zeros_like(denom), where=denom > 0)
    similarity = np.clip(similarity, -1.0, 1.0)
    rows, columns = linear_sum_assignment(-similarity)
    return rows, columns, similarity[rows, columns], similarity


def project_contributions(X, U, H, *, loss="quadratic", delta=4.0,
                          max_iter=1000, tol=1e-9, return_info=False):
    """Estimate new nonnegative G for fixed H; this is not an independent fit."""
    x, u = validate_arrays(X, U)
    h = np.asarray(H, dtype=float)
    _validate_loss(loss, delta)
    if h.ndim != 2 or h.shape[1] != x.shape[1] or not np.isfinite(h).all() or np.any(h < 0):
        raise ValueError("Invalid fixed H")
    p = h.shape[0]
    if p < 1:
        raise ValueError("At least one profile is required")
    with threadpool_limits(limits=1):
        g = np.vstack([nnls((h / u[i]).T, x[i] / u[i], maxiter=max(100 * p, 3 * p))[0] for i in range(len(x))])
        initial = objective_from_residual((x - g @ h) / u, loss, delta)
        history = [initial]
        converged, iteration = True, 0
        if loss == "huber":
            # Rescale columns for the fixed-H convex subproblems.
            col = np.median(u, axis=0)
            a, b, hc = x / col, u / col, h / col
            base = 1.0 / (b * b)
            converged = False
            stable = 0
            for iteration in range(1, max_iter + 1):
                residual = (a - g @ hc) / b
                weights = base * robust_weights_from_residual(residual, loss, delta)
                _g_block(a, weights, hc, g, sweeps=12)
                value = objective_from_residual((a - g @ hc) / b, loss, delta)
                if value > history[-1] + 1e-9 * max(1.0, history[-1]):
                    raise RuntimeError("Fixed-profile robust objective increased")
                relative_drop = (history[-1] - value) / max(history[-1], np.finfo(float).tiny)
                stable = stable + 1 if -1e-10 <= relative_drop < tol else 0
                history.append(value)
                if stable >= 3:
                    converged = True
                    break
    info = {"loss": loss, "delta": float(delta), "iterations": int(iteration), "objective_plateau": converged,
            "objective": objective_from_residual((x - g @ h) / u, loss, delta),
            "fixed_profiles": True, "history": history, "source_mass_calibrated": False}
    return (g, info) if return_info else g
