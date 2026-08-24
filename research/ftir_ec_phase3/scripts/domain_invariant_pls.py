"""Small, auditable implementation of unsupervised domain-invariant PLS.

The implementation follows Nikzad-Langerodi et al. (2018) and the reference
``diPLSlib`` algorithm.  It is kept here because diPLSlib 2.4.1 currently calls
scikit-learn validation arguments removed in scikit-learn 1.8 and its heuristic
path assumes pre-NumPy-2 scalar coercion.

Only the unsupervised, single-target-domain PLS1 case needed by this project is
implemented.  Target response values are never accepted by the API.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh, solve


def block_average(X: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Average adjacent spectral channels without dropping the final partial block."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    starts = np.arange(0, X.shape[1], block_size)
    sums = np.add.reduceat(X, starts, axis=1)
    counts = np.minimum(block_size, X.shape[1] - starts)
    return sums / counts


def _relaxed_covariance_difference(X_source: np.ndarray,
                                   X_target: np.ndarray) -> np.ndarray:
    """Return the PSD convex relaxation of the covariance difference matrix."""
    source_cov = X_source.T @ X_source / len(X_source)
    target_cov = X_target.T @ X_target / len(X_target)
    values, vectors = eigh(source_cov - target_cov, check_finite=False)
    return (vectors * np.abs(values)) @ vectors.T


@dataclass(frozen=True)
class DomainInvariantPLSFit:
    """Fitted PLS1 coefficients and the two legal prediction centerings."""

    coef: np.ndarray
    intercept: float
    source_mean: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray
    loadings: np.ndarray
    lambdas: np.ndarray
    discrepancy: np.ndarray

    def predict(self, X: np.ndarray, *, centering: str = "target") -> np.ndarray:
        """Predict with source or target-domain centering.

        ``target`` is the di-PLS prediction convention. ``source`` is exposed so
        the lambda-zero fit can be reconciled against ordinary PLS.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.coef):
            raise ValueError("X has the wrong feature shape")
        if centering == "target":
            mean = self.target_mean
        elif centering == "source":
            mean = self.source_mean
        else:
            raise ValueError("centering must be 'source' or 'target'")
        return (X - mean) @ self.coef + self.intercept


def fit_domain_invariant_pls(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    *,
    n_components: int,
    lambdas: float | np.ndarray = 0.0,
    heuristic: bool = False,
) -> DomainInvariantPLSFit:
    """Fit unsupervised single-target-domain di-PLS.

    Parameters follow the reference algorithm. If ``heuristic`` is true, a
    component-specific regularizer balances the PLS reconstruction term with
    the relaxed source/target covariance discrepancy. Target labels are not an
    argument and therefore cannot leak into model fitting.
    """
    X_source = np.asarray(X_source, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    y_source = np.asarray(y_source, dtype=float).reshape(-1)
    if X_source.ndim != 2 or X_target.ndim != 2:
        raise ValueError("source and target spectra must be two-dimensional")
    if X_source.shape[0] != len(y_source):
        raise ValueError("source spectra and responses have different row counts")
    if X_source.shape[1] != X_target.shape[1]:
        raise ValueError("source and target spectra have different feature counts")
    if not (np.isfinite(X_source).all() and np.isfinite(X_target).all()
            and np.isfinite(y_source).all()):
        raise ValueError("inputs must be finite")
    max_components = min(X_source.shape[0] - 1, X_source.shape[1])
    if not 1 <= n_components <= max_components:
        raise ValueError(f"n_components must be between 1 and {max_components}")

    source_mean = X_source.mean(axis=0)
    target_mean = X_target.mean(axis=0)
    y_mean = float(y_source.mean())
    X = X_source - source_mean
    Xs = X.copy()
    Xt = X_target - target_mean
    y = y_source - y_mean

    requested = np.asarray(lambdas, dtype=float).reshape(-1)
    if requested.size == 1:
        requested = np.repeat(requested.item(), n_components)
    if requested.size != n_components or np.any(requested < 0):
        raise ValueError("lambdas must be one non-negative value or one per component")

    p = X.shape[1]
    weights = np.zeros((p, n_components))
    loadings = np.zeros((p, n_components))
    y_loadings = np.zeros(n_components)
    selected_lambdas = np.zeros(n_components)
    discrepancy = np.zeros(n_components)
    identity = np.eye(p)
    eps = np.finfo(float).eps

    for component in range(n_components):
        y_energy = float(y @ y)
        if y_energy <= eps:
            raise np.linalg.LinAlgError("response was exhausted before all components")
        w_pls = (y @ X) / y_energy
        w_norm = w_pls / max(np.linalg.norm(w_pls), eps)
        D = _relaxed_covariance_difference(Xs, Xt)

        lam = float(requested[component])
        if heuristic:
            denominator = float(w_norm @ D @ w_norm)
            reconstruction = X - np.outer(y, w_norm)
            lam = (float(np.sum(reconstruction ** 2)) / denominator
                   if denominator > eps else 0.0)
        selected_lambdas[component] = lam

        if lam > 0:
            regularizer = identity + (lam / y_energy) * D
            weight = solve(regularizer, w_pls, assume_a="sym", check_finite=False)
        else:
            weight = w_pls
        weight /= max(np.linalg.norm(weight), eps)
        discrepancy[component] = float(weight @ D @ weight)

        score = X @ weight
        source_score = Xs @ weight
        target_score = Xt @ weight
        score_energy = float(score @ score)
        source_energy = float(source_score @ source_score)
        target_energy = float(target_score @ target_score)
        if min(score_energy, source_energy, target_energy) <= eps:
            raise np.linalg.LinAlgError("degenerate latent score")

        loading = (score @ X) / score_energy
        source_loading = (source_score @ Xs) / source_energy
        target_loading = (target_score @ Xt) / target_energy
        y_loading = float(y @ score) / score_energy

        weights[:, component] = weight
        loadings[:, component] = loading
        y_loadings[component] = y_loading

        X -= np.outer(score, loading)
        Xs -= np.outer(source_score, source_loading)
        Xt -= np.outer(target_score, target_loading)
        y -= score * y_loading

    rotation = np.linalg.pinv(loadings.T @ weights)
    coef = weights @ rotation @ y_loadings
    return DomainInvariantPLSFit(
        coef=coef,
        intercept=y_mean,
        source_mean=source_mean,
        target_mean=target_mean,
        weights=weights,
        loadings=loadings,
        lambdas=selected_lambdas,
        discrepancy=discrepancy,
    )
