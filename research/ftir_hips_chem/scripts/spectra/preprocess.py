"""Per-spectrum preprocessing and normalization."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def snv(X: np.ndarray) -> np.ndarray:
    """Apply standard normal variate scaling to each spectrum.

    Lifted from ``charcoal_spectra.snv``. The zero-sigma guard is deliberate:
    a flat spectrum remains finite rather than poisoning downstream
    correlations with NaN.
    """
    X = np.asarray(X)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def detrend(X: np.ndarray, wn: np.ndarray) -> np.ndarray:
    """Remove a per-spectrum least-squares linear trend in wavenumber.

    Lifted from ``charcoal_spectra.detrend``.
    """
    X = np.asarray(X)
    wn = np.asarray(wn)
    design = np.vstack([wn, np.ones_like(wn)]).T
    coef, *_ = np.linalg.lstsq(design, X.T, rcond=None)
    return X - (design @ coef).T


def vector_norm(X: np.ndarray) -> np.ndarray:
    """Scale each spectrum to unit L2 norm.

    Lifted from ``run_char_11.vector_norm`` and
    ``etad_spectra.vector_normalize``.
    """
    X = np.asarray(X)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norm == 0, 1.0, norm)


def normalize_area(
    X: np.ndarray,
    wn: np.ndarray,
    *,
    negatives: str = "min-shift",
) -> np.ndarray:
    """Scale each spectrum to unit area using an explicit negative-value rule.

    ``negatives="min-shift"`` preserves the shape of zero-mean/SNV spectra by
    shifting every row to a zero minimum before integration. This is
    ``charcoal_spectra.shape_norm``.

    ``negatives="clip"`` clips negative values to zero before integration.
    This is the July07 convention used by ``run_char_10/11/12``.
    """
    X = np.asarray(X)
    wn = np.asarray(wn)

    if negatives == "min-shift":
        normalized = X - np.nanmin(X, axis=1, keepdims=True)
    elif negatives == "clip":
        normalized = np.clip(X, 0, None)
    else:
        raise ValueError(
            "negatives must be either 'min-shift' or 'clip'"
        )

    order = np.argsort(wn)
    area = np.trapezoid(
        np.nan_to_num(normalized[:, order]),
        wn[order],
        axis=1,
    )[:, None]
    area[area == 0] = np.nan
    return normalized / np.abs(area)


def second_derivative(
    X: np.ndarray,
    *,
    window_length: int = 15,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply the repository's Savitzky-Golay second derivative.

    The defaults are lifted from ``run_char_11.d2``. Keyword parameters make
    the physical smoothing window explicit for grids with other resolutions.
    """
    return savgol_filter(
        X,
        window_length,
        polyorder,
        deriv=2,
        axis=1,
    )
