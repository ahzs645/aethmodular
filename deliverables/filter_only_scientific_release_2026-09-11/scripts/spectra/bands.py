"""Summary operations over spectral bands."""

from __future__ import annotations

import numpy as np


def _band_mask(
    wn: np.ndarray, window: tuple[float, float]
) -> np.ndarray:
    lo, hi = sorted(window)
    return (wn >= lo) & (wn <= hi)


def band_mean(
    X: np.ndarray,
    wn: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    """Return the mean value in an inclusive band for each spectrum.

    Lifted from ``charcoal_spectra.band_area``, whose operation is a mean
    despite its historical name.
    """
    X = np.asarray(X)
    wn = np.asarray(wn)
    mask = _band_mask(wn, window)
    if not mask.any():
        return np.full(X.shape[0], np.nan)
    return np.nanmean(X[:, mask], axis=1)


def band_integral(
    X: np.ndarray,
    wn: np.ndarray,
    window: tuple[float, float],
    *,
    baseline: str = "none",
) -> np.ndarray:
    """Integrate an inclusive spectral band on a sorted wavenumber axis.

    ``baseline="min"`` subtracts each spectrum's in-band minimum before
    integration, matching
    ``_build_13_chase_origin_and_biomass.band_area``. Unlike that local
    implementation, this function sorts the band internally, so its sign is
    correct for both ascending and descending axes without a sign hack.
    """
    X = np.asarray(X)
    wn = np.asarray(wn)
    mask = _band_mask(wn, window)
    if not mask.any():
        return np.full(X.shape[0], np.nan)

    band_wn = wn[mask]
    band_X = X[:, mask]
    if baseline == "min":
        band_X = band_X - np.nanmin(band_X, axis=1, keepdims=True)
    elif baseline != "none":
        raise ValueError("baseline must be either 'none' or 'min'")

    order = np.argsort(band_wn)
    return np.trapezoid(
        band_X[:, order],
        band_wn[order],
        axis=1,
    )
