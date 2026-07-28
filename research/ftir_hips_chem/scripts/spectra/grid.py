"""Wavenumber-grid utilities shared by FTIR analyses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


# Named spectral windows that should be excluded when their known artifacts
# would dominate a shape comparison.
MASK_REGIONS = {
    "co2_gas_phase": (2280.0, 2400.0),
    "ptfe_cf": (1100.0, 1300.0),
}


def ascending(
    wn: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return a wavenumber axis and aligned spectra sorted low-to-high.

    Lifted from ``charcoal_spectra._ascending``.
    """
    wn = np.asarray(wn)
    X = np.asarray(X)
    order = np.argsort(wn)
    return wn[order], X[..., order]


def resample(
    X: np.ndarray, wn_from: np.ndarray, wn_to: np.ndarray
) -> np.ndarray:
    """Linearly interpolate spectra onto a target grid.

    The source axis is sorted internally, so ascending and descending source
    grids behave identically. Values beyond the source range are NaN.

    Lifted from ``charcoal_spectra.resample`` and
    ``etad_spectra.resample_to``.
    """
    X = np.asarray(X)
    wn_from = np.asarray(wn_from)
    wn_to = np.asarray(wn_to)
    order = np.argsort(wn_from)
    src = wn_from[order]
    out = np.empty((X.shape[0], wn_to.size), dtype=float)
    for i in range(X.shape[0]):
        out[i] = np.interp(
            wn_to,
            src,
            X[i][order],
            left=np.nan,
            right=np.nan,
        )
    return out


def common_grid(
    step: float = 2.0, window: tuple[float, float] = (951.0, 3500.0)
) -> np.ndarray:
    """Build an inclusive ascending grid over a spectral window.

    The default window preserves ``charcoal_spectra.CHARCOAL_OVERLAP``.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    lo, hi = sorted(window)
    return np.arange(lo, hi + step / 2, step)


def _resolve_regions(
    regions: Mapping[str, tuple[float, float]]
    | Iterable[str | tuple[float, float]]
    | None,
) -> list[tuple[float, float]]:
    if regions is None:
        return list(MASK_REGIONS.values())
    if isinstance(regions, Mapping):
        return list(regions.values())

    resolved = []
    for region in regions:
        if isinstance(region, str):
            try:
                resolved.append(MASK_REGIONS[region])
            except KeyError as exc:
                names = ", ".join(sorted(MASK_REGIONS))
                raise ValueError(
                    f"unknown mask region {region!r}; choose from {names}"
                ) from exc
        else:
            resolved.append(region)
    return resolved


def exclude(
    wn: np.ndarray,
    X: np.ndarray,
    regions: Mapping[str, tuple[float, float]]
    | Iterable[str | tuple[float, float]]
    | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove named or explicit wavenumber windows from an axis and spectra.

    Parameters
    ----------
    wn
        Wavenumber axis in either direction.
    X
        Spectrum or spectra whose last dimension aligns with ``wn``.
    regions
        A mapping of names to windows, or an iterable containing names from
        :data:`MASK_REGIONS` and/or ``(low, high)`` windows. By default both
        standard artifact regions are removed.

    Returns
    -------
    tuple
        ``(wn_kept, X_kept)`` in the input axis order.
    """
    wn = np.asarray(wn)
    X = np.asarray(X)
    keep = np.ones(wn.size, dtype=bool)
    for window in _resolve_regions(regions):
        lo, hi = sorted(window)
        keep &= ~((wn >= lo) & (wn <= hi))
    return wn[keep], X[..., keep]
