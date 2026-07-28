"""Shared FTIR spectral-grid, preprocessing, and band utilities.

The submodules are exposed for notebook-style use::

    from spectra import bands, grid, preprocess

The same modules can also be imported through
``research.ftir_hips_chem.scripts.spectra``.
"""

from . import bands, grid, preprocess
from .bands import band_integral, band_mean
from .grid import MASK_REGIONS, ascending, common_grid, exclude, resample
from .preprocess import (
    detrend,
    normalize_area,
    second_derivative,
    snv,
    vector_norm,
)

__all__ = [
    "MASK_REGIONS",
    "ascending",
    "band_integral",
    "band_mean",
    "bands",
    "common_grid",
    "detrend",
    "exclude",
    "grid",
    "normalize_area",
    "preprocess",
    "resample",
    "second_derivative",
    "snv",
    "vector_norm",
]
