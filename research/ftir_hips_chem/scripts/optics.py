"""Optical property calculations for multi-wavelength aethalometer data.

Currently the Absorption Angstrom Exponent (AAE) and its source classification.
This module exists because AAE was computed inline in five notebooks under two
mutually-negated sign conventions -- see ``aae`` for the details.
"""

import numpy as np
import pandas as pd

try:
    from config import AAE_REGIONS, WAVELENGTHS_NM
except ImportError:  # Support importing as research.ftir_hips_chem.scripts.*
    from .config import AAE_REGIONS, WAVELENGTHS_NM


__all__ = ['aae', 'aae_from_columns', 'classify_aae', 'aae_source_summary']


def aae(b_short, b_long, wl_short_nm, wl_long_nm):
    """Absorption Angstrom Exponent from two absorption (or BC) channels.

    Defined from ``b_abs ~ lambda ** -AAE``, i.e.

        AAE = -ln(b_short / b_long) / ln(wl_short / wl_long)

    Because ``wl_short < wl_long`` the denominator is negative, so a sample that
    absorbs more at the short wavelength (the biomass-burning signature) yields a
    **positive** AAE. Typical values: ~1 for fossil-fuel soot, ~2 for biomass.

    .. warning::
       Two inverted conventions existed in this project. ``addis_01`` /
       ``addis_04`` computed ``ln(IR/UV) / ln(880/375)``, which is the exact
       negative of the definition above -- it returns AAE <= 0 for all real
       aerosol. Combined with ``classify_aae`` (fossil below 0.9) that put
       **100 % of samples in the fossil-fuel class** regardless of the data; a
       genuine biomass sample with UV/IR = 5.5 (true AAE 2.0) was reported as
       -2.0 and classified fossil. ``addis_05`` and
       ``multisite_diurnal_wavelength_analysis`` used the correct form. The
       inverted notebooks also carried ``clip(-1, 3)``, whose negative lower
       bound accommodated the symptom.

    Parameters
    ----------
    b_short, b_long : array-like or Series
        Absorption or BC at the shorter and longer wavelength. Non-positive and
        non-finite values yield NaN rather than raising.
    wl_short_nm, wl_long_nm : float
        Wavelengths in nm. Use ``config.WAVELENGTHS_NM`` -- these are MA350
        channel centres, not the AE33 set.

    Returns
    -------
    Same container type as the input (Series in, Series out).
    """
    if wl_short_nm >= wl_long_nm:
        raise ValueError(
            f"wl_short_nm ({wl_short_nm}) must be shorter than wl_long_nm "
            f"({wl_long_nm}); passing them the other way round silently flips "
            "the sign of every AAE."
        )

    index = b_short.index if isinstance(b_short, pd.Series) else None
    s = np.asarray(b_short, dtype=float)
    L = np.asarray(b_long, dtype=float)

    usable = np.isfinite(s) & np.isfinite(L) & (s > 0) & (L > 0)
    out = np.full(s.shape, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        out[usable] = -np.log(s[usable] / L[usable]) / np.log(wl_short_nm / wl_long_nm)

    return pd.Series(out, index=index) if index is not None else out


def aae_from_columns(df, short='UV', long='IR', kind='BCc', wavelengths=None):
    """AAE from two named aethalometer channels of ``df``.

    Column names are built as ``f'{short} {kind}'`` (e.g. ``'UV BCc'``), and the
    wavelengths come from ``config.WAVELENGTHS_NM`` so the AE33/MA350 mix-up
    cannot recur.
    """
    wl = wavelengths or WAVELENGTHS_NM
    for name in (short, long):
        if name not in wl:
            raise KeyError(f"unknown channel {name!r}; known: {sorted(wl)}")

    cols = {n: f'{n} {kind}' for n in (short, long)}
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")

    return aae(df[cols[short]], df[cols[long]], wl[short], wl[long])


def classify_aae(values, regions=None):
    """Label each AAE as ``'fossil'`` / ``'mixed'`` / ``'biomass'``.

    Boundaries come from ``config.AAE_REGIONS``. NaN in, NaN out.
    """
    r = regions or AAE_REGIONS
    fossil_max, biomass_min = r['fossil_max'], r['biomass_min']
    if fossil_max >= biomass_min:
        raise ValueError(f"fossil_max ({fossil_max}) must be < biomass_min ({biomass_min})")

    index = values.index if isinstance(values, pd.Series) else None
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, None, dtype=object)
    finite = np.isfinite(v)
    out[finite & (v <= fossil_max)] = 'fossil'
    out[finite & (v >= biomass_min)] = 'biomass'
    out[finite & (v > fossil_max) & (v < biomass_min)] = 'mixed'
    return pd.Series(out, index=index) if index is not None else out


def aae_source_summary(values, regions=None):
    """Percentage split across the three source classes, plus AAE moments."""
    labels = classify_aae(values, regions)
    labels = pd.Series(labels) if not isinstance(labels, pd.Series) else labels
    finite = pd.Series(np.asarray(values, dtype=float)).dropna()
    total = int(labels.notna().sum())
    if total == 0:
        return {'n': 0}

    counts = labels.value_counts()
    return {
        'n': total,
        'fossil_pct': 100.0 * counts.get('fossil', 0) / total,
        'mixed_pct': 100.0 * counts.get('mixed', 0) / total,
        'biomass_pct': 100.0 * counts.get('biomass', 0) / total,
        'mean_aae': float(finite.mean()),
        'median_aae': float(finite.median()),
        'std_aae': float(finite.std()),
    }
