"""Ethiopia (ETAD / Addis Ababa) FTIR spectra for the charcoal comparison.

Thin wrapper over the phase-3 loader so the charcoal notebooks get Addis
spectra on the same terms every other phase-3 analysis uses: one row per
physical filter (replicate scans averaged), complete spectra only, joined to
HIPS Fabs, deployed FTIR EC, and season.

The heavy lifting lives in ``research/ftir_ec_phase3/scripts/phase3_common.py``
and ``research/ftir_hips_chem/scripts/pls_transfer.py`` — do not re-implement
the join here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE3_SCRIPTS = REPO_ROOT / "research" / "ftir_ec_phase3" / "scripts"
PHASE2_SCRIPTS = REPO_ROOT / "research" / "ftir_hips_chem" / "scripts"

for _p in (PHASE3_SCRIPTS, PHASE2_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load_etad(baselined: bool = False) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return ``(meta, X, wn)`` for Addis Ababa filters.

    ``meta`` is one row per filter with ``season``, ``Fabs``, ``EC_deployed_ugm3``
    and sampling dates. ``X`` is the absorbance matrix aligned to ``meta`` rows
    and ``wn`` the wavenumber axis (descending, ~1.286 cm-1 spacing,
    500-3998 cm-1).

    With ``baselined=True`` the AIRSpec-baselined spectra are substituted, read
    from the phase-3 NPZ cache (``etad_corrected_df6.npz``, the validated port of
    the APRLssb R code). Those cover **1425-3998 cm-1 only** — the segmented
    spline drops everything below 1425 — so the raw offset-corrected form is the
    default here, since the charcoal comparison leans on the 1000-1400 cm-1
    fingerprint region.
    """
    from phase3_common import load_addis_evaluation
    from config import season_for_month

    meta, X, wn = load_addis_evaluation(season_for_month)
    meta = meta.reset_index(drop=True)
    if not baselined:
        return meta, X, wn

    npz_path = (
        REPO_ROOT
        / "research"
        / "ftir_ec_phase3"
        / "output"
        / "corrected"
        / "etad_corrected_df6.npz"
    )
    npz = np.load(npz_path, allow_pickle=True)
    corrected = pd.DataFrame(npz["corrected"].astype(float))
    corrected["MediaId"] = npz["media_id"].astype(int)
    by_media = corrected.groupby("MediaId").mean()
    X_b = by_media.reindex(meta["MediaId"].astype(int)).to_numpy(float)
    keep = np.isfinite(X_b).all(axis=1)
    return meta.loc[keep].reset_index(drop=True), X_b[keep], npz["wn"].astype(float)


def offset_correct_spectra(
    X: np.ndarray, wn: np.ndarray, window: tuple[float, float] = (3900.0, 4000.0)
) -> np.ndarray:
    """Subtract each spectrum's mean absorbance in a nominally featureless window.

    This is the phase-2/3 convention (see ``pls_transfer.offset_correct``) and is
    what makes ETAD filter spectra visually comparable to lab charcoal spectra,
    which carry a different additive offset entirely.
    """
    mask = (wn >= window[0]) & (wn <= window[1])
    if not mask.any():
        # Fall back to the highest-wavenumber 100 cm-1 available.
        hi = wn.max()
        mask = wn >= hi - 100.0
    return X - X[:, mask].mean(axis=1, keepdims=True)


def vector_normalize(X: np.ndarray) -> np.ndarray:
    """Unit-L2-norm each spectrum, so shape can be compared across loadings.

    Filter spectra (micrograms on Teflon) and lab charcoal spectra (KBr pellet or
    ATR on bulk material) differ in absolute absorbance by orders of magnitude;
    only band *shape* is comparable without this.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def resample_to(
    X: np.ndarray, wn_from: np.ndarray, wn_to: np.ndarray
) -> np.ndarray:
    """Linearly interpolate spectra onto a shared wavenumber grid.

    Handles descending axes (both ETAD and several charcoal sets are stored
    high-to-low) by sorting ascending before interpolation.
    """
    order = np.argsort(wn_from)
    src = wn_from[order]
    out = np.empty((X.shape[0], wn_to.size), dtype=float)
    for i in range(X.shape[0]):
        out[i] = np.interp(wn_to, src, X[i][order], left=np.nan, right=np.nan)
    return out
