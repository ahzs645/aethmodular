"""Broad IR continuum, PTFE-anchored blank backgrounds and a small prefix-k PLS (2 Nov 2026).

Why this exists
---------------
Black carbon absorbs across the whole mid-IR, so a loaded PTFE filter carries a smooth,
featureless continuum that grows with BC loading (about 1.4e-3 absorbance per µg EC at
1880 cm-1 and 6.5e-3 at 3900 cm-1 on IMPROVE filters). Both baselines used so far delete it:

* AIRSpec fits a smoothing spline through the analyte-free points, so the continuum is the
  baseline by construction (100 % removed).
* VIBES (87 blanks, rank 70) models every spectrum as blank mean + blank PCA components +
  a non-negative analyte, and its pinball loss pushes the background to the spectrum's floor.
  The blank components span smooth ramps, so an injected nu^1 continuum comes back at 0.7 %.

``anchored_correct`` keeps the same blank model (mean and components) but estimates the
component scores only from the *sharp* structure of the PTFE bands: both the spectrum and
the components are high-passed (spectrum minus a Savitzky-Golay smooth), and the scores are
fitted by least squares on the PTFE channels. A smooth continuum has no high-pass content,
so it cannot move the scores and survives; the PTFE background, including its own smooth
part, is predicted from the PTFE band shape and removed. On light IMPROVE filters this
leaves 1.1-1.7 µg-EC-equivalent of continuum scatter (3-6 µg for lot-mean subtraction)
and no lot 248/251 offset.

This is a linear stand-in for VIBES, not VIBES: no positivity, no per-spectrum noise model.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

# PTFE-dominated channels (C-F stretch doublet, the 500-700 bends, the 2365 overtone + CO2).
PTFE_WINDOWS = ((1080, 1290), (590, 690), (500, 580), (2300, 2420))
# Band-free windows for reading the continuum (avoid CO2/PTFE overtone, C-H, O-H/N-H, carbonyl).
CONTINUUM_WINDOWS = {"3900": (3850, 3950), "2600": (2550, 2650), "2050": (1980, 2120), "1880": (1850, 1910)}


def window_mask(wn, windows) -> np.ndarray:
    wn = np.asarray(wn, float)
    m = np.zeros(len(wn), bool)
    for lo, hi in windows:
        m |= (wn >= lo) & (wn <= hi)
    return m


def window_mean(X, wn, lo, hi) -> np.ndarray:
    """Mean of each row of X over lo <= wn <= hi."""
    m = (np.asarray(wn) >= lo) & (np.asarray(wn) <= hi)
    return np.asarray(X[:, m], float).mean(1)


def highpass(A, window=41, poly=2) -> np.ndarray:
    """Rows of A minus their Savitzky-Golay smooth (window in channels; ~1.29 cm-1 per channel)."""
    A = np.atleast_2d(np.asarray(A, float))
    return A - savgol_filter(A, window, poly, axis=1)


class AnchoredBackground:
    """Blank mean + components whose scores are fitted on the high-passed PTFE channels only."""

    def __init__(self, wn, mean, components, windows=PTFE_WINDOWS, window=41, ridge=1e-6):
        self.wn, self.mean, self.W = np.asarray(wn, float), np.asarray(mean, float), np.asarray(components, float)
        self.mask, self.window = window_mask(self.wn, windows), window
        D = highpass(self.W.T, window).T[self.mask]
        self._G = np.linalg.solve(D.T @ D + ridge * np.eye(self.W.shape[1]), D.T)

    def scores(self, X) -> np.ndarray:
        R = highpass(np.asarray(X, float) - self.mean, self.window)[:, self.mask]
        return (self._G @ R.T).T

    def correct(self, X, chunk=2000) -> np.ndarray:
        """Baseline-corrected spectra (float32): X - mean - scores @ components.T."""
        out = []
        for s in range(0, len(X), chunk):
            A = np.asarray(X[s:s + chunk], float)
            out.append((A - self.mean - self.scores(A) @ self.W.T).astype(np.float32))
        return np.vstack(out)


# --------------------------------------------------------------------------- PLS1 with every prefix k
def pls_fit(X, y, kmax, weights=None):
    """NIPALS PLS1; returns (x_mean, y_mean, B) with B[:, k-1] the k-component regression vector.

    ``weights`` gives a weighted fit (weighted centering, rows scaled by sqrt(w)).
    """
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    if weights is None:
        sw = np.ones(len(y))
    else:
        sw = np.asarray(weights, float) / np.mean(weights)
    xm = (sw[:, None] * X).sum(0) / sw.sum()
    ym = (sw * y).sum() / sw.sum()
    s = np.sqrt(sw)
    E, f = s[:, None] * (X - xm), s * (y - ym)
    p = X.shape[1]
    W, P, q = np.zeros((p, kmax)), np.zeros((p, kmax)), np.zeros(kmax)
    for a in range(kmax):
        w = E.T @ f
        w /= np.linalg.norm(w)
        t = E @ w
        tt = t @ t
        P[:, a], q[a], W[:, a] = E.T @ t / tt, f @ t / tt, w
        E -= np.outer(t, P[:, a])
        f = f - q[a] * t
    B = np.zeros((p, kmax))
    for k in range(1, kmax + 1):
        B[:, k - 1] = W[:, :k] @ np.linalg.solve(P[:, :k].T @ W[:, :k], q[:k])
    return xm, ym, B


def pls_predict(model, X) -> np.ndarray:
    """(n, kmax) predictions, one column per k."""
    xm, ym, B = model
    return (np.asarray(X, np.float64) - xm) @ B + ym


def ols(x, y):
    """Slope, intercept, R² and n of y on x over finite pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    s, i = np.polyfit(x[ok], y[ok], 1)
    return s, i, np.corrcoef(x[ok], y[ok])[0, 1] ** 2, int(ok.sum())
