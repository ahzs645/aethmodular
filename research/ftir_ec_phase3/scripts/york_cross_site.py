"""York (per-filter weighted errors-in-variables) re-fit of the five-site table.

Replaces the pooled-lambda Deming rows in CROSS_SITE_EVALUATION with fits that
use the real per-filter HIPS Fabs uncertainties (York et al. 2004, Am. J.
Phys. 72:367 — weighted Deming generalized to heteroscedastic errors). Also
reports the per-site range-to-noise diagnostic kappa: sites with kappa well
below ~0.8 cannot support a free EIV slope (Fuller 1987; Linnet 1998), which
is the Pasadena story. For those sites the shared-slope profile fit is the
quotable number.

Needs the explorer server on :5058 (predictions come from /api/run so the
fits are exactly the app's winner configuration).

Usage: python york_cross_site.py [cohort cutoff spectra k]   (default ocec 450 airspec 9)
"""
import json
import pickle
import sys
import urllib.request

import numpy as np
import pandas as pd

BASE = "/Users/ahmadjalil/github/aethmodular/"
PKL = BASE + "research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl"
API = "http://127.0.0.1:5058/api/run"
SITES = {"addis": "ETAD", "etbi": "ETBI", "chts": "CHTS", "indh": "INDH", "uspa": "USPA"}
MAC = 10.0


def hips_sigma_models():
    """Per-site sigma_Fabs(Fabs) from the unified dataset: join HIPS_Fabs with
    HIPS_Uncertainty by FilterId, fit sigma^2 = a^2 + (b*F)^2. ETBI has no pkl
    rows -> pooled non-ETAD model."""
    with open(PKL, "rb") as f:
        d = pickle.load(f)
    fab = d[d.Parameter == "HIPS_Fabs"][["Site", "FilterId", "Concentration"]]
    unc = d[d.Parameter == "HIPS_Uncertainty"][["Site", "FilterId", "Concentration"]]
    j = fab.merge(unc, on=["Site", "FilterId"], suffixes=("", "_u")).dropna()
    models = {}

    def fit(g):
        F, s = g["Concentration"].to_numpy(float), g["Concentration_u"].to_numpy(float)
        A = np.vstack([np.ones_like(F), F**2]).T
        coef, *_ = np.linalg.lstsq(A, s**2, rcond=None)
        a2, b2 = max(coef[0], 1e-4), max(coef[1], 0.0)
        return lambda x: np.sqrt(a2 + b2 * np.asarray(x, float) ** 2)

    for site, g in j.groupby("Site"):
        models[site] = fit(g)
    models["ETBI"] = fit(j[j.Site != "ETAD"])
    return models


def york(x, y, sx, sy, tol=1e-12, itmax=200):
    """York et al. 2004 iterative solution (uncorrelated errors). Returns
    b, a, se_b, se_a, mswd."""
    wx, wy = 1.0 / sx**2, 1.0 / sy**2
    b = np.polyfit(x, y, 1)[0]
    for _ in range(itmax):
        W = wx * wy / (wx + b**2 * wy)
        Xb, Yb = np.sum(W * x) / np.sum(W), np.sum(W * y) / np.sum(W)
        U, V = x - Xb, y - Yb
        beta = W * (U / wy + b * V / wx)
        b_new = np.sum(W * beta * V) / np.sum(W * beta * U)
        if abs(b_new - b) < tol:
            b = b_new
            break
        b = b_new
    a = Yb - b * Xb
    xi = Xb + beta
    xbar = np.sum(W * xi) / np.sum(W)
    u = xi - xbar
    se_b = np.sqrt(1.0 / np.sum(W * u**2))
    se_a = np.sqrt(1.0 / np.sum(W) + xbar**2 * se_b**2)
    mswd = np.sum(W * (y - b * x - a) ** 2) / (len(x) - 2)
    return b, a, se_b, se_a, mswd


def shared_slope_profile(data):
    """Profile total York chi^2 over a shared slope with free per-site
    intercepts; returns b*, se_b, and per-site (alpha, se_alpha)."""

    def chi2(b):
        tot = 0.0
        for x, y, sx, sy in data.values():
            W = 1.0 / (sy**2 + b**2 * sx**2)
            a = np.sum(W * (y - b * x)) / np.sum(W)
            tot += np.sum(W * (y - b * x - a) ** 2)
        return tot

    bs = np.linspace(0.3, 3.0, 541)
    c = np.array([chi2(b) for b in bs])
    i = int(np.argmin(c))
    # parabolic refine + curvature -> se_b (delta chi2 = 1)
    b0 = bs[i]
    h = bs[1] - bs[0]
    if 0 < i < len(bs) - 1:
        denom = c[i - 1] - 2 * c[i] + c[i + 1]
        b0 = bs[i] + h * 0.5 * (c[i - 1] - c[i + 1]) / denom
        se_b = h * np.sqrt(2.0 / denom)
    else:
        se_b = np.nan
    out = {}
    for site, (x, y, sx, sy) in data.items():
        W = 1.0 / (sy**2 + b0**2 * sx**2)
        a = np.sum(W * (y - b0 * x)) / np.sum(W)
        xb = np.sum(W * x) / np.sum(W)
        se_a = np.sqrt(1.0 / np.sum(W) + xb**2 * se_b**2)
        out[site] = (a, se_a)
    return b0, se_b, out


def main():
    cohort, cutoff, spectra, k = (sys.argv[1:5] + ["ocec", "450", "airspec", "9"][len(sys.argv) - 1 :])[:4]
    sig = hips_sigma_models()
    data, free = {}, {}
    print(f"config: {cohort}-{cutoff} x {spectra} k={k}  (x = Fabs/{MAC:g}, per-filter sigma_x from HIPS)")
    print(f"{'site':6s} {'n':>4s} {'kappa':>6s} | {'free York slope':>18s} {'intercept':>16s} {'mswd':>5s}")
    for target, code in SITES.items():
        body = {"cohort": cohort, "cutoff": int(cutoff), "spectra": spectra,
                "mode": "site_heldout", "target": target, "k": int(k)}
        req = urllib.request.Request(API, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        pred = np.array(r["eval"]["pred"], float)
        fabs = np.array(r["eval"]["ref"], float)
        x, sx = fabs / MAC, sig[code](fabs) / MAC
        # sigma_y: per-site scale chosen so the *free* fit has mswd ~ 1
        sy = np.full_like(x, max(np.std(pred - np.polyval(np.polyfit(x, pred, 1), x)), 1e-3))
        for _ in range(6):
            b, a, se_b, se_a, mswd = york(x, pred, sx, sy)
            sy = sy * np.sqrt(max(mswd, 1e-6))
        kappa = max(0.0, 1.0 - np.mean(sx**2) / np.var(x))
        data[target] = (x, pred, sx, sy)
        free[target] = (b, a, se_b, se_a, mswd, kappa, len(x))
        print(f"{target:6s} {len(x):4d} {kappa:6.2f} | {b:8.2f} +/- {se_b:5.2f} "
              f"{a:8.2f} +/- {se_a:5.2f} {mswd:5.2f}")

    b0, se_b, alphas = shared_slope_profile(data)
    print(f"\nshared-slope profile: b = {b0:.3f} +/- {se_b:.3f}")
    print(f"{'site':6s} {'alpha (ug/m3)':>14s} {'alpha (Mm-1)':>13s}  z")
    for site, (a, se_a) in alphas.items():
        print(f"{site:6s} {a:8.2f} +/- {se_a:4.2f} {a*MAC:8.1f} +/- {se_a*MAC:4.1f} {a/se_a:5.1f}")


if __name__ == "__main__":
    main()
