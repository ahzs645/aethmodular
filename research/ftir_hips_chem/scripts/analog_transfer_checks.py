"""Diagnostics for Satoshi's 2 Oct 2026 questions about the seasonal analog models.

Builds on ``analog_bias`` (the 17 Sep meeting module). Four questions:

1. How different are the analogs each baseline selects (shared / only-one-baseline),
   and is a difference in Addis results due to the training set or the baseline?
   Answered with a swap: select on baseline A, calibrate on baseline B.
2. What does PLS "see" as EC: VIP of the standard all-IMPROVE model vs the
   analog-trained models.
3. Is the problem in fitting or in extrapolation: the same statistics at each
   stage (calibration fit, cross-validation, IMPROVE test sites, Addis).
4. Do the analogs sit on the lower envelope of the standard model?

Units: IMPROVE comparisons in µg/m³ (TOR EC; loading / sampled volume). Addis is
FTIR EC vs HIPS Fabs / MAC 10, an optical comparison, not thermal EC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

import analog_bias as ab
from pls_transfer import (
    mahalanobis_distance_squared,
    project_scores,
    score_metric,
    spectral_q_residual,
    vip_scores,
)
from seasonal_analogs import mean_correlation_scores

# Similarity regions. "presented" is what the 1 Oct slides used (CO2 1800-2500 and
# everything above 3600 dropped; no lower bound, so VIBES selection ran to 500).
# "oct1" is the region agreed at the 1 Oct meeting: 600-3600, CO2 dropped, and the
# broad PTFE band 1050-1250 dropped for similarity only (PLS still sees it).
MASKS = {
    "presented": {"low": None, "high": 3600, "drop": [(1800, 2500)]},
    "oct1": {"low": 600, "high": 3600, "drop": [(1800, 2500), (1050, 1250)]},
}
MASK_LABELS = {
    "presented": "As presented 1 Oct (CO₂ and >3600 cm⁻¹ excluded)",
    "oct1": "Agreed 1 Oct (600–3600, CO₂ and 1050–1250 PTFE excluded)",
}
# Broad, deliberately non-specific bands for VIP summaries (cm⁻¹).
VIP_BANDS = (
    (3200, 3600, "O–H stretch"),
    (2800, 3000, "aliphatic C–H"),
    (2500, 2800, "2500–2800 shoulder"),
    (3000, 3200, "aromatic C–H / N–H"),
    (3600, 4000, "above 3600 (baseline)"),
    (1650, 1800, "carbonyl"),
    (1500, 1650, "aromatic C=C / 1600 band"),
    (1425, 1500, "1425–1500"),
    (1250, 1425, "1250–1425"),
    (1050, 1250, "PTFE C–F (1050–1250)"),
    (500, 1050, "500–1050"),
    (1800, 2500, "CO₂ / 1800–2500"),
)


def region_mask(wn, low=None, high=None, drop=()):
    wn = np.asarray(wn, float)
    keep = np.ones(wn.size, bool)
    if low is not None:
        keep &= wn >= low
    if high is not None:
        keep &= wn <= high
    for a, b in drop:
        keep &= ~((wn >= a) & (wn <= b))
    if keep.sum() < 3:
        raise ValueError("fewer than three channels retained")
    return keep


def select_analogs(ctx, method, target_mask, mask_name, n=ab.N_ANALOGS):
    """Top-n IMPROVE filters by Pearson r to the group's median Addis spectrum.

    Identical recipe to ``analog_bias.correlation_analogs`` (median target, ties
    broken by filter id), with the region given by name. With mask "presented"
    this reproduces the 17 Sep / 1 Oct selections exactly.
    """
    keep = region_mask(ctx.wn_of(method), **MASKS[mask_name])
    X, XA = ctx.X[method], ctx.XA[method][np.asarray(target_mask, bool)]
    target = np.median(XA, axis=0)[None, :]
    scores = mean_correlation_scores(X, target, keep)
    order = np.lexsort((ctx.lib.filter_id.to_numpy(), -scores))
    return order[:n], scores


def out_of_fold(X, y, groups, k, n_splits=5, seed=42):
    """Site-grouped out-of-fold predictions at a fixed number of factors.

    Same fold scheme as the protocol's CV curve (GroupKFold, shuffled, seed 42),
    so these are the predictions behind the RMSECV that chose k.
    """
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    pred = np.full(len(y), np.nan)
    folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in folds.split(X, y, groups):
        m = PLSRegression(n_components=int(min(k, len(tr) - 1)), scale=False).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te]).ravel()
    return pred


def fit_stats(x, y, sigma_x, sigma_y):
    """FTIR (y) vs reference (x): R², RMSE, bias, NRMSE and the Deming line."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    out = {"n": int(ok.sum())}
    if ok.sum() < 3:
        return out
    st = ab.xy_stats(x[ok], y[ok], sigma_x=sigma_x, sigma_y=sigma_y)
    out.update({k: st[k] for k in ("R2", "RMSE", "bias", "deming_slope", "deming_intercept",
                                    "ols_slope", "ols_intercept", "x_median")})
    out["x_mean"] = float(np.mean(x[ok]))
    out["x_p90"] = float(np.percentile(x[ok], 90))
    out["nrmse"] = out["RMSE"] / out["x_mean"] if out["x_mean"] > 0 else np.nan
    out["deming_lambda"] = (sigma_y / sigma_x) ** 2
    return out


def extrapolation(model, X_train, X_new):
    """Score leverage (T²) and spectral residual (Q) of new spectra vs the training set.

    Returns per-row T² and Q for the new spectra plus the training 95th
    percentiles (in-sample, so a conservative threshold: training rows sit inside
    their own model by construction).
    """
    scores, inv = score_metric(model)
    t2_train = mahalanobis_distance_squared(scores, inv)
    q_train = spectral_q_residual(model, X_train)
    t2_new = mahalanobis_distance_squared(project_scores(model, X_new), inv)
    q_new = spectral_q_residual(model, X_new)
    return {
        "t2": t2_new, "q": q_new,
        "t2_p95": float(np.percentile(t2_train, 95)), "q_p95": float(np.percentile(q_train, 95)),
        "t2_train_median": float(np.median(t2_train)), "q_train_median": float(np.median(q_train)),
    }


def vip_band_table(wn, vip):
    """Share of squared VIP and mean VIP per broad band (bands absent from the grid skipped)."""
    wn, vip = np.asarray(wn, float), np.asarray(vip, float)
    total = float(np.sum(vip ** 2))
    rows = []
    for lo, hi, label in VIP_BANDS:
        keep = (wn >= lo) & (wn < hi)
        if not keep.any():
            continue
        rows.append({"band": label, "low": lo, "high": hi, "n_channels": int(keep.sum()),
                     "vip2_share": float(np.sum(vip[keep] ** 2) / total),
                     "vip_mean": float(np.mean(vip[keep])),
                     "vip_ge1_fraction": float(np.mean(vip[keep] >= 1))})
    return pd.DataFrame(rows)


def conditional_percentile(x, residual, ref_x, ref_residual, n_bins=10):
    """Percentile of each residual among reference residuals at similar x.

    Bins are deciles of ``ref_x``; 50 means "in the middle of the standard model's
    scatter at that concentration", low values mean the lower envelope.
    """
    ref_x, ref_residual = np.asarray(ref_x, float), np.asarray(ref_residual, float)
    ok = np.isfinite(ref_x) & np.isfinite(ref_residual)
    ref_x, ref_residual = ref_x[ok], ref_residual[ok]
    edges = np.unique(np.quantile(ref_x, np.linspace(0, 1, n_bins + 1)))
    ref_bin = np.clip(np.searchsorted(edges, ref_x, side="right") - 1, 0, len(edges) - 2)
    sorted_by_bin = [np.sort(ref_residual[ref_bin == b]) for b in range(len(edges) - 1)]
    x, residual = np.asarray(x, float), np.asarray(residual, float)
    out = np.full(len(x), np.nan)
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    for i in np.flatnonzero(np.isfinite(x) & np.isfinite(residual)):
        ref = sorted_by_bin[b[i]]
        out[i] = 100.0 * np.searchsorted(ref, residual[i]) / len(ref)
    return out


__all__ = [
    "MASKS", "MASK_LABELS", "VIP_BANDS", "region_mask", "select_analogs", "out_of_fold",
    "fit_stats", "extrapolation", "vip_band_table", "conditional_percentile", "vip_scores",
]
