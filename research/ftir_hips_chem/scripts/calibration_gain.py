"""What a calibration responds to, scored without selecting on the reference's own error.

Used by ``workflows/run_calibration_gain_20261110.py``. Plain NumPy on spectra and per-filter reference values,
so it runs the same on raw spectra from the IMPROVE tables and on a saved bundle of baseline-corrected spectra.

The questions:

* ``response``       across site means, prediction = offset + gain x reference EC (+ response x OC). A gain below 1
                     with a positive offset is a calibration that is right in the middle of its training range and
                     reads low on heavy filters. Site means average out the reference's per-filter error.
* ``fair_scores``    prediction ÷ reference on groups chosen by site or by the previous sample at the site, never
                     by a filter's own reference value, beside the errors on light filters and overall.
* ``recalibrate_by_site``  remove the offset and divide by the gain, with both fitted on other sites only.

Nothing here uses a target-site measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import continuum_anchor as ca

__all__ = ["site_folds", "held_out_predictions", "response", "fair_scores", "recalibrate_by_site", "loading_weights"]

LIGHT_UG, HEAVY_UG = 3.0, 15.0


def site_folds(sites, n_splits=5):
    """Whole sites held out: (train, test) row indices for each fold."""
    sites = np.asarray(sites)
    return list(GroupKFold(n_splits).split(sites, groups=sites))


def loading_weights(ec, cap=8.0):
    """Weights that grow with loading: reference EC over its median, between 1 and ``cap``."""
    ec = np.asarray(ec, float)
    return np.clip(ec / np.median(ec), 1.0, cap)


def held_out_predictions(x, y, folds, kmax, *, weights=None, extra_x=None, extra_y=None, extra_weight=1.0, predict=None):
    """Held-out predictions for every component count 1..kmax, and all-data predictions for other spectra.

    ``extra_x``/``extra_y`` are rows added to every training fold and never scored (blank filters with a
    reference of zero, for a fit that has to pass through the blanks). ``predict`` maps names to spectra that
    get the all-data model (blanks, a target site). Returns ``(held_out (n, kmax), {name: (m, kmax)})``.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    base = np.ones(len(y)) if weights is None else np.asarray(weights, float)

    def fit(rows):
        if extra_x is None:
            return ca.pls_fit(x[rows], y[rows], kmax, weights=None if weights is None else base[rows])
        xs, ys = np.vstack([x[rows], extra_x]), np.r_[y[rows], extra_y]
        return ca.pls_fit(xs, ys, kmax, weights=np.r_[base[rows], np.full(len(extra_y), float(extra_weight))])

    held_out = np.full((len(y), kmax), np.nan)
    for train, test in folds:
        held_out[test] = ca.pls_predict(fit(train), x[test])
    everything = fit(np.arange(len(y)))
    return held_out, {name: ca.pls_predict(everything, np.asarray(spectra, float)) for name, spectra in (predict or {}).items()}


def _site_means(prediction, ec, oc, sites, min_filters):
    frame = pd.DataFrame({"site": np.asarray(sites), "prediction": prediction, "ec": ec, "oc": oc}).dropna()
    means = frame.groupby("site").agg(n=("ec", "size"), prediction=("prediction", "mean"), ec=("ec", "mean"), oc=("oc", "mean"))
    return means[means["n"] >= min_filters]


def response(prediction, ec, oc, sites, *, with_oc=True, min_filters=30, n_boot=500, seed=20261110) -> dict[str, float]:
    """Offset, gain per unit of reference EC and (optionally) response per unit of OC, from site means.

    Intervals are 95 % percentile bootstrap over sites.
    """
    means = _site_means(prediction, ec, oc, sites, min_filters)
    columns = ["ec", "oc"] if with_oc else ["ec"]

    def solve(frame):
        design = np.column_stack([np.ones(len(frame)), frame[columns].to_numpy()])
        return np.linalg.lstsq(design, frame["prediction"].to_numpy(), rcond=None)[0]

    centre = solve(means)
    rng = np.random.default_rng(seed)
    boot = np.array([solve(means.iloc[rng.integers(0, len(means), len(means))]) for _ in range(int(n_boot))])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0) if len(boot) else (np.full(len(centre), np.nan),) * 2
    out = {"sites": int(len(means)), "offset": float(centre[0]), "offset_lo": float(lo[0]), "offset_hi": float(hi[0]),
           "gain": float(centre[1]), "gain_lo": float(lo[1]), "gain_hi": float(hi[1])}
    if with_oc:
        out |= {"oc_response": float(centre[2]), "oc_lo": float(lo[2]), "oc_hi": float(hi[2])}
    return out


def fair_scores(prediction, ec, sites, dates, urban_sites=()) -> dict[str, float]:
    """Level on fairly chosen groups, and the errors on light filters and overall.

    Heavy filters are chosen by the *previous* sample at the site (within a week), so this filter's own
    reference error cannot decide whether it is counted. Light filters are chosen the same way.
    """
    frame = pd.DataFrame({"site": np.asarray(sites), "date": pd.to_datetime(np.asarray(dates)), "prediction": prediction, "ec": ec})
    frame = frame.sort_values(["site", "date"]).reset_index(drop=True)
    gap = (frame["date"] - frame.groupby("site")["date"].shift(1)).dt.days
    previous = frame.groupby("site")["ec"].shift(1).where(gap.between(1, 7))
    ratio, error = frame["prediction"] / frame["ec"], frame["prediction"] - frame["ec"]
    heavy, light = previous >= HEAVY_UG, previous < LIGHT_UG
    urban = frame["site"].isin(list(urban_sites))
    median = lambda mask: float(ratio[mask].median()) if mask.any() else float("nan")  # noqa: E731
    return {
        "n": int(len(frame)), "overall_median_abs_error": float(error.abs().median()), "overall_rmse": float(np.sqrt((error ** 2).mean())),
        "heavy_n": int(heavy.sum()), "heavy_level": median(heavy), "heavy_sum_ratio": float(frame.loc[heavy, "prediction"].sum() / frame.loc[heavy, "ec"].sum()) if heavy.any() else float("nan"),
        "urban_n": int(urban.sum()), "urban_level": median(urban),
        "light_n": int(light.sum()), "light_level": median(light), "light_median_abs_error": float(error[light].abs().median()) if light.any() else float("nan"),
    }


def recalibrate_by_site(prediction, ec, sites, folds, *, min_filters=30) -> np.ndarray:
    """(prediction - offset) / gain, with offset and gain fitted on the site means of the other folds' sites."""
    prediction, ec, sites = np.asarray(prediction, float), np.asarray(ec, float), np.asarray(sites)
    corrected = np.full(len(prediction), np.nan)
    for train, test in folds:
        fit = response(prediction[train], ec[train], ec[train], sites[train], with_oc=False, min_filters=min_filters, n_boot=0)
        corrected[test] = (prediction[test] - fit["offset"]) / fit["gain"]
    return corrected
