"""Trial: ikpls vs the repo's validated PLS/CV machinery, on the 800 cohort.

Three questions, per docs/package-survey-2026-08-17.md section 2:

1. Does ``ikpls`` (Improved Kernel PLS, Dayal & MacGregor) reproduce
   ``sklearn.cross_decomposition.PLSRegression(scale=False)`` predictions?
2. Do ``ikpls.cross_validate`` fold labels reproduce the repo's two CV
   protocols — site-grouped 5-fold (``pls_transfer.component_cv_curve``) and
   interleaved 10-fold (``calibration_modes.interleaved_cv_curve``) — giving
   the same RMSECV-vs-k curves?
3. How does wall-clock for the full 1..30-component curve compare?

Run:  python validate_ikpls.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

from trial_common import load_cohort_800  # also wires up repo script paths

from calibration_modes import interleaved_cv_curve  # noqa: E402  (phase-3 scripts)
from pls_transfer import component_cv_curve  # noqa: E402  (phase-2 scripts)

import ikpls
from ikpls.numpy import PLS as IkPLS
from ikpls.fast_cross_validation.numpy import PLS as IkFastPLS

MAX_COMPONENTS = 30
K = 6


# --------------------------------------------------------------------------- #
# 1. single-fit equivalence at k = 6
# --------------------------------------------------------------------------- #
def fit_equivalence(X, y):
    print(f"\n== 1. Fit equivalence at k = {K} (ikpls {ikpls.__version__}) ==")
    sk = PLSRegression(n_components=K, scale=False).fit(X, y)
    pred_sk = sk.predict(X).ravel()

    for algorithm in (1, 2):
        # Match sklearn's PLSRegression(scale=False): centering on, scaling off.
        # ikpls defaults scale_X/scale_Y to True, so both must be disabled.
        ik = IkPLS(algorithm=algorithm, center_X=True, center_Y=True,
                   scale_X=False, scale_Y=False)
        ik.fit(X, y, A=K)
        pred_ik = ik.predict(X, n_components=K).ravel()
        delta = np.abs(pred_ik - pred_sk)
        print(f"algorithm #{algorithm}: max |Δpred| = {delta.max():.3e}, "
              f"median = {np.median(delta):.3e}  "
              f"(y scale: mean {y.mean():.2f} µg/filter)")
    return pred_sk


# --------------------------------------------------------------------------- #
# fold labels reproducing the repo's two protocols
# --------------------------------------------------------------------------- #
def site_grouped_fold_labels(X, y, sites, n_splits=5, random_state=42):
    """Per-sample fold ids for the exact GroupKFold splits component_cv_curve uses."""
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    labels = np.empty(len(y), dtype=int)
    for fold, (_, test) in enumerate(splitter.split(X, y, np.asarray(sites))):
        labels[test] = fold
    return labels


def rmse_per_component(Y_val, Y_pred):
    """metric_function: per-fold RMSE for every 1..A prefix. Y_pred is (A, n, 1)."""
    residual = Y_pred[:, :, 0] - np.asarray(Y_val).reshape(1, -1)
    return np.sqrt(np.mean(residual**2, axis=1))


def press_per_component(Y_val, Y_pred):
    """metric_function: (per-component SSE, n) so folds can be pooled app-style."""
    residual = Y_pred[:, :, 0] - np.asarray(Y_val).reshape(1, -1)
    return np.sum(residual**2, axis=1), residual.shape[1]


def ikpls_curve_mean_of_folds(X, y, labels, A=MAX_COMPONENTS, n_jobs=-1):
    """RMSECV curve as component_cv_curve reports it: mean of per-fold RMSEs."""
    ik = IkPLS(algorithm=1, center_X=True, center_Y=True,
               scale_X=False, scale_Y=False)
    out = ik.cross_validate(X, y, A=A, folds=labels,
                            metric_function=rmse_per_component,
                            n_jobs=n_jobs, verbose=0)
    return np.mean([out[fold] for fold in sorted(out)], axis=0)


def ikpls_curve_pooled(X, y, labels, A=MAX_COMPONENTS, n_jobs=-1):
    """RMSECV curve as interleaved_cv_curve reports it: pooled PRESS."""
    ik = IkPLS(algorithm=1, center_X=True, center_Y=True,
               scale_X=False, scale_Y=False)
    out = ik.cross_validate(X, y, A=A, folds=labels,
                            metric_function=press_per_component,
                            n_jobs=n_jobs, verbose=0)
    press = np.sum([out[fold][0] for fold in sorted(out)], axis=0)
    n = sum(out[fold][1] for fold in sorted(out))
    return np.sqrt(press / n)


def ikpls_fast_curve_pooled(X, y, labels, A=MAX_COMPONENTS, n_jobs=-1):
    """Same pooled curve via ikpls.fast_cross_validation (cvmatrix trick)."""
    ik = IkFastPLS(algorithm=1, center_X=True, center_Y=True,
                   scale_X=False, scale_Y=False)
    out = ik.cross_validate(X, y, A=A, folds=labels,
                            metric_function=press_per_component,
                            n_jobs=n_jobs, verbose=0)
    press = np.sum([out[fold][0] for fold in sorted(out)], axis=0)
    n = sum(out[fold][1] for fold in sorted(out))
    return np.sqrt(press / n)


def cv_equivalence(X, y, sites):
    print("\n== 2. CV-curve equivalence, k = 1..30 ==")
    results = {}

    # (a) site-grouped 5-fold
    t0 = time.perf_counter()
    repo_curve = component_cv_curve(X, y, range(1, MAX_COMPONENTS + 1),
                                    groups=sites, n_splits=5, random_state=42)
    t_repo_site = time.perf_counter() - t0
    labels_site = site_grouped_fold_labels(X, y, sites)

    t0 = time.perf_counter()
    ik_site = ikpls_curve_mean_of_folds(X, y, labels_site)
    t_ik_site = time.perf_counter() - t0
    delta = np.abs(ik_site - repo_curve["rmse_mean"].to_numpy())
    print(f"site-grouped 5-fold: max |ΔRMSECV| = {delta.max():.3e} "
          f"(curve floor {repo_curve['rmse_mean'].min():.4f} µg/filter)")
    results["site"] = (repo_curve, ik_site, delta.max())

    # (b) interleaved 10-fold (the app protocol)
    t0 = time.perf_counter()
    repo_interleaved = interleaved_cv_curve(X, y)
    t_repo_int = time.perf_counter() - t0
    labels_int = np.arange(len(y)) % 10

    t0 = time.perf_counter()
    ik_int = ikpls_curve_pooled(X, y, labels_int)
    t_ik_int = time.perf_counter() - t0
    delta = np.abs(ik_int - repo_interleaved["rmsecv"].to_numpy())
    print(f"interleaved 10-fold: max |ΔRMSECV| = {delta.max():.3e} "
          f"(curve floor {repo_interleaved['rmsecv'].min():.4f} µg/filter)")
    results["interleaved"] = (repo_interleaved, ik_int, delta.max())

    # timing (single warm run each; repeat for the serial ikpls variant)
    print("\n== 3. Wall-clock, full 1..30-component curve ==")
    t0 = time.perf_counter()
    ikpls_curve_mean_of_folds(X, y, labels_site, n_jobs=1)
    t_ik_site_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    ikpls_curve_pooled(X, y, labels_int, n_jobs=1)
    t_ik_int_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    fast_int = ikpls_fast_curve_pooled(X, y, labels_int)
    t_fast_int = time.perf_counter() - t0
    delta_fast = np.abs(fast_int - results["interleaved"][0]["rmsecv"].to_numpy())
    t0 = time.perf_counter()
    fast_site = ikpls_fast_curve_pooled(X, y, labels_site)  # pooled, so compare shape only
    t_fast_site = time.perf_counter() - t0

    timing = pd.DataFrame([
        ("site-grouped 5-fold", "repo component_cv_curve", t_repo_site),
        ("site-grouped 5-fold", "ikpls cross_validate (n_jobs=-1)", t_ik_site),
        ("site-grouped 5-fold", "ikpls cross_validate (n_jobs=1)", t_ik_site_serial),
        ("site-grouped 5-fold", "ikpls fast_cross_validation", t_fast_site),
        ("interleaved 10-fold", "repo interleaved_cv_curve", t_repo_int),
        ("interleaved 10-fold", "ikpls cross_validate (n_jobs=-1)", t_ik_int),
        ("interleaved 10-fold", "ikpls cross_validate (n_jobs=1)", t_ik_int_serial),
        ("interleaved 10-fold", "ikpls fast_cross_validation", t_fast_int),
    ], columns=["protocol", "engine", "seconds"])
    timing["seconds"] = timing["seconds"].round(2)
    print(timing.to_string(index=False))
    print(f"fast_cross_validation vs repo interleaved curve: "
          f"max |ΔRMSECV| = {delta_fast.max():.3e}")
    return results, timing


def main():
    X, y, sites, _, _ = load_cohort_800()
    print(f"800 cohort: X {X.shape}, {pd.Series(sites).nunique()} sites, "
          f"y mean {y.mean():.3f} µg/filter")
    fit_equivalence(X, y)
    cv_equivalence(X, y, sites)


if __name__ == "__main__":
    main()
