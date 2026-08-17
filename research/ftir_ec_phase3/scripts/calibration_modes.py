"""The two calibration protocols, as switchable modes.

Every phase-2/3 calibration is the same PLS math; what differs between this project and
the AQRC **FTIR Calibration** Shiny app is the *protocol* wrapped around it — how the
cross-validation folds are drawn, how the component count is read off the resulting curve,
and how much of the cohort the final model is fitted on.

Two modes are defined here so any notebook can run a cohort both ways and compare:

``app``
    The Shiny app's protocol (``R/calibrateServer.R``):
    ``pls::plsr(ncomp = 80, validation = "CV", segments = 10,
    segment.type = "interleaved")`` — 10-fold *interleaved* folds with no site grouping,
    k read off the RMSEP curve (reproduced deterministically as "first k within 5% of the
    curve minimum"), and the shipped model fitted on **every** filter in the cohort. There
    is no site-disjoint hold-out by construction; that is a property of the protocol, not
    an omission here.

``site_heldout``
    The protocol locked in ftir_10/ftir_11 (referred to as the site-held-out protocol): a site-disjoint 20% TOR test split off
    *before* fitting (seed 20260717), a **site-grouped** 5-fold CV curve on the training
    part only, k by the first-major-minimum rule (earliest local minimum within one
    standard error of the global minimum), and the model fitted on the training part.

Both modes return Addis predictions. Addis is external to both training sets, so Addis
metrics are directly comparable across modes; the held-out TOR test exists only for
``site_heldout``.

Used by ftir_21; the CV-curve internals are shared with ftir_20.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupShuffleSplit

from pls_transfer import (
    component_cv_curve, predict_pls_components, regression_metrics,
    select_first_major_minimum,
)

SPLIT_SEED = 20260717          # locked in ftir_10; identical across phase 2 and 3
MAX_COMPONENTS = 30
APP_FOLDS, SITE_HELDOUT_FOLDS = 10, 5
APP_TOLERANCE = 0.05

# λ = σy²/σx² for Deming regression on the Addis crossplot at MAC = 10, from the
# HIPS_Uncertainty parameter rows (median 2.9075 Mm⁻¹ at ETAD, d1e4bce). x = Fabs/MAC,
# so σx² ∝ 1/MAC² and λ ∝ MAC² — see deming_lambda(); that scaling is the one under
# which the ftir_19 MAC pivot survives Deming exactly (ftir_31).
DEMING_LAMBDA_MAC10 = 2.96


def deming_lambda(mac: float) -> float:
    """The Deming error-variance ratio λ for an Fabs/MAC x-axis at this MAC."""
    return DEMING_LAMBDA_MAC10 * (mac / 10.0) ** 2

MODES = ('app', 'site_heldout')

# Both protocols are named for what they do, not for who uses them.
MODE_LABELS = {
    'app': 'Calibration app — interleaved CV, within-5%, fitted on all filters',
    'site_heldout': 'Site-held-out — site-grouped CV, first-major-minimum, disjoint fit',
}
MODE_SHORT = {'app': 'Calibration app', 'site_heldout': 'Site-held-out'}


def _interleaved_curve_ikpls(X, y, fold, usable):
    """Fast path for the pooled interleaved curve via `ikpls.fast_cross_validation`.

    Same math as the pure-numpy loop below — validated against it to ~3e-12 in
    `research/package_trials/chemometrics/validate_ikpls.py`, ~12x faster. ikpls
    defaults scale_X/scale_Y to True, so both must be disabled to match
    ``PLSRegression(scale=False)``.
    """
    from ikpls.fast_cross_validation.numpy import PLS as _FastPLS

    def _press(Y_val, Y_pred):
        residual = Y_pred[:, :, 0] - np.asarray(Y_val).reshape(1, -1)
        return np.sum(residual ** 2, axis=1), residual.shape[1]

    import contextlib, io
    ik = _FastPLS(algorithm=1, center_X=True, center_Y=True,
                  scale_X=False, scale_Y=False)
    with contextlib.redirect_stdout(io.StringIO()):   # it prints even at verbose=0
        out = ik.cross_validate(X, y, A=usable, folds=fold,
                                metric_function=_press, n_jobs=-1, verbose=0)
    press = np.sum([out[f][0] for f in sorted(out)], axis=0)
    n = sum(out[f][1] for f in sorted(out))
    return np.sqrt(press / n)


def interleaved_cv_curve(X, y, folds=APP_FOLDS, max_components=MAX_COMPONENTS):
    """Pooled RMSECV over interleaved folds — the app's `pls` CV scheme.

    Uses the ikpls fast-CV engine when installed (agreement ~3e-12, far below any
    reported precision); otherwise falls back to the original pure-numpy loop.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    fold = np.arange(len(y)) % folds
    usable = int(min(max_components,
                     min((fold != i).sum() for i in range(folds)) - 1, X.shape[1]))
    candidates = list(range(1, usable + 1))
    if X.size >= 1_000_000:   # below this the pure loop wins (parallel overhead)
        try:
            rmsecv = _interleaved_curve_ikpls(X, y, fold, usable)
            return pd.DataFrame({'n_components': candidates, 'rmsecv': rmsecv})
        except ImportError:
            pass
        except Exception as exc:   # never let the accelerator break a calibration
            import warnings
            warnings.warn(f'ikpls fast CV failed ({exc!r}); using the pure-numpy loop')
    press = np.zeros(usable)
    for i in range(folds):
        train, test = fold != i, fold == i
        model = PLSRegression(n_components=usable, scale=False).fit(X[train], y[train])
        press += np.sum((predict_pls_components(model, X[test], candidates)
                         - y[test][:, None]) ** 2, axis=0)
    return pd.DataFrame({'n_components': candidates, 'rmsecv': np.sqrt(press / len(y))})


def select_within_tolerance(curve, tolerance=APP_TOLERANCE):
    """The app-style read of an RMSEP curve: first k within `tolerance` of the minimum."""
    rmsecv = curve['rmsecv'].to_numpy(float)
    return int(curve['n_components'].to_numpy()[
        np.argmax(rmsecv <= rmsecv.min() * (1 + tolerance))])


@dataclass
class CalibrationFit:
    """One cohort fitted under one protocol."""

    mode: str
    cohort: str
    k: int
    n_train: int
    n_train_sites: int
    rmsecv_floor: float                     # µg/filter, at the selected k
    pct_rmsecv_floor: float                 # 100 × floor / mean cohort loading
    addis_ugm3: np.ndarray
    curve: pd.DataFrame
    heldout: dict | None = None             # site_heldout only
    model: PLSRegression | None = field(default=None, repr=False)


def protocol_train_mask(mode, X, y, sites):
    """The protocol's training mask over the cohort rows.

    ``app`` fits the shipped model on everything it was given (no hold-out, by
    construction); ``site_heldout`` is the locked ftir_10 site-disjoint 80/20 split.
    """
    if mode not in MODES:
        raise ValueError(f'unknown mode {mode!r}; expected one of {MODES}')
    y = np.asarray(y, float).reshape(-1)
    if mode == 'app':
        return np.ones(len(y), bool)
    sites = np.asarray(sites)
    train_pos, test_pos = next(GroupShuffleSplit(
        n_splits=1, test_size=.20, random_state=SPLIT_SEED).split(X, groups=sites))
    assert set(sites[train_pos]).isdisjoint(sites[test_pos])
    train = np.zeros(len(y), bool)
    train[train_pos] = True
    return train


def protocol_cv_curve(mode, X, y, sites, train, max_components=MAX_COMPONENTS):
    """The protocol's RMSECV-vs-k curve on the training rows.

    Columns: ``n_components``, ``rmsecv`` (and ``rmse_se`` for ``site_heldout``,
    which the first-major-minimum rule needs).
    """
    if mode not in MODES:
        raise ValueError(f'unknown mode {mode!r}; expected one of {MODES}')
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    if mode == 'app':
        return interleaved_cv_curve(X[train], y[train], max_components=max_components)
    return component_cv_curve(X[train], y[train], range(1, max_components + 1),
                              groups=np.asarray(sites)[train],
                              n_splits=SITE_HELDOUT_FOLDS,
                              random_state=42).rename(columns={'rmse_mean': 'rmsecv'})


def protocol_select_k(mode, curve):
    """The protocol's component-count rule applied to its own curve."""
    if mode not in MODES:
        raise ValueError(f'unknown mode {mode!r}; expected one of {MODES}')
    if mode == 'app':
        return select_within_tolerance(curve)
    k, _ = select_first_major_minimum(curve.rename(columns={'rmsecv': 'rmse_mean'}))
    return int(k)


def fit_calibration(mode, cohort, X, y, sites, X_addis, addis_volume_m3,
                    k_override=None):
    """Fit `cohort` end-to-end under `mode` and predict Addis EC in µg/m³.

    `X`/`y`/`sites` are the cohort's spectra, TOR EC loadings (µg/filter) and IMPROVE site
    labels; `X_addis` are the Addis spectra in the same representation (raw with raw,
    AIRSpec-corrected with AIRSpec-corrected). `k_override` forces the component count
    instead of the protocol's rule (for k sweeps — the meeting's scan-k item and the
    calibration_explorer app); None keeps the locked rule-choice behaviour.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    sites = np.asarray(sites)

    train = protocol_train_mask(mode, X, y, sites)
    curve = protocol_cv_curve(mode, X, y, sites, train)
    k = protocol_select_k(mode, curve)
    if k_override is not None:
        k = max(1, min(int(k_override), int(curve['n_components'].max())))
    heldout = None

    model = PLSRegression(n_components=k, scale=False).fit(X[train], y[train])
    if mode == 'site_heldout':
        heldout = regression_metrics(y[~train], model.predict(X[~train]).ravel())

    floor = float(curve.loc[curve['n_components'] == k, 'rmsecv'].iloc[0])
    return CalibrationFit(
        mode=mode, cohort=cohort, k=k,
        n_train=int(train.sum()), n_train_sites=int(pd.Series(sites[train]).nunique()),
        rmsecv_floor=floor, pct_rmsecv_floor=100 * floor / float(y[train].mean()),
        addis_ugm3=model.predict(np.asarray(X_addis, float)).ravel()
        / np.asarray(addis_volume_m3, float),
        curve=curve, heldout=heldout, model=model)


def addis_metrics(fit, fabs, mask, macs=(10.0, 6.0), cohort_label='fixed phase-2 cohort'):
    """Addis crossplot metrics for one fit: predicted EC vs HIPS EC-equivalent Fabs/MAC."""
    rows = []
    for mac in macs:
        rows.append({
            'cohort': fit.cohort, 'mode': fit.mode, 'k': fit.k,
            'evaluation_set': cohort_label, 'MAC_m2_g': mac,
            **regression_metrics(np.asarray(fabs, float)[mask] / mac,
                                 fit.addis_ugm3[mask]),
        })
    return rows
