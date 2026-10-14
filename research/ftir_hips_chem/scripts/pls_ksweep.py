"""PLS component-count sweeps and the IMPROVE smoke tracer they are checked against.

NIPALS PLS1 extracts components one at a time, so the first k components of a
k_max fit are exactly the k-component model. One fit at k_max therefore gives
predictions at every k <= k_max (``prefix_predict``; ``check_prefix`` verifies
this against direct refits). That makes a 100-split sweep over k as cheap as
100 fits.

The smoke tracer is IMPROVE non-soil potassium, nsK = K - 0.6 Fe (XRF, µg/m³),
read from the validated FED exports that ``workflows/build_improve_aerosol_archive.py``
downloads into ``output/tables/improve_validated_archive/raw``.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
IMPROVE_RAW = AREA / "output/tables/improve_validated_archive/raw"
# Soil K/Fe ratio in the IMPROVE non-soil potassium convention.
SOIL_K_PER_FE = 0.6


# --------------------------------------------------------------------------- PLS prefixes
def prefix_coefs(model, ks) -> np.ndarray:
    """Regression vectors (n_features, len(ks)) of the first-k-component submodels."""
    W, P, Q = model.x_weights_, model.x_loadings_, model.y_loadings_
    if np.any(model._x_std != 1) or np.any(model._y_std != 1):
        raise ValueError("prefix_coefs assumes scale=False")
    out = []
    for k in ks:
        if not 1 <= k <= W.shape[1]:
            raise ValueError(f"k={k} outside 1..{W.shape[1]}")
        R = W[:, :k] @ np.linalg.pinv(P[:, :k].T @ W[:, :k])
        out.append(R @ Q[:, :k].T)
    return np.hstack(out)


def prefix_predict(model, X, ks, coefs=None) -> np.ndarray:
    """Predictions (n, len(ks)) of every first-k submodel; pass ``coefs`` to reuse them."""
    B = prefix_coefs(model, ks) if coefs is None else coefs
    return (np.asarray(X, float) - model._x_mean) @ B + np.ravel(model._y_mean)[0]


def check_prefix(X, y, ks, k_max, atol=1e-6):
    """Largest |prefix - refit| prediction difference, relative to the spread of y."""
    from sklearn.cross_decomposition import PLSRegression

    X, y = np.asarray(X, float), np.asarray(y, float)
    full = PLSRegression(n_components=k_max, scale=False).fit(X, y)
    pre = prefix_predict(full, X, ks)
    worst = 0.0
    for j, k in enumerate(ks):
        ref = PLSRegression(n_components=k, scale=False).fit(X, y).predict(X).ravel()
        worst = max(worst, float(np.max(np.abs(pre[:, j] - ref)) / np.std(y)))
    if worst > atol:
        raise AssertionError(f"prefix predictions differ from refits by {worst:.2e} (relative)")
    return worst


# --------------------------------------------------------------------------- fast readouts
def ols(x, y):
    """OLS slope and intercept of y on x for each column of y (x is 1-D)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float).reshape(len(x), -1)
    xc = x - x.mean()
    slope = xc @ (y - y.mean(0)) / (xc @ xc)
    return slope, y.mean(0) - slope * x.mean()


def deming(x, y, lam):
    """Deming slope and intercept per column of y; lam = Var(y error) / Var(x error).

    Same convention as ``plotting.utils.calculate_regression_stats``.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float).reshape(len(x), -1)
    xc, yc = x - x.mean(), y - y.mean(0)
    n = len(x) - 1
    sxx, syy, sxy = xc @ xc / n, (yc * yc).sum(0) / n, xc @ yc / n
    d = syy - lam * sxx
    slope = (d + np.sqrt(d ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    return slope, y.mean(0) - slope * x.mean()


def r2(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float).reshape(len(x), -1)
    xc, yc = x - x.mean(), y - y.mean(0)
    return (xc @ yc) ** 2 / ((xc @ xc) * (yc * yc).sum(0))


# --------------------------------------------------------------------------- IMPROVE nsK
def _read_year(path, params):
    keep = []
    with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as stream:
        for chunk in pd.read_csv(stream, sep="|", usecols=["SiteCode", "POC", "FactDate", "ParamCode",
                                                           "FactValue", "Status"],
                                 chunksize=2_000_000, low_memory=False):
            keep.append(chunk.loc[chunk.ParamCode.isin(params)])
    return pd.concat(keep, ignore_index=True)


def load_improve_nsk(years, cache=None) -> pd.DataFrame:
    """One row per (Site, date): Kf, FEf and nsK = Kf - 0.6 FEf in µg/m³.

    Only validated rows (Status starting with 'V') with a value are used; where a
    site has several POCs the lowest POC wins.
    """
    if cache is not None and Path(cache).exists():
        return pd.read_csv(cache, parse_dates=["date"])
    parts = []
    for year in years:
        path = IMPROVE_RAW / f"IMPAER_{year}.txt.zip"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run workflows/build_improve_aerosol_archive.py")
        parts.append(_read_year(path, {"Kf", "FEf"}))
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.loc[raw.Status.astype(str).str.startswith("V") & raw.FactValue.gt(-900)]
    raw = raw.sort_values("POC").drop_duplicates(["SiteCode", "FactDate", "ParamCode"])
    wide = raw.pivot(index=["SiteCode", "FactDate"], columns="ParamCode", values="FactValue").reset_index()
    wide = wide.rename(columns={"SiteCode": "Site", "FactDate": "date", "Kf": "K_ugm3", "FEf": "Fe_ugm3"})
    wide["date"] = pd.to_datetime(wide.date)
    wide["nsK_ugm3"] = wide.K_ugm3 - SOIL_K_PER_FE * wide.Fe_ugm3
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(cache, index=False)
    return wide
