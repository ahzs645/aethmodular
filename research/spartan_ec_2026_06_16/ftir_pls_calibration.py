"""Local PLS calibration for FTIR OC/EC — a reproducible stand-in for the UC Davis
FTIR Calibration Shiny tool (https://shiny.aqrc.ucdavis.edu/ftir_calibration/).

Why this exists
---------------
The Shiny tool builds an OC/EC calibration by *manual* clicking: pick the PLS
component count by eye off the RMSEP curve, then iteratively click-remove outliers
("remove |residual| > 200, rerun; remove measured-negatives; remove flagged-
questionable …" — Mona's walkthrough). That is slow, server-dependent (it 502s on
large pulls), and non-reproducible.

This module does the same thing as scripted, reproducible code so we can:
  * sweep the PLS component count and pick the min-RMSEP model automatically,
  * apply Mona's outlier rules deterministically (measured-negatives dropped
    outright; iterative absolute-residual trimming; optional flagged-questionable),
  * export coefficients in the tool's shape (wavenumber, coefficient) + intercept,
  * apply a calibration to *new* spectra (e.g. the Addis SPARTAN filters) to get
    per-filter EC/OC predictions to paste into 02_biomass_calibration_comparison.

Inputs
------
X : spectra. From the Shiny "Download Spectra for offline analysis" file, which
    carries metadata (AnalysisId, FilterId, SampleDate, Site, …) + one column per
    wavenumber. `load_spectra_csv` auto-detects which columns are the spectrum.
y : the *measured* reference OC or EC. NOT in the Shiny download — get it either
    by scraping the tool's "measured vs predicted" Calibrate plot (measured value
    keyed by FilterId) or from a reference table, then join on FilterId.

This file is standalone (numpy / pandas / scikit-learn only). Run it directly to
execute a synthetic self-test:  python ftir_pls_calibration.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict


# --------------------------------------------------------------------------- #
# Loading / column detection
# --------------------------------------------------------------------------- #
_WAVENUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def detect_spectral_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns whose names are numeric (the wavenumber axis).

    The Shiny download names each spectral channel by its wavenumber (e.g.
    "4000", "3996.2", ...). Metadata columns (AnalysisId, FilterId, Site, …) are
    non-numeric and are excluded.
    """
    cols = [c for c in df.columns if _WAVENUMBER_RE.match(str(c).strip())]
    if not cols:
        raise ValueError(
            "No wavenumber-like (numeric) columns found. Pass the spectra columns "
            "explicitly, or check the file orientation (spectra may be transposed)."
        )
    # sort by numeric value so the coefficient vector is ordered by wavenumber
    return sorted(cols, key=lambda c: float(str(c)))


def load_spectra_csv(path, id_col: str = "FilterId") -> tuple[pd.DataFrame, list[str]]:
    """Load a Shiny-exported spectra CSV. Returns (dataframe, wavenumber_columns).

    `id_col` is the per-filter key used later to join the measured OC/EC.
    """
    df = pd.read_csv(path)
    wn = detect_spectral_columns(df)
    if id_col not in df.columns:
        raise KeyError(f"id_col {id_col!r} not in file; columns: {list(df.columns)[:12]}…")
    return df, wn


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationResult:
    species: str
    n_components: int
    coefficients: np.ndarray          # length == len(wavenumbers)
    intercept: float
    wavenumbers: np.ndarray
    kept_index: pd.Index              # rows surviving outlier removal
    dropped: dict[str, list]          # reason -> list of dropped ids/index
    stats: dict                       # n, r2, rmse, bias_pct, rmsep_by_ncomp
    history: list[dict] = field(default_factory=list)

    # -- application -------------------------------------------------------- #
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Apply the calibration to new spectra (must be ordered like `wavenumbers`)."""
        Xv = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return Xv @ self.coefficients + self.intercept

    def coefficient_table(self) -> pd.DataFrame:
        """Coefficients in the tool's export shape: wavenumber + coefficient.

        The intercept is appended as a sentinel row (wavenumber = NaN) so the file
        is self-contained for re-application.
        """
        tab = pd.DataFrame({"wavenumber": self.wavenumbers, "coefficient": self.coefficients})
        tab = pd.concat(
            [tab, pd.DataFrame({"wavenumber": [np.nan], "coefficient": [self.intercept]})],
            ignore_index=True,
        )
        return tab

    def to_csv(self, path) -> None:
        self.coefficient_table().to_csv(path, index=False)


def _rmsep_by_ncomp(
    X: np.ndarray, y: np.ndarray, ncomp_range: Iterable[int], cv: int, seed: int
) -> dict[int, float]:
    """Cross-validated RMSEP for each candidate component count."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    out: dict[int, float] = {}
    for k in ncomp_range:
        if k >= min(X.shape):           # PLS needs n_components < n_features, n_samples
            continue
        pred = cross_val_predict(PLSRegression(n_components=k, scale=False), X, y, cv=kf)
        out[k] = float(np.sqrt(np.mean((y - pred.ravel()) ** 2)))
    return out


def build_calibration(
    df: pd.DataFrame,
    wavenumbers: Sequence[str],
    y_col: str,
    species: str = "EC",
    *,
    ncomp_range: Iterable[int] = range(5, 31),
    residual_thresholds: Sequence[float] = (400, 300, 200),
    residual_sigma: float | None = None,
    n_sigma_rounds: int = 3,
    drop_measured_negatives: bool = True,
    status_col: str | None = None,
    drop_status_values: Sequence[str] = ("questionable", "bad", "invalid"),
    cv: int = 5,
    seed: int = 0,
) -> CalibrationResult:
    """Build a PLS OC/EC calibration, reproducing the Shiny tool's manual workflow.

    Steps (mirrors Mona's walkthrough):
      1. Drop measured-negative samples outright ("mass should never be measured
         negative") if `drop_measured_negatives`.
      2. Optionally drop rows whose `status_col` is in `drop_status_values`
         (the tool's flagged-questionable points).
      3. Iteratively: fit min-RMSEP PLS, drop samples with |measured-predicted| >
         threshold, re-fit — once per entry in `residual_thresholds` (descending,
         like the tool's "remove >400, then >300, then >200").
      4. Refit final model; choose component count by min cross-validated RMSEP.

    `residual_thresholds` are on the same scale as `y_col` (µg loading if you pass
    loading, µg/m³ if you pass concentration — match what the tool used: loading).
    """
    work = df[[*wavenumbers, y_col] + ([status_col] if status_col else [])].copy()
    work = work.dropna(subset=[y_col, *wavenumbers])
    dropped: dict[str, list] = {"measured_negative": [], "status": [], "residual": []}
    history: list[dict] = []

    if drop_measured_negatives:
        neg = work.index[work[y_col] < 0]
        dropped["measured_negative"] = list(neg)
        work = work.drop(index=neg)

    if status_col:
        bad = work.index[work[status_col].astype(str).str.lower().isin(
            [s.lower() for s in drop_status_values])]
        dropped["status"] = list(bad)
        work = work.drop(index=bad)

    # Pick the component count ONCE (CV sweep) on the post-screen data. The
    # expensive cross-validated sweep runs only twice total (here + final), NOT
    # inside every trimming iteration — trimming uses cheap single in-sample fits
    # at the fixed component count, which is enough to flag gross outliers.
    X0 = work[list(wavenumbers)].to_numpy(float)
    y0 = work[y_col].to_numpy(float)
    rmsep0 = _rmsep_by_ncomp(X0, y0, ncomp_range, cv, seed)
    if not rmsep0:
        raise ValueError("No valid component count — too few samples/features.")
    k = min(rmsep0, key=rmsep0.get)

    # iterative residual trimming at the fixed component count (single fits).
    # Two modes:
    #   * residual_sigma set  -> each round drops |resid| > residual_sigma * std(resid),
    #     auto-scaling to the data (the right choice when loadings are small, so a
    #     fixed µg threshold like 200 would never fire). Runs up to n_sigma_rounds,
    #     stopping early when a round removes nothing.
    #   * else                -> fixed descending thresholds (the tool's "remove >400,
    #     then >300, then >200" — only sensible when residuals are that large).
    if residual_sigma is not None:
        rounds = [("sigma", residual_sigma)] * n_sigma_rounds
    else:
        rounds = [("abs", thr) for thr in residual_thresholds]
    for mode, param in rounds:
        if len(work) <= k + 2:
            break
        Xw = work[list(wavenumbers)].to_numpy(float)
        yw = work[y_col].to_numpy(float)
        m = PLSRegression(n_components=k, scale=False).fit(Xw, yw)
        resid = yw - m.predict(Xw).ravel()
        thr = param * float(np.std(resid)) if mode == "sigma" else param
        drop = work.index[np.abs(resid) > thr]
        history.append({"mode": mode, "param": param, "threshold": round(float(thr), 2),
                        "n_before": len(work), "n_dropped": int(len(drop)), "n_components": k})
        if len(drop) == 0 and mode == "sigma":
            break
        dropped["residual"].extend(list(drop))
        work = work.drop(index=drop)

    # final model: re-pick components on the cleaned data, then fit
    X = work[list(wavenumbers)].to_numpy(float)
    yv = work[y_col].to_numpy(float)
    rmsep = _rmsep_by_ncomp(X, yv, ncomp_range, cv, seed)
    k = min(rmsep, key=rmsep.get)
    model = PLSRegression(n_components=k, scale=False).fit(X, yv)

    # Flatten PLS into a single coefficient vector + intercept:
    #   y = X · coef + intercept   (so it can be applied like the tool's export).
    # Derive (coef, intercept) NUMERICALLY from the fitted model rather than reading
    # model.coef_ — sklearn's PLS coef_ convention varies across versions and does
    # not always satisfy predict(X) == X @ coef_ + intercept. predict() is affine,
    # so intercept = predict(0) and coef_j = predict(e_j) − intercept reproduce it
    # exactly for any sklearn version.
    p = X.shape[1]
    intercept = float(np.asarray(model.predict(np.zeros((1, p)))).reshape(-1)[0])
    coef = np.asarray(model.predict(np.eye(p))).reshape(-1) - intercept

    pred = X @ coef + intercept
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((yv - pred) ** 2)))
    bias_pct = float(100 * np.mean(pred - yv) / np.mean(yv)) if np.mean(yv) != 0 else np.nan

    return CalibrationResult(
        species=species,
        n_components=k,
        coefficients=coef.astype(float),
        intercept=intercept,
        wavenumbers=np.array([float(str(w)) for w in wavenumbers]),
        kept_index=work.index,
        dropped=dropped,
        stats={"n": int(len(work)), "r2": r2, "rmse": rmse, "bias_pct": bias_pct,
               "n_components": k, "rmsep_by_ncomp": rmsep},
        history=history,
    )


def apply_calibration_csv(coeff_csv, spectra_df: pd.DataFrame) -> np.ndarray:
    """Apply an exported coefficient CSV (wavenumber,coefficient + intercept row) to spectra.

    Columns are matched by *numeric value*, not by string, so "400" and "400.0"
    align regardless of how either file formatted the wavenumber.
    """
    tab = pd.read_csv(coeff_csv)
    intercept = float(tab.loc[tab["wavenumber"].isna(), "coefficient"].iloc[0])
    coeffs = tab.loc[tab["wavenumber"].notna()].copy()

    # map numeric wavenumber -> actual spectra column label
    col_by_value: dict[float, str] = {}
    for c in spectra_df.columns:
        if _WAVENUMBER_RE.match(str(c).strip()):
            col_by_value[round(float(str(c)), 4)] = c

    missing = [w for w in coeffs["wavenumber"] if round(float(w), 4) not in col_by_value]
    if missing:
        raise KeyError(f"{len(missing)} calibration wavenumbers absent from spectra "
                       f"(e.g. {missing[:5]}). Check the spectra are on the same wavenumber grid.")

    use_cols = [col_by_value[round(float(w), 4)] for w in coeffs["wavenumber"]]
    X = spectra_df[use_cols].to_numpy(float)
    return X @ coeffs["coefficient"].to_numpy(float) + intercept


# --------------------------------------------------------------------------- #
# Self-test (server-independent): synthetic spectra with a known linear EC map
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Well-conditioned linear synthetic data: y is a known linear map of the
    spectra plus small noise. This validates the *machinery* (component sweep,
    outlier rules, coefficient export/re-apply) — not FTIR physics. PLS should
    recover y with high R^2.
    """
    rng = np.random.default_rng(0)
    n, p = 240, 40
    wn = np.linspace(4000, 400, p)

    # full-rank random spectra; a smooth band-shaped true coefficient over wavenumber
    X = rng.normal(0.5, 0.15, size=(n, p))
    true_beta = (np.exp(-((wn - 1600) ** 2) / (2 * 250**2))
                 + 0.6 * np.exp(-((wn - 2900) ** 2) / (2 * 200**2))) * 12.0
    y = X @ true_beta + 3.0 + rng.normal(0, 0.5, n)   # measured EC loading µg + noise

    # inject the junk the tool teaches you to remove, on known samples
    neg_idx = list(range(5))
    out_idx = list(range(5, 9))
    y[neg_idx] = -np.abs(y[neg_idx]) - 1.0          # measured negatives
    y[out_idx] += 500                                # gross residual outliers

    cols = [f"{w:.1f}" for w in wn]
    df = pd.DataFrame(X, columns=cols)
    df["FilterId"] = [f"TEST-{i:04d}" for i in range(n)]
    df["EC_meas"] = y

    res = build_calibration(df, cols, "EC_meas", species="EC",
                            ncomp_range=range(2, 25),
                            residual_thresholds=(400, 300, 200))
    print("=== self-test: synthetic EC calibration (well-conditioned linear) ===")
    print(f"  n kept            : {res.stats['n']} / {n}")
    print(f"  measured-neg dropped: {len(res.dropped['measured_negative'])} "
          f"(injected {len(neg_idx)})")
    print(f"  residual-dropped  : {len(res.dropped['residual'])} (injected {len(out_idx)} gross)")
    print(f"  n_components      : {res.n_components}")
    print(f"  R^2               : {res.stats['r2']:.4f}")
    print(f"  RMSE              : {res.stats['rmse']:.3f} µg")
    print(f"  bias %            : {res.stats['bias_pct']:+.2f}")

    # round-trip: export coefficients, re-apply to clean spectra, compare to fit
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "coeffs.csv")
        res.to_csv(path)
        clean = df.loc[res.kept_index]
        reapplied = apply_calibration_csv(path, clean)
        direct = res.predict(clean[cols])
        print(f"  export/re-apply max abs diff: {np.max(np.abs(reapplied - direct)):.2e} (should be ~0)")

    assert res.stats["r2"] > 0.95, f"self-test R^2 too low: {res.stats['r2']}"
    assert set(neg_idx).issubset(set(res.dropped["measured_negative"])), "missed injected negatives"
    assert set(out_idx).issubset(set(res.dropped["residual"])), "missed injected outliers"
    print("  OK ✓")


if __name__ == "__main__":
    _self_test()
