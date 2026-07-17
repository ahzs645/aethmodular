"""PLS interpretation and calibration-transfer helpers for the FTIR/HIPS work.

The functions in this module implement the diagnostics discussed in Section 3.4
of Takahama et al. (2019): variable importance in projection (VIP), score-space
Mahalanobis distance/leverage, and spectral residual magnitude (Q).

The module intentionally keeps data loading separate from the linear algebra so
the diagnostics can be tested with synthetic data and reused with future RDS
exports.  It does not modify any source file in the Google Drive data tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold


@dataclass(frozen=True)
class FTIRTransferPaths:
    """Resolved external source paths used by the transfer notebooks."""

    ftir_dir: Path
    etad_dir: Path
    spartan_hips_primary: Path
    spartan_hips_backup: Path

    @classmethod
    def defaults(cls) -> "FTIRTransferPaths":
        drive = (
            Path.home()
            / "Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive"
        )
        data = drive / "University/Research/Grad/UC Davis Ann/NASA MAIA/Data"
        return cls(
            ftir_dir=drive / "FTIR",
            etad_dir=data / "DAVIS/ETAD FTIR",
            spartan_hips_primary=data / "Spartan/SPARTAN_HIPS_Batch1-51.v2.csv",
            spartan_hips_backup=drive / "Downloads Backup/HIPS_all_SPARTAN_batches.csv",
        )

    def validate(self) -> pd.DataFrame:
        rows = []
        for name, path in self.__dict__.items():
            rows.append(
                {
                    "source": name,
                    "path": str(path),
                    "exists": path.exists(),
                    "size_mb": path.stat().st_size / 1e6 if path.exists() else np.nan,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class CurrentPLSModel:
    """A Python reconstruction of one exported R ``pls::mvr`` calibration."""

    species: str
    model: PLSRegression
    X: np.ndarray
    y: np.ndarray
    wavenumbers: np.ndarray
    analysis_ids: np.ndarray
    filter_ids: np.ndarray
    sites: np.ndarray
    chosen_n_components: int
    r_prediction_max_abs_error: float
    r_prediction_median_abs_error: float


def _model_prefix(species: str) -> str:
    species = species.upper()
    if species == "EC":
        return "EC"
    if species == "OC":
        return "OC_20260707"
    raise ValueError("species must be 'EC' or 'OC'")


def load_current_pls_model(
    ftir_dir: str | Path,
    species: str,
    *,
    tol: float = 2e-2,
) -> CurrentPLSModel:
    """Reconstruct a current EC or OC PLS model from its exported components.

    The RDS files themselves require R's ``pls`` package.  The FTIR directory
    already contains lossless CSV/NPZ exports of the model matrix, chosen
    component count, fitted values, and coefficients.  Fitting scikit-learn's
    univariate NIPALS PLS to the same rows reproduces the R predictions to much
    better than ``tol``; the measured discrepancy is returned for auditing.
    """

    ftir_dir = Path(ftir_dir)
    species = species.upper()
    prefix = _model_prefix(species)
    extracted = ftir_dir / "notebook/extracted"

    arrays = np.load(ftir_dir / "apps/apps_data.npz", allow_pickle=True)
    samples = pd.read_csv(extracted / f"{prefix}_samples.csv")
    stats = pd.read_csv(extracted / f"{prefix}_stats_chosen.csv").iloc[0]
    fitted = pd.read_csv(extracted / f"{prefix}_fitted_long.csv")

    available_ids = arrays[f"{species}_id"].astype(int)
    row_for_id = {analysis_id: index for index, analysis_id in enumerate(available_ids)}
    indices = np.array([row_for_id.get(int(value), -1) for value in samples["AnalysisId"]])
    if np.any(indices < 0):
        missing = samples.loc[indices < 0, "AnalysisId"].head().tolist()
        raise KeyError(f"{len(missing)} model rows missing from apps_data.npz: {missing}")

    X = arrays[f"{species}_X"][indices].astype(float)
    y = samples["measured"].to_numpy(float)
    n_components = int(stats["chosen_ncomp"])
    model = PLSRegression(
        n_components=n_components,
        scale=False,
        max_iter=1000,
        tol=1e-10,
    ).fit(X, y)

    r_fitted = (
        fitted[fitted["ncomp"] == n_components]
        .set_index("AnalysisId")
        .loc[samples["AnalysisId"], "fitted"]
        .to_numpy(float)
    )
    error = np.abs(model.predict(X).ravel() - r_fitted)
    max_error = float(error.max())
    if max_error > tol:
        raise AssertionError(
            f"Python reconstruction differs from R for {species}: max |error|={max_error:g}"
        )

    return CurrentPLSModel(
        species=species,
        model=model,
        X=X,
        y=y,
        wavenumbers=arrays["wn"].astype(float),
        analysis_ids=samples["AnalysisId"].to_numpy(int),
        filter_ids=arrays[f"{species}_fid"][indices].astype(int),
        sites=samples["Site"].astype(str).to_numpy(),
        chosen_n_components=n_components,
        r_prediction_max_abs_error=max_error,
        r_prediction_median_abs_error=float(np.median(error)),
    )


def vip_scores(model: PLSRegression) -> np.ndarray:
    """Return Wold VIP scores using response SS explained by each component.

    This is Eq. (14) in Takahama et al. (2019).  The implementation supports a
    univariate response, which is the form used by all EC, OC, and HIPS models
    in this project.
    """

    scores = np.asarray(model.x_scores_, dtype=float)
    weights = np.asarray(model.x_weights_, dtype=float)
    y_loadings = np.asarray(model.y_loadings_, dtype=float).reshape(-1)
    if y_loadings.size != scores.shape[1]:
        raise ValueError("vip_scores currently supports a univariate PLS response")

    response_ss = (y_loadings**2) * np.sum(scores**2, axis=0)
    total_ss = float(response_ss.sum())
    if total_ss <= 0:
        raise ValueError("PLS model has no explained response sum of squares")
    weight_norm_sq = np.sum(weights**2, axis=0)
    normalized_weight_sq = (weights**2) / weight_norm_sq
    n_predictors = weights.shape[0]
    return np.sqrt(n_predictors * (normalized_weight_sq @ response_ss) / total_ss)


def select_components_cv(
    X: np.ndarray,
    y: np.ndarray,
    candidates: Iterable[int],
    *,
    groups: Sequence | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[int, pd.DataFrame]:
    """Choose PLS components by held-out RMSE.

    When ``groups`` is provided, entire groups (normally IMPROVE sites) are
    held out together.  This is more conservative for geographic transfer than
    random row folds.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    candidates = sorted({int(value) for value in candidates if int(value) > 0})
    candidates = [value for value in candidates if value < min(X.shape)]
    if not candidates:
        raise ValueError("no valid PLS component candidates")

    if groups is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(X, y))
        scheme = "random K-fold"
    else:
        groups = np.asarray(groups)
        if np.unique(groups).size < n_splits:
            raise ValueError("fewer unique groups than requested folds")
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(X, y, groups))
        scheme = "group-held-out K-fold"

    rows = []
    for n_components in candidates:
        fold_rmse = []
        for train, test in splits:
            model = PLSRegression(n_components=n_components, scale=False).fit(X[train], y[train])
            prediction = model.predict(X[test]).ravel()
            fold_rmse.append(float(np.sqrt(mean_squared_error(y[test], prediction))))
        rows.append(
            {
                "n_components": n_components,
                "rmse_mean": float(np.mean(fold_rmse)),
                "rmse_sd": float(np.std(fold_rmse, ddof=1)),
                "cv_scheme": scheme,
            }
        )
    curve = pd.DataFrame(rows)
    best = int(curve.loc[curve["rmse_mean"].idxmin(), "n_components"])
    return best, curve


def predict_pls_components(
    model: PLSRegression,
    X: np.ndarray,
    candidates: Iterable[int],
) -> np.ndarray:
    """Predict with successive prefixes of one fitted PLS model.

    R's ``pls`` calibration view and the local AQRC reconstruction use one
    maximum-component fit, then truncate rotations/loadings to inspect each
    smaller component count. This is much faster than refitting per candidate.
    """

    X = np.asarray(X, dtype=float)
    candidates = sorted({int(value) for value in candidates if int(value) > 0})
    fitted_components = int(model.x_rotations_.shape[1])
    if not candidates or candidates[-1] > fitted_components:
        raise ValueError("candidate count exceeds the fitted PLS components")

    x_mean = np.asarray(model._x_mean, dtype=float)
    y_mean = float(np.asarray(model._y_mean, dtype=float).reshape(-1)[0])
    centered = X - x_mean
    prediction = np.empty((len(X), len(candidates)), dtype=float)
    for column, n_components in enumerate(candidates):
        coefficient = (
            model.x_rotations_[:, :n_components]
            @ model.y_loadings_[:, :n_components].T
        )
        prediction[:, column] = (centered @ coefficient).reshape(-1) + y_mean
    return prediction


def component_cv_curve(
    X: np.ndarray,
    y: np.ndarray,
    candidates: Iterable[int],
    *,
    groups: Sequence | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Efficient component CV curve using one maximum-component fit per fold."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    candidates = sorted({int(value) for value in candidates if int(value) > 0})
    candidates = [value for value in candidates if value < min(X.shape)]
    if not candidates:
        raise ValueError("no valid PLS component candidates")

    if groups is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(X, y))
        scheme = "random K-fold"
    else:
        groups = np.asarray(groups)
        if np.unique(groups).size < n_splits:
            raise ValueError("fewer unique groups than requested folds")
        splitter = GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(X, y, groups))
        scheme = "group-held-out K-fold"

    fold_rmse = []
    max_components = max(candidates)
    for train, test in splits:
        model = PLSRegression(
            n_components=max_components,
            scale=False,
            max_iter=1000,
            tol=1e-10,
        ).fit(X[train], y[train])
        prediction = predict_pls_components(model, X[test], candidates)
        fold_rmse.append(
            np.sqrt(np.mean((prediction - y[test, None]) ** 2, axis=0))
        )

    fold_rmse = np.asarray(fold_rmse, dtype=float)
    return pd.DataFrame(
        {
            "n_components": candidates,
            "rmse_mean": fold_rmse.mean(axis=0),
            "rmse_sd": fold_rmse.std(axis=0, ddof=1),
            "rmse_se": fold_rmse.std(axis=0, ddof=1) / np.sqrt(len(fold_rmse)),
            "n_folds": len(fold_rmse),
            "cv_scheme": scheme,
        }
    )


def select_first_major_minimum(
    curve: pd.DataFrame,
    *,
    relative_tolerance: float = 0.0,
) -> tuple[int, pd.DataFrame]:
    """Choose the earliest local CV minimum within one SE of the global minimum."""

    required = {"n_components", "rmse_mean"}
    if not required.issubset(curve.columns):
        raise ValueError(f"curve must contain {sorted(required)}")
    result = curve.sort_values("n_components").reset_index(drop=True).copy()
    if result.empty:
        raise ValueError("curve is empty")

    global_index = int(result["rmse_mean"].idxmin())
    global_mean = float(result.loc[global_index, "rmse_mean"])
    if "rmse_se" in result and np.isfinite(result.loc[global_index, "rmse_se"]):
        one_se_threshold = global_mean + float(result.loc[global_index, "rmse_se"])
    else:
        one_se_threshold = global_mean
    threshold = max(one_se_threshold, global_mean * (1 + relative_tolerance))

    values = result["rmse_mean"].to_numpy(float)
    local = np.zeros(len(result), dtype=bool)
    if len(result) >= 3:
        local[1:-1] = (values[1:-1] <= values[:-2]) & (values[1:-1] < values[2:])
    within = values <= threshold
    eligible = np.flatnonzero(local & within)
    if eligible.size == 0:
        eligible = np.flatnonzero(within)
    selected_index = int(eligible[0])

    result["is_local_minimum"] = local
    result["within_global_one_se"] = within
    result["selected_first_major_minimum"] = False
    result.loc[selected_index, "selected_first_major_minimum"] = True
    result["global_minimum_components"] = int(
        result.loc[global_index, "n_components"]
    )
    result["selection_threshold"] = threshold
    return int(result.loc[selected_index, "n_components"]), result


def local_continuum_peak_height(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    peak_band: tuple[float, float],
    left_continuum: tuple[float, float],
    right_continuum: tuple[float, float],
) -> np.ndarray:
    """Maximum peak height above a line joining two local continuum windows."""

    X = np.asarray(X, dtype=float)
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != wavenumbers.size:
        raise ValueError("X columns must match wavenumbers")

    def band_mask(bounds):
        low, high = sorted(bounds)
        mask = (wavenumbers >= low) & (wavenumbers <= high)
        if not mask.any():
            raise ValueError(f"no wavenumbers in band {bounds}")
        return mask

    peak_mask = band_mask(peak_band)
    left_mask = band_mask(left_continuum)
    right_mask = band_mask(right_continuum)
    left_x = float(wavenumbers[left_mask].mean())
    right_x = float(wavenumbers[right_mask].mean())
    left_y = X[:, left_mask].mean(axis=1)
    right_y = X[:, right_mask].mean(axis=1)
    baseline = left_y[:, None] + (right_y - left_y)[:, None] * (
        (wavenumbers[peak_mask][None, :] - left_x) / (right_x - left_x)
    )
    return np.max(X[:, peak_mask] - baseline, axis=1)


def ftir_source_band_features(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    ratio_floor: float = 1e-4,
) -> pd.DataFrame:
    """Return diagnostic CH, carbonyl, and ~1600 cm⁻¹ continuum features.

    These are interpretable cohort-selection diagnostics, not AIRSpec baseline
    replacements or chemical source assignments.
    """

    X = np.asarray(X, dtype=float)
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    ch = local_continuum_peak_height(
        X, wavenumbers, (2800, 3000), (3050, 3150), (2650, 2750)
    )
    carbonyl = local_continuum_peak_height(
        X, wavenumbers, (1650, 1775), (1800, 1900), (1500, 1550)
    )
    shoulder = local_continuum_peak_height(
        X, wavenumbers, (1550, 1650), (1750, 1800), (1500, 1530)
    )
    denominator = np.where(ch > ratio_floor, ch, np.nan)
    return pd.DataFrame(
        {
            "CH_peak": ch,
            "carbonyl_peak": carbonyl,
            "shoulder_1600_peak": shoulder,
            "carbonyl_to_CH": carbonyl / denominator,
            "shoulder_1600_to_CH": shoulder / denominator,
        }
    )


def nested_cv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    candidates: Iterable[int],
    *,
    outer_splits: int = 5,
    inner_splits: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return honest outer-fold predictions with component choice inside each fold."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    prediction = np.full(y.shape, np.nan, dtype=float)
    rows = []
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    for fold, (train, test) in enumerate(outer.split(X, y), start=1):
        chosen, curve = select_components_cv(
            X[train],
            y[train],
            candidates,
            n_splits=inner_splits,
            random_state=random_state + fold,
        )
        model = PLSRegression(n_components=chosen, scale=False).fit(X[train], y[train])
        prediction[test] = model.predict(X[test]).ravel()
        rows.append(
            {
                "outer_fold": fold,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "chosen_n_components": chosen,
                "inner_rmse": float(curve["rmse_mean"].min()),
            }
        )
    return prediction, pd.DataFrame(rows)


def regression_metrics(observed, predicted) -> dict[str, float]:
    """Regression and agreement metrics with observed values on the x-axis."""

    observed = np.asarray(observed, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    keep = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[keep]
    predicted = predicted[keep]
    if observed.size < 3:
        raise ValueError("at least three paired finite observations are required")
    slope, intercept = np.polyfit(observed, predicted, 1)
    fitted = slope * observed + intercept
    ss_res = float(np.sum((predicted - fitted) ** 2))
    ss_total = float(np.sum((predicted - predicted.mean()) ** 2))
    return {
        "n": int(observed.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(1 - ss_res / ss_total) if ss_total > 0 else np.nan,
        "RMSE": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "MAE": float(np.mean(np.abs(predicted - observed))),
        "bias": float(np.mean(predicted - observed)),
        "median_bias": float(np.median(predicted - observed)),
    }


def score_metric(model: PLSRegression) -> tuple[np.ndarray, np.ndarray]:
    """Return calibration scores and inverse score SS for Eq. (15)."""

    scores = np.asarray(model.x_scores_, dtype=float)
    inverse_ss = np.linalg.pinv(scores.T @ scores)
    return scores, inverse_ss


def project_scores(model: PLSRegression, X: np.ndarray) -> np.ndarray:
    """Project new spectra into the fitted PLS score space."""

    return np.asarray(model.transform(np.asarray(X, dtype=float)), dtype=float)


def mahalanobis_distance_squared(scores: np.ndarray, inverse_ss: np.ndarray) -> np.ndarray:
    """Squared score-space Mahalanobis distance/leverage (Eq. 15)."""

    scores = np.asarray(scores, dtype=float)
    inverse_ss = np.asarray(inverse_ss, dtype=float)
    return np.einsum("ij,jk,ik->i", scores, inverse_ss, scores)


def pairwise_score_distance_squared(
    scores: np.ndarray,
    reference: np.ndarray,
    inverse_ss: np.ndarray,
) -> np.ndarray:
    """Distance from each score vector to one reference vector in calibration metric."""

    delta = np.asarray(scores, dtype=float) - np.asarray(reference, dtype=float)
    return mahalanobis_distance_squared(delta, inverse_ss)


def spectral_q_residual(model: PLSRegression, X: np.ndarray) -> np.ndarray:
    """Squared residual norm after reconstruction from retained PLS scores."""

    X = np.asarray(X, dtype=float)
    mean = np.asarray(model._x_mean, dtype=float)  # sklearn stores fitted centering here
    scale = np.asarray(model._x_std, dtype=float)
    centered = (X - mean) / scale
    scores = centered @ np.asarray(model.x_rotations_, dtype=float)
    reconstruction = scores @ np.asarray(model.x_loadings_, dtype=float).T
    residual = centered - reconstruction
    return np.einsum("ij,ij->i", residual, residual)


def offset_correct(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    window: tuple[float, float] = (3900.0, 4000.0),
) -> np.ndarray:
    """Subtract each spectrum's mean in a nominal non-absorbing window."""

    X = np.asarray(X, dtype=float)
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    keep = (wavenumbers >= window[0]) & (wavenumbers <= window[1])
    if not np.any(keep):
        raise ValueError(f"no wavenumbers in baseline window {window}")
    return X - np.nanmean(X[:, keep], axis=1, keepdims=True)


def vip_overlap_summary(
    vip_a: np.ndarray,
    vip_b: np.ndarray,
    *,
    threshold: float = 1.0,
    top_n: int = 200,
) -> dict[str, float]:
    """Summarize similarity between two VIP profiles on the same grid."""

    vip_a = np.asarray(vip_a, dtype=float)
    vip_b = np.asarray(vip_b, dtype=float)
    if vip_a.shape != vip_b.shape:
        raise ValueError("VIP arrays must share a wavenumber grid")
    important_a = set(np.flatnonzero(vip_a >= threshold))
    important_b = set(np.flatnonzero(vip_b >= threshold))
    top_a = set(np.argsort(vip_a)[-top_n:])
    top_b = set(np.argsort(vip_b)[-top_n:])
    union = important_a | important_b
    return {
        "spearman_r": float(spearmanr(vip_a, vip_b).statistic),
        "important_a_n": int(len(important_a)),
        "important_b_n": int(len(important_b)),
        "important_jaccard": float(len(important_a & important_b) / len(union)) if union else np.nan,
        "top_n": int(top_n),
        "top_n_overlap": int(len(top_a & top_b)),
        "top_n_overlap_fraction": float(len(top_a & top_b) / top_n),
        "a_vip_mass_on_b_important": float(
            np.sum(vip_a[list(important_b)] ** 2) / np.sum(vip_a**2)
        )
        if important_b
        else np.nan,
    }


FUNCTIONAL_BANDS = (
    (3000.0, 3600.0, "O-H / N-H region"),
    (2800.0, 3000.0, "aliphatic C-H stretch"),
    (1650.0, 1850.0, "carbonyl region"),
    (1500.0, 1650.0, "amide / aromatic / N-H region"),
    (500.0, 1500.0, "fingerprint region"),
)


def summarize_vip_bands(wavenumbers: np.ndarray, vip: np.ndarray) -> pd.DataFrame:
    """Aggregate squared VIP mass across broad, deliberately non-specific bands."""

    wavenumbers = np.asarray(wavenumbers, dtype=float)
    vip = np.asarray(vip, dtype=float)
    total = float(np.sum(vip**2))
    rows = []
    for low, high, label in FUNCTIONAL_BANDS:
        keep = (wavenumbers >= low) & (wavenumbers < high)
        rows.append(
            {
                "band": label,
                "low_cm-1": low,
                "high_cm-1": high,
                "n_wavenumbers": int(keep.sum()),
                "vip2_mass_fraction": float(np.sum(vip[keep] ** 2) / total),
                "vip_ge_1_fraction": float(np.mean(vip[keep] >= 1.0)) if keep.any() else np.nan,
                "max_vip": float(np.max(vip[keep])) if keep.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def spaced_peak_table(
    wavenumbers: np.ndarray,
    score: np.ndarray,
    *,
    n_peaks: int = 20,
    min_separation_cm1: float = 20.0,
) -> pd.DataFrame:
    """Select high-scoring wavenumbers without returning the same broad peak repeatedly."""

    wavenumbers = np.asarray(wavenumbers, dtype=float)
    score = np.asarray(score, dtype=float)
    selected: list[int] = []
    for index in np.argsort(score)[::-1]:
        if not np.isfinite(score[index]):
            continue
        if all(abs(wavenumbers[index] - wavenumbers[other]) >= min_separation_cm1 for other in selected):
            selected.append(int(index))
        if len(selected) >= n_peaks:
            break
    return pd.DataFrame(
        {
            "index": selected,
            "wavenumber_cm-1": wavenumbers[selected],
            "score": score[selected],
        }
    )
