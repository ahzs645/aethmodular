"""Leakage-safe seasonal spectral diagnostics on physical filter rows.

Classification asks whether spectra distinguish canonical calendar seasons.
Regression predicts optical disagreement, which already contains an FTIR-derived
quantity; it is not independent EC validation. Fixed, restrained models are used.
All learned encoding, centering, scaling and latent components are fitted inside
each training fold. Signed spectra are retained; unsupported/NaN channels must be
removed by an explicitly declared window before this module is called.

Region importance permutes original input channels jointly, before the fitted
PCA/PLS/model. It is held-out predictive reliance, not chemical attribution or
proof that a region is uniquely informative. Impurity importance is not used.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from config import resolve_seasons, season_for_month
except ImportError:
    from .config import resolve_seasons, season_for_month

SEED = 20261103


def _dates(values):
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if dates.isna().any():
        raise ValueError("dates must be complete")
    if dates.tz is not None:
        raise ValueError("supply timezone-naive local sampling dates explicitly")
    return dates.normalize()


def _ids(values, n, *, unique=False):
    array = np.asarray(values, dtype=object)
    if array.shape != (n,) or pd.isna(array).any():
        raise ValueError("one nonmissing physical filter ID is required per row")
    array = array.astype(str)
    if np.any(np.char.str_len(array) == 0):
        raise ValueError("physical filter IDs cannot be empty")
    if unique and len(set(array)) != n:
        raise ValueError("model evaluation requires one row per physical filter; aggregate repeats first")
    return array


def season_labels(dates, convention="belg_feb"):
    """Canonical season labels; dates assign labels, never classifier predictors."""
    mapping = resolve_seasons(convention)
    return np.asarray([season_for_month(int(month), mapping) for month in _dates(dates).month], dtype=str)


def cycle_inventory(dates, physical_ids, convention="belg_feb"):
    """Coverage of Oct–Sep cycles, labelled by END year and human year range.

    An all-season block can still lack sampled months. Both facts are retained;
    neither a partial first cycle nor a truncated last cycle is called complete.
    Repeat rows may be inventoried, but each physical filter must have one date.
    """
    dates = _dates(dates)
    ids = _ids(physical_ids, len(dates))
    frame = pd.DataFrame({"physical_id": ids, "date": dates})
    if frame.groupby("physical_id").date.nunique().gt(1).any():
        raise ValueError("one physical filter has conflicting sampling dates")
    frame = frame.drop_duplicates("physical_id")
    frame["cycle"] = frame.date.dt.year + frame.date.dt.month.ge(10).astype(int)
    frame["season"] = season_labels(frame.date, convention)
    all_seasons = tuple(resolve_seasons(convention))
    rows = []
    for cycle, group in frame.groupby("cycle", sort=True):
        months = sorted(set(group.date.dt.month.astype(int)))
        counts = group.season.value_counts().to_dict()
        rows.append({
            "cycle": int(cycle), "cycle_label": f"{cycle-1}–{str(cycle)[-2:]}",
            "start": f"{cycle-1}-10-01", "end": f"{cycle}-09-30",
            "n_filters": len(group), "n_months": len(months),
            "months_present": months, "missing_months": sorted(set(range(1, 13))-set(months)),
            "all_months": len(months) == 12,
            "all_seasons": all(counts.get(label, 0) > 0 for label in all_seasons),
            "season_counts": {label: int(counts.get(label, 0)) for label in all_seasons},
            "first_date": group.date.min().date().isoformat(),
            "last_date": group.date.max().date().isoformat(), "convention": convention,
        })
    return pd.DataFrame(rows, columns=[
        "cycle", "cycle_label", "start", "end", "n_filters", "n_months", "months_present",
        "missing_months", "all_months", "all_seasons", "season_counts", "first_date", "last_date", "convention",
    ])


@dataclass(frozen=True)
class CycleFold:
    name: str
    train: np.ndarray
    test: np.ndarray
    test_cycle: int
    complete_test: bool
    train_cycles: tuple[int, ...]
    mode: str = "forward"
    train_coverage: str = "all_seasons"
    test_coverage: str = "all_months"


def make_cycle_folds(dates, physical_ids, *, convention="belg_feb", mode="forward",
                     train_coverage="all_seasons", test_coverage="all_months",
                     min_train=30, min_test=6, min_train_cycles=1):
    """Deterministic whole-cycle folds with explicit training/test coverage.

    ``all_months`` requires all 12 sampled months; ``all_seasons`` requires all
    three canonical classes, while recording missing months. Forward training
    uses earlier eligible cycles only. Leave-cycle-out uses the other eligible
    cycles and is a retrospective transfer check, not a forecast.
    """
    if mode not in {"forward", "leave_cycle_out"}:
        raise ValueError("mode must be forward or leave_cycle_out")
    if train_coverage not in {"all_months", "all_seasons"} or test_coverage not in {"all_months", "all_seasons"}:
        raise ValueError("coverage must explicitly be all_months or all_seasons")
    if min_train < 2 or min_test < 1 or min_train_cycles < 1:
        raise ValueError("invalid minimum fold sizes")
    dates = _dates(dates)
    ids = _ids(physical_ids, len(dates))
    inventory = cycle_inventory(dates, ids, convention)
    row_cycles = dates.year.to_numpy() + (dates.month.to_numpy() >= 10)
    train_ok = inventory.loc[inventory[train_coverage], "cycle"].astype(int).tolist()
    folds = []
    for row in inventory.loc[inventory[test_coverage]].itertuples():
        train_cycles = tuple(c for c in train_ok if (c < row.cycle if mode == "forward" else c != row.cycle))
        train = np.flatnonzero(np.isin(row_cycles, train_cycles))
        test = np.flatnonzero(row_cycles == row.cycle)
        if (len(train_cycles) < min_train_cycles or len(set(ids[train])) < min_train
                or len(set(ids[test])) < min_test):
            continue
        if set(ids[train]) & set(ids[test]):
            raise ValueError("physical filter overlaps training and test cycles")
        if mode == "forward" and dates[train].max() >= dates[test].min():
            raise ValueError("forward fold is not chronological")
        folds.append(CycleFold(
            name=f"{mode}_{row.cycle}", train=train, test=test, test_cycle=int(row.cycle),
            complete_test=bool(row.all_months), train_cycles=train_cycles,
            mode=mode, train_coverage=train_coverage, test_coverage=test_coverage,
        ))
    return folds, inventory


def make_feature_frame(spectra, metadata=None, *, physical_ids=None):
    """Return (DataFrame, spectral-column list), preserving signed finite input.

    Metadata are copied by row order, not joined implicitly. Unsupported spline
    padding must be removed with an explicit input mask by the caller. NaN
    spectra are rejected rather than median-imputed as if measured channels.
    """
    values = check_array(spectra, dtype=float, ensure_2d=True, ensure_min_features=0)
    columns = [f"sp_{j:04d}" for j in range(values.shape[1])]
    frame = pd.DataFrame(values, columns=columns)
    if metadata is not None:
        metadata = pd.DataFrame(metadata).reset_index(drop=True).copy()
        if len(metadata) != len(frame) or metadata.columns.duplicated().any():
            raise ValueError("metadata must have matching rows and unique columns")
        if set(columns) & set(metadata.columns):
            raise ValueError("metadata names collide with spectral columns")
        frame = pd.concat([frame, metadata], axis=1)
    if physical_ids is not None:
        frame.index = pd.Index(_ids(physical_ids, len(frame), unique=True), name="physical_id")
    return frame, columns


class FiniteSpectra(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        values = check_array(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        values = check_array(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("spectral width changed")
        return values


class CategoricalStrings(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        frame = pd.DataFrame(X)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError("metadata width changed")
        return frame.astype("string").fillna("__MISSING__").to_numpy(dtype=str)


class CappedPCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=8, random_state=SEED):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        values = check_array(X, dtype=float, ensure_min_samples=2)
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_features_in_ = values.shape[1]
        self.n_components_ = min(int(self.n_components), values.shape[1], len(values)-1)
        self.model_ = PCA(n_components=self.n_components_, svd_solver="full", random_state=self.random_state).fit(values)
        return self

    def transform(self, X):
        check_is_fitted(self, "model_")
        return self.model_.transform(check_array(X, dtype=float))


class CappedPLSRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_components=8, max_iter=1000, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        values = check_array(X, dtype=float, ensure_min_samples=2)
        target = np.asarray(y, float).reshape(-1)
        if target.shape != (len(values),) or not np.isfinite(target).all() or self.n_components < 1:
            raise ValueError("invalid PLS target or component count")
        self.n_features_in_ = values.shape[1]
        self.n_components_ = min(int(self.n_components), values.shape[1], len(values)-1)
        self.model_ = PLSRegression(n_components=self.n_components_, scale=False,
                                    max_iter=self.max_iter, tol=self.tol).fit(values, target)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(check_array(X, dtype=float))).reshape(-1)


def _calendar_column(name):
    name = str(name).lower()
    return any(token in name for token in ("season", "month", "date", "time", "year", "cycle", "day", "week", "sampling"))


def build_model(*, task, model="rf", spectral_columns=(), numeric_metadata=(),
                categorical_metadata=(), n_components=8, seed=SEED, n_jobs=2,
                scale_spectra=False, alpha=10., logistic_c=1., rf_params=None):
    """Build an unfitted pipeline; no data-dependent choice occurs outside fit.

    Spectra are mean-centered for linear models by default, with optional
    training-only channel autoscaling. PCA scores and numeric metadata are
    standardized. PLS directly uses preprocessed channels plus metadata, with
    no preceding PCA. Metadata-only RF/linear/PLS models are supported.
    """
    if task not in {"classification", "regression"} or model not in {"rf", "linear", "pls"}:
        raise ValueError("unknown task or model")
    if model == "pls" and task != "regression":
        raise ValueError("PLS is a regression comparator; use PCA/logistic for classification")
    spectra, numeric, categorical = map(list, (spectral_columns, numeric_metadata, categorical_metadata))
    chosen = spectra+numeric+categorical
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("choose nonempty, distinct model feature columns")
    if task == "classification" and any(_calendar_column(c) for c in numeric+categorical):
        raise ValueError("calendar-derived metadata would leak season labels into classification")
    transformers = []
    if spectra:
        steps = [("finite", FiniteSpectra())]
        if model in {"linear", "pls"}:
            steps.append(("center", StandardScaler(with_std=scale_spectra)))
        if model == "linear":
            steps += [("pca", CappedPCA(n_components, seed)), ("score_scale", StandardScaler())]
        transformers.append(("spectrum", Pipeline(steps), spectra))
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("strings", CategoricalStrings()),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    features = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    if model == "rf":
        params = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5,
                  "max_features": .33, "bootstrap": True}
        params.update(rf_params or {})
        params.update(random_state=seed, n_jobs=n_jobs)
        if task == "classification":
            params.setdefault("class_weight", "balanced_subsample")
            estimator = RandomForestClassifier(**params)
        else:
            estimator = RandomForestRegressor(**params)
    elif model == "pls":
        estimator = CappedPLSRegressor(n_components=n_components)
    elif task == "classification":
        estimator = LogisticRegression(C=logistic_c, class_weight="balanced", max_iter=2000,
                                       random_state=seed)
    else:
        estimator = Ridge(alpha=alpha)
    return Pipeline([("features", features), ("estimator", estimator)])


def _target(y, task, n):
    values = np.asarray(y)
    if values.ndim != 1 or len(values) != n or pd.isna(values).any():
        raise ValueError("one complete target is required per physical row")
    if task == "regression":
        values = values.astype(float)
        if not np.isfinite(values).all():
            raise ValueError("regression targets must be finite")
    elif task == "classification":
        values = values.astype(str)
    else:
        raise ValueError("unknown task")
    return values


def prediction_loss(y, prediction, task):
    """Higher is worse: RMSE or 1 - balanced accuracy."""
    prediction = np.asarray(prediction).reshape(-1)
    if task == "classification":
        return float(1-balanced_accuracy_score(y, prediction))
    if task == "regression":
        return float(np.sqrt(mean_squared_error(y, prediction)))
    raise ValueError("unknown task")


def evaluate_folds(X, y, physical_ids, folds, estimator, *, task):
    """Fit clones on training rows only and return physical-ID-level predictions."""
    if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
        raise ValueError("X must be a DataFrame with distinct feature names")
    ids = _ids(physical_ids, len(X), unique=True)
    if X.index.name == "physical_id" and not np.array_equal(X.index.astype(str), ids):
        raise ValueError("physical IDs differ from the feature-frame row order")
    target = _target(y, task, len(X))
    rows, summaries, fitted, seen = [], [], {}, set()
    for fold in folds:
        train, test = np.asarray(fold.train, int), np.asarray(fold.test, int)
        if (train.ndim != 1 or test.ndim != 1 or len(train) < 2 or not len(test)
                or min(train.min(), test.min()) < 0 or max(train.max(), test.max()) >= len(X)):
            raise ValueError("invalid fold row indices")
        if (len(set(train)) != len(train) or len(set(test)) != len(test)
                or set(ids[train]) & set(ids[test])):
            raise ValueError("duplicate or overlapping physical rows in a fold")
        if seen & set(ids[test]) or fold.name in fitted:
            raise ValueError("each physical test row and fold name must be unique in one evaluation protocol")
        seen.update(ids[test])
        if task == "classification" and len(set(target[train])) < 2:
            raise ValueError("classification training requires at least two classes")
        model = clone(estimator).fit(X.iloc[train], target[train])
        prediction = np.asarray(model.predict(X.iloc[test])).reshape(-1)
        common = {"fold": fold.name, "cycle": fold.test_cycle, "complete_test": fold.complete_test}
        record = {**common, "n_train": len(train), "n_test": len(test)}
        probabilities = None
        if task == "classification":
            labels = model.classes_
            probabilities = model.predict_proba(X.iloc[test])
            baseline = DummyClassifier(strategy="prior").fit(np.ones((len(train), 1)), target[train])
            baseline_prediction = baseline.predict(np.ones((len(test), 1)))
            record.update(
                accuracy=float(accuracy_score(target[test], prediction)),
                balanced_accuracy=float(balanced_accuracy_score(target[test], prediction)),
                macro_f1=float(f1_score(target[test], prediction, average="macro", zero_division=0)),
                log_loss=float(log_loss(target[test], probabilities, labels=labels)) if set(target[test]) <= set(labels) else np.nan,
                dummy_balanced_accuracy=float(balanced_accuracy_score(target[test], baseline_prediction)),
            )
        else:
            record.update(rmse=prediction_loss(target[test], prediction, task),
                          mae=float(mean_absolute_error(target[test], prediction)),
                          r2=float(r2_score(target[test], prediction)) if len(test) > 1 else np.nan,
                          bias=float(np.mean(prediction-target[test])),
                          dummy_rmse=prediction_loss(target[test], np.repeat(target[train].mean(), len(test)), task))
        summaries.append(record)
        for i, source in enumerate(test):
            item = {**common, "physical_id": ids[source], "truth": target[source], "prediction": prediction[i]}
            if probabilities is not None:
                item.update({f"proba__{label}": float(probabilities[i, j]) for j, label in enumerate(labels)})
            rows.append(item)
        fitted[fold.name] = {"estimator": model, "fold": fold}
    return {"predictions": pd.DataFrame(rows), "metrics": pd.DataFrame(summaries), "fitted_folds": fitted}


def spectral_regions(wavenumbers, spectral_columns, *, bin_width=100., named_regions=None,
                     include_whole=True):
    """Predeclare channel groups; named broad regions may overlap fine bins."""
    wn = np.asarray(wavenumbers, float)
    columns = list(spectral_columns)
    if (wn.ndim != 1 or len(wn) != len(columns) or not len(wn)
            or not np.isfinite(wn).all() or len(set(columns)) != len(columns)
            or len(set(wn)) != len(wn)):
        raise ValueError("wavenumbers must uniquely map to spectral columns")
    regions = {}
    if bin_width is not None:
        if not np.isfinite(bin_width) or bin_width <= 0:
            raise ValueError("bin width must be positive")
        bins = np.floor(wn/bin_width).astype(int)
        for value in sorted(set(bins)):
            name = f"bin_{value*bin_width:g}_{(value+1)*bin_width:g}"
            regions[name] = [columns[i] for i in np.flatnonzero(bins == value)]
    for name, bounds in (named_regions or {}).items():
        if name in regions or len(bounds) != 2 or not np.isfinite(bounds).all():
            raise ValueError("invalid or duplicate named spectral region")
        lo, hi = sorted(map(float, bounds))
        if lo == hi:
            raise ValueError("region must span a nonzero interval")
        group = [columns[i] for i in np.flatnonzero((wn >= lo) & (wn <= hi))]
        if group:
            regions[str(name)] = group
    if include_whole:
        if "whole_spectrum" in regions:
            raise ValueError("whole_spectrum is reserved")
        regions["whole_spectrum"] = columns
    return regions


def permute_region(X, columns, permutation):
    """Apply ONE row permutation jointly to all named channels; never mutate X."""
    if not isinstance(X, pd.DataFrame):
        raise ValueError("region permutation expects a named feature DataFrame")
    columns = list(columns)
    permutation = np.asarray(permutation)
    if (permutation.shape != (len(X),) or not np.issubdtype(permutation.dtype, np.integer)
            or not np.array_equal(np.sort(permutation), np.arange(len(X)))):
        raise ValueError("permutation must be a bijection of row positions")
    if not columns or len(set(columns)) != len(columns) or not set(columns) <= set(X.columns):
        raise ValueError("region must contain distinct available columns")
    result = X.copy(deep=True)
    # NumPy assignment avoids pandas re-aligning original labels and undoing the shuffle.
    result.loc[:, columns] = X.iloc[permutation].loc[:, columns].to_numpy()
    return result


def row_permutations(n_rows, *, repeats=10, seed=SEED, strata=None):
    """Deterministic shared permutations; optional strata preserve their labels."""
    if n_rows < 2 or repeats < 1:
        raise ValueError("permutation requires at least two test rows and one repeat")
    if strata is None:
        groups = [np.arange(n_rows)]
    else:
        labels = np.asarray(strata, dtype=object)
        if labels.shape != (n_rows,) or pd.isna(labels).any():
            raise ValueError("permutation strata must match complete test rows")
        codes, _ = pd.factorize(labels, sort=False)
        groups = [np.flatnonzero(codes == code) for code in np.unique(codes)]
    rng = np.random.default_rng(seed)
    permutations = []
    for _ in range(repeats):
        order = np.arange(n_rows)
        for group in groups:
            order[group] = rng.permutation(group)
        permutations.append(order)
    return permutations


def grouped_region_permutation(estimator, X_test, y_test, regions, *, task,
                               repeats=10, seed=SEED, strata=None):
    """Held-out block permutation loss increases, before fitted preprocessing.

    All regions share the same repeat permutations for paired comparison. A
    within-season shuffle assesses conditional reliance and can have little
    effect by construction, particularly for season classification. Negative
    importance is retained. Correlated substitute bands limit interpretation.
    """
    target = _target(y_test, task, len(X_test))
    baseline = prediction_loss(target, estimator.predict(X_test), task)
    permutations = row_permutations(len(X_test), repeats=repeats, seed=seed, strata=strata)
    rows = []
    for region, columns in regions.items():
        for repeat, order in enumerate(permutations):
            altered = permute_region(X_test, columns, order)
            loss = prediction_loss(target, estimator.predict(altered), task)
            rows.append({"region": region, "repeat": repeat, "baseline_loss": baseline,
                         "permuted_loss": loss, "importance": loss-baseline,
                         "moved_fraction": float(np.mean(order != np.arange(len(order)))),
                         "n_test": len(target), "n_channels": len(columns)})
    return pd.DataFrame(rows)


def leave_region_out_refit(estimator, X_train, y_train, X_test, y_test, regions, *, task):
    """Refit every prespecified omission, including all preprocessing, on training.

    This supports pipelines from build_model. It does not select the best region
    or change the outer split. Whole-spectrum removal without metadata yields a
    training-only intercept/prior model. DataFrames remain unchanged.
    """
    train_target = _target(y_train, task, len(X_train))
    test_target = _target(y_test, task, len(X_test))
    baseline_model = clone(estimator).fit(X_train, train_target)
    baseline_loss = prediction_loss(test_target, baseline_model.predict(X_test), task)
    rows = []
    for region, columns in regions.items():
        remove = set(columns)
        if not remove or not remove <= set(X_train.columns) or not remove <= set(X_test.columns):
            raise ValueError("ablation region must contain available feature columns")
        model = clone(estimator)
        if not isinstance(model, Pipeline) or "features" not in model.named_steps:
            raise ValueError("ablation expects a build_model pipeline")
        kept_transformers = []
        for name, transform, selected in model.named_steps["features"].transformers:
            kept = [column for column in selected if column not in remove]
            if kept:
                kept_transformers.append((name, transform, kept))
        if kept_transformers:
            model.set_params(features__transformers=kept_transformers)
            model.fit(X_train.drop(columns=list(remove)), train_target)
            prediction = model.predict(X_test.drop(columns=list(remove)))
            intercept_only = False
        else:
            model = DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor(strategy="mean")
            model.fit(np.ones((len(X_train), 1)), train_target)
            prediction = model.predict(np.ones((len(X_test), 1)))
            intercept_only = True
        loss = prediction_loss(test_target, prediction, task)
        rows.append({"region": region, "baseline_loss": baseline_loss,
                     "ablated_loss": loss, "importance": loss-baseline_loss,
                     "n_channels": len(remove), "intercept_only": intercept_only})
    return pd.DataFrame(rows)

