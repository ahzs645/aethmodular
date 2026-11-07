"""Training-only EPO projections and fixed, physical-filter site folds.

EPO estimates nuisance spectral signatures using source chemical measurements.
Consequently the projection is part of the fitted model, and is fitted anew in
each outer training fold.  Target optical measurements never enter these APIs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import continuum_anchor as ca
from candidate_metrics import validate_physical_ids


EPO_PROTOCOL = "epo_train_only_site_cv_v2"
CHEMISTRY_COLUMNS = ("TOR_EC", "TOR_OC", "other_PM", "Fe", "K")


def array_sha256(values) -> str:
    value = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def site_folds(filter_ids, sites, n_splits=5):
    """Split every bundled physical filter; label eligibility never sets folds."""
    ids = validate_physical_ids(filter_ids)
    sites = np.asarray(sites)
    if sites.shape != ids.shape or pd.isna(sites).any() or any(not str(s).strip() for s in sites):
        raise ValueError("every physical filter needs a nonmissing source site")
    return list(GroupKFold(n_splits).split(np.zeros((len(ids), 1)), groups=sites))


def chemistry_matrix(metadata: pd.DataFrame, mass: pd.DataFrame) -> np.ndarray:
    """EC, OC, residual PM, Fe and K in micrograms per physical filter."""
    validate_physical_ids(metadata.filter_id.to_numpy())
    validate_physical_ids(mass.filter_id.to_numpy(), label="IMPROVE mass")
    volume = metadata.volume_m3.to_numpy(float)
    ec = metadata.tor_ug.to_numpy(float)
    oc = metadata.tor_oc_ugm3.to_numpy(float) * volume
    pm = metadata.filter_id.map(mass.set_index("filter_id").pm25_ugm3).to_numpy(float) * volume
    other = pm - 1.8 * oc - ec
    return np.column_stack([ec, oc, other, metadata.fe_ugm3.to_numpy(float) * volume,
                            metadata.k_xrf_ugm3.to_numpy(float) * volume])


@dataclass
class Projection:
    basis: np.ndarray
    signatures: np.ndarray
    chemistry_mean: np.ndarray
    spectrum_mean: np.ndarray
    training_rows: np.ndarray
    chemistry_rank: int
    tolerance: float

    @property
    def rank(self):
        return self.basis.shape[1]

    def transform(self, spectra):
        values = np.asarray(spectra, float)
        return values - (values @ self.basis) @ self.basis.T


@dataclass
class EpoFit:
    projection: Projection
    pls_model: tuple
    training_rows: np.ndarray

    def predict(self, spectra):
        values = np.asarray(spectra, float)
        result = np.full((len(values), self.pls_model[2].shape[1]), np.nan)
        valid = np.isfinite(values).all(axis=1)
        result[valid] = ca.pls_predict(self.pls_model, self.projection.transform(values[valid]))
        return result


def training_rows(spectra, y, rows):
    """The source domain applies to the supplied training rows only."""
    X, y, rows = np.asarray(spectra, float), np.asarray(y, float), np.asarray(rows, int)
    return rows[np.isfinite(y[rows]) & (y[rows] < 150) & np.isfinite(X[rows]).all(axis=1)]


def fit_projection(spectra, chemistry, rows) -> Projection:
    X, Z, rows = np.asarray(spectra, float), np.asarray(chemistry, float), np.asarray(rows, int)
    if X.ndim != 2 or Z.shape != (len(X), len(CHEMISTRY_COLUMNS)):
        raise ValueError("EPO requires spectra and five aligned source chemistry columns")
    selected = rows[np.isfinite(X[rows]).all(axis=1) & np.isfinite(Z[rows]).all(axis=1)]
    if len(selected) <= Z.shape[1]:
        raise ValueError("EPO projection needs at least six complete training chemistry records")
    zmean, xmean = Z[selected].mean(axis=0), X[selected].mean(axis=0)
    signatures, _, design_rank, _ = np.linalg.lstsq(Z[selected] - zmean, X[selected] - xmean, rcond=None)
    if design_rank != Z.shape[1]:
        raise ValueError(f"EPO training chemistry design has rank {design_rank}, expected {Z.shape[1]}")
    nuisance = signatures[1:].T
    left, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    # Tolerance is tied to all signatures, so an entirely absent nuisance signal
    # does not acquire arbitrary directions merely through roundoff noise.
    tolerance = float(np.finfo(float).eps * max(nuisance.shape) * max(np.linalg.norm(signatures, 2), 1.0))
    rank = int(np.count_nonzero(singular > tolerance))
    return Projection(left[:, :rank], signatures, zmean, xmean, selected, int(design_rank), tolerance)


def fit_training_epo(spectra, y, chemistry, rows, kmax) -> EpoFit:
    X, y = np.asarray(spectra, float), np.asarray(y, float)
    if X.ndim != 2 or y.shape != (len(X),):
        raise ValueError("source spectra and TOR labels must have aligned rows")
    selected = training_rows(X, y, rows)
    projection = fit_projection(X, chemistry, selected)
    if kmax < 1 or kmax > min(len(selected) - 1, X.shape[1] - projection.rank):
        raise ValueError("PLS component count exceeds the available EPO training dimension")
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        model = ca.pls_fit(projection.transform(X[selected]), y[selected], kmax)
    if not all(np.isfinite(value).all() for value in model):
        raise ValueError("EPO PLS fit produced nonfinite coefficients")
    return EpoFit(projection, model, selected)


def fit_site_cv(spectra, y, chemistry, filter_ids, sites, kmax, *, splits=None):
    """Evaluate frozen site folds and then fit a separate all-source model."""
    X, y = np.asarray(spectra, float), np.asarray(y, float)
    ids = validate_physical_ids(filter_ids)
    sites = np.asarray(sites)
    if len(X) != len(ids) or y.shape != (len(ids),) or sites.shape != ids.shape or pd.isna(sites).any():
        raise ValueError("EPO source spectra, labels, IDs and sites must align")
    sites = sites.astype(str)
    splits = site_folds(ids, sites) if splits is None else splits
    fold_ids = np.full(len(ids), -1, dtype=int)
    predictions = np.full((len(ids), kmax), np.nan)
    fits, receipts = [], []
    all_rows = np.arange(len(ids))
    for fold_id, (train, test) in enumerate(splits):
        train, test = np.asarray(train, int), np.asarray(test, int)
        if (np.intersect1d(train, test).size or len(np.unique(train)) != len(train)
                or len(np.unique(test)) != len(test) or not np.array_equal(np.sort(np.r_[train, test]), all_rows)):
            raise ValueError("each outer split must partition the physical filters without overlap")
        if set(sites[train]) & set(sites[test]) or np.any(fold_ids[test] != -1):
            raise ValueError("held-out sites or physical filters overlap outer folds")
        fitted = fit_training_epo(X, y, chemistry, train, kmax)
        predictions[test] = fitted.predict(X[test])
        fold_ids[test] = fold_id
        fits.append(fitted)
        receipts.append(fit_receipt(fitted, X, y, chemistry, ids, sites, fold_id=fold_id, test_rows=test))
    if np.any(fold_ids < 0):
        raise ValueError("every source physical filter needs one outer held-out prediction")
    full = fit_training_epo(X, y, chemistry, all_rows, kmax)
    return predictions, fold_ids, fits, receipts, full


def fit_receipt(fitted, spectra, y, chemistry, ids, sites, *, fold_id=None, test_rows=()):
    projection = fitted.projection
    selected, projection_rows = fitted.training_rows, projection.training_rows
    ids, sites, test_rows = np.asarray(ids).astype(str), np.asarray(sites).astype(str), np.asarray(test_rows, int)
    return dict(
        role="all_source_target_model" if fold_id is None else "outer_training_fold",
        fold_id=fold_id, training_filter_ids=ids[selected].tolist(),
        projection_training_filter_ids=ids[projection_rows].tolist(), test_filter_ids=ids[test_rows].tolist(),
        training_sites=sorted(set(sites[selected])), test_sites=sorted(set(sites[test_rows])),
        n_train=len(selected), n_projection_train=len(projection_rows), n_test=len(test_rows),
        projection_rank=projection.rank, chemistry_design_rank=projection.chemistry_rank,
        rank_tolerance=projection.tolerance, k_max=fitted.pls_model[2].shape[1],
        training_spectra_sha256=array_sha256(np.asarray(spectra)[selected]),
        training_tor_sha256=array_sha256(np.asarray(y)[selected]),
        projection_chemistry_sha256=array_sha256(np.asarray(chemistry)[projection_rows]),
        basis_sha256=array_sha256(projection.basis),
    )


def model_arrays(prefix, fitted, ids):
    """Save actual transforms and coefficients, not just a recipe for them."""
    projection, (xm, ym, coefficients) = fitted.projection, fitted.pls_model
    ids = np.asarray(ids).astype(str)
    return {
        f"{prefix}|basis": projection.basis, f"{prefix}|signatures": projection.signatures,
        f"{prefix}|chemistry_mean": projection.chemistry_mean, f"{prefix}|spectrum_mean": projection.spectrum_mean,
        f"{prefix}|projection_training_ids": ids[projection.training_rows],
        f"{prefix}|training_ids": ids[fitted.training_rows], f"{prefix}|xm": xm,
        f"{prefix}|ym": np.asarray(ym), f"{prefix}|B": coefficients,
    }
