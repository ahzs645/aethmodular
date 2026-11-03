"""Portable, exploratory diagnostics for the gallery's packed FTIR spectra.

The gallery raw traces are six-channel means (~7.7 cm^-1), quantized to int16.
They are useful for triage, but are not the native spectra or a calibration
input. In particular, the gallery ``airspec`` view is clipped and normalized;
it must not be subtracted from raw absorbance to estimate a removed baseline.

This module preserves signed values and labels its own smoother as a sensitivity
analysis. It does not implement or reproduce AIRSpec, VIBES, EC prediction, or
chemical identification. No ambient/HIPS outcomes enter baseline fitting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


DEFAULT_EXCLUDED_WINDOWS = ((-np.inf, 700.0), (1100.0, 1300.0))
DEFAULT_BAND = (1580.0, 1650.0)
DEFAULT_FLANKS = ((1500.0, 1540.0), (1760.0, 1800.0))


def _matrix(wavenumbers, values):
    wn = np.asarray(wavenumbers, dtype=float)
    x = np.asarray(values, dtype=float)
    if wn.ndim != 1 or wn.size < 3 or not np.isfinite(wn).all():
        raise ValueError("wavenumbers must have at least three finite entries")
    if not (np.all(np.diff(wn) > 0) or np.all(np.diff(wn) < 0)):
        raise ValueError("wavenumbers must be strictly monotonic and unique")
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != len(wn) or not np.isfinite(x).all():
        raise ValueError("spectra must be finite rows on the supplied grid")
    return wn, x


def _window(wn, bounds):
    if len(bounds) != 2:
        raise ValueError("a window needs two bounds")
    lo, hi = sorted(float(v) for v in bounds)
    if np.isnan(lo) or np.isnan(hi) or lo == hi:
        raise ValueError("window bounds must be distinct and not NaN")
    return (wn >= lo) & (wn <= hi)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_packed_spectra(directory, *, views=("raw", "deriv")):
    """Decode explicitly described gallery traces, with input hashes and limits.

    Returns ``filters`` (the descriptor's column-oriented metadata), ``views``
    (each with ``wavenumbers``, ``values`` and ``spec``), and ``provenance``.
    Binary layout: float32 little-endian scale per filter, then int16
    little-endian row-major samples. All metadata rows and byte counts are
    validated. Values retain their original sign.
    """
    directory = Path(directory)
    descriptor = directory / "lot_spectra.json"
    d = json.loads(descriptor.read_text(encoding="utf-8"))
    if d.get("schema_version") != 1:
        raise ValueError("unsupported packed spectrum schema")
    filters = d.get("filters", {})
    ids = filters.get("id", [])
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("filter IDs must be nonempty and unique")
    n = len(ids)
    if any(not isinstance(v, list) or len(v) != n for v in filters.values()):
        raise ValueError("filter metadata columns have inconsistent lengths")
    out = {}
    hashes = {"lot_spectra.json": _sha256(descriptor)}
    for view in views:
        if view not in d.get("views", {}):
            raise ValueError(f"missing packed view: {view}")
        spec = d["views"][view]
        points = spec.get("points")
        if isinstance(points, bool) or not isinstance(points, int) or points < 3:
            raise ValueError("packed point count must be an integer >= 3")
        wn = np.asarray(spec.get("wn"), dtype=float)
        if wn.shape != (points,):
            raise ValueError("packed grid length differs from the point count")
        filename = spec.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("packed filenames must be local basenames")
        path = directory / filename
        expected = 4 * n + 2 * n * points
        if path.stat().st_size != expected or spec.get("bytes") != expected:
            raise ValueError("packed byte count differs from the descriptor")
        content = path.read_bytes()
        scales = np.frombuffer(content, dtype="<f4", count=n).astype(float)
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("packed scales must be finite and positive")
        quantized = np.frombuffer(content, dtype="<i2", offset=4 * n)
        values = quantized.reshape(n, points).astype(float) * scales[:, None]
        _matrix(wn, values)
        out[view] = {"wavenumbers": wn, "values": values, "spec": spec}
        hashes[filename] = hashlib.sha256(content).hexdigest()
    return {
        "filters": filters,
        "views": out,
        "provenance": {
            "input_sha256": hashes,
            "source_generated": d.get("generated"),
            "source_exporter": d.get("generated_from"),
            "filter_count": n,
            "native_spectra": False,
            "raw_binning": d["views"]["raw"].get("binning"),
            "limits": [
                "Raw traces are coarsened, quantized physical-filter averages.",
                "Analyses of the same filter cannot be recovered separately.",
                "No field blanks are included in this export.",
                "The AIRSpec view is clipped and positive-area normalized.",
                "These diagnostics do not establish EC accuracy or source identity.",
            ],
        },
    }


def local_band_contrast(
    wavenumbers, values, *, band=DEFAULT_BAND, flanks=DEFAULT_FLANKS
):
    """Signed band signal above a line joining the two flank means.

    The flank locations are the means of their actual sampled wavenumbers.
    ``height`` is the signed mean contrast, ``area`` its trapezoidal integral
    on ascending wavenumbers, and ``peak_wn`` the largest sampled contrast.
    This is an operational spectral contrast, not a chemically specific band
    concentration. Flank contamination and broad curvature remain possible.
    """
    wn, x = _matrix(wavenumbers, values)
    if len(flanks) != 2:
        raise ValueError("exactly two flank windows are required")
    order = np.argsort(wn)
    w, x = wn[order], x[:, order]
    masks = [_window(w, b) for b in (flanks[0], band, flanks[1])]
    if any(m.sum() < 2 for m in masks):
        raise ValueError("each band and flank must have at least two channels")
    left, mid, right = masks
    if not (w[left].max() < w[mid].min() < w[mid].max() < w[right].min()):
        raise ValueError("flanks must lie outside and on opposite sides of the band")
    wl, wr = w[left].mean(), w[right].mean()
    yl, yr = x[:, left].mean(axis=1), x[:, right].mean(axis=1)
    continuum = yl[:, None] + (yr - yl)[:, None] * (w[mid] - wl) / (wr - wl)
    contrast = x[:, mid] - continuum
    peak = np.argmax(contrast, axis=1)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {
        "height": contrast.mean(axis=1),
        "area": trapz(contrast, w[mid], axis=1),
        "peak_wn": w[mid][peak],
        "peak_height": contrast[np.arange(len(x)), peak],
        "n_band_channels": int(mid.sum()),
        "n_flank_channels": [int(left.sum()), int(right.sum())],
        "band": list(band),
        "flanks": [list(f) for f in flanks],
    }


def _curvature_matrix(w):
    """Second divided differences on a grid in median-channel units."""
    step = np.diff(w) / np.median(np.diff(w))
    a, b = step[:-1], step[1:]
    left = 2 / (a * (a + b))
    mid = -2 / (a * b)
    right = 2 / (b * (a + b))
    return sparse.diags((left, mid, right), (0, 1, 2), shape=(len(w) - 2, len(w)), format="csc")


def asymmetric_baseline(
    wavenumbers,
    values,
    *,
    lam=1e4,
    p=0.01,
    protected_windows: Iterable[tuple[float, float]] = (),
    excluded_windows=DEFAULT_EXCLUDED_WINDOWS,
    max_iter=100,
    tol=1e-7,
):
    """Fit an exploratory asymmetric penalized smooth baseline per row.

    Fits ``sum(w*(y-b)^2) + lam*||D2 b||^2`` with positive residual weight p
    and negative residual weight 1-p. D2 uses actual wavenumber spacing in
    median-channel units, so lambda belongs to this grid and must not be
    transferred to a different resolution unchanged. Protected windows have
    zero data weight; no claim is made that they are chemically identified.

    Excluded channels (<700 and PTFE 1100--1300 by default) are assigned zero
    fit weight and NaN in the returned baseline/corrected traces. Their values
    cannot influence the fitted baseline. Protected channels remain reported.
    Returns rows on the input grid, never clips signed corrected absorbance.
    """
    wn, x = _matrix(wavenumbers, values)
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(p) or not 0 < p < 0.5:
        raise ValueError("lam must be positive and 0 < p < 0.5")
    if not isinstance(max_iter, int) or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter and tol must be positive")
    excluded_windows = tuple(excluded_windows)
    protected_windows = tuple(protected_windows)
    order = np.argsort(wn)
    undo = np.argsort(order)
    w, y = wn[order], x[:, order]
    reported = np.ones(len(w), dtype=bool)
    for bounds in excluded_windows:
        reported &= ~_window(w, bounds)
    fit_mask = reported.copy()
    for bounds in protected_windows:
        fit_mask &= ~_window(w, bounds)
    if fit_mask.sum() < 4:
        raise ValueError("at least four fitted channels are required")
    d2 = _curvature_matrix(w)
    penalty = float(lam) * (d2.T @ d2)
    baselines = np.empty_like(y)
    converged = np.zeros(len(y), dtype=bool)
    iterations = np.zeros(len(y), dtype=int)
    relative_change = np.full(len(y), np.nan)
    for i, row in enumerate(y):
        weight = fit_mask.astype(float)
        previous = None
        for iteration in range(1, max_iter + 1):
            z = spsolve(penalty + sparse.diags(weight, format="csc"), weight * row)
            if not np.isfinite(z).all():
                raise ArithmeticError("baseline solver produced non-finite output")
            new_weight = np.where(row > z, float(p), 1.0 - p) * fit_mask
            if previous is not None:
                scale = max(np.linalg.norm(previous[reported]), np.finfo(float).eps)
                relative_change[i] = np.linalg.norm((z - previous)[reported]) / scale
                if np.array_equal(new_weight, weight) or relative_change[i] < tol:
                    converged[i] = True
                    break
            previous, weight = z, new_weight
        baselines[i] = z
        iterations[i] = iteration
    corrected = y - baselines
    baselines[:, ~reported] = np.nan
    corrected[:, ~reported] = np.nan
    return {
        "baseline": baselines[:, undo],
        "corrected": corrected[:, undo],
        "fit_mask": fit_mask[undo],
        "reported_mask": reported[undo],
        "converged": converged,
        "iterations": iterations,
        "relative_change": relative_change,
        "settings": {
            "method": "exploratory asymmetric penalized smoother",
            "lam": float(lam),
            "p": float(p),
            "grid_median_spacing_cm_1": float(np.median(np.diff(w))),
            "protected_windows": [list(v) for v in protected_windows],
            "max_iter": max_iter,
            "tol": tol,
            "clipped": False,
            "native_airspec_or_vibes": False,
        },
    }


def run_synthetic_controls(wavenumbers):
    """Known-input recovery checks for this coarse-grid smoother, not ambient EC.

    Add either a 1617 cm^-1 narrow Gaussian or a broad Gaussian input
    to a known smooth background. Recovery is the fitted correction on the
    spiked spectrum minus that on its unspiked parent. This paired difference
    avoids counting the smoother's pre-existing baseline error as recovery.
    Strong/weak amplitudes share one constructed parent and are not replicates.
    The broad case specifically tests whether a smooth real signal can be lost.
    """
    wn = np.asarray(wavenumbers, dtype=float)
    _matrix(wn, np.zeros_like(wn))
    u = (wn - 2300) / 1700
    parent = 0.15 + 0.04 * u + 0.025 * u**2
    candidates = [
        ("plain_lam1e4", 1e4, ()),
        ("plain_lam1e6", 1e6, ()),
        ("protected_1500_1760_lam1e4", 1e4, ((1500, 1760),)),
        ("protected_1500_1760_lam1e6", 1e6, ((1500, 1760),)),
    ]
    peaks = [
        ("narrow_1617_sigma25", np.exp(-0.5 * ((wn - 1617) / 25) ** 2), (1510, 1730)),
        ("broad_2800_sigma600", np.exp(-0.5 * ((wn - 2800) / 600) ** 2), (1500, 3990)),
    ]
    rows = []
    for candidate, lam, protected in candidates:
        base = asymmetric_baseline(wn, parent, lam=lam, protected_windows=protected)
        for label, shape, bounds in peaks:
            for amplitude in (0.001, 0.01, 0.05):
                addition = amplitude * shape
                spike = asymmetric_baseline(wn, parent + addition, lam=lam, protected_windows=protected)
                delta = spike["corrected"][0] - base["corrected"][0]
                mask = _window(wn, bounds) & base["reported_mask"]
                if mask.sum() < 3:
                    raise ValueError("synthetic control window has insufficient channels")
                ind = np.flatnonzero(mask)
                ind = ind[np.argsort(wn[ind])]
                trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
                recovered = float(trapz(delta[ind], wn[ind]))
                expected = float(trapz(addition[ind], wn[ind]))
                rows.append({
                    "candidate": candidate,
                    "control": label,
                    "amplitude_absorbance": amplitude,
                    "known_area": expected,
                    "recovered_area": recovered,
                    "area_recovery_fraction": recovered / expected,
                    "rms_recovery_error_absorbance": float(np.sqrt(np.mean((delta[mask] - addition[mask])**2))),
                    "parent_converged": bool(base["converged"][0]),
                    "spiked_converged": bool(spike["converged"][0]),
                    "independent_parent_count": 1,
                    "claim": "constructed-spectrum sensitivity only; not ambient EC or blank validation",
                })
    return rows
