"""Existing-data controls for signed native FTIR baselines.

This module changes no EC product or exclusion. Spline variants reuse the
repository's native spline solver; VIBES uses its vendored implementation and
explicitly local, fixed-rank ETAD blank banks. They are diagnostic candidates,
not the completed global rank-70 VIBES product.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

PHASE3 = Path(__file__).resolve().parents[2] / "ftir_ec_phase3/scripts"
if str(PHASE3) not in sys.path:
    sys.path.insert(0, str(PHASE3))
from airspec_baseline import (  # noqa: E402
    SEG1, SEG2, fit_spline_segm1, fit_spline_segm2ext, make_mask,
    smooth_spline_df,
)

METHODS = (
    {"method": "spline_default", "label": "Spline · native reference", "family": "spline",
     "fit_grid": "native segmented 1425–4000", "rank": None,
     "setting": "df1=6; df2=4; adaptive minimum 1520–1600; extension=20 native channels"},
    {"method": "spline_anchor1500", "label": "Spline · boundary 1500", "family": "spline",
     "fit_grid": "native segmented 1425–4000", "rank": None,
     "setting": "df1=6; df2=4; fixed segment-two boundary nearest 1500; extension=20 native channels"},
    {"method": "spline_anchor1560", "label": "Spline · boundary 1560", "family": "spline",
     "fit_grid": "native segmented 1425–4000", "rank": None,
     "setting": "df1=6; df2=4; fixed segment-two boundary nearest 1560; extension=20 native channels"},
    {"method": "neutral_arpls", "label": "Neutral robust spline", "family": "neutral",
     "fit_grid": "native >=700; omit 1100–1300", "rank": None,
     "setting": "repository pspline_arpls; lam=1e6; pybaselines 1.2 defaults; no band-specific anchor"},
    {"method": "neutral_arpls_iter500", "label": "Neutral spline · 500 iterations", "family": "neutral",
     "fit_grid": "native >=700; omit 1100–1300", "rank": None,
     "setting": "same pspline_arpls objective/lam=1e6/tol=1e-3; max_iter=500 numerical-convergence control"},
    {"method": "vibes_etad_r3", "label": "Local blank VIBES · rank 3", "family": "vibes",
     "fit_grid": "native full 500–4000", "rank": 3,
     "setting": "ETAD physical blank bank; fixed rank 3; PB loss; tau=0.1; not global rank 70"},
    {"method": "vibes_etad_r8", "label": "Local blank VIBES · rank 8", "family": "vibes",
     "fit_grid": "native full 500–4000", "rank": 8,
     "setting": "ETAD physical blank bank; fixed rank 8; PB loss; tau=0.1; not global rank 70"},
)

INJECTIONS = (
    {"control": "narrow_1617", "center": 1617.0, "sigma": 25.0, "bounds": (1510., 1730.)},
    {"control": "broad_2800", "center": 2800.0, "sigma": 600.0, "bounds": (1500., 3990.)},
)
AMPLITUDES = (0.001, 0.01, 0.05)


def validate_native_grid(wavenumbers, spectra):
    """Require a finite, descending, approximately native-resolution grid."""
    wn = np.asarray(wavenumbers, float)
    values = np.asarray(spectra, float)
    if wn.ndim != 1 or len(wn) < 10 or not np.isfinite(wn).all():
        raise ValueError("invalid native wavenumber grid")
    step = np.diff(wn)
    if np.any(step >= 0) or not np.allclose(step, step[0], atol=1e-8, rtol=1e-6):
        raise ValueError("native controls require a uniform descending grid")
    if abs(step[0]) > 2.0:
        raise ValueError("coarsened grids are not native spline controls")
    if values.ndim != 2 or values.shape[1] != len(wn) or not np.isfinite(values).all():
        raise ValueError("spectra must be complete finite native rows")
    return wn, values


def fixed_segment2(x, y, *, bound, df=4, n_ext=20):
    """Change only the original segment-two low boundary, on the same grid.

    The requested boundary is snapped to a native channel. The original
    effective df, protected interval to 1820, twenty-channel extension and
    below-extension zero-corrected behavior are retained.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not x.min() < bound < 1820 or np.any(np.diff(x) >= 0):
        raise ValueError("invalid fixed segment-two boundary or grid")
    imin = int(np.argmin(abs(x - bound)))
    actual = float(x[imin])
    stop = min(len(x), imin + int(n_ext) + 1)
    weights = (~make_mask(x[:stop], (1820., actual))).astype(float)
    baseline = y.copy()
    baseline[:stop] = smooth_spline_df(x[:stop], y[:stop], weights, df)
    return {"baseline": baseline, "absorbance": y-baseline,
            "bounds": np.array([1820., actual]), "stop_index": stop}


def segmented_spline_variants(wavenumbers, spectra, *, progress=None):
    """Return three signed native corrections and their actual boundaries."""
    wn, y = validate_native_grid(wavenumbers, spectra)
    m1, m2 = make_mask(wn, SEG1), make_mask(wn, SEG2)
    counts = m1.astype(int) + m2.astype(int)
    region = counts > 0
    methods = [m["method"] for m in METHODS if m["family"] == "spline"]
    baseline = {method: np.full_like(y, np.nan) for method in methods}
    records = []
    for i, row in enumerate(y):
        seg1 = fit_spline_segm1(wn[m1], row[m1], 6)
        seg2 = {
            "spline_default": fit_spline_segm2ext(wn[m2], row[m2], 4),
            "spline_anchor1500": fixed_segment2(wn[m2], row[m2], bound=1500),
            "spline_anchor1560": fixed_segment2(wn[m2], row[m2], bound=1560),
        }
        for method, second in seg2.items():
            total = np.zeros_like(wn)
            total[m1] += seg1["baseline"]
            total[m2] += second["baseline"]
            baseline[method][i, region] = total[region] / counts[region]
            bound = float(second["bounds"][1])
            anchor_index = int(np.argmin(abs(wn[m2] - bound)))
            end_index = min(m2.sum()-1, anchor_index+20)
            records.append({"row": i, "method": method,
                            "segment1_upper_boundary": float(seg1["bounds"][1]),
                            "segment2_lower_boundary": bound,
                            "extension_last_wavenumber": float(wn[m2][end_index]),
                            "extension_native_channels": 20,
                            "extension_span_cm_1": float(bound - wn[m2][end_index]),
                            "success": True, "converged": True})
        if progress is not None:
            progress(i+1, len(y))
    corrected = {key: y-value for key, value in baseline.items()}
    return baseline, corrected, pd.DataFrame(records)


def neutral_correction(wavenumbers, spectra, *, max_iter=50):
    """Repository neutral pspline_arpls, preserving its convergence evidence."""
    from pybaselines import Baseline
    from neutral_baseline import neutral_grid

    wn, y = validate_native_grid(wavenumbers, spectra)
    grid, indices = neutral_grid(wn)
    fitter = Baseline(x_data=grid)
    baseline = np.full_like(y, np.nan)
    records = []
    for i, row in enumerate(y):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            z, info = fitter.pspline_arpls(row[indices], lam=1e6, max_iter=max_iter)
        baseline[i, indices] = z
        history = np.asarray(info["tol_history"], float)
        finite = bool(np.isfinite(z).all())
        converged = bool(len(history) and np.isfinite(history[-1]) and history[-1] <= 1e-3)
        method = "neutral_arpls" if max_iter == 50 else f"neutral_arpls_iter{max_iter}"
        records.append({"row": i, "method": method, "success": finite,
                        "converged": converged, "iterations": len(history),
                        "last_tolerance": float(history[-1]) if len(history) else np.nan,
                        "warnings": " | ".join(dict.fromkeys(str(w.message) for w in caught))})
    return baseline, y-baseline, pd.DataFrame(records)


def blank_fold_assignments(physical_blanks, *, folds=5, seed=20261103):
    """Assign physical blanks, approximately balanced within lot, reproducibly."""
    required = {"MediaId", "base_filter_id", "lot_group"}
    if not required.issubset(physical_blanks.columns):
        raise ValueError("blank metadata lacks physical identity/lot")
    if physical_blanks.MediaId.duplicated().any() or physical_blanks.base_filter_id.duplicated().any():
        raise ValueError("repeat scans must be grouped into unique physical blanks")
    if folds < 2 or len(physical_blanks) < folds:
        raise ValueError("invalid number of physical-blank folds")
    data = physical_blanks.sort_values(["lot_group", "MediaId"]).copy()
    rng = np.random.default_rng(seed)
    assignment = {}
    offset = 0
    for _, group in data.groupby("lot_group", sort=True):
        shuffled = rng.permutation(group.MediaId.to_numpy())
        for j, media_id in enumerate(shuffled):
            assignment[int(media_id)] = int((j+offset) % folds)
        offset = (offset+len(group)) % folds
    data["fold"] = data.MediaId.map(assignment).astype(int)
    return data.reset_index(drop=True)


def assert_parent_holdout(training_ids, evaluation_parent_ids):
    """Guard real blank identity, including differently named injected children."""
    train = tuple(map(str, training_ids))
    parents = tuple(map(str, evaluation_parent_ids))
    if len(set(train)) != len(train):
        raise ValueError("training blank IDs are not unique physical filters")
    overlap = set(train) & set(parents)
    if overlap:
        raise ValueError("evaluation parent blank leaked into background training")


def fixed_vibes_background(wavenumbers, blanks, *, blank_ids, rank):
    """The vendored PCA basis at a declared local rank, without outcome tuning."""
    from vibes_baseline import VibesBackground
    from vibes.interference_models.pca import pca

    wn, values = validate_native_grid(wavenumbers, blanks)
    ids = tuple(map(str, blank_ids))
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError("one unique physical ID per training blank is required")
    if rank < 1 or rank > len(values)-2:
        raise ValueError("fixed rank requires at least rank+2 physical training blanks")
    start = time.perf_counter()
    mean, singular, _, components = pca(values)
    if singular[rank-1] <= np.finfo(float).eps * singular[0] * max(values.shape):
        raise ValueError("requested blank basis exceeds numerical rank")
    return VibesBackground(wn.copy(), mean, components[:, :rank],
                           pd.DataFrame(), ids, time.perf_counter()-start)


def _vibes_batch(start, wn, values, background, sample_ids):
    from vibes_baseline import vibes_baseline_matrix
    base, corrected, diagnostics = vibes_baseline_matrix(
        wn, values, background, sample_ids=sample_ids, tau=0.1,
        loss="PB", maxiter=5000, retry_failed=True,
    )
    return start, base, corrected, diagnostics


def correct_vibes(wavenumbers, spectra, background, *, sample_ids,
                  parent_ids, workers=4, batch_size=12):
    """Use upstream VIBES; failed fits stay NaN and are explicitly recorded."""
    from joblib import Parallel, delayed, parallel_config

    wn, values = validate_native_grid(wavenumbers, spectra)
    if len(sample_ids) != len(values) or len(parent_ids) != len(values):
        raise ValueError("evaluation identities do not match spectra")
    assert_parent_holdout(background.blank_ids, parent_ids)
    baseline = np.full_like(values, np.nan)
    corrected = np.full_like(values, np.nan)
    tasks = [(start, wn, values[start:start+batch_size], background,
              list(sample_ids[start:start+batch_size]))
             for start in range(0, len(values), batch_size)]
    if workers == 1:
        pieces = [_vibes_batch(*task) for task in tasks]
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            pieces = Parallel(n_jobs=workers)(delayed(_vibes_batch)(*task) for task in tasks)
    logs = []
    for start, base, corr, diagnostics in pieces:
        stop = start+len(base)
        baseline[start:stop], corrected[start:stop] = base, corr
        diagnostics["row"] = np.arange(start, stop)
        diagnostics["converged"] = diagnostics.success
        logs.append(diagnostics)
    return baseline, corrected, pd.concat(logs, ignore_index=True)


def paired_injection_metrics(wavenumbers, parent_corrected, spike_corrected,
                             addition, *, bounds):
    """Recover the added signal by paired subtraction; retain negative recovery."""
    wn = np.asarray(wavenumbers, float)
    parent, spike, known = map(lambda x: np.asarray(x, float),
                               (parent_corrected, spike_corrected, addition))
    if parent.shape != wn.shape or spike.shape != wn.shape or known.shape != wn.shape:
        raise ValueError("paired spectra and injection must share one grid")
    use = (wn >= min(bounds)) & (wn <= max(bounds))
    if use.sum() < 3 or not np.isfinite(known[use]).all():
        raise ValueError("invalid recovery window or known injection")
    expected = known[use]
    order = np.argsort(wn[use])
    x = wn[use][order]
    area = float(np.trapezoid(expected[order], x))
    if area <= 0:
        raise ValueError("known added area must be positive")
    recovered = (spike-parent)[use]
    success = bool(np.isfinite(recovered).all())
    recovered_area = float(np.trapezoid(recovered[order], x)) if success else np.nan
    return {"success": success, "known_area": area,
            "recovered_area": recovered_area,
            "area_recovery_fraction": recovered_area/area,
            "rms_recovery_error": float(np.sqrt(np.mean((recovered-expected)**2))) if success else np.nan,
            "relative_rms_recovery_error": float(np.sqrt(np.mean((recovered-expected)**2)) /
                                                  np.sqrt(np.mean(expected**2))) if success else np.nan}
