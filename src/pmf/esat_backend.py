"""Execute a source-pinned ESAT SA Python backend, never silently substitute it.

`ws-nmf-repaired` is an explicit LOCAL modification of the update rule under
ESAT's original SA controller. It is not stock ESAT or nonnegative-contribution
PMF. Upstream files and their Apache license are preserved byte for byte.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .io import sha256
from .solver import _h_block, diagnostics, validate_arrays

SOURCE = Path(__file__).parent / 'vendor/esat_source'
MODES = ('ls-nmf', 'ws-nmf', 'ws-nmf-repaired')


def verify_source():
    pin = json.loads((SOURCE / 'PINNED_SOURCES.json').read_text())
    for relative, expected in pin['files'].items():
        if sha256(SOURCE / relative) != expected:
            raise ValueError(f'Pinned ESAT source changed: {relative}')
    return pin


def _load_sa():
    verify_source()
    for name, module in list(sys.modules.items()):
        if name == 'esat' or name.startswith('esat.'):
            file = getattr(module, '__file__', '')
            if not file or not Path(file).resolve().is_relative_to(SOURCE.resolve()):
                raise RuntimeError('A different ESAT is already imported; use a fresh process for pinned ESAT')
    root = str(SOURCE.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module('esat.model.sa').SA


def repaired_ws_update(V, We, W, H, hold_h=False):
    """Signed W weighted least squares, nonnegative H coordinate NNLS.

    The repair replaces the stock signed multiplicative update and supplies the
    hold_h keyword required by SA.train. All data/weights are unchanged. The
    source module itself is not patched. Both blocks minimize/decrease the
    declared weighted quadratic, not Huber or the EPA robust diagnostic.
    """
    v, weights = np.asarray(V, float), np.asarray(We, float)
    h = np.asarray(H, float).copy()
    w = np.empty((len(v), h.shape[0]))
    for i in range(len(v)):
        root = np.sqrt(weights[i])
        w[i] = np.linalg.lstsq(h.T * root[:, None], v[i] * root, rcond=None)[0]
    if not hold_h:
        _h_block(v, weights, w, h, sweeps=24)
    norms = np.linalg.norm(h, axis=1)
    norms[norms == 0] = 1
    if not hold_h:
        h /= norms[:, None]
        w *= norms[None, :]
    return w, h


def stationarity(x, u, g, h, signed_g=False):
    w = 1 / u**2
    residual = (g @ h - x) * w
    dg, dh = residual @ h.T, g.T @ residual
    if not signed_g:
        dg = np.where((g > 0) | (dg < 0), dg, 0)
    dh = np.where((h > 0) | (dh < 0), dh, 0)
    scale = np.linalg.norm((w * x) @ h.T)**2 + np.linalg.norm(g.T @ (w * x))**2
    return float(np.sqrt((np.sum(dg**2) + np.sum(dh**2)) / max(scale, np.finfo(float).tiny)))


def fit_esat(X, U, n_factors=5, seed=42, *, method='ls-nmf', loss='quadratic',
             max_iter=2000, tol=1e-7, stationarity_tol=1e-5, **unused):
    start = time.perf_counter()
    x, u = validate_arrays(X, U)
    if method not in MODES:
        raise ValueError(f'ESAT method must be one of {MODES}')
    if loss != 'quadratic':
        raise ValueError('Pinned ESAT Python robust_mode does not optimize Huber; robust requests are rejected')
    if method == 'ls-nmf' and np.any(x < 0):
        raise ValueError('LS-NMF cannot accept signed X; no clipping or automatic method switching is allowed')
    if isinstance(n_factors, bool) or not isinstance(n_factors, (int, np.integer)) or not 1 <= n_factors <= min(x.shape):
        raise ValueError('Invalid factor count')
    if not isinstance(max_iter, int) or max_iter < 1 or not all(np.isfinite(v) and v > 0 for v in (tol, stationarity_tol)):
        raise ValueError('Invalid convergence settings')
    pin = verify_source()
    SA = _load_sa()
    col = np.median(u, axis=0)
    row = np.median(u / col, axis=1)
    xc, uc = x / row[:, None] / col, u / row[:, None] / col
    if not np.isfinite(xc).all() or not np.isfinite(uc).all() or uc.min() < 1e-12:
        raise ValueError('Conditioned inputs would trigger ESAT uncertainty flooring or overflow')
    model = SA(xc.copy(), uc.copy(), int(n_factors), method='ls-nmf' if method == 'ls-nmf' else 'ws-nmf',
               seed=int(seed), verbose=False, use_gpu=False)
    model.optimized = False  # Backend is explicitly pinned to the recovered Python implementation.
    if not all(np.isfinite(v).all() for v in (model.V, model.U, model.We)) or np.any(model.We <= 0):
        raise ValueError('ESAT input conversion or inverse-variance weights overflowed')
    if not np.array_equal(model.V, xc.astype(np.float32)) or not np.array_equal(model.U, uc.astype(np.float32)):
        raise RuntimeError('ESAT changed inputs beyond its documented float32 conversion')
    rng = np.random.default_rng(seed)
    magnitude = max(float(np.mean(np.maximum(xc, 0))), 0.01)
    g0 = rng.uniform(.2, 1.8, (len(x), n_factors))
    h0 = rng.uniform(.2, 1.8, (n_factors, x.shape[1])) * magnitude / n_factors
    model.initialize(W=g0.copy(), H=h0.copy(), init_norm=False)
    initial = float(np.sum(((model.V - model.W @ model.H) / model.U)**2))
    if method == 'ws-nmf-repaired':
        model.update_step = repaired_ws_update
    # Stock WS deliberately executes its real code path, preserving the TypeError
    # or invalid numerical result as a failed run, rather than returning fake factors.
    with threadpool_limits(limits=1):
        model.train(max_iter=max_iter, converge_delta=tol * max(1, initial), converge_n=30, robust_mode=False)
    g, h = np.asarray(model.W, float), np.asarray(model.H, float)
    if not np.isfinite(g).all() or not np.isfinite(h).all() or np.any(h < 0):
        raise FloatingPointError('ESAT returned invalid factors')
    if method == 'ls-nmf' and np.any(g < 0):
        raise FloatingPointError('LS-NMF returned signed contributions')
    history_q = np.asarray([initial, *model.q_list], float)
    if not np.isfinite(history_q).all():
        raise FloatingPointError('ESAT returned nonfinite objective history')
    maximum_increase = float(np.max(np.diff(history_q) / np.maximum(np.abs(history_q[:-1]), 1)))
    if maximum_increase > 2e-5:
        raise RuntimeError('ESAT objective increased beyond float32 tolerance')
    pg = stationarity(model.V.astype(float), model.U.astype(float), g, h, signed_g=method != 'ls-nmf')
    G, H = g * row[:, None], h * col[None, :]
    sums = H.sum(axis=1)
    active = sums > 0
    G[:, active] *= sums[active]
    H[active] /= sums[active, None]
    G[:, ~active] = 0
    metrics = diagnostics(x, u, G, H)
    # Signed-contribution semi-NMF has no nonnegative-reconstruction lower bound.
    if method != 'ls-nmf':
        for key in ('negative_observation_Q_lower_bound', 'negative_observation_objective_lower_bound'):
            metrics.pop(key, None)
    converged = bool(model.converged and pg <= stationarity_tol)
    metrics.update({'converged': converged, 'stop_reason': 'stationary_objective_plateau' if converged else 'unverified_stationarity_or_iteration_cap',
        'native_controller_converged': bool(model.converged), 'iterations': len(model.q_list),
        'relative_projected_gradient_conditioned': pg, 'seed': int(seed), 'n_factors': int(n_factors),
        'loss': 'quadratic', 'fraction_negative_contributions': float((G < 0).mean()),
        'native_reported_Q_true_conditioned_float32': float(model.Qtrue),
        'maximum_objective_relative_increase': max(0.0, maximum_increase), 'elapsed_seconds': time.perf_counter()-start})
    return {'G': G, 'H': H, 'history': np.column_stack([np.arange(len(history_q)), history_q]),
        'metrics': metrics, 'config': {'method': method, 'loss': 'quadratic', 'seed': int(seed),
            'n_factors': int(n_factors), 'max_iter': max_iter, 'tol': tol, 'stationarity_tol': stationarity_tol,
            'native_converge_delta': tol * max(1, initial), 'native_converge_n': 30},
        'provenance': {'engine': 'ESAT-SA-Python', 'upstream_repository': pin['repository'], 'upstream_commit': pin['commit'],
            'source_hashes': pin['files'], 'adapter_sha256': sha256(__file__),
            'method': method, 'upstream_update_replaced': method == 'ws-nmf-repaired',
            'signed_contributions_allowed': method != 'ls-nmf', 'data_preprocessing': 'positive row/column conditioning then upstream float32 conversion',
            'data_input_cast_max_error': float(np.max(np.abs(model.V - xc))),
            'input_verified_after_constructor': True, 'stock_ESAT_algorithm': method != 'ws-nmf-repaired',
            'native_EPA_PMF_or_ME2_executed': False, 'source_mass_calibrated': False,
            'history_columns': ['iteration', 'Q_true_conditioned_float32_input'],
            'warnings': ['WS-NMF allows signed contributions and is not conventional nonnegative-contribution PMF.'] if method != 'ls-nmf' else []}}
