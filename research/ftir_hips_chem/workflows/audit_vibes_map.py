"""Replay saved pinball MAP problems and audit status without rerunning ELBO.

Uses exactly the upstream MAP objective and CLARABEL solver, retaining status
that the upstream function does not return. Reconciles all recovery metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
import time

import cvxpy as cp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / 'research/ftir_hips_chem'
sys.path.insert(0, str(AREA / 'scripts'))
from vibes_comparison import load_comparison_data
from vibes_validation import validation_signals, incremental_recovery, diagnostic_flags
from check_vibes_continuum import sha, write_run_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=AREA / 'output/tables/vibes_continuum_check')
    args = ap.parse_args()
    out = args.output
    record = json.loads((out / 'RUN.json').read_text())
    for name in ['background_model.npz', 'RUN_MANIFEST.json', 'cases.csv', 'wn.npy']:
        if sha(args.reference_dir / name) != record['source_sha256'][name]:
            raise ValueError('Reference changed before MAP audit: ' + name)
    wn, raw, ledger, hashes = load_comparison_data(full_range=True)
    if {Path(k).name: v for k, v in hashes.items()} != record['native_source_sha256']:
        raise ValueError('Native inputs changed before MAP audit')
    with np.load(args.reference_dir / 'background_model.npz', allow_pickle=False) as bg:
        np.testing.assert_allclose(wn, bg['wn'], rtol=0, atol=1e-8)
        mean, components = bg['mean'], bg['components']
    signals = validation_signals(wn)
    positions = {f'etad:{int(media)}': i for i, media in enumerate(ledger.index)}
    diag = pd.read_csv(out / 'fit_diagnostics.csv')
    diag['success'] = diagnostic_flags(diag.success)
    replay, audit = {}, []
    for row in diag.itertuples():
        if not row.success:
            raise ValueError('Cannot replay a fit without saved successful ELBO parameters')
        base = raw[positions[row.parent_id]]
        y = base if row.amplitude == 0 else base + row.amplitude * signals[row.signal]
        latent = cp.Variable(components.shape[1])
        residual = y - mean - components @ latent
        objective = cp.sum(.5 * cp.abs(residual) + (row.tau - .5) * residual)
        if row.sigma != 0:
            objective += .5 * row.sigma * cp.sum_squares(latent)
        problem = cp.Problem(cp.Minimize(objective))
        started = time.perf_counter()
        problem.solve(solver=cp.CLARABEL)
        finite = latent.value is not None and np.isfinite(latent.value).all()
        audit.append({'parent_id': row.parent_id, 'filter_id': row.filter_id, 'signal': row.signal,
            'amplitude': row.amplitude, 'status': problem.status, 'finite': bool(finite),
            'seconds': time.perf_counter() - started, 'iterations': problem.solver_stats.num_iters})
        if not finite:
            raise ValueError('MAP replay returned nonfinite values')
        replay[(row.parent_id, row.signal, row.amplitude)] = y - (mean + components @ latent.value)
    metrics = pd.read_csv(out / 'signal_recovery.csv').query("method == 'VIBES'")
    differences = []
    for row in metrics.itertuples():
        mask = wn >= 1425. if row.window == 'Shared spline window' else np.ones(len(wn), bool)
        recovered = incremental_recovery(wn[mask], row.amplitude * signals[row.signal][mask],
            replay[(row.parent_id, 'Unspiked blank', 0.)][mask], replay[(row.parent_id, row.signal, row.amplitude)][mask])
        for name in ['area_recovery', 'gain', 'relative_RMSE']:
            differences.append(abs(recovered[name] - getattr(row, name)))
    maximum = float(max(differences))
    if not np.isfinite(maximum) or maximum > 1e-8:
        raise ValueError(f'MAP replay changed recovery metrics: {maximum}')
    frame = pd.DataFrame(audit)
    frame.to_csv(out / 'map_audit.csv', index=False)
    record.update(map_optimal=int(frame.status.eq('optimal').sum()), map_replayed=len(frame),
                  map_metric_max_abs_delta=maximum, map_audit_workflow_sha256=sha(Path(__file__)))
    write_run_manifest(out, record)
    print(json.dumps({'statuses': frame.status.value_counts().to_dict(),
                      'maximum_recovery_metric_difference': maximum}, indent=2))


if __name__ == '__main__':
    main()
