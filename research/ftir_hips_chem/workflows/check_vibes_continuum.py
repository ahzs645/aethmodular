"""Run a bounded signal-preservation check using the frozen blank background.

uv run --extra vibes python research/ftir_hips_chem/workflows/check_vibes_continuum.py --reference-dir DIR

DIR contains background_model.npz, wn.npy, cases.csv, RUN_MANIFEST.json and
fit_diagnostics.csv from the original full-range run. Raw spectra resolve through
the existing FTIR loaders. This does not launch or change a cloud runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / 'research/ftir_hips_chem'
sys.path.insert(0, str(AREA / 'scripts'))
from vibes_validation import validation_signals, incremental_recovery, diagnostic_flags, align_reference_diagnostics


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_run_manifest(output, record):
    output = Path(output)
    record = {**record, 'adapter_sha256': sha(AREA / 'scripts/vibes_baseline.py'),
              'outputs_sha256': {p.name: sha(p) for p in sorted(output.glob('*.csv'))}}
    (output / 'RUN.json').write_text(json.dumps(record, indent=2) + '\n')


def correct_parent(wn, raw, bg, parent_id, maxiter):
    from vibes_baseline import vibes_baseline_matrix
    signals = validation_signals(wn)
    labels = [('Unspiked blank', 0.)] + [(s, a) for s in signals for a in (.01, .05, .15)]
    inputs = np.stack([raw if a == 0 else raw + a * signals[s] for s, a in labels])
    ids = [f'validation:{parent_id}:{s}:{a}' for s, a in labels]
    cpu = time.process_time()
    _, values, diagnostics = vibes_baseline_matrix(wn, inputs, bg, sample_ids=ids, maxiter=maxiter)
    return values, diagnostics, time.process_time() - cpu


def main():
    from joblib import Parallel, delayed, parallel_config
    from vibes_baseline import VibesBackground
    from vibes_comparison import load_comparison_data
    from airspec_baseline import airspec_baseline_matrix

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=AREA / 'output/tables/vibes_continuum_check')
    ap.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    ref, out = args.reference_dir, args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ref / 'RUN_MANIFEST.json').read_text())
    cases = pd.read_csv(ref / 'cases.csv')
    historical = pd.read_csv(ref / 'fit_diagnostics.csv')
    historical = align_reference_diagnostics(cases, historical)
    if set(cases.sample_id) != set(historical.sample_id) or len(cases) != manifest['n_cases']:
        raise ValueError('Reference manifest, case identities and diagnostics do not align')
    historical['success'] = diagnostic_flags(historical.success)
    if int((~historical.success).sum()) != manifest['n_failed']:
        raise ValueError('Reference failure count does not match its manifest')
    with np.load(ref / 'background_model.npz', allow_pickle=False) as z:
        wn = z['wn'].copy()
        bg = VibesBackground(wn, z['mean'].copy(), z['components'].copy(), pd.DataFrame(),
                             tuple(z['blank_ids'].astype(str)), 0.)
    if bg.components.shape != (len(wn), manifest['background_rank']):
        raise ValueError('Background rank/grid does not match its manifest')
    np.testing.assert_array_equal(wn, np.load(ref / 'wn.npy', allow_pickle=False))
    native_wn, raw, ledger, raw_hashes = load_comparison_data(full_range=True)
    np.testing.assert_allclose(native_wn, wn, rtol=0, atol=1e-8)
    parents = cases[cases.kind.eq('blank') & cases.source.eq('etad')].copy().reset_index(drop=True)
    expected_ids = {f'etad:{int(i)}' for i in ledger.index[ledger.role.eq('blank_test')]}
    if set(parents.sample_id) != expected_ids or set(parents.sample_id) & set(bg.blank_ids):
        raise ValueError('Held-out physical blanks differ from the frozen split or leak into training')
    positions = {f'etad:{int(media)}': i for i, media in enumerate(ledger.index)}
    raw_parents = raw[[positions[s] for s in parents.sample_id]]
    signals = validation_signals(wn)
    labels = [('Unspiked blank', 0.)] + [(s, a) for s in signals for a in (.01, .05, .15)]
    started = time.perf_counter()
    with parallel_config(backend='loky', inner_max_num_threads=1):
        fits = Parallel(n_jobs=args.workers)(delayed(correct_parent)(wn, x, bg, sid, manifest['config']['maxiter'])
                                            for x, sid in zip(raw_parents, parents.sample_id))
    vibes_wall = time.perf_counter() - started
    recovery, traces, diagnostics, blank_rms = [], [], [], []
    spline_wall, spline_success = 0., 0
    windows = {'Full measured grid': np.ones(len(wn), bool), 'Shared spline window': wn >= 1425.}
    for p, (vibes, diag, cpu_seconds) in enumerate(fits):
        parent = parents.iloc[p]
        matrices = np.stack([raw_parents[p] if a == 0 else raw_parents[p] + a * signals[s] for s, a in labels])
        stamp = time.perf_counter()
        _, spline = airspec_baseline_matrix(wn, matrices, df1=6, df2=4)
        spline_wall += time.perf_counter() - stamp
        spline_success += int(np.isfinite(spline[:, wn >= 1425.]).all(axis=1).sum())
        diag = diag.assign(method='VIBES', parent_id=parent.sample_id, filter_id=parent.filter_id,
                           signal=[s for s, a in labels], amplitude=[a for s, a in labels])
        diagnostics.append(diag)
        values = {'VIBES': vibes, 'Spline baseline': spline}
        for method, corrected in values.items():
            for window, mask in windows.items():
                blank_rms.append({'parent_id': parent.sample_id, 'filter_id': parent.filter_id,
                    'method': method, 'window': window,
                    'RMS': np.sqrt(np.mean(corrected[0, mask] ** 2)) if np.isfinite(corrected[0, mask]).all() else np.nan})
                for j, (signal, amplitude) in enumerate(labels[1:], 1):
                    metric = incremental_recovery(wn[mask], amplitude * signals[signal][mask],
                                                   corrected[0, mask], corrected[j, mask])
                    recovery.append({'parent_id': parent.sample_id, 'filter_id': parent.filter_id,
                        'method': method, 'window': window, 'signal': signal, 'amplitude': amplitude, **metric})
            # One declared parent, every signal/amplitude; adjacent-channel means,
            # not a selected best-looking result. The metrics above use full resolution.
            if p == 0:
                bins = [np.arange(i, min(i + 3, len(wn))) for i in range(0, len(wn), 3)]
                for j, (signal, amplitude) in enumerate(labels[1:], 1):
                    recovered = corrected[j] - corrected[0]
                    for b in bins:
                        traces.append({'filter_id': parent.filter_id, 'signal': signal, 'amplitude': amplitude,
                            'method': method, 'wn': wn[b].mean(), 'value': recovered[b].mean()})
        if p == 0:
            for signal, amplitude in labels[1:]:
                for i in range(0, len(wn), 3):
                    b = slice(i, i + 3)
                    traces.append({'filter_id': parent.filter_id, 'signal': signal, 'amplitude': amplitude,
                        'method': 'Known addition', 'wn': wn[b].mean(), 'value': (amplitude * signals[signal][b]).mean()})
    pd.DataFrame(recovery).to_csv(out / 'signal_recovery.csv', index=False)
    pd.DataFrame(traces).to_csv(out / 'signal_traces.csv', index=False)
    pd.DataFrame(blank_rms).to_csv(out / 'blank_rms.csv', index=False)
    diag = pd.concat(diagnostics, ignore_index=True)
    diag.to_csv(out / 'fit_diagnostics.csv', index=False)
    reference_counts = historical.merge(cases[['sample_id', 'kind']], on='sample_id', validate='one_to_one')
    reference_counts.groupby(['kind', 'success']).size().rename('n').reset_index().to_csv(out / 'reference_fit_counts.csv', index=False)
    total = len(diag)
    runtime = [{'method': 'VIBES', 'attempted_fits': total, 'successful_fits': int(diag.success.sum()),
                'wall_seconds': vibes_wall, 'successful_fits_per_minute': 60 * diag.success.sum() / vibes_wall,
                'workers': args.workers, 'reported_logical_cpus': os.cpu_count(),
                'worker_CPU_seconds': sum(r[2] for r in fits)},
               {'method': 'Spline baseline', 'attempted_fits': total, 'successful_fits': spline_success,
                'wall_seconds': spline_wall, 'successful_fits_per_minute': 60 * spline_success / spline_wall,
                'workers': 1, 'reported_logical_cpus': os.cpu_count(), 'worker_CPU_seconds': np.nan}]
    pd.DataFrame(runtime).to_csv(out / 'runtime.csv', index=False)
    sources = {name: sha(ref / name) for name in ['background_model.npz', 'wn.npy', 'cases.csv', 'fit_diagnostics.csv', 'RUN_MANIFEST.json']}
    record = {'schema_version': 1, 'created_utc': pd.Timestamp.now(tz='UTC').isoformat(),
              'reference_signature': manifest['signature'], 'background_rank': manifest['background_rank'],
              'training_blanks': len(bg.blank_ids), 'independent_test_blanks': len(parents),
              'channels': len(wn), 'wavenumber_range': [float(wn.max()), float(wn.min())],
              'source_sha256': sources,
              'native_source_sha256': {Path(k).name: v for k, v in raw_hashes.items()},
              'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'cvxpy', 'clarabel']},
              'validation_source_sha256': sha(AREA / 'scripts/vibes_validation.py'),
              'workflow_sha256': sha(Path(__file__)),
              'scope': 'Bounded local synthetic-addition check on independent real held-out blanks; no cloud job or model refit.',
              'acceptance': 'Descriptive results; no preservation threshold was specified. Optimizer convergence does not establish continuum preservation.',
              'new_cloud_run': 'The separately reported 32,286-fit/G4 job is not connected to this export.'}
    write_run_manifest(out, record)
    print(json.dumps({'independent_blanks': len(parents), 'vibes_fits': total,
                      'successful': int(diag.success.sum()), 'wall_seconds': vibes_wall}, indent=2))


if __name__ == '__main__':
    main()
