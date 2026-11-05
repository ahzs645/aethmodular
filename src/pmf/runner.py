"""Recoverable multi-start fits, strict cache validation and frozen projection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform

import numpy as np
import scipy

from .io import archive_source, Dataset, canonical_hash, fresh_directory, load_dataset, sha256, source_snapshot, write_json, write_matrix, write_table
from .solver import fit, diagnostics, profile_match, project_contributions


def save_model(path, result, data, request_hash):
    path = Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('Model checkpoint already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, G=result['G'], H=result['H'], history=result['history'],
                        sample_ids=data.sample_ids, features=data.features)
    write_json(path.with_suffix('.json'), {'schema': 'aethmodular-pmf-model/2',
        'array_file_sha256': sha256(path), 'dataset_identity': data.identity, 'request_hash': request_hash,
        'metrics': result['metrics'], 'config': result['config'], 'provenance': result['provenance']})


def load_model(path, data, request_hash=None):
    path = Path(path)
    info = json.loads(path.with_suffix('.json').read_text())
    if info.get('schema') != 'aethmodular-pmf-model/2' or info['array_file_sha256'] != sha256(path):
        raise ValueError('Model checksum/schema mismatch')
    if info['dataset_identity'] != data.identity or (request_hash is not None and info['request_hash'] != request_hash):
        raise ValueError('Cached model input/configuration mismatch')
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z['sample_ids'], data.sample_ids) or not np.array_equal(z['features'], data.features):
            raise ValueError('Model identities do not match input')
        g, h, hist = z['G'], z['H'], z['history']
    signed = info['provenance'].get('signed_contributions_allowed', False)
    if g.ndim != 2 or h.ndim != 2 or g.shape[0] != len(data.X) or h.shape[1] != data.X.shape[1] or g.shape[1] != h.shape[0]:
        raise ValueError('Invalid checkpoint factor dimensions')
    if not all(np.isfinite(a).all() for a in (g, h, hist)) or np.any(h < 0) or (np.any(g < 0) and not signed):
        raise ValueError('Invalid checkpoint factors/history')
    calculated = diagnostics(data.X, data.U, g, h, info['metrics'].get('loss', 'quadratic'), info['config'].get('delta', 4.0))
    for key in ('Q_true', 'objective'):
        if not np.isclose(calculated[key], info['metrics'][key], rtol=1e-9, atol=1e-8):
            raise ValueError('Cached objective disagrees with factor reconstruction')
    return {'G': g, 'H': h, 'history': hist, **{k: info[k] for k in ('metrics', 'config', 'provenance')}}


def export_model(out, result, data):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g, h = result['G'], result['H']
    factor_ids = np.asarray([f'F{i+1}' for i in range(h.shape[0])])
    residuals = (data.X - g @ h) / data.U
    write_matrix(out / 'contributions.csv', data.sample_ids, factor_ids, g)
    write_matrix(out / 'profiles.csv', factor_ids, data.features, h)
    write_matrix(out / 'scaled_residuals.csv', data.sample_ids, data.features, residuals)
    write_table(out / 'sample_residuals.csv', [{'sample_id': s, 'rms': float(np.sqrt(np.mean(r*r))),
        'fraction_abs_gt3': float(np.mean(np.abs(r)>3)), **{k: str(v[i]) for k, v in data.metadata.items()}}
        for i, (s, r) in enumerate(zip(data.sample_ids, residuals))])
    write_table(out / 'feature_residuals.csv', [{'feature': f, 'rms': float(np.sqrt(np.mean(r*r))),
        'mean': float(np.mean(r)), 'fraction_abs_gt3': float(np.mean(np.abs(r)>3))}
        for f, r in zip(data.features, residuals.T)])


def _ints(values, name):
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in values):
        raise ValueError(f'{name} must be a nonempty integer list')
    if len(set(values)) != len(values):
        raise ValueError(f'{name} must not contain duplicates')
    return list(map(int, values))


def run_case(data, output, *, dataset_id, factors=(5,), seeds=(1, 10, 100), engine='python-independent',
             loss='huber', delta=4.0, max_iter=3000, tol=1e-7, stationarity_tol=1e-5,
             warmup=600, esat_method='ls-nmf', resume=False, report=True, native_runtime=None, native_timeout=600.):
    factors, seeds = _ints(list(factors), 'factor counts'), _ints(list(seeds), 'seeds')
    if any(k < 1 or k > min(data.X.shape) for k in factors) or any(s < 0 for s in seeds):
        raise ValueError('Invalid factor counts or seeds')
    if engine not in ('python-independent', 'esat', 'epa-pmf5') or loss not in ('quadratic', 'huber', 'native-robust'):
        raise ValueError('Unknown engine/loss')
    if (engine == 'epa-pmf5' and loss == 'huber') or (engine != 'epa-pmf5' and loss == 'native-robust'):
        raise ValueError('Use native-robust for EPA only; independent Huber is a distinct objective')
    if not np.isfinite(native_timeout) or native_timeout <= 0:
        raise ValueError('Invalid native timeout')
    if isinstance(max_iter,bool) or not isinstance(max_iter, int) or max_iter < 1 or isinstance(warmup,bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('Invalid iteration limits')
    if not all(np.isfinite(v) and v > 0 for v in (tol, stationarity_tol, delta)):
        raise ValueError('Invalid numerical tolerance/cutoff')
    if not str(dataset_id).strip():
        raise ValueError('A nonblank dataset ID is required')
    config = {'factors': factors, 'seeds': seeds, 'engine': engine, 'loss': loss, 'delta': delta,
        'max_iter': max_iter, 'tol': tol, 'stationarity_tol': stationarity_tol, 'warmup': warmup,
        'esat_method': esat_method if engine == 'esat' else None,
        'native_timeout': native_timeout if engine == 'epa-pmf5' else None,
        'native_runtime_hash_contract': 'official-EPA-PMF5-unchanged' if engine == 'epa-pmf5' else None,
        'native_driver_git_blob': 'da2ae8291ed0d592767f51ac37e85c72516334ab' if engine == 'epa-pmf5' else None}
    request = {'schema': 'aethmodular-pmf-request/2', 'dataset_id': str(dataset_id),
        'dataset_identity': data.identity, 'configuration': config, 'source_hashes': source_snapshot(),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}}
    request_hash = canonical_hash(request)
    out = Path(output)
    if resume:
        old = json.loads((out / 'request.json').read_text())
        if old != request or load_dataset(out / 'input.npz').identity != data.identity:
            raise ValueError('Resume refused: inputs, code, versions or settings have changed')
    else:
        fresh_directory(out)
        data.save(out / 'input.npz')
        write_json(out / 'request.json', request)
        archive_source(out / 'source.zip')
    lock = out / '.running.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    rows, selected = [], {}
    try:
        for rank in factors:
            candidates = []
            for seed in seeds:
                path = out / f'k{rank:02d}' / f'seed_{seed}.npz'
                failure = path.with_suffix('.failed.json')
                if failure.exists():
                    error = json.loads(failure.read_text())
                    if error['request_hash'] != request_hash:
                        raise ValueError('Failure receipt request mismatch')
                    rows.append(error['summary'])
                    continue
                if path.exists() or path.with_suffix('.json').exists():
                    result = load_model(path, data, request_hash)
                else:
                    try:
                        if engine == 'python-independent':
                            initial = {}
                            warmup_metrics = None
                            if loss == 'huber' and warmup:
                                prior = fit(data.X, data.U, rank, seed, loss='quadratic', max_iter=warmup,
                                    tol=tol, stationarity_tol=stationarity_tol)
                                initial = {'initial_G': prior['G'], 'initial_H': prior['H']}
                                warmup_metrics = prior['metrics']
                            result = fit(data.X, data.U, rank, seed, loss=loss, delta=delta,
                                max_iter=max_iter, tol=tol, stationarity_tol=stationarity_tol, **initial)
                            result['provenance']['engine'] = 'python-independent'
                            result['provenance']['warmup_metrics'] = warmup_metrics
                        elif engine == 'epa-pmf5':
                            from .native_backend import fit_native
                            result = fit_native(data.X, data.U, rank, seed, path.parent/f'seed_{seed}_native',
                                loss=loss, delta=delta, runtime=native_runtime, timeout=native_timeout)
                        else:
                            from .esat_backend import fit_esat
                            result = fit_esat(data.X, data.U, rank, seed, method=esat_method, loss=loss,
                                max_iter=max_iter, tol=tol, stationarity_tol=stationarity_tol)
                        save_model(path, result, data, request_hash)
                        load_model(path, data, request_hash)
                    except Exception as exc:
                        if path.exists():
                            raise  # a corrupted or half-written checkpoint is never treated as reusable
                        path.parent.mkdir(parents=True, exist_ok=True)
                        row = {'factors': rank, 'seed': seed, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
                        write_json(failure, {'request_hash': request_hash, 'summary': row})
                        rows.append(row)
                        continue
                metrics = result['metrics']
                row = {'factors': rank, 'seed': seed, 'status': 'passed' if metrics['converged'] else 'capped',
                    'objective': metrics.get('selection_objective', metrics['objective']),
                    'objective_name': metrics.get('selection_objective_name', 'independent_'+metrics.get('loss','quadratic')),
                    'comparison_objective': metrics['objective'], 'Q_true': metrics['Q_true'], 'iterations': metrics['iterations'],
                    'model': str(path.relative_to(out)), 'weighted_rmse_true': metrics['weighted_rmse_true']}
                rows.append(row)
                candidates.append((row, result))
            if candidates:
                qualified = [v for v in candidates if v[0]['status'] == 'passed']
                winner, model = min(qualified or candidates, key=lambda pair: pair[0]['objective'])
                attained = min(candidates, key=lambda pair: pair[0]['objective'])[0]
                selected[str(rank)] = {**winner, 'selection': 'lowest_converged_objective' if qualified else 'lowest_attained_unconverged',
                                      'lowest_attained_seed': attained['seed'], 'lowest_attained_objective': attained['objective']}
                export_model(out / f'k{rank:02d}' / 'selected', model, data)
                matches = []
                for other_row, other in candidates:
                    ii, jj, cos, _ = profile_match(model['H'], other['H'], 1/np.median(data.U, axis=0))
                    for i, j, c in zip(ii, jj, cos):
                        a, b = model['G'][:, i], other['G'][:, j]
                        r = float(np.corrcoef(a, b)[0, 1]) if np.std(a)>0 and np.std(b)>0 else None
                        matches.append({'seed': other_row['seed'], 'reference_factor': int(i+1), 'matched_factor': int(j+1),
                                        'profile_cosine': float(c), 'contribution_r': r})
                write_table(out / f'k{rank:02d}' / 'seed_matches.csv', matches)
        write_table(out / 'starts.csv', rows)
        counts = {s: sum(r['status']==s for r in rows) for s in ('passed','capped','failed')}
        status = 'passed' if len(selected)==len(factors) and all(v['status']=='passed' for v in selected.values()) else ('capped' if selected else 'failed')
        manifest = {'schema': 'aethmodular-pmf-run/2', 'status': status, 'operation': 'fit', 'engine': engine,
            'dataset_id': str(dataset_id), 'dataset_identity': data.identity, 'compatibility': data.compatibility,
            'samples': len(data.X), 'features': data.X.shape[1], 'configuration': config,
            'request_hash': request_hash, 'counts': counts, 'selected': selected,
            'source_mass_calibrated': False, 'native_EPA_PMF_or_ME2_executed': engine=='epa-pmf5' and any(
                json.loads(p.read_text()).get('native_engine_executed',False) for p in out.rglob('native_receipt.json')),
            'warnings': ['Numerical convergence does not establish source identities or global optimality.',
                'Spectral factors are not calibrated source-specific mass.',
                'Inspect capped/failed starts even when a selected model converged.'],
            'files': {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()
                and p.name not in ('manifest.json', '.running.lock') and 'report' not in p.parts and 'figures' not in p.parts}}
        write_json(out / 'manifest.json', manifest)
        if report:
            from .report import build_report
            build_report(out)
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def selected_model(run, rank):
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    data = load_dataset(run/'input.npz')
    if manifest['dataset_identity'] != data.identity:
        raise ValueError('Run input identity mismatch')
    selected = manifest['selected'][str(rank)]
    return data, load_model(run/selected['model'], data, manifest.get('request_hash')), manifest


def project_case(data, reference, output, rank, *, threshold=3.0, allow_unknown_protocol=False, max_iter=1000):
    reference_data, model, reference_manifest = selected_model(reference, rank)
    if not np.isfinite(threshold) or threshold <= 0 or not isinstance(max_iter,int) or max_iter < 1:
        raise ValueError('Invalid projection controls')
    if not np.array_equal(data.features, reference_data.features):
        raise ValueError('Projection feature identities/order do not match')
    unknown = data.protocol.get('status')=='unknown' or reference_data.protocol.get('status')=='unknown'
    if data.compatibility != reference_data.compatibility or (unknown and not allow_unknown_protocol):
        raise ValueError('Projection protocol mismatch or unknown baseline/units; explicit review is required')
    if model['provenance'].get('signed_contributions_allowed'):
        raise ValueError('This projector constrains G>=0; it cannot silently change an ESAT semi-NMF model')
    if model['config'].get('requested_native_loss') == 'native-robust':
        raise ValueError('Native robust fixed-profile projection is not implemented; do not substitute independent Huber')
    h = model['H'].copy()
    from .solver import array_sha256
    frozen_hash = array_sha256(h)
    g, info = project_contributions(data.X,data.U,h,loss=model['metrics'].get('loss','quadratic'),
        delta=model['config'].get('delta',4.0),max_iter=max_iter,return_info=True)
    if array_sha256(h) != frozen_hash:
        raise RuntimeError('Frozen profile changed')
    result={'G':g,'H':h,'metrics':diagnostics(data.X,data.U,g,h,info['loss'],info['delta']),
        'config':{'loss':info['loss'],'delta':info['delta']},
        'history':np.asarray(info['history'])[:,None], 'provenance':{'operation':'fixed-profile projection'}}
    out=fresh_directory(output)
    data.save(out/'input.npz')
    archive_source(out/'source.zip')
    export_model(out,result,data)
    residual=(data.X-g@h)/data.U
    rms=np.sqrt(np.mean(residual**2,axis=1))
    dates=reference_data.metadata.get('dates')
    flags=[]
    for i,fid in enumerate(data.sample_ids):
        d=data.metadata.get('dates',np.repeat('',len(data.X)))[i]
        flags.append({'sample_id':fid,'rms':float(rms[i]),'residual_review':bool(rms[i]>threshold),
            'outside_reference_date_range':bool(dates is not None and d and (d<min(dates) or d>max(dates)))})
    write_table(out/'review_flags.csv',flags)
    manifest={'schema':'aethmodular-pmf-projection/2','status':'passed' if info['objective_plateau'] else 'capped',
        'operation':'project','samples':len(data.X),'features':data.X.shape[1],'reference_dataset_identity':reference_data.identity,
        'reference_model_sha256':sha256(Path(reference)/reference_manifest['selected'][str(rank)]['model']),
        'frozen_H_sha256':frozen_hash,'profile_unchanged':True,'compatibility':data.compatibility,
        'unknown_protocol_acknowledged':bool(unknown and allow_unknown_protocol), 'threshold':threshold,
        'metrics':result['metrics'],'projection_info':info,'source_mass_calibrated':False,
        'files':{str(p.relative_to(out)):sha256(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out/'manifest.json',manifest)
    from .report import build_projection_report
    build_projection_report(out)
    return manifest
