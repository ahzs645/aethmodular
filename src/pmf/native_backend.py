"""Bridge to the unchanged EPA ME-2 driver originally merged through PR #13.

The proprietary runtime is never bundled. This module verifies the upstream
Python driver's Git blob before execution and uses its verified native results.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from .io import sha256
from .solver import diagnostics

DRIVER_PATH = 'research/ftir_hips_chem/scripts/python_pmf/me2.py'
DRIVER_GIT_BLOB = 'da2ae8291ed0d592767f51ac37e85c72516334ab'
DRIVER_COMMIT = '60ee83833571d6adad0a0c1f8e872cdbd63490de'


def load_driver():
    path = Path(__file__).resolve().parents[2] / DRIVER_PATH
    if not path.is_file():
        raise RuntimeError('Pinned EPA driver missing; use a complete aethmodular source checkout')
    raw = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if blob != DRIVER_GIT_BLOB:
        raise ValueError('EPA Python driver changed; review and update the explicit source pin')
    spec = importlib.util.spec_from_file_location('_aeth_pmf_native_me2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def result_from_native(x, u, g, h, receipt, *, loss, delta, driver_hash):
    """Keep native optimization Q separate from common quadratic diagnostics."""
    if loss not in ('quadratic', 'native-robust'):
        raise ValueError('EPA fitting accepts quadratic or native-robust, not independent Huber')
    if not receipt.get('native_engine_executed') or receipt.get('script_modified'):
        raise ValueError('A verified unchanged native engine execution is required')
    if receipt.get('status') not in ('passed', 'capped'):
        raise ValueError('Native execution did not produce a valid model')
    if bool(receipt.get('robust')) != (loss == 'native-robust'):
        raise ValueError('Native robust convention differs from the requested setting')
    g, h = np.asarray(g, dtype=float), np.asarray(h, dtype=float)
    if g.ndim != 2 or h.ndim != 2 or g.shape[0] != x.shape[0] or h.shape[1] != x.shape[1] or g.shape[1] != h.shape[0]:
        raise ValueError('Native factor dimensions disagree with the input')
    if not np.isfinite(g).all() or not np.isfinite(h).all() or np.any(g < 0) or np.any(h < 0):
        raise ValueError('Native nonnegative factors are invalid')
    metrics = diagnostics(x, u, g, h, 'quadratic', delta)
    native = receipt['metrics']
    if not np.isclose(metrics['Q_true'], native['Q_true'], rtol=1e-10, atol=1e-8):
        raise ValueError('Native receipt reconstruction Q differs from the actual factors')
    # No claim that native_Q_main equals the independent Huber loss.
    selection = float(native['native_Q_main'])
    if not np.isfinite(selection):
        raise ValueError('Native optimization objective is nonfinite')
    metrics.update(converged=receipt['status']=='passed' and bool(receipt.get('native_convergence_observed')),
        iterations=int(native['native_iterations']), selection_objective=selection,
        selection_objective_name='native_Q_main', native_metrics=native,
        relative_projected_gradient_conditioned=None)
    return {'G': g, 'H': h, 'history': np.asarray([[native['native_iterations'], selection]], dtype=float),
        'metrics': metrics, 'config': {'loss':'quadratic', 'requested_native_loss':loss, 'delta':delta,
            'factors':g.shape[1], 'seed':int(receipt['seed'])},
        'provenance': {'engine':'epa-pmf5', 'native_engine_executed':True,
            'version':receipt['version'], 'signed_contributions_allowed':False,
            'native_driver_git_blob':DRIVER_GIT_BLOB, 'native_driver_sha256':driver_hash,
            'runtime_sha256':receipt['runtime_sha256'], 'script_modified':False,
            'objective_convention':'Selection uses native_Q_main; common objective is independently recomputed Q_true.',
            'native_receipt':receipt}}


def fit_native(x, u, rank, seed, output, *, loss='quadratic', delta=4., runtime=None, timeout=600.):
    if os.name != 'nt':
        raise RuntimeError('Native EPA PMF5/ME-2 execution requires Windows; no substitute solver was run')
    if seed < 1:
        raise ValueError('EPA native seeds must be positive integers')
    runtime = runtime or os.environ.get('EPA_PMF_RUNTIME')
    if not runtime:
        raise RuntimeError('Set EPA_PMF_RUNTIME to the unmodified extracted official EPA PMF5 distribution')
    driver, path = load_driver()
    g, h, receipt = driver.run_one(x, u, runtime, output, factors=rank, seed=seed,
        robust=loss=='native-robust', threshold=delta, timeout=timeout)
    return result_from_native(x,u,g,h,receipt,loss=loss,delta=delta,driver_hash=sha256(path))


def import_verified_native_case(data, case, *, loss):
    """Verify a native driver result bundle, including exact X/U and every hash.

This is a native result import, not a claim the engine executed on this machine.
The original execution receipt and logs remain the provenance of execution.
"""
    case = Path(case)
    receipt = json.loads((case/'native_receipt.json').read_text())
    for name, digest in receipt['output_sha256'].items():
        path = (case/name).resolve()
        if path.parent != case.resolve() or sha256(path) != digest:
            raise ValueError('Native artifact checksum/path mismatch: ' + name)
    values = np.fromiter(
        (float(v) for line in (case/'pmfdata.txt').open() for v in line.split()), dtype=float)
    if not np.array_equal(values, np.concatenate([data.X.ravel(), data.U.ravel()])):
        raise ValueError('Native input values differ from this dataset')
    stdout=(case/'engine.stdout').read_text(errors='replace')
    status=(case/'pmfstatus.txt').read_text(errors='replace')
    if receipt.get('returncode') != 0 or 'PMF=ready' not in status or 'ERROR in using ME-2' in stdout:
        raise ValueError('Native output lacks successful engine completion')
    if receipt.get('status')=='passed' and 'Iteration finished, convergence achieved.' not in stdout:
        raise ValueError('Native convergence claim lacks its log evidence')
    with np.load(case/'native_factors.npz', allow_pickle=False) as z:
        g,h=z['G'],z['H']
    report=np.asarray([float(v.replace('D','E')) for v in (case/'PMFreport.txt').read_text().split()])
    if len(report)!=10 or not np.isfinite(report).all() or report[0]!=1:
        raise ValueError('Invalid native PMF report')
    residual=(data.X-g@h)/data.U
    prediction_error=((5e-5*abs(g))@abs(h)+abs(g)@(5e-5*abs(h))+(5e-5*abs(g))@(5e-5*abs(h)))/data.U
    bound=float(np.sum(2*abs(residual)*prediction_error+prediction_error**2))+1e-5*max(1.,abs(report[5]))
    if abs(float(np.sum(residual**2))-report[5])>bound:
        raise ValueError('Independently recomputed native Q exceeds its text-rounding bound')
    if receipt['metrics']['native_Q_main'] != report[3] or receipt['metrics']['native_Q_true'] != report[5]:
        raise ValueError('Native Q receipt disagrees with PMFreport.txt')
    return result_from_native(data.X,data.U,g,h,receipt,loss=loss,delta=receipt['threshold'],driver_hash='recorded-in-source-commit-'+DRIVER_COMMIT)


def ingest_case(data, case, output, *, dataset_id, loss='quadratic'):
    """Bring verified native output back into the common report/checkpoint format."""
    import shutil
    from .io import archive_source, canonical_hash, fresh_directory, write_json
    from .runner import save_model, export_model
    case = Path(case)
    result = import_verified_native_case(data,case,loss=loss)
    out = fresh_directory(output)
    data.save(out/'input.npz')
    archive_source(out/'source.zip')
    evidence = out/'native_evidence'
    evidence.mkdir()
    receipt = json.loads((case/'native_receipt.json').read_text())
    allowed = {'iniparams.txt','pmfdata.txt','engine.stdout','engine.stderr','pmfstatus.txt',
        'PMFreport.txt','native_result.dat','native_result.txt','native_result.rsd','ME2.LOG','native_factors.npz'}
    for name in receipt['output_sha256']:
        if name not in allowed:
            raise ValueError('Unexpected native artifact; proprietary runtime files must not be copied')
        shutil.copy2(case/name,evidence/name)
    shutil.copy2(case/'native_receipt.json',evidence/'native_receipt.json')
    request = {'operation':'verified-native-import','dataset_identity':data.identity,
        'native_receipt_sha256':sha256(case/'native_receipt.json'),'dataset_id':dataset_id}
    write_json(out/'request.json',request)
    request_hash=canonical_hash(request)
    rank=result['H'].shape[0]
    seed=int(receipt['seed'])
    model=out/f'k{rank:02d}'/f'seed_{seed}.npz'
    save_model(model,result,data,request_hash)
    export_model(model.parent/'selected',result,data)
    row={'factors':rank,'seed':seed,'status':receipt['status'],'objective':result['metrics']['selection_objective'],
        'objective_name':'native_Q_main','comparison_objective':result['metrics']['objective'],
        'Q_true':result['metrics']['Q_true'],'iterations':result['metrics']['iterations'],
        'model':str(model.relative_to(out)),'selection':'only_imported_native_result'}
    from .io import write_table
    write_table(out/'starts.csv',[row])
    m={'schema':'aethmodular-pmf-run/2','operation':'native-result-import','status':receipt['status'],
        'engine':'epa-pmf5','dataset_id':dataset_id,'dataset_identity':data.identity,'compatibility':data.compatibility,
        'samples':len(data.X),'features':data.X.shape[1],'configuration':{'factors':[rank],'seeds':[seed],'loss':loss},
        'request_hash':request_hash,'selected':{str(rank):row},'counts':{receipt['status']:1},
        'source_mass_calibrated':False,'native_EPA_PMF_or_ME2_executed':True,
        'execution_origin':'verified imported native driver bundle; not re-executed by this import',
        'warnings':['Native public/synthetic validation does not establish validity on private FTIR samples.',
            'Native optimization Q_main is separate from independently recomputed Q_true.',
            'This imported model is not an exhaustive multi-start scientific source solution.'],
        'files':{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}}
    write_json(out/'manifest.json',m)
    from .report import build_report
    build_report(out)
    return m
