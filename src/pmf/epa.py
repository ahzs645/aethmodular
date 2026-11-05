"""Native EPA PMF5 handoff, host checks, launch receipt and audited result import.

No undocumented ME-2 command-line options are invented. Launching a GUI is not
successful fitting. Only supplied native exports plus run evidence are imported.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from .io import Dataset, canonical_hash, fresh_directory, from_csv, load_dataset, read_matrix, sha256, write_json, write_matrix, write_table
from .runner import export_model, save_model
from .solver import diagnostics

EPA_PAGE = 'https://www.epa.gov/air-research/positive-matrix-factorization-model-environmental-data-analyses'


def runtime_probe(output=None, executable=None):
    exe = Path(executable).expanduser().resolve() if executable else None
    host_windows = os.name == 'nt' and platform.system() == 'Windows'
    found = bool(exe and exe.is_file())
    result = {'schema': 'aethmodular-epa-probe/1', 'operation': 'runtime_probe', 'status': 'available_unverified' if host_windows and found else 'blocked',
        'platform': platform.system(), 'platform_release': platform.release(), 'python': platform.python_version(),
        'windows_host': host_windows, 'executable_supplied': bool(executable), 'executable_found': found,
        'executable_sha256': sha256(exe) if found else None,
        'wine_found': bool(shutil.which('wine') or shutil.which('wine64')),
        'native_EPA_PMF_or_ME2_executed': False,
        'reason': 'Windows installation supplied; actual import/fit still needs verification' if host_windows and found else
            'Native Windows host and existing EPA PMF5 executable are required; no substitution was performed',
        'official_requirements_source': EPA_PAGE,
        'notes': 'EPA documents Windows 7–10. Other Windows versions or Wine are not validated by this adapter.'}
    if output:
        out = fresh_directory(output)
        write_json(out / 'manifest.json', result)
    return result


def export_epa(data, output, *, dataset_id, factors=(5,), seeds=(1,10,100), categories=None):
    if any(int(k)!=k or k<1 or k>min(data.X.shape) for k in factors) or not factors or not seeds:
        raise ValueError('Invalid EPA factor counts or seeds')
    if any(not isinstance(s,int) or isinstance(s,bool) or s<0 for s in seeds):
        raise ValueError('Invalid EPA seeds')
    categories = categories or {str(f): 'Strong' for f in data.features}
    if set(categories) != set(data.features) or any(v not in ('Strong','Weak','Bad') for v in categories.values()):
        raise ValueError('Categories must explicitly cover all features with Strong, Weak or Bad')
    if all(v=='Bad' for v in categories.values()):
        raise ValueError('All features classified Bad')
    out = fresh_directory(output)
    data.save(out/'input.npz')
    write_matrix(out/'concentrations.csv',data.sample_ids,data.features,data.X)
    write_matrix(out/'uncertainties.csv',data.sample_ids,data.features,data.U)
    back = from_csv(out/'concentrations.csv',out/'uncertainties.csv')
    if not np.array_equal(back.X,data.X) or not np.array_equal(back.U,data.U):
        raise RuntimeError('EPA export changed numerical values')
    write_table(out/'feature_categories.csv',[{'feature':f,'category':categories[f]} for f in data.features])
    write_table(out/'sample_metadata.csv',[{'sample_id':fid, **{k:str(v[i]) for k,v in data.metadata.items()}}
        for i,fid in enumerate(data.sample_ids)])
    instructions = '''# EPA PMF5 operator handoff

1. Use an installed EPA PMF5 on a compatible Windows host; record its actual
   executable hash/build and the project/settings file.
2. Import concentrations.csv and uncertainties.csv with first-column sample IDs.
   Verify sample/feature counts, feature order, units, signed observations and
   strict positive uncertainties. No imputation, clipping or channel deletion
   has been performed by this exporter.
3. Apply feature_categories.csv deliberately. Record any weak-variable inflation,
   bad-variable omission, added modeling uncertainty, exclusions and other changes.
4. Start with one five-factor run as a software check, then the requested ranks
   and seeds. Retain failed/capped runs and native logs. Do not call this export
   a completed model. FPEAK/DISP/BS-DISP are native operations, not replaced here.
5. Preserve original native outputs/logs. Convert the selected contribution table
   to a labeled sample-by-factor CSV, and profiles to a labeled factor-by-feature
   CSV. Do not silently transpose, reorder, normalize, or copy old source labels.
6. Supply the normalized tables to epa-import together with native evidence.json
   and the exact effective X/U tables if settings changed the supplied weights or
   feature population. No formula for unknown native scaling is guessed.

Required evidence.json keys: engine='EPA PMF5', engine_version, native_execution
(true), convergence_status ('converged' or 'capped'), settings (object), reported_q_true,
raw_files (relative paths to original logs/exports/project files), and optional
native_executable_sha256. These are operator assertions, checked for file presence
and mathematical consistency; importing them cannot independently prove origin.
Record settings.robust as true for robust native fitting (or settings.loss as
'native-robust'); false/'quadratic' identifies ordinary fitting. For compatibility,
omitting both retains the ordinary convention. Conflicting conventions are rejected.
Q(true) remains a quadratic diagnostic, not the native robust optimization loss.
'''
    (out/'NATIVE_RUN.md').write_text(instructions,encoding='utf-8')
    manifest = {'schema':'aethmodular-epa-handoff/2','operation':'epa-export','status':'not_run',
        'engine':'EPA PMF5/ME-2','dataset_id':dataset_id,'dataset_identity':data.identity,
        'samples':len(data.X),'features':data.X.shape[1],'factors':list(factors),'seeds':list(seeds),
        'categories':categories,'numerical_readback_exact':True,'native_EPA_PMF_or_ME2_executed':False,
        'warnings':['Export is not a native import or run.', 'Changing native feature weights requires exact effective inputs at result import.'],
        'files':{str(p.relative_to(out)):sha256(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out/'manifest.json',manifest)
    return manifest


def launch_native(executable, output, *, arguments=(), timeout=3600):
    """Launch only an explicitly supplied Windows installation; never claim a fit."""
    if not np.isfinite(timeout) or timeout<=0:
        raise ValueError('Timeout must be positive')
    probe = runtime_probe(executable=executable)
    out = fresh_directory(output)
    if probe['status']=='blocked':
        write_json(out/'manifest.json',probe)
        return probe
    exe=Path(executable).expanduser().resolve()
    started=time.time()
    try:
        run=subprocess.run([str(exe),*map(str,arguments)],cwd=out,capture_output=True,text=True,
            errors='replace',check=False,timeout=timeout,shell=False)
        (out/'stdout.log').write_text(run.stdout,encoding='utf-8')
        (out/'stderr.log').write_text(run.stderr,encoding='utf-8')
        result={**probe,'operation':'native_launch','status':'process_exited_unverified' if run.returncode==0 else 'failed',
            'returncode':run.returncode,'elapsed_seconds':time.time()-started,
            'native_application_launched':True,'native_EPA_PMF_or_ME2_executed':False,
            'reason':'Process exit is not proof of a completed or converged ME-2 calculation; import verified exports.'}
    except subprocess.TimeoutExpired as exc:
        result={**probe,'operation':'native_launch','status':'capped','native_application_launched':True,
                'reason':f'Native process exceeded {timeout} seconds; no fit success inferred'}
        for name,value in [('stdout.log',exc.stdout),('stderr.log',exc.stderr)]:
            (out/name).write_text(value.decode(errors='replace') if isinstance(value,bytes) else (value or ''),encoding='utf-8')
    write_json(out/'manifest.json',result)
    return result


def import_epa(handoff, contributions, profiles, evidence, output, *, effective_x=None, effective_u=None,
               q_rtol=1e-4, q_atol=1e-6):
    handoff=Path(handoff)
    evpath=Path(evidence)
    meta=json.loads((handoff/'manifest.json').read_text())
    if meta.get('schema')!='aethmodular-epa-handoff/2':
        raise ValueError('Expected a verified EPA handoff directory')
    for name,digest in meta['files'].items():
        if sha256(handoff/name)!=digest:
            raise ValueError('EPA handoff changed after export')
    original=load_dataset(handoff/'input.npz')
    ev=json.loads(evpath.read_text())
    if ev.get('engine')!='EPA PMF5' or ev.get('native_execution') is not True or not ev.get('engine_version'):
        raise ValueError('Explicit native engine/version/execution evidence is required')
    if ev.get('convergence_status') not in ('converged','capped') or not isinstance(ev.get('settings'),dict):
        raise ValueError('Native convergence and settings must be explicit')
    # Q(true) is a reconstruction diagnostic even when the native fit was robust.
    # Normalize the operator's convention for downstream projection guards while
    # preserving the original evidence/settings, without claiming native Q_main.
    settings = ev['settings']
    conventions = []
    if 'robust' in settings:
        if not isinstance(settings['robust'], bool):
            raise ValueError('Native settings.robust must be a boolean')
        conventions.append('native-robust' if settings['robust'] else 'quadratic')
    for key in ('requested_native_loss', 'loss'):
        if key in settings:
            if settings[key] not in ('quadratic', 'native-robust'):
                raise ValueError('Native loss must be quadratic or native-robust')
            conventions.append(settings[key])
    if len(set(conventions)) > 1:
        raise ValueError('Conflicting native loss settings')
    native_loss = conventions[0] if conventions else 'quadratic'
    model_config = {**settings, 'loss': 'quadratic', 'requested_native_loss': native_loss}
    if not isinstance(ev.get('raw_files'),list) or not ev['raw_files']:
        raise ValueError('Original native logs/exports are required')
    raw_hashes={}
    for name in ev['raw_files']:
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or not (evpath.parent/p).is_file() or not (evpath.parent/p).resolve().is_relative_to(evpath.parent.resolve()):
            raise ValueError('Evidence raw files must exist and be relative to evidence.json')
        raw_hashes[str(p)]=sha256(evpath.parent/p)
    if (effective_x is None)!=(effective_u is None):
        raise ValueError('Supply both effective X and effective U')
    modified=any(v!='Strong' for v in meta['categories'].values()) or bool(ev['settings'].get('inputs_modified',False))
    if modified and effective_x is None:
        raise ValueError('Changed native weighting/exclusions require the exact effective X/U pair')
    data=from_csv(effective_x,effective_u,protocol=original.protocol) if effective_x else original
    if not set(data.sample_ids)<=set(original.sample_ids) or not set(data.features)<=set(original.features):
        raise ValueError('Native effective input has unknown samples/features')
    indices = [int(np.flatnonzero(original.sample_ids == sid)[0]) for sid in data.sample_ids]
    feature_indices = [int(np.flatnonzero(original.features == fid)[0]) for fid in data.features]
    original_subset = original.X[np.ix_(indices, feature_indices)]
    changed_cells = int(np.sum(data.X != original_subset))
    if changed_cells and ev['settings'].get('inputs_modified') is not True:
        raise ValueError('Effective native observations changed without explicit settings attestation')
    if effective_x:
        data = Dataset(data.X, data.U, data.sample_ids, data.features, data.protocol,
            {k: v[indices] for k, v in original.metadata.items()}, data.sources)
    gids, factors, g=read_matrix(contributions)
    hfactors, features, h=read_matrix(profiles)
    if not np.array_equal(gids,data.sample_ids) or not np.array_equal(features,data.features) or not np.array_equal(factors,hfactors):
        raise ValueError('Native normalized output identities/order mismatch')
    if np.any(g<0) or np.any(h<0):
        raise ValueError('EPA profiles/contributions must be nonnegative')
    metrics=diagnostics(data.X,data.U,g,h)
    reported=float(ev['reported_q_true'])
    if not np.isfinite(reported) or reported<0 or not np.isclose(metrics['Q_true'],reported,rtol=q_rtol,atol=q_atol):
        raise ValueError('Reported native Q(true) disagrees with independently reconstructed residuals')
    metrics.update(loss='quadratic',converged=ev['convergence_status']=='converged',iterations=ev['settings'].get('iterations'),
                   seed=ev['settings'].get('seed'),n_factors=g.shape[1],stop_reason=ev['convergence_status'])
    result={'G':g,'H':h,'history':np.empty((0,2)),'metrics':metrics,'config':model_config,
        'provenance':{'engine':'EPA PMF5/ME-2','engine_version':ev['engine_version'],
            'operator_settings':settings,
            'native_execution_asserted_by_operator':True,'native_execution_independently_observed':False,
            'raw_file_hashes':raw_hashes,'evidence_sha256':sha256(evpath),
            'source_mass_calibrated':False,'reported_Q_true':reported,'Q_true_independently_recomputed':True,
            'factor_labels':factors.tolist(), 'changed_observation_cells': changed_cells}}
    out=fresh_directory(output)
    data.save(out/'input.npz')
    request={'engine':'epa-pmf5','dataset_identity':data.identity,'evidence_sha256':sha256(evpath)}
    rh=canonical_hash(request)
    write_json(out/'request.json',request)
    save_model(out/'native.npz',result,data,rh)
    export_model(out/'selected',result,data)
    evidence_dir=out/'native_evidence'
    evidence_dir.mkdir()
    shutil.copy2(evpath,evidence_dir/'evidence.json')
    for name in raw_hashes:
        target=evidence_dir/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(evpath.parent/name,target)
    manifest={'schema':'aethmodular-pmf-run/2','operation':'epa-import','status':'passed' if metrics['converged'] else 'capped',
        'engine':'epa-pmf5','dataset_id':meta['dataset_id'],'dataset_identity':data.identity,'request_hash':rh,
        'samples':len(data.X),'features':data.X.shape[1],'configuration':ev['settings'],
        'counts':{'passed':int(metrics['converged']),'capped':int(not metrics['converged']),'failed':0},
        'selected':{str(g.shape[1]):{'model':'native.npz','status':'passed' if metrics['converged'] else 'capped',
            'objective':metrics['objective'],'Q_true':metrics['Q_true'],'seed':metrics['seed']}},
        'warnings':['Native origin/settings are operator-attested, not independently witnessed by the importer.',
                    'Only algebraic, identity and file integrity checks are verified here.'],
        'native_execution_independently_observed':False,'source_mass_calibrated':False,
        'files':{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out/'manifest.json',manifest)
    from .report import build_report
    build_report(out)
    return manifest
