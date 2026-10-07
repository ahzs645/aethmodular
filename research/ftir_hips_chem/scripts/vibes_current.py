"""Validated completed VIBES inputs for current gallery calculations.

The recovered array is a derived artifact; original Colab exports are read-only.
Every consumer records the same hashes and aligns by sample identity.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
TABLES = AREA / 'output/tables'
CURRENT = TABLES / 'vibes_fullrange/interpretation_20261002'
EARLIER = TABLES / 'vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09'
MIGRATION = TABLES / 'vibes_fullrange/gallery_rank70_20261002'
METHODS = ('AIRSpec', 'VIBES-full-r70', 'VIBES-full-cut-r70')
MODEL_NAMES = {'AIRSpec': 'AIRSpec', 'VIBES-full-r70': 'uncapped_full', 'VIBES-full-cut-r70': 'VIBES'}
LABELS = {'AIRSpec': 'Spline baseline', 'VIBES-full-r70': 'VIBES baseline (4000–500, completed rank 70)',
          'VIBES-full-cut-r70': 'Sensitivity: VIBES fit 4000–500, predict 4000–1425 (completed rank 70)'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_completed():
    summary = json.loads((CURRENT / 'summary.json').read_text())
    sources = dict(summary['source_hashes'])
    for key, digest in sources.items():
        if sha(ROOT / key) != digest:
            raise ValueError(f'Completed VIBES source changed: {key}')
    path = CURRENT / 'corrected_VIBES_full_recovered.npy'
    digest = sha(path)
    if digest != summary['derived_array_sha256']:
        raise ValueError('Recovered VIBES array changed')
    sources[path.relative_to(ROOT).as_posix()] = digest
    cases = pd.read_csv(CURRENT / 'cases.csv')
    audit = json.loads((TABLES / 'vibes_subgroup_audit/audit_manifest.json').read_text())
    ledger_path = EARLIER / 'case_audit.csv'
    if sha(ledger_path) != audit['source_hashes'][ledger_path.relative_to(ROOT).as_posix()]:
        raise ValueError('Frozen case eligibility and split ledger changed')
    prior = pd.read_csv(ledger_path)
    if cases.sample_id.tolist() != prior.sample_id.tolist() or not cases.sample_id.is_unique:
        raise ValueError('Completed case identities do not match the frozen split')
    # The original full-range grid is in the signature-verified source list.
    grid_path = next(ROOT / key for key in sources if key.endswith('/wn.npy') and 'rank85_resume' in key)
    wn = np.load(grid_path)
    shared = wn >= 1425
    if not np.array_equal(wn[shared], np.load(EARLIER / 'wn.npy')):
        raise ValueError('Shared channels differ from the spline grid')
    full = np.load(path, mmap_mode='r')
    if full.shape != (len(cases), len(wn)) or not np.isfinite(full).all():
        raise ValueError('Completed full-range spectra are not finite and aligned')
    arrays = {'AIRSpec': np.load(EARLIER / 'corrected_AIRSpec.npy', mmap_mode='r'),
              'VIBES-full-r70': full, 'VIBES-full-cut-r70': full[:, shared]}
    grids = {'AIRSpec': wn[shared], 'VIBES-full-r70': wn, 'VIBES-full-cut-r70': wn[shared]}
    return dict(cases=prior, arrays=arrays, grids=grids, sources=sources, summary=summary)


def load_context():
    import analog_bias as ab
    data = load_completed()
    # Reuse the exact local reference join from the frozen meeting run. This
    # avoids waiting for Drive to hydrate the 220-MB TOR source on every worker.
    reference = TABLES / 'meeting_followup_20260917/improve_references.csv'
    refs = pd.read_csv(reference, parse_dates=['date'])
    tor_columns = ['Site', 'date'] + [c for c in refs if c.startswith('tor_') and c != 'tor_oc_ec']
    tor_loader, hips_loader = ab.load_tor_fractions, ab.load_improve_hips
    ab.load_tor_fractions = lambda path=None: refs[tor_columns].drop_duplicates(['Site', 'date'])
    ab.load_improve_hips = lambda path=None: refs[['filter_id', 'fabs_Mm1']].drop_duplicates('filter_id')
    try:
        ctx = ab.load_context(fullrange=False)
    finally:
        ab.load_tor_fractions, ab.load_improve_hips = tor_loader, hips_loader
    if ctx.lib.filter_id.tolist() != refs.filter_id.tolist():
        raise ValueError('Saved reference population differs from the completed library')
    data['sources'][reference.relative_to(ROOT).as_posix()] = sha(reference)
    lib_rows, addis_rows = ctx.lib.row.to_numpy(), ctx.addis.row.to_numpy()
    ctx.methods = METHODS
    ctx.X = {m: np.asarray(data['arrays'][m][lib_rows], float) for m in METHODS}
    ctx.XA = {m: np.asarray(data['arrays'][m][addis_rows], float) for m in METHODS}
    ctx.wn_by = data['grids']
    ctx.notes.append('Completed rank-70 full-range correction; original source split and calendars preserved')
    return ctx, data


def receipt(output, sources):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'rank70_inputs.json'
    value = {'background_rank': 70, 'methods': list(METHODS), 'source_sha256': sources,
             'loader_sha256': sha(Path(__file__))}
    if path.exists():
        prior = json.loads(path.read_text())
        if any(prior.get(k) != value[k] for k in ('background_rank', 'methods', 'source_sha256')):
            raise ValueError(f'Resume inputs changed: {path.relative_to(ROOT)}')
    path.write_text(json.dumps(value, indent=2) + '\n')


def study_inputs():
    """The existing paired follow-up design, using completed shared channels."""
    d = load_completed()
    cases = d['cases']
    score = pd.read_csv(CURRENT / 'benchmark/calibration_scores.csv')
    saved = pd.read_csv(CURRENT / 'benchmark/heldout_predictions.csv')
    cal = cases.kind.eq('calibration')
    train = np.flatnonzero(cal & cases.split.eq('train'))
    test = np.flatnonzero(cal & cases.split.eq('test'))
    targets = np.flatnonzero(cases.kind.eq('target'))
    if len(train) != 10066 or len(test) != 2327 or set(cases.Site.iloc[train]) & set(cases.Site.iloc[test]):
        raise ValueError('Completed follow-up split differs from the original design')
    signature_path = next(ROOT / k for k in d['sources'] if k.endswith('/RUN_MANIFEST.json') and 'rank85_resume' in k)
    return dict(source=CURRENT, manifest=json.loads(signature_path.read_text()), hashes=d['sources'],
                cases=cases, arrays={'AIRSpec': d['arrays']['AIRSpec'], 'VIBES': d['arrays']['VIBES-full-cut-r70']},
                wn=d['grids']['AIRSpec'], train=train, test=test, targets=targets,
                scores=score.loc[score.method.isin(['AIRSpec', 'VIBES'])],
                saved=saved.loc[saved.method.isin(['AIRSpec', 'VIBES'])])


def prepare_shared_run():
    """Portable paired-model inputs for diagnostics and external application."""
    import shutil
    data = study_inputs()
    out = MIGRATION / 'shared_run'
    receipt(out, data['hashes'])
    data['cases'].to_csv(out / 'cases.csv', index=False)
    data['cases'].to_csv(out / 'case_audit.csv', index=False)
    np.save(out / 'wn.npy', data['wn'])
    for method, array in data['arrays'].items():
        if not (out / f'corrected_{method}.npy').exists():
            np.save(out / f'corrected_{method}.npy', array)
        shutil.copy2(CURRENT / f'benchmark/pls_full_pool_{method}.npz', out / f'pls_full_pool_{method}.npz')
    data['saved'].to_csv(out / 'heldout_predictions.csv', index=False)
    data['scores'].to_csv(out / 'calibration_scores.csv', index=False)
    predictions = pd.read_csv(CURRENT / 'benchmark/addis_predictions.csv')
    predictions.loc[predictions.method.isin(['AIRSpec', 'VIBES'])].to_csv(out / 'addis_predictions.csv', index=False)
    manifest = {**data['manifest'], 'n_failed': 0, 'n_corrected': len(data['cases']), 'background_rank': 70,
                'n_recovered': 2, 'model_window': [4000, 1425], 'correction_window': [4000, 500],
                'source_sha256': data['hashes'], 'derived': True}
    (out / 'RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return out
