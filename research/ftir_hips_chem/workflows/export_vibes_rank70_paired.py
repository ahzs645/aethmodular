"""Export current paired residuals and exact prediction decomposition.

Retains the original physical-filter/site split and inspection case identities.
The same inspected test set cannot provide independent validation.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / 'scripts'))
from vibes_current import CURRENT, MIGRATION, sha, study_inputs  # noqa: E402
from vibes_error_audit import audit_groups, pair_predictions  # noqa: E402


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def main():
    inputs = study_inputs()
    cases, arrays, wn = inputs['cases'], inputs['arrays'], inputs['wn']
    groups, definitions, paired, influence = audit_groups(pair_predictions(inputs['saved'], cases), cases, repeats=10000)
    spec = importlib.util.spec_from_file_location('decomposition', AREA / 'workflows/trace_vibes_loading_predictions.py')
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    masks = [(wn >= low) & (wn < high) for low, high in trace.BANDS]
    if not np.equal(np.sum(masks, axis=0), 1).all():
        raise ValueError('Decomposition bands do not partition the model grid')
    positions = pd.Series(np.arange(len(cases)), index=cases.sample_id)
    contributions, checks = [], []
    sources = dict(inputs['hashes'])
    for cohort, frame in paired.groupby('cohort'):
        ix = positions.loc[frame.sample_id].to_numpy()
        xs, beta, constant, rounding = {}, {}, {}, {}
        for method in arrays:
            path = CURRENT / f'benchmark/pls_{cohort}_{method}.npz'
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
            with np.load(path) as z:
                beta[method] = z['coefficient'].ravel()
                mean = z['x_mean']
                intercept = float(z['intercept'].item())
            xs[method] = np.asarray(arrays[method][ix], float)
            constant[method] = float(intercept - mean @ beta[method])
            # sklearn centers in the dtype of each saved corrected array.
            centered = np.array(arrays[method][ix], copy=True)
            centered -= mean
            prediction = centered @ beta[method] + intercept
            np.testing.assert_allclose(prediction, frame[method], rtol=0, atol=1e-8)
            rounding[method] = prediction - (xs[method] @ beta[method] + constant[method])
            checks.append(dict(cohort=cohort, method=method, n=len(frame), max_error=float(np.max(abs(prediction-frame[method])))))
        spectral, coefficient, offset = trace.decompose(xs['AIRSpec'], xs['VIBES'], beta['AIRSpec'], beta['VIBES'], constant['AIRSpec'], constant['VIBES'])
        out = frame[['sample_id', 'cohort']].copy()
        out['prediction_delta'] = frame.VIBES-frame.AIRSpec
        out['intercept_and_centering_delta'] = offset
        out['centering_precision_delta'] = rounding['VIBES']-rounding['AIRSpec']
        for (low, high), mask in zip(trace.BANDS, masks):
            out[f'spectra_{low}_{high-1}'] = spectral[:, mask].sum(axis=1)
            out[f'coefficients_{low}_{high-1}'] = coefficient[:, mask].sum(axis=1)
        terms = out.filter(regex='^(spectra_|coefficients_)').sum(axis=1) + offset + out.centering_precision_delta
        np.testing.assert_allclose(terms, out.prediction_delta, atol=1e-8, rtol=0)
        contributions.append(out)
    old = ROOT / 'gallery/app/public/data/baseline_comparison.json'
    sources[old.relative_to(ROOT).as_posix()] = sha(old)
    old_data = json.loads(old.read_text())
    # Historical case selection identities are retained; every prediction and
    # corrected trace displayed for those cases is replaced with current data.
    spectra = []
    full = paired.loc[paired.cohort.eq('full_pool')].set_index('sample_id')
    for row in old_data['spectra']:
        sid = row['sample_id']
        p = full.loc[sid]
        ix = positions.loc[sid]
        spectra.append({**row, 'direction':'Selected in the archived benchmark',
                        'AIRSpec_prediction':float(p.AIRSpec), 'VIBES_prediction':float(p.VIBES),
                        'airspec': arrays['AIRSpec'][ix].round(9).tolist(),
                        'vibes': arrays['VIBES'][ix].round(9).tolist()})
    blank = pd.read_csv(CURRENT / 'blank_metrics.csv').query("method in ['AIRSpec', 'VIBES']").rename(columns={'rms_absorbance':'rms_from_zero'})
    payload = dict(schema_version=1, evidence_date='2026-10-02', background_rank=70,
                   model_window=[4000,1425], correction_window=[4000,500],
                   run={**inputs['manifest'], 'background_rank':70, 'n_failed':0, 'n_corrected':12808},
                   scores=records(inputs['scores']), pairs=records(paired), groups=records(groups),
                   contributions=records(pd.concat(contributions)), blank_details=records(blank),
                   spectra=spectra, wn=wn.tolist(), historical_masks=old_data['historical_masks'],
                   exclusions=old_data['exclusions'], source_sha256=sources)
    dest = ROOT / 'gallery/app/public/data'
    (dest / 'baseline_paired_r70.json').write_text(json.dumps(payload, allow_nan=False, separators=(',', ':')))
    paired.to_csv(dest / 'baseline_paired_r70_predictions.csv', index=False)
    out = MIGRATION / 'paired'
    out.mkdir(parents=True, exist_ok=True)
    groups.to_csv(out / 'groups.csv', index=False)
    definitions.to_csv(out / 'loading_definitions.csv', index=False)
    influence.to_csv(out / 'site_influence.csv', index=False)
    (out / 'checks.json').write_text(json.dumps(dict(background_rank=70, model_checks=checks, source_sha256=sources), indent=2))
    print(f'Exported rank-70 residuals, loading groups and closed decompositions for {len(paired)} paired filters')


if __name__ == '__main__':
    main()
