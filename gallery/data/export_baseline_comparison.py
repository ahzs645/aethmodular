"""Export the frozen AIRSpec/VIBES comparison and historical mask provenance.

Run with uv run --locked --no-sync python gallery/data/export_baseline_comparison.py.
No fitting, new exclusions, or replacement of existing gallery datasets.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / 'research/ftir_hips_chem/output/tables'
RUN = T / 'vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09'
STAGE = T / 'vibes_colab_bundle/stage'
AUDIT = T / 'vibes_subgroup_audit'
OLD = T / 'ann_weekly_20260910'
OUT = ROOT / 'gallery/app/public/data'
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    manifest = json.loads((STAGE/'BUNDLE_MANIFEST.json').read_text())
    run = json.loads((RUN/'RUN_MANIFEST.json').read_text())
    expected = hashlib.sha256(json.dumps(manifest['files_sha256'], sort_keys=True).encode()).hexdigest()
    assert expected == manifest['content_hash'] == run['bundle_hash']
    for name in ['data/pool_metadata.csv', 'data/etad_metadata.csv', 'data/wn.npy', 'research/ftir_hips_chem/scripts/outliers.py']:
        assert hashlib.sha256((STAGE/name).read_bytes()).hexdigest() == manifest['files_sha256'][name]
    audited = json.loads((AUDIT/'audit_manifest.json').read_text())
    for name in ['heldout_predictions.csv', 'calibration_scores.csv', 'case_audit.csv', 'RUN_MANIFEST.json']:
        file = RUN/name
        assert hashlib.sha256(file.read_bytes()).hexdigest() == audited['source_hashes'][str(file.relative_to(ROOT))]
    scores = read(RUN/'calibration_scores.csv')
    groups = read(AUDIT/'subgroup_metrics.csv')
    pairs = read(AUDIT/'paired_residuals.csv')
    original = read(RUN/'heldout_predictions.csv')
    # Reconcile the browser's inputs back to the original saved predictions.
    for r in scores.itertuples():
        d = pairs[pairs.cohort.eq(r.cohort)]
        q = original[original.cohort.eq(r.cohort) & original.method.eq(r.method)].set_index('sample_id')
        assert d.sample_id.is_unique and len(d) == r.n_test
        np.testing.assert_allclose(d[r.method], q.loc[d.sample_id, 'prediction'], atol=1e-10, rtol=0)
        e = d[r.method] - d.y
        assert np.isclose(np.sqrt(np.mean(e**2)), r.RMSE, atol=1e-10, rtol=0)
        assert np.isclose(np.mean(abs(e)), r.MAE, atol=1e-10, rtol=0)
    pool = read(STAGE/'data/pool_metadata.csv')
    etad = read(STAGE/'data/etad_metadata.csv')
    case_audit = read(RUN/'case_audit.csv')
    assert set(case_audit.query("kind == 'calibration'").filter_id.astype(int)) == set(pool.loc[pool.eligible, 'FilterId'])
    assert set(case_audit.query("kind == 'target'").source_row.astype(int)) == set(etad.index[etad.role.eq('sample')])
    assert len(case_audit) == run['n_cases'] and case_audit.paired_valid.all()
    wn = np.load(STAGE/'data/wn.npy')
    masks = read(OLD/'mask_channels.csv')
    changes = read(OLD/'mask_membership_changes.csv')
    historical = masks.merge(changes[changes.group.eq('All Addis')], on='mask', validate='one_to_one')
    # Verify that the historical upper cut is a selection mask, not a zeroed signal.
    for row in masks.to_dict('records'):
        keep = np.ones(len(wn), bool)
        if pd.notna(row['co2_low']):
            keep &= ~((wn >= row['co2_low']) & (wn <= row['co2_high']))
        if pd.notna(row['upper']):
            keep &= wn <= row['upper']
        assert keep.sum() == row['n_channels']
    for name in ['run_ann_weekly_20260910.py', 'build_vibes_colab_bundle.py']:
        f = ROOT/'research/ftir_hips_chem/workflows'/name
        SOURCES[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    cases = read(T/'vibes_case_investigation/case_evidence.csv')
    spectra_path = T/'vibes_case_investigation/inspection_spectra.npz'
    SOURCES[str(spectra_path.relative_to(ROOT))] = hashlib.sha256(spectra_path.read_bytes()).hexdigest()
    arrays = np.load(spectra_path)
    np.testing.assert_allclose(arrays['wn'], wn, rtol=0, atol=1e-8)
    spectra = []
    for i, sid in enumerate(arrays['sample_ids']):
        row = cases.set_index('sample_id').loc[sid]
        spectra.append({'sample_id': str(sid), 'site': row.Site, 'date': row.date,
                        'direction': row.inspection_direction,
                        'truth': row.y, 'AIRSpec_prediction': row.AIRSpec, 'VIBES_prediction': row.VIBES,
                        **{k: arrays[k][i].round(9).tolist() for k in ['raw','airspec','vibes']}})
    trace = read(T/'vibes_loading_trace/all_heldout_contributions.csv')
    trace_columns = ['sample_id','cohort','prediction_delta','intercept_and_centering_delta','centering_precision_delta'] + [
        f'{term}_{band}' for band in ['1425_1799','1800_2499','2500_2999','3000_4000']
        for term in ['spectra','coefficients']
    ]
    assert len(trace) == len(pairs) and not trace.duplicated(['sample_id','cohort']).any()
    traced = pairs.merge(trace[trace_columns], on=['sample_id','cohort'], validate='one_to_one')
    np.testing.assert_allclose(traced.prediction_delta, traced.VIBES-traced.AIRSpec, rtol=0, atol=1e-8)
    terms = trace[[c for c in trace_columns if c.startswith(('spectra_','coefficients_'))]].sum(axis=1)
    np.testing.assert_allclose(terms+trace.intercept_and_centering_delta+trace.centering_precision_delta,
                               trace.prediction_delta, rtol=0, atol=1e-8)
    blank_detail = read(RUN/'blank_metrics.csv')
    injection_detail = read(RUN/'injection_metrics.csv')
    assert not blank_detail.duplicated(['sample_id','method']).any()
    assert not injection_detail.duplicated(['sample_id','method']).any()
    addis_predictions = read(RUN/'addis_predictions.csv')
    assert len(addis_predictions.query("cohort == 'full_pool'")) == 506
    # Keep physical-filter ledgers inspectable, including all omitted rows.
    pool[['FilterId','Site','date','FilterPurposeId','n_scans','finite_spectrum','all_scans_have_purpose','TOR_EC_loading_ug','locked800','eligible','reason','split']].to_csv(OUT/'baseline_pool_ledger.csv', index=False)
    etad[['MediaId','filter_id','date','ExternalFilterType','n_scans','has_complete_spectrum','is_excluded','exclusion_reason','role','scope_reason']].to_csv(OUT/'baseline_addis_ledger.csv', index=False)
    pairs.to_csv(OUT/'baseline_paired_predictions.csv', index=False)
    data = {
        'schema_version': 1, 'evidence_date': '2026-09-21', 'run': run,
        'scores': records(scores),
        'groups': records(groups[groups.dimension.isin(['overall','loading_band','lot'])]),
        'pairs': records(pairs[['sample_id','Site','lot','date','y','AIRSpec','VIBES','cohort','loading_band']]),
        'blanks': records(read(AUDIT/'blank_summary.csv')),
        'injections': records(read(AUDIT/'injection_summary.csv')),
        'historical_masks': records(historical),
        'contributions': records(trace[trace_columns]),
        'blank_details': records(blank_detail[['sample_id','source','method','rms_from_zero']]),
        'injection_details': records(injection_detail[['sample_id','parent_id','amplitude','method','recovery_rmse']]),
        'addis_predictions': records(addis_predictions[['sample_id','method','cohort','prediction_ugm3','HIPS_EC_equivalent']]),
        'wn': wn.round(8).tolist(), 'spectra': spectra,
        'exclusions': {
            'pool_total': len(pool), 'pool_eligible': int(pool.eligible.sum()),
            'pool_reasons': records(pool.assign(reason=pool.reason.fillna('Eligible calibration sample')).groupby('reason').size().rename('n').reset_index()),
            'locked_eligible': int((pool.locked800 & pool.eligible).sum()),
            'locked_omitted': records(pool[pool.locked800 & ~pool.eligible].groupby('reason').size().rename('n').reset_index()),
            'addis_total': len(etad), 'addis_registry_excluded': int(etad.is_excluded.sum()),
            'addis_roles': records(etad.groupby('role').size().rename('n').reset_index()),
            'registry_unchanged': (STAGE/'research/ftir_hips_chem/scripts/outliers.py').read_bytes() == (ROOT/'research/ftir_hips_chem/scripts/outliers.py').read_bytes(),
        },
        'source_sha256': SOURCES,
    }
    (OUT/'baseline_comparison.json').write_text(json.dumps(data, allow_nan=False, separators=(',',':')))
    print(f'Exported {len(pairs)} paired test rows, {len(spectra)} individual spectra and {len(historical)} historical masks.')


if __name__ == '__main__':
    main()
