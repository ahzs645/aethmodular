"""Exploratory, training-only PCA distance diagnostic for frozen spectra.

Run from the repository root with:
uv run --locked --no-sync python gallery/data/export_baseline_applicability.py

This does not refit the EC calibration or use test/target EC to fit PCA.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09'
AUDIT = ROOT/'research/ftir_hips_chem/output/tables/vibes_subgroup_audit/audit_manifest.json'
OUT = ROOT/'gallery/app/public/data/baseline_applicability.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinates(model, matrix, train_t2, train_q):
    rows = []
    for start in range(0, len(matrix), 256):
        batch = np.asarray(matrix[start:start+256], dtype=np.float32)
        score = model.transform(batch)
        t2 = np.sum(score.astype(float)**2/model.explained_variance_, axis=1)
        residual = batch-model.mean_-score@model.components_
        q = np.sqrt(np.mean(residual.astype(float)**2, axis=1))
        rows.extend(zip(t2.tolist(), q.tolist(),
                        (100*np.searchsorted(train_t2, t2, side='right')/len(train_t2)).tolist(),
                        (100*np.searchsorted(train_q, q, side='right')/len(train_q)).tolist()))
    return rows


def main():
    audited = json.loads(AUDIT.read_text())['source_hashes']
    used = ['case_audit.csv','heldout_predictions.csv','corrected_AIRSpec.npy','corrected_VIBES.npy','RUN_MANIFEST.json']
    sources = {}
    for name in used:
        path = RUN/name
        key = str(path.relative_to(ROOT))
        assert sha(path) == audited[key], key
        sources[key] = audited[key]
    cases = pd.read_csv(RUN/'case_audit.csv')
    predictions = pd.read_csv(RUN/'heldout_predictions.csv')
    test = np.flatnonzero((cases.kind == 'calibration') & cases.split.eq('test') & cases.paired_valid)
    train = np.flatnonzero((cases.kind == 'calibration') & cases.split.eq('train') & cases.paired_valid)
    target = np.flatnonzero((cases.kind == 'target') & cases.paired_valid)
    assert len(train) == 10066 and len(test) == 2327 and len(target) == 253
    exported = []
    with threadpool_limits(limits=2):
        for method in ['AIRSpec','VIBES']:
            matrix = np.load(RUN/f'corrected_{method}.npy', mmap_mode='r', allow_pickle=False)
            assert matrix.shape == (len(cases), 2002) and np.isfinite(matrix).all()
            model = PCA(n_components=8, svd_solver='randomized', random_state=42)
            training = np.asarray(matrix[train], dtype=np.float32)
            model.fit(training)
            scores = model.transform(training)
            train_t2 = np.sort(np.sum(scores.astype(float)**2/model.explained_variance_, axis=1))
            train_q = np.empty(len(train))
            for start in range(0, len(train), 256):
                block = training[start:start+256]
                z = scores[start:start+256]
                residual = block-model.mean_-z@model.components_
                train_q[start:start+256] = np.sqrt(np.mean(residual.astype(float)**2, axis=1))
            train_q.sort()
            pool_positions = np.r_[test, target]
            coords = coordinates(model, matrix[pool_positions], train_t2, train_q)
            source = cases.iloc[pool_positions].reset_index(drop=True)
            pred = predictions.query("cohort == 'full_pool' and method == @method").set_index('sample_id')
            assert set(source.iloc[:len(test)].sample_id) == set(pred.index)
            for i, (t2, q, t2_pct, q_pct) in enumerate(coords):
                item = source.iloc[i]
                row = {'sample_id': item.sample_id, 'site': item.Site, 'kind': 'test' if i < len(test) else 'addis',
                       'method': method, 't2': t2, 'q_rms': q, 't2_train_percentile': t2_pct,
                       'q_train_percentile': q_pct}
                if i < len(test):
                    row['abs_error'] = abs(float(pred.loc[item.sample_id, 'prediction']) - float(item.y))
                else:
                    row['abs_error'] = None
                exported.append(row)
            print(f'{method}: {len(train)} training, {len(test)} test, {len(target)} Addis; '
                  f'8-PC explained variance {sum(model.explained_variance_ratio_):.3f}')
    assert len(exported) == 2*(len(test)+len(target))
    OUT.write_text(json.dumps({'schema_version': 1, 'scope': 'full_pool', 'n_components': 8,
                               'method': 'Separate 8-component randomized PCA fits to each method’s full-pool training spectra; '
                                         'T² is sum of score²/training variance; Q is spectral reconstruction RMS. '
                                         'Percentiles compare to training spectra. Exploratory, not the frozen PLS score space, '
                                         'not a validated flag or Addis accuracy.',
                               'counts': {'train': len(train), 'test': len(test), 'addis': len(target)},
                               'source_sha256': sources, 'rows': exported}, allow_nan=False, separators=(',', ':')))
    for key, digest in sources.items():
        assert sha(ROOT/key) == digest, key
    print(f'Exported {len(exported)} spectral-distance rows to {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
