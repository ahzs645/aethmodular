"""Export the completed, recovered rank-70 analysis without refitting models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'research/ftir_hips_chem/output/tables'
SRC = TABLES / 'vibes_fullrange/interpretation_20261002'
OUT = ROOT / 'gallery/app/public/data/vibes_fullrange_results.json'


def main():
    summary = json.loads((SRC / 'summary.json').read_text())
    cases = pd.read_csv(SRC / 'cases.csv')
    full = np.load(SRC / 'corrected_VIBES_full_recovered.npy', mmap_mode='r')
    wn = np.load(TABLES / 'vibes_fullrange/rank30/wn.npy')
    assert full.shape == (12808, len(wn)) and np.isfinite(full).all()
    earlier = TABLES / 'vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09'
    arrays = {'AIRSpec': (np.load(earlier / 'wn.npy'), np.load(earlier / 'corrected_AIRSpec.npy', mmap_mode='r')),
              'capped_full': (wn, np.load(TABLES / 'vibes_fullrange/rank30/corrected_VIBES_full.npy', mmap_mode='r')),
              'uncapped_full': (wn, full)}
    spectra = {}
    for kind in ['calibration', 'target']:
        spectra[kind] = {}
        rows = np.flatnonzero(cases.kind.eq(kind))
        for method, (grid, values) in arrays.items():
            bins = [np.arange(i, min(i+8, len(grid))) for i in range(0, len(grid), 8)]
            selected = values[rows]
            binned = np.stack([selected[:, b].mean(axis=1) for b in bins], axis=1)
            spectra[kind][method] = {'n': len(rows), 'wn': [float(grid[b].mean()) for b in bins],
                                    'q': np.quantile(binned, [.25, .5, .75], axis=0).tolist()}
    # Reconcile displayed scores against saved paired predictions.
    predictions = pd.read_csv(SRC / 'benchmark/heldout_predictions.csv')
    for row in summary['calibration_scores']:
        p = predictions.loc[predictions.cohort.eq(row['cohort']) & predictions.method.eq(row['method'])]
        assert len(p) == row['n_test'] and np.isclose(np.sqrt(np.mean((p.prediction-p.y)**2)), row['RMSE'])
    payload = {'schema_version': 1, 'source': SRC.relative_to(ROOT).as_posix(),
               'created_utc': summary['created_utc'], 'n_cases': summary['n_cases'],
               'recovered_rows': summary['recovered_rows'], 'labels': summary['method_labels'],
               'scores': summary['calibration_scores'], 'intervals': summary['paired_site_intervals'],
               'blank_median_rms': summary['blank_median_rms'], 'injections': summary['injection_medians'],
               'spectra': spectra, 'scope': summary['scope']}
    OUT.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
    print(f'Wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
