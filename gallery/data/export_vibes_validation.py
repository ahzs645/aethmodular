"""Export the bounded continuum check; keep it separate from model benchmarks."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_research_spectral import Page, ROOT

SOURCE = ROOT / 'research/ftir_hips_chem/output/tables/vibes_continuum_check'


def main():
    manifest = SOURCE / 'RUN.json'
    if not manifest.exists():
        raise SystemExit('Continuum-check results are unavailable; existing export left unchanged.')
    run = json.loads(manifest.read_text())
    if run.get('schema_version') != 1:
        raise ValueError('Unsupported continuum-check schema')
    hashes = run.get('outputs_sha256', {})
    required = {'signal_recovery.csv', 'signal_traces.csv', 'blank_rms.csv', 'runtime.csv',
                'reference_fit_counts.csv', 'fit_diagnostics.csv'}
    if 'map_replayed' in run:
        required.add('map_audit.csv')
    if not required <= hashes.keys():
        raise ValueError('Continuum-check output receipt is incomplete')
    for name in required:
        with (SOURCE / name).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != hashes[name]:
                raise ValueError(f'Continuum-check output changed: {name}')
    p = Page('baseline_validation', 'VIBES fit quality and continuum preservation',
             out=ROOT / 'gallery/app/public/data', exporter=Path(__file__))
    for name, filename in [('recovery', 'signal_recovery.csv'), ('traces', 'signal_traces.csv'),
                            ('blank_rms', 'blank_rms.csv'), ('runtime', 'runtime.csv'),
                            ('reference_fit_counts', 'reference_fit_counts.csv'), ('fits', 'fit_diagnostics.csv')]:
        frame = pd.read_csv(SOURCE / filename)
        p.add(name, frame, digits=None, path=str((SOURCE / filename).relative_to(ROOT)),
              read='csv', experiment='Frozen-background held-out blank controls')
    if 'map_replayed' in run:
        p.add('map_audit', pd.read_csv(SOURCE / 'map_audit.csv'), digits=None,
              path=str((SOURCE / 'map_audit.csv').relative_to(ROOT)), read='csv')
    p.add('run', pd.DataFrame([{k: v for k, v in run.items() if not isinstance(v, (dict, list))}]),
          digits=None, path=str(manifest.relative_to(ROOT)), read='json')
    p.text.update(scope=run['scope'], acceptance=run['acceptance'], new_cloud_run=run['new_cloud_run'],
        definition='Incremental recovery = corrected(blank + known addition) minus corrected(blank). '
                   'An area recovery of 1 retains the addition, 0 removes it, and negative values reverse its sign. '
                   'Full-grid spline recovery is unavailable because the spline stops at 1425 cm⁻¹.',
        trace='One declared example: the first held-out blank in the frozen case ledger. '
              'Adjacent groups of three channels are averaged for display; metrics use every native channel.',
        reference='The original fit log predates recovery of two failed blank fits. It is not the completion status '
                  'of the current recovered benchmark or the new 32,286-fit job.')
    p.write()


if __name__ == '__main__':
    main()
