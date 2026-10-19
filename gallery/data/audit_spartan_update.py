"""Compare a downloaded network export with the previous gallery and staged raw data.

Run with uv run after export_spartan_network.py. Bundle paths come from .env.
The output identifies actual added IDs and missing-value coverage, not just row
count differences. It never overwrites previous source archives or frozen runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'research/ftir_hips_chem/scripts'))
from aethmodular_cli.env import load_repo_env  # noqa: E402
from common.spartan_io import RAW_DIR  # noqa: E402
from data_paths import maia_data_root, HIPS_PULLS_SUBDIR, HIPS_RAW_CSV_NAME  # noqa: E402
from pls_transfer import FTIRTransferPaths  # noqa: E402
from export_spartan_network import read_csv, verify_bundle  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and pd.notna(value)


def compare(ftir, hips, previous_ftir, previous_addis, previous_hips, data, public, out):
    verify_bundle(ftir)
    verify_bundle(hips)
    index = json.loads((data / 'spartan_network/index.json').read_text())
    network = {r['id']: r for site in index['sites']
               for r in json.loads(gzip.decompress((data / site['filters']).read_bytes()))['rows']}
    focus = json.loads((data / 'filters.json').read_text())['rows']
    old_network = json.loads((data / 'spartan_reference.json').read_text())['rows']
    previous_ids = {r['id'] for r in [*focus, *old_network]}
    added_filters = [r for key, r in network.items() if key not in previous_ids]
    missing = sorted(previous_ids - set(network))
    fills = Counter()
    for r in focus:
        extra = network.get(r['id'], {})
        for field in index['fields']:
            if field in r and r[field] is None and numeric(extra.get(field)):
                fills[field] += 1

    added_analyses, removed_analyses, site_deltas, sources = [], [], [], {}
    old_by_site = {}
    for path in sorted(previous_ftir.glob('*/*_ftir_analysis.csv')):
        old_by_site[path.parent.name] = {r['Id'] for r in read_csv(path)}
        sources[f'previous_ftir/{path.parent.name}/{path.name}'] = sha(path)
    if previous_addis.is_file():
        old_by_site['ETAD'] = {r['SampleAnalysisId'] for r in read_csv(previous_addis)}
        sources['previous_addis_spectra'] = sha(previous_addis)
    for path in sorted(ftir.glob('*/*_ftir_analysis.csv')):
        code = path.parent.name
        rows = read_csv(path)
        new = {r['Id'] for r in rows}
        if len(new) != len(rows):
            raise ValueError(f'Duplicate FTIR analysis IDs: {code}')
        old = old_by_site.pop(code, set())
        added_analyses += [dict(site=code, analysis_id=k) for k in sorted(new-old, key=int)]
        removed_analyses += [dict(site=code, analysis_id=k) for k in sorted(old-new, key=int)]
        site_deltas.append(dict(site=code, previous_analyses=len(old), current_analyses=len(new),
                                added_analyses=len(new-old), removed_analyses=len(old-new)))
    for code, old in old_by_site.items():
        removed_analyses += [dict(site=code, analysis_id=k) for k in sorted(old, key=int)]
        site_deltas.append(dict(site=code, previous_analyses=len(old), current_analyses=0,
                                added_analyses=0, removed_analyses=len(old)))
    before = read_csv(previous_hips)
    after = read_csv(hips / HIPS_RAW_CSV_NAME)
    def sample(rows):
        return {r['SampleAnalysisId'] for r in rows if r['ResultTypeId'] == '0'}
    a, b = sample(before), sample(after)
    # Analysis × result type × timestamp × wavelength identifies readings in
    # the older joined export, which did not carry a Results.Id column.
    def key(r):
        return (r['SampleAnalysisId'], r['ResultTypeId'],
                pd.to_datetime(r['Timestamp']).isoformat(), float(r['Wavelength']))
    old_readings = Counter(key(r) for r in before)
    new_readings = Counter(key(r) for r in after)
    old_values = {key(r): r for r in before}
    changed = []
    measurements = ['Transmittance', 'Reflectance', 'TransmittanceRaw', 'ReflectanceRaw',
                    'LaserPower', 'LaserDiodeTemperature', 'TransmittanceSensorTemperature', 'ReflectanceSensorTemperature']
    for r in after:
        prior = old_values.get(key(r))
        if prior is None:
            continue
        for field in measurements:
            x, y = prior.get(field, ''), r.get(field, '')
            if (float(x) if x else None) != (float(y) if y else None):
                changed.append(dict(analysis_id=r['SampleAnalysisId'], result_type=r['ResultTypeId'],
                                    field=field, previous=x, current=y))
    chemistry = []
    for path in sorted((hips / 'public_chemspec').rglob('*.csv')):
        old = public / path.parent.name / path.name
        chemistry.append(dict(product=path.parent.name, file=path.name,
                              status='missing_locally' if not old.exists() else 'identical' if sha(old) == sha(path) else 'different'))
    sources.update({label: sha(path) for label, path in [
        ('previous_hips', previous_hips), ('ftir_manifest', ftir/'manifest.json'),
        ('hips_manifest', hips/'manifest.json'), ('previous_gallery_filters', data/'filters.json'),
        ('previous_gallery_network', data/'spartan_reference.json'), ('current_site_index', data/'spartan_network/index.json')]})
    summary = dict(generated=datetime.now(timezone.utc).isoformat(),
        previous_gallery_physical_filters=len(previous_ids), current_gallery_physical_filters=len(network),
        added_physical_pm25_filters=len(added_filters), removed_previous_filter_ids=missing,
        previous_ftir_analyses=sum(s['previous_analyses'] for s in site_deltas),
        current_ftir_analyses=sum(s['current_analyses'] for s in site_deltas),
        added_ftir_analyses=len(added_analyses), removed_ftir_analyses=len(removed_analyses),
        additional_ftir_sites=[s['site'] for s in site_deltas if s['previous_analyses'] == 0 and s['current_analyses'] > 0],
        previous_hips_result_rows=len(before), current_hips_result_rows=len(after),
        added_hips_readings=sum((new_readings-old_readings).values()), removed_hips_readings=sum((old_readings-new_readings).values()),
        added_hips_sample_analyses=len(b-a), removed_hips_sample_analyses=len(a-b),
        changed_shared_hips_measurements=len(changed), public_chemistry_files=dict(Counter(r['status'] for r in chemistry)),
        existing_focus_missing_values_filled=dict(fills), source_sha256=sources,
        note='Added filter IDs are relative to the union of both previous gallery exports. Native FTIR/HIPS analysis counts include all filter types and retain repeats; they are not additional independent PM2.5 observations. HIPS raw readings do not supply new official Fabs/BC. EC/OC ChemSpec labels duplicate the FTIR products. Previously populated focus values retain precedence; missing values are backfilled from network sources. Frozen calibration runs are not regenerated.')
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('added_physical_pm25_filters', added_filters), ('added_ftir_analyses', added_analyses),
                       ('removed_ftir_analyses', removed_analyses), ('site_changes', site_deltas),
                       ('changed_hips_measurements', changed), ('public_chemistry_files', chemistry)]:
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    load_repo_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ftir-bundle', type=Path, default=os.getenv('AETHMODULAR_SPARTAN_FTIR_BUNDLE'))
    ap.add_argument('--hips-bundle', type=Path, default=os.getenv('AETHMODULAR_SPARTAN_HIPS_BUNDLE'))
    ap.add_argument('--previous-ftir', type=Path, default=maia_data_root()/'DAVIS/SPARTAN FTIR pulls')
    ap.add_argument('--previous-addis', type=Path, default=FTIRTransferPaths.defaults().etad_dir/'ETAD_FTIR_spectra.csv')
    ap.add_argument('--previous-hips', type=Path, default=maia_data_root()/HIPS_PULLS_SUBDIR/HIPS_RAW_CSV_NAME)
    ap.add_argument('--data-dir', type=Path, default=ROOT/'gallery/app/public/data')
    ap.add_argument('--public-dir', type=Path, default=RAW_DIR/'FilterBased')
    ap.add_argument('--out', type=Path, default=ROOT/'research/ftir_hips_chem/output/tables/spartan_network_update')
    args = ap.parse_args()
    if not args.ftir_bundle or not args.hips_bundle:
        ap.error('Configure both native bundle paths in .env or pass the bundle options')
    summary = compare(args.ftir_bundle.expanduser(), args.hips_bundle.expanduser(),
        args.previous_ftir, args.previous_addis, args.previous_hips, args.data_dir, args.public_dir, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['source_sha256', 'note']}, indent=2))


if __name__ == '__main__':
    main()
