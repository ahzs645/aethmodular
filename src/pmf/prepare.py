"""Native single beams -> AIRSpec -> versioned X/U with per-filter QC.

No frozen record counts, automatic missing-value imputation, extrapolation,
negative-value clipping, or cross-site blank borrowing is performed.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .intake import evaluate
from .io import Dataset, fresh_directory, labels, read_records, sha256, write_json, write_table
from .vendor.airspec_baseline import airspec_baseline_matrix

DEFAULT_WINDOWS = ((2500.0, 4000.0), (1520.0, 2300.0))


def decode_scan(row):
    if int(row['DataPointFormat']) != 1 or row['XUnits'].strip() != 'WN':
        raise ValueError('Only documented little-endian float32 wavenumber scans are supported')
    count = int(row['NumberOfDataPoints'])
    if count < 3:
        raise ValueError('Scan has fewer than three points')
    raw = base64.b64decode(row['Data'], validate=True)
    if len(raw) != count * 4:
        raise ValueError('Scan payload length does not match point count')
    scale = float(row['YScalingFactor'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid scan scale')
    y = np.frombuffer(raw, dtype='<f4').astype(float) * scale
    w = np.linspace(float(row['FrequencyOfFirstPoint']), float(row['FrequencyOfLastPoint']), count)
    if not np.isfinite(y).all() or not np.isfinite(w).all() or w[0] == w[-1]:
        raise ValueError('Invalid scan values or wavenumber grid')
    if w[0] < w[-1]:
        w, y = w[::-1], y[::-1]
    return w, y


def absorbance(analysis, scans):
    sample = scans[analysis['SampleScanId']]
    background = scans[analysis['BackgroundScanId']]
    if sample.get('Type', 'Sample').strip().lower() != 'sample':
        raise ValueError('Sample scan reference does not identify a sample')
    if background.get('Type', 'Background').strip().lower() != 'background':
        raise ValueError('Background scan reference does not identify a background')
    for key in ('InstrumentSerialNumber', 'InstrumentType'):
        if sample.get(key) and background.get(key) and sample[key] != background[key]:
            raise ValueError('Sample/background instrument mismatch')
    w, y = decode_scan(sample)
    bw, by = decode_scan(background)
    if w[0] > bw[0] or w[-1] < bw[-1]:
        raise ValueError('Background does not cover sample grid; extrapolation is forbidden')
    bg = np.interp(w, bw[::-1], by[::-1])
    if np.any(y <= 0) or np.any(bg <= 0):
        raise ValueError('Sample/background intensities must be positive')
    answer = -np.log10(y / bg)
    if not np.isfinite(answer).all():
        raise ValueError('Nonfinite absorbance')
    return w, answer


def _grid(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 10 or not np.isfinite(a).all() or np.any(np.diff(a) >= 0):
        raise ValueError('Grid must contain at least ten finite, strictly descending wavenumbers')
    return a


def prepare_native(filters, analyses, scans, output, *, site, historical_ids=(), exclusions=None,
                   valid_qc_codes=(1,), grid=None, df1=6, df2=4, minimum_blanks=2,
                   blank_group_by=('lot',), subtract_blank=False, windows=DEFAULT_WINDOWS):
    """Prepare one site; filters without suitable measurements/blanks are held.

    `grid` is an explicit full baseline grid or None for a deterministic native
    common-overlap grid. It cannot be only the disconnected modeling windows.
    Batch matching uses an explicit BatchId column, never inferred calendar time.
    """
    site = str(site).strip()
    if not site or not site.isalnum():
        raise ValueError('A single explicit site code is required')
    valid_qc_codes = [int(c) for c in valid_qc_codes]
    if not valid_qc_codes:
        raise ValueError('Specify retained analysis QC codes')
    blank_group_by = tuple(blank_group_by)
    if not blank_group_by or 'lot' not in blank_group_by or not set(blank_group_by) <= {'lot', 'instrument', 'batch'}:
        raise ValueError('Blank matching requires lot; optional keys are instrument and batch')
    if len(set(blank_group_by)) != len(blank_group_by):
        raise ValueError('Duplicate blank grouping keys')
    if not all(np.isfinite(v) and v > 1 for v in (df1, df2)):
        raise ValueError('Baseline effective degrees of freedom must be finite and > 1')
    if minimum_blanks < 2 or int(minimum_blanks) != minimum_blanks:
        raise ValueError('At least two independent blanks are needed for a sample SD')
    windows = [tuple(sorted(map(float, x))) for x in windows]
    if not windows or any(len(x) != 2 or not np.isfinite(x).all() or x[0] >= x[1] for x in windows):
        raise ValueError('Invalid spectral windows')
    exclusions = exclusions or {}
    if any(not str(reason).strip() for reason in exclusions.values()):
        raise ValueError('Every explicit exclusion must have a reason')
    f = read_records(filters, ('MediaId', 'ExternalFilterId', 'ExternalFilterType', 'ExternalLotId',
        'SampleVolume_m3', 'SamplingStartDate'), unique='MediaId')
    a = read_records(analyses, ('Id', 'MediaId', 'AnalysisQcCode', 'SampleScanId', 'BackgroundScanId'), unique='Id')
    scan_records = read_records(scans, ('Id', 'Data'), unique='Id')
    s = {r['Id']: r for r in scan_records}
    other_sites = [{'filter_id': r['ExternalFilterId'], 'site': r['ExternalFilterId'].split('-')[0], 'status': 'not_selected_other_site'} for r in f if r['ExternalFilterId'].split('-')[0] != site]
    f = [r for r in f if r['ExternalFilterId'].split('-')[0] == site]
    if not f:
        raise ValueError(f'No physical filters for {site}')
    labels([r['ExternalFilterId'] for r in f], 'Physical filter IDs')
    analyses_by_media = {}
    for row in a:
        analyses_by_media.setdefault(row['MediaId'], []).append(row)
    normalized, analysis_audit, by_analysis = [], [], {}
    for row in f:
        for scan in analyses_by_media.get(row['MediaId'], [{}]):
            code = scan.get('AnalysisQcCode', '')
            retained = code in {str(v) for v in valid_qc_codes}
            record = {'site': site, 'filter_id': row['ExternalFilterId'],
                'particle_fraction': row['ExternalFilterType'], 'analysis_id': scan.get('Id', ''),
                'analysis_status': 'retained' if retained else 'not_retained',
                'date': row['SamplingStartDate'], 'volume_m3': row['SampleVolume_m3'],
                'lot': row['ExternalLotId'], 'instrument': scan.get('InstrumentId', ''),
                'batch': scan.get('BatchId', ''), 'exclusion_reason': exclusions.get(row['ExternalFilterId'], '')}
            normalized.append(record)
            if scan:
                by_analysis[scan['Id']] = scan
                analysis_audit.append({'filter_id': row['ExternalFilterId'], 'analysis_id': scan['Id'],
                    'qc_code': code, 'selected_by_qc_rule': retained, 'comments': scan.get('AnalysisComments', ''),
                    'analysis_date': scan.get('AnalysisDate', ''), 'filename': scan.get('Filename', '')})
    inventory = evaluate(normalized, historical_ids, minimum_blanks, check_blanks=False)
    decoded = {}
    for record in inventory:
        if record['status'] not in ('eligible', 'blank'):
            continue
        if any(not str(record.get(k, '')).strip() for k in blank_group_by):
            record.update(status='held', reason='missing explicit blank grouping metadata')
            continue
        try:
            decoded[record['filter_id']] = absorbance(by_analysis[record['analysis_id']], s)
        except (ValueError, KeyError, OverflowError) as exc:
            record.update(status='held', reason=f'native scan: {exc}')
    supplied_grid = grid is not None
    if grid is not None:
        target = _grid(grid)
    elif decoded:
        ordered = sorted(decoded)
        upper = min(decoded[k][0][0] for k in ordered)
        lower = max(decoded[k][0][-1] for k in ordered)
        first = decoded[ordered[0]][0]
        target = _grid(first[(first <= upper) & (first >= lower)])
    else:
        target = np.empty(0)
    selected = np.zeros(len(target), dtype=bool)
    for low, high in windows:
        selected |= (target >= low) & (target <= high)
    if len(target) and selected.sum() < 2:
        raise ValueError('No modeled spectral channels within grid')
    corrected, raw, baselines = {}, {}, {}
    for record in inventory:
        fid = record['filter_id']
        if record['status'] not in ('eligible', 'blank'):
            continue
        wn, values = decoded[fid]
        if target[0] > wn[0] or target[-1] < wn[-1]:
            record.update(status='held', reason='configured grid exceeds native coverage; no extrapolation')
            continue
        try:
            arr = np.interp(target, wn[::-1], values[::-1])
            base, corr = airspec_baseline_matrix(target, arr[None, :], df1=df1, df2=df2, n_jobs=1)
            if not np.isfinite(corr[0, selected]).all():
                raise ValueError('modeled channels extend outside valid baseline region')
            raw[fid], baselines[fid], corrected[fid] = arr, base[0], corr[0]
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            record.update(status='held', reason=f'baseline correction: {exc}')
    pools, pool_arrays = [], {}
    for record in inventory:
        if record['status'] == 'blank':
            key = tuple(record[k] for k in blank_group_by)
            pool_arrays.setdefault(key, []).append(record)
    summaries = {}
    for key, records in sorted(pool_arrays.items()):
        values = np.stack([corrected[r['filter_id']][selected] for r in records])
        sd = values.std(axis=0, ddof=1) if len(values) >= 2 else None
        usable = len(values) >= minimum_blanks and np.isfinite(sd).all() and np.all(sd > 0)
        summaries[key] = (sd, values.mean(axis=0), len(values), bool(usable))
        pools.append({'group': dict(zip(blank_group_by, key)), 'n_blanks': len(records), 'usable': bool(usable),
            'filter_ids': [r['filter_id'] for r in records], 'analysis_ids': [r['analysis_id'] for r in records],
            'instrument_ids': sorted(set(r['instrument'] for r in records)),
            'analysis_dates': [by_analysis[r['analysis_id']].get('AnalysisDate', '') for r in records]})
    for record in inventory:
        if record['status'] == 'eligible':
            key = tuple(record[k] for k in blank_group_by)
            pool = summaries.get(key)
            record['blank_count'] = pool[2] if pool else 0
            if pool is None or not pool[3]:
                record.update(status='held', reason='insufficient blanks or nonpositive/nonfinite blank SD')
    ready = sorted([r for r in inventory if r['status'] == 'eligible'], key=lambda r: (r['date'], r['filter_id']))
    protocol = {'conversion_source_sha256': sha256(__file__), 'site': site, 'particle_fraction': 'PM2.5', 'units': 'absorbance / m3 sampled air',
        'baseline': {'method': 'AIRSpec segmented smoothing spline', 'df1': float(df1), 'df2': float(df2),
            'source_sha256': sha256(Path(__file__).parent / 'vendor/airspec_baseline.py')},
        'grid_sha256': __import__('hashlib').sha256(target.astype('<f8').tobytes()).hexdigest(),
        'spectral_windows_cm1': windows, 'uncertainty': {'method': 'same-group independent field blank sample SD',
            'ddof': 1, 'minimum_blanks': minimum_blanks, 'group_by': ['site', *blank_group_by],
            'blank_subtracted': bool(subtract_blank), 'proxy_not_full_budget': True},
        'valid_qc_codes': valid_qc_codes, 'observations_clipped': False}
    sources = {name + '_sha256': sha256(path) for name, path in [('filters', filters), ('analyses', analyses), ('scans', scans)]}
    out = fresh_directory(output)
    write_table(out / 'inventory.csv', inventory)
    used = {r['analysis_id']: r for r in inventory if r.get('analysis_id')}
    for record in analysis_audit:
        disposition = used.get(record['analysis_id'], {})
        record['used_in_preprocessing'] = disposition.get('filter_id') in raw
        record['used_as_observation'] = disposition.get('status') == 'eligible'
        record['used_as_field_blank'] = disposition.get('status') == 'blank'
    write_table(out / 'analysis_selection.csv', analysis_audit)
    write_table(out / 'other_sites.csv', other_sites, ['filter_id', 'site', 'status'])
    write_json(out / 'blank_pools.json', pools)
    raw_ids = sorted(raw)
    if raw_ids:
        np.savez_compressed(out / 'processed_scans.npz', sample_ids=np.asarray(raw_ids), wavenumbers=target,
            raw_absorbance=np.stack([raw[k] for k in raw_ids]), baseline=np.stack([baselines[k] for k in raw_ids]),
            corrected=np.stack([corrected[k] for k in raw_ids]))
    if ready:
        x, u = [], []
        for record in ready:
            sd, mean, count, _ = summaries[tuple(record[k] for k in blank_group_by)]
            volume = float(record['volume_m3'])
            y = corrected[record['filter_id']][selected]
            x.append((y - mean if subtract_blank else y) / volume)
            u.append(sd * (np.sqrt(1 + 1 / count) if subtract_blank else 1) / volume)
        data = Dataset(np.stack(x), np.stack(u), np.asarray([r['filter_id'] for r in ready]),
            np.asarray([format(w, '.17g') for w in target[selected]]), protocol,
            {key: np.asarray([r.get(source, '') for r in ready]) for key, source in
                [('dates', 'date'), ('volumes', 'volume_m3'), ('lot_ids', 'lot'), ('analysis_ids', 'analysis_id'),
                 ('instrument_ids', 'instrument'), ('historical_102', 'historical_102')]}, sources)
        data.save(out / 'dataset.npz')
    counts = {status: sum(r['status'] == status for r in inventory) for status in ('eligible', 'held', 'excluded', 'blank')}
    receipt = {'schema': 'aethmodular-pmf-preparation/2', 'status': 'passed' if ready else 'blocked', 'site': site,
        'physical_filters': len(inventory), 'other_site_filters_not_selected': len(other_sites), 'counts': counts, 'processed_scans': len(raw),
        'model_features': int(selected.sum()), 'sources': sources, 'protocol': protocol, 'preparation_code_sha256': sha256(__file__),
        'grid_policy': 'explicit:no-extrapolation' if supplied_grid else 'first-ID native grid restricted to common overlap',
        'historical_allowlist': sorted(set(historical_ids)), 'exclusions': exclusions,
        'unused_exclusion_ids': sorted(set(exclusions) - {r['filter_id'] for r in inventory}),
        'warnings': ['Blank SD is provisional, not a full measurement error budget.',
            'Numeric QC codes require laboratory verification; explicit configured selection is recorded.',
            'No source names or mass concentrations are inferred.'],
        'files': {str(p.relative_to(out)): sha256(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out / 'preparation.json', receipt)
    return receipt
