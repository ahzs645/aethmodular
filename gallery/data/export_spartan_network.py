"""Package the native SPARTAN exports and public chemistry for lazy gallery loading.

uv run python gallery/data/export_spartan_network.py

Set AETHMODULAR_SPARTAN_FTIR_BUNDLE and AETHMODULAR_SPARTAN_HIPS_BUNDLE
in .env, or pass --ftir-bundle / --hips-bundle. Rebuild spartan_reference.json
with export_spartan_reference.py when the public/lab chemistry sources change.
Native single-beam archives stay outside the gallery. This export derives full-
resolution absorbance, retaining every analysis and its native grid. It never
converts raw HIPS readings to Fabs. PM10 and blanks remain identifiable in the
spectra archive; chemistry charts contain physical PM2.5 filters only.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research/ftir_hips_chem/scripts'))
from aethmodular_cli.env import load_repo_env  # noqa: E402
from config import SITES  # noqa: E402
from data_matching import base_filter_id  # noqa: E402
from outliers import apply_exclusion_flags  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_labels import display_site_name  # noqa: E402
from spectrum_codec import DEFAULT_STEP, encode  # noqa: E402

BASELINES = ROOT / 'research/ftir_hips_chem/output/tables/network_baselines_20261019'


def load_baselines(folder):
    """Precomputed baselines by analysis id (research/ftir_hips_chem/workflows/network_baselines_20261019.py).

    airspec: (start index on the native grid, baseline values) over the spline's 1425-4000 cm-1 span.
    vibes: the corrected spectrum on the VIBES background's own grid (resampled input minus baseline).
    Returns ({}, None) when the run is absent; the export then carries raw spectra only.
    """
    folder = Path(folder)
    out, grid = defaultdict(dict), None
    if (folder / 'airspec_baselines.npz').exists():
        z = np.load(folder / 'airspec_baselines.npz')
        native_n = pd.read_csv(folder / 'airspec_cases.csv', dtype={'analysis_id': str}).set_index('analysis_id').native_n.to_dict()
        for k in z.files:
            aid, start = k[1:].split('_')
            out[aid]['airspec'] = (int(start), z[k].astype(float))
            out[aid]['native_n'] = int(native_n[aid])
    if (folder / 'vibes_baselines_model_grid.npy').exists():
        bundle = folder / 'vibes_bundle'
        cases = pd.read_csv(bundle / 'cases.csv', dtype={'analysis_id': str})
        X = np.load(bundle / 'X_model_grid.npy').astype(float)
        base = np.load(folder / 'vibes_baselines_model_grid.npy').astype(float)
        diag = pd.read_csv(folder / 'vibes_fit_diagnostics.csv')
        ok = set(diag.loc[diag.success.astype(bool), 'case_row'].astype(int))
        for c in cases.itertuples():
            if c.case_row in ok and np.isfinite(base[c.case_row]).all():
                out[c.analysis_id]['vibes'] = X[c.case_row] - base[c.case_row]
        wn = np.load(bundle / 'background_model.npz')['wn']
        manifest = json.loads((bundle / 'MANIFEST.json').read_text())
        grid = dict(first=float(wn[0]), last=float(wn[-1]), n=int(len(wn)), model=manifest['model'],
                    model_rank=manifest['model_rank'], model_blanks=manifest['model_blanks'], resampling=manifest['resampling'],
                    fits_failed=int(len(diag) - len(ok)))
    return out, grid


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def number(v):
    return float(v) if v and np.isfinite(float(v)) else None


def date(v):
    return str(pd.to_datetime(v).date()) if v else ''


def decode_scan(scan):
    if int(scan['DataPointFormat']) != 1 or scan['XUnits'] != 'WN':
        raise ValueError(f"Unsupported scan format: {scan['Id']}")
    values = np.frombuffer(base64.b64decode(scan['Data'], validate=True), dtype='<f4').astype(float)
    n = int(scan['NumberOfDataPoints'])
    if len(values) != n or n < 2:
        raise ValueError(f"Scan length mismatch: {scan['Id']}")
    values *= float(scan['YScalingFactor'])
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite single beam: {scan['Id']}")
    wn = np.linspace(float(scan['FrequencyOfFirstPoint']), float(scan['FrequencyOfLastPoint']), n)
    if wn[0] <= wn[-1]:
        raise ValueError(f"Expected descending native grid: {scan['Id']}")
    return wn, values


def absorbance(sample, background):
    wn, s = decode_scan(sample)
    wb, b = decode_scan(background)
    if len(wn) != len(wb) or not np.allclose(wn, wb, rtol=0, atol=1e-7):
        if min(wn) < min(wb) or max(wn) > max(wb):
            raise ValueError('Background does not cover sample grid; refusing extrapolation')
        b = np.interp(wn, wb[::-1], b[::-1])
    valid = (s > 0) & (b > 0)
    a = np.full(len(s), np.nan)
    a[valid] = -np.log10(s[valid] / b[valid])
    if not valid.any():
        raise ValueError('No positive sample/background pairs')
    return wn, a.astype('<f4'), int((~valid).sum())


def json_bytes(obj):
    return json.dumps(obj, separators=(',', ':'), allow_nan=False).encode()


def verify_bundle(bundle):
    """Check the saved source receipt before deriving any gallery data."""
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8-sig'))
    for item in manifest['Files']:
        path = bundle / item['File'].replace('\\', '/')
        if not path.resolve().is_relative_to(bundle.resolve()):
            raise ValueError('Source manifest path escapes its bundle')
        if path.stat().st_size != int(item['Bytes']) or hashlib.sha256(path.read_bytes()).hexdigest().upper() != item['SHA256'].upper():
            raise ValueError(f'Source checksum mismatch: {item["File"]}')
    return manifest


def write_asset(out, stem, body, kind='json'):
    packed = gzip.compress(body, compresslevel=9, mtime=0)
    signature = hashlib.sha256(packed).hexdigest()[:16]
    rel = f'{stem}-{signature}.{kind}.gz'
    path = out / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packed)
    return 'spartan_network/' + rel


def build(ftir, hips, data, baselines_dir=BASELINES, step=DEFAULT_STEP):
    baselines, vibes_grid = load_baselines(baselines_dir)
    ftir_receipt = verify_bundle(ftir)
    verify_bundle(hips)
    out = data / 'spartan_network'
    out.mkdir(parents=True, exist_ok=True)
    ref = json.loads((data / 'spartan_reference.json').read_text())
    map_sites = {s['code']: s for s in json.loads((data / 'spartan_map.json').read_text())['sites']}
    native_sites = read_csv(hips / 'spartan_Sites.csv')
    site_ids = {s['Id']: s['SiteCode'] for s in native_sites}
    native_by_code = {s['SiteCode']: s for s in native_sites}
    filters = read_csv(hips / 'spartan_Filters.csv')
    media = {r['MediaId']: r for r in filters}
    sample_type = next(r['Id'] for r in read_csv(hips / 'hips_ResultTypes.csv') if r['Name'].lower() == 'sample')
    ha = {r['Id']: r for r in read_csv(hips / 'hips_SampleAnalysis.csv')}
    raw_hips = defaultdict(lambda: defaultdict(list))
    for result in read_csv(hips / 'hips_Results.csv'):
        analysis = ha[result['SampleAnalysisId']]
        f = media.get(analysis['MediaId'])
        if not f or not f['SiteId'] or result['ResultTypeId'] != sample_type:
            continue
        code = site_ids[f['SiteId']]
        raw_hips[code][base_filter_id(f['ExternalFilterId'])].append(dict(
            analysis_id=analysis['Id'], media_id=f['MediaId'], filter_id=f['ExternalFilterId'],
            filter_type=f['ExternalFilterType'], qc=analysis['AnalysisQcCode'],
            timestamp=result['Timestamp'], result_id=result['Id'],
            **{k: number(result[k]) for k in ['Transmittance', 'Reflectance', 'TransmittanceRaw', 'ReflectanceRaw', 'Wavelength']}))

    public = defaultdict(dict)
    for r in ref['rows']:
        if r['id'] in public[r['code']]:
            raise ValueError(f"Duplicate physical chemistry filter: {r['id']}")
        public[r['code']][r['id']] = dict(r)
    # Keep existing focus filters that predate/extend the public and native pulls.
    # Missing native metadata must not make a previously browsable filter vanish.
    for r in json.loads((data / 'filters.json').read_text())['rows']:
        if r['id'] not in public[r['code']]:
            keys = ['id', 'code', 'site', 'date', 'excluded', 'exclusion_reason', *ref['fields']]
            public[r['code']][r['id']] = {k: r[k] for k in keys if k in r}
    native_pm25 = defaultdict(dict)
    for f in filters:
        if not f['SiteId'] or f['ExternalFilterType'] != 'PM2.5':
            continue
        code = site_ids[f['SiteId']]
        key = base_filter_id(f['ExternalFilterId'])
        if key in native_pm25[code]:
            raise ValueError(f'Ambiguous native PM2.5 physical ID: {key}')
        native_pm25[code][key] = f

    focus = {cfg['code']: name for name, cfg in SITES.items()}
    sites = []
    for code in sorted(set(map_sites) | set(native_by_code) | set(public)):
        m, ns = map_sites.get(code, {}), native_by_code.get(code, {})
        name = display_site_name(code, focus.get(code), m.get('name'), ns.get('Description')).replace('_', ' ')
        spectra = defaultdict(list)
        src = ftir / code
        if src.is_dir():
            sf = {r['MediaId']: r for r in read_csv(src / f'{code}_filters.csv')}
            scans = {r['Id']: r for r in read_csv(src / f'{code}_scans_base64.csv')}
            analyses = read_csv(src / f'{code}_ftir_analysis.csv')
            if len({r['Id'] for r in analyses}) != len(analyses):
                raise ValueError(f'Duplicate FTIR analysis: {code}')
            for a in analyses:
                f = sf[a['MediaId']]
                # Referenced-scan absence is an export error, never a silent skip.
                wn, y, invalid = absorbance(scans[a['SampleScanId']], scans[a['BackgroundScanId']])
                key = base_filter_id(f['ExternalFilterId'])
                spectra[key].append((dict(analysis_id=a['Id'], media_id=a['MediaId'],
                    filter_id=f['ExternalFilterId'], filter_type=f['ExternalFilterType'],
                    date=date(f['SamplingStartDate']), analysis_date=date(a['AnalysisDate']),
                    qc=a['AnalysisQcCode'], sample_scan_id=a['SampleScanId'], background_scan_id=a['BackgroundScanId'],
                    first=float(wn[0]), last=float(wn[-1]), n=len(y), invalid_points=invalid), y))
        smeta = {}
        for key, entries in sorted(spectra.items()):
            # each trace: integer steps, differences, byte planes (spectrum_codec); offsets in bytes
            offset = 0
            descriptors, blocks = [], []

            def add(values):
                nonlocal offset
                body, desc = encode(values, step)
                blocks.append(body)
                desc['offset'] = offset
                offset += len(body)
                return desc
            for descriptor, y in entries:
                series = {'raw': add(y)}
                b = baselines.get(descriptor['analysis_id'], {})
                if b and b.get('native_n', len(y)) != len(y):
                    raise ValueError(f"Baselines for analysis {descriptor['analysis_id']} were computed on another spectrum; rerun network_baselines")
                if 'airspec' in b:
                    start, base = b['airspec']
                    if start + len(base) > len(y):
                        raise ValueError(f"AIRSpec baseline longer than its spectrum: {descriptor['analysis_id']}")
                    series['airspec_baseline'] = {**add(base), 'start': start}
                if 'vibes' in b:
                    series['vibes_corrected'] = add(b['vibes'])
                descriptors.append({**descriptor, 'series': series})
            file = write_asset(out, f'spectra/{code}/{key}', b''.join(blocks), 'bin')
            smeta[key] = dict(file=file, analyses=descriptors)
        native = dict(schema_version=2, code=code, spectra=smeta, hips=dict(raw_hips[code]))
        native_file = write_asset(out, f'native/{code}', json_bytes(native))
        rows = public[code]
        for key, f in native_pm25[code].items():
            d = date(f['SamplingStartDate'])
            if not d or d > datetime.now(timezone.utc).date().isoformat():
                continue
            r = rows.setdefault(key, dict(id=key, code=code, site=name, date=d, excluded=False, exclusion_reason=None))
            volume = number(f.get('SampleVolume_m3'))
            if r.get('Sampled volume') is None and volume is not None and volume > 0:
                r['Sampled volume'] = volume
            r['Native FTIR analyses'] = sum(e[0]['filter_type'] == 'PM2.5' for e in spectra.get(key, []))
            r['Native HIPS sample readings'] = sum(e['filter_type'] == 'PM2.5' for e in raw_hips[code].get(key, []))
        for r in rows.values():
            r['site'] = name
        # Canonical flags also cover native-only filters; never erase an existing flag.
        if rows:
            flagged = apply_exclusion_flags(pd.DataFrame([{'id': k, 'date': pd.Timestamp(r['date'])} for k, r in rows.items()]), focus.get(code, code))
            for r in flagged.to_dict('records'):
                if r['is_excluded']:
                    rows[r['id']].update(excluded=True, exclusion_reason=r['exclusion_reason'])
        records = sorted(rows.values(), key=lambda r: (r['date'], r['id']))
        file = write_asset(out, f'sites/{code}', json_bytes(dict(schema_version=1, code=code, rows=records)))
        dates = sorted(r['date'] for r in records if r['date'])
        sites.append(dict(code=code, name=name, location=display_site_name(code, m.get('name'), ns.get('Description'), name),
            lat=m.get('lat') if m.get('lat') is not None else number(ns.get('Latitude')),
            lon=m.get('lon') if m.get('lon') is not None else number(ns.get('Longitude')),
            n_filters=len(records), date_min=dates[0] if dates else '', date_max=dates[-1] if dates else '',
            filters=file, native=native_file, n_ftir_analyses=sum(len(v) for v in spectra.values()),
            n_hips_readings=sum(len(v) for v in raw_hips[code].values())))
        print(f'{code}: {len(records)} PM2.5 filters; {sites[-1]["n_ftir_analyses"]} FTIR analyses (all filter types)')
    if sum(s['n_ftir_analyses'] for s in sites) != int(ftir_receipt['Analyses']):
        raise ValueError('Exported FTIR total differs from source receipt')
    index = dict(schema_version=2, generated=datetime.now(timezone.utc).isoformat(),
        source_bundles=[ftir.name, hips.name],
        unassigned_native_filters=sum(not f['SiteId'] for f in filters),
        source_sha256={name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in [
            ('ftir_manifest', ftir / 'manifest.json'), ('hips_manifest', hips / 'manifest.json'),
            ('spartan_reference', data / 'spartan_reference.json'),
            ('focus_filters', data / 'filters.json'),
            ('exporter', Path(__file__))]},
        note='Physical PM2.5 filters in charts; native spectra retain all filter types and repeated analyses. '
             'Raw absorbance = -log10(sample/background), native resolution, no baseline correction; '
             'nonpositive single-beam pairs are NaN. Raw HIPS readings are not final Fabs or BC. '
             'Missing native records do not imply missing public measurements.',
        fields=ref['fields'], field_units=ref['field_units'], sites=sites,
        spectrum_encoding=dict(step=step, note='integer steps of `step`, differences, four byte planes per trace (gallery/data/spectrum_codec.py)'),
        baselines=dict(
            airspec=dict(method='AIRSpec/APRLssb segmented smoothing spline, DF1=6, DF2=4, on each native grid; 1425-4000 cm-1') if any('airspec' in b for b in baselines.values()) else None,
            vibes=dict(method='VIBES with the saved gallery background (no refit), ELBO + MAP, tau 0.1, PB loss', grid=vibes_grid) if vibes_grid else None,
            source=str(Path(baselines_dir).relative_to(ROOT)) if baselines else None))
    # Publish the index last: every content-addressed asset is complete before readers see it.
    temp = out / 'index.tmp'
    temp.write_bytes(json_bytes(index))
    temp.replace(out / 'index.json')
    print(f'Wrote {len(sites)} sites, {sum(s["n_filters"] for s in sites)} physical PM2.5 filters')


def main():
    load_repo_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ftir-bundle', default=os.getenv('AETHMODULAR_SPARTAN_FTIR_BUNDLE'))
    ap.add_argument('--hips-bundle', default=os.getenv('AETHMODULAR_SPARTAN_HIPS_BUNDLE'))
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'gallery/app/public/data')
    ap.add_argument('--baselines', type=Path, default=BASELINES, help='network_baselines run folder (absent: raw only)')
    ap.add_argument('--step', type=float, default=DEFAULT_STEP, help='absorbance step for stored traces (1e-8 default; 1e-7 is ~25%% smaller)')
    args = ap.parse_args()
    if not args.ftir_bundle or not args.hips_bundle:
        ap.error('Set both native bundle paths in .env or pass --ftir-bundle and --hips-bundle')
    build(Path(args.ftir_bundle).expanduser(), Path(args.hips_bundle).expanduser(), args.data_dir, args.baselines, args.step)


if __name__ == '__main__':
    main()
