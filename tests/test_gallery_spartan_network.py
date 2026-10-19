"""Scientific decoding checks for the native network gallery export."""
import base64
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'gallery/data/export_spartan_network.py'
spec = importlib.util.spec_from_file_location('spartan_network_export', PATH)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)
CODEC = Path(__file__).resolve().parents[1] / 'gallery/data/spectrum_codec.py'
codec_spec = importlib.util.spec_from_file_location('spectrum_codec', CODEC)
codec = importlib.util.module_from_spec(codec_spec)
codec_spec.loader.exec_module(codec)


def scan(values, first=4000, last=500, scale=1):
    return dict(Id='1', DataPointFormat='1', XUnits='WN',
                Data=base64.b64encode(np.asarray(values, dtype='<f4').tobytes()).decode(),
                NumberOfDataPoints=str(len(values)), FrequencyOfFirstPoint=str(first),
                FrequencyOfLastPoint=str(last), YScalingFactor=str(scale))


def test_native_absorbance_uses_scaling_and_preserves_nonpositive_gaps():
    wn, y, invalid = export.absorbance(scan([1, 2, 0, -1], scale=2), scan([4, 4, 4, 4]))
    np.testing.assert_allclose(wn[[0, -1]], [4000, 500])
    np.testing.assert_allclose(y[:2], [np.log10(2), 0], atol=1e-7)
    assert np.isnan(y[2:]).all()
    assert invalid == 2


def test_mismatched_background_grid_interpolates_without_extrapolation():
    _, y, invalid = export.absorbance(scan([1, 1, 1]), scan([2, 4]))
    np.testing.assert_allclose(y, np.log10([2, 3, 4]), atol=1e-7)
    assert invalid == 0
    with pytest.raises(ValueError, match='extrapolation'):
        export.absorbance(scan([1, 1]), scan([2, 4], first=3900))


def test_bad_scan_length_or_format_is_an_error():
    broken = {**scan([1, 2]), 'NumberOfDataPoints': '3'}
    with pytest.raises(ValueError, match='length'):
        export.decode_scan(broken)
    with pytest.raises(ValueError, match='format'):
        export.decode_scan({**scan([1, 2]), 'DataPointFormat': '2'})


def test_network_package_keeps_repeats_missing_records_and_filter_types_separate(tmp_path):
    import csv
    import gzip
    import hashlib
    import json

    ftir, hips, data = [tmp_path / name for name in ['ftir', 'hips', 'data']]
    for p in [ftir / 'ETAD', hips, data]:
        p.mkdir(parents=True)

    def write_csv(path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)

    filters = [dict(MediaId=str(i), SiteId='15', ExternalFilterId=f'ETAD-{i:04}-1',
                    ExternalFilterType=kind, SamplingStartDate='2023-01-01')
               for i, kind in [(1, 'PM2.5'), (2, 'PM10')]]
    write_csv(hips / 'spartan_Sites.csv', [dict(Id='15', SiteCode='ETAD', Description='Addis Ababa', Latitude='9', Longitude='39')])
    write_csv(hips / 'spartan_Filters.csv', filters)
    write_csv(hips / 'hips_ResultTypes.csv', [dict(Id='0', Name='Sample'), dict(Id='1', Name='Reference')])
    write_csv(hips / 'hips_SampleAnalysis.csv', [dict(Id='5', MediaId='1', AnalysisQcCode='1')])
    readings = [dict(Id=str(i), SampleAnalysisId='5', ResultTypeId=t, Timestamp='2023-02-01',
                     Transmittance='900', Reflectance='100', TransmittanceRaw='0.9', ReflectanceRaw='0.1', Wavelength='633')
                for i, t in [(1, '0'), (2, '1')]]
    write_csv(hips / 'hips_Results.csv', readings)
    src = ftir / 'ETAD'
    write_csv(src / 'ETAD_filters.csv', filters)
    write_csv(src / 'ETAD_scans_base64.csv', [{**scan([1, 1]), 'Id': '10'}, {**scan([2, 2]), 'Id': '11'}])
    analyses = [dict(Id=str(i), MediaId=str(m), SampleScanId='10', BackgroundScanId='11',
                     AnalysisDate='2023-02-01', AnalysisQcCode='1') for i, m in [(101, 1), (102, 1), (103, 2)]]
    write_csv(src / 'ETAD_ftir_analysis.csv', analyses)
    for bundle in [ftir, hips]:
        files = [dict(File=p.relative_to(bundle).as_posix(), Bytes=p.stat().st_size,
                      SHA256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in bundle.rglob('*.csv')]
        (bundle / 'manifest.json').write_text(json.dumps(dict(Analyses=3, Files=files)))
    chemistry = [dict(id='ETAD-0099', code='ETAD', site='Addis Ababa', date='2023-01-02',
                      excluded=True, exclusion_reason='source exclusion', **{'EC (FTIR)': 2.3, 'HIPS Fabs': 42.})]
    (data / 'spartan_reference.json').write_text(json.dumps(dict(rows=chemistry, fields=['EC (FTIR)', 'HIPS Fabs'], field_units={})))
    (data / 'filters.json').write_text(json.dumps(dict(rows=chemistry)))
    (data / 'spartan_map.json').write_text(json.dumps(dict(sites=[dict(code='ETAD', name='Addis Ababa', lat=9, lon=39)])))
    export.build(ftir, hips, data, baselines_dir=tmp_path / 'no-baselines')
    index = json.loads((data / 'spartan_network/index.json').read_text())
    site = index['sites'][0]
    rows = json.loads(gzip.decompress((data / site['filters']).read_bytes()))['rows']
    assert {r['id'] for r in rows} == {'ETAD-0001', 'ETAD-0099'}
    known = next(r for r in rows if r['id'] == 'ETAD-0099')
    assert known['excluded'] and known['exclusion_reason'] == 'source exclusion'
    assert known['EC (FTIR)'] == 2.3 and known['HIPS Fabs'] == 42
    assert 'HIPS Fabs' not in next(r for r in rows if r['id'] == 'ETAD-0001')
    native = json.loads(gzip.decompress((data / site['native']).read_bytes()))
    assert len(native['spectra']['ETAD-0001']['analyses']) == 2
    assert native['spectra']['ETAD-0002']['analyses'][0]['filter_type'] == 'PM10'
    assert len(native['hips']['ETAD-0001']) == 1  # reference reading never masquerades as a sample
    entry = native['spectra']['ETAD-0001']
    body = gzip.decompress((data / entry['file']).read_bytes())
    raws = [a['series']['raw'] for a in entry['analyses']]
    assert [r['offset'] for r in raws] == [0, 8]  # bytes: two 2-point traces of 4-byte steps
    for r in raws:
        np.testing.assert_allclose(codec.decode(body, r, r['offset']), np.log10(2), atol=1e-8)
    assert all(set(a['series']) == {'raw'} for a in entry['analyses'])  # no baselines folder: raw only
    assert index['baselines']['airspec'] is None and index['baselines']['vibes'] is None


def test_spectrum_codec_round_trips_within_half_a_step_and_keeps_gaps():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1e-3, 2500)) + 0.3
    y[[0, 1, 700, 701, 2499]] = np.nan
    body, desc = codec.encode(y)
    assert len(body) == 4 * len(y)
    assert desc['nan'] == [[0, 2], [700, 2], [2499, 1]]
    back = codec.decode(body, desc)
    np.testing.assert_array_equal(np.isnan(back), np.isnan(y))
    assert np.nanmax(np.abs(back - y)) <= codec.DEFAULT_STEP / 2 + 1e-15
    coarse, d = codec.encode(y, step=1e-7)
    assert np.nanmax(np.abs(codec.decode(coarse, d) - y)) <= 5e-8 + 1e-15
