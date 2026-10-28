"""Coverage must separate optical methods and preserve unknown source states."""
import csv
import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'research/ftir_ec_phase3/scripts/audit_spartan_network_coverage.py'
spec = importlib.util.spec_from_file_location('spartan_coverage_audit', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def public_row(site='OLDD', fid='OLDD-0001', parameter='Equivalent BC PM2.5', method=213,
               description='Smoke Stain Reflectometer ', value=0, year=2014, flag=''):
    return dict(Site_Code=site, Filter_ID=fid, Parameter_Name=parameter, Method_Code=method,
                Analysis_Description=description, Value=value, Start_Year_local=year,
                Start_Month_local=1, Start_Day_local=2, Start_hour_local=9,
                Units='Micrograms per cubic meter (ug/m3)', Flag=flag)


def write_public(tmp_path, rows):
    folder = tmp_path / 'public'
    folder.mkdir()
    with (folder / 'FilterBased_ChemSpecPM25_TEST.csv').open('w', newline='') as stream:
        stream.write('# File updated: fixture\n# Public data fixture\n')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return folder


def write_native(tmp_path, rows, status='ok'):
    folder = tmp_path / 'native'
    folder.mkdir()
    path = folder / 'spartan_native_filter_coverage.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame([dict(Dataset='spartan_native_filter_coverage', Status=status)]).to_csv(folder / 'exports.csv', index=False)
    pd.DataFrame([dict(File=path.name, CsvRows=len(rows), SHA256=hashlib.sha256(path.read_bytes()).hexdigest())]).to_csv(folder / 'manifest.csv', index=False)
    return folder


def native_row(media=1, fid='OLDD-0001-1', kind='PM2.5'):
    return dict(MediaId=media, ExternalFilterId=fid, ExternalFilterType=kind, SiteCode='OLDD',
                SamplingStartDate='2014-01-02', FtirAnalyses=2, FtirAnalysesWithPairedScans=1,
                HipsAnalyses=1, HipsSampleReadings=0, HipsSampleAnalyses=0, HipsSampleReadingsWithTR=0)


def test_ssr_curve_and_direct_hips_are_distinct_and_unknown_is_not_zero(tmp_path):
    rows = [public_row(flag='QC flag retained'),
            public_row(parameter='BC PM2.5', method=220, description='Estimated from the HIPS-SSR BC curve'),
            public_row(site='NEWX', fid='NEWX-0001', parameter='BC PM2.5', method=219, description='HIPS', year=2023),
            public_row(site='FUTR', fid='FUTR-0001', parameter='EC PM2.5', method=217, description='FTIR', year=2030),
            public_row(site='BADM', fid='BADM-0001', parameter='BC PM2.5', method=999, description='HIPS')]
    out = tmp_path / 'out'
    summary = audit.audit(write_public(tmp_path, rows), out, as_of='2026-10-28')
    data = pd.read_csv(out / 'filter_coverage.csv').set_index('filter_id')
    old = data.loc['OLDD-0001']
    assert old.ssr_equivalent_bc and old.hips_ssr_estimate
    assert not old.public_hips_bc and not old.direct_hips_observed
    assert old.ssr_in_legacy_hips_group and old.public_flagged_rows == 1
    assert data.lab_hips_fabs.isna().all() and data.native_filter.isna().all()
    assert data.native_ftir_analyses.isna().all()
    assert data.loc['NEWX-0001'].public_hips_bc
    assert data.loc['FUTR-0001'].future_date  # future records remain inspectable
    assert not data.loc['BADM-0001'].public_hips_bc
    assert summary['unclassified_rows'] == 1
    for entry in pd.read_csv(out / 'manifest.csv').itertuples():
        path = out / entry.File
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == entry.SHA256
        if path.suffix == '.csv':
            assert len(pd.read_csv(path)) == entry.CsvRows


def test_lab_only_filters_and_date_conflicts_are_retained(tmp_path):
    public = write_public(tmp_path, [public_row()])
    lab = tmp_path / 'hips.csv'
    pd.DataFrame([
        dict(FilterId='OLDD-0001-1', Site='OLDD', SampleDate='2014-01-03', FilterType='PM2.5', Fabs=10),
        dict(FilterId='ONLY-0001', Site='ONLY', SampleDate='2025-01-01', FilterType='PM2.5', Fabs=0),
        dict(FilterId='ONLY-0002', Site='ONLY', SampleDate='2025-01-02', FilterType='PM10', Fabs=99),
        dict(FilterId='ONLY-0003-7', Site='ONLY', SampleDate='2025-01-02', FilterType='PM2.5', Fabs=99),
    ]).to_csv(lab, index=False)
    out = tmp_path / 'out'
    audit.audit(public, out, hips_file=lab)
    data = pd.read_csv(out / 'filter_coverage.csv').set_index('filter_id')
    assert set(data.index) == {'OLDD-0001', 'ONLY-0001'}
    assert data.loc['OLDD-0001'].date_conflict
    assert not data.loc['OLDD-0001'].ssr_in_legacy_hips_group
    assert data.loc['ONLY-0001'].lab_hips_fabs and not data.loc['ONLY-0001'].public_filter
    sites = pd.read_csv(out / 'site_coverage.csv').set_index('site')
    assert not sites.loc['ONLY'].public_source_available
    assert 'unknown' in sites.loc['ONLY'].suggested_ec_message


def test_native_repeats_are_aggregated_without_counting_reference_only_as_sample(tmp_path):
    folder = write_native(tmp_path, [native_row(), native_row(media=2, fid='OLDD-0001-2'), native_row(media=3, fid='OLDD-0009', kind='PM10')])
    data, _ = audit.load_native(folder)
    assert list(data.index) == ['OLDD-0001']
    assert data.loc['OLDD-0001'].native_media_rows == 2
    assert data.loc['OLDD-0001'].native_ftir_analyses == 4
    assert data.loc['OLDD-0001'].native_hips_analyses == 2
    assert data.loc['OLDD-0001'].native_hips_sample_analyses == 0


def test_failed_or_corrupted_native_coverage_does_not_become_absence(tmp_path):
    folder = write_native(tmp_path, [native_row()], status='skipped_missing_sample_result_type')
    with pytest.raises(ValueError, match='absence is unknown'):
        audit.load_native(folder)
    pd.DataFrame([dict(Dataset='spartan_native_filter_coverage', Status='ok')]).to_csv(folder / 'exports.csv', index=False)
    with (folder / 'spartan_native_filter_coverage.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='hash differs'):
        audit.load_native(folder)


def test_duplicate_native_media_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='repeats a MediaId'):
        audit.load_native(write_native(tmp_path, [native_row(), native_row()]))


def test_fg_request_checks_identities_without_inventing_missing_groups(tmp_path):
    folder = tmp_path / 'reference'
    folder.mkdir()
    pd.DataFrame([dict(Dataset='spartan_filter_batches', Status='ok')]).to_csv(folder / 'exports.csv', index=False)
    pd.DataFrame([dict(ExternalFilterType='PM2.5', ExternalFilterId='OLDD-0001-1', MediaId='3', BatchLabel='Shipment_A')]).to_csv(folder / 'spartan_filter_batches.csv', index=False)
    request = tmp_path / 'request.csv'
    pd.DataFrame([
        dict(filter_id='OLDD-0001', shipment='Shipment_A', site='OLDD', media_id='3'),
        dict(filter_id='OLDD-0002', shipment='not_in_database', site='OLDD', media_id=''),
    ]).to_csv(request, index=False)
    public = pd.DataFrame([dict(filter_id='OLDD-0001', om_ftir=True), dict(filter_id='OLDD-0002', om_ftir=True)])
    data = audit.crosscheck_fg_request(request, folder, public, set()).set_index('filter_id')
    assert data.loc['OLDD-0001'].registered_in_reference
    assert data.loc['OLDD-0001'].requested_media_id_matches
    assert data.loc['OLDD-0001'].requested_shipment_matches
    assert not data.loc['OLDD-0002'].registered_in_reference
    assert data.public_om_reported.all() and not data.known_lab_fg_present.any()
    unknown = audit.crosscheck_fg_request(request, folder, public, None)
    assert unknown.known_lab_fg_present.isna().all()
