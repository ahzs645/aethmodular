"""Offline checks for gallery/data/export_aeth_sources.py (the instrument's own AAE per daily bin)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gallery/data'))
spec = importlib.util.spec_from_file_location('export_aeth_sources', ROOT / 'gallery/data/export_aeth_sources.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_instrument_frame_reads_the_first_datetime_column_and_drops_bins_without_an_aae():
    raw = pd.DataFrame([
        ['2023-11-09 15:00:00+05:30', 1.0, 1.2, 10.0, 2000.0, 8000.0, 1.0, 2.0],
        ['2023-11-08 15:00:00+05:30', 2.0, np.nan, np.nan, np.nan, np.nan, 1.0, 2.0],
        ['2023-11-10 15:00:00+05:30', 3.0, 1.4, 30.0, 3000.0, 7000.0, 1.0, 2.0],
    ], columns=['datetime_local', 'datetime_local', 'AAE', 'BB (%)', 'Biomass BCc  (ng/m^3)', 'Fossil fuel BCc  (ng/m^3)',
                'AAE fossil fuel', 'AAE biomass'])
    raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0])   # the second, duplicate column is junk, as in the pickles
    out = export.instrument_frame(raw)
    assert out['start'].dt.strftime('%Y-%m-%d %H:%M').tolist() == ['2023-11-09 15:00', '2023-11-10 15:00']
    assert out['aae'].tolist() == [1.2, 1.4]
    assert out['bb_bcc'].tolist() == [2.0, 3.0]          # ng/m³ -> µg/m³
    assert export.endmember(out['aae_biomass']) == 2.0
    assert export.endmember(pd.Series([1.0, 2.0])) is None


def test_hourly_bins_average_inside_each_bin_and_need_enough_hours():
    t = pd.date_range('2023-01-01 09:00', periods=48, freq='h')
    hourly = pd.DataFrame({'t': t, 'aae': np.r_[np.full(24, 1.0), np.full(10, 2.0), np.full(14, np.nan)],
                           'bb': np.r_[np.full(24, 1.0), np.full(24, 0.0)], 'ff': np.r_[np.full(24, 3.0), np.full(24, 1.0)],
                           'ir': np.ones(48)})
    out = export.hourly_bins(hourly, ['2023-01-01 09:00', '2023-01-02 09:00'], min_hours=18)
    assert out['start'].tolist() == ['2023-01-01 09:00']   # the second bin has 10 hours with an AAE
    assert out['aae'].iloc[0] == 1.0
    assert out['bb_pct'].iloc[0] == 25.0                    # 24 / (24 + 72)
    assert out['hours'].iloc[0] == 24


def hourly_frame(hours=24, start='2023-01-01 09:00', tz='UTC'):
    return pd.DataFrame({'t': pd.date_range(start, periods=hours, freq='h', tz=tz),
                         'aae': np.ones(hours), 'bb': np.ones(hours), 'ff': np.full(hours, 3.0),
                         'ir': np.ones(hours)})


def test_repeated_rows_do_not_meet_eighteen_hour_coverage():
    hourly = pd.concat([hourly_frame(1)] * 18, ignore_index=True)
    out = export.hourly_bins(hourly, ['2023-01-01 09:00'])
    assert out.empty
    rejection = out.attrs['diagnostics']['rejected_bins'][0]
    assert rejection['aae_hours'] == rejection['split_hours'] == 1
    assert rejection['raw_rows'] == 18
    assert rejection['duplicate_rows'] == 17
    assert rejection['conflict_hours'] == 0


def test_equivalent_repeats_count_once_without_changing_means_or_ratio():
    hourly = hourly_frame(18)
    hourly.loc[0, ['aae', 'bb', 'ff']] = [2.0, 3.0, 1.0]
    duplicated = pd.concat([hourly, hourly.iloc[[0, 0, 4]]], ignore_index=True)
    ordinary = export.hourly_bins(hourly, ['2023-01-01 09:00']).iloc[0]
    out = export.hourly_bins(duplicated.iloc[::-1], ['2023-01-01 09:00']).iloc[0]
    assert out['aae'] == ordinary['aae']
    assert out['bb_pct'] == ordinary['bb_pct']
    assert out['hours'] == out['aae_hours'] == out['split_hours'] == 18
    assert out['observed_hours'] == 18
    assert out['duplicate_rows'] == 3
    assert out['conflict_hours'] == 0


def test_one_paired_split_hour_never_inherits_eighteen_valid_aae_hours():
    hourly = hourly_frame(18)
    hourly.loc[1:, ['bb', 'ff']] = np.nan
    out = export.hourly_bins(hourly, ['2023-01-01 09:00']).iloc[0]
    assert out['aae'] == 1.0
    assert out['aae_hours'] == 18
    assert out['split_hours'] == 1
    assert np.isnan(out['bb_pct'])
    assert out['bb_status'] == 'insufficient_paired_hours'


def test_aae_and_complete_paired_split_have_independent_support():
    hourly = hourly_frame()
    hourly.loc[18:, 'aae'] = np.nan
    hourly.loc[:5, ['bb', 'ff']] = np.nan
    out = export.hourly_bins(hourly, ['2023-01-01 09:00']).iloc[0]
    assert out['aae_hours'] == out['split_hours'] == 18
    assert out['ir_hours'] == out['observed_hours'] == 24
    assert out['bb_pct'] == 25.0
    assert out['bb_status'] == 'available'


def test_nonfinite_values_do_not_count_toward_any_metric_coverage():
    hourly = hourly_frame()
    hourly.loc[18:, 'aae'] = np.inf
    hourly.loc[17:, 'bb'] = -np.inf
    hourly.loc[2:, 'ir'] = np.inf
    out = export.hourly_bins(hourly, ['2023-01-01 09:00'])
    row = out.iloc[0]
    assert row['aae'] == 1.0
    assert np.isnan(row['ir'])  # the IR comparison also needs its own coverage
    assert row['aae_hours'] == 18
    assert row['split_hours'] == 17
    assert row['ir_hours'] == 2
    assert np.isnan(row['bb_pct'])
    assert out.attrs['diagnostics']['input']['nonfinite_or_missing_rows'] == {
        'aae': 6, 'bb': 7, 'ff': 0, 'ir': 22,
    }
    hourly['aae'] = np.inf
    assert export.hourly_bins(hourly, ['2023-01-01 09:00']).empty
    assert export.endmember(pd.Series([np.inf, np.nan])) is None


def test_conflicting_split_duplicate_does_not_destroy_agreed_aae():
    hourly = hourly_frame(18)
    conflict = hourly.iloc[[0]].copy()
    conflict['bb'] = 9.0
    combined = pd.concat([hourly, conflict], ignore_index=True)
    out = export.hourly_bins(combined, ['2023-01-01 09:00']).iloc[0]
    assert out['aae'] == 1.0
    assert out['aae_hours'] == 18
    assert out['split_hours'] == 17
    assert np.isnan(out['bb_pct'])
    assert out['duplicate_rows'] == out['conflict_hours'] == out['split_conflict_hours'] == 1
    assert out['aae_conflict_hours'] == out['ir_conflict_hours'] == 0
    reversed_out = export.hourly_bins(combined.iloc[::-1], ['2023-01-01 09:00']).iloc[0]
    pd.testing.assert_series_equal(out, reversed_out)


def test_complementary_missing_duplicates_never_synthesize_a_split():
    hourly = hourly_frame(18)
    extra = hourly.iloc[[0]].copy()
    hourly.loc[0, 'ff'] = np.nan
    extra['bb'] = np.nan
    combined = pd.concat([hourly, extra], ignore_index=True)
    out = export.hourly_bins(combined, ['2023-01-01 09:00']).iloc[0]
    assert out['aae_hours'] == 18
    assert out['split_hours'] == 17
    assert out['split_conflict_hours'] == 1
    assert np.isnan(out['bb_pct'])


def test_missing_versus_finite_duplicate_is_reported_as_conflict_without_source_priority():
    hourly = hourly_frame(19)
    extra = hourly.iloc[[0]].copy()
    extra['aae'] = np.nan
    out = export.hourly_bins(pd.concat([hourly, extra]), ['2023-01-01 09:00']).iloc[0]
    assert out['aae_hours'] == 18
    assert out['split_hours'] == 19
    assert out['aae_conflict_hours'] == 1
    assert out['bb_pct'] == 25.0


@pytest.mark.parametrize(('start', 'start_utc', 'end_local'), [
    ('2023-11-04 15:00', '2023-11-04T22:00:00+00:00', '2023-11-05 14:00'),
    ('2023-03-11 15:00', '2023-03-11T23:00:00+00:00', '2023-03-12 16:00'),
])
def test_dst_offsets_preserve_distinct_hours_and_legacy_24_elapsed_hour_bins(tmp_path, start, start_utc, end_local):
    hourly = hourly_frame(25, start=start_utc)
    raw = hourly.rename(columns={'t': 'datetime_local', 'aae': 'AAE calculated', 'bb': 'Biomass BCc',
                                 'ff': 'Fossil fuel BCc', 'ir': 'IR BCc'})
    raw['datetime_local'] = hourly['t'].dt.tz_convert('America/Los_Angeles').astype(str)
    raw['IR BCc'] *= 1000
    path = tmp_path / 'JPL_hourly.csv'
    raw.to_csv(path, index=False)
    read = export.read_hourly(path, 'America/Los_Angeles')
    assert str(read['t'].dt.tz) == 'UTC'
    assert read['t'].nunique() == 25
    out = export.hourly_bins(read, [start], site_timezone='America/Los_Angeles').iloc[0]
    assert out['start'] == start
    assert out['start_utc'] == start_utc
    assert out['aae_hours'] == out['split_hours'] == out['observed_hours'] == 24
    assert out['duplicate_rows'] == 0
    assert out['ir'] == 1.0
    assert pd.Timestamp(out['end_utc']).tz_convert('America/Los_Angeles').strftime('%Y-%m-%d %H:%M') == end_local


def test_half_hour_zone_uses_actual_instants_and_excludes_exact_bin_end():
    hourly = hourly_frame(26, start='2023-01-01 14:00', tz='Asia/Kolkata')
    out = export.hourly_bins(hourly, ['2023-01-01 15:00'], site_timezone='Asia/Kolkata').iloc[0]
    assert out['start_utc'] == '2023-01-01T09:30:00+00:00'
    assert out['end_utc'] == '2023-01-02T09:30:00+00:00'
    assert out['aae_hours'] == out['split_hours'] == 24
    naive = hourly.copy()
    naive['t'] = naive['t'].dt.tz_localize(None)
    same = export.hourly_bins(naive, ['2023-01-01 15:00'], site_timezone='Asia/Kolkata').iloc[0]
    pd.testing.assert_series_equal(out, same)


def test_different_offset_strings_for_same_utc_hour_are_one_observation():
    hourly = hourly_frame(18)
    hourly['t'] = hourly['t'].astype(str)
    repeat = hourly.iloc[[0]].copy()
    repeat['t'] = '2023-01-01 17:00:00+08:00'
    out = export.hourly_bins(pd.concat([hourly, repeat]), ['2023-01-01 09:00']).iloc[0]
    assert out['aae_hours'] == out['split_hours'] == 18
    assert out['raw_rows'] == 19
    assert out['duplicate_rows'] == 1
    assert out['conflict_hours'] == 0


def test_ambiguous_or_nonexistent_naive_timestamps_are_flagged_not_guessed(tmp_path):
    path = tmp_path / 'hourly.csv'
    pd.DataFrame({'datetime_local': ['2023-11-05 01:00', '2023-03-12 02:00', 'broken',
                                    '2023-11-05 01:00:00-07:00', '2023-11-05 01:00:00-08:00',
                                    '2023-11-05 03:00'],
                  'AAE calculated': np.ones(6)}).to_csv(path, index=False)
    out = export.read_hourly(path, 'America/Los_Angeles')
    assert out['t'].notna().sum() == 3
    assert out['t'].dropna().dt.strftime('%H:%M').tolist() == ['08:00', '09:00', '11:00']
    assert out.attrs['input_diagnostics']['invalid_timestamp_rows'] == 3
    bins = export.hourly_bins(out, ['2023-11-05 01:00', '2023-11-05 00:00'], min_hours=1,
                             site_timezone='America/Los_Angeles')
    assert bins.attrs['diagnostics']['invalid_bin_starts'] == ['2023-11-05 01:00']
    assert bins.attrs['diagnostics']['input']['invalid_timestamp_rows'] == 3
    assert bins.iloc[0]['aae_hours'] == 3


def test_missing_split_columns_keep_aae_and_report_unavailable_biomass(tmp_path):
    path = tmp_path / 'hourly.csv'
    pd.DataFrame({'datetime_local': pd.date_range('2023-01-01 09:00', periods=18, freq='h'),
                  'AAE calculated': np.ones(18)}).to_csv(path, index=False)
    read = export.read_hourly(path, 'Asia/Shanghai')
    out = export.hourly_bins(read, ['2023-01-01 09:00'], site_timezone='Asia/Shanghai')
    assert out.iloc[0]['aae'] == 1.0
    assert out.iloc[0]['aae_hours'] == 18
    assert out.iloc[0]['split_hours'] == out.iloc[0]['ir_hours'] == 0
    assert np.isnan(out.iloc[0]['bb_pct'])
    assert out.attrs['diagnostics']['input']['missing_columns'] == ['Biomass BCc', 'Fossil fuel BCc', 'IR BCc']


def test_nonpositive_total_split_is_not_a_valid_percentage():
    hourly = hourly_frame(18)
    hourly['bb'] = -3.0
    out = export.hourly_bins(hourly, ['2023-01-01 09:00']).iloc[0]
    assert out['split_hours'] == 18
    assert np.isnan(out['bb_pct'])
    assert out['bb_status'] == 'nonpositive_split_total'


def test_daily_start_keys_survive_mixed_offsets():
    raw = pd.DataFrame({'datetime_local': ['2023-11-04 15:00:00-07:00', '2023-11-05 15:00:00-08:00'],
                        'AAE': [1.0, 1.1]})
    out = export.instrument_frame(raw)
    assert out['start'].dt.strftime('%Y-%m-%d %H:%M').tolist() == ['2023-11-04 15:00', '2023-11-05 15:00']


def test_export_preserves_legacy_keys_and_serializes_independent_coverage(tmp_path, monkeypatch):
    processed = tmp_path / 'processed'
    processed.mkdir()
    out_dir = tmp_path / 'aethalometer'
    (out_dir / 'daily').mkdir(parents=True)
    start = '2023-11-04 15:00'
    pd.DataFrame({'datetime_local': [start], 'AAE': [1.2], 'BB (%)': [25.0],
                  'AAE fossil fuel': [1.0], 'AAE biomass': [2.0]}).to_pickle(processed / 'daily.pkl')
    (out_dir / 'daily/USPA.json').write_text(json.dumps({'start': [start], 'IR': [1.0]}))
    hourly = hourly_frame(18, start='2023-11-04 22:00')
    hourly.loc[1:, ['bb', 'ff']] = np.nan
    path = tmp_path / 'JPL_hourly.csv'
    hourly.rename(columns={'t': 'datetime_local', 'aae': 'AAE calculated', 'bb': 'Biomass BCc',
                           'ff': 'Fossil fuel BCc', 'ir': 'IR BCc'}).to_csv(path, index=False)
    monkeypatch.setattr(export, 'SITE_FILES', {'USPA': ('JPL', 'daily.pkl')})
    monkeypatch.setattr(export, 'PROCESSED', processed)
    monkeypatch.setattr(export, 'OUT', out_dir)
    monkeypatch.setattr(export, 'DEST', out_dir / 'sources/instrument.json')
    monkeypatch.setattr(export, 'maia_file', lambda _: path)
    export.main()
    payload = json.loads(export.DEST.read_text())
    series = payload['sites']['USPA']['hourly']
    assert series['start'] == payload['sites']['USPA']['daily']['start'] == [start]
    assert series['aae'] == [1.0]
    assert series['bb_pct'] == [None]
    assert series['hours'] == series['aae_hours'] == [18]
    assert series['split_hours'] == [1]
    assert series['timezone'] == 'America/Los_Angeles'
    assert series['coverage_version'] == 2
    assert series['start_utc'] == ['2023-11-04T22:00:00+00:00']
    assert series['bb_status'] == ['insufficient_paired_hours']
    assert 'NaN' not in export.DEST.read_text()
