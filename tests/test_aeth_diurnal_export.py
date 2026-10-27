"""Offline checks for gallery/data/export_aeth_diurnal.py on small synthetic minute and retrieval frames."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_aeth_diurnal', ROOT / 'gallery/data/export_aeth_diurnal.py')
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

CH = ['UV', 'Blue', 'Green', 'Red', 'IR']


def minutes(days=2, start='2023-01-01', level=lambda h: 1000.0 + 100 * h):
    """`days` whole local days of minute BCc (ng/m3): every channel equal to level(hour)."""
    t = pd.date_range(start, periods=days * 1440, freq='min')
    v = np.array([level(h) for h in t.hour], dtype=float)
    return pd.DataFrame({f'{c} BCc': v for c in CH}, index=t)


def test_hourly_table_means_counts_and_absorption_aae():
    m = minutes(days=1)
    # an aerosol with a true absorption AAE of 1.4 between UV and IR: scale UV BCc so b_ATN follows it
    wl = ex.WAVELENGTHS_NM
    target = 1.4
    m['UV BCc'] = m['IR BCc'] * ex.SIGMA['IR'] / ex.SIGMA['UV'] * (wl['UV'] / wl['IR']) ** -target
    m.iloc[:30, m.columns.get_loc('IR BCc')] = np.nan            # half of hour 0 missing in IR
    out = ex.hourly_table(m)
    assert len(out) == 24 and out.day.unique().tolist() == ['2023-01-01']
    h0, h5 = out.set_index('hour').loc[0], out.set_index('hour').loc[5]
    assert h0.n == 30 and h5.n == 60
    assert h5['Red BCc'] == pytest.approx(1.5)                    # 1000 + 500 ng/m3 -> ug/m3
    assert h5.aae_uv_ir == pytest.approx(target)
    # equal BCc in Blue and IR is not AAE 0: the cross-sections differ (optics.aae_from_columns)
    assert h5.aae_blue_ir == pytest.approx(-np.log(ex.SIGMA['Blue'] / ex.SIGMA['IR']) / np.log(wl['Blue'] / wl['IR']))


def test_qc_drops_negatives_and_spikes_only():
    m = minutes(days=1, level=lambda h: 1000.0)
    m.iloc[10, 0] = -5.0
    m.iloc[20, 0] = 1e6
    out = ex.qc(m)
    assert np.isnan(out.iloc[10, 0]) and np.isnan(out.iloc[20, 0])
    assert out.iloc[:, 0].notna().sum() == 1440 - 2
    assert out.iloc[:, 1].notna().all()


def test_coverage_ratio_is_24h_over_aeronet_hours():
    # 2000 ng/m3 from 07:00 to 12:59, 1000 otherwise: 24 h mean = (6*2000 + 18*1000) / 24
    hours = [7, 8, 9, 10, 11, 12]
    m = minutes(days=2, level=lambda h: 2000.0 if h in hours else 1000.0)
    out = ex.coverage_ratios(m['Red BCc'], hours)
    assert len(out) == 2 and out.minutes.tolist() == [1440, 1440] and out.minutes_in.tolist() == [360, 360]
    assert out.ratio_mean.tolist() == pytest.approx([(6 * 2000 + 18 * 1000) / 24 / 2000] * 2)
    assert out.ratio_median.tolist() == pytest.approx([0.5, 0.5])


def test_coverage_ratio_skips_short_days():
    m = minutes(days=1).iloc[:900]                                   # 900 minutes < 1000
    assert ex.coverage_ratios(m['Red BCc'], list(range(6, 13))).empty
    m = minutes(days=1)                                              # 240 minutes inside the hours < 300
    assert ex.coverage_ratios(m['Red BCc'], [7, 8, 9, 10]).empty


def test_window_mean_takes_the_minutes_within_15_min():
    t = pd.date_range('2023-01-01', periods=120, freq='min').to_numpy()
    v = np.arange(120, dtype=float)
    v[60] = np.nan
    mean, n = ex.window_mean(t, v, np.array([np.datetime64('2023-01-01T01:00:00')]), np.timedelta64(15, 'm'))
    assert n[0] == 30                                               # 45..75 less the missing minute
    assert mean[0] == pytest.approx((sum(range(45, 76)) - 60) / 30)


def test_matched_tables_convert_retrieval_time_to_local_and_aggregate():
    m = minutes(days=1, start='2023-01-01', level=lambda h: 4000.0 if h == 9 else 1000.0)
    # 06:20 and 06:30 UTC are 09:20 and 09:30 at UTC+3, their +-15 min inside hour 9; 03:00 UTC is 06:00 local
    aod = pd.DataFrame({'time_utc': pd.to_datetime(['2023-01-01 03:00', '2023-01-01 06:20', '2023-01-01 06:30']),
                        'aod500': [0.2, 0.4, 0.4], 'ae440_870': [1.0, 1.2, 1.4]})
    hourly, daily = ex.matched_tables(m, aod, 3.0)
    h = hourly.set_index('hour')
    assert h.loc[9].n == 2 and h.loc[9].bc_ir == pytest.approx(4.0) and h.loc[9].aod500 == pytest.approx(0.4)
    assert h.loc[9].bc_aod == pytest.approx(10.0) and h.loc[9].ae == pytest.approx(1.3)
    assert h.loc[6].bc_ir == pytest.approx(1.0) and h.loc[6].bc_aod == pytest.approx(5.0)
    assert len(daily) == 1 and daily.n.iloc[0] == 3
    bc, od = np.array([1.0, 4.0, 4.0]), np.array([0.2, 0.4, 0.4])
    assert daily.bc_cv.iloc[0] == pytest.approx(bc.std(ddof=1) / bc.mean())
    assert daily.aod_cv.iloc[0] == pytest.approx(od.std(ddof=1) / od.mean())


def test_matched_tables_need_five_valid_minutes():
    m = minutes(days=1)
    m.loc['2023-01-01 08:40':'2023-01-01 09:20', 'IR BCc'] = np.nan
    m.loc['2023-01-01 09:12':'2023-01-01 09:15', 'IR BCc'] = 1000.0     # 4 valid minutes in the window
    aod = pd.DataFrame({'time_utc': pd.to_datetime(['2023-01-01 06:00']), 'aod500': [0.3], 'ae440_870': [1.0]})
    hourly, _ = ex.matched_tables(m, aod, 3.0)
    assert hourly.empty


def test_column_tables_local_hours_and_inversion_hours():
    t = pd.to_datetime(['2023-01-01 04:00', '2023-01-01 04:20', '2023-01-01 12:00', '2023-01-03 05:00'])
    aod = pd.DataFrame({'time_utc': t, 'aod500': [0.1, 0.3, 0.5, 0.2], 'ae440_870': [1.0, 1.2, 1.4, 1.1]})
    sda = pd.DataFrame({'time_utc': t[:2], 'fmf500': [0.5, 0.7]})
    inv = pd.DataFrame({'time_utc': pd.to_datetime(['2023-01-01 05:00'] * 30 + ['2023-01-01 13:00'])})
    out = ex.column_tables(aod, sda, inv, 3.0)
    assert out['first'] == '2023-01-01' and out['last'] == '2023-01-03' and out['days'] == 2
    a = out['aod']
    assert np.cumsum(a['dd']).tolist() == [0, 0, 2] and a['h'] == [7, 15, 8] and a['n'] == [2, 1, 1]
    assert a['aod500'] == [200, 500, 200] and a['ae'] == [110, 140, 110] and a['fmf'] == [60, None, None]
    assert out['inv_hours'] == [8, 16]                              # 1 of 31 is >= 2 %
    assert out['hour_counts']['aod'][7] == 2 and sum(out['hour_counts']['aod']) == 4 and out['hour_counts']['inv'][8] == 30
    assert out['hours_per_day'] == 1.5                              # 2 hours on the 1st, 1 on the 3rd
    assert ex.inversion_hours(pd.Series({8: 99, 16: 1})) == [8]       # 1 % is not


def test_read_minutes_parses_12_hour_csv_and_undoes_daylight_saving(tmp_path):
    p = tmp_path / 'm.csv'
    pd.DataFrame({ex.CSV_DATE: ['2023/07/01', '2023/07/01'], ex.CSV_TIME: ['12:30:00 AM', '01:15:00 PM'],
                  ex.CSV_TZ: [-420, -420], 'Red BCc': [500, 600], 'IR BCc': [450, 550]}).to_csv(p, index=False)
    out = ex.read_minutes(p, -8.0)
    # 00:30 PDT is 07:30 UTC, 23:30 PST the day before
    assert out.index.tolist() == [pd.Timestamp('2023-06-30 23:30'), pd.Timestamp('2023-07-01 12:15')]
    assert out.columns.tolist() == ['Red BCc', 'IR BCc'] and out['Red BCc'].tolist() == [500.0, 600.0]


def test_brc_test_on_an_aae_one_aerosol_finds_no_brown_carbon():
    wl, s = ex.WAVELENGTHS_NM, ex.SIGMA
    ir = np.array([5000.0, 8000.0, 6000.0])
    # BCc such that absorption follows lambda^-1 exactly at every channel
    daily = pd.DataFrame({f'{c} BCc': ir * s['IR'] / s[c] * (wl[c] / wl['IR']) ** -1.0 for c in CH})
    daily['day_9am'] = pd.date_range('2023-01-01', periods=3)
    out = ex.brc_test(daily)
    assert out['summary']['aae_red_ir']['mean'] == pytest.approx(1.0)
    assert max(abs(v) for v in out['brc']) < 1e-3
    assert out['summary']['uv']['out_of_range_pct'] == 0
    assert out['summary']['target']['mean'] == pytest.approx(21.716, abs=1e-3)   # ftir_28: 21.72


def test_surface_payload_round_trips_to_json_with_day_steps():
    m = minutes(days=3, start='2023-03-01')
    m.loc['2023-03-02'] = np.nan                                      # a day with no record
    aod = pd.DataFrame({'time_utc': pd.to_datetime(['2023-03-01 05:00', '2023-03-03 05:00', '2023-03-03 06:00', '2023-03-03 07:00']),
                        'aod500': [0.2, 0.3, 0.4, 0.5], 'ae440_870': [1.0, 1.1, 1.2, 1.3]})
    out = ex.surface_payload(m, aod, list(range(6, 13)), 3.0)
    json.dumps(out, allow_nan=False)
    assert out['first'] == '2023-03-01' and out['last'] == '2023-03-03' and out['days'] == 2
    h = out['hourly']
    assert len(h['h']) == 48 and sorted(set(np.cumsum(h['dd']).tolist())) == [0, 2]
    assert set(h) >= {'UV', 'Blue', 'Green', 'Red', 'IR', 'n', 'aae_uv_ir', 'aae_blue_ir'}
    assert np.cumsum(out['matched']['dd']).tolist() == [0, 2, 2, 2] and out['matched']['h'] == [8, 8, 9, 10]
    assert np.cumsum(out['intraday']['dd']).tolist() == [2]
    assert np.cumsum(out['coverage']['dd']).tolist() == [0, 2] and out['coverage']['hours'] == list(range(6, 13))


def test_read_minutes_prefers_the_utc_column(tmp_path):
    # the Jacros and Pasadena exports: "Time local" is the UTC clock and the offset is 0
    p = tmp_path / 'm.csv'
    pd.DataFrame({ex.CSV_UTC: ['2022-04-12T09:46:01Z', '2022-04-12T09:47:01Z'], ex.CSV_TZ: [0, 0],
                  ex.CSV_DATE: ['2022-04-12', '2022-04-12'], ex.CSV_TIME: ['9:46:01 AM', '9:47:01 AM'],
                  'IR BCc': [100, 200]}).to_csv(p, index=False)
    out = ex.read_minutes(p, 3.0)
    assert out.index.tolist() == [pd.Timestamp('2022-04-12 12:46:01'), pd.Timestamp('2022-04-12 12:47:01')]


def test_read_minutes_converts_an_aware_pickle_to_standard_time(tmp_path):
    p = tmp_path / 'm.pkl'
    t = pd.to_datetime(['2023-07-01 00:30', '2023-01-10 00:30']).tz_localize('America/Los_Angeles')
    pd.DataFrame({'datetime_local': t, 'IR BCc': [1.0, 2.0], 'Other': [0, 0]}).to_pickle(p)
    out = ex.read_minutes(p, -8.0)
    # 00:30 PDT is 23:30 PST the day before; January is already standard time
    assert out.index.tolist() == [pd.Timestamp('2023-01-10 00:30'), pd.Timestamp('2023-06-30 23:30')]
    assert out.columns.tolist() == ['IR BCc']


def test_hourly_means_give_cycles_and_coverage_but_no_matching(tmp_path):
    # Kyan Data/MAIA data layout: an aware local time, BCc in ng/m3, one row per hour
    t = pd.date_range('2023-01-01 00:00', periods=48, freq='h', tz='Asia/Kolkata')
    hours = [2000.0 if 7 <= h.hour <= 12 else 1000.0 for h in t]
    p = tmp_path / 'Delhi_cleaned_corrected_small_hourly.csv'
    pd.DataFrame({'datetime_local': t.astype(str), **{f'{c} BCc': hours for c in CH}, 'source': 'og'}).to_csv(p)
    raw = ex.read_hourly(p, 5.5)
    assert raw.index[0] == pd.Timestamp('2023-01-01 00:00') and len(raw) == 48
    aod = pd.DataFrame({'time_utc': pd.to_datetime(['2023-01-01 04:00']), 'aod500': [0.5], 'ae440_870': [1.2]})
    out = ex.surface_payload(raw, aod, list(range(7, 13)), 5.5, 'hourly')
    json.dumps(out, allow_nan=False)
    assert out['resolution'] == 'hourly' and 'matched' not in out and 'n' not in out['hourly']
    assert out['hourly']['IR'][7] == 2000                           # ng/m3, as integers
    cov = out['coverage']
    assert cov['minutes'] == [1440, 1440] and cov['minutes_in'] == [360, 360]
    assert cov['ratio_mean'] == pytest.approx([(6 * 2000 + 18 * 1000) / 24 / 2000] * 2, abs=1e-3)
