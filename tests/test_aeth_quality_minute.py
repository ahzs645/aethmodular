"""The minute-level QC export (gallery/data/export_aeth_quality_minute.py) on synthetic MA350 records."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gallery/data'))
spec = importlib.util.spec_from_file_location('export_aeth_quality_minute', ROOT / 'gallery/data/export_aeth_quality_minute.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def frame(n=600, start='2023-03-01 00:00', tape=None, session=None, bcc=None):
    tu = pd.date_range(start, periods=n, freq='min')
    return pd.DataFrame({
        'tu': tu, 't': tu + pd.Timedelta(hours=3),
        'Session ID': session if session is not None else np.ones(n),
        'Tape position': tape if tape is not None else np.ones(n),
        'IR BCc': bcc if bcc is not None else np.full(n, 1000.0),
    })


def test_read_minutes_takes_utc_converts_to_the_zone_and_keeps_the_repeated_dst_hour(tmp_path):
    # the raw files' "Time local" is the UTC clock; at the November change Los Angeles repeats 01:00-02:00
    utc = ['2023-11-05T09:30:00Z', '2023-11-05T08:30:00Z', '2023-11-05T09:30:00Z', '2023-11-05T10:00:00Z']
    csv = pd.DataFrame({'Time (UTC)': utc, 'Time local (hh:mm:ss)': ['x'] * 4, 'Tape position': [1, 1, 1, 2],
                        'Flow1 (mL/min)': [60, 61, 62, 63], 'IR BCc': [1, 2, 3, 4]})
    path = tmp_path / 'ma350.csv'
    csv.to_csv(path, index=False)
    df = m.read_minutes(path, 'America/Los_Angeles')
    assert df['tu'].tolist() == list(pd.to_datetime(['2023-11-05 08:30', '2023-11-05 09:30', '2023-11-05 10:00']))
    # 08:30 UTC is 01:30 PDT, 09:30 UTC is 01:30 PST: two minutes, one wall-clock time, both kept
    assert df['t'].tolist() == list(pd.to_datetime(['2023-11-05 01:30', '2023-11-05 01:30', '2023-11-05 02:00']))
    assert df['IR BCc'].tolist() == [2, 1, 4]       # the first of the duplicated stamps is kept
    assert 'Time local (hh:mm:ss)' not in df.columns


def test_segments_restart_at_a_tape_or_session_change_and_dema_restarts_with_them():
    tape = np.r_[np.ones(100), np.full(100, 2.0), np.full(100, 2.0)]
    session = np.r_[np.ones(200), np.full(100, 7.0)]
    bcc = np.r_[np.full(100, 1000.0), np.full(200, 5000.0)]
    df = m.add_segments(frame(300, tape=tape, session=session, bcc=bcc))
    assert df['segment'].tolist() == [1] * 100 + [2] * 100 + [3] * 100
    assert df['age_min'].iloc[[0, 99, 100, 250]].tolist() == [0, 99, 0, 50]
    assert df['to_next_min'].iloc[99] == 0
    smooth = m.dema(df['IR BCc'], df['segment'])
    # restarted on each spot, the constant segments stay exactly constant: no carry-over across the advance
    assert np.allclose(smooth[:100], 1000) and np.allclose(smooth[100:], 5000)
    s = pd.Series(np.arange(10.0))
    ema = s.ewm(span=10, adjust=False).mean()
    assert np.allclose(m.dema(s, pd.Series(np.ones(10))), 2 * ema - ema.ewm(span=10, adjust=False).mean())


def test_flow_flags_use_the_median_ratio_and_the_two_percent_bands():
    n = 100
    f1 = np.full(n, 60.0)
    f1[:5] = 65.0                                   # R 1.625, 8 % high: warns
    f1[5:10] = 70.0                                 # R 1.75, 17 % high: fails
    df = pd.DataFrame({'t': pd.date_range('2024-01-01', periods=n, freq='min'), 'Flow1 (mL/min)': f1, 'Flow2 (mL/min)': np.full(n, 40.0),
                       'Flow total (mL/min)': f1 + 40, 'Flow setpoint (mL/min)': np.full(n, 100.0)})
    f = m.flow_flags(df)
    assert f.attrs['R_target'] == 1.5
    assert f['R_pass'].sum() == 90 and f['R_warn'].sum() == 5 and f['R_fail'].sum() == 5
    assert f['S_pass'].all()
    assert f['T_pass'].sum() == 90                  # totals of 105 and 110 are more than 2 % off the setpoint
    assert f['QC_pass'].sum() == 90


def test_window_means_need_eight_valid_minutes_either_side():
    t = pd.date_range('2024-01-01', periods=60, freq='min').to_numpy()
    v = np.r_[np.full(30, 1.0), np.full(30, 3.0)]
    pre, post = m.window_means(t, v, np.array([t[30]]))
    assert pre[0] == 1 and post[0] == 3
    v[36:45] = np.nan                               # only 1 valid minute in [+5, +15)
    pre, post = m.window_means(t, v, np.array([t[30]]))
    assert np.isnan(pre[0]) and np.isnan(post[0])


def test_masked_errors_by_kind():
    rng = np.random.default_rng(0)
    M = np.tile(np.arange(1, 101, dtype=float), (3, 1))
    rush = np.zeros(100, dtype=bool)
    assert np.allclose(m.masked_errors(np.ones((3, 100)), 'random', 0.2, rng, rush), 0)
    high = m.masked_errors(M, 'high-pollution', 0.2, rng, rush)      # keeps 1..80: mean 40.5 against 50.5
    low = m.masked_errors(M, 'low-pollution', 0.2, rng, rush)        # keeps 21..100: mean 60.5
    assert np.allclose(high, 100 * (40.5 / 50.5 - 1)) and np.allclose(low, 100 * (60.5 / 50.5 - 1))
    assert len(m.masked_errors(M, 'contiguous', 0.1, rng, rush)) == 3 * m.N_DRAWS


def test_filter_windows_compare_the_day_definitions():
    # two local days of minutes: 1000 ng/m3 before noon on 2 Mar, 3000 after
    t = pd.date_range('2023-03-01 00:00', '2023-03-04 23:59', freq='min')
    ir = np.where(t < pd.Timestamp('2023-03-02 12:00'), 1000.0, 3000.0)
    df = pd.DataFrame({'t': t, 'tu': t - pd.Timedelta(hours=3), 'IR BCc': ir, 'Red BCc': ir})
    w = m.filter_windows(df, {'ETAD-0001': ['2023-03-02 09:00', '2023-03-03 09:00', 24.0, False], 'USPA-0001': ['2023-03-02 09:00', '2023-03-03 09:00', 24.0, False]},
                         'ETAD', 'Africa/Addis_Ababa')
    assert w['id'] == ['ETAD-0001'] and w['date'] == ['2023-03-02']
    assert w['IR_9am'][0] == round((3 * 1000 + 21 * 3000) / 24 / 1000, 4)         # 09:00-12:00 low, then high
    assert w['IR_calendar'][0] == round((12 * 1000 + 12 * 3000) / 24 / 1000, 4)
    assert w['IR_window'][0] == w['IR_9am'][0]                                    # a 09:00-09:00 window is the 9am day
    # +-1 day: the 9am days of 1, 2 and 3 Mar (1 Mar 09:00 - 2 Mar 09:00 is all low)
    first = 1.0
    third = 3.0
    assert w['IR_tolerant'][0] == round((first + w['IR_9am'][0] + third) / 3, 4)
