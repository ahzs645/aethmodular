"""Offline checks for gallery/data/export_aeth_overview.py: clock, QC, hourly means and episodes."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_aeth_overview', ROOT / 'gallery/data/export_aeth_overview.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_standard_offsets_ignore_daylight_saving():
    assert export.standard_offset('Africa/Addis_Ababa') == 3
    assert export.standard_offset('America/Los_Angeles') == -8
    assert export.standard_offset('Asia/Kolkata') == 5.5


def test_raw_csv_is_read_from_the_utc_column(tmp_path):
    # the exports' "Time local" repeats the UTC clock; local standard time must come from Time (UTC)
    csv = tmp_path / 'm.csv'
    csv.write_text('Time (UTC),Time local (hh:mm:ss),IR BCc\n'
                   '2023-01-01T04:00:00Z,4:00:00 AM,1000\n'
                   '2023-01-01T04:01:00Z,4:01:00 AM,2000\n')
    m = export.read_minutes(csv, 3.0)
    assert list(m.index) == [pd.Timestamp('2023-01-01 07:00'), pd.Timestamp('2023-01-01 07:01')]
    assert m['IR BCc'].tolist() == [1000, 2000]


def test_qc_drops_negatives_then_three_sd():
    v = pd.DataFrame({'IR BCc': [1.0] * 30 + [-5.0, 100.0]})
    out = export.qc(v)['IR BCc']
    assert out.isna().tolist() == [False] * 30 + [True, True]


def test_hourly_needs_enough_minutes_and_fills_the_grid():
    t = pd.date_range('2023-01-01 00:00', periods=60, freq='min')
    ir = pd.concat([pd.Series(2.0, index=t),                                            # a full hour
                    pd.Series(5.0, index=t[:10] + pd.Timedelta(hours=2))])              # 10 minutes only
    h = export.hourly_from_minutes(ir, min_minutes=30)
    assert list(h.index) == [pd.Timestamp('2023-01-01 00:00')]       # the short hour is dropped, so the grid ends
    ir2 = pd.concat([ir, pd.Series(4.0, index=t + pd.Timedelta(hours=3))])
    h2 = export.hourly_from_minutes(ir2, min_minutes=30)
    assert len(h2) == 4 and h2.iloc[0] == 2.0 and np.isnan(h2.iloc[1]) and np.isnan(h2.iloc[2]) and h2.iloc[3] == 4.0


def test_episodes_are_runs_above_the_quantile():
    idx = pd.date_range('2023-01-01', periods=12, freq='h')
    v = pd.Series([1, 1, 9, 12, 10, 1, 9, np.nan, 9, 9, 1, 1], index=idx, dtype=float)
    thr, eps = export.episodes(v, q=0.4, min_hours=2)
    assert thr == 1.0   # the 40th percentile of the eleven valid hours
    # 9,12,10 is one run (peak second); 9 alone is too short; the missing hour splits 9 | 9,9
    assert [(e['start'], e['hours'], e['peak'], e['rise'], e['fall']) for e in eps] == [
        (idx[2], 3, 12.0, 1, 1), (idx[8], 2, 9.0, 0, 1)]
    assert eps[0]['excess'] == (9 - thr) + (12 - thr) + (10 - thr)
