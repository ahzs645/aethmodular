"""Offline checks for gallery/data/export_aeth_sources.py (the instrument's own AAE per daily bin)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
