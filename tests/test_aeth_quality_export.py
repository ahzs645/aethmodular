"""The aethalometer quality export (gallery/data/export_aeth_quality.py): alignment with the daily export, rounding, column picking."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gallery/data'))
spec = importlib.util.spec_from_file_location('export_aeth_quality', ROOT / 'gallery/data/export_aeth_quality.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)
aeth = quality.aeth


@pytest.mark.parametrize('code', sorted(aeth.SITE_FILES))
def test_bins_line_up_with_the_daily_export(code):
    site, fname = aeth.SITE_FILES[code]
    path = aeth.PROCESSED / fname
    if not path.exists():
        pytest.skip(f'{fname} not in this checkout')
    raw = pd.read_pickle(path)
    daily = aeth.daily_frame(code, site, path)
    q = quality.quality_frame(raw)
    assert q['start'].tolist() == daily['start'].tolist()


def test_rounding_keeps_missing_values_missing():
    s = pd.Series([1.23456, np.nan, np.inf, 0.000123456, 0.0, -2.5])
    assert quality.rounded(s, ('d', 2)) == [1.23, None, None, 0.0, 0.0, -2.5]
    assert quality.rounded(s, ('s', 3)) == [1.23, None, None, 0.000123, 0.0, -2.5]


def test_frame_takes_the_first_named_column_scales_and_drops_empty_series():
    starts = pd.to_datetime(['2023-01-02 09:00', '2023-01-01 09:00', '2023-01-01 09:00']).tz_localize('Africa/Addis_Ababa')
    raw = pd.DataFrame({
        'datetime_local': starts,
        'IR BCc smoothed': [2000.0, 1000.0, 999.0],          # ng/m³, preferred over the unit-suffixed name
        'IR BCc smoothed  (ng/m^3)': [5.0, 5.0, 5.0],
        'IR ATN1': [10.0, 20.0, 30.0],
        'IR K': [np.nan, np.nan, np.nan],                     # all missing: not exported
    })
    raw = pd.concat([raw, raw[['IR ATN1']]], axis=1)          # duplicated column names, as in the legacy pickles
    q = quality.quality_frame(raw)
    assert q['start'].dt.tz is None
    assert q['start'].tolist() == list(pd.to_datetime(['2023-01-01 09:00', '2023-01-02 09:00']))
    assert q['IR.smoothed'].tolist() == [1.0, 2.0]            # sorted, first of the duplicate starts kept, µg/m³
    assert q['IR.ATN1'].tolist() == [20.0, 10.0]
    assert 'IR.K' not in q.columns


def test_flow_periods_are_the_config_periods_by_site_code():
    periods = quality.flow_periods()
    assert periods['USPA']['before_end'] == quality.FLOW_FIX_PERIODS['JPL']['before_end']
    assert periods['USPA']['after_start'] == quality.FLOW_FIX_PERIODS['JPL']['after_start']
    assert periods['ETAD']['before_end'] is None
    assert 'ETAD-JACROS' not in periods              # Jacros has no period of its own


def test_cleaning_bounds_come_from_the_cleaning_defaults():
    bounds = quality.cleaning_bounds()
    assert bounds['flow_ratio']['lower'] < bounds['flow_ratio']['upper']
    assert bounds['leak_ratio']['lower_bound'] < 1 < bounds['leak_ratio']['upper_bound']
    assert 0 < bounds['bcc_denominator']['threshold'] <= bounds['bcc_denominator']['threshold_IR']
