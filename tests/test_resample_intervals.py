"""Scientific regression tests for local-day boundaries and minute availability."""

import numpy as np
import pandas as pd
import pytest

from scripts.pipelines.create_9am_resampled_datasets import (
    resample_to_9am_daily,
    select_key_columns,
)


def test_exact_boundary_enters_next_interval_and_is_end_labelled():
    frame = pd.DataFrame({'datetime_local': pd.to_datetime([
        '2024-01-01 08:59', '2024-01-01 09:00', '2024-01-02 08:59', '2024-01-02 09:00']),
        'IR BCc': [10., 20., 40., 100.], 'index': [1., 2., 3., 4.]})
    result = resample_to_9am_daily(frame, 'Asia/Shanghai')
    assert result['IR BCc'].tolist() == [10., 30., 100.]
    assert result.datetime_local.dt.hour.tolist() == [9, 9, 9]
    assert result.interval_start.iloc[1] == pd.Timestamp('2024-01-01 09:00', tz='Asia/Shanghai')
    assert result.interval_end.iloc[1] == pd.Timestamp('2024-01-02 09:00', tz='Asia/Shanghai')
    assert result.columns.is_unique
    assert result.minutes_with_data.tolist() == [1, 2, 1]
    assert frame.columns.tolist() == ['datetime_local', 'IR BCc', 'index']


@pytest.mark.parametrize('start,end,minutes', [
    ('2024-03-09 09:00', '2024-03-10 09:00', 1380),
    ('2024-11-02 09:00', '2024-11-03 09:00', 1500),
])
def test_dst_uses_local_calendar_bounds_and_real_elapsed_minutes(start, end, minutes):
    index = pd.date_range(start, end, freq='min', inclusive='left', tz='America/Los_Angeles')
    # A UTC input must be converted to the configured local timezone.
    frame = pd.DataFrame({'IR BCc': 1.}, index=index.tz_convert('UTC'))
    result = resample_to_9am_daily(frame, 'America/Los_Angeles')
    assert len(result) == 1
    assert result.expected_minutes.iloc[0] == minutes
    assert result.minutes_with_data.iloc[0] == minutes
    assert result.data_completeness_pct.iloc[0] == 100.
    assert result.datetime_local.iloc[0] == pd.Timestamp(end, tz='America/Los_Angeles')


def test_missing_days_and_channel_specific_coverage():
    frame = pd.DataFrame({'IR BCc': [np.nan, 2.], 'Blue BCc': [3., np.inf]},
                         index=pd.to_datetime(['2024-01-01 10:00', '2024-01-03 10:00']))
    result = resample_to_9am_daily(frame, 'Asia/Shanghai')
    assert result.minutes_with_data.tolist() == [0, 0, 1]
    assert result['Blue BCc valid_minutes'].tolist() == [1, 0, 0]
    assert np.isnan(result['IR BCc'].iloc[1])
    assert result.data_completeness_pct.iloc[0] == 0
    assert result.data_completeness_pct.iloc[2] == pytest.approx(100 / 1440)
    assert result.attrs['coverage_basis'] == 'non_null_input_minutes'
    assert result.attrs['upstream_interpolation_verified'] is False


def test_subminute_rows_do_not_inflate_coverage_or_weight():
    frame = pd.DataFrame({'IR BCc': [0., 10., 20.]}, index=pd.to_datetime([
        '2024-01-01 10:00:00', '2024-01-01 10:00:30', '2024-01-01 10:01:00']))
    result = resample_to_9am_daily(frame, 'Asia/Shanghai')
    assert result.minutes_with_data.iloc[0] == 2
    assert result['IR BCc'].iloc[0] == 12.5


def test_interpolated_and_missing_observation_flags_do_not_count():
    frame = pd.DataFrame({'IR BCc': [1., 500., 999.],
                          'observed': pd.array([True, False, None], dtype='boolean')},
                         index=pd.date_range('2024-01-01 10:00', periods=3, freq='min'))
    result = resample_to_9am_daily(frame, 'Asia/Shanghai', observed_col='observed')
    assert result.minutes_with_data.iloc[0] == 1
    assert result['IR BCc'].iloc[0] == 1
    assert result.attrs['coverage_basis'] == 'observed_flag'


def test_duplicate_timestamps_require_resolution():
    frame = pd.DataFrame({'IR BCc': [1., 100.]},
                         index=pd.to_datetime(['2024-01-01 10:00'] * 2))
    with pytest.raises(ValueError, match='Duplicate timestamps'):
        resample_to_9am_daily(frame, 'Asia/Shanghai')


def test_selected_export_preserves_intervals_and_channel_counts():
    frame = pd.DataFrame({'IR BCc': [1.]}, index=pd.to_datetime(['2024-01-01 10:00']))
    result = select_key_columns(resample_to_9am_daily(frame, 'Asia/Shanghai'), 'CHTS')
    assert {'interval_start', 'interval_end', 'expected_minutes',
            'IR BCc coverage_pct', 'IR BCc valid_minutes'} <= set(result)
    assert result.attrs['resampling_version'] == 2
