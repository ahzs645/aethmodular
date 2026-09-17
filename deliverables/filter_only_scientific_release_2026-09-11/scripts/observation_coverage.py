"""Shared acquisition-slot arithmetic for daily and active-interval averages.

UTC slots prevent repeated local DST hours from colliding. Values are averaged
within a slot first, so extra sub-cadence rows cannot manufacture coverage or
overweight that slot. This module never infers whether a row was observed.
"""

import numpy as np
import pandas as pd


EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def cadence_ns(cadence_seconds):
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0:
        raise ValueError("Acquisition cadence must be positive and finite")
    result = int(round(float(cadence_seconds) * 1_000_000_000))
    if result < 1:
        raise ValueError("Acquisition cadence is below timestamp precision")
    return result


def utc_index(timestamps):
    index = pd.DatetimeIndex(timestamps)
    if index.tz is None or index.hasnans:
        raise ValueError("Acquisition timestamps must be timezone-aware and nonmissing")
    return index.tz_convert("UTC").as_unit("ns")


def slot_index(timestamps, cadence_seconds=60, origin_utc=EPOCH):
    index = utc_index(timestamps)
    origin = pd.Timestamp(origin_utc)
    if origin.tzinfo is None:
        raise ValueError("Acquisition origin must be timezone-aware")
    width = cadence_ns(cadence_seconds)
    slots = origin.value + ((index.asi8 - origin.value) // width) * width
    return pd.DatetimeIndex(pd.to_datetime(slots, utc=True))


def expected_slots(start_utc, end_utc, cadence_seconds=60, origin_utc=EPOCH):
    """Nominal acquisition instants in [start, end), on the declared cadence grid."""
    start, end, origin = map(pd.Timestamp, [start_utc, end_utc, origin_utc])
    if any(t.tzinfo is None for t in [start, end, origin]) or end <= start:
        raise ValueError("Expected-slot bounds require an increasing timezone-aware interval")
    width = cadence_ns(cadence_seconds)
    first = origin.value + (-((origin.value - start.value) // width)) * width
    return pd.DatetimeIndex(
        pd.to_datetime(np.arange(first, end.value, width, dtype=np.int64), utc=True)
    )


def distinct_slot_means(values, timestamps=None, cadence_seconds=60, origin_utc=EPOCH):
    """Equal-weight slot values. Missing/nonfinite inputs remain missing."""
    values = values.replace([np.inf, -np.inf], np.nan).copy()
    index = slot_index(
        values.index if timestamps is None else timestamps, cadence_seconds, origin_utc
    )
    values.index = index
    return values.groupby(level=0, sort=True).mean()


def boolean_slot_flags(flags, timestamps, cadence_seconds=60, origin_utc=EPOCH):
    """Return any-true and any-unknown flags without filling unknown with observed."""
    flags = pd.Series(flags).reset_index(drop=True)
    if not pd.api.types.is_bool_dtype(flags.dtype):
        raise ValueError("Observation and validity flags must have boolean dtype")
    index = slot_index(timestamps, cadence_seconds, origin_utc)
    any_true = (
        pd.Series(flags.fillna(False).to_numpy(dtype=bool), index=index).groupby(level=0).any()
    )
    unknown = pd.Series(flags.isna().to_numpy(), index=index).groupby(level=0).any()
    return any_true, unknown
