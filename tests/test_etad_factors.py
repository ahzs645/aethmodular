import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


SCRIPTS = Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"
sys.path.insert(0, str(SCRIPTS))

import etad_factors  # noqa: E402
from etad_factors import (  # noqa: E402
    GF_FRACTION_COLUMNS,
    attach_factors_by_date,
)


def _factors():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
        "GF1 (Sea Salt Mixed)": [0.01, 0.05],
        "GF2 (Wood Burning)": [0.02, 0.10],
        "GF3 (Charcoal)": [0.03, 0.20],
        "GF4 (Polluted Marine)": [0.04, 0.05],
        "GF5 (Fossil Fuel Combustion)": [0.10, 0.10],
    })


def test_attaches_to_tz_aware_datetime_index():
    frame = pd.DataFrame(
        {"bc": [1.0, 2.0]},
        index=pd.DatetimeIndex(
            ["2024-01-02 09:00", "2024-01-03 09:00"],
            tz="Africa/Addis_Ababa",
            name="sample_time",
        ),
    )

    result = attach_factors_by_date(frame, factors=_factors())

    assert result.loc[frame.index[0], "fossil_fuel_frac"] == pytest.approx(0.5)
    assert pd.isna(result.loc[frame.index[1], "fossil_fuel_frac"])
    assert result.index.equals(frame.index)


def test_attaches_to_tz_naive_datetime_index_without_localize_crash():
    frame = pd.DataFrame(
        {"bc": [1.0]},
        index=pd.DatetimeIndex(["2024-01-02 09:00"], name="sample_time"),
    )

    result = attach_factors_by_date(frame, factors=_factors())

    assert result["charcoal_frac"].iloc[0] == pytest.approx(0.15)


def test_normalization_is_applied_and_dominant_columns_are_added():
    frame = pd.DataFrame(
        {"bc": [1.0]},
        index=pd.DatetimeIndex(["2024-01-02 12:00"]),
    )

    result = attach_factors_by_date(frame, factors=_factors())

    assert result[GF_FRACTION_COLUMNS].sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert result[GF_FRACTION_COLUMNS].dtypes.eq("float64").all()
    assert result["dominant_source"].iloc[0] == "fossil_fuel"
    assert result["dominant_fraction"].iloc[0] == pytest.approx(0.5)


def test_caller_frame_is_not_mutated():
    frame = pd.DataFrame(
        {"bc": [1.0]},
        index=pd.DatetimeIndex(["2024-01-02 09:00"], tz="UTC"),
    )
    original = frame.copy(deep=True)

    result = attach_factors_by_date(frame, factors=_factors())

    assert result is not frame
    assert_frame_equal(frame, original)


def test_accepts_date_column_and_nearest_tolerance():
    frame = pd.DataFrame({
        "date": ["2024-01-03 18:00"],
        "bc": [1.0],
    })

    exact = attach_factors_by_date(frame, factors=_factors())
    nearest = attach_factors_by_date(
        frame,
        factors=_factors(),
        tolerance_days=1,
    )

    assert pd.isna(exact["dominant_source"].iloc[0])
    assert nearest["dominant_source"].iloc[0] == "fossil_fuel"


def test_loads_factors_when_not_supplied(monkeypatch):
    frame = pd.DataFrame(
        {"bc": [1.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )
    calls = []

    def fake_loader():
        calls.append(True)
        return _factors()

    monkeypatch.setattr(
        etad_factors,
        "load_etad_factors_with_filter_ids",
        fake_loader,
    )

    result = attach_factors_by_date(frame)

    assert calls == [True]
    assert result["dominant_source"].iloc[0] == "fossil_fuel"
