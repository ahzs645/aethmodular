"""Exercise the vendor-style firmware path: upper and lower clipping are audited."""

import numpy as np
import pandas as pd

from src.external.calibration import sa_aethalometer_model


def test_firmware_source_model_flags_fraction_above_one_before_percent_conversion():
    # Three physically positive ratios with raw source-model fractions -0.5, 0.5, 1.5.
    # These short records stay in the firmware smoother's documented startup bypass.
    fractions = np.array([-0.5, 0.5, 1.5])
    ratio = 880 / 470
    ir = np.full(3, 1000.0)
    blue = ir * 10.120 / 19.070 * ((1 - fractions) * ratio + fractions * ratio ** 2)
    frame = pd.DataFrame({
        "Serial number": ["MA350-SYNTHETIC"] * 3,
        "Session ID": [1] * 3, "Tape position": [1] * 3, "Relative tape position": [1] * 3,
        "datetime_local": pd.date_range("2026-01-01", periods=3, freq="min"),
        "test": [1] * 3, "Timebase (s)": [60] * 3,
        "IR BCc": ir, "IR BC1": ir, "Blue BCc": blue, "Blue BC1": blue,
        "IR ATN1": [10.0] * 3,
    })
    result = sa_aethalometer_model(
        frame, tag_data_snapped_to_0_and_100=True, integer_math=False,
        print_loading_bars=False,
    ).sort_values("datetime_local")
    assert result["bb_forced_to_between_0_and_100"].tolist() == [1, 0, 1]
    assert result["BB percent"].iloc[0] == 0
    assert result["BB percent"].iloc[2] == 100
    assert 49 <= result["BB percent"].iloc[1] <= 50  # vendor firmware truncates percent
