"""Unit-coercion helpers shared across notebooks."""

from __future__ import annotations

import pandas as pd


def to_ugm3(series, ng_threshold: float = 100.0):
    """Coerce a mass-concentration series to ug/m3, auto-detecting ng/m3.

    Some source columns arrive in ng/m3. If the median absolute value exceeds
    ng_threshold the series is assumed to be ng/m3 and divided by 1000;
    otherwise it is returned as-is. Non-numeric entries become NaN.

    Consolidated from research/catch_up (the `_to_ugm3` helper copied across 7
    notebooks).
    """
    s = pd.to_numeric(series, errors="coerce")
    med = s.dropna().abs().median()
    if pd.notna(med) and med > ng_threshold:
        return s / 1000.0
    return s
