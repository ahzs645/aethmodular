"""Shared smoothing kernels; callers choose the sampling/spot boundaries explicitly."""
from __future__ import annotations

import numpy as np
import pandas as pd


def dema(values: pd.Series, segment: pd.Series, span: int = 15) -> pd.Series:
    """2 EMA − EMA(EMA), with the pipeline's adjust=False convention per segment."""
    if span < 1:
        raise ValueError("DEMA span must be positive")

    def one(s):
        ema = s.ewm(span=span, adjust=False).mean()
        return 2 * ema - ema.ewm(span=span, adjust=False).mean()

    return values.astype(float).groupby(segment, sort=False).transform(one)


def ona(values: np.ndarray, attenuation: np.ndarray, segment: np.ndarray,
        delta_atn: float = 0.05) -> tuple[np.ndarray, pd.DataFrame]:
    """Legacy notebook ONA: non-overlapping windows until ATN > start + delta.

    The first point over the threshold starts the next window. Never average
    across a supplied boundary or an unknown attenuation. Missing BC remains
    missing rather than acquiring an apparent observation from its neighbours.
    Window sizes are returned once per window, not repeated once per minute.
    """
    if not np.isfinite(delta_atn) or delta_atn <= 0:
        raise ValueError("delta_atn must be finite and positive")
    x, atn, seg = np.asarray(values, float), np.asarray(attenuation, float), np.asarray(segment)
    if not (x.shape == atn.shape == seg.shape) or x.ndim != 1:
        raise ValueError("ONA inputs must be aligned one-dimensional arrays")
    out = np.full(len(x), np.nan)
    windows = []
    start = 0
    while start < len(x):
        if not np.isfinite(atn[start]):
            start += 1
            continue
        end = start + 1
        while (end < len(x) and seg[end] == seg[start] and np.isfinite(atn[end])
               and atn[end] <= atn[start] + delta_atn):
            end += 1
        v = x[start:end]
        good = np.isfinite(v)
        if good.any():
            out[start:end] = np.where(good, v[good].mean(), np.nan)
        windows.append({'start': start, 'end': end, 'points': end - start,
                        'threshold_reached': bool(end < len(x) and seg[end] == seg[start]
                                                  and np.isfinite(atn[end])
                                                  and atn[end] > atn[start] + delta_atn)})
        start = end
    return out, pd.DataFrame(windows, columns=['start', 'end', 'points', 'threshold_reached'])
