"""Compact, exact-enough storage for the gallery's FTIR traces (decoded by app/src/lib/spartan.ts).

A trace is stored as integer steps of ``step`` absorbance (default 1e-8), as differences from the
previous point, with the four bytes of each 32-bit difference regrouped into four planes (all low
bytes, then all second bytes, ...). Neighbouring points of a spectrum are close, so the differences
are small and the high planes are nearly all zero or 0xFF; gzip, applied to the whole file by the
exporter, then packs them far tighter than raw float32.

Measured on 400 native SPARTAN spectra (19 Oct 2026), per filter after gzip:

    float32                       10.6 KB   (45 MB for the network)
    float32, byte planes           8.0 KB
    steps of 1e-8, differences     5.5 KB   (23.5 MB)  rounding <= 5e-9, below float32's own spacing
    steps of 1e-7, differences     4.3 KB   (18 MB)    rounding <= 5e-8, ~40x below point noise
    16-bit steps, differences      2.1 KB   (9 MB)     rounding ~3e-6, at the noise level

so 1e-8 is the default: smaller than float32 and no less exact. ``step`` is recorded per trace,
so a coarser export (1e-7) needs no reader change.

Missing points (NaN) are stored as runs ``[[start, length], ...]`` and encoded as the previous
value, so they cost almost nothing.
"""
from __future__ import annotations

import numpy as np

DEFAULT_STEP = 1e-8


def encode(values, step=DEFAULT_STEP):
    """Return (bytes, descriptor) for one trace; the descriptor holds n, step and NaN runs."""
    y = np.asarray(values, dtype=float)
    nan = ~np.isfinite(y)
    filled = y.copy()
    if nan.all():
        filled[:] = 0.0
    elif nan.any():
        # carry the last finite value through each gap (the first gap takes the first finite value)
        idx = np.where(~nan, np.arange(len(y)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = y[idx]
        filled[: np.argmax(~nan)] = y[np.argmax(~nan)]
    q = np.rint(filled / step).astype(np.int64)
    d = np.diff(q, prepend=0)
    if np.abs(d).max(initial=0) >= 2**31:
        raise ValueError("trace step too fine for 32-bit differences")
    planes = d.astype("<i4").view(np.uint8).reshape(-1, 4).T
    runs, i = [], 0
    while i < len(y):
        if nan[i]:
            j = i
            while j < len(y) and nan[j]:
                j += 1
            runs.append([int(i), int(j - i)])
            i = j
        else:
            i += 1
    desc = {"n": int(len(y)), "step": step}
    if runs:
        desc["nan"] = runs
    return planes.tobytes(), desc


def decode(buf, desc, offset=0):
    """Inverse of encode; ``offset`` is in bytes within ``buf``."""
    n, step = desc["n"], desc["step"]
    planes = np.frombuffer(buf, dtype=np.uint8, count=4 * n, offset=offset).reshape(4, n)
    d = planes.T.copy().view("<i4").ravel().astype(np.int64)
    y = np.cumsum(d) * step
    for start, length in desc.get("nan", []):
        y[start:start + length] = np.nan
    return y
