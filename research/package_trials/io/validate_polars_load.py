"""polars vs pandas for the explorer's 725 MB pool-spectra CSV load.

The calibration explorer's slowest startup step is reading
`local_db/spectra_248_251.csv` (13k IMPROVE spectra x ~2.7k wavenumber
columns). This trial times the app's exact historical pandas read against
polars' multithreaded reader (the fast path now wired into
`calibration_explorer/app.py::_read_pool_spectra`) and an npz binary cache,
and verifies equivalence after the app's dedup/set_index post-processing.

Parser-precision context, established on the same-shape synthetic trial
(see README.md): pandas' default C parser is not correctly rounded; polars
is, and is bit-identical to pandas `float_precision='round_trip'`. So the
expected result here is bitwise equality vs round_trip, and at most a
handful of 1-ulp float32 differences vs the historical default read.

Run:  /Users/ahmadjalil/anaconda3/bin/python validate_polars_load.py
(the app's interpreter, from this directory — needs Google Drive access,
which sandboxed agent shells on this machine do not have)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "ftir_ec_phase3" / "scripts"))

from phase3_common import PATHS, load_addis_evaluation  # noqa: E402
from config import season_for_month  # noqa: E402

CSV = PATHS.ftir_dir / "local_db/spectra_248_251.csv"
CACHE_NPZ = Path(__file__).parent / "cache" / "pool_spectra.npz"


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """The app's exact post-read steps (app.py::_load_all)."""
    df = df[~df["AnalysisId"].duplicated()].set_index("AnalysisId")
    df.index = df.index.astype(int)
    return df


def read_pandas(wcols, **kw):
    df = pd.read_csv(CSV, usecols=["AnalysisId"] + wcols,
                     dtype={c: np.float32 for c in wcols}, **kw)
    return postprocess(df)


def read_polars(wcols):
    """Identical to app.py::_read_pool_spectra's polars branch."""
    import polars as pl
    raw = pl.read_csv(CSV, columns=["AnalysisId"] + wcols,
                      schema_overrides={c: pl.Float64 for c in wcols})
    frame = pd.DataFrame(
        raw.select(wcols).to_numpy().astype(np.float32), columns=wcols)
    frame.insert(0, "AnalysisId", raw["AnalysisId"].to_numpy())
    return postprocess(frame)


def read_npz():
    z = np.load(CACHE_NPZ)
    df = pd.DataFrame(z["values"], columns=list(z["wcols"]))
    df.insert(0, "AnalysisId", z["ids"])
    return postprocess(df)


def compare(name, got, ref):
    same = (got.index.equals(ref.index)
            and list(got.columns) == list(ref.columns))
    a, b = got.to_numpy(), ref.to_numpy()
    if a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True):
        print(f"  {name}: index/columns equal={same}, values BITWISE EQUAL")
        return
    d = a != b
    d &= ~(np.isnan(a) & np.isnan(b))
    n = int(d.sum())
    ulp = np.abs(a[d].astype(np.float64) - b[d].astype(np.float64)) / \
        np.spacing(np.abs(b[d]).astype(np.float32)).astype(np.float64)
    print(f"  {name}: index/columns equal={same}, {n}/{a.size} cells differ, "
          f"max {ulp.max():.2f} float32 ulp "
          f"(values |x| in [{np.abs(b[d]).min():.3g}, {np.abs(b[d]).max():.3g}])")


def main():
    print(f"CSV: {CSV} ({CSV.stat().st_size / 1e6:.0f} MB)")
    etad_eval, _, _ = load_addis_evaluation(season_for_month)
    wcols = list(etad_eval.attrs["wcols"])
    print(f"reading AnalysisId + {len(wcols)} wavenumber columns\n")

    timings = {}

    t = time.perf_counter()
    pol = read_polars(wcols)
    timings["polars (1st — includes any Drive streaming)"] = time.perf_counter() - t

    t = time.perf_counter()
    ref = read_pandas(wcols)
    timings["pandas default (the app's historical read)"] = time.perf_counter() - t
    print(f"frame: {ref.shape[0]} x {ref.shape[1]}, "
          f"{ref.to_numpy().nbytes / 1e6:.0f} MB float32\n")

    t = time.perf_counter()
    pol2 = read_polars(wcols)
    timings["polars (2nd — file cache warm)"] = time.perf_counter() - t

    t = time.perf_counter()
    rt = read_pandas(wcols, float_precision="round_trip")
    timings["pandas round_trip (correctly rounded)"] = time.perf_counter() - t

    CACHE_NPZ.parent.mkdir(exist_ok=True)
    t = time.perf_counter()
    np.savez(CACHE_NPZ, values=ref.to_numpy(),
             ids=ref.index.to_numpy(), wcols=np.array(wcols))
    timings["npz write (uncompressed, one-time)"] = time.perf_counter() - t
    t = time.perf_counter()
    npz = read_npz()
    timings["npz read (binary cache)"] = time.perf_counter() - t

    print("equivalence:")
    compare("polars vs pandas default", pol, ref)
    compare("polars run-to-run", pol2, pol)
    compare("polars vs pandas round_trip", pol, rt)
    compare("npz cache vs pandas default", npz, ref)

    print("\ntimings:")
    base = timings["pandas default (the app's historical read)"]
    for name, s in timings.items():
        print(f"  {s:7.1f} s  ({base / s:5.1f}x vs pandas)  {name}")


if __name__ == "__main__":
    main()
