"""char_05's McCall scores for every SPARTAN PM2.5 filter, not only Addis.

    uv run --locked --no-sync python research/ftir_etad_char/scripts/network_char_scores.py

The same recipe as char_05/char_06: AIRSpec-corrected spectrum, repeat scans of a filter averaged, cropped to
1430-3500 cm-1, linear detrend, SNV, then the best Pearson r against McCall's charred (r_char) and raw (r_raw)
feedstock. The spectra are the network run's (research/ftir_hips_chem/workflows/network_baselines_20261019.py):
each site's native raw scans and their saved AIRSpec baselines, so nothing is refitted here.

Writes output/tables/network/char_scores_per_filter.csv and prints the agreement with char_06's Addis scores, which
is the check that the recipe was carried over. gallery/data/export_research_charcoal.py adds the table to the page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "research/ftir_hips_chem/workflows")]
from charcoal_spectra import detrend, load_all, resample, snv  # noqa: E402
from network_baselines_20261019 import OUT as NETWORK, native_analyses  # noqa: E402

GRID = np.arange(1430.0, 3500.0 + 1e-9, 2.0)  # char_05's comparison window
TABLES = HERE.parent / "output/tables"
OUT = TABLES / "network/char_scores_per_filter.csv"


def zrows(X: np.ndarray) -> np.ndarray:
    sd = X.std(1, keepdims=True)
    return (X - X.mean(1, keepdims=True)) / np.where(sd == 0, 1.0, sd)


def mccall() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """McCall's references on the grid, with the masks of the raw and the charred spectra."""
    sets = load_all()
    meta = pd.concat([sets["mccall_multi"].meta, sets["mccall_barley"].meta], ignore_index=True)
    X = snv(detrend(np.vstack([resample(sets[k].X, sets[k].wn, GRID) for k in ("mccall_multi", "mccall_barley")]), GRID))
    return zrows(X), (meta["temp_c"] == 0).to_numpy(), (meta["temp_c"] > 0).to_numpy()


def corrected_scans() -> pd.DataFrame:
    """Every PM2.5 analysis's corrected spectrum on the grid, one row per analysis (the native folder holds each once or more)."""
    air = np.load(NETWORK / "airspec_baselines.npz")
    baselines = {k.split("_")[0][1:]: (int(k.split("_")[1]), k) for k in air.files}
    seen, rows, traces = set(), [], []
    for a, wn, y in native_analyses():
        aid = str(a["analysis_id"])
        if a["filter_type"] != "PM2.5" or aid in seen or aid not in baselines:
            continue
        seen.add(aid)
        start, key = baselines[aid]
        base = air[key]
        span, trace = wn[start:start + len(base)], y[start:start + len(base)] - base
        ok = np.isfinite(trace)
        if not ok.any() or span[ok].min() > GRID[0] or span[ok].max() < GRID[-1]:
            continue  # no baseline, or a scan that does not span the window
        rows.append({"id": a["filter_key"], "code": a["code"]})
        traces.append(resample(trace[ok][None, :], span[ok], GRID)[0])
    return pd.DataFrame(rows), np.vstack(traces)


def score() -> pd.DataFrame:
    refs, raw, char = mccall()
    scans, X = corrected_scans()
    groups = scans.groupby(["id", "code"], sort=True).indices  # repeat scans averaged, as char_05's loader does
    keys = list(groups)
    mean = np.vstack([X[groups[k]].mean(axis=0) for k in keys])
    R = (zrows(snv(detrend(mean, GRID))) @ refs.T) / GRID.size
    frame = pd.DataFrame(keys, columns=["id", "code"])
    frame["r_char"], frame["r_raw"] = R[:, char].max(axis=1), R[:, raw].max(axis=1)
    frame["scans"] = [len(groups[k]) for k in keys]
    return frame


def check(frame: pd.DataFrame) -> None:
    """Against char_06's own Addis scores, when that run's table is here."""
    path = TABLES / "char06/per_filter_classification.csv"
    if not path.exists():
        print("char_06's table is not here: agreement with the notebook not checked")
        return
    saved = pd.read_csv(path)
    saved["id"] = saved["FilterId"].str.replace(r"-\d+$", "", regex=True)
    both = saved.merge(frame, on="id", suffixes=("", "_network"))
    for name in ("r_char", "r_raw"):
        gap = (both[name] - both[f"{name}_network"]).abs()
        print(f"{name}: {len(both)} of {len(saved)} Addis filters, median gap {gap.median():.5f}, largest {gap.max():.4f}, "
              f"r {np.corrcoef(both[name], both[f'{name}_network'])[0, 1]:.5f}")


if __name__ == "__main__":
    scores = score()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT, index=False)
    print(f"{len(scores)} PM2.5 filters at {scores['code'].nunique()} sites → {OUT.relative_to(ROOT)}")
    check(scores)
