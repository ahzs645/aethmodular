"""Every IMPROVE filter on the HIPS bend, without grouping: a count grid and curves fitted to single filters.

Run from the repo root with the repo environment (needs the FTIR local database's TOR and HIPS tables):

    uv run python gallery/data/export_hips_improve_density.py
    uv run python gallery/data/export_hips_improve_density.py --cache /tmp/improve.parquet   # reuse a read

Ann (19 Nov 2026) asked to see the IMPROVE bend on all the data, not only the 17 group medians the ceiling was fitted
through, so the curve can be defended without the grouping. This writes, for the same 347,936 samples the 10 Nov
closure workflow uses (run_addis_reference_closure_20261110.improve_samples: 2005–2025, both volumes 20–45 m³, a
previous sample within 7 days):

* ``loading``: counts of filters on a grid of TOR EC on the filter (µg) against HIPS optical depth τ, the view the
  curve is fitted in.
* ``concentration``: counts on a grid of TOR EC (µg/m³) against HIPS Fabs (Mm⁻¹), the units HIPS is reported in.
* ``fits``: ``tau = aL / (1 + aL / tau_inf)`` fitted to single filters, by population (all sites, sites with median
  OC/EC < 4, the six urban sites), from a lowest loading (0, 5, 10, 20 µg), with residuals in τ ("absolute") or in
  log τ ("log", as the group-median fit), and a 90 % interval from resampling whole sites.

The group-median fit stays the one the gallery linearises with; this is the check on it. A fit to single filters
against their own TOR EC carries TOR's error on the x axis, which bends any fitted curve over a little more, and
a lowest loading on the x axis selects on that error, so neither is the better estimate by construction.

Writes gallery/app/public/data/hips_linearise/improve_density.json (about 60 kB).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "research/ftir_hips_chem/workflows"), str(ROOT / "research/ftir_hips_chem/scripts")]

import reference_closure as rc  # noqa: E402
import run_addis_reference_closure_20261110 as closure  # noqa: E402

OUT = ROOT / "gallery/app/public/data/hips_linearise/improve_density.json"
# grids: (lower, upper, step) per axis; cells past the upper edge are counted, not drawn
LOADING_X, LOADING_Y = (0.0, 90.0, 1.0), (0.0, 3.4, 0.04)
CONC_X, CONC_Y = (0.0, 6.0, 0.05), (0.0, 60.0, 0.5)
STARTS_UG = (0.0, 5.0, 10.0, 20.0)
RESIDUALS = ("absolute", "log")
N_BOOT = 200
SEED = 20261120


def grid(x: np.ndarray, y: np.ndarray, gx: tuple, gy: tuple) -> dict:
    """Sparse 2-D counts: [column, row, count] for every non-empty cell, and how many filters fall outside."""
    nx, ny = round((gx[1] - gx[0]) / gx[2]), round((gy[1] - gy[0]) / gy[2])
    ok = np.isfinite(x) & np.isfinite(y)
    inside = ok & (x >= gx[0]) & (x < gx[1]) & (y >= gy[0]) & (y < gy[1])
    i = ((x[inside] - gx[0]) / gx[2]).astype(int)
    j = ((y[inside] - gy[0]) / gy[2]).astype(int)
    counts = np.zeros((nx, ny), dtype=int)
    np.add.at(counts, (i, j), 1)
    cols, rows = np.nonzero(counts)
    return {"x": {"lower": gx[0], "upper": gx[1], "step": gx[2]}, "y": {"lower": gy[0], "upper": gy[1], "step": gy[2]},
            "cells": [[int(a), int(b), int(counts[a, b])] for a, b in zip(cols, rows)],
            "inside": int(inside.sum()), "outside": int(ok.sum() - inside.sum()), "max": int(counts.max())}


def fit(loading: np.ndarray, tau: np.ndarray, residual: str) -> tuple[float, float]:
    ok = (loading > 0) & (tau > 0)
    L, T = loading[ok], tau[ok]
    if residual == "log":
        r = lambda p: np.log(rc.hyperbolic(L, *p)) - np.log(T)  # noqa: E731
    else:
        r = lambda p: rc.hyperbolic(L, *p) - T  # noqa: E731
    a, tau_inf = optimize.least_squares(r, [0.04, 2.0], bounds=([0.0, 0.2], [1.0, 100.0])).x
    return float(a), float(tau_inf)


def populations(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
    site_ocec = samples.groupby("Site")["ocec"].median()
    return {"all sites": samples, "sites with median OC/EC < 4": samples[samples["Site"].map(site_ocec) < 4],
            "six urban IMPROVE sites": samples[samples["Site"].isin(closure.URBAN)]}


def fits(samples: pd.DataFrame, n_boot: int) -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for name, part in populations(samples).items():
        by_site = {s: (f["loading_ug"].to_numpy(), f["tau"].to_numpy()) for s, f in part.groupby("Site")}
        sites = np.array(sorted(by_site))
        # one set of site resamples per population, shared by every start and residual so their intervals compare
        draws = [rng.choice(sites, sites.size) for _ in range(n_boot)]
        for start in STARTS_UG:
            for residual in RESIDUALS:
                keep = part["loading_ug"] >= start
                a, tau_inf = fit(part.loc[keep, "loading_ug"].to_numpy(), part.loc[keep, "tau"].to_numpy(), residual)
                boot = []
                for draw in draws:
                    L = np.concatenate([by_site[s][0] for s in draw]); T = np.concatenate([by_site[s][1] for s in draw])
                    m = L >= start
                    if m.sum() >= 50:
                        boot.append(fit(L[m], T[m], residual)[1])
                lo, hi = np.percentile(boot, [5, 95]) if boot else (np.nan, np.nan)
                rows.append({"population": name, "from_ug": start, "residual": residual, "samples": int(keep.sum()),
                             "sites": int(part.loc[keep, "Site"].nunique()), "a_per_ug": round(a, 6),
                             "tau_inf": round(tau_inf, 4), "tau_inf_lo": round(float(lo), 4), "tau_inf_hi": round(float(hi), 4)})
                print(f"{name:28s} from {start:>4} µg {residual:8s} n={keep.sum():>7,} a={a:.4f} τ∞={tau_inf:.2f} [{lo:.2f}, {hi:.2f}]")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, help="parquet of the IMPROVE samples: read if present, else written after the read")
    parser.add_argument("--boot", type=int, default=N_BOOT, help="site resamples per fit (default %(default)s)")
    args = parser.parse_args()
    if args.cache and args.cache.is_file():
        samples = pd.read_parquet(args.cache)
    else:
        samples = closure.improve_samples()
        if samples is None:
            raise FileNotFoundError("IMPROVE TOR and HIPS tables are required; set AETHMODULAR_FTIR_LOCAL_DB")
        if args.cache:
            samples.to_parquet(args.cache)
    conc_ec = samples["EC"].to_numpy() / 1000.0                      # TOR values are ng/m³
    volume = samples["volume_m3_ptfe"]
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py (improve_samples); "
                  "gallery/data/export_hips_improve_density.py",
        "samples": int(len(samples)), "sites": int(samples["Site"].nunique()),
        "first": samples["date"].min().strftime("%Y-%m-%d"), "last": samples["date"].max().strftime("%Y-%m-%d"),
        "median_volume_m3": round(float(volume.median()), 3),
        "volume_p10_p90_m3": [round(float(volume.quantile(0.1)), 3), round(float(volume.quantile(0.9)), 3)],
        "deposit_area_cm2": closure.DEPOSIT_AREA_CM2,
        "urban_sites": list(closure.URBAN),
        "loading": grid(samples["loading_ug"].to_numpy(), samples["tau"].to_numpy(), LOADING_X, LOADING_Y),
        "concentration": grid(conc_ec, samples["fAbs"].to_numpy(), CONC_X, CONC_Y),
        "fits": fits(samples, args.boot),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1024:.0f} kB, {len(payload['fits'])} fits, "
          f"{len(payload['loading']['cells'])} + {len(payload['concentration']['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
