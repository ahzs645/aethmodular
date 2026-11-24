#!/usr/bin/env python
"""How the IMPROVE groups behind the HIPS ceiling are made, by period and season (24 Nov 2026).

The ceiling is fitted to group medians: filters are grouped by the previous sample's TOR EC loading at the same site
(17 fixed edges), and each group gives one median loading and one median optical depth. This writes those groups for two
periods (before and after the HIPS rebuild of the 2017 sample year) and two seasons (November to June, July to October),
for single years (no fit: the absorption per EC of each group), and the fitted bend in five-year windows, with the same
fit under other groupings. Aggregates of public IMPROVE data only: medians, counts and fits, no filter rows.

    uv run python gallery/data/export_hips_improve_periods.py [--cache samples.parquet] [--boot 60]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "research/ftir_hips_chem/workflows"), str(ROOT / "research/ftir_hips_chem/scripts")]

import export_hips_ceiling_checks as checks  # noqa: E402
import export_hips_warren as warren  # noqa: E402
import reference_closure as rc  # noqa: E402

OUT = ROOT / "gallery/app/public/data/hips_linearise/improve_periods.json"
EDGES, MIN_N = checks.closure.IMPROVE_EDGES, checks.closure.IMPROVE_MIN_STRATUM
YEAR_MIN_N = 30                                  # single years: a group's median is shown from this many filters (no fit)
SEASONS = (("all", "All months", tuple(range(1, 13))), ("rest", "November to June", (11, 12, 1, 2, 3, 4, 5, 6)), ("fire", "July to October", (7, 8, 9, 10)))
ERAS = (("old", "2008–2016", 2008, 2016), ("new", "2017–2025", 2017, 2025))
BASES = (("previous_ug", "Previous sample (published)"), ("two_back_ug", "Two samples back"), ("three_back_ug", "Three samples back"),
         ("next_ug", "Next sample"), ("loading_ug", "Own TOR EC"))
WINDOW = 5
SEED = 20261124


def groups(value: np.ndarray, loading: np.ndarray, tau: np.ndarray, min_n: int = MIN_N) -> list[tuple[int, int, float, float]]:
    """(group index from 0, filters, median loading, median optical depth), groups of fewer than min_n left out"""
    which = np.digitize(value, EDGES) - 1
    order = np.argsort(which, kind="stable")
    which, loading, tau = which[order], loading[order], tau[order]
    starts = np.flatnonzero(np.r_[True, which[1:] != which[:-1]])
    ends = np.r_[starts[1:], which.size]
    return [(int(which[i]), int(j - i), float(np.median(loading[i:j])), float(np.median(tau[i:j]))) for i, j in zip(starts, ends) if j - i >= min_n]


def fit(rows) -> tuple[float, float, float]:
    (a, c), rms = checks.fit_medians(np.array([r[2] for r in rows]), np.array([r[3] for r in rows]))
    return float(a), float(c), float(rms)


def arrays(frame: pd.DataFrame, column: str = "previous_ug"):
    frame = frame.dropna(subset=[column])
    return [tuple(v[[column, "loading_ug", "tau"]].to_numpy().T) for _, v in frame.groupby("Site")]


def fitted(frame: pd.DataFrame, n_boot: int, column: str = "previous_ug") -> dict:
    """The fit, with a 90 % interval on the bend 1/τ∞ (0 is a straight line) from resampling whole sites"""
    sites = arrays(frame, column)
    rows = groups(*(np.concatenate(x) for x in zip(*sites)))
    a, c, rms = fit(rows)
    rng, bend = np.random.default_rng(SEED), []
    for _ in range(n_boot):
        pick = [sites[i] for i in rng.integers(0, len(sites), len(sites))]
        b = groups(*(np.concatenate(x) for x in zip(*pick)))
        if len(b) >= 5:
            bend.append(1 / fit(b)[1])
    lo, hi = np.percentile(bend, [5, 95])
    return {"filters": int(sum(s[0].size for s in sites)), "sites": len(sites), "fabs_per_ec": round(a * 100 * warren.AREA, 3),
            "a": round(a, 6), "tau_inf": round(c, 4), "rms_log": round(rms, 4), "bend": round(1 / c, 4), "bend_lo": round(float(lo), 4), "bend_hi": round(float(hi), 4),
            "groups": [{"g": g, "n": n, "loading_ug": round(L, 4), "tau": round(t, 5)} for g, n, L, t in rows]}


def models(rows: list[dict]) -> list[dict]:
    L, t = np.array([r["loading_ug"] for r in rows]), np.array([r["tau"] for r in rows])
    out = []
    for key, label, model, start, lower, upper in (
            ("curve", "The saturating curve", rc.hyperbolic, (0.04, 2.0), (0, 0.2), (1, 100)),
            ("alpha10", "White 2025, α = 10", lambda L, a: warren.two_zone(a * L, 10), (0.04,), (0,), (1,)),
            ("free", "White 2025, α free", lambda L, a, al: warren.two_zone(a * L, al), (0.04, 10.0), (0, 1.0), (1, 1e4)),
            ("full", "White 2025, fully pixelated", lambda L, a: rc.patterned(L, a, warren.H), (0.04,), (0,), (1,))):
        p, rms = checks.fit_medians(L, t, model, start, lower, upper)
        out.append({"key": key, "label": label, "params": [round(float(v), 4) for v in p], "rms_log": round(rms, 4),
                    "aic": round(float(len(L) * np.log(rms ** 2) + 2 * len(p)), 2)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, help="parquet of every IMPROVE sample (export_hips_ceiling_checks --cache)")
    parser.add_argument("--boot", type=int, default=60, help="site resamples (default %(default)s)")
    args = parser.parse_args()
    samples = warren.with_area(checks.improve_all(args.cache))
    samples = samples[samples["area"] == warren.AREA].copy()
    year, month = samples["date"].dt.year, samples["date"].dt.month
    cells, robust, windows, years = [], [], [], []
    for skey, slabel, months in SEASONS:
        season = month.isin(months)
        for ekey, elabel, lo, hi in ERAS:
            part = samples[season & year.between(lo, hi)]
            cell = {"season": skey, "season_label": slabel, "era": ekey, "era_label": elabel, **fitted(part, args.boot)}
            cells.append(cell)
            print(f"{slabel:18s} {elabel} n={cell['filters']:>7,} τ∞ {cell['tau_inf']:.2f} bend {cell['bend']:.2f} [{cell['bend_lo']:.2f}, {cell['bend_hi']:.2f}]", flush=True)
            if skey != "all":
                for column, label in BASES:
                    r = fitted(part, args.boot, column)
                    robust.append({"season": skey, "era": ekey, "basis": column, "label": label,
                                   **{k: r[k] for k in ("filters", "tau_inf", "rms_log", "bend", "bend_lo", "bend_hi")}})
                    print(f"    {label:28s} τ∞ {r['tau_inf']:.2f} rms {r['rms_log']:.3f}", flush=True)
        last = int(year.max())
        for start in range(2008, last - WINDOW + 2):
            r = fitted(samples[season & year.between(start, start + WINDOW - 1)], args.boot)
            windows.append({"season": skey, "from": start, "to": start + WINDOW - 1, **{k: r[k] for k in ("filters", "fabs_per_ec", "tau_inf", "bend", "bend_lo", "bend_hi")},
                            "darkest": r["groups"][-1]["tau"]})
        for y in range(2005, last + 1):
            part = samples[season & (year == y)].dropna(subset=["previous_ug"])
            rows = groups(*part[["previous_ug", "loading_ug", "tau"]].to_numpy().T, min_n=YEAR_MIN_N)
            years.append({"season": skey, "year": y, "groups": [[g, n, round(L, 3), round(t, 4)] for g, n, L, t in rows]})
    pooled = []
    for skey, slabel, months in SEASONS:
        cell = fitted(samples[month.isin(months) & (year >= 2008)], args.boot)
        pooled.append({"season": skey, "label": slabel, **cell, "models": models(cell["groups"])})
        print(slabel, [(m["key"], m["params"], m["rms_log"], m["aic"]) for m in pooled[-1]["models"]], flush=True)
    payload = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "source": "gallery/data/export_hips_improve_periods.py; IMPROVE TOR and HIPS tables, standard 3.53 cm2 deposits",
               "n_boot": args.boot, "window": WINDOW, "min_n": MIN_N, "year_min_n": YEAR_MIN_N, "edges": [float(e) for e in EDGES],
               "area_cm2": warren.AREA, "cells": cells, "robust": robust, "windows": windows, "years": years, "pooled": pooled}
    OUT.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    print(f"wrote {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1024:.1f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
