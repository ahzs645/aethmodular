#!/usr/bin/env python
"""White et al. (2025)'s own test of the HIPS bend, rerun: IMPROVE against CSN at the four collocated cities (24 Nov 2026).

CSN draws about a tenth of IMPROVE's air per filter area through the same HIPS instrument, so its Fabs is the same air read at
a tenth of the darkness. White et al. set the amplitude alpha = 10 by making the two networks' means agree (2019-2022). Here
the daily pairs are grouped (by IMPROVE's TOR EC that day, a third measurement with neither HIPS error in it) and the curve and
White's model are fitted to how IMPROVE's optical depth follows the one CSN implies. No FTIR enters, and TOR only sorts the days.

CSN Fabs: EPA AQS parameter 63102, Met One SASS, public daily files (from Aug 2020). IMPROVE: the laboratory's HIPS table,
standard 3.53 cm2 deposits. Public data only; PHOE1 is the routine Phoenix sampler, never the one with drain discs.

    uv run python gallery/data/export_hips_csn.py --aqs DIR [--cache improve_all.parquet] [--boot 300]

DIR holds (or receives, about 110 MB) daily_SPEC_YYYY.zip from https://aqs.epa.gov/aqsweb/airdata/.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "research/ftir_hips_chem/workflows"), str(ROOT / "research/ftir_hips_chem/scripts")]

import data_paths  # noqa: E402
import export_hips_ceiling_checks as checks  # noqa: E402
import export_hips_warren as warren  # noqa: E402

OUT = ROOT / "gallery/app/public/data/hips_linearise/csn.json"
URL = "https://aqs.epa.gov/aqsweb/airdata/daily_SPEC_{}.zip"
YEARS = range(2020, 2026)
SITES = {"BIRM1": ("01-073-0023", "Birmingham"), "FRES1": ("06-019-0011", "Fresno"), "PHOE1": ("04-013-9997", "Phoenix"), "PUSO1": ("53-033-0080", "Seattle")}
EDGES = [0, 2, 4, 6, 8, 11, 15, 20, 27, 36, 50, 70, 300]      # µg TOR EC on the IMPROVE filter that day
MIN_N, FIRE, SEED = 20, (7, 8, 9, 10), 20261124
PREVIOUS_EDGES = [0, 4, 6, 8, 11, 15, 20, 27, 36, 300]


def csn(folder: Path) -> pd.DataFrame:
    folder.mkdir(parents=True, exist_ok=True)
    wanted = {v[0]: k for k, v in SITES.items()}
    parts = []
    for year in YEARS:
        path = folder / f"daily_SPEC_{year}.zip"
        if not path.is_file():
            urllib.request.urlretrieve(URL.format(year), path)
        with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
            d = pd.read_csv(f, low_memory=False, dtype={"State Code": str, "County Code": str, "Site Num": str},
                            usecols=["State Code", "County Code", "Site Num", "Parameter Code", "Date Local", "Arithmetic Mean", "Method Name"])
        d = d[(d["Parameter Code"] == 63102) & d["Method Name"].str.contains("SASS", na=False)].copy()
        d["Site"] = (d["State Code"] + "-" + d["County Code"] + "-" + d["Site Num"]).map(wanted)
        parts.append(d.dropna(subset=["Site"]))
    d = pd.concat(parts)
    d["date"], d["csn"] = pd.to_datetime(d["Date Local"]), d["Arithmetic Mean"] * 100     # AQS gives it per 100 Mm
    return d.groupby(["Site", "date"])["csn"].first().reset_index()


def improve() -> pd.DataFrame:
    h = pd.read_csv(data_paths.ftir_local_db() / "tables/results_hips.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Site", "SampleDate", "Parameter", "Value", "AverageFlowRate", "ElapsedTime", "SampleDepositArea"])
    h = h[h["Site"].isin(SITES) & (h["Parameter"] == "fAbs")].copy()
    h["date"] = pd.to_datetime(h["SampleDate"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    h["volume"], h["fabs"] = h["AverageFlowRate"] * h["ElapsedTime"] / 1000, pd.to_numeric(h["Value"], errors="coerce")
    h = h[h["SampleDepositArea"].round(2).eq(warren.AREA) & h["volume"].between(20, 45)].dropna(subset=["fabs"])
    h["tau"] = h["fabs"] * h["volume"] / (100 * warren.AREA)
    return h.groupby(["Site", "date"])[["fabs", "tau", "volume"]].first().reset_index()


def group(p: pd.DataFrame, by: str) -> pd.DataFrame:
    q = p.dropna(subset=[by])
    key = pd.cut(q[by], EDGES if by == "loading_ug" else PREVIOUS_EDGES) if by in ("loading_ug", "previous_ug") else pd.qcut(q[by], 14)
    g = q.groupby(key, observed=True).agg(n=("x", "size"), x=("x", "median"), tau=("tau", "median"))
    return g[g["n"] >= MIN_N]


def curve(x, c):
    return x / (1 + x / c)


# the scale s is a constant difference between the networks (inlet, deposit area, blanks), the same for a light and a dark filter
MODELS = {"curve": (lambda x, s, c: curve(s * x, c), [1.0, 2.0]), "alpha": (lambda x, s, a: warren.two_zone(s * x, a), [1.0, 10.0]),
          "curve1": (lambda x, c: curve(x, c), [2.0]), "alpha1": (lambda x, a: warren.two_zone(x, a), [10.0]),
          "alpha10": (lambda x, s: warren.two_zone(s * x, 10), [1.0]), "line": (lambda x, s: s * x, [1.0])}


def fit(g: pd.DataFrame, key: str):
    model, start = MODELS[key]
    x, t = g["x"].to_numpy(), g["tau"].to_numpy()
    r = least_squares(lambda q: np.log(np.maximum(model(x, *np.exp(q)), 1e-9)) - np.log(t), np.log(start), bounds=(np.log(0.05), np.log(1e4)))
    rms = float(np.sqrt(np.mean(r.fun ** 2)))
    return [float(v) for v in np.exp(r.x)], rms, float(len(x) * np.log(rms ** 2) + 2 * len(r.x))


def blocks(p: pd.DataFrame):
    return [v for _, v in p.groupby(p["Site"] + p["date"].dt.strftime("%Y%m"))]


def resample(bs, rng):
    return pd.concat([bs[i] for i in rng.integers(0, len(bs), len(bs))])


def fitted(p: pd.DataFrame, n_boot: int, by: str = "loading_ug") -> dict:
    g = group(p, by)
    out = {k: dict(zip(("p", "rms_log", "aic"), fit(g, k))) for k in MODELS}
    rng, bs, b = np.random.default_rng(SEED), blocks(p), {k: [] for k in MODELS}
    need = {i: [] for i in range(len(g))}
    for _ in range(n_boot):
        h = group(resample(bs, rng), by)
        if len(h) < 5:
            continue
        for k in MODELS:
            b[k].append(fit(h, k)[0])
        if by == "loading_ug":
            for i, name in enumerate(g.index):
                if name in h.index:
                    need[i].append(h.loc[name, "x"] / h.loc[name, "tau"])
    for k, v in out.items():
        lo, hi = np.percentile(b[k], [5, 95], axis=0)
        v.update(p=[round(x, 4) for x in v["p"]], lo=[round(float(x), 4) for x in lo], hi=[round(float(x), 4) for x in hi], rms_log=round(v["rms_log"], 4), aic=round(v["aic"], 2))
    return {"pairs": int(len(p)), "groups": [{"n": int(r.n), "x": round(float(r.x), 4), "tau": round(float(r.tau), 4), **({"lo": round(float(np.percentile(need[i], 5)), 4), "hi": round(float(np.percentile(need[i], 95)), 4)} if need[i] else {})} for i, r in enumerate(g.itertuples())], "fits": out}


GRID = np.linspace(0, 60, 600001)
A10 = warren.two_zone(GRID, 10)


def means(p: pd.DataFrame, label: str, n_boot: int) -> dict:
    """The test exactly as White et al. (2025) ran it: the two networks' mean Fabs, IMPROVE's under each correction"""
    def of(q):
        t, f = q["tau"].to_numpy(), q["fabs"].to_numpy()
        at = lambda c: np.nanmean(np.where(t < c, f / (1 - t / c), np.nan))  # noqa: E731
        return [q["csn"].mean(), f.mean(), np.mean(f * np.interp(t, A10, GRID) / t), at(2.2602), at(3.2552)]
    rng, bs = np.random.default_rng(SEED), blocks(p)
    b = np.array([of(resample(bs, rng)) for _ in range(n_boot)])
    ratio = b / b[:, :1]
    v = of(p)
    return {"label": label, "pairs": int(len(p)), "past": int((p["tau"] >= 2.2602).sum()),
            "rows": [{"key": k, "mean": round(float(m), 3), "ratio": round(float(m / v[0]), 4), "lo": round(float(np.percentile(ratio[:, i], 5)), 4), "hi": round(float(np.percentile(ratio[:, i], 95)), 4)}
                     for i, (k, m) in enumerate(zip(("csn", "none", "alpha10", "curve", "rest"), v))],
            "darkest_2pct_share": round(float(p["csn"].nlargest(max(1, len(p) // 50)).sum() / p["csn"].sum()), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aqs", type=Path, required=True, help="folder of EPA daily_SPEC_YYYY.zip files (downloaded when missing)")
    parser.add_argument("--cache", type=Path, help="parquet of every IMPROVE sample (export_hips_ceiling_checks --cache)")
    parser.add_argument("--boot", type=int, default=300)
    args = parser.parse_args()
    tor = checks.improve_all(args.cache)[["Site", "date", "loading_ug", "previous_ug"]]
    m = improve().merge(csn(args.aqs), on=["Site", "date"]).merge(tor, on=["Site", "date"], how="left")
    m = m[(m["csn"] > 0) & (m["tau"] > 0)].copy()
    m["x"] = m["csn"] * m["volume"] / (100 * warren.AREA)          # the depth IMPROVE's filter would have at CSN's Fabs
    month = m["date"].dt.month
    cuts = [("all", "All pairs", pd.Series(True, m.index)), ("rest", "November to June", ~month.isin(FIRE)), ("fire", "July to October", month.isin(FIRE)),
            ("white", "Aug 2020 to 2022 (White's years)", m["date"] < "2023"), ("later", "2023 to 2025 (after the paper)", m["date"] >= "2023"),
            *((s, SITES[s][1], m["Site"] == s) for s in SITES)]
    sets = []
    for key, label, keep in cuts:
        r = {"key": key, "label": label, **fitted(m[keep], args.boot)}
        sets.append(r)
        print(label, r["pairs"], {k: (v["p"], v["lo"], v["hi"], v["rms_log"]) for k, v in r["fits"].items() if k in ("curve", "curve1", "alpha", "alpha10")}, flush=True)
    by = [{"key": k, "label": label, **{n: v for n, v in fitted(m, args.boot, k).items() if n != "groups"}} for k, label in
          (("loading_ug", "IMPROVE TOR EC that day (used)"), ("previous_ug", "IMPROVE TOR EC, previous filter"), ("tau", "IMPROVE's own τ"), ("x", "CSN's own Fabs"))]
    for r in by:
        print(r["label"], r["fits"]["curve"]["p"], r["fits"]["curve1"]["p"], flush=True)
    # a mean past tau 1.5 is the curve's few filters at its own ceiling, so it is given with and without them
    mean_rows = [{"key": key, "dark": dark, **means(m[keep & (m["tau"] < top)], label, args.boot)} for key, label, keep in cuts if key in ("white", "later", "all")
                 for dark, top in (("all", np.inf), ("below", 1.5))]
    for r in mean_rows:
        print(r)
    q = [5, 25, 50, 75, 95]
    payload = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "source": "EPA AQS daily_SPEC files, parameter 63102 (Met One SASS, HIPS); IMPROVE HIPS and TOR tables, 3.53 cm2 deposits",
               "n_boot": args.boot, "first": m["date"].min().strftime("%Y-%m-%d"), "last": m["date"].max().strftime("%Y-%m-%d"),
               "sites": [{"code": s, "name": SITES[s][1], "aqs": SITES[s][0], "pairs": int((m["Site"] == s).sum())} for s in SITES],
               "edges": EDGES, "min_n": MIN_N, "sets": sets, "by": by, "means": mean_rows,
               "published": {"csn": 8.97, "improve_alpha10": 8.94, "pairs": 1628},
               "range": {"improve_tau": [round(float(v), 3) for v in np.percentile(m["tau"], q)], "csn_tau": [round(float(v), 4) for v in np.percentile(m["csn"] * 9.65 / 1186, q)]},
               "pairs": [[list(SITES).index(r.Site), r.date.strftime("%Y-%m-%d"), round(r.tau, 3), round(r.x / r.tau, 3)] for r in m.itertuples()]}
    OUT.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    print(f"wrote {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1024:.1f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
