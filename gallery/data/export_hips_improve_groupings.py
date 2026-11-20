"""How the IMPROVE ceiling depends on the grouping: data to regroup in the browser, and a sweep of groupings.

Run from the repo root with the repo environment (needs the FTIR local database's TOR and HIPS tables, or a cache):

    uv run python gallery/data/export_hips_improve_groupings.py
    uv run python gallery/data/export_hips_improve_groupings.py --cache /tmp/improve.parquet

The ceiling τ∞ = 2.26 is the saturating curve ``tau = aL / (1 + aL / tau_inf)`` fitted through the medians of 17
groups of IMPROVE filters, grouped by the *previous* sample's TOR EC loading at the same site, with hand-set edges
(run_addis_reference_closure_20261110.IMPROVE_EDGES) and groups under 150 filters left out. Ann (19 Nov 2026) asked
how much that choice matters. This writes two things for the gallery's "How the ceiling is decided" charts:

* ``bases``: for each way of grouping (the previous sample's loading, the filter's own loading, its own optical
  depth), the filters in fine bins of that quantity, and in each fine bin a histogram of TOR EC loading and of τ on
  4 % log bins. Any grouping whose edges fall on the fine edges can be rebuilt from these, its medians read off the
  merged histograms (within about 1 %), and refitted in the browser (lib/hipsGroupings.ts).
* ``sweep``: the ceiling under the published edges and under equal-count and log-spaced groupings of 4 to 40
  groups, for each basis, fitted on exact medians, with a 90 % interval from resampling whole sites.

The edge rules (published, equal count, log-spaced) are computed on the fine-bin counts, here and in the browser
alike, so a grouping in the sweep is the same grouping in the browser.

Writes gallery/app/public/data/hips_linearise/improve_groupings.json.
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

OUT = ROOT / "gallery/app/public/data/hips_linearise/improve_groupings.json"
SEED = 20261120
N_BOOT = 100
MIN_N = closure.IMPROVE_MIN_STRATUM
COUNTS = (4, 6, 8, 10, 12, 15, 17, 20, 25, 30, 40)
# value histograms: 4 % log bins; values outside fall into the end bins
RATIO = 1.04
L_RANGE, T_RANGE = (0.01, 5000.0), (0.001, 8.0)


def log_edges(lo: float, hi: float, ratio: float) -> np.ndarray:
    return np.exp(np.arange(np.log(lo), np.log(hi) + 1e-12, np.log(ratio)))


def loading_fine_edges() -> np.ndarray:
    """The published edges, each interval split in eight, then coarser steps to the top; starts at 0."""
    published = list(closure.IMPROVE_EDGES[:-1])                 # 0 … 100
    fine = []
    for a, b in zip(published[:-1], published[1:]):
        fine.extend(np.linspace(a, b, 9)[:-1])
    fine.extend([100, 120, 140, 170, 200, 250, 300, 400, 500, 700, 1000])
    return np.round(np.array(fine + [closure.IMPROVE_EDGES[-1]]), 6)    # as the browser reads them, so ties snap alike


def tau_fine_edges() -> np.ndarray:
    return np.round(np.concatenate([[0.0], log_edges(0.002, 6.0, 1.06), [100.0]]), 6)


BASES = {
    "previous": {"label": "Previous sample's TOR EC loading (µg)", "column": "previous_loading_ug", "edges": loading_fine_edges(),
                 "published": list(closure.IMPROVE_EDGES)},
    "own": {"label": "The filter's own TOR EC loading (µg)", "column": "loading_ug", "edges": loading_fine_edges(),
            "published": list(closure.IMPROVE_EDGES)},
    "tau": {"label": "The filter's own HIPS optical depth τ", "column": "tau", "edges": tau_fine_edges(), "published": None},
}


# ----------------------------------------------------------------------------- edge rules (mirrored in lib/hipsGroupings.ts)
def snap(value: float, fine: np.ndarray) -> float:
    """The fine edge nearest a value on a log scale (0 stays 0)."""
    pos = fine[fine > 0]
    return float(pos[np.argmin(np.abs(np.log(pos) - np.log(value)))])


def rule_edges(rule: str, n: int, counts: np.ndarray, fine: np.ndarray, published) -> list[float]:
    """Group edges, all on fine edges: [0, e1, …, top]. counts[i] is the filters in fine bin [fine[i], fine[i+1])."""
    top = float(fine[-1])
    if rule == "published":
        return [float(e) for e in published]
    cum = np.cumsum(counts) / counts.sum()
    if rule == "count":
        inner = [float(fine[np.searchsorted(cum, k / n) + 1]) for k in range(1, n)]
    else:                                                          # log-spaced between the 0.5 % and 99.95 % points
        lo, hi = float(fine[np.searchsorted(cum, 0.005) + 1]), float(fine[np.searchsorted(cum, 0.9995) + 1])
        inner = [snap(lo * (hi / lo) ** (k / (n - 2)), fine) for k in range(n - 1)] if n > 2 else [snap(np.sqrt(lo * hi), fine)]
    return [0.0, *sorted(set(e for e in inner if 0 < e < top)), top]


# ----------------------------------------------------------------------------- the fit (as closure._ceiling)
def ceiling(loading: np.ndarray, tau: np.ndarray) -> tuple[float, float]:
    residual = lambda p: np.log(rc.hyperbolic(loading, *p)) - np.log(tau)  # noqa: E731
    a, tau_inf = optimize.least_squares(residual, [0.04, 2.0], bounds=([0.0, 0.2], [1.0, 100.0])).x
    return float(a), float(tau_inf)


def group_medians(basis: np.ndarray, loading: np.ndarray, tau: np.ndarray, edges: list[float]):
    idx = np.digitize(basis, edges)
    frame = pd.DataFrame({"g": idx, "L": loading, "t": tau})
    g = frame.groupby("g").agg(n=("L", "size"), L=("L", "median"), t=("t", "median"))
    return g[g["n"] >= MIN_N]


def sweep(samples: pd.DataFrame, fine_counts: dict, n_boot: int) -> list[dict]:
    plans = []
    for key, b in BASES.items():
        rules = ([("published", 0)] if b["published"] else []) + [(r, n) for r in ("count", "log") for n in COUNTS]
        for rule, n in rules:
            plans.append({"basis": key, "rule": rule, "groups": n, "edges": rule_edges(rule, n, fine_counts[key], b["edges"], b["published"])})
    L, T = samples["loading_ug"].to_numpy(), samples["tau"].to_numpy()
    for p in plans:
        g = group_medians(samples[BASES[p["basis"]]["column"]].to_numpy(), L, T, p["edges"])
        p["used"], p["a"], p["tau_inf"] = int(len(g)), *ceiling(g["L"].to_numpy(), g["t"].to_numpy())
        p["boot"] = []
    rng = np.random.default_rng(SEED)
    by_site = {s: f for s, f in samples.groupby("Site")}
    sites = np.array(sorted(by_site))
    for i in range(n_boot):
        draw = pd.concat([by_site[s] for s in rng.choice(sites, sites.size)], ignore_index=True)
        L, T = draw["loading_ug"].to_numpy(), draw["tau"].to_numpy()
        for p in plans:
            g = group_medians(draw[BASES[p["basis"]]["column"]].to_numpy(), L, T, p["edges"])
            if len(g) >= 3:
                p["boot"].append(ceiling(g["L"].to_numpy(), g["t"].to_numpy())[1])
        if (i + 1) % 10 == 0:
            print(f"  resample {i + 1}/{n_boot}")
    out = []
    for p in plans:
        lo, hi = np.percentile(p.pop("boot"), [5, 95])
        out.append({**{k: p[k] for k in ("basis", "rule", "groups", "used")}, "edges": [round(e, 6) for e in p["edges"]],
                    "a_per_ug": round(p["a"], 6), "tau_inf": round(p["tau_inf"], 4), "tau_inf_lo": round(float(lo), 4), "tau_inf_hi": round(float(hi), 4)})
        print(f"{p['basis']:8s} {p['rule']:9s} {p['groups']:>2} → {p['used']:>2} groups  τ∞ {p['tau_inf']:.2f} [{lo:.2f}, {hi:.2f}]")
    return out


def histograms(samples: pd.DataFrame) -> tuple[dict, dict]:
    """Per basis and fine bin: the filter count and sparse histograms of loading and τ on the value bins."""
    l_edges, t_edges = log_edges(*L_RANGE, RATIO), log_edges(*T_RANGE, RATIO)
    li = np.clip(np.digitize(samples["loading_ug"].to_numpy(), l_edges) - 1, 0, len(l_edges) - 2)
    ti = np.clip(np.digitize(samples["tau"].to_numpy(), t_edges) - 1, 0, len(t_edges) - 2)
    bases, counts = {}, {}
    for key, b in BASES.items():
        fine = b["edges"]
        fi = np.clip(np.digitize(samples[b["column"]].to_numpy(), fine) - 1, 0, len(fine) - 2)
        bins = []
        for k in range(len(fine) - 1):
            m = fi == k
            lv, lc = np.unique(li[m], return_counts=True)
            tv, tc = np.unique(ti[m], return_counts=True)
            # sparse runs: [first value bin, count, count, …] keeps the file small; gaps inside a run are zeros
            run = lambda v, c: [] if not len(v) else [int(v[0]), *np.bincount(v - v[0], weights=c).astype(int).tolist()]  # noqa: E731
            bins.append({"n": int(m.sum()), "L": run(lv, lc), "t": run(tv, tc)})
        counts[key] = np.array([x["n"] for x in bins])
        bases[key] = {"label": b["label"], "fine": [round(float(e), 6) for e in fine], "published": b["published"], "bins": bins}
    return {"loading": {"lower": L_RANGE[0], "ratio": RATIO, "bins": len(l_edges) - 1},
            "tau": {"lower": T_RANGE[0], "ratio": RATIO, "bins": len(t_edges) - 1}, "bases": bases}, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, help="parquet of the IMPROVE samples: read if present, else written after the read")
    parser.add_argument("--boot", type=int, default=N_BOOT, help="site resamples per grouping (default %(default)s)")
    args = parser.parse_args()
    if args.cache and args.cache.is_file():
        samples = pd.read_parquet(args.cache)
    else:
        samples = closure.improve_samples()
        if samples is None:
            raise FileNotFoundError("IMPROVE TOR and HIPS tables are required; set AETHMODULAR_FTIR_LOCAL_DB")
        if args.cache:
            samples.to_parquet(args.cache)
    hist, counts = histograms(samples)
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "run_addis_reference_closure_20261110.improve_samples; gallery/data/export_hips_improve_groupings.py",
        "samples": int(len(samples)), "min_n": MIN_N, "counts": list(COUNTS), "n_boot": args.boot,
        "values": {"loading": hist["loading"], "tau": hist["tau"]}, "bases": hist["bases"],
        "sweep": sweep(samples, counts, args.boot),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1024:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
