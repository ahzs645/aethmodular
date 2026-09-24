"""Prioritization categories re-run on baselined spectra only (meeting 2026-09-17).

Levers fixed at the meeting: no raw, no 2nd derivative; AIRSpec and VIBES in
parallel; correlation selection masks CO2 (1800-2500) and >3600; CV is either
5-site grouped (``site_heldout``) or 10-fold interleaved (``app``). Cutoffs step
by 5 filters. Each row is one fitted cohort read out on Addis (all + each season)
against HIPS Fabs/MAC 10. Resumable: rows already in the JSONL are skipped.

Run from the repo root:
    uv run python research/ftir_hips_chem/workflows/run_meeting_grid_20260917.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

AREA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AREA / "scripts"))
import analog_bias as ab  # noqa: E402
from pls_transfer import spectral_q_residual  # noqa: E402

OUT = AREA / "output/tables/meeting_followup_20260917"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "category_grid.jsonl"
MODES = ("site_heldout", "app")
STEP = 5
RANGES = {"ocec": (300, 1500), "eth_shaped": (100, 600), "vip_analogs": (250, 750),
          "corr_all": (100, 1500), "corr_dry": (100, 1500), "corr_belg": (100, 1500),
          "corr_kiremt": (100, 1500)}
LABELS = {"pool": "Entire IMPROVE pool", "smoke": "Biomass smoke (906)",
          "ocec": "Lowest OC/EC", "eth_shaped": "Ethiopia-shaped smoke",
          "vip_analogs": "VIP/Euclidean analogs", "corr_all": "Correlation analogs: all Addis",
          "corr_dry": "Correlation analogs: Dry", "corr_belg": "Correlation analogs: Belg",
          "corr_kiremt": "Correlation analogs: Kiremt"}
LOCK = threading.Lock()


def rankings(ctx, method):
    gm = ab.group_masks(ctx)
    r = {"pool": np.arange(len(ctx.lib)), "smoke": ab.smoke_positions(ctx),
         "ocec": ab.ocec_ranking(ctx), "eth_shaped": ab.eth_shaped_ranking(ctx, method),
         "corr_all": ab.correlation_ranking(ctx, method, gm["All Addis"])}
    for key, season in zip(("corr_dry", "corr_belg", "corr_kiremt"), ab.SEASONS):
        r[key] = ab.correlation_ranking(ctx, method, gm[season])
    if method == "AIRSpec":      # the VIP ranking exists only in AIRSpec space
        r["vip_analogs"] = ab.vip_analog_ranking(ctx)
    return r


def configs(ctx):
    out = []
    for method in ctx.methods:
        for cat, ranked in rankings(ctx, method).items():
            if cat in RANGES:
                lo, hi = RANGES[cat]
                cuts = [c for c in range(lo, hi + 1, STEP) if c <= len(ranked)]
            else:
                cuts = [len(ranked)]
            for cut in cuts:
                for mode in MODES:
                    out.append((method, cat, cut, mode, ranked[:cut]))
    return out


def key(method, cat, cut, mode):
    return f"{method}|{cat}|{cut}|{mode}"


def run_one(ctx, gm, method, cat, cut, mode, pos, sink):
    lib = ctx.lib.iloc[pos]
    X = ctx.X[method][pos]
    f = ab.fit_cohort(X, lib.y, lib.Site, mode=mode)
    pa = f.predict(ctx.XA[method]) / ctx.addis.Volume_m3.to_numpy(float)
    # the explorer's two applicability guardrails, computed exactly as run_config does:
    # whitened score distance beyond the training p95, and spectral Q residual beyond p95
    Ttr, Tev = f.model.transform(X[f.train]), f.model.transform(ctx.XA[method])
    mu, sd = Ttr.mean(axis=0), Ttr.std(axis=0) + 1e-12
    d_tr = np.sqrt((((Ttr - mu) / sd) ** 2).sum(axis=1))
    d_ev = np.sqrt((((Tev - mu) / sd) ** 2).sum(axis=1))
    q_tr = spectral_q_residual(f.model, X[f.train])
    q_ev = spectral_q_residual(f.model, ctx.XA[method])
    row = {"key": key(method, cat, cut, mode), "method": method, "category": cat,
           "label": LABELS[cat], "cutoff": int(cut), "mode": mode, "k": f.k,
           "train_n": int(f.train.sum()), "train_sites": int(lib.Site[f.train].nunique())}
    if mode == "site_heldout":
        test = ~f.train
        tt = ab.xy_stats(lib.y.to_numpy()[test], f.predict(ctx.X[method][pos])[test])
        row["heldout_R2"] = tt.get("R2")
    x = ctx.addis.Fabs.to_numpy(float) / ab.MAC_VALUE
    for g, take in gm.items():
        st = ab.xy_stats(x[take], pa[take], sigma_x=ctx.sigma_x, sigma_y=ab.SIGMA_Y_PROXY)
        tag = "all" if g == "All Addis" else g.split()[0].lower()
        row[f"{tag}_slope"] = st["deming_slope"]
        row[f"{tag}_intercept"] = st["deming_intercept"]
        row[f"{tag}_R2"] = st["R2"]
        row[f"{tag}_median_pred"] = float(np.median(pa[take]))
    row["negative_pct"] = float(100 * (pa < 0).mean())
    row["extrap_pct"] = float(100 * (d_ev > np.percentile(d_tr, 95)).mean())
    row["q_residual_pct"] = float(100 * (q_ev > np.percentile(q_tr, 95)).mean())
    with LOCK:
        sink.write(json.dumps(row) + "\n")
        sink.flush()
    return 1


def main():
    t0 = time.time()
    ctx = ab.load_context()
    gm = ab.group_masks(ctx)
    done = set()
    if RESULTS.exists():
        done = {json.loads(l)["key"] for l in RESULTS.read_text().splitlines() if l.strip()}
    todo = [c for c in configs(ctx) if key(*c[:4]) not in done]
    # big cohorts first so the long pool fits don't straggle at the end
    todo.sort(key=lambda c: -len(c[4]))
    print(f"{len(todo)} fits to run ({len(done)} already done)", flush=True)
    with RESULTS.open("a") as sink, threadpool_limits(limits=1):
        n = 0
        for chunk in range(0, len(todo), 240):
            part = todo[chunk:chunk + 240]
            n += sum(Parallel(n_jobs=12, backend="threading")(
                delayed(run_one)(ctx, gm, *c, sink) for c in part))
            print(f"[{time.time() - t0:.0f}s] {n}/{len(todo)}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
