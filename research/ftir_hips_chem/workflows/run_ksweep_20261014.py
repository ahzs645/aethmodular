"""PLS component-count (k) sweep for the Addis intercept (14 Oct 2026).

Run from the repo root:
    uv run --no-sync python research/ftir_hips_chem/workflows/run_ksweep_20261014.py

Makes the 8 Oct scratch finding a repo workflow: the Addis intercept falls as
the number of PLS components k grows, while held-out IMPROVE R² stays flat from
k ~ 8, so IMPROVE validation cannot see it. And it adds the check that decides
what that means: do smoke-like IMPROVE filters (high non-soil potassium, high
OC/EC, the phase-2 smoke cohort) drift with k the way Addis does?

  yes  IMPROVE contains a proxy for the Addis failure, and k (or the cohort) can
       be chosen on it without Addis labels.
  no   the k-drift is an Addis-only response shift that no IMPROVE validation
       can catch.

Design (rank-70 context of the Satoshi 2 Oct / Oct 1 tabs, ``vibes_current.load_context``):

  baselines  spline (AIRSpec), VIBES capped 87-blank bank (full 4000-500 and the
             4000-1425 cut), and the two minus-LABE1 banks (lot-251 287 blanks,
             all-labeled 365 blanks) from the 8 Oct Colab reruns on Drive.
  cohorts    the standard all-IMPROVE lot-251 model, and seasonal analogs
             (500, "oct1" region) for All Addis, Dry, Belg and Kiremt.
  splits     100 site-disjoint 80/20 splits per cohort (``ab.repeated_site_splits``).
  k          1..25 from one fit at k_max per split (``pls_ksweep.prefix_predict``).

Readouts per split and k: Addis Deming slope/intercept vs HIPS Fabs / MAC 10
(sigma_x from HIPS_Uncertainty, sigma_y proxy 0.531, as every Addis panel so
far); for held-out-site IMPROVE filters in each subset, mean bias and OLS
slope/intercept vs TOR EC and vs IMPROVE HIPS Fabs / 10 (x treated as exact).
The Addis Deming lambda is NOT reused for IMPROVE: its sigma_x (0.31 µg/m³) is
the size of the whole IMPROVE smoke-subset mean (~0.34), so it inflates the
slope and drives the intercept strongly negative for reasons unrelated to k. Subsets are fixed from IMPROVE data alone, before any
Addis readout. Pooled out-of-fold predictions (each filter averaged over the
splits that held its site out) give one curve per subset as well.

Nothing is tuned on Addis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
sys.path.insert(0, str(AREA / "workflows"))
import analog_bias as ab  # noqa: E402
import analog_transfer_checks as ad  # noqa: E402
import pls_ksweep as pk  # noqa: E402
import run_oct1_remaining_20261007 as o1  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from vibes_current import load_context  # noqa: E402

OUT = AREA / "output/tables/ksweep_20261014"
SPLINE, CAPPED, CUT = "AIRSpec", "VIBES-full-r70", "VIBES-full-cut-r70"
X1_BANKS = ("VIBES-lot251-x1", "VIBES-allblanks-x1")
LABELS = {SPLINE: "Spline baseline", CAPPED: "VIBES, 87 blanks (4000–500)",
          CUT: "VIBES, 87 blanks, cut to 4000–1425",
          "VIBES-lot251-x1": "VIBES, 287 lot-251 blanks minus LABE1",
          "VIBES-allblanks-x1": "VIBES, 365 labeled blanks minus LABE1"}
K_MAX = 25
N_REPEATS = 100
LOT = 251
SUBSET_LABELS = {
    "all": "all held-out IMPROVE",
    "nsK_top10": "non-soil K (K − 0.6 Fe) ≥ library p90",
    "nsK_bottom50": "non-soil K ≤ library median",
    "ocec_top10": "TOR OC/EC ≥ library p90",
    "smoke_cohort": "phase-2 smoke cohort",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- inputs
def load_banks(ctx, data):
    """Attach only the minus-LABE1 banks (the 7 Oct big banks were set by one bad blank)."""
    keep = {m: o1.POLICIES[m] for m in X1_BANKS}
    saved, o1.POLICIES = o1.POLICIES, keep
    try:
        runs, sources = o1.load_policy_runs(ctx, data)
    finally:
        o1.POLICIES = saved
    missing = set(X1_BANKS) - set(runs)
    if missing:
        raise FileNotFoundError(f"minus-LABE1 runs not found on Drive: {sorted(missing)}")
    return runs, sources


def subsets(ctx):
    """IMPROVE-only subsets, fixed before any fit (bool masks over ctx.lib)."""
    lib = ctx.lib
    years = sorted(lib.date.dt.year.unique())
    chem = pk.load_improve_nsk(years, cache=OUT / "improve_k_fe.csv")
    nsk = lib[["Site", "date"]].merge(chem, on=["Site", "date"], how="left", validate="many_to_one")
    v = nsk.nsK_ugm3.to_numpy(float)
    ctx.lib["nsK_ugm3"] = v
    ocec = lib.tor_oc_ec.where(lib.tor_ec_ugm3 > 0.05).to_numpy(float)
    smoke = np.zeros(len(lib), bool)
    smoke[ab.smoke_positions(ctx)] = True
    masks = {"all": np.ones(len(lib), bool),
             "nsK_top10": v >= np.nanpercentile(v, 90),
             "nsK_bottom50": v <= np.nanpercentile(v, 50),
             "ocec_top10": ocec >= np.nanpercentile(ocec, 90),
             "smoke_cohort": smoke}
    return masks, float(np.nanmean(np.isfinite(v)))


def cohorts(ctx, gm, method):
    """Cohort name -> (lib positions, Addis groups it is judged on)."""
    ok = np.ones(len(ctx.lib), bool)
    ok[getattr(ctx, "failed", {}).get(method, [])] = False
    lot = np.flatnonzero(ctx.lib.lot.eq(LOT).to_numpy() & ok)
    out = {"standard": (lot, list(gm))}
    for g, take in gm.items():
        pos, _ = ad.select_analogs(ctx, method, take, "oct1")
        out[f"analog|{g}"] = (pos[ok[pos]], [g])
    return out


# --------------------------------------------------------------------------- one split
def _addis_rows(ctx, gm, groups, pa, base):
    x_all = ctx.addis.Fabs.to_numpy(float) / ab.MAC_VALUE
    rows = []
    for g in groups:
        t = gm[g]
        x, y = x_all[t], pa[t]
        ds, di = pk.deming(x, y, ctx.lam)
        os_, oi = pk.ols(x, y)
        rr = pk.r2(x, y)
        bias = y.mean(0) - x.mean()
        for j, k in enumerate(base["ks"]):
            rows.append({**base["key"], "readout": "addis", "set": g, "k": k, "n": int(t.sum()),
                         "slope": ds[j], "intercept": di[j], "ols_slope": os_[j], "ols_intercept": oi[j],
                         "R2": rr[j], "bias": bias[j], "x_mean": float(x.mean())})
    return rows


def _improve_rows(ctx, masks, pos, test, pt, base):
    lib = ctx.lib.iloc[pos]
    tor = lib.tor_ec_ugm3.to_numpy(float)[test]
    fabs10 = lib.fabs_Mm1.to_numpy(float)[test] / ab.MAC_VALUE
    rows = []
    for name, mask in masks.items():
        t = mask[pos][test] & np.isfinite(tor)
        n = int(t.sum())
        if n < 10:
            continue
        x, y = tor[t], pt[t]
        os_, oi = pk.ols(x, y)
        rr = pk.r2(x, y)
        bias = y.mean(0) - x.mean()
        h = t & np.isfinite(fabs10)
        hs, hi = pk.ols(fabs10[h], pt[h]) if h.sum() >= 10 else (np.full(len(base["ks"]), np.nan),) * 2
        hb = pt[h].mean(0) - fabs10[h].mean() if h.sum() >= 10 else np.full(len(base["ks"]), np.nan)
        for j, k in enumerate(base["ks"]):
            rows.append({**base["key"], "readout": "improve", "set": name, "k": k, "n": n,
                         "slope": os_[j], "intercept": oi[j], "R2": rr[j], "bias": bias[j],
                         "hips_slope": hs[j], "hips_intercept": hi[j], "hips_bias": hb[j],
                         "x_mean": float(x.mean()), "hips_x_mean": float(fabs10[h].mean())})
    return rows


def split_fit(ctx, gm, masks, method, cohort, pos, groups, repeat, train, Xc):
    lib = ctx.lib.iloc[pos]
    y = lib.y.to_numpy(float)
    k_max = min(K_MAX, int(train.sum()) - 2)
    ks = list(range(1, k_max + 1))
    model = PLSRegression(n_components=k_max, scale=False).fit(Xc[train], y[train])
    B = pk.prefix_coefs(model, ks)
    pa = pk.prefix_predict(model, ctx.XA[method], ks, B) / ctx.addis.Volume_m3.to_numpy(float)[:, None]
    test = ~train
    pt = pk.prefix_predict(model, Xc[test], ks, B) / lib.tor_volume_m3.to_numpy(float)[test][:, None]
    base = {"ks": ks, "key": {"baseline": method, "cohort": cohort, "repeat": repeat}}
    rows = _addis_rows(ctx, gm, groups, pa, base) + _improve_rows(ctx, masks, pos, test, pt, base)
    return rows, pos[test], pt.astype(np.float32)


def protocol_k(ctx, gm, method, cohort, pos, groups):
    """The k the frozen protocol picks (standard: frozen train split; analogs: default)."""
    lib = ctx.lib.iloc[pos]
    train = lib.split.eq("train").to_numpy() if cohort == "standard" else None
    f = ab.fit_cohort(ctx.X[method][pos], lib.y, lib.Site, train=train)
    return {"baseline": method, "cohort": cohort, "protocol_k": f.k}


# --------------------------------------------------------------------------- stages
def run_baseline(ctx, gm, masks, method, n_repeats, n_jobs):
    from joblib import Parallel, delayed

    part = OUT / "parts" / f"{method}.csv"
    pooled_part = OUT / "parts" / f"{method}_pooled.csv"
    if part.exists() and pooled_part.exists():
        log(f"{method}: cached")
        return pd.read_csv(part), pd.read_csv(pooled_part)
    rows, pooled, proto = [], [], []
    for cohort, (pos, groups) in cohorts(ctx, gm, method).items():
        t0 = time.time()
        Xc = np.ascontiguousarray(ctx.X[method][pos])
        sites = ctx.lib.Site.to_numpy()[pos]
        jobs = list(enumerate(ab.repeated_site_splits(sites, n_repeats)))
        nj = n_jobs if len(pos) < 2000 else max(1, n_jobs // 2)
        res = Parallel(n_jobs=nj, backend="threading")(
            delayed(split_fit)(ctx, gm, masks, method, cohort, pos, groups, r, tr, Xc) for r, tr in jobs)
        n_k = min(r[2].shape[1] for r in res)
        total = np.zeros((len(ctx.lib), n_k))
        count = np.zeros(len(ctx.lib))
        for r_rows, idx, pt in res:
            rows.extend(r_rows)
            total[idx] += pt[:, :n_k]
            count[idx] += 1
        held = count > 0
        oof = total[held] / count[held][:, None]
        pooled.extend(_pooled_rows(ctx, masks, np.flatnonzero(held), oof, method, cohort))
        proto.append(protocol_k(ctx, gm, method, cohort, pos, groups))
        log(f"{method} {cohort}: {len(jobs)} splits, n={len(pos)}, {time.time() - t0:.0f}s")
    part.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).merge(pd.DataFrame(proto), on=["baseline", "cohort"])
    frame.to_csv(part, index=False)
    pd.DataFrame(pooled).to_csv(pooled_part, index=False)
    return frame, pd.DataFrame(pooled)


def _pooled_rows(ctx, masks, idx, oof, method, cohort):
    lib = ctx.lib.iloc[idx]
    tor = lib.tor_ec_ugm3.to_numpy(float)
    fabs10 = lib.fabs_Mm1.to_numpy(float) / ab.MAC_VALUE
    out = []
    for name, mask in masks.items():
        t = mask[idx] & np.isfinite(tor)
        if t.sum() < 10:
            continue
        os_, oi = pk.ols(tor[t], oof[t])
        rr = pk.r2(tor[t], oof[t])
        h = t & np.isfinite(fabs10)
        hs, hi = pk.ols(fabs10[h], oof[h])
        for j in range(oof.shape[1]):
            out.append({"baseline": method, "cohort": cohort, "set": name, "k": j + 1, "n": int(t.sum()),
                        "slope": os_[j], "intercept": oi[j], "R2": rr[j],
                        "bias": float(oof[t, j].mean() - tor[t].mean()),
                        "hips_slope": hs[j], "hips_intercept": hi[j],
                        "hips_bias": float(oof[h, j].mean() - fabs10[h].mean()),
                        "x_mean": float(tor[t].mean()), "hips_x_mean": float(fabs10[h].mean())})
    return out


def reference_gap(ctx, masks):
    """TOR EC vs IMPROVE HIPS Fabs / 10 in each subset (OLS, no FTIR involved).

    A model that reproduces TOR inherits this line when it is judged against
    HIPS, so it is the floor for the subset's FTIR-vs-HIPS intercept at high k.
    """
    lib = ctx.lib
    tor, h = lib.tor_ec_ugm3.to_numpy(float), lib.fabs_Mm1.to_numpy(float) / ab.MAC_VALUE
    rows = []
    for name, mask in masks.items():
        t = mask & np.isfinite(tor) & np.isfinite(h)
        s, i = pk.ols(h[t], tor[t])
        mac = lib.mac_ratio.to_numpy(float)[t & (tor > 0.05)]
        rows.append({"set": name, "n": int(t.sum()), "slope": s[0], "intercept": i[0],
                     "intercept_rel": i[0] / tor[t].mean(), "x_mean": h[t].mean(),
                     "median_mac": float(np.median(mac))})
    return pd.DataFrame(rows)


def summarize(frame):
    """Median and IQR over splits for every (baseline, cohort, readout, set, k)."""
    keys = ["baseline", "cohort", "readout", "set", "k"]
    frame = frame.assign(intercept_rel=frame.intercept / frame.x_mean,
                         hips_intercept_rel=frame.hips_intercept / frame.get("hips_x_mean"))
    cols = ["slope", "intercept", "intercept_rel", "R2", "bias", "x_mean",
            "hips_slope", "hips_intercept", "hips_intercept_rel", "hips_bias", "hips_x_mean"]
    g = frame.groupby(keys)[cols]
    med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
    out = med.join(lo.add_suffix("_q25")).join(hi.add_suffix("_q75"))
    out["n_median"] = frame.groupby(keys)["n"].median()
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--baselines", default=",".join([SPLINE, CAPPED, CUT, *X1_BANKS]))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default=None, help="override the output folder (smoke tests)")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx, data = load_context()
    gm = ab.group_masks(ctx)
    methods = args.baselines.split(",")
    sources = {}
    if set(methods) & set(X1_BANKS):
        _, sources = load_banks(ctx, data)
    masks, nsk_coverage = subsets(ctx)
    log(f"subsets: " + ", ".join(f"{k} {int(v.sum())}" for k, v in masks.items()))
    reference_gap(ctx, masks).to_csv(OUT / "reference_gap.csv", index=False)
    frames, pooled = [], []
    for m in methods:
        f, p = run_baseline(ctx, gm, masks, m, args.repeats, args.jobs)
        frames.append(f)
        pooled.append(p)
    frame = pd.concat(frames, ignore_index=True)
    summarize(frame).to_csv(OUT / "curves.csv", index=False)
    pd.concat(pooled, ignore_index=True).to_csv(OUT / "pooled.csv", index=False)
    frame[["baseline", "cohort", "protocol_k"]].drop_duplicates().to_csv(OUT / "protocol_k.csv", index=False)
    summary = {"source": "Test 1 of the 14 Oct 2026 intercept plan (k-sweep + smoke/high-K IMPROVE check)",
               "baselines": {m: LABELS[m] for m in methods}, "subsets": SUBSET_LABELS,
               "subset_sizes": {k: int(v.sum()) for k, v in masks.items()},
               "nsK_coverage": nsk_coverage, "k_max": K_MAX, "n_repeats": args.repeats,
               "n_analogs": ab.N_ANALOGS, "analog_mask": ad.MASK_LABELS["oct1"], "mac": ab.MAC_VALUE,
               "lambda_addis": ctx.lam, "addis_n": len(ctx.addis), "improve_n": len(ctx.lib),
               "source_sha256": {**data["sources"], **sources}, "seconds": round(time.time() - t0)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
