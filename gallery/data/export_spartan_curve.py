"""SPARTAN's own HIPS curve: what SPARTAN's data already say about the loading effect, for the Experimental page "SPARTAN's own HIPS curve".

    uv run --no-sync python gallery/data/export_spartan_curve.py

The same numbers as research/improve_hips_offset/delhi_run_length_test_20261124.ipynb and
spartan_own_curve_design_20261124.ipynb (24 Nov 2026), from committed gallery exports only:

  addis      Addis sampled volume (why a dark filter is a dirty day there); HIPS optical depth against the aethalometer's
             loading (IR BC with the tape-step k x volume), fitted with tau = a L / (1 + a L / tau_inf), a free
  mac        the Addis MAC for slope 1 at each ceiling (FTIR EC on HIPS BC linearised, Deming lambda 2.96, MAC / slope)
  delhi      Delhi's filters (24, 12 and 6 h of sampling inside 9am-9am days at 5 L/min) and the regressions
             log(Fabs / ref) = b_V log V + b_C log ref + month (or season + year), SE clustered by month; placebos
  power      the paired light/dark experiment, simulated: fitted ceiling and Addis MAC against the number of pairs

Inputs: hips_linearise/linearise.json, experimental/aeth_loading_k.json (export_aeth_loading_k.py), spartan_reference.json,
weather/filters.json. Writes gallery/app/public/data/experimental/spartan_curve.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "gallery/app/public/data"
OUT = D / "experimental/spartan_curve.json"
TINF = 2.2596
K_TAPE = 0.0115          # Jacros IR tape-advance k (export_aeth_loading_k.py)
SEED = 20261124


def r(v, nd=4):
    return None if v is None or not np.isfinite(v) else float(round(float(v), nd))


def hyper(x, T):
    return x / (1 + x / T)


def inv(t, T):
    return t / (1 - t / T)


def fit_ceiling(x, t):
    f = lambda q: np.log(hyper(np.exp(q[0]) * x, np.exp(q[1]))) - np.log(t)  # noqa: E731
    return np.exp(optimize.least_squares(f, [np.log(0.1), np.log(2)], bounds=([-10, np.log(0.3)], [5, np.log(50)])).x)


def deming(x, y, lam=2.96):
    mx, my = x.mean(), y.mean()
    sxx, syy, sxy = ((x - mx) ** 2).sum(), ((y - my) ** 2).sum(), ((x - mx) * (y - my)).sum()
    d = syy - lam * sxx
    s = (d + np.sqrt(d * d + 4 * lam * sxy * sxy)) / (2 * sxy)
    return s, my - s * mx


def addis_part(lin, aeth):
    filters = [f for f in lin["filters"] if f["ftir"] is not None]
    tau = np.array([f["tau"] for f in filters]); bc = np.array([f["hips_bc"] for f in filters]); ec = np.array([f["ftir"] for f in filters])
    V = np.array([f["volume_m3"] for f in filters if f["volume_known"]])
    tv = np.array([f["tau"] for f in filters if f["volume_known"]])
    volume = {"p5": r(np.percentile(V, 5), 2), "p50": r(np.median(V), 2), "p95": r(np.percentile(V, 95), 2), "cv": r(V.std() / V.mean(), 3),
              "share_of_tau_variance": r(np.var(np.log(V)) / np.var(np.log(tv)), 4)}
    site = aeth["sites"]["ETAD-JACROS"]; grid = np.array(aeth["settings"]["k_grid"]); ch = site["channels"]["IR"]
    pos = {fid: i for i, fid in enumerate(site["filters"]["id"])}

    def aeth_bc(fid, k):
        v = ch["filters"]["bc_k"][pos[fid]]
        return float(np.interp(k, grid, np.array([np.nan if x is None else x for x in v])))

    fits, points = [], []
    for key, label, k in (("bc1", "aethalometer, no loading correction (BC1)", 0.0), ("tape", "aethalometer, tape-step k 0.0115", K_TAPE),
                          ("ftir", "FTIR EC", None)):
        X, T, ids = [], [], []
        for f in filters:
            if not f["volume_known"]:
                continue
            b = f["ftir"] if k is None else (aeth_bc(f["id"], k) if f["id"] in pos else np.nan)
            if np.isfinite(b) and b > 0:
                X.append(b * f["volume_m3"]); T.append(f["tau"]); ids.append(f["id"])
        X, T = np.array(X), np.array(T)
        a, c = fit_ceiling(X, T)
        rng = np.random.default_rng(1)
        boot = [fit_ceiling(X[j], T[j])[1] for j in (rng.integers(0, len(X), len(X)) for _ in range(300))]
        fits.append({"key": key, "label": label, "n": len(X), "a": r(a, 5), "tau_inf": r(c, 3), "lo": r(np.percentile(boot, 5), 3),
                     "hi": r(np.percentile(boot, 95), 3), "loglog": r(np.polyfit(np.log(X), np.log(T), 1)[0], 3)})
        if key == "tape":
            points = [{"id": i, "x": r(x, 3), "tau": r(t, 4)} for i, x, t in zip(ids, X, T)]
    Tg = np.round(np.linspace(1.6, 6, 89), 3)
    mac, icpt = [], []
    for Tc in Tg:
        k = 1 / (1 - tau / Tc); m = np.isfinite(k) & (k > 0)
        s, b = deming(bc[m] * k[m], ec[m]); mac.append(r(10 / s, 2)); icpt.append(r(b, 3))
    return {"volume": volume, "fits": fits, "points": points}, {"tau_inf": Tg.tolist(), "mac": mac, "intercept": icpt}, tau, (Tg, np.array(mac, float))


def delhi_part():
    windows = json.load(open(D / "weather/filters.json"))["windows"]
    ref = {row["id"]: row for row in json.load(open(D / "spartan_reference.json"))["rows"]}
    num = lambda v: v if isinstance(v, (int, float)) else np.nan  # noqa: E731
    rows = []
    for fid, w in windows.items():
        if not fid.startswith("INDH-"):
            continue
        row = ref.get(fid, {})
        if row.get("excluded"):
            continue
        fabs, V = num(row.get("HIPS Fabs")), num(row.get("Sampled volume"))
        if not (fabs > 0 and V > 0):
            continue
        rows.append(dict(id=fid, start=pd.Timestamp(w[0]), hours=w[2], V=V, fabs=fabs, tau=fabs * V / 353,
                         ec=num(row.get("EC (FTIR)")), pm=num(row.get("PM2.5 mass")), so4=num(row.get("Sulfate"))))
    d = pd.DataFrame(rows).sort_values("start")
    d["ym"] = d.start.dt.strftime("%Y-%m"); d["lv"] = np.log(d.V); d["year"] = d.start.dt.year
    d["season"] = d.start.dt.month.map(lambda m: "winter" if m in (11, 12, 1, 2) else "pre" if m in (3, 4, 5) else "monsoon" if m in (6, 7, 8, 9) else "post")
    specs = [("month", "month fixed effects", "lr ~ lc + lv + C(ym)", None),
             ("season_year", "season + year effects", "lr ~ lc + lv + C(season) + C(year)", None),
             ("y2025", "2025 only, month effects", "lr ~ lc + lv + C(ym)", lambda x: x[x.start.dt.year == 2025]),
             ("late", "Aug 2025 on, month effects", "lr ~ lc + lv + C(ym)", lambda x: x[x.start >= "2025-08-01"])]
    regs = []
    for num_, den, label, kind in (("fabs", "pm", "HIPS Fabs ÷ PM2.5 mass", "test"), ("fabs", "ec", "HIPS Fabs ÷ FTIR EC", "test"),
                                   ("fabs", "so4", "HIPS Fabs ÷ sulfate", "test"), ("ec", "pm", "FTIR EC ÷ PM2.5 mass (placebo)", "placebo"),
                                   ("so4", "pm", "sulfate ÷ PM2.5 mass (placebo)", "placebo")):
        for skey, slabel, form, sub in specs:
            x = d if sub is None else sub(d)
            x = x[(x[num_] > 0) & (x[den] > 0)].copy()
            x["lr"] = np.log(x[num_] / x[den]); x["lc"] = np.log(x[den])
            m = smf.ols(form, x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.ym)[0]})
            regs.append({"ratio": label, "kind": kind, "spec": skey, "spec_label": slabel, "n": int(len(x)),
                         "b_V": r(m.params["lv"], 3), "se_V": r(m.bse["lv"], 3), "b_C": r(m.params["lc"], 3), "se_C": r(m.bse["lc"], 3),
                         "expected": r(-x.tau.median() / TINF, 3) if kind == "test" else 0.0, "tau_median": r(x.tau.median(), 3)})
    filt = {"id": d.id.tolist(), "date": d.start.dt.strftime("%Y-%m-%d").tolist(), "hours": [r(h, 1) for h in d.hours],
            "volume": [r(v, 3) for v in d.V], "tau": [r(t, 4) for t in d.tau]}
    return {"filters": filt, "regressions": regs, "flow_l_min": 5.0}


def power_part(tau, mac_grid):
    Tg, MAC = mac_grid

    def fit_pairs(tl, td, ratio):
        def res(q):
            T = np.exp(q[0]); xl = np.where(tl < T, inv(tl, T), np.nan)
            out = np.log(hyper(ratio * xl, T)) - np.log(td); out[~np.isfinite(out)] = 5
            return out
        return np.exp(optimize.least_squares(res, [np.log(2.5)], bounds=([np.log(0.8)], [np.log(60)])).x[0])

    rng = np.random.default_rng(SEED)
    rows = []
    for T_true in (1.9, 2.26, 3.26):
        for ratio in (2, 4, 8):
            for n in (5, 10, 20, 40):
                est = []
                for _ in range(300):
                    td_t = np.minimum(rng.choice(tau, n), 0.95 * T_true)
                    tl_t = hyper(inv(td_t, T_true) / ratio, T_true)
                    est.append(fit_pairs(tl_t * np.exp(rng.normal(0, 0.04, n)), td_t * np.exp(rng.normal(0, 0.04, n)), ratio))
                q = np.percentile(est, [5, 50, 95]); m = np.interp(q, Tg, MAC)
                rows.append({"tau_inf_true": T_true, "ratio": ratio, "pairs": n, "lo": r(q[0], 3), "mid": r(q[1], 3), "hi": r(q[2], 3),
                             "mac_lo": r(m[2], 2), "mac_mid": r(m[1], 2), "mac_hi": r(m[0], 2)})
    return {"error_per_filter": 0.04, "rows": rows}


def main() -> None:
    lin = json.load(open(D / "hips_linearise/linearise.json"))
    aeth = json.load(open(D / "experimental/aeth_loading_k.json"))
    addis, mac, tau, grid = addis_part(lin, aeth)
    payload = {"schema_version": 1, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "source": "gallery/data/export_spartan_curve.py", "settings": {"tau_inf_published": TINF, "k_tape": K_TAPE, "lambda": 2.96},
               "addis": addis, "mac": mac, "delhi": delhi_part(), "power": power_part(tau, grid)}
    OUT.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
