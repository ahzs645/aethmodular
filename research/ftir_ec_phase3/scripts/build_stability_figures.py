"""Figures for the 2026-08-27 stability and mechanism findings.

Every result from that session was recorded as a terminal table only. This
builds the eight that carry a claim: the OC/EC mismatch law, the variance
decomposition, the cross-site sign flip, ensemble reproducibility, the held-out
R2 partition artifact, cohort-neighbourhood instability, blind-half rank
movement, and the 1617 band against the offset.

House style: title-free (captions carry the claim), white, dpi 168.

Run:  MPLBACKEND=Agg python3 build_stability_figures.py
Requires the calibration explorer running on :5058.
"""
from __future__ import annotations
import collections, json, statistics as st, sys, urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "research/ftir_ec_phase3/output/plots/stability"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = REPO / "calibration_explorer/cache/batch_results.jsonl"
SCRATCH = Path("/private/tmp/claude-501/-Users-ahmadjalil-github-aethmodular/"
               "ad774656-c5c2-40e0-a058-29691900e8a0/scratchpad")

INK, GREY, BLUE, PURPLE, ACCENT, AMBER = ("#22252A", "#8F8C84", "#2C6E9E",
                                          "#7A4FA3", "#B23327", "#C49442")
GREEN = "#548C66"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "savefig.dpi": 168})
BASE = "http://127.0.0.1:5058"
MARKER = 1.34          # Addis FTIR-derived OC/EC (ftir_30 composition ruler)


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return json.load(f)
    except Exception as exc:                                  # noqa: BLE001
        return {"error": str(exc)}


def finish(fig, name):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def m10(out):
    ms = out.get("metrics", [])
    for es in ("fixed", "all"):
        rows = [m for m in ms if m["evaluation_set"] == es and m["MAC"] == 10]
        if rows:
            return rows[0], es
    return None, None


def mean_resid(out, es):
    e = out["eval"]
    idx = ([i for i, f in enumerate(e["fixed"]) if f] if es == "fixed"
           else list(range(len(e["ref"]))))
    if not idx:
        idx = list(range(len(e["ref"])))
    return st.mean(e["pred"][i] - e["ref"][i] / 10.0 for i in idx)


JOBS = ([("ocec", c) for c in range(250, 2001, 150)]
        + [("analogs", c) for c in (300, 500, 800, 1200)]
        + [("eth_shaped", c) for c in (200, 300, 500, 800)]
        + [("smoke", None), ("pool", None)])
FAMILY = {"ocec": (BLUE, "lowest OC/EC"), "analogs": (PURPLE, "spectral analogs"),
          "eth_shaped": (ACCENT, "Ethiopia-shaped"), "smoke": (AMBER, "biomass smoke"),
          "pool": (GREY, "whole network")}


def sweep(target="addis", spectra="airspec", k=9):
    rows = []
    for co, cut in JOBS:
        body = {"cohort": co, "selection_space": "raw", "spectra": spectra,
                "mode": "site_heldout"}
        if cut:
            body["cutoff"] = cut
        ci = post("/api/cohort_info", body)
        out = post("/api/run", {**body, "target": target, "k": k})
        if "error" in ci or "error" in out:
            continue
        m, es = m10(out)
        if not m:
            continue
        rows.append({"cohort": co, "cutoff": cut, "ocec": ci["ocec_ratio"]["median"],
                     "resid": mean_resid(out, es), "slope": m["deming_slope"],
                     "intercept": m["deming_intercept"]})
    return rows


def ols(xs, ys):
    n = len(xs); mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((v - mx) ** 2 for v in xs)
    b = sum((p - mx) * (q - my) for p, q in zip(xs, ys)) / sxx
    a = my - b * mx
    s2 = sum((q - (a + b * p)) ** 2 for p, q in zip(xs, ys)) / (n - 2)
    return a, b, (s2 * (1 / n + mx * mx / sxx)) ** .5, (s2 / sxx) ** .5


# ------------------------------------------------------- 1. the OC/EC mismatch law
def fig_ocec_law():
    rows = sweep()
    xs = [r["ocec"] - MARKER for r in rows]
    ys = [r["resid"] for r in rows]
    a, b, sea, seb = ols(xs, ys)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for fam, (col, lab) in FAMILY.items():
        sel = [r for r in rows if r["cohort"] == fam]
        if sel:
            ax.scatter([r["ocec"] - MARKER for r in sel], [r["resid"] for r in sel],
                       s=54, color=col, label=lab, zorder=3,
                       edgecolor="white", linewidth=.6)
    grid = np.linspace(min(xs) - .3, max(xs) + .3, 50)
    ax.plot(grid, a + b * grid, color=INK, lw=1.8, zorder=2)
    ax.axvline(0, color=GREY, ls=":", lw=1)
    ax.axhline(0, color=GREY, ls="-", lw=.8)
    ax.errorbar([0], [a], yerr=[2 * sea], fmt="D", color=ACCENT, ms=9, capsize=5,
                zorder=4, label="bias at a perfectly matched cohort")
    ax.annotate(f"{a:+.2f} $\\pm$ {sea:.2f} µg m$^{{-3}}$\nremains at zero mismatch",
                xy=(0, a), xytext=(1.4, a - 1.15), color=ACCENT, fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
    ax.text(.98, .04, f"slope {b:+.3f} $\\pm$ {seb:.3f} µg m$^{{-3}}$ per OC/EC unit",
            transform=ax.transAxes, ha="right", fontsize=10.5, color=INK)
    ax.set_xlabel("cohort OC/EC median $-$ Addis OC/EC (1.34)")
    ax.set_ylabel("mean residual, FTIR $-$ HIPS/10 (µg m$^{-3}$)")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    finish(fig, "01_ocec_mismatch_law.png")


# ------------------------------------------------------ 2. variance decomposition
def fig_variance():
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    a = [r for r in rows if r.get("target") == "addis"
         and str(r.get("eval_lot")) in ("all", "None")
         and r.get("eval_split") in (None, "all")
         and r.get("deming_intercept") is not None]
    for r in a:
        c = r.get("cutoff")
        r["cutband"] = ("none" if c is None else "<400" if c < 400 else
                        "400-799" if c < 800 else "800-1199" if c < 1200 else ">=1200")

    def eta2(factor, resp):
        g = collections.defaultdict(list)
        for r in a:
            v = r.get(resp)
            if v is not None:
                g[r.get(factor)].append(v)
        g = {k: v for k, v in g.items() if len(v) >= 5}
        allv = [x for v in g.values() for x in v]
        if len(g) < 2 or not allv:
            return 0.0
        gm = st.mean(allv)
        ssb = sum(len(v) * (st.mean(v) - gm) ** 2 for v in g.values())
        sst = sum((x - gm) ** 2 for x in allv)
        return ssb / sst if sst else 0.0

    knobs = [("spectra", "baseline"), ("cohort", "cohort type"),
             ("cutband", "cohort size"), ("k", "components"),
             ("selection_space", "selection space"), ("mode", "CV protocol"),
             ("lot", "training lot")]
    resp = [("deming_intercept", "intercept", BLUE), ("deming_slope", "slope", PURPLE)]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    y = np.arange(len(knobs))
    for i, (r, lab, col) in enumerate(resp):
        vals = [eta2(f, r) * 100 for f, _ in knobs]
        ax.barh(y + (i - .5) * .38, vals, height=.36, color=col, label=lab)
        for yy, v in zip(y + (i - .5) * .38, vals):
            if v > 1.5:
                ax.text(v + .8, yy, f"{v:.0f}%", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in knobs])
    ax.invert_yaxis()
    ax.set_xlabel("share of variance explained, one knob at a time (%)")
    ax.legend(frameon=False, fontsize=10)
    finish(fig, "02_variance_decomposition.png")


# ------------------------------------------------------- 3. cross-site sign flip
def fig_sign_flip():
    sites = [("addis", "Addis (ET)"), ("etbi", "Bishoftu (ET)"), ("chts", "Beijing"),
             ("indh", "Delhi"), ("uspa", "Pasadena")]
    specs = [("raw", "raw"), ("airspec", "AIRSpec"),
             ("neutral", "pspline-arPLS"), ("deriv2", "SG 2nd deriv")]
    M = np.full((len(sites), len(specs)), np.nan)
    for i, (site, _) in enumerate(sites):
        for j, (sp, _) in enumerate(specs):
            out = post("/api/run", {"cohort": "ocec", "cutoff": 450,
                                    "selection_space": "raw", "spectra": sp,
                                    "mode": "site_heldout", "target": site, "k": 9})
            if "error" in out:
                continue
            m, es = m10(out)
            if m:
                M[i, j] = mean_resid(out, es)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    # Delhi runs to +5.5 and would flatten every other cell, but the claim here is
    # the SIGN FLIP between the Ethiopian sites and the rest, so clip the scale and
    # let Delhi saturate rather than let it set the range.
    lim = 2.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=11,
                        color=INK if abs(M[i, j]) < lim * .55 else "white")
    ax.set_xticks(range(len(specs))); ax.set_xticklabels([l for _, l in specs])
    ax.set_yticks(range(len(sites))); ax.set_yticklabels([l for _, l in sites])
    ax.axhline(1.5, color=INK, lw=2)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, extend="both",
                      label="mean residual, FTIR $-$ HIPS/10 (µg m$^{-3}$)")
    ax.text(0, -0.92, "raw fits have slope ~2.5 and are not comparable",
            fontsize=9, color=GREY, ha="left")
    ax.text(3.5, -0.92, "above the rule: FTIR reads low   ·   below it: FTIR reads high",
            fontsize=9, color=GREY, ha="right")
    finish(fig, "03_cross_site_sign_flip.png")


# ------------------------------------------- 4. ensemble reproducibility
def fig_ensemble():
    """Averaging the stable family buys reproducibility, not score."""
    sys.path.insert(0, str(REPO / "research/ftir_hips_chem/scripts"))
    from pls_transfer import deming_regression
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    def vet(r):
        return (r.get("heldout_R2") is not None and r["heldout_R2"] >= .85
                and .7 <= r["deming_slope"] <= 1.3
                and r.get("q_residual_pct") is not None and r["q_residual_pct"] <= 30
                and r.get("extrap_pct") is not None and r["extrap_pct"] <= 30
                and r.get("negative_pct") is not None and r["negative_pct"] <= 10)
    a = [r for r in rows if r.get("target") == "addis"
         and str(r.get("eval_lot")) in ("all", "None")
         and r.get("eval_split") in (None, "all") and vet(r)]
    a.sort(key=lambda r: abs(r["deming_intercept"]) + 5 * abs(r["deming_slope"] - 1))
    top = a[:10]
    got = {}
    for view in ("early", "late"):
        preds, ref, fixed = [], None, None
        for r in top:
            out = post("/api/run", {"cohort": r["cohort"], "cutoff": r["cutoff"],
                                    "selection_space": r["selection_space"],
                                    "spectra": r["spectra"], "mode": r["mode"],
                                    "lot": r["lot"], "target": "addis", "k": r["k"],
                                    "eval_split": view})
            if "error" in out:
                continue
            preds.append(np.array(out["eval"]["pred"], float))
            ref = np.array(out["eval"]["ref"], float)
            fixed = np.array(out["eval"]["fixed"], bool)
        P_ = np.vstack(preds)
        m = fixed if fixed.sum() >= 3 else np.ones(len(ref), bool)
        x = ref[m] / 10.0
        singles = [deming_regression(x, p[m], 2.96)["intercept"] for p in P_]
        ens = deming_regression(x, P_.mean(axis=0)[m], 2.96)["intercept"]
        got[view] = (singles, ens)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for i, (se, sl) in enumerate(zip(got["early"][0], got["late"][0])):
        ax.plot([0, 1], [se, sl], color=GREY, lw=1.1, alpha=.75, zorder=2)
        ax.scatter([0, 1], [se, sl], s=26, color=GREY, zorder=3, edgecolor="none")
    ax.plot([0, 1], [got["early"][1], got["late"][1]], color=ACCENT, lw=2.6, zorder=5)
    ax.scatter([0, 1], [got["early"][1], got["late"][1]], s=95, color=ACCENT,
               zorder=6, edgecolor="white", linewidth=.8)
    spread = max(got["early"][0] + got["late"][0]) - min(got["early"][0] + got["late"][0])
    ax.text(.5, got["early"][1] + .16, "ensemble of the 10", ha="center",
            color=ACCENT, fontsize=10.5)
    ax.text(.5, min(got["early"][0]) - .12,
            f"individual configurations move up to {spread:.2f} µg m$^{{-3}}$",
            ha="center", color=GREY, fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["early half", "late half"])
    ax.set_xlim(-.28, 1.28)
    ax.set_ylabel("Deming intercept (µg m$^{-3}$)")
    finish(fig, "04_ensemble_reproducibility.png")


# ------------------------------------------------- 5. held-out R2 partition artifact
def fig_partition():
    cuts, r2, frac = [], [], []
    for c in range(1300, 1471, 10):
        out = post("/api/run", {"cohort": "ocec", "cutoff": c, "selection_space": "raw",
                                "spectra": "airspec", "mode": "site_heldout",
                                "target": "addis", "k": 9})
        if "error" in out or not out.get("heldout"):
            continue
        cuts.append(c); r2.append(out["heldout"]["R2"])
        # the training FRACTION, not the site count: a site can cross from the test
        # fold into training without changing the count, and that move is exactly
        # what steps the R2 (1350 -> 1360 adds 10 filters but 64 training rows)
        frac.append(out["n_train"] / out["n_cohort"])
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.step(cuts, r2, where="mid", color=BLUE, lw=2, zorder=3)
    ax.scatter(cuts, r2, s=34, color=BLUE, zorder=4)
    for i in range(1, len(cuts)):
        if abs(frac[i] - frac[i - 1]) > 0.012:
            ax.axvline(cuts[i] - 5, color=ACCENT, ls="--", lw=1.3, zorder=2)
    ax.plot([], [], color=ACCENT, ls="--", lw=1.3,
            label="train/test split is redrawn\n(a site crosses the partition)")
    ax.set_xlabel("cohort cutoff (filters)")
    ax.set_ylabel("held-out TOR $R^2$")
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    finish(fig, "05_r2_partition_artifact.png")


# ------------------------------------------ 6. cohort-neighbourhood instability
def fig_neighbourhood():
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    a = [r for r in rows if r.get("target") == "addis" and r.get("mode") == "site_heldout"
         and str(r.get("eval_lot")) in ("all", "None") and r.get("cutoff")
         and r.get("heldout_R2") is not None]
    ser = collections.defaultdict(dict)
    for r in a:
        ser[(r["cohort"], r["selection_space"], r["spectra"], r["k"])][r["cutoff"]] = r
    ser = {k: v for k, v in ser.items() if len(v) >= 25}
    score = lambda r: abs(r["deming_intercept"]) + 5 * abs(r["deming_slope"] - 1)
    bands = [(100, 300), (300, 500), (500, 700), (700, 900),
             (900, 1200), (1200, 1600), (1600, 2001)]
    meds, labs = [], []
    for lo, hi in bands:
        vals = []
        for s in ser.values():
            cs = sorted(s)
            for c in cs:
                if lo <= c < hi:
                    w = [s[x] for x in cs if abs(x - c) <= 50]
                    if len(w) >= 5:
                        vals.append(st.pstdev([score(x) for x in w]))
        if vals:
            meds.append(st.median(vals)); labs.append(f"{lo}–{hi-1}")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    cols = [ACCENT if m > 0.215 else (AMBER if m > 0.1075 else GREEN) for m in meds]
    ax.bar(range(len(meds)), meds, color=cols, width=.68)
    ax.axhline(0.215, color=INK, ls="--", lw=1.5)
    ax.text(len(meds) - .4, 0.228, "rank 1 to rank 10 gap", ha="right",
            fontsize=10, color=INK)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=20)
    ax.set_xlabel("cohort cutoff band (filters)")
    ax.set_ylabel("score movement from a $\\pm$50-filter\nchange in the cohort")
    finish(fig, "06_cohort_neighbourhood.png")


# ------------------------------------------------- 7. blind-half rank movement
def fig_rank_movement():
    p = SCRATCH / "vetted_addis.json"
    if not p.exists():
        print("  skipped 07 (no vetted_addis.json)"); return
    d = json.loads(p.read_text())
    rows = [r for r in d["rows"]
            if all(v in r["views"] and "error" not in r["views"][v]
                   for v in ("all", "early", "late"))
            and str(r.get("eval_lot")) in ("all", "None")]
    s = lambda r, v: r["views"][v]["score"]
    be = {id(r): i for i, r in enumerate(sorted(rows, key=lambda r: s(r, "early")), 1)}
    bl = {id(r): i for i, r in enumerate(sorted(rows, key=lambda r: s(r, "late")), 1)}
    x = [be[id(r)] for r in rows]; y = [bl[id(r)] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(x, y, s=16, color=GREY, alpha=.55, edgecolor="none")
    n = len(rows)
    ax.plot([0, n], [0, n], color=INK, ls="--", lw=1.2)
    top = [r for r in rows if be[id(r)] <= 10]
    ax.scatter([be[id(r)] for r in top], [bl[id(r)] for r in top], s=52,
               color=ACCENT, zorder=4, edgecolor="white", linewidth=.7,
               label="top 10 on the early half")
    ax.set_xlabel("rank on the early half"); ax.set_ylabel("rank on the late half")
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.text(.04, .95, f"n = {n} vetted configurations", transform=ax.transAxes,
            fontsize=10, color=INK, va="top")
    finish(fig, "07_blind_half_rank.png")


# ------------------------------------------------------ 8. 1617 band vs offset
def fig_band():
    j = post("/api/site_spectra", {"space": "neutral"})
    if "error" in j:
        print("  skipped 08 (site spectra unavailable)"); return
    want = {"Addis (ETAD)": "addis", "Bishoftu (ETBI)": "etbi",
            "Beijing (CHTS)": "chts", "Delhi (INDH)": "indh",
            "Pasadena (USPA)": "uspa"}
    pts = []
    for s in j["series"]:
        key = next((k for k in want if s["label"].startswith(k)), None)
        if not key:
            continue
        wn, y = s["wn"], s["median"]
        pk = max(y[i] for i, w in enumerate(wn) if 1600 <= w <= 1640)
        sh = st.mean([y[i] for i, w in enumerate(wn) if 1550 <= w <= 1575]
                     + [y[i] for i, w in enumerate(wn) if 1680 <= w <= 1700])
        ch = max(y[i] for i, w in enumerate(wn) if 2830 <= w <= 2980)
        out = post("/api/run", {"cohort": "ocec", "cutoff": 450, "selection_space": "raw",
                                "spectra": "airspec", "mode": "site_heldout",
                                "target": want[key], "k": 9})
        if "error" in out:
            continue
        m, es = m10(out)
        pts.append((100 * (pk - sh) / ch, mean_resid(out, es), key))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for bx, by, lab in pts:
        col = ACCENT if lab.startswith(("Addis", "Bishoftu")) else BLUE
        ax.scatter([bx], [by], s=95, color=col, zorder=3,
                   edgecolor="white", linewidth=.8)
        ax.annotate(lab.split(" (")[0], (bx, by), textcoords="offset points",
                    xytext=(8, 6), fontsize=10, color=INK)
    ax.axhline(0, color=GREY, lw=.9)
    ax.set_xlabel("1617 cm$^{-1}$ band height, normalised to C$-$H (%)")
    ax.set_ylabel("mean residual, FTIR $-$ HIPS/10 (µg m$^{-3}$)")
    ax.scatter([], [], color=ACCENT, label="Ethiopian sites", s=70)
    ax.scatter([], [], color=BLUE, label="other sites", s=70)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    finish(fig, "08_band_vs_offset.png")


if __name__ == "__main__":
    print(f"writing to {OUT}")
    for fn in (fig_ocec_law, fig_variance, fig_sign_flip, fig_ensemble,
               fig_partition, fig_neighbourhood, fig_rank_movement, fig_band):
        try:
            fn()
        except Exception as exc:                              # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
