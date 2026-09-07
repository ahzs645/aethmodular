"""Figures for the DECISIONS: what is feasible, what the objective selects, and
what the guardrails do.

The stability, pathway and crossplot sets show what was found. These show what
the search can and cannot deliver, so the "what is acceptable" discussion has a
picture to point at instead of a leaderboard:

  01  every configuration as a point in (slope, intercept) - the feasible region
  02  which point the objective picks as its weight w changes
  03  the guardrail funnel: which rule does the eliminating
  04  how far the held-out R2 moves per 10-filter cohort step, across the sweep
  05  season-optimised winners, read out on every season (the cross-application)
  06  each season's winner applied to every site

House style: title-free (captions carry the claim), white, dpi 168.

Run:  MPLBACKEND=Agg python3 build_decision_figures.py
01-04 read batch_results.jsonl; 05 reads the season_opt.json the 2026-08-27
session left in its scratchpad (pass --season-json); 06 needs the explorer.
"""
from __future__ import annotations
import argparse, collections, json, statistics as st, urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output/plots/decisions"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = HERE.parents[2] / "calibration_explorer/cache/batch_results.jsonl"

INK, GREY, BLUE, PURPLE, ACCENT, AMBER = ("#22252A", "#8F8C84", "#2C6E9E",
                                          "#7A4FA3", "#B23327", "#C49442")
GREEN = "#548C66"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "savefig.dpi": 168})
BASE = "http://127.0.0.1:5058"
W_DEFAULT = 5.0
BOX = dict(min_r2=.85, slope=(.7, 1.3), extrap=30, q=30, neg=10)


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return json.load(f)
    except Exception as exc:                                   # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def finish(fig, name):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def addis_rows():
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("target") == "addis"
            and r.get("deming_intercept") is not None
            and str(r.get("eval_lot")) in ("all", "None")
            and r.get("eval_split") in (None, "all")]


def vetted(r):
    return (r.get("q_residual_pct") is not None and r.get("extrap_pct") is not None
            and r.get("negative_pct") is not None and r.get("heldout_R2") is not None
            and r["heldout_R2"] >= BOX["min_r2"]
            and BOX["slope"][0] <= r["deming_slope"] <= BOX["slope"][1]
            and r["q_residual_pct"] <= BOX["q"] and r["extrap_pct"] <= BOX["extrap"]
            and r["negative_pct"] <= BOX["neg"])


def score(r, w=W_DEFAULT):
    return abs(r["deming_intercept"]) + w * abs(r["deming_slope"] - 1)


# ------------------------------------------ 01. the feasible (slope, intercept) region
def fig_feasibility(rows):
    """Where every configuration lands, and where (1, 0) is relative to them."""
    sl = np.array([r["deming_slope"] for r in rows])
    ic = np.array([r["deming_intercept"] for r in rows])
    ok = np.array([vetted(r) for r in rows])
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    keep = (sl > -0.5) & (sl < 3.2) & (ic > -14) & (ic < 6)
    ax.scatter(sl[keep & ~ok], ic[keep & ~ok], s=4, color=GREY, alpha=.28,
               edgecolor="none", rasterized=True,
               label=f"all configurations ({keep.sum():,} shown)")
    ax.scatter(sl[ok], ic[ok], s=11, color=ACCENT, alpha=.85, edgecolor="none",
               zorder=4, label=f"pass every guardrail ({ok.sum()})")
    # the guardrail box in this plane is just the slope band; the R2/diagnostic
    # rules act on other axes, which is why the red points are a subset of it
    ax.axvspan(*BOX["slope"], color=GREEN, alpha=.09, zorder=0)
    ax.text(1.0, 5.3, "slope guardrail\n0.7 - 1.3", ha="center", fontsize=9.2,
            color=GREEN)
    ax.axhspan(-.5, .5, color=BLUE, alpha=.08, zorder=0)
    ax.text(2.95, .0, "|intercept| < 0.5", ha="right", va="center", fontsize=9.2,
            color=BLUE)
    ax.scatter([1], [0], marker="x", s=170, color=INK, linewidth=2.2, zorder=6,
               label="the target: slope 1, intercept 0")
    ax.axhline(0, color=INK, lw=.8); ax.axvline(1, color=INK, lw=.8)
    # the collapsed-slope family: near-zero intercepts bought with slope ~0.3
    near = (np.abs(ic) <= .5) & (sl < .6)
    if near.any():
        ax.annotate(f"{near.sum()} near-zero intercepts,\nall with slope ~0.3",
                    xy=(np.median(sl[near]), 0), xytext=(0.05, -6.2),
                    fontsize=9.5, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.set_xlim(-.3, 3.2); ax.set_ylim(-13.5, 6)
    ax.set_xlabel("Deming slope at MAC 10")
    ax.set_ylabel("Deming intercept at MAC 10 (µg m$^{-3}$)")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right", markerscale=2.2)
    leg.legend_handles[-1].set_sizes([90])          # the target X, not scaled up
    finish(fig, "01_feasible_region.png")


# ------------------------------------------------ 02. which point the weight picks
def fig_weight(rows):
    vet = [r for r in rows if vetted(r)]
    sl = np.array([r["deming_slope"] for r in vet])
    ic = np.array([r["deming_intercept"] for r in vet])
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(sl, ic, s=12, color=GREY, alpha=.5, edgecolor="none",
               label=f"vetted configurations ({len(vet)})")
    ws = [0, 1, 2, 5, 10, 20]
    cols = [AMBER, AMBER, PURPLE, ACCENT, BLUE, BLUE]
    seen = {}
    for w, col in zip(ws, cols):
        best = min(vet, key=lambda r: score(r, w))
        key = (best["cohort"], best["cutoff"], best["spectra"], best["k"])
        lab = f"{best['cohort']} {best['cutoff']} {best['spectra']} k={best['k']}"
        seen.setdefault(key, []).append(w)
        ax.scatter([best["deming_slope"]], [best["deming_intercept"]], s=150,
                   color=col, zorder=5, edgecolor="white", linewidth=1.2)
    # two of the winners sit 0.02 apart in slope, so their labels must be pushed
    # to opposite sides with arrows rather than stacked at a fixed offset
    placement = {}
    for key, wlist in seen.items():
        r = next(r for r in vet if (r["cohort"], r["cutoff"], r["spectra"], r["k"]) == key)
        placement[key] = (r["deming_slope"], r["deming_intercept"], wlist)
    ordered = sorted(placement.items(), key=lambda kv: kv[1][0])
    # sorted by slope the neighbours are (airspec k=9 at 1.00, deriv2 k=20 at 1.02):
    # send the lower-slope one down-left and the higher one up-right, so each leader
    # points away from the other dot and each label is nearest its own point
    offsets = [(14, 8), (-165, -40), (30, 40)][:len(ordered)]
    for (key, (sx, sy, wlist)), off in zip(ordered, offsets):
        ax.annotate(f"w = {', '.join(map(str, wlist))}\n{key[0]} {key[1]} {key[2]} k={key[3]}",
                    (sx, sy), textcoords="offset points", xytext=off,
                    fontsize=9.3, color=INK,
                    arrowprops=(dict(arrowstyle="-", color=GREY, lw=.9, shrinkB=7)
                                if abs(off[0]) > 20 else None))
    ax.scatter([1], [0], marker="x", s=140, color=INK, linewidth=2, zorder=6)
    ax.set_xlabel("Deming slope"); ax.set_ylabel("Deming intercept (µg m$^{-3}$)")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right", markerscale=1.8)
    ax.text(.02, .04, "score = |intercept| + w · |slope − 1|;  three different\n"
            "winners across w, and the shipped w = 5 picks the one that\n"
            "did not transfer to other sites",
            transform=ax.transAxes, fontsize=9.4, color=INK)
    finish(fig, "02_winner_vs_weight.png")


# --------------------------------------------------- 03. the guardrail funnel
def fig_funnel(rows):
    steps = [("all Addis rows", rows)]
    f = [r for r in rows if r.get("heldout_R2") is not None and r["heldout_R2"] >= BOX["min_r2"]]
    steps.append(("held-out R² ≥ 0.85", f))
    f = [r for r in f if BOX["slope"][0] <= r["deming_slope"] <= BOX["slope"][1]]
    steps.append(("slope 0.7 – 1.3", f))
    f = [r for r in f if r.get("extrap_pct") is not None and r["extrap_pct"] <= BOX["extrap"]]
    steps.append(("score-space OOD ≤ 30%", f))
    f = [r for r in f if r.get("q_residual_pct") is not None and r["q_residual_pct"] <= BOX["q"]]
    steps.append(("Q residual ≤ 30%", f))
    f = [r for r in f if r.get("negative_pct") is not None and r["negative_pct"] <= BOX["neg"]]
    steps.append(("negative predictions ≤ 10%", f))
    counts = [len(s) for _, s in steps]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    y = np.arange(len(steps))
    cols = [GREY] + [ACCENT if i == 1 else BLUE for i in range(1, len(steps))]
    ax.barh(y, counts, color=cols, height=.66)
    for i, (c, prev) in enumerate(zip(counts, [None] + counts[:-1])):
        drop = f"   −{prev - c:,}" if prev else ""
        ax.text(c + 180, i, f"{c:,}{drop}", va="center", fontsize=10,
                color=ACCENT if i == 1 else INK)
    ax.set_yticks(y); ax.set_yticklabels([s for s, _ in steps], fontsize=10)
    ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel("configurations surviving (log scale)")
    share = 100 * (counts[0] - counts[1]) / (counts[0] - counts[-1])
    ax.text(.98, .06, f"the R² floor alone does {share:.0f}% of all eliminating",
            transform=ax.transAxes, ha="right", fontsize=10, color=ACCENT)
    finish(fig, "03_guardrail_funnel.png")


# -------------------------------------- 04. R2 movement per 10-filter cohort step
def fig_r2_steps(rows):
    ser = collections.defaultdict(dict)
    for r in rows:
        if r.get("cutoff") and r.get("heldout_R2") is not None and r["mode"] == "site_heldout":
            ser[(r["cohort"], r["selection_space"], r["spectra"], r["k"])][r["cutoff"]] = r["heldout_R2"]
    d, used = [], 0
    for s in ser.values():
        if len(s) < 30:
            continue                 # a short series has too few steps to speak
        used += 1
        cs = sorted(s)
        d += [abs(s[cs[i]] - s[cs[i - 1]]) for i in range(1, len(cs)) if cs[i] - cs[i - 1] == 10]
    d = np.array(d)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bins = np.concatenate([[0, .005], np.linspace(.01, .95, 48)])
    ax.hist(d, bins=bins, color=BLUE, alpha=.85, edgecolor="white", linewidth=.4)
    ax.set_yscale("log")
    ax.axvline(.05, color=ACCENT, lw=1.6, ls="--")
    big = 100 * (d > .05).mean()
    ax.text(.06, ax.get_ylim()[1] * .5, f"{big:.0f}% of steps move R² by more than 0.05",
            fontsize=10, color=ACCENT, va="center")
    ax.text(.98, .95, f"{len(d):,} adjacent 10-filter steps, {used} dense series\n"
            f"median |ΔR²| {np.median(d):.4f}   p90 {np.percentile(d, 90):.2f}   "
            f"max {d.max():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.6, color=INK)
    ax.set_xlabel("|ΔR²| between cohorts that differ by 10 filters")
    ax.set_ylabel("count (log)")
    finish(fig, "04_r2_per_10_filter_step.png")


# ------------------------------------- 05. season winners x season readouts
def fig_season_matrix(path):
    if not path or not Path(path).exists():
        print("  skipped 05: no season_opt.json"); return
    res = json.load(open(path))
    labs = ["all-year", "Belg", "Kiremt", "Dry"]
    winners = {}
    for lab in labs:
        have = [(v[lab]["s"], k) for k, v in res.items() if lab in v]
        if have:
            winners[lab] = min(have)[1]
    M = np.full((len(labs), len(labs)), np.nan)
    for i, wl in enumerate(labs):
        k = winners.get(wl)
        if not k:
            continue
        for j, rl in enumerate(labs):
            if rl in res[k]:
                M[i, j] = res[k][rl]["i"]
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    for i in range(len(labs)):
        for j in range(len(labs)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=11,
                        color="white" if abs(M[i, j]) > 1.7 else INK,
                        fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels([f"winner of {l}" for l in labs])
    ax.set_xlabel("read out on"); ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, label="Deming intercept (µg m$^{-3}$)")
    ax.set_title("")
    finish(fig, "05_season_cross_application.png")


# ---------------------------------------- 06. each season's winner at every site
def fig_season_sites():
    W = {"all-year": dict(cohort="ocec", cutoff=450, spectra="deriv2", k=20),
         "Belg":     dict(cohort="ocec", cutoff=840, spectra="deriv2", k=19),
         "Kiremt":   dict(cohort="ocec", cutoff=800, spectra="airspec", k=21),
         "Dry":      dict(cohort="ocec", cutoff=440, spectra="deriv2", k=17)}
    sites = [("addis", "Addis"), ("etbi", "Bishoftu"), ("chts", "Beijing"),
             ("indh", "Delhi"), ("uspa", "Pasadena")]
    S = np.full((len(W), len(sites)), np.nan); I = S.copy()
    for i, (wl, cfg) in enumerate(W.items()):
        for j, (s, _) in enumerate(sites):
            o = post("/api/run", {**cfg, "selection_space": "raw", "mode": "site_heldout",
                                  "lot": "all", "target": s})
            if "error" in o:
                continue
            ms = o.get("metrics", [])
            for es in ("fixed", "all"):
                rr = [m for m in ms if m["evaluation_set"] == es and m["MAC"] == 10]
                if rr:
                    S[i, j], I[i, j] = rr[0]["deming_slope"], rr[0]["deming_intercept"]
                    break
    if np.isnan(S).all():
        print("  skipped 06: explorer unavailable"); return
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    # colour by |slope - 1|: how far from proportional, the transferability measure
    D = np.abs(S - 1)
    im = ax.imshow(D, cmap="YlOrRd", vmin=0, vmax=3, aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if np.isfinite(S[i, j]):
                ax.text(j, i, f"{S[i, j]:.2f}x\n{I[i, j]:+.1f}", ha="center", va="center",
                        fontsize=9.6, color="white" if D[i, j] > 1.6 else INK)
    ax.set_xticks(range(len(sites))); ax.set_xticklabels([l for _, l in sites])
    ax.set_yticks(range(len(W)))
    ax.set_yticklabels([f"{k}\n{v['cohort']} {v['cutoff']} {v['spectra']} k={v['k']}"
                        for k, v in W.items()], fontsize=9)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, label="|slope − 1|  (0 = proportional)")
    finish(fig, "06_season_winner_by_site.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-json", default=None)
    a = ap.parse_args()
    print(f"writing to {OUT}")
    rows = addis_rows()
    for fn in (lambda: fig_feasibility(rows), lambda: fig_weight(rows),
               lambda: fig_funnel(rows), lambda: fig_r2_steps(rows),
               lambda: fig_season_matrix(a.season_json), fig_season_sites):
        try:
            fn()
        except Exception as exc:                               # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
