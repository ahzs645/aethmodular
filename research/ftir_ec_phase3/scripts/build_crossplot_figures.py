"""The crossplots: the result graph, and the follow-up readouts of the same fit.

Ann's note in the 2026-08-27 run-through was that the crossplot is the figure the
group actually reads, and the stability and pathway sets did not contain one. This
builds the result itself plus the follow-ups that the evaluation-view levers make
possible: one fitted model, read out on each blind half, each season, each PMF
source class, and each site.

Convention, stated because the two differ and both are in circulation:
``--set fixed`` reproduces the fixed-190 numbers quoted throughout the
2026-08-27 working session (Deming 1.00x -1.71); ``--set all`` reproduces the
all-pairs numbers on the group deck's winner slide (0.93x -1.42). Same fit, same
model, different evaluation set. Every panel states which it used.

Run:  MPLBACKEND=Agg python3 build_crossplot_figures.py [--set fixed|all]
Requires the calibration explorer running on :5058.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output/plots/crossplots"      # HERE is .../ftir_ec_phase3/scripts
OUT.mkdir(parents=True, exist_ok=True)

INK, GREY, BLUE, PURPLE, ACCENT, AMBER = ("#22252A", "#8F8C84", "#2C6E9E",
                                          "#7A4FA3", "#B23327", "#C49442")
GREEN = "#548C66"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "savefig.dpi": 168})
BASE = "http://127.0.0.1:5058"
CFG = {"cohort": "ocec", "cutoff": 450, "selection_space": "raw",
       "spectra": "airspec", "mode": "site_heldout", "k": 9}
EVAL_SET = "fixed"          # overridden by --set

SEASON_COLOUR = {"Dry": AMBER, "Belg": GREEN, "Kiremt": BLUE}


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


def metric(out, want=None):
    """The MAC-10 row for the requested evaluation set, falling back if absent."""
    want = want or EVAL_SET
    ms = out.get("metrics", [])
    for es in (want, "all", "fixed"):
        rows = [m for m in ms if m["evaluation_set"] == es and m["MAC"] == 10]
        if rows:
            return rows[0], es
    return None, None


def points(out, es):
    """(x, y, group) for the evaluation set actually being reported."""
    e = out["eval"]
    idx = ([i for i, f in enumerate(e["fixed"]) if f] if es == "fixed"
           else list(range(len(e["ref"]))))
    if not idx:
        idx = list(range(len(e["ref"])))
    x = np.array([e["ref"][i] / 10.0 for i in idx])
    y = np.array([e["pred"][i] for i in idx])
    g = [str(e["group"][i]) for i in idx]
    return x, y, g


def draw(ax, x, y, m, es, label, colour=BLUE, groups=None, legend=False,
         loc="upper left"):
    hi = max(x.max(), y.max()) * 1.06
    lo = min(0.0, y.min() * 1.08)
    ax.plot([0, hi], [0, hi], ls=":", color=GREY, lw=1.2, zorder=1)
    if groups is not None:
        seen = {}
        for gx, gy, gg in zip(x, y, groups):
            key = gg.split(" (")[0]
            col = SEASON_COLOUR.get(key, PURPLE)
            ax.scatter([gx], [gy], s=26, color=col, alpha=.72, zorder=3,
                       edgecolor="none",
                       label=key if key not in seen else None)
            seen[key] = 1
    else:
        ax.scatter(x, y, s=26, color=colour, alpha=.65, zorder=3, edgecolor="none")
    sl, ic = m["deming_slope"], m["deming_intercept"]
    ax.plot([0, hi], [ic, sl * hi + ic], color=INK, lw=2, zorder=4)
    ax.set_xlim(0, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    # the diagonal and the fit both run corner to corner, so the only reliably
    # empty region on a crossplot is the lower right
    pos = dict(zip(("x", "y", "ha", "va"),
                   (.035, .965, "left", "top") if loc == "upper left"
                   else (.965, .045, "right", "bottom")))
    ax.text(pos["x"], pos["y"],
            f"{label}\nDeming {sl:.2f}x {ic:+.2f}\nn = {m['n']}   $R^2$ {m['R2']:.2f}",
            transform=ax.transAxes, ha=pos["ha"], va=pos["va"], fontsize=9.4,
            color=INK, linespacing=1.5)
    if legend and groups is not None:
        ax.legend(frameon=False, fontsize=9, loc="lower right", markerscale=1.4)


# ------------------------------------------------------------ 1. the result
def fig_result():
    o = post("/api/run", {**CFG, "target": "addis"})
    if "error" in o:
        print("  skipped 01:", o["error"]); return
    m, es = metric(o)
    x, y, g = points(o, es)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    draw(ax, x, y, m, es,
         f"lowest-OC/EC 450, baseline-corrected, k=9\n{es} evaluation set",
         groups=g, legend=True)
    ax.set_xlabel("HIPS EC-equivalent, Fabs/10 (µg m$^{-3}$)")
    ax.set_ylabel("predicted FTIR EC (µg m$^{-3}$)")
    finish(fig, f"01_result_crossplot_{es}.png")


# --------------------------------------------- 2. the same fit, both blind halves
def fig_halves():
    panels = [("all", "all filters"), ("early", "early half"), ("late", "late half")]
    outs = []
    for split, lab in panels:
        o = post("/api/run", {**CFG, "target": "addis", "eval_split": split})
        if "error" in o:
            continue
        m, es = metric(o)
        outs.append((lab, o, m, es))
    if not outs:
        print("  skipped 02: explorer unavailable"); return
    fig, axes = plt.subplots(1, len(outs), figsize=(4.5 * len(outs), 4.9), sharey=True)
    for ax, (lab, o, m, es) in zip(np.atleast_1d(axes), outs):
        x, y, g = points(o, es)
        draw(ax, x, y, m, es, lab, colour=BLUE, loc="lower right")
        ax.set_xlabel("HIPS EC-equivalent (µg m$^{-3}$)")
    np.atleast_1d(axes)[0].set_ylabel("predicted FTIR EC (µg m$^{-3}$)")
    finish(fig, f"02_result_by_blind_half_{EVAL_SET}.png")


# ------------------------------------------------- 3. the same fit, by season
def fig_seasons():
    groups = ["Belg (Mar-May)", "Kiremt (Jun-Sep)", "Dry (Oct-Feb)"]
    outs = []
    for g in groups:
        o = post("/api/run", {**CFG, "target": "addis",
                              "group_scheme": "season", "eval_group": g})
        if "error" in o:
            continue
        m, es = metric(o)
        outs.append((g.split(" (")[0], o, m, es))
    if not outs:
        print("  skipped 03: explorer unavailable"); return
    fig, axes = plt.subplots(1, len(outs), figsize=(4.5 * len(outs), 4.9), sharey=True)
    for ax, (lab, o, m, es) in zip(np.atleast_1d(axes), outs):
        x, y, _ = points(o, es)
        draw(ax, x, y, m, es, lab, colour=SEASON_COLOUR.get(lab, PURPLE), loc="lower right")
        ax.set_xlabel("HIPS EC-equivalent (µg m$^{-3}$)")
    np.atleast_1d(axes)[0].set_ylabel("predicted FTIR EC (µg m$^{-3}$)")
    finish(fig, f"03_result_by_season_{EVAL_SET}.png")


# ------------------------------------------- 4. the same fit, by PMF source class
def fig_pmf():
    outs = []
    for g, col in (("Marine", BLUE), ("Combustion", ACCENT)):
        o = post("/api/run", {**CFG, "target": "addis",
                              "group_scheme": "pmf_class", "eval_group": g})
        if "error" in o:
            continue
        m, es = metric(o)
        outs.append((f"PMF {g}", o, m, es, col))
    if not outs:
        print("  skipped 04: PMF scheme unavailable"); return
    fig, axes = plt.subplots(1, len(outs), figsize=(4.7 * len(outs), 5.0), sharey=True)
    for ax, (lab, o, m, es, col) in zip(np.atleast_1d(axes), outs):
        x, y, _ = points(o, es)
        draw(ax, x, y, m, es, lab, colour=col, loc="lower right")
        ax.set_xlabel("HIPS EC-equivalent (µg m$^{-3}$)")
    np.atleast_1d(axes)[0].set_ylabel("predicted FTIR EC (µg m$^{-3}$)")
    finish(fig, f"04_result_by_pmf_class_{EVAL_SET}.png")


# ------------------------------------------ 5. the same calibration, every site
def fig_sites():
    sites = [("addis", "Addis"), ("etbi", "Bishoftu"), ("chts", "Beijing"),
             ("indh", "Delhi"), ("uspa", "Pasadena")]
    outs = []
    for s, lab in sites:
        o = post("/api/run", {**CFG, "target": s})
        if "error" in o:
            continue
        m, es = metric(o)
        outs.append((lab, o, m, es))
    if not outs:
        print("  skipped 05: explorer unavailable"); return
    ncol = len(outs)
    fig, axes = plt.subplots(1, ncol, figsize=(3.5 * ncol, 4.3))
    for ax, (lab, o, m, es) in zip(np.atleast_1d(axes), outs):
        x, y, _ = points(o, es)
        col = ACCENT if lab in ("Addis", "Bishoftu") else BLUE
        draw(ax, x, y, m, es, lab, colour=col, loc="lower right")
        ax.set_xlabel("HIPS EC-equiv.", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("predicted FTIR EC (µg m$^{-3}$)")
    finish(fig, f"05_same_calibration_every_site_{EVAL_SET}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="eset", choices=("fixed", "all"), default="fixed",
                    help="fixed = the 190-filter subset used through the working "
                         "session; all = the 239 pairs on the group deck")
    a = ap.parse_args()
    EVAL_SET = a.eset
    print(f"writing to {OUT}   (evaluation set: {EVAL_SET})")
    for fn in (fig_result, fig_halves, fig_seasons, fig_pmf, fig_sites):
        try:
            fn()
        except Exception as exc:                               # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
