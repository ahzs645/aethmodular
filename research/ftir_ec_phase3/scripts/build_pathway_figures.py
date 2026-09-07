"""Figures for every investigative pathway taken on 2026-08-27, plus the
specification curve over the whole Addis search.

The stability set (build_stability_figures.py) covered the findings that
survived. This covers the PATHWAYS: one figure per hypothesis that was put to
data, including the ones that died, so the eliminations are visible rather than
asserted. Simonsohn et al. (2020) is the argument for the specification curve:
with 27,361 Addis configurations scored, the honest deliverable is the whole
distribution and its resolution, not a first place.

House style: title-free (captions carry the claim), white, dpi 168.

Run:  MPLBACKEND=Agg python3 build_pathway_figures.py
Figures 02-09 need the calibration explorer running on :5058; figure 01 does
not (it reads batch_results.jsonl directly).
"""
from __future__ import annotations
import collections, json, statistics as st, sys, urllib.request, datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "research/ftir_ec_phase3/output/plots/pathways"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = REPO / "calibration_explorer/cache/batch_results.jsonl"
sys.path.insert(0, str(HERE))

INK, GREY, BLUE, PURPLE, ACCENT, AMBER = ("#22252A", "#8F8C84", "#2C6E9E",
                                          "#7A4FA3", "#B23327", "#C49442")
GREEN = "#548C66"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "savefig.dpi": 168})
BASE = "http://127.0.0.1:5058"
CFG = {"cohort": "ocec", "cutoff": 450, "selection_space": "raw",
       "spectra": "airspec", "mode": "site_heldout", "k": 9}


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return json.load(f)
    except Exception as exc:                                   # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def finish(fig, name, tight=True):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    if tight:
        fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def load_rows():
    return [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]


def m10(out):
    """The MAC-10 metric row, preferring the fixed set when the target has one."""
    ms = out.get("metrics", [])
    for es in ("fixed", "all"):
        rows = [m for m in ms if m["evaluation_set"] == es and m["MAC"] == 10]
        if rows:
            return rows[0], es
    return None, None


def resid_of(out, es):
    e = out["eval"]
    idx = ([i for i, f in enumerate(e["fixed"]) if f] if es == "fixed"
           else list(range(len(e["ref"]))))
    if not idx:
        idx = list(range(len(e["ref"])))
    return np.array([e["pred"][i] - e["ref"][i] / 10.0 for i in idx])


def addis_run():
    return post("/api/run", {**CFG, "target": "addis"})


# ------------------------------------------------ 01. the specification curve
def fig_spec_curve():
    """Every Addis configuration ever scored, with the choices that produced it."""
    # all-lots, unsplit readouts only, so the population matches every other
    # number reported for this search; lot-specific readouts are a different view
    # of the same configurations and would double-count them here.
    rows = [r for r in load_rows() if r.get("target") == "addis"
            and r.get("deming_intercept") is not None
            and str(r.get("eval_lot")) in ("all", "None")
            and r.get("eval_split") in (None, "all")]
    def vetted(r):
        return (r.get("q_residual_pct") is not None
                and r.get("extrap_pct") is not None
                and r.get("negative_pct") is not None
                and r.get("heldout_R2") is not None and r["heldout_R2"] >= .85
                and .7 <= r["deming_slope"] <= 1.3
                and r["q_residual_pct"] <= 30 and r["extrap_pct"] <= 30
                and r["negative_pct"] <= 10)
    rows.sort(key=lambda r: r["deming_intercept"])
    y = np.array([r["deming_intercept"] for r in rows])
    n = len(y)
    ok = np.array([vetted(r) for r in rows])

    # the specification panel: one row per analytic choice, marked where used
    choices = [
        ("baseline: AIRSpec", lambda r: r["spectra"] == "airspec"),
        ("baseline: SG 2nd deriv", lambda r: r["spectra"] == "deriv2"),
        ("baseline: raw", lambda r: r["spectra"] == "raw"),
        ("cohort: lowest OC/EC", lambda r: r["cohort"] == "ocec"),
        ("cohort: spectral analogs", lambda r: r["cohort"] == "analogs"),
        ("cohort: Ethiopia-shaped", lambda r: r["cohort"] == "eth_shaped"),
        ("selection in AIRSpec", lambda r: r.get("selection_space") == "airspec"),
        ("protocol A (site-held-out)", lambda r: r["mode"] == "site_heldout"),
        ("protocol B / B2", lambda r: r["mode"] in ("app", "app_fmm")),
        ("k <= 9", lambda r: (r.get("k") or 0) <= 9),
        ("k >= 15", lambda r: (r.get("k") or 0) >= 15),
        ("cohort < 600 filters", lambda r: (r.get("cutoff") or 9999) < 600),
        ("training lot 251 only", lambda r: str(r.get("lot")) == "251"),
    ]
    M = np.zeros((len(choices), n), bool)
    for ci, (_, test) in enumerate(choices):
        for i, r in enumerate(rows):
            try:
                M[ci, i] = bool(test(r))
            except Exception:                                  # noqa: BLE001
                M[ci, i] = False

    fig, (ax, axs) = plt.subplots(
        2, 1, figsize=(11.4, 8.6), sharex=True,
        gridspec_kw={"height_ratios": [1.55, 2.0], "hspace": .07})

    x = np.arange(n)
    ax.fill_between(x, y, 0, color=BLUE, alpha=.16, linewidth=0)
    ax.plot(x, y, color=BLUE, lw=1.0)
    if ok.any():
        ax.scatter(x[ok], y[ok], s=7, color=ACCENT, zorder=4,
                   label=f"passes every guardrail ({ok.sum()} of {n:,})")
    ax.axhline(0, color=INK, lw=1.0)
    ax.axhspan(-0.5, 0.5, color=GREEN, alpha=.10, zorder=0)
    ax.set_ylabel("Deming intercept at MAC 10\n(µg m$^{-3}$)")
    lo = np.percentile(y, .3) - .6
    ax.set_ylim(lo, max(2.6, np.percentile(y, 99.7) + 1.9))
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.text(.006, .965, f"{n:,} Addis configurations, sorted by intercept",
            transform=ax.transAxes, fontsize=10.5, color=INK, va="top")
    ax.annotate("within 0.5 of zero", xy=(n * .985, 0), xytext=(n * .985, 1.8),
                ha="right", fontsize=9.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))

    axs.imshow(M, aspect="auto", interpolation="nearest",
               cmap=matplotlib.colors.ListedColormap(["#FFFFFF", INK]),
               extent=(0, n, len(choices) - .5, -.5))
    axs.set_yticks(range(len(choices)))
    axs.set_yticklabels([c[0] for c in choices], fontsize=9.5)
    axs.set_xlabel("specification, ordered by the intercept it produces")
    axs.set_xlim(0, n)
    for spine in ("top", "right", "left"):
        axs.spines[spine].set_visible(False)
    axs.tick_params(axis="y", length=0)
    finish(fig, "01_specification_curve.png", tight=False)


# ---------------------------------------------- 02. loading, within one site
def fig_loading():
    """Does the offset track filter darkness inside Addis?"""
    import pandas as pd
    from phase3_common import PATHS
    o = addis_run()
    if "error" in o:
        print("  skipped 02:", o["error"]); return
    e = o["eval"]
    ref = np.array(e["ref"], float); res = np.array(e["pred"], float) - ref / 10.0
    h = pd.read_csv(PATHS.spartan_hips_primary, encoding="cp1252",
                    usecols=["Site", "T1", "tau", "Fabs"]).query(
                    "Site=='ETAD' and T1>0 and Fabs==Fabs")
    lut = {round(float(f), 4): t for f, t in zip(h["Fabs"], h["tau"])}
    tau, rr = [], []
    for rv, rs in zip(ref, res):
        hit = lut.get(round(float(rv), 4))
        if hit is not None:
            tau.append(hit); rr.append(rs)
    tau = np.array(tau); rr = np.array(rr)
    r = float(np.corrcoef(tau, rr)[0, 1])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(tau, rr, s=26, color=GREY, alpha=.6, edgecolor="none")
    q = np.quantile(tau, [0, .25, .5, .75, 1.0])
    for i in range(4):
        m = (tau >= q[i]) & (tau <= q[i + 1])
        ax.plot([q[i], q[i + 1]], [rr[m].mean()] * 2, color=ACCENT, lw=3, zorder=4,
                solid_capstyle="butt")
    ax.plot([], [], color=ACCENT, lw=3, label="quartile mean")
    sl, ic = np.polyfit(tau, rr, 1)
    xs = np.linspace(tau.min(), tau.max(), 20)
    ax.plot(xs, sl * xs + ic, color=INK, lw=1.4, ls="--", label=f"fit, r = {r:+.2f}")
    ax.set_xlabel("filter optical depth $\\tau$  (darker $\\rightarrow$)")
    ax.set_ylabel("residual, FTIR $-$ HIPS/10 (µg m$^{-3}$)")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.text(.98, .95, "the lightest quartile already carries\nalmost the whole offset",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK)
    finish(fig, "02_loading_within_addis.png")


# ------------------------------------------ 03. additive vs multiplicative
def fig_additive():
    o = addis_run()
    if "error" in o:
        print("  skipped 03:", o["error"]); return
    e = o["eval"]
    ref = np.array(e["ref"], float) / 10.0
    pred = np.array(e["pred"], float)
    fx = np.array(e["fixed"], bool)
    x, res = ref[fx], (pred - ref)[fx]
    sl, ic = np.polyfit(x, res, 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(x, res, s=26, color=GREY, alpha=.6, edgecolor="none")
    xs = np.linspace(x.min(), x.max(), 20)
    ax.plot(xs, sl * xs + ic, color=ACCENT, lw=2,
            label=f"fit: slope {sl:+.3f}")
    ax.axhline(res.mean(), color=BLUE, lw=1.6, ls="--",
               label=f"constant offset {res.mean():+.2f}")
    ax.set_xlabel("HIPS EC-equivalent, Fabs/10 (µg m$^{-3}$)")
    ax.set_ylabel("residual, FTIR $-$ HIPS/10 (µg m$^{-3}$)")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.text(.02, .06, "a multiplicative error would tilt;\nthis is flat",
            transform=ax.transAxes, fontsize=10, color=INK)
    finish(fig, "03_additive_not_multiplicative.png")


# ------------------------------------------------------- 04. temporal drift
def fig_temporal():
    o = addis_run()
    if "error" in o:
        print("  skipped 04:", o["error"]); return
    e = o["eval"]
    ref = np.array(e["ref"], float); res = np.array(e["pred"], float) - ref / 10.0
    ds = [dt.date.fromisoformat(d) if d else None for d in e["date"]]
    keep = [i for i, d in enumerate(ds) if d]
    xs = [ds[i] for i in keep]; ys = res[keep]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.scatter(xs, ys, s=24, color=GREY, alpha=.6, edgecolor="none")
    byyear = collections.defaultdict(list)
    for d, v in zip(xs, ys):
        byyear[d.year].append(v)
    for yr, vals in sorted(byyear.items()):
        if len(vals) < 5:
            continue
        ax.plot([dt.date(yr, 1, 1), dt.date(yr, 12, 31)], [np.mean(vals)] * 2,
                color=ACCENT, lw=3, zorder=4, solid_capstyle="butt")
        ax.text(dt.date(yr, 7, 1), np.mean(vals) + .28, f"{np.mean(vals):.2f}",
                ha="center", fontsize=9.5, color=ACCENT)
    ax.plot([], [], color=ACCENT, lw=3, label="annual mean")
    ax.axhline(ys.mean(), color=BLUE, ls="--", lw=1.3, label="whole-record mean")
    ax.set_ylabel("residual, FTIR $-$ HIPS/10 (µg m$^{-3}$)")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.text(.02, .95, "no step, no drift: +0.016 µg m$^{-3}$ per year",
            transform=ax.transAxes, va="top", fontsize=10, color=INK)
    finish(fig, "04_no_temporal_drift.png")


# --------------------------------------------- 05. domain diagnostic by site
def fig_domain():
    sites = [("addis", "Addis"), ("etbi", "Bishoftu"), ("chts", "Beijing"),
             ("indh", "Delhi"), ("uspa", "Pasadena")]
    pts = []
    for s, lab in sites:
        o = post("/api/run", {**CFG, "target": s})
        if "error" in o:
            continue
        m, es = m10(o)
        if not m:
            continue
        pts.append((o["target"]["extrap_pct"], resid_of(o, es).mean(), lab))
    if not pts:
        print("  skipped 05: explorer unavailable"); return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for x, y, lab in pts:
        col = ACCENT if lab in ("Addis", "Bishoftu") else BLUE
        ax.scatter([x], [y], s=105, color=col, zorder=3,
                   edgecolor="white", linewidth=.8)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(9, 6),
                    fontsize=10.5, color=INK)
    ax.axhline(0, color=GREY, lw=.9)
    ax.axvline(30, color=INK, ls=":", lw=1.2)
    ax.text(30.8, ax.get_ylim()[1] * .92, "guardrail\n30%", fontsize=9, color=INK)
    ax.set_xlabel("score-space out-of-domain (% of filters beyond training p95)")
    ax.set_ylabel("mean residual (µg m$^{-3}$)")
    ax.scatter([], [], color=ACCENT, s=80, label="Ethiopian sites")
    ax.scatter([], [], color=BLUE, s=80, label="other sites")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.text(.98, .04, "Addis is the most in-domain site\nand still the most wrong",
            transform=ax.transAxes, ha="right", fontsize=10, color=INK)
    finish(fig, "05_domain_diagnostic.png")


# ----------------------------------------------- 06. implied MAC, by site
def fig_mac():
    sites = [("addis", "Addis"), ("etbi", "Bishoftu"), ("chts", "Beijing"),
             ("indh", "Delhi"), ("uspa", "Pasadena")]
    rows = []
    for s, lab in sites:
        o = post("/api/run", {**CFG, "target": s})
        if "error" in o:
            continue
        m, es = m10(o)
        if not m:
            continue
        e = o["eval"]
        idx = ([i for i, f in enumerate(e["fixed"]) if f] if es == "fixed"
               else list(range(len(e["ref"]))))
        if not idx:
            idx = list(range(len(e["ref"])))
        ref = np.array([e["ref"][i] for i in idx], float)
        pred = np.array([e["pred"][i] for i in idx], float)
        rows.append((lab, ref.mean() / pred.mean(), 10.0 / m["deming_slope"]))
    if not rows:
        print("  skipped 06: explorer unavailable"); return
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    y = np.arange(len(rows))
    ax.axvspan(7.5, 13.3, color=GREEN, alpha=.13, zorder=0)
    ax.text(10.4, -0.62, "published MAC range at 637 nm\n(9 European background sites)",
            ha="center", va="center", fontsize=9.2, color=GREEN)
    ax.axvline(10, color=INK, ls=":", lw=1.2)
    for i, (lab, a, b) in enumerate(rows):
        ax.plot([b, a], [i, i], color=GREY, lw=2, zorder=2)
        ax.scatter([b], [i], s=88, color=BLUE, zorder=3, label="MAC for slope 1" if i == 0 else "")
        ax.scatter([a], [i], s=88, color=ACCENT, zorder=3, label="MAC to zero bias" if i == 0 else "")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("implied mass absorption cross-section (m$^2$ g$^{-1}$)")
    ax.legend(frameon=False, fontsize=10, loc="lower right", ncol=2)
    ax.set_ylim(len(rows) - .35, -1.05)
    finish(fig, "06_implied_mac_by_site.png")


# ------------------------------------------ 07. what MAC does and does not move
def fig_mac_invariance():
    o = addis_run()
    if "error" in o:
        print("  skipped 07:", o["error"]); return
    e = o["eval"]
    ref = np.array(e["ref"], float); pred = np.array(e["pred"], float)
    fx = np.array(e["fixed"], bool)
    macs = [6.0, 10.0, 17.0]
    ints, slopes, means = [], [], []
    for mac in macs:
        row = [x for x in o["metrics"] if x["MAC"] == mac
               and x["evaluation_set"] == "fixed"][0]
        ints.append(row["deming_intercept"]); slopes.append(row["deming_slope"])
        means.append((pred[fx] - ref[fx] / mac).mean())
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(macs, ints, "o-", color=BLUE, lw=2.2, ms=9, label="Deming intercept")
    ax.plot(macs, means, "s--", color=ACCENT, lw=2.2, ms=8, label="mean residual")
    ax.axhline(0, color=GREY, lw=.9)
    for m, v in zip(macs, ints):
        ax.annotate(f"{v:.2f}", (m, v), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9.5, color=BLUE)
    for m, v in zip(macs, means):
        ax.annotate(f"{v:+.2f}", (m, v), textcoords="offset points", xytext=(0, -17),
                    ha="center", fontsize=9.5, color=ACCENT)
    ax.set_xticks(macs); ax.set_xlabel("MAC (m$^2$ g$^{-1}$)")
    ax.set_ylabel("µg m$^{-3}$")
    ax.legend(frameon=False, fontsize=10, loc="center right")
    ax.text(.03, .06, "the intercept does not move with MAC;\nthe mean residual does",
            transform=ax.transAxes, fontsize=10, color=INK)
    finish(fig, "07_mac_invariance.png")


# --------------------------------------- 08. is the residual spectrally visible
def fig_spectral():
    import warnings
    warnings.filterwarnings("ignore")
    from phase3_common import load_addis_evaluation
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    from config import season_for_month
    o = addis_run()
    if "error" in o:
        print("  skipped 08:", o["error"]); return
    etad, X, wn = load_addis_evaluation(season_for_month=season_for_month)
    e = o["eval"]
    ref = np.array(e["ref"], float)
    resid = np.array(e["pred"], float) - ref / 10.0
    if len(resid) != X.shape[0]:
        print("  skipped 08: alignment"); return
    seas = np.array([str(g) for g in e["group"]])
    dates = [dt.date.fromisoformat(d) if d else dt.date(2000, 1, 1) for d in e["date"]]
    order = np.argsort([d.toordinal() for d in dates])
    blk = np.zeros(len(dates), int); blk[order] = np.arange(len(dates)) // 30

    def cv(y, k, groups, nsplit):
        yp = np.zeros_like(y)
        for tr, te in GroupKFold(n_splits=nsplit).split(X, y, groups):
            mo = PLSRegression(n_components=k, scale=False).fit(X[tr], y[tr])
            yp[te] = mo.predict(X[te]).ravel()
        return 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    ks = [2, 4, 6, 9, 12]
    schemes = [("interleaved blocks", blk % 5, 5, BLUE),
               ("contiguous time blocks", blk, min(8, len(set(blk))), PURPLE),
               ("season-grouped", seas, len(set(seas)), ACCENT)]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for lab, g, n, col in schemes:
        ax.plot(ks, [cv(resid, k, g, n) for k in ks], "o-", color=col, lw=2, ms=7,
                label=f"residual, {lab}")
    ax.plot(ks, [cv(ref / 10.0, k, blk, min(8, len(set(blk)))) for k in ks],
            "s--", color=GREY, lw=1.8, ms=6,
            label="control: the reference itself")
    ax.axhline(0, color=INK, lw=.9)
    ax.set_xticks(ks); ax.set_xlabel("PLS components")
    ax.set_ylabel("cross-validated $R^2$")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.text(.03, .95, "the residual's variation is genuinely visible in the spectra",
            transform=ax.transAxes, va="top", fontsize=10, color=INK)
    finish(fig, "08_residual_is_spectrally_visible.png")


# --------------------------------- 09. the offset under every way of slicing it
def fig_slices():
    views = [("all filters", {}, INK)]
    for g in ("Belg (Mar-May)", "Kiremt (Jun-Sep)", "Dry (Oct-Feb)"):
        views.append((g.split(" (")[0] + " season",
                      {"group_scheme": "season", "eval_group": g}, BLUE))
    for g in ("Marine", "Combustion"):
        views.append((f"PMF {g}", {"group_scheme": "pmf_class", "eval_group": g}, PURPLE))
    for lot in ("251", "248"):
        views.append((f"lot {lot}", {"eval_lot": lot}, AMBER))
    for h in ("early", "late"):
        views.append((f"{h} half", {"eval_split": h}, GREEN))
    labs, vals, ns, cols = [], [], [], []
    for lab, extra, col in views:
        o = post("/api/run", {**CFG, "target": "addis", **extra})
        if "error" in o:
            continue
        m, es = m10(o)
        if not m:
            continue
        labs.append(lab); vals.append(m["deming_intercept"])
        # the n of the metric row actually plotted, NOT the view size: the fixed
        # subset inside a view is smaller than the view, and pairing one with the
        # other mislabels every bar
        ns.append(m["n"]); cols.append(col)
    if not labs:
        print("  skipped 09: explorer unavailable"); return
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    y = np.arange(len(labs))
    ax.barh(y, vals, color=cols, height=.66)
    for i, (v, n) in enumerate(zip(vals, ns)):
        # negative bars run leftward from zero, so "inside the bar" is v + a nudge
        ax.text(v + .05, i, f"{v:.2f}", va="center", ha="left",
                fontsize=9.5, color="white", fontweight="bold")
        ax.text(.04, i, f"n={n}", va="center", ha="left", fontsize=9, color=GREY)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Deming intercept at MAC 10, fixed evaluation set (µg m$^{-3}$)")
    ax.set_xlim(min(vals) * 1.12, max(0.34, max(vals) + .3))
    finish(fig, "09_offset_under_every_slice.png")


if __name__ == "__main__":
    print(f"writing to {OUT}")
    for fn in (fig_spec_curve, fig_loading, fig_additive, fig_temporal,
               fig_domain, fig_mac, fig_mac_invariance, fig_spectral, fig_slices):
        try:
            fn()
        except Exception as exc:                               # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
