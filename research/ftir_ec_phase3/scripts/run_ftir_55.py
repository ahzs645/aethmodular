# %% [markdown]
# # ftir_55 — every combination tried for Addis, on one plane, and how the search scores them
#
# ## tl;dr
#
# The calibration explorer's batch cache holds every configuration ever scored against
# Addis: **20,869 unique combinations** of cohort × cutoff × selection space × spectra
# treatment × CV protocol × training lot × PLS components (plus 6,500 re-scorings on
# blind halves, seasons, PMF classes and single lots that are not counted here). This
# notebook puts all of them on one figure, the **specification plane** (Deming slope on
# x, Deming intercept on y, both against HIPS Fabs/10 at MAC 10 on the fixed 190
# Addis filters), and draws on top of it what the search actually does: the objective
# `score = |intercept| + 5·|slope − 1|` as diamond-shaped contours, the guardrails as a
# box, the configurations that survive them, the ten best, and the winner. The deck's
# folded version (|slope − 1| vs |intercept|, all pairs, held-out floor as the only
# colour) is redrawn on today's cache. Then a grid of real FTIR-vs-HIPS crossplots, one
# per cohort × spectra combination at the rule k, shows what those points look like as
# data. Every number is read from `calibration_explorer/cache/batch_results.jsonl` (the
# same content-keyed cache the Colab prewarm bundle ships) or from a live explorer run
# that hits the explorer's fit cache.
#
# ## Context & Methods
#
# The explorer varies eight axes (cohort, cutoff, selection space, spectra treatment,
# protocol, training lot, evaluation lot, components) and scores each configuration on
# the same fixed Addis set. Because the reference is HIPS (not thermal EC), the search
# cannot chase "accuracy"; it chases **agreement with the HIPS axis on both slope and
# intercept** while refusing configurations that lose skill on the IMPROVE held-out
# thermal-EC test or that extrapolate. The objective and the five guardrails are the
# app's defaults (`/api/batch_start`): weight w = 5, held-out TOR R² ≥ 0.85, slope in
# [0.7, 1.3], extrapolation ≤ 30 %, Q-residual ≤ 30 %, negatives ≤ 10 %. A guardrail
# failure adds 1000 to the score, so the leaderboard is the passing set ranked by score.
#
# ### Key assumptions
#
# - Only the "all" evaluation view is counted (eval lot all, all filters, no season or
#   PMF grouping); the per-group re-scorings are views of the same fits.
# - Rows without the extrapolation / Q-residual / negatives diagnostics (older schema
#   generations) are shown but cannot pass the diagnostic guardrails; they are marked.
# - Crossplots in section 3 are explorer runs at the Option-A rule k; the cache points
#   include every k from 1 to 30, so the grid shows one representative per cell.

# %%
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from IPython.display import display, Markdown

PORT = int(os.environ.get("EXPLORER_PORT", 5058))
BASE = f"http://127.0.0.1:{PORT}"
PLOTS = Path("output/plots/ftir55")
TABLES = Path("output/tables/ftir55")
PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
CACHE = Path("../../calibration_explorer/cache/batch_results.jsonl")

INK, GREY, BLUE, PURPLE, ACCENT = "#1F1F1F", "#8F8C84", "#2C6E9E", "#7A4FA3", "#B23327"
AMBER, GREEN = "#C8862B", "#4E8A5B"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E8E6E1", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "white",
                     "savefig.dpi": 200, "savefig.facecolor": "white"})
W, MIN_R2, SLOPE_MIN, SLOPE_MAX, MAX_EXTRAP, MAX_Q, MAX_NEG = 5.0, 0.85, 0.7, 1.3, 30.0, 30.0, 10.0
SPECTRA_COLOUR = {"raw": AMBER, "deriv2": PURPLE, "airspec": BLUE}
SPECTRA_LABEL = {"raw": "raw spectra", "deriv2": "SG 2nd derivative", "airspec": "baseline-corrected (AIRSpec)"}
COHORT_LABEL = {"pool": "Entire IMPROVE network", "smoke": "Smoke-influenced 906",
                "eth_shaped": "Ethiopia-shaped smoke", "analogs": "Spectral analogs",
                "ocec": "Lowest-OC/EC"}
MANIFEST = []


def save(fig, name, settings):
    path = PLOTS / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path)
    if im.mode != "RGB":
        im.convert("RGB").save(path)
    MANIFEST.append((name, settings))
    print(f"saved {path}")


# %% [markdown]
# ## 1. The batch cache: what was tried

# %%
rows = [json.loads(line) for line in CACHE.open()]
df = pd.DataFrame(rows)
for c in ("eval_group", "eval_split"):
    df[c] = df[c].fillna("all") if c in df else "all"
KEY = ["cohort", "cutoff", "selection_space", "spectra", "mode", "lot", "k"]
addis = (df[(df.target == "addis") & (df.eval_group == "all") & (df.eval_split == "all")
            & (df.eval_lot == "all")]
         .drop_duplicates(KEY).copy())
addis["cutoff"] = addis["cutoff"].fillna(0).astype(int)
addis["score"] = addis.deming_intercept.abs() + W * (addis.deming_slope - 1).abs()
has_diag = addis[["extrap_pct", "q_residual_pct", "negative_pct"]].notna().all(axis=1)
addis["has_diagnostics"] = has_diag
addis["pass_core"] = ((addis.heldout_R2 >= MIN_R2) & (addis.deming_slope >= SLOPE_MIN)
                      & (addis.deming_slope <= SLOPE_MAX))
addis["pass_all"] = (addis.pass_core & has_diag
                     & (addis.extrap_pct.fillna(999) <= MAX_EXTRAP)
                     & (addis.q_residual_pct.fillna(999) <= MAX_Q)
                     & (addis.negative_pct.fillna(999) <= MAX_NEG))
print(f"batch cache: {len(df):,} rows across {df.target.nunique()} targets; "
      f"Addis all-view unique configurations: {len(addis):,}")
axes_tried = {
    "cohort": addis.cohort.value_counts().to_dict(),
    "spectra": addis.spectra.value_counts().to_dict(),
    "selection_space": addis.selection_space.value_counts().to_dict(),
    "mode": addis["mode"].value_counts().to_dict(),
    "lot": addis.lot.value_counts().to_dict(),
    "k": f"{int(addis.k.min())} to {int(addis.k.max())}",
    "cutoffs per cohort": addis.groupby("cohort").cutoff.nunique().to_dict(),
}
for k_, v in axes_tried.items():
    print(f"  {k_}: {v}")
print(f"passing the core guardrails (held-out R2 >= {MIN_R2}, slope {SLOPE_MIN} to {SLOPE_MAX}): "
      f"{int(addis.pass_core.sum()):,}; with all five diagnostics present and passing: {int(addis.pass_all.sum()):,}; "
      f"rows lacking diagnostics: {int((~has_diag).sum()):,}")
addis.to_csv(TABLES / "addis_all_configurations.csv", index=False)

# %% [markdown]
# ## 2. The specification plane: every combination, the objective, the guardrails

# %%
def plane(ax, data, xlim=(0.0, 4.0), ylim=(-12.5, 3.0)):
    for sp, colour in SPECTRA_COLOUR.items():
        s = data[data.spectra == sp]
        ax.scatter(s.deming_slope, s.deming_intercept, s=6, color=colour, alpha=0.18,
                   edgecolor="none", label=f"{SPECTRA_LABEL[sp]} (n = {len(s):,})", rasterized=True)
    for c in (1, 2, 3, 5):   # objective contours |b| + W|a-1| = c are diamonds centred on (1, 0)
        a = np.array([1 - c / W, 1, 1 + c / W, 1, 1 - c / W])
        b = np.array([0, c, 0, -c, 0])
        ax.plot(a, b, color=INK, lw=0.7, ls="--", alpha=0.6)
        ax.text(1 + c / W + 0.02, 0.05, f"score {c}", fontsize=7, color=INK, alpha=0.7)
    ax.axvspan(SLOPE_MIN, SLOPE_MAX, color="#F2F5F0", zorder=0)
    ax.axhline(0, color=INK, lw=0.9)
    ax.axvline(1, color=INK, lw=0.9)
    ax.set(xlim=xlim, ylim=ylim, xlabel="Deming slope vs HIPS Fabs/10 (MAC 10, fixed 190 Addis filters)",
           ylabel="Deming intercept (µg/m³)")


fig, ax = plt.subplots(figsize=(9.5, 7.0))
plane(ax, addis)
passing = addis[addis.pass_all]
ax.scatter(passing.deming_slope, passing.deming_intercept, s=16, facecolor="none",
           edgecolor=INK, lw=0.6, label=f"passes all five guardrails (n = {len(passing):,})", zorder=4)
top = passing.nsmallest(10, "score")
ax.scatter(top.deming_slope, top.deming_intercept, s=70, facecolor="none", edgecolor=ACCENT,
           lw=1.4, label="ten best by score", zorder=5)
win = top.iloc[0]
ax.annotate(f"best score: {COHORT_LABEL[win.cohort]} {win.cutoff}, {SPECTRA_LABEL[win.spectra]}, k = {int(win.k)}\n"
            f"{win.deming_slope:.2f}x {win.deming_intercept:+.2f}, score {win.score:.2f}, held-out R² {win.heldout_R2:.2f}",
            xy=(win.deming_slope, win.deming_intercept), xytext=(1.9, 1.6), fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.9),
            bbox=dict(boxstyle="round", fc="white", ec="#DDD"))
ax.text(SLOPE_MIN + 0.02, -12.2, f"slope guardrail {SLOPE_MIN} to {SLOPE_MAX}", fontsize=7.5, color=GREY)
leg = ax.legend(loc="lower right", frameon=True, fontsize=8, framealpha=0.95, markerscale=2.2)
for h in leg.legend_handles[:3]:
    h.set_alpha(0.9)
fig.tight_layout()
save(fig, "01_specification_plane_all_configurations.png",
     f"Every unique Addis configuration in the explorer batch cache (n = {len(addis):,}; all-filters "
     f"evaluation view, eval lot all), Deming vs HIPS Fabs/10 at MAC 10 on the fixed 190. Colour = spectra "
     f"treatment; dashed diamonds = objective |intercept| + {W:g}·|slope − 1|; shaded band = slope guardrail; "
     f"outlined = passes held-out TOR R² ≥ {MIN_R2}, slope, extrapolation ≤ {MAX_EXTRAP:g} %, Q ≤ {MAX_Q:g} %, "
     f"negatives ≤ {MAX_NEG:g} %.")

# %% [markdown]
# Reading the plane: the raw-spectra cloud (amber) sits far below the axis with slopes
# between 1.5 and 3; the derivative cloud (purple) is closer; only baseline-corrected
# configurations (blue) reach the diamond around the ideal point, and the guardrails
# then remove the ones that got there by losing thermal-EC skill. The objective is a
# distance from (slope 1, intercept 0) in which one unit of slope costs as much as five
# micrograms of intercept.
#
# One caution on the label: the best raw score belongs to the lowest-OC/EC 450 × SG
# 2nd-derivative × k = 20 row (1.68), a manual-k row that the Bishoftu transfer test
# showed does not generalise (0.51x, R² 0.13; asterisked in the ETBI first look). The
# vetted winner is the baseline-corrected lowest-OC/EC 440 × k = 9 row two hundredths
# behind it (0.98x −1.62, score 1.72), which transfers to Bishoftu at 0.93x. Score alone
# does not settle the choice; the out-of-site test does.
#
# ## 2a. The deck's version: folded axes, all pairs, the held-out floor as the only colour
#
# The 27 Aug group deck drew this plane folded onto |slope − 1| and |intercept| on the
# all-pairs evaluation set (239 filters), coloured only by the held-out thermal-EC floor,
# with 12,000 configurations. Same construction here on today's cache.

# %%
hon = addis[(addis["mode"] == "site_heldout") & addis.heldout_R2.notna()].copy()
sl = hon["all_deming_slope"].fillna(hon["deming_slope"]).astype(float)
ic = hon["all_deming_intercept"].fillna(hon["deming_intercept"]).astype(float)
floor = hon.heldout_R2 >= MIN_R2
XMAX, YMAX = 3.2, 14.0
ds, di = (sl - 1).abs(), ic.abs()
shown = (ds <= XMAX) & (di <= YMAX)
fig, ax = plt.subplots(figsize=(7.6, 5.4))
ax.scatter(ds[~floor & shown], di[~floor & shown], s=7, color="#D8D5CF",
           label=f"below the held-out floor (n = {int((~floor).sum()):,})", rasterized=True)
ax.scatter(ds[floor & shown], di[floor & shown], s=9, color=BLUE, alpha=0.45,
           label=f"held-out TOR R² ≥ {MIN_R2} (n = {int(floor.sum()):,})", rasterized=True)
for c in (1, 2, 3):
    ax.plot([0, c / W], [c, 0], color=INK, lw=0.7, ls="--", alpha=0.6)
    ax.text(c / W + 0.02, 0.15, f"score {c}", fontsize=7, color=INK, alpha=0.7)
vet = hon[(hon.cohort == "ocec") & (hon.cutoff == 440) & (hon.spectra == "airspec") & (hon.k == 9)]
deck = hon[(hon.cohort == "ocec") & (hon.cutoff == 450) & (hon.spectra == "airspec") & (hon.k == 9)]
for row, colour, lab in ((vet, ACCENT, "vetted winner: lowest-OC/EC 440 × AIRSpec, k = 9"),
                         (deck, AMBER, "the 27 Aug deck's star: lowest-OC/EC 450 × AIRSpec, k = 9")):
    if len(row):
        r = row.iloc[0]
        ax.scatter([abs(r.all_deming_slope - 1)], [abs(r.all_deming_intercept)], s=150, marker="*",
                   color=colour, zorder=5, label=lab)
ax.set(xlim=(0, XMAX), ylim=(0, YMAX),
       xlabel="|Deming slope − 1|  (Addis, MAC 10, all pairs, site-held-out protocol)",
       ylabel="|Deming intercept| (µg/m³)")
ax.text(0.99, 0.02, f"{int((~shown).sum())} extreme variants beyond the axes (all far from 1:1)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#6E7178")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
print(f"site-held-out configurations with a held-out score: {len(hon):,}; above the floor: {int(floor.sum()):,}")
save(fig, "01b_folded_plane_all_pairs.png",
     f"Deck-style screen: site-held-out configurations only (n = {len(hon):,}), all-pairs Addis evaluation "
     f"(239 filters), |Deming slope − 1| vs |Deming intercept| at MAC 10; grey = held-out TOR R² < {MIN_R2}; "
     "dashed = objective contours; stars = the vetted winner (440, k = 9) and the 27 Aug deck's star (450, k = 9).")

# %% [markdown]
# ## 2b. The same plane, one panel per cohort family

# %%
order = ["pool", "smoke", "eth_shaped", "analogs", "ocec"]
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
for ax, coh in zip(axes.flat, order):
    d = addis[addis.cohort == coh]
    plane(ax, d)
    p = d[d.pass_all]
    ax.scatter(p.deming_slope, p.deming_intercept, s=16, facecolor="none", edgecolor=INK, lw=0.6, zorder=4)
    best = d[d.pass_all].nsmallest(1, "score")
    txt = f"{COHORT_LABEL[coh]}\n{len(d):,} configurations, {len(p):,} pass"
    if len(best):
        b = best.iloc[0]
        ax.scatter(b.deming_slope, b.deming_intercept, s=80, facecolor="none", edgecolor=ACCENT, lw=1.4, zorder=5)
        txt += f"\nbest: cutoff {b.cutoff}, {SPECTRA_LABEL[b.spectra]}, k = {int(b.k)}: {b.deming_slope:.2f}x {b.deming_intercept:+.2f}"
    else:
        txt += "\nno configuration passes the guardrails"
    ax.set_title(txt, fontsize=8.5, loc="left")
    ax.set_xlabel(""); ax.set_ylabel("")
    if ax.get_legend():
        ax.get_legend().remove()
axes.flat[-1].axis("off")
axes.flat[-1].legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=SPECTRA_LABEL[s])
                              for s, c in SPECTRA_COLOUR.items()]
                     + [Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="passes all guardrails"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec=ACCENT, ms=9, label="best in family")],
                     loc="center", frameon=False, fontsize=9)
fig.supxlabel("Deming slope vs HIPS Fabs/10 (MAC 10, fixed 190)")
fig.supylabel("Deming intercept (µg/m³)")
fig.tight_layout()
save(fig, "02_specification_plane_by_cohort.png",
     "Same data as 01, one panel per cohort family; outlined = passes all five guardrails; red ring = "
     "best passing score in the family.")

# %% [markdown]
# ## 2c. What each cohort × spectra cell achieved

# %%
def _cell(g):
    ok = g.pass_all
    best = g[ok].score.idxmin() if ok.any() else None
    return pd.Series({
        "configurations": len(g), "passing": int(ok.sum()),
        "best_score": g.loc[best, "score"] if best is not None else np.nan,
        "best_slope": g.loc[best, "deming_slope"] if best is not None else np.nan,
        "best_intercept": g.loc[best, "deming_intercept"] if best is not None else np.nan,
        "best_cutoff": g.loc[best, "cutoff"] if best is not None else np.nan,
        "best_k": g.loc[best, "k"] if best is not None else np.nan,
        "median_intercept_all": g.deming_intercept.median(),
    })


cell = addis.groupby(["cohort", "spectra"]).apply(_cell, include_groups=False).reset_index()
cell["cohort"] = pd.Categorical(cell.cohort, order)
cell = cell.sort_values(["cohort", "spectra"])
cell.to_csv(TABLES / "cohort_by_spectra_summary.csv", index=False)
display(cell.round(2))

fig, ax = plt.subplots(figsize=(8.5, 4.2))
piv = cell.pivot(index="cohort", columns="spectra", values="best_score").loc[order, ["raw", "deriv2", "airspec"]]
cnt = cell.pivot(index="cohort", columns="spectra", values="configurations").loc[order, ["raw", "deriv2", "airspec"]]
im = ax.imshow(piv.to_numpy(float), cmap="Blues_r", vmin=0, vmax=8, aspect="auto")
for i, coh in enumerate(piv.index):
    for j, sp in enumerate(piv.columns):
        v = piv.loc[coh, sp]
        n = int(cnt.loc[coh, sp]) if pd.notna(cnt.loc[coh, sp]) else 0
        ax.text(j, i, ("no pass" if pd.isna(v) else f"best score {v:.2f}") + f"\n{n:,} tried",
                ha="center", va="center", fontsize=8.5, color=INK if (pd.isna(v) or v > 3) else "white")
ax.set(xticks=range(3), xticklabels=[SPECTRA_LABEL[s] for s in piv.columns],
       yticks=range(len(order)), yticklabels=[COHORT_LABEL[c] for c in order])
ax.grid(False)
fig.colorbar(im, ax=ax, label="best passing score (lower is better)")
fig.tight_layout()
save(fig, "03_best_score_by_cohort_and_spectra.png",
     "Best guardrail-passing score per cohort family × spectra treatment, with the number of "
     "configurations tried in each cell (all cutoffs, protocols, lots and k).")

# %% [markdown]
# ## 2d. How the search moves: score against cutoff for the winning family

# %%
oc = addis[(addis.cohort == "ocec") & (addis["mode"] == "site_heldout") & (addis.lot == "all")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for sp, colour in SPECTRA_COLOUR.items():
    d = oc[oc.spectra == sp]
    best_by_cut = d.groupby("cutoff").score.min()
    axes[0].plot(best_by_cut.index, best_by_cut.values, color=colour, lw=1.4, label=SPECTRA_LABEL[sp])
    pass_by_cut = d[d.pass_all].groupby("cutoff").score.min()
    axes[0].scatter(pass_by_cut.index, pass_by_cut.values, s=12, color=colour)
axes[0].set(xlabel="lowest-OC/EC cohort cutoff (filters)", ylabel="best score over k (line) and best passing (dots)",
            ylim=(0, 8))
axes[0].legend(frameon=False, fontsize=8.5)
d = oc[(oc.spectra == "airspec")]
for cut, colour in ((440, ACCENT), (800, BLUE), (1500, GREY)):
    s = d[d.cutoff == cut].sort_values("k")
    if len(s):
        axes[1].plot(s.k, s.score, color=colour, lw=1.4, marker="o", ms=3, label=f"cutoff {cut}")
        axes[1].plot(s.k, s.heldout_R2 * 8, color=colour, lw=0.9, ls=":")
axes[1].axhline(MIN_R2 * 8, color=GREY, lw=0.8, ls="--")
axes[1].text(1, MIN_R2 * 8 + 0.1, f"held-out R² guardrail ({MIN_R2}) on the right axis", fontsize=7.5, color=GREY)
axes[1].set(xlabel="PLS components k", ylabel="score (solid)", ylim=(0, 8))
ax2 = axes[1].twinx()
ax2.set(ylim=(0, 1), ylabel="held-out TOR R² (dotted)")
ax2.grid(False)
axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout()
save(fig, "04_search_path_ocec.png",
     "Lowest-OC/EC family, Option A, training lot all: left, best score over k at each cutoff per spectra "
     "treatment; right, score (solid) and held-out TOR R² (dotted) against k for three AIRSpec cutoffs.")

# %% [markdown]
# ## 3. What the points look like as data: one crossplot per cohort × spectra

# %%
def api_get(path):
    return json.load(urllib.request.urlopen(f"{BASE}{path}", timeout=60))


def api_post(path, body, timeout=1800):
    req = urllib.request.Request(f"{BASE}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=timeout))
    if isinstance(out, dict) and out.get("error"):
        raise RuntimeError(f"{path}: {out['error']}")
    return out


try:
    status = api_get("/api/status")
    EXPLORER = bool(status["ready"])
    print("explorer ready · commit", status["provenance"]["git_commit"])
except Exception as exc:  # noqa: BLE001
    EXPLORER = False
    print("explorer not reachable:", exc)

DEFAULT_CUTOFF = {"pool": None, "smoke": None, "eth_shaped": 300, "analogs": 500, "ocec": 800}
BASE_BODY = dict(mode="site_heldout", target="addis", lot="all", eval_lot="all")


def pick(d, eval_set="fixed", mac=10):
    rows_ = [m for m in d["metrics"] if m["MAC"] == mac and m["evaluation_set"] == eval_set]
    return (rows_ or [m for m in d["metrics"] if m["MAC"] == mac])[0]


grid_rows = []
if EXPLORER:
    fig, axes = plt.subplots(len(order), 3, figsize=(11.5, 3.6 * len(order)))
    for i, coh in enumerate(order):
        for j, sp in enumerate(["raw", "deriv2", "airspec"]):
            ax = axes[i, j]
            body = {**BASE_BODY, "cohort": coh, "spectra": sp}
            if DEFAULT_CUTOFF[coh]:
                body["cutoff"] = DEFAULT_CUTOFF[coh]
            if coh in ("analogs", "eth_shaped"):
                body["selection_space"] = "raw"
            try:
                t0 = time.time()
                d = api_post("/api/run", body)
                m = pick(d)
                ref = np.array(d["eval"]["ref"], float) / 10.0
                pred = np.array(d["eval"]["pred"], float)
                fixed = np.array(d["eval"]["fixed"], bool)
                x, y = ref[fixed], pred[fixed]
                lim = max(np.nanmax(x), np.nanmax(y), 1.0) * 1.06
                ax.plot([0, lim], [0, lim], ls=":", color=GREY, lw=1.0)
                ax.axhline(0, color="#555", lw=0.7)
                ax.scatter(x, y, s=9, color=SPECTRA_COLOUR[sp], alpha=0.6, edgecolor="none")
                xx = np.linspace(0, lim, 2)
                ax.plot(xx, m["deming_slope"] * xx + m["deming_intercept"], color=INK, lw=1.3)
                ho = d.get("heldout", {}).get("R2")
                score = abs(m["deming_intercept"]) + W * abs(m["deming_slope"] - 1)
                ax.text(0.03, 0.97, f"Deming {m['deming_slope']:.2f}x {m['deming_intercept']:+.2f}\n"
                                    f"k = {d['k']} · score {score:.2f}"
                                    + (f"\nheld-out TOR R² {ho:.2f}" if ho is not None else ""),
                        transform=ax.transAxes, va="top", fontsize=7.5,
                        bbox=dict(boxstyle="round", fc="white", ec="#DDD"))
                ax.set(xlim=(0, lim), ylim=(min(0, np.nanmin(y)) - 0.5, lim))
                grid_rows.append({"cohort": coh, "cutoff": DEFAULT_CUTOFF[coh], "spectra": sp, "k": d["k"],
                                  "n_cohort": d["n_cohort"], "deming_slope": m["deming_slope"],
                                  "deming_intercept": m["deming_intercept"], "R2": m["R2"],
                                  "heldout_R2": ho, "score": score, "seconds": round(time.time() - t0, 1)})
            except Exception as exc:  # noqa: BLE001
                ax.text(0.5, 0.5, f"not available\n{str(exc)[:60]}", ha="center", va="center", fontsize=8)
                ax.set(xticks=[], yticks=[])
            if i == 0:
                ax.set_title(SPECTRA_LABEL[sp], fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"{COHORT_LABEL[coh]}" + (f" {DEFAULT_CUTOFF[coh]}" if DEFAULT_CUTOFF[coh] else "")
                              + "\nFTIR-predicted EC (µg/m³)", fontsize=8.5)
            if i == len(order) - 1:
                ax.set_xlabel("HIPS Fabs/10 (µg/m³)", fontsize=8.5)
    fig.tight_layout()
    pd.DataFrame(grid_rows).to_csv(TABLES / "crossplot_grid_runs.csv", index=False)
    display(pd.DataFrame(grid_rows).round(3))
    save(fig, "05_crossplot_grid_cohort_by_spectra.png",
         "FTIR-predicted EC vs HIPS Fabs/10 (MAC 10, fixed 190 Addis filters) for each cohort family at its "
         "default cutoff × each spectra treatment, Option A rule k, training lot all; Deming line and 1:1.")
else:
    print("skipping the crossplot grid: start the explorer (calibration_explorer/run.sh) and rebuild.")

# %% [markdown]
# ## Manifest

# %%
lines = ["# ftir_55 every combination tried", ""] + [f"- `{n}`: {s}" for n, s in MANIFEST]
(PLOTS / "MANIFEST.md").write_text("\n".join(lines) + "\n")
with zipfile.ZipFile(PLOTS / "ftir55_every_combination.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for n, _ in MANIFEST:
        z.write(PLOTS / n, n)
    z.write(PLOTS / "MANIFEST.md", "MANIFEST.md")
display(Markdown("\n".join(lines)))

# %% [markdown]
# ## Takeaways
#
# - The search is not a black box: it is a distance to (slope 1, intercept 0) with slope
#   weighted five to one, evaluated only for configurations that keep held-out thermal-EC
#   skill and stay inside the training domain.
# - Baseline treatment separates the three clouds on the plane before any cohort choice
#   does; cohort choice then decides which blue points survive the held-out floor.
# - The best passing score sits in the lowest-OC/EC family on baseline-corrected spectra
#   around cutoff 440 to 490; the Addis score is flat there while the held-out R² is not
#   (fold-map sensitivity), so the basin should be quoted as score-stable.
#
# ## Limits
#
# - "Best" here means closest to the HIPS axis, not closest to thermal EC; ftir_44 shows
#   these same cohorts over-read Adama's quartz EC by 1.4 to 2x.
# - Older cache rows lack the extrapolation / Q / negatives fields and cannot pass the
#   full guardrail set; they appear on the plane but not among the outlined points.
