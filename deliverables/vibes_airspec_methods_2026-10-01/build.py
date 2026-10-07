"""VIBES vs AIRSpec methods explainer: figures + pptx, house style, no em dashes.

VIBES is shown as the FULL-RANGE run (pyvibes on every exported channel, 3998 to
500 cm-1), not the earlier comparison that cut VIBES to AIRSpec's 1425 to 4000 window.

Run with anaconda python from the repo root:
    ~/anaconda3/bin/python deliverables/vibes_airspec_methods_2026-10-01/build.py

Inputs (local generated tables, no explorer numbers):
- output/tables/vibes_colab_bundle_full500/stage           raw spectra + case selection
- output/tables/vibes_fullrange/rank30/                     full-range VIBES corrected (ASSEMBLY.json)
- output/tables/vibes_colab_full500_rank30/.../background_model.npz, background_rank_cv.csv
- output/tables/vibes_colab_cloud/.../full-83dcf32e86bc0f09  AIRSpec corrected, same cases
- output/tables/meeting_followup_20260917/{fits,panel_stats}.csv   EC results incl. VIBES-full
- research/ftir_hips_chem/vendor/pyvibes   (byte-identical to pyvibes-main 1.0.0)
- research/ftir_ec_phase3/scripts/airspec_baseline.py
"""

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
for old in FIG.glob("*.png"):
    old.unlink()
HC = ROOT / "research/ftir_hips_chem"
T = HC / "output/tables"
STAGE = T / "vibes_colab_bundle_full500/stage"
FULL = T / "vibes_fullrange/rank30"
BGDIR = T / "vibes_colab_full500_rank30/persistent_results/full500-rank30-137c7f6f3f9a6522"
EARLY = T / "vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"
MEET = T / "meeting_followup_20260917"

sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))
sys.path.insert(0, str(STAGE / "research/ftir_hips_chem/scripts"))
from airspec_baseline import (  # noqa: E402
    SEG1, SEG2, fit_spline_segm1, fit_spline_segm2ext, make_mask,
)
from vibes_large_run import RunConfig, prepare_experiment  # noqa: E402

GREY, BLUE, PURPLE, ACCENT, AMBER, INK = (
    "#8F8C84", "#2C6E9E", "#7A4FA3", "#B23327", "#C49442", "#22252A")
LIGHT = "#E9E7E1"
DPI = 168
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "text.color": INK, "font.size": 11,
    "axes.titlesize": 12, "legend.frameon": False,
})
WN_LABEL = "Wavenumber (cm$^{-1}$)"
CUT = 1425.0


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    Image.open(p).convert("RGB").save(p)
    return p


def wn_axis(ax, lo=500):
    ax.set_xlim(4000, lo)
    ax.set_xlabel(WN_LABEL)


def no_airspec(ax, label=True, y=None):
    ax.axvspan(CUT, 500, color=LIGHT, alpha=0.7, lw=0, zorder=0)
    if label:
        yy = ax.get_ylim()[1] if y is None else y
        ax.text(960, yy, "AIRSpec has no\noutput below 1425", ha="center", va="top",
                fontsize=9, color=GREY)


# ---------------------------------------------------------------- data
wn, X, cases, blanks, blank_ids, bmeta, _ = prepare_experiment(
    STAGE, RunConfig(profile="full", max_background_components=30))
assembly = json.loads((FULL / "ASSEMBLY.json").read_text())
assert assembly["n_failed"] == 0 and assembly["n_corrected"] == len(cases) == 12808
fcases = pd.read_csv(FULL / "cases.csv")
assert (fcases.sample_id.to_numpy() == cases.sample_id.to_numpy()).all()
V = np.load(FULL / "corrected_VIBES_full.npy").astype(float)
assert np.allclose(np.load(FULL / "wn.npy"), wn)
win = wn >= CUT
wa = np.load(EARLY / "wn.npy")
assert np.array_equal(wn[win], wa)
A = np.full(X.shape, np.nan)
A[:, win] = np.load(EARLY / "corrected_AIRSpec.npy")
bm = np.load(BGDIR / "background_model.npz")
mu, W = bm["mean"], bm["components"]
assert np.allclose(bm["wn"], wn) and np.allclose(mu, blanks.mean(axis=0))
C = W.shape[1]
rank_cv = pd.read_csv(BGDIR / "background_rank_cv.csv")
mcv, secv = rank_cv.mean_mse.values, rank_cv.se_mse.values
ONE_SE = int(1 + np.min(np.where(mcv - secv <= mcv.min())[0]))
assert ONE_SE == C == 30
diag = pd.read_csv(FULL / "fit_diagnostics.csv")
kind = cases.kind.to_numpy()
TGT = np.flatnonzero(kind == "target")
BLK = np.flatnonzero(kind == "blank")
CAL = np.flatnonzero(kind == "calibration")
INJ = np.flatnonzero(kind == "injection")
bA, bV = X - A, X - V
n_imp_bl = int((bmeta.source == "IMPROVE").sum())
n_etad_bl = int((bmeta.source == "ETAD").sum())

# Representative Addis filter: AIRSpec vs VIBES-full RMS difference (1425 to 4000)
# closest to the median of the 253 targets.
rmsd = np.sqrt(np.nanmean((A[TGT][:, win] - V[TGT][:, win]) ** 2, axis=1))
j = int(TGT[np.argmin(np.abs(rmsd - np.median(rmsd)))])
row = cases.iloc[j]
REP = f"{row.filter_id} (lot {int(row.lot)}, {row.date})"
REPID = row.filter_id
y = X[j]

# AIRSpec bounds for this filter, on the AIRSpec window grid; must reproduce the saved baseline.
xw, yw = wn[win], y[win]
m1, m2 = make_mask(xw, SEG1), make_mask(xw, SEG2)
s1 = fit_spline_segm1(xw[m1], yw[m1], 6)
s2 = fit_spline_segm2ext(xw[m2], yw[m2], 4)
acc = np.zeros_like(xw)
cnt = np.zeros_like(xw)
acc[m1] += s1["baseline"]
cnt[m1] += 1
acc[m2] += s2["baseline"]
cnt[m2] += 1
base_chk = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
ok = np.isfinite(base_chk)
assert np.max(np.abs(base_chk[ok] - bA[j][win][ok])) < 1e-5, np.max(np.abs(base_chk[ok] - bA[j][win][ok]))
B1 = (2220.0, float(s1["bounds"][1]))
B2 = (float(s2["bounds"][1]), 1820.0)
tail = xw[m2][np.flatnonzero(s2["absorbance"] == 0)]
TAIL = float(tail.max())


def weights(B):
    return np.linalg.lstsq(W, (B - mu).T, rcond=None)[0].T


x_rep = weights(bV[j][None])[0]
recon_err = np.max(np.abs(mu + W @ x_rep - bV[j]))
assert recon_err < 1e-4, recon_err
sig_rep = float(diag.loc[diag.case_row == j, "sigma"].iloc[0])

fits = pd.read_csv(MEET / "fits.csv").set_index("fit_id")
panel = pd.read_csv(MEET / "panel_stats.csv")
meet_summary = json.loads((MEET / "summary.json").read_text())
EC_METHODS = [("AIRSpec", "AIRSpec\n4000-1425", BLUE),
              ("VIBES", "VIBES cut\n4000-1425", "#B9A3CF"),
              ("VIBES-full-cut", "VIBES fit full,\nused 4000-1425", "#9C7CBD"),
              ("VIBES-full", "VIBES full\n4000-500", PURPLE)]
ec = {}
for m, _, _ in EC_METHODS:
    f = fits.loc[f"allimprove|{m}|all-IMPROVE lot 251"]
    p = panel[(panel.fit_id == f"allimprove|{m}|all-IMPROVE lot 251") & (panel.population == "addis")
              & (panel.panel == "ftir_vs_fabs") & (panel.eval_group == "All Addis")].iloc[0]
    ec[m] = dict(k=int(f.k), R2=f.TOR_test_R2, RMSE=f.TOR_test_RMSE_ug, n_test=int(f.test_n),
                 n_train=int(f.train_n), slope=p.deming_slope, icpt=p.deming_intercept,
                 n_addis=int(p.n), R2_addis=p.R2)
LAM = meet_summary["lambda"]

REGIONS = [
    ("4000-3700", 3700, 4000), ("3700-3000\nO-H, N-H", 3000, 3700),
    ("3000-2800\nC-H", 2800, 3000), ("2220-1820", 1820, 2220),
    ("1820-1600\nC=O", 1600, 1820), ("1600-1425", 1425, 1600),
    ("1425-1300", 1300, 1425), ("1300-1100\nPTFE C-F", 1100, 1300),
    ("1100-900\nC-O, Si-O", 900, 1100), ("900-500", 500, 900),
]


def region_mean(M, lo, hi):
    s = (wn >= lo) & (wn < hi)
    with np.errstate(all="ignore"):
        return np.nanmean(M[:, s], axis=1) if np.isfinite(M[:, s]).any() else np.full(len(M), np.nan)


rng = np.random.default_rng(1)

# ---------------------------------------------------------------- f1 pipeline
fig, ax = plt.subplots(figsize=(12.5, 5.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)
ax.axis("off")


def box(x, y0, w, h, text, color, size=11):
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                ec=color, fc="white", lw=1.8))
    ax.text(x + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=size,
            color=INK, linespacing=1.35)


def arrow(x0, y0, x1, y1, color=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))


ax.text(0, 49.5, "AIRSpec (segmented smoothing spline)", color=BLUE, fontsize=13, weight="bold")
box(1, 33, 15, 11, "one filter\nspectrum", GREY)
box(23, 33, 20, 11, "mask windows assumed\nto hold analyte\n(weight 0)", BLUE)
box(50, 33, 20, 11, "smoothing spline through\nthe rest (df 6 and 4,\ntwo segments)", BLUE)
box(77, 33, 21, 11, "subtract: absorbance\n4000 to 1425 cm$^{-1}$ only", BLUE)
for a_, b_ in [(16.8, 23), (43.8, 50), (70.8, 77)]:
    arrow(a_, 38.5, b_ - 0.8, 38.5, BLUE)
ax.text(0, 25.5, "VIBES (pyvibes 1.0.0, blank-trained, full range)", color=PURPLE, fontsize=13, weight="bold")
box(1, 12.5, 15, 9, f"{len(blanks)} blank\nfilters", GREY)
box(23, 12.5, 20, 9, f"PCA: mean blank +\n{C} patterns", PURPLE)
box(1, 1, 15, 8, "one filter\nspectrum", GREY)
box(50, 5, 20, 12, "fit pattern weights with\nasymmetric loss + prior;\nELBO sets the penalty", PURPLE)
box(77, 6, 21, 10, "subtract: absorbance\n4000 to 500 cm$^{-1}$", PURPLE)
arrow(16.8, 17, 22.2, 17, PURPLE)
arrow(43.8, 15, 49.2, 12.5, PURPLE)
arrow(16.8, 5, 49.2, 8.5, PURPLE)
arrow(70.8, 11, 76.2, 11, PURPLE)
ax.text(1, -2.5, "Both feed a separately fitted PLS model for EC; neither corrected spectrum is ground truth.",
        fontsize=10.5, color=GREY, style="italic")
f1 = save(fig, "f01_two_pipelines.png")

# ---------------------------------------------------------------- f2 AIRSpec mechanism
fig, (ax, bx) = plt.subplots(2, 1, figsize=(12.5, 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08))
mask_pts = np.zeros_like(wn, bool)
mask_pts[win] = make_mask(xw, B1) | make_mask(xw, B2)
fitpts = win & ~mask_pts
ax.plot(wn, y, color=GREY, lw=1.5, label="measured spectrum")
ax.plot(wn[fitpts], y[fitpts], ".", color=INK, ms=1.5, label="points the spline is fitted to")
ax.plot(wn, bA[j], color=BLUE, lw=2.2, label="AIRSpec baseline")
ax.set_ylim(min(0, np.nanmin(y[wn > 900])) - 0.02, np.nanmax(y[wn > 1400]) * 1.25)
for lo, hi in [B1, B2]:
    for a_ in (ax, bx):
        a_.axvspan(lo, hi, color=AMBER, alpha=0.13, lw=0)
no_airspec(ax, y=0.33)
ax.text(np.mean(B1), ax.get_ylim()[1] * 0.95, f"masked analyte window\n2220 to {B1[1]:.0f} cm$^{{-1}}$\n(upper edge found per spectrum)",
        ha="center", va="top", fontsize=9.5)
ax.text(np.mean(B2), ax.get_ylim()[1] * 0.95, f"masked\n1820 to\n{B2[0]:.0f}", ha="center", va="top", fontsize=9.5)
ax.set_ylabel("Absorbance")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.32), fontsize=9.5)
bx.plot(wn, A[j], color=BLUE, lw=1.4)
bx.axhline(0, color=GREY, lw=0.8, ls=":")
bx.set_ylabel("Corrected")
no_airspec(bx, label=False)
bx.text(2200, np.nanmax(A[j]) * 0.5, "outside the masks\nthe corrected spectrum\nis ~0 by construction",
        fontsize=9, color=BLUE)
wn_axis(bx)
f2 = save(fig, "f02_airspec_mechanism.png")

# ---------------------------------------------------------------- f3 AIRSpec df
fig, bx = plt.subplots(figsize=(12, 4.8))
for df1, col in {4: AMBER, 6: BLUE, 8: PURPLE}.items():
    s = fit_spline_segm1(xw[m1], yw[m1], df1)
    lab = f"df1 = {df1}" + ("  (our default)" if df1 == 6 else "  (stiffer)" if df1 == 4 else "  (bendier)")
    bx.plot(xw[m1], s["absorbance"], color=col, lw=2.2 if df1 == 6 else 1.3, label=lab)
bx.set_xlim(4000, 1820)
bx.set_xlabel(WN_LABEL)
bx.set_ylabel("Corrected absorbance, segment 1")
bx.axhline(0, color=GREY, lw=0.8, ls=":")
bx.legend(fontsize=10, loc="upper right")
f3 = save(fig, "f03_airspec_df.png")

# ---------------------------------------------------------------- f4 VIBES blank model
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1.35, 1], wspace=0.3))
ax = axs[0]
src = bmeta.source.to_numpy()
for z, s_ in zip(blanks, src):
    ax.plot(wn, z, color=GREY if s_ == "IMPROVE" else AMBER, lw=0.5, alpha=0.5)
ax.plot(wn, mu, color=INK, lw=1.8, label="model mean term (μ)")
ax.plot([], [], color=GREY, lw=0.8, label=f"{n_imp_bl} IMPROVE blanks")
ax.plot([], [], color=AMBER, lw=0.8, label=f"{n_etad_bl} ETAD blanks")
wn_axis(ax)
ax.set_ylabel("Absorbance")
ax.set_title("1. blanks: what an empty filter looks like", loc="left", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax = axs[1]
for k_, col in zip(range(3), [PURPLE, BLUE, AMBER]):
    ax.plot(wn, W[:, k_], color=col, lw=1.2, label=f"pattern {k_ + 1}")
ax.axhline(0, color=GREY, lw=0.6, ls=":")
wn_axis(ax)
ax.set_ylabel("Pattern loading (absorbance)")
ax.set_title("2. PCA turns their variation into patterns", loc="left", fontsize=11)
ax.legend(fontsize=9)
ax = axs[2]
cc = rank_cv.components.values
ax.errorbar(cc, mcv, yerr=secv, color=PURPLE, lw=1.2, ms=3, fmt="o-", capsize=0)
ax.set_yscale("log")
ax.axvline(C, color=ACCENT, lw=1.2, ls="--")
ax.text(C - 0.8, mcv.max() / 3, f"c = {C}\n(the cap;\nstill falling)", color=ACCENT, ha="right", fontsize=9.5)
ax.set_xlabel("Number of patterns c")
ax.set_ylabel("Leave-one-blank-out error")
ax.set_title("3. rank chosen on blanks only", loc="left", fontsize=11)
f4 = save(fig, "f04_vibes_blank_model.png")

# ---------------------------------------------------------------- f5 where analyte is allowed
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.28))
wts = np.where(mask_pts, 0.0, 1.0)
wts[~win] = np.nan
wts[win & (wn < TAIL)] = np.nan
ax.fill_between(wn, 0, np.nan_to_num(wts), step="mid", color=BLUE, alpha=0.25, lw=0)
ax.plot(wn, wts, color=BLUE, lw=1.6, drawstyle="steps-mid")
ax.set_ylim(-0.1, 1.4)
ax.set_yticks([0, 1])
ax.set_yticklabels(["0: free\n(analyte)", "1: must fit\n(no analyte)"])
for lo, hi in [B1, B2]:
    ax.axvspan(lo, hi, color=AMBER, alpha=0.13, lw=0)
no_airspec(ax, y=1.35)
ax.text(1680, 1.12, f"below ~{TAIL:.0f}:\nbaseline = spectrum", ha="center", fontsize=8.5, color=BLUE)
wn_axis(ax)
ax.set_title(f"AIRSpec: a yes/no weight per wavenumber ({REPID})", loc="left", fontsize=11)
a = np.linspace(-1, 1, 401)
tau = 0.1
bx.plot(a, 0.5 * np.abs(a) + (tau - 0.5) * a, color=PURPLE, lw=2.4, label="VIBES pinball loss, τ = 0.1")
bx.plot(a, 0.5 * np.abs(a), color=GREY, lw=1.4, ls="--", label="symmetric (τ = 0.5)")
bx.axvline(0, color=GREY, lw=0.6, ls=":")
bx.set_xlabel("spectrum minus background")
bx.set_ylabel("penalty")
bx.annotate("above background:\ncheap (0.1 per unit)", xy=(0.7, 0.07), xytext=(0.62, 0.6), ha="center",
            fontsize=9.5, color=PURPLE, arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.8))
bx.text(-0.25, 0.72, "below background:\ncostly (0.9 per unit)", ha="left", fontsize=9.5, color=PURPLE)
bx.set_title("VIBES: every point counts, asymmetrically", loc="left", fontsize=11)
bx.legend(fontsize=9, loc="upper center")
f5 = save(fig, "f05_where_analyte_allowed.png")

# ---------------------------------------------------------------- f6 VIBES fit on one filter
contrib = W * x_rep
top = np.argsort(np.sqrt((contrib ** 2).mean(axis=0)))[::-1][:3]
fig, (ax, bx) = plt.subplots(2, 1, figsize=(12.5, 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1.2], hspace=0.08))
ax.plot(wn, y, color=GREY, lw=1.5, label="measured spectrum")
ax.plot(wn, mu, color=INK, lw=1.1, ls="--", label="mean blank (μ)")
ax.plot(wn, bV[j], color=PURPLE, lw=1.8, label=f"VIBES baseline = μ + weighted sum of {C} patterns")
ax.fill_between(wn, bV[j], y, where=y > bV[j], color=PURPLE, alpha=0.18, lw=0, label="what is left: absorbance")
ax.set_ylabel("Absorbance")
ax.legend(fontsize=9.5, loc="upper left")
for k_, col in zip(top, [PURPLE, BLUE, AMBER]):
    bx.plot(wn, contrib[:, k_], color=col, lw=1.2, label=f"pattern {k_ + 1} × weight {x_rep[k_]:+.1f}")
bx.plot(wn, contrib.sum(axis=1), color=INK, lw=1.0, alpha=0.6, label="sum of all patterns")
bx.axhline(0, color=GREY, lw=0.6, ls=":")
bx.set_ylabel("Pattern term")
bx.legend(fontsize=8.5, ncol=2, loc="upper right")
wn_axis(bx)
f6 = save(fig, "f06_vibes_fit_one_filter.png")

# ---------------------------------------------------------------- f7 same filter, both
fig, (ax, bx) = plt.subplots(2, 1, figsize=(12.5, 6.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.5, 1.5], hspace=0.08))
ax.plot(wn, y, color=GREY, lw=1.5, label="measured spectrum")
ax.plot(wn, bA[j], color=BLUE, lw=2, label="AIRSpec baseline")
ax.plot(wn, bV[j], color=PURPLE, lw=1.6, label="VIBES baseline")
ax.set_ylabel("Absorbance")
ax.legend(fontsize=9.5, loc="upper left")
bx.plot(wn, A[j], color=BLUE, lw=1.4, label="AIRSpec corrected")
bx.plot(wn, V[j], color=PURPLE, lw=1.1, label="VIBES corrected")
bx.axhline(0, color=GREY, lw=0.6, ls=":")
bx.set_ylabel("Corrected absorbance")
lim = np.nanpercentile(np.abs(V[j]), 99.5) * 1.3
bx.set_ylim(-0.25 * lim, lim)
for lo, hi in [B1, B2]:
    for a_ in (ax, bx):
        a_.axvspan(lo, hi, color=AMBER, alpha=0.13, lw=0)
no_airspec(ax, y=0.33)
no_airspec(bx, label=False)
bx.legend(fontsize=9.5, loc="upper left")
wn_axis(bx)
f7 = save(fig, "f07_same_filter_both.png")

# ---------------------------------------------------------------- f8 population, per filter
fig, ax = plt.subplots(figsize=(13, 5.0))
reg_stats = {}
for i, (lab, lo, hi) in enumerate(REGIONS):
    ra, rv = region_mean(A[TGT], lo, hi), region_mean(V[TGT], lo, hi)
    reg_stats[lab.split("\n")[0]] = (np.nanmedian(ra) if np.isfinite(ra).any() else np.nan, np.median(rv))
    pairs = [(rv, 0.0, PURPLE)] if not np.isfinite(ra).any() else [(ra, -0.18, BLUE), (rv, 0.18, PURPLE)]
    for vals, off, col in pairs:
        ax.scatter(i + off + rng.uniform(-0.07, 0.07, vals.size), vals, s=4, color=col, alpha=0.35, lw=0)
        ax.hlines(np.median(vals), i + off - 0.13, i + off + 0.13, color=col, lw=2.6)
    if hi <= CUT:
        ax.axvspan(i - 0.5, i + 0.5, color=LIGHT, alpha=0.7, lw=0, zorder=0)
ax.axhline(0, color=GREY, lw=0.8, ls=":")
ax.set_xticks(range(len(REGIONS)))
ax.set_xticklabels([r[0] for r in REGIONS], fontsize=9)
ax.set_xlabel("Region (cm$^{-1}$)")
ax.scatter([], [], color=BLUE, s=20, label="AIRSpec")
ax.scatter([], [], color=PURPLE, s=20, label="VIBES (full range)")
ax.set_ylabel("Mean corrected absorbance in region\n(one dot per filter, bar = median)")
ax.set_ylim(-0.01, np.percentile(region_mean(V[TGT], 1300, 1425), 99) * 1.3)
ax.legend(fontsize=10, loc="upper left")
ax.text(len(REGIONS) - 2.5, ax.get_ylim()[1] * 0.97, "shaded: VIBES only", ha="center", va="top", fontsize=9, color=GREY)
f8 = save(fig, "f08_population_regions.png")

# ---------------------------------------------------------------- f9 extrapolation
groups = [
    (f"training blanks\n(n = {len(blanks)})", weights(blanks), GREY),
    (f"held-out blanks\n(n = {len(BLK)})", weights(bV[BLK]), AMBER),
    (f"IMPROVE samples\n(n = {len(CAL)})", weights(bV[CAL]), BLUE),
    (f"Addis samples\n(n = {len(TGT)})", weights(bV[TGT]), PURPLE),
]
wt_norm = {}
fig, ax = plt.subplots(figsize=(11.5, 4.9))
for i, (lab, x, col) in enumerate(groups):
    v = np.linalg.norm(x, axis=1)
    wt_norm[lab.split("\n")[0]] = (np.median(v), np.percentile(v, 90))
    if v.size > 2000:
        v = rng.choice(v, 2000, replace=False)
    ax.scatter(i + rng.uniform(-0.13, 0.13, v.size), v, s=8 if v.size > 100 else 24, color=col, alpha=0.45, lw=0)
    med = wt_norm[lab.split("\n")[0]][0]
    ax.hlines(med, i - 0.25, i + 0.25, color=INK, lw=2.2)
    ax.text(i + 0.29, med, f"median {med:.1f}", va="center", fontsize=9.5)
ax.axhline(np.sqrt(C), color=GREY, lw=0.9, ls=":")
ax.text(3.6, np.sqrt(C) * 0.86, f"typical size under the prior (√{C} ≈ {np.sqrt(C):.1f})", ha="right", fontsize=9, color=GREY, va="top")
ax.set_yscale("log")
ax.set_yticks([3, 5, 10, 20, 50, 100])
ax.set_yticklabels(["3", "5", "10", "20", "50", "100"])
ax.minorticks_off()
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups])
ax.set_xlim(-0.5, 3.75)
ax.set_ylabel("Size of fitted pattern weights\n(length of weight vector)")
f9 = save(fig, "f09_weights_extrapolation.png")

# ---------------------------------------------------------------- f10 held-out blanks
def rms(M, sel):
    with np.errstate(all="ignore"):
        return np.sqrt(np.nanmean(M[:, sel] ** 2, axis=1))


bsrc = cases.source.to_numpy()[BLK]
rA, rV = rms(A[BLK], win), rms(V[BLK], win)
rVlow = rms(V[BLK], ~win)
ra_med = np.median(rA)
k_med = int(np.argmin(np.abs(rA - ra_med)))
kb = BLK[k_med]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.28))
ax.plot(wn, A[kb], color=BLUE, lw=1.1, label="AIRSpec corrected")
ax.plot(wn, V[kb], color=PURPLE, lw=1.1, label="VIBES corrected")
ax.axhline(0, color=GREY, lw=0.6, ls=":")
no_airspec(ax, label=False)
wn_axis(ax)
ax.set_ylabel("Corrected absorbance")
ax.set_title(f"held-out blank {cases.sample_id.iloc[kb]} (median AIRSpec residual)", loc="left", fontsize=11)
ax.legend(fontsize=9.5, loc="upper left")
for xo, vals, col, lab in [(0, rA, BLUE, "AIRSpec\n4000-1425"), (1, rV, PURPLE, "VIBES\n4000-1425"),
                           (2, rVlow, PURPLE, "VIBES\n1425-500")]:
    bx.scatter(xo + rng.uniform(-0.1, 0.1, vals.size), vals, color=col, s=10, alpha=0.55 if xo < 2 else 0.3, lw=0)
    bx.hlines(np.median(vals), xo - 0.22, xo + 0.22, color=INK, lw=2.2)
bx.set_yscale("log")
bx.set_xticks([0, 1, 2])
bx.set_xticklabels(["AIRSpec\n4000-1425", "VIBES\n4000-1425", "VIBES\n1425-500"], fontsize=9.5)
bx.set_xlim(-0.5, 2.5)
bx.set_ylabel("RMS of corrected blank (absorbance)")
bx.set_title(f"all {len(BLK)} held-out blanks", loc="left", fontsize=11)
f10 = save(fig, "f10_heldout_blanks.png")

# ---------------------------------------------------------------- f11 injections
parent = cases.parent.to_numpy()
amp = cases.amplitude.to_numpy()
inj_rows = []
for i in INJ:
    p = int(parent[i])
    truth = (X[i] - X[p])[win]
    for meth, M in [("AIRSpec", A), ("VIBES", V)]:
        rec = (M[i] - M[p])[win]
        inj_rows.append(dict(amp=amp[i], method=meth, rmse=float(np.sqrt(np.mean((rec - truth) ** 2)))))
inj = pd.DataFrame(inj_rows)
injm = inj.groupby(["amp", "method"]).rmse.median().unstack()
n_parent = len(set(parent[INJ].astype(int)))
fig, ax = plt.subplots(figsize=(10, 4.8))
amps = sorted(inj.amp.unique())
for i, a_ in enumerate(amps):
    for meth, col, off in [("AIRSpec", BLUE, -0.15), ("VIBES", PURPLE, 0.15)]:
        v = inj.query("amp==@a_ and method==@meth").rmse.values
        ax.scatter(i + off + rng.uniform(-0.05, 0.05, v.size), v, color=col, s=20, alpha=0.8)
        ax.hlines(np.median(v), i + off - 0.11, i + off + 0.11, color=col, lw=2.6)
ax.set_xticks(range(len(amps)))
ax.set_xticklabels([f"added peak height {a_:g}" for a_ in amps])
ax.set_yscale("log")
ax.set_yticks([1e-4, 3e-4, 1e-3, 3e-3])
ax.set_yticklabels(["0.0001", "0.0003", "0.001", "0.003"])
ax.minorticks_off()
ax.set_ylabel("Recovery error, RMSE (absorbance)\nscored on 4000 to 1425")
ax.scatter([], [], color=BLUE, label="AIRSpec")
ax.scatter([], [], color=PURPLE, label="VIBES (full range)")
ax.legend(fontsize=10, loc="upper left")
ax.text(2.45, ax.get_ylim()[0] * 1.2, f"same {n_parent} parent blanks at each height", ha="right", fontsize=9, color=GREY)
f11 = save(fig, "f11_injection_recovery.png")

# ---------------------------------------------------------------- f12 EC downstream
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw=dict(wspace=0.28, width_ratios=[1.1, 1.1]))
for i, (m, lab, col) in enumerate(EC_METHODS):
    v = ec[m]["RMSE"]
    ax.bar(i, v, width=0.65, color=col)
    ax.text(i, v + 0.03, f"{v:.2f}\nR² {ec[m]['R2']:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(EC_METHODS)))
ax.set_xticklabels([e[1] for e in EC_METHODS], fontsize=8.5)
ax.set_ylabel("Held-out TOR EC error, RMSE (µg/filter)")
ax.set_ylim(0, max(e["RMSE"] for e in ec.values()) * 1.25)
ax.set_title(f"IMPROVE test sites (n = {ec['AIRSpec']['n_test']})", loc="left", fontsize=11)
for i, (m, lab, col) in enumerate(EC_METHODS):
    bx.scatter(ec[m]["slope"], ec[m]["icpt"], s=110, color=col, edgecolor=INK, lw=0.6, zorder=3)
    bx.annotate(lab.replace("\n", " "), (ec[m]["slope"], ec[m]["icpt"]), xytext=(8, 6),
                textcoords="offset points", fontsize=8.8)
bx.scatter([1], [0], marker="*", s=180, color=INK, zorder=3)
bx.annotate("ideal (1, 0)", (1, 0), xytext=(8, -12), textcoords="offset points", fontsize=9)
bx.axhline(0, color=GREY, lw=0.6, ls=":")
bx.axvline(1, color=GREY, lw=0.6, ls=":")
bx.set_xlim(0.3, 2.05)
bx.set_xlabel("Deming slope, FTIR EC vs HIPS Fabs/10")
bx.set_ylabel("Deming intercept (µg/m³)")
bx.set_title(f"same models applied to Addis (n = {ec['AIRSpec']['n_addis']})", loc="left", fontsize=11)
f12 = save(fig, "f12_ec_downstream.png")

# ---------------------------------------------------------------- numbers for notes
nret = int(diag.retry_count.sum())
sec_med = float(diag.seconds.median())
wn_tr, wn_hb, wn_imp, wn_add = (wt_norm[k][0] for k in ["training blanks", "held-out blanks",
                                                          "IMPROVE samples", "Addis samples"])
blank_ratio = ra_med / np.median(rV)
blank_by_src = {s_: (np.median(rA[bsrc == s_]), np.median(rV[bsrc == s_])) for s_ in ["etad", "improve"]}
reg_line = "; ".join(f"{k}: " + (f"{a_:.4f} vs {b_:.4f}" if np.isfinite(a_) else f"VIBES {b_:.4f}")
                     for k, (a_, b_) in reg_stats.items())
E = ec
print("rep", REP, "B1", B1, "B2", B2, "tail", TAIL, "sigma", sig_rep)
print("regions", reg_line)
print("weights", wt_norm)
print("blanks", ra_med, np.median(rV), np.median(rVlow), blank_by_src)
print("inj", injm.to_dict())
print("ec", {k: (round(v['RMSE'], 3), round(v['slope'], 2), round(v['icpt'], 2)) for k, v in ec.items()})

# ---------------------------------------------------------------- deck
INK_RGB = RGBColor(0x22, 0x25, 0x2A)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank_layout = prs.slide_layouts[6]


def slide(title, fig=None, lines=None, table=None, say="", notes=""):
    s = prs.slides.add_slide(blank_layout)
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.22), Inches(12.5), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = INK_RGB
    if fig is not None:
        im = Image.open(fig)
        scale = min(12.2 / (im.width / DPI), 5.9 / (im.height / DPI), 1.35)
        w = im.width / DPI * scale
        s.shapes.add_picture(str(fig), Inches((13.333 - w) / 2),
                             Inches(1.25 + (5.9 - im.height / DPI * scale) / 2), Inches(w))
    if lines:
        body = s.shapes.add_textbox(Inches(0.75), Inches(1.6), Inches(11.8), Inches(5.2))
        btf = body.text_frame
        btf.word_wrap = True
        for i, line in enumerate(lines):
            para = btf.paragraphs[0] if i == 0 else btf.add_paragraph()
            para.text = line
            para.font.size = Pt(17)
            para.font.color.rgb = INK_RGB
            para.space_after = Pt(12)
    if table:
        nr, nc = len(table), len(table[0])
        shp = s.shapes.add_table(nr, nc, Inches(0.55), Inches(1.3), Inches(12.2), Inches(0.5 * nr))
        tbl = shp.table
        tbl.columns[0].width = Inches(2.6)
        tbl.columns[1].width = Inches(4.8)
        tbl.columns[2].width = Inches(4.8)
        for ri, row_ in enumerate(table):
            for ci, txt in enumerate(row_):
                cell = tbl.cell(ri, ci)
                cell.text = txt
                cell.fill.solid()
                cell.fill.fore_color.rgb = (RGBColor(0xEE, 0xEC, 0xE6) if ri == 0
                                            else RGBColor(0xFF, 0xFF, 0xFF))
                for para in cell.text_frame.paragraphs:
                    para.font.size = Pt(12.5)
                    para.font.bold = ri == 0 or ci == 0
                    para.font.color.rgb = (
                        RGBColor(0x2C, 0x6E, 0x9E) if (ri == 0 and ci == 1) else
                        RGBColor(0x7A, 0x4F, 0xA3) if (ri == 0 and ci == 2) else INK_RGB)
    s.notes_slide.notes_text_frame.text = f"SAY: {say}\n\nNOTES: {notes}"


# 1
slide(
    "AIRSpec and VIBES both subtract a background, but they get it from different places",
    lines=[
        "How pyvibes (VIBES) and AIRSpec baselining differ, step by step",
        "VIBES shown over its full range (4000 to 500 cm⁻¹); AIRSpec over its own (4000 to 1425)",
        "Same Addis and IMPROVE spectra, same blanks, same downstream PLS",
        "Methods explainer, October 2026",
    ],
    say=("Today is a methods talk. I want everyone to leave knowing what each baseline method "
         "actually assumes, because that decides when we should trust each one. I will walk "
         "through AIRSpec first, then VIBES, then put the same filter through both. One thing up "
         "front: VIBES is shown over the whole measured spectrum, down to 500, because unlike "
         "AIRSpec it has no built-in window. Then a diagnostic on how far VIBES has to stretch "
         "to fit loaded filters, and finally what blanks, added peaks and EC prediction say. "
         "Short version: AIRSpec assumes the background is smooth and that some regions hold no "
         "analyte; VIBES assumes the background looks like our blanks."),
    notes=("VIBES = pyvibes 1.0.0 (Kamper, Krymova, Takahama; MIT; accompanying 'Towards "
           "Universal Calibration for Infrared Spectroscopy with Probabilistic Interference "
           "Removal'). research/ftir_hips_chem/vendor/pyvibes is byte-identical to "
           "~/Downloads/pyvibes-main. AIRSpec = segmented smoothing-spline baseline (NumPy port of "
           f"FitSplineKDT.R, research/ftir_ec_phase3/scripts/airspec_baseline.py). VIBES arrays: "
           f"full-range rank-30 run, {assembly['n_corrected']} of {assembly['n_cases']} cases "
           f"corrected, {assembly['n_failed']} failed, {assembly['channels']} channels "
           f"{assembly['wavenumber_range'][0]:.0f} to {assembly['wavenumber_range'][1]:.0f} cm-1 "
           f"(raw IMPROVE and Addis grids stop at 500, not 400). Receipt: "
           f"output/tables/vibes_fullrange/rank30/ASSEMBLY.json. AIRSpec arrays: the frozen "
           f"Colab run full-83dcf32e86bc0f09, same case order. Earlier results deck (VIBES cut to "
           f"1425): deliverables/airspec_vibes_2026-09-22."),
)

# 2
slide(
    "AIRSpec reads the background off each spectrum; VIBES learns it from blank filters first",
    fig=f1,
    say=("The whole difference on one page. Top row, AIRSpec: it only looks at the one "
         "spectrum in front of it. It blanks out regions where it expects analyte, draws a "
         "smooth curve through the rest, and subtracts, and it only works from 4000 down to "
         "1425. Bottom row, VIBES: before touching a sample it studies a library of blank "
         "filters and summarises how empty filters vary. Then for each sample it asks what mix "
         "of those blank patterns sits underneath the spectrum, over the whole range the blanks "
         "cover. AIRSpec needs nothing extra but assumes smoothness; VIBES needs good blanks "
         "but assumes nothing about smoothness."),
    notes=(f"AIRSpec: two segments, 4000 to 1820 (df 6) and 2000 to 1425 (df 4), averaged in "
           f"2000 to 1820; NaN outside 1425 to 4000 by construction. VIBES: PCA on centred blanks "
           f"(pca()), rank by leave-one-blank-out one-SE rule (loo_pca) capped at 30, per-spectrum "
           f"ELBO maximisation with L-BFGS-B (VibeSpec.fit), then convex MAP solve with CVXPY/"
           f"Clarabel (map_pb). pyvibes also ships EBS (sigma = 0, tuned by blocked CV); not run. "
           f"Cost: AIRSpec about 4 ms per spectrum locally; full-range VIBES median {sec_med:.1f} s "
           f"per spectrum on a Colab vCPU (different hardware, so only order of magnitude); "
           f"{nret} of {len(diag)} fits used one logged optimizer restart, none failed."),
)

# 3
slide(
    "AIRSpec fits a smooth curve only where it assumes there is no analyte, and stops at 1425",
    fig=f2,
    say=(f"One real Addis filter, {REPID}, chosen because its AIRSpec versus VIBES difference is "
         "the median of all 253 Addis filters, so it is typical, not a showcase. Grey is the "
         "measured spectrum over the full range. The shaded bands are the windows AIRSpec treats "
         "as analyte, with zero weight. Black dots are everything the spline sees, blue is the "
         "spline. The grey block on the right is where AIRSpec does nothing at all: below 1425, "
         "including the big Teflon bands. Below, the corrected spectrum: outside the shaded "
         "windows it sits on zero, which is an assumption, not a measurement."),
    notes=(f"Selection rule: target whose AIRSpec vs full-range VIBES RMS difference over 1425 to "
           f"4000 is closest to the median of 253 ETAD PM2.5 targets: {REP}. Recomputing the "
           f"AIRSpec segments on its raw spectrum reproduces the saved baseline (< 1e-5). Segment 1 "
           f"window: fixed 2220 low side; high side starts at 3710 and steps down 10 cm-1 while "
           f"the corrected sum just inside is negative (FindBound), here {B1[1]:.0f}. Segment 2: "
           f"1820 to the spectrum minimum in 1600 to 1520 (FindMinPos), here {B2[0]:.0f}; 20 "
           f"points past that the baseline is set equal to the spectrum, so corrected absorbance "
           f"is exactly zero below about {TAIL:.0f}. y axis is clipped above the PTFE C-F bands "
           f"(about 1150 and 1210) so the upper region stays readable."),
)

# 4
slide(
    "AIRSpec's main knob is spline stiffness, and it moves the broad 3000 to 3700 band",
    fig=f3,
    say=("AIRSpec has very few settings. The main one is how stiff the spline is, set as "
         "effective degrees of freedom; we use 6 for the upper segment. Here is the same "
         "filter's corrected spectrum at three settings. It changes mostly in the broad O-H and "
         "N-H region, while the sharp C-H peaks near 2900 barely move. So what AIRSpec is least "
         "sure about is broad features, because it cannot tell a broad absorption from a curving "
         "background."),
    notes=("Same filter; segment 1 only (4000 to 1820), df2 fixed at 4. df 6 and 4 are the "
           "AIRSpec/APRL defaults for PTFE. The locked reproduction found DF1 = 6 and DF1 = 8 both "
           "recover PLS k = 5 on the historical calibration. The bound search reruns for each df, "
           "so the window edge can move slightly."),
)

# 5
slide(
    f"VIBES first learns what an empty filter looks like: {len(blanks)} blanks become {C} patterns over 4000 to 500",
    fig=f4,
    say=(f"Now VIBES. Step one has nothing to do with samples. Left: {len(blanks)} real blank "
         f"filters, {n_imp_bl} from IMPROVE and {n_etad_bl} from Addis, one line each, over the "
         "full range; the big bumps near 1200 are the Teflon itself. Middle: PCA pulls out the main "
         "ways blanks differ; these are the first three patterns. Right: how many patterns to keep, "
         "by leaving one blank out and rebuilding it from the others. On the full range the error "
         f"is still falling at our cap of {C}, so the model is using as many patterns as we allowed. "
         "Everything VIBES later calls background has to be built from that mean plus those patterns."),
    notes=(f"Background training: {len(blanks)} independent physical blanks ({n_imp_bl} IMPROVE "
           f"train-site blanks across lots, {n_etad_bl} ETAD), from vibes_large_run.prepare_experiment "
           f"(seeded; outer-test-site blanks never used). Rank cap 30 for like-for-like with the "
           f"1425-cut run; one-SE rule lands on the cap ({ONE_SE}), i.e. the cap binds. Without the "
           f"cap the full-grid one-SE rank is about 67 to 70; that uncapped rerun (rank cap 85) is "
           f"only 3.5% complete (uncapped_completion_audit_20261001.json), so all results here are "
           f"rank 30. Patterns are pyvibes' W: right singular vectors scaled by singular value over "
           f"sqrt(n), so weight 1 = a typical blank-sized deviation. Rank selection never sees TOR "
           f"or samples. The mean term is a model parameter, not a sample average."),
)

# 6
slide(
    "AIRSpec decides where analyte may appear; VIBES lets it appear anywhere but only above the background",
    fig=f5,
    say=("The core philosophical difference. Left: AIRSpec's rule for this filter. Each "
         "wavenumber is either 'must fit the baseline' or 'free', yes or no, and below 1425 there "
         "is no rule because there is no correction. Right: VIBES has no windows. Every point is "
         "fitted, with a lopsided penalty: spectrum above the background is cheap, because that is "
         "what absorbance looks like; background poking above the spectrum costs nine times more. "
         "So the background is pushed underneath everywhere, rather than pinned to zero in fixed places."),
    notes=("pyvibes pb_loss: 0.5|a| + (τ minus 0.5)a with a = spectrum minus background; τa for "
           "a > 0, (1 minus τ)|a| for a < 0; ratio 9 at τ = 0.1. τ fixed at 0.1 (pyvibes default); "
           "pyvibes can also estimate τ from the ELBO (bounds 1e-5 to 0.5). ALS (asymmetric squared) "
           "loss also available; we used PB. The analyte term gets the Gibbs distribution of this "
           "loss and the pattern weights a standard normal prior, which makes VIBES a probabilistic "
           "model rather than a curve fit."),
)

# 7
slide(
    "For each filter VIBES fits weights on the blank patterns, not a free-form curve",
    fig=f6,
    say=(f"Same typical filter. Grey is the spectrum, dashed is the mean blank, purple is the "
         f"VIBES background: the mean plus a weighted sum of the {C} patterns, across the whole "
         "range including the Teflon bands. The shaded area is what VIBES calls absorbance. "
         "Bottom: the three patterns doing most of the work for this filter. Notice how far the "
         "background sits from the mean blank, and how large those weights are; I will come back "
         "to that. The key point: the background can only take shapes the blanks have shown us."),
    notes=(f"Weights recovered by least squares from the saved full-range VIBES baseline on the "
           f"saved background model (reconstruction error {recon_err:.1e}). Per spectrum: (1) "
           f"maximise the ELBO over the weights' approximate posterior and τ (fixed here), giving "
           f"the penalty strength sigma (here {sig_rep:.2e}); (2) solve the convex MAP problem, "
           f"pinball loss plus 0.5 sigma times squared weights. No cross-validation, no TOR. "
           f"Upstream VibeSpec.fit ignores the optimizer success flag; our adapter "
           f"(scripts/vibes_baseline.py) records it and allows one logged restart."),
)

# 8
slide(
    "On the same filter the backgrounds part ways under the broad bands, in AIRSpec's zero zones, and below 1425",
    fig=f7,
    say=("Both on the same filter. Blue AIRSpec, purple VIBES. Three things. Under the broad "
         "O-H region, 3000 to 3700, the AIRSpec curve sits lower, so AIRSpec calls more of that "
         "hump absorbance. In the regions AIRSpec pins to zero, above 3700 and below about 1500, "
         "VIBES leaves positive absorbance. And below 1425 only VIBES says anything at all, "
         "including through the Teflon bands. Neither picture is automatically right: they are "
         "two models of the same unknown background."),
    notes=(f"Same selected filter, {REP}. No smoothing, normalisation or averaging. Lower panel "
           f"y axis is clipped to the 99.5th percentile of |VIBES| for this filter. Shaded amber: "
           f"AIRSpec masked windows for this filter; grey: no AIRSpec output. The corrected spectra "
           f"in the C-F region (about 1100 to 1300) are a small difference between two large "
           f"numbers (raw absorbance above 0.2): treat structure there with caution."),
)

# 9
slide(
    "Across all 253 Addis filters: AIRSpec gives more to 2800 to 3700, VIBES more everywhere else",
    fig=f8,
    say=("Is that one filter typical? Each dot is one filter's average corrected absorbance in "
         "a region; blue AIRSpec, purple VIBES, bars are medians. In the O-H and C-H regions "
         "AIRSpec assigns more. At the ends of its window, AIRSpec is at or near zero by design "
         "while VIBES is consistently positive. And the right-hand shaded regions, below 1425, "
         "are new information only VIBES provides: carbonate and C-O, the Teflon region, dust. "
         "So the methods move signal between regions systematically, which matters for any PLS "
         "model that leans on them."),
    notes=(f"Per-filter means of corrected absorbance within each region, 253 ETAD targets, "
           f"full-range VIBES vs AIRSpec. Medians (AIRSpec vs VIBES): {reg_line}. Region edges "
           f"are fixed for display; AIRSpec's own window edges vary per filter. Band labels are "
           f"usual assignments, not identifications. Positive VIBES levels below 1600 could be "
           f"real absorbance or background the blanks did not capture; this plot cannot say which."),
)

# 10
slide(
    "To fit loaded filters VIBES stretches its blank patterns far past any blank: about 4x for IMPROVE, 6x for Addis",
    fig=f9,
    say=("A diagnostic that explains a lot of the disagreement. Each dot is how hard VIBES had to "
         "lean on its blank patterns to fit a spectrum. For the blanks it learned from, and for "
         "blanks it never saw, the weights are about the size the model expects. For loaded "
         "IMPROVE filters they are about four times larger, and for Addis filters about six. "
         "Loaded filters have a stronger, steeper background than any blank, so VIBES is "
         "extrapolating the blank patterns rather than interpolating between real blanks, and "
         "Addis is furthest out. That is exactly where its assumption is weakest."),
    notes=(f"New diagnostic computed for this deck (build.py), not yet in a committed notebook. "
           f"Weights recovered by least squares from the saved full-range VIBES baselines on the "
           f"saved {C}-pattern background model (training blanks: their PCA scores). Median "
           f"weight-vector length: training {wn_tr:.1f}, held-out blanks {wn_hb:.1f}, IMPROVE "
           f"calibration samples {wn_imp:.1f}, Addis {wn_add:.1f} (90th percentiles "
           f"{wt_norm['IMPROVE samples'][1]:.1f} and {wt_norm['Addis samples'][1]:.1f}). Under "
           f"pyvibes' standard normal prior a blank-like spectrum has length about sqrt({C}) = "
           f"{np.sqrt(C):.1f}. The prior penalty (0.5 sigma times squared weights, sigma of order "
           f"1e-4) is too weak to stop this. Likely cause: particle loading adds scattering and "
           f"slope that blanks never show. IMPROVE plot shows a random 2000 of {len(CAL)} for "
           f"legibility; medians use all. Same pattern on the earlier 28-blank ETAD pilot (1425 "
           f"cut): samples 31 vs blanks about 5. Not yet checked against loading or EC error."),
)

# 11
slide(
    f"On held-out blanks VIBES leaves about {blank_ratio:.0f} times less residual than AIRSpec, even over the full range",
    fig=f10,
    say=("Now evidence. The fairest test of a background method is a blank it was not trained "
         "on, where the right answer is close to zero. Left, a typical held-out blank after each "
         "correction. Right, all held-out blanks: within AIRSpec's window VIBES is far lower, and "
         "even below 1425, through the Teflon bands, the VIBES residual stays small. This is what "
         "VIBES was designed to do and it does it well."),
    notes=(f"{len(BLK)} held-out blanks ({int((bsrc == 'etad').sum())} ETAD, "
           f"{int((bsrc == 'improve').sum())} IMPROVE), never in the background library. Median "
           f"RMS over 4000 to 1425: AIRSpec {ra_med:.6f}, full-range VIBES {np.median(rV):.6f} "
           f"({blank_ratio:.0f}x); ETAD {blank_by_src['etad'][0]:.6f} vs {blank_by_src['etad'][1]:.6f}, "
           f"IMPROVE {blank_by_src['improve'][0]:.6f} vs {blank_by_src['improve'][1]:.6f}. VIBES "
           f"below 1425: {np.median(rVlow):.6f}. The 1425-cut VIBES run was about 80x better than "
           f"AIRSpec in the window; the full-range run is less extreme because the same 30 patterns "
           f"now also have to cover 1425 to 500. Caveats: blanks are what VIBES is trained to model, "
           f"so this favours it by construction; field blanks can carry handling contamination; "
           f"a clean blank is a spectroscopy result, not an EC result."),
)

# 12
slide(
    "Known added peaks come back better with full-range VIBES at every height, including the weakest",
    fig=f11,
    say=("The worry with any background method is that it eats real signal. We added a known "
         "synthetic peak pattern to held-out blanks at three heights, corrected, and checked how "
         "much came back, scored over AIRSpec's window so both methods are judged on the same "
         "channels. Lower is better. VIBES recovers the added peaks better at all three "
         "heights, and AIRSpec's error grows with peak size. At the smallest height the gap is "
         "small, and in the earlier run with VIBES cut to 1425 AIRSpec actually won there, so "
         "weak signals are still where we should keep testing."),
    notes=(f"Recovery = correct(blank + peaks) minus correct(blank), compared with the known added "
           f"signal (raw injected minus raw parent), scored 4000 to 1425. Full-range VIBES vs "
           f"AIRSpec medians: " + "; ".join(f"{a_:g}: {injm.loc[a_, 'AIRSpec']:.5f} vs "
                                             f"{injm.loc[a_, 'VIBES']:.5f}" for a_ in amps)
           + f". The 1425-cut VIBES run lost at 0.01 (0.00059 vs 0.00043); full range wins at all "
           f"three. Each height reuses the same {n_parent} parent blanks: {n_parent} independent units, "
           f"not {len(INJ)}. One synthetic peak shape only; real standards still needed."),
)

# 13
fa, fv, fc, fu = (E[m] for m in ["AIRSpec", "VIBES-full", "VIBES-full-cut", "VIBES"])
slide(
    "Full-range VIBES edges out AIRSpec on IMPROVE EC, but transfers to Addis worst",
    fig=f12,
    say=("The question we care about: EC. Each correction gets its own PLS model refit on the "
         "same IMPROVE training sites. Left, held-out IMPROVE error: full-range VIBES is the best "
         "of the four, slightly ahead of AIRSpec; cutting VIBES to 1425 is a bit worse. Right, the "
         "same models applied to Addis and compared with HIPS absorption: the ideal is the star at "
         "slope 1, intercept 0. AIRSpec under-predicts but is closest to zero intercept; full-range "
         "VIBES overshoots with the largest negative intercept. So the extra range helps where the "
         "samples look like the training set and hurts where they do not, which fits the "
         "extrapolation picture. AIRSpec stays the EC default."),
    notes=(f"Source: meeting_followup_20260917 fits.csv and panel_stats.csv (all-IMPROVE lot 251 "
           f"family). IMPROVE: train {fa['n_train']}, test {fa['n_test']} filters, site-disjoint. "
           f"Held-out TOR RMSE (µg/filter) / R²: AIRSpec {fa['RMSE']:.2f} / {fa['R2']:.3f} (k "
           f"{fa['k']}); VIBES cut {fu['RMSE']:.2f} / {fu['R2']:.3f} (k {fu['k']}); VIBES fit full "
           f"used to 1425 {fc['RMSE']:.2f} / {fc['R2']:.3f} (k {fc['k']}); VIBES full {fv['RMSE']:.2f} "
           f"/ {fv['R2']:.3f} (k {fv['k']}). No interval on these differences. Addis (n "
           f"{fa['n_addis']}): FTIR EC vs HIPS Fabs/10 (MAC 10), µg/m³, Deming with λ = {LAM:.2f} "
           f"from measurement uncertainties: AIRSpec {fa['slope']:.2f}x {fa['icpt']:+.2f}; VIBES "
           f"cut {fu['slope']:.2f}x {fu['icpt']:+.2f}; VIBES full-cut {fc['slope']:.2f}x "
           f"{fc['icpt']:+.2f}; VIBES full {fv['slope']:.2f}x {fv['icpt']:+.2f}. HIPS/MAC is an "
           f"optical equivalent, not thermal EC; no independent Addis thermal reference exists. "
           f"Protocol differs from Option A/B; do not quote as those. The only in-box (slope 0.8 to "
           f"1.2, |intercept| < 1) configs ever found were VIBES full-cut seasonal analog sets, and "
           f"they were fragile to analog count."),
)

# 14
slide(
    "Side by side: a self-contained smoothness assumption versus a blank-library assumption",
    table=[
        ["", "AIRSpec baseline", "VIBES (pyvibes)"],
        ["Information used", "the one spectrum only", f"the spectrum plus {len(blanks)} blank filters"],
        ["Background shape", "any smooth curve (spline, df 6 and 4)", f"mean blank plus weighted blank patterns (PCA, {C} here)"],
        ["Where analyte goes", "only inside masked windows; zero elsewhere", "anywhere, as long as it sits above the background"],
        ["How fit is tuned", "fixed df; window edges found per spectrum", "penalty set per spectrum by ELBO; τ fixed 0.1"],
        ["Spectral range", "4000 to 1425 cm⁻¹ only", "4000 to 500 cm⁻¹ (all the blanks cover)"],
        ["Cost", "milliseconds per spectrum", "seconds per spectrum (full range)"],
        ["Fails when", "background is not smooth, or analyte sits in a 'zero' region", "sample background unlike any blank: loaded filters already need 4 to 6x"],
        ["Our evidence", "EC default; closest Addis intercept", "far cleaner blanks and peak recovery; best IMPROVE EC; worst Addis transfer"],
    ],
    say=("The one table to remember. AIRSpec is self-contained, fast, and well tested in our "
         "calibration, but it only models smooth backgrounds, forces zero absorbance in fixed "
         "places, and cannot go below 1425. VIBES uses the blanks, models the real shape of filter "
         "background including the Teflon bands, and covers the full spectrum. Its weak point is "
         "the mirror image: it is only as good as the blank library, and our loaded filters "
         "already sit well outside it."),
    notes=("Range: AIRSpec output is NaN outside 1425 to 4000 by construction. pyvibes has no "
           "window; the first comparison cut VIBES to 1425 for like-for-like, the full-range run fit "
           "4000 to 500. Timing is order of magnitude only (different hardware). 'Best IMPROVE EC' "
           "is a point estimate on one split without an interval."),
)

# 15
slide(
    "AIRSpec stays the EC default; full-range VIBES is worth pursuing if the blanks can cover loaded filters",
    lines=[
        "Different assumptions: AIRSpec = smooth background + fixed analyte windows, 4000 to 1425; VIBES = background looks like our blanks, 4000 to 500.",
        "VIBES wins on held-out blanks and added-peak recovery; full range gives the best IMPROVE EC but the worst Addis transfer.",
        "Loaded filters push VIBES 4 to 6x past its blank library: its key assumption is under strain, most for Addis.",
        "Next: blanks or a loading term that span loaded backgrounds, the uncapped (rank ~70) run, and weak-signal standards.",
    ],
    say=("To wrap up. These make opposite bets about where background information comes from. "
         "VIBES does the background part better on blanks and opens up the region below 1425, and "
         "with the full range it even gives the best IMPROVE EC model. But it transfers worst to "
         "Addis, and the extrapolation diagnostic suggests why: our loaded filters, Addis most of "
         "all, look nothing like the blanks VIBES learned from. So AIRSpec stays the default for "
         "EC, and the next step for VIBES is giving it a background library that actually covers "
         "loaded filters. Happy to take questions on any step."),
    notes=("Proposed next steps, not completed work. Uncapped run: rank cap 85 (one-SE about 67 to "
           "70) is 3.5% complete; Colab VMs die when the controlling session ends, so it needs a "
           "kept-alive session. Loading coverage: e.g. add a smooth scattering term to the "
           "background model, or test weight size against filter loading and EC error. Weak-signal "
           "standards: at the 0.01 injection the margin is small and flipped between the cut and "
           "full-range runs. Units: EC errors in "
           "µg/filter, Addis comparison in µg/m³."),
)

out = HERE / "vibes_airspec_methods_2026-10-01.pptx"
prs.save(out)

# ---------------------------------------------------------------- checks
chk = Presentation(out)
bad = []
for i, s in enumerate(chk.slides, 1):
    txt = " ".join(sh.text_frame.text for sh in s.shapes if sh.has_text_frame)
    for sh in s.shapes:
        if sh.has_table:
            txt += " ".join(c.text for r_ in sh.table.rows for c in r_.cells)
    notes = s.notes_slide.notes_text_frame.text
    if chr(0x2014) in txt + notes:
        bad.append(f"slide {i}: em dash")
    if not notes.startswith("SAY:") or "NOTES:" not in notes:
        bad.append(f"slide {i}: notes missing")
assert chr(0x2014) not in Path(__file__).read_text(), "em dash in builder"
assert not bad, bad
n_fig = sum(1 for s in chk.slides for sh in s.shapes if sh.shape_type == 13)
print(f"saved {out.relative_to(ROOT)}: {len(chk.slides)} slides, {n_fig} figures, notes on all, no em dashes")
