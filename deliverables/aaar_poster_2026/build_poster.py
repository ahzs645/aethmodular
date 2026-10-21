"""AAAR 2026 poster (Pasadena): one 44 x 45 in slide, same frame as the 2025 poster.

Reuses build.py for every number and figure (it is run first, so the asserts there guard the poster too), then
lays out three columns. Written after the 15 Oct practice talk and Ann's 16-19 Oct notes: start at the beginning
(ambient IMPROVE samples, not lab standards, calibrate FTIR spectra to TOR EC), define every term (FTIR, IMPROVE,
SPARTAN, HIPS, TOR, MAC, calibration set ...), never say "cohort" without saying of what, point at the axes of each
crossplot, and answer "why IMPROVE?" on the poster itself.

20 Oct (Ann, 19 Oct meeting + email): title, affiliations and two headers as Ahmad edited them by hand; MAIA logo in
place of JPL; websites for SPARTAN and IMPROVE; column 1 intro and methods, column 2 the problem, column 3 what we
have tried (baseline correction, the calibration explorer, season-specific calibrations); no IMPROVE line on the
concentration bars (flow rates differ), no per-filter bar chart (the histogram says it), no site-ranking plot; say
that the SPARTAN network as a whole is predicted reasonably but Addis is not; no "held-out" anywhere. Every poster
figure is redrawn here at poster scale from build.py's data, so build.py's figures stay as the talk had them.

Run with anaconda python from the repo root:
    ~/anaconda3/bin/python deliverables/aaar_poster_2026/build_poster.py
"""

import json
import math
import re
import runpy
from pathlib import Path

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
g = runpy.run_path(str(HERE / "build.py"), run_name="aaar_build")   # figures + validated numbers
ROOT = g["ROOT"]
PRINT = HERE / "figures" / "print"

S_ADDIS, S_POOL, S_AN_REF, S_AN_HIPS, S_AN = (g[k] for k in ["S_ADDIS", "S_POOL", "S_AN_REF", "S_AN_HIPS", "S_AN"])

# Provenance: the poster's "current calibration" values (Figures 4, 7 and 9) are SPARTAN's own FTIR EC, not our
# recreation (docs/spartan-ftir-ec-recreation.md). They come from the in-house deployed-EC table; check them against
# the SPARTAN public release (v3.1.4, EC PM2.5, method 218) filter by filter.
_dep = g["ad"].assign(fid=g["ad"].ExternalFilterId.astype(str).str.rsplit("-", n=1).str[0])
_pub = g["etad"][["Filter_ID", "EC PM2.5"]].dropna()
_m = _dep.merge(_pub, left_on="fid", right_on="Filter_ID", how="inner")
_d = (_m.ftir_ec_ugm3 - _m["EC PM2.5"]).abs()
assert len(_m) >= len(_dep) - 1 and _d.median() < 0.01, (len(_m), len(_dep), _d.median())
print(f"SPARTAN provenance: {len(_m)} of {len(_dep)} poster filters are in the public release; "
      f"|deployed - public| median {_d.median():.4f}, max {_d.max():.3f} µg/m³")
sea = g["sea"]


def fit_txt(s):
    return f"slope {s.deming_slope:.2f}, intercept {s.deming_intercept:+.2f} µg/m³, R² {s.R2:.2f}"


# ---------------------------------------------------------------- locator map (IMPROVE + SPARTAN sites)
imp = pd.read_pickle(ROOT / "research/spartan/.cache/improve_chemistry_clean_cache.pkl")
imp_xy = (imp[imp.SiteCode.isin(g["imp_sites"].index)].groupby("SiteCode")[["Latitude", "Longitude"]].first())
sp_xy = {}
for f in sorted((ROOT / "data/spartan/raw/FilterBased/ChemSpecPM25").glob("*.csv")):
    d = pd.read_csv(f, comment="#", usecols=["Site_Code", "Latitude", "Longitude"], nrows=1)
    sp_xy[d.Site_Code.iloc[0]] = (d.Latitude.iloc[0], d.Longitude.iloc[0])
sp_xy = pd.DataFrame(sp_xy, index=["Latitude", "Longitude"]).T.loc[lambda t: t.index.isin(g["sp_sites"].index)]
assert len(imp_xy) == len(g["imp_sites"]) and len(sp_xy) == len(g["sp_sites"]), (len(imp_xy), len(sp_xy))

fig = plt.figure(figsize=(12, 6.0))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=-20))
ax.set_extent((-165, 175, -45, 68), crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="#ECEAE4", edgecolor="none")
ax.add_feature(cfeature.BORDERS, edgecolor="white", lw=0.4)
ax.spines["geo"].set_edgecolor("#C9C6BE")
pc = ccrs.PlateCarree()
ax.scatter(imp_xy.Longitude, imp_xy.Latitude, s=14, color=g["GREY"], transform=pc, lw=0, alpha=.8,
           label=f"IMPROVE sites ({len(imp_xy)}): FTIR + TOR, calibration")
oth = sp_xy.drop(index="ETAD")
ax.scatter(oth.Longitude, oth.Latitude, s=90, marker="*", color=g["BLUE"], transform=pc, lw=0,
           label=f"SPARTAN sites ({len(sp_xy)}): FTIR + HIPS")
a = sp_xy.loc["ETAD"]
ax.scatter([a.Longitude], [a.Latitude], s=520, marker="*", color=g["AMBER"], edgecolor=g["INK"], lw=0.8,
           transform=pc, zorder=5, label="Addis Ababa, Ethiopia (SPARTAN)")
ax.annotate("Addis Ababa", xy=(a.Longitude, a.Latitude), xycoords=pc._as_mpl_transform(ax),
            xytext=(a.Longitude + 22, a.Latitude - 20), textcoords=pc._as_mpl_transform(ax), fontsize=15,
            fontweight="bold", color=g["INK"], arrowprops=dict(arrowstyle="-", color=g["INK"], lw=1))
ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.2), ncol=2, fontsize=12.5, frameon=False, markerscale=1.2)
fig.savefig(PRINT / "f0_network_map.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- poster figures (poster-scale fonts)
SEASONS, SEASON_LABEL, SEASON_COLOR = g["SEASONS"], g["SEASON_LABEL"], g["SEASON_COLOR"]
GREY_, BLUE_, ACCENT_, AMBER_, INK_ = g["GREY"], g["BLUE"], g["ACCENT"], g["AMBER"], g["INK"]
MAC, pk, LAM = g["MAC"], g["pk"], g["LAM_ADDIS"]
POSTER_RC = {"font.size": 15, "axes.titlesize": 16, "legend.fontsize": 13.5, "axes.labelsize": 15}
X_HIPS = "HIPS EC = Fabs / MAC 10 (µg/m³)"
TICK = {"Dry (Oct-Feb)": "Dry, Bega\n(Oct-Feb)", "Belg (Mar-May)": "Belg\n(Mar-May)",
        "Kiremt (Jun-Sep)": "Kiremt\n(Jun-Sep)"}


def psave(fig, name):
    fig.savefig(PRINT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fit_box(ax, rows, loc=(0.03, 0.97)):
    ax.text(*loc, "\n".join(rows), transform=ax.transAxes, va="top", ha="left", fontsize=13.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D8D5CE"))


def fline(ax, sl, ic, lim, color=INK_, lw=2.2, one=True):
    xs = np.array([0, lim])
    if one:    # once per panel: the 1:1 line and a zero line (19 Oct: show where y = 0 is)
        ax.plot(xs, xs, ls=":", color=GREY_, lw=1.5)
        ax.axhline(0, color="#B8B5AE", lw=1.1, zorder=0)
    ax.plot(xs, ic + sl * xs, color=color, lw=lw)


def dem(x, y):
    sl, ic = pk.deming(np.asarray(x, float), np.asarray(y, float), LAM)
    return float(sl[0]), float(ic[0]), float(pk.r2(np.asarray(x, float), np.asarray(y, float))[0]), len(x)


def ftxt(s_):
    return f"{s_.deming_slope:.2f}x {s_.deming_intercept:+.2f}, R² {s_.R2:.2f}, n {int(s_.n):,}"


with plt.rc_context(POSTER_RC):
    # P2: Addis PM2.5 and EC by season, in the air. No IMPROVE line: IMPROVE samples ~4x more air per filter, so a
    # concentration comparison is not like for like (Ann, 19 Oct); the EC share of PM2.5 is the fair comparison.
    t = sea
    fig, ax = plt.subplots(figsize=(9, 5.0))
    xs = np.arange(3)
    ax.bar(xs - 0.2, t.pm, width=.38, color="#E4E1DA", label="PM2.5 mass", zorder=2)
    ax.bar(xs + 0.2, t.ec, width=.38, color=[SEASON_COLOR[k] for k in SEASONS], zorder=2,
           label="EC (from HIPS)", yerr=[t.ec - t.ec25, t.ec75 - t.ec], capsize=5, ecolor=INK_,
           error_kw=dict(lw=1.2))
    for i in range(3):
        ax.text(i + 0.2, t.ec75.iloc[i] + .4, f"{t.ec.iloc[i]:.1f}", ha="center", va="bottom", fontsize=17,
                fontweight="bold")
        ax.text(i - 0.2, t.pm.iloc[i] + .4, f"{t.pm.iloc[i]:.0f}", ha="center", va="bottom", fontsize=15,
                color="#6E6B64")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, t.pm.max() * 1.3)
    ax.set_xticks(xs, [f"{TICK[k]}\nn {int(sea.n.iloc[i])}" for i, k in enumerate(SEASONS)])
    ax.set_ylabel("Median in the air (µg/m³)")
    ax.legend(loc="upper left", fontsize=14)
    psave(fig, "fp_addis_air.png")

    # P3: the problem. Left IMPROVE (FTIR vs TOR); right SPARTAN, every other site in grey and Addis by season.
    pool, ad = g["pool"], g["ad"]
    sp = g["sp_all"]
    oth = sp[sp.Site_Code.isin(g["sp_sites"].index) & sp.Site_Code.ne("ETAD")].dropna(subset=["EC PM2.5", "BC PM2.5"])
    NET = dem(oth["BC PM2.5"], oth["EC PM2.5"])
    assert abs(NET[0] - 1.02) < 0.01 and abs(NET[1] + 0.21) < 0.01, NET     # checked 20 Oct
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw=dict(wspace=.28))
    ax = axes[0]; lim = 1.2
    ax.scatter(pool.tor_ec_ugm3, pool.ftir_ec_ugm3, s=10, color=GREY_, alpha=.45, lw=0)
    fline(ax, S_POOL.deming_slope, S_POOL.deming_intercept, lim)
    ax.set_xlim(0, lim); ax.set_ylim(-0.1, lim)
    ax.set_xlabel("TOR EC (µg/m³)"); ax.set_ylabel("FTIR EC (µg/m³)")
    ax.set_title("IMPROVE", loc="left", fontweight="bold")
    fit_box(ax, [f"{S_POOL.deming_slope:.2f}x {S_POOL.deming_intercept:+.2f}", f"R² {S_POOL.R2:.2f}, n {int(S_POOL.n):,}"])
    ax = axes[1]; lim = 12
    ax.scatter(oth["BC PM2.5"], oth["EC PM2.5"], s=8, color=GREY_, alpha=.3, lw=0, label="other SPARTAN sites")
    for k in SEASONS:
        d = ad[ad.season == k]
        ax.scatter(d.Fabs / MAC, d.ftir_ec_ugm3, s=30, color=SEASON_COLOR[k], alpha=.9, edgecolor="white", lw=.4,
                   label=f"Addis, {SEASON_LABEL[k].split(' (')[0]}")
    fline(ax, NET[0], NET[1], lim, color=GREY_, lw=2.4)
    fline(ax, S_ADDIS.deming_slope, S_ADDIS.deming_intercept, lim, one=False)
    ax.set_xlim(0, lim); ax.set_ylim(-1, lim)
    ax.set_xlabel(X_HIPS); ax.set_ylabel("FTIR EC (µg/m³)")
    ax.set_title("SPARTAN", loc="left", fontweight="bold")
    fit_box(ax, [f"Addis: {S_ADDIS.deming_slope:.2f}x {S_ADDIS.deming_intercept:+.2f}, R² {S_ADDIS.R2:.2f}",
                 f"other sites: {NET[0]:.2f}x {NET[1]:+.2f}, R² {NET[2]:.2f}"])
    ax.legend(loc="lower right", fontsize=11.5, markerscale=1.3, handletextpad=.2)
    n_out = int(((oth["BC PM2.5"] > lim) | (oth["EC PM2.5"] > lim)).sum())
    psave(fig, "fp_problem.png")
    SEAS_CUR = {k: g["addis_fit"](ad[ad.season == k]) for k in SEASONS}
    _et = g["etad"].dropna(subset=["EC PM2.5", "BC PM2.5"])
    ALL_ADDIS = dem(_et["BC PM2.5"], _et["EC PM2.5"])
    assert ALL_ADDIS[3] == 246 and abs(ALL_ADDIS[0] - 2.27) < 0.01 and abs(ALL_ADDIS[1] + 6.09) < 0.01, ALL_ADDIS

    # P4: IMPROVE filters whose spectra most resemble Addis
    an = g["an"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw=dict(wspace=.28))
    for ax, ycol, s_, ttl, ylab in [(axes[0], "tor_ec_ugm3", S_AN_REF, "TOR and HIPS agree", "TOR EC (µg/m³)"),
                                    (axes[1], "ftir_ec_ugm3", S_AN_HIPS, "FTIR is poorly predicted",
                                     "FTIR EC (µg/m³)")]:
        lim = 1.2
        ax.scatter(pool.fabs_Mm1 / MAC, pool[ycol], s=9, color=GREY_, alpha=.28, lw=0,
                   label=f"all IMPROVE ({len(pool):,})")
        ax.scatter(an.fabs_Mm1 / MAC, an[ycol], s=34, color=ACCENT_, alpha=.9, edgecolor="white", lw=.4,
                   label=f"most like Addis ({len(an)})")
        fline(ax, s_.deming_slope, s_.deming_intercept, lim, color=ACCENT_)
        ax.set_xlim(0, lim); ax.set_ylim(-0.1, lim)
        ax.set_xlabel(X_HIPS); ax.set_ylabel(ylab)
        ax.set_title(ttl, loc="left", fontweight="bold")
        fit_box(ax, [f"{s_.deming_slope:.2f}x {s_.deming_intercept:+.2f}", f"R² {s_.R2:.2f}, n {int(s_.n)}"])
    axes[0].legend(loc="lower right", fontsize=12.5, markerscale=1.3)
    psave(fig, "fp_analogs.png")

    # P5: EC per filter (each filter's own volume: flow x sampling time), histograms normalised to 100%
    imp_mass, ad_mass, p99 = g["imp_mass"], g["ad_mass"], g["p99"]
    bins = np.logspace(np.log10(min(imp_mass.min(), ad_mass.min()) * .9),
                       np.log10(max(imp_mass.max(), ad_mass.max()) * 1.1), 46)
    fig, ax = plt.subplots(figsize=(10, 3.9))
    for vals, color, lab in [(imp_mass, GREY_, f"IMPROVE calibration filters ({len(imp_mass):,})"),
                             (ad_mass, AMBER_, f"Addis Ababa ({len(ad_mass)})")]:
        w = np.full(len(vals), 100 / len(vals))
        ax.hist(vals, bins=bins, weights=w, color=color, alpha=.35, lw=0)
        ax.hist(vals, bins=bins, weights=w, histtype="step", color=color, lw=2.2, label=lab)
    top = ax.get_ylim()[1] * 1.12
    ax.set_ylim(0, top)
    for v, color, lab, side in [(imp_mass.median(), GREY_, "IMPROVE\nmedian", "right"),
                                (ad_mass.median(), AMBER_, "Addis\nmedian", "left")]:
        ax.axvline(v, color=color, ls="--", lw=1.6)
        ax.text(v * (1.07 if side == "left" else 1 / 1.07), top * .97, f"{lab}\n{v:.0f} µg", va="top", ha=side,
                fontsize=13.5)
    ax.axvline(p99, color=INK_, ls=":", lw=1.6)
    ax.text(p99 / 1.07, top * .58, f"IMPROVE\n99th pct.\n{p99:.0f} µg", va="top", ha="right", fontsize=13.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("EC on one filter (µg, log scale)")
    ax.set_ylabel("Percent of filters")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=13.5)
    psave(fig, "fp_loading.png")

    # P6: baseline correction of the spectra, before / after (same 189 Addis filters)
    base, S_BASE = g["base"], g["S_BASE"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True, gridspec_kw=dict(wspace=.08))
    lim = 12
    for ax, d, s_, ttl in [(axes[0], ad, S_ADDIS, "Current calibration"),
                           (axes[1], base, S_BASE, "Baseline-corrected spectra")]:
        for k in SEASONS:
            q = d[d.season == k]
            ax.scatter(q.Fabs / MAC, q.ftir_ec_ugm3, s=26, color=SEASON_COLOR[k], alpha=.88, edgecolor="white",
                       lw=.4, label=SEASON_LABEL[k].split(" (")[0])
        fline(ax, s_.deming_slope, s_.deming_intercept, lim)
        ax.set_xlim(0, lim); ax.set_ylim(-1, lim)
        ax.set_xlabel(X_HIPS)
        ax.set_title(ttl, loc="left", fontweight="bold")
        fit_box(ax, [f"{s_.deming_slope:.2f}x {s_.deming_intercept:+.2f}", f"R² {s_.R2:.2f}, n {int(s_.n)}"])
    axes[0].set_ylabel("FTIR EC (µg/m³)")
    axes[1].legend(loc="lower right", fontsize=13)
    psave(fig, "fp_baseline.png")

    # P7: the calibration explorer (spline baseline only), as in build.py f7
    order, inter, masks, n_spec = g["order"], g["inter"], g["masks"], g["n_spec"]
    ROWS = g["ROWS"]
    n_rows = sum(len(r) for _, r in ROWS) + len(ROWS)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.6, 5.85), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 2.6], hspace=.06))
    ax0.plot(np.arange(n_spec), np.sort(inter), color=BLUE_, lw=2.4)
    ax0.axhline(0, color=GREY_, ls=":", lw=1.4)
    ax0.axhline(S_ADDIS.deming_intercept, color=ACCENT_, ls="--", lw=1.4)
    ax0.text(n_spec * .99, S_ADDIS.deming_intercept + .3, f"current calibration {S_ADDIS.deming_intercept:.1f}",
             ha="right", fontsize=13, color=ACCENT_)
    ax0.set_ylabel("Addis\nintercept\n(µg/m³)")
    ax0.set_ylim(S_ADDIS.deming_intercept - .9, 1.2)
    y_, labels, ticks = n_rows, [], []
    for head, rows in ROWS:
        y_ -= 1
        labels.append(head); ticks.append(y_)
        for lab, key in rows:
            y_ -= 1
            on = np.flatnonzero(masks[key][order])
            ax1.vlines(on, y_ - .38, y_ + .38, color=INK_, lw=.35, alpha=.55)
            labels.append("   " + lab); ticks.append(y_)
    # rows are right-aligned tick labels; the group headers are drawn separately, left-aligned with the longest row
    heads = [(lab, tk) for lab, tk in zip(labels, ticks) if not lab.startswith(" ")]
    ax1.set_yticks(ticks, [lab.strip() if lab.startswith(" ") else "" for lab in labels], fontsize=13.5)
    ax1.set_ylim(-.6, n_rows - .4); ax1.set_xlim(-5, n_spec + 5)
    fig.canvas.draw()
    x_left = min(t_.get_window_extent().x0 for t_ in ax1.get_yticklabels() if t_.get_text())
    x_left = ax1.transAxes.inverted().transform((x_left, 0))[0]
    for lab, tk in heads:
        ax1.text(x_left, tk, lab, transform=ax1.get_yaxis_transform(), ha="left", va="center", fontsize=13.5,
                 fontweight="bold")
    ax1.tick_params(axis="y", length=0); ax1.spines["left"].set_visible(False)
    ax1.set_xlabel(f"{n_spec:,} calibrations, sorted by Addis intercept")
    psave(fig, "fp_explorer.png")
    spec = json.loads(g["SPEC"].read_text())
    sl_spec = np.array(spec["slope"])[g["keep"]]
    near0 = np.abs(inter) <= 0.5
    n_both = int((near0 & (sl_spec >= 0.8) & (sl_spec <= 1.25)).sum())
    assert int(near0.sum()) == g["n_near0"] == 254 and n_both == 20, (near0.sum(), n_both)   # checked 20 Oct

    # P8: season-specific calibrations
    SEAS, allp, SEAS_FIT, COHORT = g["SEAS"], g["allp"], g["SEAS_FIT"], g["COHORT"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.65), sharey=True, gridspec_kw=dict(wspace=.08))
    lim = 12
    for ax, k in zip(axes, SEASONS):
        b0 = base[base.season == k]
        sv = allp[(allp.fit_id == SEAS_FIT[k]) & (allp.season == k) & allp.MediaId.isin(COHORT)].dropna(
            subset=["ftir_ec_ugm3"])
        s0, s1 = SEAS[k]
        ax.scatter(b0.Fabs / MAC, b0.ftir_ec_ugm3, s=18, color=GREY_, alpha=.55, lw=0)
        ax.scatter(sv.Fabs / MAC, sv.ftir_ec_ugm3, s=26, color=SEASON_COLOR[k], alpha=.9, edgecolor="white", lw=.4)
        sc = SEAS_CUR[k]    # current SPARTAN calibration, no baseline correction (line only, to keep the panel readable)
        xs_ = np.array([0, lim])
        ax.plot(xs_, sc.deming_intercept + sc.deming_slope * xs_, color=INK_, lw=2, ls="--")
        fline(ax, s0.deming_slope, s0.deming_intercept, lim, color=GREY_)
        fline(ax, s1.deming_slope, s1.deming_intercept, lim, color=SEASON_COLOR[k], one=False)
        ax.set_xlim(0, lim); ax.set_ylim(-1, lim)
        ax.set_title(SEASON_LABEL[k].split(" (")[0], loc="left", fontweight="bold", color=SEASON_COLOR[k])
        ax.set_xlabel(X_HIPS, fontsize=13)
        # legend per panel: current calibration, baseline-corrected (Step 1), season-specific (Step 3)
        ax.legend(handles=[Line2D([], [], color=INK_, lw=2, ls="--",
                                  label=f"current: {sc.deming_slope:.2f}x {sc.deming_intercept:+.1f}"),
                           Line2D([], [], color=GREY_, lw=2.2, marker="o", ms=6, mew=0,
                                  label=f"baseline: {s0.deming_slope:.2f}x {s0.deming_intercept:+.1f}"),
                           Line2D([], [], color=SEASON_COLOR[k], lw=2.2, marker="o", ms=6, mew=0,
                                  label=f"season: {s1.deming_slope:.2f}x {s1.deming_intercept:+.1f}")],
                  loc="upper left", fontsize=12.5, frameon=True, framealpha=1, edgecolor="#D8D5CE",
                  handlelength=1.5, borderpad=.35, labelspacing=.3)
    axes[0].set_ylabel("FTIR EC (µg/m³)")
    psave(fig, "fp_seasons.png")

S_BASE = g["S_BASE"]
K0, K1 = g["SEAS"]["Kiremt (Jun-Sep)"]
cur_sl = [SEAS_CUR[k].deming_slope for k in SEASONS]

# ---------------------------------------------------------------- poster frame
W, H = 44.0, 45.0
NAVY = RGBColor(0x1B, 0x36, 0x5D)
INK = RGBColor(0x22, 0x25, 0x2A)
SOFT = RGBColor(0x55, 0x58, 0x5E)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
BACK = RGBColor(0xE4, 0xE6, 0xEA)
TINT = RGBColor(0xFB, 0xF3, 0xE4)      # amber tint for the call-out box
AMBER_RGB = RGBColor(0xC4, 0x94, 0x42)
FONT = "Calibri"
BODY, CAP, HEAD = 28, 23, 38

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
s = prs.slides.add_slide(prs.slide_layouts[6])


def rect(x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = fill
    if line is None:
        r.line.fill.background()
    else:
        r.line.color.rgb = line; r.line.width = Pt(2)
    r.shadow.inherit = False
    return r


def runs(par, text, size, color=INK, bold=False, italic=False):
    """**bold** and *italic* markup inside text."""
    for chunk in re.split(r"(\*\*.+?\*\*|\*.+?\*)", text):
        if not chunk:
            continue
        b, it = chunk.startswith("**"), chunk.startswith("*") and not chunk.startswith("**")
        r = par.add_run(); r.text = chunk.strip("*")
        f = r.font; f.name = FONT; f.size = Pt(size); f.color.rgb = color
        f.bold = bold or b; f.italic = italic or it


def text_h(paras, w, size, bullet=False, gap=0.35):
    """Estimated height (in) of wrapped paragraphs: Calibri averages ~0.5 em per character."""
    cpl = (w - 0.2 - (0.45 if bullet else 0)) * 72 / (size * 0.46)
    lines = sum(max(1, math.ceil(len(p.replace("*", "")) / cpl)) for p in paras)
    return lines * size * 1.2 / 72 + len(paras) * size * gap / 72 + 0.15


def text(x, y, w, paras, size=BODY, bullet=False, color=INK, align=PP_ALIGN.LEFT, italic=False, gap=0.35, h=None):
    h = h or text_h(paras, w, size, bullet, gap)
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.1); tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, p_ in enumerate(paras):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.space_after = Pt(size * gap)
        if bullet:
            pPr = par._p.get_or_add_pPr()
            pPr.set("marL", str(Inches(0.42))); pPr.set("indent", str(-Inches(0.42)))
            bu = pPr.makeelement(qn("a:buChar"), {"char": "•"}); pPr.append(bu)
        runs(par, p_, size, color, italic=italic)
    return h


def header(x, y, w, title):
    r = rect(x, y, w, 0.85, NAVY)
    tf = r.text_frame; tf.margin_left = Inches(0.2); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    runs(p, title, HEAD, PAPER, bold=True)
    return 0.85


def picture(x, y, w, path, center_w=None):
    im = Image.open(path)
    h = w * im.height / im.width
    xo = x + ((center_w - w) / 2 if center_w else 0)
    s.shapes.add_picture(str(path), Inches(xo), Inches(y), Inches(w), Inches(h))
    return h


rect(0, 0, W, H, BACK)
rect(0.85, 0.5, W - 1.7, 6.1, PAPER)

# title block (title and affiliations as Ahmad edited them on 19 Oct with Ann)
TITLE = ("Comparison of FTIR elemental carbon (EC) measurements in Addis Ababa, Ethiopia with rural U.S. IMPROVE EC "
         "measurements")
text(11.4, 0.75, 21.2, [TITLE], size=54, align=PP_ALIGN.CENTER, gap=0, h=1.9)
for p in s.shapes[-1].text_frame.paragraphs:
    for r in p.runs:
        r.font.bold = True; r.font.color.rgb = NAVY
text(11.4, 2.95, 21.2, ["Ahmad Jalil¹˒⁴, Sina Hasheminassab², Kyan Shlipak⁵˒², Tesfaye Mamo Mekonnen³, Araya Asfaw³, "
                        "Hossein Kazemian¹, Ann M. Dillner⁴"], size=32, align=PP_ALIGN.CENTER, gap=0, h=1.2)
text(11.4, 4.25, 21.2, ["¹ MATTER lab, University of Northern British Columbia, Prince George, BC, Canada · "
                        "² NASA Jet Propulsion Laboratory, Pasadena, CA, USA · ³ Addis Ababa University, Addis Ababa, "
                        "Ethiopia · ⁴ Air Quality Research Center, University of California Davis, Davis, CA, USA · "
                        "⁵ Northwestern University, Evanston, IL, USA"], size=21, align=PP_ALIGN.CENTER, gap=0,
     color=SOFT, h=1.3)

# logos, placed as on the 2025 poster
# Logos live in the repo (figures/logos/): UNBC, UC Davis AQRC and MAIA were taken from the 2025 AAAR poster; the
# SPARTAN logo is the cyan compact version from the vector reconstruction package (spartan-compact-cyan.png, 2,400 px,
# transparent padding trimmed).
LOGO_DIR = HERE / "figures" / "logos"
for _name in ("unbc", "ucdavis_aqrc", "maia", "spartan_cyan"):
    assert (LOGO_DIR / f"{_name}.png").exists(), f"missing logo figures/logos/{_name}.png"
picture(1.5, 1.2, 9.4, LOGO_DIR / "unbc.png")
picture(1.5, 3.6, 9.4, LOGO_DIR / "ucdavis_aqrc.png")
picture(36.45, 0.8, 2.9, LOGO_DIR / "maia.png")       # NASA MAIA (funds the work), in place of JPL
picture(34.2, 3.95, 7.4, LOGO_DIR / "spartan_cyan.png")

# ---------------------------------------------------------------- columns
TOP, BOT = 7.05, 44.3
CW, GAP, X0 = 13.7, 0.6, 0.85
XS = [X0 + i * (CW + GAP) for i in range(3)]
for x in XS:
    rect(x, TOP, CW, BOT - TOP, PAPER)
PAD = 0.3
IW = CW - 2 * PAD
fig_no = [0]


def caption(x, y, body):
    fig_no[0] += 1
    return text(x, y, IW, [f"**Figure {fig_no[0]}.** {body}"], size=CAP, color=SOFT, gap=0.2)


col_end = {}

# ---- column 1: introduction and methods
x, y = XS[0], TOP
y += header(x, y, CW, "Why elemental carbon in Addis Ababa?")
y += 0.2
y += text(x + PAD, y, IW, [
    "**Elemental carbon (EC)** is the black, soot-like carbon from burning. It warms the climate and harms health.",
    "**SPARTAN** (Surface PARTiculate mAtter Network, www.spartan-network.org) samples particles onto filters in "
    "cities around the world, mostly in the Global South. It measures EC by **FTIR** (below).",
    "**Addis Ababa**, capital of Ethiopia (about 2,400 m elevation, about 5 million people), has some of the highest "
    "EC in SPARTAN. Households cook and heat with wood and charcoal.",
    f"We use {len(g['COHORT'])} Addis filters (Dec 2022 to Sep 2024) across three seasons: **Dry** (Bega, Oct to "
    "Feb), **Belg** short rains (Mar to May) and **Kiremt** long rains (Jun to Sep). All "
    f"{ALL_ADDIS[3]} Addis filters in the public SPARTAN release give the same result."], bullet=True)
y += picture(x + PAD + 1.3, y, IW - 2.6, PRINT / "f0_network_map.png") + 0.05
y += caption(x + PAD, y, "Where the filters come from. Grey: IMPROVE sites (US), which provide the calibration. "
                         "Blue stars: SPARTAN sites, where the calibration is applied.") + 0.25

y += header(x, y, CW, "How FTIR measures EC: a calibration")
y += 0.2
y += text(x + PAD, y, IW, [
    "**FTIR** (Fourier-transform infrared spectroscopy) passes infrared light through a Teflon filter and records how "
    "much is absorbed at each wavenumber. That curve is the filter's **spectrum**. FTIR is fast, cheap and does not "
    "destroy the filter, but a spectrum does not give EC directly: it has to be **calibrated** against a reference.",
    "Most methods are calibrated with laboratory standards. FTIR EC is calibrated with **ambient samples**: real "
    "IMPROVE filters on which the same air was collected on a Teflon filter (FTIR spectrum) and on a quartz filter "
    "(EC measured by **TOR**, the reference method)."])

# flow diagram: IMPROVE (Teflon -> spectrum, quartz -> TOR EC) -> PLS calibration, then down to SPARTAN
y += 0.1
BX_H = 1.55


def box(bx, by, bw, bh, label, fill, color=INK, size=21):
    r = rect(bx, by, bw, bh, fill, line=NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    r.adjustments[0] = 0.12
    tf = r.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08)
    for i, line in enumerate(label.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        runs(p, line, size, color)
    return r


def arrow(ax_, ay, aw, ah, shape=MSO_SHAPE.RIGHT_ARROW):
    r = rect(ax_, ay, aw, ah, NAVY, shape=shape)
    return r


LIGHT = RGBColor(0xEE, 0xF2, 0xF7)
bx = x + PAD
b1w, aw_, b2w = 3.65, 0.55, 3.45
box(bx, y, b1w, 2 * BX_H + 0.25, "**IMPROVE**\nambient samples\n(US national\nparks, rural)", LIGHT)
arrow(bx + b1w + 0.05, y + BX_H / 2 - 0.22, aw_, 0.45)
arrow(bx + b1w + 0.05, y + BX_H + 0.25 + BX_H / 2 - 0.22, aw_, 0.45)
c2 = bx + b1w + aw_ + 0.1
box(c2, y, b2w, BX_H, "Teflon filter\n→ **FTIR spectrum**", PAPER)
box(c2, y + BX_H + 0.25, b2w, BX_H, "Quartz filter\n→ **TOR EC**", PAPER)
arrow(c2 + b2w + 0.05, y + BX_H / 2 + 0.05, aw_, 0.45)
arrow(c2 + b2w + 0.05, y + BX_H + 0.25 + BX_H / 2 - 0.5, aw_, 0.45)
c3 = c2 + b2w + aw_ + 0.1
pls_w = bx + IW - c3
box(c3, y, pls_w, 2 * BX_H + 0.25, "**PLS calibration**\nlearns which\nspectral features\npredict TOR EC", LIGHT)
y += 2 * BX_H + 0.25
arrow(c3 + pls_w / 2 - 0.3, y + 0.05, 0.6, 0.5, MSO_SHAPE.DOWN_ARROW)    # from the PLS box (Ann, 19 Oct)
y += 0.6
box(bx, y, IW, 1.25, "Applied to **SPARTAN** spectra (Addis Ababa): FTIR EC, checked against **HIPS** light "
                      "absorption (no TOR at Addis)", TINT)
y += 1.25 + 0.15
y += caption(x + PAD, y, "The calibration behind every SPARTAN FTIR EC value.") + 0.2
# why IMPROVE call-out (the question every audience asks; Ann: answer it on the poster)
why = ["**Why calibrate with IMPROVE filters?** Because it is the only data set that exists: no other library of "
       "matched Teflon (FTIR) and quartz (TOR) ambient filters exists anywhere, from Addis Ababa or any other city. "
       "CSN (the US urban network) filters are bigger and thicker and sample at a lower flow, so FTIR calibrations "
       "built on them predict less well."]
hh = text_h(why, IW - 0.3, 25) + 0.15
rect(x + PAD, y, IW, hh, TINT, line=AMBER_RGB)
text(x + PAD + 0.15, y + 0.08, IW - 0.3, why, size=25)
y += hh + 0.25

y += header(x, y, CW, "Key terms")
y += 0.15
terms = [
    "**IMPROVE**: Interagency Monitoring of PROtected Visual Environments, the US network sampling in national parks "
    "and rural areas (vista.cira.colostate.edu/Improve).",
    "**TOR**: thermal-optical reflectance. Heats a punch of a quartz filter and measures the carbon released; the "
    "standard reference for EC.",
    "**HIPS**: hybrid integrating plate/sphere. Measures how much red light (633 nm) a Teflon filter absorbs, called "
    "**Fabs**.",
    "**MAC**: mass absorption cross-section, the light absorbed per gram of EC (m²/g). EC = Fabs / MAC. We use "
    "MAC = 10 m²/g, a common default; the true value varies with sources. Changing MAC changes the slope in "
    "Figure 4, not the intercept, so MAC alone cannot fix Addis.",
    "**Calibration set**: the IMPROVE filters (with TOR EC) that one calibration is built from.",
    "**PLS**: partial least squares regression, which turns a whole spectrum into one number (EC).",
    "**Baseline correction of the spectra**: removing the smooth background from the Teflon filter and light "
    "scattering, so only particle absorption remains.",
    "**Deming regression**: a straight-line fit that allows for error in both measurements. Agreement means "
    "slope 1, intercept 0.",
]
y += text(x + PAD, y, IW, terms, size=24, gap=0.22) + 0.1
# acknowledgments pinned to the column foot. text_h over-reads long 24 pt lists by ~0.7 in (checked on the 20 Oct
# PowerPoint render: the key terms end about 1.3 in above their estimate), hence the 1.0 in allowance.
ack = ["**Acknowledgments:** supported by the NASA MAIA mission. Thanks to the SPARTAN team and the UC Davis FTIR group."]
h_ack = text_h(ack, IW, 21)
text(x + PAD, BOT - h_ack - 0.05, IW, ack, size=21, color=SOFT)
assert y <= BOT - h_ack + 1.0, f"key terms run into the acknowledgments ({y:.1f} in)"
col_end[0] = BOT - 0.05

# ---- column 2: the problem
x, y = XS[1], TOP
y += header(x, y, CW, "Addis Ababa EC is high, and so is its share of PM2.5")
y += 0.2
ec_lo, ec_hi = sea.ec.min(), sea.ec.max()
y += picture(x + PAD + 2.0, y, IW - 4.0, PRINT / "fp_addis_air.png") + 0.05
y += caption(x + PAD, y, f"Addis medians by season (bars on EC: interquartile range). EC is {ec_lo:.1f} to "
                         f"{ec_hi:.1f} µg/m³ and 19 to 25% of PM2.5 in every season, against about "
                         f"{100 * g['imp_ec_median'] / g['imp_pm_median']:.0f}% for a typical IMPROVE filter.") + 0.25

y += header(x, y, CW, "The problem: FTIR EC is poorly predicted but well correlated")
y += 0.2
y += picture(x + PAD, y, IW, PRINT / "fp_problem.png") + 0.05
y += caption(x + PAD, y,
             f"Each point is one filter; dotted line: 1:1. **IMPROVE:** {int(S_POOL.n):,} filters, FTIR against TOR "
             f"EC: slope near 1, intercept near 0. **SPARTAN:** FTIR against HIPS EC. The other SPARTAN sites together "
             f"are predicted reasonably (grey line), and Addis sits inside that cloud, but Addis alone is "
             f"well correlated with the wrong slope and intercept ({S_ADDIS.deming_slope:.1f}x "
             f"{S_ADDIS.deming_intercept:.1f}). Every season is off (slopes {min(cur_sl):.1f} to "
             f"{max(cur_sl):.1f}).") + 0.15
y += 0.1

y += header(x, y, CW, "IMPROVE filters that look like Addis fail too")
y += 0.2
y += text(x + PAD, y, IW, [
    f"From the IMPROVE filters we picked the {int(S_AN.n)} whose **spectra most resemble the Addis spectra**. Many are "
    "smoke-affected."], size=26) + 0.05
y += picture(x + PAD, y, IW, PRINT / "fp_analogs.png") + 0.05
y += caption(x + PAD, y,
             f"Grey: all IMPROVE filters; red: the {int(S_AN.n)} most like Addis. TOR and HIPS agree on them, but "
             f"the same FTIR calibration predicts them poorly, at ordinary IMPROVE concentrations. Same "
             "calibration; only the samples changed.") + 0.25

y += header(x, y, CW, "Addis filters carry far more EC than IMPROVE filters")
y += 0.2
y += picture(x + PAD + 0.4, y, IW - 0.8, PRINT / "fp_loading.png") + 0.05
pct_above = 100 * (g["ad_mass"] > g["p99"]).mean()
y += caption(x + PAD, y,
             f"EC on one filter, from each filter's own volume of air (flow × sampling time). The typical Addis "
             f"filter carries about {g['ratio']:.0f}× the EC of a typical IMPROVE calibration filter, and "
             f"{pct_above:.0f}% are above the IMPROVE 99th percentile.") + 0.25

col_end[1] = y

# ---- column 3: what we have tried (work in progress), conclusions
x, y = XS[2], TOP
y += header(x, y, CW, "Toward a better calibration (work in progress)")
y += 0.2
y += text(x + PAD, y, IW, ["**Step 1. Baseline correction of the spectra**, the biggest gain so far."], size=26) + 0.05
y += picture(x + PAD, y, IW, PRINT / "fp_baseline.png") + 0.05
y += caption(x + PAD, y,
             f"Same Addis filters, same axes as Figure 4. Correcting the baseline moves the intercept from "
             f"{S_ADDIS.deming_intercept:.1f} to {S_BASE.deming_intercept:.1f} µg/m³, but the slope drops to "
             f"{S_BASE.deming_slope:.2f}.") + 0.2
y += text(x + PAD, y, IW, [
    "**Step 2. Test thousands of calibrations** with an interactive app we built, changing which IMPROVE filters "
    "are in the calibration set and how it is built."], size=26) + 0.05
y += picture(x + PAD, y, IW, PRINT / "fp_explorer.png") + 0.05
y += caption(x + PAD, y,
             f"Each column is one calibration ({g['n_spec']:,}, all on baseline-corrected spectra), sorted by its "
             f"Addis intercept (top). Tick marks show which choices it used. {g['n_near0']} put the intercept within "
             f"±0.5 µg/m³ of zero; {n_both} of those also have a slope between 0.8 and 1.25. Next: find out why "
             "those work, or whether it is chance.") + 0.2
y += text(x + PAD, y, IW, ["**Step 3. One calibration per season**, built from the IMPROVE filters whose spectra "
                           "most resemble that season's Addis spectra."], size=26) + 0.05
y += picture(x + PAD, y, IW, PRINT / "fp_seasons.png") + 0.05
y += caption(x + PAD, y,
             f"Dashed: current calibration; grey: baseline-corrected (Step 1); colour: season-specific. Kiremt "
             f"improves (slope "
             f"{K0.deming_slope:.2f} to {K1.deming_slope:.2f}); Dry and Belg do not yet.") + 0.25

y += header(x, y, CW, "Conclusions and next steps")
y += 0.2
y += text(x + PAD, y, IW, [
    "FTIR EC calibrated on ambient IMPROVE filters predicts IMPROVE, and the SPARTAN network as a whole, reasonably, "
    "but not Addis Ababa.",
    "Concentration is not the whole story: IMPROVE filters with Addis-like spectra are poorly predicted at ordinary "
    "IMPROVE levels. Addis burns different fuels (wood, charcoal) from rural US sites, so its EC may differ in "
    "composition.",
    "Next: choose calibration filters by EC-to-PM2.5 ratio; check TOR on the heaviest smoke-affected IMPROVE "
    "filters; longer term, collocated Teflon and quartz filters in Addis Ababa."], bullet=True, size=24) + 0.2

# references (19 Oct; from memory: check volumes and pages before printing)
y += header(x, y, CW, "References")
y += 0.15
y += text(x + PAD, y, IW, [
    "Chow, J. C., et al. (2007). The IMPROVE_A temperature protocol for thermal/optical carbon analysis. "
    "*J. Air Waste Manag. Assoc.*, 57, 1014-1023.",
    "Dillner, A. M., & Takahama, S. (2015). Predicting ambient aerosol thermal-optical reflectance measurements from "
    "infrared spectra: elemental carbon. *Atmos. Meas. Tech.*, 8, 4013-4023.",
    "Kuzmiakova, A., Dillner, A. M., & Takahama, S. (2016). An automated baseline correction protocol for infrared "
    "spectra of atmospheric aerosols collected on polytetrafluoroethylene (Teflon) filters. *Atmos. Meas. Tech.*, 9, "
    "2615-2631.",
    "Malm, W. C., et al. (1994). Spatial and seasonal trends in particle concentration and optical extinction in the "
    "United States. *J. Geophys. Res.*, 99, 1347-1370.",
    "Snider, G., et al. (2015). SPARTAN: a global network to evaluate and enhance satellite-based estimates of "
    "ground-level particulate matter for global health applications. *Atmos. Meas. Tech.*, 8, 505-521.",
    "White, W. H., et al. (2016). A critical review of filter transmittance measurements for aerosol light "
    "absorption, and de novo calibration for a decade of monitoring on PTFE membranes. *Aerosol Sci. Technol.*, 50, "
    "984-1002.",
    "Data: SPARTAN public release v3.1.4 (www.spartan-network.org/data); IMPROVE data from the Federal Land Manager "
    "Environmental Database (FED, views.cira.colostate.edu/fed)."],
    size=17, color=SOFT, gap=0.15)
col_end[2] = y

for i, e in col_end.items():
    print(f"column {i + 1}: content ends at {e:.1f} in (column bottom {BOT})")
    assert e <= BOT + 0.01, f"column {i + 1} overflows by {e - BOT:.1f} in"

out = HERE / "aaar_poster_2026.pptx"
prs.save(out)
# rule for every pptx in this repo: no em dashes
txt = " ".join(sh.text_frame.text for sh in Presentation(out).slides[0].shapes if sh.has_text_frame)
assert "—" not in txt, "em dash on the poster"
print(f"wrote {out.relative_to(ROOT)}")
