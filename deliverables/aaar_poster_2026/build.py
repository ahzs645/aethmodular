"""AAAR 2026 practice talk (group meeting, 15 Oct 2026): figures + pptx.

Story (storyline.md, agreed with Ann 9 Oct): an established FTIR EC method that predicts EC well
across IMPROVE does much worse on the IMPROVE filters that look most like Addis Ababa, so the Addis
problem is a calibration-coverage problem that also exists inside the US network.
No PLS calibration-search material (spec curve, cohorts, VIBES) on the main slides.

Run with anaconda python from the repo root:
    ~/anaconda3/bin/python deliverables/aaar_poster_2026/build.py

Inputs (completed rank-70 meeting run, spline = AIRSpec baseline, Deming at MAC 10):
- research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/meeting/
    panel_stats.csv, addis_predictions.csv, addis_evaluation.csv, pool_test_predictions.csv,
    analog_predictions.csv, improve_references.csv
- data/spartan/raw/FilterBased/ChemSpecPM25 (SPARTAN public release v3.1.4, all sites)
- research/spartan/.cache/improve_chemistry_clean_cache.pkl (IMPROVE FED export, all sites)
- research/ftir_hips_chem/output/plots/ann_followup_20261009/addis_vs_improve_low_wavenumber.png

16 Oct, after the 15 Oct practice talk: f2b gains IMPROVE reference lines (median and 95th percentile EC, median
PM2.5) and a µg-per-filter panel; f4b is a histogram version of the f4 ECDF and is the one on the slide; axis titles
say what is plotted; speaker notes define every term and walk through the axes.
"""

from pathlib import Path
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
HC = ROOT / "research/ftir_hips_chem"
M = HC / "output/tables/vibes_fullrange/gallery_rank70_20261002/meeting"
sys.path.insert(0, str(HC / "scripts"))

GREY, BLUE, PURPLE, ACCENT, AMBER, INK = (
    "#8F8C84", "#2C6E9E", "#7A4FA3", "#B23327", "#C49442", "#22252A")
DPI = 168
MAC = 10.0
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "text.color": INK, "font.size": 12,
    "axes.titlesize": 12.5, "legend.frameon": False,
})
SEASONS = ["Dry (Oct-Feb)", "Belg (Mar-May)", "Kiremt (Jun-Sep)"]
SEASON_LABEL = {"Dry (Oct-Feb)": "Dry, Bega (Oct-Feb)", "Belg (Mar-May)": "Belg (Mar-May)",
                "Kiremt (Jun-Sep)": "Kiremt (Jun-Sep)"}
SEASON_COLOR = {"Dry (Oct-Feb)": AMBER, "Belg (Mar-May)": PURPLE, "Kiremt (Jun-Sep)": BLUE}
FIT = "allimprove|AIRSpec|all-IMPROVE lot 251"
# Plain-language axis titles (15 Oct practice-talk feedback): say what each axis physically is.
X_HIPS = "HIPS light absorption as EC (Fabs / MAC 10, µg/m³)"
Y_FTIR = "FTIR-predicted EC (calibrated on IMPROVE TOR, µg/m³)"
Y_TOR = "TOR EC, thermal-optical reference (µg/m³)"
# How the analog filters were chosen, in words (replaces shorthand such as "Addis-like" in the notes)
ANALOG_DEF = ("the IMPROVE spectra that most resemble Addis spectra (highest correlation across wavenumbers, "
              "excluding defined regions such as the CO₂ band)")


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    Image.open(p).convert("RGB").save(p)
    return p


# ---------------------------------------------------------------- data + anchor validation
stats = pd.read_csv(M / "panel_stats.csv")


def stat(fit_id, population, panel, group=None, eval_group=None):
    q = stats[(stats.fit_id == fit_id) & (stats.population == population) & (stats.panel == panel)]
    if group is not None:
        q = q[q.group == group]
    if eval_group is not None:
        q = q[q.eval_group == eval_group]
    assert len(q) == 1, (fit_id, population, panel, group, len(q))
    return q.iloc[0]


S_ADDIS = stat("deployed|SPARTAN|current SPARTAN calibration", "addis", "ftir_vs_fabs", eval_group="All Addis")
S_POOL = stat(FIT, "pool_test", "ftir_vs_tor")
S_POOL_REF = stat(FIT, "pool_test", "tor_vs_fabs")
S_AN_REF = stat(FIT, "analog_test", "tor_vs_fabs", group="All Addis")
S_AN = stat(FIT, "analog_test", "ftir_vs_tor", group="All Addis")
S_DRY = stat(FIT, "analog_test", "ftir_vs_tor", group="Dry (Oct-Feb)")
S_AN_HIPS = stat(FIT, "analog_test", "ftir_vs_fabs", group="All Addis")

# storyline.md numbers must reproduce
for got, want in [(S_ADDIS.deming_slope, 2.30), (S_ADDIS.deming_intercept, -6.09), (S_ADDIS.R2, 0.78),
                  (S_POOL.deming_slope, 0.93), (S_POOL.deming_intercept, 0.02), (S_POOL.R2, 0.73),
                  (S_AN_REF.deming_slope, 0.91), (S_AN_REF.R2, 0.82),
                  (S_AN.deming_slope, 0.48), (S_AN.R2, 0.46), (S_DRY.deming_slope, 0.44), (S_DRY.R2, 0.33)]:
    assert abs(got - want) < 0.006, (got, want)
assert int(S_ADDIS.n) == 189 and int(S_POOL.n) == 2078 and int(S_AN.n) == 97

ad = pd.read_csv(M / "addis_predictions.csv")
ad = ad[ad.fit_id == "deployed|SPARTAN|current SPARTAN calibration"].dropna(subset=["ftir_ec_ugm3", "Fabs"])
# One Addis cohort for every Addis slide: the 189 filters that have the current SPARTAN FTIR EC (of the 233 in the
# evaluation set; the other 44 have no deployed value). Not the older fixed 190-filter phase-3 cohort.
COHORT = set(ad.MediaId)
COHORT_FID = set(ad.ExternalFilterId.astype(str).str.rsplit("-", n=1).str[0])
assert len(COHORT) == 189
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402
import pls_ksweep as pk  # noqa: E402

# Addis Deming lambda (sigma_x from HIPS uncertainty, sigma_y 0.531), as in every Addis panel
LAM_ADDIS = json.loads((HC / "output/tables/ksweep_20261014/summary.json").read_text())["lambda_addis"]


def addis_fit(d):
    x, y = d.Fabs.to_numpy(float) / MAC, d.ftir_ec_ugm3.to_numpy(float)
    sl, ic = pk.deming(x, y, LAM_ADDIS)
    return SimpleNamespace(deming_slope=float(sl[0]), deming_intercept=float(ic[0]),
                           R2=float(pk.r2(x, y)[0]), n=len(x))


_chk = addis_fit(ad)
assert abs(_chk.deming_slope - S_ADDIS.deming_slope) < 1e-6 and abs(_chk.deming_intercept - S_ADDIS.deming_intercept) < 1e-6
# "Changing MAC changes the slope, not the intercept": rescale x (and its error, so lambda scales by (MAC/10)²)
MAC_CHECK = {}
for _m in (6, 14):
    _sl, _ic = pk.deming(ad.Fabs.to_numpy(float) / _m, ad.ftir_ec_ugm3.to_numpy(float), LAM_ADDIS * (_m / MAC) ** 2)
    MAC_CHECK[_m] = (float(_sl[0]), float(_ic[0]),
                     float(pk.deming(ad.Fabs.to_numpy(float) / _m, ad.ftir_ec_ugm3.to_numpy(float), LAM_ADDIS)[1][0]))
    assert abs(_ic[0] - S_ADDIS.deming_intercept) < 1e-6 and abs(_sl[0] - S_ADDIS.deming_slope * _m / MAC) < 1e-6
pool = pd.read_csv(M / "pool_test_predictions.csv")
pool = pool[pool.fit_id == FIT].dropna(subset=["ftir_ec_ugm3", "tor_ec_ugm3"])
an = pd.read_csv(M / "analog_predictions.csv")
an = an[(an.fit_id == f"{FIT}|All Addis") & (an.role == "test")]
assert len(an) == 97 and an.split.eq("test").all()   # every analog shown is from a held-out site
dry = pd.read_csv(M / "analog_predictions.csv")
dry = dry[(dry.fit_id == f"{FIT}|Dry (Oct-Feb)") & (dry.role == "test")]
assert len(dry) == 81


def stat_box(ax, s, loc=(0.03, 0.97), extra=""):
    txt = (f"Deming: {s.deming_slope:.2f}x {s.deming_intercept:+.2f}\n"
           f"R² {s.R2:.2f}   n {int(s.n)}{extra}")
    ax.text(*loc, txt, transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D8D5CE"))


def lines(ax, s, lim, color=INK):
    xs = np.array([0, lim])
    ax.plot(xs, xs, ls=":", color=GREY, lw=1.2, label="1:1")
    ax.plot(xs, s.deming_intercept + s.deming_slope * xs, color=color, lw=1.8, label="Deming fit")
    ax.set_xlim(0, lim); ax.set_ylim(min(0, ax.get_ylim()[0]), lim)


# ---------------------------------------------------------------- f1 Addis: the problem
fig, ax = plt.subplots(figsize=(6.4, 5.6))
for sname in SEASONS:
    d = ad[ad.season == sname]
    ax.scatter(d.Fabs / MAC, d.ftir_ec_ugm3, s=26, color=SEASON_COLOR[sname], alpha=.85,
               edgecolor="white", lw=.4, label=SEASON_LABEL[sname])
lim = float(np.ceil(max(ad.Fabs.max() / MAC, ad.ftir_ec_ugm3.max())))
lines(ax, S_ADDIS, lim)
ax.set_ylim(min(0, ad.ftir_ec_ugm3.min() - .3), lim)
ax.set_xlabel(X_HIPS)
ax.set_ylabel("FTIR-predicted EC, current SPARTAN calibration\n(calibrated on IMPROVE TOR, µg/m³)")
ax.legend(loc="lower right", fontsize=10.5)
stat_box(ax, S_ADDIS)
save(fig, "f1_addis_problem.png")

# ---------------------------------------------------------------- f2 IMPROVE held-out: the method works
fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.scatter(pool.tor_ec_ugm3, pool.ftir_ec_ugm3, s=9, color=GREY, alpha=.45, lw=0)
lim = 1.2
lines(ax, S_POOL, lim)
ax.set_ylim(-0.1, lim)
ax.set_xlabel(Y_TOR)
ax.set_ylabel(Y_FTIR)
stat_box(ax, S_POOL)
ax.text(0.97, 0.03, f"{(pool.tor_ec_ugm3 > lim).sum() + (pool.ftir_ec_ugm3 > lim).sum()} points beyond axes not shown",
        transform=ax.transAxes, ha="right", fontsize=9, color=GREY)
save(fig, "f2_improve_heldout.png")

# ---------------------------------------------------------------- f3 Addis-like IMPROVE filters
fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.3))
lim = 1.2
ax = axes[0]
ax.scatter(pool.fabs_Mm1 / MAC, pool.tor_ec_ugm3, s=7, color=GREY, alpha=.25, lw=0, label="all held-out IMPROVE")
ax.scatter(an.fabs_Mm1 / MAC, an.tor_ec_ugm3, s=26, color=ACCENT, alpha=.9, edgecolor="white", lw=.4,
           label="spectra most correlated\nwith Addis (97)")
lines(ax, S_AN_REF, lim, ACCENT)
ax.set_xlabel(X_HIPS); ax.set_ylabel(Y_TOR)
ax.set_title("TOR and HIPS agree", loc="left", fontsize=12.5)
stat_box(ax, S_AN_REF)
ax.legend(loc="lower right", fontsize=10)
ax = axes[1]
ax.scatter(pool.fabs_Mm1 / MAC, pool.ftir_ec_ugm3, s=7, color=GREY, alpha=.25, lw=0)
ax.scatter(an.fabs_Mm1 / MAC, an.ftir_ec_ugm3, s=26, color=ACCENT, alpha=.9, edgecolor="white", lw=.4)
lines(ax, S_AN_HIPS, lim, ACCENT)
ax.set_ylim(-0.1, lim)
ax.set_xlabel(X_HIPS); ax.set_ylabel(Y_FTIR)
ax.set_title("FTIR EC is poorly predicted", loc="left", fontsize=12.5)
stat_box(ax, S_AN_HIPS)
fig.tight_layout()
save(fig, "f3_addis_like_improve.png")

# ---------------------------------------------------------------- public SPARTAN chemistry (all sites)
# data/spartan/raw = the SPARTAN public FilterBased ChemSpecPM25 release (v3.1.4, Aug 2026). The older in-house
# unified_filter_dataset reports ETAD OC about 1.96x lower than this release, filter for filter, so it is not used.
# SPARTAN has no thermal-optical carbon: OC/EC/OM are FTIR products (method 218), BC is HIPS (method 221).
SP_RAW = ROOT / "data/spartan/raw/FilterBased/ChemSpecPM25"


def load_spartan_site(path):
    d = pd.read_csv(path, comment="#", low_memory=False)
    # BC: keep HIPS measurements only; drop values "Estimated from the HIPS-SSR BC curve" (method 220)
    d = d[~(d.Parameter_Name.eq("BC PM2.5") & ~d.Analysis_Description.fillna("").str.upper().eq("HIPS"))]
    scale = np.where(d.Units.str.startswith("Nanograms"), 1e-3, 1.0)       # elements: ng/m³ -> µg/m³
    d = d.assign(Value=d.Value * scale,
                 date=pd.to_datetime(dict(year=d.Start_Year_local, month=d.Start_Month_local, day=d.Start_Day_local),
                                     errors="coerce"))
    w = d.pivot_table(index=["Site_Code", "Filter_ID", "date"], columns="Parameter_Name", values="Value",
                      aggfunc="first").reset_index()
    name = next(l.split(":", 1)[1].strip() for l in open(path) if l.startswith("# Site:"))
    return w.assign(site_name=name.split(",")[0])


sp_all = pd.concat([load_spartan_site(f) for f in sorted(SP_RAW.glob("FilterBased_ChemSpecPM25_*.csv"))],
                   ignore_index=True)
for c in ["OC PM2.5", "BC PM2.5"]:
    sp_all[c] = sp_all.get(c)
# OC per unit HIPS absorption (µg m⁻³ per Mm⁻¹). SPARTAN "BC PM2.5" is HIPS fAbs / 10 exactly (ETAD median 10.00),
# so fAbs = 10 × BC.
sp_all["oc_ec"] = sp_all["OC PM2.5"] / (MAC * sp_all["BC PM2.5"]).where(sp_all["BC PM2.5"] > 0)
etad = sp_all[sp_all.Site_Code.eq("ETAD")]
ad_oe = etad.oc_ec[etad.oc_ec > 0]

# ---------------------------------------------------------------- f4 loading per filter
refs = pd.read_csv(M / "improve_references.csv")
ev = pd.read_csv(M / "addis_evaluation.csv")
ev = ev[ev.MediaId.isin(COHORT)]
imp_mass = (refs.fabs_Mm1 / MAC * refs.tor_volume_m3).dropna()
imp_mass = imp_mass[imp_mass > 0]
ad_mass = (ev.Fabs / MAC * ev.Volume_m3).dropna()
ratio = ad_mass.median() / imp_mass.median()
p99 = imp_mass.quantile(.99)
assert 8 < ratio < 10, ratio   # storyline: ~9x


def ecdf(ax, vals, color, lab):
    v = np.sort(np.asarray(vals))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, lw=2.4, label=lab)
    ax.axvline(np.median(v), color=color, ls="--", lw=1.1)


fig, ax = plt.subplots(figsize=(10.5, 4.6))
ecdf(ax, imp_mass, GREY, f"IMPROVE calibration pool (n {len(imp_mass):,})")
ecdf(ax, ad_mass, AMBER, f"Addis Ababa, SPARTAN (n {len(ad_mass)})")
ax.axvline(p99, color=INK, ls=":", lw=1.2)
ax.text(p99, 0.04, "IMPROVE 99th \npercentile ", fontsize=10, va="bottom", ha="right")
ax.text(ad_mass.max() * 1.15, 0.55, f"Addis median is\n{ratio:.0f}× the IMPROVE median", fontsize=11)
ax.set_xscale("log"); ax.set_ylim(0, 1.02)
ax.set_xlabel("EC on the filter, HIPS Fabs / MAC 10 × sampled volume (µg per filter)")
ax.set_ylabel("Fraction of filters")
ax.legend(loc="upper left", fontsize=10.5)
save(fig, "f4_loading_per_filter.png")

# f4b: the same two distributions as histograms (15 Oct: the ECDF needed too much explaining). Log-spaced bins,
# each network normalised to 100% so a 189-filter set and a 12,000-filter set can share one axis.
bins = np.logspace(np.log10(min(imp_mass.min(), ad_mass.min()) * .9),
                   np.log10(max(imp_mass.max(), ad_mass.max()) * 1.1), 46)
fig, ax = plt.subplots(figsize=(10.5, 4.8))
for vals, color, lab in [(imp_mass, GREY, f"IMPROVE calibration filters (n {len(imp_mass):,})"),
                         (ad_mass, AMBER, f"Addis Ababa, SPARTAN (n {len(ad_mass)})")]:
    w = np.full(len(vals), 100 / len(vals))
    ax.hist(vals, bins=bins, weights=w, color=color, alpha=.35, lw=0)
    ax.hist(vals, bins=bins, weights=w, histtype="step", color=color, lw=2, label=lab)
top = ax.get_ylim()[1] * 1.12
ax.set_ylim(0, top)
for v, color, lab, side in [(imp_mass.median(), GREY, "IMPROVE\nmedian", "right"),
                            (ad_mass.median(), AMBER, "Addis\nmedian", "left")]:
    ax.axvline(v, color=color, ls="--", lw=1.4)
    ax.text(v * (1.06 if side == "left" else 1 / 1.06), top * .97, f"{lab}\n{v:.0f} µg", va="top", ha=side,
            fontsize=10, color=INK)
ax.axvline(p99, color=INK, ls=":", lw=1.4)
ax.text(p99 / 1.06, top * .62, f"IMPROVE 99th\npercentile\n{p99:.0f} µg", va="top", ha="right", fontsize=10)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("EC on one filter (µg per filter, log scale), from HIPS: Fabs / MAC 10 × volume of air sampled")
ax.set_ylabel("Percent of filters\n(each network adds to 100%)")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=10.5)
save(fig, "f4b_loading_per_filter_hist.png")

# ---------------------------------------------------------------- IMPROVE network (all sites, FED export)
imp_net = pd.read_pickle(ROOT / "research/spartan/.cache/improve_chemistry_clean_cache.pkl")
imp_net = imp_net[imp_net.fAbs_Val > 0].assign(ec_hips=lambda d: d.fAbs_Val / MAC)
imp_ec_median = float(imp_net.ec_hips.median())
# Reference levels for slide 2, from the same filters as the typical-IMPROVE EC line (every IMPROVE filter with fAbs).
# MF = IMPROVE gravimetric PM2.5 on the same Teflon (module A) filter that fAbs and FTIR are measured on.
imp_ec_p95 = float(imp_net.ec_hips.quantile(.95))
imp_pm_median = float(imp_net.MF_Val.dropna().median())
# Per filter: module A volume = flow (L/min) × duration (min) / 1000, where the export reports both (2003 onward rows
# that have them); a few negative/zero flags are dropped.
_v = imp_net.FlowRate_Val * imp_net.SampDur_Val / 1000
imp_pf = imp_net.assign(vol=_v)[(imp_net.FlowRate_Val > 0) & (imp_net.SampDur_Val > 0)]
imp_flow_median = float(imp_pf.FlowRate_Val.median())
imp_vol_median = float(imp_pf.vol.median())
imp_pf_ec = imp_pf.ec_hips * imp_pf.vol
imp_pf_ec_median, imp_pf_ec_p95 = float(imp_pf_ec.median()), float(imp_pf_ec.quantile(.95))
imp_pf_pm_median = float((imp_pf.MF_Val * imp_pf.vol).dropna().median())
imp_share_lo, imp_share_hi = (100 * (imp_net.ec_hips / imp_net.MF_Val.where(imp_net.MF_Val > 0))).quantile([.05, .95])
print(f"IMPROVE reference (n {len(imp_net):,} filters with fAbs): EC median {imp_ec_median:.3f}, p95 {imp_ec_p95:.3f}, "
      f"PM2.5 median {imp_pm_median:.2f} µg/m³ (EC share {100 * imp_ec_median / imp_pm_median:.1f}%). Per filter "
      f"(n {len(imp_pf):,} with flow and duration; median flow {imp_flow_median:.1f} L/min, volume "
      f"{imp_vol_median:.1f} m³): EC median {imp_pf_ec_median:.2f}, p95 {imp_pf_ec_p95:.2f}, PM2.5 median "
      f"{imp_pf_pm_median:.1f} µg per filter")


def site_table(df, site, val, min_n):
    t = df[[site, val]].dropna().groupby(site)[val].agg(
        n="size", median="median", p25=lambda v: v.quantile(.25), p75=lambda v: v.quantile(.75))
    return t[t.n >= min_n]


# ---------------------------------------------------------------- f2b Addis EC by season (slide 2)
# EC = SPARTAN 'BC PM2.5' (HIPS fAbs / 10), the same quantity as IMPROVE fAbs / 10
e = etad[etad.Filter_ID.isin(COHORT_FID)].copy()
month = e.date.dt.month
e["season"] = np.select([month.isin([10, 11, 12, 1, 2]), month.isin([3, 4, 5])],
                        ["Dry (Oct-Feb)", "Belg (Mar-May)"], "Kiremt (Jun-Sep)")
e = e[e["BC PM2.5"].notna()]
sea = e.groupby("season").agg(ec=("BC PM2.5", "median"), ec25=("BC PM2.5", lambda v: v.quantile(.25)),
                             ec75=("BC PM2.5", lambda v: v.quantile(.75)), pm=("Filter PM2.5 mass", "median"),
                             n=("BC PM2.5", "size")).reindex(SEASONS)
ad_ec_median = float(e["BC PM2.5"].median())
ec_x_improve = ad_ec_median / imp_ec_median
# Addis volume per filter from the evaluation table (same 189 filters); SPARTAN samples about 5 L/min
_vol = ev.assign(fid=ev.ExternalFilterId.astype(str).str.rsplit("-", n=1).str[0]).groupby("fid").Volume_m3.first()
e["vol"] = e.Filter_ID.map(_vol)
assert e.vol.notna().all(), e.vol.isna().sum()
e["ec_pf"], e["pm_pf"] = e["BC PM2.5"] * e.vol, e["Filter PM2.5 mass"] * e.vol
sea_pf = e.groupby("season").agg(ec=("ec_pf", "median"), ec25=("ec_pf", lambda v: v.quantile(.25)),
                                ec75=("ec_pf", lambda v: v.quantile(.75)), pm=("pm_pf", "median")).reindex(SEASONS)
ad_vol_median = float(e.vol.median())
ad_pf_ec_median, ad_pf_pm_median = float(e.ec_pf.median()), float(e.pm_pf.median())
ad_pm_median = float(e["Filter PM2.5 mass"].median())
ec_x_improve_pf = ad_pf_ec_median / imp_pf_ec_median
print(f"Addis (n {len(e)}): EC median {ad_ec_median:.2f}, PM2.5 median {ad_pm_median:.1f} µg/m³ "
      f"({ec_x_improve:.0f}x IMPROVE EC); per filter (volume {ad_vol_median:.2f} m³): EC {ad_pf_ec_median:.1f}, "
      f"PM2.5 {ad_pf_pm_median:.0f} µg ({ec_x_improve_pf:.1f}x IMPROVE EC per filter)")


def ref_lines(ax, refs, x_label, min_gap=0.1):
    """Full-width IMPROVE reference lines, each labelled at the right end of the axes.

    Labels are pushed apart vertically (in axes fraction) so lines that sit close together near zero stay legible;
    a thin leader joins a label to its line when it had to move.
    """
    y0, y1 = ax.get_ylim()
    frac = lambda v: (v - y0) / (y1 - y0)  # noqa: E731
    order = sorted(refs, key=lambda r: r[0])
    placed = []
    for v, lab, style in order:
        ax.axhline(v, zorder=1.5, **style)
        y = max(frac(v) + 0.012, placed[-1] + min_gap if placed else 0.0)
        placed.append(y)
        tr = ax.get_yaxis_transform()
        ax.text(x_label + 0.014, y, lab, transform=ax.transAxes, va="bottom", ha="left", fontsize=9.6,
                color=style["color"], fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.9))
        if y - frac(v) > 0.03:
            ax.annotate("", xy=(x_label + 0.006, v), xycoords=tr, xytext=(x_label + 0.006, y + 0.03),
                        textcoords="axes fraction", arrowprops=dict(arrowstyle="-", color=style["color"], lw=.8))


STY_EC = dict(color=INK, ls=":", lw=2.0)
STY_P95 = dict(color=ACCENT, ls=(0, (6, 2, 1.5, 2)), lw=1.8)
STY_PM = dict(color="#55524C", ls="--", lw=1.6)
fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), gridspec_kw=dict(wspace=.22))
xs = np.arange(3)
for ax, t, unit, ttl, refs in [
        (axes[0], sea, "µg/m³", "In the air: concentration",
         [(imp_ec_median, f"IMPROVE median EC\n{imp_ec_median:.2f} µg/m³", STY_EC),
          (imp_ec_p95, f"IMPROVE 95th\npercentile EC {imp_ec_p95:.2f}", STY_P95),
          (imp_pm_median, f"IMPROVE median\nPM2.5 {imp_pm_median:.1f} µg/m³", STY_PM)]),
        (axes[1], sea_pf, "µg per filter", "On the filter: mass per filter (what FTIR sees)",
         [(imp_pf_ec_median, f"IMPROVE median EC\n{imp_pf_ec_median:.1f} µg", STY_EC),
          (imp_pf_ec_p95, f"IMPROVE 95th\npercentile EC {imp_pf_ec_p95:.0f} µg", STY_P95),
          (imp_pf_pm_median, f"IMPROVE median\nPM2.5 {imp_pf_pm_median:.0f} µg", STY_PM)])]:
    ax.bar(xs - 0.2, t.pm, width=.38, color="#E4E1DA", label="Addis PM2.5 mass (median)", zorder=2)
    ax.bar(xs + 0.2, t.ec, width=.38, color=[SEASON_COLOR[g] for g in SEASONS], label="Addis EC from HIPS (median, IQR)",
           yerr=[t.ec - t.ec25, t.ec75 - t.ec], capsize=4, ecolor=INK, error_kw=dict(lw=1), zorder=2)
    pad = 0.015 * t.pm.max()
    for i in range(3):
        ax.text(i + 0.2, t.ec75.iloc[i] + pad, f"{t.ec.iloc[i]:.1f}", ha="center", va="bottom", fontsize=11.5,
                fontweight="bold", zorder=3)
        ax.text(i - 0.2, t.pm.iloc[i] + pad, f"{t.pm.iloc[i]:.0f}", ha="center", va="bottom", fontsize=10.5,
                color="#6E6B64", zorder=3)
        ax.text(i - 0.2, t.pm.iloc[i] * .55, f"EC is\n{100 * t.ec.iloc[i] / t.pm.iloc[i]:.0f}%\nof\nPM2.5", ha="center",
                va="center", fontsize=9.5, color="#55524C", zorder=3)
    ax.set_xlim(-0.55, 3.3)
    ax.set_ylim(0, t.pm.max() * 1.16)
    ax.set_xticks(xs, [f"{SEASON_LABEL[g]}\nn {int(sea.n.iloc[i])}"
                       for i, g in enumerate(SEASONS)], fontsize=10)
    ax.set_ylabel(f"Median {'concentration' if unit == 'µg/m³' else 'mass on the filter'} ({unit})")
    ax.set_title(ttl, loc="left", fontsize=12.5)
    ref_lines(ax, refs, (2.45 + 0.55) / 3.85)
axes[0].legend(loc="upper left", fontsize=10)
fig.text(0.01, -0.06,
         f"IMPROVE reference lines: all {len(imp_net):,} IMPROVE filters with HIPS Fabs, 2003 to 2025 (right panel: the "
         f"{len(imp_pf):,} that also report flow and sampling time); typical IMPROVE EC is "
         f"{100 * imp_ec_median / imp_pm_median:.0f}% of PM2.5. IMPROVE samples about 22 L/min "
         f"(about {imp_vol_median:.0f} m³ per filter), SPARTAN about 5 L/min (about {ad_vol_median:.0f} m³), so the "
         f"Addis/IMPROVE gap is much larger in the air ({ec_x_improve:.0f}× for EC) than on the filter "
         f"({ec_x_improve_pf:.0f}×). FTIR measures the mass on the filter.",
         fontsize=9.6, color="#55524C", ha="left", va="top", wrap=True)
save(fig, "f2b_addis_ec_by_season.png")
# claims made in the slide 2 notes
assert (sea.ec > imp_pm_median).all() and (sea.ec > imp_ec_p95).all() and (sea_pf.ec > imp_pf_ec_p95).all()

# ---------------------------------------------------------------- f5 EC by site, every IMPROVE and SPARTAN site
# EC from HIPS for both networks (IMPROVE fAbs / 10; SPARTAN BC PM2.5 = fAbs / 10). Site medians with IQR;
# IMPROVE sites need >= 50 filters, SPARTAN >= 10. Style of research/spartan/follow_up_plots fig08.
imp_sites = site_table(imp_net, "SiteCode", "ec_hips", 50).assign(network="IMPROVE")
sp_sites = site_table(sp_all[sp_all["BC PM2.5"] > 0], "Site_Code", "BC PM2.5", 10).assign(network="SPARTAN")
sp_sites["name"] = sp_all.groupby("Site_Code").site_name.first().reindex(sp_sites.index)
cont = pd.concat([imp_sites, sp_sites]).sort_values("median").reset_index(names="site")
cont["rank"] = np.arange(len(cont))
imp_max = float(imp_sites["median"].max())
sp_above_imp_max = int((sp_sites["median"] > imp_max).sum())
sp_above_imp_med = int((sp_sites["median"] > imp_sites["median"].median()).sum())
etad_rank_top = int((cont["median"] > sp_sites.loc["ETAD", "median"]).sum()) + 1
assert sp_sites.loc["ETAD", "median"] > imp_max

fig, ax = plt.subplots(figsize=(13, 6.2))
im = cont[cont.network.eq("IMPROVE")]
ax.errorbar(im["rank"], im["median"], yerr=[im["median"] - im.p25, im.p75 - im["median"]],
            fmt="o", ms=3.2, color="#8C8C8C", ecolor="#D2D2D2", elinewidth=.8,
            label=f"IMPROVE sites (n={len(im)}, median ± IQR)")
sp = cont[cont.network.eq("SPARTAN")].reset_index(drop=True)
cmap = [*plt.get_cmap("tab20").colors, *plt.get_cmap("tab20b").colors]
for i, r in sp.iterrows():
    is_ad = r.site == "ETAD"
    col = AMBER if is_ad else cmap[i % len(cmap)]
    ax.errorbar([r["rank"]], [r["median"]], yerr=[[r["median"] - r.p25], [r.p75 - r["median"]]],
                fmt="*", ms=20 if is_ad else 12, color=col, ecolor=col, elinewidth=1.2, zorder=6 if is_ad else 5,
                mec=INK if is_ad else "white", mew=.6 if is_ad else .3,
                label=f"{r.site} {r['name']} ({r['median']:.2f})")
ax.set_yscale("log")
ax.set_xlim(-3, len(cont) + 2)
ax.set_xlabel("All IMPROVE and SPARTAN sites, ranked by median EC")
ax.set_ylabel("EC from HIPS, Fabs / 10 (µg/m³, log scale)")
ax.grid(True, color="#EEEEEE", lw=.8); ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=8.6, ncol=3, frameon=True, framealpha=.95, edgecolor="#DDDDDD",
          handletextpad=.3, columnspacing=1.0, borderpad=.5)
save(fig, "f5_ec_site_continuum.png")
cont.round(4).to_csv(FIG / "f5_ec_site_continuum.csv", index=False)
print(f"EC continuum: {len(imp_sites)} IMPROVE, {len(sp_sites)} SPARTAN; ETAD {sp_sites.loc['ETAD','median']:.2f}, "
      f"#{etad_rank_top} of {len(cont)} from the top; IMPROVE max {imp_max:.2f}; SPARTAN above IMPROVE max "
      f"{sp_above_imp_max}, above IMPROVE median {sp_above_imp_med}")


# ---------------------------------------------------------------- f6 baseline correction of the spectra helps (Addis, before/after)
base = pd.read_csv(M / "addis_predictions.csv")
base = base[(base.fit_id == FIT) & base.MediaId.isin(COHORT)].dropna(subset=["ftir_ec_ugm3", "Fabs"])
assert len(base) == len(COHORT)
S_BASE = addis_fit(base)


def addis_panel(ax, d, s, title, lim):
    for g in SEASONS:
        q = d[d.season == g]
        ax.scatter(q.Fabs / MAC, q.ftir_ec_ugm3, s=22, color=SEASON_COLOR[g], alpha=.85, edgecolor="white", lw=.4,
                   label=SEASON_LABEL[g])
    xs = np.array([0, lim])
    ax.plot(xs, xs, ls=":", color=GREY, lw=1.2)
    ax.plot(xs, s.deming_intercept + s.deming_slope * xs, color=INK, lw=1.8)
    ax.set_xlim(0, lim); ax.set_ylim(-1, lim)
    ax.set_xlabel(X_HIPS)
    ax.set_title(title, loc="left", fontsize=12.5)
    stat_box(ax, s)


lim = float(np.ceil(max(ad.Fabs.max() / MAC, ad.ftir_ec_ugm3.max(), base.ftir_ec_ugm3.max())))
fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4), sharey=True)
addis_panel(axes[0], ad, S_ADDIS, "Current calibration, no baseline correction", lim)
addis_panel(axes[1], base, S_BASE, "Baseline-corrected spectra, IMPROVE calibration", lim)
axes[0].set_ylabel(Y_FTIR)
axes[1].legend(loc="lower right", fontsize=10)
fig.tight_layout()
save(fig, "f6_baseline_before_after.png")

# ---------------------------------------------------------------- f7 the calibration explorer (spline only)

SPEC = ROOT / "gallery/app/public/data/meeting-r70/spec_curve_20260917.json"   # gallery/data export, rank-70 run
spec = json.loads(SPEC.read_text())
sp_idx = spec["categories"]["sp"].index("airspec")
keep = np.flatnonzero(np.array(spec["sp"]) == sp_idx)
inter = np.array(spec["intercept"])[keep]
order = keep[np.argsort(inter)]
n_spec = len(order)
n_better = int((inter > S_ADDIS.deming_intercept).sum())
n_near0 = int((np.abs(inter) <= 0.5).sum())
masks = {c: np.array([ch == "1" for ch in spec["masks"][i]]) for i, c in enumerate(spec["choices"])}
ROWS = [("Which IMPROVE filters calibrate", [
            ("Lowest OC/EC", "Lowest OC/EC"),
            ("Spectral analogs, all Addis", "Spectral analogs (correlation), all Addis"),
            ("Spectral analogs, Dry season", "Spectral analogs (correlation), Dry season"),
            ("Spectral analogs, Belg season", "Spectral analogs (correlation), Belg season"),
            ("Spectral analogs, Kiremt season", "Spectral analogs (correlation), Kiremt season"),
            # explorer key "Ethiopia-shaped smoke": IMPROVE smoke filters ranked by C-H, carbonyl and 1600 cm⁻¹
            # band features nearest the Addis median (pls_transfer.band_feature_distance); not the correlation pick
            ("Smoke filters nearest Addis in C-H/carbonyl bands", "Ethiopia-shaped smoke"),
            ("Spectral analogs, weighted distance", "Spectral analogs (VIP-weighted distance)")]),
        ("Calibration settings", [
            ("Site-grouped cross-validation", "Site-grouped cross-validation"),
            ("Interleaved cross-validation", "Interleaved cross-validation"),
            ("9 or fewer PLS factors", "9 or fewer PLS factors"), ("15 or more PLS factors", "15 or more PLS factors"),
            ("Cohort under 600 filters", "Cohort under 600 filters")])]
n_rows = sum(len(r) for _, r in ROWS) + len(ROWS)
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.6, 6.6), sharex=True,
                               gridspec_kw=dict(height_ratios=[1, 2.4], hspace=.06))
ax0.plot(np.arange(n_spec), np.sort(inter), color=BLUE, lw=2)
ax0.axhline(0, color=GREY, ls=":", lw=1.2)
ax0.axhline(S_ADDIS.deming_intercept, color=ACCENT, ls="--", lw=1.1)
ax0.text(n_spec * .99, S_ADDIS.deming_intercept + .25, f"current calibration {S_ADDIS.deming_intercept:.1f}",
         ha="right", fontsize=9.5, color=ACCENT)
ax0.set_ylabel("Addis intercept\n(µg/m³)")
ax0.set_ylim(S_ADDIS.deming_intercept - .8, 1.0)
y = n_rows
labels, ticks = [], []
for head, rows in ROWS:
    y -= 1
    labels.append(head); ticks.append(y)
    for lab, key in rows:
        y -= 1
        on = np.flatnonzero(masks[key][order])
        ax1.vlines(on, y - .38, y + .38, color=INK, lw=.35, alpha=.55)
        labels.append("   " + lab); ticks.append(y)
ax1.set_yticks(ticks, labels)
for t in ax1.get_yticklabels():
    if not t.get_text().startswith(" "):
        t.set_fontweight("bold")
ax1.set_ylim(-.6, n_rows - .4)
ax1.set_xlim(-5, n_spec + 5)
ax1.tick_params(axis="y", length=0)
ax1.spines["left"].set_visible(False)
ax1.set_xlabel(f"{n_spec:,} calibrations, ordered from most negative to least negative Addis intercept")
save(fig, "f7_calibration_explorer.png")

# ---------------------------------------------------------------- f8 season-specific calibrations
SEAS_FIT = {g: f"seasonal|AIRSpec|locked|{g}" for g in SEASONS}
allp = pd.read_csv(M / "addis_predictions.csv")
stitched = []
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9), sharey=True)
seas_rows = []
lim = 12
for ax, g in zip(axes, SEASONS):
    b0 = base[base.season == g]
    sv = allp[(allp.fit_id == SEAS_FIT[g]) & (allp.season == g) & allp.MediaId.isin(COHORT)].dropna(subset=["ftir_ec_ugm3"])
    s0, s1 = addis_fit(b0), addis_fit(sv)
    stitched.append(sv)
    seas_rows.append((g, s0, s1))
    ax.scatter(b0.Fabs / MAC, b0.ftir_ec_ugm3, s=16, color=GREY, alpha=.55, lw=0, label="IMPROVE calibration")
    ax.scatter(sv.Fabs / MAC, sv.ftir_ec_ugm3, s=22, color=SEASON_COLOR[g], alpha=.9, edgecolor="white", lw=.4,
               label="season-specific calibration")
    xs = np.array([0, lim])
    ax.plot(xs, xs, ls=":", color=GREY, lw=1.2, label="1:1")
    ax.plot(xs, s0.deming_intercept + s0.deming_slope * xs, color=GREY, lw=1.6, label="fit, IMPROVE calibration")
    ax.plot(xs, s1.deming_intercept + s1.deming_slope * xs, color=SEASON_COLOR[g], lw=2, label="fit, season-specific")
    ax.set_xlim(0, lim); ax.set_ylim(-1, lim)
    ax.set_title(SEASON_LABEL[g], loc="left", fontsize=12.5)
    ax.set_xlabel("HIPS light absorption as EC\n(Fabs / MAC 10, µg/m³)")
    ax.text(.03, .97, f"IMPROVE: {s0.deming_slope:.2f}x {s0.deming_intercept:+.2f}, R² {s0.R2:.2f}\n"
                      f"season:   {s1.deming_slope:.2f}x {s1.deming_intercept:+.2f}, R² {s1.R2:.2f}",
            transform=ax.transAxes, va="top", fontsize=9.8, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D8D5CE"))
axes[0].set_ylabel("FTIR-predicted EC\n(calibrated on IMPROVE TOR, µg/m³)")
# one shared legend under the panels; neutral swatches so it reads for every season
from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([], [], ls="", marker="o", ms=6, color=GREY, alpha=.7, label="IMPROVE calibration (points)"),
           Line2D([], [], color=GREY, lw=1.8, label="fit, IMPROVE calibration"),
           Line2D([], [], ls="", marker="o", ms=6, color=INK, label="season-specific calibration (points, season colour)"),
           Line2D([], [], color=INK, lw=2, label="fit, season-specific (season colour)"),
           Line2D([], [], color=GREY, ls=":", lw=1.4, label="1:1")]
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
save(fig, "f8_seasonal_calibrations.png")
SEAS = {g: (s0, s1) for g, s0, s1 in seas_rows}
S_STITCH = addis_fit(pd.concat(stitched))
assert S_STITCH.n == len(COHORT)

# ---------------------------------------------------------------- deck
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def title_box(slide, text, size=20, top=0.35):
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(top), Inches(12.2), Inches(0.9))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(size); p.font.bold = True; p.font.color.rgb = RGBColor(0x22, 0x25, 0x2A)


def notes(slide, say, note):
    slide.notes_slide.notes_text_frame.text = f"SAY: {say}\n\nNOTES: {note}"


def fig_slide(title, png, say, note):
    s = prs.slides.add_slide(BLANK)
    title_box(s, title)
    im = Image.open(png); w, h = im.size
    max_w, max_h, top = 12.2, 6.0, 1.25
    sc = min(max_w / w, max_h / h)
    W, H = w * sc, h * sc
    s.shapes.add_picture(str(png), Inches((13.333 - W) / 2), Inches(top), Inches(W), Inches(H))
    notes(s, say, note)
    return s


def text_slide(title, lines_, say, note, size=17):
    s = prs.slides.add_slide(BLANK)
    title_box(s, title)
    tb = s.shapes.add_textbox(Inches(0.9), Inches(1.7), Inches(11.5), Inches(5))
    tf = tb.text_frame; tf.word_wrap = True
    for i, line in enumerate(lines_):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.font.size = Pt(size); p.space_after = Pt(14)
        p.font.color.rgb = RGBColor(0x22, 0x25, 0x2A)
    notes(s, say, note)
    return s


# 1 title
s = prs.slides.add_slide(BLANK)
title_box(s, "FTIR elemental carbon calibrated on IMPROVE filters transfers poorly to heavily loaded "
             "samples like those in Addis Ababa, and to similar samples within IMPROVE", size=28, top=2.2)
# authors and affiliations as on the AAAR abstract
AUTHORS = ("Ahmad Jalil¹,⁴, Sina Hasheminassab², Kyan Shlipak⁵,², Tesfaye Mamo Mekonnen³, Araya Asfaw³, "
           "Hossein Kazemian¹, Ann M. Dillner⁴")
AFFILIATIONS = [
    "¹ Materials Technology & Environmental Research (MATTER) lab, University of Northern British Columbia, Prince George, BC, Canada",
    "² NASA Jet Propulsion Laboratory, Pasadena, CA, USA",
    "³ Addis Ababa University, Addis Ababa, Ethiopia",
    "⁴ University of California Davis, Air Quality Research Center, Davis, CA, USA",
    "⁵ Northwestern University, Evanston, IL, USA",
]
tb = s.shapes.add_textbox(Inches(0.55), Inches(4.25), Inches(12.2), Inches(2.6))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = AUTHORS
p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = RGBColor(0x22, 0x25, 0x2A); p.space_after = Pt(8)
for line in AFFILIATIONS:
    q = tf.add_paragraph(); q.text = line
    q.font.size = Pt(12); q.font.color.rgb = RGBColor(0x6E, 0x6B, 0x64)
notes(s,
      "This talk is about measuring elemental carbon in Addis Ababa, the capital of Ethiopia, a high-altitude city of "
      "about five million people. Elemental carbon, EC, is the black, soot-like part of fine particle pollution, "
      "mostly from burning. Organic carbon, OC, is the rest of the particle carbon, and organic matter, OM, is that "
      "carbon plus the hydrogen and oxygen attached to it. First why we care: EC in Addis is very high. Then the "
      "problem: the current FTIR calibration predicts Addis EC on the wrong scale, and the same calibration also "
      "struggles on the IMPROVE filters whose spectra most resemble Addis spectra. Then why: Addis filters carry far "
      "more EC than almost anything the calibration learned from. Then what we are doing about it: baseline "
      "correction of the spectra, a calibration explorer to test thousands of calibrations, and season-specific "
      "calibrations.",
      "Practice run of the AAAR poster, laid out as slides first. Work in progress by design: the goal is feedback. "
      "Leave at least 20 minutes for questions; keep the first part of the discussion general. Each later slide "
      "defines its terms the first time they come up; a full glossary is in talking_points.md.")

# 2 EC by season
fig_slide(
    f"Elemental carbon in Addis Ababa is very high in every season, about {ec_x_improve:.0f}× the typical IMPROVE level",
    FIG / "f2b_addis_ec_by_season.png",
    "Addis Ababa has three seasons: a long dry season, the short Belg rains, and the main Kiremt rainy season. "
    "The Addis filters come from SPARTAN, a global network of low-cost filter samplers in mostly urban, polluted "
    "places; we compare with IMPROVE, the US network that samples air in national parks and rural areas. "
    "Point at the axes first. On the left, up the side is concentration in the air, micrograms per cubic metre. "
    "The grey bars are total PM2.5, the fine particle mass, and the coloured bars are EC. EC here comes from HIPS, "
    "a measurement of how much light the filter absorbs. "
    "The three lines are IMPROVE references: the dotted line is typical IMPROVE EC, "
    f"{imp_ec_median:.2f} µg/m³; the red dash-dot line is the IMPROVE 95th percentile, {imp_ec_p95:.2f}, so 95 "
    f"percent of IMPROVE filters are below it; and the dashed line is typical IMPROVE PM2.5, {imp_pm_median:.1f}. "
    f"Addis EC is around {sea.ec.min():.0f} to {sea.ec.max():.0f} µg/m³ in every season, about "
    f"{ad_ec_median / imp_ec_p95:.0f} times even the 95th-percentile IMPROVE filter, and higher than typical "
    "IMPROVE PM2.5 as a whole. EC is also a much bigger share of the mass: about "
    f"{100 * (sea.ec / sea.pm).min():.0f} to {100 * (sea.ec / sea.pm).max():.0f} percent of PM2.5 in Addis, against "
    f"about {100 * imp_ec_median / imp_pm_median:.0f} percent at a typical IMPROVE site. The rainy Kiremt season has "
    "the worst air. "
    "The right panel is the same thing as mass on the filter, which is what FTIR actually sees. IMPROVE pulls about "
    "22 litres of air per minute through its filter, SPARTAN about 5, so an IMPROVE filter collects about four times "
    f"more air. On the filter the EC gap shrinks from about {ec_x_improve:.0f} times to about {ec_x_improve_pf:.0f} "
    "times, but Addis is still above the IMPROVE 95th percentile in every season. That is why we care about "
    "measuring EC well in Addis.",
    "EC = SPARTAN 'BC PM2.5', which is HIPS absorption / 10 (checked on Addis: median ratio 10.00), the same "
    "definition as IMPROVE fAbs / 10 for the reference lines. IMPROVE references over all "
    f"{len(imp_net):,} IMPROVE filters with fAbs in the FED export (2003 to 2025): EC median {imp_ec_median:.3f}, "
    f"95th percentile {imp_ec_p95:.3f}, PM2.5 median {imp_pm_median:.2f} µg/m³ (IMPROVE gravimetric mass, MF, on "
    f"the same Teflon filter). Per filter, IMPROVE EC is {imp_share_lo:.0f} to {imp_share_hi:.0f}% of PM2.5 for the "
    "middle 90% of filters. Per-filter panel: IMPROVE volume = flow × sampling time where the export has both "
    f"({len(imp_pf):,} filters; median {imp_flow_median:.1f} L/min, {imp_vol_median:.1f} m³), giving EC median "
    f"{imp_pf_ec_median:.1f} µg, 95th percentile {imp_pf_ec_p95:.1f} µg, PM2.5 median {imp_pf_pm_median:.0f} µg per "
    f"filter; Addis volumes from the evaluation table (median {ad_vol_median:.1f} m³). "
    f"Medians with IQR over the {len(e)} of the {len(COHORT)} Addis filters used on every Addis slide that have a "
    "public BC value; SPARTAN public release v3.1.4 (Aug 2026). Kiremt is partly worse because the boundary layer "
    "is shallow (ERA5 median ~370-410 m vs 650-800 m in Dry). Season names: Dry is also called Bega. "
    "To do: per-filter comparison ignores any difference in deposit area between the two samplers; check before "
    "quoting areal loading (µg/cm²).")

# 3 the problem
fig_slide(
    "In Addis Ababa, current SPARTAN FTIR EC is well correlated with HIPS EC but on the wrong scale",
    FIG / "f1_addis_problem.png",
    "Here is the problem. Point at the axes first. Along the bottom is EC from HIPS: HIPS measures how much light "
    "the filter absorbs, Fabs, and we turn that into EC by dividing by the mass absorption cross-section, MAC, which "
    "is how strongly one gram of EC absorbs light; we use 10 square metres per gram. Up the side is EC predicted "
    "from the FTIR spectrum with the current SPARTAN calibration. FTIR is infrared spectroscopy: infrared light goes "
    "through the filter and we record how much is absorbed at each wavenumber; that curve is the spectrum. A PLS "
    "model, partial least squares, a standard regression method for spectra, learns from a calibration set, here "
    "IMPROVE filters whose EC was measured in the lab by TOR, how to turn a spectrum into EC. "
    f"Each point is one Addis filter. R² is {S_ADDIS.R2:.2f}, so the two measurements are clearly related. But the "
    f"slope of {S_ADDIS.deming_slope:.1f} and the intercept of {S_ADDIS.deming_intercept:.0f} µg/m³ are not "
    f"physically sensible: a filter with no EC on it would be predicted at about {S_ADDIS.deming_intercept:.0f}. "
    "Lower-EC filters, mostly dry season, come out too low, and the high Kiremt filters too high. And it is not just "
    "the MAC: changing the MAC stretches the bottom axis, which changes the slope, not the intercept.",
    "Deming regression, HIPS EC = Fabs / MAC 10 on x. SPARTAN filters have no thermal-optical (TOR) analysis, so "
    f"HIPS is the independent reference at Addis. n {int(S_ADDIS.n)}. The current calibration uses spectra without "
    "baseline correction (slide 8). Colours are the three seasons from slide 2. MAC check (Deming with the HIPS "
    f"error rescaled with the axis): MAC 6 gives {MAC_CHECK[6][0]:.2f}x {MAC_CHECK[6][1]:+.2f}, MAC 14 gives "
    f"{MAC_CHECK[14][0]:.2f}x {MAC_CHECK[14][1]:+.2f}; the intercept does not move. (If the Deming error ratio is "
    f"held fixed instead, the intercept drifts a little, {MAC_CHECK[6][2]:.1f} to {MAC_CHECK[14][2]:.1f}, but stays "
    "strongly negative.)")

# 4 method works
fig_slide(
    "On typical IMPROVE filters, the same FTIR method predicts EC well",
    FIG / "f2_improve_heldout.png",
    "Now IMPROVE. Point at the axes first: along the bottom is EC from TOR, thermal-optical reflectance, the standard "
    "lab method that heats a punch of a quartz-fibre filter and measures the carbon that comes off; up the side is EC "
    "predicted from the FTIR spectrum. IMPROVE collects a Teflon filter for FTIR and a quartz filter for TOR from the "
    "same air at the same time, which is why every FTIR EC calibration is built on IMPROVE. Here the calibration is "
    "built on IMPROVE filters and tested on filters from sites it never saw. On these "
    f"{int(S_POOL.n):,} held-out filters, FTIR matches TOR well: slope {S_POOL.deming_slope:.2f}, intercept close to "
    f"zero, R² {S_POOL.R2:.2f}.",
    "Why calibrate with IMPROVE? (likely question) IMPROVE's collocated FTIR + TOR quartz library is the only one "
    "that exists anywhere; there is no Addis or urban equivalent, and CSN (the US urban network) filters are "
    "different (bigger, thicker, lower flow), so they cannot simply be added. "
    "Baseline-corrected (spline, AIRSpec) spectra, all-IMPROVE calibration on lot 251, held-out sites. "
    f"TOR vs HIPS on the same filters: {S_POOL_REF.deming_slope:.2f}x {S_POOL_REF.deming_intercept:+.2f}, "
    f"R² {S_POOL_REF.R2:.2f}. FTIR vs HIPS on the same filters: 0.88x +0.04, R² 0.47. Axis clipped at 1.2 µg/m³.")

# 5 Addis-like IMPROVE
fig_slide(
    "Current calibrations poorly predict IMPROVE FTIR EC for spectra similar to Addis Ababa spectra",
    FIG / "f3_addis_like_improve.png",
    f"Out of the held-out IMPROVE filters, we picked 97 from {ANALOG_DEF}, in red; grey is all held-out IMPROVE. "
    "Point at the axes first: both panels have HIPS EC along the bottom; up the side is TOR EC on the left and "
    "FTIR-predicted EC on the right. On the left, TOR and HIPS agree on them: "
    f"slope {S_AN_REF.deming_slope:.2f}, R² {S_AN_REF.R2:.2f}. On the right, FTIR EC against HIPS: slope "
    f"{S_AN_HIPS.deming_slope:.2f} and R² {S_AN_HIPS.R2:.2f}. FTIR is too high at the low end and too low at the high "
    "end. Same calibration as the last slide; only the samples changed. So the Addis problem also shows up inside "
    "IMPROVE, for the IMPROVE filters whose spectra most resemble Addis spectra.",
    "Analogs: the 500 IMPROVE filters whose spectra correlate best with the Addis median spectrum (Pearson "
    "correlation across wavenumbers, CO₂ region masked); 97 are from held-out sites (88 lot 251, "
    f"9 lot 248) and none trained the calibration. Against TOR instead of HIPS: {S_AN.deming_slope:.2f}x "
    f"{S_AN.deming_intercept:+.2f}, R² {S_AN.R2:.2f}. Self-prediction: IMPROVE predicts IMPROVE well (slide 4), and the "
    "analogs fit well when they are themselves the calibration set (CV R² 0.5-0.9); the failure is transfer.")

# 6 loading
fig_slide(
    f"Addis filters carry about {ratio:.0f} times more EC per filter than a typical IMPROVE filter",
    FIG / "f4b_loading_per_filter_hist.png",
    "Why? Point at the axes first. Along the bottom is how much EC sits on one filter, in micrograms, on a log "
    "scale, so each labelled tick is ten times the one before. Up the side is the percent of filters at that "
    "loading; each network adds up to 100 percent, so the small Addis set and the large IMPROVE set can share one "
    "plot. Grey is the IMPROVE filters the calibration learned from, gold is Addis. The two humps barely overlap. "
    f"The dashed lines are the medians: the typical Addis filter has about {ratio:.0f} times more EC on it. The "
    f"dotted line is IMPROVE's 99th percentile, and {100 * (ad_mass > p99).mean():.0f} percent of Addis filters are "
    "to the right of it. The calibration has almost never seen filters like these.",
    "Per filter = HIPS Fabs / MAC 10 × sampled volume, so both networks are on the same footing "
    "(IMPROVE ~33 m³ per filter at 22 L/min, SPARTAN ~7 m³ at 5 L/min). "
    f"Addis: the same {len(COHORT)} filters as slides 3, 8 and 10. IMPROVE: the calibration pool "
    f"(n {len(imp_mass):,}), volume from the TOR record. Medians {imp_mass.median():.1f} and {ad_mass.median():.1f} µg, "
    f"IMPROVE 99th percentile {p99:.1f} µg. The 15 Oct version of this slide used a cumulative distribution "
    "(figures/f4_loading_per_filter.png, still built) that needed too much explaining.")

# 7 EC by site
fig_slide(
    f"Addis Ababa EC is higher than at any IMPROVE site, and {sp_above_imp_med} of {len(sp_sites)} SPARTAN sites sit above the typical IMPROVE site",
    FIG / "f5_ec_site_continuum.png",
    f"Every site in both networks, ranked by its typical EC. Grey dots are the {len(imp_sites)} IMPROVE sites; each "
    f"coloured star is one of the {len(sp_sites)} SPARTAN sites around the world. Addis, the big gold star, is at about "
    f"{sp_sites.loc['ETAD', 'median']:.1f} µg/m³, above every IMPROVE site, and most SPARTAN sites sit at the high end. "
    "So this is not only Addis: the IMPROVE calibration network barely covers the EC levels much of the world lives with.",
    "EC from HIPS for both networks (IMPROVE fAbs / 10, SPARTAN 'BC PM2.5' = fAbs / 10), so no FTIR EC is used. "
    "SPARTAN BC values estimated from the HIPS-SSR curve (method 220) are left out; only HIPS measurements count. "
    "Site medians with IQR, log scale; every site, Addis included, uses all its public filters (Addis n "
    f"{int(sp_sites.loc['ETAD', 'n'])}), so the comparison is like for like across sites. IMPROVE sites with >= 50 filters (FED export, 1988-2025); SPARTAN >= 10 "
    f"(public release v3.1.4). Addis is number {etad_rank_top} of {len(cont)} sites from the top; {sp_above_imp_max} "
    f"SPARTAN sites are above the highest IMPROVE site ({imp_max:.2f} µg/m³). Concentration, not per-filter loading: "
    "per filter the SPARTAN values would be ~4.6x lower relative to IMPROVE. Table: figures/f5_ec_site_continuum.csv.")

# 8 baseline correction of the spectra
fig_slide(
    f"Baseline-correcting the spectra shrinks the Addis intercept from {S_ADDIS.deming_intercept:.1f} to {S_BASE.deming_intercept:.1f} µg/m³",
    FIG / "f6_baseline_before_after.png",
    "What are we doing about it? The first thing we learned is that the spectra need baseline correction. A raw "
    "spectrum sits on a sloping, curved background from the filter material and from light scattering; baseline "
    "correction fits that background and subtracts it, so only the absorption bands of the particles are left. "
    "Point at the axes first: the same as slide 3, HIPS EC along the bottom and FTIR-predicted EC up the side. On "
    "the left is the current calibration, without baseline correction. On the right, the same Addis filters "
    "predicted by a calibration built on baseline-corrected IMPROVE spectra. The intercept moves from "
    f"{S_ADDIS.deming_intercept:.1f} to {S_BASE.deming_intercept:.1f} µg/m³. The slope drops to "
    f"{S_BASE.deming_slope:.2f}, so FTIR now reads low at high EC: better, but not solved. As before, R² only says "
    "the two are related; the slope and intercept are what have to become physically sensible, and the MAC choice "
    "moves the slope, not the intercept.",
    f"Both panels: the same {len(COHORT)} Addis filters (those with a current SPARTAN FTIR EC value). Right: spline "
    f"(AIRSpec) baseline, all-IMPROVE lot-251 calibration, R² {S_BASE.R2:.2f}; on all 233 filters it is 0.56x -1.08. The two calibrations differ in more than the baseline "
    "(training set and protocol too), so this is before/after for the approach, not a single-variable test. "
    "A second baseline method (VIBES) is being tested as well; it gives a more negative Addis intercept so far.")

# 9 calibration explorer
fig_slide(
    "A calibration explorer lets us evaluate thousands of calibrations to improve EC predictions",
    FIG / "f7_calibration_explorer.png",
    f"Next, we built a tool that runs and compares many calibrations. Here are {n_spec:,} of them, all on "
    "baseline-corrected spectra, ordered by how far off their Addis intercept is, from worst on the left to best on "
    "the right. The top line is that intercept; the current calibration is the red dashed line. Underneath, each row "
    "shows which calibrations used a choice: first, which IMPROVE filters to calibrate with, for example filters with "
    "the lowest OC/EC, or the IMPROVE spectra that most resemble each Addis season's spectra; second, the "
    "calibration settings. "
    f"{n_better:,} of the {n_spec:,} do better than the current calibration, and {n_near0} get the intercept within "
    "half a microgram of zero, though not always with a good slope. "
    "We have not found a calibration that gets Addis right yet, but this is how we are searching.",
    "Spline baseline only (the VIBES baseline and no-baseline runs have their own versions of this plot). Deming "
    f"intercept at MAC 10, median {np.median(inter):.1f} µg/m³. Settings rows: cross-validation scheme, number of PLS "
    "factors and cohort size; name them if asked, but don't go into them. The explorer scores all 233 Addis filters with HIPS (its "
    "stored grid has no per-filter predictions to re-score), while slides 3, 8 and 10 use the 189 that also have a "
    "current SPARTAN value. From the gallery export "
    "gallery/app/public/data/meeting-r70/spec_curve_20260917.json (rank-70 run). Selection rows: 'Spectral analogs' "
    "= the IMPROVE spectra with the highest correlation to the Addis (or one season's) median spectrum across "
    "wavenumbers, excluding defined regions such as the CO₂ band; 'weighted distance' ranks by a distance that "
    "weights wavenumbers by their importance in the PLS model; 'Smoke filters nearest Addis in C-H/carbonyl bands' "
    "(called 'Ethiopia-shaped smoke' in the explorer) = IMPROVE smoke-affected filters ranked by how close three "
    "band heights (C-H, carbonyl, 1600 cm⁻¹ shoulder) are to the Addis median.")

# 10 seasonal calibrations
k = SEAS["Kiremt (Jun-Sep)"]; d0 = SEAS["Dry (Oct-Feb)"]; b0 = SEAS["Belg (Mar-May)"]
fig_slide(
    "Season-specific calibrations help in Kiremt, but not yet in the Dry or Belg seasons",
    FIG / "f8_seasonal_calibrations.png",
    "One option we tried: a separate calibration for each Addis season, built from the IMPROVE spectra that most "
    "resemble that season's Addis spectra (highest correlation across wavenumbers, excluding defined regions). "
    "Axes as before: HIPS EC along the bottom, FTIR-predicted EC up the side. "
    "Grey points and the grey line are the single IMPROVE calibration from slide 8; "
    "coloured points and line are the season-specific one; the dotted line is 1:1. "
    f"In Kiremt the slope improves from {k[0].deming_slope:.2f} to {k[1].deming_slope:.2f} and R² from "
    f"{k[0].R2:.2f} to {k[1].R2:.2f}. In the dry season it barely changes, and in Belg it gets worse. So seasonality "
    "matters, and it is one of the levers we are exploring.",
    "Each season: 500 IMPROVE spectral analogs (correlation, spectral region 600-3600 cm⁻¹ without CO₂ and the "
    "Teflon band), spline baseline, site-grouped calibration. "
    f"Dry {d0[0].deming_slope:.2f}x {d0[0].deming_intercept:+.2f} -> {d0[1].deming_slope:.2f}x {d0[1].deming_intercept:+.2f}; "
    f"Belg {b0[0].deming_slope:.2f}x {b0[0].deming_intercept:+.2f} -> {b0[1].deming_slope:.2f}x {b0[1].deming_intercept:+.2f}; "
    f"Kiremt {k[0].deming_slope:.2f}x {k[0].deming_intercept:+.2f} -> {k[1].deming_slope:.2f}x {k[1].deming_intercept:+.2f}. "
    f"Putting the three seasonal predictions together gives {S_STITCH.deming_slope:.2f}x "
    f"{S_STITCH.deming_intercept:+.2f} overall, because each season lands at a different offset. Same {len(COHORT)} "
    "Addis filters as slides 3 and 8.")

# 11 conclusions
text_slide(
    "Conclusions",
    [f"EC in Addis Ababa is very high, about {ec_x_improve:.0f}× typical IMPROVE, and highest in the Kiremt rainy season.",
     "Current FTIR calibrations poorly predict Addis EC, and also the IMPROVE filters whose spectra most resemble Addis.",
     f"Addis filters carry ~{ratio:.0f}× more EC than typical IMPROVE filters; most SPARTAN sites are also beyond IMPROVE.",
     f"Baseline correction helps: the Addis intercept goes from {S_ADDIS.deming_intercept:.1f} to {S_BASE.deming_intercept:.1f} µg/m³.",
     "Season-specific calibrations help in Kiremt; a calibration explorer is helping us search for better ones."],
    "To sum up. Addis has very high EC, and the calibrations built on IMPROVE do not predict it well, for Addis and "
    "for the IMPROVE filters whose spectra most resemble Addis. Addis filters are far heavier than what the "
    "calibration has seen. "
    "Baseline correction is a real improvement, season-specific calibrations help in one season, and we have a tool "
    "to test many more options. I would love your ideas on what to try next.",
    "Open for discussion: VIBES baseline; the 14 Oct k-sweep (Addis intercept gets more negative with more PLS "
    "factors: spline -0.1 to -1.6, VIBES -0.3 to -7.0 µg/m³ at k 2 to 20); OC/fAbs by site (Addis lowest of the "
    "IMPROVE sites, a possible reason; not on a slide); boundary-layer normalisation; Chile sites after AAAR. "
    "From the 15 Oct practice talk (not run yet): EC-to-PM2.5 mass ratio as a calibration-set selection criterion; "
    "check TOR optical pyrolysis (OP) on fire-affected and Addis-like IMPROVE filters.",
    size=16)

out = HERE / "aaar_practice_2026-10-15.pptx"
prs.save(out)

# ---------------------------------------------------------------- verify
chk = Presentation(out)
assert len(chk.slides) == 11
for i, sl in enumerate(chk.slides, 1):
    assert sl.notes_slide.notes_text_frame.text.startswith("SAY:"), i
    txt = sl.notes_slide.notes_text_frame.text + " ".join(sh.text_frame.text for sh in sl.shapes if sh.has_text_frame)
    assert "—" not in txt, f"em dash on slide {i}"
n_pics = sum(1 for sl in chk.slides for sh in sl.shapes if sh.shape_type == 13)
assert n_pics == 9, n_pics
print(f"wrote {out.relative_to(ROOT)}: {len(chk.slides)} slides, {n_pics} figures; "
      f"loading ratio {ratio:.1f}x, {100 * (ad_mass > p99).mean():.0f}% of Addis above IMPROVE p99")

# ---------------------------------------------------------------- copy without speaker notes (to share)
from pptx.opc.constants import RELATIONSHIP_TYPE as RT  # noqa: E402

clean = Presentation(out)
for sl in clean.slides:
    for rId, rel in list(sl.part.rels.items()):
        if rel.reltype == RT.NOTES_SLIDE:
            sl.part.drop_rel(rId)
clean_out = HERE / "aaar_practice_2026-10-15_no_notes.pptx"
clean.save(clean_out)
chk = Presentation(clean_out)
assert len(chk.slides) == 11 and not any(sl.has_notes_slide for sl in chk.slides)
print(f"wrote {clean_out.relative_to(ROOT)}: {len(chk.slides)} slides, no speaker notes")
