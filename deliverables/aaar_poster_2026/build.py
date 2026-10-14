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
"""

from pathlib import Path
import sys

import matplotlib

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

# storyline.md numbers must reproduce
for got, want in [(S_ADDIS.deming_slope, 2.30), (S_ADDIS.deming_intercept, -6.09), (S_ADDIS.R2, 0.78),
                  (S_POOL.deming_slope, 0.93), (S_POOL.deming_intercept, 0.02), (S_POOL.R2, 0.73),
                  (S_AN_REF.deming_slope, 0.91), (S_AN_REF.R2, 0.82),
                  (S_AN.deming_slope, 0.48), (S_AN.R2, 0.46), (S_DRY.deming_slope, 0.44), (S_DRY.R2, 0.33)]:
    assert abs(got - want) < 0.006, (got, want)
assert int(S_ADDIS.n) == 189 and int(S_POOL.n) == 2078 and int(S_AN.n) == 97

ad = pd.read_csv(M / "addis_predictions.csv")
ad = ad[ad.fit_id == "deployed|SPARTAN|current SPARTAN calibration"].dropna(subset=["ftir_ec_ugm3", "Fabs"])
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
           f"OLS: {s.ols_slope:.2f}x {s.ols_intercept:+.2f}\n"
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
ax.set_xlabel("HIPS EC = Fabs / MAC 10 (µg/m³)")
ax.set_ylabel("SPARTAN FTIR EC, current calibration (µg/m³)")
ax.legend(loc="lower right", fontsize=10.5)
stat_box(ax, S_ADDIS)
save(fig, "f1_addis_problem.png")

# ---------------------------------------------------------------- f2 IMPROVE held-out: the method works
fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.scatter(pool.tor_ec_ugm3, pool.ftir_ec_ugm3, s=9, color=GREY, alpha=.45, lw=0)
lim = 1.2
lines(ax, S_POOL, lim)
ax.set_ylim(-0.1, lim)
ax.set_xlabel("TOR EC (µg/m³)")
ax.set_ylabel("FTIR EC (µg/m³)")
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
           label="most Addis-like (97)")
lines(ax, S_AN_REF, lim, ACCENT)
ax.set_xlabel("HIPS EC = Fabs / MAC 10 (µg/m³)"); ax.set_ylabel("TOR EC (µg/m³)")
ax.set_title("The two reference methods agree", loc="left", fontsize=12.5)
stat_box(ax, S_AN_REF)
ax.legend(loc="lower right", fontsize=10)
ax = axes[1]
ax.scatter(pool.tor_ec_ugm3, pool.ftir_ec_ugm3, s=7, color=GREY, alpha=.25, lw=0)
ax.scatter(an.tor_ec_ugm3, an.ftir_ec_ugm3, s=26, color=ACCENT, alpha=.9, edgecolor="white", lw=.4)
lines(ax, S_AN, lim, ACCENT)
ax.set_ylim(-0.1, lim)
ax.set_xlabel("TOR EC (µg/m³)"); ax.set_ylabel("FTIR EC (µg/m³)")
ax.set_title("FTIR reads about half of TOR", loc="left", fontsize=12.5)
stat_box(ax, S_AN, extra=f"\nDry-season analogs: {S_DRY.deming_slope:.2f}x, R² {S_DRY.R2:.2f}")
fig.tight_layout()
save(fig, "f3_addis_like_improve.png")

# ---------------------------------------------------------------- public SPARTAN chemistry (all sites)
# data/spartan/raw = the SPARTAN public FilterBased ChemSpecPM25 release (v3.1.4, Aug 2026). The older in-house
# unified_filter_dataset reports ETAD OC about 1.96x lower than this release, filter for filter, so it is not used.
# SPARTAN has no thermal-optical carbon: OC/EC/OM are FTIR products (method 218), BC is HIPS (method 221).
SP_RAW = ROOT / "data/spartan/raw/FilterBased/ChemSpecPM25"


def load_spartan_site(path):
    d = pd.read_csv(path, comment="#", low_memory=False)
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
sp_all["oc_ec"] = sp_all["OC PM2.5"] / sp_all["BC PM2.5"].where(sp_all["BC PM2.5"] > 0)
etad = sp_all[sp_all.Site_Code.eq("ETAD")]
ad_oe = etad.oc_ec[etad.oc_ec > 0]

# ---------------------------------------------------------------- f4 loading per filter
refs = pd.read_csv(M / "improve_references.csv")
ev = pd.read_csv(M / "addis_evaluation.csv")
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

# ---------------------------------------------------------------- f5 OC/EC continuum, every IMPROVE and SPARTAN site
# One EC definition for both networks: HIPS absorption / MAC 10 (IMPROVE fAbs, SPARTAN BC PM2.5).
# OC: IMPROVE thermal-optical, SPARTAN FTIR. Site medians; IMPROVE sites need >= 50 filters, SPARTAN >= 10.

imp_net = pd.read_pickle(ROOT / "research/spartan/.cache/improve_chemistry_clean_cache.pkl")
imp_net = imp_net[(imp_net.fAbs_Val > 0) & (imp_net.OCf_Val > 0)]
imp_net = imp_net.assign(oc_ec=imp_net.OCf_Val / (imp_net.fAbs_Val / MAC))


def site_table(df, site, val, min_n):
    t = df[[site, val]].dropna().groupby(site)[val].agg(
        n="size", median="median", p25=lambda v: v.quantile(.25), p75=lambda v: v.quantile(.75))
    return t[t.n >= min_n]


imp_sites = site_table(imp_net, "SiteCode", "oc_ec", 50).assign(network="IMPROVE")
sp_sites = site_table(sp_all[sp_all.oc_ec > 0], "Site_Code", "oc_ec", 10).assign(network="SPARTAN")
sp_sites["name"] = sp_all.groupby("Site_Code").site_name.first().reindex(sp_sites.index)
cont = pd.concat([imp_sites, sp_sites]).sort_values("median").reset_index(names="site")
cont["rank"] = np.arange(len(cont))
n_below_imp_min = int((sp_sites["median"] < imp_sites["median"].min()).sum())
sp_below_imp_med = int((sp_sites["median"] < imp_sites["median"].median()).sum())
etad_rank_all = int(cont.index[cont.site.eq("ETAD")][0]) + 1
assert sp_sites.loc["ETAD", "median"] < imp_sites["median"].min()

fig, ax = plt.subplots(figsize=(12.4, 5.2))
im = cont[cont.network.eq("IMPROVE")]
ax.vlines(im["rank"], im.p25, im.p75, color="#D8D5CE", lw=1)
ax.scatter(im["rank"], im["median"], s=12, color=GREY, zorder=3, label=f"IMPROVE sites (n {len(im)})")
sp = cont[cont.network.eq("SPARTAN")]
ax.vlines(sp["rank"], sp.p25, sp.p75, color=PURPLE, alpha=.35, lw=1.2)
ax.scatter(sp["rank"], sp["median"], s=95, marker="*", color=PURPLE, zorder=4, edgecolor="white", lw=.4,
           label=f"SPARTAN sites (n {len(sp)})")
ea = cont[cont.site.eq("ETAD")].iloc[0]
ax.scatter([ea["rank"]], [ea["median"]], s=320, marker="*", color=AMBER, zorder=5, edgecolor=INK, lw=.6,
           label="Addis Ababa")
# label text sits below the curve with a leader line; (dx in ranks, y of the text)
LABEL = {"AEAZ": ("Abu Dhabi", 14, 0.25), "ETAD": ("Addis Ababa", 14, 0.75), "CLST": ("Santiago", 22, 1.0),
         "INDH": ("Delhi", 26, 1.55), "CLTA": ("Talca", 30, 2.05), "CHTS": ("Beijing", 26, 2.45),
         "USPA": ("Pasadena", 14, 3.0)}
for code, (lab, dx, ty) in LABEL.items():
    r = cont[cont.site.eq(code)]
    if len(r):
        r = r.iloc[0]
        ax.annotate(lab, (r["rank"], r["median"]), xytext=(r["rank"] + dx, ty), textcoords="data",
                    fontsize=10, color=INK if code == "ETAD" else PURPLE, va="center",
                    fontweight="bold" if code == "ETAD" else "normal",
                    arrowprops=dict(arrowstyle="-", color="#B9B5AC", lw=.8, shrinkA=2, shrinkB=4))
ax.set_ylim(0, 10.5)
ax.set_xlim(-3, len(cont) + 2)
ax.set_xlabel("All sites, ranked by median OC/EC")
ax.set_ylabel("OC / EC  (EC from HIPS absorption / MAC 10)")
ax.legend(loc="upper left", fontsize=10.5)
save(fig, "f5_ocec_site_continuum.png")
cont.round(3).to_csv(FIG / "f5_ocec_site_continuum.csv", index=False)

# ---------------------------------------------------------------- f6 FTIR/TOR vs OC/EC (held-out IMPROVE)
# x uses HIPS EC, not TOR EC, so x and y (FTIR/TOR) do not share a denominator
pl = pool.assign(ratio=pool.ftir_ec_ugm3 / pool.tor_ec_ugm3, load=pool.fabs_Mm1 / MAC * pool.tor_volume_m3,
                 x=pool.tor_oc_ec * pool.tor_ec_ugm3 / (pool.fabs_Mm1 / MAC))
pl = pl[(pl.tor_ec_ugm3 > 0) & (pl.tor_oc_ec > 0) & (pl.fabs_Mm1 > 0)]
al = an.assign(ratio=an.ftir_ec_ugm3 / an.tor_ec_ugm3, x=an.tor_oc_ugm3 / (an.fabs_Mm1 / MAC))
al = al[(al.tor_ec_ugm3 > 0) & (al.tor_oc_ugm3 > 0) & (al.fabs_Mm1 > 0)]
OBINS = np.array([0.5, 1.5, 2.5, 3.5, 5, 7, 10, 20, 60])


def binned(d, min_n=8):
    g = d.groupby(pd.cut(d.x, OBINS), observed=True).ratio.agg(["median", "size"])
    g = g[g["size"] >= min_n]
    return np.array([np.sqrt(i.left * i.right) for i in g.index]), g["median"].to_numpy(), g


pm, pr, pg = binned(pl)
am, ar, ag = binned(al)
oe_lo, oe_hi = np.percentile(ad_oe, [5, 95])
fig, ax = plt.subplots(figsize=(10.5, 5.0))
ax.axvspan(oe_lo, oe_hi, color=AMBER, alpha=.18, lw=0)
ax.text(np.sqrt(oe_lo * oe_hi), 1.95, "where Addis\nfilters sit\n(middle 90 %)", ha="center", va="top",
        fontsize=10.5, color="#8A6420")
ax.scatter(pl.x, pl.ratio.clip(upper=1.98), s=8, color=GREY, alpha=.3, lw=0, label=f"held-out IMPROVE (n {len(pl):,})")
ax.scatter(al.x, al.ratio.clip(upper=1.98), s=24, color=ACCENT, alpha=.85, edgecolor="white", lw=.4,
           label=f"most Addis-like IMPROVE (n {len(al)})")
ax.plot(pm, pr, color=INK, lw=2.2, marker="o", ms=5, label="median, held-out IMPROVE")
ax.plot(am, ar, color=ACCENT, lw=2.2, marker="o", ms=5, label="median, Addis-like")
ax.axhline(1, color=GREY, ls=":", lw=1.2)
ax.text(55, 1.03, "FTIR = TOR", fontsize=9.5, color=GREY, ha="right")
ax.set_xscale("log"); ax.set_xlim(0.5, 60); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])
ax.set_xlabel("OC / EC  (EC from HIPS absorption / MAC 10)")
ax.set_ylabel("FTIR EC / TOR EC")
ax.legend(loc="lower right", fontsize=10, framealpha=.9, frameon=True, edgecolor="white")
save(fig, "f6_ratio_vs_ocec.png")
lo_pool = float(pl[pl.x <= 2.5].ratio.median()); n_lo = int((pl.x <= 2.5).sum())
mid_pool = float(pl[(pl.x > 3.5) & (pl.x <= 7)].ratio.median())
hi_pool = float(pl[pl.x > 10].ratio.median())
within = pl.assign(lb=pd.cut(pl.load, [0, 4, 8, 16, 1000]), ob=pd.cut(pl.x, [0, 2.5, 6, 1e4])) \
    .pivot_table(index="lb", columns="ob", values="ratio", aggfunc="median", observed=True)
print("ratio vs OC/EC(HIPS), pool:\n", pg.round(2), "\nanalogs:\n", ag.round(2), "\nwithin loading bins:\n", within.round(2))
print(f"continuum: {len(imp_sites)} IMPROVE, {len(sp_sites)} SPARTAN; ETAD {sp_sites.loc['ETAD','median']:.2f}, "
      f"rank {etad_rank_all}/{len(cont)}; SPARTAN below lowest IMPROVE site: {n_below_imp_min}; "
      f"below IMPROVE median: {sp_below_imp_med}")

# ---------------------------------------------------------------- f7 Addis chemistry by season (public release)
e = etad.copy()
month = e.date.dt.month
e["season"] = np.select([month.isin([10, 11, 12, 1, 2]), month.isin([3, 4, 5])],
                        ["Dry (Oct-Feb)", "Belg (Mar-May)"], "Kiremt (Jun-Sep)")
e = e[e["Filter PM2.5 mass"].notna()]
g = lambda c: e.get(c, pd.Series(np.nan, index=e.index))
soil = (2.20 * g("Aluminum PM2.5") + 2.49 * g("Silicon PM2.5") + 1.63 * g("Calcium PM2.5")
        + 2.42 * g("Iron PM2.5") + 1.94 * g("Titanium PM2.5"))
comp = pd.DataFrame({
    "Sulfate": g("Sulfate Ion PM2.5"), "Nitrate": g("Nitrate Ion PM2.5"), "Ammonium": g("Ammonium Ion PM2.5"),
    "Organic matter (FTIR)": g("OM PM2.5"), "Black carbon (HIPS)": g("BC PM2.5"), "Dust (soil formula)": soil,
    "season": e.season, "PM2.5": g("Filter PM2.5 mass")})
COMP_COLORS = {"Sulfate": ACCENT, "Nitrate": BLUE, "Ammonium": AMBER, "Organic matter (FTIR)": "#5E8C4A",
               "Black carbon (HIPS)": INK, "Dust (soil formula)": "#A6774B"}
med = comp.groupby("season")[list(COMP_COLORS) + ["PM2.5"]].median().reindex(SEASONS)
counts = comp.groupby("season").size().reindex(SEASONS)
fig, ax = plt.subplots(figsize=(9.5, 5.0))
bottom = np.zeros(3)
for c, col in COMP_COLORS.items():
    ax.bar(range(3), med[c].fillna(0), bottom=bottom, color=col, width=.62, label=c, edgecolor="white", lw=.6)
    bottom += med[c].fillna(0).to_numpy()
ax.scatter(range(3), med["PM2.5"], marker="_", s=900, color=INK, lw=2, zorder=5, label="PM2.5 mass (median)")
ax.set_xticks(range(3), [f"{SEASON_LABEL[s]}\nn {counts[s]}" for s in SEASONS])
ax.set_ylabel("Median concentration (µg/m³)")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10.5)
save(fig, "f7_addis_chemistry.png")
chem_summary = med.round(2)
chem_summary.to_csv(FIG / "f7_addis_chemistry_medians.csv")

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
title_box(s, "FTIR elemental carbon calibrated on US filters transfers poorly to heavily loaded, "
             "Ethiopia-like samples, in Addis Ababa and in the US network itself", size=28, top=2.2)
tb = s.shapes.add_textbox(Inches(0.55), Inches(4.4), Inches(12), Inches(1.2))
p = tb.text_frame.paragraphs[0]; p.text = "Ahmad Jalil, Ann M. Dillner, Satoshi Takahama  [FILL IN: affiliations, co-authors]"
p.font.size = Pt(16); p.font.color.rgb = RGBColor(0x8F, 0x8C, 0x84)
notes(s,
      "Today I want to show you that a problem we first saw in Addis Ababa is not only an Ethiopia problem. "
      "First a bit of context on what the air in Addis is made of. Then the problem itself: FTIR elemental carbon "
      "in Addis correlates with an independent measurement but sits on the wrong scale. Then the twist: the same "
      "FTIR method works well on typical US filters, but not on the US filters that look most like Addis. "
      "Last, why: those samples are much heavier and much more soot-dominated than what the calibration learned from, and FTIR under-reads EC-rich filters.",
      "This is a practice run of the AAAR poster story, laid out as slides first (Ann's suggestion). "
      "PLS calibration-search material (specification curve, cohorts, VIBES baseline, k-sweep) is deliberately "
      "left out of the main slides; it is for discussion. Leave at least 20 minutes for questions.")

# 2 chemistry context
fig_slide(
    "Addis Ababa PM2.5 peaks in the Kiremt rainy season; black carbon, organics and dust are each large shares",
    FIG / "f7_addis_chemistry.png",
    "This is what PM2.5 in Addis Ababa is made of, by season. Each bar is the median of each component. "
    "Ethiopia has three seasons: a long dry season, the short Belg rains, and the main Kiremt rainy season. "
    "Kiremt has the highest concentrations, partly because the boundary layer is shallow then, so pollution is "
    "trapped close to the ground. Organic matter and dust are big pieces, and black carbon is a few µg/m³, "
    "far above typical US rural levels. Keep that in mind: these are very different filters from what US networks see.",
    "Medians of each component are stacked; medians do not add up exactly to the median PM2.5 mass (black tick), "
    "and unmeasured mass (water, other ions) is not shown. Organic matter is SPARTAN's FTIR-derived OM. "
    "Black carbon is the SPARTAN HIPS-based BC, not FTIR EC. Dust uses the IMPROVE soil formula "
    "(2.20 Al + 2.49 Si + 1.63 Ca + 2.42 Fe + 1.94 Ti). Season medians: see figures/f7_addis_chemistry_medians.csv. Source: SPARTAN public release v3.1.4 (Aug 2026). "
    "Boundary layer: ERA5 median daily BLH ~370-410 m in Kiremt vs 650-800 m in Dry; normalising by BLH is on the list, not done. "
    "Season names: 'Bega' for Dry is from one source; the literature mostly says Dry / Belg / Kiremt.")

# 3 the problem
fig_slide(
    "In Addis Ababa, FTIR EC tracks HIPS EC closely but on the wrong scale",
    FIG / "f1_addis_problem.png",
    "Here is the problem. Each point is one Addis filter. Along the bottom is EC from HIPS, an optical "
    f"absorption measurement. Up the side is EC from FTIR with the current SPARTAN calibration. The correlation is "
    f"strong, R² {S_ADDIS.R2:.2f}, but the line has a slope of {S_ADDIS.deming_slope:.1f} and an intercept of "
    f"{S_ADDIS.deming_intercept:.0f} µg/m³. The lower-EC filters, mostly dry season, come out too low, and the high Kiremt filters come out too high. "
    "So FTIR sees the EC signal, but the calibration maps it onto the wrong scale.",
    "Deming regression, HIPS EC = Fabs / MAC 10 on x. The Deming intercept does not depend on MAC; slope does. "
    "Why HIPS and not TOR: SPARTAN filters have no thermal-optical TOR analysis, so HIPS is the independent reference at Addis. "
    f"n {int(S_ADDIS.n)} filters with both. Current SPARTAN calibration = the deployed network calibration "
    "(built on IMPROVE filters). Season colours: Dry amber, Belg purple, Kiremt blue. "
    "Expected question: 'is HIPS right?' HIPS and TOR agree on IMPROVE (next slides), including on the Addis-like filters.")

# 4 method works
fig_slide(
    "On typical US filters the same FTIR method predicts EC well",
    FIG / "f2_improve_heldout.png",
    "Now the US. The IMPROVE network has both FTIR and the thermal reference method, TOR. We built the "
    "calibration on IMPROVE filters and tested it on filters from sites the calibration never saw. "
    f"On these {int(S_POOL.n):,} held-out filters, FTIR matches TOR well: slope {S_POOL.deming_slope:.2f}, "
    f"intercept close to zero, R² {S_POOL.R2:.2f}. This is the established method working as intended.",
    "Spline (AIRSpec) baseline-corrected spectra, all-IMPROVE calibration on lot 251, read out on held-out sites "
    "(site-disjoint split). Same baseline and calibration approach as the next slide; only the samples change. "
    f"TOR vs HIPS on the same held-out filters: Deming {S_POOL_REF.deming_slope:.2f}x {S_POOL_REF.deming_intercept:+.2f}, "
    f"R² {S_POOL_REF.R2:.2f}. Axis clipped at 1.2 µg/m³ for readability; count of clipped points is printed on the figure. "
    "Units µg/m³; IMPROVE samples ~33 m³ per filter, SPARTAN ~7 m³, so concentrations understate the loading gap (slide 6).")

# 5 Addis-like IMPROVE
fig_slide(
    "The same calibration under-reads the US filters whose spectra look most like Addis",
    FIG / "f3_addis_like_improve.png",
    "Here is the key result. Out of the held-out US filters, we picked the 97 whose infrared spectra look most "
    "like Addis Ababa, in red. On the left, the two reference methods, TOR and HIPS, still agree on them: "
    f"slope {S_AN_REF.deming_slope:.2f}, R² {S_AN_REF.R2:.2f}. So the reference values are fine. On the right, "
    f"FTIR reads only about half of TOR: slope {S_AN.deming_slope:.2f}, R² {S_AN.R2:.2f}. For the analogs of the dry "
    f"season it is worse, slope {S_DRY.deming_slope:.2f}. Same calibration, same baseline. Only the samples changed. "
    "So the Addis problem also shows up inside the US network, for its most Addis-like filters.",
    "Analogs: the 500 IMPROVE filters whose spectra correlate best with the Addis spectra (Pearson r on the "
    "spline-corrected spectra); 97 of them come from held-out sites and are shown. All 97 are from the test split "
    "(88 lot 251, 9 lot 248), so none trained the calibration. Grey = all held-out IMPROVE for context. "
    "Deming fit drawn in red; OLS in the box. Caveat: 'most Addis-like' depends on the similarity metric and window; "
    "the spline and VIBES baselines choose mostly different analogs (shared of 500: Dry 31, Belg 128, Kiremt 166). "
    "Ann's self-prediction question: IMPROVE predicts IMPROVE well overall (slide 4) and the analogs fit well "
    "when they are themselves the calibration set (CV R² 0.5-0.9, bias ~0); the failure is transfer.")

# 6 loading
fig_slide(
    f"Addis filters carry about {ratio:.0f} times more EC per filter than a typical IMPROVE filter",
    FIG / "f4_loading_per_filter.png",
    "Why would that happen? First, loading. This shows how much EC sits on each filter, on a log scale. "
    f"Grey is the IMPROVE calibration pool, amber is Addis. The median Addis filter has about {ratio:.0f} times "
    f"more EC on it than the median IMPROVE filter, and {100 * (ad_mass > p99).mean():.0f} percent of Addis filters are above "
    "IMPROVE's 99th percentile. The calibration has almost never seen filters like these.",
    "Per filter = HIPS Fabs / MAC 10 × sampled volume, so both networks are on the same footing "
    "(IMPROVE ~33 m³ per filter at 22 L/min, SPARTAN ~7 m³ at 5 L/min). In concentration the gap looks like ~41× "
    "because of the volume difference; per filter is the fair comparison for what the FTIR sees. "
    "Loading is not the driver on its own (next two slides): at fixed loading, OC/EC still matters.")

# 7 OC/EC site continuum
fig_slide(
    f"Addis is more EC-rich than every IMPROVE site, and {sp_below_imp_med} of {len(sp_sites)} SPARTAN sites sit below the typical US site",
    FIG / "f5_ocec_site_continuum.png",
    "Second, composition. Every site in both networks, ranked by its typical ratio of organic to elemental carbon. "
    f"Grey dots are the {len(imp_sites)} IMPROVE sites; purple stars are the {len(sp_sites)} SPARTAN sites around the "
    f"world. Addis, the big gold star, has an OC/EC of about {sp_sites.loc['ETAD', 'median']:.1f}, lower than every "
    f"single US site. {sp_below_imp_med} of the {len(sp_sites)} SPARTAN sites sit below the typical US site. So the "
    "US calibration network covers the organic-rich end well, and the soot-rich end, where much of the world lives, "
    "barely at all.",
    "One EC definition for both networks so the ranking does not lean on FTIR EC: EC = HIPS absorption / MAC 10 "
    "(IMPROVE fAbs; SPARTAN 'BC PM2.5', HIPS, method 221). OC: IMPROVE thermal-optical; SPARTAN 'OC PM2.5' is FTIR "
    "(method 218): SPARTAN has no thermal carbon, so the earlier 'thermal' labels on this plot were wrong. "
    "Site medians with IQR; IMPROVE sites with >= 50 filters (FED export, 1988-2025), SPARTAN >= 10 "
    f"(public release v3.1.4, Aug 2026). Addis ranks {etad_rank_all} of {len(cont)} sites; {n_below_imp_min} SPARTAN "
    "sites sit below the lowest IMPROVE site (Abu Dhabi is lowest overall). The older in-house file had Addis OC "
    "about 2x lower than the current release; this slide uses the release. Full table: figures/f5_ocec_site_continuum.csv.")

# 8 FTIR/TOR vs OC/EC
fig_slide(
    "On US filters, FTIR under-reads EC as OC/EC falls, and Addis sits further down still",
    FIG / "f6_ratio_vs_ocec.png",
    "This connects the two. Each point is one held-out US filter: its OC/EC along the bottom, and FTIR divided by "
    f"TOR up the side, where 1 means they agree. For typical US filters they agree, about {mid_pool:.1f}. On the most "
    f"EC-rich US filters, OC/EC below 2.5, FTIR reads only about {lo_pool:.1f} of TOR, and the Addis-like filters in red "
    "follow the same line. The shaded band is where Addis sits, at the far low end. So the more soot-dominated the "
    "sample, the more FTIR under-reads, and Addis is beyond the edge of the US data.",
    "x uses the same OC/EC as slide 7 (EC from HIPS), so x and y do not share TOR EC in the denominator. "
    "Same all-IMPROVE lot-251 spline calibration and held-out sites as slides 4 and 5; ratios above 2 drawn at the top. "
    f"Binned medians: OC/EC <= 2.5 {lo_pool:.2f} (n {n_lo}), 3.5 to 7 {mid_pool:.2f}, > 10 {hi_pool:.2f}. "
    "High ratios at high OC/EC are partly the small positive intercept on low-EC filters. The OC/EC trend holds within "
    "most loading bands (it flattens above ~16 µg per filter, where n is small). TOR's OC/EC split can still couple OC "
    "and TOR EC (charring), so treat the slope of the trend as indicative. No Addis points: Addis has no TOR.")

# 8 takeaway
text_slide(
    "Validating on the whole US network hides a calibration gap that matters for the Global South",
    ["FTIR EC calibrated on IMPROVE works well on typical US filters.",
     "It under-reads the US filters that look most like Addis Ababa, even though TOR and HIPS agree on them.",
     "FTIR under-reads EC-rich filters, and Addis filters are heavier and more EC-rich than almost any US filter.",
     "So the Addis bias is a coverage problem that also exists inside the US network, not a quirk of one site."],
    "To sum up. The established FTIR method works on the samples it was built from. It struggles on samples "
    "that look like Addis, and those exist inside the US network too, they are just rare, so whole-network "
    "validation never flags them. That matters for anyone using US-built calibrations in polluted, dusty, "
    "smoke-rich places, which is most of the Global South. Next we are working out which spectral features "
    "drive the shift and how to choose calibration samples that cover it.",
    "Open work, for discussion only: VIBES full-range baseline (more consistent on IMPROVE, worse Addis intercept); "
    "the 14 Oct k-sweep (Addis intercept falls as PLS factors increase: spline -0.1 to -1.6, VIBES -0.3 to -7.0 "
    "µg/m³ at k 2 to 20; smoke-like IMPROVE filters do not drift the same way); boundary-layer normalisation; "
    "Chile sites after AAAR.")

out = HERE / "aaar_practice_2026-10-15.pptx"
prs.save(out)

# ---------------------------------------------------------------- verify
chk = Presentation(out)
assert len(chk.slides) == 9
for i, sl in enumerate(chk.slides, 1):
    assert sl.notes_slide.notes_text_frame.text.startswith("SAY:"), i
    txt = sl.notes_slide.notes_text_frame.text + " ".join(sh.text_frame.text for sh in sl.shapes if sh.has_text_frame)
    assert "—" not in txt, f"em dash on slide {i}"
n_pics = sum(1 for sl in chk.slides for sh in sl.shapes if sh.shape_type == 13)
assert n_pics == 7, n_pics
print(f"wrote {out.relative_to(ROOT)}: {len(chk.slides)} slides, {n_pics} figures; "
      f"loading ratio {ratio:.1f}x, {100 * (ad_mass > p99).mean():.0f}% of Addis above IMPROVE p99")
print(chem_summary.to_string())

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
assert len(chk.slides) == 9 and not any(sl.has_notes_slide for sl in chk.slides)
print(f"wrote {clean_out.relative_to(ROOT)}: {len(chk.slides)} slides, no speaker notes")
