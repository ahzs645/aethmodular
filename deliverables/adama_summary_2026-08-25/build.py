"""Adama summary for Christian & Sina: 3 slides, house style, no em dashes.

Ann's ask (2026-08-19): a 2-3 slide case that Adama is unlike Addis, so more
Adama Teflon samples will not arbitrate the Addis discrepancy. Data: committed
tables only (ftir16 TOR tables + spartan_ec_2026_06_16 FTIR-vs-TOR comparison).

Run: MPLBACKEND=Agg ~/anaconda3/bin/python build.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
INKC = "#22252A"
GREY, BLUE, ACCENT, AMBER = "#8F8C84", "#2C6E9E", "#B23327", "#C49442"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E8E6E1", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "white",
                     "savefig.dpi": 165})

# ---------------------------------------------------------------- figure 2
cmp_ = pd.read_csv(REPO / "research/spartan_ec_2026_06_16/tables/"
                          "adama_ec_calibration_comparison.csv")
cmp_ = cmp_.sort_values("date").reset_index(drop=True)
labels = [f"{r.FilterId}\n{r.date[5:]}" for r in cmp_.itertuples()]
xs = np.arange(len(cmp_))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3),
                             gridspec_kw={"width_ratios": [1.5, 1]})
for name, col, c in (("production (lot 241a)", "EC_general", BLUE),
                     ("biomass, tool export", "EC_biomass_tool", AMBER),
                     ("biomass, local rebuild", "EC_biomass_local_sig", GREY)):
    r = cmp_[col] / cmp_["ECTR"]
    a1.scatter(xs, r, s=52, color=c, label=name, zorder=3)
a1.axhline(1.0, color=INKC, lw=1.4)
a1.set_xticks(xs)
a1.set_xticklabels(labels, fontsize=8.5)
a1.set_ylabel("FTIR EC / quartz TOR EC (same-date pair)")
a1.set_ylim(0, 6)
med = (cmp_["EC_general"] / cmp_["ECTR"]).median()
a1.text(0.03, 0.95, f"production median = {med:.2f}",
        transform=a1.transAxes, va="top", fontsize=9.5, color=BLUE)
a1.legend(frameon=False, fontsize=8.5, loc="upper right")

a2.bar(xs, cmp_["char_soot"], width=0.55, color=ACCENT)
a2.axhline(1.0, color=INKC, lw=1.2, ls=":")
a2.text(0.03, 0.95, "char-dominated above 1.0", transform=a2.transAxes,
        va="top", fontsize=9, color="#6E7178")
a2.set_xticks(xs)
a2.set_xticklabels([r.FilterId for r in cmp_.itertuples()], fontsize=8.5)
a2.set_ylabel("char-EC / soot-EC, (EC1-OP)/(EC2+EC3)")
a2.set_ylim(0, 1.1)
fig.tight_layout()
fig.savefig(FIG / "f_adama_ftir_vs_tor.png")
plt.close(fig)

# ---------------------------------------------------------------- figure 3
# side-by-side crossplots: Addis deployed FTIR EC vs HIPS, Adama FTIR EC vs TOR
import sys
sys.path.insert(0, str(REPO / "research/ftir_ec_phase3/scripts"))
import phase3_common as pc
ev = next(x for x in pc.load_addis_evaluation()
          if isinstance(x, pd.DataFrame) and "MediaId" in getattr(x, "columns", []))
ad = ev.dropna(subset=["EC_deployed_ugm3", "Fabs"])
ax_x = ad["Fabs"].to_numpy(float) / 10.0
ax_y = ad["EC_deployed_ugm3"].to_numpy(float)


def deming(x, y, lam=2.96):
    mx, my = x.mean(), y.mean()
    sxx, syy = ((x - mx) ** 2).mean(), ((y - my) ** 2).mean()
    sxy = ((x - mx) * (y - my)).mean()
    b = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2
                                   + 4 * lam * sxy ** 2)) / (2 * sxy)
    return b, my - b * mx


b_dem, a_dem = deming(ax_x, ax_y)
b_add, a_add = np.polyfit(ax_x, ax_y, 1)
# the committed ftir_17 deployed line is OLS 1.90x-4.17; Deming lambda* is steeper
assert abs(b_add - 1.90) < 0.05 and abs(a_add + 4.17) < 0.15, (b_add, a_add)

fig, (p1, p2) = plt.subplots(1, 2, figsize=(10.6, 5.0))
lim1 = max(ax_x.max(), ax_y.max()) * 1.06
p1.plot([0, lim1], [0, lim1], ls=":", color=GREY, lw=1.2)
p1.scatter(ax_x, ax_y, s=16, color=BLUE, alpha=0.6)
xs = np.linspace(0, lim1, 20)
p1.plot(xs, b_add * xs + a_add, color=INKC, lw=1.8)
p1.text(0.03, 0.97, f"OLS    {b_add:.2f}x {a_add:+.2f}\nDeming {b_dem:.2f}x {a_dem:+.2f}\nMAC 10, n = {len(ad)}",
        transform=p1.transAxes, va="top", fontsize=9.5, family="monospace",
        bbox=dict(fc="white", ec="#DDDAD2", alpha=0.9))
p1.set_xlim(0, lim1); p1.set_ylim(min(0, a_add) * 1.1, lim1)
p1.axhline(0, color="#CFCCC5", lw=0.8)
p1.set_xlabel("HIPS EC-equivalent, Fabs/10 (ug/m3)")
p1.set_ylabel("deployed FTIR EC (ug/m3)")
p1.set_title("Addis: FTIR vs HIPS", fontsize=11, loc="left")

t_x = cmp_["ECTR"].to_numpy(float)
t_y = cmp_["EC_general"].to_numpy(float)
lim2 = max(t_x.max(), t_y.max()) * 1.12
p2.plot([0, lim2], [0, lim2], ls=":", color=GREY, lw=1.2)
p2.scatter(t_x, t_y, s=70, color=ACCENT, zorder=3)
for xi, yi, fid in zip(t_x, t_y, cmp_["FilterId"]):
    p2.annotate(fid, (xi, yi), textcoords="offset points", xytext=(6, 5),
                fontsize=8, color="#6E7178")
med_r = float(np.median(t_y / t_x))
p2.text(0.03, 0.97, f"median FTIR/TOR = {med_r:.2f}\nn = 5 (July 2024)\nno line fitted at n = 5",
        transform=p2.transAxes, va="top", fontsize=9.5, family="monospace",
        bbox=dict(fc="white", ec="#DDDAD2", alpha=0.9))
p2.set_xlim(0, lim2); p2.set_ylim(0, lim2)
p2.set_xlabel("quartz TOR EC (ug per filter, TR basis)")
p2.set_ylabel("FTIR EC, production calibration (ug per filter)")
p2.set_title("Adama: FTIR vs quartz TOR", fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(FIG / "f_adama_addis_crossplots.png")
plt.close(fig)

# ---------------------------------------------------------------- figure 4
# where the Ethiopian sites fall within the IMPROVE network: per-site medians
# of OC/EC, OC/fAbs, and fAbs/PM2.5 (the June presentation's fig08 pair,
# extended with the one axis Bishoftu can join today)
from pls_transfer import FTIRTransferPaths
LDB = Path(FTIRTransferPaths.defaults().ftir_dir) / "local_db/tables"


def improve_daily(fname, params):
    d = pd.read_csv(LDB / fname, usecols=["Site", "SampleDate", "Parameter", "Value"])
    d = d[d.Parameter.isin(params)]
    return (d.groupby(["Site", "SampleDate", "Parameter"]).Value.median()
            .unstack("Parameter"))


tor = improve_daily("results_tor.csv", ["OC", "EC"])          # ng/m3
hip = improve_daily("results_hips.csv", ["fAbs"])             # Mm-1
grv = improve_daily("results_grav.csv", ["PM2.5"])            # ng/m3
imp = tor.join(hip, how="outer").join(grv, how="outer")
imp["oc_ec"] = imp.OC / imp.EC
imp["oc_fabs"] = (imp.OC / 1000.0) / imp.fAbs                  # ug/m3 per Mm-1
imp["fabs_pm"] = imp.fAbs / (imp["PM2.5"] / 1000.0)            # Mm-1 per ug/m3
site_meds = {}
for col in ("oc_ec", "oc_fabs", "fabs_pm"):
    g = imp[col].replace([np.inf, -np.inf], np.nan).dropna()
    g = g[(g > 0)]
    counts = g.groupby(level=0).size()
    keep = counts[counts >= 100].index
    site_meds[col] = g.groupby(level=0).median().loc[keep]

import pickle
with open(REPO / "research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl", "rb") as f:
    pkl = pickle.load(f)


def pkl_param(code, param):
    d = pkl[(pkl.Site == code) & (pkl.Parameter == param)].copy()
    d = d.dropna(subset=["Concentration"])
    # ChemSpec rows carry BASE FilterIds (ETAD-0001) while HIPS/FTIR carry
    # suffixed ones (ETAD-0001-1) -- normalize to base so cross-source joins work
    d["fid"] = d["FilterId"].astype(str).str.replace(r"-(\d)$", "", regex=True)
    d = d.drop_duplicates("fid")
    ser = d.set_index("fid")["Concentration"].astype(float)
    return ser[ser > 0]


SP = {"ETAD": "Addis", "INDH": "Delhi", "CHTS": "Beijing", "USPA": "Pasadena"}
sp_rows = {}
for code, lab in SP.items():
    oc, ec = pkl_param(code, "OC_ftir"), pkl_param(code, "EC_ftir")
    fab, pm = pkl_param(code, "HIPS_Fabs"), pkl_param(code, "ChemSpec_Filter_PM2.5_mass")
    if len(pm) and pm.median() > 1000:                        # ng/m3 trap
        pm = pm / 1000.0
    j1 = pd.concat([oc, ec], axis=1, keys=["oc", "ec"]).dropna()
    j2 = pd.concat([oc, fab], axis=1, keys=["oc", "fab"]).dropna()
    j3 = pd.concat([fab, pm], axis=1, keys=["fab", "pm"]).dropna()
    sp_rows[lab] = {
        "oc_ec": float((j1.oc / j1.ec).median()) if len(j1) else None,
        "oc_fabs": float((j2.oc / j2.fab).median()) if len(j2) else None,
        "fabs_pm": float((j3.fab / j3.pm).median()) if len(j3) else None,
        "n": (len(j1), len(j2), len(j3))}

# Bishoftu: Fabs (batch) + mass concentration from the DB export
eb = pd.read_csv(
    Path(FTIRTransferPaths.defaults().spartan_hips_primary), encoding="cp1252",
    usecols=["Site", "FilterId", "FilterType", "Fabs"])
eb = eb[(eb.Site == "ETBI") & (eb.FilterType == "PM2.5")].dropna(subset=["Fabs"])
# Canonical staged copy (stage_spartan_pull.py layout); the ~/Downloads pull is gone.
ef = pd.read_csv(Path(FTIRTransferPaths.defaults().spartan_hips_primary).parents[1]
                 / "DAVIS/SPARTAN FTIR pulls/ETBI/ETBI_filters.csv",
                 encoding="utf-8-sig")
ef.columns = [c.strip('\ufeff"') for c in ef.columns]
ef["mass_conc"] = (ef.MassCollectedOnFilter.astype(float)
                   / ef.SampleVolume_m3.astype(float))
if ef["mass_conc"].median() > 1000:                            # ug vs ng guard
    ef["mass_conc"] = ef["mass_conc"] / 1000.0
ej = eb.merge(ef[["ExternalFilterId", "mass_conc"]],
              left_on="FilterId", right_on="ExternalFilterId").dropna()
ej = ej[ej.mass_conc > 0]
etbi_fabs_pm = float((ej.Fabs / ej.mass_conc).median())
sp_rows["Bishoftu"] = {"oc_ec": None, "oc_fabs": None,
                       "fabs_pm": etbi_fabs_pm, "n": (0, 0, len(ej))}

# Adama joins the OC/EC axis on the same TOR basis as IMPROVE
sp_rows["Adama"] = {"oc_ec": float(cmp_["OC_EC"].median()),
                    "oc_fabs": None, "fabs_pm": None, "n": (5, 0, 0)}
SITE_COL = {"Addis": "#F39C12", "Delhi": "#3498DB", "Beijing": "#E74C3C",
            "Bishoftu": "#7A4FA3", "Pasadena": "#2ECC71", "Adama": "#C49442"}
RATIO_PANELS = [
    ("oc_ec", "f_ratio_oc_ec.png", "OC / EC",
     "TOR basis for IMPROVE and Adama; FTIR products for the SPARTAN cities",
     "no OC yet: Bishoftu"),
    ("oc_fabs", "f_ratio_oc_fabs.png", "OC / fAbs  (ug/m3 per 1/Mm)",
     "lower = more absorption per unit organic carbon",
     "not measurable yet: Bishoftu (no OC); Adama HIPS exists (ftir_41) but is not drawn here"),
    ("fabs_pm", "f_ratio_fabs_pm.png", "fAbs / PM2.5 mass  (1/Mm per ug/m3)",
     "higher = darker aerosol per unit mass; calibration-free",
     "not measurable: Adama (no PM2.5 mass on the PTFE twins yet)"),
]
rng = np.random.default_rng(7)
for col, fname, name, sub, missing in RATIO_PANELS:
    vals = site_meds[col]
    fig, ax = plt.subplots(figsize=(11.8, 3.4))
    jitter = rng.uniform(0.70, 1.02, len(vals))
    ax.scatter(vals, jitter, s=13, color="#C9C6BF", alpha=0.8)
    ax.text(float(np.median(vals)), 1.13,
            "199 IMPROVE sites (each dot = one site median)",
            fontsize=9, color="#8F8C84", ha="center")
    def ordinal(n):
        n = int(round(n))
        sfx = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{sfx}"

    entries = sorted(((r[col], lab) for lab, r in sp_rows.items()
                      if r[col] is not None))
    span = vals.max() - vals.min()
    last_v, last_y = None, -0.24
    for v, lab in entries:
        ax.scatter([v], [0.38], s=150, color=SITE_COL[lab], zorder=5,
                   edgecolor="white", linewidth=1.3)
        ax.plot([v, v], [0.47, 0.64], color=SITE_COL[lab], lw=1.1, alpha=0.55)
        below = float((vals < v).mean() * 100)
        tag = ("lowest of all" if below == 0 else
               "above all IMPROVE" if below == 100 else
               f"{ordinal(below)} percentile")
        # alternate label rows whenever neighbors would collide
        ytxt = 0.08 if (last_v is None or v - last_v > 0.09 * span
                        or last_y == -0.26) else -0.26
        ax.text(v, ytxt, f"{lab}\n{tag}", fontsize=9.5, ha="center",
                color=SITE_COL[lab], fontweight="bold")
        last_v, last_y = v, ytxt
    ax.text(0.99, 0.92, missing, transform=ax.transAxes, fontsize=9,
            color="#8F8C84", ha="right")
    hi = [r[col] for r in sp_rows.values() if r[col] is not None]
    ax.set_xlim(0, max(list(vals) + hi) * 1.10)
    ax.set_ylim(-0.44, 1.28)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"{name}   ({sub})", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)
plt.close(fig)

# context figure reused from the group deck staging
import shutil
shutil.copy(REPO / "deliverables/ftir_group_2026-08-27/figures/f_adama_context.png",
            FIG / "f_adama_context.png")

# flatten RGB
from PIL import Image
for p in FIG.glob("*.png"):
    im = Image.open(p)
    if im.mode != "RGB":
        im.convert("RGB").save(p)

# ---------------------------------------------------------------- deck
INK = RGBColor(0x22, 0x25, 0x2A)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]


def slide(title, fig=None, lines=None, say="", notes=""):
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.22), Inches(12.5), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = INK
    if fig is not None:
        im = Image.open(fig)
        scale = min(12.2 / (im.width / 165), 5.9 / (im.height / 165), 1.35)
        w = im.width / 165 * scale
        s.shapes.add_picture(str(fig), Inches((13.333 - w) / 2),
                             Inches(1.25 + (5.9 - im.height / 165 * scale) / 2),
                             Inches(w))
    if lines:
        body = s.shapes.add_textbox(Inches(0.75), Inches(1.6), Inches(11.8), Inches(5.2))
        btf = body.text_frame
        btf.word_wrap = True
        for i, line in enumerate(lines):
            para = btf.paragraphs[0] if i == 0 else btf.add_paragraph()
            para.text = line
            para.font.size = Pt(17)
            para.font.color.rgb = INK
            para.space_after = Pt(10)
    s.notes_slide.notes_text_frame.text = f"SAY: {say}\n\nNOTES: {notes}"


slide(
    "Adama is not Addis: OC/EC sits at the IMPROVE median, and its HIPS excess is a third of Addis's",
    fig=FIG / "f_adama_context.png",
    say=("Two panels that frame the whole question. Right: the five Adama quartz "
         "filters land at OC-to-EC around six, which is the middle of the IMPROVE "
         "distribution, nowhere near the low-OC/EC regime where the Addis "
         "calibration problem lives. Left: Bishoftu, the SPARTAN site eighty "
         "kilometres from Adama, absorbs about half of what Addis does, and its "
         "FTIR-versus-HIPS crossplot shows no Addis-style offset. The Adama "
         "region simply is not the Addis aerosol."),
    notes=("OC/EC TR basis 4.63 to 7.23, IMPROVE pool median 5.54; lowest-OC/EC "
           "cut is 2.27. Bishoftu: 26 filters, median Fabs 26.9 vs Addis 47.1 "
           "inverse megameters; transfer readout 0.93x minus 0.56 with the "
           "winner calibration. Correction (2026-09-01): HIPS on the Adama PTFE "
           "twins does exist (CSU AMOD Batch 54): median Fabs 40.5 inverse "
           "megameters in July 2024, implied MAC 16.5 against quartz EC TOR "
           "versus about 47 implied at Addis (ftir_41). Bishoftu remains the "
           "regional SPARTAN anchor."))

slide(
    "OC / EC across 199 IMPROVE sites: Addis is the lowest anywhere; Adama sits at the median",
    fig=FIG / "f_ratio_oc_ec.png",
    say=("The grey cloud is every IMPROVE site, one dot per site median; the "
         "labeled dots are our sites on the same axis. Addis is the lowest "
         "value in either network, the suspect low-OC/EC signature, with "
         "Delhi and Beijing just above it. Adama, on the identical TOR basis "
         "as IMPROVE, sits at the middle of the pack: compositionally an "
         "ordinary site. Bishoftu cannot appear here yet because no OC of any "
         "kind has been measured for it."),
    notes=("IMPROVE: per-site median of site-day TOR OC/EC, sites with 100+ "
           "days (n=199). SPARTAN cities use deployed FTIR OC/EC products, so "
           "their placement partly reflects the calibration question itself; "
           "Adama uses quartz TOR (n=5, July 2024), the same method as the "
           "IMPROVE axis. Bishoftu OC: no TOR on site, and deployed FTIR OC "
           "awaits an analysis.Results pull on the VPN machine."))

slide(
    "OC / fAbs: Addis has the least organic carbon per unit absorption of any site in either network",
    fig=FIG / "f_ratio_oc_fabs.png",
    say=("The same layout from the optical side: organic carbon per unit "
         "absorption. Addis again sits below every IMPROVE site, with Delhi "
         "and Beijing hugging the bottom edge and only Pasadena inside the "
         "pack. Together with the previous slide: the problem sites are "
         "absorption-rich relative to their organics. Neither Bishoftu nor "
         "Adama can join this axis yet; it needs OC and HIPS together."),
    notes=("This is the June presentation's fig08 claim, rebuilt: Ann asked "
           "for exactly this pair, OC/EC then OC/fAbs, and predicted Addis "
           "would move from mid-pack to dead last; on these medians Addis is "
           "lowest on both. IMPROVE OC converted ng to ug; fAbs in 1/Mm. "
           "Bishoftu lacks OC; Adama lacks HIPS (quartz filters)."))

slide(
    "Absorption per unit mass: both Ethiopian sites exceed every IMPROVE site, yet only Addis has the offset",
    fig=FIG / "f_ratio_fabs_pm.png",
    say=("The calibration-free axis, and the one Bishoftu can join today: "
         "absorption per unit PM mass. Beijing and Pasadena sit near the top "
         "of the IMPROVE pack, Delhi at its edge, and both Ethiopian sites "
         "are beyond every IMPROVE site: Bishoftu at 1.6, Addis at 2.2. The "
         "kicker: Bishoftu is nearly as dark per unit mass as Addis and shows "
         "no offset. Dark aerosol alone does not produce the discrepancy; "
         "whatever does is specific to Addis."),
    notes=("Bishoftu: batch Fabs over gravimetric mass concentration from the "
           "DB export (26 filters). SPARTAN cities: HIPS_Fabs over "
           "ChemSpec_Filter_PM2.5_mass (base-FilterId join). IMPROVE: fAbs "
           "over grav PM2.5 per site-day. No FTIR anywhere in this panel, so "
           "it is immune to the calibration circularity caveat. Adama has no "
           "PTFE optics, so no point exists for it here."))

slide(
    "Side by side: the Addis crossplot is a systematic line; the Adama one scatters around 1:1",
    fig=FIG / "f_adama_addis_crossplots.png",
    say=("The two crossplots next to each other. Left: Addis, the deployed FTIR "
         "EC against HIPS on the fixed 190 filters; a tight line at 1.90x with "
         "an intercept of minus 4.2, the systematic disagreement this whole "
         "program is about. Right: Adama, the same production calibration "
         "against collocated quartz TOR; five filters scattering around the "
         "one-to-one line, median ratio 0.77, with no line fitted because five "
         "points cannot support one. Different reference method, and a "
         "completely different picture: a tight systematic line on the left, "
         "unstructured scatter on the right."),
    notes=("Axes differ by construction: Addis is ug/m3 against HIPS Fabs/10 "
           "(MAC 10); Adama is ug per filter against TOR ECTR, pairs matched "
           "by sample date (HIPS on the Adama PTFE twins exists, see the "
           "three-method slide; no quartz exists at Addis, and that asymmetry "
           "is the point of the quartz-TOR "
           "ask). The drawn Addis line is OLS, reproducing the committed "
           "ftir_17 number 1.90x-4.17; Deming at lambda-star reads steeper "
           "(2.24x-5.86) and both are in the stat box. Adama OLS on n=5 is "
           "descriptive only. The geometry "
           "comparison, line versus scatter, is the claim; the fit numbers "
           "are context."))

slide(
    "At Adama, deployed FTIR EC sits within 15 to 30 percent of quartz TOR; the factor-of-two Addis gap is absent",
    fig=FIG / "f_adama_ftir_vs_tor.png",
    say=("The comparison Ann asked about, from the five co-located PTFE and "
         "quartz pairs. Left: FTIR EC over TOR EC per filter. The production "
         "calibration scatters around one with a median of 0.77; there is no "
         "systematic factor-of-two gap like the Addis crossplot, though five "
         "filters is five filters. The biomass-trained calibrations over-read "
         "by about one and a half times, which is a calibration-choice effect, "
         "not an Adama anomaly. Right: the thermal char-to-soot split. Every "
         "filter is soot-dominated; nothing here looks like a charcoal-heavy "
         "aerosol either."),
    notes=("Pairs matched by sample date (PTFE J1233-J1285 vs quartz "
           "J1675-J1703), July 2024 only, n=5. Ratios are unitless so the "
           "loading-vs-concentration units question drops out. char/soot = "
           "(EC1-OP)/(EC2+EC3), Han convention; all values 0.02 to 0.58, "
           "char-dominated would be above 1. Committed table: "
           "spartan_ec_2026_06_16/tables/adama_ec_calibration_comparison.csv. "
           "The AMOD export's own EC_ftir (calibration set 26) gives 0.69 vs "
           "EC TOR and 0.86 vs EC TOT (ftir_41); name the convention whenever "
           "this ratio is quoted."))

slide(
    "Adama has HIPS after all: every carbon EC sits below the HIPS band, implied MAC 16 to 21 versus about 47 at Addis",
    fig=FIG / "f_ftir41_three_method.png",
    say=("A correction to the earlier version of this deck: the CSU AMOD Batch 54 "
         "export includes HIPS on the five Adama PTFE filters. Left: TOR EC, TOT EC "
         "and deployed FTIR EC per date, with the HIPS band between MAC 6 and MAC 10. "
         "Every carbon EC sits below the band. Right: the MAC that would reconcile "
         "HIPS with each EC definition, 16 to 21 against the thermal references, "
         "versus 4 to 13 for pure EC and about 47 implied at Addis. So Adama does "
         "show absorption in excess of its EC, but at a third of the Addis "
         "magnitude. Grey circles mark the two flagged dates."),
    notes=("ftir_41. Implied MAC, unflagged medians: 16.5 vs EC TOR, 20.8 vs EC TOT, "
           "23.5 vs FTIR EC. TOR to TOT moves EC by minus 15 to minus 21 percent "
           "(median minus 19) while OC plus EC is conserved to numerical precision. "
           "Flags: July 9 quartz start 39.7 min after PTFE; July 30 PTFE volume "
           "0.46 times quartz. Neither excluded nor corrected; sampler-log review "
           "requested."))

slide(
    "The Addis-tuned calibrations over-read Adama thermal EC by 1.4 to 2 times: they are Addis instruments, not general calibrations",
    fig=FIG / "f_ftir44_models_vs_thermal.png",
    say=("The same five filters through our phase-3 models. The locked lowest-OC/EC "
         "plus baseline-corrected calibration reads 1.4 times quartz TOR EC, the "
         "sweep winner 2 times, and both land inside the HIPS band. The deployed "
         "calibration reads 0.7. Adama's OC to EC of about six is far outside the "
         "cohort those models were trained on, below 2.3, so this is extrapolation, "
         "and it says the models carry the Addis offset with them. More Adama "
         "Teflon cannot arbitrate that; only quartz at Addis can."),
    notes=("ftir_44. Unflagged medians: locked 800 plus AIRSpec 1.40 vs TOR, 1.76 vs "
           "TOT; winner 440 k=8 2.00 and 2.51; EC TOT-target model 1.01 and 1.27; "
           "deployed 0.69 and 0.86. OC and TC models refit on the same cohort leave "
           "the carbon sum at 0.47 of thermal TC, same as deployed 0.44: the carbon "
           "deficit is upstream of any calibration. Spectra id map inferred from "
           "FilterId order (CH band vs OC loading rank rho = 1.00); Davis "
           "confirmation requested."))

slide(
    "What Adama can contribute: quartz filters, not more Teflon",
    lines=["More Adama Teflon will characterize a different aerosol, not arbitrate the Addis question:",
           "   normal OC/EC, soot-dominated EC, deployed FTIR EC within 15 to 30 percent of TOR,",
           "   HIPS excess a third of Addis's, and our Addis-tuned calibrations over-read it 1.4 to 2 times",
           "The binding constraint is an independent EC measurement at Addis itself:",
           "   the quartz TOR campaign (about 36 filters across 3 seasons)",
           "If Adama sampling continues: co-located quartz would make every future pair a TOR anchor",
           "Bishoftu is the region's other live stream and it keeps collecting: 63 filters through",
           "   May 2026; 26 evaluated so far (one dry season, no offset). Two cheap asks: run HIPS",
           "   on the 22 already-scanned filters (14 are a second dry season, Jan to Feb 2026),",
           "   and scan the Belg-season batch; each lands in the evaluation with one command"],
    say=("The message for Christian and Sina in one line: the five filters we "
         "have already show Adama is a different problem from Addis, so more "
         "Adama Teflon will not help the Addis question. What would help, from "
         "anywhere in the region, is quartz: TOR EC is the measurement all "
         "three of our open forks terminate at. And Bishoftu, which is still "
         "collecting, is the ongoing regional control: its no-offset result "
         "rests on one dry season, and two cheap lab steps, HIPS on the "
         "twenty-two filters already scanned and scans for the Belg batch, "
         "would turn it into a two-dry-season, multi-season series."),
    notes=("Bishoftu census from the 2026-08-22 DB export plus the HIPS batch "
           "file: 63 PM2.5 filters dated 2025-10-20 to 2026-05-19, 48 scanned, "
           "26 with Fabs (all lot 251, dry season). Of the 22 scanned-awaiting-"
           "HIPS, the 14 dated ones are 2026-01-17 to 2026-02-19 (second dry "
           "season); 8 lack dates in the export. The Belg filters are collected "
           "but not yet FTIR-scanned; correct the earlier note that Belg "
           "filters were among the scanned backlog. "
           "Caveats to volunteer: n=5, one month (July 2024, wet season), so a "
           "dry-season Adama surprise is not excluded; the FTIR-vs-TOR "
           "agreement is production-calibration specific. The quartz-TOR "
           "campaign one-pager (ftir_ec_phase3/quartz_tor_campaign_onepager.md) "
           "carries the sizing: about 3 sigma per day separation of MAC 6 vs "
           "10 needs 11 to 13 days per season, three seasons."))

prs.save(HERE / "adama_summary_2026-09-01.pptx")
print(f"saved with {len(prs.slides._sldIdLst)} slides")
