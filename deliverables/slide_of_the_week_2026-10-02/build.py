"""Slide of the week, 2 Oct 2026 (Ann + Satoshi, 1 Oct call): figure + pptx, words over numbers.

One slide, one figure:
  left  = one real Addis Kiremt filter after the full-range VIBES baseline, with AIRSpec (stops at
          1425 cm-1) for context, and the analog-similarity window agreed in the call drawn on it
          (600 to 3600 cm-1, CO2 and the 1050 to 1250 cm-1 Teflon residual dropped), labelled in words;
  right = the blank-model ask, as a three-step ladder (no numbers on the figure).

Run with anaconda python from the repo root:
    ~/anaconda3/bin/python deliverables/slide_of_the_week_2026-10-02/build.py

Inputs (same full-range VIBES run as deliverables/vibes_airspec_methods_2026-10-01):
- research/ftir_hips_chem/output/tables/vibes_colab_bundle_full500/stage   raw spectra + cases
- research/ftir_hips_chem/output/tables/vibes_fullrange/rank30/            VIBES corrected, 3998 to 500
- research/ftir_hips_chem/output/tables/vibes_colab_cloud/.../full-83dcf32e86bc0f09   AIRSpec, same cases
- research/ftir_hips_chem/output/tables/vibes_colab_full500_rank30/.../background_model.npz
- research/ftir_hips_chem/scripts/analog_transfer_checks.py   MASKS["oct1"] (the agreed window)
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
HC = ROOT / "research/ftir_hips_chem"
T = HC / "output/tables"
STAGE = T / "vibes_colab_bundle_full500/stage"
FULL = T / "vibes_fullrange/rank30"
BGDIR = T / "vibes_colab_full500_rank30/persistent_results/full500-rank30-137c7f6f3f9a6522"
EARLY = T / "vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"

sys.path.insert(0, str(STAGE / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(HC / "scripts"))
from vibes_large_run import RunConfig, prepare_experiment  # noqa: E402
from analog_transfer_checks import MASKS  # noqa: E402

GREY, BLUE, PURPLE, ACCENT, AMBER, INK = (
    "#8F8C84", "#2C6E9E", "#7A4FA3", "#B23327", "#C49442", "#22252A")
DROP = "#ECE9E2"
DPI = 168
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "text.color": INK, "font.size": 11,
    "axes.titlesize": 12.5, "legend.frameon": False,
})
CUT = 1425.0


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    Image.open(p).convert("RGB").save(p)
    return p


# ---------------------------------------------------------------- data
OCT1 = MASKS["oct1"]
assert OCT1 == {"low": 600, "high": 3600, "drop": [(1800, 2500), (1050, 1250)]}, OCT1

wn, X, cases, blanks, blank_ids, bmeta, _ = prepare_experiment(
    STAGE, RunConfig(profile="full", max_background_components=30))
assembly = json.loads((FULL / "ASSEMBLY.json").read_text())
assert assembly["n_failed"] == 0 and assembly["n_corrected"] == len(cases) == 12808
fcases = pd.read_csv(FULL / "cases.csv")
assert (fcases.sample_id.to_numpy() == cases.sample_id.to_numpy()).all()
V = np.load(FULL / "corrected_VIBES_full.npy").astype(float)
assert np.allclose(np.load(FULL / "wn.npy"), wn)
win = wn >= CUT
A = np.full(X.shape, np.nan)
A[:, win] = np.load(EARLY / "corrected_AIRSpec.npy")
bm = np.load(BGDIR / "background_model.npz")
n_blanks, rank = len(blanks), bm["components"].shape[1]
blank_src = bmeta.source.value_counts().to_dict()
blank_lots = bmeta.groupby("source").lot.nunique().to_dict() if "lot" in bmeta.columns else {}

# Representative Addis Kiremt filter: the Kiremt target whose AIRSpec vs VIBES-full RMS difference
# over 1425 to 4000 is closest to the Kiremt median (same rule as the 1 Oct methods deck, one season).
dates = pd.to_datetime(cases.date)
TGT = np.flatnonzero((cases.kind == "target").to_numpy() & dates.dt.month.isin([6, 7, 8, 9]).to_numpy())
rmsd = np.sqrt(np.nanmean((A[TGT][:, win] - V[TGT][:, win]) ** 2, axis=1))
j = int(TGT[np.argmin(np.abs(rmsd - np.median(rmsd)))])
row = cases.iloc[j]
REP = f"{row.filter_id} (lot {int(row.lot)}, {row.date})"
v, a = V[j], A[j]

# where the Teflon residual sits relative to everything else inside 600 to 3600
inwin = (wn >= OCT1["low"]) & (wn <= OCT1["high"])
ptfe = (wn >= 1050) & (wn <= 1250)
rest = inwin & ~ptfe & ~((wn >= 1800) & (wn <= 2500))
ptfe_vs_rest = float(np.nanmax(np.abs(v[ptfe])) / np.nanmax(np.abs(v[rest])))
kir_ptfe = np.nanmax(np.abs(V[TGT][:, ptfe]), axis=1) / np.nanmax(np.abs(V[TGT][:, rest]), axis=1)
share_ptfe_largest = float(np.mean(kir_ptfe > 1))

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(14.6, 6.1))
gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1], wspace=0.08)
ax = fig.add_subplot(gs[0])
lad = fig.add_subplot(gs[1])

lo_y = min(np.nanmin(v[wn >= 500]), np.nanmin(a[win])) * 1.25
hi_y = max(np.nanmax(v[wn >= 500]), np.nanmax(a[win])) * 1.32
for x0, x1 in [(4000, OCT1["high"]), *OCT1["drop"], (OCT1["low"], 500)]:
    ax.axvspan(x0, x1, color=DROP, lw=0, zorder=0)
ax.axhline(0, color=GREY, lw=0.7, zorder=1)
ax.plot(wn[win], a[win], color=GREY, lw=1.4, zorder=2, label="AIRSpec spline baseline (stops at 1425)")
ax.plot(wn, v, color=BLUE, lw=1.6, zorder=3, label="VIBES baseline (full range)")
ax.set_xlim(4000, 500)
ax.set_ylim(lo_y, hi_y)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_ylabel("Absorbance")
ax.legend(loc="lower left", fontsize=9.8, bbox_to_anchor=(0.11, 0.0))

top = hi_y * 0.97
lab = dict(ha="center", va="top", fontsize=9.6, color=INK)
ax.text(3800, top, "Dust 'horns'\nabove 3600:\nleft out for now;\nSatoshi: worth\nkeeping", **lab)
ax.text(2150, top, "CO₂ band:\ndropped\n(mostly empty)", **lab)
ax.text(1150, top, "Teflon\nresidual:\ndropped", color=ACCENT, ha="center", va="top", fontsize=9.6,
        fontweight="bold")
ax.text(550, top, "Below\n600:\nunreliable", **lab)
# the residual itself
k = np.flatnonzero(ptfe)[np.nanargmax(np.abs(v[ptfe]))]
ax.annotate("Often the biggest feature,\nso it would steer the correlation",
            (wn[k], v[k]), xytext=(1700, lo_y * 0.55), fontsize=9.6, color=ACCENT, ha="center",
            arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.9))
ax.annotate("Broad rise from ~1600 into the fingerprint\nregion: may be defect-rich graphitic carbon,\n"
            "so this region could carry EC information",
            (1480, v[np.argmin(np.abs(wn - 1480))]), xytext=(3050, hi_y * 0.52), fontsize=9.6,
            ha="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
ax.text(3050, top, "White = what the\nanalog match now uses", fontsize=10.5, ha="center", va="top",
        color=INK, fontweight="bold")
ax.set_title("One Addis long-rains filter, both baselines", loc="left", fontsize=12)

# ladder: blanks behind the VIBES background model
lad.set_xlim(0, 1)
lad.set_ylim(0, 1)
lad.axis("off")
lad.set_title("Teflon background: train it on more blanks", loc="left", fontsize=12)
steps = [
    ("Used so far", "a handful of blanks per lot,\nthe ones closest to the blank average", GREY),
    ("Next", "every labelled blank\nfrom lot 251", BLUE),
    ("Then", "blanks from all IMPROVE lots\n(more beats pickier)", PURPLE),
]
ys = [0.80, 0.52, 0.24]
for (head, body, col), yy in zip(steps, ys):
    lad.add_patch(FancyBboxPatch((0.04, yy - 0.09), 0.92, 0.2, boxstyle="round,pad=0.012,rounding_size=0.03",
                                 fc="white", ec=col, lw=1.6))
    lad.text(0.09, yy + 0.055, head, fontsize=11, fontweight="bold", color=col, va="center")
    lad.text(0.09, yy - 0.025, body, fontsize=10.2, color=INK, va="center")
for y0, y1 in zip(ys[:-1], ys[1:]):
    lad.annotate("", (0.5, y1 + 0.115), (0.5, y0 - 0.095),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
lad.text(0.04, 0.0, "Too few blanks leave Teflon artifacts,\nworst at low loading, as at Addis",
         fontsize=9.8, color=INK, va="bottom")
fig_path = save(fig, "f1_vibes_similarity_window_and_blanks.png")

# ---------------------------------------------------------------- pptx
INK_RGB = RGBColor(0x22, 0x25, 0x2A)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
sl = prs.slides.add_slide(prs.slide_layouts[6])
tb = sl.shapes.add_textbox(Inches(0.45), Inches(0.22), Inches(12.5), Inches(0.95))
tb.text_frame.word_wrap = True
p = tb.text_frame.paragraphs[0]
p.text = ("VIBES opens up the spectrum below 1500 cm⁻¹; to use it for analogs, skip the Teflon "
          "residual and train the background on many more blanks")
p.font.size, p.font.bold, p.font.color.rgb = Pt(20), True, INK_RGB
im = Image.open(fig_path)
scale = min(12.4 / (im.width / DPI), 6.0 / (im.height / DPI), 1.35)
w = im.width / DPI * scale
sl.shapes.add_picture(str(fig_path), Inches((13.333 - w) / 2),
                      Inches(1.25 + (6.0 - im.height / DPI * scale) / 2), Inches(w))

say = (
    "This is what came out of the call with Satoshi. On the left is one real Addis filter from the long "
    "rains, after two baselines. Grey is AIRSpec, which stops at 1425. Blue is VIBES, which keeps going "
    "down to 500, so for the first time we can use the fingerprint region when we pick analogs. Two things "
    "showed up. First, the Teflon residual around 1100 to 1200 is often the biggest feature in the corrected "
    "spectrum, so if we leave it in, it decides which IMPROVE filters look similar. We agreed to drop it, "
    "drop below 600 where VIBES goes negative, and keep dropping the CO2 band; the white regions are what "
    "the match uses now. Satoshi also pointed out the broad rise from about 1600 into the fingerprint region "
    "could be defect-rich graphitic carbon, which gives a reason to keep that part in. On the right is the "
    "second fix. The background model so far was trained on a handful of blanks per lot, the ones closest "
    "to the blank average. Satoshi's experience is that more is better, so next I train it on every labelled lot 251 blank, "
    "then on blanks from all IMPROVE lots, and redo the analog selection with the new window."
)
notes = (
    f"Filter shown: {REP}, chosen as the Kiremt (Jun to Sep) Addis target whose AIRSpec vs VIBES-full "
    "RMS difference over 1425 to 4000 cm-1 is closest to the Kiremt median (same rule as the 1 Oct "
    "methods deck, restricted to one season). One real filter, not an average. VIBES = pyvibes 1.0.0, "
    f"full-range run (3998 to 500 cm-1, {assembly['n_corrected']:,} cases, 0 failures), background model "
    f"rank {rank} from {n_blanks} blanks ({', '.join(f'{k} {v_}' for k, v_ in blank_src.items())}). "
    f"For this filter the Teflon-band residual is {ptfe_vs_rest:.1f}x the largest feature elsewhere in "
    f"600 to 3600; across Kiremt Addis filters it is the largest feature in {share_ptfe_largest:.0%} of them. "
    "Agreed similarity window (mask 'oct1' in analog_transfer_checks.py): 600 to 3600 cm-1 with 1800 to "
    "2500 and 1050 to 1250 dropped; for PLS the full spectrum can stay, since PLS down-weights it. "
    "Satoshi would keep the >3600 dust horns (dust matters for the mixture); the oct1 window still stops at "
    "3600, so this is a sensitivity case to run. Fingerprint caveat (Satoshi): useful for dust, sulfate and "
    "levoglucosan, but organics with the same functional groups can differ there, so similarity on it may "
    "not mean similar EC; worth testing with and without. The ~1600 cm-1 broad absorption from defects in "
    "graphitic sheets was one mechanistic reason PLS can predict TOR EC (early FTIR EC papers). "
    "Also from the call, not on the slide: (1) with spline-baselined IMPROVE analogs matched to the dry "
    "season, IMPROVE FTIR vs Fabs/10 shows the same low slope as Addis, so it is not unique to Addis; Ann "
    "wants the intercept added to that view next. (2) Label the per-season panels as separate "
    "calibrations (separate analog sets), and say so in the title; the all-IMPROVE toggle views are one "
    "calibration. (3) Kiremt VIBES spectra share a 1500 to Teflon hump and sit high in the Addis HIPS "
    "range; Satoshi reads it as seasonal composition rather than a direct EC signal; Ahmad to try "
    "within-season correlation groupings past 1425. (4) Focus stays on Addis; skip Delhi and Pasadena. "
    "Follow-up questions arrived from Satoshi on 2 Oct (analog overlap, VIP, fit vs extrapolation, lower "
    "envelope)."
)
sl.notes_slide.notes_text_frame.text = f"SAY: {say}\n\nNOTES: {notes}"

out = HERE / "slide_of_the_week_2026-10-02.pptx"
prs.save(out)

# ---------------------------------------------------------------- checks
chk = Presentation(out)
assert len(chk.slides) == 1
for s_ in chk.slides:
    txt = " ".join(sh.text_frame.text for sh in s_.shapes if sh.has_text_frame)
    nt = s_.notes_slide.notes_text_frame.text
    assert chr(0x2014) not in txt + nt, "em dash on slide"
    assert nt.startswith("SAY:") and "NOTES:" in nt, "notes missing"
    assert sum(1 for sh in s_.shapes if sh.shape_type == 13) == 1, "figure missing"
assert chr(0x2014) not in Path(__file__).read_text(), "em dash in builder"
print(f"saved {out.relative_to(ROOT)}: 1 slide, 1 figure, notes present, no em dashes")
print(f"rep {REP}; blanks {n_blanks} {blank_src} lots {blank_lots}; ptfe/rest {ptfe_vs_rest:.2f}; "
      f"kiremt share ptfe largest {share_ptfe_largest:.2f} of {len(TGT)}")
