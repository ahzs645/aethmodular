"""spectral_comparison_2026-09-01.pptx: a plain figure deck for the 2026-09-01
work (Bishoftu confirmation, cross-site crossplots, spectral comparison methods
ftir_50, and the network spectral map ftir_52).

Deliberately undesigned: one figure per slide, a factual title, nothing else.
No bullets, no speaker notes, no styling beyond a readable title. Add the
narrative in the room, or rebuild through the ann-deck skill when a claim-title
deck is wanted.

Run: ~/anaconda3/bin/python build_deck.py
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "spectral_comparison_2026-09-01.pptx"
INK = RGBColor(0x22, 0x25, 0x2A)
MUTED = RGBColor(0x6B, 0x6F, 0x76)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]


def section(title):
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(0.8), Inches(3.1), Inches(11.7), Inches(1.3))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = INK


def slide(title, figure, sub=None):
    path = FIG / figure
    if not path.is_file():
        raise FileNotFoundError(path)
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.22), Inches(12.5), Inches(0.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = INK
    top = 1.05
    if sub:
        sb = s.shapes.add_textbox(Inches(0.45), Inches(0.78), Inches(12.5), Inches(0.4))
        sp = sb.text_frame.paragraphs[0]
        sp.text = sub
        sp.font.size = Pt(13)
        sp.font.color.rgb = MUTED
        top = 1.35
    iw, ih = Image.open(path).size
    maxh = 7.35 - top
    scale = min(12.6 / (iw / 165), maxh / (ih / 165), 1.7)
    w, h = iw / 165 * scale, ih / 165 * scale
    s.shapes.add_picture(str(path), Inches((13.333 - w) / 2),
                         Inches(top + (maxh - h) / 2), Inches(w))


# title
s = prs.slides.add_slide(blank)
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.7), Inches(11.7), Inches(2.0))
tf = tb.text_frame
p = tf.paragraphs[0]
p.text = "FTIR spectral comparison: figures"
p.font.size = Pt(34)
p.font.bold = True
p.font.color.rgb = INK
p2 = tf.add_paragraph()
p2.text = ("Bishoftu confirmation, cross-site calibrations, comparison methods (ftir_50), "
           "network spectral map (ftir_52)")
p2.font.size = Pt(16)
p2.font.color.rgb = MUTED
p3 = tf.add_paragraph()
p3.text = "2026-09-01"
p3.font.size = Pt(14)
p3.font.color.rgb = MUTED

section("1. Bishoftu and the five sites")
slide("Bishoftu filter inventory by sampling month", "fig1_etbi_inventory.png",
      "26 accepted, 14 confirmed holdout, 21 sampled but not yet analysed")
slide("Bishoftu confirmation: the near-zero intercept replicates on unseen filters",
      "fig2_confirmation_crossplot.png", "locked Addis winner k=8, no re-selection")
slide("Locked configurations on the reconstructed holdouts",
      "fig3_locked_configs_holdouts.png", "dots = fit, bars = bootstrap 95% CI")
slide("One locked calibration across five SPARTAN sites", "figA_cross_site_crossplots.png",
      "lowest-OC/EC 440 + AIRSpec, k=8, site-held-out; York fits per site")
slide("Addis under every optimised calibration", "crossplot_allcalibs_addis.png",
      "the negative intercept is present in all of them")
slide("HIPS blank lines: exposure and the effect on York fits", "figB_blank_line_story.png",
      "see ftir_47 for the per-deployed-line correction")
slide("The raw HIPS archive: 4,115 filters across 28 sites", "figD_hips_archive.png")

section("2. Spectral comparison methods (ftir_50)")
slide("In-plane versus off-plane distance to the library", "domain_t2_q.png",
      "Hotelling T2 and Q residual flag different filters")
slide("Analog lists concentrate on a few library spectra", "hubness_lorenz.png")
slide("Mutual analogs per filter, selectivity-matched", "viz_mutual_analogs.png",
      "36 percent of Addis filters have none")
slide("Agreement is band-dependent", "band_correlations.png",
      "dotted line = whole-spectrum r")
slide("Where each site departs from the library", "viz_moving_window.png",
      "moving-window Pearson r against the library median")
slide("A typical Addis filter and its five nearest library spectra",
      "viz_nearest_analogs.png")
slide("How much the similarity metrics agree", "viz_metric_agreement.png",
      "median-based and neighbour-based similarity are different measurements")

section("3. The network spectral map (ftir_52)")
slide("Median spectrum per Ward class", "class_spectra.png",
      "dashed = library median; two of four classes are low-deposit")
slide("The library by Ward class, with the SPARTAN targets overlaid", "class_map_pca.png")
slide("The IMPROVE network by median corrected spectrum", "site_dendrogram.png",
      "SPARTAN sites in red")
slide("How similar IMPROVE sites are to each other", "improve_pair_distribution.png",
      "vertical lines mark each SPARTAN site's best match")
slide("Raw versus baselined site similarity", "raw_vs_baselined_sites.png",
      "plotted as 1 minus r on log axes; 92 percent of sites change nearest neighbour")
slide("Addis analogs chosen in raw space versus baselined space",
      "addis_raw_vs_baselined_analogs.png", "median overlap of the two top-50 sets is 12 percent")
slide("The same raw-chosen analogs, redrawn after baselining",
      "addis_raw_analogs_after_baselining.png")
slide("Where a spectral match becomes trustworthy", "threshold_curve.png",
      "EC agreement never reaches 75 percent precision at any cutoff")
slide("Addis median spectrum by Ethiopian season", "season_medians.png")
slide("Each season's five nearest IMPROVE spectra", "season_top_analogs.png",
      "top-200 analog overlap is 0 percent dry versus either wet season")
slide("What each season draws from the network", "season_analog_character.png")

prs.save(OUT)
print(f"wrote {OUT} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
