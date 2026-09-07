"""Ann 1:1 deck, house style (docs/deck-style-prompt.md).

16:9 white, claim-as-title, one title-free figure per slide, SAY/NOTES on every
slide. Numbers validated against the calibration explorer on :5058 before build
(anchors: OCEC-800 Option A raw = k6, OLS 1.5854x-3.2215, held-out TOR R2 0.9107;
+AIRSpec = k5, OLS 0.857x-1.615, Deming 0.9539x-2.089, held-out 0.9042).

Run: ~/anaconda3/bin/python build_deck.py
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from PIL import Image

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "ann_1on1_2026-08-25.pptx"
INK = RGBColor(0x22, 0x25, 0x2A)
GREY = RGBColor(0x8F, 0x8C, 0x84)
W, H = 13.333, 7.5

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]


def _notes(slide, say, notes):
    slide.notes_slide.notes_text_frame.text = f"SAY: {say}\n\nNOTES: {notes}"


def _title(slide, text, size=20):
    box = slide.shapes.add_textbox(Inches(0.62), Inches(0.42),
                                   Inches(W - 1.24), Inches(0.95))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = True
    r.font.color.rgb = INK; r.font.name = "Calibri"
    return box


def fig_slide(title, png, say, notes):
    """Claim title + one title-free figure, centred in the space below it."""
    s = prs.slides.add_slide(BLANK)
    _title(s, title)
    p = FIG / png
    if not p.exists():
        raise SystemExit(f"missing figure: {p}")
    iw, ih = Image.open(p).size
    top_y, avail_h = 1.52, H - 1.52 - 0.42
    avail_w = W - 1.5
    scale = min(avail_w / (iw / 96), avail_h / (ih / 96))
    w_in, h_in = (iw / 96) * scale, (ih / 96) * scale
    s.shapes.add_picture(str(p), Inches((W - w_in) / 2),
                         Inches(top_y + (avail_h - h_in) / 2),
                         width=Inches(w_in), height=Inches(h_in))
    _notes(s, say, notes)
    return s


def text_slide(title, lines, say, notes, size=17):
    s = prs.slides.add_slide(BLANK)
    _title(s, title)
    box = s.shapes.add_textbox(Inches(0.78), Inches(1.9),
                               Inches(W - 1.9), Inches(H - 2.5))
    tf = box.text_frame; tf.word_wrap = True
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.color.rgb = INK if not ln.startswith("   ") else GREY
        r.font.name = "Calibri"
        p.space_after = Pt(13)
    _notes(s, say, notes)
    return s



def divider(text, sub=""):
    s = prs.slides.add_slide(BLANK)
    b = s.shapes.add_textbox(Inches(0.9), Inches(3.0), Inches(W - 1.8), Inches(1.6))
    tf = b.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = text
    r.font.size = Pt(34); r.font.bold = True; r.font.color.rgb = INK; r.font.name = "Calibri"
    if sub:
        p = tf.add_paragraph(); r2 = p.add_run(); r2.text = sub
        r2.font.size = Pt(15); r2.font.color.rgb = GREY; r2.font.name = "Calibri"
    _notes(s, f"Divider: {text}.", sub or "Section break.")
    return s

# ------------------------------------------------------------------ 1. title
s = prs.slides.add_slide(BLANK)
_title(s, "Your lot hypothesis was right \u2014 and baselining already fixes it", size=27)
b = s.shapes.add_textbox(Inches(0.66), Inches(1.75), Inches(W - 1.4), Inches(1.4))
tf = b.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = "Ann 1:1 \u2014 25 Aug 2026 \u2014 the week of 19\u201325 August"
r.font.size = Pt(16); r.font.color.rgb = GREY; r.font.name = "Calibri"
_notes(s,
   "Roadmap. Slide 2 is your August list. Slide 3 is the week in order. Slide 4, the locked "
   "numbers still reproduce. Then the measurement block \u2014 the York re-fit and the "
   "five-site ladder under three blank lines. Then the HIPS decode, which is the big one: we "
   "took the instrument math apart and Pasadena's anomaly dissolved into it. Then your lot "
   "question, answered on slide 9 \u2014 you were right. Then the grid and the slope trap, the "
   "AERONET witness, the filters we recovered, what the app bought us, what didn't survive, "
   "and Adama. Backup block at the end has six more.",
   "This is the Ann 1:1, distinct from the 27 Aug FTIR-group deck "
   "(deliverables/ann_update_2026-08-25/). Shared figures, different framing: this one follows "
   "her August asks and the week's arc.")

# ------------------------------------------------------------------ 2. status
fig_slide("Every item from 12 August is closed or answered \u2014 except Adama",
   "f_status_asks.png",
   "Top to bottom: baseline both selection and calibration, shipped on the 18th. Cutoff sweep "
   "done. Evaluate-on-same-lot is in the app. Your two questions \u2014 is baselining helping "
   "because of the lot, and why does 248 look so small \u2014 both answered later in the deck. "
   "Bishoftu spectra pulled. Adama is the one still open.",
   "'ANSWERED' is used where you asked a question rather than assigned a task. Adama is PART: "
   "the seed figure exists but the TOR-vs-FTIR OC/EC comparison you named is not built.")

# ------------------------------------------------------------------ 3. the week
text_slide("The week in order",
   ["19\u201320 Aug \u2014 your meeting actions became app features; server-side batches; the "
    "dense cutoff sweep found an ocec 440\u2013490 \u00d7 AIRSpec basin beating the locked 800.",
    "22 Aug \u2014 direct SQL to the SPARTAN database; spectra for Bishoftu, Beijing, Delhi, "
    "Pasadena. The pre-registered Bishoftu test: the offset largely vanishes.",
    "23 Aug \u2014 adjudication. York re-fit with real HIPS uncertainties; the HIPS math decoded "
    "from the batch export; the 1617-band story corrected twice; the 9,555-job grid.",
    "24\u201325 Aug \u2014 validation layer (winner takes only 62.5% of bootstraps); LOCAL "
    "per-filter calibration; the lot question answered."],
   "One slide of context before the results. The shape of the week: it went from a one-city "
   "mystery to a five-city, multi-instrument adjudication. The turning point was the 22nd, when "
   "SQL access let me pull four more sites' spectra myself \u2014 which is the thing you told me "
   "to do. Everything after that is comparative.",
   "Three concurrent Claude sessions worked the repo this week, which is why some findings were "
   "produced in parallel and had to be reconciled (OFFSET_ADJUDICATION_2026-08-23.md exists "
   "because three docs reached different conclusions within 24 hours).")

# ------------------------------------------------------------------ 4. anchors
text_slide("Before anything new: the locked numbers still reproduce",
   ["Lowest-OC/EC 800, Option A, raw  \u2192  k = 6,  OLS 1.5854x \u2212 3.2215,  held-out TOR R\u00b2 0.9107",
    "Lowest-OC/EC 800, Option A, +AIRSpec  \u2192  k = 5,  OLS 0.857x \u2212 1.615,  Deming 0.9539x \u2212 2.089",
    "   Option A = site-grouped 5-fold CV, first major minimum, site-disjoint 80/20 fit",
    "   Every number in this deck was pulled from the explorer today."],
   "Both locked anchors reproduce exactly against the live app, so everything after this is on "
   "the same footing as August. Option A is the site-grouped protocol \u2014 the only one that "
   "gives a held-out TOR test, so I quote that R\u00b2 wherever I use it.",
   "Deming is primary for the intercept (lambda* = 2.96 at MAC 10, scaled by MAC^2); OLS "
   "alongside. The Deming intercept is MAC-invariant. Anchors validated before any figure "
   "in this deck was generated \u2014 and that validation caught a real error, slide 18.")

fig_slide("Where we started: what the network ships today",
   "f_deployed_shape.png",
   "One slide of before, so the rest has a baseline. This is SPARTAN's currently deployed FTIR "
   "EC plotted against HIPS absorption \u2014 no work of ours in it. Addis: slope 2.26 and an "
   "intercept of minus 5.9 micrograms. Delhi: 1.81 and minus 4.8. Both sit well off the "
   "one-to-one line, and both have large negative intercepts. For scale, the corrected "
   "calibration we will get to brings the Addis intercept from minus 5.9 to about minus 1.4 "
   "\u2014 roughly a four-fold reduction. So the offset we are chasing is what is left after "
   "the calibration work, not the whole gap.",
   "ChemSpec 'EC PM2.5' IS the deployed EC_ftir (ftir_27 showed ChemSpec_EC = round(EC_ftir, 2) "
   "at r2 = 0.9997), so this is the deployed product, NOT an independent reference. That is the "
   "right thing to plot here \u2014 the question is what the network currently produces. "
   "Addis n = 175, Delhi n = 27 (Delhi's ChemSpec coverage is thin). Deming lambda* = 2.96 at "
   "MAC 10; OLS shown alongside and is markedly flatter, which is the estimator point again. "
   "Do not read Delhi's -4.77 as comparable to the York table's Delhi number \u2014 different "
   "estimator, different n, deployed vs our calibration.")

divider("The measurement")

# ---------------------------------------------------------- 5a. estimator explainer
fig_slide("Why the estimator matters: a y-only fit hides part of the intercept",
   "f_york_explainer.png",
   "Before the result, one slide on how it's fitted, because the choice moves the number. "
   "Blue is an ordinary fit that assumes the x-axis is exact. But the x-axis is HIPS absorption, "
   "which carries its own measurement error \u2014 so a y-only fit flattens the slope and drags "
   "the intercept toward zero. Red treats both axes as uncertain. On the same filters the "
   "intercept goes from minus 1.28 to minus 1.71. The red diamond on the left is the quantity we "
   "care about: the EC the model predicts where the filter shows no absorption at all.",
   "Addis, n = 239, lowest-OC/EC-800 x AIRSpec, Option A. lambda* = 2.96 at MAC 10. "
   "This is why the deck quotes Deming/York rather than OLS for the intercept \u2014 OLS "
   "understates it. Both are shown on every crossplot elsewhere. The full weighted treatment "
   "(York with per-filter HIPS uncertainties) is the next slide; this one is the intuition.")

# ------------------------------------------------------------------ 5. York
fig_slide("Weighted per-filter fitting leaves Addis as the only certain intercept",
   "f_york_cross_site.png",
   "This replaces the Deming table I showed you before. Per-filter weighted errors-in-variables, "
   "using each site's real HIPS uncertainties, with the errors inflated until the fit is "
   "consistent \u2014 so any lack of fit, including curvature, is absorbed into the bars rather "
   "than into the intercept. Addis sits at minus 1.38 with a z of nearly eight. Bishoftu, same "
   "country, same instrument, same lot, is indistinguishable from zero. Beijing is positive.",
   "York (2004) per-filter weighting; empirical per-site sigma(Fabs) from unified_filter_dataset; "
   "sigma_y inflated until MSWD = 1 \u2014 conservative, because it hands curvature to the error "
   "bars. Bars are 95% CI. This supersedes the pooled-lambda Deming table. "
   "Beijing was never anomalous: its unweighted slope of 1.48 was leverage from a few points and "
   "falls to 0.98 under weighting.")

# ------------------------------------------------------------------ 6. C invariant
fig_slide("The offset is a property of the data, not of the MAC we assume",
   "f_mac_invariance.png",
   "A fair objection is that the intercept just reflects the mass absorption cross-section we "
   "picked. It doesn't. The quantity on the right \u2014 intercept times MAC over slope \u2014 "
   "is algebraically invariant to MAC, and empirically it barely moves: 18.8 to 26.1 across six "
   "setups whose slopes span a factor of three and a half. Median 21.5 inverse megametres, about "
   "46 percent of the median Addis absorption.",
   "ftir_25_intercept_invariant.md \u2014 pure algebra on committed constants, hence markdown not "
   "a notebook. Open circles are MAC 6, filled are MAC 10; they coincide by construction. "
   "This is why the offset survives every reframing: it isn't a modelling artifact.")

# ------------------------------------------------------------------ 7. ladder
fig_slide("Addis holds its intercept under every HIPS blank-line variant",
   "f1_intercept_blankline_ladder.png",
   "We recomputed every filter's absorption under three different forms of the HIPS calibration "
   "line \u2014 the deployed one, a lot-common linear one, and a quadratic. Addis holds between "
   "minus 1.3 and minus 1.5 under all three. Delhi's bar reaches zero, so Delhi's intercept is "
   "uncertain. The others bracket them.",
   "The three variants are re-derivations of tau from raw T1/R1 with different blank-line fits, "
   "not re-weightings of the same Fabs. Harshest variant for Addis is -1.27 \u00b1 0.17, z ~ -7. "
   "This is the figure that moved the story from 'Addis and Delhi' to 'Addis certain, Delhi "
   "uncertain'.")

divider("The HIPS calibration, decoded", "what the instrument is actually doing")

# ------------------------------------------------------------------ 8. geometry
fig_slide("The scattering correction is a regression through field blanks \u2014 and 36% of Addis sits past its end",
   "f3_blankline_geometry.png",
   "This is the piece I didn't understand before this week. The HIPS scattering correction isn't "
   "a physical model \u2014 it's a straight line fitted through the field blanks of that filter "
   "lot. Tau is the log ratio of that line's prediction to the measured transmittance. The shaded "
   "region is where a filter is darker than every blank that defines its own calibration. "
   "Thirty-six percent of Addis filters live there, so their absorption rests on an extrapolated "
   "line.",
   "tau = ln((Intercept + Slope*R1)/T1), decoded from the SPARTAN batch export and confirmed "
   "against the database: the formula reproduces shipped tau for 100% of filters to within 1e-4. "
   "The PTFE zero itself is clean: +0.32 \u00b1 0.46 Mm-1 from 575 blanks, so the extrapolation "
   "is the issue, not the origin.")

# ------------------------------------------------------------------ 9. slope ladder
fig_slide("Pasadena's anomaly was the blank line; Delhi's slope is real",
   "f2_slope_blankline_ladder.png",
   "And this is what that bought us. Pasadena looked wildly wrong \u2014 slope 3.15. Recompute it "
   "under a lot-common line and it falls to 1.47; under a quadratic, 0.91. Its anomaly was "
   "blank-line shape error at very low loading, not aerosol. Delhi's 1.8 barely moves across all "
   "three, so Delhi's slope is a real, unexplained result.",
   "Pasadena has sub-detection filters with up to 442% relative uncertainty \u2014 exactly where "
   "a mis-shaped blank line does the most damage. Delhi's stability across variants is what "
   "promoted it from 'probably the same story as Addis' to its own open question. "
   "Candidate explanations for Delhi: low true MAC, or FTIR organic interference \u2014 the "
   "four-site MA350 anchor would split them.")

# ------------------------------------------------------------------ 10. lot swap
fig_slide("Mis-assigning a lot's blank line costs 2.6% of the offset",
   "f_blankline_lot_swap.png",
   "The natural follow-up: if the blank line is lot-specific, how much damage does using the "
   "wrong lot's line do? I applied lot 251's line to every lot-253 filter. Median shift is "
   "0.57 inverse megametres \u2014 the dashed line on the right is the offset we're chasing. "
   "It's about two and a half percent of it. So lot-line bookkeeping is a nuisance, not the "
   "explanation.",
   "n = 330 lot-253 filters. Blank lines by lot: 248 (1383.2, -2.544), 251 (1416.2, -2.783), "
   "253 (1375.1, -2.608). Lot 253's line is built from 33 lab blanks vs lot 248's 10, so it is "
   "the better-determined one \u2014 I initially guessed the opposite and the database corrected me.")

divider("Your lot question", "answered")

# ------------------------------------------------------------------ 11. HEADLINE
fig_slide("Baseline correction removes the lot-to-lot difference \u2014 the mechanism you predicted",
   "f_lot_baseline_removes_lot_effect.png",
   "In August you said baselining might be helping because it minimises the difference between "
   "lot 248 and lot 251. This is that, measured. Beijing, because it's the one site that ran both "
   "lots at the same time \u2014 178 days of overlap, so season and period are held fixed. Grey is "
   "raw spectra: the lots differ by about four and a half and the interval never touches zero, at "
   "either cohort size. Blue is the same comparison on baseline-corrected spectra: it collapses to "
   "about one and straddles zero. Baselining isn't just removing Teflon background in general "
   "\u2014 it's specifically absorbing the between-lot substrate difference.",
   "ocec-450 raw -4.35 [-7.63,-1.20], AIRSpec -1.43 [-4.01,+0.94]; ocec-800 raw -4.49 [-7.01,-1.96], "
   "AIRSpec -0.86 [-3.66,+2.02]. At MAC 10 that is -0.45 vs -0.09 ug/m3. Bootstrapped mean residual "
   "vs a common line, 4,000 draws \u2014 does not depend on a per-lot slope from small n. "
   "CAVEATS: n = 14 vs 34; within-window dates not matched further; loading differs slightly "
   "(median predicted EC 1.18 vs 1.66 at ocec-800 AIRSpec). "
   "It does NOT explain the Addis offset: ETBI is 100% lot 251 and ETAD 80%, so Addis-vs-Bishoftu "
   "is a within-lot-251 contrast, and the residual lot term after baselining is ~5% of 21.5 Mm-1.")

# ------------------------------------------------------------------ 12. lot 248
fig_slide("Lot 248 isn't a partial download \u2014 it's a two-month lot",
   "f_lot_census_improve.png",
   "You flagged that my 248 pool looked too small \u2014 about 1,300 when 251 had eleven thousand. "
   "It's not a bad pull. Lot 248 ran for two months, mid-December 2020 to mid-February 2021, and "
   "produced 1,362 analyses network-wide. My 1,299 was essentially the whole lot. 251 ran a full year.",
   "IMPROVE ftir_catalog: 169,566 analyses, 182 sites. Bar labels are analysis counts. "
   "This also reframes the lot picture: IMPROVE runs roughly two years ahead of SPARTAN on the same "
   "lot numbering, so IMPROVE already holds the lots SPARTAN will use next.")

# ------------------------------------------------------------------ 13. lot 253
fig_slide("The lot that matters now is 253 \u2014 and the calibration has never seen it",
   "f_lot253_takeover.png",
   "The forward-looking version of your lot question. Lot 253 entered SPARTAN in March 2025 and is "
   "now 58 percent of everything sampled since September, including the newest Addis filters. Our "
   "basis is IMPROVE lots 248 and 251 only. The good news: IMPROVE has already been through 253 "
   "\u2014 five thousand filters with matched TOR EC, and the pull manifest is written.",
   "433 lot-253 SPARTAN filters, 13 sites; 340 carry a SampleDate and the stackplot uses that "
   "subset. IMPROVE lot 253: 7,631 analyses, 5,050 with matched TOR EC, 83 sites, 2023-04 to "
   "2024-04. Lot 255 deliberately sealed for replication. "
   "The obvious Delhi lot-253 test is NOT identifiable: 4x more loaded, seasonally disjoint, "
   "6 filters in the overlap, CI [-53, +304].")

divider("The exhaustive search")

# ------------------------------------------------------------------ 14. grid
fig_slide("The Addis basin held when the search got 4\u20137\u00d7 wider and ~28\u00d7 denser",
   "f4_screening_cloud.png",
   "Every Addis variant we can identify, scored the same way. One dot per configuration: "
   "cohort size across, the Addis intercept up. Grey fails the held-out floor, blue passes it, "
   "and the star is the winner. The point is the shape \u2014 there is a basin, and it stayed "
   "put when we widened the cutoff ranges four to seven times and sampled them roughly "
   "twenty-eight times more finely. Nine and a half thousand jobs, seventy-one thousand scored "
   "rows, zero errors. So the Addis answer is not sitting on a knife edge of cohort size; it is "
   "a broad, stable region.",
   "7,234 honest-protocol (Option A) variants at Addis shown. Search widened from the original "
   "ladders \u2014 eth_shaped 200-400, analogs 400-600, ocec 600-1000, five cutoffs each \u2014 "
   "to eth_shaped 100-900, analogs 100-1500, ocec 100-2000 in steps of 10, so 15 cutoff values "
   "became 413. "
   "IMPORTANT CAVEAT to state if asked: these screening numbers are DESCRIPTIVE only. Every "
   "configuration here was scored on filters that informed the selection, so the cloud shows "
   "structure, not skill. Anything quoted as a result comes from the held-out TOR test or an "
   "out-of-country site. "
   "The methods lesson lives on its own slide later: |intercept| + w|slope-1| is gamed by flat "
   "slopes, so every ranking is slope-boxed to 0.85-1.18. "
   "The two-city question is the NEXT slide \u2014 this figure is Addis only and cannot speak "
   "to Delhi.")

# ------------------------------------------------------- 14a. the target region is empty
fig_slide("Across 19,275 configurations, the region we actually want is empty",
   "f_two_site_reconcile.png",
   "Here is the same search drawn as one picture. Every configuration is a point: its Delhi "
   "intercept across, its Addis intercept up. The dashed box top-centre is what we want \u2014 "
   "both intercepts near zero. It is empty. The blue points are the 850 configurations that hold "
   "a sane slope at both cities, and the best Addis intercept among them is minus 0.66. So the "
   "two cities cannot be reconciled by choosing a cohort, a baseline or a component count.",
   "19,275 configurations run at both sites, Option A; one fitted calibration applied to each "
   "site. Slope box 0.85-1.18 at BOTH sites gives n = 850. The grey cloud's diagonal streaks are "
   "cohort families. This is the single strongest statement that the offset is not a tuning "
   "artifact \u2014 and it is descriptive, so quoted results still come from held-out sets.")

# ------------------------------------------------------- 14b. joint winners
fig_slide("Three configurations do hold a sane slope at both cities \u2014 and they are the analogs",
   "f_best_joint_crossplots.png",
   "These are the three. Top row Addis, bottom row Delhi, coloured by filter lot. Delhi sits on "
   "the one-to-one line across the whole range, and lot 253 \u2014 the red points, the new lot "
   "\u2014 lies on the same line as lot 251, which is a real cross-lot result. Addis is the "
   "tight band sitting parallel to and below the line: that shape is an additive offset, not "
   "noise and not a slope error. And note what they are: the spectral analog cohorts you pushed "
   "me toward in August.",
   "analogs-540 x deriv2 k17: Addis 0.86x-1.19, Delhi 1.11x-0.01, held-out TOR R2 0.59. "
   "analogs-550 x deriv2 k17: 0.85x-1.18 / 1.09x+0.03, R2 0.56. "
   "eth_shaped-400 x deriv2 k7: 1.01x-1.96 / 1.15x+0.18, R2 0.34 \u2014 best Addis slope but "
   "worst intercept and half the held-out skill, a case where slope alone flatters a weaker model. "
   "Samples: Addis 239 (lot 248:34, 251:191, 253:14), Delhi 152 (248:15, 251:76, 253:61). "
   "Addis lots recovered by joining the evaluation date field to ETAD metadata \u2014 validated "
   "against the HIPS-side counts, which match exactly.")

# ------------------------------------------------------- 14c. addis-only collapses
fig_slide("At Addis, tuning on Addis alone genuinely wins \u2014 that is the trap",
   "f_collapse_1_addis.png",
   "Now the counterpart, in three parts. First, Addis on its own terms. Red is tuned on Addis "
   "alone, blue on both cities, and the amber line is what SPARTAN ships today. Both tuned "
   "sets sit in the slope band on the left; the deployed calibration is up at 2.26, well "
   "outside it. But look at the "
   "intercept on the right: the red bars are almost exactly zero, and the blue ones are down at "
   "minus 1.2. On Addis's own scorecard, tuning on Addis wins outright. If this were the only "
   "slide, you would pick red.",
   "Top-8 under each strategy, Option A. Addis-tuned = best |intercept| + 0.5|slope-1| with the "
   "Addis slope boxed to 0.85-1.18; both-cities = the same score summed over Addis and Delhi "
   "with both slopes boxed. This slide is deliberately the case FOR the wrong answer \u2014 the "
   "next two are why it is wrong. "
   "The amber deployed line (Deming 2.26x -5.86, n = 175) is a REFERENCE LEVEL, not a "
   "like-for-like row: it is fitted on the ChemSpec-covered filters, a different and smaller "
   "set than our 239-filter evaluation, and ChemSpec EC is EC_ftir rounded (ftir_27).")

fig_slide("The same eight configurations collapse at Delhi",
   "f_collapse_2_delhi.png",
   "Same eight configurations, now applied to Delhi. Slopes of four to nine, and intercepts of "
   "minus fifteen to minus forty-five micrograms. For scale, the entire Addis offset we have "
   "been chasing all deck is about minus one point four, and even the deployed calibration "
   "\u2014 the amber line, which nobody defends \u2014 only reaches minus 4.8. The Addis-tuned "
   "set is three to nine times worse than the thing we are trying to improve on. The blue set "
   "sits on the line at both cities. So the near-zero Addis intercept on the previous slide was "
   "bought by breaking the model everywhere else.",
   "Delhi n = 152. The magnitude of the intercept collapse is the point: this is not a mild "
   "degradation, it is a different model. Same failure the validation layer caught for Delhi's "
   "own grid winner (analogs-530 x deriv2: 96.1% of Delhi filters beyond the training domain). "
   "Deployed reference at Delhi: Deming 1.81x -4.77 on n = 27 ChemSpec-covered filters \u2014 "
   "thin coverage, so treat it as a level, not a precise value.")

fig_slide("And they had almost no held-out skill to begin with",
   "f_collapse_3_skill.png",
   "And the honest skill test, on filters the cohort selection never saw. The Addis-tuned set "
   "sits between 0.10 and 0.33. The both-cities set is 0.34 to 0.80. So they were overfit the "
   "whole time \u2014 it was visible without ever looking at Delhi, if you check held-out skill "
   "rather than the fit you optimised. That is the argument for freezing lot 253 as a validation "
   "set nothing is screened on.",
   "Held-out TOR R2 under Option A \u2014 site-disjoint 80/20, so the test sites are not in the "
   "fit at all. This is the cleanest statement of the slope trap in the deck: a near-zero "
   "intercept at one site is cheap to buy and worth nothing on its own. "
   "Screening is not evaluation \u2014 VALIDATION_LAYER_2026-08-24.md.")

# ------------------------------------------------------------------ 15. slope trap
fig_slide("A flat enough line always has a small intercept \u2014 rankings need a slope box",
   "f_slope_trap.png",
   "The methods lesson from running the grid at scale, and it's the kind of thing you only see "
   "with twenty thousand configurations. If you rank naively on intercept plus half the slope "
   "error, the winners are the red points \u2014 median slope 0.48. They have small intercepts "
   "because they're nearly flat, not because they're good. The blue points are the same ranking "
   "with slope constrained to a sane band. Every ranking in this deck uses the box.",
   "19,561 Addis configurations, Option A. The score is |intercept| + 0.5|slope-1|. This is "
   "ftir_17's full-scale slope trap reappearing at grid scale. The Optimize tab's score alone is "
   "gameable \u2014 the app now ships with the slope box in the leaderboard.")

divider("Independent witnesses")

# ------------------------------------------------------------------ 16. AERONET
fig_slide("Addis reports more absorption than its own column can account for",
   "f_aeronet_column_check.png",
   "AERONET looking down at the same air. The ratio of column absorption to our surface Fabs gives "
   "an effective scale height. Delhi, Beijing and Pasadena cluster between 544 and 894 metres. "
   "Addis sits at 232. The sharpest contrast is Delhi: 1.6 times our surface absorption but 4.3 "
   "times the column. Delhi's column scales with its surface; Addis's doesn't.",
   "H = AAOD675/Fabs is NOT a physical boundary-layer height \u2014 meaningful only compared "
   "across sites measured the same way. Level 1.5; AERONET inversion products are formally "
   "reliable only above AOD440 ~ 0.4. Gurgaon is ~30 km from the Delhi SPARTAN site. "
   "Separately and more strongly: the corrected 1500-1700 cm-1 organic envelope at Addis predicts "
   "column absorption beyond loading, Fabs and month (r = +0.28, p = 0.002) \u2014 no filter "
   "artifact can produce that.")

# ------------------------------------------------------------------ 17. recovered
fig_slide("Bishoftu grew from 26 to 40 filters \u2014 recovered from raw instrument data",
   "f_recovered_filters.png",
   "You told me to get the Bishoftu spectra myself, and that turned into more than spectra. The "
   "database holds raw transmittance and reflectance for 256 filters that never reached the "
   "shipped file. Because the raw values are bit-identical to the shipped ones and the tau formula "
   "reproduces shipped tau exactly, I can recompute absorption for them. Bishoftu gains fourteen "
   "\u2014 a 54 percent increase on the site carrying our whole Addis-is-specific argument.",
   "Validation: raw T/R identical to shipped T1/R1 on 3,859 filters (max |diff| = 0); tau formula "
   "reproduces shipped tau for 100% within 1e-4. Recovered medians land on each site's shipped "
   "median. 42 of the 256 come out at tau < 0.05 with no volume \u2014 the field-blank signature. "
   "CAVEAT: RECONSTRUCTED, not official. Production QC (MDL, uncertainty, comment-based rejection) "
   "not replicated; DepositArea is the site median. Report headlines with and without them.")

divider("The tool, and what it taught us")

# ------------------------------------------------------------------ 18. app built
text_slide("The app stopped being a viewer and became the experiment",
   ["Your actions became features: evaluate-on-same-lot masking, selection on corrected spectra, "
    "the cutoff slider you asked for.",
    "Server-side batch optimizer \u2014 survives the browser tab, runs multi-site sweeps unattended; "
    "targets are the inner loop so one fit is reused across five cities.",
    "New tabs this week: Sites (cross-site + extrapolation diagnostic), HIPS lab (York \u00d7 "
    "blank-line variants, blank ledger), Analogs, Optimize with slope-boxed leaderboards.",
    "   Validation layer: Q-residual and score-distance guardrails; LOCAL per-filter endpoints."],
   "You asked in August how I'd keep track once I could make a million crossplots. This is the "
   "answer. The app went from something I click through to the thing that runs the experiment "
   "\u2014 I describe a grid, it runs overnight, and the leaderboard is scored the same way the "
   "interactive readout is. The HIPS lab is the piece that made this week's decode possible: it "
   "recomputes every filter under alternative blank lines.",
   "Flask on :5058 under anaconda python (the uv venv lacks flask). Caches are keyed and "
   "SHA-256-manifested. Curve caches from the 08-20 dense sweeps carried most of the 9,555-job "
   "grid's cost, which is why it finished same-day.")

# ------------------------------------------------------------------ 19. app learned
text_slide("Three things we only learned because the search was exhaustive",
   ["The slope trap. Only visible with ~20,000 configurations \u2014 naive scoring picks flat lines.",
    "The winner is not stable. Bootstrap the filters and the top configuration wins only 62.5% "
    "of draws \u2014 so 'the best cohort' is a range, not a point.",
    "Screening is not evaluation. Delhi's grid winner had 96.1% of its filters outside the "
    "training domain and died on fresh filters \u2014 which is why lot 253 is now frozen as an "
    "untouched validation set.",
    "   Your August worry \u2014 that selection becomes 'Ahmad liked this one' \u2014 is exactly "
    "what these three guard against."],
   "This is the answer to the concern you raised in August, that picking configurations by hand "
   "would come down to taste. Running everything doesn't remove judgement, but it makes the "
   "judgement auditable: I can show you the whole cloud, the winner's stability, and whether a "
   "result held on filters nothing was screened on.",
   "Winner stability from the validation layer's bootstrap (VALIDATION_LAYER_2026-08-24.md). "
   "Lot 253 is the first untouched spectral-lot challenge; lot 255 stays sealed for replication. "
   "Nested selection lives in ftir_36; it validates the selection algorithm on source sites, "
   "which is NOT the same as turning the repeatedly-inspected Addis readout into an independent "
   "estimate (Varma & Simon 2006).")

divider("Corrections, and what's owed")

# ------------------------------------------------------------------ 20. didn't survive
text_slide("What didn't survive \u2014 and why that's the point",
   ["The 1617 cm\u207b\u00b9 band: retracted twice. Delhi matched Addis (wrong metric \u2014 it "
    "was reading a carbonyl flank); then Bishoftu had the band without the offset. Three baselines "
    "gave three answers.",
    "'The offset is Addis-specific' \u2192 'a two-city signature' \u2192 'Addis certain, Delhi is "
    "a slope story'. Each doc carries a banner pointing forward.",
    "Closed branches: dust/Fe (7% of Addis), PTFE zero (+0.3 Mm\u207b\u00b9), MA350 wavelength "
    "spread, ChemSpec columns (both circular), shape-based cohorts, pooled-\u03bb Deming.",
    "   Method rule now standing: any claim about a band under 1520\u20131600 cm\u207b\u00b9 "
    "needs two independent baselines, because AIRSpec anchors its spline there."],
   "I want to show the failures as prominently as the results, because most of this week was "
   "killing my own findings. The band story is the clearest case: I wrote it up as a five-site "
   "result, then found the metric was measuring the wrong thing at Delhi, then found Bishoftu has "
   "the band and no offset. It's an Ethiopian regional marker, not the offset's carrier.",
   "Every superseded doc is retained with a banner rather than rewritten \u2014 BAND1617_LEAD has "
   "the full reversal trail. The AIRSpec spline anchor at 1520-1600 is a structural property of "
   "the method (find_min_pos with interval (1600,1520)), not a tuning choice.")

# ------------------------------------------------------------------ 21. the mistake
text_slide("The mistake you flagged in August was still live in my code",
   ["In August you caught me selecting on baseline-corrected spectra but calibrating on raw.",
    "The app takes spectra = \"airspec\" / \"neutral\" / \"deriv2\" \u2014 anything else falls "
    "through to raw, silently.",
    "I had been passing \"corrected\" \u2014 the filename convention. It ran on raw and looked fine.",
    "   Caught by an anchor refusing to reproduce. Both anchors now match; the lot result on "
    "slide 11 is the re-run."],
   "I want to flag this rather than bury it. The exact error you caught in August was still in my "
   "API calls, because the token I was passing isn't valid and the app falls back to raw without "
   "complaining. It was caught because the locked anchor wouldn't reproduce, which is why I "
   "validate them before building anything. The first version of the lot result showed a big lot "
   "effect that was really a raw-spectra effect.",
   "The dispatch in app.py is an if/elif chain with a bare else returning raw, so an unrecognised "
   "spectra token is indistinguishable from asking for raw. Recorded in "
   "SPARTAN_LOT_INVENTORY_2026-08-23.md and in the group deck's talking points.")

# ------------------------------------------------------------------ 22. Adama
fig_slide("Adama: this is a seed, not yet the case you asked for",
   "f_adama_context.png",
   "The one I owe you. You asked for two or three slides making the case to Christian and Sina "
   "that Adama is unlike Addis, so they stop putting effort into collecting more. What I have is "
   "context \u2014 Adama sits at the IMPROVE median rather than the Addis extreme. What I don't "
   "have is the comparison you named: TOR OC and EC against FTIR OC and EC on the same filters. "
   "That's the panel that makes the argument, and it's next.",
   "Her exact ask (12 Aug): communicate to Christian and Sina that this site is very different "
   "from Addis \u2014 no HIPS-EC difference, different OC/EC ratios. Timeline: two-to-three weeks. "
   "Context in hand: Adama TOR OC/EC 4.6-7.2, about the pool median, which challenges the "
   "OC/EC-extreme premise. Blocked on nothing.")

# ------------------------------------------------------------------ 23. asks
text_slide("What I need from other people",
   ["Quartz TOR on collocated Addis filters \u2014 three independent lines now terminate there.",
    "Kirchstetter solvent extraction + HIPS re-measurement \u2014 cheap, uses archived filters, "
    "and decisive between organic absorption and an instrument artifact. This is the ask for Davis.",
    "IMPROVE lot-253 spectra \u2014 pull script written, needs a Windows/VPN session.",
    "   Adama TOR-vs-FTIR panel is mine to build; blocked on nothing."],
   "Three asks, and the second is new and cheap. If you extract the organics off an archived Addis "
   "filter with solvent and re-measure it on HIPS, an organic absorber shows up as a drop and an "
   "instrument artifact doesn't. That uses filters we already have. The quartz TOR campaign is "
   "still the primary, because it's the only thing that separates an additive offset from "
   "high-loading curvature \u2014 they're degenerate on the axes we have.",
   "Degeneracy: a per-site Fabs = alpha*EC^beta explains ~100% of the Addis and Beijing offsets as "
   "well as an additive constant does, and no rearrangement of FTIR-EC data separates them because "
   "the x-axis is the quantity in question. One-pager: quartz_tor_campaign_onepager.md.")

# ------------------------------------------------------------------ 24. logistics
text_slide("Logistics",
   ["Committee meeting \u2014 2 Sept, 3 pm.  [FILL IN: invite resent? who has confirmed?]",
    "FTIR group talk \u2014 27 Aug. Deck built.",
    "AAAR \u2014 poster Thursday, session 9, 1\u20133 pm.  [FILL IN: forwarded to Ann?]",
    "   [FILL IN: funding / anything else she asked about]"],
   "Quick logistics. Committee meeting is the 2nd at three. Group talk on the 27th, deck is built. "
   "AAAR poster Thursday session nine.",
   "From 12 Aug: she asked to be forwarded the AAAR acceptance and the committee invite, and "
   "flagged that nobody appeared to have received the committee email. Fill the bracketed items "
   "before presenting.")

# ------------------------------------------------------------------ BACKUP
divider("Backup", "asked-for detail, and the threads that are still moving")

fig_slide("Backup \u2014 correcting for daytime-only sampling widens the Addis gap, not closes it",
   "f_aeronet_diurnal.png",
   "The obvious objection to the column check is that the photometer only sees daylight, and Addis "
   "has a nearly five-fold day-night swing. Addis's two retrieval windows straddle the cycle and "
   "cancel; Pasadena's single midday window sits in its own minimum. Correcting pushes the sites "
   "further apart.",
   "Ratios are medians of per-day 24h/AERONET-hours values: Addis 0.92-1.05, Pasadena 1.07-1.09. "
   "Separation 2.9x -> 3.0-3.3x. Red BCc is 625 nm, effectively the HIPS wavelength. Assumes the "
   "column tracks the surface in relative diurnal shape. Delhi and Beijing have no co-located "
   "MA350. Uses the RAW 1-min files; the 9am-resampled pickles are daily and cannot answer this.")

fig_slide("Backup \u2014 the AERONET pairing was right all along",
   "f_truewindow_vs_sameday.png",
   "A literature reading said SPARTAN filters are eight staggered three-hour windows over nine "
   "days, which would have invalidated the whole column comparison. The metadata says otherwise: "
   "every clean Addis filter is exactly 24 hours, local midnight to midnight. I rebuilt the match "
   "on true windows anyway and it is identical \u2014 every point on the one-to-one line.",
   "95 filters; median |difference| 0 m; 0.0% relative. Identical because every almucantar "
   "retrieval is daytime and therefore always lands on the same UTC calendar day as the local "
   "filter day \u2014 the timezone edge never bites at these longitudes. My 'correction' was the "
   "error; the original pairing was sound.")

fig_slide("Backup \u2014 the HIPS instrument has three configuration epochs, and Addis straddles all of them",
   "f_instrument_epochs.png",
   "The lab documented two unexplained re-calibrations. The database's instrument telemetry "
   "columns are entirely empty, but the ratio of shipped transmittance to its normalised twin "
   "recovers the epoch structure anyway \u2014 two sharp network-wide breaks, then stable for "
   "three years. Addis has filters in all three epochs, so the offset can be re-fitted per epoch.",
   "Breaks at 2023-05-03 and 2023-09-22, matching hips.CalibrationSets events (collimator "
   "replacement, lab move). IMPORTANT: T1 itself is stable (674-818) across all three, and T1 is "
   "what enters tau \u2014 so this marks when the configuration changed, NOT that Fabs moved. "
   "ETAD: 8 filters in E1, 48 in E2, 240 in E3. Flat across the boundary kills the instrument-shift "
   "hypothesis; a step is a measurement-side account of the intercept. Untested.")

fig_slide("Backup \u2014 per-filter local calibration helps where the library has neighbours, and fails where it doesn't",
   "f_local_transfer_v1.png",
   "First pass at per-filter calibration: for each target filter, find its nearest library spectra "
   "and fit a small local model. Delhi and Pasadena improve markedly. Addis gets worse \u2014 its "
   "neighbourhood is Phoenix-dominated, which tells you something in itself about how unlike the "
   "library Addis is. Version two needs score-space similarity and a refuse-to-predict gate.",
   "k = 200 by Pearson-on-deriv2, ncomp = 8, Deming MAC-10. The textbook off-manifold failure: "
   "local is not the same as in-domain. A methods survey found no per-sample kNN-PLS prior art in "
   "the FTIR-aerosol literature \u2014 the template comes from soil and feed NIR (Shenk & "
   "Westerhaus 1997; Ramirez-Lopez 2013). TRANSFER_METHODS_2026-08-24.md.")

text_slide("Backup \u2014 baselining relocates seasonality rather than removing it",
   ["raw          all-year 2.35x\u22126.08   Dry 1.27x\u22121.80   Belg 2.21x\u22125.15   Kiremt 2.25x\u22125.06",
    "AIRSpec      all-year 0.93x\u22121.42   Dry 0.60x\u22120.22   Belg 0.84x\u22120.91   Kiremt 0.86x\u22120.73",
    "deriv2       all-year 1.61x\u22123.45   Dry 1.44x\u22122.86   Belg 1.36x\u22122.04   Kiremt 1.71x\u22123.93",
    "   Raw fails in the wet seasons; AIRSpec cleans them but leaves a dry-season slope deficit."],
   "One more thing the baseline does that we didn't expect. Raw spectra fail in the wet seasons. "
   "AIRSpec fixes the wet seasons but opens a dry-season slope deficit \u2014 0.60. So baselining "
   "doesn't remove the seasonal structure, it moves it. The corrected model's anomaly is the "
   "charcoal-heavy dry regime.",
   "Consistent with the char_06 dry-season spectral anomaly. This is the seed of ftir_24, the "
   "dry/wet deliverable still owed. Also relevant: the seasonal pattern in the predicted EC series "
   "itself is stable across every calibration choice \u2014 which she called 'hugely important' in "
   "August, because a seasonal signal that flipped with the calibration would be uninterpretable.")

prs.save(OUT)
print(f"saved {OUT.name}")
