"""ftir_group_2026-08-27.pptx; the comprehensive group-talk deck.

Everything the program has done, in the house style (docs/deck-style-prompt.md):
white 16:9, claim-as-title, one title-free figure per slide, SAY/NOTES on all.
Foundation-slide claims adapt the validated 2026-08-18 deliverable; this week's
slides carry the adjudication arc. Anchors re-validated live this session
(k6 / 1.585x-3.221 / 0.911 and k5 / 0.86x-1.615 / Deming 0.95x-2.09, exact).

Run: ~/anaconda3/bin/python build_deck.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
INK = RGBColor(0x22, 0x25, 0x2A)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]


def slide(title, fig=None, lines=None, say="", notes="", title_size=20):
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.22), Inches(12.5), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(title_size)
    p.font.bold = True
    p.font.color.rgb = INK
    if fig is not None:
        from PIL import Image
        iw, ih = Image.open(fig).size
        scale = min(12.2 / (iw / 165), 5.95 / (ih / 165), 1.35)
        w, h = iw / 165 * scale, ih / 165 * scale
        s.shapes.add_picture(str(fig), Inches((13.333 - w) / 2),
                             Inches(1.2 + (6.0 - h) / 2), Inches(w))
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
    return s


F = lambda n: FIG / n  # noqa: E731

# ============================================================ ACT 0; framing
slide(
    "FTIR–EC at Addis Ababa: from one bad crossplot to a five-city adjudication",
    lines=["Ahmad Jalil; UNBC / UC Davis AQRC; FTIR group, 27 Aug 2026",
           "Conventions: Deming λ* primary (MAC 10, λ ∝ MAC²), µg/m³ everywhere,",
           "protocol named on every figure; screening vs held-out results labelled."],
    say=("The full story in one talk, in five acts: what the problem is and what a "
         "raw spectrum actually contains; how cohort selection and baselining "
         "interact; including the analog fix; what the HIPS instrument itself "
         "turned out to be and the offset verdict; the five-site and season "
         "results; and the physics context plus the two measurements that settle "
         "the rest. Live app at the end; any variant anyone asks for, we can run "
         "in the room. If Satoshi joins late: the selection science starts at "
         "slide 9, the instrument verdict at slide 15, the new cross-site "
         "results at slide 20."),
    notes=("Option A = site-grouped 5-fold CV, first major minimum, site-disjoint "
           "80/20 (only option with a held-out TOR test). Option B = interleaved "
           "10-fold, within-5%-of-min, fit on all filters (network protocol). "
           "B2 = interleaved folds, first-major-minimum rule. Anchors validated "
           "live before this deck was generated."))

slide(
    "The problem: the deployed calibration reads Addis at 1.90x − 4.17",
    fig=F("deployed_alldata_crossplot.png"),
    say=("This is where the whole program starts: the network's deployed "
         "calibration against HIPS at Addis; slope 1.9, intercept minus 4.2 at "
         "MAC 10. About 20 inverse megameters of absorption with no EC to explain "
         "it, on a site where the satellite work needs ground truth. Everything "
         "in this talk is the anatomy of that line: how much is spectra "
         "processing, how much is cohort, how much is the instrument, and how "
         "much is Addis itself."),
    notes=("Deployed EC exists only for the fixed 190; at MAC 6 the same "
           "predictions read 1.14x-4.17 (intercept is MAC-invariant, ftir_19). "
           "The deployed model was trained under the network's interleaved "
           "protocol with operator-picked k (ftir_17 provenance)."))

# ==================================================== ACT 1; spectra basics
slide(
    "What a raw spectrum mostly is: ~90% Teflon background, and Addis rides a higher one",
    fig=F("f14_teflon.png"),
    say=("Before any calibration talk, what the instrument hands us. Left: all 239 "
         "Addis filters faint, one real filter in black; never an average; and "
         "the AIRSpec baseline under it dashed. Nearly everything is a smooth "
         "scattering slope from the PTFE membrane and the deposit; at the CH band "
         "about ninety percent of the measured absorbance is baseline. Right: "
         "that background per filter; Addis at a median of 0.17 versus 0.10 for "
         "the low-OC/EC IMPROVE cohort. Raw-spectra models can learn background "
         "that does not transfer; that is the mechanism behind half this talk."),
    notes=("Representative filter picked within 10 percentile points of the Addis "
           "median on four bands AND baseline height (median-CH-alone is a trap: "
           "that filter sits at the 92nd percentile of O-H). Background here = "
           "AIRSpec df1=6 baseline at 2920 cm-1."))

slide(
    "Baseline correction, shown on one real filter",
    fig=F("airspec_1_baseline.png"),
    say=("And what 'baseline-corrected, AIRSpec' means operationally: the "
         "smoothing-spline baseline fitted under the analyzed segments of this "
         "one real Addis filter. Subtract it and what remains is the chemistry "
         "the calibration should see. Keep this picture in mind whenever a slide "
         "says corrected."),
    notes=("APRLssb/AIRSpec, df1=6; validated Python port matches the R original "
           "to ~1e-7. Standing caveat for band work: the spline anchors at "
           "1520-1600 cm-1, which suppresses features near 1617 (slide 27's "
           "methods lesson)."))

# ============================================= ACT 2; protocols & the matrix
slide(
    "Six setups × three protocols: Option B reproduces the historical numbers; protocol moves less than setup",
    fig=F("f1_ladder.png"),
    say=("Provenance and naming. Option A is our site-grouped CV with a real "
         "held-out test; Option B is the network's interleaved protocol and "
         "reproduces the historical slide intercepts exactly; B2 separates the "
         "fold-structure from the k-rule. Two readings: which cohort you pick "
         "moves the answer far more than which protocol; the comforting result "
         "- and the baselined composition row sits nearest zero under all three."),
    notes=("Historical Option-B intercepts reproduced: -5.76 / -10.16 / -6.74 / "
           "-2.17. Fixed 190, Deming lambda*, MAC 10. Interleaved CV is "
           "row-order dependent (ftir_22); the honest reproducibility argument "
           "for Option A lives in NOTES if asked."))

slide(
    "Raw spectra, Option A: every cohort lands a negative intercept; selection alone does not fix raw",
    fig=F("f11_grid_raw_A.png"),
    say=("The six-panel grid grammar we will reuse: filled and solid is MAC 10, "
         "open and dashed is MAC 6, the black diamond is the shared Deming "
         "intercept; MAC choice pivots only the slope, never the intercept. On "
         "raw spectra every cohort lands between minus 2 and minus 10; "
         "lowest-OC/EC raw is minus 4.34 even with the best held-out score. No "
         "selection rescues raw spectra at Addis."),
    notes=("k under A: 15/4/10/4/6/5; blocked CV picks parsimonious models, as "
           "Satoshi predicted. Ethiopia-shaped's held-out 0.00 asterisk is the "
           "raw-selection artifact fixed on slide 14."))

slide(
    "Baseline-corrected, Option A: intercepts shrink everywhere; only lowest-OC/EC + AIRSpec keeps slope and TOR skill",
    fig=F("f12_grid_airspec_A.png"),
    say=("Same grid, corrected. Every diamond moves toward zero; but look at the "
         "held-out column for which small intercepts you can trust: network 0.66, "
         "smoke 0.40, Ethiopia-shaped 0.24; and lowest-OC/EC plus AIRSpec at "
         "0.90 with slope 0.95 and intercept minus 2.09. Baselining buys small "
         "intercepts almost anywhere; only the composition cohort also keeps a "
         "usable slope and passes the TOR test. That pairing is the program's "
         "locked headline."),
    notes=("Same conclusion under B and B2 (backup slides): -2.17 / -1.95 / "
           "-2.09 across protocols; the protocol-invariance Ann called the "
           "strongest property on record."))

slide(
    "How the composition cohort is picked: rank the pool by OC/EC, keep the lowest",
    fig=F("filtering_by_ocec.png"),
    say=("Mechanics for completeness, since selection is half this talk: rank "
         "every TOR-eligible IMPROVE filter by its OC-to-EC ratio and keep the "
         "lowest N; filters whose carbon is mostly elemental, so the EC signal "
         "isn't buried under organics. No spectra involved in the selection at "
         "all. The cutoff N is a lever we return to on slide 16."),
    notes=("Cut at 800 = OC/EC <= 2.27. Contrast with shape-based selection "
           "(next slides): eth-shaped ∩ lowest-OC/EC = exactly 1 filter; shape "
           "does not find composition (Ann's own overlap analysis, reproduced)."))

# ================================================ ACT 3; selection science
slide(
    "How spectral analogs are picked: score-space closeness to Addis, in either spectra space",
    fig=F("f13_analog_explainer.png"),
    say=("The other selection family. Two metrics per filter: Mahalanobis "
         "distance to the nearest Addis spectrum in a PLS score space, and a "
         "VIP-weighted spectral RMSE against the Addis median; where the model "
         "looks, mismatch counts more. Rank on the mean percentile, keep the top "
         "500. Top row is the committed raw-space selection; bottom is the "
         "identical recipe on baseline-corrected spectra; and the two pick "
         "almost completely different filters, which is the tell for the next "
         "slide."),
    notes=("'Top 500' resolves to 477 after TOR-eligibility (486 corrected); "
           "state resolved n. Machinery: IMPROVE-HIPS PLS (k=17, n=916), full "
           "ftir_09 recipe refit in corrected space."))

slide(
    "Selecting analogs on corrected spectra fixes them: −9.7 → −4.3, with real TOR skill",
    fig=F("f2_analog_fix.png"),
    say=("The analog headline from last week, kept because it frames everything "
         "since: same top-500 selection, same calibration, only the similarity "
         "matching moves to corrected spectra; and the Deming intercept goes "
         "from minus 9.7 to minus 4.3, the slope from 3.2 to 1.9, held-out from "
         "0.62 to 0.71. The analogs start behaving like a composition cohort "
         "instead of the failure case."),
    notes=("Ann's follow-up ('use corrected for BOTH selection and calibration') "
           "is what led into the dense sweep and the 440-490 basin (slide 16)."))

slide(
    "Why it works: raw analog similarity was matching Teflon; 4 of 477 filters survive corrected re-selection",
    fig=F("f3_overlap.png"),
    say=("Satoshi's Teflon suspicion, confirmed: switch the similarity space and "
         "the analog pool changes almost completely; four of 477 filters "
         "survive. The raw-space matching was ranking filter background, not "
         "aerosol. Ethiopia-shaped is the control: 285 of 300 survive, because "
         "band-shape features are less baseline-dominated. And the overlap "
         "request from the meeting: Ethiopia-shaped and lowest-OC/EC share "
         "exactly one filter; shape and composition select disjoint worlds."),
    notes=("Companion analog-lab result (this week): the committed analog "
           "ranking's agreement with plain similarity collapses on deriv2 "
           "(rho=-0.25) and AIRSpec (~0); baseline-dominated by construction."))

slide(
    "The Ethiopia-shaped rescue: nearly the same filters, but held-out TOR goes 0.00 → 0.63",
    fig=F("f4_eth_rescue.png"),
    say=("The mirror image. Ethiopia-shaped keeps 285 of 300 filters under "
         "corrected selection; the pool barely changes; but the model goes "
         "from zero held-out skill to 0.63. So the asterisk it carried in the "
         "matrix was a selection-space artifact, not a failure of shape-matching "
         "itself. Fifteen filters were enough to poison the calibration."),
    notes=("The 15 swapped filters also move k 10 -> 20 (same first-major-min "
           "rule both sides). Shape cohorts still fail the held-out floor at "
           "essentially every cutoff in the wide grid; rescued, not competitive."))

slide(
    "Satoshi's k question, answered: 21 components instead of 9 does not fix the analogs",
    fig=F("f5_k_sweep.png"),
    say=("The direct question from the Satoshi meeting; should the analogs use "
         "around 21 components? Scanning k from 4 to 24 under both CV schemes: "
         "the intercept never beats about minus 9 and drifts worse past 12, and "
         "held-out skill peaks exactly at k equals 9. It was never the component "
         "count; it was the selection space."),
    notes=("Also covers the one-consistent-k-rule request: rule choices are "
           "marked; conclusion is not rule-sensitive here. Quote k always with "
           "scheme incl. fold count (grouped floors are count-sensitive, "
           "ftir_32)."))

slide(
    "Cohort size is a real lever: the dense sweep found a basin below the locked 800",
    fig=F("f_cutoff_basin.png"),
    say=("Cutoffs, taken seriously: sweep the lowest-OC/EC cutoff every ten "
         "filters under corrected calibration. Intercept on top, held-out below. "
         "The shaded 440-to-490 basin is where both are best simultaneously; "
         "held-out 0.92 against 0.87 at our locked 800, intercept around minus "
         "1.2. And the honesty number: under bootstrap re-selection this winner "
         "takes 62.5 percent of draws; a leading frozen candidate, not a final "
         "answer."),
    notes=("ftir_11's '800 sweet spot' was a coarse-grid artifact (never scanned "
           "below 600). Basin wants k=8-9; at rule-k=5 it ties the 800; 'rules "
           "stop early' (ftir_23). Runner-up family: corrected analog-440, ~1/3 "
           "of draws. Source-site slope interval still wide: 1.03 [0.65, 1.50]. "
           "Coarse-cutoff prequel figure (f6) available if asked."))

# ================================================== ACT 4; the instrument
slide(
    "The HIPS scattering correction is a regression through blanks, and heavy filters sit beyond its support",
    fig=F("f3_blankline_geometry.png"),
    say=("Now the instrument itself. We decoded the HIPS math from the batch "
         "export: tau is the log of intercept-plus-slope-times-R1 over T1, and "
         "that line is fitted through field blanks. Top: the lot-251 blank cloud "
         "with the deployed linear line and a quadratic refit. Bottom: each "
         "site's darkness distribution; the shaded zone is darker than every "
         "blank, and 36 percent of Addis filters live there. Their Fabs stands "
         "on an extrapolated calibration. So we measured what that is worth."),
    notes=("Formula verified <1e-4 on 3,859 shipped tau values. PTFE zero from "
           "575 blanks: +0.32 +/- 0.46 Mm-1 at Addis geometry; that branch is "
           "dead. Lot-253 blanks are ~10x tighter than the big lots (rms 3.8 vs "
           "31-42 counts)."))

slide(
    "The Addis intercept survives the instrument check: the blank-line share is ~15%",
    fig=F("f1_intercept_blankline_ladder.png"),
    say=("Every filter's Fabs recomputed under three blank-line variants, then "
         "refit with per-filter weighted York regression; errors-in-variables "
         "with the real HIPS uncertainties. Addis moves from minus 1.51 to minus "
         "1.27 and stays seven sigma from zero under the harshest variant. "
         "Delhi's point estimate matches Addis but its error bar reaches zero. "
         "Beijing, Bishoftu and Pasadena bracket zero. One certain offset in "
         "five cities: Addis."),
    notes=("York 2004; sigma_y inflated to MSWD=1 (conservative). Scope caveat: "
           "blanks sample tau~0 only; high-loading nonlinearity is unprobed and "
           "still degenerate with real absorption on FTIR-EC axes; that is what "
           "quartz TOR terminates. Addis offset ~13-15 Mm-1."))

slide(
    "Pasadena's slope anomaly was the blank line; Delhi's 1.8x is real",
    fig=F("f2_slope_blankline_ladder.png"),
    say=("Same treatment, slopes; and two anomalies dissolve into methodology. "
         "Pasadena read 3.15x on the deployed line; under a quadratic blank line "
         "it is 0.91; at their loadings, blank-line shape error dominates tau. "
         "Beijing's old 1.48 was unweighted leverage; properly weighted, 0.98. "
         "Delhi's 1.8 survives every variant; that one is real, and still open: "
         "genuinely low site MAC, or the model reading oxidized organics as EC."),
    notes=("Delhi correlate: within-site r(residual, carbonyl/m3)~0.9 raw, "
           "collapsing when detrended by the site's own line; carbonyl marks "
           "which SITES over-read, not which filters scatter. Queued splitter: "
           "the MA350 880-nm channel as an independent per-filter anchor at all "
           "four sites (ftir_28 has only ever run at Addis)."))

slide(
    "Lot 248 isn't a partial download, it's a two-month lot",
    fig=F("f_lot_census_improve.png"),
    say=("The pool mystery, closed with a census: the database holds about "
         "fourteen hundred lot-248 FTIR analyses network-wide; the lot served "
         "for roughly two months around January 2021. The 1,299-filter download "
         "was essentially complete. So lot-248 cohorts are winter-2021 cohorts: "
         "a season confound, not missing data."),
    notes=("IMPROVE ftir_catalog: lot 248 = 1,362 analyses, 2020-12-17 to "
           "2021-02-15; lot 251 = 11,843 over a year. Red = 248, blue = 251, "
           "amber = 253; and 253/255 being the recent big lots is what enables "
           "the independent-lot validation (backup slide)."))

slide(
    "Baseline correction removes the lot-to-lot difference: the mechanism Ann predicted",
    fig=F("f_lot_baseline_removes_lot_effect.png"),
    say=("Your hypothesis from two weeks ago, tested where both lots ran "
         "simultaneously; Beijing, 178 days. On raw spectra the two lots "
         "disagree by four and a half inverse megameters against a common line; "
         "baseline-corrected, that collapses to about one and is "
         "indistinguishable from zero, at both cohort sizes. So part of what "
         "AIRSpec buys is lot insensitivity. But the lot is not the Addis story "
         "- Bishoftu shares Addis's lots and has no offset."),
    notes=("ocec-800: raw -4.49 [-7.01,-1.96] vs airspec -0.86 [-3.66,+2.02] "
           "n.s.; ocec-450: -4.35 vs -1.43 n.s. (4,000 bootstrap draws, Option "
           "A, n=14 vs 34; small; loading not fully matched, 1.18 vs 1.66 "
           "ug/m3 median). Residual post-baseline lot term ~1 Mm-1 = ~5% of the "
           "~21.5 Mm-1 offset; Addis-vs-Bishoftu is a within-lot-251 contrast."))

# ============================================ ACT 5; five sites & seasons
slide(
    "Four more cities' spectra, pulled and evaluated, plus a lesson about baselines",
    fig=F("cross_site_spectra_2026-08-23.png"),
    say=("The new data this month: raw FTIR spectra for Bishoftu, Beijing, Delhi "
         "and Pasadena, pulled straight from the SPARTAN database and built into "
         "the explorer as evaluation sites. Median spectra by site, in three "
         "baseline treatments; and the treatments disagree about chemistry: the "
         "1617 band we chased tracked the offset in raw space, was Addis-only in "
         "AIRSpec space, and under a neutral baseline shows at Addis AND "
         "Bishoftu; an Ethiopian regional marker, decoupled from the offset, "
         "because Bishoftu has the band and no offset. Methods rule extracted: "
         "any claim in 1500-to-1650 needs two baselines, because AIRSpec anchors "
         "right under that band."),
    notes=("Three baselines gave three answers; full reversal arc preserved in "
           "BAND1617_LEAD_2026-08-23.md; presentable as tested-and-corrected. "
           "Composition evidence for the offset now rests on the AERONET "
           "envelope link and BC/PM2.5 = 23% at Addis (3x any other site), not "
           "the band."))

slide(
    "An exhaustive search: no cohort, baseline, or k reconciles Addis and Delhi",
    fig=F("f4_screening_cloud.png"),
    say=("With five sites in the tool, the brute-force question: is there ANY "
         "combination that works everywhere? Every cohort, cutoffs 100 to 2000 "
         "in steps of ten, three spectral spaces, swept components; nine and a "
         "half thousand configuration-by-site jobs. Grey fails the held-out "
         "floor, blue passes, the star is the winner at Addis. Exactly one "
         "configuration lands near one-to-one at both Addis and Delhi, it still "
         "carries material offsets, and it fails freshly reconstructed holdout "
         "filters. The Addis-Delhi difference is not a calibration choice."),
    notes=("Axes are capped at |slope-1| <= 3.2 and |intercept| <= 14; the 58 "
           "extreme variants beyond them (all far from 1:1) are counted on the "
           "figure, never silently dropped. 71,263 scored rows, dedup-keyed, "
           "resumable. Grid-scale scoring "
           "lesson: |intercept|+w|slope-1| is gamed by flat slopes (0.44x with "
           "~0 intercept 'wins'); rankings are slope-boxed 0.85-1.18. Delhi's "
           "screening 'winner' (analogs-530 x deriv2) was 96% extrapolated and "
           "collapsed on holdout; the validation layer catching exactly what "
           "it was built to catch."))

slide(
    "Baselining relocates seasonality: the dry season becomes the anomaly",
    fig=F("f_season_panels.png"),
    say=("The season check you called hugely important, stratified under both "
         "baselines. Left, raw: the year-round 2.35x hides wet seasons at 2.2 "
         "and Dry at 1.3; raw's failure is a wet-season phenomenon. Right, "
         "corrected: wet seasons snap to about 0.85 with small intercepts, and "
         "Dry drops to 0.60. Baselining does not remove the seasonal structure; "
         "it moves it onto the charcoal-heavy dry season; the reverse of the "
         "direction we guessed. The overall seasonal ordering of concentrations "
         "stays stable across calibrations, which remains the good news."),
    notes=("Per-season Deming at lambda* on restricted ranges; indicative; the "
           "interaction fit is the ftir_24 deliverable. Consistent with "
           "char_06's dry-season spectral anomaly and ftir_22's dry-separates "
           "residuals. Season schemes differ across sites (Ethiopian vs "
           "quarters); pooling needs a mapping decision."))

# ====================================== ACT 6; physics & independent checks
slide(
    "Mie theory grounds the MAC fork: bare BC tops out near 6; MAC 10 means aged, thickly coated particles",
    fig=F("f8_mac_mie.png"),
    say=("The physics frame for the 6-versus-10 fork. Mie calculations at the "
         "HIPS wavelength: bare soot ensembles top out around 5.6 square meters "
         "per gram; bare BC never reaches 6. Getting to 10 requires absorption "
         "enhancement near 2 from thick coatings; aged, internally mixed "
         "particles. So the MAC fork encodes a fresh-versus-aged mixing-state "
         "assumption, not calibration noise. For Addis-like fresh-combustion "
         "aerosol, that leans low-MAC; but the IMPROVE implied-MAC bridge "
         "centered Addis-like filters at 10, so the fork stays open pending "
         "quartz TOR."),
    notes=("PyMieScatt sweep at 633 nm; enhancement 1.8-2.0 for the coated "
           "branch. Implied-MAC bridge (ftir_16): 151,843 matched filters, "
           "median 11.96 overall, 10.05 in the Addis-like OC/EC<=2.27 subset."))

slide(
    "The AERONET diurnal confound is measured, and the Addis anomaly widens, not closes",
    fig=F("f_aeronet_diurnal.png"),
    say=("Independent witness one: the sun photometer above the same air. The "
         "obvious objection to comparing 24-hour filters with daytime-only "
         "retrievals is now measured with the co-located minute-resolution "
         "MA350: Addis has a five-fold diurnal swing, but its two retrieval "
         "windows straddle the cycle and the biases cancel; Pasadena's midday "
         "window sits in its own minimum. Correcting for all of it moves the "
         "Addis-to-Pasadena separation from 2.9x to about 3.2x; the outlier "
         "gets more anomalous."),
    notes=("Ratios are medians of per-day ratios; assumes the column tracks the "
           "surface's relative diurnal shape; Delhi/Beijing have no co-located "
           "MA350 and stay uncorrected. Red BCc at 625 nm; effectively the "
           "HIPS wavelength."))

slide(
    "Addis reports more absorption than its own column can account for",
    fig=F("f_aeronet_column_check.png"),
    say=("Witness one, continued: effective absorption scale height; column "
         "absorption over surface absorption; is 232 meters at Addis against "
         "544 to 894 everywhere else. And the right panel is the part no filter "
         "artifact can produce: the filters' corrected organic envelope predicts "
         "the PHOTOMETER's column absorption, beyond loading, beyond the "
         "filter's own Fabs, within months. The filter chemistry knows something "
         "real about the sky above it."),
    notes=("H is comparative-only, not a physical height. Envelope partial "
           "r=+0.28, perm p~0.002, n=125 (the discrete-1620-band version of "
           "this died under month controls; envelope is the honest carrier). "
           "Level 1.5 retrievals; Gurgaon ~30 km from the Delhi site."))

# ==================================================== ACT 7; close
slide(
    "Adama is not Addis: normal OC/EC, no HIPS-EC gap; more Adama samples won't arbitrate this",
    fig=F("f_adama_context.png"),
    say=("For Christian and Sina, previewed here: the five Adama TOR filters sit "
         "at OC-to-EC around six; the IMPROVE pool median, nowhere near the "
         "low-OC/EC regime the Addis question lives in; and Bishoftu absorbs "
         "half of what Addis does. Adama characterizes a different aerosol; the "
         "decisive Adama-side contribution would be quartz filters for TOR, not "
         "more Teflon."),
    notes=("This figure is the SEED of the two-pager; the HIPS + TOR-OC/EC vs "
           "FTIR-OC/EC comparison panel still needs adding (ftir16 "
           "adama_batch54 tables). Draft this week."))

slide(
    "The calibration explorer: the Shiny-app idea, rebuilt in Python for this research",
    fig=F("f_app_calibrate.png"),
    say=("A big part of the month's work is this tool, so it gets its own "
         "section. It is a Flask app, the same idea as the group's Shiny "
         "calibration app but built for research iteration: the sidebar holds "
         "the whole configuration space, cohort by cutoff by baseline by "
         "protocol by k, and Run gives the Addis readout in seconds because "
         "everything is cached. Auto-run re-fits on any change. Every number "
         "in this deck came out of this app, and every figure regenerates "
         "from it."),
    notes=("Flask on :5058; presets mirror the six setup-matrix rows; "
           "data-checks verify the locked cohorts at startup (green checks in "
           "the sidebar). Ann has seen an earlier version; new since then: "
           "the sidebar itself, the Sites and HIPS tabs, validation "
           "guardrails (Q-residual + score distance on every run), and tab "
           "deep-links for live demos."))

slide(
    "Five-site evaluation is one click, and every prediction carries a domain check",
    fig=F("f_app_sites.png"),
    say=("The Sites tab: the current configuration evaluated against all five "
         "cities at once, with the extrapolation diagnostics on every row. "
         "The on-screen run is the default raw-800 setup, and you can see "
         "Addis and Delhi carrying the big intercepts while the score and "
         "Q-residual columns flag which rows are out of domain. Below it, the "
         "median spectra of every site under switchable baselines. The HIPS "
         "blank-line lab is the same idea, one click. [Live now if time "
         "allows: switch to AIRSpec, re-run, watch the ladder move.]"),
    notes=("Screenshot shows raw-800 Option A (the demo default), NOT the "
           "winner; the winner readouts are the earlier slides. Q% column = "
           "share above training-p95 orthogonal spectral residual, the "
           "validation-layer addition. HIPS-tab screenshot is in backup. "
           "Fallback if a live demo dies: deep-links #tab=sites&run=1 and "
           "#tab=hips&run=1 reproduce these views headlessly."))

slide(
    "The leaderboard by city: every site optimizes to a different cohort size",
    fig=F("f_out_leaderboard_by_site.png"),
    say=("First app output worth a slide on its own. Ask the 71,000-row grid "
         "for each city's best configuration, slope-constrained and held-out "
         "gated, and no two cities agree: Pasadena wants 120 filters, Addis "
         "440, Bishoftu a thousand, Beijing eighteen hundred and fifty. "
         "Cohort choice is doing per-site work that one global calibration "
         "cannot do; and Delhi's apparent optimum is the cautionary tale, "
         "96 percent extrapolated and dead on holdout. This is the "
         "quantitative version of 'one calibration will not serve the "
         "network'."),
    notes=("Slope box 0.85-1.18 (the flat-slope scoring trap otherwise "
           "'wins' at 0.44x), Option A, held-out TOR R2 >= 0.85, score = "
           "|intercept| + 0.5|slope-1|. Screening numbers; per-site winners "
           "are candidates, not conclusions (winner's-curse discipline as "
           "on the stability slide). Winner readouts per city in "
           "FIVE_SITE_GRID_2026-08-23.md with the corrected-audit table."))

slide(
    "Cross-applied, every 'best' fails abroad; only the Ethiopian pair transfers",
    fig=F("f_out_cross_application.png"),
    say=("The natural follow-up: take each city's best and apply it to the "
         "other four. Rows are calibrations, columns are cities, dark is "
         "good, bold is the home fit, and an exclamation mark means over 30 "
         "percent of that city's filters are extrapolations for that model. "
         "Three readings. The diagonal is dark: each best works at home. The "
         "off-diagonal mostly is not: Delhi's best collapses everywhere else, "
         "and Pasadena's tiny 120-filter cohort is unusable abroad. And the "
         "one genuine transfer: Addis's best lands 0.99x at Bishoftu; the "
         "two Ethiopian sites cohere, which is exactly what you would want "
         "a regional calibration to do."),
    notes=("Numbers pulled live from /api/run per (config, city): all-pairs "
           "Deming at MAC 10, score = |intercept| + 0.5|slope-1| clipped at "
           "4 for color. Delhi's row inherits its 96%-extrapolated screening "
           "artifact (every cell flagged); Beijing's 1850-filter best is the "
           "most tolerable row abroad but still misses Delhi (1.56x-2.20). "
           "Reinforces both the no-universal-calibration grid result and "
           "the Addis-Bishoftu regional coherence from ETBI_FIRST_LOOK."))

slide(
    "The analog lab's verdict: the committed analog ranking is baseline-dominated",
    fig=F("f_out_analog_agreement.png"),
    say=("Second output: the Analogs tab compares our committed analog score "
         "against plain textbook similarity metrics, in three spectral "
         "spaces, live. On raw spectra they agree, rho about 0.7. Correct "
         "the baseline and the agreement collapses to zero; on second "
         "derivatives it goes negative. And the top-500 overlap is five to "
         "eight percent everywhere. Translation: what the committed score "
         "was ranking is mostly the baseline, not the chemistry; that is "
         "the same lesson as the four-of-477 slide, now measured across "
         "every metric the literature uses."),
    notes=("Numbers pulled live from /api/analog_lab: rho(committed, cosine) "
           "= 0.68 raw / 0.07 AIRSpec / -0.25 deriv2; top-500 overlap "
           "4.8-8.2%. Metrics: cosine/SAM, Pearson-to-median (LOCAL's "
           "metric), Mahalanobis PCA-10 (Reggente 2016). Motivated the "
           "corrected-selection fix and, later, the per-filter LOCAL work."))

slide(
    "Validate-top-5: the winner takes 62.5% of re-selection draws, not all of them",
    fig=F("f_out_winner_stability.png"),
    say=("Third output, and the honesty slide: the validation layer re-runs "
         "the whole selection under bootstrap resampling of the target "
         "filters and under re-draws of the IMPROVE source sites. The "
         "frozen winner takes about 63 percent of draws both ways; the "
         "corrected-selection analog family takes a third. So OCEC-440 is "
         "the leading frozen candidate, not a definitive universal "
         "calibration, and we say so with a number."),
    notes=("From VALIDATION_LAYER_2026-08-24.md: target-filter wins "
           "62.5/37.5/0%; source-site wins 63.0/31.5/5.5%. Source-site "
           "slope interval for the winner stays wide: 1.03 [0.65, 1.50]. "
           "Frozen protocol for the lot-253 independent validation: "
           "OCEC-440 x AIRSpec, Option A, k=5; lot 255 sealed for "
           "replication."))

slide(
    "Three lines of evidence end at the same two measurements",
    lines=["Quartz TOR EC on collocated Addis filters: the only measurement that is a",
           "   function of neither axis; breaks the offset-vs-curvature degeneracy",
           "   (campaign one-pager written: ~36 filters, 3 seasons)",
           "Solvent extraction + HIPS re-measurement on archived Addis + Delhi filters,",
           "   Beijing/Pasadena controls: separates real absorbers from residual artifact",
           "Both use existing samples or the existing sampling plan."],
    say=("Where this all lands. The MAC fork, the Bishoftu contrast, and the "
         "curvature degeneracy all terminate at an independent EC measurement; "
         "quartz TOR is decisive, and the extraction test is the cheap companion "
         "on filters already in the archive. These are the two asks I want to "
         "leave the room with."),
    notes=("Extraction design: Kirchstetter/Novakov/Hobbs JGR 2004; water then "
           "methanol; drop only at Addis/Delhi = real organic absorbers; "
           "methanol-resistant dark residue = char/tar. TOR sizing from ftir_16: "
           "~3 sigma/day separation of MAC 6 vs 10 -> 11-13 days/season x 3."))

slide(
    "Logistics",
    lines=["AAAR: poster Thursday 1-3 pm, session 9. Flights + conference hotel booked",
           "Committee meeting: Sep 2, 3 pm. [FILL IN: confirm invites arrived]",
           "Adama two-pager for Christian & Sina: draft this week",
           "Satoshi 1:1: ready when scheduled; his variants run live in the app",
           "[FILL IN: lot-253 spectra pull; needs a session on the Windows/VPN machine]"],
    say=("Housekeeping: AAAR booked, poster Thursday one to three, session nine. "
         "Committee September 2 at three. Adama two-pager this week. And the "
         "Satoshi meeting can happen whenever it schedules; the tool is ready "
         "for it."),
    notes=("Lot-253 spectra pull staged (get_improve_lot253_spectra.ps1; pilot "
           "mode; lot 255 sealed for replication; TOR y-values already local)."))

# ==================================================== backups
slide("Backup", say="Backup material from here.", notes="", title_size=40)

slide(
    "Backup: raw grid under Option B2: the interleaved protocols agree with A on the story",
    fig=F("f15_grid_raw_B2.png"),
    say=("Raw six-panel under B2; for protocol-triangle completeness."),
    notes=("B2 fits on all filters (no held-out). Eth-shaped+AIRSpec under B2 "
           "(k=21) is the standing unstable cell."))

slide(
    "Backup: corrected grid under Option B: the AIRSpec verdict is protocol-robust (−2.17)",
    fig=F("f10_grid_airspec.png"),
    say=("Corrected under Option B: the locked setup reads minus 2.17 against "
         "minus 2.09 under A and minus 1.95 under B2; the number that does not "
         "move."),
    notes=("Option B reproduces historical numbers but has no held-out test; "
           "continuity only, never ranking."))

slide(
    "Backup: corrected grid under Option B2",
    fig=F("f16_grid_airspec_B2.png"),
    say=("And the B2 corrected grid closing the triangle at minus 1.95."),
    notes=("k under B2 corrected: 7/5/21/5/5; the eth-shaped 21 is the "
           "unstable cell."))

slide(
    "Backup: the corrected model's residuals no longer track extrapolation distance",
    fig=F("residual_vs_d2.png"),
    say=("The ftir_15 diagnostic under Option A: raw-model residuals track "
         "score-space distance; extrapolation-driven error; while the "
         "corrected model's residuals are a flat, season-stable constant. "
         "That flat line is what makes the remaining offset a real quantity "
         "instead of a model artifact."),
    notes=("Raw r=0.71, corrected ~-0.05 (site-held-out); reproduced under the "
           "app protocol at 0.87/0.33 (ftir_22)."))

slide(
    "Backup: bootstrap CIs: the corrected intercept is bounded away from zero",
    fig=F("bootstrap_intercept_ci.png"),
    say=("Bootstrap confidence intervals under Option A: the corrected "
         "lowest-OC/EC intercept sits at minus 1.2 to minus 1.8; disjoint from "
         "the raw model's interval and excluding zero."),
    notes=("ftir_15 committed draws: corrected CI [-1.78,-1.06]; survives the "
           "protocol change (ftir_22: app-protocol CI [-2.07,-1.03])."))

slide(
    "Backup: SPARTAN has moved to a lot the calibration has never seen",
    fig=F("f_lot253_takeover.png"),
    say=("Forward risk and opportunity: from early 2025 SPARTAN filters shift "
         "to lot 253; outside the 248-plus-251 calibration basis. The "
         "opportunity: IMPROVE ran five thousand TOR-EC filters on that same "
         "lot, so a frozen protocol tested there is a genuine independent-lot "
         "validation. Protocol frozen; lot 255 sealed for replication; only "
         "the scan spectra remain to pull."),
    notes=("433 SPARTAN lot-253 filters, 13 sites, 58% of everything since "
           "2025-09. Frozen: OCEC-440 x AIRSpec, Option A, k=5; tests "
           "lot+time transfer, not unseen-site. Delhi lot-253 subset is NOT "
           "identifiable (n=6 overlap, 4x loading, CI [-53,+304])."))

slide(
    "Backup: the HIPS lab tab, populated",
    fig=F("f_app_hips.png"),
    say=("The HIPS tab live: York fits under three blank-line variants per "
         "site, and the blank ledger underneath, keyed per deployed "
         "calibration line."),
    notes=("On-screen config is the raw-800 demo default. Ledger shows "
           "per-line blank counts, linear vs quadratic rms, R1 validity "
           "range, and the blank-tau zero."))

slide(
    "Backup: per-filter LOCAL calibration: helps where the library has support",
    lines=["Per target filter: k most-similar library spectra → small PLS (novel in the FTIR-aerosol literature)",
           "Delhi 1.37x (from 2.36x) and Pasadena 1.04x (from 4.22x): real improvement",
           "Addis 4.5x collapses: neighbors are the library's edge; local is not in-domain",
           "v2: score-space similarity + refuse-to-predict gates; then CORAL / di-PLS"],
    say=("First pass at per-filter local calibration: strong where the library "
         "has support, honest failure where it does not; which is itself the "
         "result. No selection scheme manufactures Addis support that isn't "
         "there."),
    notes=("local_lab.py (/api/local_cross); survey with citations in "
           "TRANSFER_METHODS_2026-08-24.md. Decisive next evaluation: simulated "
           "transfer at IMPROVE sites with collocated HIPS; pretend Fabs-only, "
           "run the full pipeline, unblind TOR once."))

prs.save(HERE / "ftir_group_2026-08-27.pptx")
print(f"saved ftir_group_2026-08-27.pptx with {len(prs.slides._sldIdLst)} slides")

# notes-free twin for sharing/projection
for s_ in prs.slides:
    if s_.has_notes_slide:
        s_.notes_slide.notes_text_frame.text = ""
prs.save(HERE / "ftir_group_2026-08-27_no_notes.pptx")
print("saved ftir_group_2026-08-27_no_notes.pptx (notes stripped)")
