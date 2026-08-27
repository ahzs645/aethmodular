"""ftir_group_2026-08-27.pptx; the group-talk deck, rebuilt to Ann's
Aug 27 run-through edit plan.

Order: problem -> what a spectrum is + baselining -> how cohorts are picked
-> one before/after + the six baselined crossplots -> three ways of picking
the model -> cohort size -> filters/lots -> cross-site spectra -> seasons ->
the exhaustive search + winner crossplot + per-city results -> discussion
questions. Language: "baseline-corrected (using AIRSpec)" on first mention,
then "baseline-corrected"; calibration-set vs prediction-target phrasing;
intercepts in ug/m3 everywhere; no CV jargon in the main arc.

Run: ~/anaconda3/bin/python build_deck.py   (writes the notes deck and the
stripped no-notes twin)
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
        body = s.shapes.add_textbox(Inches(0.75), Inches(1.55), Inches(11.8), Inches(5.4))
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

# ---------------------------------------------------------------- 1 title
slide(
    "FTIR EC at Addis Ababa: one intercept problem, a huge space of choices",
    lines=["Ahmad Jalil; UNBC / UC Davis AQRC; FTIR group, 28 Aug 2026",
           "The arc: Addis is different; depending on which samples we calibrate on,",
           "how we baseline, and how we pick the model, we get a huge array of outcomes;",
           "then: what constraints and validation make a choice defensible?"],
    say=("Four parts today. One: Addis is different, and you can see it in the "
         "spectra themselves. Two: depending on which samples we use as the "
         "calibration set, whether we baseline-correct, and how we pick the "
         "model, we get a huge array of possible answers; I will show a few "
         "illustrative ones, then the full search. Three: what the filters and "
         "lots contribute. Four, and this is the part I want your input on: "
         "what constraints and what validation would make a choice defensible. "
         "Discussion questions are at the end."),
    notes=("Conventions: Deming regression (errors in both variables) at MAC "
           "10, intercepts in ug/m3 everywhere; 'baseline-corrected (using "
           "AIRSpec)' on first mention, then 'baseline-corrected'. Every "
           "result slide names its calibration set and its prediction "
           "target."))

# ---------------------------------------------------------------- 2 problem
slide(
    "The problem: the deployed calibration predicts Addis at 1.90x with a -4.2 µg/m³ intercept",
    fig=F("deployed_alldata_crossplot.png"),
    say=("Where this starts. Using the network's deployed calibration to "
         "predict the Addis samples, against HIPS: slope 1.9, intercept minus "
         "4.2 micrograms per cubic meter. Something reads absorption at Addis "
         "that the predicted EC cannot explain, at a site where the satellite "
         "work needs ground truth. Everything today is the anatomy of that "
         "line."),
    notes=("Deployed EC exists only for the fixed 190 filters. At MAC 6 the "
           "same predictions read 1.14x-4.17: the Deming intercept is "
           "MAC-invariant, MAC only pivots the slope. x-axis is Fabs/MAC in "
           "ug/m3."))

# ------------------------------------------------- 3-5 spectra + baselining
slide(
    "A raw FTIR spectrum is mostly filter background, not chemistry",
    fig=F("f14a_spectrum.png"),
    say=("Before any calibration: what the instrument hands us. Every faint "
         "grey line is one of the 239 Addis filters; the black line is a "
         "single real filter, not an average; the red dashed line is the "
         "background fitted underneath it. Nearly everything is a smooth "
         "scattering slope from the PTFE filter and the deposit. At the CH "
         "band, about ninety percent of what we measure is background."),
    notes=("Representative filter picked within 10 percentile points of the "
           "Addis median on four bands AND background height; picking on the "
           "CH band alone is a trap (that filter sits at the 92nd percentile "
           "of O-H). Split from the old two-panel figure at full resolution "
           "per the run-through (the pasted pair rendered blurry)."))

slide(
    "and Addis rides a higher background than the calibration samples",
    fig=F("f14b_background_hist.png"),
    say=("The size of that background, one value per filter: Addis in red at "
         "a median of 0.17, the low-OC/EC IMPROVE calibration samples at "
         "0.10. A model trained on raw spectra can learn background that "
         "does not transfer from the calibration set to Addis. That is the "
         "mechanism behind most of what follows."),
    notes=("Background = fitted baseline height at 2920 cm-1. Companion "
           "half of the previous figure, full resolution."))

slide(
    "Baseline correction (using AIRSpec): fit the background and remove it",
    fig=F("airspec_1_baseline.png"),
    say=("What baseline-correcting means operationally: fit a smooth baseline "
         "under the analyzed segments of each spectrum and subtract it, so "
         "the calibration sees chemistry instead of background. We use the "
         "baseline-correction step of AIRSpec for this; from here on I will "
         "just say baseline-corrected."),
    notes=("AIRSpec also computes functional groups; we use ONLY its "
           "baseline-correction step (Ann's language rule). Validated Python "
           "port of APRLssb, matches R to ~1e-7. Nagendra, Mona and Naveed "
           "know AIRSpec, so the name lands with this group."))

# ------------------------------------------------- 6-7 how cohorts are picked
slide(
    "Choosing calibration samples by composition: keep the lowest-OC/EC IMPROVE samples",
    fig=F("filtering_by_ocec.png"),
    say=("First way of choosing a calibration set. Addis sits at the extreme "
         "low end of the OC-to-EC ratio, in the red range, where the IMPROVE "
         "pool has very few samples. So: rank every IMPROVE sample by OC/EC "
         "and keep the lowest N as the calibration set, to predict the Addis "
         "samples. We started with N of 800 as a compromise between being "
         "composition-like and having enough samples; how arbitrary that was "
         "comes up in a few slides."),
    notes=("Cut at 800 = OC/EC <= 2.27. Addis barely overlaps the pool at "
           "all, which seeds discussion question 4 (is spectral similarity "
           "more defensible than OC/EC similarity, given the non-overlap?)."))

slide(
    "Choosing by spectral shape, and baselining changes which samples get picked",
    fig=F("f13b_analog_ranked.png"),
    say=("Second way: rank the whole pool by how close each spectrum is to "
         "the Addis spectra, and keep the closest 500 as the calibration "
         "set. Top: ranking computed on raw spectra. Bottom: the identical "
         "recipe on baseline-corrected spectra. The shape of the ranking "
         "changes completely, and a clear near-Addis population emerges "
         "only after baselining; the two rankings share almost no filters. "
         "So even the choice of similarity space is a choice that matters."),
    notes=("Left diagnostic scatter panels removed per the run-through (not "
           "intuitive). Under corrected selection only 4 of 477 raw-selected "
           "filters survive: the raw similarity was matching filter "
           "background, exactly Satoshi's suspicion. 'Top 500' resolves to "
           "477 after TOR eligibility."))

# --------------------------------------- 8-9 before/after + baselined grid
slide(
    "Baselining transforms the answer: the intercept moves from -4.3 to -0.6 µg/m³, but the slope collapses",
    fig=F("f_before_after_baseline.png"),
    say=("The one before-and-after graph. Same calibration set, the entire "
         "IMPROVE network; same Addis prediction target; the only change is "
         "baseline correction. The intercept goes from minus 4.3 to minus "
         "0.6. But look at the slope: 0.43. Baselining alone buys a small "
         "intercept almost anywhere, at the price of a collapsed slope; "
         "which samples you calibrate on decides whether you keep both. "
         "That is the next slide."),
    notes=("Site-grouped model selection (the held-out variant); network raw "
           "k=15, corrected k=7. This replaces the two full six-panel grids "
           "(raw and corrected) from the earlier cut, per the run-through: "
           "one before/after, then the six baselined crossplots only."))

slide(
    "All six calibration sets, baseline-corrected: only lowest-OC/EC keeps slope AND held-out skill",
    fig=F("f12_grid_airspec_A.png"),
    say=("The six candidate calibration sets, all baseline-corrected, all "
         "predicting Addis. Filled points are MAC 10, open are MAC 6, and "
         "the black diamond is the shared Deming intercept, which MAC cannot "
         "move. Every intercept is now small. The held-out test in each box "
         "tells you which small intercepts to trust: the network at 0.66, "
         "smoke 0.40, Ethiopia-shaped 0.24, analogs 0.59, and lowest-OC/EC "
         "at 0.90 with slope 0.95 and intercept minus 2.09. Composition "
         "selection plus baselining is the pairing that keeps everything."),
    notes=("Held-out here = TOR test on held-out IMPROVE sites (calibration "
           "quality), not an Addis holdout; the Addis-side validation gap is "
           "discussion question 2. Suptitle on the figure says "
           "'baseline-corrected (AIRSpec)' which matches the language rule."))

# ---------------------------------------------------- 10 model selection
slide(
    "Three ways of selecting the model; the red circles are how all deployed SPARTAN calibrations are made",
    fig=F("f1_ladder.png"),
    say=("One more choice: how the model itself is selected. There are three "
         "reasonable recipes; I will not walk through their internals. The "
         "red circles are the way the current calibration app, the Shiny "
         "app, does it, which is how every deployed SPARTAN calibration was "
         "made; the blue and gold are two other defensible ways stacked on "
         "top. Two things to see: the choice of calibration set moves the "
         "answer far more than the choice of recipe, which is comforting; "
         "and the baseline-corrected composition row sits nearest zero "
         "under all three."),
    notes=("Jargon stripped per the run-through (no interleaved/site-grouped "
           "/ within-5% language without an RMSE-vs-components CV curve, "
           "which the group agreed not to show). If pressed: red = "
           "interleaved 10-fold CV, k within 5% of minimum, fit on all "
           "filters; blue = site-grouped 5-fold, first major minimum, "
           "site-disjoint 80/20; gold = interleaved folds with the "
           "first-major-minimum rule. Historical deployed intercepts "
           "reproduce exactly on the red circles: -5.76/-10.16/-6.74/-2.17."))

# ---------------------------------------------------- 11 cohort size
slide(
    "Cohort size: the intercept barely cares, the held-out R² does; 800 was somewhat arbitrary",
    fig=F("f_cutoff_basin.png"),
    say=("How many calibration samples should the cut keep? Sweeping the "
         "cutoff every ten filters: the intercept, top panel, is remarkably "
         "stable across the whole range. The held-out R-squared, bottom, is "
         "not: it bounces around at small N and only settles once the "
         "calibration set gets large. So our 800 was somewhat arbitrary; "
         "the shaded 440-to-490 basin is where both are best simultaneously, "
         "and the number of filters is a genuine decision, not a detail. "
         "That is one of the questions at the end."),
    notes=("This sweep is under the site-grouped selection recipe; the "
           "Shiny-style recipe's version would differ (flagged in the "
           "run-through). Intercept stability + R2 instability was Ann's "
           "own reading. Basin held-out 0.92 vs 0.87 at 800; under "
           "re-selection the basin winner takes 62.5% of draws "
           "(stability slide later)."))

# ------------------------------------------------- 12-14 filters and lots
slide(
    "The filters themselves differ by site: what's on the filter, not what's in the air",
    fig=F("f_site_darkness.png"),
    say=("Now the filter side. R1 is the reflectance measured at the filter, "
         "in detector counts; lower means a darker deposit. The sites "
         "separate: Addis filters are the darkest population, Pasadena the "
         "lightest, and Delhi sits in the middle despite Delhi's air, "
         "because sampled volume differs by site; Delhi collects less "
         "volume per filter. So this is a statement about deposits, not "
         "about the atmosphere, and it means the calibration meets "
         "filter-darkness regimes it has rarely seen."),
    notes=("Rebuilt per the run-through: axes defined in words, fit lines "
           "and blank cloud removed, and the mislabeled blank count "
           "dropped (a lot can carry more than one calibration set; the "
           "old 'lot-251 blanks n=373' conflated two). T1 = transmittance "
           "counts if asked. The blank-line extrapolation analysis lives "
           "in backup."))

slide(
    "Lot 248 isn't a partial download; it's a genuine two-month lot",
    fig=F("f_lot_census_improve.png"),
    say=("The lot timeline, which also shows when filters were analyzed. "
         "Lot 248 really does hold only about fourteen hundred analyses, "
         "December 2020 to February 2021; my 1,299-filter pull was "
         "essentially all of it. Lot 251 is the year-long workhorse with "
         "nearly twelve thousand. So lot-248-trained calibrations are "
         "winter-2021 calibrations; a season note, not missing data. And "
         "the older lots were analyzed years ago, on instruments that have "
         "changed since."),
    notes=("Ann liked this slide as-is. Red = 248, blue = 251, amber = 253. "
           "IMPROVE is done with 253/255; SPARTAN's 253 is being analyzed "
           "now. 253/255 being large recent lots is what enables an "
           "independent-lot validation later."))

slide(
    "Baseline correction removes the lot-to-lot difference; the mechanism Ann predicted",
    fig=F("f_lot_baseline_removes_lot_effect.png"),
    say=("Do the lots behave differently? Beijing used lots 248 and 251 "
         "side by side for 178 days, so we can test it. On raw spectra the "
         "two lots disagree by about four and a half inverse megameters "
         "against a common line; baseline-corrected, the difference "
         "collapses to about one and is indistinguishable from zero. So "
         "part of what baselining buys is lot insensitivity. And the lot "
         "is not the Addis story: Bishoftu shares Addis's lots and has no "
         "offset."),
    notes=("n = 14 vs 34, bootstrapped; loadings not perfectly matched "
           "(median predicted EC 1.18 vs 1.66 ug/m3). Residual "
           "post-baseline lot term ~1 Mm-1, about 5% of the Addis offset."))

# ---------------------------------------------------- 15 cross-site spectra
slide(
    "Median baseline-corrected spectra, five cities: Addis has the least organic signal",
    fig=F("f_cross_site_spectra_v2.png"),
    say=("The new data this month: spectra for four more SPARTAN cities, "
         "pulled from the database and evaluated alongside Addis. Median "
         "baseline-corrected spectrum per city. Left: the O-H and N-H "
         "region and the sharp CH peaks near 2920. Right: the carbonyl "
         "region around 1700. Delhi and Beijing carry the biggest organic "
         "signals; Bishoftu is surprisingly high in O-H; and Addis, the "
         "most polluted-feeling site, has close to the least organic "
         "absorbance. Which is actually consistent: Addis's OC-to-EC is "
         "the lowest we have; most of its carbon is black carbon, and "
         "these bands are organic functional groups. The spectra are "
         "partly telling us what OC/EC already said."),
    notes=("Ann's talking point from the run-through folded in (low "
           "organics at Addis consistent with very low OC/EC). The 1617 "
           "band panel was cut per the run-through (its rationale needs "
           "the other deck's material); the carbonyl-vs-intercept scatter "
           "in ug/m3 is in backup if it comes up. Bands labeled: CH ~2920, "
           "carbonyl ~1700."))

# ---------------------------------------------------- 16 seasons
slide(
    "Baselining shifts Addis seasonality but does not remove it; the dry season becomes the anomaly",
    fig=F("f_season_panels.png"),
    say=("Seasonality, which we have tracked all along. Left, raw: the wet "
         "seasons are the outliers. Right, baseline-corrected: the wet "
         "seasons snap to about 0.85x and the dry season drops to 0.60x. "
         "So baselining does not remove the seasonal structure; it moves "
         "it onto the charcoal-heavy dry season, while improving the "
         "intercept a lot. Seasonality is shifted, not solved."),
    notes=("Per-season Deming on restricted ranges; indicative, the formal "
           "interaction fit is the ftir_24 deliverable. The overall "
           "seasonal ordering of concentrations stays stable across "
           "calibrations, which remains the good news."))

# ------------------------------------- 17-21 the search and its outputs
slide(
    "So we tried everything: 12,000+ configurations per city, scored on the same two numbers",
    fig=F("f4_screening_cloud.png"),
    say=("Instead of graph after graph, we let the computer run the whole "
         "space: every calibration-set family, cutoffs from 100 to 2000 in "
         "steps of ten, raw and two baseline treatments, swept components; "
         "each dot is one configuration predicting Addis. Horizontal axis: "
         "how far the slope is from one. Vertical: the intercept size in "
         "micrograms per cubic meter. You want the bottom-left corner. "
         "Grey fails a held-out quality floor; blue passes; the star is "
         "the best passing configuration. Notice the V: you can buy a zero "
         "intercept with a terrible slope, so we always constrain both."),
    notes=("58 extreme variants beyond the capped axes, counted on-figure. "
           "71k scored rows total across five cities, all reproducible "
           "from the app. The scoring lesson: |intercept| alone is gamed "
           "by flat slopes; rankings are slope-boxed 0.85-1.18."))

slide(
    "How the search runs: the calibration explorer (live, if we have time)",
    fig=F("f_app_calibrate.png"),
    say=("How those thousands of runs happen: a Flask app in the spirit of "
         "the Shiny calibration app, but built for research; every choice "
         "in this talk is a control, every run is cached, and any variant "
         "this room proposes we can run before the meeting ends. I will "
         "demo it live if time allows; the search results on the next "
         "slides all come out of it."),
    notes=("On-screen config is the demo default (network raw 800), not "
           "the winner; say so if asked. Batches run server-side and "
           "survive closing the page; results land in a shared results "
           "file the leaderboard reads."))

slide(
    "The winner, as a crossplot; because the crossplot is what we look at",
    fig=F("f_winner_crossplot.png"),
    say=("The starred configuration, drawn the way we actually judge "
         "calibrations. Lowest-OC/EC 450 as the calibration set, "
         "baseline-corrected, predicting all 239 Addis samples: 0.93x "
         "minus 1.4, held-out quality 0.90. Better than anything we had, "
         "and it is exactly our locked family with a smaller cohort. Two "
         "honesty notes: minus 1.4 is not zero, and under re-selection "
         "this winner is chosen 62.5 percent of the time, not always."),
    notes=("Added per the run-through ('you want to show the crossplot... "
           "that's what we look at'). All-pairs readout at MAC 10, Deming "
           "lambda*. The k=9 is above the rule choice of 5 (the basin "
           "needs k=8-9; rules stop early)."))

slide(
    "Ask each city for its own best and no two agree; cohort size is doing per-site work",
    fig=F("f_out_leaderboard_by_site.png"),
    say=("Now run the same search for each city separately. Each row is a "
         "city's best configuration under the same constraints: Pasadena "
         "wants 120 calibration samples, Addis 440, Bishoftu about a "
         "thousand, Beijing eighteen hundred and fifty. And Delhi's "
         "apparent best is the cautionary tale: it looks lovely and it is "
         "96 percent extrapolation, and it fails fresh holdout filters. "
         "Per-city optimization is doing real work that one global "
         "calibration cannot."),
    notes=("Slope-boxed 0.85-1.18, held-out floor 0.85, score = |intercept| "
           "plus half the slope error. Screening numbers: per-site winners "
           "are candidates, not conclusions."))

slide(
    "Cross-applied, every city's best fails abroad; only the Ethiopian pair transfers",
    fig=F("f_out_cross_application.png"),
    say=("Then take each city's best and apply it to the other four. Rows "
         "are calibrations, columns are prediction targets. The color is "
         "simply how far the fit is from ideal, combining the intercept "
         "size in micrograms per cubic meter with the slope error; darker "
         "is better. Bold is the home fit; an exclamation mark means the "
         "model is extrapolating for most of that city's filters. The "
         "diagonal is dark: every best works at home. The off-diagonal "
         "mostly is not: Delhi's best collapses everywhere else, "
         "Pasadena's tiny cohort is unusable abroad. The one genuine "
         "transfer: Addis's best lands at 0.99x at Bishoftu. The two "
         "Ethiopian sites cohere, which is exactly what you would want a "
         "regional calibration to do."),
    notes=("Metric in words per the run-through: color = |intercept in "
           "ug/m3| + 0.5 x |slope - 1|, capped at 4. Delhi's row inherits "
           "its extrapolated screening artifact (every cell flagged). "
           "Beijing's 1850-sample best is the most tolerable abroad but "
           "still misses Delhi."))

slide(
    "How stable is the winner? Chosen in 62.5% of re-selection draws, not all of them",
    fig=F("f_out_winner_stability.png"),
    say=("The 'add twenty filters and the winner changes' worry, "
         "quantified. Re-run the entire selection under resampling of the "
         "Addis filters, and again under re-draws of the IMPROVE "
         "calibration sites: the winner is re-chosen about 63 percent of "
         "the time both ways, and the corrected-selection spectral-analog "
         "family takes about a third. So lowest-OC/EC 440 is the leading "
         "candidate, not a settled answer, and the runner-up is the "
         "spectral-similarity family; which connects directly to the "
         "question of which selection principle is more defensible."),
    notes=("From the validation layer: target-filter wins 62.5/37.5/0%; "
           "source-site wins 63.0/31.5/5.5%. Source-site slope interval "
           "for the winner is wide: 1.03 [0.65, 1.50]."))

# ------------------------------------------------ 22-25 discussion questions
slide(
    "Question 1: what is good enough?",
    lines=["Is an intercept of -1 µg/m³ acceptable? (deployed today: -4.2)",
           "What slope range counts as usable: 0.9 to 1.1? 0.85 to 1.18?",
           "Is held-out R² of at least 0.85 the right quality floor?",
           "Should the number of calibration samples itself be a constraint",
           "   (no cohorts under ~400, whatever the score says)?"],
    say=("The search gives us thousands of candidates and a ranking, but "
         "the ranking needs a definition of good enough that the group "
         "owns. Concretely: is minus one acceptable for the intercept, "
         "given we are at minus four today? What slope range is usable? "
         "Is the 0.85 quality floor right? And should cohort size be a "
         "constraint on its own?"),
    notes=("Anchor numbers if asked: winner reads 0.93x-1.4 at Addis with "
           "held-out 0.90; deployed reads 1.90x-4.2. The tuned-on-both "
           "two-city compromise sits near -1 at both cities."))

slide(
    "Question 2: how should we validate the choice? (the current gap)",
    lines=["Today, ALL Addis samples are used both to pick the winning",
           "   configuration and to judge it; the win is partly self-graded.",
           "Proposal from Tuesday: split the Addis set into equal halves",
           "   (first year vs second year), pick the calibration on one half,",
           "   predict the blind other half; equal test sizes keep R² comparable.",
           "Also available: a never-screened IMPROVE lot (253, ~5,000 TOR",
           "   filters) as a true independent check."],
    say=("The honest gap in everything I showed: the same Addis samples "
         "pick the winner and grade it. The fix we discussed Tuesday: "
         "split the Addis samples into equal halves by time; select the "
         "calibration using only the first half; predict the untouched "
         "second half blind. Equal test sizes so the R-squared numbers "
         "are comparable. And beyond Addis, IMPROVE lot 253 gives a "
         "five-thousand-filter never-screened set for a true independent "
         "test. Does the group agree this is the standard the choice has "
         "to pass?"),
    notes=("The IMPROVE-side held-out TOR test already exists (that is "
           "the 0.85 floor); the gap is on the TARGET side. Lot-253 "
           "protocol is frozen (winner family, k=5) with lot 255 sealed "
           "for replication; only the scan spectra remain to pull."))

slide(
    "Question 3: one calibration, or per-site calibrations, and on what grounds?",
    lines=["The cross-application matrix: every city's best fails abroad;",
           "   only Addis-to-Bishoftu transfers (0.99x).",
           "A per-site calibration helps exactly where it is hardest to defend:",
           "   among 19,000 configurations there is always a flattering one.",
           "Middle ground: one calibration per REGION or aerosol regime",
           "   (the Ethiopian pair coheres), chosen under pre-agreed constraints?"],
    say=("The transfer matrix says one global calibration does not exist "
         "in the space we searched, and per-city tuning works at home "
         "while failing abroad. But a per-site calibration is also the "
         "easiest thing to fool ourselves with: among nineteen thousand "
         "configurations there is always a flattering one. The Ethiopian "
         "pair cohering suggests a middle ground: regional calibrations, "
         "chosen under constraints we write down in advance. Is that "
         "defensible, and what would the grounds be?"),
    notes=("A tuned-on-both compromise exists: near-1 slopes at Addis AND "
           "Delhi with intercepts near -1; is one compromise calibration "
           "better than two tuned ones? Connects to Q1's 'is -1 "
           "acceptable'."))

slide(
    "Question 4: is spectral similarity more defensible than OC/EC similarity?",
    lines=["Addis OC/EC barely overlaps the IMPROVE pool; selecting on OC/EC",
           "   means selecting the pool's thin extreme tail.",
           "Selecting on baseline-corrected spectral shape selects on what the",
           "   model actually sees; and it is the re-selection runner-up (~1/3).",
           "Do the top performers share a common thread (all low-OC/EC?",
           "   all 400-500 samples?), or are they unrelated configurations?",
           "External constraints not yet used: the co-located sun photometer",
           "   (AERONET) as an independent absorption check."],
    say=("Last one, conceptual, from Ann. Lowest-OC/EC wins our "
         "re-selection test, but Addis's OC/EC barely overlaps IMPROVE at "
         "all, so selecting on it means living in the pool's thin tail. "
         "Selecting on baseline-corrected spectral shape selects on what "
         "the calibration actually sees, and it is the consistent "
         "runner-up. Is it more defensible in principle? Related: do our "
         "top performers share a common thread, or are they unrelated "
         "configurations that happen to score well? And there are "
         "external constraints we have not used, like the co-located sun "
         "photometer, if we want an independent check."),
    notes=("Top-of-leaderboard composition can be answered live in the "
           "app (leaderboard site filter + export). AERONET detail in "
           "backup: the diurnal-corrected column check hardens the Addis "
           "anomaly."))

# ------------------------------------------------------------- backups
slide("Backup", say="Backup material from here.", notes="", title_size=40)

slide(
    "Backup: the six calibration sets on raw spectra; no selection fixes raw",
    fig=F("f11_grid_raw_A.png"),
    say=("The raw-spectra companion grid: every calibration set lands a "
         "materially negative intercept on raw spectra."),
    notes=("Main deck shows one before/after + the corrected grid per the "
           "run-through; this is the full raw grid if someone asks."))

slide(
    "Backup: the HIPS correction is a regression through blanks; heavy filters sit beyond it",
    fig=F("f3_blankline_geometry.png"),
    say=("The instrument-side analysis behind the darkness slide: the "
         "scattering correction is a line fitted through blanks, and 36 "
         "percent of Addis filters are darker than every blank that "
         "defines it."),
    notes=("Blank-count caveat: a lot can carry more than one deployed "
           "calibration line; the pooled n on this figure conflates two "
           "sets within lot 251 (flagged in the run-through). The "
           "per-line audit exists in the app's HIPS tab."))

slide(
    "Backup: an errors-in-both-variables refit with per-filter uncertainties (York) confirms the intercepts",
    fig=F("f1_intercept_blankline_ladder.png"),
    say=("York regression is an errors-in-both-variables fit, like Deming "
         "but with each point's own uncertainty. Refitting every city "
         "with real per-filter HIPS uncertainties, under three versions "
         "of the blank line: the Addis intercept survives everything, "
         "minus 1.3 to minus 1.5 micrograms per cubic meter; Delhi "
         "matches in size but its error bar reaches zero; the rest "
         "bracket zero."),
    notes=("Plain-language definition added per the run-through (group "
           "unfamiliar with York). Instrument share of the Addis "
           "intercept ~15% (blank-line mechanism only)."))

slide(
    "Backup: the same refit for slopes; Pasadena's anomaly was the blank line, Delhi's is real",
    fig=F("f2_slope_blankline_ladder.png"),
    say=("Slopes under the same treatment: Pasadena's wild 3.15x dissolves "
         "to 0.91 under a quadratic blank line; Delhi's 1.8x survives "
         "every variant."),
    notes=("Pasadena's loadings are so light that blank-line shape error "
           "dominates; never an aerosol anomaly."))

slide(
    "Backup: carbonyl band strength vs each city's intercept (µg/m³)",
    fig=F("f_band_vs_intercept_v2.png"),
    say=("The band-versus-offset view, rebuilt on the carbonyl band at "
         "1700 and with the intercept in micrograms per cubic meter: no "
         "clean five-city relationship; the offset is not simply "
         "organic-band-driven."),
    notes=("Replaces the earlier 1617-band panel (its rationale lives in "
           "the working deck, where it was also retired: the band is an "
           "Ethiopian regional marker, decoupled from the offset)."))

slide(
    "Backup: Mie theory grounds the MAC fork; bare BC tops out near 6",
    fig=F("f8_mac_mie.png"),
    say=("Physics context for MAC 6 versus 10: bare soot ensembles top "
         "out near 5.6; reaching 10 requires aged, thickly coated "
         "particles. The fork encodes a mixing-state assumption."),
    notes=("PyMieScatt at 633 nm; enhancement 1.8-2.0 for the coated "
           "branch. IMPROVE implied-MAC bridge centers Addis-like "
           "filters at 10."))

slide(
    "Backup: the sun-photometer cross-check; the Addis anomaly widens after the diurnal correction",
    fig=F("f_aeronet_diurnal.png"),
    say=("The AERONET check mentioned on question 4: correcting for the "
         "photometer's daytime-only sampling with co-located minute "
         "aethalometer data widens the Addis column-versus-surface "
         "anomaly rather than closing it."),
    notes=("Level 1.5 retrievals; Delhi/Beijing lack minute-resolution "
           "MA350 and stay uncorrected. Companion: the filters' organic "
           "envelope predicts column absorption beyond loading and "
           "season (r=+0.28, p~0.002)."))

slide(
    "Backup: SPARTAN has moved to a lot the calibration has never seen",
    fig=F("f_lot253_takeover_v2.png"),
    say=("From early 2025 SPARTAN filters shift to lot 253, outside the "
         "248-plus-251 calibration basis; the same lot has five thousand "
         "IMPROVE TOR filters, which is what makes the independent-lot "
         "validation in question 2 possible."),
    notes=("Rebuilt with black in-plot text per the run-through (old "
           "version had a clipped red annotation). Addis today: 34 x 248, "
           "191 x 251, 14 x 253; Delhi leans newer."))

slide(
    "Backup: the two lab measurements that settle the remainder",
    lines=["Quartz TOR EC on collocated Addis filters: the only measurement that",
           "   is a function of neither axis (campaign one-pager written)",
           "Solvent extraction + HIPS re-measurement on archived Addis and Delhi",
           "   filters, Beijing/Pasadena as controls: separates real absorbers",
           "   from any residual filter artifact"],
    say=("If the discussion reaches 'what would settle this': quartz TOR "
         "on collocated filters, and the extraction re-measurement on "
         "archived filters."),
    notes=("Three lines of evidence (MAC fork, Bishoftu contrast, "
           "offset-vs-curvature degeneracy) terminate at an independent "
           "EC measurement."))

prs.save(HERE / "ftir_group_2026-08-27.pptx")
print(f"saved ftir_group_2026-08-27.pptx with {len(prs.slides._sldIdLst)} slides")

for s_ in prs.slides:
    if s_.has_notes_slide:
        s_.notes_slide.notes_text_frame.text = ""
prs.save(HERE / "ftir_group_2026-08-27_no_notes.pptx")
print("saved ftir_group_2026-08-27_no_notes.pptx (notes stripped)")
