"""Assemble ann_update_2026-08-25.pptx — FTIR group talk 2026-08-27.

House style (docs/deck-style-prompt.md): 16:9 white, claim-as-title ~20pt
top-left ink, one title-free figure centered, SAY/NOTES on every slide.
Run: ~/anaconda3/bin/python build_deck.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
REPO = HERE.parents[1]
INK = RGBColor(0x22, 0x25, 0x2A)
W, H = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
blank = prs.slide_layouts[6]


def add_slide(title, fig=None, lines=None, say="", notes="", title_size=20):
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.22), Inches(12.5),
                              Inches(0.9))
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
        maxw, maxh = 12.2, 6.0
        scale = min(maxw / (iw / 165), maxh / (ih / 165), 1.35)
        w = iw / 165 * scale
        h = ih / 165 * scale
        s.shapes.add_picture(str(fig), Inches((13.333 - w) / 2),
                             Inches(1.15 + (6.1 - h) / 2), Inches(w))
    if lines:
        body = s.shapes.add_textbox(Inches(0.75), Inches(1.6), Inches(11.8),
                                    Inches(5.2))
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


# 1 — title
add_slide(
    "FTIR–EC at Addis: the offset, adjudicated — group update, 27 Aug 2026",
    lines=["Ahmad Jalil — UNBC / UC Davis AQRC",
           "Everything in this deck reproduces from the calibration explorer;",
           "screening numbers are labelled; quoted results are held-out or out-of-country."],
    say=("Since the last update the Addis question went from a one-city mystery to a "
         "five-city adjudication. Roadmap: I'll close out the asks from our Aug 19 "
         "meeting, show what the HIPS calibration itself turned out to be and why the "
         "Addis intercept survives it, what an exhaustive search says about Addis vs "
         "Delhi, the season-baseline interaction, and finish with the two lab "
         "measurements that would settle the rest. If Satoshi joins late: the "
         "cross-site results start at slide 7."),
    notes=("Deck convention: Deming primary (lambda*=2.96 at MAC 10, scaled by MAC^2), "
           "OLS alongside; Option A = site-grouped 5-fold CV, first major minimum, "
           "site-disjoint 80/20 (the only option with a held-out TOR test); Option B = "
           "interleaved 10-fold, within-5%; B2 = interleaved, first major minimum. "
           "Units ug/m3, Mm-1 parenthetical (x10 at MAC 10)."))

# 2 — status
add_slide(
    "Every ask from the Aug 19 meeting is closed or in motion",
    fig=FIG / "f_status.png",
    say=("Quick score-card on what you asked for. Select-and-calibrate on corrected "
         "spectra: done, and it found something better than our locked 800. Lot-251-only "
         "evaluation: built into the app. The lot-248 mystery: solved — it was a "
         "two-month lot. Bishoftu spectra: I pulled them myself from the SPARTAN "
         "database like you suggested, and grabbed Beijing, Delhi and Pasadena while I "
         "was in there. And your big question — lot or aerosol — has an answer now."),
    notes=("ETBI pull route: Networks_1_0 over SqlClient on the VPN machine (Sean's "
           "config); the same pipeline builds any SPARTAN site as an explorer target. "
           "Adama slides: in progress this week, figure already drafted (slide 13)."))

# 3 — provenance
add_slide(
    "Before anything new: the locked numbers reproduce exactly",
    fig=FIG / "f1_ladder.png",
    say=("Same anchor check as always: the six setups' Addis intercepts under all "
         "three protocols, Deming lambda-star, fixed 190. Option B reproduces the "
         "historical slide numbers, and the locked anchors reproduce to the third "
         "decimal — lowest-OC/EC Option A k=6, OLS 1.585x minus 3.221, held-out "
         "0.911; plus AIRSpec k=5, 0.86x minus 1.615, Deming 0.95x minus 2.09. "
         "Two readings to carry forward: protocol choice moves things far less than "
         "the setup does — the comforting result from last time — and the "
         "AIRSpec row sits nearest zero under every protocol."),
    notes=("Validated live against the explorer before this deck was built. Standing "
           "caveats carried: analog 'top 500' resolves to 477 after TOR eligibility; "
           "entire-network Option A picks k=15 vs the committed pool run's k=10 "
           "(unexplained — don't quote that cell); Eth-shaped+AIRSpec under B2 (k=21) "
           "is an unstable cell."))

# 3b — grid block, raw, Option A
add_slide(
    "Raw spectra fail the same way across every cohort — no selection fixes it",
    fig=FIG / "f11_grid_raw_A.png",
    say=("The six-panel view, raw spectra, Option A. Filled points and solid line "
         "are MAC 10, open and dashed are MAC 6, the black diamond is the shared "
         "Deming intercept — which is MAC-invariant, so the diamond is the number "
         "to watch. Every cohort lands a materially negative intercept on raw "
         "spectra; lowest-OC/EC raw sits at minus 4.34 despite the best held-out "
         "TOR R-squared. Selection alone does not rescue raw spectra."),
    notes=("Grid grammar: dotted grey 1:1; stat box = Deming intercept (both MACs), "
           "Deming slopes @10/@6, OLS intercept, k, held-out TOR R2 (Option A only). "
           "Per-panel square-ish limits — never shared clipped axes. Fixed 190 "
           "readout."))

add_slide(
    "Baseline-corrected, every cohort tightens — and lowest-OC/EC + AIRSpec leads",
    fig=FIG / "f12_grid_airspec_A.png",
    say=("Same six panels, baseline-corrected. Every diamond moves toward zero, and "
         "lowest-OC/EC plus AIRSpec is the standout: Deming minus 2.09, slope 0.95 "
         "at MAC 10, held-out TOR R-squared 0.90. That pairing — composition "
         "selection plus baseline correction — is the story the rest of the deck "
           "keeps confirming: baselining is the lever, and it needs the right cohort "
         "under it."),
    notes=("The shape cohorts (Ethiopia-shaped, analogs) improve cosmetically but "
           "keep weak held-out R2 (0.24 / 0.59) — shape selection does not find "
           "composition (ftir_33). 'both MACs' on the intercept: the Deming "
           "intercept is exactly MAC-invariant under lambda scaled with MAC^2 "
           "(ftir_31)."))

# 4 — mechanism
add_slide(
    "The HIPS scattering correction is a regression through blanks — heavily loaded filters sit beyond its support",
    fig=FIG / "f3_blankline_geometry.png",
    say=("We decoded the HIPS math from the batch export: tau is ln of (intercept plus "
         "slope times R1) over T1, and that line is fitted through field blanks. Top "
         "panel: the lot-251 blank cloud with the deployed linear line and a quadratic "
         "refit. Bottom: where each site's filters sit. The shaded zone is darker than "
         "every blank — 36% of Addis filters live there. Their Fabs stands on an "
         "extrapolated calibration, so we tested what that's worth."),
    notes=("Formula verified to <1e-4 against 3,859 shipped tau values. Blank-tau zero "
           "bound: +0.32 +/- 0.46 Mm-1 at Addis geometry (575 blanks) — the PTFE-zero "
           "branch is dead. Lot 253's blanks are ~10x tighter (rms 3.8 vs 31-42 counts) "
           "than the big lots — a HIPS-precision finding on its own. A lot can carry "
           "more than one deployed line; the per-line audit is in hips_lab."))

# 5 — intercept survives
add_slide(
    "The Addis intercept survives the instrument check — the blank-line share is ~15%",
    fig=FIG / "f1_intercept_blankline_ladder.png",
    say=("Every filter's Fabs recomputed under three blank-line variants, then refit "
         "with per-filter weighted York regression — that's errors-in-variables with "
         "the real HIPS uncertainties, replacing pooled-lambda Deming. Addis moves "
         "from minus 1.51 to minus 1.27 and stays seven sigma from zero. Delhi's "
         "point estimate matches Addis but its error bar reaches zero. Beijing, "
         "Bishoftu, Pasadena bracket zero. One certain offset: Addis."),
    notes=("York et al. 2004; sigma_y inflated to MSWD=1 (absorbs lack-of-fit — "
           "conservative). Scope caveat (state if asked): blanks only sample tau~0, so "
           "high-loading response nonlinearity is unprobed by this test and still "
           "degenerate with real absorption on FTIR-EC axes — that's what quartz TOR "
           "terminates. In Mm-1: Addis offset ~13-15."))

# 6 — slopes
add_slide(
    "Pasadena's slope anomaly was the blank line; Delhi's 1.8x is real",
    fig=FIG / "f2_slope_blankline_ladder.png",
    say=("Same treatment, slopes. Pasadena read 3.15x with the deployed line — switch "
         "to a quadratic blank line and it's 0.91x. At their loadings the blank-line "
         "shape error dominates tau, so Pasadena was never an aerosol anomaly. "
         "Beijing's old 1.48x was unweighted leverage; properly weighted it's 0.98. "
         "Delhi's 1.8x barely moves under any variant — that one is real and still "
         "open: genuinely low site MAC, or the model reading oxidized organics as EC."),
    notes=("Delhi correlate: within-site r(residual, carbonyl/m3) ~0.9 raw, collapsing "
           "when detrended by the site's own line — carbonyl marks WHICH sites "
           "over-read, not which filters scatter. The MA350 880-nm channel as an "
           "independent per-filter anchor at all four sites is the queued splitter "
           "(ftir_28 has only ever run at Addis)."))

# 7 — lot census
add_slide(
    "Lot 248 isn't a partial download — it's a two-month lot",
    fig=FIG / "f_lot_census_improve.png",
    say=("The pool mystery from our meeting. The database really does hold only about "
         "1,400 lot-248 FTIR analyses network-wide — the lot was in service for "
         "roughly two months around January 2021. So lot-248-trained cohorts are "
         "winter-2021 cohorts; that's a season confound, not missing data. Your guess "
         "in the meeting — 'somebody decided it wasn't any good and moved on' — looks "
         "right."),
    notes=("IMPROVE ftir_catalog: lot 248 = 1,362 analyses, 2020-12-17 to 2021-02-15; "
           "Ahmad's 1,299 download was essentially the whole thing. Lot 251 = 11,843 "
           "over a year. Bar labels are analysis counts; red = 248, blue = 251, "
           "amber = 253. Lots 253/255 being the recent large lots is what makes the "
           "independent-lot validation possible (backup slide)."))

# 8 — lot effect
add_slide(
    "Baseline correction removes the lot-to-lot difference — the mechanism you predicted",
    fig=FIG / "f_lot_baseline_removes_lot_effect.png",
    say=("You suggested baselining may help partly by erasing 248-versus-251 media "
         "differences. Beijing used both lots in one 178-day window, so we can test "
         "it: on raw spectra the two lots disagree by about 4.5 inverse megameters on "
         "a common line; baseline-corrected, that collapses to about one, "
         "indistinguishable from zero. So yes — part of what AIRSpec buys is lot "
         "insensitivity. But Bishoftu shares Addis's lots and has no offset, so the "
         "lot is not the Addis story."),
    notes=("Numbers (Beijing, both lots in use 2022-10-13..2023-04-09, n=14 vs 34, "
           "4,000 bootstrap draws, Option A): ocec-800 raw -4.49 [-7.01,-1.96] "
           "significant vs airspec -0.86 [-3.66,+2.02] n.s.; ocec-450 raw -4.35 vs "
           "airspec -1.43 n.s. Caveats: n small; lot-248 filters slightly less loaded "
           "than the 251 comparators (median pred EC 1.18 vs 1.66). Does NOT explain "
           "Addis: ETBI is 100% lot 251 and ETAD ~80%, so Addis-vs-Bishoftu is a "
           "within-lot-251 contrast, and the residual post-baseline lot term (~1 Mm-1) "
           "is ~5% of the ~21.5 Mm-1 offset."))

# 9 — grid
add_slide(
    "An exhaustive search: no cohort, baseline, or k reconciles Addis and Delhi",
    fig=FIG / "f4_screening_cloud.png",
    say=("To your 'million crossplots' worry — this is what keeping track looks like. "
         "Every variant we've identified — five cohorts, cutoffs 100 to 2000 in steps "
         "of ten, three spectral spaces, swept components — evaluated at all five "
         "sites: nine and a half thousand jobs. Grey fails the held-out floor, blue "
         "passes, the star is the winner. The punchline for Delhi: exactly one "
         "configuration lands near-1:1 at both cities, it still carries an offset at "
         "Addis, and it fails fresh holdout filters. The Addis-Delhi difference is "
         "not a knob we can turn."),
    notes=("71,263 scored rows in batch_results.jsonl, dedup-keyed, resumable. "
           "Methods lesson learned at grid scale: |intercept|+w|slope-1| is gamed by "
           "flat slopes (0.44x with ~0 intercept 'wins'); rankings are slope-boxed "
           "0.85-1.18. Screening numbers are descriptive only — quoted results come "
           "from held-out / out-of-country sets."))

# 10 — basin
add_slide(
    "Lowest-OC/EC 440–490 × AIRSpec beats the locked 800 — and wins 62.5% of bootstraps",
    fig=FIG / "f_cutoff_basin.png",
    say=("The corrected-selection run you asked for, taken dense. Intercept on top, "
         "held-out TOR R-squared below, as the cohort cutoff sweeps. The shaded basin "
         "at 440 to 490 is where everything is best simultaneously — held-out 0.92 "
         "versus 0.87 at our locked 800. The honest footnote: under bootstrap "
         "reselection this winner takes 62.5% of draws, so it's the leading frozen "
         "candidate, not a final answer."),
    notes=("ftir_11's '800 sweet spot' was a coarse-grid artifact (never scanned below "
           "600). Basin needs k=8-9 — rule-k at 5 ties the 800 (connects to ftir_23's "
           "'rules stop early'). Corrected analog-440 wins ~1/3 of draws — the "
           "runner-up family. Source-site slope interval remains wide: 1.03 "
           "[0.65, 1.50]."))

# 11 — seasons
add_slide(
    "Baselining relocates seasonality: the dry season becomes the anomaly",
    fig=FIG / "f_season_panels.png",
    say=("You flagged season-stability as hugely important, so we stratified the "
         "winner by season under both baselines. Left, raw: the whole-year 2.35x "
         "hides wet seasons at 2.2x and Dry at 1.3x — raw's failure is a wet-season "
         "phenomenon. Right, corrected: the wet seasons snap to about 0.85x with "
         "small intercepts, but Dry drops to 0.60x. Baselining doesn't remove the "
         "seasonal structure — it moves it, and the charcoal-heavy dry season becomes "
         "the anomalous regime. That's the reverse of the direction we'd guessed."),
    notes=("Per-season Deming at lambda*=2.96 on restricted ranges — indicative, not "
           "final; the proper interaction fit is the ftir_24 deliverable. Consistent "
           "with char_06's dry-season spectral anomaly and ftir_22's dry-separates "
           "residuals. Season schemes differ across sites (Ethiopian vs quarters) — "
           "cross-site seasonal pooling needs a mapping decision."))

# 12 — band lesson
add_slide(
    "Three baselines gave three answers about the 1617 band — a methods rule, not a mechanism",
    fig=REPO / "research/ftir_ec_phase3/output/plots/cross_site_spectra_2026-08-23.png",
    say=("A cautionary result we're glad we caught ourselves. The 1617 band looked "
         "like it tracked the offset across five sites; on corrected spectra the "
         "match was Delhi's carbonyl flank; under a neutral baseline the band shows "
         "at Addis AND Bishoftu — an Ethiopian regional marker, decoupled from the "
         "offset, because Bishoftu has the band and no offset. The mechanism: "
         "APRLssb anchors its spline at 1520 to 1600 — directly under the band. "
         "House rule going forward: any claim in 1500-1650 needs two baselines."),
    notes=("Full arc with both retractions preserved in BAND1617_LEAD_2026-08-23.md. "
           "This is presentable credibility: tested, corrected in-place, rule "
           "extracted. The composition evidence for the offset now rests on the "
           "envelope-AERONET link and BC/PM2.5=23% (3x any other site), not the band."))

# 13 — adama
add_slide(
    "Adama is not Addis: Batch-54 sits at the IMPROVE median, with no HIPS–EC gap",
    fig=FIG / "f_adama_context.png",
    say=("Seed of the two-pager for Christian and Sina. The five Adama TOR filters "
         "sit at OC-to-EC around six — the IMPROVE pool median, nowhere near the "
         "low-OC/EC tail that Addis calibrations need — and Bishoftu, eighty "
         "kilometres away, absorbs half of what Addis does. More Adama samples "
         "characterize a different problem than the Addis one."),
    notes=("This figure is the SEED, not the full case — Ann's ask is a 2-3 slide "
           "HIPS + TOR-OC/EC vs FTIR-OC/EC comparison telling Christian & Sina that "
           "more Adama samples won't arbitrate Addis; the TOR-vs-FTIR panel still "
           "needs to be added (data: ftir16 adama_batch54 tables). Figure provenance: "
           "ftir31 figure_manifest. The decisive Adama-side contribution would be "
           "quartz filters for TOR."))

# 14 — app
add_slide(
    "Every variant in this deck is a click — live demo",
    lines=["Sites tab — the five-site table with the extrapolation flag, one click",
           "HIPS tab — York fits under all three blank lines, plus the blank ledger",
           "Optimize — multi-site batches, dense sweeps, slope-boxed leaderboards",
           "New since last time: config sidebar, validation guardrails (Q-residual + score distance) on every run"],
    say=("Rather than more slides — the app. [Switch to the explorer: run the Sites "
         "tab on the winner, open the HIPS tab, show the blank-line ladder live. If "
         "Satoshi asks a variant, run it now.]"),
    notes=("Explorer on :5058 under anaconda python; warm cache first. If the live "
           "demo dies: every figure here regenerates from build_figures.py in the "
           "deliverable folder."))

# 15 — asks
add_slide(
    "Three lines of evidence now terminate at the same two measurements",
    lines=["Quartz TOR EC on collocated Addis filters — breaks the offset-vs-curvature degeneracy",
           "   (campaign one-pager written: ~36 filters across 3 seasons)",
           "Solvent extraction + HIPS re-measurement on archived Addis + Delhi filters,",
           "   Beijing/Pasadena as controls — separates real absorbers from any residual artifact",
           "Both use existing samples or the existing sampling plan; neither needs new field work"],
    say=("The statistics have gone as far as statistics go: the MAC fork, the "
         "Bishoftu contrast, and the curvature degeneracy all end at an independent "
         "EC measurement. Quartz TOR is the decisive one; the extraction test is the "
         "cheap companion that uses filters already in the archive. These are the "
         "two asks."),
    notes=("Extraction design: Kirchstetter/Novakov/Hobbs JGR 2004 — water then "
           "methanol; a drop only at Addis/Delhi = real organic absorbers; "
           "methanol-resistant dark residue = char/tar specifically. TOR sizing from "
           "ftir_16: ~3 sigma/day separation of MAC 6 vs 10 -> 11-13 days/season x 3 "
           "seasons."))

# 16 — logistics
add_slide(
    "Logistics",
    lines=["AAAR: poster Thursday 1–3 pm, session 9 — flights + conference hotel booked",
           "Committee meeting: Sep 2, 3 pm — invite emails [FILL IN: confirm they arrived this time]",
           "Adama two-pager for Christian & Sina: draft this week",
           "Satoshi meeting: ready when scheduled — the app makes his variants runnable live",
           "[FILL IN: lot-253 spectra pull status — needs a session on the Windows/VPN machine]"],
    say=("Housekeeping: AAAR is booked, poster Thursday one to three. Committee "
         "meeting September 2 at three. The Adama two-pager lands this week. And "
         "whenever the Satoshi meeting happens, the tool is ready for it."),
    notes=("The lot-253 pull script (get_improve_lot253_spectra.ps1) is staged: "
           "pilot mode first, resumable parts, lot 255 sealed as the replication "
           "set. TOR y-values for lot 253 are already local — only spectra missing."))

# 17 — backup divider
add_slide("Backup", say="Backup material from here.", notes="", title_size=40)

# 17b — backup grids, Option B
add_slide(
    "Backup — the same grids under Option B reproduce the historical slide numbers",
    fig=FIG / "f9_grid_raw.png",
    say=("For reference against the network's own protocol: raw spectra under "
         "Option B — interleaved ten-fold, fitted on all filters. These reproduce "
         "the historical deck intercepts."),
    notes=("Option B has no held-out TOR test (fits on all filters) — its "
           "leaderboard passes are vacuous; shown for continuity with the "
           "network's numbers, never for ranking."))

add_slide(
    "Backup — Option B, baseline-corrected: the AIRSpec conclusion is protocol-robust",
    fig=FIG / "f10_grid_airspec.png",
    say=("And baseline-corrected under Option B: lowest-OC/EC plus AIRSpec lands "
         "within a tenth of the Option A answer — the protocol-robustness result "
         "Ann called the good news."),
    notes=("A-vs-B spread on the AIRSpec row is ~0.1 ug/m3 (-1.65 vs -1.62 OLS, "
           "ftir_21) — the strongest robustness claim in the matrix; contrast "
           "smoke-906, which swings 2.43x-6.35 to 0.50x-0.99 across protocols."))

# 18 — aeronet diurnal
add_slide(
    "Backup — correcting for AERONET's daytime-only sampling widens the Addis gap, not closes it",
    fig=FIG / "f_aeronet_diurnal.png",
    say=("The day-night confound we flagged is now measured with the co-located "
         "MA350 minute data: correcting for AERONET's daytime-only sampling leaves "
         "Addis at ~190 metres of effective absorption scale height while Pasadena "
         "moves up — the outlier gets more anomalous, not less."),
    notes=("The photometer sees ~8 of 24 hours. Addis has a 4.8x diurnal swing in "
           "Red BCc (625 nm ~ HIPS wavelength) but its two retrieval windows straddle "
           "the cycle, so the biases cancel (ratio 0.92-1.05); Pasadena's midday "
           "window sits in its own BC minimum (1.07-1.09) — correction moves the "
           "Addis/Pasadena separation from 2.9x to 3.0-3.3x. Assumes the column "
           "tracks the surface's relative diurnal shape; Delhi/Beijing have no "
           "co-located MA350 and stay uncorrected. Companion result: the corrected "
           "1500-1700 organic envelope predicts column AAOD675 beyond loading, Fabs "
           "and month (r=+0.28, p~0.002)."))

# 19 — lot 253 takeover
add_slide(
    "Backup — SPARTAN has moved to a lot the calibration has never seen",
    fig=FIG / "f_lot253_takeover.png",
    say=("Forward-looking risk and opportunity: from early 2025 the network's filters "
         "shift to lot 253 — a lot our calibration basis has never seen. That's the "
         "risk. The opportunity: IMPROVE ran 5,050 TOR-EC filters on that same lot, "
         "so a frozen protocol tested there is a true independent-lot validation — "
         "protocol frozen, lot 255 sealed for replication; only the scan spectra "
         "remain to pull."),
    notes=("433 lot-253 SPARTAN filters (stackplot uses the 340 dated), 13 sites, "
           "58% of everything sampled since 2025-09; basis is IMPROVE 248+251 only. "
           "IMPROVE holds 5,050 lot-253 TOR-EC filters; pull manifest at "
           "output/tables/improve_pull/ftir_list_253.csv; lot 255 sealed for "
           "replication. Frozen protocol: OCEC-440 x AIRSpec, Option A, k=5 — tests "
           "lot+time transfer, not unseen-site. The direct Delhi lot-253 test is NOT "
           "identifiable (6-filter overlap, 4x loading, CI [-53,+304])."))

# 19b — column check backup
add_slide(
    "Backup — Addis reports more absorption than its own column can account for",
    fig=FIG / "f_aeronet_column_check.png",
    say=("The column-side view: effective absorption scale height — column AAOD "
         "over surface Fabs — is 232 metres at Addis against 544 to 894 everywhere "
         "else, and the filters' organic envelope predicts the photometer's column "
         "absorption beyond loading and season."),
    notes=("H is NOT a physical boundary-layer height — comparative across sites "
           "only. Sharpest contrast is Delhi: 1.6x Addis's surface Fabs but 4.3x "
           "the column. Level 1.5 retrievals (not QA'd 2.0), formally reliable only "
           "above AOD440~0.4; Gurgaon is ~30 km from the Delhi SPARTAN site."))

# 20 — local v1
add_slide(
    "Backup — per-filter LOCAL calibration: helps where the library has support",
    lines=["Per target filter: k most-similar library spectra -> small PLS (novel in FTIR-aerosol lit)",
           "Delhi 1.37x (from 2.36x) and Pasadena 1.04x (from 4.22x) — real improvement",
           "Addis 4.5x — collapses: nearest neighbors are the library's edge, local ≠ in-domain",
           "v2: score-space similarity + refuse-to-predict gates; then CORAL / di-PLS"],
    say=("First pass at per-filter local calibration: strong where the library has "
         "support, honest failure where it doesn't — which is itself the result: no "
         "selection scheme manufactures Addis support that isn't there."),
    notes=("Module: calibration_explorer/local_lab.py (/api/local_cross). Survey with "
           "citations: TRANSFER_METHODS_2026-08-24.md. The decisive evaluation to "
           "run: simulated transfer at IMPROVE sites with collocated HIPS — pretend "
           "Fabs-only, run the whole pipeline, unblind TOR once."))

prs.save(HERE / "ann_update_2026-08-25.pptx")
ns = len(prs.slides.__iter__.__self__._sldIdLst)
print(f"saved ann_update_2026-08-25.pptx with {len(prs.slides._sldIdLst)} slides")
