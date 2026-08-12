# Task spec — build the Ann briefing deck for today (2026-08-12)

Self-contained instructions for building `deliverables/ann_briefing_2026-08-12.html`, the
deck for today's 1:1 with Ann (the day before the Satoshi meeting, 2026-08-13 10:00).
Everything needed is on branch `claude/data-analysis-u9r3h7` or stated here.

## Purpose and audience

Working 1:1, ~30 minutes, one attendee (Ann) who has seen the July 31 deck walkthrough
and knows the project cold. This is NOT a re-presentation of phase 3 — it is a
decision meeting. Every slide must either (a) report something that changed since she
last saw the work, or (b) put a decision or question in front of her. Target
**8 slides**, hard cap 9.

## Format and placement

- Single self-contained HTML, figures base64-embedded, in
  `research/ftir_ec_phase3/deliverables/` (NOT `output/` — `research/**/output/` is
  gitignored and a `!`-negation silently fails because the parent dir is excluded;
  this is why `deliverables/` exists). Track it in git.
- House style: copy the inline `<style>` block and slide anatomy verbatim from
  `deliverables/satoshi_deck_2026-08-13.html` (kicker → h1 → scorecard → parts →
  slides with `.finding` / `.plate` / `.says` / `.chips` → `.close` with
  `.open-card`s → footer). Figures available via
  `deliverables/extract_deck_figures.py` output or the paths below.
- Write a spoken-notes companion `deck_notes_ann_2026-08-12.md` (the say-out-loud
  format of `deck_notes_airspec.md`) — one short paragraph per slide plus
  "if she pushes" caveats. Ahmad presents from these.
- Before calling it done: run the same claims audit that caught the random-cohort
  error. Every number on a slide must trace to a committed table/notebook or be
  explicitly badged `preliminary` (see provenance table below). No exceptions.

## Slide plan

**Header** — kicker: `FTIR · EC · Addis Ababa · Ann 1:1 · 12 Aug 2026`. H1 along the
lines of "What changed overnight, and what needs your call." Meta chips: Satoshi
meeting tomorrow 10:00; deck awaiting her approval; two results, one negative result,
three decisions.

**Scorecard** — left column "Since you saw it": audit corrections applied; intercept
restated as ~21 Mm⁻¹ invariant (committed); FTIR side exonerated vs MA350
(preliminary); MA350 BrC route tested and closed; ftir_24 feasible today. Right
column "Needs you today": approve Satoshi deck; HIPS wavelength 405 vs 633 (OPEN —
her lab runs HIPS); meaning of `HIPS_Uncertainty`/`HIPS_MDL` fields; ftir_24
go/no-go; ask-assignments for tomorrow; Hossein outcome for the co-author email.

**Slide 01 — Corrections caught before Satoshi.** The audit found the slide-02
random-cohort claim wrong on all three counts (5 cohorts not ten; range 3.10–5.00
not 4.4–7.3; random #3 at 3.097 beats OCEC-800's 3.415) AND structurally invalid
(each cohort scored on its own disjoint-site split — not comparable). Replacement
evidence, like-for-like and committed: same 190 Addis filters RMSE **1.16 vs
1.48–2.24**; held-out TOR R² **0.911 vs 0.594–0.775**. Also: VIP ρ = 0.74/0.12
flagged uncorroborated (no executed notebook computes it; removed from the email
draft); −1.61 → −1.62 propagated; ETBI contrast now vs Addis *Dry* median 43.2 (not
all-season 47.1); "~91% baseline" qualified as one representative filter. Framing in
notes: this is the check working, not damage — fixed and pushed (53c332e).
No figure; chips carry the numbers.

**Slide 02 — Satoshi deck: what she's approving.** One-slide map of the corrected
deck's five parts + the six asks, with the two tone-sensitive items called out for
her sign-off: the component-selection slide (critiques the group's Calibration app;
framing is "different question, not wrong tool") and how hard to push MAC ≈ 10
("lands closest", NOT "self-consistent" — 0.86 fails ftir_19's |slope−1| ≤ 0.1 bar).
Figure: `deliverables/` copy of `calibration_setup_matrix` (site_held_out) as the
visual anchor, or no figure and a compact HTML list.

**Slide 03 — Result: the intercept is a ~21 Mm⁻¹ absorption target (committed,
ftir_25).** C = |intercept|·MAC/slope is MAC-invariant by construction and spans
only 18.8–26.1 Mm⁻¹ (median 21.5 ≈ 46% of median Addis Fabs 47.11) across all six
setups while slopes span 3.4×. State the caveat before she does: C is invariant to
any multiplicative rescaling of the EC scale, so cross-setup stability is partly
structural — the claim is "the additive offset is robust to every scaling choice",
not "six independent routes agree". Render the six-row table from
`ftir_25_intercept_invariant.md` as an HTML table in the plate (no PNG exists).
This slide answers her July question "why is the intercept negative when no data is".

**Slide 04 — Result (preliminary): the FTIR side is exonerated.** EC_ftir ~ MA350
BC(880): intercept **+0.285, CI [−0.022, 0.593]** — includes zero — R² **0.870**,
better than the HIPS comparison (0.743). By the attack plan's pre-registered rule,
the additive offset localizes to the HIPS axis. Note the surfaced ~2.2× absolute
scale gap (HIPS Fabs mean 49.7 vs MA350 b_ATN(625) mean 111.6) is mostly the
multiple-scattering C-factor, i.e. expected — the *additive* result is the news.
Badge: `preliminary — agent-reproduced, commit pending`. If the fit lands as a
committed script before build time, drop the badge and cite it.

**Slide 05 — Negative result: the MA350 cannot measure the BrC share.** Tested, not
speculated: AAE(625,880) = 0.944 ± 0.060 (no red excess; implied Babs_BrC −2.06
Mm⁻¹, negative on 84.5% of days vs the +21.7 needed; closing requires AAE_BC = 0.32,
unphysical). Instrument health: Green channel gives unphysical negative AAE, UV
clips on 35% of days, Red sits on IR within channel reproducibility; only IR is
trustworthy. Phrase the conclusion exactly: **"the MA350 cannot answer this
question"** — NOT "there is no BrC at Addis". Mention the two repo traps fixed en
route (AE33-vs-MA350 wavelengths in the README; BCc-AAE offset identity ≈ −1.0,
which had 47% biomass reading as 12%). Badge preliminary as slide 04.

**Slide 06 — The question only she can answer today: what wavelength is HIPS?**
`docs/filter-optics-reference.md` marks it OPEN ("do not quote without checking with
SPARTAN"); `RESEARCH_PROGRESS.md:112` says 405 nm; phase-3 prose has assumed ~633.
Why it matters: at 405 nm BrC and dust absorb several-fold more than at 633, making
a ~40% non-EC share of Fabs *more* plausible, and any red-channel reasoning moot.
Second question: at ETAD, `HIPS_Uncertainty` (190 filled) and `HIPS_MDL` exist as
their own parameter rows — what are their semantics? If usable as σ_x, the Deming
bound (Tier-1 item 3) runs with real uncertainties and `docs/open-items.md`'s
"no uncertainties" item is wrong. Badge the field-existence claim `verification in
progress`. No figure; this is a question slide.

**Slide 07 — ftir_24: your hypothesis, tested against committed numbers.** Ann's
"dry is right, wet breaks" is inverted for the raw model: per-season residuals Dry
**−1.25** (worst), Belg −0.47, Kiremt **+0.49** (nearest zero); corrected-model
residuals are season-stable (−2.0…−2.6), so season sensitivity is a raw-model
phenomenon. Confound to design around: Dry spans ~55% of the wet x-range, so naive
per-season slopes attenuate mechanically — the notebook needs a pooled fit with a
season×x interaction to distinguish "different line" from "shorter segment".
Feasibility: buildable today (~1 GB pull, ~5 min regen); the 43-undated worry is
retired (all 43 outside the 239-filter eval set; split Dry 105/Belg 61/Kiremt 73;
`dry_feb` convention, belg_feb as sensitivity row). Decision: build today or after
tomorrow. Figure: `output/plots/ftir17/addis_spectra_by_season.png` (or its
deliverables extraction) as backdrop.

**Slide 08 / close — Tomorrow's plan + decisions.** Revised next-steps for the
Satoshi meeting: Tier 1 (IMPROVE Fabs-at-EC=0 intercept — agent running; free-offset
bootstrap CI; Deming with real σ_x if slide 06 unblocks it), HIPS-side scale/
wavelength investigation, quartz-TOR campaign one-pager (spec: 11–13 days/season,
~36 filters TOTAL, quartz only). Tier 2 reported as closed. Then the decision
checklist as `.open-card`s: ① approve Satoshi deck (with her edits); ② HIPS
wavelength + uncertainty semantics; ③ ftir_24 today?; ④ ask-owners for tomorrow's
six asks; ⑤ quartz one-pager go-ahead; ⑥ Hossein outcome → co-author email.
`.decisive` box: the standing rule — nothing reaches Satoshi or co-authors before
her approval, which is what today is for.

## Number provenance (audit against this before shipping)

| Claim | Source | Status |
|---|---|---|
| random-cohort correction (5, 3.10–5.00, #3 = 3.097 vs 3.415) | run_ftir_11.py / ftir_11 committed tables | committed (53c332e) |
| RMSE 1.16 vs 1.48–2.24; TOR R² 0.911 vs 0.594–0.775 | ftir_11/21 committed tables | committed |
| C invariant 18.8–26.1, median 21.5, 46% of 47.11 | ftir_25_intercept_invariant.md + script | committed (fc93862) |
| EC_ftir~BC880: +0.285 [−0.022, 0.593], R² 0.870 vs 0.743; means 49.7/111.6 | tier-2 agent run | **preliminary — badge or commit first** |
| AAE(625,880) 0.944 ± 0.060; −2.06 Mm⁻¹; 84.5%; AAE_BC 0.316; channel health | tier-2 agent run | **preliminary — badge or commit first** |
| BCc-AAE offset −1.0157/−0.9674; 47%→12% | optics.py docstring (3aedfdc) | committed |
| per-season raw residuals −1.25/−0.47/+0.49; corrected −2.0…−2.6 | ftir_15 committed tables | committed |
| eval-set split 239 = 105/61/73; 43 undated all outside | ftir_24 feasibility agent | preliminary (verified vs ftir_15 table) |
| HIPS_Uncertainty/HIPS_MDL populated at ETAD (190) | unified_filter_dataset.pkl probe | **verification in progress** |
| HIPS wavelength 405 vs 633 | filter-optics-reference.md (OPEN) / RESEARCH_PROGRESS.md:112 | genuinely open — that's the slide |

## Do-nots

- Do NOT reuse the old random-cohort chip or the "ten cohorts / 4.4–7.3" numbers anywhere.
- Do NOT state a HIPS wavelength as fact; the slide asks, it doesn't assert.
- Do NOT phrase tier 2 as "no BrC at Addis" — the instrument can't see it either way.
- Do NOT quote VIP ρ = 0.74/0.12 (uncorroborated), MAC 10.05 without "Addis-like
  6,503-filter subset", "~36 filters per season" (it's the total), or
  "self-consistent at MAC 10" (say "lands closest").
- Do NOT block the build on the still-running Tier-1 agents; if item 1's IMPROVE
  intercept lands before the meeting, add one chip to slide 08, else say "running".
- Deck goes to Ann only — it is itself subject to her approval rule for anything
  reused tomorrow.
