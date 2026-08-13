# Ann 1:1 (12 Aug) — every consistency issue she raised, assessed and addressed

Source: today's meeting transcript. For each item: what she flagged, the root cause,
whether it was a real error, and the action taken (or queued for tonight).
Tomorrow's 10:00 now includes **Naveed** (Ann invited him — he had the SPARTAN EC
work before the handoff).

## 1. Units flip mid-deck (Mm⁻¹ vs µg/m³) — real presentation defect
She caught intercepts quoted in Mm⁻¹ on the IMPROVE-origin slide against µg/m³
everywhere else ("it's one instead of six, but it's actually 0.1"). **Fix applied**:
intercept-scale claims now carry µg/m³-equivalents wherever they appear
(pooled IMPROVE bound 1.3 Mm⁻¹ = **0.13 µg/m³** at MAC 10, vs an Addis offset of
1.6–4.2 µg/m³). Standing rule for all future slides: y-intercepts in µg/m³, with
Mm⁻¹ only as a parenthetical.

## 2. "−0.99 vs −6.91" biomass-smoke intercept — NOT an error, a protocol label gap
The new matrix says smoke-906 = −0.99; the old deck said −6.91. Both are committed
and both are right: **−0.99 is the site-held-out protocol (k = 4)**, −6.91 the
app-protocol fixed-cohort value (k = 19). This is literally ftir_21's headline
("the protocol moves the Addis answer more than the MAC fork does") surfacing as an
apparent inconsistency because the slide didn't carry its protocol label loudly
enough. **Fix applied**: every table/crossplot now states protocol + units in the
header, and a new comparison slide (her request) shows all six setups under both
protocols side by side (`intercept_slope_by_mode`). Do not re-derive the intercept
by hand — the number was never wrong.

## 3. "K says four but the curve bottoms near six" — known subtlety, wrong figure shown
The displayed CV curves (ftir_20 overlay) are computed on **full cohorts**; the
locked k was selected on the **training split**. The repo's own caveat notes say the
two need not agree — the annotation was right, the figure invited the wrong reading.
**Fix applied**: the how-was-k-chosen slide now uses ftir_23's
`selection_curves_site_heldout.png`, which draws the rule's machinery and the chosen
k on the curve the rule actually ran on; the notes carry the full-cohort-vs-training
explanation. (Sanity check from committed tables: the smoke full-cohort site-grouped
raw curve's global min is at k = 26 with a huge ±SE — consistent with the documented
"long flat tail" behaviour, not with a mis-selection.)

## 4. "Use one k-selection method consistently" — already true, now visible
Confirmed: every headline number uses site-grouped 5-fold CV + first-major-minimum.
The confusion came from the ftir_20 comparison overlay showing both schemes at once.
**Fix applied**: headline slides show one protocol; the both-protocols slide is the
explicit comparison she asked for, with the app-protocol k values (≈17–27) and the
order-dependence result (same 800 filters → k = 15/18/19 on reorderings;
site-grouped → 5 every time) in the notes for Satoshi.

## 5. OLS vs Deming — her methods request, partially pre-answered
She asked for errors-in-both-axes regression since the intercept is the quantity of
interest. Already in hand: `HIPS_Uncertainty` is populated (190/190, median
2.9075 Mm⁻¹) → λ* ≈ 2.96, and the committed result is that EIV moves the corrected
intercept **−1.62 → −2.09** (λ = 1 would overstate it at −2.66). **Fix applied**: the
residual/intercept slides now state "OLS headline; Deming with measured σ_x makes
the offset larger, so OLS is the conservative bound", with the numbers in notes.
Her "you shouldn't have to define any zero-crossing yourself" point stands as a
process rule; no committed fit does that.

## 6. Ethiopia-shaped-300 ∩ lowest-OC/EC-800 overlap — her analysis request: DONE
From committed membership tables (`smoke_cohort_spectral_selection.csv`,
`lowest_ocec_800_cohort.csv`): **overlap = 1 filter of 300** (and only 3 of the full
smoke-906 appear in the 800). Shape-similarity and composition-similarity select
essentially disjoint filters — the "shape should reflect OC/EC" premise fails in
the data, which is consistent with the two cohorts' opposite test behaviour
(Ethiopia-shaped fails the held-out TOR test; OCEC-800 passes). Per her own rule
("almost no overlap → don't worry about it"), no three-panel spectra figure is
needed; the count is now a chip on the cohort slide.

## 7. "Picture can't be displayed" on the MAC slide — real defect, fixed
The embedded figures were RGBA PNGs, which PowerPoint/OneDrive intermittently
refuses to render. **Fix applied**: every plate is flattened to RGB before embedding.

## 8. Titles must carry the claim
Her approved example: "A loading-dependent HIPS artifact can't be closed with
IMPROVE data", with the reasons listed under it. Applied to the deck's findings
slides; recorded here as the house rule for future slides.

## 9. Smaller wording items — applied
- "AIRSpec" reads as a package name: first uses now say **"baseline-corrected
  (AIRSpec)"**, since only the baseline-correction part is being used.
- The **C column is now defined on-slide** (C = |intercept|·MAC/slope — the implied
  constant absorption excess, Mm⁻¹). In the meeting it was mis-glossed as a mean
  Fabs; the notes carry the correct one-liner so that cannot recur with Satoshi.
  (Related: the AIRSpec row is the **same 800 filters**, baselined — the committed
  design has no filter-dropping step; if any local variant dropped "far" filters,
  that variant is not what the committed numbers describe.)
- **Native-MAC framing** added to the MAC slide: with no MAC applied, the deployed
  fit implies MAC ≈ 10/1.90 ≈ **5.3** — noting this trusts FTIR-EC, which is the
  same circularity that bans ChemSpec as a reference; the implied-MAC bridge
  (10.05 at Addis-like composition) is the non-circular version of the same idea.

## Queued for tonight (not blocking the deck)
- Verify the smoke-panel k annotation renders as 4 in the ftir_23 figure (it is 4
  in the committed selector table).
- If Ann still wants it after the overlap result: the three-panel shared/only
  spectra figure (needs the Drive spectra pull; ~1 GB).
