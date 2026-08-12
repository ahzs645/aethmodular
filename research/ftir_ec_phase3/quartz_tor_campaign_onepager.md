# Quartz TOR at ETAD — the measurement that decides the Addis intercept

For Ann and Satoshi · 2026-08-12 · decision requested: go/no-go on ~36 quartz filters

## Ask

Green-light a co-located quartz TOR campaign at ETAD: **12 Belg + 13 Dry + 11 Kiremt = ~36
quartz filters in total** — the per-season counts are 11–13 days each, and 36 is the sum
across all three seasons, not a per-season figure — sampled alongside
the routine Teflon so every quartz day carries paired FTIR and HIPS, analysed IMPROVE_A TOR
exactly as Adama Batch-54 was. Second ask, free: add **ETBI (Bishoftu)** to the FTIR spectra
pull already being requested for INDH/CHTS.

## Why now — this was one of six asks yesterday; today it is the only one left

The Addis calibration intercept restates as a constant absorption excess,
`C = |intercept| · MAC / slope` ≈ **21 Mm⁻¹** (18.8–26.1 across the six setup-matrix
calibrations, median 21.5) — about **46% of median Addis Fabs (47.11 Mm⁻¹)**. C is invariant
to the MAC fork by algebra, so the target is well posed whether or not MAC 6 vs 10 is ever
settled (`ftir_25`, committed). It is not a marginal correction.

Today's four results each removed a candidate owner and none of them chose a winner:

- **Not a generic HIPS zero.** IMPROVE HIPS runs through the origin: pooled OLS intercept
  **+1.345 Mm⁻¹** as an upper bound (trimmed at EC ≤ p95, **+0.200**; Addis-like subset,
  trimmed, **+0.097**). At EC ≤ 0 the median Fabs is **+0.12** with 23.8% of filters reading
  negative, and only **186 of 160,023** IMPROVE filters reach 21.5 Mm⁻¹ of *total* absorption.
  (Preliminary — ftir_26, in preparation.)
- **Not on the FTIR axis.** FTIR-EC vs MA350 BC(880) has an intercept whose CI includes zero:
  **+0.285 [−0.022, 0.593]**, R² 0.870. (Preliminary — in preparation.)
- **Not measurable by the MA350.** AAE(625,880) = **0.944 ± 0.060**; implied BrC absorption
  **−2.06 Mm⁻¹**, negative on 84.5% of days against the **+21.7** required. The instrument
  cannot see a red excess in either direction — this is *not* evidence that Addis has no brown
  carbon. (Preliminary — ftir_28, in preparation.)
- **And no reference in hand can arbitrate.** Both candidate EC references in the committed
  dataset are circular. `ChemSpec_BC` is Fabs/10 rounded (R² 0.9982, implied MAC 10.0003) —
  x-circular. `ChemSpec_EC` is `EC_ftir` rounded (r² 0.999693, ratio 1.0000, median |Δ| 0.0030,
  the 2-dp half-width) — y-circular. Committed in `ftir_25` (commit `00ab987`).

That leaves exactly three live explanations, and nothing already collected separates them:
a **loading-dependent HIPS artifact**; **curve geometry** (a straight line fitted to a concave
Fabs-vs-EC relationship manufactures a negative intercept with no offset present — IMPROVE
per-site intercepts run at a median 35% of site mean Fabs, IQR 28–53%, and Addis's 46% sits
inside that band); and **real non-EC absorption**. Quartz TOR is the only route to an EC
reference that is circular with neither axis.

(Honest caveat, since Satoshi will ask: the six setups agreeing on C is *not* six independent
measurements. C is invariant under any multiplicative rescaling of the predicted-EC axis, so
the agreement says the additive discrepancy survives every scaling choice we have made — not
that it is real. Same reason we need an outside axis.)

**Dust is already disfavoured, on prior work worth re-checking.** Within the "real non-EC
absorption" branch, dust is the sub-case an earlier analysis argues against: the HIPS/FTIR
ratio correlates with coarse AOD at **r ≈ −0.33**, i.e. more dust goes with HIPS reading
*lower* relative to FTIR, where dust inflating Fabs would give a positive correlation
(`ftir_hips_chem/RESEARCH_PROGRESS.md:108-114`). Two reasons not to lean on it hard yet: it
reasons from the 405 nm assumption that the section below flags as open, and it rests on
AERONET data that is not present in this checkout, so it could not be re-derived here. If it
holds, it narrows the absorption branch toward brown carbon; either way TOR still arbitrates.

## What it measures

Paired quartz gives EC that is derived from neither Fabs nor FTIR, which turns three
questions into direct regressions on the same days:

1. **MAC 6 vs 10** — implied MAC = Fabs / EC_TOR, per day. This is what the campaign is
   powered for.
2. **The offset itself** — Fabs regressed on EC_TOR, with an intercept read directly in Mm⁻¹
   against the ~21 target.
3. **Artifact vs constant** — whether the low-loading days still carry ~21 Mm⁻¹ (constant
   non-EC absorption) or fall toward the origin with the excess appearing only at high loading
   (loading-dependent artifact / curvature). Readouts 2 and 3 come free with the same filters,
   but the power calculation below sizes only readout 1 — so spread days across the loading
   range within each season, not just across seasons.

## Design and cost

- **~3σ per day** separates MAC 6 from MAC 10 (median z/day: 2.89 Belg, 2.82 Dry, 3.03
  Kiremt). Statistics are not the constraint; seasonal and protocol systematics are.
- **11–13 days per season for 5σ per season** (12 Belg / 13 Dry / 11 Kiremt) → **~36 filters
  TOTAL**, spread across the three seasons rather than concentrated.
- Two caveats carried from `ftir_16` and not to be dropped in retelling: the 5σ figure
  **already includes a 0.5 de-rating** for protocol systematics (quartz artifacts,
  face-velocity, bounce); and σ is a **modelled** TOR uncertainty (10% relative + 0.3 µg/m³),
  **not a measured** one. If real TOR reproducibility is worse than modelled, the day counts
  rise.
- **Quartz is required.** TOR cannot run on the archived Teflon, so no retrospective
  substitute exists. Cost is filters, sampling days and TOR analysis slots — no new
  instrumentation.

## Precedent — Adama Batch-54

Five quartz filters, all sampled July 2024, TOR OC/EC **4.63–7.23 (median 6.08)**. The chain
— sample, ship, IMPROVE_A TOR, results in hand — is proven end to end in-country. Note what it
is and is not: there is **no paired FTIR/HIPS on the Adama filters**, which is precisely why
Batch-54 is *feasibility precedent and not evidence*. The pairing is the whole point of the
new campaign.

## What each outcome would mean

- **Implied MAC ≈ 10 and Fabs-vs-EC_TOR through the origin.** The ~21 Mm⁻¹ is not an
  absorption excess in the world. It is our line fitted to a curve, or FTIR-EC reading low at
  Addis loadings. The fix is a model/geometry fix, and the calibration story simplifies.
- **Implied MAC ≈ 6, or an intercept near +21 Mm⁻¹ that persists at the lowest loadings.**
  A real non-EC absorber is carrying roughly half of Addis Fabs, and every Fabs-derived EC at
  Addis is biased high by that amount until it is apportioned.
- **Intercept present but only at high loading.** Loading-dependent HIPS artifact — a
  correctable instrument effect, and one that would propagate to every high-Fabs SPARTAN site,
  not just ours.
- **A null is informative here.** Because C is invariant to gain, no further refitting of
  Fabs-derived quantities can produce this answer; a clean origin result closes the hypothesis
  in a way nothing in the current dataset can.

## ETBI pre-registration — recorded now, before the data exists

ETBI (Bishoftu) is a second Ethiopian SPARTAN site nobody has analysed: **32 filters
(Oct–Dec 2025), 26 with HIPS Fabs, median 26.9 Mm⁻¹**. It is circular with neither axis and
costs nothing beyond the spectra pull, which rides along with the INDH/CHTS export already
being requested.

**Prediction, registered 2026-08-12:** if ~21 Mm⁻¹ is a regional or persistent background,
then ETBI is *mostly* background — and when its spectra arrive, ETBI FTIR-EC should come out
very low, with the same C recovered on ETBI filters.

**Like-for-like caution:** ETBI's window is Oct–Dec, i.e. **Dry only**. The correct comparison
is against Addis's **Dry-season** median Fabs of **43.2 Mm⁻¹**, not the all-season 47.1. The
like-for-like statement is 26.9 vs 43.2.

**What would falsify it:** ETBI FTIR-EC that tracks ETBI Fabs through the origin, or an ETBI-C
materially smaller than Addis's ~21 Mm⁻¹. Either would mean the offset is site-specific to
ETAD — sampling, handling, or local composition — rather than a regional absorbing background.

## One open question that changes the prior (not the plan)

**Do not quote a HIPS wavelength as settled.** The repo carries both **405 nm**
(`ftir_hips_chem/RESEARCH_PROGRESS.md:112`, `ftir_hips_chem/COMPLETE_RESEARCH_SUMMARY.md:19`)
and **~633 nm**
(White 2025, assumed throughout phase-3 prose); `docs/filter-optics-reference.md` marks it
**OPEN** — "do not quote without checking with SPARTAN". It matters for plausibility: BrC and
dust absorb several-fold more at 405 nm than at 633 nm, so at 405 a ~46% non-EC share of Fabs
is considerably more plausible, and at 633 correspondingly less. It does not change the
campaign — TOR arbitrates at either wavelength — but it changes how surprised we should be by
each outcome. Ann, this is a one-email question to SPARTAN.

## Decision requested

1. **Go/no-go on ~36 quartz filters at ETAD**, 12 / 13 / 11 across Belg / Dry / Kiremt,
   co-located with routine Teflon, IMPROVE_A TOR as per Batch-54. Earliest season start sets
   the timeline; the seasonal spread is not compressible.
2. **Add ETBI to the INDH/CHTS spectra pull** — no marginal cost, and the prediction above is
   on the record before the data lands.
3. **Ann:** confirm the HIPS wavelength with SPARTAN.

Preliminary results above (ftir_26, ftir_28, the MA350 crossplot) are being committed in
parallel and are marked as such; `ftir_25` and the `ftir_16` campaign spec are committed.
