# Data opportunity audit — 2026-08-24

## Bottom line

The strongest remaining analysis is a **paired residual attribution** on the
same filters:

1. `HIPS / MAC - MA350`
2. `FTIR EC - MA350`

Explain both with HIPS reflectance/transmittance terms, loading, potassium,
normalized PMF factors, month/site, and the committed `char_06` spectral class.
This directly separates three competing interpretations:

- a predictor acting mainly on the HIPS residual supports an optical/HIPS path;
- a predictor acting mainly on the FTIR residual supports an FTIR interference path;
- a predictor acting on both supports aerosol optical-property/MAC variation.

This is more discriminating than another pairwise crossplot because the two
residuals share the same independent MA350 anchor and filter window.

## What is actually available

| Source | Grain and usable coverage |
|---|---|
| Unified filter chemistry | 44,493 long rows; 1,603 filters; 4 sites; 62 parameters |
| Same-filter HIPS + FTIR | ETAD 190; CHTS 163; INDH 63; USPA 130 |
| HIPS + FTIR + K | ETAD 188; CHTS 147; INDH 27; USPA 107 |
| HIPS + FTIR + Al/Si/Ti/Fe | ETAD 188; CHTS 148; INDH 27; USPA 128 |
| Exact-window HIPS + FTIR + MA350 | Addis 173; Beijing 65; Delhi 24; Pasadena 65 |
| Those triplets plus K | Addis 171; Beijing 60; Delhi 24; Pasadena 62 |
| Processed MA350 windows | Addis 515; Beijing 222; Delhi 113; Pasadena 489 valid IR days |
| Raw MA350 | about 4.16 million minute rows across four sites |
| SPARTAN HIPS batch | 3,963 rows; 27 sites; 3,175 complete PM2.5 R1/T1 rows; 582 complete blanks |
| IMPROVE scan fields | 147,380 filters, 167 sites with initial/minimum/final 635-nm R/T; 145,200 with volume |
| Addis PMF | 102 dates, all exact joins to locked predictions |
| `char_06` classes | 239 exact joins to locked Addis predictions |
| Audited explorer grid | 61,635 rows; 12,327 configuration-k results per target; 5 targets |

Important limitations:

- HIPS laser power and sensor-temperature telemetry are entirely null.
- Delhi's 24 MA350 triplets are confirmation-only, not a standalone mechanism
  sample.
- AERONET is Level 1.5; there is no strict Level 2/U27 set in hand.
- Reconstructed and augmented targets are provisional and must not be optimized
  against as though they were ordinary screening sets.

## Ranked analyses that are answerable now

### 1. Paired HIPS/FTIR residual attribution against MA350

Use the exact-window triplets and fit a multilevel or site-stratified model for
both residual outcomes. Start with Addis, replicate in Beijing/Pasadena, and use
Delhi only as a direction check. Use held-out or repeated cross-validation and
report incremental out-of-sample R² rather than a large in-sample coefficient
table.

Primary inputs:

- `output/tables/variation_closure/four_site_ma350_pairs.csv`
- `research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl`

### 2. Operational HIPS reflectance/scattering decomposition

Build operational predictors from the raw R1/T1 signals and the deployed line,
then test incremental association with `HIPS-MA350` and locked calibration
residuals. Call these *operational reflectance/scattering indices* unless and
until a physical decomposition is validated.

This is scientifically motivated: HIPS explicitly combines simultaneous
reflectance and transmittance to address sample/filter scattering, while its
calibration is derived from contemporary blanks. The 2024 patterned-deposit
study also stresses that deposit geometry affects filter photometry. See
[White et al. (2016)](https://doi.org/10.1080/02786826.2016.1211615) and
[White et al. (2024)](https://doi.org/10.1080/10962247.2024.2442634).

Do not pool only by lot when reconstructing blank lines. In the current batch,
four lots resolve to nine distinct deployed calibration lines; lots 248, 250,
and 251 each contain multiple lines.

### 3. Locked residuals versus PMF factors and `char_06`

Fit normalized source fractions using a compositional parameterization (for
example an ilr transform), controlling loading and cyclic month. Separately
compare the locked residual distribution for the 98 low-similarity and 141
normal `char_06` filters. Do not enter all five raw GF fractions in an
unconstrained OLS model.

### 4. Cross-site PLS domain geometry: score distance plus Q residual

For every locked calibration and target, report both:

- score-space distance / Hotelling-like T²; and
- spectral reconstruction Q residual.

The explorer currently has the first family only. The second distinguishes
spectra that are ordinary in the EC-relevant latent space from spectra carrying
large unmodelled bands/background. This follows the same external-validation
logic as Reggente et al., who showed that score-space squared Mahalanobis
distance provides a rough indication of mean prediction error and found that
some new sites required a different calibration
([Reggente, Dillner & Takahama, 2016](https://doi.org/10.5194/amt-9-441-2016)).

### 5. Raw MA350 DualSpot reconstruction

Recompute the loading correction from spot-1/spot-2 attenuation and flows, then
re-integrate over the exact filter windows. Compare raw and vendor-corrected
MA350 against HIPS and FTIR. This tests whether MA350 is independent at the
algorithm level, not merely a third exported column. DualSpot is specifically a
real-time filter-loading correction; its parameter is composition- and
time-dependent ([Drinovec et al., 2015](https://doi.org/10.5194/amt-8-1965-2015)).

### 6. IMPROVE initial/minimum/final 635-nm scan dynamics

Derive within-filter reflectance/transmittance scan-shape indices for 147,380
filters and model them against loading, EC, fAbs, site, and lot. Use this as a
large-network mechanism/plausibility benchmark, not as a direct SPARTAN
correction: the geometries differ.

### 7. Grid factor importance and model-choice uncertainty

Use the audited 61,635-row grid for blocked functional ANOVA or permutation
importance across cohort, cutoff, preprocessing, selection space, CV mode, k,
and target. Add bootstrap Pareto stability and a prediction envelope across
prequalified models. This would replace anecdotal statements such as
“baselining matters most” with variance shares and stability probabilities.

## Valuable but limited

- Low-wavenumber charcoal ratios: substrate/baseline dominated below about
  1425 cm-1.
- Shared-component MCR-ALS: exploratory and strongly baseline-sensitive.
- Reverse-orientation HIPS comparisons: 2,747 paired filters, but the result
  orientation semantics are unconfirmed.
- Wind-sector attribution: station representativeness and season-wind
  confounding remain substantial.
- AERONET surface/column refinements: Level 1.5 only.
- Adama: five quartz TOR observations and no same-filter Teflon twins.

## Blocked by new information or measurements

- Collocated quartz TOR EC.
- Solvent extraction plus HIPS remeasurement; TEM/tar-ball counts.
- Deposit images and confirmed SPARTAN support-screen geometry.
- Official QC'd HIPS release and uncertainty semantics.
- AERONET Level 2/U27 exports.
- IMPROVE scan pulls for lots 253/255/259/264.
- Teflon twins for Adama.

## Do not propose again as new

Already completed or substantially run: linear/quadratic HIPS blank-line tests,
HIPS epoch/loading diagnostics, four-site York/EIV, four-site MA350 crossplots,
potassium/dust residual tests, three-wavelength AERONET, both February season
conventions, locked reconstructed holdouts, hybrid low-OC/EC, di-PLS,
cross-site 1617-cm-1/baseline adjudication, and the grid winner audit.

## Recommended implementation order

1. Build one locked, filter-level table joining MA350, HIPS R/T, FTIR prediction,
   chemistry, normalized PMF and `char_06`, with a join ledger.
2. Freeze the model specification before inspecting coefficient signs.
3. Fit the two residual outcomes together, with site replication and bootstrap
   uncertainty.
4. Add score distance plus Q residual to the explorer as a domain panel.
5. Only then decide whether raw MA350 reconstruction or the 147k-filter IMPROVE
   scan analysis has the higher expected value.
