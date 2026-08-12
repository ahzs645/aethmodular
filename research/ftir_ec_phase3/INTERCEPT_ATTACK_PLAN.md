# Attacking the intercept — candidate analyses, ranked

The target, stated precisely. After the best locked calibration (lowest-OC/EC 800 +
AIRSpec), the Addis residual is a season-stable **−2.0 to −2.6 µg/m³ constant**,
independent of score-space extrapolation (ftir_15/22). An additive constant in y-vs-x with
x = Fabs/MAC means a constant absorption excess C = |intercept|·MAC/slope sitting in Fabs
with no FTIR-EC counterpart:

- corrected branch (MAC 10): 1.62 × 10 / 0.86 ≈ **19 Mm⁻¹**
- raw branch (MAC 6):        3.22 × 6 / 0.95 ≈ **20 Mm⁻¹**

**Both MAC branches imply the same ~19–20 Mm⁻¹ of constant, EC-free absorption at the
HIPS wavelength (~633 nm).** That convergence is the reframe: the intercept problem is
"find (or rule out) ~20 Mm⁻¹ of constant non-EC absorption at Addis." Roughly 40% of the
median Addis Fabs (47 Mm⁻¹). Candidate owners: brown carbon, dust, a HIPS-generic
artifact, or an FTIR-side zero error. Each analysis below discriminates among them, uses
only data we already hold (resolve via `ftir_hips_chem/scripts/data_paths.py` — never
hardcode Drive paths), and names the lineage it extends.

## Tier 1 — cheap, in-hand, do first

### 1. Does IMPROVE HIPS itself have a Fabs offset at EC = 0?
`local_db/tables/results_hips.csv` + `results_tor.csv` (the 151,843 matches ftir_16
already built). Regress Fabs on TOR EC — pooled, by site, and in the Addis-like
OC/EC ≤ 2.27 subset — and read the intercept **in Mm⁻¹**. If HIPS generically reads
several Mm⁻¹ at zero EC (filter scattering, tau→Fabs conversion, loading correction),
part of the Addis offset is instrument-generic and scales to Addis conditions; if IMPROVE
runs through zero, the offset is Addis-specific. One afternoon, pure in-hand data,
directly on point. *Extends ftir_16's implied-MAC machinery.*

### 2. Fit the offset as a parameter and demand consistency
Refit all six setup-matrix calibrations as y = a·(x − c), c a free additive Fabs offset
(site-cluster bootstrap for CI, reusing ftir_15's machinery). If the offset physically
lives in Addis Fabs, every setup should recover the *same* c (~1.9–2.0 µg/m³ at MAC 10)
regardless of training cohort; if c varies with the model, it's a calibration artifact,
not an absorption component. Also fit per-season c to re-confirm constancy inside the
new parameterization. *Extends ftir_15/19; `calibration_modes.py` as-is.*

### 3. Errors-in-variables sensitivity (the Deming bound)
`docs/open-items.md`: HIPS Fabs has no uncertainty estimates, so current fits assume
λ = 1; measured comparisons show 22–37% slope attenuation is plausible. OLS attenuation
biases slope low and intercept toward less-negative — so the true intercept may be *more*
negative than −1.62, or the fork's slopes shift. Sweep λ over a defensible range and
report the (slope, intercept) trajectory for both branches. Bounds how much of the
intercept is regression artifact: probably small, but it should be a slide footnote, not
an open question. *Extends the `calculate_regression_stats` Deming path already in
`src/`.*

## Tier 2 — the decisive physics, needs the MA350 leg

### 4. MA350 spectral decomposition: how much 633-nm absorption is not BC?
The MA350 measures at 375/470/528/625/880 nm; BrC absorbs steeply in the UV and barely at
880. Using collocated filter-day averages (minute files under `aethalometry_dir()`;
processing lineage in `src/analysis/bc/` and ftir_hips_chem): assume AAE_BC ≈ 1 anchored
at 880 nm, attribute the excess at 625 nm to BrC, and get **Babs_BrC(≈633) per filter
day**. Two tests: (a) magnitude — is it of order 20 Mm⁻¹? (b) shape — is it roughly
*constant* across seasons (matching the season-stable residual) or loading-proportional
(which would argue against it being our offset)? This is the single most decisive
analysis available without new sampling. *Extends ftir_hips_chem aethalometer processing
+ phase-3 crossplots.*

### 5. Subtract it and re-cross-plot
If (4) is plausible in magnitude: form Fabs_EC = Fabs − Babs_BrC(633) per filter and
redo the locked-model crossplots at both MACs. The success criterion is written in
advance: **intercept CI includes 0 while the held-out TOR test is untouched** (the
correction only moves x at Addis; the calibration itself never changes). If the intercept
closes AND the corrected-branch slope stays ~0.86–1.0 at MAC 10, the whole story
(MAC ≈ 10 + BrC in Fabs) locks. *New notebook; candidate ftir_25.*

### 6. Localize the offset by third instrument
The ftir_hips_chem paper skeleton already notes the Addis anomaly is stronger in
HIPS-vs-FTIR than in MA350-vs-FTIR. Make that quantitative on phase-3 conventions: fit
FTIR-EC against MA350 BC(880) — where BrC contamination is minimal — on the fixed cohort
days. If the intercept against BC(880) is ≈ 0 while against HIPS it is −2, the offset is
localized to 633-nm optics (BrC/dust/HIPS), exonerating the FTIR side. If both show −2,
the problem is in FTIR-EC's zero, and Tier-3 item 8 becomes urgent. *This is also the
AAAR three-way-comparison bridge — one notebook serves both purposes.*

## Tier 3 — supporting discriminants

### 7. Dust's share of the 20 Mm⁻¹
Dust absorbs at 633 nm with AAE ≈ 2–3, partially degenerate with BrC in (4). Two
independent handles: SPARTAN chemical speciation for ETAD (dust/Al-Fe proxies, RCFM dust
mass — check availability in the Spartan tables; note `ChemSpec_EC` is Fabs-derived and
banned as a reference) regressed against the per-filter residual; and Dry-season
behaviour — dust should peak Dry (consistent with ftir_17's relatively-more-O–H in Dry)
while the residual is season-stable, so a large dust share is already disfavoured.
Deliverable: an upper bound on dust's contribution. *Extends ftir_15 residuals + context
tables.*

### 8. The FTIR zero: blanks and low-EC behaviour
If ETAD field blanks exist in the pull (check `etad_dir()` metadata), predict them with
the locked models: they should read ≈ 0. Same for the lowest-loading IMPROVE filters
under the corrected model. If the model reads ≈ 0 at blank/low spectra, the FTIR side's
zero is clean and the offset must live in x. Cheap, closes a loophole. *Extends ftir_13.*

### 9. AERONET column check
`aeronet.aeronet_dir()`: Addis absorption AOD + columnar AAE by season. Independent
(different airmass, column vs surface) confirmation that UV-enhanced absorption is
large and persistent at Addis. Supporting context for (4), not a quantitative anchor.

### 10. ETBI as the background probe — blocked, but pre-register it
If ~20 Mm⁻¹ were a regional/persistent background, Bishoftu (median Fabs 26.9 Mm⁻¹)
would be *mostly* background: its FTIR-EC should come out very low once spectra arrive.
Write the prediction down now; it's a clean out-of-sample test the day the INDH/CHTS/ETBI
pull lands.

## What would settle it regardless
The quartz-TOR campaign (ftir_16 spec: 11–13 days/season) remains the direct measurement.
Everything above is what we can do **before** any new filter is sampled — and (4)+(5)
could plausibly close the intercept with data already on the Drive.

## Suggested order
1 → 2 → 3 (one notebook, all in-hand tables) · then 4 → 5 → 6 as the MA350 notebook
(doubles as the AAAR bridge) · 7–9 as sidebars where they fit · 10 pre-registered.
