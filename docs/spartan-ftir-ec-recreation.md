# SPARTAN FTIR EC: what the calibration is, and our recreation

Written 20 Oct 2026, during the AAAR 2026 poster work. This records what is documented about the calibration behind
SPARTAN's published FTIR elemental carbon (EC), how we rebuilt an equivalent model, how closely it matches, and what is
still unknown.

**Use SPARTAN's published values for anything presented as SPARTAN's EC.** The recreation is an analysis tool for asking
"what if the calibration were changed?" (baseline correction, spectral window, number of components). It is not
SPARTAN's model. The poster build checks this
(`deliverables/aaar_poster_2026/build_poster.py`, "Provenance").

## Summary

- No paper or data file gives SPARTAN's EC calibration settings in full. Four sources together get close (below).
- SPARTAN's EC is a PLS calibration on **raw** FTIR spectra of IMPROVE filters against IMPROVE TOR EC, built with the
  AQRC "FTIR Calibration" Shiny app on about 22 IMPROVE sites. The coefficients are applied directly to SPARTAN spectra,
  and the predicted µg per filter is divided by the SPARTAN air volume.
- Our recreation uses raw spectra over 4000–500 cm⁻¹ and every QC'd lot-251 IMPROVE filter, with k = 18. It reproduces
  SPARTAN's published Addis Ababa EC filter by filter: r 0.992, RMSE 0.30 µg/m³, mean difference +0.02 µg/m³, median
  ratio 0.993. The difference has no offset, no gain error and no trend with EC level, season, air volume or date.
- The likely reason it is not exact: SPARTAN's model was trained on IMPROVE years we do not have spectra for. Our local
  spectra cover filter lots 248 and 251 only, 2021–2023. The training sites and the hand-picked number of components
  may also differ.
- Only AQRC's saved model file would settle it. The app's "Export Model" RDS stores the coefficients for every k and
  the list of training filters.

## What is documented

| Source | What it says |
|---|---|
| AQRC "FTIR Calibration" Shiny app (`global.R`, `R/selectorServer.R`, `R/calibrateServer.R`; source shared by Sean Raffuse, July 2026) | Raw absorbance; every 4th point above 500 cm⁻¹ (`FREQ_MINIMUM_IMP <- 500`, 2,722 points). Default calibration sites: a hard-coded list of 21 IMPROVE sites, with an "All sites" box. The analyst picks dates and filter lots. Field blanks are included by default with the target set to 0. Optional biomass-burning selection from `data/BiomassDetected_*.csv`. Target: TOR EC × volume, in µg per filter. `pls::plsr(ncomp = 80, validation = "CV", segments = 10, segment.type = "interleaved")`. **The number of components is picked by clicking the RMSEP curve**, so it is not recorded anywhere. The app does not contain the step that applies a model to SPARTAN. |
| Anwar (2025), UC Davis dissertation, [eScholarship 0nm8n2xr](https://escholarship.org/uc/item/0nm8n2xr) (chair A. M. Dillner) | "The FT-IR spectra from twenty-two diverse IMPROVE sites and parallel [TOR] EC measurements were used as input to PLS regression"; SPARTAN spectra were "matrix multiplied" by the regression coefficients. SPARTAN deposit area is 3.53 cm², the same as IMPROVE, so no area correction. Network FTIR EC vs HIPS: −0.3 + 1.1x, R² 0.84 (1,822 samples, 22 sites). Addis 2023 (Fig. S2.2): −3.9 + 1.8x, R² 0.75. Christiansen-effect scattering at 2500–3500 cm⁻¹ in sulfate-rich samples, Addis included. |
| Debus et al. (2022), AMT 15, 2685, [doi:10.5194/amt-15-2685-2022](https://doi.org/10.5194/amt-15-2685-2022), and supplement | IMPROVE's operational design: a Global model (all sites, every 5th sample), a Limited model (21 sites from a Gaussian-mixture clustering of 2015 spectra, the app's default list; Table S6), and a Biomass Burning model (Kennard–Stone two-thirds subset). Calibrations use raw spectra (supplement S5). Nothing SPARTAN-specific. |
| SPARTAN papers and data release (Ren et al. 2025, Nat. Commun. 16, 7010; Dillner et al., SPARTAN meeting slides, May 2023; public release v3.1.4) | Only "calibrated to IMPROVE TOR" and "FTIR" (method code 218 at MAIA sites such as ETAD). 25 mm PTFE, about 5 L/min. No training years, components or spectral window. |

## How the recreation was built

Code: `research/ftir_hips_chem/workflows/raw_vs_spline_20261020.py`. Run from the repo root with
`uv run python research/ftir_hips_chem/workflows/raw_vs_spline_20261020.py <stage>`:

| Stage | What it does |
|---|---|
| *(none)* | Same lot-251 calibration on raw 4000–500, raw 1425–4000 and spline-corrected spectra, under our site-held-out protocol. |
| `app` | The same three spectra under the Shiny app's protocol (interleaved 10-fold CV, up to 80 components, first k within 5% of the minimum, fit on all filters). |
| `contrib` | Splits each Addis prediction into per-wavenumber contributions. |
| `spartan` | The app-options grid below: sites × field blanks × biomass, with k matched to SPARTAN's published values. |
| `freeze` | Saves the recreation in the app's coefficient format and checks the saved file reproduces the predictions. |

Inputs:

- IMPROVE and Addis raw spectra: the frozen VIBES Colab bundles under `research/ftir_hips_chem/output/tables/`.
  - `vibes_colab_bundle/stage/` covers 4000–1425 cm⁻¹. Its content hash, 57c7c3c6…, matches the spline run in
    `vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09`, and the script checks this.
  - `vibes_colab_bundle_full500/stage/` covers 4000–500 cm⁻¹ and has the same filters in the same order (checked).
  - Both use the app's 2,722-point grid exactly; the exported app coefficient files share it.
- TOR EC, in µg per filter, comes from the same run's case table (`ab.load_context()`).
- The 189 Addis filters on the poster are those with a published SPARTAN FTIR EC value.
- HIPS EC = Fabs / 10. Fits are Deming with the poster's error ratio (λ ≈ 3.34).

Choices:

1. **Spectra:** raw absorbance, 4000–500 cm⁻¹, no baseline correction (the app's recipe).
2. **Calibration set:** every lot-251 IMPROVE filter (11,107 filters, 151 sites, May 2022 to May 2023), no field
   blanks, no biomass selection.
   - One filter is excluded: TONT1, 19 Nov 2022. Its raw spectrum sits about 3 absorbance units high (row maximum 3.19
     against a 99.9th percentile of 0.61). Raw-spectra models predicted it at about 1,400 µg, and it dominated their
     results. Baseline correction hides the offset, so spline-based runs kept it.
   - **Exclude this filter from any raw-spectra fit.** A first pass that kept it reached the opposite conclusion about
     baseline correction.
3. **Fit:** PLS on all filters, as the app ships its model (`sklearn` `PLSRegression(scale=False)`, which is the same
   calculation as `pls::plsr`; see the exactness check below).
4. **Components:** k = 18, chosen as the k whose Addis predictions best match SPARTAN's published values (RMSE
   minimum). k = 16–20 all match about as well.
5. **Applying it:** µg per filter = b0 + Σ b_j x_j on raw spectra, divided by the filter's air volume for µg/m³. No
   deposit-area scaling is needed (3.53 cm² for both networks).

Saved model: `research/ftir_hips_chem/output/tables/raw_vs_spline_20261020/spartan_equivalent/`. It holds
`calibration-EC-spartan-equivalent-k18.csv` (the app's "Export Coefficients" layout; the row with Wavenumber 0 is the
intercept), `addis_spartan_vs_recreation.csv` and a README. `output/` is git-ignored, so rebuild these with the
`freeze` stage.

## How well it matches

Addis Ababa, 189 filters, FTIR EC against HIPS EC (Deming):

| | SPARTAN published | Recreation (k 18) | Mean difference (µg/m³) |
|---|---|---|---|
| All | 2.30x − 6.09, R² 0.78 | 2.32x − 6.20, R² 0.78 | +0.02 (RMSE 0.30) |
| Dry (Oct–Feb), 73 | 1.41x − 2.62 | 1.37x − 2.44 | 0.00 |
| Belg (Mar–May), 51 | 1.89x − 4.17 | 1.84x − 3.87 | +0.07 |
| Kiremt (Jun–Sep), 65 | 2.19x − 5.09 | 2.32x − 5.83 | +0.01 |

Filter by filter: r 0.992, median ratio 0.993 (middle half 0.97 to 1.03). SPARTAN against the recreation fits
0.99x + 0.03. Correlations of the difference with EC level, HIPS EC, air volume and date are 0.01, 0.04, −0.06 and
−0.10.

### Options that match worse

Each row was rebuilt from lot 251 unless noted, with the best-matching k:

| Calibration sites | Field blanks at 0 | Biomass filters | Best k | RMSE vs SPARTAN | r | Addis fit |
|---|---|---|---|---|---|---|
| all 151 sites | no | any | 18 | 0.303 | 0.992 | 2.32x − 6.20 |
| all 151 sites | no | excluded | 20 | 0.292 | 0.992 | 2.33x − 6.23 |
| all 151 sites | yes (323) | any | 18 | 0.302 | 0.992 | 2.32x − 6.18 |
| all 151 sites | no | only | 5 | 0.664 | 0.965 | 2.18x − 5.77 |
| app's 21 sites | no | any | 13 | 0.479 | 0.985 | 2.53x − 7.13 |
| app's 21 sites, lots 248 + 251 | no | any | 17 | 0.375 | 0.991 | 2.53x − 7.19 |
| all 157 sites, lots 248 + 251 | no | any | 19 | 0.323 | 0.991 | 2.21x − 5.76 |
| all 151 sites, raw 1425–4000 instead of 4000–500 | no | any | 18 (fixed) | 0.99 | 0.980 | 1.90x − 4.96 |

- Field blanks make no difference.
- Smoke-only calibrations match badly.
- The app's 21-site list on one lot matches worse than all sites, but adding the second lot closes most of the gap.
  That is consistent with SPARTAN's model being a multi-year calibration on about 22 selected sites.
- Cutting the spectra at 1425 cm⁻¹ moves the match off, so SPARTAN's model very likely uses the full range.

### Exactness check against real app exports

The app exports from 24 June 2026 were the smoke-only selection: 916 biomass-listed lot-251 filters, 906 of them
with TOR EC, May 2022 to April 2023. Refitting those exact spectra with our code reproduces both exported coefficient
files to within 6 × 10⁻¹² (relative), at k = 31 and k = 27. Our PLS therefore equals the app's, and the remaining
difference from SPARTAN is the training set and k, not the method. Those exports are not SPARTAN's model: on Addis
they give 3.65x − 11.73 and 3.43x − 10.25 (r 0.963 against SPARTAN). This is the model the phase-3 notes call
"smoke-906".

## What the recreation shows (comparison page)

`deliverables/aaar_poster_2026/addis_intercept.html` (published as a private artifact) has:

- SPARTAN and the recreation side by side;
- the same calibration on raw versus spline-corrected spectra under both protocols;
- the per-wavenumber contribution map.

At k = 18, the poster's −6.1 to −1.2 µg/m³ intercept change splits into:

- calibration set: about 0;
- leaving out 500–1425 cm⁻¹: about +1.2;
- baseline correction: about +3.4;
- the poster calibration's smaller k: about +0.3.

Raw-spectra models build the Addis prediction from large, cancelling contributions, including bands with no particle
absorbers (3600–4000 and 1800–2800 cm⁻¹). After baseline correction the model reads EC mostly through the C–H band,
and Addis is low in organic carbon per unit EC.

## Still unknown, and how to settle it

- SPARTAN's training years, exact sites (the 22nd beyond Debus's 21), number of components, and whether field blanks
  or outlier removal were used.
- Whether a separate smoke model was applied to any SPARTAN filters. Anwar (2025) describes a single 22-site
  calibration.
- Settle it with AQRC's exported model file (RDS) for the SPARTAN EC calibration. `readRDS()` gives
  `$model` (training rows), `$coefficients` (all k) and `$metadata` (training filters with site, date and matched TOR).
  The k actually used would still need to come from whoever built it.

## Related

- `docs/spartan-network-data-pulls.md`: where the SPARTAN FTIR and HIPS exports come from.
- `docs/filter-optics-reference.md`: SPARTAN and IMPROVE filter and HIPS optics, MAC = 10.
- `research/ftir_ec_phase3/PHASE3_SUMMARY.md`: the earlier deployed-model lineage notes (smoke-906, the HIPS-916
  transfer).
- `research/ftir_hips_chem/addis_public246_20261020.ipynb`: the same analysis on all 246 public Addis filters.
