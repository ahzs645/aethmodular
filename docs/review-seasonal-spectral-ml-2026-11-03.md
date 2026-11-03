# Whole-spectrum seasonal ML: Addis, existing data

Executed 3 November 2026. This extends the private **Research → Baseline lab** gallery. No new sampling, laboratory reference measurements, production EC recalibration or canonical exclusions were introduced.

## Finding

The existing spectra contain transferable seasonal information. They also predict the existing FTIR–HIPS disagreement, but the spectral regions and preferred representation depend on the task, season and test cycle. Random forest is useful for these diagnostics; it does not consistently outperform the fixed linear and PLS comparisons.

**Revised after the lot and FTIR-EC checks (below).** The first version of this report named corrected **1800–2250 cm⁻¹** (especially 1800–1900) as the most actionable baseline lead. That does not hold up:

- **Only the random forests rely on it.** PCA-ridge and PLS predict D equally well or better while putting essentially no weight there.
- **It disappears once FTIR EC is accounted for.** D = FTIR EC(X) − HIPS BC, and a straight line on FTIR EC alone (no spectrum) predicts D as well as the best spectral model (RMSE 0.488 vs 0.487 µg/m³ in the complete test year).
- **What the spectrum knows beyond EC** sits in **3500–4000 cm⁻¹** (the one interval all three model types use), **2800–3000 cm⁻¹ C–H**, and **1560–1800 cm⁻¹** (the overlapping mid-infrared bands and the carbonyl region).

Separately, filter lot is aliased with season in the first training year: every 2022–23 Bega filter is lot 248, and every Kiremt filter is lot 251. Corrected spectra still recognise season better than raw when train and test share lot 251, but the forward-test gap between raw and corrected is partly a lot effect.

A single Ethiopian correction, or an automatic seasonal multiplier, is not established by these results.

## Cohort and evaluation

Both tasks start from the same 232 direct-HIPS Addis physical filters. Repeated native scans are averaged within their physical filter before splitting. No scan appears on both sides of a split. Public-HIPS extensions are not mixed into this primary experiment.

| October–September cycle, labelled by ending year | Filters | Role |
| --- | ---: | --- |
| 2023 | 91 | Initial training; December–September observations, missing October/November |
| 2024 | 97 | Primary held-out test; all 12 months sampled |
| 2025 | 30 | Separately reported sparse later test; September absent; trained on the earlier 188 |
| 2026 | 14 | Latest Bega-only observations; descriptive spectra only |

Thus **218 filters participate in model fits/evaluation**, and **127 unique filters receive held-out predictions**. “232 filters” is the source cohort size, not 232 independent test observations. The weather subset is exactly the same 91 training plus 97 primary test filters; later observations lack weather.

Primary calendar: **Bega October–January, Belg February–May, Kiremt June–September** (`belg_feb`). The historical `dry_feb` convention remains an explicit sensitivity. Dates, month, year and season are not predictors in season classification. The primary test has 37 Bega, 34 Belg and 26 Kiremt filters. Bishoftu lacks the necessary three-season coverage for an equivalent validation.

## Models and representations actually run

The measured raw spectrum contains 2,722 native channels, approximately 500–3998 cm⁻¹. Raw, spline-corrected and removed-component representations are also compared on the same 1,897-channel 1560–3998 cm⁻¹ window. This avoids confusing a preprocessing change with a wavelength-window change. The cutoff excludes every selected scan's algorithmically padded spline zeros. A 1600 cm⁻¹ sensitivity additionally excludes the small extrapolated lower tail.

Other controls include raw second derivatives, omission of the PTFE-rich 1000–1300 and CO₂-prone 2250–2400 cm⁻¹ windows, the alternative season calendar, and omission of the three filters flagged by the earlier scan audit. Signed absorbance is retained throughout; there is no positive-area normalization or clipping.

The fixed forest uses 200 trees, maximum depth 8, minimum leaf size 5 and feature fraction 0.33. The linear comparison uses eight training-fitted PCA components followed by logistic regression or ridge regression. PLS regression uses eight components directly, without a preceding PCA. Scaling, imputation, encoding and fitted transforms remain inside the training fold. These are prespecified configuration comparisons, not a test-set-tuned winner search.

Benchmarks explicitly include **HIPS, PM₂.₅, collection volume and filter lot**; additional regression benchmarks add season or airport sample-day relative humidity/rainfall. HIPS is a component of the outcome, so these are HIPS-conditioned diagnostics, not an independent reference. Airport sample-day weather is not collection-interval or instrument-chamber humidity.

## Prediction results

Balanced accuracy is mean recall across the three seasons; the training-only reference is 0.333. Disagreement is current FTIR EC minus HIPS-derived BC at the existing MAC of 10 m²/g. Lower RMSE means better prediction of that difference, not better EC accuracy.

| Fixed model | Primary season balanced accuracy | Primary disagreement RMSE, µg/m³ | Sparse later season balanced accuracy | Sparse later disagreement RMSE, µg/m³ |
| --- | ---: | ---: | ---: | ---: |
| RF, raw full spectrum | 0.4884 | 0.5782 | 0.6111 | 0.5676 |
| RF, raw common window | 0.5550 | 0.6000 | 0.6389 | 0.5746 |
| RF, spline common window | 0.6990 | 0.6365 | 0.7778 | 0.4559 |
| RF, removed component | 0.5150 | 0.5942 | 0.6111 | 0.5634 |
| RF, second derivative | 0.3392 | 0.6839 | 0.8611 | 0.4956 |
| PCA + linear, raw full spectrum | 0.5842 | 0.4870 | 0.8056 | 0.5591 |
| PCA + linear, spline common window | 0.7817 | 0.6708 | 0.7500 | 0.4723 |
| PLS, raw full spectrum | — | 0.5549 | — | 0.5053 |
| PLS, spline common window | — | 0.5119 | — | 0.4750 |

The corrected representation distinguishes seasons more clearly in the primary test. Raw/removed information helps predict disagreement. The raw-versus-corrected RF ranking for disagreement reverses in the sparse later test. The derivative classifier also changes sharply between cycles. These are reasons to preserve the separate test populations rather than pool them into one headline score.

On the primary test, the linear HIPS/PM₂.₅/volume/lot benchmark has RMSE 0.9531; adding season gives 0.8761. Adding available weather gives 0.8135, and adding both weather and season gives 0.8311. Those comparisons use the same 97 test filters. They support conditional diagnostics, not a claim that weather causes the observed spectral behavior.

Separate component checks expose the target coupling. Raw-spectrum RF predicts FTIR EC itself with primary R² 0.9528, versus 0.7594 for HIPS BC. Since disagreement contains spectrum-derived EC(X), its predictability partly recovers a quantity already derived from the predictors. Neither this result nor improved agreement validates Ethiopian EC accuracy.

## Which spectral regions matter?

The workflow jointly shuffles all channels in each predeclared interval, then measures held-out loss. It also removes each broad interval and refits preprocessing and the model on the training filters. Fine 100 cm⁻¹ bins, named broad intervals and the whole included spectrum remain separate. Negative effects are retained. Ten-repeat variation is Monte Carlo variation, not a population confidence interval.

For primary **raw-spectrum disagreement prediction**, shuffling 1300–1560 cm⁻¹ raises RMSE by 0.4304 µg/m³ across filters and 0.1877 when shuffles stay within seasons. Yet removing that interval and refitting raises RMSE by only 0.0062. The forest relies on it, but much of its information can be replaced by correlated regions. The corresponding whole-spectrum shuffle is not plotted as a localized spectral peak.

For primary **corrected-spectrum season classification**, shuffling 1800–2250 cm⁻¹ raises balanced error by 0.3444; removing and refitting raises it by 0.1570. The strongest fine interval is 2100–2200 cm⁻¹, followed by 1900–2000 and 1800–1900. For corrected disagreement, removing the entire 1800–2250 interval increases RMSE by about 0.0682, substantially less than its permutation effect.

### Seasonal interpretation on the same held-out year

The models are trained once on all 91 earlier filters. These checks then shuffle only among test filters in the selected season; they do not fit three small seasonal forests.

| Season | Test n | Raw RF baseline RMSE | Corrected RF baseline RMSE | Strongest named corrected interval | Mean within-season ΔRMSE |
| --- | ---: | ---: | ---: | --- | ---: |
| Bega | 37 | 0.456 | 0.638 | 1560–1700 cm⁻¹ | 0.117 |
| Belg | 34 | 0.584 | 0.491 | 1800–2250 cm⁻¹ | 0.580 |
| Kiremt | 26 | 0.711 | 0.786 | 1800–2250 cm⁻¹ | 0.878 |

Among the fine bins, **1800–1900 cm⁻¹ leads corrected disagreement in all three seasons**, with ΔRMSE approximately 0.056, 0.520 and 0.748 respectively. The broad and fine rankings need not coincide: grouped importance is nonadditive and group widths differ. Raw 1300–1560 leads among named raw intervals in all three seasons, with ΔRMSE 0.128, 0.202 and 0.254.

These are descriptive differences in predictive reliance. Their magnitudes also depend on each season's target variation and starting model error; they are not a formal season interaction test, percentage source contribution or chemical mass fraction.

**Superseded for baseline work.** Both tables above use RF fitted to D directly, so they show where the forest reads FTIR EC. See *Checks: filter lot and the FTIR EC term* below. On what the EC line misses, 1800–2250 carries nothing in any season (≤ 0.02 µg/m³ for every model). The per-season leads become 2800–3000 and 3500–4000 cm⁻¹ for PCA-ridge, and 1700–1800, 1560–1700 and 3500–4000 cm⁻¹ for PLS. The disagreement's spread also differs by season (SD 0.61, 0.83 and 1.36 µg/m³ for Bega, Belg and Kiremt in 2023–24), which inflates the Kiremt numbers.

## Direct baseline mechanism check

The inspected implementation is `research/ftir_ec_phase3/scripts/airspec_baseline.py`, SHA-256 `394d72cfe2214b94f1293b11704d774b15bfc952c041d74b05bfa5bd7939008c`. The transition receipt retains the exact source locations and a source snapshot.

- Segment one covers 1820–4000 cm⁻¹; segment two covers 1425–2000 cm⁻¹.
- Their independently fitted baselines are averaged in the 1820–2000 overlap, without an explicit continuity constraint at its edges.
- 2220 cm⁻¹ is a fitting-weight boundary, not a segment join.

All 350 native channels in 1800–2250 are finite supported measurements. The region is nevertheless algorithmically structured. This makes broader baseline behavior a testable explanation, not a proven cause or an exclusive chemical-band assignment.

The countercheck masks ±20 cm⁻¹ around all three documented transitions, removing 93 channels from both raw and corrected common windows. Primary raw RF disagreement RMSE changes 0.6000 → 0.5994; corrected changes 0.6365 → 0.6748. Corrected season balanced accuracy changes 0.6990 → 0.7351. On the sparse later test, corrected disagreement RMSE changes 0.4559 → 0.4809 and season accuracy 0.7778 → 0.8056.

The signal therefore survives narrow masking. This does not eliminate broader baseline effects because retained corrected channels still reflect the original fit. Masking also removes real measurements and changes RF feature availability. The diagnostic does not justify replacing the production baseline. Filter-level adjacent-channel differences at 1820/2000 are saved with exact channel coordinates and orientation; they also contain slope and noise and are not automatically artifacts.

## Scan quality and filters to inspect

The preexisting flags are ETAD-0037 (nearly flat acquisition), ETAD-0139 (discordant repeat scans) and ETAD-0158 (documented physical damage). The sensitivity omits them without changing canonical exclusions. On the same 95 retained primary test IDs, raw RF RMSE changes 0.5814 → 0.5742 and corrected 0.6356 → 0.6178. Season balanced accuracy changes 0.4920 → 0.4634 and 0.6998 → 0.6808 respectively. The qualitative result survives; interval importance was not rerun for this QC subset.

For inspection, the saved examples select the two largest absolute differences between raw and corrected RF held-out predictions within each season:

- **Bega:** ETAD-0161 (10 January 2024), ETAD-0157 (1 January 2024).
- **Belg:** ETAD-0181 (4 March 2024), ETAD-0202 (27 April 2024).
- **Kiremt:** ETAD-0243 (25 August 2024), ETAD-0240 (4 August 2024).

They are contrasting model examples, not six newly identified bad filters. Their actual disagreement, both predictions, dates and selection rule are downloadable in `priority_filter_examples.csv`. The existing linked native-spectrum viewer retains original scans and signed raw/corrected/removed curves.

## Existing PMF triangulation

The existing `etad246_huber` result supplies five finite factor amplitudes for 231 of the 232 direct filters, with no date mismatches; only damaged ETAD-0158 is absent. The joined counts are 80 Bega, 81 Belg and 70 Kiremt. No new factor fit was performed.

Factor 3 correlates with disagreement at Spearman ρ 0.899 overall and 0.649/0.844/0.797 within Bega/Belg/Kiremt. Its overall correlation with FTIR EC is 0.900, versus 0.654 with HIPS, so spectral target coupling remains prominent. Factor 2 correlates with the signed 1800–2250 continuum at 0.812 overall and 0.473/0.824/0.560 within seasons. All five factors against all four targets are retained; no p-values or selective source labels are supplied.

These globally fitted factors are descriptive spectral amplitudes per volume, not calibrated chemical masses. They are not leakage-safe forward-prediction covariates and are not an independent validation of EC or source attribution. Within-season association does not additionally control year, loading, lot and volume.

## Checks: filter lot and the FTIR EC term

`research/ftir_hips_chem/workflows/run_seasonal_ml_checks.py` reruns three controls with the first analysis's own models, folds, windows and seed. Its outputs are in `output/tables/seasonal_ml_checks_20261103/`, and it is shown in the gallery card **Seasonal ML: checks for filter lot and FTIR EC**. Spectra come from the native arrays when present; otherwise from the per-filter curves exported to the gallery. With the gallery curves, all 24 saved D scores it re-derives agree within 0.001 µg/m³, on the same 1,897-channel window.

**Lot.** The native lot map covers 188 of the 232 filters. The remaining 44 (2024–25 and 2025–26) are filled from SPARTAN's filter table (`spartan_Filters.csv`, `ExternalLotId`): the 2024–25 filters are lot 251 and the 2025–26 Bega filters are lot 253. Recorded lots agree with that table wherever both exist.

| Cycle | Bega | Belg | Kiremt |
| --- | --- | --- | --- |
| 2022–23 (training) | 18 × lot 248 | 16 × 248, 19 × 251 | 38 × 251 |
| 2023–24 (primary test) | 37 × 251 | 34 × 251 | 26 × 251 |
| 2024–25 | 12 × 251 | 12 × 251 | 6 × 251 |
| 2025–26 | 14 × 253 | — | — |

The saved loading + lot classifier gets 0 % of 2023–24 Bega filters right, because it learned "lot 248 = Bega". Season accuracy when training on 2023–24, which is all lot 251:

| Model · spectra | 2022–23 lot 248 | 2022–23 lot 251 | 2024–25 lot 251 | 2025–26 lot 253 (Bega) |
| --- | ---: | ---: | ---: | ---: |
| RF · raw | 0.62 | 0.54 | 0.60 | 0.86 |
| RF · corrected | 0.56 | 0.79 | 0.80 | 0.86 |
| PCA-logistic · raw | 0.56 | 0.67 | 0.77 | 0.86 |
| PCA-logistic · corrected | 0.53 | 0.75 | 0.87 | 0.71 |

Within lot 251, corrected spectra still beat raw, so baselining does help season recognition. On the lot-248 filters, every representation falls to 0.53–0.62. The lot-253 test has only 14 filters, all Bega. Inside 2022–23 Belg, PCA-logistic tells lot 248 from lot 251 perfectly from raw spectra and from the removed spline component, and at 0.88 from corrected spectra. Lot and date coincide there (2 Feb–22 Mar vs 29 Mar–29 May), so this is indicative only.

**The EC line.** D correlates with FTIR EC at r = 0.95, 0.94 and 0.88 in the first three cycles. A line D ≈ a + b·FTIR EC, fitted on training filters only, is now scored on every saved D fold. It reaches RMSE 0.488 µg/m³ in 2023–24 and 0.640 in 2024–25. Against it:

| RMSE, µg/m³ | 2023–24 · D directly | 2023–24 · line + spectral residual | 2024–25 · D directly | 2024–25 · line + residual |
| --- | ---: | ---: | ---: | ---: |
| EC line only (no spectrum) | 0.488 | — | 0.640 | — |
| PCA-ridge · raw full | 0.487 | 0.401 | 0.559 | 0.580 |
| PCA-ridge · corrected | 0.671 | 0.407 | 0.472 | 0.440 |
| PLS · corrected | 0.512 | 0.444 | 0.475 | 0.378 |
| RF · raw full | 0.578 | 0.465 | 0.568 | 0.611 |
| RF · corrected | 0.635 | 0.427 | 0.456 | 0.496 |

- In 2023–24, no model of D directly beats the EC line. The raw-vs-corrected RMSE ranking is mostly a ranking of how well each representation recovers FTIR EC: RF predicts EC with R² 0.95 from raw spectra, but 0.78 from corrected.
- A spectral model of what the line misses does add information. Every model beats the line in 2023–24 (0.40–0.47 vs 0.488, up to 18 % lower). In 2024–25 the corrected-spectrum residual models do best (0.38–0.50 vs 0.640, up to 41 % lower), while the raw-spectrum ones barely help.

**Regions beyond EC.** The same grouped permutation is run on corrected-spectrum RF, PCA-ridge and PLS models (raw spectra: RF only), for D and for the residual. Mean RMSE increase on the 2023–24 test, µg/m³:

| Model | Target | 1560–1700 | 1700–1800 | 1800–2250 | 2800–3000 | 3000–3500 | 3500–4000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF | D | 0.03 | 0.01 | **1.11** | 0.00 | 0.01 | 0.01 |
| PCA-ridge | D | 0.01 | 0.01 | 0.00 | 1.92 | 1.23 | 0.01 |
| PLS | D | 0.65 | 0.21 | 0.01 | 1.05 | 0.39 | 0.01 |
| RF | residual | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 |
| PCA-ridge | residual | 0.01 | 0.01 | 0.00 | 0.29 | 0.00 | 0.13 |
| PLS | residual | 0.45 | 0.34 | 0.00 | 0.35 | 0.17 | 0.38 |

The 2024–25 test gives the same picture: 1800–2250 is at most 0.02 for every residual model. The regions to examine for baseline-related disagreement are therefore 3500–4000, 2800–3000 and 1560–1800 cm⁻¹, not 1800–2250. Shuffling a raw continuum block pushes linear and PLS models far off their training data (importances of 3–15 µg/m³), so raw-spectrum region tests are RF only.

**PMF.** Factor 3 correlates with FTIR EC (ρ 0.900) as strongly as with D (0.899). It is an EC factor, not specifically a disagreement factor.

## Gallery and reproducibility

The Baseline lab gains two graph cards: held-out ML performance/confusion, and spectral-region tests with a seasonal-spectrum view. Individual predictions are downloadable. Controls distinguish target, calendar, source cohort, temporal test, model family, representation, broad/fine/whole regions, permutation/refitting and held-out season. Descriptive median/IQR spectra use the full source cohort and are explicitly separated from validation. Methods and caveats are collapsed. The live Ethiopia season view now explicitly defaults to `belg_feb`, while preserving `dry_feb` as an option.

The served merged data contain 33 model specifications, 10 representations, 120 score rows, 7,746 model/filter prediction rows, 888 regional results and six separate seasonal performance rows. Those counts include repeated predictions from different models; they are not independent sample sizes. Shared source/input hashes, model/fold identities and held-out physical IDs are validated before add-ons merge. Original analysis JSON files remain unchanged downloads. PMF is download-only and is never silently supplied to the ML models.

Reproduce with the locked existing environment:

```bash
OPENBLAS_NUM_THREADS=1 uv run --locked --no-sync python research/ftir_hips_chem/workflows/run_seasonal_spectral_ml.py
OPENBLAS_NUM_THREADS=1 uv run --locked --no-sync python research/ftir_hips_chem/workflows/run_seasonal_spectral_ml_by_season.py
OPENBLAS_NUM_THREADS=1 uv run --locked --no-sync python research/ftir_hips_chem/workflows/run_spline_transition_sensitivity.py
uv run --locked --no-sync python research/ftir_hips_chem/workflows/run_seasonal_pmf_triangulation.py
uv run --locked --no-sync python research/ftir_hips_chem/workflows/plot_seasonal_spectral_ml.py
```

The lot / FTIR-EC checks, and a refresh of only their gallery file:

```bash
.venv/bin/python research/ftir_hips_chem/workflows/run_seasonal_ml_checks.py --lots "<SPARTAN HIPS bundle>/spartan_Filters.csv"
.venv/bin/python gallery/data/export_baseline_lab.py --seasonal-checks-only
```

Then run the existing Baseline lab exporter with its native-data directory configured. All steps are registered in `gallery/data/rebuild.py`; generated CSV/JSON/figures remain private artifacts, outside Git and public gallery builds. Receipts identify the base Git revision and exact executed worktree source hashes. This avoids implying that an uncommitted execution used only the base revision's bytes.

Scientific-core and export-contract validation: **34 focused cases passed** (12 core, 22 export). Five publication tests passed. Frontend checks include signed/null handling, season membership, confusion normalization, exact fold/cohort selection, unavailable regions, whole-spectrum placement and profile colours. The complete unrelated repository suite is not claimed green; previously identified missing rank-70/similarity exports remain outside this change.

## Research basis and decision

The companion [literature review](review-seasonal-spectral-ml-literature-2026-11-03.md) records eight primary sources with stable links. Particularly relevant are [Reggente et al. 2016](https://doi.org/10.5194/amt-9-441-2016) on year/site transfer and [Weakley et al. 2018](https://doi.org/10.1080/02786826.2018.1439571) on spectral-population classification followed by separate TOR-calibrated EC models. [Ghasemi and Tavakoli 2013](https://doi.org/10.1039/C3AY26338J) supports RF as a spectral calibration comparator, without universal superiority. [Strobl et al. 2008](https://doi.org/10.1186/1471-2105-9-307) and [Gregorutti et al. 2015](https://doi.org/10.1016/j.csda.2015.04.002) explain why correlated intervals need more careful interpretation than a single-channel importance ranking.

Any future spectral model of D should be reported next to the EC line, and its region tests run on what the line misses. With current data, use the seasonal and regional diagnostics to identify baseline sensitivity, prioritize concrete filters, and flag where a calibration transfers poorly. Preserve raw, corrected and removed representations and report the temporal/seasonal sensitivity. A future algorithm change should be judged against the existing blank, injection, replicate and source-calibration controls as well as these temporal tests. The present evidence does not warrant a country multiplier, three unconstrained seasonal correction factors, or selecting a baseline solely because it makes FTIR agree with HIPS.
