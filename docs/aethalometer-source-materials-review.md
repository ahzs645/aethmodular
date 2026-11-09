# Aethalometer source-material review

Reviewed 2026-11-09. The supplied EC–BC presentation and JPL source archive add useful methods, figure provenance and a saved source-model sensitivity table. They do not supply the missing complete Addis BC inputs or establish a fully reproducible final-paper run. The curated, machine-readable record is [ec_bc_jpl_review.json](../research/ftir_hips_chem/workflows/profiles/ec_bc_jpl_review.json).

## Source identity and recoverable evidence

| Material | Verified SHA256 | Recoverable content |
|---|---|---|
| `EC_BC.pptx`, 2,376,593 bytes | `956d9d49ab69c3f14868c350c075bbc33ed0fca1fa5cf234da736033be7fb674` | 17 visually reviewed slides, 36 raster images and one editable regression table. No embedded workbooks/chart data, populated notes, serial manifest or exact matched dates. |
| `JPL-main.zip`, 94,837,746 bytes | `d303eb2ff6decaa1ca8ac2c1dd411a23f53a1771e1a5c217389c225eba39b3d3` | 44 files, including 24 Python files, 11 notebooks and 3 R Markdown files. Saved tables, plot outputs and some numeric Plotly traces are recoverable. The complete Central/Jacros BC input pickles and external `aethpy` implementation are absent. |

The archive's entries are stamped 2026-06-11. Its comment is `b4526955b7ab2c02e4d3530678207ce42de084dc`, a commit-shaped marker that has **not** been verified against a remote repository. Notebook cell references below are **zero-based JSON cell indices**, not execution counts. No archived notebook or fitting code was executed in this review.

## Loading correction: candidates, not an adopted calibration

Slides 2–4 give the dual-spot relationship

\[
k=\frac{BC_2-BC_1}{BC_2 ATN_1-BC_1 ATN_2},\qquad
BC_c=\frac{BC_1}{1-k ATN_1}.
\]

Slide 5 contains three different fixed-parameter choices:

| Candidate | k | Stated objective or provenance |
|---|---:|---|
| EC-residual objective | 0.011723 | Minimize correlation between residuals labelled EC−BC and IR ATN1; bounds 0.004–0.016. |
| BC–ATN objective | 0.011846 | Heading and plot indicate absolute BC–ATN correlation; body text also mentions residuals, so the exact objective remains ambiguous. Same stated bounds. |
| Median reported IR K | 0.0129 | Median marked on the IR K histogram. Slide 6 explicitly labels its monthly diurnal comparison `median(k)`. |

Slide 9 says “optimized k” without identifying which optimization objective produced that row. These values must remain separately named sensitivity candidates. An analytic gain curve `1/(1−k·ATN1)` explains their consequences but is not a measured-data correction. Plot it only on its positive-denominator domain. Slide 4's preference against a correction larger than threefold is not an established QC threshold.

A rebuild requires minute timestamps, device/deployment identity, BC1/BC2, ATN1/ATN2, reported K, flow, tape/session and status records, plus the matched FTIR inputs, model version and eligibility rules. Correct minutes before averaging: a nonlinear correction of mean inputs is generally different from the mean of corrected observations. Select and evaluate k on separated time or tape blocks before claiming transferability.

The existing Quality page already provides flow/tape/environment checks and minute K versus closure K*. In `quality/minute/ETAD-JACROS.json`, the five IR attenuation-bin medians have compensated spot-2/reported-BCc ratios of 1.0 and nearly equal K/K*. This checks internal algebra at the saved precision. It does not prove independent accuracy or exclude a shared flow bias. The deck's historical “best practice” conclusion does not supersede this finer-resolution evidence.

## Historical EC–BC fits

The following values transcribe slide 9. The **y variable is FTIR-derived EC; x is the specified BC variant**. Intercepts are in µg/m³.

| BC variant | Slope | Reported r | R² | Intercept | Residual r with IR ATN1 |
|---|---:|---:|---:|---:|---:|
| Reported IR BCc | 0.60 | 0.83 | 0.69 | 0.77 | 0.06 |
| IR BC1 | 1.03 | 0.92 | 0.84 | 0.25 | 0.28 |
| IR BC2 | 0.94 | 0.91 | 0.82 | 0.31 | 0.26 |
| IR BCc with optimized k | 0.65 | 0.94 | 0.88 | 0.14 | 0.01 |

Slide 17 states 140 Addis and 52 JPL points. It does not establish exact membership of every panel. Addis plots span approximately late 2022 to May 2024; the JPL comparison spans approximately June–November 2023. These visually read ranges are not exact sampling endpoints. The counts are historical comparison counts, not the current gallery's 18,280 valid Jacros hours or its current matched-filter cohort.

Exact decoded RGBA pixel matches link slide 8's BC1 and BC2 images to `k_exploration.ipynb` cells 27/28, executions 635/636. Their plotting function uses `statsmodels.OLS` with an intercept. The notebook hash is `eadb3db8bad545c75cfde605446b9a72270d91fcfa8df84bfd9bc03557c44ac0`. The BC1 and BC2 pixel hashes are, respectively, `eca316a1e2ed62f5076e5c5ba8f9d7df6251c9c8fa10a17e65434a06f6f3d7fd` and `d2013800b92d68f7338ccc50136838e3023e01b9a66fcc0175627130c0e38466`. This establishes OLS provenance for those two linked figures only. Slide 6's manual/instrument fit and slide 12's MAC fit also match saved images, but their source cells 33/32 have null execution counts. No fit was rerun.

The remaining fits are described as **linear fits reported in the deck**, without assigning an unverified estimator. Exact cohort IDs, coverage rules, FTIR model version and training/evaluation split remain missing. Rescaling BC with a fitted slope/intercept predicts FTIR-derived EC. It must not replace the measured BC series in the paper comparison or be described as validation against independent thermal EC.

Other deck evidence supplies useful questions, not new calibration constants: slide 10 compares HIPS optical BC with aethalometer BC and FTIR EC; slides 11–12 explore MAC values 8.4 and 12.8 without a complete assignment rule; slide 16 pools four sites. The pooled R² of 0.91 does not establish site-specific performance, especially given JPL's reported R² of 0.30 on slide 13.

## Archived optical sensitivity and the published 5.8%

`data_cleaning_pipeline.py` visibly calls fossil/biomass exponents **1.0/1.8** after hourly resampling. `Figures.ipynb` cells 215–218 also contain nine exponent pairs, including the final paper's **1.0/2.0** pair. It would be incorrect to describe the entire archive as using only 1.0/1.8.

Cell 216, execution 131, computes

\[
100\frac{\sum BC_{bb}}{\sum BC_{IR}}
\]

after removing paired missing/nonfinite components. It then saves 2.5th/97.5th bootstrap percentiles from 5,000 independent row resamples, with no seed shown. Cell 218, execution 132, reproduces the 18 values in a site pivot. All 18 reported estimates and intervals are preserved in the curated profile and checked against both outputs. `Figures.ipynb` has SHA256 `1cc78ec634b6b390a391783c9b0fbd5d58b3ffa72cbadcd4eaeeadca13331bd0`.

At 1.0/1.8, the archive reports 7.7% at each site. At 1.0/2.0, it reports **5.8% at each site**, with saved intervals 5.6–5.9%. Across the full grid, Jacros spans 0.5–20.3% and Central 1.2–17.3%. The grid therefore provides a concrete display of end-member sensitivity.

This supplies an archive-specific aggregation definition consistent with the paper's discussion value. It does not establish the final paper's exact input bytes, time window, paired coverage, wavelength pair, clipping or correction sequence. Keep these as **archived model results**, separate from the current observation export. The narrow saved intervals neither preserve time dependence nor include end-member uncertainty; they are not a new validated uncertainty estimate.

The archive does include a separate `calibration.py.sa_aethalometer_model`, with defaults IR/Blue 880/470 nm, C=1.3, exponents 1.0/2.0, integer arithmetic and clipping. The pipeline instead invokes the absent external `aethpy.sourceApportionment` implementation. That unresolved code linkage prevents treating the bundled defaults as the final paper's settings.

## Calendar, identity and implementation findings

`Figures.ipynb` cell 22 uses October–January Bega, February–May Belg and June–September Kiremt, matching `belg_feb`. Cell 303 explicitly selects Meskel noon-to-noon in `Africa/Addis_Ababa`; cell 300 selects the Hidar calendar day on November 21, 2022; November 22, 2023; and November 21, 2024. These support display-window provenance. They do not settle the exact exclusions in final-paper climatologies: cell 126 constructs an event-excluded frame but calls its plotting function with the unfiltered frame. The event “other September/November” baselines in cells 303/300 likewise omit the month restriction. Do not copy those baselines.

The archived `rafael_processing/jacros_data_processing_rafael_cmu.py` assigns MA350-0238. `all_data_processing_rafael_cmu.py` assigns MA350-0225, a useful Central identity lead that still needs a deployment manifest, dates and matching source files. Neither script supplies the missing complete measurements.

The calibration audit compared 47 distinct top-level function names, using the last definition where a name was duplicated: **43 are AST-identical** to current `src/external/calibration.py`, ignoring comments/formatting. The differences are `set_up_environment`, `histogram_plot`, `read_AE33` and `sa_aethalometer_model`. The archive's AE33 reader has serial-header extraction worth evaluating against real fixtures. Its source-apportionment upper-clipping diagnostic uses a percentage threshold on a fractional quantity; the repository already has the corrected fractional threshold. Another archive file, `AAE_method_comparison.py`, uses 450 in a Blue/IR denominator despite declaring Blue as 470 nm. These are reasons to review narrow differences rather than vendor the archive.

Some notebook outputs do contain numeric observations: JPL fire traces in June–August 2025 and T640/BAM comparisons in August–November 2022. They are recoverable plotted subsets with their own unresolved QC and coverage, not missing Addis BC records. No private access links or raw external code are included in the curated profile.

## Where the findings belong

| Finding | Current integration point | Defensible next use |
|---|---|---|
| Three fixed-k candidates and correction gain | Addis study methods evidence; existing Quality page | Explain model sensitivity; obtain minute inputs and a held-out evaluation before applying a correction. |
| Dual-spot, flow and tape diagnostics | `gallery/data/export_aeth_quality_minute.py`; `quality/MinuteCharts.tsx` | Reuse the current checks and distinguish internal consistency from accuracy. |
| Historical EC–BC coefficients | Curated source profile; `src/data/processors/site_corrections.py` | Display provenance and fit direction; report whether an eligible correction was actually applied. |
| Archived source-model sensitivity | Curated source profile; `src/analysis/bc/source_model.py` | Display the saved grid separately; use explicit end members, optical basis, paired coverage and clipping diagnostics for any new run. |
| Named calendar and festival windows | `config.py`; `export_addis_study.py` | Retain explicit calendar/window metadata and current missingness. Do not infer missing 2024 observations. |
| Final study reproduction | `paper_comparison.py`; `compare_shlipak_2026.py` | Resolve deployment manifests, final inputs and correction lineage before upgrading descriptive comparison to reproducibility. |
