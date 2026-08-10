# Plot taxonomy

Earlier passes characterised exactly one family (crossplots) and counted inline
helper redefinitions. This is the full census: **3,942 plotting calls across 180
files**, forming **654 figure-producing cells**.

Figure recipes, by primitive combination:

| Recipe | Cells | What it is |
|---|---:|---|
| `plot+scatter` | 216 | the crossplot family (scatter + fit/1:1 line) |
| `plot` | 82 | time series |
| `scatter` | 50 | bare scatter, no fit |
| `fill_between+plot` | 49 | series with an uncertainty/spread band |
| `bar` / `barh` | 46 | summary/count bars |
| `boxplot` | 21 (+20 mixed) | distribution by season/site/wavelength |
| `hist` | 19 (+14 mixed) | distributions |
| `hexbin+plot+scatter` | 8 | density crossplot for the large IMPROVE pool |

Variant axes *within* those families -- these are the knobs that actually differ,
and the reason the panels look repetitive but are not interchangeable:

| Variant feature | Figures |
|---|---:|
| hand-built stats box | 231 |
| 1:1 reference line | 187 |
| a fitted regression | 185 |
| equal/square axes | 140 |
| colour-by-third-variable (loading, iron, biomass %) | 110 |
| uncertainty band | 104 |
| shared axes across a grid | 98 |
| log axis | 31 |
| seasonal shading | 28 |

## Findings

1. **Regression method is the substantive one.** *(Acted on 2026-07-28 --
   `calculate_regression_stats(..., errors_in_variables=True)` now returns
   `deming_slope`, `deming_intercept`, `deming_lambda`, and
   `slope_attenuation_pct`; `crossplot_on_axes` enables it automatically
   whenever `one_to_one and equal_axes` and prints both slopes in the stats box.
   Off by default in `calculate_regression_stats` so the 10+ existing callers'
   boxes are unchanged, and `crossplot_on_axes` had zero notebook adoption, so
   no published figure moved. Validated against a synthetic set with a known
   slope of 1.00 and equal error in both variables: OLS returns 0.859, Deming
   1.004. 10 tests.)* Of the **115** figures that draw
   a 1:1 line -- which asserts both axes measure the same quantity, so both carry
   error -- **114 fit with an OLS-family estimator only** (`linregress` 216 uses,
   `polyfit` 162, `OLS` 135) and just **one** also computes an
   errors-in-variables fit. The project already owns the right tool
   (`plotting.utils.deming`, plus a whole `addis_fabs_ec_deming` research dir),
   so this is under-adoption, not a missing capability.

   Measured on the real filter data, orthogonal (lambda=1) vs OLS:

   | Comparison | n | OLS | Deming | shift |
   |---|---:|---:|---:|---:|
   | ETAD Fabs/MAC vs FTIR EC | 190 | 1.898 | 2.379 | +25.3 % |
   | CHTS Fabs/MAC vs FTIR EC | 160 | 1.079 | 1.428 | +32.4 % |
   | USPA Fabs/MAC vs TOR EC | 114 | 0.651 | 0.890 | +36.9 % |
   | FTIR EC vs TOR EC (all sites) | 149-175 | ~1.00 | ~1.00 | 0.0 % |

   Median shift **24 %** across the ten site-pairs. The zero-shift rows are the
   control: where the two variables track each other almost exactly, OLS and
   Deming agree, so the 24 % is regression dilution and not an artefact of the
   estimator. **Caveat: lambda=1 (orthogonal) is an assumption here** -- the
   `Uncertainty` column in `unified_filter_dataset.pkl` is entirely null for
   `HIPS_Fabs`, `EC_ftir`, and `ChemSpec_EC_PM2.5`, so a data-driven lambda is
   not currently computable. The direction is robust; the exact magnitude is not.

2. **`EC_ftir` and `ChemSpec_EC_PM2.5` agree to about 1 %** -- slope 1.00,
   intercept 0.00, r2 = 0.99992, median ratio 1.000 at every site. For two
   nominally independent analytical methods (FTIR vs thermal-optical) that is not
   physically plausible; published FTIR-vs-TOR EC calibrations reach r2 ~0.8-0.95.
   Investigated 2026-07-28. What the data settles:

   - **Not a copy or join-fill.** 103 filters carry `EC_ftir` with **no** TOR EC
     at all (494 have both, 49 TOR-only). A copied column could not exist where
     the source is absent.
   - **Not an arithmetic restatement of the reported TOR value.** `EC_ftir` is
     stored at full float precision (14-16 decimals, 597 distinct values) while
     `ChemSpec_EC_PM2.5` is a reported figure rounded to 2 decimals (343 distinct
     values). The residual sd between them, 0.0254 ug/m3, is **8.8x** larger than
     the 0.0029 explainable by TOR's rounding, and **0 of 494** pairs are exactly
     equal.
   - So `EC_ftir` is an independently *computed* quantity that nonetheless tracks
     a 2-decimal TOR figure to ~1 %. That is the signature of a calibrated
     prediction whose training target was TOR EC.

   **Where the values come from.** The pickle's builder did exist in this repo:
   `research/filter_combine/` (`main.py` + three `load_filter_sample_data_*.py`
   variants), deleted in `aa9714b` on 2026-07-27. Recovered from git, `main.py`
   is a pure reshaper -- its FTIR branch is commented *"Load FTIR data (already
   in long format)"*, and it only tags `DataSource='FTIR'`, sets units, and
   concatenates. It never computes EC. `EC_ftir` arrives **already computed** in
   `Four_Sites_FTIR_data.v2.csv`, at full float precision
   (`MassLoading_ug = 6.50430770368977`).

   **So the calibration -- and whether these filters were in its training set --
   lives upstream of this repository entirely.** That remains the crux: in-sample
   fit would explain r2 = 0.9999 and would mean the 18 files crossplotting the
   pair report calibration fit rather than independent method agreement. It
   cannot be settled from this checkout. **Ask whoever produced
   `Four_Sites_FTIR_data.v2.csv`**; until then treat FTIR-EC-vs-TOR-EC panels as
   calibration diagnostics, not validation.

5. **`MDL` is populated where `Uncertainty` is not, and it pins lambda.**
   `Uncertainty` is empty for every parameter, but `MDL` is present for `EC_ftir`
   (597/750), `OC_ftir` (597/750), and `ChemSpec_EC_PM2.5` (1000/1043) -- and
   **absent for `HIPS_Fabs` (0/546)**. Using median MDL as the error proxy gives
   lambda = (0.4143/0.0578)^2 = **51** for the TOR-vs-FTIR EC pair: FTIR's own
   stated detection limit is 7.2x TOR's, so nearly all the error is on the FTIR
   axis, lambda >> 1, and OLS-of-y-on-x is already close to correct **for that
   pair**. That independently explains the 0.0 % rows in the table above rather
   than leaving them as a coincidence.

   The consequence is specific and actionable: for the Fabs-vs-EC comparisons --
   the ones showing the 22-37 % attenuation -- `HIPS_Fabs` carries no MDL, so
   lambda = 1 (orthogonal) is the only available assumption and the magnitude of
   that correction stays uncertain. **Obtaining HIPS Fabs uncertainties would
   firm up the largest correction in the estate.** Pass them via
   `calculate_regression_stats(..., sigma_x=, sigma_y=)` when available.

3. **MAC reference lines are healthy** -- worth stating because it looked like a
   risk. 36 files draw MAC=10 only (matching `config.MAC_VALUE`), **zero** draw 6
   alone, and the 14 files carrying both are the deliberate side-by-side protocol
   comparisons (`ftir_19_mac_effect_on_calibrations`,
   `ftir_22_figures_under_both_protocols`). The 10/20 pairs in the IMPROVE
   notebooks are bracketing reference lines, not a rival convention. The open
   MAC 6-vs-10 fork is being handled explicitly rather than drifting.

4. **Cosmetic drift confirms the panels are hand-built.** *(dpi acted on
   2026-07-28: `config.SAVEFIG_DPI` (200) and `config.FIGURE_DPI` (110) are now
   applied by `apply_default_style()`. 462 savefig calls pin a dpi explicitly and
   still win; the 224 that never did were silently getting matplotlib's 100 and
   now match the pinned majority. Resolution only -- no data or layout change.)* R-squared is spelled
   five different ways in stats boxes (`R2` 352, `r2` 167, escaped `R\u00b2` 55,
   `R2` 28, `R^2` 15), and `savefig` uses **13 distinct dpi values** from 120 to
   1000 (160/150/140 most common). Harmless individually; together they are the
   signature of 231 independently written stats boxes.

The two unresolved measurement questions are tracked in
[`open-items.md`](open-items.md): the provenance of `EC_ftir`, and HIPS Fabs
uncertainties needed to pin Deming lambda.
