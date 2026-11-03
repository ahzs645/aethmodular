# Existing-data calibration controls: 3 November 2026

## What was completed

Existing source files were recovered and used to **recalibrate every preprocessing
arm against the same IMPROVE thermal-optical EC measurements**. This is a new,
matched experiment using existing measurements; it does not require additional
sampling or funding. It goes beyond comparing the visual appearance of corrected
Addis spectra.

The main finding is a tradeoff. Raw spectra have lower point estimates of
source-domain cross-validation error in this cohort, while the spline models
transfer substantially better to the existing Addis field blanks. The uncertainty
interval for raw versus spline source RMSE includes zero. Neither a cleaner blank
nor closer agreement with HIPS establishes an accurate Addis EC calibration.
The isolated 1580–1650 cm⁻¹ window does not explain the predictive information by
itself, and the continuum removed by the spline retains substantial predictive
covariance with source EC.

**Scope:** these are historical smoke-selected calibration experiments. They are
not the published SPARTAN model, the later full-pool raw recreation, or the current
rank-70 VIBES calibration. No production EC values were replaced.

## Recovered inputs and source cohort

The recovered `apps_data.npz` contains 935 raw spectra on 2,722 channels spanning
3998.423–500.124 cm⁻¹. The official `EC_samples.csv` selects 906 unique physical
filters, at 140 IMPROVE sites, dated 17 May 2022–30 April 2023. EC and OC labels
join all 906 rows. As documented in
[the SPARTAN recreation audit](spartan-ftir-ec-recreation.md), this is the old
smoke-selected export, not the full operational calibration population.

The reconstruction of its original 31-component R model agrees with the saved R
fitted values to a maximum absolute difference of **0.0000508 µg/filter**. This
verifies the recovered matrix, labels, ordering, and PLS implementation; it is a
reproduction check, not held-out validation.

The archive's 29 other rows remain in `archive_selection_ledger.csv`, with the
reason that they are not listed in the official calibration manifest. They were
not newly rejected using the outcomes of this experiment.

### Previously documented source artifact: apply the same flag to every arm

TONT1, 19 November 2022, analysis **164549**, physical filter **2008808**, has an
approximately +3 absorbance-unit offset. Its mean raw absorbance in the shared
1425–4000 cm⁻¹ window is 3.129. This was already documented and excluded from raw
fits on 20 October in `docs/spartan-ftir-ec-recreation.md`; it is not a new discovery.

It materially changes the apparent method ranking:

| Source experiment | n | Raw full-range RMSE | Raw shared-window RMSE | Spline RMSE |
|---|---:|---:|---:|---:|
| Historical 906, including the known offset scan | 906 | 63.423 | 75.658 | 18.779 |
| Same original fitted models, scored only on the other 905 | 905 | 16.224 | 19.875 | 18.789 |
| Refit every arm with that previously documented scan flagged | 905 | **16.904** | **17.652** | **18.787** |

All entries are µg/filter against source TOR EC. The middle row only changes
evaluation membership; the last row also removes the flagged scan from model
fitting and inner component selection. Their difference must not be conflated.
The primary experiment uses the last row's policy for **every** arm. Original
measurements and all 906-run outputs are retained separately. The 905 common
filters have exactly the same outer site-fold assignments in both runs.

The primary cohort has 886 confirmed lot-251 records and 19 without a matching
catalog record. It contains no confirmed lot-248 reference cohort. Thus a
lot-248-versus-lot-251 source-calibration comparison is not identifiable from
this bounded recovered calibration.

## Matched protocol

- Five outer folds hold out entire IMPROVE sites, using seed 20261103. Every
  source filter has exactly one outer prediction; no site crosses a fold.
- Each outer training fold selects 1–40 PLS components using five inner folds
  grouped by site. The existing repository rule selects the earliest local
  minimum within one standard error of the global minimum. It uses source TOR
  labels only.
- All arms use identical source membership and folds. Baselines are fit
  independently to each spectrum. Neither Addis spectra nor HIPS values select
  components, baseline parameters, or a winning model.
- A separate chronological check trains on 807 filters before 1 November 2022
  and evaluates 98 later filters. It holds time out; it does not also hold all
  sites out. It is reported separately.
- Final models are selected and fit on all 905 source filters, then applied to
  the original 319 native Addis scans. Correction precedes averaging by the
  296 physical-filter identifiers. Native sampling volume converts predicted
  loading to concentration only when the volume is finite and positive.
- Target wavenumbers are aligned to the archive's float32 grid before all
  corrections; the maximum displacement is 0.000122 cm⁻¹. This is far below
  the approximately 1.286 cm⁻¹ channel spacing.
- Signed spectra and negative predictions are retained. ETAD-0139 has an
  additional, explicitly conditional scan-1668-only sensitivity; the original
  two-scan average is retained as the primary record.

The neutral arm is the existing `neutral_baseline.py` implementation:
`pspline_arpls(lam=1e6)` with the pybaselines 1.2.1 defaults. Its fit uses the
native grid at or above 700 cm⁻¹, excluding 1100–1300 cm⁻¹, before evaluation
on the common 1425–4000 cm⁻¹ window. Finite output does not guarantee that this
iterative method converged. Its exact stopping-status audit is recorded in
`neutral_default_convergence.csv` and `diagnostics_receipt.json`; this arm must
be described as the fixed default implementation, not a proven converged
neutral baseline. **126/905 source filters and 51/319 native target scans** did
not reach the default tolerance of 0.001. The latter include 44/274 PM2.5 scans,
6/39 field-blank scans, and 1/6 PM10 scans; these are scan counts, not independent
physical-filter counts. Recomputing these diagnostics reproduced the calibration
corrections exactly, with maximum difference zero. The separate native-control
audit examines a larger iteration allowance.

## Source-reference results

These are pooled outer-fold predictions from **905 filters at 140 sites**.
RMSE and MAE are µg/filter. Predictive R² is 1 − SSE/SST; it is not Pearson r².

| Matched preprocessing and recalibration | Channels | RMSE | MAE | Predictive R² |
|---|---:|---:|---:|---:|
| Raw, 500–4000 cm⁻¹ | 2722 | 16.904 | 6.690 | 0.574 |
| Raw, shared 1425–4000 cm⁻¹ | 2002 | 17.652 | 6.575 | 0.535 |
| Spline df1=6, df2=4 | 2002 | 18.787 | 7.105 | 0.474 |
| Removed spline continuum only | 2002 | 17.134 | 6.341 | 0.562 |
| Spline df1=6, df2=3 | 2002 | 18.733 | 7.083 | 0.477 |
| Spline df1=6, df2=6 | 2002 | 19.777 | 7.466 | 0.417 |
| Neutral arPLS, repository default | 2002 | 19.738 | 7.483 | 0.419 |
| Raw, excluding 1580–1650 cm⁻¹ | 1947 | 17.710 | 6.615 | 0.532 |
| Spline, excluding 1580–1650 cm⁻¹ | 1947 | 18.900 | 7.128 | 0.467 |
| Spline, only 1580–1650 cm⁻¹ | 55 | 20.729 | 7.859 | 0.359 |
| Second derivative, 31-point/degree-3 Savitzky–Golay | 2002 | 18.726 | 6.877 | 0.477 |

The paired bootstrap resamples 140 sites, with 2,000 repetitions, using fixed
out-of-fold predictions. Against the spline df6/4 reference:

- Raw full-range ΔRMSE = **−1.883**, 95% interval **−5.891 to +1.732**.
- Raw shared-window ΔRMSE = **−1.135**, interval **−3.886 to +1.370**.
- Removing 1580–1650 from the spline ΔRMSE = **+0.113**, interval
  **−0.073 to +0.325**.
- Neutral default ΔRMSE = **+0.950**, interval **+0.273 to +1.704**, subject
  to the numerical stopping-status qualification above.
- Changing the spline's second-segment df from 4 to 3 changes RMSE by only
  **−0.054**, interval **−0.199 to +0.097**.

These exploratory comparisons do not provide an untouched external validation,
and the bootstrap does not refit models. They do show why an apparent dramatic
baseline improvement from the unflagged 906 run would be misleading.

The broad removed component being predictive does **not** identify that component
as pure soot absorption. Loading, composition, substrate, and scattering may
covary with source EC. Conversely, it cannot be assumed to be disposable noise.
The small effect of deleting the 1600 cm⁻¹ window does not prove that the band is
absent; correlated information can be carried elsewhere in the spectrum.

In the chronological check, raw full-range, raw shared-window, and spline RMSE
are 9.568, 9.121, and 11.204 µg/filter, respectively. Their mean prediction biases
are −3.981, −4.038, and −1.221 µg/filter. Lower RMSE and smaller signed bias do
not rank these arms identically.

OC/EC-stratified diagnostics are also saved. Source errors are much larger in
the low-OC/EC and high-OC/EC tails than in the middle 5–10 range. For example,
raw full-range RMSE is 31.552 for OC/EC <5 (122 filters), 7.361 for 5–10 (530),
and 20.702 for ≥10 (253). These strata use measured source thermal carbon;
they are descriptive, not target-label proxies.

## Existing Addis controls and transfer consequences

### Field blanks expose a different part of the problem

All 37 physical field blanks are outside the source calibration. Their model
predictions are compared with nominal zero loading, without dividing by an
air volume and without adding assumed zero labels to training.

| Source-trained arm | Median blank prediction, µg/filter | RMS about nominal zero, µg/filter |
|---|---:|---:|
| Raw full-range | +6.306 | 6.755 |
| Raw shared-window | +7.041 | 7.397 |
| Spline df6/4 | −0.367 | 0.472 |
| Spline df6/3 | −0.338 | 0.471 |
| Spline df6/6 | +0.858 | 0.908 |
| Neutral default | −1.431 | 2.049 |
| Removed continuum only | +3.856 | 4.895 |

These are model background-transfer diagnostics. Nominal zero is not a measured
TOR result for each blank. They nevertheless demonstrate why source prediction
error alone is an inadequate criterion for choosing a transferable pipeline.
The spline's better blank behavior is useful evidence even though it does not
beat raw spectra on every source metric.

### Experimental target predictions are not new SPARTAN EC products

The current chemistry ledger supplies **246** eligible Addis optical pairs,
including **232 direct HIPS** pairs and 14 public-HIPS extensions. Dates,
physical-filter IDs, PM2.5 type, and native volume are checked. This comparison
keeps its current-cohort identity; it is not pooled with the older 189-pair
frozen unified cohort.

Median experimental FTIR minus selected HIPS BC on the 246 pairs is:

| Arm | Median difference, µg/m³ |
|---|---:|
| Raw full-range | −2.031 |
| Raw shared-window | −0.346 |
| Spline df6/4 | −3.688 |
| Neutral default | −3.774 |
| Removed continuum only | −0.179 |
| Spline excluding 1580–1650 | −3.347 |
| Second derivative | −2.610 |

The source-cohort selection and component-selection rule matter. These smoke
models are not expected to reproduce the published SPARTAN product. In
particular, the closer optical agreement of a raw-shared or continuum-only model
does not make it an EC accuracy winner; HIPS was never used as a thermal target.
The results show that changing the baseline and refitting can strongly alter
Addis predictions while source scores remain comparatively similar.

For ETAD-0139, the spline experiment gives 0.557 µg/m³ from the original average
and 1.176 µg/m³ when only the previously documented good scan 1668 is used.
Those values are an experiment-specific sensitivity, not replacements for its
published EC value. Analogous rows for every arm are retained in
`target_predictions.csv`. The shared target sensitivity omitting the three
previously documented problem filters, 0037/0139/0158, contains 243 pairs and
does not materially remove the differences between these arms.

### Source support is descriptive

PLS score distance T² and reconstruction residual Q are calculated using only
the source-trained final models and compared with the fitted-source empirical
95th percentiles. Among the 253 native PM2.5 physical spectra, 53 exceed either
threshold for raw full-range, 113 for raw shared-window, and 3 for spline df6/4.
These are calibration-specific geometric diagnostics, not calibrated rejection
probabilities. The models have different selected component counts. Their
envelopes cannot establish target EC accuracy or be interpreted as proof that
the more strongly corrected spectrum is scientifically correct.

## Files and reproducibility

Workflows:

- [Run matched calibration controls](../research/ftir_hips_chem/workflows/run_existing_calibration_controls.py).
- [Summarize flags, blanks, and source support](../research/ftir_hips_chem/workflows/summarize_existing_calibration_controls.py).

Set `FTIR_CALIBRATION_INPUT_DIR` to the recovered FTIR input tree,
`ETAD_NATIVE_INPUT_DIR` to the native Addis input folder, and make the optional
pybaselines dependency available. If that dependency is installed separately,
`BASELINE_CONTROL_DEPS` identifies its directory. No source download location
is embedded in either workflow.

```bash
PYTHONPATH="$BASELINE_CONTROL_DEPS${PYTHONPATH:+:$PYTHONPATH}" uv run python research/ftir_hips_chem/workflows/run_existing_calibration_controls.py \
  --ftir-dir "$FTIR_CALIBRATION_INPUT_DIR" --native-dir "$ETAD_NATIVE_INPUT_DIR" --source-policy documented_qc

PYTHONPATH="$BASELINE_CONTROL_DEPS${PYTHONPATH:+:$PYTHONPATH}" uv run python research/ftir_hips_chem/workflows/summarize_existing_calibration_controls.py \
  --ftir-dir "$FTIR_CALIBRATION_INPUT_DIR" --native-dir "$ETAD_NATIVE_INPUT_DIR" \
  --run-dir research/ftir_hips_chem/output/tables/calibration_existing_data_20261103
```

To regenerate the retained comparison cohort, run the first command with
`--source-policy all_original` and `--output-dir` pointing to its separate
`historical906_all_scans` subdirectory before running the summary workflow.

Derived tables are in
`research/ftir_hips_chem/output/tables/calibration_existing_data_20261103/`.
The 906-filter audit is confined to `historical906_all_scans/`. The primary
905 cohort, historical audit, chronological split, and optical diagnostics must
not be pooled in the gallery.

The primary `metrics.csv` uses these exact evaluation values:

| Evaluation value | Meaning | n |
|---|---|---:|
| `site_grouped_outer_fold` | One of five held-site folds | varies |
| `site_grouped_pooled` | All independent outer predictions | 905 |
| `time_block_2022_11_01` | Later dates after training on earlier dates | 98 |

Other key outputs are `heldout_predictions.csv`, `component_selection.csv`,
`paired_site_bootstrap.csv`, `source_stratum_metrics.csv`,
`source_flag_sensitivity.csv`, `target_predictions.csv`,
`target_optical_pairs.csv`, `target_optical_summary.csv`,
`target_blank_predictions.csv`, `target_source_support.csv`, and the two
machine-readable summary/receipt JSON files. Source filenames and SHA-256
hashes are recorded in `summary.json`; raw inputs and private download URLs
are not committed.

Reproduction uses Python 3.12.14, NumPy 2.4.2, pandas 2.3.3, SciPy 1.17.0,
scikit-learn 1.8.0, and pybaselines 1.2.1. Complete source hashes are recorded
in the ignored run receipt. The targeted tests protect physical averaging,
the conditional scan sensitivity, date matching, missing-volume handling,
retention of negative predictions, and the distinction between correlation
and predictive accuracy.

The bounded inventory check did not find newer Addis lot metadata: the available
ETAD source folder still has the June raw/metadata files and July lot map, while
the separate SPARTAN pulls have ETBI, USPA, INDH, and CHTS folders. No lot was
inferred from a sampling date for the recent filters with unresolved lot IDs.
