# Closeout reproduction — 7 November 2026

## Purpose and evidence boundary

This run repairs the source evaluation and preserves the existing target-comparison evidence. It uses the recovered October decision bundle, the original production VIBES background, and a new, versioned regeneration of the two continuum variants. It does not introduce new Ethiopian reference measurements, tune a new model on the outer scoreboard, or promote numerical VIBES v2 into the saved studies.

The primary comparison is fixed at 18 PLS components and five site-grouped outer folds. Each physical source filter has one held-out prediction per candidate. All candidate error and composition-subset summaries use the same scored physical filters; unavailable predictions remain visible in the ledger. The complete-pipeline comparison includes raw, spline, production VIBES, deflated VIBES, anchored VIBES and anchored EPO. A separate matched-resolution control compares pair-averaged anchored spectra with and without EPO.

Execution results and the exact tested source revision belong in the release receipt. A prepared input bundle, a passing synthetic test or a browser success is not itself a completed scientific rerun.

## Recovered inputs

| Input | Identity and scope |
|---|---|
| Original decision bundle | 18 data files plus its original manifest; 12,393 IMPROVE, 233 Addis and 3,517 network rows. All 18 file hashes match the manifest. |
| Decision manifest SHA-256 | `121557f8bf470816e02ddeafed8b7acf91b1c2cd50a59eab08088b7af169db48` |
| Original production background | 87 blank IDs, rank 70, 2,722 channels; recovered bytes match both the original background manifest and the parent checksum list. |
| Background SHA-256 | `8b717b6b988465083e0e102f887da4f61c63aed86cce43be640e52dece2165e3` |
| Pinned public archive | `data-82d701248232`; SHA-256 `82d701248232b1975ea2e452bc1d682877e811103a2aa1040aff9d4adc989051`. |
| Frozen continuum display export | Generated 4 November 2026; SHA-256 `90b8e16bfcb002db7e6bc7553d190c94f462753182fcb80d5bacc1c65a2111c9`. Its historical target values are retained. |
| Pinned filter/exclusion map | `filters.json`, SHA-256 `a043679b673e82cdc6b539c6aeab38f97855f142ca7f1901a61bc66ff416089d`; preserved with the frozen export because coverage and target views use this map. |

The background stores its grid as float64 and the decision bundle as float32. They are equal after casting the background grid to the decision grid's storage dtype. The check requires that exact equality; no interpolation or nearby-grid tolerance is used.

The old November corrected arrays were not located. Their absence is resolved for this run by regenerating them from the verified originals. The historical transfer/intercept tables remain unavailable, so the new export replaces only the source scoreboard on the frozen public JSON. It retains the historical target figures and date and withholds unverified EPO sweep/simulation derivatives.

## Cohort rule fixed before examining new scores

The source population has 12,393 unique physical filters across 157 sites. There are 12,387 finite TOR observations below 150 µg/filter before the recorded spectrum-quality rule.

The original decision workbook explicitly removes one corrupt raw spectrum from **every comparison**, not just the raw candidate. The decision metadata marks this row `raw_qc_ok=false`. The new calibration and scoring stages honor that pre-existing rule consistently, leaving **12,386 eligible source filters** before any candidate prediction failure. The rule is documented in [the original workbook builder](../research/ftir_hips_chem/workflows/build_decision_workbook_20261022.py); it is not a new exclusion chosen after looking at the scores.

All 12,393 rows remain in saved identity vectors, predictions and ledgers. Site folds are fixed from the complete source population before label eligibility or QC is applied. Each training fold, including EPO projection fitting, applies the fixed QC mask and its own training-label eligibility. The held-out QC flag is retained as a reason for not including that filter in the primary score.

The new eligibility rule must be shown when comparing these results with historical November scalars that did not apply the recorded QC flag. Differences between the old and new tables cannot all be attributed to EPO's leakage correction: the numerical runtime, regenerated continuum spectra and restored QC rule also have explicit versions.

## Numerical procedure

The continuum regeneration uses the original `vibes_baseline_matrix` adapter, pinball loss (`PB`), `tau=0.1`, initial `maxiter=10000` and `retry_failed=True`. The original retry policy remains in effect. Deflation and PTFE anchoring use the existing construction. Newton, GPU fitting, interpolation, clipping and imputation are outside this rerun.

Preparation writes immutable inputs, copied source code, installed package versions and solver settings. Run/resume verifies those identities and reuses only matching, complete checkpoints. Collection requires all expected row/variant jobs exactly once. A failed fit remains failed; default collection stops rather than filling the result. A completed collection receipt binds the corrected arrays and diagnostics to the bundle.

The prepared bundle contains **16,143 spectra × two variants = 32,286 fit jobs**, with content hash `3583bdb74db1ac2d111d9cdfb6365c1e2a692c545fb8459e507fe9af4633362d`. Its runner SHA-256 is `cb7e689475b31d567558782ad046e7cc1e18b1d117baa6f9277ad584fb1f7ec0`. This is a newly recorded numerical environment, not a claim of byte identity with the original Colab execution.

Collection completed on 7 November: **32,286 expected and collected fits, zero recorded fit failures**, six corrected arrays and 270 verified checkpoints. The complete continuum archive has SHA-256 `cd44e62160e3717afcfc13f793eea294852e139983c25139ed31cf32ac4d4853` and size 980,487,622 bytes. Its three supplied binary parts reassemble to those exact bytes; the parts manifest records their ordered sizes and hashes. Reassemble before extracting the archive beneath `research/ftir_hips_chem/output/tables` into an empty run destination:

```bash
cat Aethmodular_Continuum_Run_20261107.tar.gz.part01 \
    Aethmodular_Continuum_Run_20261107.tar.gz.part02 \
    Aethmodular_Continuum_Run_20261107.tar.gz.part03 \
    > Aethmodular_Continuum_Run_20261107.tar.gz
sha256sum Aethmodular_Continuum_Run_20261107.tar.gz
```

The archive's restoration prefix is `closeout_20261107/continuum_v1/`. Its earlier checkpoint package is superseded. The research-results ZIP includes the part manifest, portable Python reconstruction instructions and the separate worker-interruption history.

## Reproduction commands

Restore the input archive beneath `research/ftir_hips_chem/output/tables`. It creates `closeout_20261107/decision_bundle`, the original `background_model.npz` and `frozen_public/continuum_baseline.json`. Verify its input manifest before use. From the matching source checkout:

```bash
uv sync --locked --python 3.13.5 --extra vibes
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export AETHMODULAR_CLOSEOUT=research/ftir_hips_chem/output/tables/closeout_20261107

uv run --no-sync python research/ftir_hips_chem/workflows/vibes_continuum_20261102.py prepare \
  --source "$AETHMODULAR_CLOSEOUT/decision_bundle" \
  --background-model "$AETHMODULAR_CLOSEOUT/background_model.npz" \
  --output "$AETHMODULAR_CLOSEOUT/continuum_v1"

uv run --no-sync python research/ftir_hips_chem/workflows/vibes_continuum_20261102.py run \
  --output "$AETHMODULAR_CLOSEOUT/continuum_v1" --workers 4

uv run --no-sync python research/ftir_hips_chem/workflows/vibes_continuum_20261102.py collect \
  --output "$AETHMODULAR_CLOSEOUT/continuum_v1"

uv run --no-sync python research/ftir_hips_chem/workflows/run_transfer_diagnosis_20261102.py \
  --bundle "$AETHMODULAR_CLOSEOUT/decision_bundle" \
  --continuum-output "$AETHMODULAR_CLOSEOUT/continuum_v1" \
  --vibes-model "$AETHMODULAR_CLOSEOUT/continuum_v1/bundle/background_model.npz" \
  --output "$AETHMODULAR_CLOSEOUT/baseline_site_cv_v1" \
  --stages fit --models raw spline vibes vibes_deflate vibes_anchorhp --k-max 18

uv run --no-sync python research/ftir_hips_chem/workflows/run_intercept_fixes_20261103.py \
  --bundle "$AETHMODULAR_CLOSEOUT/decision_bundle" \
  --continuum-output "$AETHMODULAR_CLOSEOUT/continuum_v1" \
  --output "$AETHMODULAR_CLOSEOUT/epo_site_cv_v2" \
  --stages fit --models base epo --k-max 18

uv run --no-sync python research/ftir_hips_chem/workflows/run_hips_response_20261103.py \
  --bundle "$AETHMODULAR_CLOSEOUT/decision_bundle" \
  --baseline-output "$AETHMODULAR_CLOSEOUT/baseline_site_cv_v1" \
  --epo-output "$AETHMODULAR_CLOSEOUT/epo_site_cv_v2" \
  --scores-output "$AETHMODULAR_CLOSEOUT/source_scores_v2" --stages scores

uv run --no-sync python research/ftir_hips_chem/workflows/verify_closeout_source_20261107.py \
  --bundle "$AETHMODULAR_CLOSEOUT/decision_bundle" \
  --continuum-output "$AETHMODULAR_CLOSEOUT/continuum_v1" \
  --baseline-output "$AETHMODULAR_CLOSEOUT/baseline_site_cv_v1" \
  --epo-output "$AETHMODULAR_CLOSEOUT/epo_site_cv_v2" \
  --scores-output "$AETHMODULAR_CLOSEOUT/source_scores_v2" \
  --output "$AETHMODULAR_CLOSEOUT/source_acceptance.json"

uv run --no-sync aeth gallery rebuild --only continuum-closeout-review closeout-figure-ledger
```

Each scientific output destination must be new. Do not rerun preparation or collection over a completed bundle. To resume fitting, use `run` on the existing bundle; it validates and reuses its checkpoints. Preserve earlier artifacts when changing the code, settings or population and use a new versioned destination.

The recorded execution initially used eight workers. After 22,800 jobs had been checkpointed, the runtime killed one worker (`SIGKILL`); the process exited with `TerminatedWorkerError`. It was resumed with four workers to reduce memory pressure, using the same verified input, code, package and solver identities. The interrupted batch is recomputed, and completed batches are reused. Worker count controls parallel execution, not the individual fit. The attempt history is preserved separately from numerical fit-failure counts. Monitor the process exit as well as the progress file; only the complete collection receipt authorizes downstream fitting.

The fit stages also save separate all-source Addis/network predictions. Those new predictions are not substituted into the historical optical-response tables by the source-only export. The acceptance step reconstructs every saved EPO fold and the separate all-source target models, then refits only EPO fold 0 after changing held-out TOR and nuisance chemistry. It holds spectra, splits, QC and training records fixed, records actual numerical differences under predeclared tolerances, and labels the altered-label scores as artificial integrity checks. The figure ledger separately reconciles historical populations; it does not refit their models.

## Reading the release

The primary source table compares complete candidate pipelines. Raw and VIBES variants use 2,722 channels spanning approximately 500.12–3998.42 cm⁻¹. The saved spline representation has 2,002 channels spanning 1425.80–3998.42 cm⁻¹. The EPO workflow averages adjacent full-grid channels to 1,361. The additional `anchorhp_bin2_base` versus `epo_bin2` control holds that resolution fixed and reports its own population identity. It is not silently substituted for the full-grid anchored candidate.

The [figure-population ledger](gallery-closeout-figure-populations-2026-11-07.md) explains the 233-filter Addis export, 172 optical pairs, 170 seasonal complete cases, 232 historical response cases, and the separate 16-parent ambient injection study. Seasonal uncertainty endpoints are transformed on the correct log scale using saved rounded summaries. These display corrections do not refit historical outcomes or silently harmonize their populations.

The source receipt binds metrics and physical-filter ledgers to the saved prediction hashes and identical site folds. MedAE, true MAE, RMSE, mean/median signed bias, P90/P95, composition subsets and coverage have distinct definitions. The gallery carries compact counts/hashes; the complete ledgers remain in the scientific package.

The public review combines the pinned historical archive and a separately hashed source overlay. Its receipt identifies both inputs. The base archive checksum does not describe the modified served export. Browser acceptance tests both an unchanged historical snapshot, where EPO remains withheld, and the actual corrected overlay used for the figure package.

Absolute Addis EC accuracy remains unresolved. A completed source evaluation, a transformed seasonal interval, a successful browser journey and a loaded timeline export each support their own stated conclusions; none creates an independent Addis TOR measurement.
