# Actual PLS calibration-domain screen: native Addis spectra

**2 November 2026 — existing data, frozen historical model.**

The original calibration exports are available, so the actual fitted PLS score
space can now be examined. This run adds a model-specific diagnostic to the
previous descriptive spectral screens. It does not replace the current public
FTIR EC product or establish Ethiopian EC accuracy.

**Result:** 80 of 253 native Addis PM2.5 physical-filter mean spectra exceed at
least one of the calibration set's marginal 95th-percentile distance/residual
references. Eighteen exceed a marginal 99th-percentile reference. However,
ETAD-0037 and both ETAD-0139 scans remain below both references despite the
previously documented acquisition concerns. Calibration proximity therefore
cannot replace scan-quality checks.

## What was reconstructed and checked

| Item | Verified result |
|---|---|
| Model | Historical EC calibration reconstructed from the archived R exports |
| Calibration physical filters | 906 unique filters, with unique analysis IDs |
| Calibration sites | 140 IMPROVE sites |
| Calibration collection dates | 17 May 2022–30 April 2023 |
| Selected PLS components | 31, inherited from the export; not retuned |
| Spectral representation | Full raw absorbance; 2,722 descending channels; `scale=False` |
| Approximate spectral extent | 3998.423–500.124 cm⁻¹ |
| Maximum Python/R fitted-response difference | 0.0000507763 in exported response units |
| Median absolute Python/R difference | 0.00000912924 in exported response units |
| Native Addis input | 319 scans of 296 physical filters |
| Native physical-filter types | 253 PM2.5, 37 field blanks, 6 PM10 |
| Filters with repeated scans | 22 |

The workflow reuses `load_current_pls_model`, `score_metric`, `project_scores`,
`mahalanobis_distance_squared`, and `spectral_q_residual` from
[pls_transfer.py](../research/ftir_hips_chem/scripts/pls_transfer.py).
The export's four file hashes are pinned in the new workflow: a different
calibration cannot silently inherit the same representation claim.

The preprocessing contract is the archived July transfer workflow, which
explicitly treats the source and native target as raw full-grid absorbance.
The new run reads the original `ETAD_FTIR_spectra.csv`, verifies that native and
model wavenumbers are identical after casting to the model export's float32
precision, and rejects different channel counts, ordering, representations,
or grid values. The largest double-versus-float32 wavenumber difference is
0.000121953 cm⁻¹. No interpolation, clipping, baseline correction, normalization,
or target-label adjustment is applied.

These checks establish compatibility with the reviewed archived representation.
They do not independently reconstruct instrument sample/reference beams or
prove that this historical calibration generated the current public EC values.

All original scans remain in a scan ledger. Physical IDs and metadata are
checked before arithmetic means of repeated raw spectra are formed. Repeated
target scans never become additional calibration observations. Existing
canonical exclusion flags are retained without removing their records; all
253 PM2.5 physical filters are included in the primary descriptive counts.

## The two diagnostics answer different questions

The source model returns retained score coordinates \(t\) for each spectrum.
The existing helper uses the score sum-of-squares matrix:

\[
h=t^\mathsf{T}(T^\mathsf{T}T)^{-1}t.
\]

This run labels that output **score leverage**. It separately reports the
conventional sample-covariance normalization:

\[
D^2=(n_{\mathrm{train}}-1)h=905h.
\]

The training scores are centered; the new tests verify the result against a
direct sample-covariance calculation. Score distance asks whether retained
predictive coordinates are unusually large relative to this model's training
population.

**Q** is the sum of squared residuals after reconstructing the full centered
spectrum from the retained PLS scores and loadings. It measures spectral
structure the retained model does not reconstruct. Because this is the full
raw grid, Q can reflect substrate/background differences as well as aerosol
composition. It is not a chemical identification or an EC error estimate.

The use of actual PLS score space follows the calibration-transfer approach in
[Reggente, Dillner & Takahama (2016)](https://doi.org/10.5194/amt-9-441-2016).
The distinction from the previous PCA screen is documented in the
[research and experiment note](research-transfer-and-optics-2026-11-02.md).

## Reference levels and counts

| Reference | D² | Q, summed absorbance² |
|---|---:|---:|
| Calibration-set marginal 95th percentile | 93.3798 | 0.00381926 |
| Calibration-set marginal 99th percentile | 315.325 | 0.00754055 |

These are **descriptive in-sample references**. They are not externally
validated error limits, acceptance thresholds, or a joint 95%/99% confidence
region. Combining two marginal thresholds by their union does not preserve
their individual nominal coverage. For example, 62/906 calibration filters
exceed at least one marginal 95th-percentile reference, while 17/906 exceed
at least one marginal 99th-percentile reference.

| Native PM2.5 physical-filter means | Number | Fraction of 253 |
|---|---:|---:|
| D² above calibration marginal 95th percentile | 75 | 29.6% |
| Q above calibration marginal 95th percentile | 53 | 20.9% |
| Either above marginal 95th percentile | 80 | 31.6% |
| D² above calibration marginal 99th percentile | 0 | 0.0% |
| Q above calibration marginal 99th percentile | 18 | 7.1% |
| Either above marginal 99th percentile | 18 | 7.1% |

The separate field-blank result is 5/37 above either marginal 95th-percentile
reference and 3/37 above either marginal 99th-percentile reference. Blank and
PM10 rows are retained for diagnosis; passing a reference screen does not
validate EC prediction for those sample types.

The lot breakdown is informative but confounded:

| Native PM2.5 lot | Filters | Above either marginal 95th reference | Above either marginal 99th reference |
|---|---:|---:|---:|
| 248 | 35 | 31 | 16 |
| 251 | 160 | 34 | 0 |
| Unknown | 58 | 15 | 2 |

This is a descriptive association with the full raw-spectrum/model geometry.
It does not establish a manufacturing effect: lot, collection dates,
composition, and acquisition conditions are not separated by this table.

## Filter shortlist, with original scan context

The rankings identify filters to inspect, not filters to exclude. The CSV
shortlist keeps the physical-filter mean, scan IDs/counts, and minimum/maximum
scan-level D² and Q. All original scan values remain available in the full
scan ledger.

### Five highest Q values among physical PM2.5 filters

| Rank | Filter | Original scan IDs | D² | Q |
|---|---|---|---:|---:|
| 1 | ETAD-0308 | 4343 | 239.932 | 0.014520 |
| 2 | ETAD-0006 | 3205 | 170.595 | 0.012647 |
| 3 | ETAD-0008 | 3207 | 201.489 | 0.012575 |
| 4 | ETAD-0004 | 3203 | 187.391 | 0.012335 |
| 5 | ETAD-0022 | 385 | 178.865 | 0.011135 |

These five filters each have one native scan. Their scan and physical-mean
metrics are therefore identical.

### Five highest D² values among physical PM2.5 filters

| Rank | Filter | Original scan IDs | D² of mean | Q of mean |
|---|---|---|---:|---:|
| 1 | ETAD-0308 | 4343 | 239.932 | 0.014520 |
| 2 | ETAD-0021 | 384 | 218.407 | 0.008772 |
| 3 | ETAD-0008 | 3207 | 201.489 | 0.012575 |
| 4 | ETAD-0020 | 383 | 201.196 | 0.008236 |
| 5 | ETAD-0070 | 900, 915, 932 | 201.008 | 0.003603 |

ETAD-0070 merits repeat-level review:

| ETAD-0070 representation | D² | Q |
|---|---:|---:|
| Scan 900 | 676.638 | 0.012058 |
| Scan 915 | 114.789 | 0.002772 |
| Scan 932 | 130.839 | 0.003433 |
| Mean of all three raw scans | 201.008 | 0.003603 |

The mean is not a summary of the three diagnostic distances: projecting a mean
is linear, but squared distance and Q are nonlinear. The marked difference
between scan 900 and the other two scans is a reason to examine acquisition
records and signed spectral differences. This analysis does not identify
which acquisition is correct, and it makes no scan-selection change.

### Previously flagged acquisitions can remain inside the reference range

| Filter/scan | D² | Q | Above either marginal 95th reference? |
|---|---:|---:|---|
| ETAD-0037 / 406 | 67.708 | 0.002429 | No |
| ETAD-0139 / 1626 | 71.927 | 0.002300 | No |
| ETAD-0139 / 1668 | 34.795 | 0.001267 | No |
| ETAD-0139 / physical mean | 34.265 | 0.001030 | No |
| ETAD-0158 / 1995 | 66.735 | 0.003356 | No |

The documented nearly flat ETAD-0037 and ETAD-0139 acquisition, and the recorded
physical-damage concern for ETAD-0158, therefore remain separate quality
questions. A spectrum can be close to a calibration population while the
measurement fails to represent its intended sample.

## Source fit is not held-out source validation

The reconstructed calibration's training RMSE is **9.24235** and training MAE
is **5.30182** in exported response units, which the archived workflow treats as
µg/filter loading. Those values are calculated from fitted training responses.

The additional historical `EC_rmsep.csv` reports **CV = 71.74466** and
**adjusted CV = 68.15058** at 31 components. This run preserves that row and its
hash, but the supplied files contain no per-filter held-out predictions or
original fold membership. Consequently, the historical aggregate cannot be
reconstructed, matched to independent residuals, or substituted for a new
held-out validation result here. The large distinction from training RMSE is
another reason to avoid presenting the training fit as predictive accuracy.

No target EC concentrations, HIPS residual correction, or recalibrated EC
product is generated by this workflow. Source-domain validation of alternative
pipelines remains a separate experiment.

## Reproduction and output files

Run with the verified existing FTIR export directory and native four-file
export directory:

```bash
uv run python research/ftir_hips_chem/workflows/run_calibration_domain_screen.py \
  --ftir-dir "$AETHMODULAR_FTIR_SPECTRA_DIR" \
  --native-dir "$AETHMODULAR_ETAD_NATIVE_DIR"
```

The workflow writes to
`research/ftir_hips_chem/output/tables/calibration_domain_20261102/`:

- `manifest.json`: model identity, reconstruction error, conventions,
  representation checks, and input/code hashes captured at the start of the run.
- `source_training_domain.csv`: calibration reference metrics and explicitly
  labelled training fitted responses/residuals.
- `native_scan_domain.csv`: every original scan and its diagnostic values.
- `native_physical_filter_domain.csv`: physical means and complete scan-ID lists.
- `native_pm25_domain_shortlist.csv`: separate top-five Q and D² rankings with
  scan-level ranges; overlap between the rankings is preserved.
- `training_reference_thresholds.csv` and `domain_summary.csv`: references and
  sample-type-specific counts.
- `historical_exported_CV_summary.csv`: historical aggregate only, with its
  unverified fold scheme explicitly labelled.

The six focused tests in
[test_calibration_domain_screen.py](../tests/test_calibration_domain_screen.py)
passed. They check covariance normalization, source-only metric construction,
grid/product rejection, frozen-export identity, noncontiguous repeat averaging,
and rejection of ambiguous physical-filter IDs. These are numerical and
provenance checks, not a validation of Ethiopian EC accuracy.
