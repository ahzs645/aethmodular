# Baseline lab: existing-data investigation, 3 November 2026

The new **Research diagnostics → Baseline lab** page collects the completed
baseline investigation in seven reusable charts. It uses existing measurements
only. Methods, provenance and downloads are collapsed; the chart controls retain
units, population and validation choices. The page is part of the existing
private Research section and is omitted from public gallery builds.

Open the full gallery at `#tab=research&aep=baseline-lab`. A checkout without the
optional SPARTAN network export can use
`#tab=research&aep=baseline-lab&net=focus`.

## What the page contains

| Chart | Completed evidence available |
| --- | --- |
| Filters and optical disagreement | Five spectral sites; direct HIPS or the documented HIPS extension; baseline sensitivity, band contrasts and reference-envelope diagnostics; linked physical-filter selection. |
| Raw, removed and signed spectra | 908 inspectable physical filters, including 296 with original native Addis spectra; seven native correction methods; native repeats and physical-filter means; signed values and explicit unsupported gaps. |
| Baseline controls | 37 held-out physical blank parents; known-lot transfer; 10 input-selected ambient injection parents; three injection amplitudes on blank parents; separate constructed controls; absolute and relative repeat differences. |
| Spectral associations | Current portable and native populations kept separate; signed partial-rank association and population size in hover. |
| Chronological evaluation | Earlier-quarter training and later-filter testing; all eligible, direct-HIPS and known-lot populations; pooled and individual quarters. |
| Ethiopia | Dates, recorded lots, seasons, overlapping periods and fixed-caliper matching between Addis and Bishoftu; unavailable same-lot matching remains unavailable. |
| Matched recalibration | Eleven source-trained preprocessing/refit arms; 905-filter primary and 906-filter artifact sensitivity; site-held-out and chronological scores; separate predictions on 37 Addis blanks. |

There are 865 filters in the packed five-site spectrum export and 43 additional
native-only identities (37 field blanks and six undated PM2.5 filters). The
current ledger contains **747 eligible spectral/optical pairs**, including
**689 direct-HIPS pairs**. Native-only records do not acquire invented optical
measurements or eligibility.

## Findings that change the working diagnosis

### 1. The actual native spline preserves the specified narrow test peak

The native implementation reproduces the saved R result within 6.1 × 10⁻⁷
absorbance units. On actual blank parents it recovers about **99.93%** of the
specified narrow Gaussian centered at 1617 cm⁻¹. On the ten separately selected
ambient parents it recovers **99.826–99.961%**. Earlier losses under generic
smoothers are not evidence that this native spline removes the same fraction.
These controls do not support a general narrow-peak-destruction explanation.

The broad Gaussian is a different result: the native spline retains about
**12%**. Local blank VIBES also removes much of it. These are known-signal
recovery fractions for defined shapes, not estimates of actual ambient EC loss.
See [the native-control report](review-native-baseline-controls-2026-11-03.md).

### 2. Cleaner blanks, numerical convergence and useful calibration are separate checks

Local rank-8 VIBES reduces median held-out blank RMS from **0.000551** to
**0.000136 AU**, but loses its advantage when a filter lot is absent from its
small local training bank. Neither local VIBES arm is the existing global
rank-70 product.

Raising neutral arPLS's iteration ceiling from 50 to 500 resolves the recorded
51 native-scan convergence failures. It does not resolve instability: one weak
injection still yields **42.30-fold** apparent recovery. Finite, converged and
successful output flags remain distinct, and the graph retains failed outcomes.

### 3. A known source artifact can reverse the apparent baseline ranking

All eleven calibration arms were refitted against the same existing source TOR
measurements, with identical site folds and source-only component selection.
The already documented TONT1 offset scan is flagged equally in the primary
905-filter comparison; the complete 906-filter run remains a separate sensitivity.

| Existing source experiment | Source cross-validation RMSE, µg/filter | Median prediction on 37 Addis blanks, µg/filter |
| --- | ---: | ---: |
| Raw, 500–4000 cm⁻¹ | 16.904 | +6.306 |
| Raw, 1425–4000 cm⁻¹ | 17.652 | +7.041 |
| Spline df6/4 and matched refit | 18.787 | −0.367 |
| Neutral default and matched refit | 19.738 | −1.431 |

The raw-full minus spline RMSE difference is −1.883 µg/filter, with paired
site-bootstrap 95% interval **−5.891 to +1.732**. That interval does not establish
a firm source-performance winner. The blank check shows why a lower source RMSE
alone is insufficient for choosing an Ethiopian pipeline. Removing the
1580–1650 cm⁻¹ window from the spline changes source RMSE by only +0.113
µg/filter, while the removed continuum alone remains predictive.
See [the matched calibration report](review-existing-calibration-2026-11-03.md).

### 4. Ethiopia can be investigated more narrowly with the current records

Matching uses direct HIPS, positive loading/volume, unique physical identities,
and prespecified calipers: collection dates within 30 days, optical BC ratio at
most 1.35, PM2.5 ratio at most 1.5, and volume ratio at most 1.2. FTIR EC never
selects the matches. The one-to-one assignment maximizes cardinality before
minimizing the normalized covariate distance.

The result is **eight Addis–Bishoftu pairs**. Their site-specific median
FTIR-minus-optical differences are **−1.687** and **−1.345 µg/m³**, respectively.
These are descriptive medians of matched site populations, not a calibrated
correction factor. Requiring a recorded common lot yields **zero matches**;
later Addis lot metadata are missing. The available Bishoftu spectral period
does not cover Kiremt.

The documented ETAD-0037 near-flat acquisition, ETAD-0139 discordant scans
1626/1668, and ETAD-0158 physical-hole note remain inspectable. No canonical
exclusion or published concentration is changed. A useful next action with
existing resources is to reconcile these acquisition and filter records, retain
recorded lot/season strata, and compare complete preprocessing-plus-calibration
pipelines through the existing controls. A country-wide multiplier is not
identified by these results.

## Display and statistical safeguards

- Signed absorbance, negative predictions and negative optical differences are retained.
- The spline's forced-zero tail is hidden outside each scan's supported range.
  A physical-filter mean uses the intersection of its scans' support. Neutral
  method masks remain gaps; the browser never bridges null channels or converts
  them to measured zeros. The numerical experiments retain their stated original
  scoring-window convention.
- Optical disagreement is not independently measured Ethiopian EC error.
  Its response contains FTIR-derived EC, so spectral prediction of disagreement
  is not causal proof of a baseline problem.
- Saved model and matching scores keep their exported populations when Samples
  changes. Live filter plots follow shared selection and exclusion controls.
- Blank children do not leak into background training, reused amplitudes do not
  inflate independent-parent counts, and repeated acquisitions do not create
  additional independent filters.
- Leave-lot-out checks are blank-residual checks. Injection schemes are the
  37-blank, ten-ambient and one-constructed-parent experiments, kept separate.
- The source calibration is the historical smoke-selected cohort, dominated by
  known lot 251; it does not contain a confirmed lot-248 reference population.

## Reproduction

The exporter is a registered gallery data step. The portable analysis needs the
repository's packed spectra and current chemistry snapshot. Native controls and
source recalibration require the existing external input files, configured via
environment variables; see the linked reports for their commands.

```bash
uv run python research/ftir_hips_chem/workflows/run_portable_baseline_review.py
uv run python gallery/data/export_baseline_lab.py
npm ci --prefix gallery/app
npm --prefix gallery/app run dev
```

Set `AETHMODULAR_ETAD_NATIVE_DIR` locally to include original native traces.
The exporter consumes optional completed tables from
`baseline_existing_data_20261103` and `calibration_existing_data_20261103`.
Missing inputs remain absent, rather than being replaced with synthetic observations.
Generated graph data are under `gallery/app/public/data/research/baseline_lab/`.
The data bundle preserves JSON/CSV evidence and lazy target spectra; source
preprocessing caches containing raw calibration matrices are not exported.

## Verification

Targeted scientific checks cover physical identity, source/target membership,
held-out blank parents, independent chronological fitting, signed/null values,
native numerical fidelity, convergence flags and matching without FTIR-outcome
selection. Frontend checks cover cohort separation, units, null spectral gaps,
repeat display and shared chart interaction standards. The page was inspected
in a browser at desktop and 390-pixel mobile widths.

The full gallery check passes its index, TypeScript, 104 Python checks and public
visibility checks. Its Node suite retains seven known failures caused by absent
gitignored rank-70/similarity exports in this checkout. Those unrelated data
products were not fabricated to make the suite green. No new test failure was
introduced by this section.
