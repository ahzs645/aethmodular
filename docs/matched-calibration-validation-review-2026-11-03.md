# Independent review of the matched calibration experiment

Review date: 3 November 2026. Scope: the completed eleven-arm matched
calibration run, its optical join, the added 500-iteration arPLS arm, and the
seasonal extension. This independent review did not refit the models or change
the original data. The initial source-results comparison below concerns the
eleven-arm run; the final section separately checks the completed extensions.

## Review conclusion

The completed source-calibration outputs can be reused for seasonal diagnostics.
The reviewed implementation preserves physical-filter identity, keeps source
sites out of their own outer validation folds, and selects PLS components inside
the outer training data. Every compared arm uses the same source cohort and
folds. No target optical outcome enters model fitting or component selection.

The important remaining limits are scientific: the source population is an old
smoke-selected IMPROVE cohort, season and lot coverage are uneven, and Addis
lacks an independent thermal-optical EC reference in this experiment. Lower
source prediction error, cleaner field blanks, and closer HIPS agreement are
different forms of evidence and should be presented separately.

## Reviewed files and verification

The calculation is implemented in
[run_existing_calibration_controls.py](../research/ftir_hips_chem/workflows/run_existing_calibration_controls.py)
and its blank, convergence, and support audit in
[summarize_existing_calibration_controls.py](../research/ftir_hips_chem/workflows/summarize_existing_calibration_controls.py).
The detailed scientific account is
[review-existing-calibration-2026-11-03.md](review-existing-calibration-2026-11-03.md).

Generated evidence is the `calibration_existing_data_20261103` run under
`research/ftir_hips_chem/output/tables/`. These generated files are kept outside
version control; the report bundle includes the relevant tables and receipts.

The review verified:

- All ten calibration and native input-file hashes match `summary.json`, both
  in the existing local copies and in the newly downloaded files from the
  supplied Drive folder.
- The calculation workflow hash matches the recorded workflow hash.
- All eleven arms contain exactly 905 distinct source analysis IDs and 905
  distinct physical-filter IDs, with one held-out prediction per source filter.
- Each source site occurs in one outer fold, and every source analysis has the
  same fold assignment across all arms.
- The 905 common filters also retain the same outer-fold assignments in the
  separate original-906 sensitivity run.
- The existing two focused calibration tests pass. They check that correlation
  is not reported as predictive accuracy, and exercise scan aggregation,
  missing volume, negative predictions, date mismatches, and the separately
  labeled ETAD-0139 scan sensitivity.
- Frozen optical values, sources, dates, and eligibility flags reproduce from
  the active public chemistry export, as detailed below.

These checks support reuse of the completed calculation. They do not establish
target-domain EC accuracy or a universally superior preprocessing method.
The fresh-file hashes and independent arithmetic checks are recorded in
`drive_seasonal_verification_20261103/provenance_receipt.json` and
`drive_seasonal_verification_20261103/independent_optical_lineage_receipt.json`
under the generated tables directory.

## Source cohort, labels, and units

The raw archive contains 935 spectra. The official historical EC manifest
selects 906 unique physical filters at 140 IMPROVE sites, dated 17 May 2022
through 30 April 2023. The primary matched experiment applies the previously
documented TONT1 offset flag to every arm, leaving 905 source filters. The
original 906 records remain in the flagged ledger and a separate sensitivity
run. The archive's other 29 rows are not in the official manifest; their absence
is not a new outcome-based exclusion.

The source response is the archived measured TOR EC loading, in µg/filter,
following the calibration contract documented in
[spartan-ftir-ec-recreation.md](spartan-ftir-ec-recreation.md). Its renamed
`TOR_EC_loading_ug` values equal the exported `MatchedValue` values exactly.
Reconstruction of the original 31-component R model differs from its saved
fitted values by at most 0.0000508 µg/filter. This checks matrix ordering,
response reconstruction, and the numerical PLS implementation; fitted-value
agreement is not held-out validation.

The primary source cohort has 886 confirmed lot-251 filters and 19 without a
matching lot record. It has no confirmed lot-248 reference subset. A calibration
comparison between source lots 248 and 251 is therefore unavailable here.

There are 319 native Addis scans representing 296 physical filters: 253 PM2.5,
37 field blanks, and 6 PM10. Corrections precede averaging of repeated scans.
Concentration predictions divide loading by the native sampling volume only
when that volume is finite and positive. Field-blank predictions are reported
as loadings; nominal zero is a diagnostic assumption, not a measured TOR label.

## Validation boundaries and statistical interpretation

Five outer folds hold out entire source sites. Five inner folds, also grouped
by site, select from 1–40 PLS components using the repository's first-major-
minimum rule. Preprocessing in these eleven arms is deterministic and fitted
independently to each spectrum, so caching those transformations before the
split does not transfer fitted population information across folds.

A separate chronological check trains on 807 filters collected before
1 November 2022 and tests on the 98 later filters. It does not also separate all
sites. It should remain labeled as a time holdout, not combined site-and-time
validation.

Raw full-range spectra have a source outer-fold RMSE of 16.904 µg/filter versus
18.787 for the df6/4 spline. However, the paired site-bootstrap difference is
−1.883 µg/filter with a 95% interval of −5.891 to +1.732. The interval spans zero.
This is not evidence for a decisive raw-spectrum winner. The bootstrap
resamples sites using the fixed held-out predictions; it does not refit the
models and does not account for all model-selection uncertainty.

The raw-full model predicts a median 6.306 µg/filter on the 37 Addis field blanks,
whereas the spline predicts −0.367. This exposes a transfer tradeoff that source
error alone would miss. A near-zero blank response also does not prove retention
of real broad aerosol signal.

The eleven preprocessing arms are exploratory comparisons on the same source
data. Reporting the best observed arm as if it had passed an untouched external
validation would overstate the evidence. Predictive R² is 1 − SSE/SST; Pearson
r² describes association and can remain high despite substantial bias.

## Independent optical-join verification

The active file `gallery/app/public/data/spartan_reference.json` has:

- SHA256: `11d545419f0d56259304fefa71c313f2f3cb53125cb8c87066424e927c1c92be`.
- 5,069 physical-filter rows.
- Export timestamp: `2026-10-28T20:11:53.339295+00:00`.

The frozen `target_optical_pairs.csv` has SHA256
`fd4fdd8e6e708144e7fedd7e709b89e5c8c37a9c7038c7a7c62d33e595e48bfb`.
All 3,267 rows were checked: eleven arms, each containing 296 physical-filter
records plus the separately labeled ETAD-0139 alternative.

| Reconstructed field | Result |
|---|---|
| Existing public FTIR EC | Identical values and missingness; maximum absolute difference 0 |
| HIPS-derived optical BC | Identical missingness; maximum absolute difference 8.88 × 10⁻¹⁶ µg/m³ |
| Public sampling date | Every date and missing-date position matches |
| Optical source label | Every direct-HIPS, extension, and missing-source record matches |
| Existing pair-eligibility flag | Every flag and missing-record position matches |
| Experiment pair eligibility | Exactly reproduced from existing eligibility, PM2.5 type, valid native volume, and equal native/public date |

The numerical tolerance used for the comparison was 10⁻¹². HIPS absorption is
exported in Mm⁻¹ and divided by the declared MAC of 10 m²/g to produce the
optical-equivalent BC value in µg/m³. This is not an independent EC reference.
The public `EC (ChemSpec FTIR)` and `EC (FTIR)` fields both map to `EC PM2.5`;
they must not be counted as separate EC measurements.

Eligibility intentionally follows the existing public paired ledger, including
its requirement for a finite existing public FTIR EC value. Consequently, the
232 matched direct-HIPS pairs are a fixed comparison cohort, not a claim that
every native filter with HIPS is included. Nonpositive measurements remain
eligible when otherwise valid; this is not a positivity-selected analysis.

### Fresh Drive HIPS source check

The newly retrieved `SPARTAN_HIPS_Batch1-51.v2.csv` from the supplied data folder
has SHA256
`4276eaa23b9be587a0c240d7faba53526858443c2b45110481f5c2e02eccf4fb`.
It contains all 232 eligible physical filters after the canonical
`base_filter_id` normalization and restriction to PM2.5. Each has exactly one
HIPS source record: no duplicate-filter averaging is needed for this cohort.
All 232 HIPS sampling dates equal the native sampling dates, and the HIPS and
native sampling volumes agree exactly.

Comparison with unrounded source `Fabs / 10` gives a maximum absolute difference
of 4.986 × 10⁻⁸ µg/m³ and a mean absolute difference of 2.630 × 10⁻⁸ µg/m³.
The difference is explained by the explicit six-decimal rounding of laboratory
fields in
[export_spartan_reference.py](../gallery/data/export_spartan_reference.py).
After applying that documented `round(Fabs, 6) / 10` conversion, all 232 values
match the frozen optical pairs at tolerance 10⁻¹²; the maximum remaining
difference is 8.88 × 10⁻¹⁶ µg/m³. Thus the optical endpoint of this experiment
traces to the supplied Drive HIPS file, with only the recorded export rounding.

The older `FilterBased_ChemSpecPM25_ETAD.csv` in the supplied folder identifies
itself as version 2.5, updated 17 April 2025. It is a separate FTIR-derived
public-product snapshot. It neither supersedes the more recent export used
here nor provides an independent TOR EC validation target.

## Season coverage and identifiable comparisons

Season names must come from `config.resolve_seasons` and
`config.season_for_month`, with the convention recorded beside the result.
Under the canonical `dry_feb` convention, Dry is October–February, Belg is
March–May, and Kiremt is June–September. The registered `belg_feb` sensitivity
moves February from Dry to Belg.

| Native PM2.5 coverage under `dry_feb` | Dry | Belg | Kiremt | Undated |
|---|---:|---:|---:|---:|
| All physical PM2.5 filters | 115 | 61 | 71 | 6 |
| Matched direct-HIPS pairs | 102 | 60 | 70 | 0 |

Under `belg_feb`, the matched direct-HIPS counts become 81 Dry, 81 Belg, and
70 Kiremt: 21 February pairs move. The six undated filters remain in the ledger
without an assigned season. The completed seasonal extension uses `belg_feb`
as its primary local calendar and reports `dry_feb` as the explicit sensitivity;
the tables above identify their own convention rather than silently mixing it
with that primary choice.

| Lot among matched direct-HIPS pairs, `dry_feb` | Dry | Belg | Kiremt |
|---|---:|---:|---:|
| 248 | 27 | 7 | 0 |
| 251 | 46 | 44 | 64 |
| Unknown | 29 | 9 | 6 |

There is no lot-248 Kiremt comparison. Whole-cohort lot contrasts therefore
cannot identify a lot effect independently of season. The lot-251-only
sensitivity has observations in all three seasons, but its date coverage still
needs to be shown. Most direct-HIPS observations are from 2023–2024; later years
do not provide equally complete season coverage.

The source IMPROVE population is also temporally uneven: shared March–May bins
contain 24 filters, June–September 637, and October–February 244. These are
calendar bins for a US source population. They should not be described as
observed Ethiopian rainy seasons or used to imply equally supported seasonal
calibrations.

## Checks of the completed extensions

The 500-iteration arPLS arm retains exactly the original 905 source analysis IDs,
TOR responses, and outer-fold assignments. Its 140 sites each occur in a single
outer fold. Source outer-fold RMSE is 19.745 µg/filter, compared with 19.738 for
the 50-iteration default. Raising the ceiling therefore does not materially
change this error estimate. At 500 iterations, 899 of 905 primary source fits
and all 319 native target fits converge; six source fits remain finite but
unconverged. It is a 500-iteration sensitivity, not an entirely converged source
calibration.

The additional 297 target rows match the active public chemistry export with
the same numerical tolerance, date checks, and eligibility reconstruction as
the initial 3,267 rows. Together, 3,564 target rows across twelve fitted arms
have verified optical joins.

Every one of the 1,040 rows in the final `seasonal_optical_summary.csv` was
independently recomputed from the original frozen prediction files, using the
canonical calendar functions. This covers twelve matched arms, the existing
published-product context, both February conventions, all four illustrative
MAC values, and the separately labeled scan and prior-problem-filter
sensitivities. Physical-pair counts, month and year counts, mean and median
differences, quartiles, median absolute disagreement, RMS disagreement, negative
prediction counts, and below-optical counts all match. The largest arithmetic
difference is 1.78 × 10⁻¹⁵, below the 10⁻¹⁰ check tolerance.

For the primary `belg_feb`, MAC-10, all-original-scan comparison, the independently
checked median FTIR-minus-optical differences are:

| Product or experiment | Dry/Bega | Belg | Kiremt |
|---|---:|---:|---:|
| Raw shared window | −0.896 | −0.268 | +1.420 |
| Spline df6/4 | −3.428 | −3.728 | −3.823 |
| Existing published FTIR context | −1.046 | −0.510 | +1.547 |

All values are µg/m³, on the same 81, 81, and 70 physical-filter pairs. The sign
change in the raw shared-window and published-product comparisons survives
explicit seasonal reporting; it is hidden by a single annual median. These are
optical disagreements, not measured seasonal EC biases.

## Guard deficits and disposition

1. **Default arPLS stopping status:** 126 of 905 source filters and 51 of 319
   native target scans did not reach the default tolerance within the
   50-iteration implementation. The exact corrections are reproduced and
   the statuses are retained. Finite output must not be called proof of
   convergence. The completed larger-iteration arm preserves the same cohort,
   folds, and component-selection rule; its remaining six unconverged source
   cases must stay visible.
2. **Original receipt completeness:** the original calibration summary hashes
   the ten calibration/native inputs and the main baseline workflows, but not
   the public chemistry export or every imported model-selection helper.
   This review closes the optical-file identity gap for the frozen result.
   The new seasonal receipt should record its consumed output hashes, public
   export hash, helper-code hashes, season convention, and runtime versions.
3. **Seasonal support:** no method should acquire a season or lot effect merely
   from changing its eligible cohort. Keep paired arms on identical physical
   filters, retain unassigned dates, and display sparse or absent cells rather
   than filling them with modeled certainty.

No additional destructive filtering, production-product replacement, or raw-data
modification was required by this review.
