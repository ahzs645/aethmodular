# Optical/filter comparison: measured readiness audit

The first milestone is now a reproducible catalog and candidate-match audit.
**The existing inputs do not yet support a validated interval-matched
aethalometer calibration.** This audit inspected the files and ran the code;
counts below are measured from the current inputs, not estimates from the
repository overview.

## Reproduce

From the repository root:

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/audit_matched_samples.py
uv run pytest tests/test_resample_intervals.py tests/test_matched_sample_audit.py
```

The environment check passed with Python 3.13.9. The workflow uses the existing
config paths (`AETHMODULAR_DATA_ROOT` is supported), canonical base-filter-ID
helper, and existing exclusion registry. It does not edit input pickles or
notebooks. `--filter-path`, `--sites-dir` and `--output-dir` override locations.
Portal exports resolve through `--portal-dir`, `AETHMODULAR_PORTAL_DIR`, a local
`Filter Data/SPARTAN portal downloads` directory, then the existing Drive
resolver. The current site-level files are used without mixing archive
versions. `--skip-portal` reproduces the audit from local pickles alone.

Outputs under `research/ftir_hips_chem/output/tables/matched_sample_audit/`:

- `matched_sample_candidates.parquet`: one row per site/base FilterId,
  measurements with explicit units, separate eligibility/quality flags,
  and original filter IDs and source row numbers.
- `filter_measurements.parquet`: all 44,493 original measurement rows with
  source offsets, retaining conflicting and missing observations.
- `aeth_candidate_links.parquet`: 2,248 candidate row links into the hashed
  saved aethalometer files, including which rows supply finite raw IR values.
- `portal_filter_metadata.parquet`, `portal_measurement_rows.parquet`: 453
  original portal filter records and 14,419 parameter rows from the current
  Addis/JPL exports. These retain local timing fields, sampled hours, collection
  descriptions, reference conditions, parameter codes, and source-row links.
- `selection_summary.parquet`, `selection_report.md`, `aeth_inventory.json`:
  measured sample selection and data-quality findings.
- `manifest.json`: input/code/output SHA-256 hashes, Git revision and dirty
  status, resolved configuration, lockfile hash, Python and installed packages.

The input pickles remain the source of truth. Object fields in the evidence
export are serialized to strings for Parquet compatibility; numbers retain
numeric storage. Source row positions refer to `.iloc` in the input pickle.

## What the files contain

| Site | Physical filter IDs | Finite FTIR EC | Finite HIPS | Same-filter HIPS/FTIR pair | HIPS + finite IR date candidate | Missing sample date |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beijing | 405 | 186 | 163 | 163 | 73 | 30 |
| Addis Ababa | 224 | 190 | 190 | 190 | 181 | 34 |
| Delhi | 120 | 63 | 63 | 63 | 25 | 23 |
| JPL | 306 | 158 | 130 | 130 | 68 | 26 |
| Total | 1,055 | 597 | 546 | 546 | 347 | 113 |

These are separate subsets, not a complete-case intersection. All 546 finite
HIPS measurements have finite FTIR EC on the same base ID. Of those pairs, 545
pass the existing registry and unambiguous filter-type checks for a **filter
measurement diagnostic**, including 62/63 in Delhi after the registered
INDH-0172 exclusion. The registry flags three catalog entries in total;
the other two are not HIPS/FTIR pairs. Six FTIR EC values are nonpositive;
89 are below their stored MDL. All are retained with flags. A ratio analysis
needs an explicit low-denominator eligibility rule.

The 347 HIPS/IR candidates are **not** 347 validated co-sampled air comparisons.
They reproduce a ±1-calendar-day candidate search without applying a coverage
threshold. Observed coverage is unknown, not zero or 100%.

### Collection metadata recovered from original exports

The unified table and lab database omitted useful timing information that
**is present in the original portal CSVs**. The audit now restores it by
physical filter ID, converts local clocks using the configured timezone and
retains the original fields and source hashes.

| Site | Portal records | Positive-duration windows | HIPS with a window | Timing issues | Start-date disagreements |
| --- | ---: | ---: | ---: | ---: | ---: |
| Addis Ababa | 188 | 187 | 187 | 7 | 1 |
| JPL | 265 | 265 | 130 | 16 | 0 |
| Total | 453 | 452 | 317 | 23 | 1 |

The current portal directory contains no Beijing/Delhi exports. This is a
coverage limit of the sources inspected, not proof those records do not exist.

Most retrieved windows are consistent with continuous 24-hour collection, but
important exceptions prevent treating every filter as a daily sample:

- ETAD-0163 reports 22.8 sampled hours within 24 elapsed hours.
- ETAD-0178 reports identical start/end timestamps despite 24 sampled hours.
- ETAD-0241 through ETAD-0245 report 24 sampled hours across 192–216 elapsed
  hours. Their actual active periods cannot be recovered from these fields.
- Fifteen JPL filters report 48 sampled hours across 237–238 elapsed hours.
  USPA-0243 separately has a 23-hour window but reports 24 sampled hours.
- ETAD-0243 starts 2024-08-24 in the portal, whereas the unified `SampleDate`
  is 2024-08-25. Both are retained; no date is silently corrected.

The consistency flag permits 0.05 hours of rounding difference. It is only a
metadata check, not proof of an observed on/off schedule. Positive-duration
windows are source-backed collection bounds; verified active coverage remains
unavailable in the saved aethalometer aggregates.

## Findings that change the recommended next steps

1. **The historical 9 AM builder actually creates 15:00 boundaries.**
   `scripts/pipelines/create_9am_resampled_datasets.py` subtracted 15 hours,
   resampled midnight bins, then added 15 hours. That creates 15:00 bins labelled
   at their start, despite the comment claiming 09:00 interval ends. Saved
   Beijing/Delhi timestamps are all 15:00; JPL has 762 at 15:00, two at 14:00
   and one at 16:00. All three inputs have duplicate `datetime_local` columns.
   The source is fixed and tested; existing pickles need regeneration from
   original timestamped data, not a timestamp relabel.
2. **Coverage was not a count of observed IR minutes.** The builder counted
   non-null records from the first raw BC column and divided by 1,440,
   regardless of DST or duplicate/subminute rows. JPL has one saved day above
   100%; Addis has no coverage column. The corrected function counts distinct
   minutes per channel, handles local DST intervals, and ties compatibility
   fields explicitly to IR. Optional boolean observation flags exclude
   interpolated/non-observed input from means and counts. Without provenance
   flags, this remains input availability, not verified observed coverage.
3. **Actual collection intervals are absent from the unified dataset.**
   `SampleDate`, `FilterType`, and some volumes exist, but no collection
   start/end or active-period schedule. A read-only schema inspection of the
   original `spartan_ftir_hips.db` on Drive also found only `sample_date` in its
   `filters` table and no interval fields in its measurement tables. The portal
   CSVs recover 452 positive-duration windows as described above. The shared `match_aeth_filter_data`
   averages a ±1-day window; `match_all_parameters` also averages filter
   parameters across nearby dates without preserving physical filter identity.
   Their historical results should be treated as diagnostics pending rematching.
4. **ChemSpec EC is ambiguous within the unified table and is not independent.**
   500 physical-filter groups have conflicting `ChemSpec_EC_PM2.5` concentrations
   (Beijing 156, Addis 175, Delhi 27, JPL 142). For example, CHTS-0658 contains
   both 0.93 and 0.06 µg/m³ under this name. The audit does not choose the first
   or average them; source rows remain available for mapping back to the
   original portal parameter definitions. The repo's existing provenance
   investigation also identifies the concentration product as FTIR EC and
   ChemSpec BC as HIPS/10, subject to rounding. See [open items](open-items.md).
5. **Known uncertainty must not be lost or overinterpreted.** All 546 finite
   HIPS rows have corresponding `HIPS_Uncertainty` parameter rows. The audit
   reads those rows, including their `Uncertainty` values, and reads HIPS MDL
   from its sibling parameter. Error semantics remain unresolved; the AIRSpec
   held-out RMSE is not automatically the uncertainty of every FTIR product.
6. **Measurement conversion history still needs verification.** HIPS is in
   Mm⁻¹ and IR BCc is eBC in ng/m³ at the configured 880 nm MA350 channel.
   The audit only converts ng/m³ to µg/m³. `MAC_VALUE` is explicitly a HIPS
   conversion assumption, not a verified coefficient for reconstructing
   aethalometer absorption. No wavelength-aligned absorption fit is claimed.

## Smallest defensible next implementation

1. Recover the remaining Beijing/Delhi metadata and resolve the 23 retrieved
   timing issues, including active periods for intermittent collection. The
   recovered Addis/JPL bounds, timezone and reported collection/reference
   conditions are already in the catalog. Do not infer timing from volume.
2. Resolve the original BCc correction/observation history and rebuild from
   timestamped data. The Drive mount is available, and the archive README lists
   original MA350 files. Listing availability is not raw-data integrity or
   correction verification; large files may require Drive hydration.
3. Match active intervals and export observed per-channel coverage, within-window
   distribution, sample/interval source links and separate eligibility flags.
4. Verify wavelength and optical conversion metadata, then run the three
   measurement relationships. Label FTIR predictions explicitly; absent an
   independent EC reference, do not call those comparisons validation of EC.
5. Evaluate adjustments in withheld calendar blocks with each filter kept in
   one group. Predefine coverage, smoothing, timing, low-EC and influence
   sensitivities before fitting; report bias and errors alongside R² and
   measurement-error-aware slopes.

Completion means every plotted point has a physical-filter identity, an actual
sampling interval and its source, observed coverage, a documented unit and
correction chain, inclusion reasons and a validation group. The current audit
delivers the identity/provenance baseline and makes the remaining gaps explicit.

## Verification

Focused synthetic tests cover local interval boundaries, timezone conversion,
DST duration, channel-specific minute availability, observation flags,
duplicate timestamps and filter identity, conflicting values, missing dates,
unit checks, sibling HIPS uncertainty/MDL, exclusion preservation, and portal
window recovery including intermittent and invalid timing. Relevant existing
CLI, helper, loader, import and exclusion tests also pass. Generated rows are
checked against source offsets, and reruns are checked for identical table
content. This validates the audit software; it does not validate the historical
measurement processing or a calibration model.
