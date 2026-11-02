# SPARTAN mass-balance release reconciliation

This follow-up audits the gallery's 141 filters without a current compatible
reconstruction. The earlier count refers to the full focus-site cohort of 938
filters after the gallery's central exclusion flags. The quoted 494-filter UI
selection was not supplied and is not the denominator used here.

The reproducible analysis is
`research/ftir_hips_chem/workflows/reconcile_spartan_mass_balance.py`. Run it with
`uv run` after `uv run aeth doctor`. Inputs resolve through `maia_data_root()`
and `common.spartan_io.RAW_DIR`; no machine-specific paths belong in this file.
Its separate audit outputs are under
`research/ftir_hips_chem/output/tables/spartan_mass_balance_audit/`.

## Definition and numerical checks

The current release carries three parameters under method 618:
`Residual Matter Only`, `Residual Matter with OC`, and `Residual Matter with OM`.
On the paired filters, `Only - with OC = OC` to the published 0.01 µg/m³
rounding precision. This establishes a within-release relationship. It does
not establish how to convert an older release.

The legacy v3.0 files updated 2025-07-29 have one parameter, `Residual Matter`.
Its formula text matches the current `Only` description, but neither that name
mapping nor the OC-subtraction mapping reproduces the current values. The
actual comparison against current `with OC`, restricted to identical physical
IDs and sampling dates, is:

| Site | Paired residuals | Median absolute difference (µg/m³) | 95th percentile absolute difference (µg/m³) |
| --- | ---: | ---: | ---: |
| Beijing | 330 | 0.435 | 13.903 |
| Addis Ababa | 172 | 0.635 | 2.829 |
| Delhi | 26 | 1.435 | 8.080 |
| JPL | 223 | 0.500 | 2.471 |

The differences include revisions of soil, sea salt, trace-element oxides and
some carbon values. Comparing releases must retain each release's complete
component bundle. A sum forced to equal PM₂.₅ by assigning the leftover to
water would be an accounting identity, not independent validation of a residual
conversion. No universal legacy-to-current residual conversion was established.

## Historical coverage and flags

Sixteen currently unmatched JPL filters have a historical balance with matching
IDs and dates. Fourteen occur in the July 2025 v3.0 archive; the November 2024
v2.4 archive contains fourteen, including two additional physical filters.
Both versions are retained in `archived_missing_filter_components.csv`, with
the original values, units, methods, flags, release identifier and source hash.
Nanogram values are also provided in explicitly named µg/m³ fields.

The official methods workbook defines `SGP` as **Screen or Grid Pattern**,
`VIH` as **Visible Inhomogeneous Deposition**, and `SUS` as **Suspect or
Compromised Data**. These flags are distinct. A morphology flag alone is not
a new exclusion rule, and the historical records have not been silently
reinstated or excluded. The publisher's reason for their later absence remains
unresolved.

Keep these observations as a versioned historical source. Do not substitute
them into the gallery's current twelve-component stack by renaming a column or
subtracting OC. The warning should describe absence of a compatible
reconstruction in the loaded release, rather than asserting that a filter
never had a published reconstruction.

## AQRC extraction and local import

The verified 2026-10-28 AQRC reference export contains 90 of the requested
physical IDs: Delhi 69, Beijing 15, Addis 3 and JPL 3. These are catalog and
mass/volume metadata, not 90 completed chemical balances.

The scoped 90-row subset is retained under
`data/aqrc/spartan_mass_balance/reference_20261028_subset/`, alongside a
141-filter coverage table and a receipt linking the subset to its verified
source hashes. The native mass and volume values are preserved without a
concentration conversion. This is a subset of the existing October 28 export,
not a new SQL pull. Its transfer archive is
`aqrc_existing_reference_missing_filter_metadata.zip` in the audit output folder.

`research/ftir_ec_phase3/scripts/export_spartan_mass_balance_followup.ps1`
defines a SELECT-only follow-up for all 141 requested IDs. It retains unmatched
requests and exports:

- Physical media, sampling intervals, native mass/volume fields, batch and
  sample/blank metadata.
- Actual raw HIPS readings, analysis records and qualifier flags.
- FTIR analysis/QC metadata and sample/background scan availability.
- Available media comments, filter-integrity records, QC definitions and the
  visible schema describing potential result stores.

It exports no final ion concentrations or reconstructed balance unless those
products are actually found in a subsequent source. Raw HIPS readings are not
official Fabs/BC, and FTIR scan availability is not an EC/OC result. The script
preserves existing authentication/TLS settings, changes no database contents or
permissions, and makes no execution-policy changes itself.

The bundle builder is
`research/ftir_hips_chem/workflows/build_aqrc_mass_balance_bundle.py`. The Davis
run requires the existing configured Windows session. Temporary Process
`RemoteSigned` and removal of download markers from the three reviewed files
were explicitly approved in this chat. The run instructions restore the prior
Process policy in `finally`. The October 31 run completed and its output archive
was received and independently verified locally. Preparing or transferring a
bundle alone must not be described as a completed pull.

`research/ftir_hips_chem/workflows/import_aqrc_mass_balance_export.py` verifies
the output manifest, hashes, CSV counts, receipt, exact request cohort and
canonical replicate identities before retaining raw files under the gitignored
`data/aqrc/spartan_mass_balance/` directory. Its coverage table keeps missing
registry records, repeated analyses, raw flags, negative mass and zero volume
inspectable. It does not manufacture chemical concentrations or replace gallery
values. Its failure/identity/raw-value tests are in
`tests/test_aqrc_mass_balance_import.py`.

### Verified October 31 results

Run `20261031_234146_395a58fd` completed nine datasets with zero errors. The
received archive matches the SHA256 printed on Davis and its checksum sidecar:
`30b7efe8e29bb9535515b6a4fdde667f530c1c24de515a0ddc7dbedd69620bb2`.
All 14 manifest entries passed hash and applicable CSV-count checks. The exact
141 requested physical IDs were retained, including unmatched requests.

The raw files are retained under
`data/aqrc/spartan_mass_balance/20261031_234146_395a58fd/`. The generated
`aqrc_physical_filter_coverage.csv` and `aqrc_import_receipt.json` in the audit
output folder link physical IDs, dates, metadata and source QC evidence.

| Result | Physical filters | Rows |
| --- | ---: | ---: |
| Registry matches | 90 | 90 |
| Raw HIPS sample readings | 90 | 94 |
| Raw HIPS reference readings | 53 | 57 |
| FTIR analysis/scan metadata | 90 | 94 |
| Media comments | 32 | 33 |

Registry matches comprise Delhi 69, Beijing 15, Addis 3 and JPL 3; none has a
sampling-date conflict. The other 51 requests had no registry match in this
accessible export. That does not prove that they are absent from all AQRC data.
The schema inventory contains 486 columns across scientific tables, not 486
chemical observations. The pull contains no completed chemical mass balances.

All 90 matched filters have an analysis marked Valid (`QcCode=1`) in both HIPS
and FTIR. Four additional HIPS analyses are Invalid (`0`), with comments saying
they were software tests; four additional FTIR analyses are Replicates (`6`).
Keep these records and their codes distinct. Neither a valid analysis code nor
empty structured integrity/qualifier tables overrides an invalid physical
sample documented in a media comment.

Six requested filters carry explicit `INVALID --` comments: `INDH-0169`,
`INDH-0170`, `INDH-0171`, `INDH-0173`, `INDH-0174` and `INDH-0176`. The comments
record WashU's decision to reject cartridge INDH-022, covering filters
INDH-0169 through INDH-0176, because of water infiltration and disagreement
between mass and BAM observations. This establishes documented sample rejection
for those six members of this request cohort. It does not establish the complete
public-release processing history. `INDH-0172` was already centrally excluded
for extreme FTIR EC; this scoped export does not contain a fresh comment row
for it or for `INDH-0175`.

Other comments record grid patterns, holes, uneven deposits, a barcode
correction and an unsealed station (`INDH-0161`). These observations are retained
for review, without promoting every morphology note to a rejection. The
coverage table explicitly flags the six literal source invalidations and keeps
analysis QC counts alongside them. No new central exclusions or gallery
chemical backfills were applied as part of the initial import.

### Handling agreed on November 1

The six Delhi filters with explicit source invalidations are now in the central
`outliers.EXCLUDED_SAMPLES` registry, keyed by their physical/external identities
and sampling dates. The default clean gallery view excludes them; their records,
values and water-infiltration reasons remain inspectable through the existing
excluded-filter controls. The scope is the six approved filters, not an inferred
exclusion of every filter with a morphology note.

The original 938-filter/141-gap audit and AQRC request remain a dated snapshot.
New exclusion flags change the current clean population; they do not rewrite
the historical request cohort or make missing reconstructions available.
The rebuilt focus export retains all 942 physical filters, with 10 centrally
excluded and 932 clean. Of the clean filters, 797 have a current reconstruction
and 135 do not. These are full-cohort counts, not the unreproduced 494-filter
selection. `current_stack_coverage.json` records this check.

The full reference/native rebuild needs an unavailable batch-4 workbook and
native scan bundles. `exclusion-flags` instead refreshes the canonical flags in
the existing reference and content-addressed network filter snapshots; it leaves
measurement values and native spectrum assets unchanged and records source
hashes for the flag update. The focus export was rebuilt from its dataset.

The 16 JPL filters remain versioned historical records. The generated
`jpl_historical_balance_review.csv` has one row per filter and archived release
(28 rows for 16 physical filters), retaining dates, original PM and residual,
reported parameter names, flags and source paths. None of these archived entries
reports separate OC. Work through them by reproducing each release's original
component definitions, comparing entire same-date bundles on overlapping
filters, and resolving changed inputs before promoting any balance to the
current stack. Keep an unexplained difference explicit; do not label the
leftover water or substitute zero for an unreported parameter to force closure.

AQRC records without completed chemistry remain staged. They are not used to
fill missing stack components. Missing reconstruction alone does not exclude a
filter from analyses of its other available measurements.

### Live network-database continuation on November 1

The existing AQRC CLI connection successfully probed `Networks_1_0`. A live
listing returned 75 visible physical user tables (`limit=1000`,
`hasMore=false`). Column metadata was inspected for all 69 scientific tables,
546 columns, leaving authentication tables and `dbo.sysdiagrams` uninspected.
None of the column reads hit its 1000-row limit. These are current-login
physical-table results, not an inventory of every server object.

Five-row live previews returned zero visible rows, with `hasMore=false`, for
`qcarbon.ProcessedData`, `qcarbon.SampleAnalysis`, `qcarbon.SourceFiles`,
`xrf.DeviceCounts`, `xrf.SampleAnalysis`, `xrf.TransmissionFiles` and
`xrf.LabBlankStats`. `hips.Results` describes raw transmittance/reflectance;
FTIR stores scans and calibration coefficients rather than a discovered final
reconstruction table. The FRM and AMOD schemas contain metadata/mass fields for
different projects; their column definitions do not identify SPARTAN balances.

The shared parameter dictionary returned 91 definitions (`limit=200`,
`hasMore=false`), including `RCMN`, Reconstructed Mass. This is a lookup
definition, not a measurement result. It also contains thermal-optical carbon
names for other network workflows; those names do not establish TOR reference
measurements on the four focus-site SPARTAN filters.

All 88 visible import-batch headers were returned (`limit=200`,
`hasMore=false`); their types are Filter and HIPS. No chemical-result import
type was found in this log, although other processing may use different logs.
The import-note preview returned 200 unordered rows with `hasMore=true`; none
of those rows matched the media IDs of the 141-filter request. This leaves
cohort-specific notes unresolved. The 10-row FTIR calibration-set preview was
also capped; calibration labels are not per-filter chemical results.

No additional completed balance was recovered in that first CLI pass. Reader v1
could not list views/synonyms or filter/join notes by FilterId. The reader-v2
follow-up below supersedes those tool limitations; it uses the guarded SELECT
path rather than requiring a new named endpoint for each query.

Raw structured responses, their limits and `hasMore` values, a summary, and a
SHA256 manifest are retained in the audit output's
`database_search_20261101/` directory. This was a live read, not a fixture. The
database-name listing showed `Networks_1_0` and `Improve_2.1`; no contents of
the other catalog were queried for this follow-up. No helper permissions,
execution policies, database contents or gallery exclusions were changed.

### Reader-v2 follow-up on November 1

The restarted reader was independently verified live with protocol 2 and
`queryAvailable=true`. The existing connection was reused; no policy, token,
allowlist or server configuration was changed by this investigation.

Both `views` and `synonyms` returned zero visible rows (`limit=1000`,
`hasMore=false`). The combined table/view/synonym inventory returned 75 objects
without hitting its limit. An expanded metadata SELECT found the scientific
tables and database-diagram helper routines, but no visible reconstruction
view, synonym or scientific processing routine. No routine was executed.

An explicit permission read returned database `SELECT=1` and database-wide
`VIEW DEFINITION=0`. This does not prove that a hidden reconstruction object
exists; it means complete definition/metadata visibility has not been
established. Object access and visibility remain subject to the current login.

Exact `COUNT_BIG` queries confirmed zero visible rows in the seven XRF/carbon
stores checked earlier. Other counts describe raw acquisition and metadata:
5,261 FTIR analysis rows, 5,979 scans, 5,111 HIPS analysis rows, 8,605 raw HIPS
result rows, 4,491 SPARTAN registry rows and 1,062 import-note rows. These are
row counts, not distinct physical-filter counts or completed balances.

A parameterized CTE searched all 141 original requested IDs against stored
ExternalFilterId and import Identifier, allowing only numeric physical `-N`
suffixes. The query returned nine notes for nine filters, one page with
`limit=1000`, `hasMore=false`, `byteBudgetReached=false` and zero truncated
cells at a 1,048,576-character cell limit. The other 132 IDs had no matching
notes under this explicit identity rule; unknown alternate identifiers remain
outside it. The query and its parameters are retained for reproduction.

The nine notes concern sampling durations: eight Beijing filters and one
Addis filter. Live media comments identify the corresponding 8-day or 54-day
SPARTAN protocols. `CHTS-0889` has a historical 1,149-day warning based on
2021-10-24, whereas the live stored start is 2024-10-24 and its media comment
states a 54-day protocol. These notes do not establish fresh sample rejection
or explain the publisher's reconstruction decisions. No new exclusions were
applied.

No additional completed balance was recovered. The two reader-v1 search gaps
have now been checked to the extent of current metadata permissions and the
defined ID match. Remaining uncertainty concerns hidden objects, other stores
or servers, alternate identifiers and upstream reconstruction processing files.
An owner-confirmed object/export location or the publisher's versioned outputs
would be more informative than repeating the same visible-table search.

`reader_v2_search_summary.json` and `reader_v2_import_note_coverage.csv` in the
audit output summarize the result. The summary points to the gitignored raw
query folder under `data/aqrc/spartan_mass_balance/`, with SQL, parameter files,
complete structured responses and a verified SHA256 manifest. These bounded
live queries are distinct from the earlier full export receipt.

### Further independent probing on November 2

This pass went beyond the visible Networks tables into saved lab deliveries,
lab QC notes, the other allowlisted database and a fresh publisher check. It
used the existing CLI and connected Drive access without computer use, changing
the helper or contacting anyone. **It recovered useful organic-result leads and
identified missing volume metadata, but no additional compatible reconstruction.**
The counts below retain the original 141-filter request, not the rebuilt
135-filter current gap cohort.

Five complete source workbooks were downloaded and read internally: batch 2
resubmitted, batch 3 with MDLs, the combined batch 2/3 delivery with MDLs, and
both the original and V2 batch-4 deliveries. An ID-only scratch sheet was not
counted as a result. Matching used the physical base ID plus the workbook's
explicit sampling-date column; sample names, suffixes, row numbers and source
versions remain in the audit.

| Delivery evidence | Original requested filters |
| --- | ---: |
| Numeric four-group/OM results with matching ID and sampling date, across versions | 56 |
| Of those, already carrying OM rows in the unified dataset | 41 |
| Of those, JPL organic-result candidates absent from the unified dataset | 15 |
| Other JPL IDs with conflicting delivery dates | 2 |
| Batch-4 Delhi rows with volume and derived concentrations reported as `NA` | 27 |

The 56 are 7 Beijing, 2 Addis, 29 Delhi and 18 JPL filters. These are FTIR
organic measurements, not 56 complete chemical balances. The 15 additional JPL
candidates have numeric aCOH, aCH, naCO, COOH, OM, FTIR OC and FTIR EC; their
values agree between the separate batch-2 and combined batch-2/3 deliveries.
Twelve also belong to the previously documented historical-balance set. They
therefore extend the evidence on organic measurements, not the count of
recovered current reconstructions. No values have been imported.

The two date conflicts are `USPA-0034` (request 2021-11-12, delivery date column
2021-12-11) and `USPA-0037` (request 2021-11-21, delivery date column 2021-11-12).
Filenames and date columns must be reconciled rather than selecting whichever
date makes a join succeed.

The batch-4 version difference is material: **all 26 numeric matched OM pairs
differ between original and V2**, while reported volume, FTIR OC and FTIR EC
are unchanged. Individual group values also differ. Neither the filename nor
its modification date establishes which FG/OM values are approved. Both
versions are retained; no correction or common conversion has been inferred.

For the 27 Delhi `NA` rows, a new bounded live join returned all 27 physical
registry records. Both `spartan.Filters.SampleVolume_m3` and
`analysis.Media.SampleVolume` are null for every one. This confirms a missing
concentration-conversion input in these sources. It does not establish why the
publisher omitted a reconstruction, or justify reconstructing volume from an
assumed sampling schedule. These are distinct from the six rejected Delhi
filters already excluded centrally.

Three additional laboratory comparison workbooks were inspected internally.
The workbook explicitly named `OCEC_OM_Ion_soil_residual.xlsx` contains none of
the requested IDs. `OM_PM_Ions_Soil.xlsx` contains chemistry/organic plot inputs
for 11 requested JPL IDs, with repeated lot-comparison sheets; it does not
provide a compatible current reconstruction. `uspa.xlsx` is a carbon/HIPS
comparison, not a chemical balance. The accompanying July 2024 lab note
explicitly says missing nitrate or sea salt was replaced with zero for certain
QC plots, while sulfate and soil were required. Those plot conventions are
not a publisher release rule or evidence that the missing species were measured
as zero. They must not be imported into the gallery's measured-species view.

The saved filter-integrity document names damaged Beijing filters outside this
request, describes heavy deposits/uneven sampling on some requested filters,
and distinguishes invalid first analyses from valid reanalyses. It supplies no
new explicit rejection of the requested cohort. A broader live integrity-check
match returned zero rows. Morphology and absent integrity rows do not determine
final publication status; no exclusions were added.

`Improve_2.1` was independently probed successfully. Its metadata inventory
returned 249 visible tables/views/synonyms without a cap. A bounded search of
`filter.Filters.FilterBarcode` for CHTS, ETAD, INDH and USPA returned zero rows.
This only tests those recognizable barcode forms; it does not rule out unknown
alternate IDs. The roughly 159-million-row IMPROVE result store was not scanned,
and its statuses must not be assigned to SPARTAN filters without an identity link.
Networks QC-code definitions were also read in full. Their Valid, Invalid,
Reanalyzed, Superseded, Reprocessed and Replicate meanings describe analysis
records, not the publisher's per-filter reconstruction workflow.

Fresh downloads of ChemSpecPM25 and ReconstrPM25 for all four sites match the
existing cache byte for byte. The publisher's GroupBySite and GroupByProduct
copies also match each other for all eight files. They remain version 3.1.4,
updated October 13 for Beijing/Addis/Delhi and August 4 for JPL. No newer or
alternate balance was recovered through those two public layouts. The linked
known-issues document discusses nitrate loss, without per-filter processing
states. This pass did not find an explicit pending-QC, pending-publication,
withdrawn or unfinished-analysis status for the remaining gaps.

`research/ftir_hips_chem/workflows/audit_spartan_lab_deliveries.py` reproduces
the delivery comparisons and checks the saved live-query caps. The audit output
contains `further_probe_summary.json`, `lab_delivery_matched_rows.csv`,
`lab_delivery_additional_organic_candidates.csv`, `lab_delivery_date_conflicts.csv`
and `lab_delivery_batch4_version_differences.csv`. Original downloads, source
metadata, SQL and scalar parameters, full uncapped responses, publisher snapshots
and a SHA256 manifest are retained under the gitignored
`data/aqrc/spartan_mass_balance/further_probe_20261102/` directory. The live SQL
queries have `hasMore=false`, `byteBudgetReached=false` and zero truncated cells.

The interface wording **“reconstruction unavailable in this release”** remains
supported. The investigation now gives concrete follow-up targets: reconcile
the 15 JPL organic candidates and two dates, establish the approved batch-4 FG
version, and locate source volume records for the 27 Delhi filters. None of
those requires assuming that a missing reconstruction is awaiting processing.

## Sources

- [SPARTAN data and product definitions](https://www.spartan-network.org/data).
- [SPARTAN product documentation](https://spartan-cloud.readthedocs.io/en/latest/data_type.html#rcfm).
- [Sampling Parameters and Methods v2.3](https://www.spartan-network.org/_files/ugd/6321a4_219864c4f1454c58a956339bc58b9120.xlsx?dn=Sampling_Parameters_Methods_2.3.xlsx), sheets PM2.5 RCFM and Flags.
- [July 2025 JPL reconstruction archive](https://drive.google.com/file/d/1B3H6W4Y6ZG1v7xqeNC6uh7GyIJZlo9x8/view).
- [November 2024 JPL reconstruction archive](https://drive.google.com/file/d/1ooJE6kPj5FdcEEUm05PzJQM3k_CwUnXU/view).
- The current public ReconstrPM25 files, their parameter-level metadata, and the
  input hashes in `reconciliation.json`.
- [Batch-2 resubmitted laboratory delivery](https://drive.google.com/file/d/1A2IB2abz8-r_QCc4K6ctkcLTRiTUOxzy/view) and
  [combined batch-2/3 delivery with MDLs](https://drive.google.com/file/d/17SmNP9O6D63bZnI5OTr7z-cphGqSKn7Y/view).
- [Original batch-4 delivery](https://drive.google.com/file/d/1XLK7jE6hBqzdf1PM8kH3EjOsLJzD8qwT/view) and
  [batch-4 V2 delivery](https://drive.google.com/file/d/1FigoWyRQgDX_avk06jcuNCLm86-p58Bi/view).
- [Batch-4 laboratory report](https://drive.google.com/file/d/1bTXLejoz-371GZ3wziN42Fbsl1nIWnzg/view),
  [filter-integrity notes](https://drive.google.com/file/d/19AmC2RgaNKaE9sFqMFyEacLyNFsb0iQO/view), and
  [laboratory QC-plot conventions](https://drive.google.com/file/d/1EvyL1hi6LQ5qvdZajQd2srlrBVM0JmFx/view).
