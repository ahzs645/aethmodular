# Davis reference exports

`research/ftir_ec_phase3/scripts/export_aqrc_reference_data.ps1` defines
`Invoke-AqrcReferenceExport`. It exports the small reference datasets identified
in the October 2026 SQL survey, plus the parameter names actually stored in
IMPROVE results. Loading the file does not connect to a database.

## Run on Davis

Copy these two files into the same folder on the Davis machine, for example
Downloads:

- `research/ftir_ec_phase3/scripts/export_aqrc_reference_data.ps1`
- `scripts/diagnostics/inspect_sql_access.ps1` (shared connection/CSV helpers;
  use the corrected version from this repo).

Use the existing local connection setup to populate
`AETHMODULAR_AQRC_CONNECTION_STRING`. The exporter reads that environment variable;
it never records its value, the server endpoint or the login. It preserves the
configured authentication and TLS settings.

```powershell
. "$HOME\Downloads\export_aqrc_reference_data.ps1"
$export = Invoke-AqrcReferenceExport
$export
```

From a repo checkout, dot-source the file at its repo location instead; it finds
the shared helper automatically. An explicit `-HelperScript` path also works.
The script does not change execution policies or database contents.

Each run creates a new folder under `Downloads\aqrc_reference_exports\`.
`-OutputRoot` changes the parent folder. The returned value is the full run path.

## Exported datasets

| CSV | Source and purpose |
| --- | --- |
| `ftir_calibrations.csv` | All coefficient rows, retaining `CalibrationSetId`, `RowNumber`, `Wavenumber`, `b` and `Id`. |
| `ftir_calibration_sets.csv` | Parameter, lot, QC, labels, comments and processing time. |
| `ftir_calibration_shape.csv` | Coefficient counts, distinct row numbers/wavenumbers, ranges and null counts per set. |
| `csu_amod_filters.csv` | All AMOD filters, including mass difference, volume, sampling time, device, lot and error flags. |
| `adama_filter_coverage.csv` | Exact-ID lookup of the five quartz/PTFE pairs in `research/ftir_ec_calibration_2026_06_25/tables/adama_tor_vs_ftir.csv`. Reports missing and duplicate matches explicitly. |
| `analysis_batches.csv` | Batch labels and network IDs. |
| `analysis_media.csv` | Media IDs, batch IDs, labels, barcodes and media metadata. |
| `analysis_parameters.csv` | SPARTAN parameter dictionary with AQS codes and uncertainty metadata. |
| `analysis_networks.csv`, `analysis_media_purposes.csv` | Context for network IDs and sample/blank purposes. |
| `spartan_filter_batches.csv` | Physical external filter IDs and sites joined through `MediaId` to batch IDs/labels. Left joins retain unmatched filters. |
| `improve_aqs_parameters.csv` | IMPROVE's `analysis.AqsParameters` dictionary, including units and methods. |
| `improve_result_parameters.csv` | `SELECT DISTINCT Parameter` from `Improve_2.1.analysis.Results`, with an explicit null indicator. No measurement values are exported from this large table. |

Only `Networks_1_0` and `Improve_2.1` are queried. FRM filters and shared-drive
searches are outside this export. Reads cover the rows visible to the current
login, with no row sampling and no `NOLOCK`. Separate queries do not constitute
an atomic snapshot.

The default timeout is 60 seconds per reference query and 180 seconds for the
IMPROVE results parameter check. `DISTINCT` can read a large index or table even
though its output is small. A timeout is recorded as a failed check; other
exports continue. To allow more time:

```powershell
$export = Invoke-AqrcReferenceExport -ImproveTimeoutSeconds 600
```

`-SkipImproveParameterCheck` omits that query, records a skip and leaves the
IMPROVE question open. It still exports the IMPROVE dictionary.

## Verification and interpretation

Start with `exports.csv`: each dataset has a status, verified CSV row count,
filename and query ID. A successful empty export has a header and zero rows.
Failed or skipped exports are explicitly marked and do not establish an empty
source. `queries.jsonl` records exact SQL, timestamps, timeout and query outcome;
`errors.csv` records exception types and SQL error numbers without potentially
sensitive exception messages. `receipt.json` records options and script hashes.
`manifest.csv` records file hashes, sizes and CSV row counts, excluding itself.

CSV dates use ISO 8601 and decimals use invariant formatting. SQL float values
are converted with style 3 on the server to preserve their precision on Windows
PowerShell 5.1. Empty fields generally represent SQL NULL or empty text; use
`ParameterIsNull` in the IMPROVE results export to distinguish these cases.

Calibration coefficients alone do not establish that raw spectra reproduce the
official product: confirm preprocessing, spectral-grid alignment, intercept and
calibration-selection conventions first. The shape export makes the extra-row
question inspectable without assuming what it means.

Database batch IDs/labels require a crosswalk before treating them as the lab's
delivery batch numbers. AMOD exact-ID matches establish catalog coverage, not
sampling comparability or valid gravimetric measurements. The mass/volume ratio
in micrograms per cubic metre is `1000 * MassDifference_micrograms /
SampleVolume_liters`; check positive volume, flags, weighing conventions and
sampler size cut before interpreting it as gravimetric PM2.5. Raw values remain
unchanged.

Absence of functional-group column names does not rule out long-format results.
Inspect the successfully exported IMPROVE result parameter names and their
dictionary descriptions. A failed/skipped parameter query cannot settle that
question.

## Offline verification

```powershell
pwsh -NoProfile -File tests/powershell/test_aqrc_reference_export.ps1
pwsh -NoProfile -File tests/powershell/test_sql_access_survey.ps1
```

The fixtures replace database transport and exercise CSV precision, exact Adama
ID coverage, failure/skip handling and manifest integrity without credentials.

## Completed run and follow-up work

The returned run `20261028_105305_0098a639` completed on Davis with 13 datasets
and zero recorded errors. Its 18 manifest entries were verified locally after
transfer. The subsequent policy-restoration error came from a malformed paste
that never assigned the saved policy; it does not invalidate the completed data
export. Recovery to the session's previously observed `Undefined` Process policy
was supplied separately; do not assume a future user's prior policy was Undefined.

The optional `-IncludeSpartanCoverage` adds a result-type lookup, site metadata,
and native FTIR/HIPS counts per registered SPARTAN media. To export only these
three new datasets without repeating the completed 13-dataset pull:

```powershell
. "$HOME\Downloads\export_aqrc_reference_data.ps1"
$coverage = Invoke-AqrcReferenceExport -SpartanCoverageOnly
```

Sample results are resolved by the `hips.ResultTypes` label. A missing label
skips the dependent query; it does not report false zero counts. FTIR and HIPS
are aggregated separately before joining. Paired-scan counts check referenced
scan IDs, while sample-reading counts are raw availability, not final products.

`research/ftir_ec_phase3/scripts/export_aqrc_followup.ps1` is a separate read-only
follow-up for AMOD FTIR spectra and functional-group file leads. It requires the
same `inspect_sql_access.ps1` helper. It exports all AMOD analyses, the union of
referenced sample/background scans as base64, per-scan byte lengths/hashes and a
reference-completeness check. It also exports media-type definitions rather than
inferring quartz from numeric media-type IDs.

```powershell
. "$HOME\Downloads\export_aqrc_followup.ps1"
$followup = Invoke-AqrcFollowupExport
```

The FG search uses explicit `-FgSearchRoot` folders or
`AETHMODULAR_FG_SEARCH_ROOT`. Otherwise it derives only the specific
`Chelsey_SPARTAN_MAIA` project folder from the deployed calibration comments.
Unavailable roots, read errors and search limits are explicit in
`fg_search_roots.csv`. It searches filenames and the first 12 lines of candidate
CSV/text files; it does not read workbook contents, compute missing groups or
certify a candidate as an official corrected delivery. The default per-root
budgets are 10,000 files, 2,000 directories, depth 6 and 120 seconds of loop time;
an individual network I/O call can still block. Outputs go under
`Downloads\aqrc_followup_exports\` in a fresh run folder.

The follow-up run `20261028_112459_ba004525` completed with zero errors. After
transfer, all 13 manifest entries, 652 scan byte hashes and 1,160 sample/background
references were verified. It contains 669 AMOD filters and 580 analyses across
556 media. The scoped shared-folder search examined 7,309 files in 245 directories
and returned 142 candidates, with no recorded read errors, caps or depth limit.
Its CSV-header probe found one exact group-column match; the 42 workbook
candidates were not inspected internally. The batch-4 `V2.xlsx` and earlier FG
combine/prediction files are leads to collect and compare with the request list,
not evidence of a corrected or complete delivery. Candidate paths were exported;
the candidate file contents were not copied by this run.

For future paste blocks, put policy capture and cleanup inside one `& { ... }`
block, start from a clean prompt with Ctrl+C, and use the skill's full runbook.
This avoids executing policy setup when an earlier malformed line prevented the
prior value from being captured.

## Local source-coverage audit

Run `research/ftir_ec_phase3/scripts/audit_spartan_network_coverage.py` on the
analysis machine, from the repo with `uv run`. It reads the public mirror and,
when available through configured data paths, the lab HIPS delivery. It separates
direct public HIPS, lab HIPS Fabs, HIPS–SSR curve estimates and SSR equivalent BC.
It writes per-filter, site, year and method coverage with input hashes and an
output manifest. It does not rebuild gallery data or change gallery labels.

```sh
uv run aeth doctor
uv run python research/ftir_ec_phase3/scripts/audit_spartan_network_coverage.py \
  --reference-folder "$REFERENCE_EXPORT" \
  --fg-request "$FG_REQUEST_CSV" \
  --out research/ftir_ec_phase3/output/tables/spartan_coverage_new
```

Set the two shell variables to the user's local inputs; do not commit their
absolute paths. `--native-folder` optionally adds the new coverage-only pull.
Absent native coverage remains unknown. The request crosscheck compares physical
IDs, public OM, exported shipments/media IDs and the known batch-2/3 FG delivery;
it does not obtain the missing concentrations. An earliest reported sample date
is not evidence of an instrument's introduction date. Missing public values are
not proof that the lab never measured a filter.
