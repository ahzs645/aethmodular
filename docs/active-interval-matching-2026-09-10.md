# Active-interval evidence build — 10 September 2026

The interval-matching machinery is implemented and the available evidence is
cataloged. **A primary observed-data cohort is still unavailable:** no inspected
source establishes a filter's actual pump on/off periods, and the timestamped
instrument exports do not establish their observation or correction histories.
The build preserves these gaps instead of inventing intervals or fitting a
calibration to date candidates.

## Results from the current local inputs

| Result | Count | Meaning |
|---|---:|---|
| Physical filters retained | 1,055 | All parent identities, including blanks and flagged records |
| Filter/channel decisions | 5,275 | Five MA350 channels per filter; these are decisions, not successful matches |
| Eligible HIPS/FTIR diagnostics | 545 | Existing registry and filter-type rules retained |
| Eligible HIPS/FTIR EC ratios | 480 | FTIR EC is resolved, positive and at/above its reported, unambiguous MDL |
| Lab date rows recovered | 680 | Beijing 232, Delhi 256, JPL 192; original date strings preserved |
| Lab date rows whose identity is in the parent | 531 | Beijing 227, Delhi 120, JPL 184 |
| Conflicting ChemSpec EC groups | 500 | None adjudicated by value sorting or averaging |
| Timestamped instrument rows staged | 1,566,318 | Addis 1,095,086; JPL 471,232 |
| Verified active intervals | 0 | No inferred daily or continuous intervals |
| Eligible primary interval comparisons | 0 | Explicit schedule, provenance and coverage gates |

The 317 HIPS records with positive reported windows and 347 historical HIPS/IR
date candidates remain separate from the 545 filter-only pairs. Subtracting the
23 timing issues from 317 does not establish a matched cohort.

Of the parent filters, 430 have hour-consistent but unverified envelopes, 373
have only recovered lab date bounds, 22 have unresolved active periods, one has
contradictory timing, and 229 lack recovered schedule evidence. ETAD-0163 retains
its 22.8 sampled hours inside 24 elapsed hours. ETAD-0178 retains its zero-length
envelope contradiction. ETAD-0241–0245 and the intermittent JPL records have no
speculative active intervals.

## EC interpretation

The evidence table preserves all 1,043 ChemSpec EC rows: 500 conflicting groups
and 43 single-record groups. In every conflicting group, one reported
concentration equals its reported MDL rounded to two decimals. For CHTS-0658,
the 0.06 value matches the rounding of MDL 0.06375, while 0.93 matches the rounded
FTIR prediction 0.932718. The original unified rows are 5859 and 5860; the
FTIR prediction is row 5887, using zero-based positional rows.

This is evidence for a **possible MDL field flattened into the concentration
field**, not enough to relabel either row as an authoritative source parameter.
The inspected current and archived portal exports did not recover EC parameter
definitions. The inspected combined FTIR/HIPS SQLite database did not contain a
ChemSpec source table. Original EC codes, role definitions, reference conditions,
export versions and revision precedence remain unresolved. The generated EC
table records null selected values and retains all competing rows, methods,
units, MDLs, volume and FTIR links.

ChemSpec EC is not an independent EC reference. The usable FTIR field remains
explicitly FTIR-predicted EC. Its ratio eligibility is separate from unresolved
ChemSpec EC; comparisons using only HIPS and eBC do not depend on ChemSpec EC.

## Instrument staging

The canonical Drive resolver locates the two timestamped exports:

- `Aethalometry Data/Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv`
- `Aethalometry Data/Raw/Pasadena_MA350_1-min_2023-2024_Cleaned.csv`

Both have valid UTC timestamps, no duplicate timestamps or acquisition IDs,
and a reported 60-second timebase. Addis spans 12 April 2022–20 August 2024 UTC;
JPL spans 23 June 2023–31 May 2024 UTC. These are exported-input ranges, not claims
of continuous or genuinely observed coverage.

The staged Parquets preserve original BC1/BC2/BCc strings, datum/session IDs,
firmware/app versions, flow/status fields, zero-based CSV data rows and full
source SHA-256 hashes. Numeric BCc copies are added without conversion or
interpolation. Unknown observation flags retain nullable boolean type through
Parquet serialization. Numeric finiteness is the declared initial validity
rule; it does not establish instrument QC or replace a documented status rule.

Neither the filename nor an acquisition ID proves that the exported values were
never replaced or interpolated. Cadence-grid origin and acquisition timestamp
meaning remain unverified. BCc units and wavelengths are retained as repository
configuration, with export-specific verification separately false. BCc processing
and optical conversion history remain unresolved. No HIPS `MAC_VALUE` is used
as an aethalometer conversion coefficient. Existing daily pickles remain
historical inputs; relabeling their timestamps is not a regeneration.

Beijing/Delhi processed manual-BCc pickles were located, but original observation
and processing provenance has not been established. Direct portal retrieval for
their metadata timed out during this investigation; local lab dates were
recovered instead. These are access/provenance gaps, not claims that data do
not exist.

## Reproduction

From the repository root:

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/build_active_interval_matches.py
uv run pytest -o addopts='' -q tests/test_active_interval_matching.py tests/test_resample_intervals.py
```

The build validates the source audit artifacts against their manifest and
checks the unified source hash, identity/unit rules and exclusion-registry hashes.
It then hashes local evidence, stages the
instrument exports and writes to
`research/ftir_hips_chem/output/tables/active_interval_matches/`:

| Artifact | Contents |
|---|---|
| `sampling_intervals.parquet` | Typed table of verified active periods; currently empty |
| `filter_schedule_catalog.parquet` | Every parent filter, unresolved schedules and evidence links |
| `lab_date_evidence.parquet` | Original lab metadata, date strings, source hashes/rows |
| `ec_provenance_resolution.parquet` | One resolved-or-unresolved decision per ChemSpec EC group |
| `ec_provenance_source_rows.parquet` | All original EC rows and separately labeled numerical clues |
| `instrument_processing_provenance.parquet` | Per-site/channel units, cadence, corrections, observation and optical gates |
| `matched_active_intervals.parquet` | Every filter/channel with separate means, coverage and eligibility reasons |
| `active_segment_coverage.parquet` | Coverage and source-row links for each admitted period/channel; currently empty |
| `observations/*.parquet` | Timestamped inputs with nullable observation flags |
| `selection_summary.parquet`, `selection_report.md` | Cohort counts and limitations |
| `source_inventory.json`, `manifest.json` | Inspected sources and input/code/output hashes |

Generated tables and instrument copies are local outputs, not source-controlled
datasets. Full source content is rehashed before cache reuse. A failed Drive
hydration/read is not treated as proof of corruption or permission to delete a
source archive.

## Adding a verified subset

Supply reviewed evidence with:

```bash
uv run python research/ftir_hips_chem/workflows/build_active_interval_matches.py \
  --sampling-intervals /absolute/path/reviewed_sampling_intervals.parquet \
  --processing-evidence /absolute/path/reviewed_stream_evidence.json
```

Each interval requires the columns in
`scripts/active_interval_matching.py::INTERVAL_COLUMNS`: site, base filter ID,
unique interval ID, timezone-aware UTC bounds, original local clocks and named
timezone, documented continuous/intermittent mode, evidence type, source path,
SHA-256, zero-based source row and resolution reason. Local clocks must agree
with UTC, periods must have positive duration and not overlap, and continuous
collection has exactly one interval. No date-only lab or hour-consistency rule
automatically creates this table.

The processing JSON is a list keyed by the exported `stream_id`. Each update
requires the staged source's hash, `provenance_evidence_file`,
`provenance_evidence_hash` and a human-readable `resolution_reason`. It may point
to a documented observation Parquet via `staged_file`/`staged_file_hash`, with
the original source-row identities and boolean observation/validity columns.
Explicit booleans gate cadence, timestamp role, observation history, correction
history, units and optical conversion. An optical conversion also requires a
positive coefficient, its evidence and verified wavelength treatment. Hashes
prove content identity; a reviewer must establish the evidence's meaning.

The initial coverage policy is a configurable analyst screening choice:
**at least 75% valid observed slots overall and 50% in each active segment**.
It was declared before inspecting matched performance, not selected to improve
a fit. Change it explicitly using `--minimum-coverage` and
`--minimum-segment-coverage`; the manifest retains the chosen values.

Slots use the declared UTC cadence and origin. Nominal instants belong to
`[start, end)`; raw timestamps are clipped before grouping into slots. The first
partial slot is excluded if its nominal instant precedes the active start.
Sub-cadence rows are averaged within slots and cannot inflate coverage or weight.
Means pool distinct valid observed slots across periods. Input availability,
observed coverage and valid observed coverage remain separate; unknown
observations never become 100% observed. Repeated acquisition timestamps require
source resolution.

The synthetic tests include the requested two 12-hour periods in an eight-day
envelope, an end-to-end reviewed subset, unknown observation history, per-channel
missingness, slot weighting, DST, schedule/source integrity and question-specific
EC/optical gates. Synthetic fixtures are confined to temporary test directories;
none is mixed into the research evidence tables.

HIPS uncertainty and MDL values remain linked to their source rows with unresolved
semantics. AIRSpec held-out RMSE is not assigned as every FTIR sample's uncertainty.
Calibration and withheld-block evaluation await a defensible observed cohort.

## Subsequent filter-only results and evidence review

The [filter-only report](filter-only-results-2026-09-10.md) now provides the
545-pair diagnostics, 480-pair ratios and six figure families. An earlier local
ChemSpec export was recovered: EC code 28203 and FTIR labeling are established,
but both competing values already occur in its Value field. This updates the
inventory gap above while leaving the authoritative value roles unresolved.

Processing evidence now additionally requires explicit `evidence_scope` and
`scope_justification`, with UTC bounds for `bounded_period` and optional
`scope_session_ids_json`. Assertions cannot certify observations outside that
scope. Four successive quarters of each active interval are exported as
descriptive coverage diagnostics; the 75%/50% policy is unchanged.
