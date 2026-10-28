# SPARTAN network exports: FTIR, HIPS and public ChemSpec

Working procedure and findings from the Davis session on 2026-10-19. This is
the current network workflow; the August SQL notes are historical. Database
reads were SELECT-only. Exports were left in Davis's Downloads directory for
the user to place in the archive. Nothing was published or added to the gallery.

## Which source answers which question?

For a catalog of **all accessible SQL databases**, including columns, declared
joins and optional range/missingness profiles, use the
[general SQL discovery guide](sql-data-discovery.md). The procedure below is
the SPARTAN-specific export workflow.

| Source | Grain and purpose | Important limitation |
|---|---|---|
| Public `ChemSpecPM25` / `ChemSpecPM10` | Filter × parameter × method; concentrations, units, MDLs, uncertainty, flags | Public products, not the raw spectra or HIPS instrument observations |
| `spartan.Filters`, `spartan.Sites` | Media identity, external filter ID, site, filter type, sampling metadata | A registered filter need not have a completed measurement |
| `ftir.SampleAnalysis`, `ftir.Scan` | Analysis and referenced sample/background scans | Repeats are real records; scans are native binary single-beam data |
| `hips.SampleAnalysis`, `hips.Results` | Analysis and sample/reference instrument readings | Raw T/R readings are not official Fabs/BC concentrations |
| Remaining `hips.*` tables | Calibration, instruments, QC types/flags | Empty source tables are possible and must be reported as empty |
| `SPARTAN_HIPS_Batch1-51.v2.csv` | Lab-delivered HIPS optical products and applied calibration lines | Older delivery; preserve alongside a new raw pull |

Public EC/OC are FTIR-derived (method codes 217/218). They are not TOR, and
the public ChemSpec carbon and in-house FTIR carbon must not be counted as
independent measurements. Public BC methods 219/221 are HIPS; method 220 is
estimated from the HIPS–SSR curve. Keep these sources separate.

## Local ChemSpec mirror

The repo already has the public files under
`data/spartan/raw/FilterBased/{ChemSpecPM25,ChemSpecPM10}/` (gitignored).
`scripts/common/spartan_io.py` owns the paths, header reader and ID normalizer.
Do not use plain `read_csv` without skipping the comment header.

```sh
uv run aeth doctor
uv run aeth spartan pull --check-updates
# If there are changes, synchronize and regenerate the inventory:
uv run aeth spartan pull
# Offline inventory rebuild:
uv run aeth spartan pull --skip-download
```

The pull command covers other public SPARTAN products too. Its downloader
compares SHA256 of the response with the local file and installs changed files
atomically. `research/spartan/inventory/overview.csv` records object metadata.
Use `gallery/data/export_spartan_reference.py` only when intentionally rebuilding
the gallery after reviewing changes; an archive pull does not require rebuilding it.

On 2026-10-19 all **74 ChemSpec files** (37 PM2.5 + 37 PM10) were compared with
the public objects byte for byte and were unchanged. Their sizes were
48,549,807 and 2,821,955 bytes respectively. PM2.5 had 160,677 parameter rows,
4,449 distinct filter IDs, and 3,581 filters with public EC/OC at 32 sites.
The gallery's 3,581-filter count is therefore a carbon-coverage count, not the
size of the entire public chemistry population.

Within those 3,581 EC filters, 3,375 have a non-null public BC value from direct
HIPS methods, 198 have a HIPS–SSR curve estimate, and eight have neither public
BC product. These are distinct filter-ID intersections with method codes
retained, not a claim of available raw instrument readings. The counts refer to
the current public mirror; query the new native HIPS export separately.

Public source: [SPARTAN data directory](https://www.spartan-network.org/data),
with the CSVs obtained from its public `GroupByProduct/FilterBased/` S3 mirror
by `scripts/pipelines/spartan_pull_and_summarize.py`. Carbon/method provenance
is also documented in
[`research/spartan_ec_2026_06_16/README.md`](../research/spartan_ec_2026_06_16/README.md).

## Connect on Davis

Use the Windows account with existing database read access and the required
institutional network/VPN. Windows PowerShell's built-in .NET
`System.Data.SqlClient` worked; no ODBC installation was needed.

Keep the SQL endpoint and machine configuration outside version control. Set
`AETHMODULAR_AQRC_CONNECTION_STRING` in the Windows session from the local
configuration, using `Database=Networks_1_0`, `Integrated Security=SSPI` and
`Connect Timeout=60`. Do not print or commit its value. The October session
connected without a certificate-validation override. A pre-login timeout
resolved on retry with the longer timeout; do not treat a timeout as evidence
that the database is empty or change TLS settings to address it.

```powershell
$ErrorActionPreference = 'Stop'
if (-not $env:AETHMODULAR_AQRC_CONNECTION_STRING) {
    throw 'Set the connection string in this Windows session first.'
}
$c = New-Object System.Data.SqlClient.SqlConnection(
    $env:AETHMODULAR_AQRC_CONNECTION_STRING)
$c.Open()

function Q([string]$sql) {
    $cmd = $c.CreateCommand()
    $cmd.CommandText = $sql
    $cmd.CommandTimeout = 600
    $dt = New-Object System.Data.DataTable
    (New-Object System.Data.SqlClient.SqlDataAdapter($cmd)).Fill($dt) | Out-Null
    return ,$dt
}

function E($dt, [string]$path) {
    $rows = foreach ($r in $dt.Rows) {
        $p = [ordered]@{}
        foreach ($cl in $dt.Columns) {
            $v = $r[$cl]
            if ($v -is [DBNull]) { $v = '' }
            elseif ($v -is [byte[]]) { $v = [Convert]::ToBase64String($v) }
            elseif ($v -is [datetime]) { $v = $v.ToString('o') }
            $p[$cl.ColumnName] = $v
        }
        [pscustomobject]$p
    }
    if ($dt.Rows.Count) {
        $rows | Export-Csv $path -NoTypeInformation -Encoding UTF8
    } else {
        # Preserve the schema even when the source has zero rows.
        $header = ($dt.Columns | ForEach-Object {
            '"' + $_.ColumnName.Replace('"', '""') + '"'
        }) -join ','
        Set-Content $path $header -Encoding UTF8
    }
    if (@(Import-Csv $path).Count -ne $dt.Rows.Count) {
        throw "CSV row mismatch: $path"
    }
}
```

`return ,$dt` matters: PowerShell otherwise enumerates the DataTable into rows,
and downstream code can lose its column schema. Do not export `byte[]` as
`System.Byte[]`; the base64 conversion is required. ISO timestamps avoid
locale-dependent dates. Empty strings represent SQL NULL in these CSVs, so
consult the saved SQL schema when restoring types.

## Inventory before exporting

For a fresh search for FTIR functional-group products, use
[`find_ftir_fg_tables.ps1`](../research/ftir_ec_phase3/scripts/find_ftir_fg_tables.ps1).
Set the session connection variable as above, load the script, then run
`Find-FtirFgTables`. Its SQL reads metadata only; it writes a new timestamped
folder under `Downloads\ftir_fg_discovery` with all visible tables/views/synonyms,
all visible columns, keyword candidates, database permission context, query text
and file hashes. CSV row counts are checked after export. Table row counts are
metadata estimates; views and synonyms need separate inspection.

A keyword candidate is not a confirmed functional-group product. Review generic
parameter/species/result tables as well as obvious group names. No match only
describes objects visible to the current account in `Networks_1_0`; it does not
rule out hidden objects, other databases or lab CSV deliveries. In particular,
an absent `VIEW DEFINITION` permission limits any database-wide absence claim.

Save `INFORMATION_SCHEMA.COLUMNS` for `spartan`, `ftir` and `hips`, including
column order and types. Discover tables before assuming an old schema still
applies. In particular, SPARTAN HIPS is under `hips.*`, not IMPROVE's
`analysis.Results`. `hips.Results` joins to
`hips.SampleAnalysis.Id` through `SampleAnalysisId`; it has no direct MediaId.

Start with one row per registered media, using correlated counts or aggregated
subqueries. Joining both analysis tables directly multiplies repeated analyses.

```sql
SELECT s.SiteCode, f.ExternalFilterId, f.MediaId, f.ExternalFilterType,
       f.SamplingStartDate,
       (SELECT COUNT(*) FROM ftir.SampleAnalysis a
        WHERE a.MediaId = f.MediaId) AS FtirAnalyses,
       (SELECT COUNT(*) FROM hips.SampleAnalysis a
        WHERE a.MediaId = f.MediaId) AS HipsAnalyses,
       (SELECT COUNT(*) FROM hips.SampleAnalysis a
        JOIN hips.Results r ON r.SampleAnalysisId = a.Id
        WHERE a.MediaId = f.MediaId AND r.ResultTypeId = 0) AS HipsSampleResults
FROM spartan.Filters f
JOIN spartan.Sites s ON s.Id = f.SiteId;
```

Resolve `ResultTypeId` from `hips.ResultTypes` on every pull; 0 meant Sample
and 1 meant Reference in the inspected schema. A HIPS analysis row without a
sample result is not evidence of a measured optical value.

## Export FTIR without losing repeats or background scans

Create a fresh timestamped output folder. Export all `spartan.Filters` and
`spartan.Sites`, then, per site with FTIR analyses:

1. `<SITE>_filters.csv`: filter metadata for that site.
2. `<SITE>_ftir_analysis.csv`: **every** analysis whose MediaId belongs to it.
3. `<SITE>_scans_base64.csv`: the union of SampleScanId and BackgroundScanId
   referenced by those analyses, preserving all scan columns and base64 Data.

The SELECTs in
[`get_spartan_spectra.ps1`](../research/ftir_ec_phase3/scripts/get_spartan_spectra.ps1)
show the established join path. Use the connection and export helpers above
instead of its older hardcoded connection settings. Derive the site list from
the database inventory, never from the four focus sites or from stale console
variables. Parameterize site predicates in reusable implementations.

Verify analysis counts against the starting inventory, unique analysis and scan
IDs, and that **every referenced scan ID** is present. Keep PM2.5, PM10, field
blanks, lab blanks and repeat analyses identifiable; do not collapse MediaIds
in the source archive.

The native pull is not the 2,722-point absorbance table. The existing
[`build_spartan_target.py`](../research/ftir_ec_phase3/scripts/build_spartan_target.py)
decodes little-endian float32 scans, forms `-log10(sample/background)` and
interpolates onto the calibration grid. It selects among repeated MediaIds:
review/change that selection before using it for an all-analysis archive.
Validate scaling, grid endpoints/length, finite values and agreement with a known
Addis spectrum before generating gallery-ready spectra. Keep source and derived
artifacts separate.

## Export the complete HIPS source

Export all current base tables in the `hips` schema, rather than only the
flattened drift-diagnostic CSV. The October schema had 11 tables:
`AnalysisFlags`, `AnalysisTypes`, `Calibrations`, `CalibrationSetFilters`,
`CalibrationSets`, `Instruments`, `QcSamples`, `QcSampleTypes`, `Results`,
`ResultTypes`, `SampleAnalysis`.

```powershell
$out = Join-Path $HOME ('Downloads\spartan_hips_network_' +
    (Get-Date -Format yyyyMMdd_HHmmss))
New-Item $out -ItemType Directory | Out-Null
$tables = Q @"
SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'hips' AND TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_NAME
"@
$counts = foreach ($t in $tables.Rows) {
    $name = $t.TABLE_NAME
    $dt = Q ('SELECT * FROM hips.[' + $name.Replace(']', ']]') + ']')
    E $dt (Join-Path $out ('hips_' + $name + '.csv'))
    [pscustomobject]@{Table = $name; Rows = $dt.Rows.Count}
}
$counts | Export-Csv (Join-Path $out 'counts.csv') -NoTypeInformation
E (Q 'SELECT * FROM spartan.Filters') (Join-Path $out 'spartan_Filters.csv')
E (Q 'SELECT * FROM spartan.Sites') (Join-Path $out 'spartan_Sites.csv')
```

Also save the schema and filter inventory. Export a convenient joined view:

```sql
SELECT s.SiteCode, f.ExternalFilterId, f.ExternalLotId, f.SamplingStartDate,
       a.MediaId, r.*
FROM hips.Results r
JOIN hips.SampleAnalysis a ON a.Id = r.SampleAnalysisId
JOIN spartan.Filters f ON f.MediaId = a.MediaId
JOIN spartan.Sites s ON s.Id = f.SiteId;
```

Retaining the unjoined tables preserves calibration/QC records that would be
lost by restricting the entire schema to registered SPARTAN filters. Check
orphan result/analysis keys explicitly. Keep the older delivered HIPS CSV:
it contains Fabs, MDL, uncertainty and applied lines that the joined raw data
does not replace. `reconstruct_hips_fabs.py` produces provisional values with
assumptions, not a new official lab release.

## Reconcile cohorts and verify the archive

Compare physical filter keys with the public PM2.5 cohort, preserving original
IDs. Use `base_filter_id` / `add_base_filter_id` or
`config.BASE_FILTER_ID_PATTERN`; the operational public pipeline exposes the
equivalent `common.spartan_io.normalize_fid`. Keep PM10 and blanks separate.
Compare coverage by method/parameter, not only by site presence or row totals.

Save one row per public filter with public EC/OC/BC method coverage, database
presence, FTIR analysis/paired-scan coverage and HIPS sample-result coverage.
Save the anti-join lists and per-site rollup as well. Missing records are a
source gap to investigate, not a reason to manufacture a measurement or silently
drop the filter. A matched raw result still does not establish final QC-passed
Fabs availability.

After placing the unpacked HIPS bundle on the analysis machine, the companion
[`audit_spartan_network_parity.py`](../research/ftir_ec_phase3/scripts/audit_spartan_network_parity.py)
compares its `filter_inventory.csv` and `spartan_hips_raw_all.csv` with the
local public mirror. Set `AETHMODULAR_SPARTAN_HIPS_BUNDLE` in the session to
the chosen unpacked folder, then run:

```sh
uv run python research/ftir_ec_phase3/scripts/audit_spartan_network_parity.py \
  "$AETHMODULAR_SPARTAN_HIPS_BUNDLE" \
  --out research/ftir_ec_phase3/output/tables/spartan_network_parity
```

It writes filter parity, gaps, per-site counts and a source-hash receipt.
It counts distinct sample-analysis IDs for HIPS, keeps reference-only records
from masquerading as samples, and rejects ambiguous PM2.5 metadata keys.
It tests FTIR analysis presence; scan-reference completeness is the separate
export check above. Its fixture verification covered repeats, reference-only
rows, missing metadata and separation of direct HIPS from SSR estimates.

Before compressing, re-import CSVs for row counts and write a manifest with
relative filenames, sizes, SHA256, query text, UTC retrieval time and database
name (no endpoint, account or secrets). Include zero-row table counts. Create a
new ZIP, then read **every ZIP entry as a stream** and compare its SHA256 with
the original file. Windows `Compress-Archive` may use backslashes in entry
names: normalize `\` to `/` before matching manifest names. Save a separate
SHA256 checksum for the ZIP itself. Close the SQL connection in `finally`.

Leave the ZIP, checksum and unpacked folder together until the user chooses the
archive destination. Once moved, configure paths through `.env` / existing
`data_paths` resolvers and record paths with `env.display_path()`; do not commit
raw spectra, machine paths or account-specific Drive locations.

## Windows App input lessons

If Davis is visible but computer-use clicks return `noWindowsAvailable`, open
Windows App's **Window > Connection Center**, then double-click the saved Davis
device to reactivate the existing session. This restored input in the October 27
session after merely raising the window did not. Remote screen updates can lag
behind completed input calls; verify the full command and resulting prompt.

The remote terminal was usable through Windows App with Unicode keyboard input.
Scancode input dropped Shift on this session (`$` became `4`, parentheses became
digits); stop immediately if a harmless test command is corrupted. Clipboard
paste also failed in this session. Click the terminal prompt first, send short
commands, inspect the typed command, then press Return and inspect its output.
Long simulated typing lost chunks. Use ASCII script text and a saved script file
when available; a `param()` block belongs at the start of a script file and is
not safe to paste into an already-running console. Escape cleared unfinished
input. An unchanged screen does not establish command completion.

## October 19 FTIR export receipt

`spartan_ftir_network_20261019_072611.zip`, left in Davis Downloads:

| Quantity | Count |
|---|---:|
| Sites with FTIR scans | 28 |
| Registered filter metadata rows exported | 4,272 |
| Distinct filters with FTIR analyses | 4,270 |
| FTIR analyses, all with both scan IDs | 4,452 |
| Per-site exported scan rows (sample + background) | 5,233 |
| Unpacked source/audit bytes | 95,140,843 |
| ZIP bytes | 55,739,025 |
| Files verified inside ZIP | 88 |

Scanned-media inventory: 3,517 PM2.5 filters / 3,685 analyses; 219 PM10 /
222 analyses; 534 field blanks / 545 analyses. These sum to 4,270 media and
4,452 analyses. The extra two metadata rows have no FTIR analysis. The native
DB cohort and the 3,581 public-carbon filters are different populations.

The old sizing argument `319 Addis spectra / 246 public filters` mixed cohorts.
Only 267 of those existing Addis analyses match the 246 public filters, so the
matched repeat factor is about 1.085, not 1.30. Existing staged FTIR pulls also
existed for CHTS, INDH, USPA and ETBI; Addis was not the only spectra source.
For storage estimates specify whether counting native scans, derived absorbance
spectra, repeat analyses, or public physical filters. At 4,650 × 2,722 points,
float32 values alone occupy 50,629,200 bytes; that is an estimate for a derived
array, not the observed native archive size.

## October 19 HIPS and parity findings

The fresh export folder is `spartan_hips_network_20261019_074717` in Davis
Downloads. It contains all 11 native HIPS base tables, the schema, full SPARTAN
filter/site metadata, a joined raw HIPS view, the 74 public ChemSpec CSVs and
their object index, and filter-level/per-site parity tables. It supplements the
separate FTIR archive above. The joined HIPS view contains raw readings; final
Fabs remains a separate delivered product.

`spartan_hips_network_20261019_074717.zip` contains **99 files**; each ZIP
entry was read back and its SHA256 matched to the source file. The unpacked
folder and a `.zip.sha256.txt` checksum are beside it. Native CSV row counts
were also checked against all 11 table counts, including the four empty tables.

| Quantity | Count |
|---|---:|
| Native `hips.SampleAnalysis` rows (whole table) | 5,111 |
| Native `hips.Results` rows (whole table) | 8,605 |
| Result rows joined to registered SPARTAN filters | 7,180 |
| SPARTAN sample readings / distinct sample analyses | 4,274 |
| Distinct SPARTAN filters with sample readings, all types | 4,267 |
| HIPS results without a parent analysis | 0 |
| Public ChemSpec downloads with mismatched object MD5 | 0 / 74 |

The older staged raw HIPS CSV had 6,876 result rows and 4,122 sample analyses;
the new pull adds 304 result rows / 152 sample analyses by count. The whole native
tables also preserve records outside the registered-SPARTAN join. Do not call
the whole-table count a SPARTAN physical-filter count.

The public EC cohort was reconciled against native PM2.5 metadata by normalized
physical filter ID, not by approximate site totals:

| Public EC cohort | Present in this native pull | Missing from this native pull |
|---|---:|---:|
| FTIR analysis | 2,754 | 827 |
| HIPS sample reading | 2,753 | 828 |

Denominator: 3,581 public EC filters. **Full raw-data parity is not established.**
The database's complete available export is not the entire public cohort.
Among sites with public EC, CADO (28), CALE (13), INKA (15), VNHN (12) and
SGSU (13) have no matched native FTIR analysis or HIPS sample in this pull.
Other sites also have partial coverage. `INDH-0280` has a matched native FTIR
analysis but no native HIPS sample reading. See `public_ec_gaps.csv`,
`public_ec_site_parity.csv`, `public_filter_parity.csv` and
`parity_summary.json` inside the export for the exact missing cohort.

Preserve those missing IDs for the next source request: locate additional lab
deliveries or ask the database owner which source/import holds them. Do not
replace the public HIPS product with a reconstructed raw-data value or infer
that a public measurement did not happen because this database lacks its raw row.
`hips.AnalysisFlags`, `hips.CalibrationSetFilters`, `hips.QcSamples` and
`hips.QcSampleTypes` were empty in this source; an export cannot supply absent
calibration-filter membership or QC history.

## October 27 functional-group discovery

Read-only discovery ran successfully on Davis against `Networks_1_0`. The
starting script came from commit `4d761ed6b0f52a97be90e3daecc2e2605d9171de`
on `claude/eloquent-clarke-2e11yo`; its downloaded SHA256 matched
`8325278c60a801af94fe93c1b0394c6b92fd35d9899b6a670404cefa42be056d`.
Before loading its helpers, the older certificate-validation override was
removed, connection timeout was set to 60 seconds, and the output directory was
changed to a fresh timestamped folder. Supplementary SELECTs saved the complete
column and table/view inventories, parameter lookup, calibration descriptions,
effective permissions and a direct count of the generic carbon-processing table.

Observed results:

- **75 visible base tables, 570 visible columns, no visible views or synonyms.**
- No functional-group result table or group-named column was found by the
  discovery search. The broad keyword candidates included instrument-parameter
  metadata and raw HIPS readings; a keyword match is not a final product.
- All **20 `ftir.CalibrationSets` records** were labelled for `EC_ftir` or
  `OC_ftir`. `ftir.Calibrations` had **54,460 coefficient records** with
  `CalibrationSetId`, `RowNumber`, `Wavenumber` and `b`, rather than per-filter
  functional-group results.
- The **91-row `analysis.Parameters` lookup** contained `EC_ftir`, `OC_ftir`
  and the composite `OMC` (Organic Mass by Carbon) entry. Searches of its names
  and codes found no functional-group entries. A lookup entry does not establish
  populated results or identify the public FTIR OM product.
- A direct `COUNT_BIG(*)` confirmed **zero rows** in
  `qcarbon.ProcessedData`.
- Effective database permissions included `SELECT` and `CONNECT`, but not
  `VIEW DEFINITION`. These findings describe the current account's visible
  objects in this database; they do not rule out hidden objects, other databases,
  network-drive files or separate lab deliveries.

The evidence was left in Davis Downloads in
`ftir_fg_discovery_20261027_230608`. It contains the original `tables.csv` and
candidate `columns.csv`, plus `all_columns.csv`, `objects.csv`,
`calibration_labels.csv`, `parameter_lookup.csv`, `permissions.csv`,
`context.csv` and `qcarbon_count.csv`. All nine CSVs were re-imported; row counts
were 75 tables, 129 keyword-candidate columns, 570 full columns, 75 objects,
20 calibration labels, 91 lookup entries, four permissions, one database-context
row and one carbon-table count row. `manifest.csv` records each file's relative
name, re-imported row count and SHA256. The database connections were closed after
inspection. The next source to request is the lab's official functional-group
deliveries, including the corrected batch 4; this search did not recover them.

## October 28 survey: no functional-group results in SQL

A read-only metadata survey of every database the login can open (`Networks_1_0`
and `Improve_2.1`; 324 objects, 2,955 columns) found **no stored FTIR results** for
SPARTAN: no functional-group parameters (aCH, aCOH, COOH, naCO, FTIR OM) in
`analysis.Parameters` or any column, and no results table in `Networks_1_0`
(`qcarbon` is empty). `ftir.Calibrations` (54,460 rows) holds coefficients, not
results: one `(Wavenumber, b)` vector per `ftir.CalibrationSets` row, all 20 of them
OC/EC by lot (234, 241/241a, 245, 248, 250, 251, 253, a dust-site set). Their
comments point at the lab share (`U:/FTIR_spectra/Chelsey_SPARTAN_MAIA/…`), which
is where the functional-group deliveries should be requested or looked for.

[`get_networks_extras.ps1`](../research/ftir_ec_phase3/scripts/get_networks_extras.ps1)
exports those calibrations, the Adama `csu_amod` filters (gravimetric mass
difference and volume), the batch tables, `frm.*` and a parameter check on
`Improve_2.1`, with a manifest and a verified ZIP like the exports above.

## October 28 reference export: what it settled

`aqrc_reference_exports/20261028_105305_0098a639` (13 exports, 0 errors, every
manifest SHA256 re-verified) answered four questions:

- **IMPROVE stores no FTIR products.** The 137 parameters observed in
  `Improve_2.1` `analysis.Results` include TOR fractions, seven-wavelength
  thermal-optical carbon (`EC405TR` … `OC980TT`), HIPS raw intensities and
  `fAbs`, but no FTIR OC/EC and no functional groups.
- **The deployed SPARTAN calibrations are complete and usable.** All 20
  `ftir.Calibrations` sets have 2,723 rows: row 1 (wavenumber 0) is the
  intercept, then 2,722 coefficients on 3998.42 → 500.12 cm⁻¹, the explorer's
  app grid. Lot 245's rejected sets (QC 0) are identical to lot 241's and its
  accepted sets equal lot 241a's. Preprocessing (raw or baselined absorbance,
  units) still has to be confirmed against a public OC/EC value before predicting.
- **Functional groups stop at shipment `Shipment_2023.05.08`.** The database
  batches are shipment labels, not the lab's numbered delivery batches. Every
  shipment from `Shipment_2023.06.14` to `Shipment_2026.08.07` (filters sampled
  Nov 2022 → Jul 2026) has public OM and no functional groups: 2,082 filters,
  plus 291 older filters with public OM that are not in `spartan.Filters`
  (CAHA, AEAZ, CASH, CADO, NGIL …). That 2,373-filter list is the request to
  the lab.
- **Adama is covered.** All five quartz TOR filters (J1675, J1679, J1693, J1701,
  J1703) and their PTFE partners (J1285, J1270, J1269, J1266, J1233) are in
  `csu_amod.Filters`. The PTFE partners carry weighed mass: 31.3, 41.3, 38.0, 34.4
  and 26.8 µg/m³ (1000 × `MassDifference_micrograms` / `SampleVolume_liters`;
  J1270's volume is 1,161 L against ~2,880 L for the others, so check it).
  `csu_amod` holds 669 filters at about ten locations, including **101 quartz
  filters** (Kaohsiung 66, Delhi 14, Pretoria/Johannesburg 12, Adama 6). Their
  TOR results are not in SQL; if CSU analysed them as it did Batch 54, they are a
  thermal EC/OC reference at SPARTAN sites.

## October 28 follow-up: AMOD spectra and the calibration convention

`aqrc_followup_exports/20261028_112459_ba004525` (0 errors, hashes verified) holds
580 FTIR analyses on 556 CSU AMOD filters with all 652 referenced scans. AMOD
quartz filters have no FTIR analyses; the five Adama PTFE filters have two each.

**The deployed calibrations are reproducible.** Prediction = row-1 intercept +
Σ b × raw absorbance (−log10 sample/background, each scan interpolated onto the
2,722-point app grid, no baseline), which gives µg per filter; ÷ sampled volume
(m³) gives µg/m³. Applied with lot 245's accepted EC set (= lot 241a) to the five
Adama PTFE spectra, it reproduces the lab's Batch 54 FTIR EC
(`research/ftir_ec_calibration_2026_06_25/tables/adama_new_ec_by_variant.csv`)
to 0.974–0.996 of the delivered values. With the 2026-10-19 spectra this can
regenerate SPARTAN FTIR OC/EC under any of the 20 lot calibrations.

The same export searched `U:\FTIR_spectra\Chelsey_SPARTAN_MAIA` (7,309 files,
names and CSV headers only). Leads: a batch 4 report **V2**
(`Spartan batch4\report\…Nov2022 to March2024 V2.xlsx`, 2025-02-03, possibly the
corrected organics), `Spartan batch4\FG and BLC\`, `Spartan batch5\`, and
`MAIA\MAIA batch4\FTIR by diff. cal vs. TOR data\` (likely the AMOD quartz TOR).
[`get_chelsey_share_files.ps1`](../research/ftir_ec_phase3/scripts/get_chelsey_share_files.ps1)
copies the small files from those folders with a full listing and a verified ZIP.

## October 28 lab share: corrected batch 4, batch 5 and the AMOD TOR

A copy of `U:\FTIR_spectra\Chelsey_SPARTAN_MAIA` shared on Drive (7,677 files;
fetched selectively with the `mirror_davis_drive.py` helpers) settles the
functional-group gap and the AMOD quartz question:

- **Batch 4 V2 is the corrected delivery.**
  `Spartan batch4/report/SPARTAN_FTIR data_batch 4_ Nov2022 to March2024 V2.xlsx`
  (1,060 filters, sampled 2022-11 → 2024-03): OM / public OM median 0.948 (p5–p95
  0.78–1.05), where the submitted V1 runs 1.936. OC and EC equal the public values
  (ratio 1.000) in both. Two header rows; filter IDs are in `sample`.
- **Batch 5 exists.** `Spartan batch5/spartan5_OCEC_V_F_OM.csv` (373 filters,
  2023-05 → 2024-10): the four groups as `OM_aCOH`, `OM_aCH`, `OM_naCO`, `OM_COOH`
  plus `OM_total` (µg/m³; OM / public OM median 0.954), OC and EC equal to public.
  The raw group outputs per spectrum are in `Spartan batch{4,5}/FG/OM_FG.csv`.
- Together they cover 1,163 of the 2,373 filters with public OM and no groups.
  Left: 1,210, mostly filters sampled after 2024-10 (2024: 150, 2025: 481, 2026:
  254; CLST, INDH, CLTA, ETAD, INJA, ETBI) and 291 from 2018–2020 that predate
  `Networks_1_0`.
- **AMOD quartz TOR** is in `MAIA/MAIA batch3/TOR for batch3 and 4.xlsx`
  (OCTR/ECTR per quartz filter with volume) and `MAIA/MAIA batch4/Quartz_FilterId_OCEC.csv`:
  9 Pretoria filters (batch 3) and 22 Tainan filters (batch 4). Batch 3 ECTR is
  ≈ 0 on all nine (median −0.001 µg/m³) and batch 4 on 5 of 22, with OC 8–9 µg/m³.
  The paired PTFE filters read FTIR EC 1–5 µg/m³ and Fabs/10 0.5–1.7, so
  the AMOD quartz ECTR is not a usable EC reference without a lab explanation;
  OCTR may be. `MAIA/MAIA batch4/FTIR by diff. cal vs. TOR data/` holds the
  lab's own FTIR-by-calibration vs TOR comparison for those filters.

**In the gallery (2026-10-28).** `export_spartan_reference.py` reads all three
deliveries (`FG_DELIVERIES`); the two new files sit beside the batch 2–3 CSV in
Davis Data `EC-HIPS-Aeth Comparison/Data/Lab deliverables/FTIR EC/`. Rebuilt with
the Drive inputs, the previous exporter reproduced the committed JSON exactly
(0 field differences) before the change; after it, functional groups cover
2,780 filters at 26 sites (was 1,473 at 24; Addis 21 → 162, Delhi 15 → 49), the
batch 2–3 values and every volume are unchanged, and five lab-only filters gain
rows. Twelve Dhaka filters in V2 (BDDU-0401 … 0414) have no sample date anywhere
and stay out. `availability.json` and `aeronet/stations.json` still count the
filters without those five until the AERONET export reruns with its cache.

## Reference exports after the access survey

`research/ftir_ec_phase3/scripts/export_aqrc_reference_data.ps1` exports the
SPARTAN calibration coefficients and descriptions, AMOD filter metadata, batch
lookups and a physical-filter-to-batch map. It also exports the IMPROVE parameter
dictionary and checks the distinct parameter names actually present in
`Improve_2.1.analysis.Results`. The large results table is queried only for names.

See [Davis reference exports](aqrc-reference-exports.md) for setup, output files,
timeouts and interpretation. This exporter has offline fixture coverage; writing
it does not establish that its new queries have run successfully on Davis.
