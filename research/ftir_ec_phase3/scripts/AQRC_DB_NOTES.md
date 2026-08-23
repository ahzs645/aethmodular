# AQRC database structure — mined from Sean's Shiny app code (2026-08-20)

Source: the "FTIR Calibration" app zip Sean shared 2026-07-02, now at
`My Drive/University/Research/Grad/FTIR Calibration/` (`global.R` +
`R/selectorServer.R`). This is the "knowledge of the IMPROVE database
structure" his email warned about.

## The IMPROVE side (database `Improve_2.1` on AQRC-SQL)

Metadata lives in SQL, organized by schema:

| schema.table | what |
|---|---|
| `ftir.SampleAnalysis` | one row per FTIR analysis — Id (= AnalysisId), FilterId, scan ids, AnalysisDate, AnalysisQcCode, InstrumentId… |
| `filter.Filters` / `filter.Lots` / `filter.Statuses` | filter identity, LotNumber, status codes |
| `filter.FilterComments` | the comments the app doesn't export |
| `module.Modules`, `sampler.Samplers`, `sampler.AverageFlows` | site/module/volume chain |
| `analysis.Results` + `analysis.Sets` | TOR/HIPS-style results rows |
| `grav.SampleAnalysis` | same pattern per analysis type (schema-per-method) |
| `ftir.Instrument`, `ftir.FilterIntegrityChecks` | instrument + integrity |

## Discovery run 2026-08-20 (SqlClient, Windows machine on VPN)

`sys.databases` on AQRC-SQL: ApexSQL*, ArchiveCrd_*, `CSN_1.0`,
`Field_Group_Inventory`, FWS_*, `Improve`, `Improve_2.1`, `Networks_1_0`,
`Spada`, TestDB, ReportServer*. The user's AD account can open **only
`Improve_2.1`** — everything else is access-denied. ETBI/SPARTAN presumably
lives in one of the denied DBs (`Spada` and `Networks_1_0` are the natural
candidates) → ask Sean for read access by name. ALSO: `Improve_2.1` has
**`ftir.Scan`** — Id, NumberOfDataPoints, FrequencyOfFirst/LastPoint,
YScalingFactor, YMax/YMin, XUnits, Type, `Data` (binary blob), instrument
fields. So IMPROVE spectra ARE in SQL as packed binary (the HTTP service is a
decoder in front of this table), joined from
`ftir.SampleAnalysis.SampleScanId`/`BackgroundScanId`. Full-resolution spectra
are therefore reachable by SQL — the app's CSV export subsamples every 4th
point, so a direct `ftir.Scan` decode would upgrade the pool spectra if ever
needed. Note ODBC was a dead end on that machine (no SQL drivers installed);
.NET SqlClient (`Integrated Security=SSPI`) is the working route.

Follow-up probes (same night): denied DBs fail with db-level "Login failed for
user 'AD3\ajalil'" — server login fine, no db grant. Sites in `Improve_2.1`
live in **`sampler.Samplers`** (`Name` = site code e.g. ACAD1, `SiteName`,
`SiteTypeCode` = 'IMPROVE', `UCCode`, module-slot count), reached from
`module.Modules` by SamplerId; `module.Modules` itself has no SiteCode column.
A full-text sweep of `sampler.Samplers` for ET%/Addis/Bishoftu/SPARTAN returns
only "Addison Pinnacle" (ADPI1) — **SPARTAN/ETBI is definitively NOT in
`Improve_2.1`**. Critical path: Sean grants `db_datareader` on the SPARTAN
database (likely `Spada` or `Networks_1_0`) to `AD3\ajalil`; the export
one-liner is ready in `get_etbi_spectra.ps1` the moment access lands.

## The spectra are NOT in SQL — superseded (see discovery above); the HTTP
## service is a convenience decoder over ftir.Scan

`R/selectorServer.R`: spectra come per-AnalysisId from an HTTP service —
`ftir::get_spectrum(id, "improve.aqrc.ucdavis.edu", <secret>)` (the internal
AQRC `ftir` R package; the token sits in `global.R` line ~203 of the app copy —
**do not commit it anywhere**). The service returns `$Absorbance$Records`
(Frequency, Value); the app keeps every 4th point above 500 cm⁻¹ and
pivots wide. So the 13k-pool CSV's shape is exactly this pipeline's output.

## Consequences for the ETBI pull

- SQL (PowerShell route) gets the ETBI **metadata** for sure; whether SPARTAN
  spectra are in SQL or behind the same kind of HTTP service is what the
  discovery pass answers (it searches for wavenumber-shaped columns).
- The SPARTAN database name is unknown — discovery lists all databases on
  AQRC-SQL; expect something parallel to `Improve_2.1`.
- If SPARTAN spectra are API-served like IMPROVE's, the fallback is a five-line
  R script on the Davis remote machine (which has the internal `ftir` package):
  loop `ftir::get_spectrum()` over the ETBI AnalysisIds from SQL, using the
  token from the app copy, and `pivot_wider` — literally the app's own
  `get_a_spectrum` with a different id list and possibly a different host.
- The app has never been run locally against production: the zip references
  `../config/config.yml`, and no config dir was ever created. Nothing lost —
  the direct SQL + API route supersedes it.

## Also of note

`global.R` hardcodes the 21 calibration sites ("someday in the database") and
`FREQ_MINIMUM_IMP = 500`; biomass selection comes from
`data/BiomassDetected_2019..2025.csv` shipped with the app — consistent with
the phase-3 read of the deployed-calibration lineage (ftir_19 notes).
