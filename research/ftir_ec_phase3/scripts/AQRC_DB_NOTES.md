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

## SPARTAN side: the `hips` schema (confirmed 2026-08-23)

Earlier notes guessed HIPS results would sit in `analysis.Results` + `analysis.Sets`.
**That is the IMPROVE-side layout and those objects do not exist in `Networks_1_0`**
(`Invalid object name 'analysis.Sets'`). A live `INFORMATION_SCHEMA.TABLES` sweep
of `Networks_1_0` returned:

| schema.table | note |
|---|---|
| `hips.Results` | the HIPS result rows |
| `hips.ResultTypes` | which quantities are stored (tau / Fabs / R1 / T1 …) |
| `hips.CalibrationSets` | **likely the per-lot blank lines** (Intercept/Slope) |
| `hips.CalibrationSetFilters` | the blank filters behind each calibration set |
| `ftir.CalibrationSets`, `xrf.CalibrationSets` | same pattern per method |
| `import.ImportNoteSets` | import bookkeeping |

So the schema-per-method convention holds on the SPARTAN side too, just under
`hips.*` rather than `analysis.*`. `hips.CalibrationSets` is the table to read
for the lot-253 blank-line question in
`../SPARTAN_LOT_INVENTORY_2026-08-23.md` — the shipped CSV only exposes the
Intercept/Slope already applied, not the blanks or the fit behind them.

### Gotcha: `param()` is not paste-safe

`param(...)` is only legal as the **first statement of a script file**. Pasted
into an interactive console it fails to parse, and PowerShell then continues with
every declared variable **empty** — so `WHERE SiteCode = ''` matches nothing and
the output looks like an empty database rather than an error. `check_aqrc_gaps.ps1`
now uses `if (-not $X) { $X = ... }` defaults instead, and asserts the site
lookup returns a row before reporting anything else.

### hips schema, confirmed contents (2026-08-23 live run)

**`hips.Results`** — one row per HIPS measurement, keyed on **`SampleAnalysisId`
(not MediaId)**:

| column | note |
|---|---|
| `Transmittance`, `Reflectance` | the T1/R1 scale (e.g. 989.000, 150.000) |
| `TransmittanceRaw`, `ReflectanceRaw` | normalised (0.688, 0.306) |
| `Wavelength` | **633** — confirms the He-Ne wavelength from the instrument itself |
| `LaserPower`, `LaserDiodeTemperature` | **instrument drift diagnostics** |
| `TransmittanceSensorTemperature`, `ReflectanceSensorTemperature` | per-sensor temps |
| `ResultTypeId` | `hips.ResultTypes`: 0 = Sample, 1 = Reference |

None of the drift diagnostics reach `SPARTAN_HIPS_Batch1-51.v2.csv`, which
exposes only final tau/Fabs plus R1/T1. Pull them with
[get_hips_internals.ps1](get_hips_internals.ps1).

**`hips.CalibrationSets`** (Id, LotNumber, Label, Comments) — the per-lot blank
calibrations, and **a lot can have several over time**, each tied to a named
instrument event:

| lot | sets | events |
|---|---|---|
| 248 | 3, 8, 14 | reconfigured (fibre optics, optics cleaned/realigned) · collimator replaced · 10 lab blanks |
| 250 | 6, 7, 9, 12 | shift · shift 2 · collimator · lab move |
| 251 | 4, 5, 10, 13 | reconfigured · **shift** · collimator · lab move (room 132C -> 138) |
| 253 | 16 | "253 initial", **33 lab blanks** (SPARTAN BatchId 45) |
| 241a | 15 | 16 lab blanks |
| 245 | 2, 11 | move · collimator |
| unk-01 | 1 | initial SPARTAN calibration, lot number not provided |

Two of these were triggered by an **unexplained instrument shift** ("performed
due to an apparent shift in the last calibration of this lot", lots 250 and 251).
That is a measurement-side mechanism that would produce a time-correlated offset,
and it has not been checked against the Addis intercept.

`hips.CalibrationSetFilters` is just (MediaId, CalibrationSetId) — the blanks
behind each set. The Intercept/Slope coefficients are **not** on
`hips.CalibrationSets`; where they live is still unknown (query [B] of
`get_hips_internals.ps1` searches the whole DB for them).

### Currency: the database is not ahead of our exports (ETAD, 2026-08-23)

| | database | our local files |
|---|---|---|
| ETAD filters | **296**, through **2026-03-01** | `ETAD_metadata.csv` 296 rows, same max date |
| ETAD FTIR analyses | **319** (296 filters, all with both scans) | `ETAD_FTIR_spectra.csv` 319 rows |

`newer_than_local = 0`. Filters do **not** reach the database at the 3-day
sampling cadence — Addis sampling stops at 2026-03-01 in the DB itself, though
the newest FTIR analysis ran 2026-04-28. Do not extrapolate "filters we must be
missing" from cadence; check the DB.

### hips.Results: the drift diagnostics are NULL (2026-08-23, all-sites pull)

`LaserPower`, `LaserDiodeTemperature`, `TransmittanceSensorTemperature` and
`ReflectanceSensorTemperature` are **100% NULL across all 6,876 SPARTAN rows**.
The columns exist; SPARTAN never populated them.

**However** `TransmittanceRaw`/`ReflectanceRaw` (also absent from the shipped CSV,
and not derivable from T1/R1) do recover the instrument epochs: `T1/TransmittanceRaw`
is flat within a configuration and steps hard at **2023-05-03** and **2023-09-22**,
matching the events in `hips.CalibrationSets`. T1 itself is stable across those
boundaries, so treat it as an epoch marker rather than evidence of drift in tau.

What the pull *is* good for: raw `Transmittance`/`Reflectance` are bit-identical
to the shipped `T1`/`R1`, `tau = ln((Intercept + Slope*R1)/T1)` reproduces shipped
tau exactly, and 256 filters appear here that the shipped CSV omits — enough to
reconstruct Fabs for them. See `../HIPS_RAW_PULL_2026-08-23.md`.

`ResultTypeId = 1` ("Reference") is **not** a per-filter reference beam — it
cannot substitute for the lot blank line.
