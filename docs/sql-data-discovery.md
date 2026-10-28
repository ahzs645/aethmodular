# Catalog and profile accessible SQL data

[`scripts/diagnostics/inspect_sql_access.ps1`](../scripts/diagnostics/inspect_sql_access.ps1)
surveys the SQL Server databases available to the existing Davis login. It is
broader than the SPARTAN search: it inventories visible tables, views, external
tables, columns, declared keys and relationships across accessible databases.
Optional aggregate queries add ranges and missingness. It does not enumerate
files, network drives, APIs, stored procedures, table-valued functions or synonyms.

The script issues SELECT queries, leaves database contents and permissions
unchanged, and writes its reports locally. Loading the file only defines
functions. It needs Windows PowerShell 5.1 or later and SQL Server 2016 or later;
no additional PowerShell modules are required on Davis.

To export the calibration, AMOD and batch reference rows identified by the
survey, see [Davis reference exports](aqrc-reference-exports.md). That separate
exporter also checks the parameter names actually present in IMPROVE results.

## Run on Davis

Copy the script to the machine and open PowerShell in its directory. Use the
existing institutional connection configuration to set
`AETHMODULAR_AQRC_CONNECTION_STRING` in that session. Follow the connection
instructions in [the Davis pull guide](spartan-network-data-pulls.md#connect-on-davis).
Do not paste connection details into a report or commit them. The survey
preserves the connection's authentication and TLS settings.

```powershell
. .\inspect_sql_access.ps1

# First pass: metadata across all visible, accessible online user databases.
$catalog = Invoke-SqlAccessSurvey
Invoke-Item $catalog

# Second pass: aggregate profiles, up to 10,000 rows per table.
$profiles = Invoke-SqlAccessSurvey -Profile
Invoke-Item $profiles
```

Each run creates a fresh folder under `Downloads\sql_access_discovery\`.
`-OutputRoot` changes the parent folder. System databases are listed but excluded
from inspection unless `-IncludeSystemDatabases` is supplied. An inaccessible
database is recorded rather than interpreted as an empty database.

Narrow a follow-up using exact database names and wildcard schema/object names:

```powershell
Invoke-SqlAccessSurvey -Profile -Database 'Networks_1_0','Improve_2.1' `
    -Schema 'analysis','filter','spartan' -MaxProfileRows 10000

# An exact profile of a selected table, subject to the query timeout.
Invoke-SqlAccessSurvey -Profile -FullScan -Database 'Networks_1_0' `
    -Schema 'analysis' -Object 'Parameters'
```

Database names in these examples are known sources, not a promise of current
access. Explicit database names are attempted even when database enumeration
does not expose them. The script never seeks additional privileges.

## Read the reports

| File | Contents |
|---|---|
| `databases.csv` | Visible/requested databases, access checks, connection status and metadata visibility |
| `objects.csv` | Tables/views/external tables, SELECT permission and approximate metadata row counts |
| `columns.csv` | Types, sizes, precision, nullability, identity/computed/masked/encrypted flags, column SELECT permission and any `MS_Description` |
| `keys.csv` | Declared primary keys and unique indexes, in key order; filtered indexes are identified |
| `foreignkeys.csv` | Declared relationships, including disabled/untrusted flags |
| `profiles.csv` | Per-column counts, null percentage, distinct count and supported ranges; each record includes coverage and query ID |
| `profileobjects.csv` | Profile outcome per object, including objects skipped because of scope, unsupported columns or the table budget |
| `errors.csv` | Failed operations, exception class and SQL error number; exception messages and connection strings are omitted |
| `queries.jsonl` | Exact queries, parameters, timing and outcomes for reproducibility |
| `receipt.json` | Options, run status, script hash and limitations |
| `manifest.csv` | Output filenames, sizes, verified CSV row counts and SHA256 hashes |

Numeric and temporal columns get minimum/maximum values. Bounded short text
gets minimum/maximum byte lengths and distinct counts; text category values
are not exported. Identifiers get counts. Binary scans, large text, computed,
hidden and encrypted columns remain in the inventory without value profiling.
Masked columns are skipped unless effective database-level UNMASK is present.
Skip reasons are explicit in `profiles.csv`.

Use the profile alongside the catalog: a foreign key suggests a join, and a
column description may explain a field. Neither establishes units, method
provenance or scientific validity. In particular, public SPARTAN FTIR-derived
EC/OC must not be relabelled as thermal-optical reference measurements.

## Interpret coverage correctly

- **Default profiling is capped.** It uses primary-key order when a selectable
  primary key is visible, otherwise unordered TOP. Neither is a random sample.
  Minimum/maximum, null percentage and distinct count describe only
  `RowsProfiled`. They are not population estimates.
- **`CoversAllVisibleRows` is conservative.** It is true when fewer rows than
  the limit were returned, or an explicit full scan succeeded. Reaching the
  limit always gives false, even if the table happens to have exactly that many
  rows. Full coverage means rows visible to this login at query time.
- **Empty is different from failed.** A successful empty profile has zero rows
  and unavailable ranges/null percentage. A failed query has `Status=failed`
  with an error record. A missing CSV statistic is not a zero.
- **Metadata is permission-limited.** Tables and columns can be hidden by SQL
  Server metadata visibility rules. Approximate counts from `sys.partitions`
  describe physical table metadata; they are not exact SELECT counts and are
  not adjusted for row-level security. No visible object is not proof that no
  object exists. See Microsoft's [metadata visibility documentation](https://learn.microsoft.com/en-us/sql/relational-databases/security/metadata-visibility-configuration).
- **Reads are separate queries.** Profiles use batches of up to 12 columns;
  concurrent changes or unordered selection can produce different row sets.
  They do not form an atomic snapshot, and the script does not use NOLOCK.

The defaults profile at most **200 tables across the run**, with a **30-second
timeout per aggregate query**. Metadata is still inventoried after the profile
budget is exhausted. Adjust `-MaxProfileTables`, `-MaxProfileRows` and
`-ProfileTimeoutSeconds`, or target a smaller scope. Views require
`-IncludeViews`; external tables remain metadata-only. A bounded TOP query can
still do substantial work; `-FullScan` removes the row cap and should target a
reviewed table. A completed run means its requested operations finished; check
`profileobjects.csv` for deliberate skips and `errors.csv` for failed reads.

## Validation

The offline PowerShell fixtures exercise identifier quoting, aggregate query
generation, exact decimal serialization, missingness, empty results, skipped
columns, access failures, table budgets, CSV round trips and manifest hashes:

```powershell
pwsh -NoLogo -NoProfile -File tests/powershell/test_sql_access_survey.ps1
```

These fixtures passed on PowerShell 7 on macOS. The corrected survey also
completed on Davis on 2026-10-28 using Windows PowerShell: the uploaded receipt
records 26 visible databases, 324 objects, 2,955 columns, 2,230 profile records
and zero recorded errors. It used `-Profile -MaxProfileTables 10000`, a 10,000-row
cap and the default 30-second query timeout. The source SHA256 was
`C7C37383DA2334038049BEB09BF91625CCCF0EF4A5A0F249FF5A9338BBCE668E`.
The successful run folder was `20261028_095232_825968bf` under the default output
root. Profiles remain capped observations, and the separate reference exporter
has not yet completed on Davis.

For a reusable Windows App procedure, connection preflight and recovery from
execution-policy errors, use the locally installed `$davis-data-discovery` skill.
