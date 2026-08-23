# How to query the AQRC SQL Server directly (cookbook)

Established 2026-08-20. This is the practical guide; the schema knowledge and
the investigation trail live in [AQRC_DB_NOTES.md](AQRC_DB_NOTES.md), and the
ready-made ETBI script is [get_etbi_spectra.ps1](get_etbi_spectra.ps1).

## Prerequisites

- A **Windows machine on the UCD VPN** (macOS can't do the Windows AD trusted
  auth, and the server's firewall drops everything off-VPN — from outside,
  port 443/1433 time out silently).
- Your **Windows AD account** (`AD3\ajalil`) — no separate SQL password;
  `Integrated Security=SSPI` uses your login. This is what Sean's
  `Trusted_Connection: "yes"` means.
- Nothing to install: **do not bother with ODBC** (the lab Windows machine has
  no SQL ODBC drivers at all). .NET's built-in `System.Data.SqlClient` ships
  with Windows and talks to SQL Server directly.

## The recipe

Open PowerShell (Start → type `powershell`; no admin needed) and paste:

```powershell
function Open-AqrcConn([string]$db = "") {
    $cs = "Server=AQRC-SQL,1433;Integrated Security=SSPI;TrustServerCertificate=True;"
    if ($db) { $cs += "Database=$db;" }
    $conn = New-Object System.Data.SqlClient.SqlConnection($cs)
    $conn.Open()
    return $conn
}

function Invoke-AqrcQuery($conn, [string]$sql) {
    $cmd = $conn.CreateCommand(); $cmd.CommandText = $sql; $cmd.CommandTimeout = 300
    $table = New-Object System.Data.DataTable
    (New-Object System.Data.SqlClient.SqlDataAdapter($cmd)).Fill($table) | Out-Null
    return $table
}
```

Then query at will:

```powershell
$c = Open-AqrcConn "Improve_2.1"
Invoke-AqrcQuery $c "SELECT TOP 5 * FROM sampler.Samplers" | Format-Table -AutoSize
Invoke-AqrcQuery $c "SELECT ..." | Export-Csv "$HOME\Downloads\out.csv" -NoTypeInformation
$c.Close()
```

## Hard-won gotchas (each one cost a round-trip)

1. **Functions die with the window.** Every new PowerShell window starts
   empty — re-paste the two functions first. Symptom: `The term 'Open-AqrcConn'
   is not recognized`.
2. **Saving as .ps1 via Notepad breaks on non-ASCII.** Notepad writes UTF-8
   without BOM; Windows PowerShell 5.1 reads that as ANSI, and an em-dash's
   bytes include one that decodes as a *curly quote* — strings terminate early
   and you get cascading `Missing closing '}'` parse errors. Keep scripts pure
   ASCII (the repo script now is), or just paste into the console instead.
   In-place fix if it happens:
   `(Get-Content in.ps1 -Raw) -replace '[^\x00-\x7F]+','-' | Set-Content out.ps1 -Encoding ASCII`
3. **Running a .ps1 file** needs `powershell -ExecutionPolicy Bypass -File
   "C:\path\script.ps1"` (a bare quoted path just echoes the string; policy
   blocks plain invocation). Console pastes bypass policy entirely.
4. **`$db:` inside a double-quoted string is a parse error** (scope-qualifier
   syntax) — write `${db}:`.
5. **"Login failed for user" per-database is a permissions grant, not a broken
   login** — the server authenticated you; that database has no read grant.
   Ask Sean for `db_datareader` on it by name.

## What you can reach today

- **`Improve_2.1`** — the only database `AD3\ajalil` can open (of ~20 on the
  server). Key tables: `ftir.SampleAnalysis` (AnalysisId, FilterId, scan ids,
  QC codes), `ftir.Scan` (**full-resolution spectra** as scaled binary blobs —
  `Data` decoded with `YScalingFactor`/`NumberOfDataPoints`/frequency bounds;
  the Shiny app's HTTP service and its every-4th-point CSVs are a convenience
  layer over this), `filter.Filters`/`Lots`/`Statuses`/`FilterComments`,
  `sampler.Samplers` (site codes in `Name`, e.g. ACAD1), `module.Modules`,
  `analysis.Results`/`Sets`.
- **NOT reachable yet: SPARTAN/ETBI.** Definitively absent from `Improve_2.1`
  (full-text site sweep returns only "Addison Pinnacle" for `%Addis%`). It's
  behind one of the denied databases — `Spada` or `Networks_1_0` most likely —
  pending Sean's grant. Once granted: rerun the discovery pass in
  `get_etbi_spectra.ps1`, then `Export-Etbi -DbName ... -TblSpectra ...
  -TblMeta ...` drops the CSVs in Downloads.

## Useful one-liners already proven

```powershell
# lot inventory across the network (matches the Shiny-app counts)
Invoke-AqrcQuery $c "SELECT l.LotNumber, COUNT(*) AS n FROM filter.Filters f JOIN filter.Lots l ON l.Id = f.LotId GROUP BY l.LotNumber ORDER BY n DESC"
# (verify the join column names with INFORMATION_SCHEMA first — pattern below)

# what columns does a table have?
Invoke-AqrcQuery $c "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='ftir' AND TABLE_NAME='SampleAnalysis'"

# which tables mention a column like X anywhere?
Invoke-AqrcQuery $c "SELECT TABLE_SCHEMA + '.' + TABLE_NAME AS tbl, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME LIKE '%Lot%'"
```

Everything here is SELECT-only by construction; there is no write path in any
of these snippets. Keep it that way — it's the production database.

## Related sources

- Sean's email (2026-07-02) — the original connection config (ODBC flavor, for
  the R Shiny app), account/PR offer, "we do not track flag history".
- The Shiny app copy (`My Drive/University/Research/Grad/FTIR Calibration/`) —
  authoritative schema documentation in R code; also holds the spectra-service
  token in `global.R` (~line 203). **Never commit that token to the repo.**
- The Shiny app itself (`https://shiny.aqrc.ucdavis.edu/ftir_calibration/`,
  VPN-only) — quickest for eyeballing counts/lots without writing SQL.
