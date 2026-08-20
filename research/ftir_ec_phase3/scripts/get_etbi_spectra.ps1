# Pull the ETBI (Bishoftu) FTIR spectra + metadata from AQRC-SQL.
# PASTE-SAFE: copy this whole file into a PowerShell window on the Windows
# machine that is on the UCD VPN (or run it as a script - both work).
# Connection per Sean Raffuse's 2026-07-02 email: AQRC-SQL:1433, Windows AD
# trusted auth - via .NET's built-in SqlClient, so no ODBC driver is needed
# (the machine we tried had none installed). Everything here is SELECT-only.
#
# Pasting runs the DISCOVERY pass immediately: it lists reachable databases,
# flags spectra/FTIR/site-shaped tables, and counts ETBI rows where it can.
# Then export with the table names discovery found, e.g.:
#
#   Export-Etbi -DbName "Spartan" -TblSpectra "dbo.FtirSpectra" -TblMeta "dbo.Filters"
#
# CSVs land in your Downloads folder.

function Open-AqrcConn([string]$db = "") {
    # .NET's built-in SqlClient: ships with every Windows, no ODBC driver
    # needed, and Integrated Security = the Windows AD auth Sean described
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

function Find-EtbiTables {
    Write-Host "== discovery: connecting to AQRC-SQL =="
    $conn = Open-AqrcConn
    $dbs = (Invoke-AqrcQuery $conn "SELECT name FROM sys.databases WHERE database_id > 4 ORDER BY name").name
    Write-Host "`n-- databases visible to your account --"
    $dbs | ForEach-Object { Write-Host "  $_" }
    $conn.Close()

    foreach ($db in $dbs) {
        try { $dbc = Open-AqrcConn $db } catch { Write-Host "  (no access to $db)"; continue }
        try {
            $hits = Invoke-AqrcQuery $dbc @"
SELECT t.TABLE_SCHEMA + '.' + t.TABLE_NAME AS tbl,
       STRING_AGG(c.COLUMN_NAME, ', ') AS cols
FROM INFORMATION_SCHEMA.TABLES t
JOIN INFORMATION_SCHEMA.COLUMNS c
  ON c.TABLE_SCHEMA = t.TABLE_SCHEMA AND c.TABLE_NAME = t.TABLE_NAME
WHERE t.TABLE_TYPE = 'BASE TABLE'
  AND (t.TABLE_NAME LIKE '%spectr%' OR t.TABLE_NAME LIKE '%ftir%'
       OR t.TABLE_NAME LIKE '%absorb%' OR t.TABLE_NAME LIKE '%scan%'
       OR c.COLUMN_NAME LIKE '%wavenumber%' OR c.COLUMN_NAME LIKE '%SiteCode%'
       OR c.COLUMN_NAME LIKE '%MediaId%')
GROUP BY t.TABLE_SCHEMA, t.TABLE_NAME
"@
        } catch { Write-Host "  (${db}: schema query failed - $($_.Exception.Message))"; $dbc.Close(); continue }
        if ($hits.Rows.Count) {
            Write-Host "`n-- $db : candidate tables --"
            foreach ($r in $hits) {
                Write-Host "  $($r.tbl)"
                Write-Host "      $($r.cols)"
                if ($r.cols -match "SiteCode|(^|, )Site(, |$)") {
                    $col = if ($r.cols -match "SiteCode") { "SiteCode" } else { "Site" }
                    try {
                        $n = (Invoke-AqrcQuery $dbc "SELECT COUNT(*) AS n FROM $($r.tbl) WHERE $col = 'ETBI'").n
                        Write-Host "      -> ETBI rows: $n"
                    } catch {}
                }
            }
        }
        $dbc.Close()
    }
    Write-Host "`nNext: Export-Etbi -DbName <db> -TblSpectra <schema.table> -TblMeta <schema.table>"
}

function Export-Etbi {
    param(
        [Parameter(Mandatory)][string]$DbName,
        [Parameter(Mandatory)][string]$TblSpectra,
        [Parameter(Mandatory)][string]$TblMeta,
        [string]$SiteColumn = "SiteCode",
        [string]$SiteValue = "ETBI"
    )
    $outDir = Join-Path $HOME "Downloads"
    $conn = Open-AqrcConn $DbName
    Write-Host "exporting $SiteValue rows from $DbName ..."
    $meta = Invoke-AqrcQuery $conn "SELECT * FROM $TblMeta WHERE $SiteColumn = '$SiteValue'"
    $meta | Export-Csv (Join-Path $outDir "etbi_metadata_raw.csv") -NoTypeInformation
    Write-Host "  metadata: $($meta.Rows.Count) rows"
    $spectra = Invoke-AqrcQuery $conn @"
SELECT s.* FROM $TblSpectra s
WHERE EXISTS (SELECT 1 FROM $TblMeta m
              WHERE m.$SiteColumn = '$SiteValue' AND m.MediaId = s.MediaId)
"@
    $spectra | Export-Csv (Join-Path $outDir "etbi_spectra_raw.csv") -NoTypeInformation
    Write-Host "  spectra:  $($spectra.Rows.Count) rows"
    $conn.Close()
    Write-Host "wrote etbi_metadata_raw.csv + etbi_spectra_raw.csv to $outDir"
    Write-Host "(long-format spectra are fine - they get pivoted to the ETAD wide shape on the analysis side)"
}

Find-EtbiTables
