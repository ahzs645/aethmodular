# Read-only export of the small Networks_1_0 tables the 2026-10-28 SQL survey
# turned up, plus a parameter check on Improve_2.1. PASTE-SAFE, pure ASCII.
# Run on the Windows machine on the UCD VPN, in a session where
# $env:AETHMODULAR_AQRC_CONNECTION_STRING is set (docs/spartan-network-data-pulls.md;
# do not print or commit its value). Nothing is written to the database.
#
# What it pulls (whole tables, all small):
#   ftir.CalibrationSets, ftir.Calibrations   deployed SPARTAN OC/EC calibrations per lot
#                                             (Wavenumber, b): the coefficient vectors
#   csu_amod.Filters, csu_amod.Shipments      Adama CSU AMOD filters: mass difference, volume, lot, flag
#   analysis.Batches, Media, MediaComments,   batch labels and per-filter batch ids; parameter list
#     Parameters, Networks, MediaPurposes,
#     MediaTypes, QcCodes
#   frm.Filters, frm.Sites                    the FRM filter set with FTIR ids
#   spartan.Filters, spartan.Sites            for the joins (also in the 2026-10-19 pull)
# Plus joined/summary views (calibration_summary, spartan_filter_batches,
# batch_site_counts, adama_tor_match) and, from Improve_2.1, the distinct
# parameters in its small lookup tables (does IMPROVE store FTIR functional groups?).
#
# Usage: paste the whole file, then run
#     Get-NetworksExtras
# Optional and slow (a GROUP BY over ~159M rows; run only if the lookup check is unclear):
#     Get-ImproveResultParameters
#
# Writes Downloads\networks_extras_<yyyyMMdd_HHmmss>\ with one CSV per query,
# manifest.csv (file, rows, bytes, SHA256, database, query, UTC time), a ZIP of
# the folder verified entry by entry, and <zip>.sha256.txt. Leave all three
# together; send the ZIP back.

# The five Adama quartz TOR filters (Batch 54; research/ftir_ec_calibration_2026_06_25/README.md)
$AdamaTorIds = @("J1675", "J1679", "J1693", "J1701", "J1703")

function Open-Db([string]$db) {
    if (-not $env:AETHMODULAR_AQRC_CONNECTION_STRING) {
        throw "Set AETHMODULAR_AQRC_CONNECTION_STRING in this Windows session first."
    }
    $b = New-Object System.Data.SqlClient.SqlConnectionStringBuilder($env:AETHMODULAR_AQRC_CONNECTION_STRING)
    $b["Database"] = $db
    if (-not $b.ConnectTimeout -or $b.ConnectTimeout -lt 60) { $b.ConnectTimeout = 60 }
    $c = New-Object System.Data.SqlClient.SqlConnection($b.ConnectionString)
    $c.Open()
    return $c
}

function Q($conn, [string]$sql) {
    $cmd = $conn.CreateCommand()
    $cmd.CommandText = $sql
    $cmd.CommandTimeout = 600
    $dt = New-Object System.Data.DataTable
    (New-Object System.Data.SqlClient.SqlDataAdapter($cmd)).Fill($dt) | Out-Null
    return ,$dt
}

# DataTable -> CSV: base64 for binary, ISO-8601 for dates, empty for NULL; the
# row count is read back and checked. Zero-row results keep their header.
function E($dt, [string]$path) {
    $rows = foreach ($r in $dt.Rows) {
        $p = [ordered]@{}
        foreach ($cl in $dt.Columns) {
            $v = $r[$cl]
            if ($v -is [DBNull]) { $v = "" }
            elseif ($v -is [byte[]]) { $v = [Convert]::ToBase64String($v) }
            elseif ($v -is [datetime]) { $v = $v.ToString("o") }
            elseif ($v -is [double] -or $v -is [single]) { $v = $v.ToString("R", [Globalization.CultureInfo]::InvariantCulture) }
            elseif ($v -is [decimal]) { $v = $v.ToString([Globalization.CultureInfo]::InvariantCulture) }
            $p[$cl.ColumnName] = $v
        }
        [pscustomobject]$p
    }
    if ($dt.Rows.Count) {
        $rows | Export-Csv $path -NoTypeInformation -Encoding UTF8
    } else {
        $header = ($dt.Columns | ForEach-Object { '"' + $_.ColumnName.Replace('"', '""') + '"' }) -join ","
        Set-Content $path $header -Encoding UTF8
    }
    if (@(Import-Csv $path).Count -ne $dt.Rows.Count) { throw "CSV row mismatch: $path" }
}

function Save-Query($conn, [string]$db, [string]$name, [string]$sql, $out, $manifest) {
    $dt = Q $conn $sql
    $path = Join-Path $out ($name + ".csv")
    E $dt $path
    $manifest.Add([pscustomobject]@{
        File = $name + ".csv"; Rows = $dt.Rows.Count; Bytes = (Get-Item $path).Length
        SHA256 = (Get-FileHash $path -Algorithm SHA256).Hash; Database = $db
        Query = ($sql -replace "\s+", " ").Trim(); RetrievedUtc = (Get-Date).ToUniversalTime().ToString("o")
    }) | Out-Null
    Write-Host ("  {0,-34} {1,8} rows" -f $name, $dt.Rows.Count)
    return ,$dt
}

function Get-NetworksExtras {
    $out = Join-Path $HOME ("Downloads\networks_extras_" + (Get-Date -Format yyyyMMdd_HHmmss))
    New-Item $out -ItemType Directory | Out-Null
    $manifest = New-Object System.Collections.ArrayList
    $db = "Networks_1_0"
    $c = Open-Db $db
    try {
        Write-Host "== Networks_1_0: whole tables =="
        $tables = @(
            "ftir.CalibrationSets", "ftir.Calibrations",
            "csu_amod.Filters", "csu_amod.Shipments",
            "analysis.Batches", "analysis.Media", "analysis.MediaComments", "analysis.Parameters",
            "analysis.Networks", "analysis.MediaPurposes", "analysis.MediaTypes", "analysis.QcCodes",
            "frm.Filters", "frm.Sites", "spartan.Filters", "spartan.Sites")
        foreach ($t in $tables) {
            $s, $n = $t.Split(".")
            $sql = "SELECT * FROM [" + $s + "].[" + $n + "]"
            if ($t -eq "ftir.Calibrations") { $sql += " ORDER BY CalibrationSetId, RowNumber" }
            $null = Save-Query $c $db ($s + "_" + $n) $sql $out $manifest
        }

        Write-Host "== Networks_1_0: views for checking =="
        # one row per calibration: how many coefficients, the wavenumber span, and the row numbering
        # (an intercept, if stored, shows up as an extra row outside the spectral grid)
        $sql = @"
SELECT cs.Id AS CalibrationSetId, cs.Parameter, cs.LotNumber, cs.Label, cs.AnalysisQcCode, cs.ProcessedTime,
       COUNT(k.Id) AS Coefficients, MIN(k.RowNumber) AS FirstRow, MAX(k.RowNumber) AS LastRow,
       MIN(k.Wavenumber) AS MinWavenumber, MAX(k.Wavenumber) AS MaxWavenumber,
       SUM(CASE WHEN k.b = 0 THEN 1 ELSE 0 END) AS ZeroCoefficients
FROM ftir.CalibrationSets cs
LEFT JOIN ftir.Calibrations k ON k.CalibrationSetId = cs.Id
GROUP BY cs.Id, cs.Parameter, cs.LotNumber, cs.Label, cs.AnalysisQcCode, cs.ProcessedTime
ORDER BY cs.Id
"@
        $null = Save-Query $c $db "calibration_summary" $sql $out $manifest
        # every registered SPARTAN filter with its batch, and how many FTIR analyses it has
        $sql = @"
SELECT s.SiteCode, f.ExternalFilterId, f.ExternalFilterType, f.ExternalLotId, f.SamplingStartDate,
       f.MassCollectedOnFilter, f.SampleVolume_m3, f.MediaId, m.Label AS MediaLabel, m.Barcode,
       m.BatchId, b.Label AS BatchLabel,
       (SELECT COUNT(*) FROM ftir.SampleAnalysis a WHERE a.MediaId = f.MediaId) AS FtirAnalyses
FROM spartan.Filters f
JOIN spartan.Sites s ON s.Id = f.SiteId
LEFT JOIN analysis.Media m ON m.Id = f.MediaId
LEFT JOIN analysis.Batches b ON b.Id = m.BatchId
ORDER BY s.SiteCode, f.SamplingStartDate
"@
        $null = Save-Query $c $db "spartan_filter_batches" $sql $out $manifest
        $sql = @"
SELECT b.Id AS BatchId, b.Label AS BatchLabel, n.Id AS NetworkId, s.SiteCode,
       COUNT(*) AS Filters, MIN(f.SamplingStartDate) AS FirstSample, MAX(f.SamplingStartDate) AS LastSample
FROM analysis.Media m
LEFT JOIN analysis.Batches b ON b.Id = m.BatchId
LEFT JOIN analysis.Networks n ON n.Id = m.NetworkId
LEFT JOIN spartan.Filters f ON f.MediaId = m.Id
LEFT JOIN spartan.Sites s ON s.Id = f.SiteId
GROUP BY b.Id, b.Label, n.Id, s.SiteCode
ORDER BY b.Id, s.SiteCode
"@
        $null = Save-Query $c $db "batch_site_counts" $sql $out $manifest
        # are the five Adama TOR filters registered? Looked for by filter id, media label and barcode
        $like = ($AdamaTorIds | ForEach-Object { "'%" + $_ + "%'" })
        $where = (@("cf.ExternalFilterId", "m.Label", "m.Barcode") | ForEach-Object {
            $col = $_; ($like | ForEach-Object { $col + " LIKE " + $_ }) -join " OR " }) -join " OR "
        $sql = "SELECT cf.*, m.Id AS AnalysisMediaId, m.Label AS MediaLabel, m.Barcode, b.Label AS BatchLabel " +
            "FROM analysis.Media m LEFT JOIN csu_amod.Filters cf ON cf.MediaId = m.Id " +
            "LEFT JOIN analysis.Batches b ON b.Id = m.BatchId WHERE " + $where
        $adama = Save-Query $c $db "adama_tor_match" $sql $out $manifest
        Write-Host ("  Adama TOR ids: {0} matching rows for {1} ids (expect 5; 0 means they are not registered here)" -f $adama.Rows.Count, $AdamaTorIds.Count)
    } finally {
        $c.Close()
    }

    # Improve_2.1: the distinct parameters in every small table that has a Parameter column
    $db = "Improve_2.1"
    $c = Open-Db $db
    try {
        Write-Host "== Improve_2.1: parameter lookups =="
        $sql = @"
SELECT c.TABLE_SCHEMA, c.TABLE_NAME, SUM(p.rows) AS RowCnt
FROM INFORMATION_SCHEMA.COLUMNS c
JOIN sys.tables t ON t.name = c.TABLE_NAME
JOIN sys.schemas sc ON sc.schema_id = t.schema_id AND sc.name = c.TABLE_SCHEMA
JOIN sys.partitions p ON p.object_id = t.object_id AND p.index_id IN (0, 1)
WHERE c.COLUMN_NAME = 'Parameter'
GROUP BY c.TABLE_SCHEMA, c.TABLE_NAME
"@
        $cands = Q $c $sql
        $parts = foreach ($r in $cands.Rows) {
            if ([int64]$r.RowCnt -le 200000) {
                "SELECT DISTINCT '" + $r.TABLE_SCHEMA + "." + $r.TABLE_NAME + "' AS SourceTable, CAST(Parameter AS nvarchar(255)) AS Parameter FROM [" + $r.TABLE_SCHEMA + "].[" + $r.TABLE_NAME + "]"
            }
        }
        $null = Save-Query $c $db "improve_parameter_tables" "SELECT c.TABLE_SCHEMA, c.TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS c WHERE c.COLUMN_NAME = 'Parameter' ORDER BY 1, 2" $out $manifest
        if ($parts) {
            $null = Save-Query $c $db "improve_parameters" (($parts -join " UNION ") + " ORDER BY Parameter, SourceTable") $out $manifest
        }
    } finally {
        $c.Close()
    }

    $manifest | Export-Csv (Join-Path $out "manifest.csv") -NoTypeInformation -Encoding UTF8
    Write-Host "== packing =="
    $zip = $out + ".zip"
    Compress-Archive -Path (Join-Path $out "*") -DestinationPath $zip
    # read every entry back and compare with the source file's SHA256
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $z = [System.IO.Compression.ZipFile]::OpenRead($zip)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $checked = 0
        foreach ($entry in $z.Entries) {
            $name = $entry.FullName.Replace("\", "/")
            $src = Join-Path $out $name
            $stream = $entry.Open()
            try { $h = -join ($sha.ComputeHash($stream) | ForEach-Object { $_.ToString("X2") }) } finally { $stream.Close() }
            if ($h -ne (Get-FileHash $src -Algorithm SHA256).Hash) { throw "ZIP entry differs from source: $name" }
            $checked++
        }
    } finally {
        $z.Dispose()
    }
    $zh = (Get-FileHash $zip -Algorithm SHA256).Hash
    Set-Content ($zip + ".sha256.txt") ($zh + "  " + (Split-Path $zip -Leaf)) -Encoding ASCII
    Write-Host ("  {0} entries verified; {1}" -f $checked, $zip)
}

# Optional: every parameter actually stored in Improve_2.1 analysis.Results, with counts.
# Slow (a full scan of ~159M rows); the 600 s timeout may need raising.
function Get-ImproveResultParameters {
    $out = Join-Path $HOME ("Downloads\improve_result_parameters_" + (Get-Date -Format yyyyMMdd_HHmmss))
    New-Item $out -ItemType Directory | Out-Null
    $manifest = New-Object System.Collections.ArrayList
    $c = Open-Db "Improve_2.1"
    try {
        $null = Save-Query $c "Improve_2.1" "improve_result_parameters" "SELECT Parameter, COUNT_BIG(*) AS [Rows] FROM analysis.Results GROUP BY Parameter ORDER BY Parameter" $out $manifest
    } finally {
        $c.Close()
    }
    $manifest | Export-Csv (Join-Path $out "manifest.csv") -NoTypeInformation -Encoding UTF8
    Write-Host "  wrote $out"
}
