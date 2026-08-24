# Pull IMPROVE lot-253 FTIR spectra from AQRC-SQL (Improve_2.1) in batches.
# PASTE-SAFE, pure ASCII. Run on the Windows machine on the UCD VPN.
#
# Purpose: the independent-lot validation frozen in
# research/ftir_ec_phase3/VALIDATION_LAYER_2026-08-24.md needs the lot-253
# scan spectra (~5,050 TOR-EC filters, 83 sites, Apr 2023 - Apr 2024). TOR EC
# for these filters is already local (local_db results_tor.csv); ONLY the
# ftir.Scan blobs are missing. Lot 255 is the reserved replication set - do
# NOT pull it until the 253 protocol has run.
#
# Usage: paste the whole file, then
#     Show-ImproveLots                 # discovery: filters + scans per lot
#     Get-ImproveLot "253"             # full pull, batched (start here)
#     Get-ImproveLot "253" -Pilot      # first 300 analyses only, to validate
#
# Writes into Downloads\improve_lot253\:
#     lot253_analyses.csv                          (one row per FTIR analysis)
#     lot253_scans_base64_partNNN.csv              (500 analyses per part)
# Batching matters: ~5k filters x 2 scans x full resolution is a multi-GB
# pull; per-part CSVs survive an interrupted session (rerun skips done parts).

function Open-AqrcConn([string]$db = "Improve_2.1") {
    $cs = "Server=AQRC-SQL,1433;Integrated Security=SSPI;TrustServerCertificate=True;Database=$db;"
    $conn = New-Object System.Data.SqlClient.SqlConnection($cs)
    $conn.Open()
    return $conn
}

function Invoke-AqrcQuery($conn, [string]$sql) {
    $cmd = $conn.CreateCommand(); $cmd.CommandText = $sql; $cmd.CommandTimeout = 1200
    $table = New-Object System.Data.DataTable
    (New-Object System.Data.SqlClient.SqlDataAdapter($cmd)).Fill($table) | Out-Null
    return $table
}

function Export-AqrcTable($table, [string]$path) {
    if ($table -isnot [System.Data.DataTable]) { $table = @($table)[0].Table }
    $list = foreach ($row in $table.Rows) {
        $props = [ordered]@{}
        foreach ($c in $table.Columns) {
            $v = $row[$c]
            if ($v -is [byte[]]) { $v = [Convert]::ToBase64String($v) }
            elseif ($v -is [System.DBNull]) { $v = "" }
            $props[$c.ColumnName] = $v
        }
        [pscustomobject]$props
    }
    $list | Export-Csv $path -NoTypeInformation -Encoding UTF8
    Write-Host "  wrote $($table.Rows.Count) rows -> $path"
}

function Show-ImproveLots {
    $c = Open-AqrcConn
    Write-Host "== IMPROVE filters + FTIR analyses per lot (filter.Filters x ftir.SampleAnalysis) =="
    Invoke-AqrcQuery $c @"
SELECT l.LotNumber,
       COUNT(DISTINCT f.Id) AS Filters,
       COUNT(DISTINCT sa.Id) AS FtirAnalyses,
       MIN(f.SampleDate) AS FirstSample, MAX(f.SampleDate) AS LastSample
FROM filter.Filters f
JOIN filter.Lots l ON l.Id = f.LotId
LEFT JOIN ftir.SampleAnalysis sa ON sa.FilterId = f.Id
GROUP BY l.LotNumber
ORDER BY FtirAnalyses DESC
"@ | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
    Write-Host "If the join above errors on a column name, run these and adjust:"
    Write-Host "  Invoke-AqrcQuery `$c `"SELECT TOP 3 * FROM filter.Filters`" | Format-List"
    Write-Host "  Invoke-AqrcQuery `$c `"SELECT TOP 3 * FROM filter.Lots`" | Format-List"
    $c.Close()
}

function Get-ImproveLot([Parameter(Mandatory)][string]$Lot, [switch]$Pilot) {
    $out = Join-Path $HOME "Downloads\improve_lot$Lot"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $c = Open-AqrcConn
    Write-Host "== IMPROVE lot $Lot -> $out =="

    $analyses = Invoke-AqrcQuery $c @"
SELECT sa.Id AS AnalysisId, sa.FilterId, sa.SampleScanId, sa.BackgroundScanId,
       sa.AnalysisDate, sa.AnalysisQcCode, sa.InstrumentId, l.LotNumber
FROM ftir.SampleAnalysis sa
JOIN filter.Filters f ON f.Id = sa.FilterId
JOIN filter.Lots l ON l.Id = f.LotId
WHERE l.LotNumber = '$Lot'
ORDER BY sa.Id
"@
    Export-AqrcTable $analyses (Join-Path $out "lot${Lot}_analyses.csv")
    $ids = @($analyses.Rows | ForEach-Object {
        @([int]$_["SampleScanId"], [int]$_["BackgroundScanId"]) }) |
        ForEach-Object { $_ } | Sort-Object -Unique
    Write-Host "  $($analyses.Rows.Count) analyses, $($ids.Count) distinct scans"
    if ($Pilot) {
        $take = [Math]::Min(600, $ids.Count); $ids = $ids[0..($take-1)]
        Write-Host "  PILOT MODE: first $take scan ids only"
    }

    $chunk = 1000     # scans per part (= ~500 analyses)
    for ($i = 0; $i -lt $ids.Count; $i += $chunk) {
        $part = [int]($i / $chunk) + 1
        $path = Join-Path $out ("lot${Lot}_scans_base64_part{0:d3}.csv" -f $part)
        if (Test-Path $path) { Write-Host "  part $part exists, skipping"; continue }
        $hi = [Math]::Min($i + $chunk, $ids.Count) - 1
        $idList = ($ids[$i..$hi] -join ",")
        $scans = Invoke-AqrcQuery $c @"
SELECT Id, NumberOfDataPoints, FrequencyOfFirstPoint, FrequencyOfLastPoint,
       YScalingFactor, DataPointFormat, Data
FROM ftir.Scan WHERE Id IN ($idList)
"@
        Export-AqrcTable $scans $path
    }
    $c.Close()
    Write-Host "== done. Copy the improve_lot$Lot folder to the analysis machine. =="
}
