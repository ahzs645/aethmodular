# Pull deployed FTIR results (OC/EC predictions) for ETBI from AQRC-SQL.
# PASTE-SAFE, pure ASCII. Run on the Windows machine on the UCD VPN.
#
# Purpose: Bishoftu has NO organic carbon value of any kind locally, so it
# cannot be placed on the OC/EC or OC/fAbs by-site figures
# (deliverables/adama_summary_2026-08-25). The deployed FTIR OC/EC products
# should live somewhere in Networks_1_0 (results/analysis-style tables whose
# exact names we have never enumerated). This script is discovery-first:
# step 1 lists every table so we learn the real names; step 2 exports any
# candidate rows tied to ETBI media.
#
# Usage: paste the whole file, then run IN ORDER:
#     Show-NetworksTables            # step 1: table census (small, fast)
#     Get-EtbiResults                # step 2: export ETBI rows from candidates
#
# Step 1 prints every table with its row count and flags likely candidates
# (names containing result / predict / calib / carbon / oc / ec). If step 2
# reports "no candidate matched", send me the step-1 listing and I will name
# the exact tables for a follow-up one-liner.
#
# Writes into Downloads\etbi_results\:
#     networks_tables.csv                    (the census, always)
#     etbi_<schema>_<table>.csv              (one per candidate with ETBI rows)

function Open-AqrcConn([string]$db = "Networks_1_0") {
    $cs = "Server=AQRC-SQL,1433;Integrated Security=SSPI;TrustServerCertificate=True;Database=$db;"
    $conn = New-Object System.Data.SqlClient.SqlConnection($cs)
    $conn.Open()
    return $conn
}

function Invoke-AqrcQuery($conn, [string]$sql) {
    $cmd = $conn.CreateCommand(); $cmd.CommandText = $sql; $cmd.CommandTimeout = 600
    $table = New-Object System.Data.DataTable
    (New-Object System.Data.SqlClient.SqlDataAdapter($cmd)).Fill($table) | Out-Null
    return $table
}

# DataTable -> CSV. The .Table recovery matters: piping DataRows directly
# makes PowerShell drop the columns and emit junk.
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

function Show-NetworksTables {
    $c = Open-AqrcConn
    $sql = @"
SELECT s.name AS SchemaName, t.name AS TableName, p.rows AS RowCnt
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id
JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0,1)
ORDER BY s.name, t.name
"@
    $tbl = Invoke-AqrcQuery $c $sql
    $out = Join-Path $HOME "Downloads\etbi_results"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Export-AqrcTable $tbl (Join-Path $out "networks_tables.csv")
    Write-Host ""
    Write-Host "== all tables in Networks_1_0 =="
    foreach ($r in $tbl.Rows) {
        $name = "$($r.SchemaName).$($r.TableName)"
        $flag = ""
        if ($name -match "result|predict|calib|carbon|report") { $flag = "   <-- CANDIDATE" }
        Write-Host ("  {0,-45} {1,10} rows{2}" -f $name, $r.RowCnt, $flag)
    }
    $c.Close()
    Write-Host ""
    Write-Host "Now run: Get-EtbiResults"
}

function Get-EtbiResults {
    $c = Open-AqrcConn
    $out = Join-Path $HOME "Downloads\etbi_results"
    New-Item -ItemType Directory -Force -Path $out | Out-Null

    # ETBI media + analysis ids (the join keys any results table could use)
    $media = Invoke-AqrcQuery $c @"
SELECT f.MediaId, f.ExternalFilterId
FROM spartan.Filters f
JOIN spartan.Sites s ON f.SiteId = s.Id
WHERE s.SiteCode = 'ETBI'
"@
    $ana = Invoke-AqrcQuery $c @"
SELECT a.Id AS SampleAnalysisId, a.MediaId
FROM ftir.SampleAnalysis a
WHERE a.MediaId IN (SELECT f.MediaId FROM spartan.Filters f
                    JOIN spartan.Sites s ON f.SiteId = s.Id
                    WHERE s.SiteCode = 'ETBI')
"@
    Export-AqrcTable $media (Join-Path $out "etbi_media_keys.csv")
    Export-AqrcTable $ana   (Join-Path $out "etbi_analysis_keys.csv")
    $mediaIds = ($media.Rows | ForEach-Object { $_.MediaId }) -join ","
    $anaIds   = ($ana.Rows   | ForEach-Object { $_.SampleAnalysisId }) -join ","
    if ($mediaIds.Length -eq 0) { Write-Host "no ETBI media found - stop"; return }

    # every candidate table, probed for a usable key column
    $tables = Invoke-AqrcQuery $c @"
SELECT s.name AS SchemaName, t.name AS TableName
FROM sys.tables t JOIN sys.schemas s ON t.schema_id = s.schema_id
WHERE t.name LIKE '%result%' OR t.name LIKE '%predict%'
   OR t.name LIKE '%calib%'  OR t.name LIKE '%carbon%'
   OR t.name LIKE '%report%'
"@
    $hits = 0
    foreach ($r in $tables.Rows) {
        $full = "[$($r.SchemaName)].[$($r.TableName)]"
        $cols = Invoke-AqrcQuery $c @"
SELECT c.name FROM sys.columns c
JOIN sys.tables t ON c.object_id = t.object_id
JOIN sys.schemas s ON t.schema_id = s.schema_id
WHERE s.name = '$($r.SchemaName)' AND t.name = '$($r.TableName)'
"@
        $colNames = $cols.Rows | ForEach-Object { $_.name }
        $key = $null; $ids = $null
        if ($colNames -contains "MediaId") { $key = "MediaId"; $ids = $mediaIds }
        elseif ($colNames -contains "SampleAnalysisId") { $key = "SampleAnalysisId"; $ids = $anaIds }
        elseif ($colNames -contains "AnalysisId") { $key = "AnalysisId"; $ids = $anaIds }
        if ($null -eq $key) {
            Write-Host "  $full : no MediaId/SampleAnalysisId/AnalysisId column, skipped"
            continue
        }
        try {
            $rows = Invoke-AqrcQuery $c "SELECT * FROM $full WHERE $key IN ($ids)"
            if ($rows.Rows.Count -gt 0) {
                $dest = Join-Path $out ("etbi_{0}_{1}.csv" -f $r.SchemaName, $r.TableName)
                Export-AqrcTable $rows $dest
                $hits += 1
            } else {
                Write-Host "  $full : keyed on $key, 0 ETBI rows"
            }
        } catch {
            Write-Host "  $full : query failed ($($_.Exception.Message))"
        }
    }
    $c.Close()
    if ($hits -eq 0) {
        Write-Host ""
        Write-Host "No candidate table held ETBI rows. Send networks_tables.csv"
        Write-Host "back and the exact tables will be named for a follow-up."
    } else {
        Write-Host ""
        Write-Host "Done: $hits table(s) exported into Downloads\etbi_results\."
        Write-Host "Copy that folder to the Mac (Downloads) and hand it back."
    }
}

Write-Host "Loaded. Run: Show-NetworksTables   then   Get-EtbiResults"
