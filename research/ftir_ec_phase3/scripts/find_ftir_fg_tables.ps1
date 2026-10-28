# Does Networks_1_0 hold FTIR functional-group results? Read-only discovery.
# PASTE-SAFE, pure ASCII. Run on the Windows machine on the UCD VPN.
#
# Why: the gallery has SPARTAN functional groups (aCH, aCOH, COOH, naCO) only
# from the lab's batch 2-3 deliverable (2020-01 to 2023-03; 1,473 filters at 24
# sites). Public OM PM2.5 exists for 3,532 filters, and the batch 2-3 OM equals
# the public OM, so the groups were computed for the other ~2,370 filters too.
# The known schema (AQRC_DB_NOTES.md) has FTIR spectra and analyses
# (ftir.SampleAnalysis, ftir.Scan, ftir.CalibrationSets) but no results table.
# This checks that properly before asking the lab for the deliverables.
#
# Usage: paste the whole file, then run
#     Find-FtirFgTables
#
# It lists every table in every schema with its row count, and every column
# whose name looks like a functional group, OM or a prediction/result. Nothing
# is changed. Writes into Downloads\ftir_fg_discovery\:
#     tables.csv     every table, with row counts
#     columns.csv    the matching columns (schema, table, column, type)
# Send both back; if a results table turns up, the export is a one-liner.

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
    return ,$table
}

function Export-AqrcTable($table, [string]$path) {
    $list = foreach ($row in $table.Rows) {
        $props = [ordered]@{}
        foreach ($c in $table.Columns) {
            $v = $row[$c]
            if ($v -is [System.DBNull]) { $v = "" }
            $props[$c.ColumnName] = $v
        }
        [pscustomobject]$props
    }
    $list | Export-Csv $path -NoTypeInformation -Encoding UTF8
    Write-Host "  wrote $($table.Rows.Count) rows -> $path"
}

function Find-FtirFgTables {
    $out = Join-Path $HOME "Downloads\ftir_fg_discovery"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $c = Open-AqrcConn
    try {
        $tables = Invoke-AqrcQuery $c @"
SELECT s.name AS SchemaName, t.name AS TableName, SUM(p.rows) AS RowCnt
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id
JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0,1)
GROUP BY s.name, t.name
ORDER BY s.name, t.name
"@
        Export-AqrcTable $tables (Join-Path $out "tables.csv")
        $cols = Invoke-AqrcQuery $c @"
SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE COLUMN_NAME LIKE '%aCH%' OR COLUMN_NAME LIKE '%aCOH%' OR COLUMN_NAME LIKE '%COOH%'
   OR COLUMN_NAME LIKE '%naCO%' OR COLUMN_NAME LIKE '%alkane%' OR COLUMN_NAME LIKE '%alcohol%'
   OR COLUMN_NAME LIKE '%carbox%' OR COLUMN_NAME LIKE '%carbonyl%' OR COLUMN_NAME LIKE '%functional%'
   OR COLUMN_NAME LIKE 'OM%' OR COLUMN_NAME LIKE '%Predict%' OR COLUMN_NAME LIKE '%Concentration%'
   OR COLUMN_NAME LIKE '%Parameter%' OR COLUMN_NAME LIKE '%Species%'
   OR TABLE_NAME LIKE '%Result%' OR TABLE_NAME LIKE '%Predict%' OR TABLE_NAME LIKE '%Parameter%'
ORDER BY TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME
"@
        Export-AqrcTable $cols (Join-Path $out "columns.csv")
        Write-Host ""
        Write-Host "== tables outside hips/spartan with rows =="
        foreach ($r in $tables.Rows) {
            if ($r.SchemaName -notin @("hips", "spartan") -and $r.RowCnt -gt 0) {
                Write-Host ("  {0,-45} {1,10} rows" -f "$($r.SchemaName).$($r.TableName)", $r.RowCnt)
            }
        }
        Write-Host ""
        Write-Host "== candidate columns =="
        foreach ($r in $cols.Rows) {
            Write-Host ("  {0}.{1}.{2} ({3})" -f $r.TABLE_SCHEMA, $r.TABLE_NAME, $r.COLUMN_NAME, $r.DATA_TYPE)
        }
    } finally {
        $c.Close()
    }
}
