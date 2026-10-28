# Read-only SQL metadata discovery for FTIR functional-group products.
# Paste this file into Windows PowerShell, then run: Find-FtirFgTables
# Set AETHMODULAR_AQRC_CONNECTION_STRING from local configuration first.
# No endpoint, account, credentials, or connection string is written to output.
# SQL is SELECT-only. Local CSVs and a query/manifest receipt are written to a
# fresh timestamped folder under Downloads\ftir_fg_discovery.
# A missing keyword match is not proof that products do not exist: generic
# key/value tables, external lab files, other databases, or hidden objects may
# hold them. The complete visible column inventory is retained for review.

function Find-FtirFgTables {
    [CmdletBinding()]
    param(
        [string]$OutputRoot = (Join-Path $HOME 'Downloads\ftir_fg_discovery')
    )

    $ErrorActionPreference = 'Stop'
    if (-not $env:AETHMODULAR_AQRC_CONNECTION_STRING) {
        throw 'Set AETHMODULAR_AQRC_CONNECTION_STRING from local configuration first.'
    }
    $builder = New-Object System.Data.SqlClient.SqlConnectionStringBuilder
    $builder.set_ConnectionString($env:AETHMODULAR_AQRC_CONNECTION_STRING)
    if ($builder['Initial Catalog'] -ne 'Networks_1_0') {
        throw 'Discovery is scoped to Networks_1_0. Check the local connection configuration.'
    }
    $builder['Connect Timeout'] = 60
    $conn = New-Object System.Data.SqlClient.SqlConnection($builder.ConnectionString)

    function Query-FgMetadata([string]$Sql) {
        $cmd = $conn.CreateCommand()
        $cmd.CommandText = $Sql
        $cmd.CommandTimeout = 600
        $adapter = New-Object System.Data.SqlClient.SqlDataAdapter($cmd)
        try {
            $dt = New-Object System.Data.DataTable
            $adapter.Fill($dt) | Out-Null
            return ,$dt
        } finally {
            $adapter.Dispose()
            $cmd.Dispose()
        }
    }

    function Export-FgMetadata($Table, [string]$Path) {
        $rows = foreach ($r in $Table.Rows) {
            $p = [ordered]@{}
            foreach ($cl in $Table.Columns) {
                $v = $r[$cl]
                if ($v -is [DBNull]) { $v = '' }
                elseif ($v -is [byte[]]) { $v = [Convert]::ToBase64String($v) }
                elseif ($v -is [datetime]) { $v = $v.ToString('o') }
                $p[$cl.ColumnName] = $v
            }
            [pscustomobject]$p
        }
        if ($Table.Rows.Count) {
            $rows | Export-Csv $Path -NoTypeInformation -Encoding UTF8
        } else {
            $header = ($Table.Columns | ForEach-Object {
                '"' + $_.ColumnName.Replace('"', '""') + '"'
            }) -join ','
            Set-Content $Path $header -Encoding UTF8
        }
        if (@(Import-Csv $Path).Count -ne $Table.Rows.Count) {
            throw "CSV row mismatch: $Path"
        }
    }

    $queries = [ordered]@{
        context = @'
SELECT DB_NAME() AS DatabaseName, SYSUTCDATETIME() AS RetrievedUtc,
       HAS_PERMS_BY_NAME(DB_NAME(), 'DATABASE', 'VIEW DEFINITION') AS CanViewDefinition,
       HAS_PERMS_BY_NAME(DB_NAME(), 'DATABASE', 'SELECT') AS CanSelectDatabase,
       IS_MEMBER('db_owner') AS IsDbOwner,
       IS_MEMBER('db_datareader') AS IsDataReader;
'@
        objects = @'
SELECT s.name AS SchemaName, o.name AS ObjectName, o.type_desc AS ObjectType,
       (SELECT SUM(p.rows) FROM sys.partitions p
        WHERE p.object_id = o.object_id AND p.index_id IN (0,1)) AS ApproxRows,
       HAS_PERMS_BY_NAME(QUOTENAME(s.name)+'.'+QUOTENAME(o.name),
                         'OBJECT', 'SELECT') AS CanSelect
FROM sys.objects o JOIN sys.schemas s ON s.schema_id = o.schema_id
WHERE o.is_ms_shipped = 0 AND o.type IN ('U','V','SN','ET')
ORDER BY s.name, o.name;
'@
        columns = @'
SELECT s.name AS SchemaName, o.name AS ObjectName, o.type_desc AS ObjectType,
       c.column_id AS ColumnOrdinal, c.name AS ColumnName, ty.name AS DataType,
       c.max_length AS MaxLengthBytes, c.precision AS NumericPrecision,
       c.scale AS NumericScale, c.is_nullable AS IsNullable
FROM sys.objects o JOIN sys.schemas s ON s.schema_id = o.schema_id
JOIN sys.columns c ON c.object_id = o.object_id
JOIN sys.types ty ON ty.user_type_id = c.user_type_id
WHERE o.is_ms_shipped = 0 AND o.type IN ('U','V','ET')
ORDER BY s.name, o.name, c.column_id;
'@
    }
    # Broad discovery terms; candidate status is a prompt for inspection only.
    $pattern = '(?i)functional|alkane|alkene|aromatic|alcohol|hydrox|carbox|carbonyl|amine|amino|naco|cooh|ach|acoh|organ|carbon|predict|result|report|model|calib|concentr|parameter|species|analyte|component|(^|[^a-z])(fg|om|oc|ec|ch|co)([^a-z]|$)'

    try {
        $conn.Open()
        $context = Query-FgMetadata $queries.context
        $objects = Query-FgMetadata $queries.objects
        $columns = Query-FgMetadata $queries.columns
        $hits = $columns.Clone()
        foreach ($r in $columns.Rows) {
            if (($r.SchemaName + '.' + $r.ObjectName + '.' + $r.ColumnName) -match $pattern) {
                $hits.ImportRow($r)
            }
        }
        $out = Join-Path $OutputRoot ((Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8))
        New-Item -ItemType Directory -Path $out | Out-Null
        Export-FgMetadata $context (Join-Path $out 'context.csv')
        Export-FgMetadata $objects (Join-Path $out 'networks_objects.csv')
        Export-FgMetadata $columns (Join-Path $out 'networks_columns.csv')
        Export-FgMetadata $hits (Join-Path $out 'candidate_columns.csv')
        $queryText = foreach ($name in $queries.Keys) {
            '-- ' + $name
            $queries[$name]
            ''
        }
        $queryText | Set-Content (Join-Path $out 'queries.sql') -Encoding UTF8
        $pattern | Set-Content (Join-Path $out 'candidate_pattern.txt') -Encoding UTF8
        Get-ChildItem $out -File | Sort-Object Name | ForEach-Object {
            [pscustomobject]@{
                File = $_.Name
                Bytes = $_.Length
                SHA256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
            }
        } | Export-Csv (Join-Path $out 'manifest.csv') -NoTypeInformation -Encoding UTF8

        Write-Host '=== Metadata visibility ==='
        $context | Format-Table -AutoSize | Out-Host
        Write-Host '=== All visible tables, views and synonyms (row counts are estimates) ==='
        $objects | Format-Table -AutoSize | Out-Host
        Write-Host '=== Candidate columns (name matches, not confirmed products) ==='
        $hits | Format-Table SchemaName,ObjectName,ColumnName,DataType -AutoSize | Out-Host
        Write-Host ("Objects: {0}; columns: {1}; candidate columns: {2}" -f $objects.Rows.Count,$columns.Rows.Count,$hits.Rows.Count)
        if ($context.Rows[0].CanViewDefinition -ne 1) {
            Write-Warning 'Database-wide VIEW DEFINITION is not granted; hidden objects may be absent.'
        }
        Write-Host 'No SQL data or schema was modified. This search covers the current database and account only.'
        Write-Host "Saved and row-verified CSVs plus query/hash receipt: $out"
        return $out
    } finally {
        $conn.Dispose()
    }
}

Write-Host 'Loaded. Run: Find-FtirFgTables'
