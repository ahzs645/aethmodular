# Read-only SQL Server discovery. Windows PowerShell 5.1 / SQL Server 2016+.
# Load: . .\inspect_sql_access.ps1
# Run:  Invoke-SqlAccessSurvey                 # metadata, all accessible user DBs
#       Invoke-SqlAccessSurvey -Profile        # bounded aggregate profiles
# Connection: AETHMODULAR_AQRC_CONNECTION_STRING, configured locally.
# This file defines functions only; loading it does not connect or query.

$script:SqlSurveySourcePath = $PSCommandPath

function Quote-SqlSurveyIdentifier([string]$Name) {
    if ([string]::IsNullOrEmpty($Name)) { throw 'SQL identifier must not be empty.' }
    return '[' + $Name.Replace(']', ']]') + ']'
}

function Test-SqlSurveyPattern([string]$Value, [string[]]$Patterns) {
    foreach ($pattern in $Patterns) { if ($Value -like $pattern) { return $true } }
    return $false
}

function Get-SqlSurveyErrorNumber($ErrorRecord) {
    $exception = $ErrorRecord.Exception
    while ($null -ne $exception) {
        if ($exception.PSObject.Properties['Number']) { return $exception.Number }
        $exception = $exception.InnerException
    }
    return $null
}

function Add-SqlSurveyError($State, $Database, $Stage, $Object, $ErrorRecord) {
    # Exception messages can contain endpoints, accounts or connection details.
    # Retain only the exception class and SQL error number, never the message.
    $number = Get-SqlSurveyErrorNumber $ErrorRecord
    $State.Errors.Add([pscustomobject]@{
        Database = $Database; Stage = $Stage; Object = $Object
        SqlErrorNumber = $number
        ExceptionType = $ErrorRecord.Exception.GetType().FullName
        Utc = [datetime]::UtcNow.ToString('o')
    })
    Write-Warning ("{0}: {1} failed for {2}; SQL error {3}. See errors.csv." -f $Database,$Stage,$Object,$number)
}

function Open-SqlSurveyConnection([string]$ConnectionString, [string]$Database) {
    $builder = New-Object System.Data.SqlClient.SqlConnectionStringBuilder
    $builder.set_ConnectionString($ConnectionString)
    if ($Database) { $builder["Initial Catalog"] = $Database }
    $builder["Connect Timeout"] = 60
    $builder["Application Name"] = 'aethmodular SQL discovery'
    # Preserve local authentication/TLS settings. Do not add a certificate override.
    $connection = New-Object System.Data.SqlClient.SqlConnection($builder.ConnectionString)
    try { $connection.Open(); return $connection }
    catch { $connection.Dispose(); throw }
}

function Invoke-SqlSurveyQuery {
    param($Connection, [string]$Sql, $State, [string]$Database,
          [string]$Stage, [string]$Object = '', [int]$Limit = 0, [int]$Timeout = 60)
    $entry = [pscustomobject]@{
        QueryId = ('q{0:d6}' -f ($State.Queries.Count + 1))
        Database = $Database; Stage = $Stage; Object = $Object
        Sql = $Sql; ProfileLimit = $Limit; TimeoutSeconds = $Timeout
        StartedUtc = [datetime]::UtcNow.ToString('o'); FinishedUtc = ''
        Status = 'running'; RowsReturned = $null; SqlErrorNumber = $null
    }
    $State.Queries.Add($entry)
    $command = $Connection.CreateCommand()
    $command.CommandText = $Sql
    $command.CommandTimeout = $Timeout
    if ($Limit -gt 0) {
        $parameter = $command.Parameters.Add('@profile_limit', [System.Data.SqlDbType]::Int)
        $parameter.Value = $Limit
    }
    $adapter = New-Object System.Data.SqlClient.SqlDataAdapter($command)
    try {
        $table = New-Object System.Data.DataTable
        $adapter.Fill($table) | Out-Null
        $entry.Status = 'ok'; $entry.RowsReturned = $table.Rows.Count
        return ,$table
    } catch {
        $entry.Status = 'failed'; $entry.SqlErrorNumber = Get-SqlSurveyErrorNumber $_
        Add-SqlSurveyError $State $Database $Stage $Object $_
        return $null
    } finally {
        $entry.FinishedUtc = [datetime]::UtcNow.ToString('o')
        $adapter.Dispose(); $command.Dispose()
    }
}

function Convert-SqlSurveyRows($Table, [string]$Database) {
    if ($null -eq $Table) { return }
    foreach ($row in $Table.Rows) {
        $values = [ordered]@{ Database = $Database }
        foreach ($column in $Table.Columns) {
            $value = $row[$column]
            if ($value -is [DBNull]) { $value = $null }
            $values[$column.ColumnName] = $value
        }
        [pscustomobject]$values
    }
}

function Export-SqlSurveyCsv($Rows, [string[]]$Columns, [string]$Path) {
    $items = @($Rows | ForEach-Object {
        $values = [ordered]@{}
        foreach ($name in $Columns) {
            $value = $_.$name
            if ($null -eq $value -or $value -is [DBNull]) { $value = '' }
            elseif ($value -is [datetime] -or $value -is [datetimeoffset]) { $value = $value.ToString('o') }
            elseif ($value -is [System.IFormattable]) {
                $value = $value.ToString($null, [cultureinfo]::InvariantCulture)
            }
            $values[$name] = $value
        }
        [pscustomobject]$values
    })
    if ($items.Count) { $items | Export-Csv $Path -NoTypeInformation -Encoding UTF8 }
    else {
        $header = ($Columns | ForEach-Object { '"' + $_.Replace('"','""') + '"' }) -join ','
        Set-Content $Path $header -Encoding UTF8
    }
    $readCount = @(Import-Csv $Path).Count
    if ($readCount -ne $items.Count) {
        throw ('CSV verification failed for {0}: expected {1} rows, read {2}.' -f (Split-Path $Path -Leaf),$items.Count,$readCount)
    }
}

function Get-SqlSurveyColumnPlan($Column, [bool]$CanUnmask) {
    $reason = ''
    $kind = ''
    if ($Column.CanSelectColumn -ne 1) { $reason = 'column_select_not_granted_or_unknown' }
    elseif ($Column.IsHidden) { $reason = 'hidden_column' }
    elseif ($Column.IsComputed) { $reason = 'computed_column_metadata_only' }
    elseif ($Column.EncryptionType) { $reason = 'encrypted_column_metadata_only' }
    elseif ($Column.IsMasked -and -not $CanUnmask) { $reason = 'masked_without_database_unmask' }
    elseif ($Column.BaseType -in @('tinyint','smallint','int','bigint','decimal','numeric','money','smallmoney','float','real','bit')) { $kind = 'numeric' }
    elseif ($Column.BaseType -in @('date','datetime','datetime2','smalldatetime','datetimeoffset','time')) { $kind = 'temporal' }
    elseif ($Column.BaseType -eq 'uniqueidentifier') { $kind = 'identifier' }
    elseif ($Column.BaseType -in @('char','varchar','nchar','nvarchar') -and
            $Column.MaxLengthBytes -gt 0 -and $Column.MaxLengthBytes -le 512) { $kind = 'text_bytes' }
    else { $reason = 'binary_lob_large_text_or_unsupported_type_metadata_only' }
    [pscustomobject]@{ Column = $Column; Kind = $kind; SkipReason = $reason }
}

function New-SqlSurveyProfileQuery {
    param([string]$Schema, [string]$Object, [object[]]$Plans,
          [string[]]$OrderColumns = @(), [switch]$FullScan)
    if (-not $Plans.Count) { throw 'A profile query needs at least one column.' }
    $sourceColumns = @(); $aggregates = @('COUNT_BIG(*) AS [ProfileRows]')
    for ($i = 0; $i -lt $Plans.Count; $i++) {
        $plan = $Plans[$i]
        if ($plan.SkipReason -or -not $plan.Kind) { throw 'Skipped columns cannot be queried.' }
        $column = Quote-SqlSurveyIdentifier $plan.Column.ColumnName
        $sourceColumns += $column
        $aggregates += "COUNT_BIG($column) AS [n$i]"
        $aggregates += "COUNT_BIG(DISTINCT $column) AS [d$i]"
        if ($plan.Kind -eq 'numeric') {
            $expression = $column
            if ($plan.Column.BaseType -eq 'bit') { $expression = "CAST($column AS tinyint)" }
            $style = ''
            if ($plan.Column.BaseType -in @('float','real')) { $style = ',3' }
            elseif ($plan.Column.BaseType -in @('money','smallmoney')) { $style = ',2' }
            # Convert extrema on the server so SQL decimal(38,...) is never
            # coerced into the narrower CLR decimal type or local number format.
            $aggregates += "CONVERT(varchar(100),MIN($expression)$style) AS [lo$i]"
            $aggregates += "CONVERT(varchar(100),MAX($expression)$style) AS [hi$i]"
        } elseif ($plan.Kind -eq 'temporal') {
            $aggregates += "CONVERT(varchar(64),MIN($column),127) AS [lo$i]"
            $aggregates += "CONVERT(varchar(64),MAX($column),127) AS [hi$i]"
        } elseif ($plan.Kind -eq 'text_bytes') {
            $aggregates += "MIN(DATALENGTH($column)) AS [lo$i]"
            $aggregates += "MAX(DATALENGTH($column)) AS [hi$i]"
        }
    }
    $top = ''; $order = ''
    if (-not $FullScan) {
        $top = 'TOP (@profile_limit) '
        if ($OrderColumns.Count) {
            $order = ' ORDER BY ' + (($OrderColumns | ForEach-Object { Quote-SqlSurveyIdentifier $_ }) -join ',')
        }
    }
    $qualified = (Quote-SqlSurveyIdentifier $Schema) + '.' + (Quote-SqlSurveyIdentifier $Object)
    return 'WITH [profile_source] AS (SELECT ' + $top + ($sourceColumns -join ',') +
        ' FROM ' + $qualified + $order + ') SELECT ' + ($aggregates -join ',') + ' FROM [profile_source];'
}

function Convert-SqlSurveyProfileResult {
    param($Row, [object[]]$Plans, [int]$Limit, [switch]$FullScan,
          [string]$Database, [string]$Schema, [string]$Object,
          [string]$Selection, [string]$QueryId)
    $count = [long]$Row.ProfileRows
    $complete = $FullScan -or $count -lt $Limit
    for ($i = 0; $i -lt $Plans.Count; $i++) {
        $plan = $Plans[$i]; $nonNull = [long]$Row["n$i"]; $nulls = $count - $nonNull
        $pct = $null
        if ($count) { $pct = [math]::Round(100.0 * $nulls / $count, 4) }
        $low = $null; $high = $null
        if ($plan.Kind -ne 'identifier') {
            if ($Row["lo$i"] -isnot [DBNull]) { $low = $Row["lo$i"] }
            if ($Row["hi$i"] -isnot [DBNull]) { $high = $Row["hi$i"] }
        }
        $minimum = $null; $maximum = $null; $minBytes = $null; $maxBytes = $null
        if ($plan.Kind -eq 'text_bytes') { $minBytes = $low; $maxBytes = $high }
        else { $minimum = $low; $maximum = $high }
        [pscustomobject]@{
            Database = $Database; SchemaName = $Schema; ObjectName = $Object
            ColumnName = $plan.Column.ColumnName; Status = 'ok'; Reason = ''
            StatisticKind = $plan.Kind; RowsProfiled = $count; NullCount = $nulls
            NullPct = $pct; DistinctNonNull = [long]$Row["d$i"]
            Minimum = $minimum; Maximum = $maximum; MinBytes = $minBytes; MaxBytes = $maxBytes
            CoversAllVisibleRows = [bool]$complete; Selection = $Selection
            QueryId = $QueryId; ProfileLimit = $Limit
        }
    }
}

function Invoke-SqlAccessSurvey {
    [CmdletBinding()]
    param(
        [string[]]$Database = @(),
        [string[]]$Schema = @('*'), [string[]]$Object = @('*'),
        [switch]$Profile, [switch]$FullScan, [switch]$IncludeViews,
        [switch]$IncludeSystemDatabases,
        [ValidateRange(1,1000000)][int]$MaxProfileRows = 10000,
        [ValidateRange(1,100000)][int]$MaxProfileTables = 200,
        [ValidateRange(1,600)][int]$ProfileTimeoutSeconds = 30,
        [string]$OutputRoot = (Join-Path $HOME 'Downloads\sql_access_discovery')
    )
    $ErrorActionPreference = 'Stop'
    if ($FullScan -and -not $Profile) { throw '-FullScan requires -Profile.' }
    if (-not $env:AETHMODULAR_AQRC_CONNECTION_STRING) {
        throw 'Set AETHMODULAR_AQRC_CONNECTION_STRING from local configuration first.'
    }
    $out = Join-Path $OutputRoot ((Get-Date -Format yyyyMMdd_HHmmss) + '_' + [guid]::NewGuid().ToString('N').Substring(0,8))
    New-Item -ItemType Directory -Path $out | Out-Null
    $state = [pscustomobject]@{}
    foreach ($name in @('Databases','Objects','Columns','Keys','ForeignKeys','Profiles','ProfileObjects','Errors','Queries')) {
        $state | Add-Member NoteProperty $name (New-Object 'System.Collections.Generic.List[object]')
    }
    $started = [datetime]::UtcNow.ToString('o'); $status = 'interrupted_or_failed'
    $profiledTables = 0; $bootstrap = $null
    $objectSql = @'
SELECT o.object_id AS ObjectId,s.name AS SchemaName,o.name AS ObjectName,o.type_desc AS ObjectType,
       (SELECT SUM(p.rows) FROM sys.partitions p WHERE p.object_id=o.object_id AND p.index_id IN (0,1)) AS ApproxRows,
       HAS_PERMS_BY_NAME(QUOTENAME(s.name)+'.'+QUOTENAME(o.name),'OBJECT','SELECT') AS CanSelect
FROM sys.objects o JOIN sys.schemas s ON s.schema_id=o.schema_id
WHERE o.is_ms_shipped=0 AND o.type IN ('U','V','ET') ORDER BY s.name,o.name;
'@
    $columnSql = @'
SELECT o.object_id AS ObjectId,s.name AS SchemaName,o.name AS ObjectName,c.column_id AS ColumnOrdinal,
       c.name AS ColumnName,t.name AS DeclaredType,TYPE_NAME(c.system_type_id) AS BaseType,
       c.max_length AS MaxLengthBytes,c.precision AS NumericPrecision,c.scale AS NumericScale,
       c.is_nullable AS IsNullable,c.is_identity AS IsIdentity,c.is_computed AS IsComputed,
       c.is_hidden AS IsHidden,c.encryption_type_desc AS EncryptionType,
       ISNULL(mc.is_masked,0) AS IsMasked,c.collation_name AS CollationName,
       HAS_PERMS_BY_NAME(QUOTENAME(s.name)+'.'+QUOTENAME(o.name),'OBJECT','SELECT',c.name,'COLUMN') AS CanSelectColumn,
       CONVERT(nvarchar(4000),ep.value) AS Description
FROM sys.objects o JOIN sys.schemas s ON s.schema_id=o.schema_id
JOIN sys.columns c ON c.object_id=o.object_id JOIN sys.types t ON t.user_type_id=c.user_type_id
LEFT JOIN sys.masked_columns mc ON mc.object_id=c.object_id AND mc.column_id=c.column_id
LEFT JOIN sys.extended_properties ep ON ep.class=1 AND ep.major_id=c.object_id AND ep.minor_id=c.column_id AND ep.name='MS_Description'
WHERE o.is_ms_shipped=0 AND o.type IN ('U','V','ET') ORDER BY s.name,o.name,c.column_id;
'@
    $keySql = @'
SELECT i.object_id AS ObjectId,s.name AS SchemaName,o.name AS ObjectName,i.name AS KeyName,
       i.is_primary_key AS IsPrimaryKey,i.is_unique AS IsUnique,i.has_filter AS HasFilter,
       ic.key_ordinal AS KeyOrdinal,c.name AS ColumnName
FROM sys.indexes i JOIN sys.objects o ON o.object_id=i.object_id JOIN sys.schemas s ON s.schema_id=o.schema_id
JOIN sys.index_columns ic ON ic.object_id=i.object_id AND ic.index_id=i.index_id
JOIN sys.columns c ON c.object_id=ic.object_id AND c.column_id=ic.column_id
WHERE o.is_ms_shipped=0 AND i.is_unique=1 AND ic.key_ordinal>0 ORDER BY s.name,o.name,i.name,ic.key_ordinal;
'@
    $foreignSql = @'
SELECT fk.parent_object_id AS ObjectId,OBJECT_SCHEMA_NAME(fk.parent_object_id) AS SchemaName,
       OBJECT_NAME(fk.parent_object_id) AS ObjectName,fk.name AS ForeignKeyName,
       fc.constraint_column_id AS KeyOrdinal,COL_NAME(fc.parent_object_id,fc.parent_column_id) AS ColumnName,
       OBJECT_SCHEMA_NAME(fk.referenced_object_id) AS ReferencedSchema,
       OBJECT_NAME(fk.referenced_object_id) AS ReferencedObject,
       COL_NAME(fc.referenced_object_id,fc.referenced_column_id) AS ReferencedColumn,
       fk.is_disabled AS IsDisabled,fk.is_not_trusted AS IsNotTrusted
FROM sys.foreign_keys fk JOIN sys.foreign_key_columns fc ON fc.constraint_object_id=fk.object_id
ORDER BY SchemaName,ObjectName,ForeignKeyName,KeyOrdinal;
'@
    try {
        try { $bootstrap = Open-SqlSurveyConnection $env:AETHMODULAR_AQRC_CONNECTION_STRING '' }
        catch { Add-SqlSurveyError $state '' 'bootstrap_connect' '' $_; throw 'Could not open the locally configured SQL connection. See errors.csv.' }
        $dbTable = Invoke-SqlSurveyQuery $bootstrap @'
SELECT database_id AS DatabaseId,name AS DatabaseName,state_desc AS State,
       HAS_DBACCESS(name) AS HasAccess FROM sys.databases ORDER BY name;
'@ $state '' 'databases'
        $known = @()
        if ($null -ne $dbTable) { $known = @($dbTable.Rows | ForEach-Object { $_.DatabaseName }) }
        foreach ($r in @($dbTable.Rows)) {
            if ($null -eq $r) { continue }
            $selected = ($Database.Count -eq 0 -or $r.DatabaseName -in $Database)
            if ($r.DatabaseId -le 4 -and -not $IncludeSystemDatabases) { $selected = $false }
            $state.Databases.Add([pscustomobject]@{
                Database = $r.DatabaseName; DatabaseId = $r.DatabaseId; State = $r.State
                HasAccess = $r.HasAccess; Selected = $selected; Source = 'sys.databases'
                OpenStatus = 'not_selected'; InspectionStatus = ''; CanSelectDatabase = $null
                CanViewDefinition = $null; CanUnmask = $null
            })
        }
        $requested = @($Database)
        if ($null -eq $dbTable -and -not $requested.Count) { $requested = @($bootstrap.Database) }
        foreach ($name in $requested) {
            if ($name -and $name -notin $known) {
                $state.Databases.Add([pscustomobject]@{
                    Database = $name; DatabaseId = $null; State = 'unknown'; HasAccess = $null
                    Selected = $true; Source = 'requested_or_bootstrap'; OpenStatus = 'not_opened'
                    InspectionStatus = ''; CanSelectDatabase = $null; CanViewDefinition = $null; CanUnmask = $null
                })
                $known += $name
            }
        }
        $bootstrap.Dispose(); $bootstrap = $null
        foreach ($db in $state.Databases) {
            if (-not $db.Selected) { continue }
            if ($db.State -notin @('ONLINE','unknown') -or $db.HasAccess -eq 0) {
                $db.OpenStatus = 'not_accessible_or_not_online'; continue
            }
            $connection = $null
            try {
                try { $connection = Open-SqlSurveyConnection $env:AETHMODULAR_AQRC_CONNECTION_STRING $db.Database }
                catch { $db.OpenStatus = 'connect_failed'; Add-SqlSurveyError $state $db.Database 'connect' '' $_; continue }
                $db.OpenStatus = 'connected'; $db.InspectionStatus = 'running'
                Write-Host ("Inspecting {0}" -f $db.Database)
                $context = Invoke-SqlSurveyQuery $connection @'
SELECT HAS_PERMS_BY_NAME(DB_NAME(),'DATABASE','SELECT') AS CanSelectDatabase,
       HAS_PERMS_BY_NAME(DB_NAME(),'DATABASE','VIEW DEFINITION') AS CanViewDefinition,
       HAS_PERMS_BY_NAME(DB_NAME(),'DATABASE','UNMASK') AS CanUnmask;
'@ $state $db.Database 'visibility'
                if ($null -ne $context -and $context.Rows.Count) {
                    foreach ($name in @('CanSelectDatabase','CanViewDefinition','CanUnmask')) {
                        if ($context.Rows[0][$name] -isnot [DBNull]) { $db.$name = $context.Rows[0][$name] }
                    }
                }
                $objects = @(Convert-SqlSurveyRows (Invoke-SqlSurveyQuery $connection $objectSql $state $db.Database 'objects') $db.Database |
                    Where-Object { (Test-SqlSurveyPattern $_.SchemaName $Schema) -and (Test-SqlSurveyPattern $_.ObjectName $Object) })
                $ids = New-Object 'System.Collections.Generic.HashSet[int]'
                foreach ($item in $objects) { $state.Objects.Add($item); $ids.Add([int]$item.ObjectId) | Out-Null }
                $columns = @(Convert-SqlSurveyRows (Invoke-SqlSurveyQuery $connection $columnSql $state $db.Database 'columns') $db.Database |
                    Where-Object { $ids.Contains([int]$_.ObjectId) })
                foreach ($item in $columns) { $state.Columns.Add($item) }
                $keys = @(Convert-SqlSurveyRows (Invoke-SqlSurveyQuery $connection $keySql $state $db.Database 'keys') $db.Database |
                    Where-Object { $ids.Contains([int]$_.ObjectId) })
                foreach ($item in $keys) { $state.Keys.Add($item) }
                foreach ($item in @(Convert-SqlSurveyRows (Invoke-SqlSurveyQuery $connection $foreignSql $state $db.Database 'foreign_keys') $db.Database)) {
                    if ($ids.Contains([int]$item.ObjectId)) { $state.ForeignKeys.Add($item) }
                }
                foreach ($item in $objects) {
                    $tableStatus = [pscustomobject]@{
                        Database = $db.Database; SchemaName = $item.SchemaName; ObjectName = $item.ObjectName
                        Status = 'metadata_only'; Batches = 0; FailedBatches = 0
                    }
                    $state.ProfileObjects.Add($tableStatus)
                    if (-not $Profile) { continue }
                    if ($item.ObjectType -ne 'USER_TABLE' -and -not ($IncludeViews -and $item.ObjectType -eq 'VIEW')) {
                        $tableStatus.Status = 'skipped_view_or_external_table'; continue
                    }
                    if ($profiledTables -ge $MaxProfileTables) { $tableStatus.Status = 'skipped_profile_table_budget'; continue }
                    $profiledTables++
                    $objectColumns = @($columns | Where-Object { $_.ObjectId -eq $item.ObjectId })
                    $plans = @()
                    foreach ($column in $objectColumns) {
                        $plan = Get-SqlSurveyColumnPlan $column ($db.CanUnmask -eq 1)
                        if ($plan.SkipReason) {
                            $state.Profiles.Add([pscustomobject]@{
                                Database = $db.Database; SchemaName = $item.SchemaName; ObjectName = $item.ObjectName
                                ColumnName = $column.ColumnName; Status = 'skipped'; Reason = $plan.SkipReason
                            })
                        } else { $plans += $plan }
                    }
                    if (-not $plans.Count) { $tableStatus.Status = 'no_supported_selectable_columns'; continue }
                    $orderColumns = @($keys | Where-Object { $_.ObjectId -eq $item.ObjectId -and $_.IsPrimaryKey } |
                        Sort-Object KeyOrdinal | ForEach-Object { $_.ColumnName })
                    foreach ($name in $orderColumns) {
                        if (-not @($objectColumns | Where-Object { $_.ColumnName -eq $name -and $_.CanSelectColumn -eq 1 }).Count) {
                            $orderColumns = @(); break
                        }
                    }
                    $selection = 'unordered_TOP_not_random'
                    if ($orderColumns.Count) { $selection = 'primary_key_ordered_TOP_not_random' }
                    if ($FullScan) { $selection = 'full_scan_current_login' }
                    $limit = $MaxProfileRows
                    if ($FullScan) { $limit = 0 }
                    for ($offset = 0; $offset -lt $plans.Count; $offset += 12) {
                        $batch = @($plans[$offset..([math]::Min($offset+11,$plans.Count-1))])
                        $sql = New-SqlSurveyProfileQuery $item.SchemaName $item.ObjectName $batch $orderColumns -FullScan:$FullScan
                        $result = Invoke-SqlSurveyQuery $connection $sql $state $db.Database 'profile' ($item.SchemaName+'.'+$item.ObjectName) $limit $ProfileTimeoutSeconds
                        $tableStatus.Batches++
                        $queryId = $state.Queries[$state.Queries.Count-1].QueryId
                        if ($null -eq $result) {
                            $tableStatus.FailedBatches++
                            foreach ($plan in $batch) {
                                $state.Profiles.Add([pscustomobject]@{
                                    Database = $db.Database; SchemaName = $item.SchemaName; ObjectName = $item.ObjectName
                                    ColumnName = $plan.Column.ColumnName; Status = 'failed'; Reason = 'query_failed_see_errors'
                                    QueryId = $queryId
                                })
                            }
                        } else {
                            foreach ($record in @(Convert-SqlSurveyProfileResult $result.Rows[0] $batch $limit -FullScan:$FullScan -Database $db.Database -Schema $item.SchemaName -Object $item.ObjectName -Selection $selection -QueryId $queryId)) {
                                $state.Profiles.Add($record)
                            }
                        }
                    }
                    $tableStatus.Status = 'profiled'
                    if ($tableStatus.FailedBatches) { $tableStatus.Status = 'partial_or_failed' }
                }
                $db.InspectionStatus = 'finished'
                if (@($state.Errors | Where-Object { $_.Database -eq $db.Database }).Count) { $db.InspectionStatus = 'finished_with_errors' }
            } finally { if ($null -ne $connection) { $connection.Dispose() } }
        }
        $status = 'complete'
        if ($state.Errors.Count) { $status = 'partial_with_errors' }
    } finally {
        if ($null -ne $bootstrap) { $bootstrap.Dispose() }
        $exports = [ordered]@{
            Databases = @('Database','DatabaseId','State','HasAccess','Selected','Source','OpenStatus','InspectionStatus','CanSelectDatabase','CanViewDefinition','CanUnmask')
            Objects = @('Database','ObjectId','SchemaName','ObjectName','ObjectType','ApproxRows','CanSelect')
            Columns = @('Database','ObjectId','SchemaName','ObjectName','ColumnOrdinal','ColumnName','DeclaredType','BaseType','MaxLengthBytes','NumericPrecision','NumericScale','IsNullable','IsIdentity','IsComputed','IsHidden','EncryptionType','IsMasked','CollationName','CanSelectColumn','Description')
            Keys = @('Database','ObjectId','SchemaName','ObjectName','KeyName','IsPrimaryKey','IsUnique','HasFilter','KeyOrdinal','ColumnName')
            ForeignKeys = @('Database','ObjectId','SchemaName','ObjectName','ForeignKeyName','KeyOrdinal','ColumnName','ReferencedSchema','ReferencedObject','ReferencedColumn','IsDisabled','IsNotTrusted')
            Profiles = @('Database','SchemaName','ObjectName','ColumnName','Status','Reason','StatisticKind','RowsProfiled','NullCount','NullPct','DistinctNonNull','Minimum','Maximum','MinBytes','MaxBytes','CoversAllVisibleRows','Selection','QueryId','ProfileLimit')
            ProfileObjects = @('Database','SchemaName','ObjectName','Status','Batches','FailedBatches')
            Errors = @('Database','Stage','Object','SqlErrorNumber','ExceptionType','Utc')
        }
        foreach ($export in $exports.GetEnumerator()) {
            $name = $export.Key
            Export-SqlSurveyCsv $state.$name $export.Value (Join-Path $out ($name.ToLowerInvariant()+'.csv'))
        }
        $queryLines = @($state.Queries | ForEach-Object { $_ | ConvertTo-Json -Depth 6 -Compress })
        [System.IO.File]::WriteAllLines((Join-Path $out 'queries.jsonl'),[string[]]$queryLines,(New-Object System.Text.UTF8Encoding($false)))
        $sourceHash = $null
        if ($script:SqlSurveySourcePath -and (Test-Path $script:SqlSurveySourcePath)) { $sourceHash = (Get-FileHash $script:SqlSurveySourcePath -Algorithm SHA256).Hash }
        $receipt = [ordered]@{
            Status = $status; StartedUtc = $started; FinishedUtc = [datetime]::UtcNow.ToString('o')
            SourceSHA256 = $sourceHash; DatabaseFilter = $Database; SchemaFilter = $Schema; ObjectFilter = $Object
            Profile = [bool]$Profile; FullScan = [bool]$FullScan; IncludeViews = [bool]$IncludeViews
            IncludeSystemDatabases = [bool]$IncludeSystemDatabases; MaxProfileRows = $MaxProfileRows
            MaxProfileTables = $MaxProfileTables; ProfileTimeoutSeconds = $ProfileTimeoutSeconds
            VisibleDatabases = $state.Databases.Count; Objects = $state.Objects.Count; Columns = $state.Columns.Count
            ProfileRecords = $state.Profiles.Count; Errors = $state.Errors.Count
            Visibility = 'Current login only. Metadata, row security, masking and permissions can limit observations.'
            Consistency = 'Separate queries; not an atomic snapshot. Queries never use NOLOCK.'
            Sampling = 'TOP is not random. A capped profile does not establish population extrema or uniqueness.'
            Values = 'No raw rows, binary scans or text category values exported. Numeric and temporal extrema are aggregates.'
            CsvNulls = 'Empty CSV fields represent unavailable values; use Status/Reason to distinguish skips and failures.'
        }
        $receipt | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $out 'receipt.json') -Encoding UTF8
        @(
            'SQL access survey: '+$status
            'Start with databases.csv, objects.csv and columns.csv.'
            'keys.csv and foreignkeys.csv describe declared joins; they do not prove data integrity.'
            'profiles.csv: ranges/missingness/distinct counts within RowsProfiled only.'
            'CoversAllVisibleRows=False means sampled/capped; TOP is not random.'
            'profileobjects.csv records views, budget limits and objects without supported columns.'
            'errors.csv distinguishes failed reads from empty sources. queries.jsonl preserves exact SQL.'
            'receipt.json records scope and limits; manifest.csv records verified CSV counts and hashes.'
        ) | Set-Content (Join-Path $out 'START_HERE.txt') -Encoding UTF8
        $manifest = @(Get-ChildItem $out -File | Sort-Object Name | ForEach-Object {
            $rows = $null
            if ($_.Extension -eq '.csv') { $rows = @(Import-Csv $_.FullName).Count }
            [pscustomobject]@{ File = $_.Name; Bytes = $_.Length; CsvRows = $rows; SHA256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash }
        })
        Export-SqlSurveyCsv $manifest @('File','Bytes','CsvRows','SHA256') (Join-Path $out 'manifest.csv')
        Write-Host ("Survey {0}: {1} objects, {2} columns, {3} recorded errors." -f $status,$state.Objects.Count,$state.Columns.Count,$state.Errors.Count)
        Write-Host "Saved verified exports: $out"
    }
    return $out
}
