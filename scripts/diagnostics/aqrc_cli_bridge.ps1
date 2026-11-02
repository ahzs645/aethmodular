# Windows PowerShell 5.1+. Defines functions when dot-sourced; no policy changes.
# Requests select metadata operations or a parser-validated, single SELECT.
[CmdletBinding()]
param(
    [string]$TokenFile = (Join-Path $PSScriptRoot 'bridge-token.txt'),
    [ValidateRange(1024,65535)][int]$Port = 8765,
    [string[]]$AllowedDatabase = @('Networks_1_0','Improve_2.1'),
    [ValidateRange(0,240)][int]$LifetimeMinutes = 120
)

. (Join-Path $PSScriptRoot 'aqrc_read_queries.ps1')

function Get-AqrcCliSqlErrorNumber($ErrorRecord) {
    $exception = $ErrorRecord.Exception
    while ($null -ne $exception) {
        if ($exception.PSObject.Properties['Number']) { return $exception.Number }
        $exception = $exception.InnerException
    }
    return $null
}

function Assert-AqrcCli($Condition, [string]$Code) {
    if (-not $Condition) { throw [ArgumentException]::new($Code) }
}

function Get-AqrcCliFailure($ErrorRecord) {
    $code = 'operation_failed'
    # Match our explicit validation codes, not arbitrary ArgumentException text.
    $safe = @('unknown_operation','invalid_name','database_not_allowed','invalid_limit',
              'select_1_to_20_columns','duplicate_columns','column_not_visible','unsupported_column',
              'invalid_sql_length','sql_parser_unavailable','sql_parser_hash_mismatch','sql_parse_error',
              'single_select_required','sql_write_not_allowed','sql_hints_not_allowed','sql_too_complex',
              'sql_table_source_not_allowed','sql_function_target_not_allowed','sql_remote_source_not_allowed',
              'invalid_query_bound','invalid_query_parameters','unsupported_query_value','too_many_query_columns')
    if ($ErrorRecord.Exception.Message -cin $safe) { $code = $ErrorRecord.Exception.Message }
    $failure=@{ ok=$false; error=$code; sqlErrorNumber=(Get-AqrcCliSqlErrorNumber $ErrorRecord) }
    if ($code -eq 'sql_parse_error' -and $ErrorRecord.Exception.Data.Contains('parseErrors')) {
        $failure.parseErrors=$ErrorRecord.Exception.Data['parseErrors']
    }
    return $failure
}

function Get-AqrcCliName($Value) {
    Assert-AqrcCli ($Value -is [string] -and $Value.Length -gt 0 -and $Value.Length -le 128) 'invalid_name'
    return $Value
}

function Quote-AqrcCliName([string]$Name) { return '[' + $Name.Replace(']', ']]') + ']' }

function New-AqrcCliPlan($Request, [string[]]$Allowed) {
    $operation = [string]$Request.operation
    Assert-AqrcCli ($operation -cin @('probe','databases','tables','views','synonyms','objects','columns','preview')) 'unknown_operation'
    $database = ''
    if ($operation -ne 'databases') {
        $database = Get-AqrcCliName $Request.database
        Assert-AqrcCli ($database -cin $Allowed) 'database_not_allowed'
    }
    $limit = 25
    if ($null -ne $Request.limit) {
        Assert-AqrcCli ($Request.limit -is [int] -or $Request.limit -is [long]) 'invalid_limit'
        $limit = [long]$Request.limit
    }
    $max = 1000
    if ($operation -eq 'preview') { $max = 200 }
    Assert-AqrcCli ($limit -ge 1 -and $limit -le $max) 'invalid_limit'
    $parameters = @{ '@fetch' = [int]($limit + 1) }
    $sql = ''
    switch ($operation) {
        'probe' { $sql = 'SELECT DB_NAME() AS DatabaseName, CAST(1 AS int) AS Connected;' }
        'databases' {
            $sql = 'SELECT TOP (@fetch) name AS DatabaseName FROM sys.databases WHERE database_id > 4 AND state = 0 AND HAS_DBACCESS(name) = 1 ORDER BY name;'
        }
        'tables' {
            $sql = 'SELECT TOP (@fetch) s.name AS SchemaName, t.name AS TableName FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id WHERE t.is_ms_shipped=0 AND t.is_external=0 ORDER BY s.name,t.name;'
        }
        'views' {
            $sql = 'SELECT TOP (@fetch) s.name AS SchemaName,v.name AS ViewName FROM sys.views v JOIN sys.schemas s ON s.schema_id=v.schema_id WHERE v.is_ms_shipped=0 ORDER BY s.name,v.name;'
        }
        'synonyms' {
            $sql = 'SELECT TOP (@fetch) s.name AS SchemaName,n.name AS SynonymName,n.base_object_name AS BaseObjectName FROM sys.synonyms n JOIN sys.schemas s ON s.schema_id=n.schema_id ORDER BY s.name,n.name;'
        }
        'objects' {
            $sql = 'SELECT TOP (@fetch) s.name AS SchemaName,o.name AS ObjectName,o.type_desc AS ObjectType,n.base_object_name AS BaseObjectName FROM sys.objects o JOIN sys.schemas s ON s.schema_id=o.schema_id LEFT JOIN sys.synonyms n ON n.object_id=o.object_id WHERE o.is_ms_shipped=0 AND o.type IN (''U'',''V'',''SN'') ORDER BY s.name,o.name;'
        }
        { $_ -in @('columns','preview') } {
            $parameters['@schema'] = Get-AqrcCliName $Request.schema
            $parameters['@table'] = Get-AqrcCliName $Request.table
            $sql = @'
SELECT TOP (@fetch) c.name AS ColumnName, c.column_id AS Ordinal,
 TYPE_NAME(c.system_type_id) AS BaseType, c.max_length AS MaxLength,
 c.is_nullable AS IsNullable, c.is_computed AS IsComputed,
 c.is_hidden AS IsHidden, c.encryption_type AS EncryptionType
FROM sys.columns c JOIN sys.objects t ON t.object_id=c.object_id
JOIN sys.schemas s ON s.schema_id=t.schema_id
WHERE s.name=@schema AND t.name=@table AND t.is_ms_shipped=0 AND t.type IN ('U','V')
 AND NOT EXISTS (SELECT 1 FROM sys.external_tables e WHERE e.object_id=t.object_id)
ORDER BY c.column_id;
'@
        }
    }
    $selected = @()
    if ($operation -eq 'preview') {
        $selected = @($Request.columns)
        Assert-AqrcCli ($selected.Count -ge 1 -and $selected.Count -le 20) 'select_1_to_20_columns'
        foreach ($column in $selected) { $null = Get-AqrcCliName $column }
        Assert-AqrcCli (@($selected | Sort-Object -Unique).Count -eq $selected.Count) 'duplicate_columns'
    }
    return [pscustomobject]@{
        Operation=$operation; Database=$database; Limit=$limit; Sql=$sql
        Parameters=$parameters; Columns=$selected
    }
}

function Open-AqrcCliConnection([string]$ConnectionString, [string]$Database) {
    $builder = New-Object System.Data.SqlClient.SqlConnectionStringBuilder
    $builder.set_ConnectionString($ConnectionString)
    if ($Database) { $builder['Initial Catalog'] = $Database }
    $builder['Connect Timeout'] = 60
    $builder['Application Name'] = 'aethmodular bounded CLI reads'
    # Authentication and TLS settings are inherited unchanged from local config.
    $connection = New-Object System.Data.SqlClient.SqlConnection($builder.ConnectionString)
    try { $connection.Open(); return $connection }
    catch { $connection.Dispose(); throw }
}

function Invoke-AqrcCliSql($Connection, [string]$Sql, [hashtable]$Parameters) {
    $command = $Connection.CreateCommand()
    $command.CommandText = $Sql
    $command.CommandTimeout = 30
    foreach ($key in $Parameters.Keys) {
        $value = $Parameters[$key]
        if ($value -is [int]) { $p = $command.Parameters.Add($key, [Data.SqlDbType]::Int) }
        else { $p = $command.Parameters.Add($key, [Data.SqlDbType]::NVarChar, 128) }
        $p.Value = $value
    }
    $reader = $null
    try {
        $reader = $command.ExecuteReader()
        $rows = New-Object 'System.Collections.Generic.List[object]'
        $columns = @(); for ($i=0; $i -lt $reader.FieldCount; $i++) { $columns += $reader.GetName($i) }
        while ($reader.Read()) {
            $row = [ordered]@{}
            for ($i=0; $i -lt $reader.FieldCount; $i++) {
                $value = $reader.GetValue($i)
                if ($value -is [DBNull]) { $value = $null }
                $row[$columns[$i]] = $value
            }
            $rows.Add([pscustomobject]$row)
        }
        return [pscustomobject]@{ Columns=$columns; Rows=$rows.ToArray() }
    } finally {
        if ($null -ne $reader) { $reader.Dispose() }
        $command.Dispose()
    }
}

function New-AqrcCliPreviewSql($Plan, [object[]]$Metadata) {
    $expressions = @()
    foreach ($name in $Plan.Columns) {
        $match = @($Metadata | Where-Object { $_.ColumnName -ceq $name })
        Assert-AqrcCli ($match.Count -eq 1) 'column_not_visible'
        $c = $match[0]
        Assert-AqrcCli (-not $c.IsComputed -and -not $c.IsHidden -and $null -eq $c.EncryptionType) 'unsupported_column'
        Assert-AqrcCli ($c.BaseType -in @('tinyint','smallint','int','bigint','bit','decimal','numeric','money','smallmoney','float','real','char','varchar','nchar','nvarchar','date','datetime','datetime2','smalldatetime','datetimeoffset','time','uniqueidentifier')) 'unsupported_column'
        $quoted = Quote-AqrcCliName $name
        $style = ''
        if ($c.BaseType -in @('float','real')) { $style = ',3' }
        elseif ($c.BaseType -in @('money','smallmoney')) { $style = ',2' }
        elseif ($c.BaseType -in @('date','datetime','datetime2','smalldatetime','datetimeoffset','time')) { $style = ',127' }
        $expressions += "CONVERT(nvarchar(512),$quoted$style) AS $quoted"
    }
    $qualified = (Quote-AqrcCliName $Plan.Parameters['@schema']) + '.' + (Quote-AqrcCliName $Plan.Parameters['@table'])
    return 'SELECT TOP (@fetch) ' + ($expressions -join ',') + ' FROM ' + $qualified + ';'
}

function Invoke-AqrcCliOperation($Request, [string]$ConnectionString, [string[]]$Allowed) {
    if ($Request.operation -ceq 'capabilities') {
        $available=$true
        try { Initialize-AqrcScriptDom } catch { $available=$false }
        return [pscustomobject]@{ok=$true;operation='capabilities';protocolVersion=2;
            operations=@('probe','databases','tables','views','synonyms','objects','columns','preview','query','capabilities');
            allowedDatabases=$Allowed;queryAvailable=$available;queryMaxRows=10000;queryMaxChars=1048576;
            queryMaxTimeoutSeconds=120;requestMaxBytes=131072;responseMaxBytes=33554432;
            sqlPolicy='one SELECT; ScriptDom AST validation; no writes, EXEC, ad hoc external sources or hints'}
    }
    if ($Request.operation -ceq 'query') { return Invoke-AqrcReadQuery $Request $ConnectionString $Allowed }
    $plan = New-AqrcCliPlan $Request $Allowed
    $connection = Open-AqrcCliConnection $ConnectionString $plan.Database
    try {
        if ($plan.Operation -eq 'preview') {
            # A table can have up to 1024 columns; metadata itself is bounded.
            $metadataParameters = $plan.Parameters.Clone()
            $metadataParameters['@fetch'] = 1025
            $metadata = Invoke-AqrcCliSql $connection $plan.Sql $metadataParameters
            $sql = New-AqrcCliPreviewSql $plan @($metadata.Rows)
            $result = Invoke-AqrcCliSql $connection $sql @{ '@fetch' = [int]($plan.Limit+1) }
        } else { $result = Invoke-AqrcCliSql $connection $plan.Sql $plan.Parameters }
        $all = @($result.Rows)
        $more = $all.Count -gt $plan.Limit
        $rows = @($all | Select-Object -First $plan.Limit)
        return [pscustomobject]@{
            ok=$true; protocolVersion=2; operation=$plan.Operation; database=$connection.Database
            columns=@($result.Columns); rows=$rows; rowCount=$rows.Count
            limit=$plan.Limit; hasMore=$more
            selection=$(if ($plan.Operation -eq 'preview') { 'unordered_bounded_preview' } else { 'metadata_or_probe' })
            valuesAsText=($plan.Operation -eq 'preview')
            cellTextLimit=$(if ($plan.Operation -eq 'preview') { 512 } else { $null })
        }
    } finally { $connection.Dispose() }
}

function Read-AqrcCliLine($Stream, [int]$MaxBytes = 131072) {
    $buffer = New-Object IO.MemoryStream
    try {
        for ($i=0; $i -le $MaxBytes; $i++) {
            $value = $Stream.ReadByte()
            if ($value -eq 10) {
                return ([Text.UTF8Encoding]::new($false, $true)).GetString($buffer.ToArray())
            }
            if ($value -lt 0 -or $i -eq $MaxBytes) { throw 'invalid_request_frame' }
            $buffer.WriteByte([byte]$value)
        }
    } finally { $buffer.Dispose() }
}

function Test-AqrcCliToken([string]$Candidate, [byte[]]$ExpectedHash) {
    if ($Candidate.Length -gt 256) { return $false }
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $actual = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Candidate)) }
    finally { $sha.Dispose() }
    $difference = 0
    for ($i=0; $i -lt 32; $i++) { $difference = $difference -bor ($actual[$i] -bxor $ExpectedHash[$i]) }
    return $difference -eq 0
}

function Start-AqrcCliBridge {
    param([string]$TokenFile, [int]$Port=8765, [string[]]$AllowedDatabase=@('Networks_1_0','Improve_2.1'), [int]$LifetimeMinutes=120)
    $ErrorActionPreference = 'Stop'
    Assert-AqrcCli ($Port -ge 1024 -and $Port -le 65535 -and $LifetimeMinutes -ge 0 -and $LifetimeMinutes -le 240) 'invalid_listener_settings'
    Assert-AqrcCli ([bool]$env:AETHMODULAR_AQRC_CONNECTION_STRING) 'connection_not_configured_in_this_tab'
    $token = [IO.File]::ReadAllText($TokenFile).Trim()
    Assert-AqrcCli ($token -cmatch '^[A-Za-z0-9_-]{43}$') 'invalid_token_file'
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $tokenHash = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($token)) }
    finally { $sha.Dispose() }
    $token = $null
    $connectionString = $env:AETHMODULAR_AQRC_CONNECTION_STRING
    # No listener starts until an actual SELECT succeeds using existing settings.
    try {
        $probe = Invoke-AqrcCliOperation ([pscustomobject]@{ operation='probe'; database=$AllowedDatabase[0] }) $connectionString $AllowedDatabase
    } catch {
        throw ('Database probe failed; SQL error number: ' + (Get-AqrcCliSqlErrorNumber $_))
    }
    Write-Host ('AQRC database probe succeeded: ' + $probe.database)
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
    $deadline = $null
    if ($LifetimeMinutes -gt 0) { $deadline = [datetime]::UtcNow.AddMinutes($LifetimeMinutes) }
    try {
        $listener.Start(8)
        $lifetimeMessage = 'Runs until Ctrl+C.'
        if ($null -ne $deadline) { $lifetimeMessage = 'Stops after ' + $LifetimeMinutes + ' minutes or Ctrl+C.' }
        Write-Host ('AQRC CLI ready on loopback port ' + $Port + '. ' + $lifetimeMessage)
        while ($null -eq $deadline -or [datetime]::UtcNow -lt $deadline) {
            if (-not $listener.Pending()) { Start-Sleep -Milliseconds 200; continue }
            $client = $listener.AcceptTcpClient()
            $stream = $null
            try {
                $stream = $client.GetStream()
                $stream.ReadTimeout = 10000; $stream.WriteTimeout = 10000
                $response = @{ ok=$false; error='invalid_request' }
                try {
                    $request = Read-AqrcCliLine $stream | ConvertFrom-Json
                    if (-not (Test-AqrcCliToken ([string]$request.token) $tokenHash)) {
                        $response = @{ ok=$false; error='unauthorized' }
                    } else {
                        try { $response = Invoke-AqrcCliOperation $request $connectionString $AllowedDatabase }
                        catch {
                            # Never return exception text or the connection string.
                            $response = Get-AqrcCliFailure $_
                        }
                    }
                } catch { $response = @{ ok=$false; error='invalid_request' } }
                $json = ConvertTo-Json -InputObject $response -Depth 12 -Compress
                $bytes = [Text.Encoding]::UTF8.GetBytes($json + "`n")
                if ($bytes.Length -gt 33554432) { $bytes = [Text.Encoding]::UTF8.GetBytes('{"ok":false,"error":"response_too_large"}' + "`n") }
                $stream.Write($bytes, 0, $bytes.Length)
            } catch { Write-Host 'A client disconnected or exceeded the request timeout.' }
            finally {
                if ($null -ne $stream) { $stream.Dispose() }
                $client.Dispose()
            }
        }
    } finally { $listener.Stop(); $connectionString=$null }
}

if ($MyInvocation.InvocationName -ne '.') {
    Start-AqrcCliBridge -TokenFile $TokenFile -Port $Port -AllowedDatabase $AllowedDatabase -LifetimeMinutes $LifetimeMinutes
}
