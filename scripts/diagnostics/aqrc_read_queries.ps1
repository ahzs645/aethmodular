# Shared read-query policy and bounded result streaming. Loading defines functions
# only. Microsoft's pinned ScriptDom dependency is loaded before accepting SQL.
$script:AqrcReadSupportRoot = $PSScriptRoot
$script:AqrcScriptDomHashes = @{
    net472 = 'e138533c4b8fc51d7346ee511650b68ea58a2ff5f0fef8e943b9e173d6abad55'
    'netstandard2.0' = 'b74a080ea142c51831ce5c21d7d376f99bce423d1fb695ae025c3adee48eaf74'
}

function Get-AqrcScriptDomRoot {
    if ($env:AETHMODULAR_AQRC_SCRIPTDOM_DIR) { return $env:AETHMODULAR_AQRC_SCRIPTDOM_DIR }
    return Join-Path $script:AqrcReadSupportRoot 'lib/ScriptDom'
}

function Get-AqrcScriptDomFramework {
    if ($PSVersionTable.PSEdition -eq 'Desktop') { return 'net472' }
    return 'netstandard2.0'
}

function Install-AqrcScriptDom {
    $framework = Get-AqrcScriptDomFramework
    $root = Get-AqrcScriptDomRoot
    $dll = Join-Path (Join-Path $root $framework) 'Microsoft.SqlServer.TransactSql.ScriptDom.dll'
    if (Test-Path -LiteralPath $dll -PathType Leaf) {
        Assert-AqrcCli ((Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash -eq $script:AqrcScriptDomHashes[$framework]) 'sql_parser_hash_mismatch'
        return
    }
    $null = New-Item -ItemType Directory -Path $root -Force
    $package = Join-Path $root 'scriptdom-180.107.0.nupkg'
    try {
        Invoke-WebRequest -UseBasicParsing -Uri 'https://api.nuget.org/v3-flatcontainer/microsoft.sqlserver.transactsql.scriptdom/180.107.0/microsoft.sqlserver.transactsql.scriptdom.180.107.0.nupkg' -OutFile $package
        Assert-AqrcCli ((Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash -eq 'd815856f46486ffb576096da86bd7613f80d3e3f333be17bc6d05c2eb913e21a') 'sql_parser_hash_mismatch'
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip = [IO.Compression.ZipFile]::OpenRead($package)
        try {
            $entry = $zip.GetEntry('lib/' + $framework + '/Microsoft.SqlServer.TransactSql.ScriptDom.dll')
            Assert-AqrcCli ($null -ne $entry) 'sql_parser_unavailable'
            $null = New-Item -ItemType Directory -Path (Split-Path $dll) -Force
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dll, $false)
        } finally { $zip.Dispose() }
        Assert-AqrcCli ((Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash -eq $script:AqrcScriptDomHashes[$framework]) 'sql_parser_hash_mismatch'
    } finally {
        if (Test-Path -LiteralPath $package) { Remove-Item -LiteralPath $package }
    }
}

function Initialize-AqrcScriptDom {
    if ($null -ne $script:AqrcScriptDomAssembly) { return }
    $framework = Get-AqrcScriptDomFramework
    $dll = Join-Path (Join-Path (Get-AqrcScriptDomRoot) $framework) 'Microsoft.SqlServer.TransactSql.ScriptDom.dll'
    Assert-AqrcCli (Test-Path -LiteralPath $dll -PathType Leaf) 'sql_parser_unavailable'
    Assert-AqrcCli ((Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash -eq $script:AqrcScriptDomHashes[$framework]) 'sql_parser_hash_mismatch'
    # Load normally; do not remove download markers or bypass a load restriction.
    $script:AqrcScriptDomAssembly = [Reflection.Assembly]::LoadFrom($dll)
}

function Assert-AqrcReadSql([string]$Sql, [string[]]$Allowed) {
    Assert-AqrcCli ($Sql.Length -gt 0 -and $Sql.Length -le 32768) 'invalid_sql_length'
    Initialize-AqrcScriptDom
    $asm = $script:AqrcScriptDomAssembly
    $fragmentType = $asm.GetType('Microsoft.SqlServer.TransactSql.ScriptDom.TSqlFragment')
    $tableType = $asm.GetType('Microsoft.SqlServer.TransactSql.ScriptDom.TableReference')
    $parser = [Activator]::CreateInstance($asm.GetType('Microsoft.SqlServer.TransactSql.ScriptDom.TSql160Parser'), @($true))
    $reader = [IO.StringReader]::new($Sql)
    $errors = $null
    try { $tree = $parser.Parse($reader, [ref]$errors) } finally { $reader.Dispose() }
    if ($errors.Count) {
        $exception=[ArgumentException]::new('sql_parse_error')
        $exception.Data['parseErrors']=@($errors | Select-Object -First 10 | ForEach-Object {
            [pscustomobject]@{number=$_.Number;line=$_.Line;column=$_.Column}
        })
        throw $exception
    }
    Assert-AqrcCli ($tree.Batches.Count -eq 1 -and $tree.Batches[0].Statements.Count -eq 1) 'single_select_required'
    $statement = $tree.Batches[0].Statements[0]
    Assert-AqrcCli ($statement.GetType().Name -eq 'SelectStatement') 'single_select_required'
    Assert-AqrcCli ($null -eq $statement.Into) 'sql_write_not_allowed'
    Assert-AqrcCli ($statement.OptimizerHints.Count -eq 0) 'sql_hints_not_allowed'
    $tables = @('NamedTableReference','QueryDerivedTable','InlineDerivedTable','QualifiedJoin',
        'UnqualifiedJoin','JoinParenthesisTableReference','OdbcQualifiedJoinTableReference',
        'PivotedTableReference','UnpivotedTableReference','OpenJsonTableReference','FullTextTableReference')
    $pending = New-Object 'System.Collections.Generic.Stack[object]'
    $seen = New-Object 'System.Collections.Generic.HashSet[object]'
    $pending.Push($statement)
    while ($pending.Count) {
        $node = $pending.Pop()
        if (-not $seen.Add($node)) { continue }
        Assert-AqrcCli ($seen.Count -le 20000) 'sql_too_complex'
        $kind = $node.GetType().Name
        Assert-AqrcCli ($kind -notin @('NextValueForExpression','SelectSetVariable')) 'sql_write_not_allowed'
        if ($tableType.IsInstanceOfType($node)) {
            Assert-AqrcCli ($kind -in $tables) 'sql_table_source_not_allowed'
        }
        if ($kind -eq 'NamedTableReference') {
            Assert-AqrcCli ($node.TableHints.Count -eq 0) 'sql_hints_not_allowed'
        }
        if ($kind -eq 'FunctionCall') {
            Assert-AqrcCli ($null -eq $node.CallTarget) 'sql_function_target_not_allowed'
        }
        if ($kind -eq 'SchemaObjectName') {
            Assert-AqrcCli ($null -eq $node.ServerIdentifier) 'sql_remote_source_not_allowed'
            if ($null -ne $node.DatabaseIdentifier) {
                Assert-AqrcCli ($node.DatabaseIdentifier.Value -cin $Allowed) 'database_not_allowed'
            }
        }
        # Traverse only AST children, never the repeated token stream.
        foreach ($property in $node.GetType().GetProperties()) {
            if ($property.Name -eq 'ScriptTokenStream' -or $property.GetIndexParameters().Length) { continue }
            $value = $property.GetValue($node, $null)
            if ($null -eq $value) { continue }
            if ($fragmentType.IsInstanceOfType($value)) { $pending.Push($value) }
            elseif ($value -is [Collections.IEnumerable] -and $value -isnot [string]) {
                foreach ($item in $value) {
                    if ($null -ne $item -and $fragmentType.IsInstanceOfType($item)) { $pending.Push($item) }
                }
            }
        }
    }
}

function Get-AqrcReadBound($Request, [string]$Name, [int]$Default, [int]$Maximum) {
    $value = $Request.$Name
    if ($null -eq $value) { return $Default }
    Assert-AqrcCli (($value -is [int] -or $value -is [long]) -and $value -ge 1 -and $value -le $Maximum) 'invalid_query_bound'
    return [int]$value
}

function Add-AqrcReadParameters($Command, $Parameters) {
    if ($null -eq $Parameters) { return }
    Assert-AqrcCli ($Parameters -is [pscustomobject] -or $Parameters -is [Collections.IDictionary]) 'invalid_query_parameters'
    $entries = @()
    if ($Parameters -is [Collections.IDictionary]) {
        foreach ($key in $Parameters.Keys) { $entries += [pscustomobject]@{Name=$key;Value=$Parameters[$key]} }
    } else { $entries = @($Parameters.PSObject.Properties) }
    Assert-AqrcCli ($entries.Count -le 64) 'invalid_query_parameters'
    foreach ($entry in $entries) {
        $key = [string]$entry.Name; $value = $entry.Value
        Assert-AqrcCli ($key -cmatch '^@[A-Za-z_][A-Za-z0-9_]{0,127}$' -and $key -notmatch '^@__aqrc_') 'invalid_query_parameters'
        if ($null -eq $value) { $p=$Command.Parameters.Add($key,[Data.SqlDbType]::NVarChar,-1); $value=[DBNull]::Value }
        elseif ($value -is [bool]) { $p=$Command.Parameters.Add($key,[Data.SqlDbType]::Bit) }
        elseif ($value -is [int] -or $value -is [long]) { $p=$Command.Parameters.Add($key,[Data.SqlDbType]::BigInt) }
        elseif ($value -is [double] -or $value -is [single]) {
            Assert-AqrcCli (-not [double]::IsNaN($value) -and -not [double]::IsInfinity($value)) 'invalid_query_parameters'
            $p=$Command.Parameters.Add($key,[Data.SqlDbType]::Float)
        }
        elseif ($value -is [string]) {
            Assert-AqrcCli ($value.Length -le 16384) 'invalid_query_parameters'
            $p=$Command.Parameters.Add($key,[Data.SqlDbType]::NVarChar,-1)
        }
        else { throw [ArgumentException]::new('invalid_query_parameters') }
        $p.Value=$value
    }
}

function Convert-AqrcReadCell($Reader, [int]$Index, [int]$MaxChars) {
    if ($Reader.IsDBNull($Index)) { return [pscustomobject]@{Value=$null;Truncated=$false;Length=0;Encoding='text'} }
    $kind = $Reader.GetDataTypeName($Index).ToLowerInvariant()
    if ($kind -in @('char','varchar','nchar','nvarchar','text','ntext','xml')) {
        $length = $Reader.GetChars($Index,0,$null,0,0)
        $count = [int][Math]::Min($length,$MaxChars)
        $buffer = New-Object char[] $count
        $read = 0
        while ($read -lt $count) {
            $n = $Reader.GetChars($Index,$read,$buffer,$read,$count-$read)
            if ($n -eq 0) { break }; $read += [int]$n
        }
        if ($read -gt 0 -and [char]::IsHighSurrogate($buffer[$read-1])) { $read-- }
        $value = ''
        if ($read -gt 0) { $value = -join $buffer[0..($read-1)] }
        return [pscustomobject]@{Value=$value;Truncated=($length -gt $read);Length=$length;Encoding='text'}
    }
    if ($kind -in @('binary','varbinary','image','timestamp','rowversion')) {
        $length = $Reader.GetBytes($Index,0,$null,0,0)
        $count = [int][Math]::Min($length,$MaxChars)
        $buffer = New-Object byte[] $count
        $read = 0
        while ($read -lt $count) {
            $n = $Reader.GetBytes($Index,$read,$buffer,$read,$count-$read)
            if ($n -eq 0) { break }; $read += [int]$n
        }
        return [pscustomobject]@{Value=[Convert]::ToBase64String($buffer,0,$read);Truncated=($length -gt $read);Length=$length;Encoding='base64'}
    }
    if ($kind -in @('decimal','numeric')) { $value = $Reader.GetSqlDecimal($Index).ToString() }
    else {
        $value = $Reader.GetValue($Index)
        if ($value -is [datetime] -or $value -is [datetimeoffset]) { $value=$value.ToString('o') }
        elseif ($value -is [double]) { $value=$value.ToString('G17',[cultureinfo]::InvariantCulture) }
        elseif ($value -is [single]) { $value=$value.ToString('G9',[cultureinfo]::InvariantCulture) }
        elseif ($value -is [System.IFormattable]) { $value=$value.ToString($null,[cultureinfo]::InvariantCulture) }
        elseif ($value -is [bool] -or $value -is [guid] -or $value -is [timespan]) { $value=$value.ToString() }
        else { throw [ArgumentException]::new('unsupported_query_value') }
    }
    return [pscustomobject]@{Value=[string]$value;Truncated=$false;Length=([string]$value).Length;Encoding='text'}
}

function Read-AqrcQueryRows($Reader, [int]$Limit, [int]$MaxChars, [int]$ByteBudget=33500000) {
    Assert-AqrcCli ($Reader.FieldCount -ge 1 -and $Reader.FieldCount -le 256) 'too_many_query_columns'
    $names = @(); $original=@(); $types=@(); $encoding=@()
    $used = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    for ($i=0; $i -lt $Reader.FieldCount; $i++) {
        $name=$Reader.GetName($i); $original += $name
        if (-not $name) { $name='column_'+($i+1) }
        $base=$name; $suffix=2
        while (-not $used.Add($name)) { $name=$base+'__'+$suffix; $suffix++ }
        $typeName=$Reader.GetDataTypeName($i)
        $names += $name; $types += $typeName
        if ($typeName.ToLowerInvariant() -in @('binary','varbinary','image','timestamp','rowversion')) { $encoding += 'base64' }
        else { $encoding += 'text' }
    }
    $rows=New-Object 'System.Collections.Generic.List[object]'
    $truncated=New-Object 'System.Collections.Generic.List[object]'
    $truncatedCount=0; $spent=0; $more=$false; $budgetReached=$false
    while ($Reader.Read()) {
        if ($rows.Count -eq $Limit) { $more=$true; break }
        $row=[ordered]@{}; $rowCost=32; $rowTruncated=@()
        for ($i=0; $i -lt $names.Count; $i++) {
            # Six bytes per UTF-16 unit bounds even fully escaped JSON strings.
            $remaining=[int][Math]::Floor(($ByteBudget-$spent-$rowCost)/6)
            if ($remaining -le 0) { $budgetReached=$true; break }
            $cell=Convert-AqrcReadCell $Reader $i ([int][Math]::Min($MaxChars,$remaining))
            if ($remaining -lt $MaxChars -and $cell.Truncated) { $budgetReached=$true; break }
            $row[$names[$i]]=$cell.Value
            $rowCost += 6*($names[$i].Length+([string]$cell.Value).Length)+32
            if ($cell.Truncated) {
                $rowTruncated += [pscustomobject]@{row=$rows.Count;column=$names[$i];originalLength=$cell.Length;returnedTextLength=([string]$cell.Value).Length;encoding=$cell.Encoding}
            }
        }
        if ($budgetReached -or $spent+$rowCost -gt $ByteBudget) { $budgetReached=$true; $more=$true; break }
        $spent += $rowCost; $rows.Add([pscustomobject]$row)
        $truncatedCount += $rowTruncated.Count
        foreach ($item in $rowTruncated) { if ($truncated.Count -lt 1000) { $truncated.Add($item) } }
    }
    return [pscustomobject]@{Columns=$names;OriginalColumns=$original;Types=$types;Encodings=$encoding;Rows=$rows.ToArray();HasMore=$more;ByteBudgetReached=$budgetReached;TruncatedCells=$truncated.ToArray();TruncatedCellCount=$truncatedCount}
}

function Invoke-AqrcReadQuery($Request,[string]$ConnectionString,[string[]]$Allowed) {
    $database=Get-AqrcCliName $Request.database
    Assert-AqrcCli ($database -cin $Allowed) 'database_not_allowed'
    $limit=Get-AqrcReadBound $Request 'limit' 100 10000
    $maxChars=Get-AqrcReadBound $Request 'max_chars' 65536 1048576
    $timeout=Get-AqrcReadBound $Request 'timeout' 30 120
    Assert-AqrcCli ($Request.sql -is [string]) 'invalid_sql_length'
    Assert-AqrcReadSql $Request.sql $Allowed
    $connection=Open-AqrcCliConnection $ConnectionString $database
    $command=$null; $reader=$null
    try {
        $command=$connection.CreateCommand()
        $command.CommandTimeout=$timeout
        $command.CommandText='SET NOCOUNT ON; SET ROWCOUNT '+($limit+1)+";`n"+$Request.sql
        Add-AqrcReadParameters $command $Request.parameters
        $reader=$command.ExecuteReader([Data.CommandBehavior]::SequentialAccess)
        $result=Read-AqrcQueryRows $reader $limit $maxChars
        return [pscustomobject]@{
            ok=$true;operation='query';database=$database;columns=@($result.Columns)
            originalColumns=@($result.OriginalColumns);columnTypes=@($result.Types);encodings=@($result.Encodings)
            rows=@($result.Rows);rowCount=@($result.Rows).Count;limit=$limit;hasMore=$result.HasMore
            byteBudgetReached=$result.ByteBudgetReached;valuesAsText=$true;cellTextLimit=$maxChars
            truncatedCells=@($result.TruncatedCells);truncatedCellCount=$result.TruncatedCellCount
            truncationDetailsCapped=($result.TruncatedCellCount -gt 1000);timeoutSeconds=$timeout
        }
    } finally {
        if ($null -ne $reader) { $reader.Dispose() }
        if ($null -ne $command) { $command.Dispose() }
        $connection.Dispose()
    }
}
