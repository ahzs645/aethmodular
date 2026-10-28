# Offline fixtures: no database or credentials are needed.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../../scripts/diagnostics/inspect_sql_access.ps1')
. (Join-Path $PSScriptRoot '../../research/ftir_ec_phase3/scripts/find_ftir_fg_tables.ps1')

function Assert-Survey($Condition, [string]$Message) {
    if (-not $Condition) { throw "Assertion failed: $Message" }
}
function New-FixtureTable([object[]]$Records) {
    $table = New-Object System.Data.DataTable
    foreach ($name in $Records[0].PSObject.Properties.Name) { $table.Columns.Add($name,[object]) | Out-Null }
    foreach ($record in $Records) {
        $row = $table.NewRow()
        foreach ($column in $table.Columns) {
            $value = $record.($column.ColumnName)
            if ($null -eq $value) { $value = [DBNull]::Value }
            $row[$column.ColumnName] = $value
        }
        $table.Rows.Add($row)
    }
    return ,$table
}
function New-FixtureColumn([string]$Name, [string]$Type = 'decimal', [int]$ObjectId = 10) {
    [pscustomobject]@{
        ObjectId = $ObjectId; SchemaName = 'lab'; ObjectName = 'Measures'; ColumnOrdinal = 1
        ColumnName = $Name; DeclaredType = $Type; BaseType = $Type; MaxLengthBytes = 40
        NumericPrecision = 38; NumericScale = 8; IsNullable = $true; IsIdentity = $false
        IsComputed = $false; IsHidden = $false; EncryptionType = $null; IsMasked = $false
        CollationName = $null; CanSelectColumn = 1; Description = $null
    }
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('aeth-sql-survey-test-'+[guid]::NewGuid().ToString('N'))
New-Item $tempRoot -ItemType Directory | Out-Null
try {
    # Use the real connection-string builder, but intercept SqlConnection so
    # authentication/TLS/catalog handling is tested without opening a socket.
    # PowerShell adapts builder property assignment as dictionary-key writes;
    # assigning .ConnectionString used to fail before any SQL connection.
    $script:ProbeConnectionString = ''
    $script:ProbeStopOnOpen = $false
    function New-Object {
        param([string]$TypeName, [object[]]$ArgumentList)
        if ($TypeName.StartsWith('System.Data.SqlClient.SqlConnection') -and
            -not $TypeName.StartsWith('System.Data.SqlClient.SqlConnectionStringBuilder')) {
            $script:ProbeConnectionString = [string]$ArgumentList[0]
            $probe = [pscustomobject]@{ WasOpened=$false; WasDisposed=$false }
            $probe | Add-Member ScriptMethod Open {
                $this.WasOpened = $true
                if ($script:ProbeStopOnOpen) { throw 'fixture-stop-after-builder' }
            }
            $probe | Add-Member ScriptMethod Dispose { $this.WasDisposed = $true }
            return $probe
        }
        Microsoft.PowerShell.Utility\New-Object @PSBoundParameters
    }
    try {
        $fixtureConnection = 'Data Source=fixture;Initial Catalog=Seed;Integrated Security=True;Encrypt=True;TrustServerCertificate=False;Connect Timeout=4'
        foreach ($catalog in @('', 'Other')) {
            $probe = Open-SqlSurveyConnection $fixtureConnection $catalog
            Assert-Survey $probe.WasOpened 'connection opened through the intercepted transport'
            $parsed = New-Object System.Data.SqlClient.SqlConnectionStringBuilder -ArgumentList $script:ProbeConnectionString
            $expected = 'Seed'; if ($catalog) { $expected = $catalog }
            Assert-Survey ($parsed['Initial Catalog'] -eq $expected) 'bootstrap catalog preserved or explicitly overridden'
            Assert-Survey ($parsed['Connect Timeout'] -eq 60) 'connection timeout uses a valid keyword'
            Assert-Survey ($parsed['Application Name'] -eq 'aethmodular SQL discovery') 'application name uses a valid keyword'
            Assert-Survey ($parsed['Integrated Security'] -and $parsed['Encrypt'] -and -not $parsed['TrustServerCertificate']) 'authentication and TLS settings preserved'
            $probe.Dispose()
            Assert-Survey $probe.WasDisposed 'probe connection disposed'
        }
        # The older FG helper had the same adapted-property regression. Stop
        # after construction so this uses the real builder without SQL transport.
        $savedProbeEnvironment = $env:AETHMODULAR_AQRC_CONNECTION_STRING
        try {
            $env:AETHMODULAR_AQRC_CONNECTION_STRING = $fixtureConnection.Replace('Initial Catalog=Seed','Initial Catalog=Networks_1_0')
            $script:ProbeStopOnOpen = $true
            try { Find-FtirFgTables | Out-Null; throw 'Expected the intercepted open to stop.' }
            catch { Assert-Survey ($_.Exception.ToString().Contains('fixture-stop-after-builder')) 'FG discovery reaches connection opening with the corrected builder' }
            $parsed = New-Object System.Data.SqlClient.SqlConnectionStringBuilder -ArgumentList $script:ProbeConnectionString
            Assert-Survey ($parsed['Initial Catalog'] -eq 'Networks_1_0' -and $parsed['Connect Timeout'] -eq 60) 'FG catalog and timeout use supported builder operations'
            Assert-Survey ($parsed['Encrypt'] -and -not $parsed['TrustServerCertificate']) 'FG helper preserves TLS verification'
        } finally {
            $env:AETHMODULAR_AQRC_CONNECTION_STRING = $savedProbeEnvironment
            $script:ProbeStopOnOpen = $false
        }
    } finally { Remove-Item Function:\New-Object }

    Assert-Survey ((Quote-SqlSurveyIdentifier 'a]b') -eq '[a]]b]') 'bracket escaping'
    $numeric = Get-SqlSurveyColumnPlan (New-FixtureColumn 'EC]value') $false
    $sql = New-SqlSurveyProfileQuery 'lab]schema' 'Daily]results' @($numeric) @('Id')
    Assert-Survey ($sql.Contains('[lab]]schema].[Daily]]results]')) 'schema/object quoting'
    Assert-Survey ($sql.Contains('[EC]]value]')) 'column quoting'
    Assert-Survey ($sql.Contains('TOP (@profile_limit)')) 'parameterized row bound'
    Assert-Survey ($sql.Contains('ORDER BY [Id]')) 'stable primary-key selection'
    Assert-Survey ($sql.Contains('CONVERT(varchar(100),MIN(')) 'decimal extrema are strings'
    $full = New-SqlSurveyProfileQuery 'lab' 'Measures' @($numeric) @('Id') -FullScan
    Assert-Survey ($full -notmatch 'TOP|ORDER BY|NOLOCK') 'full scan is explicit and has no dirty-read hint'
    $bit = Get-SqlSurveyColumnPlan (New-FixtureColumn 'Flag' 'bit') $false
    Assert-Survey ((New-SqlSurveyProfileQuery 'lab' 't' @($bit)) -match 'MIN\(CAST\(\[Flag\] AS tinyint\)\)') 'bit uses a supported aggregate'
    $money = Get-SqlSurveyColumnPlan (New-FixtureColumn 'Cost' 'money') $false
    Assert-Survey ((New-SqlSurveyProfileQuery 'lab' 't' @($money)).Contains('MIN([Cost]),2)')) 'money retains four decimal places'
    $textColumn = New-FixtureColumn 'Label' 'nvarchar'
    $text = Get-SqlSurveyColumnPlan $textColumn $false
    $textSql = New-SqlSurveyProfileQuery 'lab' 't' @($text)
    Assert-Survey ($textSql.Contains('DATALENGTH([Label])') -and -not $textSql.Contains('MIN([Label])')) 'no text values are exported'
    foreach ($type in @('varbinary','image','xml','sql_variant')) {
        Assert-Survey ((Get-SqlSurveyColumnPlan (New-FixtureColumn 'Payload' $type) $false).SkipReason) "skip unsupported $type"
    }
    $textColumn.MaxLengthBytes = -1
    Assert-Survey ((Get-SqlSurveyColumnPlan $textColumn $false).SkipReason) 'LOB text skipped'
    $masked = New-FixtureColumn 'Masked'; $masked.IsMasked = $true
    Assert-Survey ((Get-SqlSurveyColumnPlan $masked $false).SkipReason -eq 'masked_without_database_unmask') 'masked values cannot become false ranges'
    Assert-Survey (-not (Get-SqlSurveyColumnPlan $masked $true).SkipReason) 'explicit effective UNMASK permits profiling'
    $denied = New-FixtureColumn 'Denied'; $denied.CanSelectColumn = 0
    Assert-Survey ((Get-SqlSurveyColumnPlan $denied $false).SkipReason) 'column permissions respected'

    $result = New-FixtureTable @([pscustomobject]@{ProfileRows=3;n0=2;d0=2;lo0='-2';hi0='9999999999999999999999999999.12345678'})
    $converted = @(Convert-SqlSurveyProfileResult $result.Rows[0] @($numeric) 3 -Database 'Fixture')
    Assert-Survey ($converted[0].NullCount -eq 1 -and $converted[0].NullPct -eq 33.3333) 'null denominator is rows actually profiled'
    Assert-Survey (-not $converted[0].CoversAllVisibleRows) 'reaching the cap never implies full coverage'
    Assert-Survey ($converted[0].Maximum -eq '9999999999999999999999999999.12345678') 'decimal(38) precision retained'
    $converted = @(Convert-SqlSurveyProfileResult $result.Rows[0] @($numeric) 4)
    Assert-Survey ($converted[0].CoversAllVisibleRows) 'fewer rows than cap covers visible rows'
    $empty = New-FixtureTable @([pscustomobject]@{ProfileRows=0;n0=0;d0=0;lo0=$null;hi0=$null})
    $converted = @(Convert-SqlSurveyProfileResult $empty.Rows[0] @($numeric) 3)
    Assert-Survey ($null -eq $converted[0].NullPct -and $null -eq $converted[0].Minimum) 'empty is not zero-percent missing or a zero range'
    $oldCulture = [threading.thread]::CurrentThread.CurrentCulture
    try {
        [threading.thread]::CurrentThread.CurrentCulture = [cultureinfo]'de-DE'
        $csv = Join-Path $tempRoot 'culture.csv'
        Export-SqlSurveyCsv @([pscustomobject]@{Name="quoted,`"value`"`nnext";Value=12.5}) @('Name','Value') $csv
        $read = @(Import-Csv $csv)
        Assert-Survey ($read.Count -eq 1 -and $read[0].Value -eq '12.5') 'CSV invariant numbers and multiline quoting'
        Export-SqlSurveyCsv @() @('Name','Value') (Join-Path $tempRoot 'empty.csv')
        Assert-Survey ((Get-Content (Join-Path $tempRoot 'empty.csv')) -eq '"Name","Value"') 'empty exports preserve headers'
    } finally { [threading.thread]::CurrentThread.CurrentCulture = $oldCulture }

    # Exercise the whole exporter with a synthetic accessible/denied/system DB
    # inventory, including profiles, a skipped view, a binary field and budgets.
    $script:Opened = @(); $script:Disposed = @(); $script:FailColumns = $false
    function Open-SqlSurveyConnection([string]$ConnectionString,[string]$Database) {
        $script:Opened += $Database
        $connection = [pscustomobject]@{Database=$Database}
        if (-not $Database) { $connection.Database = 'Fixture' }
        $connection | Add-Member ScriptMethod Dispose { $script:Disposed += $this.Database }
        return $connection
    }
    function Invoke-SqlSurveyQuery {
        param($Connection,[string]$Sql,$State,[string]$Database,[string]$Stage,[string]$Object='', [int]$Limit=0,[int]$Timeout=60)
        $State.Queries.Add([pscustomobject]@{QueryId=('q'+$State.Queries.Count);Sql=$Sql;Stage=$Stage;Status='ok'})
        switch ($Stage) {
            'databases' { return ,(New-FixtureTable @(
                [pscustomobject]@{DatabaseId=1;DatabaseName='master';State='ONLINE';HasAccess=1},
                [pscustomobject]@{DatabaseId=5;DatabaseName='Fixture';State='ONLINE';HasAccess=1},
                [pscustomobject]@{DatabaseId=6;DatabaseName='Denied';State='ONLINE';HasAccess=0}
            )) }
            'visibility' { return ,(New-FixtureTable @([pscustomobject]@{CanSelectDatabase=1;CanViewDefinition=0;CanUnmask=0})) }
            'objects' { return ,(New-FixtureTable @(
                [pscustomobject]@{ObjectId=10;SchemaName='lab';ObjectName='Measures';ObjectType='USER_TABLE';ApproxRows=3;CanSelect=1},
                [pscustomobject]@{ObjectId=11;SchemaName='lab';ObjectName='Other';ObjectType='USER_TABLE';ApproxRows=20;CanSelect=1},
                [pscustomobject]@{ObjectId=12;SchemaName='lab';ObjectName='SlowView';ObjectType='VIEW';ApproxRows=$null;CanSelect=1}
            )) }
            'columns' {
                if ($script:FailColumns) {
                    try { throw 'connection-secret-MUST-NOT-BE-EXPORTED' }
                    catch { Add-SqlSurveyError $State $Database $Stage $Object $_ }
                    return $null
                }
                return ,(New-FixtureTable @((New-FixtureColumn 'Value'),(New-FixtureColumn 'Payload' 'varbinary'),(New-FixtureColumn 'Id' 'int' 11)))
            }
            'keys' { return ,(New-FixtureTable @([pscustomobject]@{ObjectId=10;SchemaName='lab';ObjectName='Measures';KeyName='PK';IsPrimaryKey=1;IsUnique=1;HasFilter=0;KeyOrdinal=1;ColumnName='Value'})) }
            'foreign_keys' { return ,(New-FixtureTable @([pscustomobject]@{ObjectId=10;SchemaName='lab';ObjectName='Measures';ForeignKeyName='FK';KeyOrdinal=1;ColumnName='Value';ReferencedSchema='lab';ReferencedObject='Other';ReferencedColumn='Id';IsDisabled=0;IsNotTrusted=1})) }
            'profile' {
                Assert-Survey ($Sql.StartsWith('WITH [profile_source]')) 'profile uses only the generated aggregate SELECT'
                return ,(New-FixtureTable @([pscustomobject]@{ProfileRows=3;n0=2;d0=2;lo0='-2';hi0='5'}))
            }
            default { throw "Unexpected fixture query stage: $Stage" }
        }
    }
    $oldConnection = $env:AETHMODULAR_AQRC_CONNECTION_STRING
    $env:AETHMODULAR_AQRC_CONNECTION_STRING = 'connection-secret-MUST-NOT-BE-EXPORTED'
    try {
        $metadataOut = Invoke-SqlAccessSurvey -OutputRoot $tempRoot
        Assert-Survey (@(Import-Csv (Join-Path $metadataOut 'profiles.csv')).Count -eq 0) 'metadata-only is the default'
        Assert-Survey ('master' -notin $script:Opened -and 'Denied' -notin $script:Opened) 'no system/denied database connection attempts'
        $databaseRows = @(Import-Csv (Join-Path $metadataOut 'databases.csv'))
        Assert-Survey (($databaseRows | Where-Object Database -eq 'Denied').OpenStatus -eq 'not_accessible_or_not_online') 'denied is recorded, not interpreted as empty'
        $profileOut = Invoke-SqlAccessSurvey -Profile -MaxProfileRows 3 -MaxProfileTables 1 -OutputRoot $tempRoot
        $profiles = @(Import-Csv (Join-Path $profileOut 'profiles.csv'))
        Assert-Survey (($profiles | Where-Object ColumnName -eq 'Value').CoversAllVisibleRows -eq 'False') 'whole-run sample coverage'
        Assert-Survey (($profiles | Where-Object ColumnName -eq 'Payload').Status -eq 'skipped') 'binary data stays metadata-only'
        $objects = @(Import-Csv (Join-Path $profileOut 'profileobjects.csv'))
        Assert-Survey (($objects | Where-Object ObjectName -eq 'Other').Status -eq 'skipped_profile_table_budget') 'profile budget is explicit'
        Assert-Survey (($objects | Where-Object ObjectName -eq 'SlowView').Status -eq 'skipped_view_or_external_table') 'views are not executed by default'
        $manifest = @(Import-Csv (Join-Path $profileOut 'manifest.csv'))
        foreach ($entry in $manifest) {
            $path = Join-Path $profileOut $entry.File
            Assert-Survey ($entry.SHA256 -eq (Get-FileHash $path).Hash) 'manifest read-back hash'
            Assert-Survey ($entry.File -notmatch '[/\\]') 'manifest paths are relative filenames'
        }
        $script:FailColumns = $true
        $failureOut = Invoke-SqlAccessSurvey -Database Fixture -OutputRoot $tempRoot
        Assert-Survey ((Get-Content (Join-Path $failureOut 'receipt.json') -Raw | ConvertFrom-Json).Status -eq 'partial_with_errors') 'metadata failures remain visible'
        foreach ($path in Get-ChildItem $tempRoot -Recurse -File) {
            Assert-Survey ((Get-Content $path.FullName -Raw) -notmatch 'connection-secret-MUST-NOT-BE-EXPORTED') 'no credential or exception-message leakage'
        }
        Assert-Survey ($script:Disposed.Count -ge 6) 'connections disposed in all fixture runs'
    } finally { $env:AETHMODULAR_AQRC_CONNECTION_STRING = $oldConnection }
    Write-Host 'PASS: SQL survey quoting, ranges, nulls, CSVs, budgets, visibility, failures and manifests.'
} finally {
    Remove-Item $tempRoot -Recurse -Force
}
