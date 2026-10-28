# Offline contract tests. Database transport is replaced; no credentials needed.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../../research/ftir_ec_phase3/scripts/export_aqrc_reference_data.ps1')

function Assert-Reference($Condition, [string]$Message) {
    if (-not $Condition) { throw "Assertion failed: $Message" }
}

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ('aeth-reference-test-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot | Out-Null
$oldConnection = $env:AETHMODULAR_AQRC_CONNECTION_STRING
$oldCulture = [threading.thread]::CurrentThread.CurrentCulture
try {
    $plan = @(Get-AqrcReferenceExportPlan 42 137)
    Assert-Reference ($plan.Count -eq 12) 'all requested source exports are planned'
    foreach ($item in $plan) {
        Assert-Reference ($item.Sql -match '^SELECT\b') 'each database command is a SELECT'
        Assert-Reference ($item.Sql -notmatch '\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|EXEC|NOLOCK)\b') 'no writes, dynamic SQL or dirty reads'
    }
    $resultPlan = $plan | Where-Object Name -eq 'improve_result_parameters'
    Assert-Reference ($resultPlan.Timeout -eq 137) 'large distinct check has a separate timeout'
    Assert-Reference ($resultPlan.Sql -match 'SELECT DISTINCT \[Parameter\]' -and
        $resultPlan.Sql -notmatch '\[Value\]|COUNT|TOP') 'actual IMPROVE result names without exporting measurements or sampling'
    Assert-Reference (($plan | Where-Object Name -eq 'ftir_calibrations').Sql -match 'CONVERT\(varchar\(100\), \[b\], 3\)') 'coefficient precision retained on SQL Server'
    Assert-Reference (($plan | Where-Object Name -eq 'spartan_filter_batches').Sql -match 'LEFT JOIN \[analysis\].\[Media\]') 'unmatched physical filters retained in batch map'

    $global:AqrcReferenceFixture = [pscustomobject]@{
        RealHelper=(Join-Path $PSScriptRoot '../../scripts/diagnostics/inspect_sql_access.ps1')
        Plan=$plan; FailDataset=''; DeniedDatabase=''; EmptyAmod=$false; NoSampleLabel=$false; Opened=@(); Disposed=@()
    }
    $helper = Join-Path $tempRoot 'fixture_helper.ps1'
    @'
. $global:AqrcReferenceFixture.RealHelper
function Open-SqlSurveyConnection([string]$ConnectionString, [string]$Database) {
    $global:AqrcReferenceFixture.Opened += $Database
    if ($global:AqrcReferenceFixture.DeniedDatabase -eq $Database) { throw 'fixture-secret-must-not-appear' }
    $connection = [pscustomobject]@{Database=$Database}
    $connection | Add-Member ScriptMethod Dispose { $global:AqrcReferenceFixture.Disposed += $this.Database }
    return $connection
}
function Invoke-SqlSurveyQuery {
    param($Connection,[string]$Sql,$State,[string]$Database,[string]$Stage,[string]$Object='', [int]$Limit=0,[int]$Timeout=60)
    $entry = [pscustomobject]@{
        QueryId=('q{0:d6}' -f ($State.Queries.Count+1)); Database=$Database; Stage=$Stage
        Object=$Object; Sql=$Sql; TimeoutSeconds=$Timeout; Status='ok'; RowsReturned=0
    }
    $State.Queries.Add($entry)
    if ($global:AqrcReferenceFixture.FailDataset -eq $Object) {
        $entry.Status='failed'
        try { throw [TimeoutException]::new('fixture-secret-must-not-appear') }
        catch { Add-SqlSurveyError $State $Database $Stage $Object $_ }
        return $null
    }
    $item = $global:AqrcReferenceFixture.Plan | Where-Object Name -eq $Object
    if (-not $item) { throw 'Unexpected export requested.' }
    $table = New-Object System.Data.DataTable
    foreach ($name in $item.Columns) { $table.Columns.Add($name,[object]) | Out-Null }
    $records = @(@{})
    switch ($Object) {
        'analysis_batches' { $records=@() }
        'ftir_calibrations' {
            $records=@(@{Id=1;CalibrationSetId=10;RowNumber=0;Wavenumber='1.2345678901234567e+003';b='-9.8765432109876543e-012'})
        }
        'ftir_calibration_sets' {
            $records=@(@{Id=10;Parameter='EC';Comments="line one, `"quoted`"`nline two";ProcessedTime=[datetime]'2026-10-28T00:01:02'})
        }
        'csu_amod_filters' {
            $records=@(
                @{MediaId=1;ExternalFilterId='J1269';MassDifference_micrograms=[decimal]::Parse('-2.500',[cultureinfo]::InvariantCulture);SampleVolume_liters=[decimal]0;ExternalErrorFlag='retain'},
                @{MediaId=2;ExternalFilterId='J1266';SampleVolume_liters=[decimal]1234.567},
                @{MediaId=3;ExternalFilterId='J1266'},
                @{MediaId=4;ExternalFilterId='j1285'},
                @{MediaId=5;ExternalFilterId='J1693'}
            )
        }
        'improve_result_parameters' { $records=@(@{Parameter='ECf';ParameterIsNull=$false},@{Parameter=$null;ParameterIsNull=$true}) }
        'spartan_coverage_result_types' {
            $label='Sample'; if ($global:AqrcReferenceFixture.NoSampleLabel) { $label='Reference' }
            $records=@(@{Id=5;Name=$label})
        }
    }
    if ($Object -eq 'csu_amod_filters' -and $global:AqrcReferenceFixture.EmptyAmod) { $records=@() }
    foreach ($record in $records) {
        $row = $table.NewRow()
        foreach ($name in $item.Columns) {
            if ($record.ContainsKey($name)) { $row[$name]=$record[$name] }
            else { $row[$name]=[DBNull]::Value }
        }
        $table.Rows.Add($row)
    }
    $entry.RowsReturned=$table.Rows.Count
    return ,$table
}
'@ | Set-Content -LiteralPath $helper -Encoding UTF8

    $env:AETHMODULAR_AQRC_CONNECTION_STRING = 'fixture-secret-must-not-appear'
    [threading.thread]::CurrentThread.CurrentCulture = [cultureinfo]'de-DE'
    $out = Invoke-AqrcReferenceExport -HelperScript $helper -OutputRoot $tempRoot -QueryTimeoutSeconds 42 -ImproveTimeoutSeconds 137
    $exports = @(Import-Csv (Join-Path $out 'exports.csv'))
    Assert-Reference ($exports.Count -eq 13 -and @($exports | Where-Object Status -ne 'ok').Count -eq 0) 'complete run reports every source plus coverage'
    Assert-Reference ((Get-Content (Join-Path $out 'receipt.json') -Raw | ConvertFrom-Json).Status -eq 'complete') 'complete receipt'
    Assert-Reference (($exports | Where-Object Dataset -eq 'analysis_batches').CsvRows -eq '0') 'successful empty read keeps a header and zero count'
    $coefficients = @(Import-Csv (Join-Path $out 'ftir_calibrations.csv'))
    Assert-Reference ($coefficients[0].b -eq '-9.8765432109876543e-012') 'coefficient string survives CSV round trip'
    $sets = @(Import-Csv (Join-Path $out 'ftir_calibration_sets.csv'))
    Assert-Reference ($sets[0].Comments -eq "line one, `"quoted`"`nline two" -and $sets[0].ProcessedTime.StartsWith('2026-10-28T00:01:02')) 'multiline comments and ISO dates retained'
    $amod = @(Import-Csv (Join-Path $out 'csu_amod_filters.csv'))
    Assert-Reference ($amod[0].MassDifference_micrograms -eq '-2.500' -and $amod[0].SampleVolume_liters -eq '0' -and $amod[0].ExternalErrorFlag -eq 'retain') 'negative mass, zero volume and error flags remain unchanged'
    Assert-Reference ($amod[1].SampleVolume_liters -eq '1234.567') 'decimal format independent of machine locale'
    $coverage = @(Import-Csv (Join-Path $out 'adama_filter_coverage.csv'))
    Assert-Reference ($coverage.Count -eq 10) 'five pairs checked with separate quartz/PTFE identities'
    Assert-Reference (($coverage | Where-Object ExternalFilterId -eq 'J1269').Status -eq 'matched_exact_id') 'PTFE exact match'
    Assert-Reference (($coverage | Where-Object ExternalFilterId -eq 'J1266').Status -eq 'multiple_exact_id_matches') 'duplicate external IDs are visible'
    Assert-Reference (($coverage | Where-Object ExternalFilterId -eq 'J1285').Status -eq 'not_found_exact_id') 'no case folding or approximate ID matching'
    $nullParameter = @(Import-Csv (Join-Path $out 'improve_result_parameters.csv')) | Where-Object ParameterIsNull -eq 'True'
    Assert-Reference ($nullParameter.Parameter -eq '') 'NULL parameter is explicitly distinguishable'
    $manifest = @(Import-Csv (Join-Path $out 'manifest.csv'))
    foreach ($entry in $manifest) {
        $path = Join-Path $out $entry.File
        Assert-Reference ($entry.File -notmatch '[/\\]' -and $entry.SHA256 -eq (Get-FileHash $path).Hash) 'manifest uses relative names and valid hashes'
        if ($entry.File.EndsWith('.csv')) { Assert-Reference ($entry.CsvRows -eq @(Import-Csv $path).Count) 'manifest count matches CSV read back' }
    }
    Assert-Reference ($global:AqrcReferenceFixture.Disposed.Count -eq 2) 'all successful connections disposed'

    $global:AqrcReferenceFixture.EmptyAmod=$true
    $emptyOut = Invoke-AqrcReferenceExport -HelperScript $helper -OutputRoot $tempRoot
    $emptyCoverage = @(Import-Csv (Join-Path $emptyOut 'adama_filter_coverage.csv'))
    Assert-Reference ($emptyCoverage.Count -eq 10 -and @($emptyCoverage | Where-Object Status -ne 'not_found_exact_id').Count -eq 0) 'successful empty AMOD source differs from an unavailable source'
    $global:AqrcReferenceFixture.EmptyAmod=$false

    $global:AqrcReferenceFixture.FailDataset='improve_result_parameters'
    $partial = Invoke-AqrcReferenceExport -HelperScript $helper -OutputRoot $tempRoot
    Assert-Reference ((Get-Content (Join-Path $partial 'receipt.json') -Raw | ConvertFrom-Json).Status -eq 'partial_with_errors') 'timeout does not imply completion'
    $failed = @(Import-Csv (Join-Path $partial 'exports.csv')) | Where-Object Dataset -eq 'improve_result_parameters'
    Assert-Reference ($failed.Status -eq 'query_failed' -and -not (Test-Path (Join-Path $partial 'improve_result_parameters.csv'))) 'failed check has no misleading empty CSV'
    Assert-Reference (@(Import-Csv (Join-Path $partial 'csu_amod_filters.csv')).Count -eq 5) 'unrelated exports survive a failed check'
    $allText = (Get-ChildItem $partial -File | ForEach-Object { Get-Content $_.FullName -Raw }) -join "`n"
    Assert-Reference ($allText -notmatch 'fixture-secret-must-not-appear') 'neither connection secrets nor exception messages leak'

    $skipped = Invoke-AqrcReferenceExport -HelperScript $helper -OutputRoot $tempRoot -SkipImproveParameterCheck
    Assert-Reference ((Get-Content (Join-Path $skipped 'receipt.json') -Raw | ConvertFrom-Json).Status -eq 'complete_with_skips') 'skip is explicit'
    $queries = @(Get-Content (Join-Path $skipped 'queries.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
    Assert-Reference (@($queries | Where-Object Object -eq 'improve_result_parameters').Count -eq 0) 'skipped large query never runs'

    $global:AqrcReferenceFixture.FailDataset=''
    $global:AqrcReferenceFixture.DeniedDatabase='Networks_1_0'
    $denied = Invoke-AqrcReferenceExport -HelperScript $helper -OutputRoot $tempRoot
    $deniedExports = @(Import-Csv (Join-Path $denied 'exports.csv'))
    Assert-Reference (($deniedExports | Where-Object Dataset -eq 'csu_amod_filters').Status -eq 'connection_failed') 'access failure is recorded'
    Assert-Reference (($deniedExports | Where-Object Dataset -eq 'adama_filter_coverage').Status -eq 'skipped_source_unavailable') 'missing AMOD source does not create ten false absences'
    Assert-Reference (($deniedExports | Where-Object Dataset -eq 'improve_result_parameters').Status -eq 'ok') 'other database still exported'
    $global:AqrcReferenceFixture.DeniedDatabase=''
    $global:AqrcReferenceFixture.Plan=@(Get-AqrcReferenceExportPlan -IncludeSpartanCoverage)
    $global:AqrcReferenceFixture.Opened=@()
    $coverageOut=Invoke-AqrcReferenceExport -HelperScript $helper -OutputRoot $tempRoot -SpartanCoverageOnly
    $coverageExports=@(Import-Csv (Join-Path $coverageOut 'exports.csv'))
    Assert-Reference ($coverageExports.Count -eq 3 -and @($coverageExports | Where-Object Status -ne 'ok').Count -eq 0) 'coverage-only exports exactly the three new datasets'
    Assert-Reference ($global:AqrcReferenceFixture.Opened.Count -eq 1 -and $global:AqrcReferenceFixture.Opened[0] -eq 'Networks_1_0') 'coverage-only never reconnects to IMPROVE'
    $nativeSql=($global:AqrcReferenceFixture.Plan | Where-Object Name -eq 'spartan_native_filter_coverage').Sql
    Assert-Reference ($nativeSql -match 'GROUP BY a.\[MediaId\]' -and $nativeSql -notmatch '\.\[Data\]') 'native instrument counts aggregate before joining and never read scan blobs'
    Assert-Reference ($nativeSql -match "N'SAMPLE'" -and $nativeSql -notmatch 'ResultTypeId\] = 0') 'sample type resolved through labels rather than fixed IDs'
    $global:AqrcReferenceFixture.NoSampleLabel=$true
    $unknownTypes=Invoke-AqrcReferenceExport -HelperScript $helper -OutputRoot $tempRoot -SpartanCoverageOnly
    $unknownExport=@(Import-Csv (Join-Path $unknownTypes 'exports.csv')) | Where-Object Dataset -eq 'spartan_native_filter_coverage'
    Assert-Reference ($unknownExport.Status -eq 'skipped_missing_sample_result_type' -and -not (Test-Path (Join-Path $unknownTypes 'spartan_native_filter_coverage.csv'))) 'unknown sample labels never become false zero HIPS coverage'
    Write-Host 'PASS: reference export precision, coverage, failures, skips, disposal and manifests.'
} finally {
    $env:AETHMODULAR_AQRC_CONNECTION_STRING=$oldConnection
    [threading.thread]::CurrentThread.CurrentCulture=$oldCulture
    Remove-Variable AqrcReferenceFixture -Scope Global -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $tempRoot -Recurse -Force
}
