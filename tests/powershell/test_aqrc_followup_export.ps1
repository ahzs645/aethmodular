$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../../research/ftir_ec_phase3/scripts/export_aqrc_followup.ps1')
function Assert-Followup($Condition,[string]$Message) { if (-not $Condition) { throw "Assertion failed: $Message" } }
$root=Join-Path ([IO.Path]::GetTempPath()) ('aqrc-followup-test-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root | Out-Null
$oldConnection=$env:AETHMODULAR_AQRC_CONNECTION_STRING
try {
    $lab=Join-Path $root 'lab'; New-Item -ItemType Directory -Path $lab | Out-Null
    Set-Content (Join-Path $lab 'FTIR_batch.csv') 'sample,aCH,aCOH,COOH,naCO'
    Set-Content (Join-Path $lab 'unrelated.csv') 'a,b'
    $found=Find-AqrcFgFiles -Root @($lab)
    Assert-Followup ($found.Candidates.Count -eq 1 -and $found.Candidates[0].FgColumns -match 'COOH') 'FG columns identified within a scoped folder'
    $missing=Find-AqrcFgFiles -Root @((Join-Path $root 'missing'))
    Assert-Followup ($missing.Roots[0].Status -eq 'unavailable') 'unmapped folders are unknown rather than empty'
    $global:AqrcFollowupFixture=[pscustomobject]@{
        Helper=(Join-Path $PSScriptRoot '../../scripts/diagnostics/inspect_sql_access.ps1')
        Plan=@(Get-AqrcFollowupPlan); MissingBackground=$false; Disposed=$false
    }
    $helper=Join-Path $root 'fixture.ps1'
    @'
. $global:AqrcFollowupFixture.Helper
function Open-SqlSurveyConnection {
    param($ConnectionString,$Database)
    $c=[pscustomobject]@{}
    $c | Add-Member ScriptMethod Dispose { $global:AqrcFollowupFixture.Disposed=$true }
    return $c
}
function Invoke-SqlSurveyQuery {
    param($Connection,$Sql,$State,$Database,$Stage,$Object,$Timeout)
    $State.Queries.Add([pscustomobject]@{Sql=$Sql;Object=$Object;Status='ok'})
    $item=$global:AqrcFollowupFixture.Plan | Where-Object Name -eq $Object
    $table=New-Object Data.DataTable
    $columns=$item.Columns
    if ($Object -eq 'amod_ftir_scans_base64') { $columns=@($columns | Where-Object { $_ -notin @('DataBytes','DataSHA256','DataBase64') })+@('Data') }
    foreach ($name in $columns) { $table.Columns.Add($name,[object]) | Out-Null }
    $records=@(@{Id=1})
    if ($Object -eq 'amod_ftir_analyses') {
        $records=@(@{Id=10;MediaId=3;SampleScanId=101;BackgroundScanId=102},@{Id=11;MediaId=3;SampleScanId=101;BackgroundScanId=102})
    }
    if ($Object -eq 'amod_ftir_scans_base64') {
        $records=@(@{Id=101;Data=[byte[]]@(0,1,2,253,254,255)},@{Id=102;Data=[byte[]]@(4,5,6)})
        if ($global:AqrcFollowupFixture.MissingBackground) { $records=@($records[0]) }
    }
    foreach ($record in $records) {
        $row=$table.NewRow()
        foreach ($name in $columns) { if ($record.ContainsKey($name)) { $row[$name]=$record[$name] } else { $row[$name]=[DBNull]::Value } }
        $table.Rows.Add($row)
    }
    return ,$table
}
'@ | Set-Content $helper -Encoding UTF8
    $env:AETHMODULAR_AQRC_CONNECTION_STRING='fixture-secret-never-log'
    $out=Invoke-AqrcFollowupExport -HelperScript $helper -OutputRoot $root -FgSearchRoot $lab
    $receipt=Get-Content (Join-Path $out 'receipt.json') -Raw | ConvertFrom-Json
    Assert-Followup ($receipt.Status -eq 'complete' -and $receipt.Analyses -eq 2 -and $receipt.Scans -eq 2) 'repeat analyses retained while shared scan IDs remain unique'
    $scans=@(Import-Csv (Join-Path $out 'amod_ftir_scans_base64.csv'))
    foreach ($scan in $scans) {
        $bytes=[Convert]::FromBase64String($scan.DataBase64)
        Assert-Followup ($bytes.Length -eq [int]$scan.DataBytes) 'native bytes survive base64 CSV round trip'
        $hasher=[Security.Cryptography.SHA256]::Create()
        try { $hash=[BitConverter]::ToString($hasher.ComputeHash($bytes)).Replace('-','') } finally { $hasher.Dispose() }
        Assert-Followup ($hash -eq $scan.DataSHA256) 'per-scan SHA256 matches restored bytes'
    }
    $checks=@(Import-Csv (Join-Path $out 'amod_scan_reference_check.csv'))
    Assert-Followup ($checks.Count -eq 4 -and @($checks | Where-Object ScanFound -ne 'True').Count -eq 0) 'sample and background references checked for every analysis'
    $manifest=@(Import-Csv (Join-Path $out 'manifest.csv'))
    foreach ($entry in $manifest) {
        $path=Join-Path $out $entry.File
        Assert-Followup ($entry.SHA256 -eq (Get-FileHash $path).Hash) 'manifest hashes verified'
        if ($entry.File.EndsWith('.csv')) { Assert-Followup ($entry.CsvRows -eq @(Import-Csv $path).Count) 'manifest CSV counts verified' }
    }
    Assert-Followup $global:AqrcFollowupFixture.Disposed 'SQL connection disposed'
    $global:AqrcFollowupFixture.MissingBackground=$true
    $partial=Invoke-AqrcFollowupExport -HelperScript $helper -OutputRoot $root -FgSearchRoot $lab
    $r=Get-Content (Join-Path $partial 'receipt.json') -Raw | ConvertFrom-Json
    Assert-Followup ($r.Status -eq 'partial_with_errors_or_missing_scans' -and $r.MissingOrEmptyScanReferences -eq 2) 'missing shared background is recorded against each analysis'
    $scanSql=($global:AqrcFollowupFixture.Plan | Where-Object Name -eq 'amod_ftir_scans_base64').Sql
    Assert-Followup ($scanSql -match '\bUNION\b' -and $scanSql -notmatch 'UNION ALL') 'sample/background union avoids duplicate scan records'
    $all=(Get-ChildItem $out -File | ForEach-Object { Get-Content $_.FullName -Raw }) -join "`n"
    Assert-Followup ($all -notmatch 'fixture-secret-never-log|System.Byte\[\]') 'no connection secret or lossy byte-array formatting'
    Write-Host 'PASS: scoped FG search, AMOD repeats, sample/background completeness, base64 bytes, hashes and manifests.'
} finally {
    $env:AETHMODULAR_AQRC_CONNECTION_STRING=$oldConnection
    Remove-Variable AqrcFollowupFixture -Scope Global -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $root -Recurse -Force
}
