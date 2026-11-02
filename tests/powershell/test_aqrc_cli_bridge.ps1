# Offline checks: the real SQL transport is never invoked.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../../scripts/diagnostics/aqrc_cli_bridge.ps1')

function Assert-Cli($Condition, [string]$Message) {
    if (-not $Condition) { throw ('Assertion failed: ' + $Message) }
}
function Assert-Rejected($Block, [string]$Expected) {
    try { & $Block; throw 'expected_rejection' }
    catch { Assert-Cli ($_.Exception.Message -eq $Expected) ('reject ' + $Expected) }
}
$allowed = @('Networks_1_0')
Assert-Rejected { Start-AqrcCliBridge -LifetimeMinutes -1 } 'invalid_listener_settings'
Assert-Rejected { Start-AqrcCliBridge -LifetimeMinutes 241 } 'invalid_listener_settings'
$request = [pscustomobject]@{operation='columns';database='Networks_1_0';schema="lab';DROP TABLE x;--";table='a]b';limit=3}
$plan = New-AqrcCliPlan $request $allowed
Assert-Cli (-not $plan.Sql.Contains($request.schema)) 'schema data stays in parameters'
Assert-Cli ($plan.Parameters['@schema'] -eq $request.schema -and $plan.Parameters['@fetch'] -eq 4) 'bound and parameters retained'
Assert-Cli ((Quote-AqrcCliName 'a]b') -eq '[a]]b]') 'identifier quoting'
Assert-Rejected { New-AqrcCliPlan ([pscustomobject]@{operation='query';sql='DELETE x'}) $allowed } 'unknown_operation'
Assert-Rejected { New-AqrcCliPlan ([pscustomobject]@{operation='probe';database='master'}) $allowed } 'database_not_allowed'
Assert-Rejected { New-AqrcCliPlan ([pscustomobject]@{operation='tables';database='Networks_1_0';limit=0}) $allowed } 'invalid_limit'
Assert-Rejected { New-AqrcCliPlan ([pscustomobject]@{operation='tables';database='Networks_1_0';limit='1;DELETE x'}) $allowed } 'invalid_limit'
Assert-Rejected { New-AqrcCliPlan ([pscustomobject]@{operation='preview';database='Networks_1_0';schema='s';table='t';columns=@('x');limit=201}) $allowed } 'invalid_limit'
Assert-Rejected { New-AqrcCliPlan ([pscustomobject]@{operation='preview';database='Networks_1_0';schema='s';table='t';columns=@()}) $allowed } 'select_1_to_20_columns'
$preview = New-AqrcCliPlan ([pscustomobject]@{operation='preview';database='Networks_1_0';schema='s]x';table='t]x';columns=@('value]x')}) $allowed
$metadata = [pscustomobject]@{ColumnName='value]x';BaseType='decimal';IsComputed=$false;IsHidden=$false;EncryptionType=$null}
$sql = New-AqrcCliPreviewSql $preview @($metadata)
Assert-Cli ($sql -eq 'SELECT TOP (@fetch) CONVERT(nvarchar(512),[value]]x]) AS [value]]x] FROM [s]]x].[t]]x];') 'preview quotes identifiers and bounds rows and cells'
$metadata.IsComputed=$true
Assert-Rejected { New-AqrcCliPreviewSql $preview @($metadata) } 'unsupported_column'
$metadata.IsComputed=$false; $metadata.BaseType='varbinary'
Assert-Rejected { New-AqrcCliPreviewSql $preview @($metadata) } 'unsupported_column'
Assert-Rejected { New-AqrcCliPreviewSql $preview @() } 'column_not_visible'

$sha = [Security.Cryptography.SHA256]::Create()
try { $digest=$sha.ComputeHash([Text.Encoding]::UTF8.GetBytes('secret')) }
finally { $sha.Dispose() }
Assert-Cli (Test-AqrcCliToken 'secret' $digest) 'matching token'
Assert-Cli (-not (Test-AqrcCliToken 'wrong' $digest)) 'wrong token'
Assert-Cli (-not (Test-AqrcCliToken ('x'*257) $digest)) 'oversized token'
try { throw [ArgumentException]::new('credential-like-sensitive-detail') }
catch {
    $failure = Get-AqrcCliFailure $_
    Assert-Cli ($failure.error -eq 'operation_failed') 'arbitrary exception text never leaves the helper'
}
$stream = [IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes('{"x":1}' + "`n"))
try { Assert-Cli ((Read-AqrcCliLine $stream) -eq '{"x":1}') 'line frame' } finally { $stream.Dispose() }
$stream = [IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes('incomplete'))
try { Assert-Rejected { Read-AqrcCliLine $stream } 'invalid_request_frame' } finally { $stream.Dispose() }
$stream = [IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes('12345' + "`n"))
try { Assert-Rejected { Read-AqrcCliLine $stream 4 } 'invalid_request_frame' } finally { $stream.Dispose() }

$script:Disposed=$false
function Open-AqrcCliConnection {
    $connection=[pscustomobject]@{Database='Fixture'}
    $connection | Add-Member ScriptMethod Dispose { $script:Disposed=$true }
    return $connection
}
function Invoke-AqrcCliSql {
    param($Connection,$Sql,$Parameters)
    return [pscustomobject]@{Columns=@('Value');Rows=@([pscustomobject]@{Value=1},[pscustomobject]@{Value=2})}
}
$reply = Invoke-AqrcCliOperation ([pscustomobject]@{operation='tables';database='Networks_1_0';limit=1}) 'fixture' $allowed
Assert-Cli ($reply.ok -and $reply.rowCount -eq 1 -and $reply.hasMore) 'overflow row distinguishes capped results'
Assert-Cli $script:Disposed 'connection closed'
function Invoke-AqrcCliSql { throw 'fixture-transport-failure' }
$script:Disposed=$false
try { $null=Invoke-AqrcCliOperation ([pscustomobject]@{operation='probe';database='Networks_1_0'}) 'fixture' $allowed }
catch { Assert-Cli ($_.Exception.Message -eq 'fixture-transport-failure') 'failure stays a failure' }
Assert-Cli $script:Disposed 'connection closed after error'
Write-Host 'AQRC CLI bridge offline checks passed.'
