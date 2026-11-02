# Run normally in the existing database-configured PowerShell tab. No elevation,
# execution-policy changes, service installation or public tunnel access.
[CmdletBinding()]
param([ValidateRange(0,240)][int]$LifetimeMinutes = 120)

$ErrorActionPreference = 'Stop'
# Dot-sourcing the bridge also binds its default script parameters; retain the
# caller's lifetime separately so an explicit zero cannot become 120 again.
$requestedLifetimeMinutes = $LifetimeMinutes
if (-not $env:AETHMODULAR_AQRC_CONNECTION_STRING) {
    throw 'Use the PowerShell tab with the existing AQRC connection variable.'
}
$tokenFile = Join-Path $PSScriptRoot 'bridge-token.txt'
if (-not (Test-Path -LiteralPath $tokenFile -PathType Leaf)) { throw 'The setup token file is missing.' }
$tool = Join-Path $PSScriptRoot 'devtunnel.exe'
if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) {
    Invoke-WebRequest 'https://aka.ms/TunnelsCliDownload/win-x64' -OutFile $tool -UseBasicParsing
}
$signature = Get-AuthenticodeSignature -LiteralPath $tool
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
    throw 'The Microsoft tunnel client signature could not be verified.'
}
& $tool user login -g
if ($LASTEXITCODE -ne 0) { throw 'Tunnel sign-in did not complete.' }
. (Join-Path $PSScriptRoot 'aqrc_cli_bridge.ps1')
Install-AqrcScriptDom
Initialize-AqrcScriptDom
$tunnelArguments = @('host','-p','8765')
if ($requestedLifetimeMinutes -ne 0) { $tunnelArguments += @('--expiration','4h') }
$tunnelProcess = Start-Process -FilePath $tool -ArgumentList $tunnelArguments -NoNewWindow -PassThru
try {
    Start-AqrcCliBridge -TokenFile $tokenFile -Port 8765 -LifetimeMinutes $requestedLifetimeMinutes
} finally {
    if (-not $tunnelProcess.HasExited) { Stop-Process -Id $tunnelProcess.Id }
}
