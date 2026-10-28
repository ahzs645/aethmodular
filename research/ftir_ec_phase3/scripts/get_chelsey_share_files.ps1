# Copy the FTIR delivery files the 2026-10-28 share search pointed at.
# PASTE-SAFE, pure ASCII. Run on the Windows machine where U: is mapped.
# Read-only on U: (files are copied, never moved or changed).
#
# Why: the share search (aqrc_followup_exports/20261028_112459) found, under
# U:\FTIR_spectra\Chelsey_SPARTAN_MAIA:
#   - "Spartan batch4\report\SPARTAN_FTIR data_batch 4_ Nov2022 to March2024 V2.xlsx"
#     (2025-02-03), a later version of the batch 4 delivery whose organic groups run
#     about 2x the public OM; possibly the correction
#   - "Spartan batch4\FG and BLC\" (functional groups from baselined spectra) and
#     "Spartan batch5\" (batch 5 baselined spectra, reports from Alex and Sean)
#   - "MAIA\MAIA batch4\FTIR by diff. cal vs. TOR data\" (FTIR on CSU AMOD quartz
#     filters against TOR: the AMOD TOR values may be in there)
# The search read only file names and the first lines of CSVs, so .xlsx files and
# any functional-group file named differently were not inspected.
#
# Usage: paste the whole file, then run
#     Get-ChelseyShareFiles
#
# Writes Downloads\chelsey_share_<yyyyMMdd_HHmmss>\:
#     listing.csv     every file under the root (path, bytes, modified UTC)
#     files\...       copies of every file up to 5 MB in the folders below, same layout
#     manifest.csv    copied files with SHA256 (source and copy must match)
# and a verified ZIP of the folder with its .sha256.txt beside it.

$ShareRoot = "U:\FTIR_spectra\Chelsey_SPARTAN_MAIA"
$CopyFolders = @(
    "Spartan batch4",
    "Spartan batch5",
    "Spartan submitted data and reports",
    "Spartan by lot",
    "MAIA\MAIA batch4"
)
$MaxBytes = 5MB

function Get-ChelseyShareFiles {
    if (-not (Test-Path $ShareRoot)) { throw "Not reachable: $ShareRoot (is U: mapped?)" }
    $out = Join-Path $HOME ("Downloads\chelsey_share_" + (Get-Date -Format yyyyMMdd_HHmmss))
    $filesOut = Join-Path $out "files"
    New-Item $filesOut -ItemType Directory -Force | Out-Null
    $rootFull = (Resolve-Path $ShareRoot).ProviderPath.TrimEnd("\")

    Write-Host "== listing $ShareRoot =="
    $all = Get-ChildItem -LiteralPath $rootFull -Recurse -File -ErrorAction SilentlyContinue
    $all | ForEach-Object {
        [pscustomobject]@{
            RelativePath = $_.FullName.Substring($rootFull.Length + 1)
            Bytes = $_.Length
            ModifiedUtc = $_.LastWriteTimeUtc.ToString("o")
        }
    } | Export-Csv (Join-Path $out "listing.csv") -NoTypeInformation -Encoding UTF8
    Write-Host ("  {0} files listed" -f @($all).Count)

    Write-Host "== copying files up to 5 MB =="
    $manifest = New-Object System.Collections.ArrayList
    foreach ($folder in $CopyFolders) {
        $src = Join-Path $rootFull $folder
        if (-not (Test-Path -LiteralPath $src)) { Write-Host "  missing: $folder"; continue }
        foreach ($f in (Get-ChildItem -LiteralPath $src -Recurse -File -ErrorAction SilentlyContinue)) {
            if ($f.Length -gt $MaxBytes) { continue }
            $rel = $f.FullName.Substring($rootFull.Length + 1)
            $dest = Join-Path $filesOut $rel
            New-Item (Split-Path $dest -Parent) -ItemType Directory -Force | Out-Null
            Copy-Item -LiteralPath $f.FullName -Destination $dest
            $hs = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash
            $hd = (Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash
            if ($hs -ne $hd) { throw "Copy differs from source: $rel" }
            $manifest.Add([pscustomobject]@{ RelativePath = $rel; Bytes = $f.Length; ModifiedUtc = $f.LastWriteTimeUtc.ToString("o"); SHA256 = $hd }) | Out-Null
        }
        Write-Host ("  {0}: done" -f $folder)
    }
    $manifest | Export-Csv (Join-Path $out "manifest.csv") -NoTypeInformation -Encoding UTF8
    Write-Host ("  {0} files copied" -f $manifest.Count)

    Write-Host "== packing =="
    $zip = $out + ".zip"
    Compress-Archive -Path (Join-Path $out "*") -DestinationPath $zip
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $z = [System.IO.Compression.ZipFile]::OpenRead($zip)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $checked = 0
        foreach ($entry in $z.Entries) {
            if (-not $entry.Name) { continue }
            $name = $entry.FullName.Replace("/", "\")
            $stream = $entry.Open()
            try { $h = -join ($sha.ComputeHash($stream) | ForEach-Object { $_.ToString("X2") }) } finally { $stream.Close() }
            if ($h -ne (Get-FileHash -LiteralPath (Join-Path $out $name) -Algorithm SHA256).Hash) { throw "ZIP entry differs from source: $name" }
            $checked++
        }
    } finally {
        $z.Dispose()
    }
    $zh = (Get-FileHash $zip -Algorithm SHA256).Hash
    Set-Content ($zip + ".sha256.txt") ($zh + "  " + (Split-Path $zip -Leaf)) -Encoding ASCII
    Write-Host ("  {0} entries verified; {1}" -f $checked, $zip)
}
