# Pull FTIR spectra + filter metadata for ANY SPARTAN site(s) from AQRC-SQL.
# Generalized from the ETBI pull (2026-08-22). PASTE-SAFE, pure ASCII.
#
# Run on the Windows machine that is on the UCD VPN. Requires read access to
# the Networks_1_0 database (granted via ticket INC2653758).
#
# Usage: paste the whole file, then call e.g.
#     Get-SpartanSite "INDH"                      # Delhi
#     Get-SpartanSite "CHTS"                      # Beijing
#     "INDH","CHTS","USPA","ETAD" | % { Get-SpartanSite $_ }
#     Show-SpartanSites                           # list every site + scan counts
#
# Writes <SITE>_filters.csv, <SITE>_ftir_analysis.csv, <SITE>_scans_base64.csv
# into Downloads. Hand those to build_spartan_target.py on the analysis machine.

function Open-AqrcConn([string]$db = "Networks_1_0") {
    $cs = "Server=AQRC-SQL,1433;Integrated Security=SSPI;TrustServerCertificate=True;Database=$db;"
    $conn = New-Object System.Data.SqlClient.SqlConnection($cs)
    $conn.Open()
    return $conn
}

function Invoke-AqrcQuery($conn, [string]$sql) {
    $cmd = $conn.CreateCommand(); $cmd.CommandText = $sql; $cmd.CommandTimeout = 600
    $table = New-Object System.Data.DataTable
    (New-Object System.Data.SqlClient.SqlDataAdapter($cmd)).Fill($table) | Out-Null
    return $table
}

# DataTable -> CSV. The explicit .Table recovery matters: piping DataRows
# directly makes PowerShell drop $table.Columns and emit junk.
function Export-AqrcTable($table, [string]$path) {
    if ($table -isnot [System.Data.DataTable]) { $table = @($table)[0].Table }
    $list = foreach ($row in $table.Rows) {
        $props = [ordered]@{}
        foreach ($c in $table.Columns) {
            $v = $row[$c]
            if ($v -is [byte[]]) { $v = [Convert]::ToBase64String($v) }
            elseif ($v -is [System.DBNull]) { $v = "" }
            $props[$c.ColumnName] = $v
        }
        [pscustomobject]$props
    }
    $list | Export-Csv $path -NoTypeInformation -Encoding UTF8
    Write-Host "  wrote $($table.Rows.Count) rows -> $path"
}

function Show-SpartanSites {
    $c = Open-AqrcConn
    Write-Host "== SPARTAN sites, filters, and FTIR-scanned counts =="
    Invoke-AqrcQuery $c @"
SELECT s.SiteCode, s.Name,
       COUNT(DISTINCT f.MediaId) AS Filters,
       COUNT(DISTINCT sa.MediaId) AS FtirScanned,
       MIN(f.SamplingStartDate) AS FirstSample,
       MAX(f.SamplingStartDate) AS LastSample
FROM spartan.Sites s
LEFT JOIN spartan.Filters f ON f.SiteId = s.Id
LEFT JOIN ftir.SampleAnalysis sa ON sa.MediaId = f.MediaId
GROUP BY s.SiteCode, s.Name
ORDER BY FtirScanned DESC, s.SiteCode
"@ | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
    $c.Close()
}

function Get-SpartanSite([Parameter(Mandatory)][string]$SiteCode) {
    $out = Join-Path $HOME "Downloads"
    $c = Open-AqrcConn
    Write-Host "== $SiteCode =="
    Export-AqrcTable (Invoke-AqrcQuery $c @"
SELECT f.* FROM spartan.Filters f
JOIN spartan.Sites s ON s.Id = f.SiteId WHERE s.SiteCode = '$SiteCode'
"@) (Join-Path $out "${SiteCode}_filters.csv")

    Export-AqrcTable (Invoke-AqrcQuery $c @"
SELECT sa.* FROM ftir.SampleAnalysis sa
WHERE sa.MediaId IN (SELECT f.MediaId FROM spartan.Filters f
                     JOIN spartan.Sites s ON s.Id = f.SiteId WHERE s.SiteCode = '$SiteCode')
"@) (Join-Path $out "${SiteCode}_ftir_analysis.csv")

    Export-AqrcTable (Invoke-AqrcQuery $c @"
SELECT sc.* FROM ftir.Scan sc
WHERE sc.Id IN (SELECT SampleScanId FROM ftir.SampleAnalysis WHERE MediaId IN
                  (SELECT f.MediaId FROM spartan.Filters f
                   JOIN spartan.Sites x ON x.Id = f.SiteId WHERE x.SiteCode = '$SiteCode'))
   OR sc.Id IN (SELECT BackgroundScanId FROM ftir.SampleAnalysis WHERE MediaId IN
                  (SELECT f.MediaId FROM spartan.Filters f
                   JOIN spartan.Sites x ON x.Id = f.SiteId WHERE x.SiteCode = '$SiteCode'))
"@) (Join-Path $out "${SiteCode}_scans_base64.csv")
    $c.Close()
}

Show-SpartanSites
Write-Host "`nNext: Get-SpartanSite ""INDH""   (or CHTS / USPA / ETAD / any code above)"
