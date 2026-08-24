# check_aqrc_gaps.ps1 - is the database holding more than our local exports?
#
# Run on a WINDOWS machine on the UCD VPN (AD trusted auth). See AQRC_SQL_HOWTO.md.
# Read-only: every statement is a SELECT. Pure ASCII on purpose.
#
# PASTE-SAFE: no param() block, because param() is only legal as the first
# statement of a script FILE - pasting it into an interactive console fails to
# parse and silently leaves every variable EMPTY, which makes every query match
# nothing and look like an empty database (cost one round-trip, 2026-08-23).
#
# Either paste this whole file into PowerShell, or run it as a file:
#   powershell -ExecutionPolicy Bypass -File .\check_aqrc_gaps.ps1
# To use a different site, set $SiteCode before pasting, or edit the line below.

if (-not $SiteCode)  { $SiteCode  = "ETAD" }        # SPARTAN site code
if (-not $LocalMeta) { $LocalMeta = "2026-03-01" }  # newest local ETAD_metadata.csv sample
if (-not $LocalHips) { $LocalHips = "2026-01-04" }  # newest local SPARTAN HIPS ETAD sample
if (-not $OutDir)    { $OutDir    = "$HOME\Downloads" }

function Open-AqrcConn([string]$db = "Networks_1_0") {
    $cs = "Server=AQRC-SQL,1433;Integrated Security=SSPI;TrustServerCertificate=True;Database=$db;"
    $conn = New-Object System.Data.SqlClient.SqlConnection($cs)
    $conn.Open()
    return $conn
}
function Invoke-AqrcQuery($conn, [string]$sql) {
    $cmd = $conn.CreateCommand(); $cmd.CommandText = $sql; $cmd.CommandTimeout = 300
    $table = New-Object System.Data.DataTable
    (New-Object System.Data.SqlClient.SqlDataAdapter($cmd)).Fill($table) | Out-Null
    return ,$table
}
# DataRow piping to Export-Csv produces junk - unwrap to the DataTable first
function Export-AqrcTable($table, [string]$path) {
    if ($table -isnot [System.Data.DataTable]) { $table = @($table)[0].Table }
    $table | Export-Csv $path -NoTypeInformation -Encoding UTF8
    Write-Host ("      -> " + $path)
}
function Try-Aqrc($conn, [string]$label, [string]$sql) {
    Write-Host ("      " + $label)
    try { Invoke-AqrcQuery $conn $sql | Format-Table -AutoSize }
    catch { Write-Host ("      SKIP: " + $_.Exception.Message) -ForegroundColor DarkGray }
}

if (-not $SiteCode) { Write-Host "SiteCode is empty - aborting" -ForegroundColor Red; return }
Write-Host ""
Write-Host "=== $SiteCode : database vs our local exports ===" -ForegroundColor Cyan
$c = Open-AqrcConn

$mediaFor = "SELECT f.MediaId FROM spartan.Filters f JOIN spartan.Sites s ON s.Id = f.SiteId WHERE s.SiteCode = '$SiteCode'"

# --- 0. sanity: the site must actually exist -------------------------------
Write-Host ""
Write-Host "[0] site lookup (if this is empty, nothing below is meaningful)" -ForegroundColor Yellow
Invoke-AqrcQuery $c "SELECT Id, SiteCode, Name, Latitude, Longitude FROM spartan.Sites WHERE SiteCode = '$SiteCode'" | Format-Table -AutoSize

# --- 1. filter counts and recency ------------------------------------------
Write-Host "[1] spartan.Filters" -ForegroundColor Yellow
Invoke-AqrcQuery $c @"
SELECT COUNT(*) AS filters_in_db,
       MIN(f.SamplingStartDate) AS first_sample,
       MAX(f.SamplingStartDate) AS last_sample,
       SUM(CASE WHEN f.SamplingStartDate > '$LocalMeta' THEN 1 ELSE 0 END) AS newer_than_local
FROM spartan.Filters f
JOIN spartan.Sites s ON s.Id = f.SiteId WHERE s.SiteCode = '$SiteCode'
"@ | Format-Table -AutoSize

# --- 2. the actual new filters ---------------------------------------------
Write-Host "[2] filters sampled after $LocalMeta (new to us)" -ForegroundColor Yellow
$new = Invoke-AqrcQuery $c @"
SELECT f.ExternalFilterId, f.MediaId, f.SamplingStartDate, f.ExternalFilterType,
       f.ExternalLotId, f.SampleVolume_m3,
       (SELECT COUNT(*) FROM ftir.SampleAnalysis sa WHERE sa.MediaId = f.MediaId) AS n_ftir_analyses
FROM spartan.Filters f
JOIN spartan.Sites s ON s.Id = f.SiteId
WHERE s.SiteCode = '$SiteCode' AND f.SamplingStartDate > '$LocalMeta'
ORDER BY f.SamplingStartDate
"@
Write-Host ("      rows: " + $new.Rows.Count)
if ($new.Rows.Count -gt 0) {
    $new | Select-Object -First 15 | Format-Table -AutoSize
    Export-AqrcTable $new "$OutDir\${SiteCode}_new_filters.csv"
}

# --- 3. FTIR analyses / scans ----------------------------------------------
Write-Host "[3] ftir.SampleAnalysis" -ForegroundColor Yellow
Invoke-AqrcQuery $c @"
SELECT COUNT(*) AS analyses_in_db,
       COUNT(DISTINCT sa.MediaId) AS filters_with_ftir,
       MIN(sa.AnalysisDate) AS first_analysis,
       MAX(sa.AnalysisDate) AS last_analysis,
       SUM(CASE WHEN sa.SampleScanId IS NOT NULL
                 AND sa.BackgroundScanId IS NOT NULL THEN 1 ELSE 0 END) AS with_both_scans
FROM ftir.SampleAnalysis sa WHERE sa.MediaId IN ($mediaFor)
"@ | Format-Table -AutoSize

# --- 4. HIPS - real schema, discovered 2026-08-23 ---------------------------
# Networks_1_0 has a dedicated `hips` schema (NOT analysis.Results, which is the
# IMPROVE-side layout and does not exist here).
Write-Host "[4] hips.* schema" -ForegroundColor Yellow
Try-Aqrc $c "hips.Results columns:" @"
SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'hips' AND TABLE_NAME = 'Results' ORDER BY ORDINAL_POSITION
"@
Try-Aqrc $c "hips.ResultTypes (what quantities are stored):" "SELECT * FROM hips.ResultTypes"
Try-Aqrc $c "hips.Results sample row:" "SELECT TOP 3 * FROM hips.Results"
Try-Aqrc $c "$SiteCode rows in hips.Results:" @"
SELECT COUNT(*) AS hips_rows, COUNT(DISTINCT r.MediaId) AS filters_with_hips
FROM hips.Results r WHERE r.MediaId IN ($mediaFor)
"@
# per-lot blank lines almost certainly live here - this is what the lot-253
# question needs (see SPARTAN_LOT_INVENTORY_2026-08-23.md)
Try-Aqrc $c "hips.CalibrationSets:" "SELECT TOP 40 * FROM hips.CalibrationSets ORDER BY Id DESC"
Try-Aqrc $c "hips.CalibrationSetFilters columns:" @"
SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'hips' AND TABLE_NAME = 'CalibrationSetFilters' ORDER BY ORDINAL_POSITION
"@

$c.Close()
Write-Host ""
Write-Host "Local baseline: ETAD_metadata.csv 296 rows through $LocalMeta | ETAD_FTIR_spectra.csv 319 analyses | SPARTAN HIPS 280 ETAD filters (239 Fabs) through $LocalHips" -ForegroundColor Cyan
Write-Host ""
