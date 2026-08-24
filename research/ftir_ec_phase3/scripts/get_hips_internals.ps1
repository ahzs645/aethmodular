# get_hips_internals.ps1 - pull the HIPS internals the shipped CSV does not expose
#
# Follow-up to check_aqrc_gaps.ps1, which established (2026-08-23) that
# Networks_1_0 has a dedicated `hips` schema and that hips.Results carries RAW
# instrument data: Transmittance/Reflectance (+Raw), Wavelength (633),
# LaserPower, LaserDiodeTemperature and two sensor temperatures - none of which
# reach SPARTAN_HIPS_Batch1-51.v2.csv.
#
# Goal: (a) map the hips schema properly instead of guessing column names,
#       (b) count the lab blanks behind each per-lot calibration set,
#       (c) find where the Intercept/Slope coefficients actually live,
#       (d) export ETAD raw HIPS rows with the drift diagnostics attached.
#
# Read-only. Pure ASCII. Paste-safe (no param block).
#   powershell -ExecutionPolicy Bypass -File .\get_hips_internals.ps1

# NOTE: deliberately NOT named $SiteCode - that variable survives in the console
# from check_aqrc_gaps.ps1 and would silently narrow this pull to a single site.
# "" = ALL SPARTAN sites (small table: roughly 2 rows per analysis).
$HipsSite = ""   # set to e.g. "ETAD" for one site only
if (-not $OutDir)   { $OutDir   = "$HOME\Downloads" }

function Open-AqrcConn([string]$db = "Networks_1_0") {
    $cs = "Server=AQRC-SQL,1433;Integrated Security=SSPI;TrustServerCertificate=True;Database=$db;"
    $conn = New-Object System.Data.SqlClient.SqlConnection($cs); $conn.Open(); return $conn
}
function Invoke-AqrcQuery($conn, [string]$sql) {
    $cmd = $conn.CreateCommand(); $cmd.CommandText = $sql; $cmd.CommandTimeout = 600
    $table = New-Object System.Data.DataTable
    (New-Object System.Data.SqlClient.SqlDataAdapter($cmd)).Fill($table) | Out-Null
    return ,$table
}
function Export-AqrcTable($table, [string]$path) {
    if ($table -isnot [System.Data.DataTable]) { $table = @($table)[0].Table }
    $table | Export-Csv $path -NoTypeInformation -Encoding UTF8
    Write-Host ("      -> " + $path + "  (" + $table.Rows.Count + " rows)")
}
function Try-Aqrc($conn, [string]$label, [string]$sql) {
    Write-Host ("      " + $label)
    try { Invoke-AqrcQuery $conn $sql | Format-Table -AutoSize }
    catch { Write-Host ("      SKIP: " + $_.Exception.Message) -ForegroundColor DarkGray }
}

$c = Open-AqrcConn
Write-Host ""
Write-Host "=== [A] full hips.* schema (stop guessing column names) ===" -ForegroundColor Cyan
Invoke-AqrcQuery $c @"
SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'hips'
ORDER BY TABLE_NAME, ORDINAL_POSITION
"@ | Format-Table -AutoSize

Write-Host "=== [B] where do Intercept/Slope live, anywhere in the DB? ===" -ForegroundColor Cyan
Try-Aqrc $c "columns named like slope/intercept/coeff:" @"
SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE COLUMN_NAME LIKE '%lope%' OR COLUMN_NAME LIKE '%ntercept%'
   OR COLUMN_NAME LIKE '%oeff%' OR COLUMN_NAME LIKE '%Tau%'
ORDER BY TABLE_SCHEMA, TABLE_NAME
"@

Write-Host "=== [C] lab blanks behind each per-lot calibration set ===" -ForegroundColor Cyan
Try-Aqrc $c "blanks per calibration set:" @"
SELECT cs.Id, cs.LotNumber, cs.Label, COUNT(csf.MediaId) AS n_blanks
FROM hips.CalibrationSets cs
LEFT JOIN hips.CalibrationSetFilters csf ON csf.CalibrationSetId = cs.Id
GROUP BY cs.Id, cs.LotNumber, cs.Label
ORDER BY cs.LotNumber, cs.Id
"@

Write-Host "=== [D] how hips.Results reaches a filter ===" -ForegroundColor Cyan
# hips.Results keys on SampleAnalysisId - find the analysis table that resolves it
Try-Aqrc $c "candidate analysis tables carrying MediaId:" @"
SELECT TABLE_SCHEMA, TABLE_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE COLUMN_NAME = 'MediaId' AND TABLE_SCHEMA IN ('hips','spartan','analysis','lab')
GROUP BY TABLE_SCHEMA, TABLE_NAME ORDER BY TABLE_SCHEMA, TABLE_NAME
"@
Try-Aqrc $c "hips.SampleAnalysis columns (if it exists):" @"
SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'hips' AND TABLE_NAME = 'SampleAnalysis' ORDER BY ORDINAL_POSITION
"@

$siteLabel = if ($HipsSite) { $HipsSite } else { "ALL SITES" }
$siteWhere = if ($HipsSite) { "WHERE s.SiteCode = '$HipsSite'" } else { "" }
$siteTag   = if ($HipsSite) { $HipsSite } else { "all" }
Write-Host "=== [E] $siteLabel raw HIPS with drift diagnostics ===" -ForegroundColor Cyan
# Join path assumes hips.SampleAnalysis(Id, MediaId). If [D] shows otherwise,
# swap the table name here and re-run - everything else stands.
$sql = @"
SELECT s.SiteCode, f.ExternalFilterId, f.ExternalLotId, f.SamplingStartDate,
       r.SampleAnalysisId, r.ResultTypeId, r.Timestamp, r.Wavelength,
       r.Transmittance, r.Reflectance, r.TransmittanceRaw, r.ReflectanceRaw,
       r.LaserPower, r.LaserDiodeTemperature,
       r.TransmittanceSensorTemperature, r.ReflectanceSensorTemperature
FROM hips.Results r
JOIN hips.SampleAnalysis sa ON sa.Id = r.SampleAnalysisId
JOIN spartan.Filters f ON f.MediaId = sa.MediaId
JOIN spartan.Sites s ON s.Id = f.SiteId
$siteWhere
ORDER BY f.SamplingStartDate, r.Timestamp
"@
try {
    $t = Invoke-AqrcQuery $c $sql
    Write-Host ("      rows: " + $t.Rows.Count)
    $t | Select-Object -First 5 | Format-Table -AutoSize
    Export-AqrcTable $t "$OutDir\spartan_hips_raw_${siteTag}.csv"
} catch {
    Write-Host ("      SKIP (fix the join from [D] and re-run): " + $_.Exception.Message) -ForegroundColor DarkGray
}

Write-Host "=== [F] ALL sites' laser power / temperature over time (drift check) ===" -ForegroundColor Cyan
# network-wide, so an instrument shift shows up as a step shared across sites
try {
    $t2 = Invoke-AqrcQuery $c @"
SELECT CAST(r.Timestamp AS date) AS day, COUNT(*) AS n,
       AVG(r.LaserPower) AS laser_power, AVG(r.LaserDiodeTemperature) AS laser_temp,
       AVG(r.TransmittanceSensorTemperature) AS t_sensor_temp,
       AVG(r.ReflectanceSensorTemperature) AS r_sensor_temp
FROM hips.Results r GROUP BY CAST(r.Timestamp AS date) ORDER BY day
"@
    Write-Host ("      days: " + $t2.Rows.Count)
    Export-AqrcTable $t2 "$OutDir\hips_instrument_daily.csv"
} catch { Write-Host ("      SKIP: " + $_.Exception.Message) -ForegroundColor DarkGray }

$c.Close()
Write-Host ""
