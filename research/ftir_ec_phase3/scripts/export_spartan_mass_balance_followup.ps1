# Read-only, requested-filter export. No connection settings or policy changes.
$script:SpartanMassBalanceSource = $PSCommandPath

function Get-SpartanMassBalancePlan([string[]]$FilterId) {
    if (-not $FilterId.Count -or $FilterId.Count -gt 1000) { throw 'Supply 1-1000 physical filter IDs.' }
    foreach ($id in $FilterId) {
        if ($id -cnotmatch '^[A-Z]{4}-[0-9]{4}$') { throw 'Invalid physical filter ID.' }
    }
    $values = (@($FilterId | Sort-Object -Unique | ForEach-Object { "('$_')" }) -join ',')
    # Prefix matching selects candidate replicates; the importer validates each
    # against config.BASE_FILTER_ID_PATTERN before joining physical filters.
    $request = "WITH requested AS (SELECT v.FilterId FROM (VALUES $values) v(FilterId))"
    $media = $request + ", cohort AS (SELECT r.FilterId RequestedFilterId,f.* FROM requested r JOIN spartan.Filters f ON f.ExternalFilterId=r.FilterId OR f.ExternalFilterId LIKE r.FilterId+'-%') "
    @(
        [pscustomobject]@{Name='requested_filter_coverage'; Sql=$request+@'
 SELECT r.FilterId RequestedFilterId,f.ExternalFilterId,f.MediaId,s.SiteCode,
 f.SamplingStartDate,f.SamplingEndDate,f.ExternalFilterType,f.ExternalLotId,
 f.ExternalShipmentId,f.CartridgeNumber,f.ExternalAnalysisId,f.ProjectId,
 f.MassCollectedOnFilter,f.SampleVolume_m3,m.SampleVolume MediaSampleVolume,
 m.MediaPurposeId,m.MediaTypeId,m.BatchId,b.Label BatchLabel,
 p.Value MediaPurpose,p.Description MediaPurposeDescription,p.IsFieldBlank
 FROM requested r LEFT JOIN spartan.Filters f ON f.ExternalFilterId=r.FilterId OR f.ExternalFilterId LIKE r.FilterId+'-%'
 LEFT JOIN spartan.Sites s ON s.Id=f.SiteId
 LEFT JOIN analysis.Media m ON m.Id=f.MediaId
 LEFT JOIN analysis.Batches b ON b.Id=m.BatchId
 LEFT JOIN analysis.MediaPurposes p ON p.Id=m.MediaPurposeId
 ORDER BY r.FilterId,f.MediaId;
'@}
        [pscustomobject]@{Name='hips_analyses'; Sql=$media+@'
 SELECT c.RequestedFilterId,c.ExternalFilterId,c.MediaId,a.Id AnalysisId,
 a.AnalysisDate,a.AnalysisQcCode,a.InstrumentId,a.ImportDate,a.Comments,
 a.AnalysisTypeId,t.Name AnalysisType,a.RegistrationFilterId,a.RegistrationAnalysisId
 FROM cohort c JOIN hips.SampleAnalysis a ON a.MediaId=c.MediaId
 LEFT JOIN hips.AnalysisTypes t ON t.Id=a.AnalysisTypeId
 ORDER BY c.RequestedFilterId,a.Id;
'@}
        [pscustomobject]@{Name='hips_readings'; Sql=$media+@'
 SELECT c.RequestedFilterId,c.ExternalFilterId,c.MediaId,a.Id AnalysisId,
 r.Id ReadingId,r.Timestamp,r.ResultTypeId,t.Name ResultType,r.Wavelength,
 r.Transmittance,r.Reflectance,r.TransmittanceRaw,r.ReflectanceRaw,
 r.TransmittanceSensorTemperature,r.ReflectanceSensorTemperature,
 r.LaserDiodeTemperature,r.LaserPower
 FROM cohort c JOIN hips.SampleAnalysis a ON a.MediaId=c.MediaId
 JOIN hips.Results r ON r.SampleAnalysisId=a.Id
 LEFT JOIN hips.ResultTypes t ON t.Id=r.ResultTypeId
 ORDER BY c.RequestedFilterId,a.Id,r.Id;
'@}
        [pscustomobject]@{Name='hips_flags'; Sql=$media+@'
 SELECT c.RequestedFilterId,c.ExternalFilterId,c.MediaId,a.Id AnalysisId,
 f.Parameter,f.AqsCodeId,q.QualifierCode,q.QualifierDescription,q.QualifierTypeDescription
 FROM cohort c JOIN hips.SampleAnalysis a ON a.MediaId=c.MediaId
 JOIN hips.AnalysisFlags f ON f.SampleAnalysisId=a.Id
 LEFT JOIN analysis.AqsCodes q ON q.Id=f.AqsCodeId
 ORDER BY c.RequestedFilterId,a.Id,f.Parameter,f.AqsCodeId;
'@}
        [pscustomobject]@{Name='ftir_analyses'; Sql=$media+@'
 SELECT c.RequestedFilterId,c.ExternalFilterId,c.MediaId,a.Id AnalysisId,
 a.SampleName,a.SampleScanId,a.BackgroundScanId,a.IngestDate,a.AnalysisDate,
 a.AnalysisQcCode,a.AnalysisComments,a.InstrumentId,
 s.NumberOfDataPoints SamplePoints,s.DataPointFormat SampleFormat,
 s.FrequencyOfFirstPoint SampleFirst,s.FrequencyOfLastPoint SampleLast,
 DATALENGTH(s.Data) SampleBytes,DATALENGTH(b.Data) BackgroundBytes
 FROM cohort c JOIN ftir.SampleAnalysis a ON a.MediaId=c.MediaId
 LEFT JOIN ftir.Scan s ON s.Id=a.SampleScanId
 LEFT JOIN ftir.Scan b ON b.Id=a.BackgroundScanId
 ORDER BY c.RequestedFilterId,a.Id;
'@}
        [pscustomobject]@{Name='media_comments'; Sql=$media+@'
 SELECT c.RequestedFilterId,c.ExternalFilterId,c.MediaId,m.Id CommentId,
 m.Entry,m.EntryDate,m.ModifiedDate
 FROM cohort c JOIN analysis.MediaComments m ON m.MediaId=c.MediaId
 ORDER BY c.RequestedFilterId,m.Id;
'@}
        [pscustomobject]@{Name='filter_integrity'; Sql=$media+@'
 SELECT c.RequestedFilterId,c.ExternalFilterId,c.MediaId,i.Id IntegrityCheckId,
 i.FilterIdentifier,i.BatchIdentifier,i.ReportedDate,i.FilterIntegrityIssueId,
 d.Issue,i.FilterIssueLocation,i.Comments,i.Processed
 FROM cohort c JOIN ftir.FilterIntegrityChecks i ON i.MediaId=c.MediaId
 LEFT JOIN ftir.FilterIntegrityIssues d ON d.Id=i.FilterIntegrityIssueId
 ORDER BY c.RequestedFilterId,i.Id;
'@}
        [pscustomobject]@{Name='qc_codes'; Sql='SELECT QcCode,Definition,Description FROM analysis.QcCodes ORDER BY QcCode;'}
        [pscustomobject]@{Name='visible_result_objects'; Sql=@'
SELECT s.name SchemaName,o.name ObjectName,o.type_desc ObjectType,
 c.column_id ColumnOrdinal,c.name ColumnName,t.name DataType
 FROM sys.objects o JOIN sys.schemas s ON s.schema_id=o.schema_id
 JOIN sys.columns c ON c.object_id=o.object_id
 JOIN sys.types t ON t.user_type_id=c.user_type_id
 WHERE o.type IN ('U','V') AND s.name IN ('spartan','analysis','hips','ftir','xrf','qcarbon')
 ORDER BY s.name,o.name,c.column_id;
'@}
    )
}

function Invoke-SpartanMassBalanceExport {
    [CmdletBinding()]
    param(
        [string]$RequestCsv = (Join-Path (Split-Path $script:SpartanMassBalanceSource) 'requested_filters.csv'),
        [string]$HelperScript = (Join-Path (Split-Path $script:SpartanMassBalanceSource) 'inspect_sql_access.ps1'),
        [string]$OutputRoot = (Join-Path $HOME 'Downloads/aqrc_mass_balance_exports'),
        [ValidateRange(1,600)][int]$TimeoutSeconds = 60
    )
    $ErrorActionPreference='Stop'
    if (-not $env:AETHMODULAR_AQRC_CONNECTION_STRING) { throw 'Configure the existing AQRC session first.' }
    . $HelperScript
    $requests=@(Import-Csv -LiteralPath $RequestCsv)
    $ids=@($requests | ForEach-Object { [string]$_.FilterId })
    if (@($ids | Sort-Object -Unique).Count -ne $ids.Count) { throw 'Duplicate requested physical filters.' }
    $plan=@(Get-SpartanMassBalancePlan $ids)
    foreach ($item in $plan) {
        if ($item.Sql -notmatch '^\s*(WITH|SELECT)\b' -or $item.Sql -match '\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|EXEC|NOLOCK)\b') {
            throw 'Query plan must remain SELECT-only.'
        }
    }
    $out=Join-Path $OutputRoot ((Get-Date -Format yyyyMMdd_HHmmss)+'_'+[guid]::NewGuid().ToString('N').Substring(0,8))
    New-Item -ItemType Directory -Path $out | Out-Null
    Copy-Item -LiteralPath $RequestCsv -Destination (Join-Path $out 'requested_filters.csv')
    $state=[pscustomobject]@{
        Queries=(New-Object 'Collections.Generic.List[object]')
        Errors=(New-Object 'Collections.Generic.List[object]')
        Exports=(New-Object 'Collections.Generic.List[object]')
    }
    $started=[datetime]::UtcNow.ToString('o');$status='interrupted';$connection=$null
    try {
        try { $connection=Open-SqlSurveyConnection $env:AETHMODULAR_AQRC_CONNECTION_STRING 'Networks_1_0' }
        catch { Add-SqlSurveyError $state 'Networks_1_0' 'connect' '' $_ }
        foreach ($item in $plan) {
            $entry=[pscustomobject]@{Dataset=$item.Name;Status='pending';CsvRows=$null;File='';QueryId=''}
            $state.Exports.Add($entry)
            if ($null -eq $connection) { $entry.Status='connection_failed';continue }
            Write-Host ('Reading requested cohort: '+$item.Name)
            try {
                $table=Invoke-SqlSurveyQuery -Connection $connection -Sql $item.Sql -State $state `
                    -Database 'Networks_1_0' -Stage 'mass_balance_followup' -Object $item.Name -Timeout $TimeoutSeconds
                if ($state.Queries.Count) { $entry.QueryId=$state.Queries[$state.Queries.Count-1].QueryId }
                if ($null -eq $table) { $entry.Status='query_failed';continue }
                $rows=@(Convert-SqlSurveyRows $table 'Networks_1_0')
                $columns=@('Database')+@($table.Columns | ForEach-Object { $_.ColumnName })
                $entry.File=$item.Name+'.csv'
                Export-SqlSurveyCsv $rows $columns (Join-Path $out $entry.File)
                $entry.CsvRows=$rows.Count;$entry.Status='ok'
            } catch {
                $entry.Status='export_failed';Add-SqlSurveyError $state 'Networks_1_0' 'export' $item.Name $_
            }
        }
        $status='complete'
        if ($state.Errors.Count -or @($state.Exports | Where-Object Status -ne 'ok').Count) { $status='partial_with_errors' }
    } finally {
        if ($null -ne $connection) { $connection.Dispose() }
        Export-SqlSurveyCsv $state.Exports @('Dataset','Status','CsvRows','File','QueryId') (Join-Path $out 'exports.csv')
        Export-SqlSurveyCsv $state.Errors @('Database','Stage','Object','SqlErrorNumber','ExceptionType','Utc') (Join-Path $out 'errors.csv')
        [IO.File]::WriteAllLines((Join-Path $out 'queries.jsonl'),[string[]]@($state.Queries | ForEach-Object { $_ | ConvertTo-Json -Depth 5 -Compress }),(New-Object Text.UTF8Encoding($false)))
        [ordered]@{
            Status=$status;StartedUtc=$started;FinishedUtc=[datetime]::UtcNow.ToString('o')
            RequestedFilters=$ids.Count;Errors=$state.Errors.Count
            SuccessfulExports=@($state.Exports | Where-Object Status -eq 'ok').Count
            SourceSHA256=(Get-FileHash -LiteralPath $script:SpartanMassBalanceSource -Algorithm SHA256).Hash
            HelperSHA256=(Get-FileHash -LiteralPath $HelperScript -Algorithm SHA256).Hash
            RequestSHA256=(Get-FileHash -LiteralPath $RequestCsv -Algorithm SHA256).Hash
            Scope='Requested physical filters and candidate replicate identities only; contextual QC dictionary and visible schema metadata.'
            Limitations='Raw HIPS readings and FTIR analysis metadata are not final BC, EC, ions or a chemical mass balance. FTIR scan payloads are not exported.'
            Missingness='A successful zero-row CSV is empty. A failed dataset is unresolved. Missing registry rows remain in requested_filter_coverage.'
            MassUnits='MassCollectedOnFilter and volume are exported under native field names; conversion requires unit/QC confirmation.'
            Consistency='Separate read-only SELECT queries; no database, permission, TLS or execution-policy changes.'
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $out 'receipt.json') -Encoding UTF8
        $manifest=@(Get-ChildItem -LiteralPath $out -File | Sort-Object Name | ForEach-Object {
            $n=$null;if ($_.Extension -eq '.csv') { $n=@(Import-Csv -LiteralPath $_.FullName).Count }
            [pscustomobject]@{File=$_.Name;Bytes=$_.Length;CsvRows=$n;SHA256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}
        })
        Export-SqlSurveyCsv $manifest @('File','Bytes','CsvRows','SHA256') (Join-Path $out 'manifest.csv')
    }
    # Verify the actual files before packaging. Never substitute partial for complete.
    foreach ($entry in @(Import-Csv (Join-Path $out 'manifest.csv'))) {
        $path=Join-Path $out $entry.File
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.SHA256) { throw 'Output hash mismatch.' }
        if ($entry.CsvRows -ne '' -and @(Import-Csv -LiteralPath $path).Count -ne [long]$entry.CsvRows) { throw 'Output row count mismatch.' }
    }
    $zip=$out+'.zip'
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory($out,$zip)
    $zipHash=(Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash
    ($zipHash+'  '+[IO.Path]::GetFileName($zip)) | Set-Content ($zip+'.sha256.txt') -Encoding ASCII
    Write-Host ('AQRC cohort export: '+$status+'; '+$ids.Count+' requested filters; '+$state.Errors.Count+' errors.')
    Write-Host ('Verified ZIP SHA256: '+$zipHash)
    Write-Host ('Output archive: '+$zip)
    return $zip
}
