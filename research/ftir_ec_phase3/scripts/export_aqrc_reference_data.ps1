# Read-only reference exports for the two databases verified by the Davis survey.
# Windows PowerShell 5.1 / SQL Server 2016+. Loading this file only defines functions.
# Connection: AETHMODULAR_AQRC_CONNECTION_STRING, configured locally (never logged).
# Helper: scripts/diagnostics/inspect_sql_access.ps1, or a copy beside this file.

$script:AqrcReferenceSourcePath = $PSCommandPath

function Get-AqrcReferenceExportPlan {
    param([ValidateRange(1,3600)][int]$QueryTimeoutSeconds = 60,
          [ValidateRange(1,3600)][int]$ImproveTimeoutSeconds = 180,
          [switch]$IncludeSpartanCoverage)

    # Explicit projections also make unexpected schema changes fail visibly.
    # SQL style 3 preserves all significant digits of SQL float values, including
    # on Windows PowerShell 5.1, whose default double formatting is less precise.
    @(
        [pscustomobject]@{
            Database='Networks_1_0'; Name='ftir_calibrations'; Timeout=$QueryTimeoutSeconds
            Columns=@('Id','CalibrationSetId','RowNumber','Wavenumber','b')
            Sql=@'
SELECT [Id], [CalibrationSetId], [RowNumber],
       CONVERT(varchar(100), [Wavenumber], 3) AS [Wavenumber],
       CONVERT(varchar(100), [b], 3) AS [b]
FROM [ftir].[Calibrations]
ORDER BY [CalibrationSetId], [RowNumber], [Id];
'@
        }
        [pscustomobject]@{
            Database='Networks_1_0'; Name='ftir_calibration_sets'; Timeout=$QueryTimeoutSeconds
            Columns=@('Id','Parameter','LotNumber','AnalysisQcCode','Label','Comments','ProcessedTime')
            Sql=@'
SELECT [Id], [Parameter], [LotNumber], [AnalysisQcCode], [Label], [Comments], [ProcessedTime]
FROM [ftir].[CalibrationSets] ORDER BY [Id];
'@
        }
        [pscustomobject]@{
            Database='Networks_1_0'; Name='ftir_calibration_shape'; Timeout=$QueryTimeoutSeconds
            Columns=@('CalibrationSetId','CoefficientRows','DistinctRowNumbers','MinRowNumber','MaxRowNumber',
                'DistinctWavenumbers','MinWavenumber','MaxWavenumber','NullRowNumbers','NullWavenumbers','NullCoefficients')
            Sql=@'
SELECT [CalibrationSetId], COUNT_BIG(*) AS [CoefficientRows],
       COUNT_BIG(DISTINCT [RowNumber]) AS [DistinctRowNumbers],
       MIN([RowNumber]) AS [MinRowNumber], MAX([RowNumber]) AS [MaxRowNumber],
       COUNT_BIG(DISTINCT [Wavenumber]) AS [DistinctWavenumbers],
       CONVERT(varchar(100), MIN([Wavenumber]), 3) AS [MinWavenumber],
       CONVERT(varchar(100), MAX([Wavenumber]), 3) AS [MaxWavenumber],
       COUNT_BIG(*) - COUNT_BIG([RowNumber]) AS [NullRowNumbers],
       COUNT_BIG(*) - COUNT_BIG([Wavenumber]) AS [NullWavenumbers],
       COUNT_BIG(*) - COUNT_BIG([b]) AS [NullCoefficients]
FROM [ftir].[Calibrations] GROUP BY [CalibrationSetId] ORDER BY [CalibrationSetId];
'@
        }
        [pscustomobject]@{
            Database='Networks_1_0'; Name='csu_amod_filters'; Timeout=$QueryTimeoutSeconds
            Columns=@('MediaId','ExternalFilterId','ExternalDeviceId','SamplingStart','Latitude','Longitude',
                'SamplingRunTime_hours','MassDifference_micrograms','SampleVolume_liters','ExternalErrorFlag',
                'LotNumber','TubeNumber','UcdShipDate')
            Sql=@'
SELECT [MediaId], [ExternalFilterId], [ExternalDeviceId], [SamplingStart], [Latitude], [Longitude],
       [SamplingRunTime_hours], [MassDifference_micrograms], [SampleVolume_liters], [ExternalErrorFlag],
       [LotNumber], [TubeNumber], [UcdShipDate]
FROM [csu_amod].[Filters] ORDER BY [MediaId], [ExternalFilterId];
'@
        }
        [pscustomobject]@{
            Database='Networks_1_0'; Name='analysis_batches'; Timeout=$QueryTimeoutSeconds
            Columns=@('Id','Label','NetworkId')
            Sql='SELECT [Id], [Label], [NetworkId] FROM [analysis].[Batches] ORDER BY [Id];'
        }
        [pscustomobject]@{
            Database='Networks_1_0'; Name='analysis_media'; Timeout=$QueryTimeoutSeconds
            Columns=@('Id','NetworkId','MediaPurposeId','SampleVolume','AnalysisType','MediaTypeId',
                'SiteAnalysisPathId','Label','Barcode','BatchId')
            Sql=@'
SELECT [Id], [NetworkId], [MediaPurposeId], [SampleVolume], [AnalysisType], [MediaTypeId],
       [SiteAnalysisPathId], [Label], [Barcode], [BatchId]
FROM [analysis].[Media] ORDER BY [Id];
'@
        }
        [pscustomobject]@{
            Database='Networks_1_0'; Name='analysis_parameters'; Timeout=$QueryTimeoutSeconds
            Columns=@('Parameter','Name','Category','Reported','AQSParameterCode','AQSParameterName',
                'AQSUnitCode','AQSMethodCode','ReportToDart','MdlFloor','ProportionalUncertainty')
            Sql=@'
SELECT [Parameter], [Name], [Category], [Reported], [AQSParameterCode], [AQSParameterName],
       [AQSUnitCode], [AQSMethodCode], [ReportToDart],
       CONVERT(varchar(100), [MdlFloor], 3) AS [MdlFloor],
       CONVERT(varchar(100), [ProportionalUncertainty], 3) AS [ProportionalUncertainty]
FROM [analysis].[Parameters] ORDER BY [Parameter];
'@
        }
        [pscustomobject]@{
            Database='Networks_1_0'; Name='analysis_networks'; Timeout=$QueryTimeoutSeconds
            Columns=@('Id','Name','Summary','Description','StartDate','EndDate')
            Sql='SELECT [Id], [Name], [Summary], [Description], [StartDate], [EndDate] FROM [analysis].[Networks] ORDER BY [Id];'
        }
        [pscustomobject]@{
            Database='Networks_1_0'; Name='analysis_media_purposes'; Timeout=$QueryTimeoutSeconds
            Columns=@('Id','Value','Description','IsFieldBlank')
            Sql='SELECT [Id], [Value], [Description], [IsFieldBlank] FROM [analysis].[MediaPurposes] ORDER BY [Id];'
        }
        [pscustomobject]@{
            Database='Networks_1_0'; Name='spartan_filter_batches'; Timeout=$QueryTimeoutSeconds
            Columns=@('MediaId','ExternalFilterId','SiteId','SiteCode','SiteName','SamplingStartDate','SamplingEndDate',
                'ExternalShipmentId','CartridgeNumber','ExternalAnalysisId','ExternalFilterType','ProjectId','ExternalLotId',
                'MassCollectedOnFilter','SampleVolume_m3','MediaFound','MediaNetworkId','MediaPurposeId','MediaTypeId',
                'MediaLabel','Barcode','BatchId','BatchFound','BatchLabel','BatchNetworkId')
            Sql=@'
SELECT f.[MediaId], f.[ExternalFilterId], f.[SiteId], s.[SiteCode], s.[Name] AS [SiteName],
       f.[SamplingStartDate], f.[SamplingEndDate], f.[ExternalShipmentId], f.[CartridgeNumber],
       f.[ExternalAnalysisId], f.[ExternalFilterType], f.[ProjectId], f.[ExternalLotId],
       f.[MassCollectedOnFilter], f.[SampleVolume_m3],
       CAST(CASE WHEN m.[Id] IS NULL THEN 0 ELSE 1 END AS bit) AS [MediaFound],
       m.[NetworkId] AS [MediaNetworkId], m.[MediaPurposeId], m.[MediaTypeId],
       m.[Label] AS [MediaLabel], m.[Barcode], m.[BatchId],
       CAST(CASE WHEN b.[Id] IS NULL THEN 0 ELSE 1 END AS bit) AS [BatchFound],
       b.[Label] AS [BatchLabel], b.[NetworkId] AS [BatchNetworkId]
FROM [spartan].[Filters] f
LEFT JOIN [spartan].[Sites] s ON s.[Id] = f.[SiteId]
LEFT JOIN [analysis].[Media] m ON m.[Id] = f.[MediaId]
LEFT JOIN [analysis].[Batches] b ON b.[Id] = m.[BatchId]
ORDER BY f.[MediaId], f.[ExternalFilterId];
'@
        }
        [pscustomobject]@{
            Database='Improve_2.1'; Name='improve_aqs_parameters'; Timeout=$QueryTimeoutSeconds
            Columns=@('Parameter','AQSParameterCode','AQSParameterName','AQSUnitCode','AQSMethodCode','Category',
                'AnalysisMethod','Unit','ReportToAQS','ReportToFED','FEDParameterName','AQSUnit',
                'AnalyticalMDLUnit','RawUnit','AnalysisTypeCode')
            Sql=@'
SELECT [Parameter], [AQSParameterCode], [AQSParameterName], [AQSUnitCode], [AQSMethodCode], [Category],
       [AnalysisMethod], [Unit], [ReportToAQS], [ReportToFED], [FEDParameterName], [AQSUnit],
       [AnalyticalMDLUnit], [RawUnit], [AnalysisTypeCode]
FROM [analysis].[AqsParameters] ORDER BY [Parameter];
'@
        }
        [pscustomobject]@{
            Database='Improve_2.1'; Name='improve_result_parameters'; Timeout=$ImproveTimeoutSeconds
            Columns=@('Parameter','ParameterIsNull')
            # Do not substitute AqsParameters: a dictionary is not observed data.
            # This can scan a large index/table; the separate timeout is intentional.
            Sql=@'
SELECT DISTINCT [Parameter],
       CAST(CASE WHEN [Parameter] IS NULL THEN 1 ELSE 0 END AS bit) AS [ParameterIsNull]
FROM [analysis].[Results] ORDER BY [Parameter];
'@
        }
    )
    if ($IncludeSpartanCoverage) {
        [pscustomobject]@{
            Database='Networks_1_0'; Name='spartan_coverage_result_types'; Timeout=$QueryTimeoutSeconds
            Columns=@('Id','Name')
            Sql='SELECT [Id], [Name] FROM [hips].[ResultTypes] ORDER BY [Id];'
        }
        [pscustomobject]@{
            Database='Networks_1_0'; Name='spartan_coverage_sites'; Timeout=$QueryTimeoutSeconds
            Columns=@('Id','Name','SiteCode','Description','StartDate','EndDate')
            Sql='SELECT [Id], [Name], [SiteCode], [Description], [StartDate], [EndDate] FROM [spartan].[Sites] ORDER BY [Id];'
        }
        [pscustomobject]@{
            Database='Networks_1_0'; Name='spartan_native_filter_coverage'; Timeout=$QueryTimeoutSeconds
            Columns=@('MediaId','ExternalFilterId','ExternalFilterType','SiteId','SiteCode','SiteName',
                'SamplingStartDate','SamplingEndDate','FtirAnalyses','FtirAnalysesWithPairedScans',
                'FtirFirstAnalysis','FtirLastAnalysis','HipsAnalyses','HipsFirstAnalysis','HipsLastAnalysis',
                'HipsSampleReadings','HipsSampleAnalyses','HipsSampleReadingsWithTR')
            # Aggregate each instrument before joining: repeated FTIR analyses and
            # HIPS readings must not multiply each other's counts. Read scan IDs,
            # never scan Data. Resolve sample types from names, not hardcoded IDs.
            Sql=@'
WITH ft AS (
    SELECT a.[MediaId], COUNT_BIG(*) AS [FtirAnalyses],
           SUM(CAST(CASE WHEN s.[Id] IS NOT NULL AND b.[Id] IS NOT NULL THEN 1 ELSE 0 END AS bigint)) AS [FtirAnalysesWithPairedScans],
           MIN(a.[AnalysisDate]) AS [FtirFirstAnalysis], MAX(a.[AnalysisDate]) AS [FtirLastAnalysis]
    FROM [ftir].[SampleAnalysis] a
    LEFT JOIN [ftir].[Scan] s ON s.[Id] = a.[SampleScanId]
    LEFT JOIN [ftir].[Scan] b ON b.[Id] = a.[BackgroundScanId]
    GROUP BY a.[MediaId]
), ha AS (
    SELECT [MediaId], COUNT_BIG(*) AS [HipsAnalyses],
           MIN([AnalysisDate]) AS [HipsFirstAnalysis], MAX([AnalysisDate]) AS [HipsLastAnalysis]
    FROM [hips].[SampleAnalysis] GROUP BY [MediaId]
), hs AS (
    SELECT a.[MediaId], COUNT_BIG(*) AS [HipsSampleReadings],
           COUNT_BIG(DISTINCT a.[Id]) AS [HipsSampleAnalyses],
           SUM(CAST(CASE WHEN r.[Transmittance] IS NOT NULL AND r.[Reflectance] IS NOT NULL THEN 1 ELSE 0 END AS bigint)) AS [HipsSampleReadingsWithTR]
    FROM [hips].[SampleAnalysis] a
    JOIN [hips].[Results] r ON r.[SampleAnalysisId] = a.[Id]
    JOIN [hips].[ResultTypes] t ON t.[Id] = r.[ResultTypeId]
    WHERE UPPER(LTRIM(RTRIM(t.[Name]))) = N'SAMPLE'
    GROUP BY a.[MediaId]
)
SELECT f.[MediaId], f.[ExternalFilterId], f.[ExternalFilterType], f.[SiteId],
       s.[SiteCode], s.[Name] AS [SiteName], f.[SamplingStartDate], f.[SamplingEndDate],
       COALESCE(ft.[FtirAnalyses],0) AS [FtirAnalyses],
       COALESCE(ft.[FtirAnalysesWithPairedScans],0) AS [FtirAnalysesWithPairedScans],
       ft.[FtirFirstAnalysis], ft.[FtirLastAnalysis],
       COALESCE(ha.[HipsAnalyses],0) AS [HipsAnalyses], ha.[HipsFirstAnalysis], ha.[HipsLastAnalysis],
       COALESCE(hs.[HipsSampleReadings],0) AS [HipsSampleReadings],
       COALESCE(hs.[HipsSampleAnalyses],0) AS [HipsSampleAnalyses],
       COALESCE(hs.[HipsSampleReadingsWithTR],0) AS [HipsSampleReadingsWithTR]
FROM [spartan].[Filters] f
LEFT JOIN [spartan].[Sites] s ON s.[Id] = f.[SiteId]
LEFT JOIN ft ON ft.[MediaId] = f.[MediaId]
LEFT JOIN ha ON ha.[MediaId] = f.[MediaId]
LEFT JOIN hs ON hs.[MediaId] = f.[MediaId]
ORDER BY f.[SiteId], f.[MediaId];
'@
        }
    }
}

function Get-AqrcAdamaFilterCoverage {
    param([object[]]$Filters)
    # Exact IDs from the existing five-pair table, not inferred from spectra order:
    # research/ftir_ec_calibration_2026_06_25/tables/adama_tor_vs_ftir.csv
    $pairs = @(
        @('2024-07-09','J1693','J1269'), @('2024-07-20','J1701','J1266'),
        @('2024-07-23','J1675','J1285'), @('2024-07-26','J1703','J1233'),
        @('2024-07-30','J1679','J1270')
    )
    foreach ($pair in $pairs) {
        foreach ($role in @('quartz','ptfe')) {
            $id = $pair[1]; if ($role -eq 'ptfe') { $id = $pair[2] }
            $matches = @($Filters | Where-Object { $_.ExternalFilterId -ceq $id })
            $status = 'not_found_exact_id'
            if ($matches.Count -eq 1) { $status = 'matched_exact_id' }
            elseif ($matches.Count -gt 1) { $status = 'multiple_exact_id_matches' }
            [pscustomobject]@{
                PairDate=$pair[0]; Role=$role; ExternalFilterId=$id; Status=$status
                MatchedRows=$matches.Count; MediaIds=(($matches | ForEach-Object { $_.MediaId }) -join ';')
            }
        }
    }
}

function Invoke-AqrcReferenceExport {
    [CmdletBinding()]
    param(
        [string]$HelperScript,
        [string]$OutputRoot = (Join-Path $HOME 'Downloads/aqrc_reference_exports'),
        [ValidateRange(1,3600)][int]$QueryTimeoutSeconds = 60,
        [ValidateRange(1,3600)][int]$ImproveTimeoutSeconds = 180,
        [switch]$SkipImproveParameterCheck,
        [switch]$IncludeSpartanCoverage,
        [switch]$SpartanCoverageOnly
    )
    $ErrorActionPreference = 'Stop'
    $sourcePath = $script:AqrcReferenceSourcePath
    if (-not $HelperScript) {
        $candidates = @(
            (Join-Path (Split-Path $sourcePath) 'inspect_sql_access.ps1'),
            (Join-Path (Split-Path $sourcePath) '../../../scripts/diagnostics/inspect_sql_access.ps1')
        )
        foreach ($candidate in $candidates) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { $HelperScript = $candidate; break }
        }
    }
    if (-not $HelperScript -or -not (Test-Path -LiteralPath $HelperScript -PathType Leaf)) {
        throw 'Place inspect_sql_access.ps1 beside this script, or supply -HelperScript with its location.'
    }
    # Always load the on-disk helper to avoid stale definitions in a long session.
    . $HelperScript
    if (-not $env:AETHMODULAR_AQRC_CONNECTION_STRING) {
        throw 'Set AETHMODULAR_AQRC_CONNECTION_STRING from local configuration first.'
    }
    $out = Join-Path $OutputRoot ((Get-Date -Format yyyyMMdd_HHmmss) + '_' + [guid]::NewGuid().ToString('N').Substring(0,8))
    New-Item -ItemType Directory -Path $out | Out-Null
    $state = [pscustomobject]@{
        Errors=(New-Object 'System.Collections.Generic.List[object]')
        Queries=(New-Object 'System.Collections.Generic.List[object]')
        Exports=(New-Object 'System.Collections.Generic.List[object]')
    }
    $started = [datetime]::UtcNow.ToString('o'); $status = 'interrupted'; $amodRows = $null; $amodQueryId = ''
    $plan = @(Get-AqrcReferenceExportPlan $QueryTimeoutSeconds $ImproveTimeoutSeconds `
        -IncludeSpartanCoverage:($IncludeSpartanCoverage -or $SpartanCoverageOnly))
    if ($SpartanCoverageOnly) { $plan = @($plan | Where-Object { $_.Name -like 'spartan_*coverage*' }) }
    $databases = @($plan | Select-Object -ExpandProperty Database -Unique)
    $sampleTypeConfirmed = $false
    try {
        foreach ($database in $databases) {
            $connection = $null
            try {
                try { $connection = Open-SqlSurveyConnection $env:AETHMODULAR_AQRC_CONNECTION_STRING $database }
                catch { Add-SqlSurveyError $state $database 'connect' '' $_ }
                foreach ($item in @($plan | Where-Object { $_.Database -eq $database })) {
                    $entry = [pscustomobject]@{
                        Database=$database; Dataset=$item.Name; Status='pending'; CsvRows=$null; File=''; QueryId=''
                    }
                    $state.Exports.Add($entry)
                    if ($item.Name -eq 'improve_result_parameters' -and $SkipImproveParameterCheck) {
                        $entry.Status='skipped_by_option'; continue
                    }
                    if ($null -eq $connection) { $entry.Status='connection_failed'; continue }
                    if ($item.Name -eq 'spartan_native_filter_coverage' -and -not $sampleTypeConfirmed) {
                        $entry.Status='skipped_missing_sample_result_type'; continue
                    }
                    Write-Host ('Reading {0}: {1}' -f $database,$item.Name)
                    try {
                        $table = Invoke-SqlSurveyQuery -Connection $connection -Sql $item.Sql -State $state `
                            -Database $database -Stage 'reference_export' -Object $item.Name -Timeout $item.Timeout
                        if ($state.Queries.Count) { $entry.QueryId=$state.Queries[$state.Queries.Count-1].QueryId }
                        if ($null -eq $table) { $entry.Status='query_failed'; continue }
                        $rows = @(Convert-SqlSurveyRows $table $database)
                        $file = $item.Name+'.csv'
                        Export-SqlSurveyCsv $rows (@('Database')+$item.Columns) (Join-Path $out $file)
                        $entry.Status='ok'; $entry.CsvRows=$rows.Count; $entry.File=$file
                        if ($item.Name -eq 'spartan_coverage_result_types') {
                            $sampleTypeConfirmed = @($rows | Where-Object { ([string]$_.Name).Trim() -ieq 'Sample' }).Count -gt 0
                        }
                        if ($item.Name -eq 'csu_amod_filters') {
                            $amodRows = $rows; $amodQueryId = $entry.QueryId
                        }
                    } catch {
                        $entry.Status='export_failed'
                        Add-SqlSurveyError $state $database 'reference_export' $item.Name $_
                    }
                }
            } finally { if ($null -ne $connection) { $connection.Dispose() } }
        }
        $coverageEntry = [pscustomobject]@{
            Database='Networks_1_0'; Dataset='adama_filter_coverage'; Status='skipped_source_unavailable'
            CsvRows=$null; File=''; QueryId=$amodQueryId
        }
        if (@($plan | Where-Object Name -eq 'csu_amod_filters').Count) { $state.Exports.Add($coverageEntry) }
        if ($null -ne $amodRows) {
            try {
                $coverage = @(Get-AqrcAdamaFilterCoverage $amodRows)
                $coverageFile='adama_filter_coverage.csv'
                Export-SqlSurveyCsv $coverage @('PairDate','Role','ExternalFilterId','Status','MatchedRows','MediaIds') (Join-Path $out $coverageFile)
                $coverageEntry.Status='ok'; $coverageEntry.CsvRows=$coverage.Count; $coverageEntry.File=$coverageFile
            } catch {
                $coverageEntry.Status='export_failed'
                Add-SqlSurveyError $state 'Networks_1_0' 'derived_export' 'adama_filter_coverage' $_
            }
        }
        $status='complete'
        if ($state.Errors.Count -or @($state.Exports | Where-Object { $_.Status -notin @('ok','skipped_by_option') }).Count) {
            $status='partial_with_errors'
        } elseif (@($state.Exports | Where-Object Status -eq 'skipped_by_option').Count) { $status='complete_with_skips' }
    } finally {
        Export-SqlSurveyCsv $state.Exports @('Database','Dataset','Status','CsvRows','File','QueryId') (Join-Path $out 'exports.csv')
        Export-SqlSurveyCsv $state.Errors @('Database','Stage','Object','SqlErrorNumber','ExceptionType','Utc') (Join-Path $out 'errors.csv')
        $lines = @($state.Queries | ForEach-Object { $_ | ConvertTo-Json -Depth 6 -Compress })
        [System.IO.File]::WriteAllLines((Join-Path $out 'queries.jsonl'),[string[]]$lines,(New-Object System.Text.UTF8Encoding($false)))
        [ordered]@{
            Status=$status; StartedUtc=$started; FinishedUtc=[datetime]::UtcNow.ToString('o')
            SourceSHA256=(Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash
            HelperSHA256=(Get-FileHash -LiteralPath $HelperScript -Algorithm SHA256).Hash
            Databases=$databases; QueryTimeoutSeconds=$QueryTimeoutSeconds
            ImproveTimeoutSeconds=$ImproveTimeoutSeconds; SkipImproveParameterCheck=[bool]$SkipImproveParameterCheck
            IncludeSpartanCoverage=[bool]($IncludeSpartanCoverage -or $SpartanCoverageOnly); SpartanCoverageOnly=[bool]$SpartanCoverageOnly
            SuccessfulExports=@($state.Exports | Where-Object { $_.Status -eq 'ok' }).Count; Errors=$state.Errors.Count
            Visibility='Rows visible to the current login only; permission and row-security limits still apply.'
            Consistency='Separate SELECT queries, not an atomic snapshot. No NOLOCK or session policy changes.'
            Values='Selected reference rows and metadata; IMPROVE Results exports distinct parameter names only.'
            Nulls='Empty CSV fields can represent SQL NULL or empty text; ParameterIsNull distinguishes these in the IMPROVE check.'
            Precision='SQL float fields are converted on the server with style 3. Dates use ISO 8601; decimals use invariant culture.'
        } | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $out 'receipt.json') -Encoding UTF8
        @(
            ('AQRC reference export: '+$status)
            'Start with exports.csv: ok means the CSV was written and its row count verified by reading it back.'
            'Failed/skipped exports have no promised data file; they do not establish an empty source.'
            'queries.jsonl records exact SELECT statements and durations; errors.csv records safe error codes/types.'
            'manifest.csv records filenames, verified CSV row counts and SHA256 hashes (excluding itself).'
            'improve_result_parameters.csv contains names actually observed in analysis.Results. Check its export status first.'
            'A missing functional-group column or dictionary entry alone cannot rule out a long-format result.'
            'Calibration rows are coefficients. Establish preprocessing, grid alignment, intercept and selection conventions before prediction.'
            'Database batch IDs/labels need a lab crosswalk before equating them with delivery batch numbers.'
            'adama_filter_coverage.csv compares exact quartz/PTFE IDs from the existing five-pair table; unmatched IDs need investigation.'
            'Raw AMOD mass, volume and error flags are retained. ug/m3 = 1000 * MassDifference_micrograms / SampleVolume_liters.'
            'Check positive volume, QC, sampler size cut and weighing conventions before interpreting that ratio as gravimetric PM2.5.'
            'Optional spartan_native_filter_coverage.csv inventories raw analyses/readings per media; these are not official FTIR/HIPS products.'
            'Sample-result labels come from hips.ResultTypes. A missing lookup skips the native coverage query instead of reporting false zeroes.'
            'Compare native coverage with public ChemSpec and lab deliveries using audit_spartan_network_coverage.py in the repo.'
        ) | Set-Content (Join-Path $out 'START_HERE.txt') -Encoding UTF8
        $manifest = @(Get-ChildItem -LiteralPath $out -File | Sort-Object Name | ForEach-Object {
            $count=$null; if ($_.Extension -eq '.csv') { $count=@(Import-Csv -LiteralPath $_.FullName).Count }
            [pscustomobject]@{File=$_.Name; Bytes=$_.Length; CsvRows=$count; SHA256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}
        })
        Export-SqlSurveyCsv $manifest @('File','Bytes','CsvRows','SHA256') (Join-Path $out 'manifest.csv')
        Write-Host ('Export {0}: {1} successful datasets, {2} recorded errors.' -f $status,@($state.Exports | Where-Object { $_.Status -eq 'ok' }).Count,$state.Errors.Count)
        Write-Host "Saved verified exports: $out"
    }
    return $out
}
