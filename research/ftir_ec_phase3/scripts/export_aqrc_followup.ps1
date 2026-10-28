# Read-only AMOD FTIR export and a bounded search of user-specified lab folders.
# Requires inspect_sql_access.ps1 beside this file, or the repository helper.
# Loading only defines functions. No policy changes, SQL writes or uploads.
$script:AqrcFollowupSourcePath = $PSCommandPath

function Get-AqrcFollowupPlan {
    @(
        [pscustomobject]@{ Name='fg_calibration_sources'; Columns=@('Id','Parameter','LotNumber','Comments'); Sql='SELECT [Id], [Parameter], [LotNumber], [Comments] FROM [ftir].[CalibrationSets] ORDER BY [Id];' }
        [pscustomobject]@{ Name='amod_filters'; Columns=@('MediaId','ExternalFilterId','ExternalDeviceId','SamplingStart','Latitude','Longitude','SamplingRunTime_hours','MassDifference_micrograms','SampleVolume_liters','ExternalErrorFlag','LotNumber','TubeNumber','UcdShipDate'); Sql=@'
SELECT [MediaId], [ExternalFilterId], [ExternalDeviceId], [SamplingStart], [Latitude], [Longitude],
       [SamplingRunTime_hours], [MassDifference_micrograms], [SampleVolume_liters], [ExternalErrorFlag],
       [LotNumber], [TubeNumber], [UcdShipDate]
FROM [csu_amod].[Filters] ORDER BY [MediaId];
'@ }
        [pscustomobject]@{ Name='amod_media'; Columns=@('Id','MediaTypeId','MediaPurposeId','NetworkId','BatchId','Label','Barcode'); Sql=@'
SELECT m.[Id], m.[MediaTypeId], m.[MediaPurposeId], m.[NetworkId], m.[BatchId], m.[Label], m.[Barcode]
FROM [analysis].[Media] m WHERE EXISTS (SELECT 1 FROM [csu_amod].[Filters] f WHERE f.[MediaId]=m.[Id])
ORDER BY m.[Id];
'@ }
        [pscustomobject]@{ Name='media_types'; Columns=@('Id','TypeName','SampleDepositArea','FilterMaterial','NominalVolume','SamplerTypeId'); Sql=@'
SELECT [Id], [TypeName], CONVERT(varchar(100),[SampleDepositArea],3) AS [SampleDepositArea],
       [FilterMaterial], CONVERT(varchar(100),[NominalVolume],3) AS [NominalVolume], [SamplerTypeId]
FROM [analysis].[MediaTypes] ORDER BY [Id];
'@ }
        [pscustomobject]@{ Name='amod_ftir_analyses'; Columns=@('Id','MediaId','SampleScanId','BackgroundScanId','InstrumentId','SampleName','SampleForm','Filename','IngestDate','AnalysisDate','AnalysisQcCode','AnalysisComments'); Sql=@'
SELECT a.[Id], a.[MediaId], a.[SampleScanId], a.[BackgroundScanId], a.[InstrumentId], a.[SampleName],
       a.[SampleForm], a.[Filename], a.[IngestDate], a.[AnalysisDate], a.[AnalysisQcCode], a.[AnalysisComments]
FROM [ftir].[SampleAnalysis] a
WHERE EXISTS (SELECT 1 FROM [csu_amod].[Filters] f WHERE f.[MediaId]=a.[MediaId])
ORDER BY a.[MediaId], a.[Id];
'@ }
        [pscustomobject]@{ Name='amod_ftir_scans_base64'; Columns=@('Id','DataPointFormat','NumberOfDataPoints','FrequencyOfFirstPoint','FrequencyOfLastPoint','YScalingFactor','YMaximum','YMinimum','DateOfMeasurement','TimeOfMeasurement','XUnits','Type','InstrumentType','InstrumentSerialNumber','DataBytes','DataSHA256','DataBase64'); Sql=@'
WITH ids AS (
 SELECT a.[SampleScanId] AS [Id] FROM [ftir].[SampleAnalysis] a
 WHERE EXISTS (SELECT 1 FROM [csu_amod].[Filters] f WHERE f.[MediaId]=a.[MediaId])
 UNION
 SELECT a.[BackgroundScanId] AS [Id] FROM [ftir].[SampleAnalysis] a
 WHERE EXISTS (SELECT 1 FROM [csu_amod].[Filters] f WHERE f.[MediaId]=a.[MediaId])
)
SELECT s.[Id], s.[DataPointFormat], s.[NumberOfDataPoints],
       CONVERT(varchar(100),s.[FrequencyOfFirstPoint],3) AS [FrequencyOfFirstPoint],
       CONVERT(varchar(100),s.[FrequencyOfLastPoint],3) AS [FrequencyOfLastPoint],
       CONVERT(varchar(100),s.[YScalingFactor],3) AS [YScalingFactor],
       CONVERT(varchar(100),s.[YMaximum],3) AS [YMaximum], CONVERT(varchar(100),s.[YMinimum],3) AS [YMinimum],
       s.[DateOfMeasurement], s.[TimeOfMeasurement], s.[XUnits], s.[Type], s.[InstrumentType], s.[InstrumentSerialNumber], s.[Data]
FROM [ftir].[Scan] s JOIN ids ON ids.[Id]=s.[Id] ORDER BY s.[Id];
'@ }
    )
}

function Find-AqrcFgFiles {
    param([string[]]$Root, [int]$MaxFiles=10000, [int]$MaxDirectories=2000, [int]$MaxDepth=6, [int]$TimeoutSeconds=120)
    $candidates=New-Object 'System.Collections.Generic.List[object]'
    $roots=New-Object 'System.Collections.Generic.List[object]'
    foreach ($path in $Root) {
        if ([string]::IsNullOrWhiteSpace($path)) { continue }
        $record=[pscustomobject]@{Root=$path;Status='running';FilesExamined=0;DirectoriesExamined=0;Candidates=0;ReadErrors=0;DepthLimited=$false}
        $roots.Add($record)
        if (-not (Test-Path -LiteralPath $path -PathType Container)) { $record.Status='unavailable'; continue }
        $timer=[Diagnostics.Stopwatch]::StartNew()
        $queue=New-Object 'System.Collections.Generic.Queue[object]'
        $queue.Enqueue([pscustomobject]@{Path=$path;Depth=0})
        while ($queue.Count) {
            if ($record.FilesExamined -ge $MaxFiles -or $record.DirectoriesExamined -ge $MaxDirectories -or $timer.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
                $record.Status='capped'; break
            }
            $directory=$queue.Dequeue(); $record.DirectoriesExamined++
            try { $items=@(Get-ChildItem -LiteralPath $directory.Path -ErrorAction Stop) }
            catch { $record.ReadErrors++; continue }
            foreach ($item in $items) {
                if ($timer.Elapsed.TotalSeconds -ge $TimeoutSeconds -or $record.FilesExamined -ge $MaxFiles) { $record.Status='capped'; break }
                if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { continue }
                if ($item.PSIsContainer) {
                    if ($directory.Depth -lt $MaxDepth) { $queue.Enqueue([pscustomobject]@{Path=$item.FullName;Depth=$directory.Depth+1}) }
                    else { $record.DepthLimited=$true }
                    continue
                }
                $record.FilesExamined++
                if ($item.Extension -notin @('.csv','.txt','.xlsx','.xls','.zip') -or $item.Name -notmatch '(?i)ftir|functional|group|batch|ach|acoh|cooh|naco') { continue }
                $groups=New-Object 'System.Collections.Generic.HashSet[string]'
                $inspection='not_csv'
                if ($item.Extension -in @('.csv','.txt')) {
                    $reader=$null
                    try {
                        $reader=New-Object IO.StreamReader($item.FullName)
                        for ($i=0; $i -lt 12 -and -not $reader.EndOfStream; $i++) {
                            $line=$reader.ReadLine()
                            foreach ($match in [regex]::Matches($line,'(?i)(?:^|[,;\t])\s*"?(aCH|aCOH|COOH|naCO)"?\s*(?=[,;\t]|$)')) { $groups.Add($match.Groups[1].Value) | Out-Null }
                        }
                        $inspection='first_12_lines_checked'
                    } catch { $inspection='read_failed'; $record.ReadErrors++ }
                    finally { if ($reader) { $reader.Dispose() } }
                }
                $candidates.Add([pscustomobject]@{Root=$path;FullPath=$item.FullName;Name=$item.Name;Bytes=$item.Length;ModifiedUtc=$item.LastWriteTimeUtc.ToString('o');FgColumns=(@($groups | Sort-Object) -join ';');Inspection=$inspection})
                $record.Candidates++
            }
        }
        if ($record.Status -eq 'running') { $record.Status='finished_within_scope' }
        if ($record.ReadErrors -and $record.Status -eq 'finished_within_scope') { $record.Status='finished_with_read_errors' }
        $timer.Stop()
    }
    [pscustomobject]@{Candidates=$candidates;Roots=$roots}
}

function Invoke-AqrcFollowupExport {
    [CmdletBinding()]
    param(
        [string]$HelperScript,
        [string[]]$FgSearchRoot=@($env:AETHMODULAR_FG_SEARCH_ROOT),
        [string]$OutputRoot=(Join-Path $HOME 'Downloads/aqrc_followup_exports'),
        [ValidateRange(1,3600)][int]$QueryTimeoutSeconds=300,
        [ValidateRange(1,1000000)][int]$MaxSearchFiles=10000,
        [ValidateRange(1,20)][int]$MaxSearchDepth=6,
        [ValidateRange(1,600)][int]$SearchTimeoutSeconds=120
    )
    $ErrorActionPreference='Stop'
    $sourcePath=$script:AqrcFollowupSourcePath
    if (-not $HelperScript) {
        foreach ($candidate in @((Join-Path (Split-Path $sourcePath) 'inspect_sql_access.ps1'),(Join-Path (Split-Path $sourcePath) '../../../scripts/diagnostics/inspect_sql_access.ps1'))) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { $HelperScript=$candidate; break }
        }
    }
    if (-not $HelperScript) { throw 'Place inspect_sql_access.ps1 beside this script or supply -HelperScript.' }
    . $HelperScript
    if (-not $env:AETHMODULAR_AQRC_CONNECTION_STRING) { throw 'Use the configured PowerShell tab: AETHMODULAR_AQRC_CONNECTION_STRING is missing.' }
    $out=Join-Path $OutputRoot ((Get-Date -Format yyyyMMdd_HHmmss)+'_'+[guid]::NewGuid().ToString('N').Substring(0,8))
    New-Item -ItemType Directory -Path $out | Out-Null
    $state=[pscustomobject]@{Queries=(New-Object 'System.Collections.Generic.List[object]');Errors=(New-Object 'System.Collections.Generic.List[object]');Exports=(New-Object 'System.Collections.Generic.List[object]')}
    $started=[datetime]::UtcNow.ToString('o'); $connection=$null; $status='interrupted'; $analyses=$null; $scans=$null; $fgComments=@()
    $references=New-Object 'System.Collections.Generic.List[object]'
    $search=[pscustomobject]@{Candidates=@();Roots=@()}
    try {
        try { $connection=Open-SqlSurveyConnection $env:AETHMODULAR_AQRC_CONNECTION_STRING 'Networks_1_0' }
        catch { Add-SqlSurveyError $state 'Networks_1_0' 'connect' '' $_ }
        foreach ($item in @(Get-AqrcFollowupPlan)) {
            $entry=[pscustomobject]@{Dataset=$item.Name;Status='pending';CsvRows=$null;File=''}; $state.Exports.Add($entry)
            if (-not $connection) { $entry.Status='connection_failed'; continue }
            Write-Host ('Reading '+$item.Name)
            try {
                $table=Invoke-SqlSurveyQuery -Connection $connection -Sql $item.Sql -State $state -Database 'Networks_1_0' -Stage 'amod_followup' -Object $item.Name -Timeout $QueryTimeoutSeconds
                if ($null -eq $table) { $entry.Status='query_failed'; continue }
                $rows=@(Convert-SqlSurveyRows $table 'Networks_1_0')
                if ($item.Name -eq 'amod_ftir_scans_base64') {
                    foreach ($row in $rows) {
                        $data=$row.Data; $length=0; $hash=''; $encoded=''
                        if ($null -ne $data) {
                            if ($data -isnot [byte[]]) { throw 'Scan Data was not returned as bytes.' }
                            $length=$data.Length; $encoded=[Convert]::ToBase64String($data)
                            $hasher=[Security.Cryptography.SHA256]::Create()
                            try { $hash=[BitConverter]::ToString($hasher.ComputeHash($data)).Replace('-','') } finally { $hasher.Dispose() }
                        }
                        $row | Add-Member NoteProperty DataBytes $length
                        $row | Add-Member NoteProperty DataSHA256 $hash
                        $row | Add-Member NoteProperty DataBase64 $encoded
                    }
                    $scans=$rows
                }
                if ($item.Name -eq 'amod_ftir_analyses') { $analyses=$rows }
                if ($item.Name -eq 'fg_calibration_sources') { $fgComments=$rows }
                $entry.File=$item.Name+'.csv'
                Export-SqlSurveyCsv $rows (@('Database')+$item.Columns) (Join-Path $out $entry.File)
                $entry.Status='ok'; $entry.CsvRows=$rows.Count
            } catch { $entry.Status='export_failed'; Add-SqlSurveyError $state 'Networks_1_0' 'amod_followup' $item.Name $_ }
        }
        if ($null -ne $analyses -and $null -ne $scans) {
            $index=@{}
            foreach ($scan in $scans) {
                if ($index.ContainsKey([string]$scan.Id)) { throw 'Duplicate scan ID in the export.' }
                $index[[string]$scan.Id]=$scan
            }
            foreach ($analysis in $analyses) {
                foreach ($role in @('Sample','Background')) {
                    $id=$analysis.($role+'ScanId'); $found=$false; $hasData=$false
                    if ($null -ne $id -and $index.ContainsKey([string]$id)) { $found=$true; $hasData=$index[[string]$id].DataBytes -gt 0 }
                    $references.Add([pscustomobject]@{AnalysisId=$analysis.Id;MediaId=$analysis.MediaId;Role=$role;ScanId=$id;ScanFound=$found;DataPresent=$hasData})
                }
            }
        }
        $actualRoots=@($FgSearchRoot | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if (-not $actualRoots.Count) {
            # Derive only the specific lab project mentioned in the prior notes;
            # never follow arbitrary comments into an entire drive or a new host.
            $derived=New-Object 'System.Collections.Generic.HashSet[string]'
            foreach ($comment in $fgComments) {
                foreach ($match in [regex]::Matches([string]$comment.Comments,'(?i)[A-Z]:[\\/][^\r\n]*?Chelsey_SPARTAN_MAIA(?=[\\/]|$)')) {
                    $derived.Add($match.Value.Replace('/','\')) | Out-Null
                }
            }
            $actualRoots=@($derived | Sort-Object)
        }
        Write-Host 'Searching only specified folders or the lab project paths in calibration comments (bounded, read-only).'
        $search=Find-AqrcFgFiles -Root $actualRoots -MaxFiles $MaxSearchFiles -MaxDepth $MaxSearchDepth -TimeoutSeconds $SearchTimeoutSeconds
        $status='complete'
        if ($state.Errors.Count -or @($state.Exports | Where-Object Status -ne 'ok').Count -or @($references | Where-Object { -not $_.ScanFound -or -not $_.DataPresent }).Count) { $status='partial_with_errors_or_missing_scans' }
        elseif (-not $search.Roots.Count -or @($search.Roots | Where-Object { $_.Status -ne 'finished_within_scope' -or $_.DepthLimited }).Count) { $status='amod_complete_fg_search_unresolved_or_limited' }
    } finally {
        if ($connection) { $connection.Dispose() }
        Export-SqlSurveyCsv $state.Exports @('Dataset','Status','CsvRows','File') (Join-Path $out 'exports.csv')
        Export-SqlSurveyCsv $state.Errors @('Database','Stage','Object','SqlErrorNumber','ExceptionType','Utc') (Join-Path $out 'errors.csv')
        Export-SqlSurveyCsv $references @('AnalysisId','MediaId','Role','ScanId','ScanFound','DataPresent') (Join-Path $out 'amod_scan_reference_check.csv')
        Export-SqlSurveyCsv $search.Candidates @('Root','FullPath','Name','Bytes','ModifiedUtc','FgColumns','Inspection') (Join-Path $out 'fg_file_candidates.csv')
        Export-SqlSurveyCsv $search.Roots @('Root','Status','FilesExamined','DirectoriesExamined','Candidates','ReadErrors','DepthLimited') (Join-Path $out 'fg_search_roots.csv')
        [IO.File]::WriteAllLines((Join-Path $out 'queries.jsonl'),[string[]]@($state.Queries | ForEach-Object { $_ | ConvertTo-Json -Depth 5 -Compress }),(New-Object Text.UTF8Encoding($false)))
        [ordered]@{
            Status=$status;StartedUtc=$started;FinishedUtc=[datetime]::UtcNow.ToString('o');Errors=$state.Errors.Count
            SourceSHA256=(Get-FileHash -LiteralPath $sourcePath).Hash;HelperSHA256=(Get-FileHash -LiteralPath $HelperScript).Hash
            Analyses=if ($null -eq $analyses) { $null } else { $analyses.Count }
            Scans=if ($null -eq $scans) { $null } else { $scans.Count }
            MissingOrEmptyScanReferences=@($references | Where-Object { -not $_.ScanFound -or -not $_.DataPresent }).Count
            SearchRootCount=$search.Roots.Count;SearchCandidates=$search.Candidates.Count
            SearchLimits=[ordered]@{FilesPerRoot=$MaxSearchFiles;DirectoriesPerRoot=2000;Depth=$MaxSearchDepth;SecondsPerRoot=$SearchTimeoutSeconds}
            Note='SELECT-only. All matching AMOD analyses retained; scans are the union of sample/background IDs, in native binary format encoded as base64. Separate queries are not an atomic snapshot. File candidates are leads, not confirmed official deliveries. No FG values are computed, no CSU TOR results are implied, and no files are uploaded.'
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $out 'receipt.json') -Encoding UTF8
        $manifest=@(Get-ChildItem -LiteralPath $out -File | Sort-Object Name | ForEach-Object {
            $count=$null; if ($_.Extension -eq '.csv') { $count=@(Import-Csv -LiteralPath $_.FullName).Count }
            [pscustomobject]@{File=$_.Name;Bytes=$_.Length;CsvRows=$count;SHA256=(Get-FileHash -LiteralPath $_.FullName).Hash}
        })
        Export-SqlSurveyCsv $manifest @('File','Bytes','CsvRows','SHA256') (Join-Path $out 'manifest.csv')
        Write-Host ('Follow-up '+$status); Write-Host "Saved verified exports: $out"
    }
    return $out
}
