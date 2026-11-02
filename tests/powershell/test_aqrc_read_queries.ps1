# Offline policy/serialization checks. No SQL connection is opened.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../../scripts/diagnostics/aqrc_cli_bridge.ps1')
function Assert-Read($Condition,[string]$Message) {
    if (-not $Condition) { throw ('Assertion failed: '+$Message) }
}
$allowed=@('Networks_1_0','Improve_2.1')
Initialize-AqrcScriptDom
foreach ($sql in @(
    'SELECT 1 AS n',
    "SELECT 'DELETE x; EXEC evil; GO' AS harmless -- DROP is just a comment",
    'WITH c AS (SELECT 1 AS n) SELECT n FROM c',
    'SELECT f.MediaId,n.Message FROM spartan.Filters f LEFT JOIN [import].ImportNotes n ON n.MediaId=f.MediaId WHERE f.ExternalFilterId=@id ORDER BY n.Id OFFSET 0 ROWS FETCH NEXT 20 ROWS ONLY',
    'SELECT ROW_NUMBER() OVER(ORDER BY n.Id) AS n FROM [import].ImportNotes n',
    'SELECT * FROM dbo.SomeViewOrSynonym',
    'SELECT * FROM [Improve_2.1].analysis.Results',
    'SELECT * FROM OPENJSON(@json) WITH (id int)',
    'SELECT 1 AS n UNION ALL SELECT 2'
)) { Assert-AqrcReadSql $sql $allowed }

$rejected=@(
    'DELETE dbo.x','UPDATE dbo.x SET n=1','INSERT INTO dbo.x VALUES(1)',
    'DROP TABLE dbo.x','CREATE TABLE dbo.x(n int)','TRUNCATE TABLE dbo.x',
    'EXEC sys.sp_executesql N''SELECT 1''','WAITFOR DELAY ''00:00:01''',
    'SET ROWCOUNT 0; SELECT 1','SELECT 1; DELETE dbo.x','SELECT 1; SELECT 2',
    "SELECT 1`nGO`nSELECT 2",'WITH c AS (SELECT * FROM dbo.x) DELETE c',
    'SELECT 1 INTO dbo.x','SELECT NEXT VALUE FOR dbo.seq','SELECT @x=1',
    'SELECT * FROM OPENQUERY(server,''SELECT 1'')',
    'SELECT * FROM OPENROWSET(BULK ''secret.txt'',SINGLE_CLOB) AS x',
    'SELECT * FROM server.db.dbo.x','SELECT * FROM master.sys.objects',
    'SELECT dbo.SomeClrFunction(1)','SELECT * FROM dbo.SomeFunction()',
    'SELECT * FROM dbo.x WITH (UPDLOCK)','SELECT * FROM dbo.x OPTION(MAXRECURSION 0)',
    'SELECT FROM'
)
foreach ($sql in $rejected) {
    $failed=$false
    try { Assert-AqrcReadSql $sql $allowed } catch { $failed=$true }
    Assert-Read $failed ('unsafe SQL rejected: '+$sql)
}

$command=[Data.SqlClient.SqlCommand]::new()
try {
    Add-AqrcReadParameters $command ([pscustomobject]@{'@id'="x' OR 1=1--";'@n'=12;'@flag'=$true;'@missing'=$null})
    Assert-Read ($command.Parameters['@id'].Value -eq "x' OR 1=1--") 'parameter value is data'
    Assert-Read ($command.Parameters['@id'].SqlDbType -eq [Data.SqlDbType]::NVarChar) 'typed string binding'
    Assert-Read ($command.Parameters['@n'].SqlDbType -eq [Data.SqlDbType]::BigInt) 'typed integer binding'
    Assert-Read ($command.Parameters['@missing'].Value -is [DBNull]) 'null binding'
} finally { $command.Dispose() }

function New-ReadFixture {
    $r=[pscustomobject]@{
        Index=-1;FieldCount=6;Names=@('note','note','empty','missing','bytes','exact');
        Types=@('nvarchar','nvarchar','nvarchar','nvarchar','varbinary','decimal');
        Values=@(('abc'+[char]::ConvertFromUtf32(0x1F600)+'tail'),'second','',$null,([byte[]]@(0,1,2,3,4)),'9999999999999999999999999999.12345678')
    }
    $r | Add-Member ScriptMethod Read { $this.Index++; return $this.Index -eq 0 }
    $r | Add-Member ScriptMethod GetName { param($i) return $this.Names[$i] }
    $r | Add-Member ScriptMethod GetDataTypeName { param($i) return $this.Types[$i] }
    $r | Add-Member ScriptMethod IsDBNull { param($i) return $null -eq $this.Values[$i] }
    $r | Add-Member ScriptMethod GetValue { param($i) return $this.Values[$i] }
    $r | Add-Member ScriptMethod GetSqlDecimal { param($i) return [Data.SqlTypes.SqlDecimal]::Parse($this.Values[$i]) }
    $r | Add-Member ScriptMethod GetChars {
        param($i,$offset,$buffer,$bufferOffset,$length)
        $text=[string]$this.Values[$i]
        if ($null -eq $buffer) { return [long]$text.Length }
        $count=[Math]::Min($length,$text.Length-$offset)
        $text.CopyTo($offset,$buffer,$bufferOffset,$count)
        return [long]$count
    }
    $r | Add-Member ScriptMethod GetBytes {
        param($i,$offset,$buffer,$bufferOffset,$length)
        $bytes=$this.Values[$i]
        if ($null -eq $buffer) { return [long]$bytes.Length }
        $count=[Math]::Min($length,$bytes.Length-$offset)
        [Array]::Copy($bytes,$offset,$buffer,$bufferOffset,$count)
        return [long]$count
    }
    return $r
}
$result=Read-AqrcQueryRows (New-ReadFixture) 10 4
Assert-Read ($result.Rows[0].note -eq 'abc') 'truncation does not split a surrogate pair'
Assert-Read ($result.Rows[0].note__2 -eq 'seco') 'duplicate labels retain both columns'
Assert-Read ($result.Rows[0].empty -ceq '' -and $null -eq $result.Rows[0].missing) 'empty and null differ'
Assert-Read ($result.Rows[0].bytes -eq 'AAECAw==') 'bounded binary is base64'
Assert-Read ($result.Rows[0].exact -eq '9999999999999999999999999999.12345678') 'decimal precision preserved'
Assert-Read ($result.TruncatedCellCount -eq 3) 'all truncated text/binary cells are reported'
$full=Read-AqrcQueryRows (New-ReadFixture) 10 1000
Assert-Read ($full.TruncatedCellCount -eq 0) 'complete fields are distinct from truncated fields'
$nullBinary=New-ReadFixture
$nullBinary.Values[4]=$null
$nullResult=Read-AqrcQueryRows $nullBinary 10 1000
Assert-Read ($nullResult.Encodings[4] -eq 'base64') 'binary encoding survives a null first row'
$capped=Read-AqrcQueryRows (New-ReadFixture) 10 1000 16
Assert-Read ($capped.ByteBudgetReached -and $capped.HasMore -and $capped.Rows.Count -eq 0) 'budget cap never reports a partial row as complete'
Write-Host 'AQRC SELECT policy and result-stream checks passed.'
