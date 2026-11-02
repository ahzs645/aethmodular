# AQRC database access from the CLI

Runbook for setup, reconnection, bounded database reads and recovery without
administrator rights on the AQRC Windows machine. Prefer this route for supported
reads; routine operation needs no computer use. The client is currently a
standalone Python command, not an `aeth aqrc` command.

## Verified setup and evidence

On **2026-11-01 (America/Vancouver; Windows logs used November 2 UTC)**:

1. An existing VS Code Remote Tunnel reached the Windows desktop, but its
   documented CLI did not offer a general remote-shell command.
2. A fresh VS Code terminal had no SQL configuration. An older Windows
   PowerShell tab had `AETHMODULAR_AQRC_CONNECTION_STRING` and the trusted
   `Open-AqrcConn` function. That configured tab was used for the helper.
3. The private setup ZIP was dragged through the connected VS Code Explorer
   to the Windows Desktop, its SHA256 checked, then extracted to Downloads.
4. Script loading was initially blocked. All policy scopes were `Undefined`;
   the effective Windows default was restrictive. The user explicitly approved
   temporary `RemoteSigned` at **Process** scope, wrapped in `try/finally` to
   restore its prior value. No administrator elevation was used.
5. The launcher downloaded and verified Microsoft's portable tunnel client.
   Both ends authenticated with the same GitHub account. SQL configuration
   stayed on Windows; a separate random bridge token authenticated requests.
6. The Windows helper successfully probed `Networks_1_0` and listened on
   loopback port 8765. The Mac connected through a separate private dev tunnel.
7. From the Mac, `probe` returned `Connected=1`; `tables --limit 10` returned
   ten table names with `hasMore=true`; `columns` returned five requested
   metadata rows for `spartan.Filters`; a one-row, two-column filter preview
   also succeeded. These were live reads through the CLI, without computer use.
8. A later runtime update ZIP was copied from the Mac to the AQRC Desktop by
   driving VS Code's remote filesystem API from its CLI. Reading the file back
   through that API verified the full byte count and SHA256. The database
   helper itself was not given filesystem or command-execution capabilities.

`Improve_2.1` is in the implementation's default allowlist but was **not
live-tested** in this setup session. The restoration wrapper was installed;
policy restoration was not observed while the Windows helper remained running.
An old successful probe does not establish current availability. The later
`-LifetimeMinutes 0` option was subsequently activated in the configured Windows
tab; the restarted reader was verified from the Mac CLI as described below.

## Architecture and maintained files

```mermaid
flowchart LR
    A[Agent: Python CLI] --> B[Mac loopback port 8765]
    B --> C[Authenticated Microsoft dev tunnel]
    C --> D[Windows loopback helper]
    D --> E[SQL Server through existing Windows connection]
```

The tunnel authenticates its owner, and every bridge request additionally
carries a random 256-bit token. Reader v2 accepts metadata operations and single
SELECT queries validated with Microsoft ScriptDom. It has no shell, filesystem,
EXEC, DDL or DML endpoint.
This constrains submitted SQL; it does not make the SQL account itself read-only
or independently audit the definitions of database-owned views/synonyms.
Authentication and SQL TLS settings are inherited from the existing Windows
connection without overrides.
Windows PowerShell's built-in `System.Data.SqlClient` supplies the SQL driver;
the Mac does not need an ODBC/SQL Server driver. The bridge uses newline-delimited
UTF-8 JSON over TCP, not HTTP. Use `devtunnel connect` for local TCP forwarding;
the browser/inspect URLs printed by the tunnel host are not a database web UI.

| File | Purpose |
| --- | --- |
| `scripts/diagnostics/aqrc_cli.py` | Standard-library Python client and private bundle builder |
| `scripts/diagnostics/aqrc_cli_bridge.ps1` | Windows loopback server and operation dispatch |
| `scripts/diagnostics/aqrc_read_queries.ps1` | Pinned ScriptDom loader, SELECT policy, parameters and bounded streaming |
| `scripts/diagnostics/start_aqrc_cli.ps1` | Vendor download/signature check, GitHub login, private tunnel and helper lifecycle |
| `tests/powershell/test_aqrc_cli_bridge.ps1` | Offline validation, quoting, caps, redaction and connection cleanup |
| `tests/powershell/test_aqrc_read_queries.ps1` | Real parser policy and bounded long-text/binary serialization fixtures |
| `tests/test_aqrc_cli.py` | Packaging, framing and real local TCP fixture with mocked SQL |

The personal skill is `$aqrc-cli-access`, installed under the user's Codex
skills directory. It includes portable runtime scripts and a runbook snapshot;
prefer current repository sources when available. Its source map records hashes.
A **secret-free skill ZIP** can be shared with other agents. The **private Windows
setup ZIP** is a different artifact and contains a bridge access key.

## Resume an existing setup first

From the repo use its environment (`uv sync --python 3.13` if needed, then
`uv run aeth doctor`). The client itself needs only Python's standard library.
Try the existing config and forward before asking the user to do setup:

```sh
uv run python scripts/diagnostics/aqrc_cli.py probe
uv run python scripts/diagnostics/aqrc_cli.py capabilities
```

Reader v2 reports `protocolVersion=2` and `queryAvailable=true` when the pinned
parser can load. An old helper can return `unknown_operation` for capabilities;
that is a reader-version limit, not a database permission denial.

Default config: `~/.config/aethmodular/aqrc-cli.json`. Check its existence
without printing the token. Do not recreate it for an ordinary reconnect.
The verified setup installed the portable Mac executable at:

```sh
aqrc_tunnel_bin="$HOME/.local/share/aethmodular/aqrc-cli/runtime/devtunnel"
test -x "$aqrc_tunnel_bin"
"$aqrc_tunnel_bin" list --json
"$aqrc_tunnel_bin" port list TUNNEL_ID --json
"$aqrc_tunnel_bin" connect TUNNEL_ID
```

Use a current ID discovered from the account or supplied by the user, not an ID
from old chat history. Identify a currently hosted tunnel with port **8765**.
The original VS Code tunnel has another purpose and labels such as
`vscode-server-launcher`; its presence does not prove the database helper is up.
Resolve ambiguity if multiple matching tunnels exist. Keep `connect` running
and use a second CLI process for queries. Reuse a working existing forward.

The executable may instead already be on PATH as `devtunnel`. For a new Mac,
use Microsoft's [portable downloads](https://learn.microsoft.com/en-us/azure/developer/dev-tunnels/get-started)
for its observed CPU architecture (`osx-arm64-zip` or `osx-x64-zip`), extracted
into a user-owned folder. A system installation is unnecessary.
If login is missing/expired, authenticate with the same GitHub account as Windows:

```sh
"$aqrc_tunnel_bin" user login -g -d
```

The user completes device sign-in. Device codes expire; generate a new one when
needed. GitHub tunnel login, Windows SQL login and the bridge token are separate.
Never copy tokens or device codes into durable documentation.

## First setup or deliberate token rotation

`prepare` creates a new matching token pair. It refuses existing config/archive
paths. Do not delete a working config merely to get past that check:

```sh
uv run python scripts/diagnostics/aqrc_cli.py prepare \
  --output "$HOME/Downloads/aqrc-cli-setup.zip"
# For a separate setup, choose fresh private paths; --config precedes the command:
# uv run python scripts/diagnostics/aqrc_cli.py --config PRIVATE_NEW_CONFIG prepare \
#   --output PRIVATE_NEW_ARCHIVE
```

The config and ZIP are mode 0600 on the Mac. The ZIP contains four reviewed runtime scripts, the token file, a short README
and a SHA256 manifest. It does not
copy `.env`, SQL configuration, runtime binaries or datasets. The printed archive
hash is the independent transfer check. The config/ZIP contain the bridge token:
keep them private and out of Git, public links and chat. Record machine paths
through the repository's environment/path display conventions.

### Transfer and Windows readiness

Copy the ZIP directly into the AQRC user's account using the approved transfer
route. The verified route was dragging from Mac Downloads into **Desktop** in
VS Code's connected Explorer. Another route may land in Windows Downloads;
use the path where the file actually arrived.

An additional **CLI-driven copy** was subsequently verified through the existing
VS Code tunnel. VS Code's built-in CLI has no simple `cp` command; a temporary
local UI-kind development extension can use `vscode.workspace.fs.writeFile`
against the observed remote Desktop URI, then `readFile` and SHA256 to verify
the result. Launch it with `--extensionDevelopmentPath`, `--remote` and the
observed `--folder-uri`; no permanent extension installation or new listener
is needed. Use only the explicitly requested source/destination and refuse an
unexpected existing file. Close only that temporary development window afterward.
See Microsoft's [filesystem API](https://code.visualstudio.com/api/references/vscode-api#FileSystem)
and [extension host documentation](https://code.visualstudio.com/api/advanced-topics/extension-host).

The UI extension can activate before `workspaceFolders` is populated. Resolve
the intended URI from the already observed workspace metadata, or await that
workspace, rather than guessing a host/path or assuming activation means the
remote folder is ready. An ambiguous destination or failed readback is not a
completed copy. This separate VS Code transport does not add a file-upload
endpoint to `aqrc_cli.py`; the SQL helper remains restricted to guarded reads.

Use the existing database-configured PowerShell tab, not the terminal displaying
`Visual Studio Code Tunnel` host logs. Check without revealing secrets:

```powershell
Get-ExecutionPolicy -List
Get-ExecutionPolicy
[pscustomobject]@{
    ConnectionConfigured = [bool]$env:AETHMODULAR_AQRC_CONNECTION_STRING
    ConnectionHelperLoaded = [bool](Get-Command Open-AqrcConn -CommandType Function -ErrorAction SilentlyContinue)
    TunnelClientAvailable = [bool](Get-Command devtunnel -ErrorAction SilentlyContinue)
}
```

The launcher requires the environment variable. A loaded, trusted `Open-AqrcConn`
can supply it in the configured tab: see the
[Davis connection procedure](spartan-network-data-pulls.md#connect-on-davis).
Do not dot-source unknown legacy files, invent an endpoint or request a connection
string in chat. New tabs do not inherit variables/functions assigned in an older
PowerShell process.

Verify the **current** archive hash before extraction, then launch normally:

```powershell
$zip = Join-Path $HOME 'Desktop\aqrc-cli-setup.zip'
$expectedHash = 'PASTE_THE_HASH_PRINTED_FOR_THIS_ARCHIVE'
if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'Setup archive hash mismatch'
}
$setup = Join-Path $HOME 'Downloads\aqrc-cli-setup'
Expand-Archive -LiteralPath $zip -DestinationPath $setup
& (Join-Path $setup 'start_aqrc_cli.ps1')
```

For an existing extraction, skip `Expand-Archive` and rerun the launcher. Do not
overwrite an active setup with a newly generated, different token.

### Execution policy: only when needed and authorized

The launcher does not change policy. If blocked, inspect both scoped policies
and the effective policy. Undefined scopes still leave an OS default; they do
not mean scripts are allowed. MachinePolicy/UserPolicy can impose restrictions
that Process cannot override. Do not evade enforced policy.

The user explicitly authorized the temporary change in the November 1 session.
That is historical evidence, not blanket authorization for unrelated future
sessions/users. Reuse applicable current-conversation approval; otherwise explain
and obtain authorization before executing this policy-changing wrapper:

```powershell
& {
    $previousPolicy = Get-ExecutionPolicy -Scope Process
    $ErrorActionPreference = 'Stop'
    try {
        Set-ExecutionPolicy RemoteSigned -Scope Process -Force
        if ((Get-ExecutionPolicy) -in @('Restricted','AllSigned')) {
            throw 'An enforced policy still blocks these scripts; use the approved institutional process.'
        }
        $launcher = Join-Path $HOME 'Downloads\aqrc-cli-setup\start_aqrc_cli.ps1'
        & $launcher
    }
    finally {
        Set-ExecutionPolicy $previousPolicy -Scope Process -Force
        Get-ExecutionPolicy -Scope Process
    }
}
```

No administrator rights are required. The wrapper restores Process policy when
the helper exits or Ctrl+C unwinds the block; verify the reported value.
Terminating the entire PowerShell process also ends its Process policy. Do not
change CurrentUser/LocalMachine, use `Bypass`, or reinterpret/encode script text
to defeat a restriction. `Unblock-File` is a separate persistent marker removal
that needs authorization for exact reviewed files if necessary. It was not part
of this setup.

### Expected startup and lifecycle

The launcher downloads Microsoft's x64 `devtunnel.exe` into the setup directory
if absent, verifies a valid Microsoft Authenticode signature, and prompts for
GitHub login. By default it starts `host -p 8765 --expiration 4h`, then a timed
helper. With `-LifetimeMinutes 0` it omits the custom tunnel expiration and
starts a helper without a deadline. It does not enable anonymous access.
Other Windows architectures need the matching vendor executable.

```text
AQRC database probe succeeded: Networks_1_0
AQRC CLI ready on loopback port 8765. Stops after 120 minutes or Ctrl+C.
Ready to accept connections for tunnel: <CURRENT_TUNNEL_ID>
```

Ordering can vary because the tunnel is a child process. Tunnel-ready alone
is not a database probe, and a Windows probe is not an end-to-end Mac test.
Connect from the Mac and run `probe` there.

Keep the Windows helper and Mac tunnel client running. The guided launcher stops
its own tunnel child when the helper exits. The default helper checks its
**two-hour** deadline between requests, so an in-flight request can finish later.
Both launcher and bridge accept `-LifetimeMinutes 0` for no scheduled deadline;
positive values 1–240 set a timed session. The default remains 120.
Direct bridge options also include `-Port` and `-AllowedDatabase`. Port changes
need matching tunnel/client config. An allowlist change does not grant SQL access.

Microsoft's `--expiration` is an **inactivity window**, not a fixed session
deadline: activity pushes it out. The default is 30 days without activity;
custom values range from one hour to 30 days. Temporary tunnels are removed
when hosting ends. This corrects the earlier interpretation of `4h` as a
possible fixed disconnect. See the [Microsoft tunnel FAQ](https://learn.microsoft.com/en-us/azure/developer/dev-tunnels/faq).

The original VS Code tunnel can coexist. Preserve it if the helper was started
inside VS Code's remote integrated terminal, or its parent process is uncertain.
A helper in an independent native Windows PowerShell window does not need that
editor tunnel. Avoid closing the entire terminal application while needed
processes are running.

### Run until Ctrl+C instead of the two-hour cutoff

Use the new option only when the user requests a session without the helper's
deadline. No Windows service, startup task, public access or persistent policy
change is needed:

```powershell
& (Join-Path $HOME 'Downloads\aqrc-cli-setup\start_aqrc_cli.ps1') -LifetimeMinutes 0
```

When policy requires the already-authorized Process wrapper, pass
`-LifetimeMinutes 0` on the `& $launcher` line in that wrapper. Its `finally`
still restores the previous policy after Ctrl+C. The ready message becomes
`AQRC CLI ready on loopback port 8765. Runs until Ctrl+C.`

For an existing old deployment, first transfer `aqrc-cli-runtime-update.zip`
directly to the Windows Desktop. The original timer-only update contains two PowerShell scripts and a manifest.
For reader v2 use `aqrc-cli-reader-v2-update.zip`, which includes three PowerShell
scripts, pinned Microsoft parser DLLs for Windows/PowerShell Core, their license
and a hash manifest. Neither runtime update contains a bridge token. Stop the old
helper with Ctrl+C, verify the supplied current archive hash, then extract:

```powershell
$updateZip = Join-Path $HOME 'Desktop\aqrc-cli-reader-v2-update.zip'
$expectedHash = 'PASTE_THE_HASH_PRINTED_FOR_THIS_UPDATE_ARCHIVE'
if ((Get-FileHash -LiteralPath $updateZip -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'Runtime update hash mismatch'
}
Expand-Archive -LiteralPath $updateZip -DestinationPath (Join-Path $HOME 'Downloads\aqrc-cli-setup') -Force
```

Restart in the same database-configured tab using the option above (and the
authorized policy wrapper when needed). Reuse the existing matching token;
do not run `prepare`. Restarting the temporary host produces a current tunnel
ID. Reconnect the Mac client to that ID, stopping only its known old forward
if that process still holds local port 8765, then verify with a live CLI probe.
Local file edits never change the already-running Windows helper.

Here, "no deadline" means the helper stays up while its PowerShell process
and hosting/network/login remain usable. Closing PowerShell, logging off,
sleep/reboot, a lost network connection or an authentication failure can still
interrupt access. This option does not install unattended restart or promise
availability forever. Per-query limits, private access, token authentication
and SQL TLS checks remain the same.

## Reader v2 commands and capabilities

Reader v2 was implemented after the first live session. The schema for the notes
query was checked live with the old reader: `import.ImportNotes` has `MediaId`,
`Identifier`, `Message`, `SourceData` and `ImportNoteSetId`; `ImportNoteSets`
provides batch context. After verified deployment and the user's Windows restart,
the Mac CLI confirmed protocol 2, `queryAvailable=true`, and these live checks:

- A SELECT of the database name and `1` succeeded in `Networks_1_0`.
- `views` and `synonyms` each returned zero visible rows without SQL errors.
  That is visibility-limited metadata, not proof that no hidden objects exist.
- A joined filter import-note read returned one requested row, `hasMore=true`,
  no truncated cells, and a 594-character batch note. This verifies the command,
  not completeness of that filter's note history.
- A controlled synthetic SELECT returned 2048 text characters, an exact large
  decimal and base64 binary. A text cap of 16 returned 16 characters and an
  explicit truncation. These are transport checks, not research observations.
- A harmless two-SELECT batch was rejected with `single_select_required`.

The user-started helper reports `Runs until Ctrl+C.` Future source edits still
need deployment/restart; dated checks do not establish future availability.

The default catalog is `Networks_1_0`. `Improve_2.1` remains allowlisted but
requires its own live probe before reporting access. No database grants change.

```sh
uv run python scripts/diagnostics/aqrc_cli.py capabilities
uv run python scripts/diagnostics/aqrc_cli.py views --limit 1000
uv run python scripts/diagnostics/aqrc_cli.py synonyms --limit 1000
uv run python scripts/diagnostics/aqrc_cli.py objects --limit 1000
uv run python scripts/diagnostics/aqrc_cli.py columns --schema spartan --table Filters --limit 1000
uv run python scripts/diagnostics/aqrc_cli.py notes --filter-id FILTER_ID --max-chars 1048576
```

| Command | Capability | Main bound |
| --- | --- | --- |
| `prepare` | Matching private config/Windows bundle; no SQL | Refuses existing config/archive |
| `capabilities` | Current remote protocol, parser availability, allowed catalogs and limits | Does not establish SELECT permission on an object |
| `probe` | Opens allowed catalog and SELECTs `DB_NAME()`/`1` | Connectivity check, not coverage audit |
| `databases` | Accessible online user catalog names | Default 25, max 1000; does not extend the allowlist |
| `tables`, `views`, `synonyms`, `objects` | Physical tables, views, synonym targets, or combined inventory | Default 25, max 1000; metadata visibility is permission-limited |
| `columns` | Columns on ordinary tables and views | Default 25, max 1000; for synonym result columns use a SELECT TOP (0) query |
| `preview` | Legacy explicit-column preview on tables/views | Max 200 rows/20 columns, 512 characters per cell; retained for compatibility |
| `query` | One parsed SELECT, including joins, CTEs, aggregates, WHERE, ordering and SQL pagination | Default 100 rows, max 10,000; 30-second default/120-second max query timeout |
| `notes` | Parameterized SELECT of a filter's import notes plus import-set context | Same query bounds; exact stored FilterId/Identifier by default |

### Flexible read-only SQL

Prefer `--sql-file` and `--params-file` so shell interpolation cannot change a
query or expose literal values. For example, save this as a local `.sql` file:

```sql
SELECT f.ExternalFilterId, n.Id AS NoteId, n.Category, n.Message, n.SourceData
FROM spartan.Filters AS f
JOIN [import].ImportNotes AS n ON n.MediaId = f.MediaId
WHERE f.ExternalFilterId = @filter_id
ORDER BY n.Id
```

Save parameter values separately as JSON, for example
`{"@filter_id":"REPLACE_WITH_STORED_FILTER_ID"}`, then run:

```sh
uv run python scripts/diagnostics/aqrc_cli.py query --sql-file read.sql \
  --params-file params.json --limit 1000 --max-chars 1048576 --timeout 60
```

Scalar parameter values may be strings, signed 64-bit integers, finite floating
point numbers, booleans or null. Pass large exact decimal values as strings and
CAST them in SQL. There is no template substitution: values bind through typed
SQL parameters. CLI `--sql` is also available when normal shell quoting is safe.

The parser requires one SELECT and checks the complete syntax tree, not a prefix
or keyword regex. Writes, SELECT INTO, sequence advancement, variable assignment,
EXEC, multiple statements/batches, hints, four-part server references, disallowed
database prefixes and ad hoc sources such as OPENROWSET/OPENQUERY are blocked.
Schema-qualified UDF/method calls and table-function sources outside the small
approved set are blocked as well. Ordinary built-ins, joins, CTEs, window
functions, UNION, OPENJSON, WHERE and ORDER BY/OFFSET/FETCH are accepted.
These rules apply before opening/executing the submitted query. A rejected
statement is a helper-policy result, not a SQL permission denial.

Existing views, computed columns and registered synonyms can be read with query
according to the login's database permissions. Their definitions and downstream
ownership chains are database-owned; the helper does not recursively audit those
definitions or establish a database-enforced read-only role. Direct object
references are checked against the two configured catalogs. Native SQL permission
errors remain authoritative; no new login, role or grant is created here.

For object definitions use SELECTs against `sys.sql_modules` or
`OBJECT_DEFINITION`. Null/hidden metadata may mean encryption, missing visibility
or an absent object; do not equate it with data absence. For a synonym's result
schema a query such as `SELECT TOP (0) * FROM dbo.SYNONYM_NAME` returns column
metadata without returning data rows. Discover the actual names first.

### Filter import notes

`notes` joins `import.ImportNotes.MediaId` to `spartan.Filters.MediaId` and adds
`ImportNoteSets` context. It matches either stored `ExternalFilterId` or the
note's `Identifier` exactly. It returns Message/SourceData and clearly named
batch-level `ImportSetNotes`/`ImportSetComments`, not the entire batch Records
payload. This template targets the verified Networks schema.

Gallery FilterIds can omit the physical `-N` suffix. Use `--include-suffixes`
when intentionally including those numeric physical-filter suffixes; the option
escapes LIKE metacharacters and retains each source ID rather than silently
normalizing distinct records:

```sh
uv run python scripts/diagnostics/aqrc_cli.py notes --filter-id BASE_FILTER_ID \
  --include-suffixes --max-chars 1048576 --limit 1000
# Continue after the prior page's last NoteId if hasMore is true:
uv run python scripts/diagnostics/aqrc_cli.py notes --filter-id BASE_FILTER_ID \
  --include-suffixes --after-id LAST_NOTE_ID --max-chars 1048576 --limit 1000
```

Only say the notes are complete when the ID matching is appropriate,
`hasMore=false`, `byteBudgetReached=false` and `truncatedCellCount=0`. A zero-row
exact match does not rule out notes on a suffixed ID or an upstream identifier.

### Result semantics and budgets

- Query values are strings or JSON null. Decimal(38) precision, float round-trip
  digits and ISO timestamps are preserved. Empty text differs from null. Binary
  values are base64, marked in the column-aligned `encodings` array.
- Duplicate column labels receive `__2`, `__3`, etc., without dropping either
  value; `originalColumns` records the original labels. Alias important join
  columns explicitly. `columnTypes` describes their SQL types.
- The long-field limit defaults to 65,536 and can be raised to 1,048,576 UTF-16
  units for text or bytes for binary. Streaming avoids reading an unbounded LOB
  into memory. `truncatedCells` reports row/column, original length, returned text
  length and encoding. `truncatedCellCount` counts all returned truncated cells;
  details are capped at 1000 and `truncationDetailsCapped` states that condition.
  Do not claim full notes/spectra from a truncated cell.
- Query rows default to 100, max 10,000, with at most 256 columns. A conservative
  response budget can stop before the row cap; `byteBudgetReached` and `hasMore`
  identify that condition. Partial rows are not returned as complete records.
- The server prepends its own SET ROWCOUNT bound and reads at most one overflow
  row. This caps returned rows, not the work required for joins/aggregates. A
  COUNT result can still scan a large table; respect the time budget.
- Query pagination belongs in the SQL (ordered OFFSET/FETCH or a stable keyset
  predicate). `notes --after-id` provides keyset paging for note IDs. Metadata
  commands still have no pagination, so repeating a capped listing is not a way
  to retrieve later entries. `hasMore` is not a total count.
- Request cap: 128 KiB, SQL text cap: 32,768 characters, max 64 scalar parameters.
  Response cap: 32 MiB; one server request at a time. Server-side socket read/write
  timeout stays 10 seconds; SQL connect timeout 60 seconds; per-query timeout
  30 seconds by default, maximum 120; client timeout 240 seconds.
- Legacy metadata/preview result semantics and smaller caps remain unchanged.
  Preview is unordered, not a random sample or a population summary. Query's
  scientific grain is determined by its joins and filters, not by the row limit.
- Exit codes remain 0 success, 1 structured server failure, 2 local config/
  protocol/transport failure. Parser errors report only number/line/column;
  native SQL failures report an error number. Raw exception text, SQL credentials
  and connection strings are not returned.

### Parser dependency and update

`aqrc_read_queries.ps1` uses Microsoft's MIT-licensed ScriptDom 180.107.0 from
NuGet, with pinned package and DLL SHA256 checks. The launcher loads the matching
net472 DLL for Windows PowerShell (.NET Framework 4.7.2+) or netstandard2.0 for
PowerShell Core. Fresh setups download the verified package if needed; the v2
runtime update carries both DLLs and the license so Windows needs no NuGet
connection during this upgrade. It is a user-folder library, not an admin install.
Do not bypass a DLL load restriction, remove download markers or skip a hash
check. Missing/invalid parser means query is unavailable; it never falls back to
prefix matching. Metadata reads can still be inspected through capabilities.

For tests/development only, `AETHMODULAR_AQRC_SCRIPTDOM_DIR` can point to an
existing verified cache with the expected framework subdirectories. The normal
Windows setup uses `lib/ScriptDom` under its private setup folder.

## Troubleshooting

| Symptom | Next action |
| --- | --- |
| `running scripts is disabled` | Launcher never loaded; no SQL attempted. Inspect effective/scoped policies and use the authorized Process wrapper only when applicable. |
| `not digitally signed` with RemoteSigned | Review download marker, source and hash. Do not automatically unblock files or weaken policy. |
| `connection_not_configured_in_this_tab` | Reuse the configured tab or trusted institutional local setup, without printing configuration. |
| Only the VS Code tunnel appears | The separate database helper tunnel is not established; inspect Windows launcher output. |
| Mac refusal/generic CLI error | Check config existence, forward, Windows lifetime, current tunnel ID and login. Reconnect before rebuilding. |
| `unauthorized` | Mac/Windows tokens differ. Check which private bundle was extracted; do not print tokens or rotate blindly. |
| `database_not_allowed` | Catalog is outside helper configuration; appearing in `databases` does not authorize an allowlist change. |
| `invalid_name`, `invalid_limit`, `unknown_operation` | Check remote capabilities/version and documented arguments. An old helper must be updated for query/views/synonyms; this is not an SQL denial. |
| `column_not_visible`, `unsupported_column` | Legacy preview has narrower rules; use the guarded query path or reviewed exporters. |
| `sql_parser_unavailable`, `sql_parser_hash_mismatch` | Install/transfer the pinned verified library; never bypass validation. |
| `sql_parse_error` | Inspect safe parser number/line/column and correct the query syntax. |
| `single_select_required`, `sql_write_not_allowed`, `sql_*_not_allowed` | The helper rejected the syntax before SQL execution. Use a permitted SELECT; do not reinterpret this as database access denial. |
| `operation_failed` with SQL number | Preserve as a failed read; diagnose the number/connectivity/permissions without exposing raw exceptions or changing TLS. |
| `response_too_large` | Reduce requested rows/columns rather than removing the guard. |
| `hasMore=true` | Result is capped; do not report full coverage or loop a preview as pagination. |
| Timed session expired | Restart existing launcher in the configured Windows tab; reuse matching tokens and reconnect. For a user-requested session without a deadline, updated scripts accept `-LifetimeMinutes 0`. |
| Local port occupied | Reuse a working forward; do not kill an unidentified listener. |

Diagnose the failing layer before retrying it. Do not restart Windows, close
unrelated terminals, change sharing, expand privileges or disable validation as
an inferred cure.

## Extending and maintaining the workflow

For ordinary investigations use the validated query path rather than adding a
named operation for every SELECT. Add new server capabilities only when needed,
with typed parameters, catalog checks, bounded streaming/timeouts and focused
policy tests. Record scientific grain/units where applicable. Redeploy reviewed
script changes to Windows: editing the Mac source does not change an already
loaded helper. Do not remove the parser/statement checks or add unvalidated SQL, EXEC, writes
or command execution.

Broader [SQL surveys](sql-data-discovery.md),
[reference exporters](aqrc-reference-exports.md) and
[network pulls](spartan-network-data-pulls.md) remain useful for established
profiles, large/binary exports and full receipts/manifests. Requested key metadata
and aggregate checks can also use guarded SELECT queries within their budgets.
Transport does
not change provenance: SPARTAN ChemSpec EC/OC are FTIR products, not independent
TOR measurements. Follow `AGENTS.md` and the Davis schema/provenance reference.

```sh
# Initialize the pinned dependency once if no verified cache is already configured:
pwsh -NoProfile -Command '. ./scripts/diagnostics/aqrc_cli_bridge.ps1; Install-AqrcScriptDom'
pwsh -NoProfile -File tests/powershell/test_aqrc_cli_bridge.ps1
pwsh -NoProfile -File tests/powershell/test_aqrc_read_queries.ps1
AQRC_TEST_PWSH="$(command -v pwsh)" uv run pytest tests/test_aqrc_cli.py
```

A verified portable PowerShell runtime was also placed beside the Mac tunnel
binary under the user runtime directory. Set `AQRC_TEST_PWSH` to its
`powershell/pwsh` when pwsh is not on PATH. The protocol fixtures exercise both
timed and no-deadline helper modes, while the PowerShell checks cover bounds,
quoting and cleanup. Fixtures replace SQL transport; only the real Mac-side
probe establishes live access.

When sources change, update the installed skill's four portable scripts,
runbook snapshot and `references/source-map.json` hashes. Keep paths relative to
the repository and omit tokens, SQL endpoints, usernames, expiring device codes
and transient tunnel IDs. Never package local config, `bridge-token.txt`, runtime
binaries or the private Windows setup ZIP inside a shareable skill. The separate
reviewed Windows v2 update intentionally includes the pinned parser DLLs/license.

### Give the skill to another agent

Extract the **secret-free** `aqrc-cli-access-skill.zip` into that agent user's
Codex skills directory, preserving the top-level `aqrc-cli-access/` folder:

```sh
aqrc_skill_home="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$aqrc_skill_home"
unzip -n "$HOME/Downloads/aqrc-cli-access-skill.zip" -d "$aqrc_skill_home"
```

`-n` preserves an existing skill; review differences before replacing one.
Ask the agent to use `$aqrc-cli-access`, or give it the installed `SKILL.md`
path if it has not discovered the new skill. The skill carries instructions
and source snapshots only: it does not carry login state, the bridge token,
current host availability or a grant of permission to change policies.

Microsoft references: [portable downloads](https://learn.microsoft.com/en-us/azure/developer/dev-tunnels/get-started),
[private hosting and CLI forwarding](https://learn.microsoft.com/en-us/azure/developer/dev-tunnels/cli-commands),
[PowerShell execution policies](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies),
[Microsoft ScriptDom](https://github.com/microsoft/SqlScriptDOM),
[SET ROWCOUNT semantics](https://learn.microsoft.com/en-us/sql/t-sql/statements/set-rowcount-transact-sql).
