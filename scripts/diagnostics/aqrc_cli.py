"""Bounded AQRC reads over an authenticated, loopback-only forwarded connection.

The bridge accepts metadata operations and parser-validated SELECT queries, with
no database-write or shell endpoint. This client uses only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import socket
import sys
import zipfile
from pathlib import Path

CONFIG_PATH = Path.home() / ".config" / "aethmodular" / "aqrc-cli.json"
MAX_RESPONSE = 33_554_432
MAX_REQUEST = 131_072

IMPORT_NOTES_SQL = """SELECT n.Id AS NoteId, n.MediaId, f.ExternalFilterId AS FilterId,
n.Identifier, n.Category, n.Severity, n.Message, n.SourceData, n.ImportNoteSetId,
s.Type AS ImportType, s.ImportedDate, s.Filename,
s.Notes AS ImportSetNotes, s.Comments AS ImportSetComments
FROM [import].ImportNotes AS n
LEFT JOIN spartan.Filters AS f ON f.MediaId = n.MediaId
LEFT JOIN [import].ImportNoteSets AS s ON s.Id = n.ImportNoteSetId
WHERE (f.ExternalFilterId = @filter_id OR n.Identifier = @filter_id
 OR (@include_suffixes = 1 AND
   ((f.ExternalFilterId LIKE @filter_pattern ESCAPE '~'
     AND SUBSTRING(f.ExternalFilterId,LEN(@filter_id)+2,128) NOT LIKE '%[^0-9]%')
    OR (n.Identifier LIKE @filter_pattern ESCAPE '~'
     AND SUBSTRING(n.Identifier,LEN(@filter_id)+2,128) NOT LIKE '%[^0-9]%'))))
 AND n.Id > @after_id
ORDER BY n.Id"""


def prepare(output: Path, config: Path = CONFIG_PATH) -> dict:
    """Create a private transfer bundle; never read .env or SQL configuration."""
    if config.exists() or output.exists():
        raise ValueError("Config or archive already exists; choose fresh paths.")
    config.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    server = Path(__file__).with_name("aqrc_cli_bridge.ps1").read_bytes()
    launcher = Path(__file__).with_name("start_aqrc_cli.ps1").read_bytes()
    query_support = Path(__file__).with_name("aqrc_read_queries.ps1").read_bytes()
    readme = (
        "Private AQRC CLI setup. This archive contains an authentication secret.\n"
        "Keep it private; do not commit or upload it to a public location.\n\n"
        "Extract on AQRC, then use the PowerShell tab whose connection variable\n"
        "is already configured. Run the reviewed start_aqrc_cli.ps1 normally.\n"
        "It downloads/verifies Microsoft's client and prompts for sign-in.\n"
        "It starts a private tunnel and helper until Ctrl+C. Do not bypass an enforced\n"
        "execution policy. See docs/aqrc-cli-access.md in the source repository.\n"
        "The helper allows bounded SELECT queries validated by Microsoft ScriptDom.\n"
        "No database writes, EXEC, ad hoc external sources or shell commands.\n"
    ).encode()
    files = {"aqrc_cli_bridge.ps1": server, "start_aqrc_cli.ps1": launcher,
             "aqrc_read_queries.ps1": query_support,
             "bridge-token.txt": token.encode(), "README.txt": readme}
    manifest = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name, value in files.items():
                bundle.writestr(name, value)
            bundle.writestr("manifest.json", json.dumps(manifest, indent=2))
        fd = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump({"host": "127.0.0.1", "port": 8765, "token": token}, stream)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {"prepared": True, "serverSha256": manifest["aqrc_cli_bridge.ps1"],
            "archiveSha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def request(operation: str, config: Path = CONFIG_PATH, **arguments) -> dict:
    settings = json.loads(config.read_text())
    if settings["host"] not in {"127.0.0.1", "::1"}:
        raise ValueError("Client connects only to a local tunnel endpoint.")
    payload = json.dumps({**arguments, "operation": operation, "token": settings["token"]})
    if len(payload.encode()) > MAX_REQUEST:
        raise ValueError("Request too large.")
    with socket.create_connection((settings["host"], settings["port"]), timeout=240) as connection:
        connection.sendall(payload.encode() + b"\n")
        with connection.makefile("rb") as stream:
            line = stream.readline(MAX_RESPONSE + 1)
    if len(line) > MAX_RESPONSE or not line.endswith(b"\n"):
        raise ValueError("Incomplete or oversized bridge response.")
    result = json.loads(line)
    if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
        raise ValueError("Invalid bridge response.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    sub = parser.add_subparsers(dest="operation", required=True)
    bundle = sub.add_parser("prepare", help="Prepare a private Windows transfer archive")
    bundle.add_argument("--output", type=Path, required=True)
    sub.add_parser("capabilities", help="Inspect the remote reader version and limits")
    for name in ("probe", "databases", "tables", "views", "synonyms", "objects", "columns", "preview", "query", "notes"):
        command = sub.add_parser(name)
        if name != "databases":
            command.add_argument("--database", default="Networks_1_0")
        if name in {"columns", "preview"}:
            command.add_argument("--schema", required=True)
            command.add_argument("--table", required=True)
        if name == "preview":
            command.add_argument("--columns", nargs="+", required=True)
        if name == "query":
            sql = command.add_mutually_exclusive_group(required=True)
            sql.add_argument("--sql-file", type=Path)
            sql.add_argument("--sql")
            command.add_argument("--params-file", type=Path)
        if name == "notes":
            command.add_argument("--filter-id", required=True, help="Exact stored FilterId or import Identifier")
            command.add_argument("--after-id", type=int, default=0, help="Continue after the previous page's last NoteId")
            command.add_argument("--include-suffixes", action="store_true", help="Also match numeric -N physical-filter suffixes for a base FilterId")
        if name in {"query", "notes"}:
            command.add_argument("--max-chars", type=int, default=65536)
            command.add_argument("--timeout", type=int, default=30)
        if name != "probe":
            command.add_argument("--limit", type=int, default=100 if name in {"query", "notes"} else 25)
    args = vars(parser.parse_args())
    config = args.pop("config")
    operation = args.pop("operation")
    try:
        if operation == "prepare":
            result = prepare(args["output"], config)
        else:
            if operation == "query":
                sql_file = args.pop("sql_file")
                if sql_file:
                    args["sql"] = sql_file.read_text(encoding="utf-8-sig")
                params_file = args.pop("params_file")
                args["parameters"] = json.loads(params_file.read_text(encoding="utf-8-sig")) if params_file else {}
            if operation == "notes":
                filter_id = args.pop("filter_id")
                pattern = filter_id
                for literal in ("~", "%", "_", "["):
                    pattern = pattern.replace(literal, "~" + literal)
                args["parameters"] = {
                    "@filter_id": filter_id, "@after_id": args.pop("after_id"),
                    "@include_suffixes": args.pop("include_suffixes"), "@filter_pattern": pattern + "-[0-9]%",
                }
                args["sql"] = IMPORT_NOTES_SQL
                operation = "query"
            result = request(operation, config, **args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("ok", result.get("prepared", False)) else 1
    except (OSError, ValueError, KeyError, TypeError):
        # Never echo settings, tokens, or the raw server response in failures.
        print("AQRC CLI failed: check the local config, private tunnel, and bridge status.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
