"""Protocol and packaging tests; real database/tunnel access is not exercised."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aqrc_cli", ROOT / "scripts/diagnostics/aqrc_cli.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def test_private_bundle_contains_only_setup_files(tmp_path):
    config, archive = tmp_path / "config.json", tmp_path / "setup.zip"
    result = cli.prepare(archive, config)
    assert result["prepared"]
    assert config.stat().st_mode & 0o777 == 0o600
    assert archive.stat().st_mode & 0o777 == 0o600
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {
            "aqrc_cli_bridge.ps1", "start_aqrc_cli.ps1", "aqrc_read_queries.ps1",
            "bridge-token.txt", "manifest.json", "README.txt"
        }
        assert bundle.read("bridge-token.txt").decode() == json.loads(config.read_text())["token"]
        manifest = json.loads(bundle.read("manifest.json"))
        for name, digest in manifest.items():
            assert cli.hashlib.sha256(bundle.read(name)).hexdigest() == digest
    with pytest.raises(ValueError):
        cli.prepare(tmp_path / "another.zip", config)


def mock_endpoint(tmp_path, response):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    config = tmp_path / "client.json"
    config.write_text(json.dumps({"host": "127.0.0.1", "port": server.getsockname()[1], "token": "secret"}))
    received = []

    def serve():
        try:
            connection, _ = server.accept()
            with connection, connection.makefile("rb") as stream:
                received.append(json.loads(stream.readline()))
                connection.sendall(response)
        finally:
            server.close()

    worker = threading.Thread(target=serve)
    worker.start()
    return config, received, worker


def test_protocol_preserves_null_empty_and_exact_numeric_text(tmp_path):
    body = {"ok": True, "rows": [{"id": "999999999999999999999999999999", "x": None, "empty": ""}]}
    config, received, worker = mock_endpoint(tmp_path, json.dumps(body).encode() + b"\n")
    assert cli.request("preview", config, columns=["id", "x", "empty"]) == body
    worker.join(2)
    assert received[0]["operation"] == "preview"
    assert received[0]["token"] == "secret"


@pytest.mark.parametrize("response", [b'{"ok":true}', b'not-json\n', b'[]\n', b'{"ok":"true"}\n'])
def test_invalid_frames_are_rejected(tmp_path, response):
    config, _, worker = mock_endpoint(tmp_path, response)
    with pytest.raises(ValueError):
        cli.request("probe", config)
    worker.join(2)


def test_client_rejects_nonlocal_destination(tmp_path):
    config = tmp_path / "client.json"
    config.write_text(json.dumps({"host": "example.com", "port": 8765, "token": "secret"}))
    with pytest.raises(ValueError, match="local tunnel"):
        cli.request("probe", config)


def test_notes_binds_id_and_escapes_optional_like_pattern(monkeypatch, capsys):
    calls = []

    def capture(operation, config, **arguments):
        calls.append((operation, arguments))
        return {"ok": True, "rows": []}

    monkeypatch.setattr(cli, "request", capture)
    monkeypatch.setattr(sys, "argv", ["aqrc_cli.py", "notes", "--filter-id", "ET_AD%[1]'", "--include-suffixes", "--after-id", "12"])
    assert cli.main() == 0
    _, args = calls[0]
    assert "ET_AD" not in args["sql"]
    assert args["parameters"]["@filter_id"] == "ET_AD%[1]'"
    assert args["parameters"]["@filter_pattern"] == "ET~_AD~%~[1]'-[0-9]%"
    assert args["parameters"]["@after_id"] == 12
    assert args["parameters"]["@include_suffixes"] is True
    assert "--" not in capsys.readouterr().out


def test_query_reads_sql_and_parameters_without_shell_interpolation(tmp_path, monkeypatch):
    sql_file, params_file = tmp_path / "query.sql", tmp_path / "params.json"
    sql_file.write_text("SELECT '$(literal)' AS value WHERE @id = @id")
    params_file.write_text(json.dumps({"@id": "x' OR 1=1--"}))
    calls = []
    monkeypatch.setattr(cli, "request", lambda operation, config, **args: calls.append(args) or {"ok": True})
    monkeypatch.setattr(sys, "argv", ["aqrc_cli.py", "query", "--sql-file", str(sql_file), "--params-file", str(params_file)])
    assert cli.main() == 0
    assert calls[0]["sql"] == sql_file.read_text()
    assert calls[0]["parameters"] == {"@id": "x' OR 1=1--"}


@pytest.mark.parametrize("lifetime_minutes", [0, 1])
def test_real_powershell_listener_auth_and_operations(tmp_path, lifetime_minutes):
    pwsh = os.environ.get("AQRC_TEST_PWSH")
    if not pwsh:
        pytest.skip("Set AQRC_TEST_PWSH to run the real PowerShell protocol fixture")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    token = "a" * 43
    token_file = tmp_path / "token.txt"
    token_file.write_text(token)
    script = tmp_path / "server.ps1"
    bridge = str(ROOT / "scripts/diagnostics/aqrc_cli_bridge.ps1").replace("'", "''")
    token_path = str(token_file).replace("'", "''")
    script.write_text(f"""
. '{bridge}'
$env:AETHMODULAR_AQRC_CONNECTION_STRING='fixture'
function Open-AqrcCliConnection {{
  $c=[pscustomobject]@{{Database='Fixture'}}
  $c | Add-Member ScriptMethod Dispose {{}}
  $c | Add-Member ScriptMethod CreateCommand {{
    $command=[pscustomobject]@{{CommandText='';CommandTimeout=0;Parameters=([Data.SqlClient.SqlCommand]::new()).Parameters}}
    $command | Add-Member ScriptMethod Dispose {{}}
    $command | Add-Member ScriptMethod ExecuteReader {{
      $table=[Data.DataTable]::new()
      $null=$table.Columns.Add('Connected',[int])
      $null=$table.Rows.Add(1)
      return ,$table.CreateDataReader()
    }}
    return $command
  }}
  return $c
}}
function Invoke-AqrcCliSql {{
  param($Connection,$Sql,$Parameters)
  return [pscustomobject]@{{Columns=@('Connected');Rows=@([pscustomobject]@{{Connected=1}})}}
}}
Start-AqrcCliBridge -TokenFile '{token_path}' -Port {port} -LifetimeMinutes {lifetime_minutes}
""")
    process = subprocess.Popen([pwsh, "-NoLogo", "-NoProfile", "-File", str(script)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() >= deadline or process.poll() is not None:
                    raise AssertionError("Fixture did not start")
                time.sleep(0.1)
        config = tmp_path / "client.json"
        config.write_text(json.dumps({"host": "127.0.0.1", "port": port, "token": token}))
        result = cli.request("probe", config, database="Networks_1_0")
        assert result["ok"] and result["rows"] == [{"Connected": 1}]
        caps = cli.request("capabilities", config)
        assert caps["protocolVersion"] == 2
        assert caps["queryAvailable"]
        assert cli.request("query", config, database="Networks_1_0", sql="DELETE x")["error"] == "single_select_required"
        assert cli.request("query", config, database="Networks_1_0", sql="SELECT 1 AS Connected")["rows"] == [{"Connected": "1"}]
        notes = cli.request("query", config, database="Networks_1_0", sql=cli.IMPORT_NOTES_SQL,
                            parameters={"@filter_id": "ETAD-0035", "@after_id": 0,
                                        "@include_suffixes": True, "@filter_pattern": "ETAD-0035-[0-9]%"})
        assert notes["ok"]
        assert cli.request("views", config, database="Networks_1_0")["ok"]
        assert cli.request("synonyms", config, database="Networks_1_0")["ok"]
        assert cli.request("probe", config, database="master")["error"] == "database_not_allowed"
        config.write_text(json.dumps({"host": "127.0.0.1", "port": port, "token": "wrong"}))
        assert cli.request("probe", config)["error"] == "unauthorized"
        with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
            connection.sendall(b"not-json\n")
            assert b'"error":"invalid_request"' in connection.recv(4096)
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
        expected = b"Runs until Ctrl+C." if lifetime_minutes == 0 else b"Stops after 1 minutes or Ctrl+C."
        assert expected in output
