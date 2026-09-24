"""Mirror a CLI-managed Colab experiment locally and release the VM on success."""

import argparse
import json
from pathlib import Path
import subprocess
import shutil
import tarfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "mirror_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    prefix = ["colab", "--auth=adc"]
    last_progress = None
    last_refresh = 0.0
    cli_python = Path(shutil.which("colab")).resolve().parent / "python"
    refresh_script = Path(__file__).with_name("refresh_vibes_colab_session.py")

    def cli(command, **kwargs):
        result = subprocess.run(prefix + command, capture_output=True, text=True, timeout=180, **kwargs)
        if result.returncode:
            raise RuntimeError(result.stderr + result.stdout)
        return result.stdout

    while True:
        try:
            # Runtime proxy tokens expire after one hour. The installed CLI
            # otherwise treats that 401 as a lost runtime and prunes its record.
            if time.monotonic() - last_refresh > 1200:
                refresh = subprocess.run(
                    [str(cli_python), str(refresh_script), "--record", str(output / "CLOUD_RUN.json")],
                    capture_output=True, text=True, timeout=90,
                )
                if refresh.returncode:
                    raise RuntimeError(refresh.stderr + refresh.stdout)
                last_refresh = time.monotonic()
            # Only immutable, atomically completed checkpoint files are mirrored
            # during execution. Final reports are mirrored after execution ends.
            script = '''
from pathlib import Path
import hashlib, json, tarfile
root = Path('/content/aeth_vibes')
status_file = root / 'execution_status.json'
status = json.loads(status_file.read_text()) if status_file.exists() else {'state':'starting'}
known = json.loads(LEDGER)
paths = list(root.glob('persistent_results/*/checkpoints/*.npz'))
if status['state'] in ('complete', 'failed'):
    paths += [p for p in root.glob('persistent_results/**/*') if p.is_file() and p.suffix != '.partial']
    paths += list(root.glob('*_executed.ipynb'))
    paths += list(root.glob('*_result.json'))
paths += [p for p in [status_file, root / 'execution.log'] if p.exists()]
paths = sorted(set(paths))
changed = [p for p in paths if known.get(str(p.relative_to(root))) != [p.stat().st_size, p.stat().st_mtime_ns]]
manifest = {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns] for p in changed}
parts = []
if changed:
    with tarfile.open('/content/aeth_vibes_snapshot.tar', 'w') as archive:
        for p in changed:
            archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
    with open('/content/aeth_vibes_snapshot.tar', 'rb') as source:
        while block := source.read(8 * 1024 * 1024):
            path = Path(f'/content/aeth_snapshot.part{len(parts):03d}')
            path.write_bytes(block)
            parts.append({'path':str(path), 'sha256':hashlib.sha256(block).hexdigest()})
counts = {}
for p in root.glob('persistent_results/*/checkpoints/*.npz'):
    name = p.parent.parent.name
    counts[name] = counts.get(name, 0) + 1
print('AETH_STATUS=' + json.dumps({'status':status, 'checkpoints':counts, 'changed':manifest, 'parts':parts}))
'''.replace("LEDGER", repr(json.dumps(ledger)))
            raw = cli(["exec", "-s", args.session, "--timeout", "120"], input=script)
            line = next(line for line in raw.splitlines() if line.startswith("AETH_STATUS="))
            record = json.loads(line.split("=", 1)[1])
            if record["changed"]:
                archive_path = output / "snapshot.tar"
                import hashlib

                with archive_path.open("wb") as archive_file:
                    for part in record["parts"]:
                        part_path = output / "snapshot.part"
                        cli(["download", "-s", args.session, part["path"], str(part_path)])
                        block = part_path.read_bytes()
                        if hashlib.sha256(block).hexdigest() != part["sha256"]:
                            raise RuntimeError("Snapshot download checksum mismatch")
                        archive_file.write(block)
                        part_path.unlink()
                with tarfile.open(archive_path) as archive:
                    archive.extractall(output, filter="data")
                ledger.update(record["changed"])
                ledger_path.write_text(json.dumps(ledger, indent=2))
                archive_path.unlink()
            progress = {"status": record["status"], "checkpoints": record["checkpoints"]}
            observed = dict(progress, monitor_connection="connected",
                            checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            (output / "monitor_status.json").write_text(json.dumps(observed, indent=2))
            if progress != last_progress:
                print(json.dumps(progress), flush=True)
                last_progress = progress
            state = record["status"]["state"]
            if state == "complete":
                print(cli(["stop", "-s", args.session]), flush=True)
                print("All results downloaded; Colab runtime stopped.", flush=True)
                return
            if state == "failed":
                raise SystemExit("Notebook failed; downloaded diagnostics. Runtime retained for recovery.")
        except (RuntimeError, subprocess.TimeoutExpired, StopIteration, json.JSONDecodeError) as exc:
            last_refresh = 0.0
            status_path = output / "monitor_status.json"
            observed = json.loads(status_path.read_text()) if status_path.exists() else {}
            observed.update(monitor_connection="error", monitor_error=str(exc),
                            checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            status_path.write_text(json.dumps(observed, indent=2))
            print(f"Monitor connection issue; retrying: {exc}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
