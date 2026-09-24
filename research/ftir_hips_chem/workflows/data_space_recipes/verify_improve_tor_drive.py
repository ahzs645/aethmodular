"""Compare the Drive-mounted TOR folder with the ingested repository copy.

Google Drive File Provider can hang on byte reads. Each read is isolated in a
subprocess with a timeout so an unavailable placeholder is reported rather
than silently treated as verified. This script does not modify either source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ingest_improve_tor_archive import DEFAULT_DRIVE, DEFAULT_SOURCE

HERE = Path(__file__).resolve().parent
DEFAULT_REPORT = HERE / "improve_tor_drive_verification.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(relative: str, source: Path, drive: Path, timeout: int) -> dict:
    expected = source / relative
    mounted = drive / relative
    result = {"path": relative, "expectedSha256": sha256(expected),
              "expectedBytes": expected.stat().st_size}
    if not mounted.is_file():
        return {**result, "status": "missing"}
    result["mountedBytes"] = mounted.stat().st_size
    if result["mountedBytes"] != result["expectedBytes"]:
        return {**result, "status": "size_mismatch"}
    try:
        completed = subprocess.run(["shasum", "-a", "256", str(mounted)],
                                   capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {**result, "status": "read_timeout"}
    if completed.returncode != 0:
        return {**result, "status": "read_error", "exitCode": completed.returncode}
    result["mountedSha256"] = completed.stdout.split()[0]
    result["status"] = ("matched" if result["mountedSha256"] == result["expectedSha256"]
                        else "hash_mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--drive", type=Path, default=DEFAULT_DRIVE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    drive = args.drive.expanduser()
    paths = sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file())
    if len(paths) != 14:
        raise RuntimeError(f"Expected 14 TOR source files, got {len(paths)}")
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(verify, relative, source, drive, args.timeout): relative
                   for relative in paths}
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(f"{result['status']} {result['path']}", flush=True)
    results.sort(key=lambda item: item["path"])
    report = {"source": "Davis Data/improve_tor_fractions",
              "comparison": "Google Drive File Provider mounted bytes against ingested repository copy",
              "files": results,
              "matched": sum(item["status"] == "matched" for item in results),
              "total": len(results)}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"matched": report["matched"], "total": report["total"],
                      "report": str(args.report)}))


if __name__ == "__main__":
    main()
