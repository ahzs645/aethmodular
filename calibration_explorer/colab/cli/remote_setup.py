"""Start the calibration explorer inside a Colab kernel driven by the `colab` CLI.

Run with `colab exec -s <session> -f calibration_explorer/colab/cli/remote_setup.py`
after uploading the prewarm bundle and mounting Drive (see ../README.md, "Colab
CLI"). It is the launcher notebook's setup cells as one script: unpack, runtime
layer, Drive paths, explorer server. Kernel state persists between `colab exec`
calls, so the server thread and the helpers defined here (`batch_status`,
`pack_results`) stay available to later calls in the same session.
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

BUNDLE = Path("/content/aethmodular_calibration_explorer_prewarm.zip")
WORK_ROOT = Path("/content/aethmodular")
PORT = 5058

if not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError("Google Drive is not mounted: run `colab drivemount -s <session>` in a terminal first")
if not BUNDLE.is_file():
    raise FileNotFoundError(f"Upload the prewarm bundle first: colab upload -s <session> <zip> {BUNDLE}")

# unpack once per VM; a second exec keeps the running server and its caches
if not (WORK_ROOT / "PREWARM_MANIFEST.json").is_file():
    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    WORK_ROOT.mkdir(parents=True)
    with zipfile.ZipFile(BUNDLE) as archive:
        archive.extractall(WORK_ROOT)
print("bundle:", json.loads((WORK_ROOT / "PREWARM_MANIFEST.json").read_text()))

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "Flask>=3,<4", "numpy>=2,<3", "pandas>=2,<3", "scipy>=1.13",
    "scikit-learn>=1.5", "matplotlib>=3.8", "pybaselines>=1.2",
    "polars>=1", "ikpls>=6.1",
])

os.environ["AETHMODULAR_DRIVE_ROOT"] = "/content/drive/MyDrive"
for p in (WORK_ROOT, WORK_ROOT / "calibration_explorer", WORK_ROOT / "research/ftir_hips_chem/scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pls_transfer import FTIRTransferPaths  # noqa: E402

resolved = FTIRTransferPaths.defaults().validate()
missing = resolved.loc[~resolved["exists"], "path"].tolist()
if missing:
    raise FileNotFoundError("Drive mounted, but these source paths are missing:\n" + "\n".join(missing))
print("Drive source paths: all", len(resolved), "found")

if "explorer_server" not in globals():
    from werkzeug.serving import make_server

    explorer = importlib.import_module("calibration_explorer.app")
    for elapsed in range(900):
        if explorer.STATE["ready"] or explorer.STATE["error"]:
            break
        if elapsed % 30 == 0:
            print(f"[{elapsed:>3}s] {explorer.STATE['message']}", flush=True)
        time.sleep(1)
    if not explorer.STATE["ready"]:
        raise RuntimeError(explorer.STATE)
    explorer_server = make_server("0.0.0.0", PORT, explorer.app, threaded=True)
    threading.Thread(target=explorer_server.serve_forever, daemon=True).start()

print("explorer ready on port", PORT)
for check in explorer.STATE["checks"]:
    print("✓" if check["ok"] else "⚠", check["name"], "-", check["detail"])


def batch_status():
    """One progress line; call with `echo 'batch_status()' | colab exec -s <session>`."""
    import requests

    s = requests.get(f"http://127.0.0.1:{PORT}/api/batch_status", timeout=30).json()
    pct = 100 * s["done"] / max(1, s["total"])
    print(f"{'running' if s['running'] else 'stopped'}  {s['done']}/{s['total']} ({pct:.1f} %)"
          f"  new rows {s['new_rows']}  failed {s['skipped']}  now: {s['current']}")
    for e in (s.get("errors") or [])[-3:]:
        print("  error:", e)
    return s


def pack_results(tag):
    """Zip the explorer cache (content-keyed fits + batch_results.jsonl) for `colab download`."""
    out = shutil.make_archive(f"/content/explorer_cache_{tag}", "zip",
                              root_dir=WORK_ROOT / "calibration_explorer", base_dir="cache")
    print(out, f"{Path(out).stat().st_size / 1e6:.1f} MB")
    return out
