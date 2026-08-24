"""Build the Google Colab launcher and its prewarmed explorer bundle.

The bundle intentionally contains only repository-side inputs. The multi-gigabyte
FTIR source tree remains on Google Drive and is read through
``AETHMODULAR_DRIVE_ROOT=/content/drive/MyDrive`` after Colab mounts Drive.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOTEBOOK = HERE / "Calibration_Explorer_Colab.ipynb"
BUNDLE = HERE / "aethmodular_calibration_explorer_prewarm.zip"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def add_tree(archive: zipfile.ZipFile, path: Path) -> None:
    for source in sorted(path.rglob("*")):
        if not source.is_file():
            continue
        if "__pycache__" in source.parts or source.suffix in {".pyc", ".DS_Store"}:
            continue
        archive.write(source, source.relative_to(REPO))


def build_bundle() -> None:
    files = [
        REPO / "calibration_explorer/app.py",
        REPO / "calibration_explorer/hips_lab.py",
        REPO / "calibration_explorer/target_registry.json",
        REPO / "calibration_explorer/README.md",
        REPO / "calibration_explorer/ANALOG_CUTOFF_AUDIT_2026-08-18.md",
        REPO / "calibration_explorer/cache/analog_corrected_ranking.npz",
        REPO / "research/ftir_ec_phase3/output/corrected/improve_pool_corrected_df6.npz",
        REPO / "research/ftir_ec_phase3/output/corrected/etad_corrected_df6.npz",
        REPO / "research/ftir_ec_phase3/output/corrected/improve_pool_neutral_pspline_arpls_lam1e6.npz",
        REPO / "research/ftir_ec_phase3/output/corrected/etad_neutral_pspline_arpls_lam1e6.npz",
        REPO / "research/ftir_ec_phase3/output/corrected/neutral_pspline_arpls_lam1e6_manifest.json",
        REPO / "research/ftir_ec_phase3/output/tables/ftir11/lowest_ocec_800_cohort.csv",
        REPO / "research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl",
    ]
    trees = [
        REPO / "calibration_explorer/static",
        REPO / "calibration_explorer/targets",
        REPO / "research/ftir_ec_phase3/scripts",
        REPO / "research/ftir_hips_chem/scripts",
        REPO / "research/ftir_hips_chem/output/tables/pls_calibration_phase2",
        REPO / "research/ftir_hips_chem/output/tables/pls_transfer",
    ]
    missing = [path for path in files + trees if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prewarm inputs:\n" + "\n".join(map(str, missing)))

    manifest = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "drive_root_expected": "/content/drive/MyDrive",
        "analog_cutoff_semantics": (
            "Eligibility-first patch present: cutoff N counts TOR-eligible filters. "
            "The raw top-500 startup check must reproduce the locked 500/500 cohort."
        ),
    }
    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            archive.write(source, source.relative_to(REPO))
        for tree in trees:
            add_tree(archive, tree)
        archive.writestr("PREWARM_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "colab": {"name": NOTEBOOK.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Aethmodular Calibration Explorer: Colab launcher

## Goal

Mount Google Drive, unpack the prewarmed calibration-explorer bundle, validate the
required Drive data, start Flask, and open it through Colab's authenticated port proxy.

The prewarm bundle contains the AIRSpec and neutral-baseline caches, cohort tables,
shared scripts, filter dataset, and corrected-analog ranking cache. Disposable per-run CV-fit caches are omitted
to keep the download compact. The 4.6 GB FTIR source tree is **not duplicated**;
it is read from your mounted Drive.

> **Analog cutoff correction:** the bundled explorer counts TOR-eligible filters before
> applying cutoff N. The startup check should therefore report 500/500 for the locked raw
> analog cohort. Earlier cached 477/486 headline values must be regenerated under this rule.
"""
        ),
        nbf.v4.new_markdown_cell("## Setup\n\nRun the next cells in order."),
        nbf.v4.new_code_cell(
            """from pathlib import Path

BUNDLE_NAME = "aethmodular_calibration_explorer_prewarm.zip"
DRIVE_FOLDER = Path("/content/drive/MyDrive/Aethmodular Colab")
BUNDLE_PATH = DRIVE_FOLDER / BUNDLE_NAME
WORK_ROOT = Path("/content/aethmodular")
PORT = 5058
print("Expected bundle:", BUNDLE_PATH)"""
        ),
        nbf.v4.new_markdown_cell("### 1. Mount Drive"),
        nbf.v4.new_code_cell(
            """from google.colab import drive

drive.mount("/content/drive")
if not BUNDLE_PATH.is_file():
    candidates = list(Path("/content/drive/MyDrive").rglob(BUNDLE_NAME))
    if len(candidates) == 1:
        BUNDLE_PATH = candidates[0]
    elif not candidates:
        raise FileNotFoundError(f"Upload {BUNDLE_NAME} to Google Drive first")
    else:
        raise RuntimeError(f"Multiple bundles found; set BUNDLE_PATH explicitly: {candidates}")
print("Using:", BUNDLE_PATH)"""
        ),
        nbf.v4.new_markdown_cell("### 2. Unpack the prewarmed app"),
        nbf.v4.new_code_cell(
            """import json
import shutil
import zipfile

if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)
WORK_ROOT.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE_PATH) as archive:
    archive.extractall(WORK_ROOT)
manifest = json.loads((WORK_ROOT / "PREWARM_MANIFEST.json").read_text())
print(json.dumps(manifest, indent=2))"""
        ),
        nbf.v4.new_markdown_cell("### 3. Install the small runtime layer"),
        nbf.v4.new_code_cell(
            """import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "Flask>=3,<4", "numpy>=2,<3", "pandas>=2,<3", "scipy>=1.13",
    "scikit-learn>=1.5", "matplotlib>=3.8", "pybaselines>=1.2",
    "polars>=1", "ikpls>=6.1",
])
print("Runtime installed")"""
        ),
        nbf.v4.new_markdown_cell("### 4. Point the repo at mounted Drive and validate inputs"),
        nbf.v4.new_code_cell(
            """import os
import sys

os.environ["AETHMODULAR_DRIVE_ROOT"] = "/content/drive/MyDrive"
sys.path.insert(0, str(WORK_ROOT))
sys.path.insert(0, str(WORK_ROOT / "research/ftir_hips_chem/scripts"))

from pls_transfer import FTIRTransferPaths

resolved = FTIRTransferPaths.defaults().validate()
display(resolved)
missing = resolved.loc[~resolved["exists"], "path"].tolist()
if missing:
    raise FileNotFoundError(
        "Drive mounted, but required source paths were not found. "
        "Check AETHMODULAR_DRIVE_ROOT or your Drive folder layout:\\n" + "\\n".join(missing)
    )"""
        ),
        nbf.v4.new_markdown_cell("## Start and open the explorer"),
        nbf.v4.new_code_cell(
            """import importlib
import threading
import time
from werkzeug.serving import make_server

if "explorer_server" in globals():
    explorer_server.shutdown()

explorer = importlib.import_module("calibration_explorer.app")
for elapsed in range(240):
    if explorer.STATE["ready"] or explorer.STATE["error"]:
        break
    if elapsed % 10 == 0:
        print(explorer.STATE["message"])
    time.sleep(1)
if not explorer.STATE["ready"]:
    raise RuntimeError(explorer.STATE)

explorer_server = make_server("0.0.0.0", PORT, explorer.app, threaded=True)
explorer_thread = threading.Thread(target=explorer_server.serve_forever, daemon=True)
explorer_thread.start()
print("Explorer ready on port", PORT)
print("Startup checks:")
for check in explorer.STATE["checks"]:
    print("✓" if check["ok"] else "⚠", check["name"], "-", check["detail"])"""
        ),
        nbf.v4.new_code_cell(
            """from google.colab import output

output.serve_kernel_port_as_window(PORT)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Optional: exhaustive batch pre-compute

Runs the full cohort x cutoff-ladder x selection-space x spectra x protocol grid
**inside the server**, scoring every (configuration, k) row and appending it to
`calibration_explorer/cache/batch_results.jsonl`. Safe to leave running for hours;
progress prints below. Trim the GRID dict for a smaller pass.
"""
        ),
        nbf.v4.new_code_cell(
            """import time
import requests

# FLEET PATTERN: run several Colab sessions in parallel, each with a
# different slice of the grid (e.g. one protocol per session) -- the caches
# are content-keyed, so every session's cache zip merges cleanly back into
# calibration_explorer/cache/ on your own machine.
GRID = {
    "cohorts": ["eth_shaped", "analogs", "ocec", "smoke", "pool"],
    "spectra": ["raw", "airspec", "deriv2"],
    "modes": ["app"],         # THIS SESSION'S SLICE -- e.g. ["app"] here,
                              # ["app_fmm"] in a second session, etc.
    "corrsel": True,          # also select in AIRSpec-corrected space
    "cutoff_step": 10,        # dense mode: every 10th cutoff across the full
                              # range (eth 100-600, analogs 250-750,
                              # ocec 300-1500); set 0 for the 5-point ladder
    "cutoff_ladder": True,    # used only when cutoff_step is 0
    "sweep_k": True,          # sparse ladder always includes k=21 and k_max
    "k_min": 1,
    "k_max": 30,
    "dense_k": False,        # True = every integer 1..30 for every config (large)
    "lots": ["all", "251"],
    "match_eval_lot": True,  # train lot 251 -> Addis eval lot 251
    "target": "addis",
    "eval_lot": "all",
}
r = requests.post(f"http://127.0.0.1:{PORT}/api/batch_start", json=GRID, timeout=30).json()
print(r)
while True:
    s = requests.get(f"http://127.0.0.1:{PORT}/api/batch_status", timeout=30).json()
    print(f"{s['done']}/{s['total']}  {s['current']}  new rows: {s['new_rows']}"
          + (f"  failed: {s['skipped']}" if s["skipped"] else ""), flush=True)
    if not s["running"]:
        break
    time.sleep(30)
print("batch finished;", s["new_rows"], "rows saved")
if s.get("errors"):
    print("last errors:", *s["errors"], sep="\\n  ")"""
        ),
        nbf.v4.new_markdown_cell(
            """### Copy the cache (and batch results) back to Drive

Everything the batch computed: CV-curve caches, per-k fits, and
`batch_results.jsonl`: zips back to your Drive folder. On your own machine,
unzip it into `calibration_explorer/cache/` (merging is safe: files are
content-keyed) and click **Load saved results** in the app's Optimize tab.
"""
        ),
        nbf.v4.new_code_cell(
            """import shutil
import time

stamp = time.strftime("%Y%m%d_%H%M")
archive = shutil.make_archive(
    str(DRIVE_FOLDER / f"explorer_cache_{stamp}"), "zip",
    root_dir=WORK_ROOT / "calibration_explorer", base_dir="cache")
print("cache copied to Drive:", archive)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Checks and next steps

- Keep this Colab runtime open while using the explorer.
- Custom presets live in the browser's local storage, not in the bundle.
- New CV fits are cached in `/content/aethmodular/calibration_explorer/cache` for the
  current runtime. Copy that folder back to Drive if you want to preserve newly warmed
  configurations.
- The startup check for the locked raw analog cohort must report 500/500. A 477/500 result
  means an older bundle or server is still running.
"""
        ),
    ]
    nbf.write(notebook, NOTEBOOK)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    build_notebook()
    build_bundle()
    print(NOTEBOOK)
    print(BUNDLE)


if __name__ == "__main__":
    main()
