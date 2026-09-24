"""Execute the portable notebook on a CLI-managed Colab VM without Drive auth.

Upload the notebook and bundle to /content first. Run this script as a detached
subprocess, then download persistent_results/checkpoints periodically with the CLI.
The directory named persistent_results is local to the VM until downloaded.
"""

from pathlib import Path
import json
import os
import time
import traceback

import nbformat
from nbclient import NotebookClient


def main():
    root = Path("/content/aeth_vibes")
    root.mkdir(exist_ok=True)
    os.environ.update(
        AETH_VIBES_LOCAL_BUNDLE="/content/vibes_large_run_bundle.zip",
        AETH_VIBES_LOCAL_WORK=str(root),
        AETH_VIBES_PROFILE="full",
    )
    notebook = nbformat.read("/content/VIBES_AIRSpec_Large_Run_Colab.ipynb", as_version=4)
    output = root / "VIBES_AIRSpec_Large_Run_Colab_executed.ipynb"
    status = {"state": "running", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def save_status():
        path = root / "execution_status.json"
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(path)

    def cell_start(cell, cell_index, **kwargs):
        status["cell_index"] = cell_index
        save_status()
        print(f"Starting notebook cell {cell_index}", flush=True)

    def cell_done(cell, cell_index, **kwargs):
        nbformat.write(notebook, output)
        print(f"Finished notebook cell {cell_index}", flush=True)

    save_status()
    try:
        NotebookClient(
            notebook, timeout=None, kernel_name="python3",
            resources={"metadata": {"path": "/content"}},
            on_cell_start=cell_start, on_cell_executed=cell_done,
        ).execute()
        status["state"] = "complete"
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        nbformat.write(notebook, output)
        status["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_status()


if __name__ == "__main__":
    main()
