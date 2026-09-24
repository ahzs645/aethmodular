"""Start one protocol's slice of the batch grid in a kernel prepared by remote_setup.py.

Set the slice first, then run this file:

    echo 'BATCH_MODES = ["app_fmm"]' | colab exec -s aeth-b2
    colab exec -s aeth-b2 -f calibration_explorer/colab/cli/remote_batch.py

The batch runs in the explorer's own worker thread and this call returns at
once; poll with `echo 'batch_status()' | colab exec -s aeth-b2`.

Scope (2026-09-23): fill protocols B (`app`) and B2 (`app_fmm`) to the same
dense cutoff grid and five sites that protocol A already has, AIRSpec
calibration spectra only (the 2026-09-17 meeting's corrected-spectra rule).
Held-out TOR R² stays empty for B/B2 by construction: they fit on every
IMPROVE site, so no sites are held out to score.
"""

import requests

GRID = {
    "cohorts": ["eth_shaped", "analogs", "ocec", "smoke", "pool"],
    "spectra": ["airspec"],
    "modes": globals().get("BATCH_MODES", ["app"]),
    "corrsel": True,           # select on raw and on AIRSpec-corrected spectra (eth_shaped, analogs)
    "cutoff_step": 10,         # dense: eth 100-600, analogs 250-750, ocec 300-1500
    "sweep_k": True,
    "k_min": 1,
    "k_max": 30,
    "dense_k": False,
    "lots": ["all"],
    "targets": ["addis", "etbi", "indh", "chts", "uspa"],
    "eval_lot": "all",
    "eval_group": "all",
    "eval_splits": ["all"],
}

r = requests.post("http://127.0.0.1:5058/api/batch_start", json=GRID, timeout=60).json()
print({k: v for k, v in r.items() if k != "configs"} if isinstance(r, dict) else r)
