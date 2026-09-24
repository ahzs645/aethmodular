"""Merge full-range VIBES checkpoints from every VM mirror into one run folder.

The full-range job (``scripts/vibes_fullrange_run.py``) was split across two Colab
VMs (the second running ``scripts/vibes_fullrange_shard.py`` on a batch range).
Each VM's checkpoints are mirrored to its own folder by ``monitor_vibes_colab.py``.
This script collects every ``batch_*.npz`` for one rank cap, checks that all carry
the same signature and that their indices tile 0..N-1 exactly, and writes:

    output/tables/vibes_fullrange/rank<cap>/corrected_VIBES_full.npy   (N x 2722, float32)
    output/tables/vibes_fullrange/rank<cap>/wn.npy                      (3998 -> 500 cm-1)
    output/tables/vibes_fullrange/rank<cap>/cases.csv                   (row order = earlier run)
    output/tables/vibes_fullrange/rank<cap>/fit_diagnostics.csv
    output/tables/vibes_fullrange/rank<cap>/ASSEMBLY.json

Cases are rebuilt with the bundle's own ``prepare_experiment`` (seeded, identical
to the earlier 1425-4000 run's cases.csv, which is asserted).

    uv run --no-sync --extra vibes python research/ftir_hips_chem/workflows/assemble_vibes_fullrange.py --rank 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
TABLES = AREA / "output/tables"
STAGE = TABLES / "vibes_colab_bundle_full500/stage"
EARLIER = TABLES / "vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    files = sorted(TABLES.glob(f"vibes_colab_full500_rank{a.rank}*/persistent_results/full500-rank{a.rank}-*/checkpoints/batch_*.npz"))
    if not files:
        raise SystemExit("no checkpoints mirrored yet")
    sys.path.insert(0, str(STAGE / "research/ftir_hips_chem/scripts"))
    from vibes_large_run import RunConfig, prepare_experiment

    wn, X, cases, *_ = prepare_experiment(STAGE, RunConfig(profile="full", max_background_components=a.rank))
    earlier = pd.read_csv(EARLIER / "cases.csv")
    assert (cases.sample_id.to_numpy() == earlier.sample_id.to_numpy()).all(), "case order changed"
    n = len(cases)
    vib = np.full((n, len(wn)), np.nan, np.float32)
    seen = np.zeros(n, bool)
    signatures, diags, sources = set(), [], {}
    for f in files:
        with np.load(f) as z:
            sig, idx, v = str(z["signature"]), z["indices"], z["vibes"]
            d = json.loads(str(z["diagnostics"]))
        signatures.add(sig)
        if seen[idx].any():
            # the same batch mirrored from two VMs: must agree exactly
            if not np.allclose(vib[idx], v, equal_nan=True):
                raise ValueError(f"conflicting duplicate batch {f.name}")
            continue
        vib[idx], seen[idx] = v, True
        diags.extend(d)
        sources[f.name] = str(f.relative_to(TABLES))
    if len(signatures) != 1:
        raise ValueError(f"mixed signatures: {signatures}")
    if not seen.all() and not a.allow_incomplete:
        missing = np.flatnonzero(~seen)
        raise SystemExit(f"{(~seen).sum()} cases not yet corrected (first {missing[:5]})")
    out = TABLES / f"vibes_fullrange/rank{a.rank}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "corrected_VIBES_full.npy", vib)
    np.save(out / "wn.npy", wn)
    cases.to_csv(out / "cases.csv", index=False)
    diag = pd.DataFrame(diags)
    diag.to_csv(out / "fit_diagnostics.csv", index=False)
    info = {"signature": signatures.pop(), "rank_cap": a.rank, "n_cases": n, "n_corrected": int(seen.sum()),
            "n_failed": int((~diag.success.astype(bool)).sum()) if len(diag) else None,
            "channels": len(wn), "wavenumber_range": [float(wn.max()), float(wn.min())],
            "checkpoint_files": len(sources),
            "sha256_corrected": hashlib.sha256(vib.tobytes()).hexdigest()}
    (out / "ASSEMBLY.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
