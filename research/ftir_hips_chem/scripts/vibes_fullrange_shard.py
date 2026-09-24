"""Run a contiguous range of batches of a vibes_fullrange_run job on another VM.

Colab allows two GPU-class runtimes (12 vCPU each) on this account, so a long
full-range job is split: one VM runs ``vibes_fullrange_run.py`` from batch 0, and
this script runs batches ``[--batch-from, --batch-to)`` elsewhere. The checkpoint
signature and output folder are computed exactly as in ``vibes_fullrange_run.run``
(bundle content hash + scientific config + package versions), so shards from
different VMs are interchangeable and ``workflows/assemble_vibes_fullrange.py``
merges them after verifying signatures and indices.

This file is uploaded next to the bundle's scripts; it is not part of the bundle
manifest, so adding it does not change the bundle hash or the signature.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import traceback

import numpy as np
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from vibes_large_run import RunConfig, prepare_experiment, verify_bundle
from vibes_baseline import fit_vibes_background
from vibes_fullrange_run import _correct, _status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-components", type=int, default=30)
    ap.add_argument("--batch-from", type=int, required=True, help="first batch number (batch = 64 cases)")
    ap.add_argument("--batch-to", type=int, required=True, help="one past the last batch number")
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    _status(a.root, state="running", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            shard=[a.batch_from, a.batch_to])
    try:
        config = RunConfig(profile="full", workers=a.workers, max_background_components=a.max_components)
        manifest = verify_bundle(a.bundle)
        versions = {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn", "cvxpy", "clarabel"]}
        sci = asdict(config)
        sci.pop("workers")
        signature = hashlib.sha256(json.dumps({"bundle": manifest["content_hash"], "config": sci, "versions": versions,
                                               "methods": ["VIBES"], "range": "full"}, sort_keys=True).encode()).hexdigest()
        out = a.root / "persistent_results" / f"full500-rank{a.max_components}-{signature[:16]}"
        ck = out / "checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        wn, X, cases, blanks, blank_ids, _, _ = prepare_experiment(a.bundle, config)
        with threadpool_limits(limits=max(1, a.workers)):
            bg = fit_vibes_background(wn, blanks, blank_ids=blank_ids, max_components=config.max_background_components)
        print(f"shard {a.batch_from}-{a.batch_to}: blank PCA rank {bg.components.shape[1]}, signature {signature[:16]}", flush=True)
        start = time.perf_counter()
        batches = [b for b in range(a.batch_from, a.batch_to) if b * config.batch_size < len(X)]
        for n, b in enumerate(batches, 1):
            first = b * config.batch_size
            idx = np.arange(first, min(first + config.batch_size, len(X)))
            path = ck / f"batch_{first:06d}.npz"
            if not path.exists():
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    rows = Parallel(n_jobs=a.workers)(delayed(_correct)(int(i), X[i], wn, bg, config) for i in idx)
                v = np.stack([r[0] for r in rows]).astype(np.float32)
                diag = [r[1] for r in rows]
                tmp = path.with_suffix(".partial")
                with tmp.open("wb") as fh:
                    np.savez_compressed(fh, signature=np.array(signature), indices=idx, vibes=v,
                                        diagnostics=np.array(json.dumps(diag, default=str)))
                os.replace(tmp, path)
            _status(a.root, state="running", done_batches=n, shard_batches=len(batches),
                    elapsed_min=round((time.perf_counter() - start) / 60, 1))
            print(f"batch {b} ({n}/{len(batches)}) elapsed {(time.perf_counter() - start) / 60:.1f} min", flush=True)
        _status(a.root, state="complete", result=str(out), finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except BaseException as exc:
        _status(a.root, state="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
