"""VIBES-only correction over the full exported grid (4000-500 cm-1), for Colab.

The 2026-09-21 comparison corrected both methods on AIRSpec's 4000-1425 cm-1
window. VIBES itself has no window (upstream pyvibes corrects whatever columns
it is given), so this run re-corrects the same cases on every channel. Case
selection, blank bank and site split come from ``vibes_large_run.prepare_experiment``
with the same seed, so rows pair one-to-one with the earlier run's cases.csv.

AIRSpec is not recomputed: the spline method stops at 1425 cm-1 by design and
its arrays from the earlier run are unchanged.

Layout matches the earlier cloud run so ``workflows/monitor_vibes_colab.py``
mirrors it unchanged:  <root>/persistent_results/<name>/checkpoints/*.npz  and
<root>/execution_status.json.

    python vibes_fullrange_run.py --bundle DIR --root /content/aeth_vibes --workers 12
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
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from vibes_large_run import RunConfig, prepare_experiment, verify_bundle
from vibes_baseline import fit_vibes_background, vibes_baseline_matrix


def _correct(i, y, wn, bg, config):
    with threadpool_limits(limits=1):
        _, vib, diag = vibes_baseline_matrix(wn, y[None], bg, sample_ids=[str(i)], tau=config.tau,
                                             loss=config.loss, maxiter=config.maxiter)
    rec = diag.iloc[0].to_dict()
    rec["case_row"] = i
    return vib[0], rec


def _status(root, **kw):
    path = root / "execution_status.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(kw)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(cur, indent=2))
    os.replace(tmp, path)


def run(bundle, root, workers, max_components=30):
    config = RunConfig(profile="full", workers=workers, max_background_components=max_components)
    manifest = verify_bundle(bundle)
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn", "cvxpy", "clarabel"]}
    sci = asdict(config)
    sci.pop("workers")
    signature = hashlib.sha256(json.dumps({"bundle": manifest["content_hash"], "config": sci, "versions": versions,
                                           "methods": ["VIBES"], "range": "full"}, sort_keys=True).encode()).hexdigest()
    out = root / "persistent_results" / f"full500-rank{max_components}-{signature[:16]}"
    ck = out / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    wn, X, cases, blanks, blank_ids, bmeta, _ = prepare_experiment(bundle, config)
    cases.to_csv(out / "cases.csv", index=False)
    bmeta.to_csv(out / "background_training_blanks.csv", index=False)
    print(f"{len(cases)} cases, {len(blanks)} blanks, {len(wn)} channels {wn.max():.0f}-{wn.min():.0f} cm-1", flush=True)
    with threadpool_limits(limits=max(1, workers)):
        bg = fit_vibes_background(wn, blanks, blank_ids=blank_ids, max_components=config.max_background_components)
    bg.cv_errors.to_csv(out / "background_rank_cv.csv", index=False)
    np.savez_compressed(out / "background_model.npz", wn=wn, mean=bg.mean, components=bg.components,
                        blank_ids=np.asarray(blank_ids, dtype=str))
    print(f"blank PCA rank {bg.components.shape[1]}", flush=True)
    vib = np.full(X.shape, np.nan, dtype=np.float32)
    records, start = [], time.perf_counter()
    for first in range(0, len(X), config.batch_size):
        idx = np.arange(first, min(first + config.batch_size, len(X)))
        path = ck / f"batch_{first:06d}.npz"
        if path.exists():
            with np.load(path) as z:
                if str(z["signature"]) != signature or not np.array_equal(z["indices"], idx):
                    raise ValueError(f"stale checkpoint {path}")
                v, diag = z["vibes"], json.loads(str(z["diagnostics"]))
        else:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                rows = Parallel(n_jobs=workers)(delayed(_correct)(int(i), X[i], wn, bg, config) for i in idx)
            v = np.stack([r[0] for r in rows]).astype(np.float32)
            diag = [r[1] for r in rows]
            tmp = path.with_suffix(".partial")
            with tmp.open("wb") as fh:
                np.savez_compressed(fh, signature=np.array(signature), indices=idx, vibes=v,
                                    diagnostics=np.array(json.dumps(diag, default=str)))
            os.replace(tmp, path)
        vib[idx] = v
        records.extend(diag)
        _status(root, state="running", done=int(idx[-1] + 1), total=len(X),
                elapsed_min=round((time.perf_counter() - start) / 60, 1))
        print(f"{idx[-1] + 1}/{len(X)} elapsed {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    np.save(out / "corrected_VIBES_full.npy", vib)
    np.save(out / "wn.npy", wn)
    diagnostics = pd.DataFrame(records)
    diagnostics.to_csv(out / "fit_diagnostics.csv", index=False)
    (out / "RUN_MANIFEST.json").write_text(json.dumps({
        "signature": signature, "config": asdict(config), "methods": ["VIBES"],
        "wavenumber_range": [float(wn.max()), float(wn.min())], "channels": int(len(wn)),
        "bundle_hash": manifest["content_hash"], "versions": versions, "n_cases": len(cases),
        "n_background_blanks": len(blanks), "background_rank": int(bg.components.shape[1]),
        "n_failed": int((~diagnostics.success.astype(bool)).sum()),
        "correction_wall_seconds": time.perf_counter() - start, "workers": workers}, indent=2) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-components", type=int, default=30,
                    help="blank PCA rank cap; 30 matches the 1425-4000 run, 85 = upstream one-SE rule uncapped")
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    _status(a.root, state="running", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        out = run(a.bundle, a.root, a.workers, a.max_components)
        _status(a.root, state="complete", result=str(out), finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except BaseException as exc:
        _status(a.root, state="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
