"""Audit failed full-range cases; optionally retry them in separate outputs.

The frozen result, checkpoints, background model and sample selection are never
modified. A larger optimizer budget is recorded as a numerical recovery attempt.
"""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

AREA = Path(__file__).resolve().parents[1]
REPO = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
from vibes_baseline import VibesBackground, vibes_baseline_matrix


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def retry_one(case, wn, y, background, config, maxiter):
    with threadpool_limits(limits=1):
        _, corrected, diag = vibes_baseline_matrix(
            wn, y[None], background, sample_ids=[case["sample_id"]],
            tau=config["tau"], loss=config["loss"], maxiter=maxiter,
        )
    record = diag.iloc[0].to_dict()
    record["case_row"] = case["case_row"]
    print(json.dumps(record, default=str), flush=True)
    return corrected[0], record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--retry-maxiter", type=int, default=0)
    args = ap.parse_args()
    assert args.retry_maxiter >= 0
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.run / "RUN_MANIFEST.json").read_text())
    bundle_manifest = json.loads((args.bundle / "BUNDLE_MANIFEST.json").read_text())
    assert bundle_manifest["content_hash"] == manifest["bundle_hash"]
    cases = pd.read_csv(args.run / "cases.csv")
    diag = pd.read_csv(args.run / "fit_diagnostics.csv")
    success = diag.success.astype(str).str.lower()
    assert success.isin(["true", "false"]).all()
    failed = diag.loc[success.eq("false")].copy()
    assert diag.case_row.tolist() == list(range(len(cases)))
    values = np.load(args.run / "corrected_VIBES_full.npy", mmap_mode="r")
    wn = np.load(args.run / "wn.npy")
    finite = np.isfinite(values).all(axis=1)
    assert np.array_equal(np.flatnonzero(~finite), failed.case_row.to_numpy())
    versions = {p: importlib.metadata.version(p) for p in manifest["versions"]}
    assert versions == manifest["versions"], "Recovery requires the frozen package versions"
    adapter = AREA / "scripts/vibes_baseline.py"
    assert sha(adapter) == sha(args.bundle / "research/ftir_hips_chem/scripts/vibes_baseline.py")
    for name in ["data/pool_raw.npy", "data/pool_metadata.csv", "data/wn.npy"]:
        assert sha(args.bundle / name) == bundle_manifest["files_sha256"][name]
    assert np.array_equal(wn, np.load(args.bundle / "data/wn.npy"))
    pool = pd.read_csv(args.bundle / "data/pool_metadata.csv")
    raw = np.load(args.bundle / "data/pool_raw.npy", mmap_mode="r")
    with np.load(args.run / "background_model.npz") as z:
        background = VibesBackground(z["wn"], z["mean"], z["components"],
                                     pd.DataFrame(), tuple(z["blank_ids"]), 0.0)
    assert np.array_equal(wn, background.wavenumbers)
    inputs, records = [], []
    for original in failed.to_dict("records"):
        i = int(original["case_row"])
        case = cases.iloc[i].to_dict()
        assert case["source"] == "improve", "This audit expects IMPROVE failures"
        source = pool.iloc[int(case["source_row"])]
        assert case["sample_id"] == f"improve:{int(source.FilterId)}"
        assert case["sample_id"] not in background.blank_ids
        y = np.asarray(raw[int(case["source_row"])], dtype=float)
        assert np.isfinite(y).all()
        with threadpool_limits(limits=1):
            residual = y - background.mean
            residual -= background.components @ np.linalg.lstsq(background.components, residual, rcond=None)[0]
        records.append(dict(case, case_row=i, original_diagnostics=original,
                            filter_purpose_id=int(source.FilterPurposeId),
                            all_scans_have_purpose=bool(source.all_scans_have_purpose),
                            raw_finite=True, raw_min=float(y.min()), raw_max=float(y.max()),
                            raw_sd=float(y.std()), pca_residual_rms=float(np.sqrt(np.mean(residual**2)))))
        inputs.append(y)
    coverage = cases.assign(finite=finite).groupby("kind").finite.agg(["count", "sum"])
    audit = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "original_signature": manifest["signature"],
        "original_result": args.run.resolve().relative_to(REPO).as_posix(),
        "original_array_sha256": sha(args.run / "corrected_VIBES_full.npy"),
        "background_sha256": sha(args.run / "background_model.npz"),
        "background_rank": int(background.components.shape[1]),
        "coverage": {kind: {"total": int(r["count"]), "finite": int(r["sum"])} for kind, r in coverage.iterrows()},
        "failed_cases": records,
        "versions": versions,
        "recovery": {"state": "running" if args.retry_maxiter and records else "not_requested",
                     "maxiter": args.retry_maxiter, "automatic_retry_maxiter": 5 * args.retry_maxiter,
                     "settings": "Same raw spectra, frozen rank-70 background, PB loss, tau and bounds; only optimizer iteration budget increases."},
    }
    audit_path = args.output / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str) + "\n")
    print(json.dumps({"coverage": audit["coverage"], "failed_ids": [r["sample_id"] for r in records]}, indent=2), flush=True)
    if args.retry_maxiter and records:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            rows = Parallel(n_jobs=min(2, len(records)))(
                delayed(retry_one)(r, wn, y, background, manifest["config"], args.retry_maxiter)
                for r, y in zip(records, inputs)
            )
        retry_diag = pd.DataFrame([r[1] for r in rows])
        retry_values = np.stack([r[0] for r in rows])
        assert np.array_equal(np.isfinite(retry_values).all(axis=1), retry_diag.success.to_numpy(bool))
        np.savez_compressed(args.output / "retry_spectra.npz", wn=wn, corrected=retry_values,
                            case_rows=np.array([r["case_row"] for r in records]),
                            sample_ids=np.array([r["sample_id"] for r in records]))
        retry_diag.to_csv(args.output / "retry_diagnostics.csv", index=False)
        audit["recovery"].update(state="complete", succeeded=int(retry_diag.success.sum()),
                                 failed=int((~retry_diag.success).sum()), diagnostics=retry_diag.to_dict("records"))
        assert audit["original_array_sha256"] == sha(args.run / "corrected_VIBES_full.npy")
        audit_path.write_text(json.dumps(audit, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
