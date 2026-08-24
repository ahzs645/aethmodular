#!/usr/bin/env python3
"""Build reproducible neutral-baseline caches for the pool and every target."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd

from neutral_baseline import DEFAULT_LAM, MASK_WINDOWS, _init_worker, neutral_baseline_matrix
from phase3_common import PATHS

REPO = Path(__file__).resolve().parents[3]
DEFAULT_POOL = PATHS.ftir_dir / "local_db/spectra_248_251.csv"
DEFAULT_ETAD = PATHS.etad_dir / "ETAD_FTIR_spectra.csv"
DEFAULT_OUTPUT = REPO / "research/ftir_ec_phase3/output/corrected"
DEFAULT_TARGETS = REPO / "calibration_explorer/targets"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid(columns: pd.Index) -> tuple[list[str], np.ndarray]:
    numeric = pd.to_numeric(columns, errors="coerce").to_numpy(float)
    mask = np.isfinite(numeric) & (numeric >= 100) & (numeric <= 5000)
    names = columns[mask].astype(str).tolist()
    return names, numeric[mask]


def _save_npz(path: Path, wn: np.ndarray, corrected: np.ndarray, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp,
        wn=np.asarray(wn, np.float64),
        corrected=np.asarray(corrected, np.float32),
        **{key: np.asarray(value) for key, value in metadata.items()},
    )
    temp.replace(path)


def process_pool(path: Path, output: Path, pool, read_chunksize: int, task_chunksize: int):
    header = pd.read_csv(path, nrows=0)
    names, wn = _grid(header.columns)
    metadata = ["AnalysisId", "FilterId", "SampleDate", "Site"]
    pieces, ids, filters, dates, sites = [], [], [], [], []
    source_rows = skipped = 0
    for number, chunk in enumerate(
        pd.read_csv(path, usecols=metadata + names, chunksize=read_chunksize), start=1
    ):
        source_rows += len(chunk)
        raw = chunk[names].to_numpy(float)
        good = np.isfinite(raw).all(axis=1)
        skipped += int((~good).sum())
        if good.any():
            retained_wn, corrected = neutral_baseline_matrix(
                wn, raw[good], pool=pool, chunksize=task_chunksize
            )
            pieces.append(corrected.astype(np.float32, copy=False))
            ids.append(chunk.loc[good, "AnalysisId"].astype(str).to_numpy())
            filters.append(chunk.loc[good, "FilterId"].astype(str).to_numpy())
            dates.append(chunk.loc[good, "SampleDate"].astype(str).to_numpy())
            sites.append(chunk.loc[good, "Site"].astype(str).to_numpy())
        print(f"pool chunk {number}: {int(good.sum())} kept, {int((~good).sum())} skipped")
    values = np.concatenate(pieces)
    _save_npz(
        output / "improve_pool_neutral_pspline_arpls_lam1e6.npz",
        retained_wn,
        values,
        analysis_id=np.concatenate(ids),
        filter_id=np.concatenate(filters),
        sample_date=np.concatenate(dates),
        site=np.concatenate(sites),
    )
    return {"source_rows": source_rows, "output_rows": len(values), "skipped_rows": skipped}


def process_etad(path: Path, output: Path, pool, task_chunksize: int):
    frame = pd.read_csv(path)
    names, wn = _grid(frame.columns)
    raw = frame[names].to_numpy(float)
    good = np.isfinite(raw).all(axis=1)
    retained_wn, corrected = neutral_baseline_matrix(
        wn, raw[good], pool=pool, chunksize=task_chunksize
    )
    _save_npz(
        output / "etad_neutral_pspline_arpls_lam1e6.npz",
        retained_wn,
        corrected,
        sample_analysis_id=frame.loc[good, "SampleAnalysisId"].astype(str).to_numpy(),
        media_id=frame.loc[good, "MediaId"].astype(str).to_numpy(),
    )
    return {"source_rows": len(frame), "output_rows": int(good.sum()),
            "skipped_rows": int((~good).sum())}


def process_targets(directory: Path, pool, task_chunksize: int) -> list[dict]:
    rows = []
    for folder in sorted(p for p in directory.iterdir() if p.is_dir()):
        source = folder / "spectra.csv"
        if not source.exists():
            continue
        frame = pd.read_csv(source)
        names, wn = _grid(frame.columns)
        raw = frame[names].to_numpy(float)
        good = np.isfinite(raw).all(axis=1)
        if not good.all():
            raise ValueError(f"{source}: neutral cache requires finite spectra")
        retained_wn, corrected = neutral_baseline_matrix(
            wn, raw, pool=pool, chunksize=task_chunksize
        )
        target = folder / "spectra_neutral.csv"
        out = pd.DataFrame(corrected, columns=[f"{value:g}" for value in retained_wn])
        out.insert(0, frame.columns[0], frame.iloc[:, 0].to_numpy())
        temp = target.with_suffix(".tmp.csv")
        out.to_csv(temp, index=False)
        temp.replace(target)
        rows.append({"target": folder.name, "rows": len(out), "path": str(target)})
        print(f"target {folder.name}: {len(out)} rows")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--etad", type=Path, default=DEFAULT_ETAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets-dir", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--read-chunksize", type=int, default=1024)
    parser.add_argument("--task-chunksize", type=int, default=8)
    parser.add_argument("--skip-targets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in methods else methods[0])
    # All source grids are identical here; initialize on the canonical pool grid.
    header = pd.read_csv(args.pool, nrows=0)
    _, raw_wn = _grid(header.columns)
    from neutral_baseline import neutral_grid
    retained_wn, _ = neutral_grid(raw_wn)
    with context.Pool(
        processes=args.jobs, initializer=_init_worker, initargs=(retained_wn, DEFAULT_LAM)
    ) as pool:
        pool_summary = process_pool(
            args.pool, args.output_dir, pool, args.read_chunksize, args.task_chunksize
        )
        etad_summary = process_etad(args.etad, args.output_dir, pool, args.task_chunksize)
        targets = [] if args.skip_targets else process_targets(
            args.targets_dir, pool, args.task_chunksize
        )
    manifest = {
        "method": "pybaselines.Baseline.pspline_arpls",
        "lambda": DEFAULT_LAM,
        "mask_windows_cm-1": [[str(lo), str(hi)] for lo, hi in MASK_WINDOWS],
        "wavenumber_order": "ascending",
        "wavenumber_count": len(retained_wn),
        "sources": {
            "pool": {"path": str(args.pool), "sha256": _sha256(args.pool)},
            "etad": {"path": str(args.etad), "sha256": _sha256(args.etad)},
        },
        "pool": pool_summary,
        "etad": etad_summary,
        "targets": targets,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    path = args.output_dir / "neutral_pspline_arpls_lam1e6_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

