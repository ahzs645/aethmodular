#!/usr/bin/env python3
"""Create ETAD and IMPROVE AIRSpec-corrected NPZ deliverables."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd

from airspec_baseline import SEG1, SEG2, airspec_baseline_matrix, make_mask


DEFAULT_ETAD = Path(
    "/Users/ahmadjalil/Library/CloudStorage/GoogleDrive-"
    "ahzs645@gmail.com/My Drive/University/Research/Grad/UC Davis Ann/"
    "NASA MAIA/Data/DAVIS/ETAD FTIR/ETAD_FTIR_spectra.csv"
)
DEFAULT_IMPROVE = Path(
    "/Users/ahmadjalil/Library/CloudStorage/GoogleDrive-"
    "ahzs645@gmail.com/My Drive/FTIR/local_db/spectra_248_251.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output/corrected"


def _column_grid(columns: pd.Index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(columns, errors="coerce").to_numpy(dtype=float)
    spectral = np.isfinite(numeric) & (numeric >= 100.0) & (numeric <= 5000.0)
    x = numeric[spectral]
    if np.any(np.diff(x) >= 0):
        raise ValueError("wavenumbers must be strictly descending")
    analyzed = make_mask(x, SEG1) | make_mask(x, SEG2)
    return x, spectral, analyzed


def _strings(series: pd.Series) -> np.ndarray:
    return series.astype("string").fillna("").to_numpy(dtype=str)


def _save_npz(path: Path, corrected: np.ndarray, wn: np.ndarray, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in metadata.items()}
    arrays["wn"] = np.asarray(wn, dtype=np.float64)
    arrays["corrected"] = np.asarray(corrected, dtype=np.float32)
    np.savez_compressed(path, **arrays)
    print(f"wrote {path}: {corrected.shape[0]} rows x {corrected.shape[1]} wavenumbers")


def process_etad(path: Path, output_dir: Path, pool, chunksize: int) -> dict:
    started = time.perf_counter()
    data = pd.read_csv(path)
    x, spectral, analyzed = _column_grid(data.columns)
    y_all = data.loc[:, spectral].to_numpy(dtype=float)
    good = np.isfinite(y_all).all(axis=1)
    y = y_all[good]
    meta = {
        "sample_analysis_id": _strings(data.loc[good, "SampleAnalysisId"]),
        "media_id": _strings(data.loc[good, "MediaId"]),
    }
    for df1 in (6, 8):
        _, corrected = airspec_baseline_matrix(
            x, y, df1=df1, df2=4, pool=pool, chunksize=chunksize
        )
        _save_npz(
            output_dir / f"etad_corrected_df{df1}.npz",
            corrected[:, analyzed],
            x[analyzed],
            **meta,
        )
    elapsed = time.perf_counter() - started
    return {
        "source_rows": int(len(data)),
        "output_rows": int(good.sum()),
        "skipped_rows": int((~good).sum()),
        "wn_count": int(analyzed.sum()),
        "runtime_s": elapsed,
    }


def process_improve(
    path: Path,
    output_dir: Path,
    pool,
    task_chunksize: int,
    read_chunksize: int,
) -> dict:
    started = time.perf_counter()
    header = pd.read_csv(path, nrows=0)
    x, spectral, analyzed = _column_grid(header.columns)
    spectral_names = header.columns[spectral].to_numpy(dtype=str)
    analyzed_names = spectral_names[analyzed]
    metadata_names = ["AnalysisId", "FilterId", "SampleDate", "Site"]
    missing = [name for name in metadata_names if name not in header.columns]
    if missing:
        raise ValueError(f"IMPROVE input is missing metadata columns: {missing}")

    usecols = metadata_names + analyzed_names.tolist()
    ids: list[np.ndarray] = []
    filters: list[np.ndarray] = []
    dates: list[np.ndarray] = []
    sites: list[np.ndarray] = []
    corrected = {6: [], 8: []}
    source_rows = 0
    skipped_rows = 0

    reader = pd.read_csv(
        path,
        usecols=usecols,
        chunksize=read_chunksize,
        dtype={name: "string" for name in metadata_names},
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        source_rows += len(chunk)
        y_all = chunk.loc[:, analyzed_names].to_numpy(dtype=float)
        good = np.isfinite(y_all).all(axis=1)
        skipped_rows += int((~good).sum())
        if not good.any():
            print(f"IMPROVE chunk {chunk_number}: no finite analyzed rows")
            continue
        y = y_all[good]
        ids.append(_strings(chunk.loc[good, "AnalysisId"]))
        filters.append(_strings(chunk.loc[good, "FilterId"]))
        dates.append(_strings(chunk.loc[good, "SampleDate"]))
        sites.append(_strings(chunk.loc[good, "Site"]))
        for df1 in (6, 8):
            _, values = airspec_baseline_matrix(
                x[analyzed],
                y,
                df1=df1,
                df2=4,
                pool=pool,
                chunksize=task_chunksize,
            )
            corrected[df1].append(values.astype(np.float32, copy=False))
        print(
            f"IMPROVE chunk {chunk_number}: {int(good.sum())} kept, "
            f"{int((~good).sum())} skipped"
        )

    metadata = {
        "analysis_id": np.concatenate(ids),
        "filter_id": np.concatenate(filters),
        "site": np.concatenate(sites),
        "sample_date": np.concatenate(dates),
    }
    output_rows = len(metadata["analysis_id"])
    for df1 in (6, 8):
        values = np.concatenate(corrected[df1], axis=0)
        if values.shape != (output_rows, int(analyzed.sum())):
            raise RuntimeError(f"unexpected IMPROVE df{df1} output shape {values.shape}")
        _save_npz(
            output_dir / f"improve_pool_corrected_df{df1}.npz",
            values,
            x[analyzed],
            **metadata,
        )
    elapsed = time.perf_counter() - started
    return {
        "source_rows": source_rows,
        "output_rows": output_rows,
        "skipped_rows": skipped_rows,
        "wn_count": int(analyzed.sum()),
        "runtime_s": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etad", type=Path, default=DEFAULT_ETAD)
    parser.add_argument("--improve", type=Path, default=DEFAULT_IMPROVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--task-chunksize", type=int, default=4)
    parser.add_argument("--read-chunksize", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in methods else methods[0])
    total_started = time.perf_counter()
    with context.Pool(processes=args.jobs) as pool:
        etad = process_etad(args.etad, args.output_dir, pool, args.task_chunksize)
        improve = process_improve(
            args.improve,
            args.output_dir,
            pool,
            args.task_chunksize,
            args.read_chunksize,
        )
    total_runtime = time.perf_counter() - total_started
    print(f"ETAD summary: {etad}")
    print(f"IMPROVE summary: {improve}")
    print(f"total_runtime_s: {total_runtime}")


if __name__ == "__main__":
    main()
