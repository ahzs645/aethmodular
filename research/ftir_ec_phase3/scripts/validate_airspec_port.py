#!/usr/bin/env python3
"""Validate the Python AIRSpec baseline port against the saved R output."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from airspec_baseline import SEG1, SEG2, airspec_baseline_matrix, make_mask


DEFAULT_ETAD_DIR = Path(
    "/Users/ahmadjalil/Library/CloudStorage/GoogleDrive-"
    "ahzs645@gmail.com/My Drive/University/Research/Grad/UC Davis Ann/"
    "NASA MAIA/Data/DAVIS/ETAD FTIR"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "output/tables/airspec_port_validation.csv"
)
RATIO_COLUMN = "max_abs_err_over_p95_abs_corrected"


def _spectral_columns(columns: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(columns, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(numeric) & (numeric >= 100.0) & (numeric <= 5000.0)
    return numeric, mask


def validate(raw_path: Path, truth_path: Path, output_path: Path, jobs: int) -> dict:
    raw = pd.read_csv(raw_path)
    truth = pd.read_csv(truth_path)
    numeric, wn_mask = _spectral_columns(raw.columns)
    non_wn = np.flatnonzero(~wn_mask)
    if non_wn.size == 0:
        raise ValueError("raw input has no non-wavenumber identifier column")

    x = numeric[wn_mask]
    if np.any(np.diff(x) >= 0):
        raise ValueError("raw wavenumbers are not strictly descending")
    y_all = raw.loc[:, wn_mask].to_numpy(dtype=float)
    good = np.isfinite(y_all).all(axis=1)
    y = y_all[good]
    ids = raw.iloc[good, non_wn[0]].astype("string").to_numpy(dtype=str)

    if truth.columns[0] != "id":
        raise ValueError(f"expected first ground-truth column 'id', got {truth.columns[0]!r}")
    truth_ids = truth.iloc[:, 0].astype("string").to_numpy(dtype=str)
    if not np.array_equal(ids, truth_ids):
        raise ValueError("ground-truth ids/order do not match filtered raw input")

    truth_labels = truth.columns[1:].to_numpy(dtype=str)
    raw_labels = raw.columns[wn_mask].to_numpy(dtype=str)
    if not np.array_equal(raw_labels, truth_labels):
        raise ValueError("ground-truth wavenumber columns do not match raw input")
    r_corrected = truth.iloc[:, 1:].to_numpy(dtype=float)

    analyte = make_mask(x, SEG1) | make_mask(x, SEG2)
    if not np.array_equal(np.isfinite(r_corrected).all(axis=0), analyte):
        raise ValueError("finite R output columns do not equal the strict AIRSpec region")
    if not np.isfinite(r_corrected[:, analyte]).all():
        raise ValueError("R output has non-finite values inside the AIRSpec region")

    started = time.perf_counter()
    _, py_corrected = airspec_baseline_matrix(
        x, y, df1=6, df2=4, n_jobs=jobs, chunksize=2
    )
    runtime_s = time.perf_counter() - started

    signed_error = py_corrected[:, analyte] - r_corrected[:, analyte]
    abs_error = np.abs(signed_error)
    max_abs = abs_error.max(axis=1)
    median_abs = np.median(abs_error, axis=1)
    rms_truth = np.sqrt(np.mean(np.square(r_corrected[:, analyte]), axis=1))
    p95_truth = np.percentile(np.abs(r_corrected[:, analyte]), 95, axis=1)
    ratio = np.divide(
        max_abs, p95_truth, out=np.full_like(max_abs, np.inf), where=p95_truth > 0
    )

    result = pd.DataFrame(
        {
            "id": ids,
            "max_abs_err": max_abs,
            "median_abs_err": median_abs,
            "rms_of_ground_truth_corrected": rms_truth,
            RATIO_COLUMN: ratio,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    wn = x[analyte]
    residual_profile = np.median(signed_error, axis=0)
    profile_r = float(np.corrcoef(wn, residual_profile)[0, 1])
    global_p95 = float(np.percentile(np.abs(r_corrected[:, analyte]), 95))
    max_profile = float(np.max(np.abs(residual_profile)))
    summary = {
        "raw_rows": int(len(raw)),
        "validated_rows": int(len(result)),
        "dropped_raw_rows": int((~good).sum()),
        "spectral_columns": int(wn_mask.sum()),
        "analyzed_columns": int(analyte.sum()),
        "runtime_s": runtime_s,
        "median_ratio": float(np.median(ratio)),
        "max_ratio": float(np.max(ratio)),
        "median_max_abs_err": float(np.median(max_abs)),
        "max_abs_err": float(np.max(max_abs)),
        "median_median_abs_err": float(np.median(median_abs)),
        "all_cell_rmse": float(np.sqrt(np.mean(np.square(signed_error)))),
        "median_signed_error": float(np.median(signed_error)),
        "max_abs_median_residual_profile": max_profile,
        "profile_to_global_p95_ratio": max_profile / global_p95,
        "residual_profile_wn_r": profile_r,
        "residual_profile_wn_r2": profile_r * profile_r,
        "accepted": bool(np.median(ratio) <= 0.02),
        "output_path": str(output_path),
    }
    print("AIRSpec Python-port validation against R ground truth")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_ETAD_DIR / "ETAD_FTIR_spectra.csv")
    parser.add_argument(
        "--truth",
        type=Path,
        default=DEFAULT_ETAD_DIR / "baseline_plots_AIRSPEC/spectra_baselined_AIRSPEC.csv",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        summary = validate(args.raw, args.truth, args.output, args.jobs)
    except Exception as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise
    raise SystemExit(0 if summary["accepted"] else 1)
