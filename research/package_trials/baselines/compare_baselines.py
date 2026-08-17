#!/usr/bin/env python3
"""Trial pybaselines / rampy baseline algorithms against the AIRSpec port.

Compares candidate baseline-correction algorithms from the 2026-08-17 package
survey (docs/package-survey-2026-08-17.md, section 1) against the validated
AIRSpec/APRLssb port (research/ftir_ec_phase3/scripts/airspec_baseline.py) on
Addis (ETAD) PTFE-filter FTIR spectra.

Ground truth is the cached AIRSpec output
research/ftir_ec_phase3/output/corrected/etad_corrected_df6.npz.  The
evaluation subset is restricted to MediaIds with exactly ONE replicate row in
the cache, so the replicate-averaged spectrum from
phase3_common.load_addis_evaluation() is bit-identical (modulo float32 cache
storage) to the raw row the cache was computed from — the comparison is then
exact, with no average-then-correct vs correct-then-average ambiguity.

Candidates (all run on the analyzed 1425.8-3998.4 cm^-1 window, ascending):
  - pybaselines mixture_model   (penalized spline + EM; survey's nearest cousin)
  - pybaselines pspline_arpls   (P-spline arPLS)
  - pybaselines arpls           (full-rank Whittaker arPLS; closest classic variant)
  - rampy gcvspline             (GCV smoothing spline through anchor windows;
                                 rampy 0.6.4 backs this with
                                 scipy.interpolate.make_smoothing_spline, so no
                                 FORTRAN gcvspline build is needed)

Each candidate is briefly tuned (lam / num_knots / s grids) on a small tuning
subset by minimizing the median RMS difference of the corrected spectra vs the
AIRSpec ground truth, then evaluated on the full subset: per-spectrum RMS,
Pearson correlation, deltas in the three diagnostic band features from
pls_transfer.ftir_source_band_features (CH_peak, carbonyl_peak,
shoulder_1600_peak), and runtime per spectrum vs the port.

Run from the repo root:
    python research/package_trials/baselines/compare_baselines.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "research/ftir_ec_phase3/scripts"))
sys.path.insert(0, str(REPO_ROOT / "research/ftir_hips_chem/scripts"))

from airspec_baseline import airspec_baseline_matrix  # noqa: E402
from phase3_common import load_addis_evaluation  # noqa: E402
from pls_transfer import ftir_source_band_features  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pybaselines  # noqa: E402
import rampy  # noqa: E402
from pybaselines import Baseline  # noqa: E402

CACHE = REPO_ROOT / "research/ftir_ec_phase3/output/corrected/etad_corrected_df6.npz"
OUT_DIR = Path(__file__).resolve().parent

# Anchor (non-analyte) windows for the rampy gcvspline ROI, mirroring the
# AIRSpec segment logic: below the ~1520-1600 minimum, the 1820-2220 fixed
# inter-segment window, and the high-wavenumber tail above the typical
# adaptive OH/CH bound.
GCV_ROI = np.array([[1426.0, 1560.0], [1820.0, 2250.0], [3600.0, 3998.0]])


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------

def load_aligned_subset(n_spectra: int, seed: int = 0):
    """Return (media_ids, wn_analyzed_desc, Y_raw_desc_full, wn_full_desc,
    truth_corrected) for singleton-replicate MediaIds."""
    cache = np.load(CACHE)
    wn_cached = cache["wn"]  # descending, 2002 points
    media = cache["media_id"].astype(str)
    counts = pd.Series(media).value_counts()
    singleton = set(counts[counts == 1].index)

    etad_eval, X_full, wn_full = load_addis_evaluation()
    eval_media = etad_eval["MediaId"].astype(str).to_numpy()
    usable = [i for i, m in enumerate(eval_media) if m in singleton]
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(usable, size=min(n_spectra, len(usable)),
                               replace=False))

    media_ids = eval_media[picked]
    Y_raw = X_full[picked]  # descending full grid (2722 points)

    cache_row = {m: i for i, m in enumerate(media)}
    truth = np.stack([cache["corrected"][cache_row[m]] for m in media_ids])
    truth = truth.astype(np.float64)

    # Column alignment of the analyzed window inside the full grid.
    col_of = {round(v, 6): i for i, v in enumerate(wn_full)}
    cols = np.array([col_of[round(v, 6)] for v in wn_cached])
    return media_ids, wn_cached, Y_raw, wn_full, cols, truth


# --------------------------------------------------------------------------
# Candidate runners: (wn_ascending, y_ascending, params) -> baseline_ascending
# --------------------------------------------------------------------------

def run_pybaselines(method: str):
    def _run(wn_asc, y_asc, params):
        fitter = Baseline(x_data=wn_asc)
        baseline, _ = getattr(fitter, method)(y_asc, **params)
        return baseline
    return _run


def run_rampy_gcvspline(wn_asc, y_asc, params):
    _, baseline = rampy.baseline(wn_asc, y_asc, GCV_ROI, "gcvspline", **params)
    return baseline.ravel()


CANDIDATES = {
    "mixture_model": run_pybaselines("mixture_model"),
    "pspline_arpls": run_pybaselines("pspline_arpls"),
    "arpls": run_pybaselines("arpls"),
    "rampy_gcvspline": run_rampy_gcvspline,
}

TUNING_GRIDS = {
    "mixture_model": [
        {"lam": lam, "num_knots": nk, "p": p}
        for lam in (1e4, 1e5, 1e6, 1e7, 1e8)
        for nk in (50, 100, 200)
        for p in (1e-3, 1e-2)
    ],
    "pspline_arpls": [
        {"lam": lam, "num_knots": nk}
        for lam in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
        for nk in (50, 100, 200)
    ],
    "arpls": [{"lam": lam} for lam in (1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11)],
    "rampy_gcvspline": [
        {"s": s} for s in (None, 0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    ],
}


def correct_matrix(runner, wn_desc, Y_desc, params):
    """Apply a candidate to each descending-grid spectrum; return corrected
    (descending orientation to match the cache) and elapsed seconds."""
    wn_asc = wn_desc[::-1].copy()
    out = np.empty_like(Y_desc)
    t0 = time.perf_counter()
    for i, row in enumerate(Y_desc):
        y_asc = row[::-1].copy()
        baseline = runner(wn_asc, y_asc, params)
        out[i] = (y_asc - baseline)[::-1]
    return out, time.perf_counter() - t0


def rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def pearson_rows(a, b):
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    return np.sum(a0 * b0, axis=1) / np.sqrt(
        np.sum(a0**2, axis=1) * np.sum(b0**2, axis=1)
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-spectra", type=int, default=50)
    parser.add_argument("--tune-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"pybaselines {pybaselines.__version__}, numpy {np.__version__}")
    media_ids, wn_an, Y_raw, wn_full, cols, truth = load_aligned_subset(
        args.n_spectra, args.seed
    )
    n = len(media_ids)
    print(f"subset: {n} singleton-replicate spectra; analyzed window "
          f"{wn_an[-1]:.1f}-{wn_an[0]:.1f} cm^-1 ({wn_an.size} points)")

    # --- AIRSpec port: sanity check vs cache + runtime -------------------
    t0 = time.perf_counter()
    _, port_corr = airspec_baseline_matrix(wn_full, Y_raw, df1=6, df2=4)
    port_s = time.perf_counter() - t0
    port_corr = port_corr[:, cols]
    port_vs_cache = rms(port_corr, truth)
    print(f"AIRSpec port: {port_s / n * 1000:.1f} ms/spectrum; "
          f"max RMS vs cache {port_vs_cache.max():.3e} (float32 storage)")

    Y_an = Y_raw[:, cols]  # analyzed window, descending
    tune_idx = np.arange(min(args.tune_n, n))

    # --- Tune ------------------------------------------------------------
    best = {}
    for name, grid in TUNING_GRIDS.items():
        runner = CANDIDATES[name]
        scores = []
        for params in grid:
            corr, _ = correct_matrix(runner, wn_an, Y_an[tune_idx], params)
            scores.append(np.median(rms(corr, truth[tune_idx])))
        k = int(np.argmin(scores))
        best[name] = grid[k]
        print(f"tuned {name}: {grid[k]} (median tuning RMS {scores[k]:.4f})")

    # --- Evaluate --------------------------------------------------------
    rows, per_spec = [], {}
    truth_feat = ftir_source_band_features(truth, wn_an)
    for name, runner in CANDIDATES.items():
        corr, elapsed = correct_matrix(runner, wn_an, Y_an, best[name])
        per_spec[name] = corr
        r = rms(corr, truth)
        cc = pearson_rows(corr, truth)
        feat = ftir_source_band_features(corr, wn_an)
        rec = {
            "method": name,
            "params": str(best[name]),
            "median_RMS": np.median(r),
            "p90_RMS": np.percentile(r, 90),
            "median_pearson_r": np.median(cc),
            "min_pearson_r": np.min(cc),
            "ms_per_spectrum": elapsed / n * 1000,
        }
        for band in ("CH_peak", "carbonyl_peak", "shoulder_1600_peak"):
            delta = feat[band].to_numpy() - truth_feat[band].to_numpy()
            rec[f"med_abs_d_{band}"] = np.median(np.abs(delta))
            scale = np.median(np.abs(truth_feat[band].to_numpy()))
            rec[f"med_abs_d_{band}_pct"] = (
                100 * rec[f"med_abs_d_{band}"] / scale if scale > 0 else np.nan
            )
        rows.append(rec)
        print(f"{name}: median RMS {rec['median_RMS']:.4f}, "
              f"median r {rec['median_pearson_r']:.4f}, "
              f"{rec['ms_per_spectrum']:.1f} ms/spectrum")

    results = pd.DataFrame(rows)
    results.insert(6, "port_ms_per_spectrum", port_s / n * 1000)
    results.to_csv(OUT_DIR / "results_summary.csv", index=False)
    print(f"wrote {OUT_DIR / 'results_summary.csv'}")

    # --- Plots -----------------------------------------------------------
    # Representative spectrum: median CH_peak in ground truth.
    rep = int(np.argsort(truth_feat["CH_peak"].to_numpy())[n // 2])
    rep_id = media_ids[rep]
    colors = {
        "mixture_model": "#D55E00", "pspline_arpls": "#0072B2",
        "arpls": "#009E73", "rampy_gcvspline": "#CC79A7",
    }

    # 1. raw + baselines
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(wn_an, Y_an[rep], color="0.25", lw=1.4, label="raw (analyzed window)")
    ax.plot(wn_an, Y_an[rep] - truth[rep], color="black", lw=2.0, ls="--",
            label="AIRSpec baseline (ground truth)")
    for name in CANDIDATES:
        ax.plot(wn_an, Y_an[rep] - per_spec[name][rep], lw=1.1,
                color=colors[name], label=f"{name} baseline")
    ax.set_xlabel("wavenumber (cm$^{-1}$)")
    ax.set_ylabel("absorbance")
    ax.set_title(f"MediaId {rep_id}: raw spectrum and fitted baselines")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_baselines_overlay.png", dpi=150)

    # 2. corrected overlays
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(wn_an, truth[rep], color="black", lw=2.0, label="AIRSpec corrected")
    for name in CANDIDATES:
        ax.plot(wn_an, per_spec[name][rep], lw=1.0, color=colors[name],
                label=f"{name} corrected")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xlabel("wavenumber (cm$^{-1}$)")
    ax.set_ylabel("baseline-corrected absorbance")
    ax.set_title(f"MediaId {rep_id}: corrected spectra")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_corrected_overlay.png", dpi=150)

    # 3. band-region zoom (1425-1900: shoulder + carbonyl, where methods differ)
    fig, ax = plt.subplots(figsize=(9, 5))
    zoom = wn_an <= 1950
    ax.plot(wn_an[zoom], truth[rep][zoom], color="black", lw=2.0,
            label="AIRSpec corrected")
    for name in CANDIDATES:
        ax.plot(wn_an[zoom], per_spec[name][rep][zoom], lw=1.1,
                color=colors[name], label=name)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xlabel("wavenumber (cm$^{-1}$)")
    ax.set_ylabel("corrected absorbance")
    ax.set_title(f"MediaId {rep_id}: carbonyl / 1600 shoulder region")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_carbonyl_zoom.png", dpi=150)
    print("wrote 3 figures")

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
