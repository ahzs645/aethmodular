"""Reproducible paired Addis FTIR comparison: VIBES versus AIRSpec DF1=6/DF2=4."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))
from phase3_common import PATHS
from airspec_baseline import airspec_baseline_matrix, make_mask, SEG1, SEG2
from data_matching import base_filter_id
from outliers import apply_exclusion_flags, get_clean_data
from pls_transfer import ftir_source_band_features
from vibes_baseline import fit_vibes_background, vibes_baseline_matrix, VENDOR

SEED = 20260920
BANDS = ("CH_peak", "carbonyl_peak", "shoulder_1600_peak")


def load_comparison_data(full_range=False):
    """Average scans by physical filter, with an explicit full inclusion ledger.

    ``full_range`` keeps every channel the instrument exported (4000-500 cm-1)
    instead of AIRSpec's 4000-1425 segments, for VIBES runs that model the
    whole spectrum.
    """
    source = PATHS.etad_dir / "ETAD_FTIR_spectra.csv"
    metadata = PATHS.etad_dir / "ETAD_metadata.csv"
    lot_path = PATHS.etad_dir / "etad_spectra_lotmap.csv"
    raw = pd.read_csv(source)
    meta = pd.read_csv(metadata)
    columns = [c for c in raw if c not in ("SampleAnalysisId", "MediaId")]
    columns.sort(key=lambda c: -float(c))
    wn_full = np.array(columns, dtype=float)
    keep = np.ones(len(wn_full), bool) if full_range else (make_mask(wn_full, SEG1) | make_mask(wn_full, SEG2))
    columns = list(np.asarray(columns)[keep])
    wn = wn_full[keep]
    # pandas.mean can skip NaNs; flag any incomplete contributing scan first.
    raw["finite_scan"] = np.isfinite(raw[columns].to_numpy(float)).all(axis=1)
    grouped = raw.groupby("MediaId")
    spectra = grouped[columns].mean()
    ledger = meta.set_index("MediaId").join(grouped.size().rename("n_scans"))
    ledger["has_complete_spectrum"] = grouped.finite_scan.all().reindex(ledger.index).fillna(False)
    lots = pd.read_csv(lot_path).groupby("MediaId").LotId
    if (lots.nunique() > 1).any():
        raise ValueError("A physical filter maps to more than one lot")
    ledger["LotId"] = lots.first()
    ledger["date"] = pd.to_datetime(ledger.SamplingStartDate, errors="coerce")
    ledger["filter_id"] = ledger.ExternalFilterId.map(base_filter_id)
    ledger = apply_exclusion_flags(ledger, "Addis_Ababa")
    ledger["role"] = "outside_scope"
    eligible = get_clean_data(ledger).index
    valid = ledger.index.isin(eligible) & ledger.has_complete_spectrum
    ledger.loc[valid & ledger.ExternalFilterType.eq("PM2.5"), "role"] = "sample"
    ledger.loc[valid & ledger.ExternalFilterType.eq("FB"), "role"] = "blank_train"
    rng = np.random.default_rng(SEED)
    # Hold out whole blanks, within each lot; no replicate leakage.
    for _, group in ledger[ledger.role.eq("blank_train")].groupby("LotId", dropna=False):
        if len(group) >= 4:
            chosen = rng.choice(group.index, max(1, round(len(group) * 0.25)), replace=False)
            ledger.loc[chosen, "role"] = "blank_test"
    ledger["scope_reason"] = np.select(
        [ledger.is_excluded, ~ledger.has_complete_spectrum, ledger.ExternalFilterType.eq("PM10")],
        [ledger.exclusion_reason, "Missing/nonfinite spectrum", "PM10 outside PM2.5 comparison"],
        default="",
    )
    if (ledger.role == "blank_test").sum() < 2:
        raise ValueError("Need at least two held-out blanks")
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, metadata, lot_path)
    }
    return wn, spectra.reindex(ledger.index).to_numpy(float), ledger, hashes


def _airspec(wn, x):
    start = time.perf_counter()
    _, corrected = airspec_baseline_matrix(wn, x, df1=6, df2=4)
    return corrected, time.perf_counter() - start


def run_comparison(output_dir, *, progress=print):
    """Run real samples, independent blank checks, and controlled peak injection."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wn, x, ledger, hashes = load_comparison_data()
    train = ledger.role.eq("blank_train").to_numpy()
    sample = ledger.role.eq("sample").to_numpy()
    test = ledger.role.eq("blank_test").to_numpy()
    ids = ledger.index.to_numpy(dtype=str)
    progress(
        f"Included {sample.sum()} PM2.5 filters; {train.sum()} training and {test.sum()} held-out field blanks"
    )
    bg = fit_vibes_background(wn, x[train], blank_ids=ids[train])
    progress(f"Blank-only LOO selected {bg.components.shape[1]} PCA components")
    bg.cv_errors.to_csv(out / "blank_rank_cv.csv", index=False)

    def report(done, total):
        if done % 25 == 0 or done == total:
            progress(f"VIBES {done}/{total}")

    air, air_seconds = _airspec(wn, x[sample])
    _, vib, diag = vibes_baseline_matrix(wn, x[sample], bg, sample_ids=ids[sample], progress=report)
    diag["role"] = "sample"
    air_blank, air_blank_seconds = _airspec(wn, x[test])
    _, vib_blank, bd = vibes_baseline_matrix(wn, x[test], bg, sample_ids=ids[test])
    bd["role"] = "blank_test"
    # Fixed artificial peaks span narrow and broad bands. This tests recovery
    # of added absorbance, not chemical identity or actual analyte-free truth.
    centers = np.array([1610.0, 1720.0, 2920.0, 3400.0])
    widths = np.array([30.0, 22.0, 35.0, 130.0])
    shape = np.exp(-0.5 * ((wn[None, :] - centers[:, None]) / widths[:, None]) ** 2)
    amplitudes = [0.01, 0.05, 0.15]
    signals = np.stack([a * np.array([0.7, 0.5, 1.0, 0.8]) @ shape for a in amplitudes])
    truth = np.repeat(signals, test.sum(), axis=0)
    blanks = np.tile(x[test], (len(amplitudes), 1))
    injections = blanks + truth
    injection_ids = [f"{sid}:added:{a}" for a in amplitudes for sid in ids[test]]
    air_inj, air_inj_seconds = _airspec(wn, injections)
    _, vib_inj, sd = vibes_baseline_matrix(wn, injections, bg, sample_ids=injection_ids)
    sd["role"] = "injection"
    records = []
    recovery = {}
    for method, corrected, blank in [
        ("AIRSpec", air_inj, air_blank),
        ("VIBES", vib_inj, vib_blank),
    ]:
        recovered = corrected - np.tile(blank, (len(amplitudes), 1))
        recovery[method] = recovered
        for i, sid in enumerate(injection_ids):
            records.append(
                {
                    "method": method,
                    "sample_id": sid,
                    "added_amplitude": np.repeat(amplitudes, test.sum())[i],
                    "increment_rmse": np.sqrt(np.mean((recovered[i] - truth[i]) ** 2)),
                    "increment_bias": np.mean(recovered[i] - truth[i]),
                    "absolute_rmse_to_added_signal": np.sqrt(
                        np.mean((corrected[i] - truth[i]) ** 2)
                    ),
                }
            )
    pd.DataFrame(records).to_csv(out / "injection_metrics.csv", index=False)
    blank_rows = []
    for method, values in [("AIRSpec", air_blank), ("VIBES", vib_blank)]:
        for i, sid in enumerate(ids[test]):
            blank_rows.append(
                {
                    "method": method,
                    "sample_id": sid,
                    "rms_from_zero": np.sqrt(np.mean(values[i] ** 2)),
                    "mean_residual": values[i].mean(),
                }
            )
    pd.DataFrame(blank_rows).to_csv(out / "heldout_blank_metrics.csv", index=False)
    paired = np.isfinite(air).all(axis=1) & np.isfinite(vib).all(axis=1)
    metrics = ledger.loc[sample, ["ExternalFilterId", "LotId", "date", "n_scans"]].copy()
    metrics["paired_valid"] = paired
    metrics["rms_difference"] = np.sqrt(np.mean((vib - air) ** 2, axis=1))
    metrics["mean_difference"] = np.mean(vib - air, axis=1)
    correlation_r2 = np.full(len(vib), np.nan)
    for i in np.flatnonzero(paired):
        if np.std(air[i]) > 0 and np.std(vib[i]) > 0:
            correlation_r2[i] = np.corrcoef(air[i], vib[i])[0, 1] ** 2
    metrics["spectral_correlation_R2"] = correlation_r2
    for method, values in [("AIRSpec", air), ("VIBES", vib)]:
        features = ftir_source_band_features(values, wn)
        for band in BANDS:
            metrics[f"{method}_{band}"] = features[band].to_numpy()
        metrics[f"{method}_negative_fraction"] = np.mean(values < 0, axis=1)
        metrics.loc[~np.isfinite(values).all(axis=1), f"{method}_negative_fraction"] = np.nan
    metrics.to_csv(out / "paired_sample_metrics.csv")
    all_diag = pd.concat([diag, bd, sd], ignore_index=True)
    all_diag.to_csv(out / "vibes_fit_diagnostics.csv", index=False)
    ledger.to_csv(out / "inclusion_and_blank_split.csv")
    np.savez_compressed(
        out / "comparison_arrays.npz",
        wn=wn,
        raw=x[sample],
        airspec=air,
        vibes=vib,
        sample_ids=ids[sample],
        blank_ids=ids[test],
        blank_raw=x[test],
        blank_train=x[train],
        airspec_blank=air_blank,
        vibes_blank=vib_blank,
        injection_truth=truth,
        airspec_recovered=recovery["AIRSpec"],
        vibes_recovered=recovery["VIBES"],
    )
    timings = pd.DataFrame(
        [
            {
                "method": "AIRSpec",
                "stage": "samples",
                "seconds": air_seconds,
                "n": int(sample.sum()),
            },
            {
                "method": "VIBES",
                "stage": "samples",
                "seconds": diag.seconds.sum(),
                "n": int(sample.sum()),
            },
            {
                "method": "VIBES",
                "stage": "blank PCA + LOO",
                "seconds": bg.fit_seconds,
                "n": int(train.sum()),
            },
        ]
    )
    timings["ms_per_spectrum"] = 1000 * timings.seconds / timings.n
    timings.to_csv(out / "timing.csv", index=False)
    summary = {
        "seed": SEED,
        "sample_count": int(sample.sum()),
        "paired_count": int(paired.sum()),
        "blank_train_count": int(train.sum()),
        "blank_test_count": int(test.sum()),
        "vibes_components": int(bg.components.shape[1]),
        "vibes_tau": 0.1,
        "vibes_loss": "PB",
        "maxiter": 10000,
        "retry_maxiter": 50000,
        "retry_maxls": 50,
        "retried_fits": int(all_diag.retry_count.sum()),
        "rank_cap": int(train.sum()) - 2,
        "airspec_df1": 6,
        "airspec_df2": 4,
        "vibes_failures": int((~all_diag.success).sum()),
        "median_spectral_R2": float(metrics.spectral_correlation_R2.median()),
        "median_rms_difference": float(metrics.rms_difference.median()),
        "input_sha256": hashes,
        "vendor_manifest": json.loads((VENDOR / "SOURCE.json").read_text()),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "scipy", "pandas", "cvxpy", "clarabel", "scikit-learn"]
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    progress(f"Finished: {paired.sum()} paired samples; {summary['vibes_failures']} VIBES failures")
    return summary
