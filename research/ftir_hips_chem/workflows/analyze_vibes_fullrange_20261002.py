"""Paired full-range VIBES diagnostics and a frozen-split EC benchmark.

Retains every frozen case. Numerical recoveries replace only the two failed
blank rows in a derived copy. Component selection uses calibration sites only;
the same held-out sites evaluate every rank/window variant. No test-case tuning.
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
TABLES = AREA / "output/tables"
OUT = TABLES / "vibes_fullrange/interpretation_20261002"
PLOTS = AREA / "output/plots/vibes_fullrange_20261002"
sys.path.insert(0, str(AREA / "scripts"))
from plotting import PlotConfig, overlays
from plotting.utils import style_axes
from vibes_large_run import RunConfig, compare_calibrations, paired_site_bootstrap

LABELS = {
    "AIRSpec": "Spline baseline · shared window",
    "VIBES": "VIBES rank 70 · shared window",
    "capped_shared": "VIBES rank 30 · shared window",
    "uncapped_full": "VIBES rank 70 · full window",
    "capped_full": "VIBES rank 30 · full window",
    "archived_truncated": "VIBES rank 30 · fitted on shared window",
}
COLORS = {"AIRSpec": "#2171b5", "VIBES": "#d62728", "capped_shared": "#9467bd",
          "uncapped_full": "#d62728", "capped_full": "#9467bd", "archived_truncated": "#666666"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(fig, name):
    fig.savefig(PLOTS / f"{name}.png", bbox_inches="tight")
    fig.savefig(PLOTS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reuse-benchmark", action="store_true", help="Reuse saved predictions only when all input hashes match")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    PlotConfig.set(sites="all", layout="individual", figsize=(12, 9),
                   figsize_grid=(13, 7), font_size=10, title_size=12)
    run = TABLES / "vibes_colab_full500_rank85_resume_20261001/persistent_results/full500-rank85-9345e818c61929fe"
    recovery = TABLES / "vibes_fullrange/failure_check_20261002"
    capped = TABLES / "vibes_fullrange/rank30"
    earlier = TABLES / "vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"
    inputs = [run / "corrected_VIBES_full.npy", run / "cases.csv", run / "wn.npy",
              run / "RUN_MANIFEST.json", recovery / "audit.json", recovery / "retry_spectra.npz",
              capped / "corrected_VIBES_full.npy", capped / "cases.csv",
              earlier / "corrected_AIRSpec.npy", earlier / "corrected_VIBES.npy", earlier / "cases.csv", earlier / "wn.npy"]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    cases = pd.read_csv(run / "cases.csv")
    for other in (capped, earlier):
        assert cases.sample_id.tolist() == pd.read_csv(other / "cases.csv").sample_id.tolist()
    wn = np.load(run / "wn.npy")
    shared = wn >= 1425
    assert np.array_equal(wn[shared], np.load(earlier / "wn.npy"))
    v70 = np.load(run / "corrected_VIBES_full.npy").copy()
    audit = json.loads((recovery / "audit.json").read_text())
    assert audit["original_array_sha256"] == hashes[(run / "corrected_VIBES_full.npy").relative_to(ROOT).as_posix()]
    with np.load(recovery / "retry_spectra.npz") as z:
        rows = z["case_rows"]
        assert np.array_equal(rows, np.flatnonzero(~np.isfinite(v70).all(axis=1)))
        assert cases.iloc[rows].sample_id.tolist() == z["sample_ids"].tolist()
        assert np.array_equal(wn, z["wn"]) and np.isfinite(z["corrected"]).all()
        v70[rows] = z["corrected"]
    assert np.isfinite(v70).all()
    np.save(OUT / "corrected_VIBES_full_recovered.npy", v70)
    cases.to_csv(OUT / "cases.csv", index=False)
    v30 = np.load(capped / "corrected_VIBES_full.npy", mmap_mode="r")
    air = np.load(earlier / "corrected_AIRSpec.npy", mmap_mode="r")
    old = np.load(earlier / "corrected_VIBES.npy", mmap_mode="r")
    assert np.isfinite(v30).all() and np.isfinite(air).all() and np.isfinite(old).all()
    arrays = {"AIRSpec": air, "VIBES": v70[:, shared], "capped_shared": v30[:, shared],
              "uncapped_full": v70, "capped_full": v30, "archived_truncated": old}
    grids = {m: wn if m.endswith("full") else wn[shared] for m in arrays}
    spectral_records = []
    for kind in ["calibration", "target", "blank"]:
        idx = np.flatnonzero(cases.kind.eq(kind))
        rms = np.sqrt(np.mean((v70[idx] - v30[idx]) ** 2, axis=1))
        spectral_records.append({"kind": kind, "n": len(idx), "median_rank70_minus_rank30_rms": float(np.median(rms))})
    pd.DataFrame(spectral_records).to_csv(OUT / "rank_sensitivity.csv", index=False)

    # Spectral comparison: paired filters, absorbance units, no extrapolation.
    fig, axs = plt.subplots(2, 1, figsize=PlotConfig.get("figsize"), layout="constrained")
    for ax, kind, title in zip(axs, ["calibration", "target"], ["IMPROVE source population (calibration + held-out)", "Addis test-set spectra"]):
        idx = np.flatnonzero(cases.kind.eq(kind))
        for method in ["AIRSpec", "capped_full", "uncapped_full"]:
            q = np.quantile(arrays[method][idx], [0.25, 0.5, 0.75], axis=0)
            ax.plot(grids[method], q[1], color=COLORS[method], lw=1.4, label=LABELS[method])
            ax.fill_between(grids[method], q[0], q[2], color=COLORS[method], alpha=0.10)
        ax.axvline(1425, ls="--", lw=0.8, color="0.45")
        ax.axhline(0, lw=0.6, color="0.6")
        ax.set_xlim(4000, 500)
        style_axes(ax, "Wavenumber (cm⁻¹)", "Corrected absorbance", f"{title}: same {len(idx):,} physical filters")
    fig.suptitle("Full-range correction: median and interquartile range\nSpline baseline ends at 1425 cm⁻¹; exported full grid ends at 500 cm⁻¹")
    save(fig, "01_spectral_overlay")

    # Held-out blanks: same common window for the primary paired comparison.
    blank = np.flatnonzero(cases.kind.eq("blank"))
    blank_records = []
    primary = ["AIRSpec", "capped_shared", "VIBES"]
    fig, axs = plt.subplots(2, 1, figsize=PlotConfig.get("figsize"), layout="constrained")
    for ax, methods, title in [(axs[0], primary, "Shared 4000–1425 cm⁻¹ window"),
                               (axs[1], ["capped_full", "uncapped_full"], "Full 4000–500 cm⁻¹ window")]:
        distributions = []
        for method in methods:
            rms = np.sqrt(np.mean(arrays[method][blank] ** 2, axis=1))
            distributions.append(rms)
            for i, value in zip(blank, rms):
                blank_records.append({"sample_id": cases.iloc[i].sample_id, "source": cases.iloc[i].source,
                                      "method": method, "rms_absorbance": value, "numerically_recovered": bool(i in rows)})
        bp = ax.boxplot(distributions, tick_labels=[LABELS[m].replace(" · ", "\n") for m in methods], patch_artist=True)
        for box, m in zip(bp["boxes"], methods):
            box.set_facecolor(COLORS[m])
            box.set_alpha(0.4)
        ax.set_yscale("log")
        style_axes(ax, "", "Residual RMS absorbance (log scale)", title, show_legend=False)
    fig.suptitle("135 held-out blanks: smaller residual is useful, but does not prove signal preservation\nThe two numerically recovered blanks remain included")
    save(fig, "02_blank_residuals")
    blanks = pd.DataFrame(blank_records)
    blanks.to_csv(OUT / "blank_metrics.csv", index=False)

    # Synthetic additions are a known within-blank contrast, not EC standards.
    injection_records = []
    for i in np.flatnonzero(cases.kind.eq("injection")):
        parent = int(cases.iloc[i].parent)
        amplitude = float(cases.iloc[i].amplitude)
        for method, values in arrays.items():
            w = grids[method]
            shape = sum(a * np.exp(-0.5 * ((w - c) / width) ** 2) for a, c, width in
                        zip([0.7, 0.5, 1.0, 0.8], [1610., 1720., 2920., 3400.], [30., 22., 35., 130.]))
            truth = amplitude * shape
            contrast = values[i] - values[parent]
            injection_records.append({"sample_id": cases.iloc[i].sample_id, "method": method,
                                      "amplitude": amplitude, "parent": parent,
                                      "signal_area_recovery": float(np.sum(contrast) / np.sum(truth)),
                                      "relative_rms_error": float(np.sqrt(np.mean((contrast-truth)**2)) / np.sqrt(np.mean(truth**2)))})
    injections = pd.DataFrame(injection_records)
    injections.to_csv(OUT / "injection_metrics.csv", index=False)
    fig, axs = plt.subplots(2, 1, figsize=PlotConfig.get("figsize"), layout="constrained")
    for ax, metric, ideal, label in [(axs[0], "signal_area_recovery", 1, "Recovered / added spectral area"),
                                      (axs[1], "relative_rms_error", 0, "RMS error / known-signal RMS")]:
        for method in primary:
            grouped = injections.loc[injections.method.eq(method)].groupby("amplitude")[metric]
            q = grouped.quantile([.25, .5, .75]).unstack()
            ax.plot(q.index, q[.5], marker="o", color=COLORS[method], label=LABELS[method])
            ax.fill_between(q.index, q[.25], q[.75], color=COLORS[method], alpha=.12)
        ax.axhline(ideal, color="0.5", ls="--", lw=.8)
        style_axes(ax, "Synthetic addition amplitude (absorbance)", label, "Shared window; median and IQR across nine Addis blank parents")
    fig.suptitle("Signal preservation: 27 synthetic spectral additions\nA numerical stress test, not independent EC accuracy or chemical identification")
    save(fig, "03_known_signal_recovery")
    print("Spectral, blank and injection figures saved. Starting paired calibration benchmark…", flush=True)

    benchmark = OUT / "benchmark"
    benchmark.mkdir(exist_ok=True)
    if args.reuse_benchmark:
        previous = json.loads((OUT / "summary.json").read_text())
        assert previous["source_hashes"] == hashes, "Cached benchmark inputs changed"
        scores = pd.read_csv(benchmark / "calibration_scores.csv")
    else:
        with threadpool_limits(limits=4):
            scores = compare_calibrations(cases, arrays, benchmark, RunConfig())
    # Reproduce the archived two-method benchmark before interpreting new variants.
    prior = pd.read_csv(earlier / "calibration_scores.csv")
    for cohort in ["locked800", "full_pool"]:
        for method, previous_method in [("AIRSpec", "AIRSpec"), ("archived_truncated", "VIBES")]:
            new = scores.loc[scores.cohort.eq(cohort) & scores.method.eq(method)].iloc[0]
            original = prior.loc[prior.cohort.eq(cohort) & prior.method.eq(previous_method)].iloc[0]
            assert new.k == original.k and np.isclose(new.RMSE, original.RMSE, atol=1e-6)
    predictions = pd.read_csv(benchmark / "heldout_predictions.csv")
    intervals = []
    for cohort in ["locked800", "full_pool"]:
        p = predictions.loc[predictions.cohort.eq(cohort)].pivot(index="sample_id", columns="method", values="prediction")
        truth = predictions.loc[predictions.cohort.eq(cohort)].drop_duplicates("sample_id").set_index("sample_id").loc[p.index]
        for method in arrays:
            if method != "AIRSpec":
                intervals.append(dict(cohort=cohort, method=method, comparator="AIRSpec", **paired_site_bootstrap(
                    truth.y, p.AIRSpec, p[method], truth.Site, repeats=2000, seed=20260920)))
    intervals = pd.DataFrame(intervals)
    intervals.to_csv(benchmark / "all_paired_site_intervals.csv", index=False)
    fig, axs = plt.subplots(2, 1, figsize=PlotConfig.get("figsize"), layout="constrained")
    for ax, cohort in zip(axs, ["locked800", "full_pool"]):
        s = scores.loc[scores.cohort.eq(cohort)]
        xs = np.arange(len(s))
        ax.bar(xs, s.RMSE, color=[COLORS[m] for m in s.method], alpha=.8)
        ax.set_xticks(xs, [LABELS[m].replace(" · ", "\n") for m in s.method], rotation=12, ha="right", fontsize=8)
        for x, row in zip(xs, s.itertuples()):
            ax.text(x, row.RMSE, f"{row.RMSE:.3f}\nK={row.k}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, s.RMSE.max()*1.24)
        style_axes(ax, "", "Held-out TOR EC RMSE (µg/filter)",
                   f"{'Lowest OC/EC cohort' if cohort=='locked800' else 'Full calibration pool'}: {int(s.n_train.iloc[0]):,} calibration filters, {int(s.n_test.iloc[0]):,} held-out filters", show_legend=False)
    fig.suptitle("Source-domain prediction: fixed held-out sites for all methods\nPLS factor count K selected by site-grouped cross-validation on calibration sites only")
    save(fig, "04_source_prediction_benchmark")
    fig, axs = plt.subplots(1, 2, figsize=PlotConfig.get("figsize_grid"), layout="constrained")
    plotted = predictions.loc[predictions.cohort.eq("full_pool") & predictions.method.isin(["AIRSpec", "uncapped_full"])]
    lo = min(0, plotted.prediction.min(), plotted.y.min())
    hi = max(plotted.prediction.max(), plotted.y.max()) * 1.04
    for ax, method in zip(axs, ["AIRSpec", "uncapped_full"]):
        p = predictions.loc[predictions.cohort.eq("full_pool") & predictions.method.eq(method)]
        overlays.crossplot_on_axes(ax, p.y, p.prediction, "Held-out TOR EC (µg/filter)",
                                  "Predicted FTIR EC (µg/filter)", color=COLORS[method], stats_box=False, fit_line=False)
        for points in ax.collections:
            points.set_sizes([15])
            points.set_edgecolor("none")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        row = scores.loc[scores.cohort.eq("full_pool") & scores.method.eq(method)].iloc[0]
        ax.set_title(f"{LABELS[method]}\nPredictive R²={row.predictive_R2:.3f}; RMSE={row.RMSE:.3f}")
    fig.suptitle("Full-pool held-out predictions: negative predictions remain in the audit\nThese are IMPROVE results, not independent validation at Addis")
    save(fig, "05_source_crossplots")

    fig, axs = plt.subplots(2, 1, figsize=PlotConfig.get("figsize"), layout="constrained")
    focus = ["AIRSpec", "VIBES", "uncapped_full"]
    s = scores.loc[scores.cohort.eq("full_pool")].set_index("method").loc[focus]
    names = ["Spline baseline\n4000–1425", "VIBES corrected full range\nuse 4000–1425", "VIBES corrected full range\nuse 4000–500"]
    axs[0].bar(range(3), s.RMSE, color=[COLORS[m] for m in focus], alpha=.8)
    axs[0].set_xticks(range(3), names)
    for i, row in enumerate(s.itertuples()):
        axs[0].text(i, row.RMSE+.04, f"RMSE {row.RMSE:.3f}\nPredictive R² {row.predictive_R2:.3f}", ha="center", fontsize=10)
    axs[0].set_ylim(0, 4.5)
    style_axes(axs[0], "Model spectral input (cm⁻¹)", "Held-out TOR EC RMSE (µg/filter)",
               "Same 10,066 calibration filters and 2,327 held-out filters", show_legend=False)
    ci = intervals.loc[intervals.cohort.eq("full_pool")].set_index("method").loc[["VIBES", "uncapped_full"]]
    delta = ci.delta_rmse.to_numpy()
    axs[1].errorbar(delta, [1, 0], xerr=np.array([delta-ci.ci_low, ci.ci_high-delta]), fmt="o", capsize=6, color="#d62728")
    axs[1].axvline(0, color="0.4", ls="--", lw=1)
    axs[1].set_yticks([1, 0], ["Use 4000–1425", "Use 4000–500"])
    axs[1].set_ylim(-.6, 1.6)
    style_axes(axs[1], "RMSE difference from spline baseline (µg/filter): negative favours VIBES", "",
               "Paired 95% bootstrap intervals: 32 held-out sites resampled, 2,000 repeats", show_legend=False)
    fig.suptitle("A cleaner baseline does not guarantee a better EC prediction\nRank-70 VIBES: spectral-window comparison on the frozen source-site split")
    save(fig, "06_main_result")

    fig, axs = plt.subplots(2, 1, figsize=PlotConfig.get("figsize"), layout="constrained")
    for ax, kind in zip(axs, ["calibration", "target"]):
        idx = np.flatnonzero(cases.kind.eq(kind))
        q = np.quantile(np.abs(v70[idx]-v30[idx]), [.25, .5, .75], axis=0)
        ax.plot(wn, q[1], color="#d62728", lw=1.2)
        ax.fill_between(wn, q[0], q[2], color="#d62728", alpha=.15)
        ax.set_xlim(4000, 500)
        style_axes(ax, "Wavenumber (cm⁻¹)", "Absolute rank-70 minus rank-30 difference (absorbance)",
                   f"{'IMPROVE source population' if kind=='calibration' else 'Addis test set'}: n={len(idx):,}", show_legend=False)
    fig.suptitle("Background rank sensitivity: paired median and interquartile range\nDifferences are small in aerosol spectra; full-pool full-window RMSE changes by <0.001 µg/filter")
    save(fig, "07_rank_sensitivity")

    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "n_cases": len(cases),
               "recovered_rows": rows.tolist(), "rank_sensitivity": spectral_records,
               "blank_median_rms": blanks.groupby("method").rms_absorbance.median().to_dict(),
               "injection_medians": injections.groupby(["method", "amplitude"])[["signal_area_recovery", "relative_rms_error"]].median().reset_index().to_dict("records"),
               "calibration_scores": scores.to_dict("records"), "paired_site_intervals": intervals.to_dict("records"),
               "scope": "Exploratory paired rank/window comparison. Same source-site split and factor-selection rule; test results did not tune preprocessing. Synthetic additions are not chemical standards. No Addis thermal reference is supplied. Exported spectra end at 500, not 400 cm-1.",
               "method_labels": LABELS, "source_hashes": hashes,
               "derived_array_sha256": sha(OUT / "corrected_VIBES_full_recovered.npy")}
    assert all(sha(ROOT / p)==h for p, h in hashes.items())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(scores.to_string(index=False), flush=True)
    print(intervals.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
