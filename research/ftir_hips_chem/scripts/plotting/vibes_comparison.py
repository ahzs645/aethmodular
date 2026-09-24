"""Figures for paired blank-trained VIBES / AIRSpec baseline assessment."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .overlays import crossplot_on_axes
from .utils import style_axes as _style_axes


def style_axes(ax, xlabel, ylabel):
    _style_axes(ax, xlabel, ylabel, show_legend=False)


METHODS = {"AIRSpec": "C0", "VIBES": "C1"}
BANDS = ("CH_peak", "carbonyl_peak", "shoulder_1600_peak")


def _spectral(ax, wn, values, label, color):
    q25, med, q75 = np.nanpercentile(values, [25, 50, 75], axis=0)
    ax.plot(wn, med, color=color, label=label)
    ax.fill_between(wn, q25, q75, color=color, alpha=0.15)


def _spectrum_axes(ax, ylabel="Absorbance"):
    style_axes(ax, "Wavenumber (cm⁻¹)", ylabel)
    ax.set_xlim(4000, 1425)
    ax.axhline(0, color="0.6", lw=0.7)


def comparison_figures(table_dir):
    """Yield named figures so notebooks can display/save one section at a time."""
    folder = Path(table_dir)
    z = np.load(folder / "comparison_arrays.npz")
    wn = z["wn"]
    m = pd.read_csv(folder / "paired_sample_metrics.csv")
    cv = pd.read_csv(folder / "blank_rank_cv.csv")
    manifest = json.loads((folder / "run_manifest.json").read_text())
    ledger = pd.read_csv(folder / "inclusion_and_blank_split.csv")
    valid = m.paired_valid.to_numpy(bool)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    _spectral(axs[0], wn, z["blank_train"], "Training blanks", "C2")
    _spectral(axs[0], wn, z["blank_raw"], "Held-out blanks", "C3")
    _spectrum_axes(axs[0])
    axs[0].legend()
    axs[0].set_title("Field-blank spectra: median and interquartile range")
    axs[1].errorbar(cv.components, cv.mean_mse, yerr=cv.se_mse, fmt="o-", ms=4)
    axs[1].axvline(manifest["vibes_components"], color="C1", ls="--", label="Selected rank")
    axs[1].set_yscale("log")
    axs[1].legend()
    style_axes(axs[1], "PCA components", "Blank-only LOO reconstruction MSE")
    axs[1].set_title("Upstream one-standard-error selection")
    fig.tight_layout()
    yield "01_blank_model", fig

    ordered = np.flatnonzero(valid)[np.argsort(m.loc[valid, "rms_difference"].to_numpy())]
    representatives = ordered[np.round(np.array([0.1, 0.5, 0.9]) * (len(ordered) - 1)).astype(int)]
    fig, axs = plt.subplots(3, 3, figsize=(15, 11), sharex="col")
    for row, idx in enumerate(representatives):
        axs[row, 0].plot(wn, z["raw"][idx], color="0.5", label="Raw")
        for method, color in METHODS.items():
            vals = z[method.lower()][idx]
            axs[row, 0].plot(wn, z["raw"][idx] - vals, color=color, label=method + " baseline")
            axs[row, 1].plot(wn, vals, color=color, label=method)
            axs[row, 2].plot(wn, vals, color=color, label=method)
        for ax in axs[row]:
            _spectrum_axes(ax)
        axs[row, 2].set_xlim(1900, 1450)
        axs[row, 0].set_title(f"{m.ExternalFilterId.iloc[idx]} · difference q{[10, 50, 90][row]}")
        axs[row, 1].set_title("Corrected: identical input spectrum")
        axs[row, 2].set_title("Carbonyl / 1600 cm⁻¹ region")
    for ax in axs[0]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    yield "02_matched_examples", fig

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for method, color in METHODS.items():
        _spectral(axs[0], wn, z[method.lower()][valid], method, color)
    _spectral(axs[1], wn, (z["vibes"] - z["airspec"])[valid], "VIBES − AIRSpec", "C4")
    for ax in axs:
        _spectrum_axes(ax, "Corrected absorbance")
        ax.legend()
    axs[0].set_title(f"{valid.sum()} paired filters: median and interquartile range")
    axs[1].set_title("Difference by wavenumber (not error against truth)")
    fig.tight_layout()
    yield "03_population_spectra", fig

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, band in zip(axs, BANDS):
        crossplot_on_axes(
            ax,
            1000 * m.loc[valid, "AIRSpec_" + band],
            1000 * m.loc[valid, "VIBES_" + band],
            "AIRSpec height (milli-absorbance)",
            "VIBES height (milli-absorbance)",
            color="C1",
        )
        ax.set_title(band.replace("_", " "))
    fig.suptitle(
        "Identical local-continuum band metrics · OLS + Deming (λ=1, descriptive)", fontsize=12
    )
    fig.tight_layout()
    yield "04_band_comparison", fig

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    axs[0].hist(m.loc[valid, "rms_difference"], bins=22, color="C4", alpha=0.8)
    style_axes(axs[0], "RMS(VIBES − AIRSpec), absorbance", "Filters")
    axs[0].set_title("Whole-spectrum disagreement")
    axs[1].hist(m.loc[valid, "spectral_correlation_R2"], bins=22, color="C2", alpha=0.8)
    style_axes(axs[1], "Squared spectral correlation R²", "Filters")
    axs[1].set_title("Shape agreement, not predictive R²")
    groups = list(m.loc[valid].groupby(m.loc[valid, "LotId"].fillna("Unknown"), sort=False))
    axs[2].boxplot([g.rms_difference for _, g in groups], tick_labels=[str(k) for k, _ in groups])
    style_axes(axs[2], "Filter lot (unknown kept)", "RMS difference, absorbance")
    axs[2].set_title("Background-domain diagnostic")
    fig.tight_layout()
    yield "05_agreement_and_lots", fig

    blanks = pd.read_csv(folder / "heldout_blank_metrics.csv")
    inj = pd.read_csv(folder / "injection_metrics.csv")
    # Identical valid pairs for both methods, including the same test blanks.
    blank_paired = blanks.pivot(
        index="sample_id", columns="method", values="rms_from_zero"
    ).dropna()
    inj_paired = inj.pivot(
        index=["sample_id", "added_amplitude"], columns="method", values="increment_rmse"
    ).dropna()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for j, (method, color) in enumerate(METHODS.items()):
        axs[0].scatter(np.full(len(blank_paired), j), blank_paired[method], color=color)
        axs[0].plot([j - 0.15, j + 0.15], [blank_paired[method].median()] * 2, color=color, lw=3)
        grouped = inj_paired[method].groupby(level="added_amplitude")
        axs[1].plot(grouped.median().index, grouped.median(), "o-", label=method, color=color)
        axs[1].fill_between(
            grouped.median().index,
            grouped.quantile(0.25),
            grouped.quantile(0.75),
            color=color,
            alpha=0.15,
        )
        _spectral(
            axs[2],
            wn,
            z[method.lower() + "_recovered"][len(z["blank_ids"]) : 2 * len(z["blank_ids"])],
            method,
            color,
        )
    axs[0].set_xticks([0, 1], list(METHODS))
    style_axes(axs[0], "Method", "Held-out blank RMS from zero")
    axs[0].set_title(f"{len(blank_paired)} independent field blanks")
    style_axes(axs[1], "Added peak amplitude (absorbance)", "Increment recovery RMSE")
    axs[1].set_title("Known additions: median and IQR")
    axs[1].legend()
    axs[2].plot(
        wn,
        z["injection_truth"][len(z["blank_ids"])],
        color="black",
        ls="--",
        label="Known addition",
    )
    _spectrum_axes(axs[2], "Recovered added absorbance")
    axs[2].set_title("0.05-amplitude addition")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    yield "06_independent_checks", fig

    timing = pd.read_csv(folder / "timing.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    t = timing[timing.stage.eq("samples")]
    axs[0].bar(t.method, t.ms_per_spectrum, color=list(METHODS.values()))
    style_axes(axs[0], "Method", "ms / real sample (serial)")
    axs[0].set_title("Correction time; excludes one-time blank training")
    for j, value in enumerate(t.ms_per_spectrum):
        axs[0].text(j, value, f"{value:.1f}", ha="center", va="bottom")
    axs[0].margins(y=0.2)
    di = pd.read_csv(folder / "vibes_fit_diagnostics.csv")
    groups = di.groupby("role").success.agg(["sum", "count"])
    axs[1].bar(groups.index, groups["count"], color="0.8", label="Attempted")
    axs[1].bar(groups.index, groups["sum"], color="C2", label="Converged + finite")
    style_axes(axs[1], "Evaluation set", "Fits")
    axs[1].legend()
    axs[1].set_title(f"Blank PCA + LOO: {timing.iloc[2].seconds:.2f} s")
    fig.tight_layout()
    yield "07_runtime_and_convergence", fig
