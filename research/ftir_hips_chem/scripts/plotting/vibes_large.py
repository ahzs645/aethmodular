"""Shared plotting primitives for the portable large VIBES comparison."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .overlays import crossplot_on_axes as _canonical_crossplot
from .utils import style_axes

COLORS = {"AIRSpec": "C0", "VIBES": "C1"}


def crossplot_on_axes(ax, x, y, *args, **kwargs):
    """Use the shared panel, extending its zero-based axes for negative values."""
    result = _canonical_crossplot(ax, x, y, *args, **kwargs)
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.any():
        xmin, ymin = x[valid].min(), y[valid].min()
        if kwargs.get("equal_axes", True) and min(xmin, ymin) < 0:
            upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
            lower = min(xmin, ymin) - 0.03 * (upper - min(xmin, ymin))
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
        else:
            if xmin < 0:
                ax.set_xlim(left=xmin - 0.03 * (ax.get_xlim()[1] - xmin))
            if ymin < 0:
                ax.set_ylim(bottom=ymin - 0.03 * (ax.get_ylim()[1] - ymin))
    return result


def large_run_figures(output):
    out = Path(output)
    scores = pd.read_csv(out / "calibration_scores.csv")
    predictions = pd.read_csv(out / "heldout_predictions.csv")
    cv = pd.read_csv(out / "cv_curves.csv")
    cohorts = list(scores.cohort.unique())
    fig, axs = plt.subplots(1, len(cohorts), figsize=(13, 4.3), squeeze=False)
    for ax, cohort in zip(axs[0], cohorts):
        for method, color in COLORS.items():
            d = cv[(cv.cohort == cohort) & (cv.method == method)]
            ax.plot(d.n_components, d.rmse_mean, color=color, label=method)
            ax.fill_between(
                d.n_components,
                d.rmse_mean - d.rmse_se,
                d.rmse_mean + d.rmse_se,
                color=color,
                alpha=0.15,
            )
            k = d.selected_k.iloc[0]
            ax.scatter([k], d.loc[d.n_components.eq(k), "rmse_mean"], color=color, s=65, zorder=3)
        style_axes(ax, "PLS components", "Training site-CV RMSE (µg/filter)", title=cohort)
    fig.tight_layout()
    yield "01_training_only_cv", fig

    fig, axs = plt.subplots(len(cohorts), 2, figsize=(12, 5 * len(cohorts)))
    for row, cohort in enumerate(cohorts):
        for col, (method, color) in enumerate(COLORS.items()):
            d = predictions[(predictions.cohort == cohort) & (predictions.method == method)]
            crossplot_on_axes(
                axs[row, col],
                d.y,
                d.prediction,
                "Held-out TOR EC (µg/filter)",
                "Predicted EC (µg/filter)",
                color=color,
            )
            score = scores[(scores.cohort == cohort) & (scores.method == method)].iloc[0]
            axs[row, col].set_title(
                f"{cohort} · {method} · predictive R²={score.predictive_R2:.3f}"
            )
    fig.suptitle(
        "Identical held-out rows · box R² is correlation; title R² is predictive\nDeming λ=1 is descriptive, not an uncertainty-calibrated fit",
        fontsize=11,
    )
    fig.tight_layout()
    yield "02_heldout_tor", fig

    intervals = pd.read_csv(out / "paired_site_bootstrap.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    pos = np.arange(len(cohorts))
    for j, (method, color) in enumerate(COLORS.items()):
        d = scores[scores.method.eq(method)].set_index("cohort").loc[cohorts]
        axs[0].bar(pos + (j - 0.5) * 0.32, d.RMSE, width=0.32, label=method, color=color)
    axs[0].set_xticks(pos, cohorts)
    style_axes(axs[0], "Cohort", "Held-out RMSE (µg/filter)")
    for i, r in intervals.iterrows():
        axs[1].plot([r.ci_low, r.ci_high], [i, i], color="C4", lw=3)
        axs[1].scatter(r.delta_rmse, i, color="C4")
    axs[1].axvline(0, color="black", ls="--")
    axs[1].set_yticks(pos, intervals.cohort)
    style_axes(axs[1], "RMSE(VIBES) − RMSE(AIRSpec), µg/filter", "", show_legend=False)
    axs[1].set_title("Paired site-bootstrap 95% interval; negative favors VIBES")
    fig.tight_layout()
    yield "03_paired_model_comparison", fig

    ext = pd.read_csv(out / "addis_predictions.csv")
    fig, axs = plt.subplots(len(cohorts), 2, figsize=(12, 5 * len(cohorts)))
    for row, cohort in enumerate(cohorts):
        for col, (method, color) in enumerate(COLORS.items()):
            d = ext[(ext.cohort == cohort) & (ext.method == method)]
            crossplot_on_axes(
                axs[row, col],
                d.HIPS_EC_equivalent,
                d.prediction_ugm3,
                "HIPS EC-equivalent (µg/m³)",
                "FTIR-predicted EC (µg/m³)",
                color=color,
                one_to_one=False,
                equal_axes=False,
            )
            axs[row, col].set_title(f"{cohort} · {method} · external Addis")
    fig.suptitle(
        "HIPS is an optical comparator, not independent thermal EC truth\nOLS describes the association; no slope/intercept acceptance rule is applied",
        fontsize=11,
    )
    fig.tight_layout()
    yield "04_addis_transfer", fig

    features = pd.read_csv(out / "spectral_features.csv")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, band in zip(axs, ["CH_peak", "carbonyl_peak", "shoulder_1600_peak"]):
        d = (
            features[features.kind.eq("target")]
            .pivot(index="sample_id", columns="method", values=band)
            .dropna()
        )
        crossplot_on_axes(
            ax,
            1000 * d.AIRSpec,
            1000 * d.VIBES,
            "AIRSpec (milli-absorbance)",
            "VIBES (milli-absorbance)",
            color="C1",
        )
        ax.set_title(band.replace("_", " "))
    fig.suptitle("Same Addis band features · descriptive Deming λ=1", fontsize=11)
    fig.tight_layout()
    yield "05_addis_band_features", fig

    blanks = pd.read_csv(out / "blank_metrics.csv")
    injections = pd.read_csv(out / "injection_metrics.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    b = blanks.pivot(
        index=["sample_id", "source"], columns="method", values="rms_from_zero"
    ).dropna()
    for j, source in enumerate(b.index.get_level_values("source").unique()):
        values = b.xs(source, level="source")
        for k, (method, color) in enumerate(COLORS.items()):
            x = j + (k - 0.5) * 0.3
            axs[0].scatter(
                np.full(len(values), x),
                values[method],
                color=color,
                alpha=0.35,
                s=12,
                label=method if j == 0 else None,
            )
            axs[0].plot([x - 0.08, x + 0.08], [values[method].median()] * 2, color=color, lw=3)
    axs[0].set_xticks(
        range(b.index.get_level_values("source").nunique()),
        b.index.get_level_values("source").unique(),
    )
    axs[0].set_yscale("log")
    style_axes(axs[0], "Held-out blank source", "Residual RMS (absorbance)")
    inj = injections.pivot(
        index=["sample_id", "amplitude"], columns="method", values="recovery_rmse"
    ).dropna()
    for method, color in COLORS.items():
        grouped = inj[method].groupby(level="amplitude")
        axs[1].plot(grouped.median().index, grouped.median(), "o-", label=method, color=color)
        axs[1].fill_between(
            grouped.median().index,
            grouped.quantile(0.25),
            grouped.quantile(0.75),
            color=color,
            alpha=0.15,
        )
    style_axes(axs[1], "Added peak amplitude", "Incremental recovery RMSE (absorbance)")
    axs[1].set_title("Known additions on independent ETAD blanks")
    fig.tight_layout()
    yield "06_blanks_and_recovery", fig

    di = pd.read_csv(out / "fit_diagnostics.csv")
    run = json.loads((out / "RUN_MANIFEST.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    times = di[di.kind.eq("calibration")][["airspec_seconds", "seconds"]].median() * 1000
    axs[0].bar(["AIRSpec", "VIBES"], times, color=list(COLORS.values()))
    style_axes(axs[0], "Method", "Median worker time per library filter (ms)", show_legend=False)
    g = di.groupby("kind").success.agg(["sum", "count"])
    axs[1].bar(g.index, g["count"], color="0.8", label="Attempted")
    axs[1].bar(g.index, g["sum"], color="C2", label="Successful")
    style_axes(axs[1], "Evaluation set", "Fits")
    axs[1].set_title(f"{run['n_retried']} logged retries; {run['n_failed']} failures")
    fig.tight_layout()
    yield "07_runtime_and_coverage", fig

    with np.load(out / "spectral_examples.npz") as z:
        fig, axs = plt.subplots(3, 3, figsize=(15, 11))
        for row in range(3):
            wn = z["wn"]
            axs[row, 0].plot(wn, z["raw"][row], color="0.5", label="Raw")
            for method, color in COLORS.items():
                values = z[method.lower()][row]
                axs[row, 0].plot(
                    wn, z["raw"][row] - values, color=color, label=method + " baseline"
                )
                axs[row, 1].plot(wn, values, color=color, label=method)
                axs[row, 2].plot(wn, values, color=color, label=method)
            for col, ax in enumerate(axs[row]):
                style_axes(ax, "Wavenumber (cm⁻¹)", "Absorbance", show_legend=False)
                ax.set_xlim((1900, 1450) if col == 2 else (4000, 1425))
                ax.axhline(0, color="0.7", lw=0.7)
            axs[row, 0].set_title(f"{z['ids'][row]} · RMS difference q{[10, 50, 90][row]}")
            axs[row, 1].set_title("Corrected spectra")
            axs[row, 2].set_title("Carbonyl / 1600 cm⁻¹ region")
        for ax in axs[0]:
            ax.legend(fontsize=8)
        fig.tight_layout()
        yield "08_matched_spectral_overlays", fig
