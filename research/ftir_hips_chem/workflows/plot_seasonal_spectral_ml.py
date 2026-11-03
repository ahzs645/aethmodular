"""Export the fixed seasonal-ML comparisons using the shared plotting package.

The gallery reads the complete saved results. These figures are compact views
of the same complete-cycle results, not independently fitted models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from plotting import PlotConfig, comparisons  # noqa: E402

REP_LABELS = {
    "raw_full": "Raw\n500–4000",
    "raw_common": "Raw\n1560–4000",
    "spline_common": "Spline corrected\n1560–4000",
    "removed_common": "Removed component\n1560–4000",
    "derivative_common": "Second derivative\n1560–4000",
}


def save_bars(values, metric, title, ylabel, footer, output, name, *, reference=None, limits=None):
    comparisons.summary_bars(values, metric=metric)
    fig = plt.gcf()
    ax = fig.axes[0]
    fig.set_size_inches(13, 6.2)
    ax.set_title(title, pad=18)
    ax.set_ylabel(ylabel)
    # The shared helper labels positive bars. Replace with signed labels for
    # these diagnostics, retaining negative importance instead of truncating it.
    for label in list(ax.texts):
        label.remove()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    if reference is not None:
        ax.axhline(reference[0], color="black", linestyle="--", linewidth=1, label=reference[1])
    ax.axhline(0, color="black", linewidth=.7)
    if limits is not None:
        ax.set_ylim(*limits)
    else:
        low, high = ax.get_ylim()
        span = max(high-low, .05)
        ax.set_ylim(min(0, low)-span*.08, high+span*.15)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    if len(values) > 8:
        ax.tick_params(axis="x", labelsize=9)
    fig.text(.5, .025, footer, ha="center", va="bottom", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, .10, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "research/ftir_hips_chem/output/tables/seasonal_spectral_ml_20261103/seasonal_ml.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research/ftir_hips_chem/output/figures/seasonal_spectral_ml_20261103")
    parser.add_argument("--by-season", type=Path, default=ROOT / "research/ftir_hips_chem/output/tables/seasonal_spectral_ml_by_season_20261103/by_season.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.input.read_text())
    predictions = pd.DataFrame(data["predictions"])
    predictions = predictions.loc[predictions.calendar.eq("belg_feb")]
    paired_qc = []
    for task in ("season", "disagreement"):
        for model in ("rf_raw_full", "rf_spline_common"):
            for fold in ("forward_2024", "forward_2025"):
                base = predictions.loc[predictions.task.eq(task) & predictions.model_id.eq(model)
                                       & predictions.fold_id.eq(fold)].set_index("id")
                qc = predictions.loc[predictions.task.eq(task) & predictions.model_id.eq(model+"_scan_qc")
                                     & predictions.fold_id.eq(fold)].set_index("id")
                if qc.empty:
                    continue
                base = base.loc[qc.index]
                for label, rows in (("original_fit_same_test_ids", base), ("scan_qc_refit", qc)):
                    score = (balanced_accuracy_score(rows.actual, rows.predicted) if task == "season"
                             else mean_squared_error(rows.actual.astype(float), rows.predicted.astype(float))**.5)
                    paired_qc.append({"task": task, "model_id": model, "fold_id": fold,
                                      "comparison": label, "n_test": len(rows),
                                      "metric": "balanced_accuracy" if task == "season" else "rmse", "value": score})
    pd.DataFrame(paired_qc).to_csv(args.input.parent / "scan_qc_matched_scores.csv", index=False)
    candidates = predictions.loc[predictions.task.eq("disagreement") & predictions.fold_id.eq("forward_2024")
                                 & predictions.model_id.isin(["rf_raw_full", "rf_spline_common"])]
    candidates = candidates.pivot(index=["id", "date", "season", "actual"], columns="model_id", values="predicted").reset_index()
    candidates["raw_minus_corrected_prediction"] = candidates.rf_raw_full.astype(float)-candidates.rf_spline_common.astype(float)
    candidates["absolute_prediction_difference"] = candidates.raw_minus_corrected_prediction.abs()
    candidates = candidates.sort_values("absolute_prediction_difference", ascending=False).groupby("season", sort=False).head(2)
    candidates["selection_rule"] = "Two largest raw-versus-corrected RF prediction differences within each held-out season; examples, not an exclusion rule"
    candidates.to_csv(args.input.parent / "priority_filter_examples.csv", index=False)
    scores = pd.DataFrame(data["scores"])
    scores = scores.loc[scores.calendar.eq("belg_feb") & scores.cohort.eq("direct232")
                        & scores.fold_id.eq("forward_2024")]
    PlotConfig.set(sites="all", layout="individual", figsize=(13, 6.2), equal_axes=False)
    footer = "Addis • 91 earlier training filters → 97 held-out filters, Oct 2023–Sep 2024 • Bega Oct–Jan / Belg Feb–May / Kiremt Jun–Sep"
    for task, metric, title, ylabel, name, ref, limits in (
        ("season", "balanced_accuracy", "Corrected spectra identify seasons better in the complete-year test",
         "Balanced accuracy · mean recall across three seasons", "season_classification",
         (1/3, "Training-only reference = 1/3"), (0, 1.03)),
        ("disagreement", "rmse", "Raw spectra predict disagreement better in the complete-year test",
         "Held-out RMSE of FTIR EC − HIPS-derived BC (µg/m³)", "disagreement_prediction",
         None, (0, 1.03)),
    ):
        subset = scores.loc[scores.task.eq(task)].set_index("model_id")
        linear_label = "PCA + logistic regression" if task == "season" else "PCA + ridge regression"
        values = {label: {"Random forest": {metric: float(subset.at["rf_"+rep, metric])},
                          linear_label: {metric: float(subset.at["linear_"+rep, metric])}}
                  for rep, label in REP_LABELS.items()}
        if task == "disagreement":
            ref = (float(subset.at["linear_metadata", "rmse"]), "HIPS, PM₂.₅, volume and lot benchmark")
        save_bars(values, metric, title, ylabel, footer, args.output, name, reference=ref, limits=limits)

    importance = pd.DataFrame(data["importance"])
    importance = importance.loc[importance.fold_id.eq("forward_2024")
                                & importance.scheme.eq("across_filters")
                                & ~importance.region_id.str.startswith("bin_")
                                & importance.region_id.ne("whole_spectrum")]
    for task, model_id, title, ylabel, name in (
        ("season", "rf_spline_common", "Which corrected-spectrum regions support season recognition?",
         "Increase in balanced classification error", "season_region_tests"),
        ("disagreement", "rf_raw_full", "Which raw-spectrum regions support disagreement prediction?",
         "Increase in disagreement RMSE (µg/m³)", "disagreement_region_tests"),
    ):
        subset = importance.loc[importance.task.eq(task) & importance.model_id.eq(model_id)]
        ordered = subset.drop_duplicates("region_id").sort_values("low")
        values = {}
        for region_id in ordered.region_id:
            rows = subset.loc[subset.region_id.eq(region_id)].set_index("kind")
            label = region_id.split(" · ")[0].replace("–", "–\n", 1)
            values[label] = {
                "Shuffle interval · 10-repeat mean": {"delta": float(rows.at["permutation", "delta_mean"])},
                "Remove interval + refit": {"delta": float(rows.at["ablation", "delta_mean"])},
            }
        save_bars(values, "delta", title, ylabel,
                  "Same 97-filter test • Positive = worse prediction; negative = improvement • Spectral intervals (cm⁻¹) are not chemical mass fractions",
                  args.output, name)
    if args.by_season.is_file():
        seasonal = pd.DataFrame(json.loads(args.by_season.read_text())["performance"])
        values = {}
        for season in ("Bega", "Belg", "Kiremt"):
            rows = seasonal.loc[seasonal.season.str.startswith(season+" (")].set_index("model_id")
            n = int(rows.n_test.iloc[0])
            values[f"{season}\nn = {n}"] = {
                "RF · raw full spectrum": {"rmse": float(rows.at["rf_raw_full", "rmse"])},
                "RF · spline corrected ≥1560": {"rmse": float(rows.at["rf_spline_common", "rmse"])},
            }
        save_bars(values, "rmse", "Disagreement prediction behaves differently across the three seasons",
                  "Held-out disagreement RMSE (µg/m³)",
                  "Same pooled 91-filter training set • Separate subsets of the complete 97-filter test • Lower = better prediction, not greater EC accuracy",
                  args.output, "season_specific_prediction", limits=(0, 1.03))
    print(json.dumps({"output": str(args.output), "figures": sorted(p.name for p in args.output.glob("*.png"))}))


if __name__ == "__main__":
    main()
