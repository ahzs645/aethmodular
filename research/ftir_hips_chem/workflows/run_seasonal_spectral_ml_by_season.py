"""Season-specific held-out reliance of the existing pooled spectral RF models.

The two fixed models train on the same 91 earlier physical filters as the main
screen. Joint interval permutations are evaluated separately within each season
of the complete 97-filter test cycle. No seasonal model, tuning or ablation is
introduced. Outputs describe predictive reliance, not chemical source fractions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "research/ftir_hips_chem/output/tables"
MODEL_IDS = ("rf_raw_full", "rf_spline_common")
CALENDAR = "belg_feb"


def fingerprint(paths):
    return {name: {"filename": path.name,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in paths.items()}


def run(args):
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    code_paths = {
        "addon": Path(__file__).resolve(),
        "main_workflow": ROOT / "research/ftir_hips_chem/workflows/run_seasonal_spectral_ml.py",
        "core": ROOT / "research/ftir_hips_chem/scripts/seasonal_spectral_ml.py",
        "calendar_config": ROOT / "research/ftir_hips_chem/scripts/config.py",
    }
    paths = {**code_paths, "ledger": args.ledger, "arrays": args.arrays,
             "boundaries": args.boundaries, "weather": args.weather}
    hashes_start = fingerprint(paths)
    snapshot_dir = out / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    for name, path in code_paths.items():
        (snapshot_dir / f"{name}.py").write_bytes(path.read_bytes())

    # Import only after recording the source/input snapshot. The main runner is
    # not executed; its data preparation and model specifications are reused.
    sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/workflows"))
    from run_seasonal_spectral_ml import (
        NAMED_REGIONS, SEED, feature_input, load_inputs, log, model_specs, write_json,
    )
    from seasonal_spectral_ml import (
        build_model, evaluate_folds, grouped_region_permutation, make_cycle_folds,
        prediction_loss, spectral_regions,
    )
    from config import resolve_seasons

    if fingerprint(paths) != hashes_start:
        raise RuntimeError("Source/input changed during import; start a fresh run")
    write_json(out / "receipt_start.json", {"started_at": started_at, "hashes_start": hashes_start})
    cohort, representations, support, _ = load_inputs(args)
    folds, _ = make_cycle_folds(
        cohort.date.to_numpy(), cohort.id.to_numpy(), convention=CALENDAR,
        mode="forward", train_coverage="all_seasons", test_coverage="all_seasons",
        min_train=50, min_test=20,
    )
    primary = [fold for fold in folds if fold.test_cycle == 2024 and fold.complete_test]
    if len(primary) != 1:
        raise ValueError("Exactly one complete primary cycle ending2024 is required")
    fold = primary[0]
    if (len(cohort), len(fold.train), len(fold.test)) != (232, 91, 97):
        raise ValueError("Frozen232/91/97 cohort changed; do not silently substitute rows")
    seasons = list(resolve_seasons(CALENDAR))
    season_counts = [int(cohort.iloc[fold.test].season.eq(label).sum()) for label in seasons]
    if season_counts != [37, 34, 26]:
        raise ValueError("Expected held-out Bega/Belg/Kiremt counts37/34/26")
    y = cohort.optical_disagreement_ugm3.to_numpy(float)
    specs = {spec["id"]: spec for spec in model_specs()}
    importance, repeats, performance, predictions, pooled = [], [], [], [], []
    for model_id in MODEL_IDS:
        spec = specs[model_id]
        if spec["family"] != "rf" or spec["covariates"] != "none":
            raise ValueError("This add-on requires the unchanged spectrum-only RF specifications")
        X, columns, numeric, categorical = feature_input(cohort, representations, spec)
        estimator = build_model(
            task="regression", model="rf", spectral_columns=columns,
            numeric_metadata=numeric, categorical_metadata=categorical,
            n_components=8, seed=SEED, n_jobs=args.jobs,
        )
        log(f"By-season pooled fit: {model_id}")
        result = evaluate_folds(X, y, cohort.id.to_numpy(), [fold], estimator, task="regression")
        fitted = result["fitted_folds"][fold.name]["estimator"]
        pooled.extend({"model_id": model_id, **row} for row in result["metrics"].to_dict("records"))
        wn = representations[spec["representation"]][1]
        regions = spectral_regions(wn, columns, bin_width=100,
                                   named_regions=NAMED_REGIONS, include_whole=True)
        column_index = {column: index for index, column in enumerate(columns)}
        for season in seasons:
            indices = fold.test[cohort.iloc[fold.test].season.eq(season).to_numpy()]
            X_test, y_test = X.iloc[indices], y[indices]
            predicted = fitted.predict(X_test)
            prefix = {"task": "disagreement", "calendar": CALENDAR, "cohort": "direct232",
                      "model_id": model_id, "fold_id": fold.name, "season": season}
            baseline = prediction_loss(y_test, predicted, "regression")
            performance.append({**prefix, "n_train": len(fold.train), "n_test": len(indices),
                                "complete_test": True, "rmse": baseline,
                                "mae": float(mean_absolute_error(y_test, predicted)),
                                "r2": float(r2_score(y_test, predicted)),
                                "bias": float(np.mean(predicted-y_test)),
                                "dummy_rmse": prediction_loss(
                                    y_test, np.repeat(y[fold.train].mean(), len(indices)), "regression")})
            for index, truth, pred in zip(indices, y_test, predicted):
                predictions.append({**prefix, "id": cohort.id.iloc[index],
                                    "date": cohort.date.iloc[index].date().isoformat(),
                                    "actual": float(truth), "predicted": float(pred)})
            log(f"By-season permutations: {model_id} / {season} / n={len(indices)}")
            detail = grouped_region_permutation(
                fitted, X_test, y_test, regions, task="regression",
                repeats=args.permutations, seed=SEED,
            )
            repeats.extend({**prefix, "scheme": "within_season", **row}
                           for row in detail.to_dict("records"))
            for region_id, group in detail.groupby("region", sort=False):
                channel_idx = [column_index[column] for column in regions[region_id]]
                values = group.importance.to_numpy(float)
                importance.append({
                    **prefix, "kind": "permutation", "scheme": "within_season",
                    "region_id": region_id, "region_label": region_id,
                    "low": float(wn[channel_idx].min()), "high": float(wn[channel_idx].max()),
                    "n_channels": len(channel_idx), "n_test": len(indices),
                    "delta_mean": float(values.mean()),
                    "delta_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.,
                    "q05": float(np.quantile(values, .05)), "q95": float(np.quantile(values, .95)),
                    "repeats": len(values), "baseline_loss": baseline, "metric": "rmse",
                    "moved_fraction": float(group.moved_fraction.mean()),
                })
    pd.DataFrame(repeats).to_csv(out / "permutation_repeats.csv", index=False)
    pd.DataFrame(importance).to_csv(out / "region_importance.csv", index=False)
    pd.DataFrame(performance).to_csv(out / "per_season_performance.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "heldout_predictions.csv", index=False)
    pd.DataFrame(pooled).to_csv(out / "pooled_prediction_checks.csv", index=False)
    hashes_end = fingerprint(paths)
    receipt = {
        "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
        "hashes_start": hashes_start, "hashes_end": hashes_end,
        "source_and_inputs_unchanged": hashes_start == hashes_end,
        "source_snapshot": {name: f"source_snapshot/{name}.py" for name in code_paths},
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "elapsed_seconds": time.monotonic()-started,
        "settings": {"model_ids": MODEL_IDS, "calendar": CALENDAR, "fold_id": fold.name,
                     "n_source": len(cohort), "n_train": len(fold.train), "n_test": len(fold.test),
                     "seed": SEED, "permutations": args.permutations, "n_jobs": args.jobs,
                     "support": support, "fit": "Pooled91trainingfilters; no seasonal refits or tuning",
                     "permutation": "Joint channels shuffled only among filters of the selected held-out season"},
    }
    data = {
        "schema_version": 1, "generated_at": receipt["completed_at"],
        "counts": {"source_filters": len(cohort), "training_filters": len(fold.train),
                   "test_filters": len(fold.test), "season_counts": dict(zip(seasons, season_counts))},
        "importance": importance, "performance": performance, "pooled_checks": pooled,
        "provenance": receipt,
        "limitations": [
            "Each model trains once on pooled earlier seasons; these are season-specific held-out diagnostics, not separately fitted seasonal models.",
            "The target is FTIR EC minus HIPS; dependence on spectrum-derived EC remains mathematical coupling, not independent EC validation.",
            "Shuffling stays within the selected season but does not condition on loading, lot or correlated intervals.",
            "Absolute RMSE changes depend on season-specific target variation and model error; differences are descriptive, not significance tests between seasons.",
            "Ten-repeat spread is Monte Carlo variability, not a population confidence interval. Negative importance is retained.",
            "Interval names are candidate spectral neighborhoods, not chemical species, causal sources or mass fractions.",
        ],
    }
    write_json(out / "receipt.json", receipt)
    write_json(out / "by_season.json", data)
    if hashes_start != hashes_end:
        raise RuntimeError("Source/input changed during computation; inspect snapshots before merging output")
    log(f"By-season complete: {len(importance)} interval summaries, {len(repeats)} repeat rows")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=TABLES / "baseline_review_20261102/filter_diagnostic_ledger.csv")
    parser.add_argument("--arrays", type=Path, default=TABLES / "baseline_existing_data_20261103/native_method_arrays.npz")
    parser.add_argument("--boundaries", type=Path, default=TABLES / "baseline_existing_data_20261103/boundary_sensitivity.csv")
    parser.add_argument("--weather", type=Path, default=ROOT / "gallery/app/public/data/research/addis_met/filters.json")
    parser.add_argument("--output", type=Path, default=TABLES / "seasonal_spectral_ml_by_season_20261103")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
