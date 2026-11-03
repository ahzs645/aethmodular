"""Fixed-model sensitivity to omitting declared AIRSpec transition neighborhoods.

Masks +/-20 cm^-1 around1820,2000,2220 in both raw and signed corrected common
windows. No baseline is changed, no model is tuned, and no importance is fitted.
Adjacent-channel differences are descriptive, not evidence of artifact causation.
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

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "research/ftir_hips_chem/output/tables"
TRANSITIONS = (1820., 2000., 2220.)
HALF_WIDTH = 20.
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
        "airspec": ROOT / "research/ftir_ec_phase3/scripts/airspec_baseline.py",
    }
    paths = {**code_paths, "ledger": args.ledger, "arrays": args.arrays,
             "boundaries": args.boundaries, "weather": args.weather}
    hashes_start = fingerprint(paths)
    snapshot_dir = out / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    for name, path in code_paths.items():
        (snapshot_dir / f"{name}.py").write_bytes(path.read_bytes())
    sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/workflows"))
    from run_seasonal_spectral_ml import (
        SEED, feature_input, load_inputs, log, model_specs, write_json,
    )
    from seasonal_spectral_ml import build_model, evaluate_folds, make_cycle_folds

    if fingerprint(paths) != hashes_start:
        raise RuntimeError("Source/input changed during import; start a fresh run")
    write_json(out / "receipt_start.json", {"started_at": started_at, "hashes_start": hashes_start})
    cohort, representations, support, _ = load_inputs(args)
    folds, _ = make_cycle_folds(
        cohort.date.to_numpy(), cohort.id.to_numpy(), convention=CALENDAR,
        mode="forward", train_coverage="all_seasons", test_coverage="all_seasons",
        min_train=50, min_test=20,
    )
    if len(cohort) != 232 or [(f.test_cycle, len(f.train), len(f.test)) for f in folds] != [
        (2024, 91, 97), (2025, 188, 30),
    ]:
        raise ValueError("Frozen physical cohort or forward folds changed")
    source_specs = {spec["id"]: spec for spec in model_specs()}
    scores, predictions, models, rep_specs, mask_rows = [], [], [], [], []
    for source_rep in ("raw_common", "spline_common"):
        matrix, wn, label = representations[source_rep]
        remove = np.any(np.abs(wn[:, None]-np.asarray(TRANSITIONS)[None, :]) <= HALF_WIDTH, axis=1)
        masked_rep = source_rep + "_transitions_masked"
        masked_id = "rf_" + masked_rep
        masked_label = label + " · transition windows omitted"
        representations[masked_rep] = (matrix[:, ~remove], wn[~remove], masked_label)
        spec = {**source_specs["rf_"+source_rep], "id": masked_id,
                "label": "Random forest · "+masked_label, "representation": masked_rep,
                "sensitivity": True}
        models.append(spec)
        rep_specs.append({"id": masked_rep, "label": masked_label,
                          "n_channels": int((~remove).sum()), "low": float(wn[~remove].min()),
                          "high": float(wn[~remove].max()), "source_representation": source_rep,
                          "n_removed": int(remove.sum()), "mask_half_width_cm_1": HALF_WIDTH,
                          "transition_centers_cm_1": TRANSITIONS})
        for center in TRANSITIONS:
            selected = np.abs(wn-center) <= HALF_WIDTH
            mask_rows.append({"source_representation": source_rep, "center_cm_1": center,
                              "nominal_low": center-HALF_WIDTH, "nominal_high": center+HALF_WIDTH,
                              "actual_low": float(wn[selected].min()), "actual_high": float(wn[selected].max()),
                              "n_removed": int(selected.sum())})
        for active_spec, role in ((source_specs["rf_"+source_rep], "unmasked_verification"),
                                  (spec, "transition_masked")):
            X, columns, numeric, categorical = feature_input(cohort, representations, active_spec)
            for task, core_task in (("season", "classification"), ("disagreement", "regression")):
                target = cohort.season.to_numpy() if task == "season" else cohort.optical_disagreement_ugm3.to_numpy()
                estimator = build_model(task=core_task, model="rf", spectral_columns=columns,
                                        numeric_metadata=numeric, categorical_metadata=categorical,
                                        n_components=8, seed=SEED, n_jobs=args.jobs)
                log(f"Transition check: {task} / {active_spec['id']}")
                result = evaluate_folds(X, target, cohort.id.to_numpy(), folds, estimator, task=core_task)
                prefix = {"task": task, "calendar": CALENDAR, "cohort": "direct232",
                          "model_id": active_spec["id"], "role": role}
                for row in result["metrics"].to_dict("records"):
                    row["fold_id"] = row.pop("fold")
                    scores.append({**prefix, **row})
                by_id = cohort.set_index("id")
                for row in result["predictions"].to_dict("records"):
                    identifier = row.pop("physical_id")
                    row["fold_id"] = row.pop("fold")
                    row["actual"] = row.pop("truth")
                    row["predicted"] = row.pop("prediction")
                    predictions.append({**prefix, **row, "id": identifier,
                                        "date": by_id.at[identifier, "date"].date().isoformat(),
                                        "season": by_id.at[identifier, "season"]})

    # Retain signed adjacent-channel differences with an explicit orientation.
    wn = representations["raw_common"][1]
    raw = representations["raw_common"][0]
    corrected = representations["spline_common"][0]
    roles = np.repeat("not_modeled", len(cohort)).astype(object)
    roles[folds[0].train] = "first_cycle_training"
    roles[folds[0].test] = "primary_test"
    roles[folds[1].test] = "secondary_test"
    join_rows = []
    for center in (1820., 2000.):
        lower = int(np.flatnonzero(wn < center)[0])
        upper = int(np.flatnonzero(wn > center)[-1])
        for index, row in cohort.iterrows():
            raw_difference = float(raw[index, upper]-raw[index, lower])
            corrected_difference = float(corrected[index, upper]-corrected[index, lower])
            join_rows.append({"id": row.id, "date": row.date.date().isoformat(), "season": row.season,
                              "lot_group": row.lot_group, "evaluation_role": roles[index],
                              "transition_cm_1": center, "low_wavenumber": float(wn[lower]),
                              "high_wavenumber": float(wn[upper]), "spacing_cm_1": float(wn[upper]-wn[lower]),
                              "raw_high_minus_low": raw_difference,
                              "spline_high_minus_low": corrected_difference,
                              "removed_high_minus_low": raw_difference-corrected_difference})
    joins = pd.DataFrame(join_rows)
    join_summaries = []
    for scope, frame in (("all_direct232", joins), ("primary_test97", joins.loc[joins.evaluation_role.eq("primary_test")])):
        for (center, season), group in frame.groupby(["transition_cm_1", "season"], sort=False):
            item = {"scope": scope, "transition_cm_1": center, "season": season, "n_filters": len(group)}
            for column in ("raw_high_minus_low", "spline_high_minus_low", "removed_high_minus_low"):
                item.update({column+"_median": float(group[column].median()),
                             column+"_q25": float(group[column].quantile(.25)),
                             column+"_q75": float(group[column].quantile(.75))})
            join_summaries.append(item)
    pd.DataFrame(scores).to_csv(out / "scores.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(mask_rows).to_csv(out / "masked_channels.csv", index=False)
    joins.to_csv(out / "adjacent_channel_differences.csv", index=False)
    pd.DataFrame(join_summaries).to_csv(out / "adjacent_channel_season_summary.csv", index=False)
    hashes_end = fingerprint(paths)
    source_note = {
        "path": "research/ftir_ec_phase3/scripts/airspec_baseline.py",
        "sha256": hashes_start["airspec"]["sha256"],
        "locations": {
            "SEG1_SEG2": "lines32–33: segment1(1820,4000), segment2(1425,2000), strict endpoints",
            "weights": "lines249–251: points strictly inside the declared analyte interval receive zero fitting weight",
            "segment1": "lines269–282: fixed2220 is the low edge of the segment1 excluded interval",
            "segment2": "lines285–305: fixed1820 is the high edge of the segment2 excluded interval",
            "combination": "lines308–323: arithmetic mean where segments overlap; no explicit cross-segment continuity constraint",
        },
        "interpretation": "1820 and2000 change segment membership/averaging;2220 changes fitting weights, not a segment join. Their presence does not establish an observed artifact.",
    }
    receipt = {
        "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
        "hashes_start": hashes_start, "hashes_end": hashes_end,
        "source_and_inputs_unchanged": hashes_start == hashes_end,
        "source_snapshot": {name: f"source_snapshot/{name}.py" for name in code_paths},
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "elapsed_seconds": time.monotonic()-started,
        "settings": {"calendar": CALENDAR, "cohort": "direct232", "seed": SEED, "n_jobs": args.jobs,
                     "transitions_cm_1": TRANSITIONS, "half_width_cm_1": HALF_WIDTH,
                     "selection": "Declared from AIRSpec constants, not optimized on outcomes",
                     "support": support, "baseline_changed": False, "importance_computed": False},
        "airspec_transition_definition": source_note,
    }
    data = {
        "schema_version": 1, "generated_at": receipt["completed_at"],
        "models": models, "representations": rep_specs,
        "scores": [row for row in scores if row["role"] == "transition_masked"],
        "predictions": [row for row in predictions if row["role"] == "transition_masked"],
        "provenance": receipt, "transition_definition": source_note, "masked_channels": mask_rows,
        "adjacent_channel_summary": join_summaries,
        "limitations": [
            "This omits the same declared channel neighborhoods from raw and corrected inputs; it does not alter or re-estimate their baselines.",
            "Changes in prediction do not prove that joins caused the disagreement; genuine spectral information may also be omitted.",
            "Retained corrected channels were already influenced by the full spline fit, so masking does not remove all possible baseline effects.",
            "Targets remain season labels and FTIR-minus-HIPS disagreement, not independently measured Ethiopian EC.",
            "Adjacent differences are high-wavenumber minus low-wavenumber absorbance at neighboring native channels; they include physical slope and noise and are not automatically artifacts.",
        ],
    }
    write_json(out / "receipt.json", receipt)
    write_json(out / "transition_sensitivity.json", data)
    if hashes_start != hashes_end:
        raise RuntimeError("Source/input changed during computation; inspect snapshots before merging output")
    log(f"Transition check complete: {len(data['scores'])} new scores; unmasked controls retained in CSV")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=TABLES / "baseline_review_20261102/filter_diagnostic_ledger.csv")
    parser.add_argument("--arrays", type=Path, default=TABLES / "baseline_existing_data_20261103/native_method_arrays.npz")
    parser.add_argument("--boundaries", type=Path, default=TABLES / "baseline_existing_data_20261103/boundary_sensitivity.csv")
    parser.add_argument("--weather", type=Path, default=ROOT / "gallery/app/public/data/research/addis_met/filters.json")
    parser.add_argument("--output", type=Path, default=TABLES / "spline_transition_sensitivity_20261103")
    parser.add_argument("--jobs", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
