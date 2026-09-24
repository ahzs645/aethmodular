"""Export the executed AIRSpec/VIBES follow-up without changing the frozen gallery run.

Run with uv run --locked --no-sync python gallery/data/export_baseline_followup.py.
All metrics are reconciled against per-filter notebook predictions before export.
"""

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "research/ftir_hips_chem"
RUN = AREA / "output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"
RESULTS = AREA / "output/tables/vibes_followup_notebook"
EXECUTED = AREA / "notebooks/archive/executed/vibes_followup_experiments_executed.ipynb"
OUT = ROOT / "gallery/app/public/data"
TABLES = [
    "common_test_metrics.csv", "common_test_predictions.csv",
    "analog_membership.csv", "restricted_membership_sensitivity.csv",
    "routing_metrics.csv", "routing_nested_training_predictions.csv",
    "routing_prior_test_predictions.csv", "routing_largest_VIBES_errors.csv",
]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def check_score(frame: pd.DataFrame, metric: pd.Series, prefix: str) -> None:
    error = frame.prediction.to_numpy(float) - frame.y.to_numpy(float)
    measured = {
        "n": len(frame),
        "RMSE": np.sqrt(np.mean(error**2)),
        "MAE": np.mean(abs(error)),
        "bias": np.mean(error),
        "predictive_R2": 1 - np.sum(error**2) / np.sum((frame.y-frame.y.mean())**2),
    }
    for key, value in measured.items():
        if not np.isclose(value, metric[key], rtol=0, atol=1e-9):
            raise ValueError(f"{prefix} {key} did not reconcile")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run = json.loads((RUN / "RUN_MANIFEST.json").read_text())
    frozen = json.loads((RESULTS / "input_manifest.json").read_text())
    if frozen["source_signature"] != run["signature"]:
        raise ValueError("Follow-up source signature differs from completed Colab run")
    for name, expected in frozen["source_sha256"].items():
        if sha(RUN / name) != expected:
            raise ValueError(f"Frozen input changed after follow-up: {name}")
    notebook = nbformat.read(EXECUTED, as_version=4)
    if any(output.output_type == "error" for cell in notebook.cells for output in cell.get("outputs", [])):
        raise ValueError("Executed notebook contains saved cell errors")

    metrics = pd.read_csv(RESULTS / "common_test_metrics.csv")
    predictions = pd.read_csv(RESULTS / "common_test_predictions.csv")
    cases = pd.read_csv(RUN / "case_audit.csv")
    test = cases.loc[cases.kind.eq("calibration") & cases.split.eq("test")]
    train = cases.loc[cases.kind.eq("calibration") & cases.split.eq("train")]
    if len(test) != 2327 or len(train) != 10066 or metrics.shape[0] != 12:
        raise ValueError("Follow-up population or candidate count changed")
    test_ids = set(test.sample_id)
    if predictions.duplicated(["model", "method", "sample_id"]).any():
        raise ValueError("Duplicate common-test predictions")
    for metric in metrics.itertuples(index=False):
        part = predictions.loc[predictions.model.eq(metric.model) & predictions.method.eq(metric.method)]
        if set(part.sample_id) != test_ids:
            raise ValueError(f"Different test population for {metric.model}/{metric.method}")
        check_score(part, pd.Series(metric._asdict()), f"{metric.model}/{metric.method}")
    original = pd.read_csv(RUN / "heldout_predictions.csv")
    for method in ("AIRSpec", "VIBES"):
        full = predictions.loc[predictions.model.eq("full_pool") & predictions.method.eq(method)]
        old = original.loc[original.cohort.eq("full_pool") & original.method.eq(method)]
        joined = full.merge(old[["sample_id", "prediction"]], on="sample_id", validate="one_to_one")
        np.testing.assert_allclose(joined.prediction_x, joined.prediction_y, rtol=0, atol=1e-8)

    restricted = pd.read_csv(RESULTS / "restricted_membership_sensitivity.csv")
    if len(restricted) != 4 or set(restricted.n) != {137, 2190}:
        raise ValueError("Restricted-membership audit changed")
    selected = pd.read_csv(RESULTS / "analog_membership.csv")
    overlap = json.loads((RESULTS / "selection_overlap.json").read_text())
    if selected.duplicated(["model", "sample_id"]).any():
        raise ValueError("Duplicate analog filter")
    for name, part in selected.groupby("model"):
        if len(part) != 500 or not set(part.sample_id).issubset(set(train.sample_id)):
            raise ValueError(f"Analog selection escaped training: {name}")
        if int(part.matching_channels.max()) != int(part.matching_channels.min()) or not part.pls_channels.eq(2002).all():
            raise ValueError("Selection and PLS channel grids were conflated")
        baseline = set(selected.loc[selected.model.eq("analog_full"), "sample_id"])
        if overlap[name]["replaced_vs_unmasked"] != 500-len(set(part.sample_id) & baseline):
            raise ValueError("Analog overlap did not reconcile")

    routed = pd.read_csv(RESULTS / "routing_metrics.csv")
    nested = pd.read_csv(RESULTS / "routing_nested_training_predictions.csv")
    prior = pd.read_csv(RESULTS / "routing_prior_test_predictions.csv")
    if set(nested.sample_id) != set(train.sample_id) or set(prior.sample_id) != test_ids:
        raise ValueError("Routing populations differ from frozen split")
    if nested.sample_id.duplicated().any() or prior.sample_id.duplicated().any():
        raise ValueError("Duplicate routing predictions")
    for population, frame in (("training_site_nested", nested), ("prior_inspected_test", prior)):
        np.testing.assert_allclose(frame.routed,
                                   np.where(frame.choice.eq("VIBES"), frame.VIBES, frame.AIRSpec),
                                   rtol=0, atol=1e-10)
        for method in ("AIRSpec", "VIBES", "routed"):
            metric = routed.loc[routed.population.eq(population) & routed.method.eq(method)]
            if len(metric) != 1:
                raise ValueError(f"Missing routing score: {population}/{method}")
            check_score(frame.rename(columns={method: "prediction"}) if method == "routed"
                        else frame[["y", method]].rename(columns={method: "prediction"}),
                        metric.iloc[0], f"{population}/{method}")
    for method in ("AIRSpec", "VIBES"):
        old = original.loc[original.cohort.eq("full_pool") & original.method.eq(method)]
        joined = prior.merge(old[["sample_id", "prediction"]], on="sample_id", validate="one_to_one")
        np.testing.assert_allclose(joined[method], joined.prediction, rtol=0, atol=1e-8)
    readiness = json.loads((RESULTS / "addis_readiness.json").read_text())
    if readiness["status"] != "reference_unavailable" or readiness["n_independent_thermal_matches"] != 0:
        raise ValueError("Addis status changed; review before publishing")

    source_paths = [RESULTS / name for name in TABLES]
    source_paths.extend([RESULTS / "selection_overlap.json", RESULTS / "input_manifest.json",
                         RESULTS / "addis_readiness.json", EXECUTED,
                         AREA / "workflows/vibes_followup_studies.py",
                         AREA / "workflows/build_vibes_followup_notebook.py"])
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    data = {
        "schema_version": 1,
        "source_run_signature": run["signature"],
        "scope": "new_exploratory_followup_on_saved_corrections",
        "common_test_n": len(test), "common_test_sites": test.Site.nunique(),
        "train_n": len(train), "train_sites": train.Site.nunique(),
        "selection_metrics": rows(metrics),
        "selection_overlap": overlap,
        "restricted_sensitivity": rows(restricted),
        "routing_metrics": rows(routed),
        "routing_extreme": rows(pd.read_csv(RESULTS / "routing_largest_VIBES_errors.csv").head(1)),
        "addis_readiness": readiness,
        "source_sha256": hashes,
    }
    (OUT / "baseline_followup.json").write_text(json.dumps(data, allow_nan=False, separators=(",", ":")))
    for name in TABLES:
        shutil.copy2(RESULTS / name, OUT / f"baseline_followup_{name}")
    shutil.copy2(EXECUTED, OUT / "baseline_followup_executed.ipynb")
    shutil.copy2(AREA / "vibes_followup_experiments.ipynb", OUT / "baseline_followup_source.ipynb")
    print(f"Exported {len(metrics)} common-test scores and {len(routed)} routing scores for {len(test)} shared test filters.")


if __name__ == "__main__":
    main()
