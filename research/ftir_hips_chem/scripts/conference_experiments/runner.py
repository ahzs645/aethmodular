"""Run explicit experiment manifests and retain failures and missing evidence."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

import pandas as pd

from .io import (load_dataset, new_output_directory, runtime_receipt, safe_name,
                 sha256, write_json, write_report, write_result)


KINDS = {
    "humidity_memory", "episode_evolution", "calibration_coverage",
    "reference_panel", "selection_simulation", "regime_interactions",
    "chemistry_prediction", "rain_history", "rain_evaluation",
    "measurement_plan", "colocation_eligibility",
}


class MissingEvidence(ValueError):
    """A declared input is absent or an upstream experiment could not run."""


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected conference manifest schema_version=1")
    if manifest.get("data_scope") not in {"synthetic", "research"}:
        raise ValueError("Declare data_scope as synthetic or research")
    if not isinstance(manifest.get("datasets"), dict):
        raise ValueError("datasets must be a mapping")
    for name in manifest["datasets"]:
        safe_name(name)
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Provide at least one experiment")
    seen = set()
    for item in experiments:
        identifier = safe_name(item["id"])
        if identifier in seen:
            raise ValueError("Experiment identifiers must be unique")
        if item["kind"] not in KINDS:
            raise ValueError(f"Unsupported experiment kind: {item['kind']}")
        if not isinstance(item.get("parameters", {}), dict):
            raise ValueError("Experiment parameters must be an object")
        if not isinstance(item.get("inputs", {}), dict):
            raise ValueError("Experiment inputs must be an object")
        for reference in item.get("inputs", {}).values():
            if isinstance(reference, str):
                if reference not in manifest["datasets"]:
                    raise ValueError(f"Unknown dataset: {reference}")
            elif isinstance(reference, dict) and set(reference) == {"experiment", "table"}:
                if reference["experiment"] not in seen:
                    raise ValueError("Result inputs must refer to an earlier experiment")
                safe_name(reference["table"])
            else:
                raise ValueError("Input must name a dataset or an earlier experiment/table")
        seen.add(identifier)


def _configuration(cls, options):
    if options is None:
        return None
    unknown = set(options) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**options)


def dispatch(kind: str, inputs: dict, parameters: dict) -> dict:
    parameters = dict(parameters)
    data = inputs.get("data")
    if kind in {"calibration_coverage", "reference_panel", "selection_simulation"}:
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError("This experiment requires a spectra dataset with aligned metadata")
        if kind == "calibration_coverage":
            from .coverage import run_coverage_experiment
            return run_coverage_experiment(*data, **parameters)
        from .selection import build_reference_panel, simulate_reference_selection
        function = build_reference_panel if kind == "reference_panel" else simulate_reference_selection
        return function(*data, **parameters)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("This experiment requires a table dataset named data")
    if kind in {"humidity_memory", "episode_evolution"}:
        from .humidity import HumidityConfig, reconstruct_ma350_attenuation, run_humidity_memory
        conversion = None
        raw = parameters.pop("raw_ma350", None)
        if raw is not None:
            data, derived, conversion = reconstruct_ma350_attenuation(data, **raw)
            selected = parameters.get("signal_columns", derived)
            if any(derived.get(name) != column for name, column in selected.items()):
                raise ValueError("Raw MA350 signal selection must use the declared attenuation conversion")
            if parameters.get("signal_quantity", "b_atn") != "b_atn":
                raise ValueError("Raw BC times sigma reconstructs b_atn, not corrected b_abs")
            parameters["signal_columns"], parameters["signal_quantity"] = selected, "b_atn"
        if kind == "humidity_memory":
            parameters["config"] = _configuration(HumidityConfig, parameters.get("config"))
            for name in ("active_intervals", "filter_catalog"):
                if name in inputs:
                    parameters[name] = inputs[name]
            result = run_humidity_memory(data, **parameters)
        else:
            from .episodes import EpisodeConfig, run_episode_evolution
            parameters["config"] = _configuration(EpisodeConfig, parameters.get("config"))
            if "episodes" not in inputs:
                raise MissingEvidence("Independent episode definitions are required")
            result = run_episode_evolution(data, inputs["episodes"], **parameters)
        if conversion is not None:
            result["summary"]["raw_ma350_conversion"] = conversion
        return result
    if kind in {"regime_interactions", "chemistry_prediction"}:
        from .chemistry import ChemicalMeasurement, evaluate_chemistry_prediction, evaluate_regime_interactions
        key = "chemistry" if kind == "regime_interactions" else "target"
        parameters[key] = ChemicalMeasurement(**parameters[key])
        function = evaluate_regime_interactions if kind == "regime_interactions" else evaluate_chemistry_prediction
        return function(data, **parameters)
    if kind == "rain_history":
        from .rain import RainConfig, build_rain_history
        if "weather" not in inputs:
            raise MissingEvidence("Rain history needs an independently specified weather table")
        parameters["config"] = _configuration(RainConfig, parameters.get("config")) or RainConfig()
        return build_rain_history(data, inputs["weather"], active_intervals=inputs.get("active_intervals"), **parameters)
    if kind == "rain_evaluation":
        from .rain import evaluate_rain_history
        return evaluate_rain_history(data, **parameters)
    if kind == "measurement_plan":
        from .measurements import measurement_plan
        return measurement_plan(data, evidence=inputs.get("evidence"), **parameters)
    if kind == "colocation_eligibility":
        from .measurements import assess_colocation_pairs
        return assess_colocation_pairs(data, **parameters)
    raise ValueError(f"Unsupported experiment kind: {kind}")


def run_manifest(manifest_path: Path, output: Path, *, output_prepared=False) -> dict:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    output = Path(output)
    if not output_prepared:
        new_output_directory(output)
    elif not output.is_dir() or (output / "receipt.json").exists():
        raise ValueError("Prepared output must be new and must not contain a prior receipt")
    receipt = {"schema_version": 1, "data_scope": manifest["data_scope"],
               "manifest_sha256": sha256(manifest_path), "runtime": runtime_receipt(),
               "inputs": {}, "experiments": []}
    cache, results = {}, {}

    def materialize(reference):
        if isinstance(reference, dict):
            parent = results[reference["experiment"]]
            if parent["summary"].get("status") in {"blocked", "failed", "insufficient_data"}:
                raise MissingEvidence(f"Upstream experiment {reference['experiment']} did not complete")
            return parent["tables"][reference["table"]].copy()
        if reference not in cache:
            spec = manifest["datasets"][reference]
            if spec.get("unavailable_reason"):
                raise MissingEvidence(spec["unavailable_reason"])
            try:
                dataset, files = load_dataset(spec, manifest_path.parent)
            except FileNotFoundError as exc:
                raise MissingEvidence(f"Dataset {reference} is unavailable; configure its manifest input") from exc
            cache[reference] = dataset
            receipt["inputs"][reference] = {"kind": spec.get("kind", "table"), "files": files,
                                           "description": spec.get("description", "")}
        return cache[reference]

    for experiment in manifest["experiments"]:
        identifier, kind = experiment["id"], experiment["kind"]
        print(f"Running {identifier} ({kind})", flush=True)
        try:
            inputs = {name: materialize(ref) for name, ref in experiment.get("inputs", {}).items()}
            result = dispatch(kind, inputs, experiment.get("parameters", {}))
        except MissingEvidence as exc:
            result = {"summary": {"status": "blocked", "reason": str(exc)}, "tables": {}}
        except Exception as exc:
            # Preserve every other independent experiment, but never report an
            # exception as success or silently drop it from the research record.
            result = {"summary": {"status": "failed", "error_type": type(exc).__name__, "reason": str(exc)}, "tables": {}}
        result["summary"]["data_scope"] = manifest["data_scope"]
        result["summary"]["requested_parameters"] = experiment.get("parameters", {})
        results[identifier] = result
        item = write_result(output, identifier, kind, result)
        receipt["experiments"].append(item)
        write_json(output / "receipt.json", receipt)
        write_report(output, receipt)
        print(f"  {item['status']}", flush=True)
    receipt["status"] = ("failed" if any(r["status"] == "failed" for r in receipt["experiments"])
                         else "incomplete" if any(r["status"] in {"blocked", "insufficient_data"} for r in receipt["experiments"])
                         else "completed")
    write_json(output / "receipt.json", receipt)
    return receipt
