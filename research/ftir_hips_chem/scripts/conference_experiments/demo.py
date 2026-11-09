"""Deterministic artificial inputs for the entire conference workflow.

These generated observations and evidence strings are exclusively a software
demonstration. They are never mixed with field files or labelled research data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import MAC_VALUE, WAVELENGTHS_NM

from .io import write_json


SEED = 20261107


def _source(folder: Path, rng) -> None:
    matrix = rng.normal(size=(96, 8))
    matrix[:, 0] += np.tile(np.linspace(0, 5, 12), 8)
    ids = np.array([f"DEMO-A{i:03d}" for i in range(len(matrix))])
    metadata = pd.DataFrame({
        "analysis_id": ids, "filter_id": [f"DEMO-F{i:03d}" for i in range(len(matrix))],
        "site": np.repeat([f"DEMO-S{i}" for i in range(8)], 12),
        "ec_ug": 12 + 3 * matrix[:, 0] + 0.5 * matrix[:, 1],
        "reference_kind": "thermal_optical_ec", "season": np.tile(["A", "B", "C"], 32),
        "lot": np.tile(["DEMO-L1", "DEMO-L2"], 48), "is_blank": False,
        "is_excluded": False, "exclusion_reason": "",
    })
    np.savez_compressed(folder / "source.npz", X=matrix, analysis_ids=ids)
    # Deliberately shuffle metadata: the reader must align explicit row IDs.
    metadata.sample(frac=1, random_state=SEED).to_csv(folder / "source.csv", index=False)


def _filters_and_weather(folder: Path, rng) -> None:
    months = pd.date_range("2022-10-01", periods=24, freq="MS")
    dates = pd.DatetimeIndex([m + pd.Timedelta(days=d) for m in months for d in (4, 14, 24)])
    n = len(dates)
    ec, sulfate = rng.uniform(0.5, 8, n), rng.uniform(0.1, 1.9, n)
    difference = (0.2 + 0.15 * ec + 0.25 * (ec - 4) * (sulfate - 1)
                  + rng.normal(0, 0.02, n))
    frame = pd.DataFrame({
        "physical_id": [f"DEMO-ETAD-{5000+i}" for i in range(n)], "date": dates,
        "lot": np.where(np.arange(n) % 2, "DEMO-A", "DEMO-B"),
        "ftir_ec_ugm3": ec, "hips_fabs_Mm1": (ec - difference) * MAC_VALUE,
        "hips_source": "direct_HIPS", "pm25_ugm3": 4 + 3 * ec + rng.normal(0, 0.5, n),
        "sulfate": sulfate, "optical_ir": rng.uniform(1, 10, n),
        "temperature": rng.uniform(10, 25, n),
        "shape_uv_ir": 1 + 0.6 * sulfate + rng.normal(0, 0.1, n),
        "within_window_iqr": 0.2 + sulfate + rng.normal(0, 0.1, n),
        "sampling_schedule_verified": True, "observation_provenance_verified": True,
        "processing_history_verified": True, "valid_observed_coverage_fraction": 0.95,
        "minimum_segment_valid_observed_coverage": 0.9,
        "window_evidence_id": "SYNTHETIC known acquisition and active-pump schedule",
        "optical_quantity": "b_ATN", "optical_units": "Mm-1",
    })
    frame["window_start_utc"] = frame.date.dt.tz_localize("Africa/Addis_Ababa").dt.tz_convert("UTC")
    frame["window_end_utc"] = frame.window_start_utc + pd.Timedelta(hours=24)
    frame.to_csv(folder / "filters.csv", index=False)
    weather_dates = pd.date_range("2022-09-01", "2024-10-01", freq="D")
    rain = np.where(np.arange(len(weather_dates)) % 10 < 3, rng.uniform(2, 15, len(weather_dates)), 0)
    pd.DataFrame({"date": weather_dates, "rain_mm": rain}).to_csv(folder / "weather.csv", index=False)


def _minute_inputs(folder: Path) -> None:
    humidity, evolution, events = [], [], []
    for day in range(8):
        time = np.arange(600)
        sign = 1 if day % 2 == 0 else -1
        rh = np.full(600, 40.0 if sign == 1 else 80.0)
        rh[150:400] += sign * 20
        response = np.zeros(600)
        response[150:] += sign * 3 * np.exp(-(time[150:] - 150) / 20)
        response[400:] -= sign * 3 * np.exp(-(time[400:] - 400) / 20)
        common = {"Sample temp (C)": 23.0, "Session ID": day, "Tape position": 1,
                  "IR ATN1": 12.0, "Flow1 (mL/min)": 70.0, "Flow2 (mL/min)": 35.0}
        start = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)
        humidity.append(pd.DataFrame({
            **common, "tu": pd.date_range(start, periods=600, freq="min"),
            "Sample RH (%)": rh, "IR_atn": 10 + response,
            "UV_atn": 25 + 2 * response, "IR_spot1_atn": 9 + 0.8 * response,
        }))
        start = pd.Timestamp("2024-02-01", tz="UTC") + pd.Timedelta(days=day, hours=(day * 5) % 19)
        exponent = 1.8 - 0.002 * np.arange(180)
        evolution.append(pd.DataFrame({
            **common, "tu": pd.date_range(start, periods=180, freq="min"),
            "Sample RH (%)": 50.0, "IR_atn": 10.0,
            "UV_atn": 10 * (WAVELENGTHS_NM["UV"] / WAVELENGTHS_NM["IR"]) ** -exponent,
        }))
        events.append({"event_id": f"DEMO-E{day}", "start_utc": start,
                       "end_utc": start + pd.Timedelta(minutes=180),
                       "definition": "independent", "evidence": "SYNTHETIC externally fixed chamber schedule"})
    pd.concat(humidity, ignore_index=True).to_csv(folder / "humidity_minutes.csv", index=False)
    pd.concat(evolution, ignore_index=True).to_csv(folder / "episode_minutes.csv", index=False)
    pd.DataFrame(events).to_csv(folder / "episodes.csv", index=False)


def _measurement_inputs(folder: Path) -> None:
    pd.DataFrame({"physical_id": ["DEMO-panel-1", "DEMO-panel-2"],
                  "panel_role": ["representative", "challenge"]}).to_csv(folder / "candidates.csv", index=False)
    pair = {
        "pair_id": "DEMO-pair-1", "comparison": "ftir_thermal_ec",
        "left_quantity": "ec_ftir", "right_quantity": "thermal_ec",
        "left_unit": "ug/m3", "right_unit": "ug/m3", "right_method": "TOR",
        "left_start_utc": "2024-01-01T00:00:00Z", "left_end_utc": "2024-01-02T00:00:00Z",
        "right_start_utc": "2024-01-01T00:00:00Z", "right_end_utc": "2024-01-02T00:00:00Z",
        "left_size_fraction": "PM2.5", "right_size_fraction": "PM2.5",
        "sampling_verified": True, "calibration_verified": True,
        "identity_evidence": "SYNTHETIC identity crosswalk", "sampling_evidence": "SYNTHETIC pump log",
        "calibration_evidence": "SYNTHETIC independent method certificate",
    }
    # The second candidate must fail even though its quantity label says EC.
    invalid = {**pair, "pair_id": "DEMO-duplicate-product", "right_method": "ChemSpec_FTIR"}
    pd.DataFrame([pair, invalid]).to_csv(folder / "pairs.csv", index=False)


def create_demo_inputs(folder: Path) -> Path:
    """Write a small reproducible manifest that exercises all eleven operations."""
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    _source(folder, rng)
    _filters_and_weather(folder, rng)
    _minute_inputs(folder)
    _measurement_inputs(folder)
    datasets = {name: {"path": f"{name}.csv", "description": "Entirely synthetic demonstration"}
                for name in ("filters", "weather", "humidity_minutes", "episode_minutes", "episodes", "candidates", "pairs")}
    datasets["source"] = {"kind": "spectra", "path": "source.npz", "metadata": "source.csv",
                          "description": "Artificial spectra and artificial reference EC, not field measurements"}
    source_parameters = {
        "reference_kind": "thermal_optical_ec", "test_sites": ["DEMO-S6", "DEMO-S7"],
        "response_unit": "ug/filter", "budget": 30, "models": ["ridge", "pls", "rf"],
        "inner_folds": 3, "bootstrap_repeats": 20,
        "model_options": {"ridge": [{"n_components": 3, "alpha": 0.1}, {"n_components": 3, "alpha": 5}],
                          "pls": [{"n_components": 2}],
                          "rf": [{"rf_params": {"n_estimators": 12, "min_samples_leaf": 2, "max_depth": 4}}]},
    }
    chemical = {"column": "sulfate", "analyte": "sulfate", "method": "ion_chromatography",
                "units": "ug/m3", "source_evidence": "SYNTHETIC independent chemistry certificate",
                "independence_verified": True}
    optical = {"signal_columns": {"IR": "IR_atn", "UV": "UV_atn"}, "signal_quantity": "b_atn",
               "timezone": "UTC", "rh_provenance": "SYNTHETIC sample-stream humidity"}

    def experiment(identifier, kind, data, parameters=None, **inputs):
        return {"id": identifier, "kind": kind, "inputs": {"data": data, **inputs},
                "parameters": parameters or {}}

    manifest = {"schema_version": 1, "data_scope": "synthetic", "datasets": datasets, "experiments": [
        experiment("humidity", "humidity_memory", "humidity_minutes", {
            **optical, "signal_columns": {**optical["signal_columns"], "IR_spot1": "IR_spot1_atn"},
            "config": {"n_bootstrap": 20}}),
        experiment("coverage", "calibration_coverage", "source", {
            **source_parameters, "retain_fractions": [0, 0.5, 1], "seed": SEED}),
        experiment("regimes", "regime_interactions", "filters", {"chemistry": chemical, "bootstrap_reps": 20}),
        experiment("evolution", "episode_evolution", "episode_minutes", {
            **optical, "config": {"n_bootstrap": 20}}, episodes="episodes"),
        experiment("chemistry", "chemistry_prediction", "filters", {
            "target": chemical, "baseline_numeric": ["optical_ir", "temperature"],
            "shape_columns": ["shape_uv_ir"], "temporal_columns": ["within_window_iqr"], "bootstrap_reps": 20}),
        experiment("rain_features", "rain_history", "filters", {"config": {"mode": "daily"}}, weather="weather"),
        experiment("rain", "rain_evaluation", {"experiment": "rain_features", "table": "features"}, {"bootstrap_reps": 20}),
        experiment("reference_panel", "reference_panel", "source", {
            "representative_n": 12, "challenge_n": 8, "strata": ["season", "lot"], "seed": SEED}),
        experiment("selection", "selection_simulation", "source", {**source_parameters, "seeds": [SEED]}),
        experiment("measurement_readiness", "measurement_plan", "candidates"),
        experiment("colocation", "colocation_eligibility", "pairs"),
    ]}
    path = folder / "manifest.json"
    write_json(path, manifest)
    return path
