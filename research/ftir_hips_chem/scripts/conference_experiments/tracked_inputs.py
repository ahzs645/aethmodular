"""Reproduce the bounded filter/chemistry/rain run from tracked repository data.

This recipe reads the historical unified filter CSV and its ChemSpec method
source. It never downloads data, infers active sampling times, or marks the
processed daily optical exports as verified minute observations.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from data_matching import base_filter_id
except ImportError:
    from ..data_matching import base_filter_id

from .chemistry import ChemicalMeasurement, adapt_unified_filters
from .io import new_output_directory, sha256, write_json


SOURCE_FILES = {
    "unified_filters": "research/ftir_hips_chem/Filter Data/unified_filter_dataset.csv",
    "chemical_methods": "research/filter_combine/FilterBased_ChemSpecPM25_ETAD.csv",
    "daily_weather": "research/ftir_hips_chem/Weather Data/Meteostat/Addis Ababa daily Average met Data.csv",
}


def _method_evidence(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(source, comment="#", low_memory=False)
    required = {"Site_Code", "Filter_ID", "Parameter_Name", "Method_Code", "Analysis_Description", "Units", "Value"}
    if not required <= set(table):
        raise ValueError("tracked ChemSpec file lacks chemical method/provenance columns")
    rows = table.loc[table.Site_Code.eq("ETAD") & table.Parameter_Name.eq("Sulfate Ion PM2.5")].copy()
    if rows.empty or not rows.Analysis_Description.fillna("").str.contains("Ion Chromatography", regex=False).all():
        raise ValueError("Sulfate method evidence does not consistently identify ion chromatography")
    codes = pd.to_numeric(rows.Method_Code, errors="coerce")
    if codes.isna().any() or not set(codes) <= {316, 318}:
        raise ValueError("Sulfate method codes changed; review the source before declaring independence")
    units = rows.Units.astype(str)
    ug = units.str.contains("(ug/m3)", regex=False)
    ng = units.str.contains("(ng/m3)", regex=False)
    if not (ug | ng).all() or (ug & ng).any():
        raise ValueError("Sulfate source units are missing or ambiguous")
    rows["physical_id"] = rows.Filter_ID.map(base_filter_id).mask(rows.Filter_ID.isna())
    if rows.physical_id.isna().any():
        raise ValueError("Sulfate method source contains missing physical filter IDs")
    rows["sulfate_ugm3"] = pd.to_numeric(rows.Value, errors="coerce") * np.where(ug, 1.0, 0.001)
    evidence = rows[["Parameter_Name", "Method_Code", "Analysis_Description", "Units"]].drop_duplicates().sort_values(["Method_Code", "Units"])
    return evidence, rows[["physical_id", "sulfate_ugm3"]]


def prepare_tracked_inputs(repo_root: Path, output: Path) -> Path:
    """Write input tables, a source-hash receipt and an executable manifest.

    Strict all-month coverage remains a separate support audit. The additional
    all-seasons analysis is explicitly exploratory and reports incomplete test
    years. A failed support gate is never repaired by lowering thresholds here.
    """
    repo_root = Path(repo_root).resolve()
    output = new_output_directory(Path(output))
    paths = {name: repo_root / path for name, path in SOURCE_FILES.items()}
    hashes = {name: sha256(path) for name, path in paths.items()}
    evidence, source_values = _method_evidence(paths["chemical_methods"])
    source_evidence = (
        f"{SOURCE_FILES['chemical_methods']}: Sulfate Ion PM2.5; Method_Code 316/318; "
        f"Analysis_Description Ion Chromatography; source SHA-256 {hashes['chemical_methods']}"
    )
    spec = ChemicalMeasurement(
        column="sulfate_ugm3", analyte="sulfate", method="ion_chromatography",
        units="ug/m3", source_evidence=source_evidence, independence_verified=True,
    )
    long = pd.read_csv(paths["unified_filters"], low_memory=False)
    long["SampleDate"] = pd.to_datetime(long.SampleDate)
    chemical = adapt_unified_filters(long, chemical_specs={"ChemSpec_Sulfate_Ion_PM2.5": spec})
    # Require identity/value agreement with the method-bearing original source,
    # not just a global assertion that a similarly named species was measured.
    source_lookup = source_values.groupby("physical_id").sulfate_ugm3.apply(lambda s: s.dropna().to_numpy(float))
    verified = []
    for row in chemical.itertuples():
        expected = source_lookup.get(row.physical_id, np.array([], dtype=float))
        valid = np.isfinite(row.sulfate_ugm3) and len(expected) > 0 and bool(np.isclose(expected, row.sulfate_ugm3, rtol=1e-10, atol=1e-12).any())
        verified.append(valid)
    chemical["chemical_method_value_verified"] = verified
    missing_proof = chemical.sulfate_ugm3.notna() & ~chemical.chemical_method_value_verified
    chemical.loc[missing_proof, "sulfate_ugm3"] = np.nan
    chemical.loc[missing_proof, "is_excluded"] = True
    chemical.loc[missing_proof, "exclusion_reason"] = chemical.loc[missing_proof, "exclusion_reason"].map(
        lambda text: "; ".join(filter(None, [text, "chemical_value_not_verified_against_method_source"]))
    )
    # A sulfate repeat conflict has no relevance to a rain-only hypothesis.
    rain = adapt_unified_filters(long, chemical_specs={})
    weather = pd.read_csv(paths["daily_weather"], encoding="utf-8-sig")
    if not {"date", "prcp", "tavg", "wspd"} <= set(weather):
        raise ValueError("tracked daily weather lacks date/prcp/tavg/wspd")
    weather = weather[["date", "prcp", "tavg", "wspd"]].rename(columns={
        "prcp": "rain_mm", "tavg": "sample_day_temperature_c", "wspd": "sample_day_wind_kmh"})
    weather["date"] = pd.to_datetime(weather.date)
    if weather.date.duplicated().any():
        raise ValueError("tracked weather contains duplicate local dates")
    daily_controls = weather[["date", "sample_day_temperature_c", "sample_day_wind_kmh"]]
    chemical = chemical.merge(daily_controls, on="date", how="left", validate="many_to_one")
    rain = rain.merge(daily_controls, on="date", how="left", validate="many_to_one")
    tables = {"chemical_filters": chemical, "rain_filters": rain,
              "daily_weather": weather, "chemical_method_evidence": evidence}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    if hashes != {name: sha256(path) for name, path in paths.items()}:
        raise ValueError("tracked source inputs changed during preparation")
    preparation = {
        "data_scope": "research", "population": "historical tracked ETAD filter and Bole weather records",
        "adapter": "conference_experiments.chemistry.adapt_unified_filters",
        "method_review": "Sulfate identified as ion chromatography by original method code and analysis description, with physical-filter/value matching.",
        "source_inputs": [{"name": name, "path": SOURCE_FILES[name], "sha256": digest} for name, digest in hashes.items()],
        "outputs": [{"path": f"{name}.csv", "sha256": sha256(output / f"{name}.csv"), "rows": len(frame)} for name, frame in tables.items()],
        "verified_sulfate_values": int(chemical.chemical_method_value_verified.sum()),
        "weather_unit_contract": {"source": "https://dev.meteostat.net/api/stations/daily",
                                  "tavg": "degrees C", "wspd": "km/h", "prcp": "mm",
                                  "interpretation": "archived export interpreted in Meteostat's metric field units"},
        "notes": [
            "Chemical and rain cohorts have separate requested-parameter conflict audits.",
            "The strict all-month protocol is retained beside the explicitly exploratory all-seasons variant.",
            "Collection bounds/Hours_sampled in a public table do not establish verified active schedules.",
            "No optical feature, event definition or independent EC label is invented by this recipe.",
            "Loading is PM25 ug/m3 times archived collection volume m3 divided by deposit area cm2; it does not validate an active schedule.",
            "Bole temperature (degrees C) and wind speed (km/h) are sample-calendar-day station controls, not collection-window/chamber measurements.",
            "This archived weather export lacks per-value observed/model provenance; do not call every value an observed local rain measurement.",
        ],
    }
    write_json(output / "preparation_receipt.json", preparation)
    datasets = {
        name: {"kind": "table", "path": f"{name}.csv"} for name in tables
    }
    datasets.update({
        "verified_optical_filter_windows": {"kind": "table", "unavailable_reason":
            "Verified active filter schedules, observed-stream provenance and optical summaries over those intervals are not supplied by the tracked daily exports."},
        "independent_episodes": {"kind": "table", "unavailable_reason":
            "No independently defined pollution-episode table with evidence has been supplied; episodes must not be selected from the optical response being tested."},
        "minute_stream": {"kind": "table", "unavailable_reason":
            "This recipe covers tracked filter/weather inputs; provide a validated minute stream for an event analysis."},
    })
    controls = ["pm25_ugm3", "pm25_areal_ugcm2", "sample_day_temperature_c", "sample_day_wind_kmh"]
    interactions = {"chemistry": asdict(spec), "bootstrap_reps": 300, "numeric_controls": controls}
    manifest = {
        "schema_version": 1, "data_scope": "research",
        "preparation_receipt": "preparation_receipt.json",
        "preparation_receipt_sha256": sha256(output / "preparation_receipt.json"),
        "datasets": datasets,
        "experiments": [
            {"id": "sulfate_strict_coverage", "kind": "regime_interactions",
             "inputs": {"data": "chemical_filters", "method_evidence": "chemical_method_evidence"}, "parameters": interactions},
            {"id": "sulfate_partial_cycle_exploratory", "kind": "regime_interactions",
             "inputs": {"data": "chemical_filters", "method_evidence": "chemical_method_evidence"},
             "parameters": {**interactions, "test_coverage": "all_seasons"}},
            {"id": "daily_rain_features", "kind": "rain_history",
             "inputs": {"data": "rain_filters", "weather": "daily_weather"}, "parameters": {}},
            {"id": "rain_strict_coverage", "kind": "rain_evaluation",
             "inputs": {"data": {"experiment": "daily_rain_features", "table": "features"}}, "parameters": {"bootstrap_reps": 300, "numeric_controls": controls}},
            {"id": "rain_partial_cycle_exploratory", "kind": "rain_evaluation",
             "inputs": {"data": {"experiment": "daily_rain_features", "table": "features"}},
             "parameters": {"test_coverage": "all_seasons", "bootstrap_reps": 300, "numeric_controls": controls}},
            {"id": "chemistry_prediction_readiness", "kind": "chemistry_prediction",
             "inputs": {"data": "verified_optical_filter_windows"},
             "parameters": {"target": asdict(spec), "baseline_numeric": ["optical_ir_Mm1", "temperature_c"],
                            "shape_columns": ["shape_uv_ir"], "temporal_columns": ["within_window_ir_iqr"]}},
            {"id": "episode_evolution_readiness", "kind": "episode_evolution",
             "inputs": {"episodes": "independent_episodes", "data": "minute_stream"}, "parameters": {}},
        ],
    }
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"
