"""Coverage, identity and optical-quantity contracts for the named paper comparison."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "paper_comparison", ROOT / "research/ftir_hips_chem/scripts/paper_comparison.py"
)
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


@pytest.fixture
def metadata():
    return {
        "schema_version": 1, "timezone": "Africa/Addis_Ababa", "bc_unit": "ug/m3",
        "timestamp_basis": "hour_start_with_offset", "instrument_model": "MA350",
        "inlet": "no_size_selective_inlet", "qc": {"description": "Synthetic fixture", "reference": "test only"},
        "deployments": {
            "central": {"serial": "synthetic-central", "start": "2022-04-01T00:00:00+03:00",
                        "end_exclusive": "2022-04-01T03:00:00+03:00"},
            "jacros": {"serial": "synthetic-jacros", "start": "2022-04-01T00:00:00+03:00",
                       "end_exclusive": "2022-04-01T03:00:00+03:00"},
        },
    }


def frame():
    return pd.DataFrame({
        "deployment_id": ["central"] * 3 + ["jacros"] * 3,
        "timestamp": ["2022-04-01T00:00:00+03:00", "2022-04-01T01:00:00+03:00", "2022-04-01T02:00:00+03:00"] * 2,
        "bc_ug_m3": [1, 3, 5, 4, 8, 12], "qc_pass": [True, False, True, True, True, True],
    })


def test_coverage_and_common_hours_use_only_eligible_absolute_hours(metadata):
    result = paper.compare_hourly(frame(), metadata)
    assert result["study_calendar_hours"] == 26304
    central = result["deployments"]["central"]
    assert central["expected_hours"] == 3
    assert central["valid_hours"] == 2
    assert central["coverage_percent"] == pytest.approx(200 / 3)
    assert central["qc_rejected_hours"] == 1
    assert central["seasonal"]["Belg"]["valid_hours"] == 2
    assert result["common_valid_hours"] == 2
    assert result["common_hour_means_ug_m3"] == {"central": 3, "jacros": 8}
    assert result["paper_reproduction_complete"] is False
    assert result["independent_validation"] is False
    json.dumps(result, allow_nan=False)


def test_same_hour_expressed_with_different_offsets_is_duplicate(metadata):
    data = frame()
    duplicate = data.iloc[[0]].copy()
    duplicate["timestamp"] = "2022-03-31T21:00:00Z"
    with pytest.raises(ValueError, match="Duplicate"):
        paper.compare_hourly(pd.concat([data, duplicate]), metadata)


@pytest.mark.parametrize("value", ["2022-04-01T00:00:00", "2022-04-01T00:01:00+03:00", None])
def test_ambiguous_or_nonhourly_timestamps_fail(metadata, value):
    data = frame()
    data.loc[0, "timestamp"] = value
    with pytest.raises(ValueError, match="timestamp"):
        paper.compare_hourly(data, metadata)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1, "not a number"])
def test_invalid_bc_is_unavailable_not_zero(metadata, value):
    data = frame().astype({"bc_ug_m3": object})
    data.loc[0, "bc_ug_m3"] = value
    central = paper.compare_hourly(data, metadata)["deployments"]["central"]
    assert central["valid_hours"] == 1
    assert central["mean_ug_m3"] == 5
    assert central["invalid_bc_after_qc_hours"] == 1


def test_same_instrument_cannot_be_two_overlapping_sites(metadata):
    metadata["deployments"]["central"]["serial"] = "MA350-0238"
    metadata["deployments"]["jacros"]["serial"] = "ma350 0238"
    with pytest.raises(ValueError, match="same instrument serial"):
        paper.compare_hourly(frame(), metadata)


@pytest.mark.parametrize("bad", [None, [], {"schema_version": 1, "qc": None}])
def test_malformed_metadata_fails_as_contract_error(bad):
    with pytest.raises(ValueError):
        paper.compare_hourly(frame(), bad)


def test_null_qc_or_deployment_fail_as_contract_error(metadata):
    bad = copy.deepcopy(metadata)
    bad["qc"] = None
    with pytest.raises(ValueError, match="metadata.qc"):
        paper.compare_hourly(frame(), bad)
    metadata["deployments"]["central"] = None
    with pytest.raises(ValueError, match="deployments.central"):
        paper.compare_hourly(frame(), metadata)


def test_empty_input_and_different_inlet_remain_explicit(metadata):
    metadata["inlet"] = "PM2.5 inlet"
    result = paper.compare_hourly(frame().iloc[:0], metadata)
    assert result["deployments"]["central"]["mean_ug_m3"] is None
    assert result["deployments"]["central"]["valid_hours"] == 0
    assert result["protocol_differences"]["inlet"] == {
        "declared": "PM2.5 inlet", "paper": "no_size_selective_inlet"
    }
    json.dumps(result, allow_nan=False)


def test_optical_fraction_has_separate_support_and_declared_quantity(metadata):
    data = frame()
    ratio = 880 / 470
    data["absorption_long_mm1"] = 10.0
    data["absorption_short_mm1"] = 10 * (0.5 * ratio + 0.5 * ratio**2)
    data.loc[2, "absorption_short_mm1"] = np.nan
    metadata["source_model"] = {
        "signal_basis": "absorption", "short_wavelength_nm": 470, "long_wavelength_nm": 880,
        "correction_reference": "synthetic physical mixture, not a firmware reproduction",
    }
    result = paper.compare_hourly(data, metadata)
    source = result["deployments"]["central"]["optical_source_model"]
    assert source["valid_hours"] == 1
    assert source["mean_fraction"] == pytest.approx(0.5)
    assert source["absorption_weighted_fraction"] == pytest.approx(0.5)
    assert source["quantity"] == "fraction_of_long_wavelength_absorption"
    assert source["comparable_to_reported_biomass_bc_percent"] is False
    assert source["clip"] is True
    assert source["invalid_reasons"] == {"nonfinite_signal": 1}
    metadata["source_model"]["clip"] = False
    with pytest.raises(ValueError, match="clipping are fixed"):
        paper.compare_hourly(data, metadata)
    metadata["source_model"].pop("clip")
    metadata["source_model"]["signal_basis"] = "BCc"
    with pytest.raises(ValueError, match="BCc is not absorption"):
        paper.compare_hourly(data, metadata)


def test_cli_pins_inputs_and_refuses_overwrite(tmp_path, metadata):
    hourly, manifest, output = (tmp_path / name for name in ("hourly.csv", "deployment.json", "report.json"))
    frame().to_csv(hourly, index=False)
    manifest.write_text(json.dumps(metadata))
    command = [sys.executable, "-m", "aethmodular_cli", "study", "shlipak-2026", "--hourly", str(hourly),
               "--metadata", str(manifest), "--output", str(output)]
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    report = json.loads(output.read_text())
    assert set(report["inputs"]) == {"hourly", "metadata", "profile"}
    assert len(report["inputs"]["hourly"]["sha256"]) == 64
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode == 2


def test_duplicate_csv_columns_are_rejected(tmp_path, metadata):
    with pytest.raises(ValueError, match="duplicate column"):
        paper.compare_hourly(pd.DataFrame(columns=["deployment_id", "deployment_id"]), metadata)
    hourly, manifest = tmp_path / "hourly.csv", tmp_path / "metadata.json"
    hourly.write_text("deployment_id,timestamp,bc_ug_m3,qc_pass,bc_ug_m3\n")
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="duplicate column"):
        paper.compare_files(hourly, manifest)
