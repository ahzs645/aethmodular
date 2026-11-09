"""Independent-method and collection equivalence gates must not be inferred."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from conference_experiments.measurements import assess_colocation_pairs, measurement_plan


def pair():
    return {"pair_id": "pair-1", "comparison": "ftir_thermal_ec", "left_quantity": "ec_ftir",
            "right_quantity": "thermal_ec", "left_unit": "ug/m3", "right_unit": "ug/m3",
            "right_method": "TOR", "left_start_utc": "2024-01-01T00:00:00Z",
            "left_end_utc": "2024-01-02T00:00:00Z", "right_start_utc": "2024-01-01T00:00:00Z",
            "right_end_utc": "2024-01-02T00:00:00Z", "left_size_fraction": "PM2.5",
            "right_size_fraction": "PM2.5", "sampling_verified": True, "calibration_verified": True,
            "identity_evidence": "synthetic authoritative crosswalk", "sampling_evidence": "synthetic pump log",
            "calibration_evidence": "synthetic laboratory certificate"}


def test_readiness_rejects_blank_or_colliding_physical_identities():
    for ids in ([], [" "], ["f1", " f1 "]):
        with pytest.raises(ValueError, match="identities"):
            measurement_plan(pd.DataFrame({"physical_id": ids}))


def test_absorption_per_refractory_bc_requires_absorption_wavelength():
    record = {**pair(), "comparison": "absorption_sp2", "left_quantity": "b_abs",
              "right_quantity": "rbc", "left_unit": "Mm^-1", "right_method": "SP2"}
    missing = assess_colocation_pairs(pd.DataFrame([record]))["tables"]["colocation_eligibility"]
    assert not missing.eligible.any()
    assert "absorption_wavelength_missing" in missing.iloc[0].exclusion_reason
    known = assess_colocation_pairs(pd.DataFrame([{**record, "left_wavelength_nm": 870}]))
    assert known["summary"]["n_eligible"] == 1


def test_ftir_carbon_cannot_be_declared_independent_by_a_quantity_label():
    valid = pair()
    invalid = {**pair(), "pair_id": "duplicate-product", "right_method": "ChemSpec_FTIR"}
    result = assess_colocation_pairs(pd.DataFrame([valid, invalid]))
    ledger = result["tables"]["colocation_eligibility"]
    assert ledger.eligible.tolist() == [True, False]
    assert "reference_method_not_independent" in ledger.iloc[1].exclusion_reason


def test_overlap_must_cover_both_intervals_and_clocks_need_timezones():
    short = {**pair(), "right_end_utc": "2024-01-01T01:00:00Z"}
    naive = {**pair(), "pair_id": "naive", "left_start_utc": "2024-01-01 00:00"}
    ledger = assess_colocation_pairs(pd.DataFrame([short, naive]))["tables"]["colocation_eligibility"]
    assert not ledger.eligible.any()
    assert "insufficient_shared_sampling" in ledger.iloc[0].exclusion_reason
    assert "invalid_or_unzoned" in ledger.iloc[1].exclusion_reason


def test_false_strings_and_unmatched_size_do_not_pass():
    row = {**pair(), "sampling_verified": "False", "right_size_fraction": "unrestricted"}
    result = assess_colocation_pairs(pd.DataFrame([row]))
    assert result["summary"]["status"] == "insufficient_data"
    reasons = result["tables"]["colocation_eligibility"].iloc[0].exclusion_reason
    assert "sampling_unverified" in reasons and "size_fraction_unmatched" in reasons


def test_optical_wavelengths_require_an_exact_match():
    row = {**pair(), "comparison": "ma350_photoacoustic", "left_quantity": "b_atn", "right_quantity": "b_abs",
           "left_unit": "Mm^-1", "right_unit": "Mm^-1", "right_method": "photoacoustic",
           "left_wavelength_nm": 625., "right_wavelength_nm": 633.}
    result = assess_colocation_pairs(pd.DataFrame([row]))
    assert not result["tables"]["colocation_eligibility"].eligible.iloc[0]
    row.update(left_wavelength_nm=880., right_wavelength_nm=880.)
    assert assess_colocation_pairs(pd.DataFrame([row]))["tables"]["colocation_eligibility"].eligible.iloc[0]


def test_measurement_plan_never_claims_new_observations_or_auto_verifies_checks():
    candidates = pd.DataFrame({"physical_id": ["candidate-1"], "panel": ["representative"]})
    planned = measurement_plan(candidates)
    checks = planned["tables"]["evidence_requirements"].copy()
    assert planned["summary"]["n_ready"] == 0
    checks["verified"] = True  # A check without evidence is still incomplete.
    result = measurement_plan(candidates, evidence=checks)
    assert result["summary"]["n_ready"] == 0
    assert not result["tables"]["measurement_plan"].independent_addis_ec_observed.any()
    checks["evidence"] = "synthetic laboratory review"
    assert measurement_plan(candidates, evidence=checks)["summary"]["n_ready"] == 4
