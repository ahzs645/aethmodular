"""The frozen workbook's raw-spectrum QC applies before every source fit/score."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research/ftir_hips_chem/scripts"), str(ROOT / "research/ftir_hips_chem/workflows")]
import epo_validation as ev  # noqa: E402
import run_intercept_fixes_20261103 as workflow  # noqa: E402
from candidate_metrics import (  # noqa: E402
    SOURCE_QC_PROTOCOL, paired_candidate_metrics, recorded_source_qc, sha256_file, validate_archive_identity,
)
import test_epo_validation as epo_cases  # noqa: E402

frozen_inputs = epo_cases.frozen_inputs


def test_recorded_qc_is_explicit_and_absence_is_reported():
    metadata = pd.DataFrame(dict(filter_id=["A", "B", "C"], site=["A", "B", "C"], tor_ug=1., ocec=2.))
    mask, policy = recorded_source_qc(metadata)
    np.testing.assert_array_equal(mask, [True, True, True])
    assert policy["protocol"] == SOURCE_QC_PROTOCOL and policy["flag_present"] is False
    _, population, _ = paired_candidate_metrics(metadata, {"candidate": np.ones(3)})
    assert population["domain"].startswith("no recorded raw-spectrum QC flag supplied;")
    flags = pd.DataFrame({"raw_qc_ok": [True, "false", " TRUE "]})
    mask, policy = recorded_source_qc(flags)
    np.testing.assert_array_equal(mask, [True, False, True])
    assert policy["flag_present"] is True
    assert (policy["n_total"], policy["n_pass"], policy["n_excluded"]) == (3, 2, 1)
    assert len(policy["mask_sha256"]) == 64


@pytest.mark.parametrize("invalid", [None, np.nan, pd.NA, "unknown", "", 0, 1])
def test_present_null_or_ambiguous_qc_fails_closed(invalid):
    with pytest.raises(ValueError, match="raw_qc_ok contains"):
        recorded_source_qc(pd.DataFrame({"raw_qc_ok": [True, invalid]}))


def test_qc_excluded_filters_keep_predictions_and_reasons_in_the_score_ledger():
    metadata = pd.DataFrame(dict(filter_id=list("ABCD"), site=list("ABCD"), tor_ug=[1., 2., 150., np.nan],
                                 ocec=2., raw_qc_ok=[True, False, False, True]))
    scores, population, ledger = paired_candidate_metrics(metadata, {"first": np.array([2., 99., 3., 4.]),
                                                                  "second": np.array([1., 999., 2., 3.])})
    assert population["n_total"] == 4 and population["n_reference_domain"] == 2
    assert population["n_eligible"] == population["n_common"] == 1
    assert population["n_excluded_raw_qc"] == 1
    assert population["source_qc_policy"]["n_excluded"] == 2
    assert set(scores.n_excluded_raw_qc) == {1}
    assert scores.set_index("model").loc["first", "improve_median_absolute_error"] == 1.
    flagged = ledger.set_index("filter_id").loc["B"]
    assert not flagged.eligible and not flagged.common and not flagged.raw_qc_ok
    assert flagged.eligibility_reason == "recorded_raw_spectrum_QC"
    assert flagged.first_predicted_ug == 99. and flagged.second_predicted_ug == 999.


@pytest.mark.parametrize("mask", [np.array([True, False]), np.array(["true"] * 90)])
def test_epo_rejects_nonboolean_or_misaligned_qc_masks(mask):
    X, y, Z, _, _ = epo_cases.synthetic_source()
    with pytest.raises(ValueError, match="aligned Boolean vector"):
        ev.fit_training_epo(X, y, Z, np.arange(len(y)), 3, qc_mask=mask)


def refresh_decision_manifest(bundle):
    epo_cases.write_manifest(bundle, ("improve_meta.csv", "addis_meta.csv", "network_meta.csv", "improve_mass.csv", "wn_full.npy"))


def test_actual_epo_and_base_fits_exclude_recorded_bad_spectrum_without_changing_folds(frozen_inputs, tmp_path):
    bundle, corrected, _, _, _ = frozen_inputs
    metadata = pd.read_csv(bundle / "improve_meta.csv")
    metadata["raw_qc_ok"] = True
    metadata.loc[0, "raw_qc_ok"] = False
    metadata.to_csv(bundle / "improve_meta.csv", index=False)
    refresh_decision_manifest(bundle)
    before_dir = workflow.fit(bundle, models=("base", "epo"), output=tmp_path / "before", kmax=3)
    with np.load(before_dir / "predictions.npz", allow_pickle=False) as archive:
        before = {key: archive[key] for key in archive.files}
    before_receipt = json.loads((before_dir / "fit_receipt.json").read_text())
    qc = before["improve_raw_qc_ok"]
    assert qc.dtype == bool and qc.sum() == 89
    assert before_receipt["schema_version"] == 3
    assert before_receipt["source_qc_policy"]["flag_present"] is True
    assert before_receipt["source_domain_rows_before_qc"] == 90
    assert before_receipt["source_domain_qc_excluded"] == 1
    assert before_receipt["source_domain_rows"] == before_receipt["fit_rows"] == 89
    bad_id = before["improve_filter_ids"][0]
    for fold in before_receipt["epo"]["folds"]:
        assert bad_id not in fold["training_filter_ids"]
        assert bad_id not in fold["projection_training_filter_ids"]
    assert bad_id not in before_receipt["epo"]["final_model"]["training_filter_ids"]
    assert bad_id not in before_receipt["epo"]["final_model"]["projection_training_filter_ids"]
    assert np.isfinite(before["epo|improve_oof"][0]).all()
    assert np.isfinite(before["base|improve_oof"][0]).all()
    membership = pd.read_csv(before_dir / "fold_membership.csv")
    assert len(membership) == 90 and not membership.loc[0, "source_domain"]
    assert membership.loc[0, "source_domain_before_qc"] and not membership.loc[0, "raw_qc_ok"]

    # Perturb the QC-excluded spectrum and every associated source outcome.
    # It may change its own saved prediction, but cannot affect any fitted model
    # or another physical filter's held-out prediction.
    metadata.loc[0, ["ec_ug", "tor_ec_ugm3", "tor_oc_ugm3", "fe_ugm3", "k_xrf_ugm3"]] = [10000., 4000., 500., 60., 90.]
    metadata.to_csv(bundle / "improve_meta.csv", index=False)
    mass = pd.read_csv(bundle / "improve_mass.csv")
    mass.loc[0, "pm25_ugm3"] = 100000.
    mass.to_csv(bundle / "improve_mass.csv", index=False)
    X = np.load(corrected / "corrected_anchorhp_improve.npy")
    X[0] += np.linspace(20., 50., X.shape[1])
    np.save(corrected / "corrected_anchorhp_improve.npy", X)
    refresh_decision_manifest(bundle)
    after_dir = workflow.fit(bundle, models=("base", "epo"), output=tmp_path / "after", kmax=3)
    with np.load(after_dir / "predictions.npz", allow_pickle=False) as after:
        np.testing.assert_array_equal(before["improve_fold_id"], after["improve_fold_id"])
        for model in ("base", "epo"):
            np.testing.assert_array_equal(before[f"{model}|improve_oof"][qc], after[f"{model}|improve_oof"][qc])
            for key in ("B", "xm", "ym", "addis", "network"):
                np.testing.assert_array_equal(before[f"{model}|{key}"], after[f"{model}|{key}"])
        for prefix in ("epo", *[f"epo_fold_{fold}" for fold in range(5)]):
            for key in ("basis", "signatures", "B", "xm", "ym", "training_ids", "projection_training_ids"):
                np.testing.assert_array_equal(before[f"{prefix}|{key}"], after[f"{prefix}|{key}"])
        im, _, _ = workflow.td.load(bundle)
        receipt = json.loads((after_dir / "fit_receipt.json").read_text())
        validate_archive_identity(after, receipt, im, bundle / "improve_meta.csv", after_dir / "predictions.npz")


def test_old_fit_cannot_claim_a_recorded_qc_rule_it_did_not_certify(frozen_inputs, tmp_path):
    bundle, _, _, _, _ = frozen_inputs
    metadata = pd.read_csv(bundle / "improve_meta.csv")
    metadata["raw_qc_ok"] = True
    metadata.loc[0, "raw_qc_ok"] = False
    metadata.to_csv(bundle / "improve_meta.csv", index=False)
    refresh_decision_manifest(bundle)
    output = workflow.fit(bundle, models=("base", "epo"), output=tmp_path / "fitted", kmax=3)
    im, _, _ = workflow.td.load(bundle)
    receipt = json.loads((output / "fit_receipt.json").read_text())
    with np.load(output / "predictions.npz", allow_pickle=False) as archive:
        missing_policy = dict(receipt)
        missing_policy.pop("source_qc_policy")
        with pytest.raises(ValueError, match="does not certify the recorded source QC"):
            validate_archive_identity(archive, missing_policy, im, bundle / "improve_meta.csv", output / "predictions.npz")
        changed = {key: archive[key] for key in archive.files}
    changed["improve_raw_qc_ok"] = np.ones(len(im), dtype=bool)
    np.savez(output / "predictions.npz", **changed)
    receipt["source_identity"]["predictions_sha256"] = sha256_file(output / "predictions.npz")
    with np.load(output / "predictions.npz", allow_pickle=False) as archive:
        with pytest.raises(ValueError, match="saved source QC mask differs"):
            validate_archive_identity(archive, receipt, im, bundle / "improve_meta.csv", output / "predictions.npz")
