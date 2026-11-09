"""Figure audit attachments must belong to the supplied scientific bundle."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research/ftir_hips_chem/scripts"), str(ROOT / "research/ftir_hips_chem/workflows")]
from candidate_metrics import ids_sha256, recorded_source_qc, sha256_file, verified_score_receipt  # noqa: E402
from epo_validation import site_folds  # noqa: E402
import run_hips_response_20261103 as response  # noqa: E402

PRIMARY_MODELS = {"raw", "spline", "vibes", "vibes_deflate", "vibes_anchorhp", "epo"}


def write_decision_manifest(bundle):
    names = ("improve_meta.csv", "addis_meta.csv", "network_meta.csv", "wn_full.npy")
    (bundle / "MANIFEST.json").write_text(json.dumps({"files": {
        name: {"sha256": sha256_file(bundle / name)} for name in names}}))


@pytest.fixture
def corrected_source_attachment(tmp_path, monkeypatch):
    bundle = tmp_path / "decision_bundle"
    bundle.mkdir()
    reference = np.arange(1., 51.)
    metadata = pd.DataFrame(dict(filter_id=[f"F{i:03d}" for i in range(50)],
                                 site=np.repeat([f"SITE{i}" for i in range(10)], 5), ec_ug=reference,
                                 volume_m3=2.5, tor_ec_ugm3=reference / 2.5,
                                 tor_oc_ugm3=reference * 2 / 2.5, fabs_Mm1=20., raw_qc_ok=True))
    metadata.loc[0, ["ec_ug", "tor_ec_ugm3"]] = np.nan
    metadata.loc[1, ["ec_ug", "tor_ec_ugm3"]] = [150., 60.]
    metadata.loc[2, "raw_qc_ok"] = False
    metadata.to_csv(bundle / "improve_meta.csv", index=False)
    pd.DataFrame(dict(filter_id=["ADDIS1"], fabs_Mm1=[20.], volume_m3=[2.5],
                      ec_published_ugm3=[3.])).to_csv(bundle / "addis_meta.csv", index=False)
    pd.DataFrame(dict(filter_id=["NETWORK1"], fabs_Mm1=[20.], volume_m3=[2.5],
                      ec_ftir_ugm3=[3.])).to_csv(bundle / "network_meta.csv", index=False)
    grid = np.linspace(4000, 500, 2722, dtype=np.float32)
    np.save(bundle / "wn_full.npy", grid)
    write_decision_manifest(bundle)
    qc_mask, qc_policy = recorded_source_qc(metadata)
    ids = metadata.filter_id.to_numpy().astype(str)
    sites = metadata.site.to_numpy().astype(str)
    folds = np.full(len(metadata), -1, dtype=int)
    for fold, (_, test) in enumerate(site_folds(ids, sites)):
        folds[test] = fold

    baseline, epo = tmp_path / "baseline", tmp_path / "epo"
    for directory, offsets in ((baseline, dict(raw=5., spline=4., vibes=3., vibes_deflate=2.5, vibes_anchorhp=10.)),
                               (epo, dict(base=2., epo=.5))):
        directory.mkdir()
        arrays = dict(improve_filter_ids=ids, improve_site_ids=sites, improve_fold_id=folds,
                      improve_raw_qc_ok=qc_mask)
        arrays.update({f"{model}|improve_oof": np.repeat((reference + offset)[:, None], 18, axis=1)
                       for model, offset in offsets.items()})
        if directory == epo:
            arrays["wn_bin2"] = grid.reshape(-1, 2).mean(axis=1)
        prediction_path = directory / "predictions.npz"
        np.savez(prediction_path, **arrays)
        receipt = dict(source_qc_policy=qc_policy, source_identity=dict(
            protocol="physical_filter_order_and_site_folds_v1", filter_ids_sha256=ids_sha256(ids),
            improve_metadata_sha256=sha256_file(bundle / "improve_meta.csv"),
            predictions_sha256=sha256_file(prediction_path)))
        if directory == epo:
            receipt["epo_protocol"] = "epo_train_only_site_cv_v2"
        (directory / "fit_receipt.json").write_text(json.dumps(receipt))

    def forbidden_fit(*args, **kwargs):
        pytest.fail("figure audit and saved-score fixtures must not refit a model")

    monkeypatch.setattr(response.td, "fit", forbidden_fit)
    monkeypatch.setattr(response.ix, "fit", forbidden_fit)
    monkeypatch.setattr(response.ix.ev, "fit_training_epo", forbidden_fit)
    scores = response.source_scores(bundle, baseline_output=baseline, epo_output=epo,
                                    scores_output=tmp_path / "source_scores")
    assert verified_score_receipt(scores)["status"] == "corrected"
    return bundle, scores, qc_policy


def assert_valid_source_summary(bundle, scores, qc_policy):
    from audit_closeout_figures_20261107 import verified_source_summary

    summary = verified_source_summary(scores, bundle)
    assert summary["status"] == "verified_corrected"
    population = summary["population"]
    assert population["n_total"] == 50
    assert population["n_reference_domain"] == 48
    assert population["n_eligible"] == population["n_common"] == 47
    assert population["n_excluded_raw_qc"] == 1
    assert "common_filter_ids" not in population
    assert len(population["common_filter_ids_sha256"]) == 64
    assert summary["recorded_qc"] == qc_policy
    assert {row["model"] for row in summary["scores"]} == PRIMARY_MODELS
    assert len(summary["scores"]) == 6 and all(row["k"] == 18 for row in summary["scores"])
    control = summary["matched_resolution_control"]
    assert control["primary_comparison_member"] is False
    assert control["same_common_filters_as_primary"] is True
    assert control["resolution"]["primary_anchorhp_channels"] == 2722
    assert control["resolution"]["control_channels"] == 1361
    assert {row["model"] for row in control["summary"]} == {"anchorhp_bin2_base", "epo_bin2"}
    assert "common_filter_ids" not in control["population"]


def test_valid_scores_cannot_attach_to_same_identity_bundle_with_changed_tor(corrected_source_attachment, tmp_path):
    from audit_closeout_figures_20261107 import verified_source_summary

    original, scores, qc_policy = corrected_source_attachment
    assert_valid_source_summary(original, scores, qc_policy)
    changed_bundle = tmp_path / "another_decision_bundle"
    shutil.copytree(original, changed_bundle)
    before = pd.read_csv(original / "improve_meta.csv")
    changed = before.copy()
    changed.loc[5, "ec_ug"] += .25
    changed.loc[5, "tor_ec_ugm3"] = changed.loc[5, "ec_ug"] / changed.loc[5, "volume_m3"]
    changed.to_csv(changed_bundle / "improve_meta.csv", index=False)
    write_decision_manifest(changed_bundle)
    assert len(before) == len(changed)
    pd.testing.assert_frame_equal(before[["filter_id", "site", "raw_qc_ok"]], changed[["filter_id", "site", "raw_qc_ok"]])
    assert recorded_source_qc(before)[1] == recorded_source_qc(changed)[1]
    assert verified_score_receipt(scores)["status"] == "corrected"  # Internal hashes alone still pass.
    with pytest.raises(ValueError):
        verified_source_summary(scores, changed_bundle)
