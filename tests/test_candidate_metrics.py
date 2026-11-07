"""Source-evaluation invariants and export boundaries, without private data."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research/ftir_hips_chem/scripts"), str(ROOT / "gallery/data")]
from candidate_metrics import (  # noqa: E402
    SCORE_PROTOCOL, export_scoreboard, ids_sha256, paired_candidate_metrics, sha256_file,
    validate_archive_identity, verified_score_receipt,
)
from export_continuum_baseline import source_tables  # noqa: E402


def example_scores():
    metadata = pd.DataFrame({
        "filter_id": ["A-1", "B-1", "C-1", "D-1", "E-1", "F-1"],
        "site": ["A", "B", "C", "D", "E", "F"],
        "tor_ug": [1., 2., 3., 4., 200., np.nan], "ocec": [1., 1., 1., 1., 1., 1.],
    })
    return paired_candidate_metrics(metadata, {
        "raw": np.array([2., 4., 12., np.nan, 222., 5.]),
        "epo": np.array([1., 1., 5., 4., 200., 6.]),
    })


def test_metrics_share_physical_filters_and_retain_differential_coverage():
    scores, population, ledger = example_scores()
    raw, epo = scores.set_index("model").loc[["raw", "epo"]].to_dict("records")
    assert raw["improve_median_absolute_error"] == raw["improve_error"] == 2
    assert raw["improve_mae"] == 4
    assert raw["improve_rmse"] == pytest.approx(np.sqrt(86 / 3))
    assert raw["improve_mean_bias"] == 4
    assert raw["improve_p95_absolute_error"] == pytest.approx(8.3)
    assert epo["improve_median_absolute_error"] == 1
    assert epo["improve_mean_bias"] == pytest.approx(1 / 3)
    assert raw["n_eligible"] == epo["n_eligible"] == 4
    assert raw["n_common"] == epo["n_common"] == 3
    assert raw["coverage"] == .75 and epo["coverage"] == 1
    assert raw["n_failed"] == 1 and epo["n_failed"] == 0
    assert population["common_filter_ids"] == ["A-1", "B-1", "C-1"]
    assert population["n_missing_reference"] == population["n_outside_domain"] == 1
    assert ledger.filter_id.is_unique and ledger.common.sum() == 3
    assert ledger.set_index("filter_id").loc["E-1", "eligibility_reason"] == "TOR_at_or_above_150_ug"


def test_repeated_scans_cannot_silently_reweight_a_physical_filter():
    _, _, ledger = example_scores()
    ledger.loc[1, "filter_id"] = ledger.loc[0, "filter_id"]
    with pytest.raises(ValueError, match="repeated physical-filter IDs"):
        paired_candidate_metrics(ledger, {"raw": ledger.raw_predicted_ug.to_numpy()})


def test_empty_common_population_is_missing_not_zero_error():
    _, _, ledger = example_scores()
    scores, population, _ = paired_candidate_metrics(ledger, {"raw": np.full(len(ledger), np.nan)})
    assert pd.isna(scores.iloc[0].improve_median_absolute_error)
    assert scores.iloc[0].coverage == 0 and population["n_common"] == 0


def test_heavy_and_light_ratios_are_also_common_population_metrics():
    metadata = pd.DataFrame({"filter_id": ["A", "B", "C", "D"], "site": ["X"] * 4,
                             "tor_ug": [20., 30., 8., 9.], "ocec": [1., 1., 5., 5.]})
    scores, _, _ = paired_candidate_metrics(metadata, {"raw": [10., np.nan, 8., np.nan], "epo": [20., 90., 16., 90.]})
    assert scores.n_heavy.tolist() == [1, 1] and scores.n_light.tolist() == [1, 1]
    assert scores.improve_heavy_ratio.tolist() == [.5, 1.]
    assert scores.improve_light_ratio.tolist() == [1., 2.]


def write_fixture(directory):
    scores, population, ledger = example_scores()
    scores.loc[scores.model.eq("epo"), "evaluation_status"] = "corrected"
    scores.to_csv(directory / "baseline_scoreboard.csv", index=False)
    ledger.to_csv(directory / "baseline_scoreboard_filters.csv", index=False)
    receipt = {
        "status": "corrected", "protocol": SCORE_PROTOCOL, "population": population,
        "provenance": {"epo_protocol": "fixture_fold_local_epo_v1", "epo_predictions_sha256": "a" * 64},
        "outputs": {name: {"sha256": sha256_file(directory / name)} for name in
                    ("baseline_scoreboard.csv", "baseline_scoreboard_filters.csv")},
    }
    (directory / "baseline_scoreboard_receipt.json").write_text(json.dumps(receipt))
    return receipt


def test_legacy_scoreboard_withholds_epo_without_relabeling_median_as_mean(tmp_path):
    frame = pd.DataFrame({"model": ["vibes", "epo"], "improve_error": [1.1, 1.24],
                          "improve_heavy_ratio": [.77, .91], "n_heavy": [20, 20]})
    status = verified_score_receipt(tmp_path)
    result = export_scoreboard(frame, status).set_index("model")
    assert result.loc["vibes", "improve_median_absolute_error"] == 1.1
    assert pd.isna(result.loc["vibes", "improve_mae"])
    assert pd.isna(result.loc["epo", "improve_error"])
    assert pd.isna(result.loc["epo", "improve_heavy_ratio"])
    assert result.loc["epo", "evaluation_status"] == "legacy_withheld"


def test_receipt_cannot_certify_a_changed_scoreboard_or_ledger(tmp_path):
    write_fixture(tmp_path)
    assert verified_score_receipt(tmp_path)["status"] == "corrected"
    with (tmp_path / "baseline_scoreboard_filters.csv").open("a") as stream:
        stream.write("unexpected replacement\n")
    status = verified_score_receipt(tmp_path)
    assert status["status"] == "legacy_unverified"
    assert "does not match" in status["reason"]


def test_new_scores_do_not_promote_an_old_sweep_or_rule_simulation(tmp_path):
    receipt = write_fixture(tmp_path)
    pd.DataFrame({"model": ["vibes", "epo"], "k": [18, 18], "improve_mae": [1.1, 1.24],
                  "improve_heavy_lowocec_ratio": [.77, .91], "slope_hips_bc": [1.5, 1.8]}).to_csv(tmp_path / "summary.csv", index=False)
    pd.DataFrame({"model": ["VIBES k18 (before)", "EPO k18"], "pass": [.9, .95]}).to_csv(tmp_path / "pass_rule_simulation.csv", index=False)
    scoreboard, sweep, simulation, status = source_tables(tmp_path)
    assert status["status"] == "corrected" and status["sweep_status"] == "legacy_unverified"
    assert scoreboard.set_index("model").loc["epo", "evaluation_status"] == "corrected"
    epo = sweep.set_index("model").loc["epo"]
    assert epo.slope_hips_bc == 1.8  # Saved full-source optical diagnostic retained.
    assert pd.isna(epo.improve_median_absolute_error) and epo.evaluation_status == "legacy_withheld"
    assert not simulation.model.str.contains("EPO").any()
    assert sweep.set_index("model").loc["vibes", "improve_median_absolute_error"] == 1.1
    assert pd.isna(sweep.set_index("model").loc["vibes", "improve_mae"])
    table_receipt = {
        "epo_protocol": receipt["provenance"]["epo_protocol"],
        "predictions_sha256": receipt["provenance"]["epo_predictions_sha256"],
        "outputs": {"summary.csv": {"sha256": sha256_file(tmp_path / "summary.csv")}},
    }
    (tmp_path / "summary_receipt.json").write_text(json.dumps(table_receipt))
    _, sweep, _, status = source_tables(tmp_path)
    assert status["sweep_status"] == "corrected"
    assert sweep.set_index("model").loc["epo", "evaluation_status"] == "corrected"
    assert sweep.set_index("model").loc["epo", "improve_median_absolute_error"] == 1.24
    # Same table bytes with a receipt from another fitted archive remain unsafe.
    table_receipt["predictions_sha256"] = "b" * 64
    (tmp_path / "summary_receipt.json").write_text(json.dumps(table_receipt))
    _, sweep, _, status = source_tables(tmp_path)
    assert status["sweep_status"] == "legacy_unverified"


def test_versioned_source_scores_preserve_historical_tables(tmp_path):
    original = pd.DataFrame({"model": ["raw", "epo"], "improve_error": [8., 9.], "saved_target_intercept": [-1., .2]})
    historical_path = tmp_path / "baseline_scoreboard.csv"
    original.to_csv(historical_path, index=False)
    historical_bytes = historical_path.read_bytes()
    current = tmp_path / "source_scores_v2"
    current.mkdir()
    write_fixture(current)
    pd.DataFrame({"model": ["vibes", "epo"], "k": [18, 18], "improve_mae": [1.1, 1.24]}).to_csv(tmp_path / "summary.csv", index=False)
    pd.DataFrame({"model": ["VIBES k18 (before)", "EPO k18"], "pass": [.9, .95]}).to_csv(tmp_path / "pass_rule_simulation.csv", index=False)
    scoreboard, _, _, status = source_tables(tmp_path)
    assert status["status"] == "corrected"
    epo = scoreboard.set_index("model").loc["epo"]
    assert epo.improve_median_absolute_error == 1.
    assert epo.saved_target_intercept == .2
    assert historical_path.read_bytes() == historical_bytes


def write_source_archive(directory, metadata, metadata_path, models, *, epo=False, fold_id=None, ids=None):
    directory.mkdir(parents=True)
    identities = metadata.filter_id.to_numpy().astype(str) if ids is None else np.asarray(ids).astype(str)
    arrays = {"improve_filter_ids": identities, "improve_site_ids": metadata.site.to_numpy().astype(str),
              "improve_fold_id": np.array([0, 1, 2, 3, 4, 0]) if fold_id is None else np.asarray(fold_id)}
    arrays.update({f"{model}|improve_oof": np.repeat(np.arange(1., len(metadata) + 1)[:, None], 18, axis=1) for model in models})
    path = directory / "predictions.npz"
    np.savez(path, **arrays)
    receipt = {"source_identity": {
        "protocol": "physical_filter_order_and_site_folds_v1", "filter_ids_sha256": ids_sha256(metadata.filter_id),
        "improve_metadata_sha256": sha256_file(metadata_path), "predictions_sha256": sha256_file(path),
    }}
    if epo:
        receipt["epo_protocol"] = "epo_train_only_site_cv_v2"
    (directory / "fit_receipt.json").write_text(json.dumps(receipt))
    return path, receipt


def test_archive_identity_requires_saved_ids_not_only_matching_row_counts(tmp_path):
    _, _, metadata = example_scores()
    metadata_path = tmp_path / "improve_meta.csv"
    metadata.to_csv(metadata_path, index=False)
    path, receipt = write_source_archive(tmp_path / "archive", metadata, metadata_path, ["epo"],
                                         ids=metadata.filter_id.to_numpy()[::-1])
    with np.load(path) as archive:
        with pytest.raises(ValueError, match="differ from bundle order"):
            validate_archive_identity(archive, receipt, metadata, metadata_path, path)
        with pytest.raises(ValueError, match="legacy prediction archive"):
            validate_archive_identity(archive, {}, metadata, metadata_path, path)


def test_score_stage_verifies_common_fold_ids_without_target_spectra(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/workflows"))
    import run_hips_response_20261103 as response
    _, _, metadata = example_scores()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    metadata.to_csv(bundle / "improve_meta.csv", index=False)
    baseline, corrected, outputs = tmp_path / "baseline", tmp_path / "corrected", tmp_path / "outputs"
    write_source_archive(baseline, metadata, bundle / "improve_meta.csv", ["raw", "spline", "vibes", "vibes_deflate", "vibes_anchorhp"])
    write_source_archive(corrected, metadata, bundle / "improve_meta.csv", ["epo"], epo=True)
    monkeypatch.setattr(response, "XD", baseline)
    monkeypatch.setattr(response, "OUT", outputs)
    monkeypatch.setattr(response.td, "load", lambda _: (metadata, None, None))
    response.source_scores(bundle, corrected)
    output_scores = outputs / 'source_scores_v2'
    status = verified_score_receipt(output_scores)
    assert status["status"] == "corrected"
    assert status["population"]["n_common"] == 4
    scored = pd.read_csv(output_scores / "baseline_scoreboard.csv")
    assert scored.set_index("model").loc["epo", "evaluation_status"] == "corrected"
    assert scored.improve_mae.eq(0).all()
    assert set(pd.read_csv(output_scores / "baseline_scoreboard_filters.csv").fold_id) == {0, 1, 2, 3, 4}
    assert not (outputs / "baseline_scoreboard.csv").exists()
    # Different valid site splits cannot become a matched candidate comparison.
    receipt_path = corrected / "fit_receipt.json"
    with np.load(corrected / "predictions.npz") as archive:
        changed = {key: archive[key] for key in archive.files}
    changed["improve_fold_id"] = np.array([1, 0, 2, 3, 4, 0])
    np.savez(corrected / "predictions.npz", **changed)
    receipt = json.loads(receipt_path.read_text())
    receipt["source_identity"]["predictions_sha256"] = sha256_file(corrected / "predictions.npz")
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="outer folds differ"):
        response.source_scores(bundle, corrected)
