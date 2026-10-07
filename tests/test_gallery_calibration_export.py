"""The cutoff chart must not substitute target error or nominal cohort size."""
import json
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from gallery.data.export_calibration import cache_key, cache_readout, compact
from gallery.data import export_calibration as exporter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ftir_ec_phase3" / "scripts"))


def cached_case(tmp_path, *, mode="site_heldout", provenance=False):
    row = {"cohort": "ocec", "cutoff": 800, "selection_space": "raw",
           "spectra": "airspec", "mode": mode, "lot": "all", "target": "addis",
           "k": 2, "auto_k": 2, "heldout_R2": .9 if mode == "site_heldout" else None}
    schema, source = ("schema-v3", "source-fingerprint") if provenance else (None, None)
    fingerprint = "target-fingerprint"
    if provenance:
        row["provenance"] = {"cache_schema": schema, "source_fingerprint": source,
                             "target_fingerprint": fingerprint,
                             "resolved_cohort_hash": "cohort-hash"}
    curve_key = cache_key(("curve", "ocec", "raw", "airspec", mode, 30,
                           "all", "cohort-hash"), schema, source)
    (tmp_path / f"{curve_key}.json").write_text(json.dumps([
        {"n_components": 1, "rmsecv": 4., "rmse_se": .1},
        {"n_components": 2, "rmsecv": 2., "rmse_se": .1}]))
    parts = ("fit", curve_key, 2, "addis")
    if provenance:
        parts += (fingerprint,)
    n_fit = 623 if mode == "site_heldout" else 800
    fit = {"n_train": n_fit,
           "heldout": {"R2": .9, "RMSE": 2.75} if mode == "site_heldout" else None}
    (tmp_path / f"{cache_key(parts, schema, source)}.json").write_text(json.dumps(fit))
    return row, n_fit


@pytest.mark.parametrize("provenance", [False, True])
def test_recovers_exact_cache_counts_and_distinct_errors(tmp_path, provenance):
    row, n_fit = cached_case(tmp_path, provenance=provenance)
    metadata = cache_readout(row, "cohort-hash", 800, n_fit, tmp_path, [], {})
    assert metadata["n_train"] == 623
    assert metadata["n_heldout"] == 177
    assert metadata["heldout_RMSE"] == 2.75
    assert metadata["rmsecv_floor"] == 2.
    compacted = compact({**row, **metadata, "RMSE": .53})
    assert compacted["rmse"] == .53  # target concentrations stay separate
    assert compacted["cv_rmse"] == 2.75  # IMPROVE TOR loadings
    assert compacted["rmsecv"] == 2.


@pytest.mark.parametrize("field,value", [("k", 1), ("auto_k", 1),
                                         ("heldout_R2", .8)])
def test_rejects_cache_for_other_k_rule_or_holdout_score(tmp_path, field, value):
    row, n_fit = cached_case(tmp_path)
    row[field] = value
    assert cache_readout(row, "cohort-hash", 800, n_fit, tmp_path, [], {}) == {}


def test_rejects_count_or_membership_mismatch(tmp_path):
    row, n_fit = cached_case(tmp_path, provenance=True)
    assert cache_readout(row, "cohort-hash", 800, n_fit + 1, tmp_path, [], {}) == {}
    assert cache_readout(row, "another-cohort", 800, n_fit, tmp_path, [], {}) == {}


def test_interleaved_final_fit_has_no_outer_holdout(tmp_path):
    row, n_fit = cached_case(tmp_path, mode="app")
    metadata = cache_readout(row, "cohort-hash", 800, n_fit, tmp_path, [], {})
    assert metadata["n_heldout"] == 0
    assert metadata["heldout_RMSE"] is None
    assert metadata["rmsecv_floor"] == 2.


def test_missing_caches_do_not_create_counts_or_error(tmp_path):
    row, n_fit = cached_case(tmp_path)
    for p in tmp_path.iterdir():
        p.unlink()
    assert cache_readout(row, "cohort-hash", 800, n_fit, tmp_path, [], {}) == {}


def test_staged_resolver_counts_eligible_physical_filters_and_corrected_coverage(tmp_path, monkeypatch):
    scripts = tmp_path / "research/ftir_hips_chem/scripts"
    tables = scripts.parent / "output/tables"
    transfer = tables / "pls_transfer"
    phase2 = tables / "pls_calibration_phase2"
    corrected = tmp_path / "research/ftir_ec_phase3/output/corrected"
    for p in (transfer, phase2, corrected):
        p.mkdir(parents=True)
    monkeypatch.setattr(exporter, "REPO", tmp_path)
    monkeypatch.setattr(exporter, "SCRIPTS", scripts)
    monkeypatch.setattr(exporter, "EXPLORER", tmp_path / "explorer")
    ids = np.arange(1, 8)
    metadata = pd.DataFrame({"AnalysisId": ids, "Site": [f"S{i}" for i in ids],
                             "date": ["2020-01-01"] * 7,
                             "TOR_EC_loading_ug": [1] * 6 + [-1],
                             "TOR_EC_ugm3": [1] * 7, "TOR_OC_ugm3": ids,
                             "OC_EC_ratio": ids, "LotNumber": [251] * 7})
    metadata_path = tmp_path / "pool_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    pd.DataFrame({"AnalysisId": ids, "FilterId": [1, 2, 3, 4, 5, 5, 7],
                  "SampleDate": ["2020-01-01"] * 7,
                  "Site": metadata.Site, "analog_rank_score": ids}).to_csv(
                      transfer / "improve_full_pool_addis_similarity.csv", index=False)
    pd.DataFrame({"AnalysisId": ids, "Addis_band_feature_distance": ids}).to_csv(
        phase2 / "smoke_cohort_spectral_selection.csv", index=False)
    pd.DataFrame({"AnalysisId": [1, 2, 3]}).to_csv(
        phase2 / "locked_analog_train_test_split.csv", index=False)
    np.savez(corrected / "improve_pool_corrected_df6.npz", analysis_id=[1, 3, 4, 5])
    np.savez(corrected / "improve_pool_neutral_pspline_arpls_lam1e6.npz", analysis_id=ids)
    resolve = exporter.staged_cohort_resolver(metadata_path)
    pool, _ = resolve("pool", None, "raw", "raw", "all")
    assert pool.tolist() == [1, 2, 3, 4, 5]  # EC eligibility + physical-filter dedup
    raw, _ = resolve("ocec", 3, "raw", "raw", "all")
    baseline, _ = resolve("ocec", 3, "raw", "airspec", "all")
    assert raw.tolist() == [1, 2, 3]
    assert baseline.tolist() == [1, 3]  # real coverage after selection, not nominal 3
