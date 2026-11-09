"""Reference-panel population, independence, QC and source-simulation controls."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from conference_experiments.selection import build_reference_panel, simulate_reference_selection


def fixture():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(96, 6))
    X[:, 0] += np.tile(np.linspace(0, 4, 12), 8)
    meta = pd.DataFrame({"filter_id": [f"F{i:03d}" for i in range(len(X))],
        "site": np.repeat([f"S{i}" for i in range(8)], 12), "ec_ug": 10+2*X[:, 0],
        "season": np.tile(["A", "B", "C"], 32), "lot": np.tile(["L1", "L2"], 48),
        "is_excluded": False, "exclusion_reason": "", "is_blank": False})
    return X, meta


def selected(result):
    frame = result["tables"]["manifest"]
    return frame.loc[frame.panel.isin(["representative", "challenge"]), ["filter_id", "panel", "selection_reason"]].reset_index(drop=True)


def test_disjoint_representative_and_challenge_panels_have_exact_budget_and_probabilities():
    X, meta = fixture()
    result = build_reference_panel(X, meta, representative_n=12, challenge_n=8,
                                   strata=("season", "lot"), blank_n=2, replicate_n=3)
    assert result["summary"]["status"] == "completed"
    manifest = result["tables"]["manifest"]
    counts = manifest.panel.value_counts()
    assert counts["representative"] == 12
    assert counts["challenge"] == 8
    assert manifest.filter_id.is_unique
    assert manifest.representative_inclusion_probability.eq(12/96).all()
    challenge = manifest.query("panel == 'challenge'")
    assert {"low_loading_proxy", "high_loading_proxy", "spectral_extreme"} <= set(challenge.selection_reason)
    assert result["summary"]["reference_assays"] == 20
    assert result["summary"]["extra_requested_qc_assays"] == 5
    assert result["summary"]["missing_blank_specimens"] == 2
    blank_plan = result["tables"]["qc_plan"].query("assay_role == 'blank'")
    assert blank_plan.filter_id.isna().all()
    assert blank_plan.status.eq("specimen_not_available").all()
    repeats = result["tables"]["qc_plan"].query("assay_role == 'blind_repeat'")
    assert len(repeats) == 3
    assert set(repeats.filter_id) <= set(selected(result).filter_id)


def test_reference_labels_and_extra_outcomes_cannot_change_the_plan():
    X, meta = fixture()
    first = build_reference_panel(X, meta, representative_n=12, challenge_n=8, strata=("season", "lot"))
    changed = meta.assign(ec_ug=np.arange(len(meta))*1000, residual=np.arange(len(meta))[::-1])
    second = build_reference_panel(X, changed, representative_n=12, challenge_n=8, strata=("season", "lot"))
    pd.testing.assert_frame_equal(selected(first), selected(second))
    for kwargs in ({"loading_proxy_col": "ec_ug"}, {"strata": ("ec_ug",)}, {"strata": ("reference_ec",)}):
        with pytest.raises(ValueError, match="independent reference"):
            build_reference_panel(X, changed, representative_n=12, challenge_n=8, **kwargs)


def test_row_order_does_not_change_representative_or_challenge_identities():
    X, meta = fixture()
    first = build_reference_panel(X, meta, representative_n=10, challenge_n=9, strata=("lot",))
    order = np.random.default_rng(52).permutation(len(X))
    second = build_reference_panel(X[order], meta.iloc[order], representative_n=10, challenge_n=9, strata=("lot",))
    pd.testing.assert_frame_equal(selected(first), selected(second))


def test_small_budget_fallback_preserves_nonzero_probability_for_all_eligible_filters():
    X, meta = fixture()
    result = build_reference_panel(X, meta, representative_n=2, challenge_n=2, strata=("site", "season", "lot"))
    assert result["summary"]["representative_strategy"].startswith("simple_random_fallback")
    assert result["tables"]["manifest"].representative_inclusion_probability.eq(2/96).all()


def test_canonical_flags_unavailable_material_and_blanks_have_separate_roles():
    X, meta = fixture()
    meta.loc[0, ["is_excluded", "exclusion_reason"]] = [True, "damaged"]
    meta.loc[1, "is_blank"] = True
    meta["available_for_assay"] = True
    meta.loc[2, "available_for_assay"] = False
    meta["is_outlier"], meta["outlier_reason"] = False, ""
    meta.loc[3, ["is_outlier", "outlier_reason"]] = [True, "prior_threshold"]
    result = build_reference_panel(X, meta, representative_n=20, challenge_n=20, blank_n=2)
    manifest = result["tables"]["manifest"].set_index("filter_id")
    assert manifest.loc["F000", "panel"] == "not_selected"
    assert manifest.loc["F001", "panel"] == "qc_blank"
    assert manifest.loc["F002", "panel"] == "not_selected"
    assert manifest.loc["F003", "panel"] == "not_selected"
    assert "specimen_unavailable" in manifest.loc["F002", "eligibility_reason"]
    scan_ledger = result["tables"]["scan_ledger"].set_index("filter_id")
    assert not scan_ledger.loc["F002", "experiment_eligible"]
    assert scan_ledger.loc["F002", "eligibility_reason"] == manifest.loc["F002", "eligibility_reason"]
    assert result["summary"]["blank_specimens_available"] == 1
    assert len(selected(result)) == 40


def test_duplicate_physical_scans_do_not_increase_assay_count_or_duplicate_samples():
    X, meta = fixture()
    repeat = meta.iloc[[0, 2, 3]].copy()
    repeat["scan_note"] = "repeat"
    result = build_reference_panel(np.vstack([X, X[[0, 2, 3]]]), pd.concat([meta, repeat]),
                                   representative_n=12, challenge_n=8)
    manifest = result["tables"]["manifest"]
    assert len(manifest) == 96
    assert manifest.n_scans.sum() == 99
    assert len(selected(result)) == 20


def test_infeasible_panel_retains_ledger_and_claims_no_completed_selection():
    X, meta = fixture()
    result = build_reference_panel(X, meta, representative_n=70, challenge_n=30)
    assert result["summary"]["status"] == "insufficient_data"
    assert len(result["tables"]["manifest"]) == len(meta)
    assert "panel" not in result["tables"]["manifest"]


def simulation(X, meta, **options):
    kwargs = dict(reference_kind="thermal_optical_ec", test_sites=["S6", "S7"], budget=30,
        strata=("season", "lot"), seeds=(23,), models=("ridge",), inner_folds=3,
        model_options={"ridge": [{"n_components": 3, "alpha": 1.}]}, bootstrap_repeats=20)
    kwargs.update(options)
    return simulate_reference_selection(X, meta, **kwargs)


def test_simulation_keeps_equal_budgets_and_same_test_sites_for_each_strategy():
    X, meta = fixture()
    result = simulation(X, meta)
    assert result["summary"]["status"] == "completed"
    chosen = result["tables"]["selection"].query("panel in ['representative','challenge']")
    assert chosen.groupby(["strategy", "seed"]).size().eq(30).all()
    expected_ids = set(meta.loc[meta.site.isin(["S6", "S7"]), "filter_id"])
    assert not set(chosen.filter_id) & expected_ids
    for _, frame in result["tables"]["predictions"].groupby(["strategy", "seed", "model"]):
        assert set(frame.filter_id) == expected_ids
    assert result["tables"]["paired_contrasts"].comparison.eq("strategy_minus_random").all()
    assert not result["summary"]["selection_uses_reference_labels"]


def test_simulation_selection_ignores_training_labels_and_heldout_covariates():
    X, meta = fixture()
    first = simulation(X, meta)
    modified = meta.copy()
    modified["ec_ug"] += np.sin(np.arange(len(meta)))*500
    shifted = X.copy()
    shifted[meta.site.isin(["S6", "S7"])] += 10000
    second = simulation(shifted, modified)
    columns = ["filter_id", "panel", "selection_reason", "strategy", "seed"]
    pd.testing.assert_frame_equal(first["tables"]["selection"][columns], second["tables"]["selection"][columns])


def test_simulation_rejects_ftir_reference_and_missing_test_identity():
    X, meta = fixture()
    with pytest.raises(ValueError, match="thermal_optical_ec"):
        simulation(X, meta, reference_kind="FTIR_EC")
    with pytest.raises(ValueError, match="missing"):
        simulation(X, meta, test_sites=["unavailable"])


def test_simulation_fit_failure_preserves_failure_ledger(monkeypatch):
    from conference_experiments import selection

    def fail(*args, **kwargs):
        raise ValueError("synthetic estimator convergence failure")

    monkeypatch.setattr(selection, "fit_source_model", fail)
    X, meta = fixture()
    result = simulation(X, meta)
    assert result["summary"]["status"] == "failed"
    assert result["summary"]["completed_runs"] == 0
    assert result["tables"]["runs"].status.eq("failed").all()
    assert result["tables"]["runs"].reason.eq("synthetic estimator convergence failure").all()
    assert len(result["tables"]["physical_ledger"]) == len(meta)
