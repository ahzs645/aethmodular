"""Counterfactual tests for signal preservation and independent blank fitting."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from baseline_challenge import (
    fit_additive_blank_model, independent_blank_bank, injected_shapes,
    recovery_metrics, whittaker_arpls,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/workflows"))
from run_baseline_challenge import (
    METHODS, challenge_statistics, challenge_trial_metrics, mark_common_converged,
)


def bank_fixture():
    wn = np.linspace(3998, 1426, 500)
    rng = np.random.default_rng(5185)
    basis = np.array([np.sin(wn/150), np.cos(wn/90), np.sin(wn/310)])
    bank = .01 + .003*(wn/4000) + rng.normal(scale=.001, size=(10, 3)) @ basis
    bank += rng.normal(scale=1e-7, size=bank.shape)
    ids = np.array([f"blank-{i}" for i in range(10)])
    lots = np.array(["a"]*5 + ["b"]*4 + ["unknown"])
    return wn, bank, ids, lots


def test_physical_blank_holdout_never_includes_target():
    _, bank, ids, lots = bank_fixture()
    selected, train_ids, _ = independent_blank_bank(bank, ids, lots, ids[0])
    assert len(selected) == 9 and ids[0] not in train_ids
    changed = bank.copy()
    changed[0] += 100
    np.testing.assert_array_equal(independent_blank_bank(changed, ids, lots, ids[0])[0], selected)


def test_known_lot_holdout_excludes_unknown_and_same_lot():
    _, bank, ids, lots = bank_fixture()
    selected, _, selected_lots = independent_blank_bank(bank, ids, lots, ids[0], holdout="known_lot")
    assert len(selected) == 4 and set(selected_lots) == {"b"}
    with pytest.raises(ValueError, match="known target lot"):
        independent_blank_bank(bank, ids, lots, ids[-1], holdout="known_lot")


def test_repeat_scan_ids_cannot_inflate_blank_bank():
    _, bank, ids, lots = bank_fixture()
    ids[-1] = ids[0]
    with pytest.raises(ValueError, match="independent physical blank"):
        independent_blank_bank(bank, ids, lots, ids[0])


def test_background_model_rejects_training_filter_as_test():
    wn, bank, ids, _ = bank_fixture()
    model = fit_additive_blank_model(wn, bank, blank_ids=ids)
    with pytest.raises(ValueError, match="evaluation"):
        model.correct(bank[:1], sample_ids=[ids[0]])


def test_protected_band_prevents_local_signal_from_changing_background_fit():
    wn, bank, ids, _ = bank_fixture()
    model = fit_additive_blank_model(wn, bank, blank_ids=ids, protected=True)
    signal = np.where((wn >= 1580) & (wn <= 1650), .01, 0.0)
    sample = bank.mean(axis=0) + .0001*np.sin(wn/80)
    result = model.correct(np.array([sample, sample+signal]), sample_ids=["heldout", "heldout"])
    np.testing.assert_allclose(result[1]-result[0], signal, atol=1e-12)


def test_additive_linear_continuum_is_unidentifiable_even_with_protected_bands():
    wn, bank, ids, _ = bank_fixture()
    model = fit_additive_blank_model(wn, bank, blank_ids=ids, protected=True)
    signal = .01 * injected_shapes(wn)["broad_linear_continuum"]
    result = model.correct(np.array([bank.mean(axis=0), bank.mean(axis=0)+signal]),
                           sample_ids=["heldout", "heldout"])
    assert abs(recovery_metrics(result[0], result[1], signal)["signal_gain"]) < 1e-10


def test_gain_measures_increment_and_detects_shape_loss():
    signal = np.linspace(.001, .01, 100)
    background = np.linspace(-100, 50, 100)
    metrics = recovery_metrics(background, background+.4*signal, signal)
    assert metrics["signal_gain"] == pytest.approx(.4)
    assert metrics["relative_recovery_error"] == pytest.approx(.6)
    assert metrics["relative_shape_distortion"] < 1e-10


def test_arpls_is_equivariant_to_signal_sign_independent_offset_and_channel_reversal():
    rng = np.random.default_rng(34)
    x = np.linspace(0, 1, 400)
    y = .01*x + .1*np.exp(-.5*((x-.5)/.01)**2) + rng.normal(scale=1e-4, size=400)
    result, status = whittaker_arpls(y, lam=1e5)
    shifted, status2 = whittaker_arpls(y+.05, lam=1e5)
    reversed_result, status3 = whittaker_arpls(y[::-1], lam=1e5)
    assert status["converged"] and status2["converged"] and status3["converged"]
    np.testing.assert_allclose(result, shifted, atol=1e-8)
    np.testing.assert_allclose(result, reversed_result[::-1], atol=1e-8)
    assert result.max() > .08


def test_arpls_iteration_limit_is_not_marked_success():
    x = np.linspace(0, 1, 200)
    _, status = whittaker_arpls(.03*x + np.exp(-.5*((x-.6)/.05)**2), max_iter=1)
    assert not status["converged"] and status["reason"] == "iteration_limit"


def test_arpls_linear_input_returns_zero_signed_residual():
    result, status = whittaker_arpls(np.linspace(-.01, .03, 100), lam=1e4)
    assert status["converged"]
    np.testing.assert_allclose(result, 0, atol=1e-8)


def test_nonfinite_solver_trial_is_retained_without_recovery_metrics():
    signal = np.ones(10)
    metrics = challenge_trial_metrics(np.zeros(10), np.full(10, np.nan), signal,
                                      original_converged=True, injected_converged=False)
    assert not metrics["converged"]
    assert metrics["metric_status"] == "nonfinite_correction"
    assert np.isnan(metrics["signal_gain"])
    rows = pd.DataFrame([{"holdout": "filter", "method": "arpls_1e4", "shape": "test",
                          "amplitude": 1.0, **metrics}])
    summary = challenge_statistics(rows).iloc[0]
    assert summary.trials == 1 and summary.failed_trials == 1 and summary.converged_trials == 0
    assert np.isnan(summary.signal_gain_median)


def test_common_cohort_rejects_failed_or_missing_methods():
    rows = []
    for fid in ("complete", "failed", "missing"):
        for method in METHODS:
            if fid == "missing" and method == METHODS[-1]:
                continue
            rows.append({"holdout": "filter", "id": fid, "shape": "test", "amplitude": .01,
                         "method": method, "converged": not (fid == "failed" and method == METHODS[1])})
    marked = mark_common_converged(pd.DataFrame(rows))
    assert marked.loc[marked.id.eq("complete"), "common_converged"].all()
    assert not marked.loc[marked.id.isin(["failed", "missing"]), "common_converged"].any()
