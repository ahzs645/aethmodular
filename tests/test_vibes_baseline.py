"""Numerical parity and safeguards for the optional blank-trained baseline."""

from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("cvxpy", reason="Install the vibes extra for these tests")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from vibes_baseline import fit_vibes_background, vibes_baseline_matrix
from vibes.absorbance_estimators.vibes import VibeSpec


@pytest.fixture
def fixture_data():
    rng = np.random.default_rng(12)
    wn = np.linspace(3990, 1430, 180)
    t = (wn - wn.min()) / np.ptp(wn)
    blanks = 0.1 + rng.normal(0.2, 0.04, (12, 1)) * (1 + t) + rng.normal(0, 0.01, (12, 1)) * t**2
    blanks += rng.normal(0, 1e-4, blanks.shape)
    signal = 0.1 * np.exp(-0.5 * ((wn - 1700) / 30) ** 2)
    sample = blanks[[0]] + signal
    ids = [f"blank-{i}" for i in range(1, 12)]
    bg = fit_vibes_background(wn, blanks[1:], blank_ids=ids, max_components=3)
    return wn, blanks, sample, signal, bg


def test_matches_unmodified_upstream_and_preserves_signal(fixture_data):
    wn, _, y, signal, bg = fixture_data
    baseline, corrected, diag = vibes_baseline_matrix(wn, y, bg, sample_ids=["test"])
    assert diag.success.all(), diag.to_dict("records")
    mod = VibeSpec(c=bg.components.shape[1], loss="PB", tau_init=0.1)
    mod.fit(y[0], bg.mean, bg.components, tau_min=0.1, tau_max=0.1, mit=2000)
    expected, _ = mod.map_solver.solve(y[0], bg.mean, bg.components)
    np.testing.assert_allclose(baseline[0], expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(baseline + corrected, y, atol=1e-14)
    assert np.sqrt(np.mean((corrected - signal) ** 2)) < 0.005


def test_direction_invariance(fixture_data):
    wn, blanks, y, _, bg = fixture_data
    reverse_bg = fit_vibes_background(
        wn[::-1], blanks[1:, ::-1], blank_ids=bg.blank_ids, max_components=3
    )
    _, a, d1 = vibes_baseline_matrix(wn, y, bg, sample_ids=["test"])
    _, b, d2 = vibes_baseline_matrix(wn[::-1], y[:, ::-1], reverse_bg, sample_ids=["test"])
    assert d1.success.all() and d2.success.all()
    np.testing.assert_allclose(a, b[:, ::-1], atol=2e-6)


def test_prevents_training_leakage_and_misalignment(fixture_data):
    wn, _, y, _, bg = fixture_data
    with pytest.raises(ValueError, match="training blanks"):
        vibes_baseline_matrix(wn, y, bg, sample_ids=[bg.blank_ids[0]])
    with pytest.raises(ValueError, match="grids"):
        vibes_baseline_matrix(wn + 0.01, y, bg, sample_ids=["test"])
    with pytest.raises(ValueError, match="non-finite"):
        vibes_baseline_matrix(wn, y * np.nan, bg, sample_ids=["test"])


def test_failed_optimizer_is_flagged_not_silently_accepted(fixture_data):
    wn, _, y, _, bg = fixture_data
    baseline, corrected, diag = vibes_baseline_matrix(
        wn, y, bg, sample_ids=["test"], maxiter=1, retry_failed=False
    )
    assert not diag.success.any()
    assert np.isnan(baseline).all() and np.isnan(corrected).all()
    assert "ELBO optimization failed" in diag.message.iloc[0]


def test_invalid_blank_libraries(fixture_data):
    wn, blanks, _, _, _ = fixture_data
    with pytest.raises(ValueError, match="unique physical"):
        fit_vibes_background(wn, blanks, blank_ids=["same"] * len(blanks))
    with pytest.raises(ValueError, match="no background variation"):
        fit_vibes_background(wn, np.ones_like(blanks), blank_ids=range(len(blanks)))
    with pytest.raises(ValueError, match="monotonic"):
        fit_vibes_background(np.ones_like(wn), blanks, blank_ids=range(len(blanks)))


def test_restart_is_auditable_and_uses_same_objective(fixture_data, monkeypatch):
    import vibes_baseline as adapter

    wn, _, y, _, bg = fixture_data
    minimize = adapter.minimize
    calls = []

    def first_attempt_fails(fun, init, **kwargs):
        calls.append(fun)
        result = minimize(fun, init, **kwargs)
        if len(calls) == 1:
            result.success = False
            result.message = "Simulated optimizer stop for retry-path verification"
        return result

    monkeypatch.setattr(adapter, "minimize", first_attempt_fails)
    _, corrected, diag = adapter.vibes_baseline_matrix(wn, y, bg, sample_ids=["test"])
    assert len(calls) == 2 and calls[0] is calls[1]
    assert diag.success.all() and np.isfinite(corrected).all()
    assert diag.retry_count.iloc[0] == 1
    assert not diag.initial_optimizer_success.iloc[0]
    assert "Simulated" in diag.initial_message.iloc[0]
