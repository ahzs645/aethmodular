"""Synthetic checks of the response-exponent, triple-collocation and saturation helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
import reference_closure as rc  # noqa: E402


def _instruments(n=4000, seed=0, exponents=(1.0, 1.0, 0.45), noise=(0.15, 0.12, 0.08)):
    """Three instruments reading one lognormal truth with given power-law responses and independent errors."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(0.0, 0.45, n)
    return [np.exp(b * truth + rng.normal(0.0, s, n)) for b, s in zip(exponents, noise)], truth


def test_response_exponent_is_not_flattened_by_noise_in_x():
    (a, b, c), _ = _instruments(noise=(0.15, 0.30, 0.08))
    # a log-log regression of the compressed instrument on a noisy x is biased low; the instrumented estimate is not
    ols = np.polyfit(np.log(b), np.log(c), 1)[0]
    result = rc.response_exponent(c, b, a, n_boot=200)
    assert ols < 0.36
    assert result["exponent"] == pytest.approx(0.45, abs=0.03)
    assert result["lo"] < 0.45 < result["hi"]


def test_response_exponent_of_proportional_instruments_is_one():
    (a, b, c), _ = _instruments()
    assert rc.response_exponent(a, b, c, n_boot=200)["exponent"] == pytest.approx(1.0, abs=0.05)


def test_response_exponent_drops_missing_and_nonpositive_rows():
    (a, b, c), _ = _instruments(n=500)
    a = a.copy()
    a[:10], b[10:20], c[20:30] = np.nan, 0.0, -1.0
    assert rc.response_exponent(a, b, c, n_boot=50)["n"] == 470
    with pytest.raises(ValueError):
        rc.response_exponent(a[:5], b[:5], c[:5])


def test_triple_collocation_recovers_errors_and_exponents():
    (a, b, c), _ = _instruments(n=20000)
    result = rc.triple_collocation(a, b, c, n_boot=100)
    first, second, third = result["measurements"]
    assert result["signal_sd"] == pytest.approx(0.45, abs=0.02)
    assert first["exponent"] == 1.0
    assert second["exponent"] == pytest.approx(1.0, abs=0.03)
    assert third["exponent"] == pytest.approx(0.45, abs=0.02)
    assert first["error_sd"] == pytest.approx(0.15, abs=0.02)
    assert second["error_sd"] == pytest.approx(0.12, abs=0.02)
    # the compressed instrument's error is reported on the first instrument's scale: 0.08 / 0.45
    assert third["error_sd"] == pytest.approx(0.08 / 0.45, abs=0.02)


def test_fit_response_prefers_the_saturating_curve_and_recovers_tau_inf():
    rng = np.random.default_rng(3)
    loading = np.exp(rng.normal(3.6, 0.45, 400))
    tau = rc.hyperbolic(loading, 0.04, 2.0) * np.exp(rng.normal(0.0, 0.05, loading.size))
    curve = rc.fit_response(loading, tau, "hyperbolic", n_boot=50)
    line = rc.fit_response(loading, tau, "proportional", n_boot=0)
    assert curve["params"][1] == pytest.approx(2.0, rel=0.1)
    assert curve["aic"] < line["aic"] - 100
    assert curve["intervals"][1][0] < 2.0 < curve["intervals"][1][1]


def test_fit_response_does_not_invent_a_bend():
    rng = np.random.default_rng(4)
    loading = np.exp(rng.normal(3.6, 0.45, 400))
    tau = 0.01 * loading * np.exp(rng.normal(0.0, 0.05, loading.size))
    power = rc.fit_response(loading, tau, "power", n_boot=0)
    assert power["params"][1] == pytest.approx(1.0, abs=0.03)


def test_linearise_inverts_the_hyperbolic_response():
    true = np.array([0.1, 0.5, 1.0, 3.0])
    measured = rc.hyperbolic(true, 1.0, 2.0)
    assert np.allclose(rc.linearise(measured, 2.0), true)
    assert np.isnan(rc.linearise(np.array([2.0, 2.5]), 2.0)).all()
    # the patterned-deposit form is also proportional at low loading
    assert rc.patterned(np.array([1e-4]), 1.0, 0.8)[0] == pytest.approx(1e-4, rel=1e-3)
