"""Synthetic checks of the calibration response, fair scoring and site-held-out recalibration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
import calibration_gain as cg  # noqa: E402


def _network(seed=0, n_sites=40, per_site=60, gain=0.8, offset=1.5, noise=0.4):
    """Sites with different typical loadings; a 'prediction' that under-responds to EC and adds an offset."""
    rng = np.random.default_rng(seed)
    level = np.exp(rng.normal(1.2, 0.8, n_sites))
    sites = np.repeat([f"S{i:02d}" for i in range(n_sites)], per_site)
    ec = np.repeat(level, per_site) * np.exp(rng.normal(0, 0.5, n_sites * per_site))
    dates = np.tile(pd.date_range("2024-01-01", periods=per_site, freq="3D"), n_sites)
    measured = ec * np.exp(rng.normal(0, 0.25, ec.size))          # the reference has its own per-filter error
    prediction = offset + gain * ec + rng.normal(0, noise, ec.size)
    return sites, dates, ec, measured, prediction


def test_response_recovers_gain_and_offset_despite_reference_error():
    sites, _, _, measured, prediction = _network()
    fit = cg.response(prediction, measured, measured, sites, with_oc=False, n_boot=200)
    # a per-filter regression on the noisy reference would be flattened; site means are not
    per_filter = np.polyfit(measured, prediction, 1)[0]
    assert per_filter < 0.74
    assert fit["gain"] == pytest.approx(0.8, abs=0.05)
    assert fit["offset"] == pytest.approx(1.5, abs=0.5)
    assert fit["gain_lo"] < fit["gain"] < fit["gain_hi"]


def test_recalibration_uses_other_sites_only_and_restores_heavy_filters():
    sites, dates, _, measured, prediction = _network()
    folds = cg.site_folds(sites)
    for train, test in folds:
        assert not set(sites[train]) & set(sites[test])
    before = cg.fair_scores(prediction, measured, sites, dates)
    after = cg.fair_scores(cg.recalibrate_by_site(prediction, measured, sites, folds), measured, sites, dates)
    assert before["heavy_sum_ratio"] < 0.92
    assert after["heavy_sum_ratio"] == pytest.approx(1.0, abs=0.06)
    assert cg.response(cg.recalibrate_by_site(prediction, measured, sites, folds), measured, measured, sites, with_oc=False, n_boot=0)["gain"] == pytest.approx(1.0, abs=0.06)


def test_fair_groups_do_not_select_on_the_filters_own_reference():
    # an unbiased prediction: choosing heavy filters by their own noisy reference makes it look low, the fair grouping does not
    sites, dates, ec, measured, _ = _network(gain=1.0, offset=0.0)
    unbiased = ec.copy()
    own = measured >= cg.HEAVY_UG
    assert np.median(unbiased[own] / measured[own]) < 0.97
    assert cg.fair_scores(unbiased, measured, sites, dates)["heavy_level"] == pytest.approx(1.0, abs=0.04)


def test_held_out_predictions_and_weights():
    rng = np.random.default_rng(1)
    sites = np.repeat([f"S{i}" for i in range(10)], 30)
    ec = np.exp(rng.normal(1.5, 0.7, sites.size))
    basis = rng.normal(size=(3, 40))
    x = np.outer(ec, basis[0]) + rng.normal(size=(sites.size, 2)) @ basis[1:] + rng.normal(0, 0.01, (sites.size, 40))
    folds = cg.site_folds(sites)
    blanks = rng.normal(0, 0.01, (12, 40))
    held_out, extra = cg.held_out_predictions(x, ec, folds, 4, weights=cg.loading_weights(ec), extra_x=blanks, extra_y=np.zeros(12),
                                              extra_weight=5.0, predict={"blanks": blanks})
    assert held_out.shape == (sites.size, 4) and np.isfinite(held_out).all()
    assert np.corrcoef(held_out[:, -1], ec)[0, 1] > 0.99
    assert abs(np.median(extra["blanks"][:, -1])) < 0.2
    weights = cg.loading_weights(ec)
    assert weights.min() == 1.0 and weights.max() <= 8.0
