"""Scientific audit guards: pairing, train/test separation and cluster inference."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from vibes_error_audit import loading_bands, metrics, pair_predictions


@pytest.fixture
def frozen_pair():
    cases = pd.DataFrame(dict(
        sample_id=["train", "test1", "test2"], Site=["A", "B", "C"], lot=[248]*3,
        y=[1., 2., 3.], kind=["calibration"]*3, split=["train", "test", "test"],
        locked800=[True]*3, paired_valid=[True]*3,
    ))
    rows = [dict(cohort=c, sample_id=s, method=m, Site=site, y=y, prediction=y+.5)
            for c in ("full_pool", "locked800") for m in ("AIRSpec", "VIBES")
            for s, site, y in [("test1", "B", 2.), ("test2", "C", 3.)]]
    return pd.DataFrame(rows), cases


def test_pairing_is_order_independent_and_complete(frozen_pair):
    pred, cases = frozen_pair
    got = pair_predictions(pred.sample(frac=1, random_state=2), cases)
    assert len(got) == 4
    assert (got.AIRSpec == got.VIBES).all()
    with pytest.raises(ValueError, match="membership"):
        pair_predictions(pred.iloc[:-1], cases)
    with pytest.raises(ValueError, match="unique"):
        pair_predictions(pd.concat([pred, pred.iloc[:1]]), cases)


def test_changed_truth_and_site_leakage_rejected(frozen_pair):
    pred, cases = frozen_pair
    bad = pred.copy()
    bad.loc[0, "y"] = 100
    with pytest.raises(ValueError, match="truth"):
        pair_predictions(bad, cases)
    cases.loc[0, "Site"] = "B"
    with pytest.raises(ValueError, match="overlap"):
        pair_predictions(pred, cases)


def test_training_quartiles_and_boundary_ownership():
    labels, cuts = loading_bands([0., 4., 8., 12., 16.], [-100., 4., 4.001, 8., 12., 1000.])
    np.testing.assert_array_equal(cuts, [4., 8., 12.])
    assert list(labels) == ["Q1", "Q1", "Q2", "Q2", "Q3", "Q4"]


def test_site_bootstrap_matches_explicit_cluster_resampling():
    frame = pd.DataFrame(dict(Site=["A", "A", "A", "B"], y=[1., 3., 2., 4.],
                              AIRSpec=[2., 4., 3., 2.], VIBES=[1., 2., 2., 7.]))
    result = metrics(frame, repeats=100, seed=13)
    draws = np.random.default_rng(13).integers(0, 2, (100, 2))
    values = {k: [] for k in ("RMSE", "MAE", "bias")}
    for draw in draws:
        sample = pd.concat([frame.loc[frame.Site.eq(["A", "B"][i])] for i in draw])
        a, v = sample.AIRSpec - sample.y, sample.VIBES - sample.y
        values["RMSE"].append(np.sqrt(np.mean(v**2))-np.sqrt(np.mean(a**2)))
        values["MAE"].append(np.mean(abs(v))-np.mean(abs(a)))
        values["bias"].append(np.mean(v)-np.mean(a))
    for key, samples in values.items():
        np.testing.assert_allclose([result[f"delta_{key}_ci_low"], result[f"delta_{key}_ci_high"]], np.quantile(samples, [.025, .975]))
    single = metrics(frame.loc[frame.Site.eq("A")])
    assert np.isnan(single["delta_RMSE_ci_low"])
    assert single["interval_status"] == "unavailable_single_site"
