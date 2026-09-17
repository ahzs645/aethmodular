"""Validate split integrity and influence comparisons, not plotting details."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from filter_relationship_stability import evaluate, fit_training, paired_influence, TARGET


def fixture_points():
    dates = (
        list(pd.date_range("2023-01-01", periods=12, freq="D"))
        + list(pd.date_range("2023-07-01", periods=12, freq="D"))
        + list(pd.date_range("2023-10-01", periods=12, freq="D"))
    )
    x = np.arange(len(dates), dtype=float) + 1
    d = pd.DataFrame(
        {
            "point_id": [f"Addis_Ababa:T{i}" for i in range(len(x))],
            "site": "Addis_Ababa",
            "date": dates,
            "ftir_ec_ugm3": x,
            "hips_fabs_Mm1": 3 + 2 * x,
            "eligible_filter_diagnostic": True,
        }
    )
    d.loc[0, "point_id"] = TARGET
    d["reported_date_block"] = d.date.dt.to_period("Q").astype(str)
    return d


def test_calendar_identity_and_future_leakage():
    p = fixture_points()
    m = p.assign(population="diagnostic")
    b, pred, _, _ = evaluate(p, m)
    base = b.loc[b.variant.eq("baseline") & b.population.eq("diagnostic")]
    assert base.loc[base.block.eq("2023Q2"), "status"].eq("empty_test_block").all()
    later = base.loc[base.scheme.eq("later_period") & base.status.eq("evaluated")]
    assert (later.train_date_max < later.test_date_min).all()
    import json

    assert all(
        not set(json.loads(r.train_ids)) & set(json.loads(r.test_ids)) for r in b.itertuples()
    )
    primary = pred.loc[pred.variant.eq("baseline") & pred.scheme.eq("leave_quarter_out")]
    assert primary.point_id.nunique() == 36
    assert np.allclose(primary.ols_error, 0, atol=1e-10)
    # Changing held-out HIPS must not change either model fitted for that fold.
    altered = p.copy()
    altered.loc[altered.reported_date_block.eq("2023Q3"), "hips_fabs_Mm1"] += 10000
    b2, _, _, _ = evaluate(altered, altered.assign(population="diagnostic"))

    def select(f):
        return f.loc[
            f.population.eq("diagnostic")
            & f.variant.eq("baseline")
            & f.block.eq("2023Q3")
            & f.scheme.eq("leave_quarter_out")
        ]

    for col in ["ols_slope", "ols_intercept", "training_median"]:
        assert np.allclose(select(b)[col], select(b2)[col])


def test_small_population_stays_unavailable():
    p = fixture_points().iloc[:7]
    b, pred, _, cohorts = evaluate(p, p.assign(population="mdl_2x"))
    assert pred.ols_prediction.isna().all()
    assert b.loc[b.test_n.gt(0), "status"].eq("insufficient_training_n").all()
    assert (
        cohorts.loc[cohorts.population.eq("mdl_2x") & cohorts.variant.eq("baseline"), "n"].item()
        == 7
    )
    assert fit_training(fixture_points().assign(ftir_ec_ugm3=1))["status"] == "constant_training_ec"


def test_individual_influence_uses_common_filters():
    p = fixture_points()
    p.loc[p.point_id.eq(TARGET), "hips_fabs_Mm1"] += 50
    b, pred, c, _ = evaluate(p, p.assign(population="diagnostic"))
    influence, paired = paired_influence(pred, c, b)
    assert not paired.point_id.eq(TARGET).any()
    own = paired.loc[paired.block.eq("2023Q1") & paired.scheme.eq("leave_quarter_out")]
    assert np.allclose(own.ols_prediction_baseline, own.ols_prediction_without)
    assert influence.common_n.gt(0).all()
