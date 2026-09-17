"""Training-only proportional fits, common metadata holdouts, and weighting estimands."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from filter_proportionality import proportional_fit, benchmark, summaries, id11_sensitivity
from filter_relationship_stability import evaluate


def fixture():
    dates = [
        *pd.date_range("2023-01-01", periods=12),
        *pd.date_range("2023-04-01", periods=12),
        *pd.date_range("2023-07-01", periods=12),
    ]
    x = np.arange(36, dtype=float) - 3
    p = pd.DataFrame(
        {
            "point_id": [f"Addis_Ababa:T{i}" for i in range(36)],
            "site": "Addis_Ababa",
            "date": dates,
            "ftir_ec_ugm3": x,
            "hips_fabs_Mm1": 4 + 2 * x,
            "eligible_filter_diagnostic": True,
            "ftir_CalibrationSetId": ["17"] * 12 + ["11"] * 24,
        }
    )
    p["reported_date_block"] = p.date.dt.to_period("Q").astype(str)
    m = p.assign(population="diagnostic")
    b, pred, _, _ = evaluate(p, m)
    return p, m, b, pred


def test_proportional_formula_and_nonpositive_preservation():
    p, m, b, old = fixture()
    new, pred = benchmark(b, old, m)
    assert (
        pred.loc[pred.point_id.eq("Addis_Ababa:T0"), "proportional_prediction"].dropna().lt(0).all()
    )
    assert pred.loc[pred.scheme.eq("leave_quarter_out")].point_id.nunique() == 36
    train = pd.DataFrame({"ftir_ec_ugm3": [-1.0, 0.0, 2.0], "hips_fabs_Mm1": [1.0, 3.0, 7.0]})
    assert proportional_fit(train) == pytest.approx(13 / 5)
    with pytest.raises(ValueError):
        proportional_fit(train.assign(ftir_ec_ugm3=0))
    baseline = old.loc[old.variant.eq("baseline")].set_index(
        ["site", "population", "scheme", "point_id"]
    )
    actual = pred.set_index(["site", "population", "scheme", "point_id"])
    pd.testing.assert_series_equal(actual.ols_prediction, baseline.ols_prediction)


def test_heldout_outcomes_cannot_change_proportional_training():
    p, m, b, old = fixture()
    new, _ = benchmark(b, old, m)
    changed = m.copy()
    changed.loc[changed.reported_date_block.eq("2023Q2"), "hips_fabs_Mm1"] += 10000
    new2, _ = benchmark(b, old, changed)

    def select(d):
        return d.loc[
            d.block.eq("2023Q2") & d.scheme.eq("leave_quarter_out") & d.population.eq("diagnostic"),
            "proportional_k",
        ]

    assert np.allclose(select(new), select(new2))


def test_id11_requires_common_support_and_strictly_earlier_data():
    p, m, b, old = fixture()
    blocks, pred, pairs, changes = id11_sensitivity(b, m)
    q2 = blocks.loc[blocks.block.eq("2023Q2") & blocks.population.eq("diagnostic")]
    assert not q2.common_support.any()  # All-earlier has 12, ID-11-earlier has zero.
    assert pred.loc[pred.block.eq("2023Q2"), "ols_prediction"].isna().all()
    q3 = blocks.loc[blocks.block.eq("2023Q3") & blocks.population.eq("diagnostic")]
    assert q3.common_support.all()
    assert q3.test_ids.nunique() == 1
    assert (q3.train_date_max < q3.test_date_min).all()
    assert pred.loc[pred.ols_prediction.notna(), "ftir_CalibrationSetId"].eq("11").all()
    assert len(pairs.loc[pairs.ols_error_all.notna()]) == 12


def test_equal_quarter_weighting_is_not_equal_filter_weighting():
    p, m, b, old = fixture()
    blocks, pred = benchmark(b, old, m)
    select = blocks.loc[
        blocks.population.eq("diagnostic") & blocks.scheme.eq("leave_quarter_out")
    ].copy()
    # Unequal evaluated blocks with known opposing biases.
    select = select.iloc[:2].copy()
    select["test_n"] = [1, 3]
    errors = [1.0, -1.0, -1.0, -1.0]
    pt = pd.DataFrame(
        {
            "site": "Addis_Ababa",
            "population": "diagnostic",
            "scheme": "leave_quarter_out",
            "block": ["2023Q1"] + ["2023Q2"] * 3,
        }
    )
    for model in ["median", "proportional", "ols"]:
        pt[model + "_error"] = errors
        pt[model + "_prediction"] = 5.0
        select[model + "_mean_signed_error"] = [1.0, -1.0]
        select[model + "_mae"] = 1.0
        select[model + "_rmse"] = 1.0
    result = summaries(select, pt).set_index("weighting")
    assert result.loc["equal_filter", "ols_mean_signed_error"] == -0.5
    assert result.loc["equal_quarter", "ols_mean_signed_error"] == 0
    assert result.loc["equal_filter", "largest_test_block_fraction"] == 0.75
