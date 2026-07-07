"""Tests for the consolidated src/common helpers."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common import (
    base_filter_id,
    deming,
    deming_lambda,
    find_repo_root,
    fit_fabs_ec,
    normalize_filter_id,
    regression_stats,
    season_for_month,
    to_ugm3,
)


class TestDeming:
    def test_perfect_line_recovers_slope_intercept(self):
        x = np.arange(1.0, 50.0)
        y = 2.0 * x + 3.0
        slope, intercept = deming(x, y, lam=1.0)
        assert slope == pytest.approx(2.0, abs=1e-6)
        assert intercept == pytest.approx(3.0, abs=1e-6)

    def test_drops_non_finite(self):
        x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, np.inf])
        slope, intercept = deming(x, y, lam=1.0)
        assert slope == pytest.approx(2.0, abs=1e-6)

    def test_too_few_points_returns_nan(self):
        slope, intercept = deming([1.0, 2.0], [2.0, 4.0])
        assert np.isnan(slope) and np.isnan(intercept)

    def test_lambda_from_sigmas(self):
        assert deming_lambda(0.2, 1.0) == pytest.approx(25.0)


class TestRegressionStats:
    def test_basic_ols(self):
        x = np.arange(1.0, 20.0)
        y = 3.0 * x + 1.0
        s = regression_stats(x, y)
        assert s["n"] == 19
        assert s["slope"] == pytest.approx(3.0, abs=1e-6)
        assert s["intercept"] == pytest.approx(1.0, abs=1e-6)
        assert s["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_infinities_dropped(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, np.inf])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = regression_stats(x, y)
        assert s["n"] == 4

    def test_positive_only_filters_nonpositive(self):
        x = np.array([-1.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([-5.0, 1.0, 2.0, 3.0, 4.0])
        assert regression_stats(x, y, positive_only=True)["n"] == 4
        # default keeps the negative pair
        assert regression_stats(x, y)["n"] == 5

    def test_too_few_returns_nan_slope(self):
        s = regression_stats([1.0, 2.0], [1.0, 2.0])
        assert s["n"] == 2 and np.isnan(s["slope"])


class TestFitFabsEc:
    def test_returns_ols_and_deming(self):
        ec = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        fabs = 8.0 * ec + 2.0
        out = fit_fabs_ec(ec, fabs)
        assert out["slope"] == pytest.approx(8.0, abs=1e-6)
        assert out["deming_slope"] == pytest.approx(8.0, abs=1e-3)
        assert out["lambda"] == pytest.approx(deming_lambda(0.2, 1.0))


class TestFilterIds:
    @pytest.mark.parametrize("raw,expected", [
        ("ETAD-0035-3", "ETAD-0035"),
        ("ETAD-0035", "ETAD-0035"),
        ("MEXX-1234-12", "MEXX-1234"),
        (None, None),
        (np.nan, None),
    ])
    def test_base_filter_id(self, raw, expected):
        assert base_filter_id(raw) == expected

    def test_normalize_filter_id_collapses_extra_suffix(self):
        assert normalize_filter_id("ETAD-0035-3-extra") == "ETAD-0035"
        assert normalize_filter_id("SITE") == "SITE"
        assert normalize_filter_id(None) is None


class TestUnits:
    def test_ngm3_autoconverted(self):
        s = pd.Series([1000.0, 2000.0, 3000.0])  # median 2000 > 100 -> ng/m3
        out = to_ugm3(s)
        assert out.tolist() == [1.0, 2.0, 3.0]

    def test_ugm3_left_alone(self):
        s = pd.Series([1.0, 2.0, 3.0])
        out = to_ugm3(s)
        assert out.tolist() == [1.0, 2.0, 3.0]

    def test_non_numeric_coerced(self):
        out = to_ugm3(pd.Series(["a", "2.0", None]))
        assert out.isna().tolist() == [True, False, True]


class TestSeasons:
    @pytest.mark.parametrize("month,expected_prefix", [
        (1, "Dry"), (2, "Dry"), (10, "Dry"), (12, "Dry"),
        (3, "Belg"), (4, "Belg"), (5, "Belg"),
        (6, "Kiremt"), (8, "Kiremt"), (9, "Kiremt"),
    ])
    def test_addis_calendar(self, month, expected_prefix):
        assert season_for_month(month).startswith(expected_prefix)

    def test_unknown_site_raises(self):
        with pytest.raises(KeyError):
            season_for_month(1, site="Atlantis")


class TestPaths:
    def test_finds_repo_root(self):
        root = find_repo_root(__file__)
        assert (root / "pyproject.toml").exists()
