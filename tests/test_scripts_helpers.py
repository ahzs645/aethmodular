"""Tests for the consolidated helpers in research/ftir_hips_chem/scripts.

These replace the notebook-inlined copies of deming, season lookup, filter-id
normalization, unit coercion, and repo-root discovery. The scripts package is
the sanctioned reusable-logic home for the active research area (see AGENTS.md),
so the tests add it to sys.path the same way the notebooks do.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..",
                 "research", "ftir_hips_chem", "scripts")
)
sys.path.insert(0, _SCRIPTS)

from plotting.utils import (                                 # noqa: E402
    deming, deming_lambda, calculate_regression_stats,
)
from config import ETHIOPIA_SEASONS, season_for_month        # noqa: E402
from data_matching import base_filter_id, normalize_filter_id  # noqa: E402
from prep import to_ugm3, find_repo_root                     # noqa: E402


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
        slope, _ = deming(x, y, lam=1.0)
        assert slope == pytest.approx(2.0, abs=1e-6)

    def test_too_few_points_returns_nan(self):
        slope, intercept = deming([1.0, 2.0], [2.0, 4.0])
        assert np.isnan(slope) and np.isnan(intercept)

    def test_lambda_from_sigmas(self):
        assert deming_lambda(0.2, 1.0) == pytest.approx(25.0)


def _improve_regression_stats(df, x_col, y_col):
    """Verbatim copy of the improve_hips_offset inline variant, for equivalence
    testing against the consolidated calculate_regression_stats."""
    d = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d[x_col] > 0) & (d[y_col] > 0)]
    if len(d) < 3:
        return {'n': len(d), 'slope': np.nan, 'intercept': np.nan,
                'r2': np.nan, 'origin_slope': np.nan}
    x = d[x_col].to_numpy(float)
    y = d[y_col].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    origin_slope = np.sum(x * y) / np.sum(x ** 2)
    return {'n': int(len(d)), 'slope': slope, 'intercept': intercept,
            'r2': 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            'origin_slope': origin_slope}


class TestCalculateRegressionStats:
    def test_array_form_backward_compatible(self):
        x = np.arange(1.0, 20.0)
        y = 3.0 * x + 1.0
        s = calculate_regression_stats(x, y)
        assert s['n'] == 19
        assert s['slope'] == pytest.approx(3.0, abs=1e-9)
        assert s['intercept'] == pytest.approx(1.0, abs=1e-9)
        assert s['r_squared'] == pytest.approx(1.0, abs=1e-12)
        assert s['r2'] == s['r_squared']          # alias
        assert 'correlation' in s and 'origin_slope' in s

    def test_dataframe_form(self):
        df = pd.DataFrame({'a': np.arange(1.0, 20.0), 'b': 2.0 * np.arange(1.0, 20.0)})
        s = calculate_regression_stats(df, 'a', 'b')
        assert s['slope'] == pytest.approx(2.0, abs=1e-9)

    def test_inf_dropped(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, np.inf])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert calculate_regression_stats(x, y)['n'] == 4

    def test_positive_only(self):
        x = np.array([-1.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([-5.0, 1.0, 2.0, 3.0, 4.0])
        assert calculate_regression_stats(x, y, positive_only=True)['n'] == 4
        assert calculate_regression_stats(x, y)['n'] == 5

    def test_too_few_returns_none(self):
        assert calculate_regression_stats([1.0, 2.0], [1.0, 2.0]) is None

    @pytest.mark.parametrize("seed_shift", [0.0, 3.7, -2.1])
    def test_matches_improve_variant(self, seed_shift):
        # representative data with negatives, a zero, and an inf that both impls drop
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf]) + 0.0
        y = np.array([-2.0, 1.0, 1.4, 8.0 + seed_shift, 18.0, 26.0, 33.0, 41.0, 7.0])
        df = pd.DataFrame({'EC': x, 'FABS': y})
        got = calculate_regression_stats(df, 'EC', 'FABS', positive_only=True)
        exp = _improve_regression_stats(df, 'EC', 'FABS')
        assert got['n'] == exp['n']
        assert got['slope'] == pytest.approx(exp['slope'], rel=1e-9)
        assert got['intercept'] == pytest.approx(exp['intercept'], rel=1e-9)
        assert got['r2'] == pytest.approx(exp['r2'], rel=1e-9)
        assert got['origin_slope'] == pytest.approx(exp['origin_slope'], rel=1e-9)


class TestSeasons:
    @pytest.mark.parametrize("month,prefix", [
        (1, "Dry"), (2, "Dry"), (10, "Dry"), (12, "Dry"),
        (3, "Belg"), (4, "Belg"), (5, "Belg"),
        (6, "Kiremt"), (8, "Kiremt"), (9, "Kiremt"),
    ])
    def test_addis_calendar(self, month, prefix):
        assert season_for_month(month).startswith(prefix)

    def test_every_month_covered(self):
        assert all(season_for_month(m) is not None for m in range(1, 13))

    def test_table_shape(self):
        assert set(ETHIOPIA_SEASONS) == {
            "Dry (Oct-Feb)", "Belg (Mar-May)", "Kiremt (Jun-Sep)"
        }


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

    def test_normalize_filter_id(self):
        assert normalize_filter_id("ETAD-0035-3-extra") == "ETAD-0035"
        assert normalize_filter_id("SITE") == "SITE"
        assert normalize_filter_id(None) is None


class TestUnits:
    def test_ngm3_autoconverted(self):
        assert to_ugm3(pd.Series([1000.0, 2000.0, 3000.0])).tolist() == [1.0, 2.0, 3.0]

    def test_ugm3_left_alone(self):
        assert to_ugm3(pd.Series([1.0, 2.0, 3.0])).tolist() == [1.0, 2.0, 3.0]

    def test_non_numeric_coerced(self):
        assert to_ugm3(pd.Series(["a", "2.0", None])).isna().tolist() == [True, False, True]


class TestPaths:
    def test_finds_repo_root(self):
        root = find_repo_root(__file__)
        assert (root / "pyproject.toml").exists()
