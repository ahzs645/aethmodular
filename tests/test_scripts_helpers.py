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

from plotting.utils import deming, deming_lambda            # noqa: E402
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
