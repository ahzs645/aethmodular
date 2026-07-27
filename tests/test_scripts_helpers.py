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
from data_matching import (                                   # noqa: E402
    add_base_filter_id, base_filter_id, normalize_filter_id,
)
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


class TestFlowPeriodLabelVocabularies:
    """comparisons.flow_periods must accept both flow-period label spellings.

    Two producers exist and both reach this plot function:
        flow_periods.add_flow_period         -> 'before' / 'after' / 'gap'
        data_matching.add_flow_period_column -> 'before_fix' / 'after_fix' / 'gap_period'

    Matching only the bare form made the plot silently print "skipping" and draw
    nothing for every caller using the data_matching column.
    """

    @staticmethod
    def _frame(before_label, after_label, n=40):
        rng = np.random.default_rng(0)
        half = n // 2
        return pd.DataFrame({
            "aeth_bc": rng.uniform(1, 10, n),
            "filter_ec": rng.uniform(1, 10, n),
            "flow_period": [before_label] * half + [after_label] * half,
        })

    @pytest.mark.parametrize(
        "before_label,after_label",
        [("before", "after"), ("before_fix", "after_fix")],
    )
    def test_both_vocabularies_produce_results(self, before_label, after_label):
        import matplotlib
        matplotlib.use("Agg")
        from plotting import comparisons, PlotConfig

        PlotConfig.set(sites=["Beijing"])
        results = comparisons.flow_periods(
            {"Beijing": self._frame(before_label, after_label)}
        )

        assert "Beijing" in results, f"{before_label}/{after_label} was skipped"
        # Result keys are canonical regardless of the input spelling.
        assert sorted(results["Beijing"]) == ["after", "before"]

    def test_missing_after_period_still_skips(self):
        """A genuinely incomplete pair must still be skipped, not force-plotted."""
        import matplotlib
        matplotlib.use("Agg")
        from plotting import comparisons, PlotConfig

        PlotConfig.set(sites=["Beijing"])
        results = comparisons.flow_periods({"Beijing": self._frame("before", "gap")})
        assert not results.get("Beijing")


class TestPlottingOverlays:
    """Axes-level primitives that replaced plotting_legacy.

    plotting_legacy was deleted on 2026-07-26. These lock in the contract three
    notebooks depend on: draw onto a caller-supplied axes, return regression
    stats, and keep the legacy styling so migrated figures did not change.
    """

    @staticmethod
    def _axes():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt.subplots()

    def test_scatter_returns_stats_and_draws_on_given_axes(self):
        from plotting import overlays

        rng = np.random.default_rng(7)
        x = rng.uniform(1, 20, 60)
        y = 1.8 * x + rng.normal(0, 2, 60)

        fig, ax = self._axes()
        stats = overlays.scatter_on_axes(ax, x, y, "X", "Y")

        assert stats["n"] == 60
        assert stats["slope"] == pytest.approx(1.8, abs=0.1)
        assert ax.collections, "nothing was drawn on the supplied axes"

    def test_scatter_drops_non_finite_instead_of_raising(self):
        """Legacy masked with ~isnan, so +/-inf reached polyfit and axis limits."""
        from plotting import overlays

        rng = np.random.default_rng(7)
        x = np.append(rng.uniform(1, 20, 40), [np.inf, np.nan])
        y = np.append(rng.uniform(1, 20, 40), [5.0, 5.0])

        fig, ax = self._axes()
        stats = overlays.scatter_on_axes(ax, x, y, "X", "Y")

        assert stats["n"] == 40
        assert all(np.isfinite(lim) for lim in ax.get_xlim() + ax.get_ylim())

    def test_scatter_returns_none_below_three_points(self):
        from plotting import overlays

        fig, ax = self._axes()
        assert overlays.scatter_on_axes(ax, [1.0, 2.0], [1.0, 2.0], "X", "Y") is None

    def test_outliers_excluded_from_fit_but_still_drawn(self):
        from plotting import overlays

        rng = np.random.default_rng(3)
        x = rng.uniform(1, 20, 30)
        y = 2.0 * x
        mask = np.zeros(30, bool)
        mask[:5] = True

        fig, ax = self._axes()
        stats = overlays.scatter_on_axes(ax, x, y, "X", "Y", outlier_mask=mask)

        assert stats["n"] == 25
        assert len(ax.collections) == 2, "expected retained + excluded point sets"

    def test_iron_gradient_requires_iron_present(self):
        """Rows with missing iron are excluded from the fit, unlike
        crossplots.with_iron_gradient. Preserved so migrated numbers match."""
        from plotting import overlays

        rng = np.random.default_rng(11)
        x = rng.uniform(1, 20, 30)
        y = 2.0 * x
        iron = rng.uniform(1, 5, 30)
        iron[:6] = np.nan

        fig, ax = self._axes()
        stats, scatter = overlays.iron_gradient_on_axes(ax, x, y, iron, "X", "Y")

        assert stats["n"] == 24
        assert scatter is not None

    def test_bc_timeseries_labels_series_with_site_name(self):
        """timeseries.bc draws without a legend; this keeps the site label so a
        caller can overlay several sites on one shared axes."""
        from plotting import overlays

        df = pd.DataFrame({
            "day_9am": pd.date_range("2024-01-01", periods=10),
            "IR BCc": np.arange(10.0),
        })

        fig, ax = self._axes()
        overlays.bc_timeseries_on_axes(ax, "Beijing", df, {"color": "#E74C3C"})

        assert [line.get_label() for line in ax.get_lines()] == ["Beijing"]


class TestBaseFilterIdColumnForm:
    """add_base_filter_id must agree with the scalar base_filter_id.

    It used a bare r'-\\d+$', which strips the 4-digit sample number from ids
    already in base form -- 'ETAD-0035' collapsed to 'ETAD', silently mapping
    every sample at a site onto one join key.
    """

    IDS = ["ZAJB-0041-12", "ETAD-0035-3", "ETAD-0035", "ZAJB-0007-1", "ETAD-0035-10"]

    def test_column_form_matches_scalar_form(self):
        out = add_base_filter_id(pd.DataFrame({"FilterId": self.IDS}))
        assert list(out["base_filter_id"]) == [base_filter_id(i) for i in self.IDS]

    def test_already_base_ids_are_unchanged(self):
        out = add_base_filter_id(pd.DataFrame({"FilterId": ["ETAD-0035", "ZAJB-0007"]}))
        assert list(out["base_filter_id"]) == ["ETAD-0035", "ZAJB-0007"]

    def test_multi_digit_replicates_are_stripped(self):
        out = add_base_filter_id(pd.DataFrame({"FilterId": ["ZAJB-0041-12"]}))
        assert out["base_filter_id"][0] == "ZAJB-0041"


class TestLayoutFallback:
    """Plots that don't implement every layout must say so, not draw nothing.

    `resolve_layout` accepts 'combined' as valid, but six plots had if/elif
    chains with no else, so `PlotConfig.set(layout='combined')` globally made
    them silently return None having drawn nothing.
    """

    def test_resolve_layout_passes_through_supported(self):
        from plotting import resolve_layout

        assert resolve_layout("grid", supported=("individual", "grid")) == "grid"

    def test_resolve_layout_falls_back_and_warns(self):
        from plotting import resolve_layout

        with pytest.warns(UserWarning, match="not implemented"):
            got = resolve_layout("combined", supported=("individual", "grid"))
        assert got == "individual"

    def test_resolve_layout_still_rejects_nonsense(self):
        from plotting import resolve_layout

        with pytest.raises(ValueError, match="Invalid layout"):
            resolve_layout("sideways")

    def test_no_supported_arg_keeps_old_behaviour(self):
        from plotting import resolve_layout

        assert resolve_layout("combined") == "combined"

    @pytest.mark.parametrize(
        "module,func",
        [
            ("distributions", "smooth_raw_histogram"),
            ("distributions", "uv_ir_ratio_histogram"),
            ("distributions", "correlation_matrix"),
            ("timeseries", "data_completeness"),
            ("timeseries", "filter_vs_aeth"),
            ("timeseries", "flow_ratio"),
        ],
    )
    def test_partial_layout_plots_declare_what_they_support(self, module, func):
        """Guards the wiring: these six must keep passing `supported=`."""
        import importlib
        import inspect

        mod = importlib.import_module(f"plotting.{module}")
        src = inspect.getsource(getattr(mod, func))
        assert "resolve_layout(layout, supported=" in src, (
            f"{module}.{func} no longer declares its supported layouts, so an "
            "unsupported layout would silently draw nothing again"
        )
