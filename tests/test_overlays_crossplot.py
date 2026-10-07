"""Tests for the generic axes-level cross-plot overlay."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from research.ftir_hips_chem.scripts.plotting import PlotConfig
from research.ftir_hips_chem.scripts.plotting import overlays
from research.ftir_hips_chem.scripts.plotting import utils


@pytest.fixture(autouse=True)
def _reset_plotting():
    PlotConfig.reset()
    yield
    PlotConfig.reset()
    plt.close("all")


def test_exported_and_returns_calculated_stats():
    assert "crossplot_on_axes" in overlays.__all__

    x = np.arange(1.0, 7.0)
    y = 2.0 * x + 1.0
    _, ax = plt.subplots()

    stats = overlays.crossplot_on_axes(ax, x, y, "Observed", "Predicted")

    assert stats["n"] == 6
    assert stats["slope"] == pytest.approx(2.0)
    assert stats["intercept"] == pytest.approx(1.0)
    assert ax.get_xlabel() == "Observed"
    assert ax.get_ylabel() == "Predicted"
    assert {line.get_label() for line in ax.lines} == {"Deming fit", "1:1 line"}
    assert ax.get_aspect() == pytest.approx(1.0)


def test_non_finite_and_outliers_are_drawn_but_excluded_from_fit():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.inf, np.nan])
    y = np.array([2.0, 4.0, 6.0, 40.0, 10.0, 12.0, 14.0])
    outliers = np.array([False, False, False, True, False, False, False])
    _, ax = plt.subplots()

    stats = overlays.crossplot_on_axes(
        ax, x, y, "X", "Y", outlier_mask=outliers
    )

    assert stats["n"] == 4
    assert stats["slope"] == pytest.approx(2.0)
    assert len(ax.collections) == 2
    assert {collection.get_label() for collection in ax.collections} == {
        "Data (n=4)",
        "Excluded (n=1)",
    }
    assert all(np.isfinite(limit) for limit in ax.get_xlim() + ax.get_ylim())


def test_returns_none_and_marks_axes_with_fewer_than_three_points():
    _, ax = plt.subplots()

    stats = overlays.crossplot_on_axes(
        ax, [1.0, 2.0, np.inf], [1.0, 2.0, 3.0], "X", "Y"
    )

    assert stats is None
    assert [text.get_text() for text in ax.texts] == ["Insufficient data"]
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"
    assert not ax.collections


def test_precomputed_pls_metrics_are_returned_without_recalculation(monkeypatch):
    supplied = {
        "n": 5,
        "slope": 1.1,
        "intercept": -0.2,
        "R2": 0.975,
        "RMSE": 0.42,
        "MAE": 0.31,
        "bias": 0.05,
        "median_bias": 0.02,
        "deming_slope": 1.2,
        "deming_intercept": -0.3,
        "deming_lambda": 3.3354,
    }

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("precomputed stats should prevent recalculation")

    monkeypatch.setattr(overlays, "calculate_regression_stats", fail_if_called)
    _, ax = plt.subplots()

    result = overlays.crossplot_on_axes(
        ax,
        np.arange(1.0, 6.0),
        np.arange(1.0, 6.0),
        "Observed",
        "Predicted",
        stats=supplied,
    )

    assert result is supplied
    assert "r_squared" not in supplied
    box_text = ax.texts[-1].get_text()
    assert "R² = 0.975" in box_text
    assert "Deming: y = 1.200x - 0.30" in box_text
    assert "RMSE" not in box_text
    fit = next(line for line in ax.lines if line.get_label() == 'Deming fit')
    np.testing.assert_allclose(fit.get_ydata(), 1.2 * fit.get_xdata() - 0.3)


def test_panel_toggles_suppress_optional_artists():
    _, ax = plt.subplots()

    stats = overlays.crossplot_on_axes(
        ax,
        np.arange(1.0, 6.0),
        np.arange(1.0, 6.0),
        "X",
        "Y",
        one_to_one=False,
        stats_box=False,
        equal_axes=False,
        fit_line=False,
    )

    assert stats["n"] == 5
    assert not ax.lines
    assert not ax.texts
    assert ax.get_aspect() == "auto"
    assert ax.get_xlim()[0] == pytest.approx(0)
    assert ax.get_ylim()[0] == pytest.approx(0)


def test_explicit_toggles_override_and_restore_plot_config():
    PlotConfig.set(show_1to1=False, show_stats=False)
    _, ax = plt.subplots()

    overlays.crossplot_on_axes(
        ax,
        np.arange(1.0, 6.0),
        np.arange(1.0, 6.0),
        "X",
        "Y",
        one_to_one=True,
        stats_box=True,
    )

    assert "1:1 line" in {line.get_label() for line in ax.lines}
    assert ax.texts
    assert PlotConfig.get("show_1to1") is False
    assert PlotConfig.get("show_stats") is False


# --- errors-in-variables regression -----------------------------------------
# A 1:1 line asserts both axes measure the same quantity, so x carries error and
# OLS of y-on-x is biased shallow (regression dilution). These pin that the
# Deming equation and line replace the displayed OLS fit on comparison panels;
# the returned OLS metrics remain available for calculations.

def _errors_in_both(seed=0, n=400, true_slope=1.0, err=1.2):
    rng = np.random.default_rng(seed)
    true = rng.uniform(1.0, 10.0, n)
    x = true + rng.normal(0, err, n)
    y = true_slope * true + rng.normal(0, err, n)
    return x, y


def test_deming_recovers_a_known_slope_that_ols_attenuates():
    x, y = _errors_in_both()
    stats = utils.calculate_regression_stats(x, y, errors_in_variables=True)
    # OLS is biased low against the known truth; Deming is not.
    assert stats["slope"] < 0.95
    assert abs(stats["deming_slope"] - 1.0) < 0.06
    assert stats["slope_attenuation_pct"] > 5.0


def test_errors_in_variables_is_off_by_default():
    """Existing callers' stats boxes must not change."""
    x, y = _errors_in_both()
    stats = utils.calculate_regression_stats(x, y)
    for key in ("deming_slope", "deming_intercept", "deming_lambda",
                "slope_attenuation_pct"):
        assert key not in stats


def test_ols_keys_are_untouched_when_eiv_is_enabled():
    x, y = _errors_in_both()
    plain = utils.calculate_regression_stats(x, y)
    eiv = utils.calculate_regression_stats(x, y, errors_in_variables=True)
    for key, value in plain.items():
        assert eiv[key] == value, key


def test_lambda_defaults_to_orthogonal_and_honours_supplied_sigmas():
    x, y = _errors_in_both()
    default = utils.calculate_regression_stats(x, y, errors_in_variables=True)
    assert default["deming_lambda"] == 1.0
    # lambda = (sigma_y / sigma_x) ** 2
    supplied = utils.calculate_regression_stats(
        x, y, errors_in_variables=True, sigma_x=1.0, sigma_y=2.0
    )
    assert supplied["deming_lambda"] == pytest.approx(4.0)
    assert supplied["deming_slope"] != default["deming_slope"]


def test_a_large_lambda_moves_deming_toward_ols():
    """lambda -> inf means all the error is in y, which is the OLS assumption."""
    x, y = _errors_in_both()
    stats = utils.calculate_regression_stats(
        x, y, errors_in_variables=True, sigma_x=1.0, sigma_y=300.0
    )
    ols = stats["slope"]
    assert abs(stats["deming_slope"] - ols) < abs(1.0 - ols)


def test_crossplot_enables_eiv_automatically_for_a_one_to_one_panel():
    x, y = _errors_in_both()
    _, ax = plt.subplots()
    result = overlays.crossplot_on_axes(ax, x, y, "x", "y")
    plt.close("all")
    assert "deming_slope" in result
    assert "Deming" in ax.texts[-1].get_text()
    assert 'OLS' not in ax.texts[-1].get_text()
    assert len(ax.texts[-1].get_text().splitlines()) == 3
    fit = next(line for line in ax.lines if line.get_label() == 'Deming fit')
    np.testing.assert_allclose(fit.get_ydata(), result['deming_slope'] * fit.get_xdata() + result['deming_intercept'])


def test_crossplot_leaves_non_comparison_panels_on_ols():
    x, y = _errors_in_both()
    _, ax = plt.subplots()
    result = overlays.crossplot_on_axes(ax, x, y, "x", "y", one_to_one=False, equal_axes=False)
    plt.close("all")
    assert "deming_slope" not in result


def test_crossplot_eiv_can_be_forced_off_on_a_one_to_one_panel():
    x, y = _errors_in_both()
    _, ax = plt.subplots()
    result = overlays.crossplot_on_axes(ax, x, y, "x", "y", errors_in_variables=False)
    plt.close("all")
    assert "deming_slope" not in result
    assert "Deming" not in ax.texts[-1].get_text()


def test_precomputed_ols_metrics_are_preserved_and_supplemented_only_for_display():
    x, y = _errors_in_both()
    supplied = {"n": len(x), "slope": 0.5, "intercept": 0.0, "R2": 0.9}
    _, ax = plt.subplots()
    result = overlays.crossplot_on_axes(ax, x, y, "x", "y", stats=supplied)
    plt.close("all")
    assert result is supplied
    assert "deming_slope" not in result
    assert 'Deming: y =' in ax.texts[-1].get_text()
    assert '0.500x' not in ax.texts[-1].get_text()


def test_legacy_deming_line_switch_does_not_duplicate_or_add_ols():
    x, y = _errors_in_both()
    _, ax_off = plt.subplots()
    overlays.crossplot_on_axes(ax_off, x, y, "x", "y")
    labels_off = [ln.get_label() for ln in ax_off.lines]

    _, ax_on = plt.subplots()
    overlays.crossplot_on_axes(ax_on, x, y, "x", "y", deming_line=True)
    labels_on = [ln.get_label() for ln in ax_on.lines]
    plt.close("all")

    assert labels_off == labels_on == ['Deming fit', '1:1 line']


def test_crossplot_uses_the_supplied_measurement_error_ratio():
    x, y = _errors_in_both()
    _, ax = plt.subplots()
    stats = overlays.crossplot_on_axes(ax, x, y, 'HIPS BC', 'FTIR EC', sigma_x=0.3, sigma_y=0.6)
    assert stats['deming_lambda'] == pytest.approx(4.0)
    expected = utils.deming(x, y, 4.0)
    assert stats['deming_slope'] == pytest.approx(expected[0])


def test_degenerate_deming_does_not_fall_back_to_an_ols_equation_or_line():
    _, ax = plt.subplots()
    overlays.crossplot_on_axes(ax, [1, 2, 3, 4], [1, 3, 3, 1], 'x', 'y')
    assert 'Deming unavailable' in ax.texts[-1].get_text()
    assert [line.get_label() for line in ax.lines] == ['1:1 line']


@pytest.mark.parametrize('kind', ['scatter', 'iron'])
def test_other_axes_comparison_helpers_draw_the_same_single_deming_fit(kind):
    x, y = _errors_in_both()
    _, ax = plt.subplots()
    if kind == 'scatter':
        stats = overlays.scatter_on_axes(ax, x, y, 'x', 'y', sigma_x=0.3, sigma_y=0.6)
    else:
        stats, _ = overlays.iron_gradient_on_axes(ax, x, y, np.arange(len(x)), 'x', 'y', sigma_x=0.3, sigma_y=0.6)
    assert stats['deming_lambda'] == pytest.approx(4)
    assert [line.get_label() for line in ax.lines] == ['Deming fit', '1:1 line']
    assert 'Deming: y =' in ax.texts[-1].get_text()
    assert 'OLS' not in ax.texts[-1].get_text()


def test_ols_only_saved_stats_use_the_saved_error_ratio_for_display():
    x, y = _errors_in_both()
    saved = utils.calculate_regression_stats(x, y)
    saved['deming_lambda'] = 3.3354
    _, ax = plt.subplots()
    result = overlays.crossplot_on_axes(ax, x, y, 'x', 'y', stats=saved)
    assert result is saved and 'deming_slope' not in saved
    slope, intercept = utils.deming(x, y, saved['deming_lambda'])
    assert f'Deming: {utils.format_equation(slope, intercept)}' in ax.texts[-1].get_text()


def test_explicit_error_ratio_and_legacy_deming_line_request():
    x, y = _errors_in_both()
    _, ax = plt.subplots()
    result = overlays.crossplot_on_axes(ax, x, y, 'x', 'y', one_to_one=False,
                                       fit_line=False, deming_line=True, lambda_ratio=3.3354)
    assert result['deming_lambda'] == 3.3354
    assert [line.get_label() for line in ax.lines] == ['Deming fit']
