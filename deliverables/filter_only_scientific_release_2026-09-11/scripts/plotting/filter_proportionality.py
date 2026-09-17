"""Graph-focused proportionality extension with canonical site colors."""

import matplotlib.pyplot as plt
import numpy as np

from config import SITES
from filter_relationship_stability import SITE_ORDER, POPULATIONS
from .filter_relationship_stability import grid, DISPLAY
from .utils import style_axes


def select(d, scheme="leave_quarter_out", population="diagnostic"):
    return d.loc[d.scheme.eq(scheme) & d.population.eq(population)]


def finish(ax, site, ylabel):
    ax.set_title(DISPLAY[site])
    ax.set_ylabel(ylabel)
    style_axes(ax, ax.get_xlabel(), ylabel, show_legend=False)


def paired_blocks(blocks):
    fig, axes = grid(
        "Does an intercept improve prediction beyond proportionality?",
        "Diagnostic population · identical withheld quarters · positive difference favors the intercept",
    )
    for site, ax in zip(SITE_ORDER, axes):
        g = select(blocks)
        g = g.loc[g.site.eq(site)]
        ax.bar(np.arange(len(g)), g.delta_mae_proportional_minus_ols, color=SITES[site]["color"])
        ax.axhline(0, color=".25", lw=0.8)
        ax.set_xticks(
            np.arange(len(g)),
            [f"{r.block}\nn={r.test_n}" for r in g.itertuples()],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        finish(ax, site, "MAE proportional − MAE intercept (Mm⁻¹)")
    return fig


def bias_blocks(blocks):
    fig, axes = grid(
        "Prediction bias remains after choosing the model form",
        "Diagnostic population · withheld quarters · predicted minus reported HIPS",
    )
    for site, ax in zip(SITE_ORDER, axes):
        g = select(blocks)
        g = g.loc[g.site.eq(site)]
        x = np.arange(len(g))
        ax.plot(x, g.proportional_mean_signed_error, "s--", color=".45", label="Proportional")
        ax.plot(
            x, g.ols_mean_signed_error, "o-", color=SITES[site]["color"], label="With intercept"
        )
        ax.axhline(0, color=".25", lw=0.8)
        ax.set_xticks(x, g.block, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=9)
        finish(ax, site, "Mean signed error (Mm⁻¹)")
    return fig


def forward_errors(blocks):
    fig, axes = grid(
        "Later-period prediction: proportional, intercept and constant",
        "Diagnostic population · training only on earlier filters · all models use identical supported test filters",
    )
    for site, ax in zip(SITE_ORDER, axes):
        g = select(blocks, "later_period")
        g = g.loc[g.site.eq(site)]
        x = np.arange(len(g))
        ax.plot(x, g.median_mae, ":", color=".7", lw=2, label="Training median")
        ax.plot(x, g.proportional_mae, "s--", color=".4", label="Proportional")
        ax.plot(x, g.ols_mae, "o-", color=SITES[site]["color"], label="With intercept")
        ax.set_xticks(
            x,
            [f"{r.block}\nn={r.test_n}" for r in g.itertuples()],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        for i, r in enumerate(g.itertuples()):
            if r.status != "evaluated":
                ax.text(i, 0.02, "NA", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        finish(ax, site, "Held-out MAE (Mm⁻¹)")
    return fig


def denominator_sensitivity(summary, cohorts):
    fig, axes = grid(
        "The proportionality conclusion depends on site and population",
        "Leave-quarter-out · equal-filter MAE · n labels show frozen population size, not supported evaluations",
    )
    labels = ["Diag.", "Ratio", "1.5×", "2×", "3×", "5×"]
    for site, ax in zip(SITE_ORDER, axes):
        g = (
            summary.loc[
                summary.site.eq(site)
                & summary.scheme.eq("leave_quarter_out")
                & summary.weighting.eq("equal_filter")
            ]
            .set_index("population")
            .reindex(POPULATIONS)
        )
        counts = (
            cohorts.loc[cohorts.site.eq(site) & cohorts.variant.eq("baseline")]
            .set_index("population")
            .reindex(POPULATIONS)
            .n
        )
        x = np.arange(6)
        ax.plot(x, g.median_mae, ":", color=".7", lw=2, label="Training median")
        ax.plot(x, g.proportional_mae, "s--", color=".4", label="Proportional")
        ax.plot(x, g.ols_mae, "o-", color=SITES[site]["color"], label="With intercept")
        ax.set_xticks(x, [f"{lab}\nn={n}" for lab, n in zip(labels, counts)], fontsize=10)
        for i, v in enumerate(g.ols_mae):
            if not np.isfinite(v):
                ax.text(i, 0.04, "NA", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
        finish(ax, site, "Held-out MAE (Mm⁻¹)")
    return fig


def weighting(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for scheme, ax, title in zip(
        ["leave_quarter_out", "later_period"],
        axes,
        ["Withheld quarters", "Later-period prediction"],
    ):
        g = select(summary, scheme)
        for i, site in enumerate(SITE_ORDER):
            d = g.loc[g.site.eq(site)].set_index("weighting")
            a = d.loc["equal_filter", "ols_mean_signed_error"]
            b = d.loc["equal_quarter", "ols_mean_signed_error"]
            ax.plot([i - 0.12, i + 0.12], [a, b], color=SITES[site]["color"], lw=2)
            ax.scatter(
                i - 0.12,
                a,
                color=SITES[site]["color"],
                marker="o",
                s=80,
                label="Equal filters" if i == 0 else None,
            )
            ax.scatter(
                i + 0.12,
                b,
                color=SITES[site]["color"],
                marker="s",
                s=80,
                facecolors="none",
                label="Equal quarters" if i == 0 else None,
            )
            ax.annotate(
                f"{a:+.2f}",
                (i - 0.12, a),
                xytext=(-4, 10),
                textcoords="offset points",
                ha="right",
                fontsize=9,
            )
            ax.annotate(
                f"{b:+.2f}",
                (i + 0.12, b),
                xytext=(4, -15),
                textcoords="offset points",
                ha="left",
                fontsize=9,
            )
        ax.set_xticks(range(4), ["Addis", "Beijing", "Delhi", "JPL"])
        ax.axhline(0, color=".3", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("Intercept-model mean signed error (Mm⁻¹)")
        ax.margins(x=0.18, y=0.2)
        ax.legend(fontsize=9)
        style_axes(ax, "", ax.get_ylabel(), show_legend=False)
    fig.suptitle(
        "Averaging filters and averaging quarters answer different questions", fontsize=17, y=0.98
    )
    fig.text(
        0.5,
        0.90,
        "Diagnostic population · identical supported filters and folds within each evaluation scheme",
        ha="center",
    )
    fig.subplots_adjust(top=0.79, bottom=0.14, wspace=0.28)
    return fig


def id11_errors(blocks):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    g = blocks.loc[blocks.population.eq("diagnostic") & blocks.status.eq("evaluated")]
    a = g.loc[g.training_choice.eq("all_earlier")]
    i = g.loc[g.training_choice.eq("id11_earlier")]
    x = np.arange(len(a))
    for ax, col, ylabel in zip(
        axes, ["ols_mae", "ols_mean_signed_error"], ["MAE (Mm⁻¹)", "Mean signed error (Mm⁻¹)"]
    ):
        ax.plot(x, a[col], "s--", color=".4", label="All earlier training")
        ax.plot(x, i[col], "o-", color=SITES["Addis_Ababa"]["color"], label="Earlier ID 11 only")
        ax.set_xticks(
            x, [f"{r.block}\nn={r.test_n}" for r in a.itertuples()], rotation=35, ha="right"
        )
        ax.axhline(0, color=".3", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        style_axes(ax, "", ylabel, show_legend=False)
    fig.suptitle(
        "Addis: compare training choices on the same later ID-11 filters", fontsize=17, y=0.98
    )
    fig.text(
        0.5,
        0.90,
        "Intercept model · 127 common test filters across five quarters · identifier and date remain confounded",
        ha="center",
    )
    fig.subplots_adjust(top=0.81, bottom=0.21, wspace=0.28)
    return fig


def id11_support(blocks):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    g = blocks.loc[blocks.population.eq("diagnostic") & blocks.training_choice.eq("all_earlier")]
    x = np.arange(len(g))
    axes[0].plot(x, g.all_earlier_train_n, "s--", color=".4", label="All earlier")
    axes[0].plot(
        x, g.id11_earlier_train_n, "o-", color=SITES["Addis_Ababa"]["color"], label="Earlier ID 11"
    )
    axes[0].axhline(10, color=".3", ls=":", label="Minimum training n = 10")
    axes[0].set_ylabel("Training filters")
    axes[0].legend(fontsize=9)
    axes[1].bar(
        x,
        g.test_n,
        color=[SITES["Addis_Ababa"]["color"] if yes else ".75" for yes in g.common_support],
    )
    for j, r in enumerate(g.itertuples()):
        if r.test_n and not r.common_support:
            axes[1].text(
                j, r.test_n + 0.8, "Unavailable", ha="center", fontsize=9, rotation=90, va="bottom"
            )
    axes[1].set_ylim(0, max(g.test_n) + 14)
    axes[1].set_ylabel("ID-11 test filters")
    for ax in axes:
        ax.set_xticks(x, g.block, rotation=45, ha="right")
        style_axes(ax, "", ax.get_ylabel(), show_legend=False)
    fig.suptitle("Restricted training changes which early folds are supported", fontsize=18, y=0.98)
    fig.text(
        0.5,
        0.90,
        "Addis diagnostic population · test filters are compared only when both training choices meet the same rule",
        ha="center",
    )
    fig.subplots_adjust(top=0.81, bottom=0.23, wspace=0.27)
    return fig


def test_weights(blocks):
    fig, axes = grid(
        "Unequal quarter sizes change the contribution to overall error",
        "Diagnostic later-period evaluation · each bar is a quarter’s share of supported test filters",
    )
    for site, ax in zip(SITE_ORDER, axes):
        g = select(blocks, "later_period")
        g = g.loc[g.site.eq(site) & g.status.eq("evaluated")]
        frac = 100 * g.test_n / g.test_n.sum()
        x = np.arange(len(g))
        ax.bar(x, frac, color=SITES[site]["color"])
        for j, (v, n) in enumerate(zip(frac, g.test_n)):
            ax.text(j, v + 1, f"{v:.1f}%\nn={n}", ha="center", fontsize=10)
        ax.set_xticks(x, g.block, rotation=35, ha="right")
        ax.set_ylim(0, max(frac) + 15)
        finish(ax, site, "Share of evaluated filters (%)")
    return fig


def make_figures(tables, cohorts, directory):
    plt.rcParams["svg.hashsalt"] = "filter-proportionality-v2"
    directory.mkdir(parents=True, exist_ok=True)
    b = tables["proportionality_blocks"]
    s = tables["proportionality_summary"]
    ib = tables["id11_training_blocks"]
    builders = [
        ("01_paired_proportionality", lambda: paired_blocks(b)),
        ("02_temporal_bias", lambda: bias_blocks(b)),
        ("03_later_period_models", lambda: forward_errors(b)),
        ("04_denominator_sensitivity", lambda: denominator_sensitivity(s, cohorts)),
        ("05_weighting_and_bias", lambda: weighting(s)),
        ("06_id11_common_test_errors", lambda: id11_errors(ib)),
        ("07_id11_common_support", lambda: id11_support(ib)),
        ("08_quarter_contributions", lambda: test_weights(b)),
    ]
    paths = []
    for name, builder in builders:
        fig = builder()
        for ext in ["png", "svg"]:
            fig.savefig(
                directory / (name + "." + ext),
                bbox_inches="tight",
                metadata={"Date": None} if ext == "svg" else None,
            )
        paths.append(directory / (name + ".png"))
        plt.close(fig)
    return paths
