"""Reported-date validation figures; canonical site colors and comparison overlays."""

import matplotlib.pyplot as plt
import numpy as np

from config import SITES
from filter_relationship_stability import SITE_ORDER, POPULATIONS
from .overlays import crossplot_on_axes
from .utils import style_axes

DISPLAY = {s: s.replace("_", " ") for s in SITE_ORDER}


def grid(title, subtitle):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=19, y=0.99)
    fig.text(0.5, 0.943, subtitle, ha="center", fontsize=11, color=".3")
    fig.subplots_adjust(top=0.875, bottom=0.1, hspace=0.45, wspace=0.27)
    return fig, axes.ravel()


def subset(df, scheme="leave_quarter_out", population="diagnostic"):
    return df.loc[df.variant.eq("baseline") & df.population.eq(population) & df.scheme.eq(scheme)]


def mae_blocks(blocks, scheme):
    title = "Withheld quarters: does the linear model improve on a constant?"
    subtitle = (
        "Diagnostic population · MAE in HIPS units · training uses earlier and later quarters"
    )
    if scheme == "later_period":
        title = "Later-period prediction: train only on earlier quarters"
        subtitle = "Diagnostic population · expanding training · unavailable early folds retain their calendar position"
    fig, axes = grid(title, subtitle)
    for site, ax in zip(SITE_ORDER, axes):
        g = subset(blocks, scheme)
        g = g.loc[g.site.eq(site)]
        x = np.arange(len(g))
        ax.plot(x, g.median_mae, "o--", color=".55", label="Training median")
        ax.plot(x, g.ols_mae, "o-", color=SITES[site]["color"], label="OLS with intercept")
        ax.set_xticks(
            x,
            [f"{r.block}\nn={r.test_n}" for r in g.itertuples()],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        for i, r in enumerate(g.itertuples()):
            if r.status != "evaluated":
                ax.text(i, 0.03, "NA", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        ax.set_title(DISPLAY[site])
        ax.set_ylabel("Held-out MAE (Mm⁻¹)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        style_axes(ax, ax.get_xlabel(), ax.get_ylabel(), show_legend=False)
    return fig


def bias_blocks(blocks):
    fig, axes = grid(
        "Error direction changes across reported-date blocks",
        "Diagnostic population · OLS prediction − reported HIPS · zero indicates no mean bias",
    )
    for site, ax in zip(SITE_ORDER, axes):
        g = subset(blocks)
        g = g.loc[g.site.eq(site)]
        ax.bar(np.arange(len(g)), g.ols_mean_signed_error, color=SITES[site]["color"])
        ax.axhline(0, color=".3", lw=0.8)
        ax.set_xticks(np.arange(len(g)), g.block, rotation=45, ha="right", fontsize=9)
        ax.set_title(DISPLAY[site])
        ax.set_ylabel("Mean signed error (Mm⁻¹)")
        style_axes(ax, ax.get_xlabel(), ax.get_ylabel(), show_legend=False)
    return fig


def sensitivity(summary, cohorts):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    names = list(POPULATIONS)
    vals = np.full((4, 6), np.nan)
    ns = np.zeros((4, 6), int)
    for i, site in enumerate(SITE_ORDER):
        for j, pop in enumerate(names):
            r = subset(summary, population=pop)
            r = r.loc[r.site.eq(site)].iloc[0]
            vals[i, j] = r.mae_improvement_pct
            ns[i, j] = cohorts.loc[
                cohorts.site.eq(site) & cohorts.population.eq(pop) & cohorts.variant.eq("baseline"),
                "n",
            ].iloc[0]
    bound = max(10, np.nanmax(np.abs(vals)))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#ededed")
    im = ax.imshow(vals, cmap=cmap, vmin=-bound, vmax=bound, aspect="auto")
    for i in range(4):
        for j in range(6):
            text = (
                f"{vals[i, j]:+.1f}%\nn={ns[i, j]}"
                if np.isfinite(vals[i, j])
                else f"Unavailable\nn={ns[i, j]}"
            )
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=12,
                color="white" if abs(vals[i, j]) > 0.6 * bound else "#222222",
            )
    ax.set_xticks(range(6), ["Diagnostic", "Ratio / 1×", "1.5× MDL", "2× MDL", "3× MDL", "5× MDL"])
    ax.set_yticks(range(4), [DISPLAY[s] for s in SITE_ORDER])
    ax.set_title(
        "Selection sensitivity: held-out MAE improvement over the median", pad=38, fontsize=17
    )
    fig.text(
        0.48,
        0.865,
        "Leave-quarter-out · filter-weighted errors · positive = lower OLS error",
        ha="center",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, label="MAE improvement (%)", fraction=0.045, pad=0.03)
    fig.subplots_adjust(top=0.81, bottom=0.12, left=0.14, right=0.9)
    return fig


def coefficients(blocks, coeff):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    b = subset(blocks)
    b = b.loc[b.site.eq("Addis_Ababa") & b.test_n.gt(0)]
    c = coeff.loc[
        coeff.site.eq("Addis_Ababa")
        & coeff.population.eq("diagnostic")
        & coeff.variant.eq("without_ETAD_0037")
    ].iloc[0]
    labels = ["ETAD-0037", *b.block.tolist()]
    for ax, col, single, label in zip(
        axes,
        ["slope_delta_from_full", "intercept_delta_from_full"],
        [c.delta_slope_from_original, c.delta_intercept_from_original],
        ["Change in OLS slope [(Mm⁻¹)/(µg m⁻³)]", "Change in OLS intercept (Mm⁻¹)"],
    ):
        ax.barh(labels, [single, *b[col]], color=[SITES["Addis_Ababa"]["color"]] + [".6"] * len(b))
        ax.axvline(0, color=".2", lw=0.8)
        ax.set_xlabel(label)
        ax.invert_yaxis()
        style_axes(ax, ax.get_xlabel(), ax.get_ylabel(), show_legend=False)
    fig.suptitle("Addis: one-filter influence versus whole-quarter omission", fontsize=18, y=0.98)
    fig.text(
        0.5,
        0.91,
        "Diagnostic cohort · coefficient changes relative to the full-cohort descriptive fit",
        ha="center",
    )
    fig.subplots_adjust(top=0.82, bottom=0.15, wspace=0.35, left=0.11, right=0.98)
    return fig


def prediction_scatter(predictions):
    fig, axes = grid(
        "Held-out HIPS predictions retain block-specific errors",
        "Diagnostic population · leave-quarter-out · 1:1 reference, no fitted comparison line",
    )
    for site, ax in zip(SITE_ORDER, axes):
        p = subset(predictions)
        p = p.loc[p.site.eq(site) & p.ols_prediction.notna()]
        crossplot_on_axes(
            ax,
            p.hips_fabs_Mm1,
            p.ols_prediction,
            "Reported HIPS (Mm⁻¹)",
            "Held-out HIPS prediction (Mm⁻¹)",
            color=SITES[site]["color"],
            one_to_one=True,
            equal_axes=True,
            fit_line=False,
            stats_box=False,
        )
        for c in ax.collections:
            c.set_sizes([24])
            c.set_alpha(0.7)
        ax.set_title(f"{DISPLAY[site]} · n={len(p)}")
        ax.text(
            0.03, 0.96, f"MAE {p.ols_error.abs().mean():.2f} Mm⁻¹", transform=ax.transAxes, va="top"
        )
        ax.legend().remove()
    return fig


def concentration_ranges(blocks):
    fig, axes = grid(
        "Test concentration ranges differ from their training ranges",
        "Diagnostic population · leave-quarter-out · labels give test filters outside the training EC range",
    )
    fig.subplots_adjust(left=0.18, right=0.98, wspace=0.65, top=0.84)
    for site, ax in zip(SITE_ORDER, axes):
        b = subset(blocks)
        b = b.loc[b.site.eq(site) & b.test_n.gt(0)]
        for i, r in enumerate(b.itertuples()):
            ax.plot(
                [r.train_ec_min, r.train_ec_max],
                [i + 0.12, i + 0.12],
                color=".65",
                lw=4,
                label="Training range" if i == 0 else None,
            )
            ax.plot(
                [r.test_ec_min, r.test_ec_max],
                [i - 0.12, i - 0.12],
                color=SITES[site]["color"],
                lw=4,
                label="Test range" if i == 0 else None,
            )
        ax.set_yticks(
            range(len(b)),
            [
                f"{r.block}  ({int(r.outside_training_ec_n)}/{r.test_n} outside)"
                for r in b.itertuples()
            ],
            fontsize=9,
        )
        ax.set_title(DISPLAY[site])
        ax.set_xlabel("FTIR-predicted EC (µg m⁻³)")
        ax.invert_yaxis()
        style_axes(ax, ax.get_xlabel(), ax.get_ylabel(), show_legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.917), ncol=2, frameon=False
    )
    return fig


def metadata_plot(predictions):
    fig, axes = grid(
        "Reported calibration-set identifiers track different filter groups",
        "Diagnostic population · held-out OLS errors · identifiers are not documented model versions",
    )
    for site, ax in zip(SITE_ORDER, axes):
        p = subset(predictions)
        p = p.loc[p.site.eq(site) & p.ols_error.notna()]
        groups = list(p.groupby("ftir_CalibrationSetId", sort=True))
        ax.boxplot(
            [g.ols_error for _, g in groups],
            tick_labels=[f"ID {label}\nn={len(g)}" for label, g in groups],
            patch_artist=True,
            boxprops={"facecolor": SITES[site]["color"], "alpha": 0.45},
            medianprops={"color": "black"},
        )
        ax.axhline(0, color=".3", lw=0.8)
        ax.set_title(DISPLAY[site])
        ax.set_ylabel("Prediction − HIPS (Mm⁻¹)")
        style_axes(ax, ax.get_xlabel(), ax.get_ylabel(), show_legend=False)
    return fig


def paired_plot(influence):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    g = influence.loc[
        influence.population.eq("diagnostic") & influence.scheme.eq("leave_quarter_out")
    ]
    labels = g.block.replace({"ALL_COMMON_FILTERS": "All common\nfilters"})
    ax.bar(labels, g.mae_change_without_minus_baseline, color=SITES["Addis_Ababa"]["color"])
    ax.axhline(0, color=".2", lw=0.8)
    ax.set_ylabel("MAE change after omission (Mm⁻¹)")
    ax.set_title("ETAD-0037: changes in error on the same held-out filters", fontsize=17, pad=35)
    fig.text(
        0.5,
        0.86,
        "Leave-quarter-out · diagnostic population · negative = lower error after omission",
        ha="center",
        fontsize=11,
    )
    for i, r in enumerate(g.itertuples()):
        ax.text(
            i, 0.02, f"n={r.common_n}", transform=ax.get_xaxis_transform(), ha="center", fontsize=9
        )
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.12, right=0.97)
    style_axes(ax, ax.get_xlabel(), ax.get_ylabel(), show_legend=False)
    return fig


def make_figures(b, p, c, s, inf, cohorts, members, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "filter-relationship-stability-frozen-v1"
    builders = [
        ("01_withheld_block_mae", lambda: mae_blocks(b, "leave_quarter_out")),
        ("02_withheld_block_bias", lambda: bias_blocks(b)),
        ("03_population_sensitivity", lambda: sensitivity(s, cohorts)),
        ("04_addis_coefficient_influence", lambda: coefficients(b, c)),
        ("05_heldout_predictions", lambda: prediction_scatter(p)),
        ("06_later_period_mae", lambda: mae_blocks(b, "later_period")),
        ("07_concentration_ranges", lambda: concentration_ranges(b)),
        ("08_reported_metadata_errors", lambda: metadata_plot(p)),
        ("09_addis_common_filter_influence", lambda: paired_plot(inf)),
    ]
    paths = []
    for name, builder in builders:
        fig = builder()
        for ext in ["png", "svg"]:
            path = directory / (name + "." + ext)
            fig.savefig(
                path, bbox_inches="tight", metadata={"Date": None} if ext == "svg" else None
            )
        paths.append(directory / (name + ".png"))
        plt.close(fig)
    return paths
