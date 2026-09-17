"""Additional weekly calculations and figures, executed by the companion notebook."""

import json
from hashlib import sha256

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analog_diagnostics import (
    analog_month_stability,
    discrepancy_stats,
    paired_discrepancy_bootstrap,
    site_concentration,
    spectral_shape_projection,
)
from config import ETHIOPIA_SEASONS
from seasonal_analogs import spectral_region_mask
from .ann_weekly_figures import AnnWeeklyFigures, BLUE, GREY, INK
from .overlays import scatter_on_axes


class AnnWeeklyDiagnostics(AnnWeeklyFigures):
    def __init__(self, repo_root):
        super().__init__(repo_root)
        self.extra = self.tables.parent / "ann_weekly_20260910_diagnostics"
        self.extra.mkdir(exist_ok=True)
        self.output = self.output.parent / "ann_weekly_20260910_diagnostics"
        self.output.mkdir(exist_ok=True)
        self.groups = ["All Addis", *self.seasons]
        self.group_colors = [BLUE, *self.colors]
        self.labels = ["All Addis", "Dry", "Belg", "Kiremt"]
        self.models = {g: self.stats(g).fit_id for g in self.groups}
        self.reference = "historical_ocec440__All_Addis"
        self.meta = self.read("library_eligibility")
        assert np.array_equal(self.meta.AnalysisId, self.library["analysis_id"])
        self.mask = spectral_region_mask(self.wn)
        self.paired = self.addis[["MediaId", "date", "season", "month_block"]].merge(
            self.predictions.pivot(index="MediaId", columns="fit_id", values="prediction_ugm3"),
            on="MediaId",
            validate="one_to_one",
        )
        proxy = self.predictions.groupby("MediaId").hips_equivalent_ugm3
        assert proxy.nunique().eq(1).all()
        self.paired["proxy"] = self.paired.MediaId.map(proxy.first())
        self.paired["seasonal_routed"] = [
            r[self.models[r.season]] for _, r in self.paired.iterrows()
        ]
        assert not self.paired.isna().any().any()

    def save_table(self, frame, name):
        frame.to_csv(self.extra / f"{name}.csv", index=False)
        return frame

    def calculate(self):
        rows = []
        for i, season in enumerate(self.seasons):
            p = self.paired.loc[self.paired.season.eq(season)]
            for label, model in [
                ("Season-specific", self.models[season]),
                ("All-Addis", self.models["All Addis"]),
            ]:
                result = paired_discrepancy_bootstrap(
                    p.proxy, p[self.reference], p[model], p.month_block, seed=20260911 + i
                )
                rows.append(dict(season=season, candidate=label, fit_id=model, **result))
        self.pair_stats = self.save_table(pd.DataFrame(rows), "paired_model_comparison")

        rows = []
        for label, model in [
            ("Historical", self.reference),
            ("All Addis", self.models["All Addis"]),
            *[(s.split(" (")[0], self.models[s]) for s in self.seasons],
        ]:
            for season in self.seasons:
                p = self.paired.loc[self.paired.season.eq(season)]
                rows.append(
                    dict(model=label, season=season, **discrepancy_stats(p.proxy, p[model]))
                )
        self.transfer = self.save_table(pd.DataFrame(rows), "cross_season_transfer")

        rows = []
        for month, p in self.paired.groupby("month_block"):
            for label, model in [
                ("Historical", self.reference),
                ("All Addis", self.models["All Addis"]),
                ("Season-specific", "seasonal_routed"),
            ]:
                rows.append(dict(month=month, model=label, **discrepancy_stats(p.proxy, p[model])))
        self.monthly = self.save_table(pd.DataFrame(rows), "monthly_discrepancy")

        draws, frequencies, summaries = [], [], []
        for i, group in enumerate(self.groups):
            selected = (
                np.ones(len(self.addis), bool)
                if group == "All Addis"
                else self.addis.season.eq(group).to_numpy()
            )
            print(f"Reselecting analogs: {group}, 200 month resamples", flush=True)
            d, f, base = analog_month_stability(
                self.library["corrected"],
                self.X[selected],
                self.mask,
                self.meta,
                self.addis.loc[selected, "month_block"],
                seed=20260912 + i,
            )
            saved = (
                self.read("analog_membership")
                .query("mask == 'no_co2' and target_group == @group")
                .FilterId
            )
            assert set(base) == set(saved), "Bootstrap baseline must reproduce the saved cohort"
            d["group"], f["group"] = group, group
            draws.append(d)
            frequencies.append(f)
            original = f.loc[f.original_rank.notna()]
            summaries.append(
                dict(
                    group=group,
                    target_n=int(selected.sum()),
                    target_months=self.addis.loc[selected, "month_block"].nunique(),
                    median_retained=d.retained_fraction.median(),
                    p05_retained=d.retained_fraction.quantile(0.05),
                    p95_retained=d.retained_fraction.quantile(0.95),
                    stable_original_n=int(original.selection_frequency.ge(0.8).sum()),
                    union_selected_n=int(f.selection_frequency.gt(0).sum()),
                )
            )
        self.stability_draws = self.save_table(pd.concat(draws), "analog_bootstrap_draws")
        self.frequencies = self.save_table(pd.concat(frequencies), "analog_selection_frequencies")
        self.stability = self.save_table(pd.DataFrame(summaries), "analog_stability_summary")

        self.projection = spectral_shape_projection(
            self.library["corrected"], self.X, self.mask, self.meta
        )
        z = self.projection
        source = self.meta.iloc[z["source_positions"]][["AnalysisId", "FilterId", "Site"]].copy()
        for df, scores, q in [(source, z["source_scores"], z["source_q"])]:
            df["PC1"], df["PC2"], df["reconstruction_q"] = scores[:, 0], scores[:, 1], q
        self.save_table(source, "pca_source_scores")
        target = self.addis[["MediaId", "ExternalFilterId", "season"]].copy()
        target["PC1"], target["PC2"] = z["target_scores"][:, 0], z["target_scores"][:, 1]
        target["reconstruction_q"] = z["target_q"]
        target["above_source_q95"] = target.reconstruction_q.gt(z["source_q95"])
        self.pca_target = self.save_table(target, "pca_target_scores")
        self.pca_summary = self.save_table(
            target.groupby("season", sort=False)
            .agg(
                n=("MediaId", "size"),
                above_source_q95=("above_source_q95", "sum"),
                fraction_above=("above_source_q95", "mean"),
                median_q=("reconstruction_q", "median"),
            )
            .reset_index(),
            "pca_season_summary",
        )
        np.savez_compressed(
            self.extra / "pca_basis.npz",
            wavenumbers=self.wn[self.mask],
            components=z["components"],
            variance_ratio=z["variance_ratio"],
            source_mean=z["source_mean"],
        )

        summaries, curves = [], []
        members = self.read("analog_membership")
        for group in self.groups:
            part = members.query("mask == 'no_co2' and target_group == @group")
            s, curve = site_concentration(part)
            summaries.append(dict(group=group, **s))
            curve["group"] = group
            curves.append(curve)
        self.diversity = self.save_table(pd.DataFrame(summaries), "site_concentration")
        self.site_curves = self.save_table(pd.concat(curves), "site_concentration_curves")
        self.save_table(self.paired, "paired_addis_predictions")
        methods = dict(
            season_convention="dry_feb",
            proxy="HIPS Fabs/MAC; not chemical EC truth",
            paired_bootstrap_n=4000,
            analog_bootstrap_n=200,
            unit="observed year-month",
            seed_paired=20260911,
            seed_analog=20260912,
            pca_n_components=10,
            pca_source_n=len(source),
            source_q95=z["source_q95"],
            pca_variance_ratio=z["variance_ratio"].tolist(),
            input_table_sha256={
                name: sha256((self.tables / f"{name}.csv").read_bytes()).hexdigest()
                for name in [
                    "predictions",
                    "addis_evaluation",
                    "analog_membership",
                    "library_eligibility",
                ]
            },
            original_source_sha256=self.summary["source_sha256"],
            limits="Exploratory existing Addis data. Fixed predictions in discrepancy intervals. No model refit, no new holdout scoring. Month resampling ignores dependence between months.",
        )
        (self.extra / "methods.json").write_text(json.dumps(methods, indent=2))
        print(self.pair_stats.to_string(index=False))
        print(self.stability.to_string(index=False))
        print(self.pca_summary.to_string(index=False))
        print(self.diversity.to_string(index=False))

    def paired_comparison(self):
        fig, axes = self.canvas(2)
        for ax, candidate in zip(axes.ravel(), ["Season-specific", "All-Addis"]):
            for i, (season, color) in enumerate(zip(self.seasons, self.colors)):
                r = self.pair_stats.query("season == @season and candidate == @candidate").iloc[0]
                ax.errorbar(
                    r.delta_rmse,
                    i,
                    xerr=[[r.delta_rmse - r.delta_rmse_low], [r.delta_rmse_high - r.delta_rmse]],
                    fmt="o",
                    color=color,
                    capsize=6,
                    lw=3,
                    ms=10,
                )
            ax.axvline(0, color=GREY, ls="--")
            ax.set(
                yticks=range(3),
                yticklabels=self.labels[1:],
                ylim=(2.55, -0.55),
                xlim=(
                    self.pair_stats.delta_rmse_low.min() - 0.2,
                    self.pair_stats.delta_rmse_high.max() + 0.2,
                ),
            )
            self.style(
                ax, "Change in RMS discrepancy (µg/m³)", title=f"{candidate} − historical", grid="x"
            )
        return fig

    def transfer_heatmap(self):
        fig, axes = self.canvas(2)
        models = ["Historical", *self.labels]
        for ax, metric, title, limits in zip(
            axes.ravel(),
            ["rmse", "r_squared"],
            ["RMS discrepancy (µg/m³)", "Squared correlation (R²)"],
            [(0, 4.5), (0, 1)],
        ):
            a = (
                self.transfer.pivot(index="model", columns="season", values=metric)
                .loc[models, self.seasons]
                .to_numpy()
            )
            im = ax.imshow(
                a,
                cmap="YlOrRd" if metric == "rmse" else "Blues",
                vmin=limits[0],
                vmax=limits[1],
                aspect="auto",
            )
            ax.set(
                xticks=range(3), xticklabels=self.labels[1:], yticks=range(5), yticklabels=models
            )
            self.style(ax, "Addis evaluation season", "Calibration selection", title, grid=None)
            for (i, j), value in np.ndenumerate(a):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=21,
                    color="white" if value > limits[1] * 0.6 else INK,
                )
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        return fig

    def monthly_bias(self):
        fig, axes = self.canvas()
        ax = axes[0, 0]
        for label, color, marker in [
            ("Historical", INK, "o"),
            ("All Addis", BLUE, "s"),
            ("Season-specific", "#B56636", "^"),
        ]:
            p = self.monthly.query("model == @label").copy()
            # Reindex explicitly so missing calendar months break the lines.
            p.index = pd.PeriodIndex(p.month, freq="M")
            p = p.reindex(pd.period_range(p.index.min(), p.index.max(), freq="M"))
            ax.plot(
                p.index.to_timestamp() + pd.Timedelta(days=14),
                p.bias,
                label=label,
                color=color,
                marker=marker,
                ms=5,
                lw=2,
            )
        ax.axhline(0, color=GREY, ls="--")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        self.style(ax, ylabel="Monthly mean FTIR − HIPS/MAC (µg/m³)")
        ax.legend(
            loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), frameon=False, fontsize=18
        )
        return fig

    def stability_plot(self):
        fig, axes = self.canvas(2)
        ax = axes[0, 0]
        values = [
            100 * self.stability_draws.query("group == @g").retained_fraction for g in self.groups
        ]
        boxes = ax.boxplot(
            values,
            tick_labels=self.labels,
            patch_artist=True,
            widths=0.55,
            whis=(5, 95),
            showfliers=False,
        )
        for patch, color in zip(boxes["boxes"], self.group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for line in boxes["medians"]:
            line.set(color=INK, linewidth=2)
        ax.set(ylim=(0, 103))
        self.style(ax, ylabel="Original 500 retained (%)", title="200 month resamples")
        ax = axes[0, 1]
        for group, label, color in zip(self.groups, self.labels, self.group_colors):
            p = self.frequencies.query("group == @group and original_rank.notna()", engine="python")
            ax.plot(
                np.arange(1, 501),
                100 * np.sort(p.selection_frequency)[::-1],
                label=label,
                color=color,
                lw=2.7,
            )
        ax.axhline(80, color=GREY, ls="--")
        ax.set(ylim=(0, 103))
        self.style(
            ax,
            "Original analogs, ordered by stability",
            "Selected in resamples (%)",
            "Stable core versus changing edge",
        )
        ax.legend(frameon=False, fontsize=15, ncol=2, loc="lower left")
        return fig

    def shape_map(self):
        fig, axes = self.canvas(2)
        ax = axes[0, 0]
        z = self.projection
        # PCA maps are an extension: the standard scatter overlay supplies axes-level drawing.
        scatter_on_axes(
            ax,
            z["source_scores"][:, 0],
            z["source_scores"][:, 1],
            "PC1",
            "PC2",
            color=GREY,
            equal_axes=False,
            show_stats=False,
            show_1to1=False,
        )
        background = ax.collections[-1]
        background.set_sizes([5])
        background.set_alpha(0.13)
        background.set_edgecolor("none")
        background.set_label("IMPROVE")
        for line in list(ax.lines):
            line.remove()
        coordinates = np.vstack([z["source_scores"][:, :2], z["target_scores"][:, :2]])
        low, high = coordinates.min(axis=0), coordinates.max(axis=0)
        margin = (high - low) * 0.04
        ax.set(
            xlim=(low[0] - margin[0], high[0] + margin[0]),
            ylim=(low[1] - margin[1], high[1] + margin[1]),
        )
        for season, label, color in zip(self.seasons, self.labels[1:], self.colors):
            p = self.pca_target.query("season == @season")
            ax.scatter(
                p.PC1,
                p.PC2,
                color=color,
                s=25,
                alpha=0.8,
                label=label,
                edgecolor="white",
                linewidth=0.2,
            )
        variance = 100 * z["variance_ratio"]
        self.style(
            ax,
            f"PC1 ({variance[0]:.1f}% source variance)",
            f"PC2 ({variance[1]:.1f}%)",
            "Spectral shape, CO₂ region removed",
            grid=None,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, frameon=False, fontsize=15, ncol=2)
        ax = axes[0, 1]
        q = self.pca_summary.set_index("season").loc[self.seasons]
        bars = ax.bar(self.labels[1:], 100 * q.fraction_above, color=self.colors, width=0.6)
        ax.bar_label(
            bars,
            labels=[f"{r.above_source_q95}/{r.n}" for r in q.itertuples()],
            padding=5,
            fontsize=20,
        )
        ax.axhline(5, color=GREY, ls="--")
        ax.set(ylim=(0, max(25, 5 * np.ceil(100 * q.fraction_above.max() * 1.4 / 5))))
        self.style(
            ax,
            ylabel="Addis above source residual threshold (%)",
            title="10-PC reconstruction diagnostic",
        )
        return fig

    def diversity_plot(self):
        fig, axes = self.canvas(2)
        ax = axes[0, 0]
        p = self.diversity.set_index("group").loc[self.groups]
        y = np.arange(4)
        a = ax.barh(y - 0.18, p.n_sites, height=0.34, color=GREY, label="Distinct sites")
        b = ax.barh(
            y + 0.18,
            p.effective_sites,
            height=0.34,
            color=self.group_colors,
            label="Effective sites",
        )
        ax.bar_label(a, fmt="%.0f", padding=4, fontsize=16)
        ax.bar_label(b, fmt="%.1f", padding=4, fontsize=16)
        ax.set(
            yticks=y, yticklabels=self.labels, ylim=(3.7, -0.7), xlim=(0, p.n_sites.max() * 1.24)
        )
        self.style(ax, "Sites", title="500 filters in every cohort", grid="x")
        ax.legend(frameon=False, fontsize=16, loc="lower right")
        ax = axes[0, 1]
        for group, label, color in zip(self.groups, self.labels, self.group_colors):
            p = self.site_curves.query("group == @group")
            ax.plot(p["rank"], 100 * p.cumulative_fraction, color=color, label=label, lw=2.7)
        ax.set(ylim=(0, 103))
        self.style(
            ax,
            "Sites, largest contribution first",
            "Cumulative filters (%)",
            "How quickly do sites dominate?",
        )
        ax.legend(frameon=False, fontsize=16, loc="lower right")
        return fig
