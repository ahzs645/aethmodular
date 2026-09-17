"""Presentation figures from the completed September 10 FTIR analysis.

The notebook owns execution and exports. Reuse the standard comparison overlay;
the other panels extend the plotting package for spectral masks, cohort counts
and asymmetric confidence intervals, which its existing APIs do not cover.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ETHIOPIA_SEASONS, MAC_VALUE
from outliers import apply_exclusion_flags, get_clean_data
from .overlays import crossplot_on_axes
from .utils import style_axes

BLUE, GREY, INK = "#246A9B", "#AAB4BC", "#17384A"


class AnnWeeklyFigures:
    def __init__(self, repo_root):
        self.root = Path(repo_root)
        self.tables = self.root / "research/ftir_hips_chem/output/tables/ann_weekly_20260910"
        self.output = (
            self.root / "research/ftir_hips_chem/output/plots/ann_weekly_20260910_notebook"
        )
        self.output.mkdir(parents=True, exist_ok=True)
        self.summary = json.loads((self.tables / "summary.json").read_text())
        self.metrics = self.read("regression_metrics")
        self.predictions = self.read("predictions")
        self.addis = get_clean_data(
            apply_exclusion_flags(self.read("addis_evaluation"), "Addis_Ababa")
        )
        assert len(self.addis) == 239, "Exclusion changes require an analysis rerun"
        self.seasons = list(ETHIOPIA_SEASONS)
        self.colors = [ETHIOPIA_SEASONS[s]["color"] for s in self.seasons]
        cache = self.root / "research/ftir_ec_phase3/output/corrected"
        self.library = np.load(cache / "improve_pool_corrected_df6.npz", allow_pickle=True)
        target = np.load(cache / "etad_corrected_df6.npz", allow_pickle=True)
        self.wn = target["wn"].astype(float)
        self.X = np.vstack(
            [
                target["corrected"][target["media_id"].astype(int) == m].mean(axis=0)
                for m in self.addis.MediaId
            ]
        )
        self.positions = pd.Series(
            np.arange(len(self.library["analysis_id"])), index=self.library["analysis_id"]
        )
        assert np.allclose(self.library["wn"], self.wn)

    def read(self, name):
        return pd.read_csv(self.tables / f"{name}.csv")

    def stats(self, group, mask="no_co2"):
        m = self.metrics
        return m.loc[
            m["mask"].eq(mask) & m.selection_group.eq(group) & m.evaluation_group.eq(group)
        ].iloc[0]

    @staticmethod
    def canvas(ncols=1):
        return plt.subplots(1, ncols, figsize=(14.8, 5.6), squeeze=False, layout="constrained")

    @staticmethod
    def style(ax, xlabel="", ylabel="", title=None, grid="y"):
        style_axes(ax, xlabel, ylabel, title, show_legend=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=16)
        ax.grid(False)
        if grid:
            ax.grid(axis=grid, color="#E1E7EB", alpha=0.8)
        ax.set_axisbelow(True)

    def save(self, fig, slide):
        path = self.output / f"slide_{slide:02d}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        return path

    def spectral_regions(self):
        fig, axes = self.canvas()
        ax = axes[0, 0]
        ax.plot(self.wn, np.median(self.X, axis=0), color=BLUE, lw=2.5)
        ax.axvspan(1800, 2500, color="#C64F44", alpha=0.15, label="CO₂ cut: 1800–2500")
        ax.axvspan(3500, self.wn.max(), color="#BD8A29", alpha=0.16, label="Upper cut: >3500")
        ax.axvline(3600, color="#946713", ls="--", lw=1.6, label="Alternative: >3600")
        ax.set_xlim(self.wn.max(), self.wn.min())
        self.style(ax, "Wavenumber (cm⁻¹)", "Corrected absorbance")
        ax.legend(
            loc="upper center", ncol=3, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 1.13)
        )
        return fig

    def membership_changes(self):
        fig, axes = self.canvas(2)
        m = self.read("mask_membership_changes").query("group == 'All Addis'").set_index("mask")
        values = [
            int(m.loc[k, "changed_vs_full"]) for k in ["no_co2", "no_co2_max3600", "no_co2_max3500"]
        ]
        pairs = self.read("seasonal_overlap").query("mask == 'no_co2'")
        for ax, labels, heights, title, ymax in [
            (
                axes[0, 0],
                ["CO₂ cut", "+ >3600", "+ >3500"],
                values,
                "Replaced after spectral cuts",
                100,
            ),
            (
                axes[0, 1],
                ["Dry / Belg", "Dry / Kiremt", "Belg / Kiremt"],
                pairs.intersection,
                "Shared between seasons",
                200,
            ),
        ]:
            bars = ax.bar(labels, heights, color=BLUE, width=0.58)
            ax.bar_label(bars, padding=5, fontsize=20, fontweight="bold", color=INK)
            ax.set_ylim(0, ymax)
            self.style(ax, ylabel="Filters", title=title)
        return fig

    def seasonal_crossplots(self):
        fig, axes = self.canvas(3)
        for i, (ax, season, color) in enumerate(zip(axes.ravel(), self.seasons, self.colors)):
            r = self.stats(season)
            p = self.predictions.query("fit_id == @r.fit_id and season == @season")
            crossplot_on_axes(
                ax,
                p.hips_equivalent_ugm3,
                p.prediction_ugm3,
                "HIPS Fabs/MAC (µg/m³)",
                "FTIR EC (µg/m³)" if i == 0 else "",
                stats=r.to_dict(),
                stats_box=False,
                fit_line=False,
                deming_line=True,
                color=color,
            )
            ax.set(xlim=(0, 10), ylim=(0, 10))
            for line in ax.lines:
                if line.get_label().startswith("Deming"):
                    line.set_data(
                        [0, 10], [r.deming_intercept, 10 * r.deming_slope + r.deming_intercept]
                    )
                    line.set(color=INK, lw=2, ls="-")
                elif "1:1" in line.get_label():
                    line.set_data([0, 10], [0, 10])
                    line.set(color=GREY, lw=1.4, ls="--")
            for collection in ax.collections:
                collection.set_sizes([35])
                collection.set_edgecolor("white")
                collection.set_linewidth(0.3)
                collection.set_alpha(0.8)
            if ax.get_legend():
                ax.get_legend().remove()
            ax.text(
                0.04,
                0.96,
                f"R² = {r.R2:.3f}\nSlope = {r.deming_slope:.2f}\nIntercept = {r.deming_intercept:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=15,
                color=INK,
            )
            ax.set_title(
                f"{season.split(' ')[0]} (n = {len(p)})",
                color=color,
                fontsize=22,
                fontweight="bold",
            )
            ax.tick_params(labelsize=15)
            ax.spines[["top", "right"]].set_visible(False)
        return fig

    def pmf_comparison(self):
        fig, axes = self.canvas()
        ax = axes[0, 0]
        names = sorted(set(self.summary["pmf_counts"]) - {"unmatched"})
        rows = [self.stats(n) for n in names]
        x, width = np.arange(len(rows)), 0.34
        for offset, field, color, label in [
            (-0.5, "R2", BLUE, "Addis R²"),
            (0.5, "TOR_R2", "#7B959F", "TOR R²"),
        ]:
            b = ax.bar(
                x + offset * width, [r[field] for r in rows], width, color=color, label=label
            )
            ax.bar_label(b, fmt="%.3f", padding=4, fontsize=17, color=INK)
        ax.axhline(0.85, color="#A85F55", ls="--", lw=1.3, label="Prior TOR screen: 0.85")
        ax.set_xticks(x, ["Charcoal", "Fossil fuel", "Polluted\nmarine", "Sea salt\nmixed", "Wood"])
        ax.set_ylim(0, 1.06)
        self.style(ax, ylabel="R²")
        ax.legend(loc="upper center", ncol=3, fontsize=17, frameon=False)
        return fig

    def provenance(self):
        fig, axes = self.canvas(2)
        history = self.read("historical_addis_winner_membership")
        lots = self.read("bishoftu_primary_lot_check")
        for ax, vals, labels, title, ymax in [
            (
                axes[0, 0],
                [history.role.eq("train").sum(), history.role.eq("TOR_test").sum()],
                ["Fit", "TOR test"],
                "Historical 440-filter cohort",
                480,
            ),
            (
                axes[0, 1],
                [
                    lots.ReferenceSource.eq("shipped").sum(),
                    lots.ReferenceSource.eq("reconstructed").sum(),
                ],
                ["Shipped", "Reconstructed"],
                "Bishoftu: all 40 filters are lot 251",
                44,
            ),
        ]:
            bottom = 0
            for v, label, color in zip(vals, labels, [BLUE, GREY]):
                b = ax.bar([0], [v], bottom=bottom, width=0.5, color=color, label=label)
                ax.bar_label(
                    b,
                    label_type="center",
                    fontsize=23,
                    color="white" if color == BLUE else INK,
                    fontweight="bold",
                    labels=[str(v)],
                )
                bottom += v
            ax.set(ylim=(0, ymax), xlim=(-0.7, 0.7), xticks=[])
            self.style(ax, ylabel="Filters", title=title)
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=18, frameon=False
            )
        return fig

    def proposed_split(self):
        fig, axes = self.canvas()
        ax = axes[0, 0]
        split = self.read("split_counts")
        parts = [split.loc[split.season.eq(s)].set_index("role") for s in self.seasons]
        x, bottom = np.arange(3), np.zeros(3)
        for role, color in [("selection", BLUE), ("validation", GREY)]:
            values = np.array([p.loc[role, "n"] for p in parts])
            b = ax.bar(x, values, bottom=bottom, color=color, width=0.58, label=role.title())
            ax.bar_label(
                b,
                label_type="center",
                fontsize=25,
                fontweight="bold",
                color="white" if role == "selection" else INK,
                labels=[str(v) for v in values],
            )
            bottom += values
        ax.set_xticks(x, [s.split(" ")[0] for s in self.seasons])
        ax.set_ylim(0, 125)
        self.style(ax, ylabel="Addis filters")
        ax.legend(loc="upper right", ncol=2, fontsize=18, frameon=False)
        return fig

    def uncertainty(self):
        fig, axes = self.canvas(2)
        for j, term in enumerate(["slope", "intercept"]):
            ax = axes[0, j]
            for i, (season, color) in enumerate(zip(self.seasons, self.colors)):
                r = self.stats(season)
                estimate = r["deming_" + term]
                error = np.array(
                    [[estimate - r[term + "_ci_low"]], [r[term + "_ci_high"] - estimate]]
                )
                ax.errorbar(
                    estimate,
                    2 - i,
                    xerr=error,
                    fmt="o",
                    color=color,
                    capsize=6,
                    markersize=9,
                    elinewidth=3,
                )
            ax.axvline(1 if j == 0 else 0, color="#7B8990", ls="--", lw=1.5)
            ax.set_yticks([2, 1, 0], [s.split(" ")[0] for s in self.seasons])
            ax.set_ylim(-0.6, 2.6)
            ax.set_xlim((0, 1.25) if j == 0 else (-2.5, 1))
            self.style(ax, xlabel="Deming slope" if j == 0 else "Intercept (µg/m³)", grid="x")
        return fig

    def benchmark(self):
        fig, axes = self.canvas()
        ax = axes[0, 0]
        values = [self.stats(s).TOR_R2 for s in self.seasons]
        values += [
            self.stats("All Addis").TOR_R2,
            self.stats("All Addis", "historical_ocec440").TOR_R2,
        ]
        b = ax.bar(
            ["Dry", "Belg", "Kiremt", "All Addis", "Historical\nreference"],
            values,
            color=self.colors + ["#7B959F", BLUE],
            width=0.62,
        )
        ax.bar_label(b, fmt="%.3f", padding=5, fontsize=22, fontweight="bold", color=INK)
        ax.axhline(0.85, color="#A85F55", ls="--", lw=1.4)
        ax.text(
            0.01,
            0.865,
            "Prior screen: 0.85",
            transform=ax.get_yaxis_transform(),
            color="#914D44",
            fontsize=16,
        )
        ax.set_ylim(0, 1.06)
        self.style(ax, ylabel="TOR R²")
        return fig

    def full_spectra(self, season):
        fig, axes = self.canvas(2)
        slug = "".join(c if c.isalnum() else "_" for c in season).strip("_")
        cohort = self.read(f"cohort_no_co2__{slug}")
        ids = cohort.loc[cohort.role.eq("train"), "AnalysisId"]
        training = self.library["corrected"][self.positions.loc[ids].to_numpy()]
        target = self.X[self.addis.season.eq(season)]
        color = ETHIOPIA_SEASONS[season]["color"]
        lo = min(training.min(), target.min())
        hi = max(training.max(), target.max())
        for ax, values, c, title in [
            (axes[0, 0], training, "#687A84", f"{len(training)} IMPROVE training filters"),
            (axes[0, 1], target, color, f"All {len(target)} Addis filters"),
        ]:
            ax.plot(self.wn, values.T, color=c, alpha=0.15, lw=0.5)
            ax.plot(self.wn, np.median(values, axis=0), color=c, lw=2.7)
            ax.axvspan(1800, 2500, color="#ADB6BB", alpha=0.15)
            ax.set(xlim=(self.wn.max(), self.wn.min()), ylim=(lo - (hi - lo) * 0.04, hi * 1.04))
            self.style(
                ax, "Wavenumber (cm⁻¹)", "Corrected absorbance" if ax is axes[0, 0] else "", title
            )
        return fig
