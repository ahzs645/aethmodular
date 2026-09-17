"""Rebuild the meeting-requested Adama evidence from measured filter data.

Run from the repository root with ``uv run python .../prepare_adama_summary_20260910.py``.
No exclusions are added. Pairing flags identify comparability questions and remain
visible. All filter concentrations use each sampler's own recorded air volume.
"""

from pathlib import Path
import hashlib
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
ANALYSIS = REPO / "research/ftir_hips_chem"
sys.path.insert(0, str(ANALYSIS / "scripts"))
sys.path.insert(0, str(REPO / "research/ftir_ec_phase3/scripts"))
from config import MAC_VALUE, SITES
from outliers import apply_exclusion_flags, get_clean_data
from data_matching import load_filter_data, base_filter_id
from plotting import PlotConfig
from plotting.overlays import crossplot_on_axes
from plotting.utils import calculate_regression_stats, style_axes
from phase3_common import PATHS
from theory_test_suite import davis_root

PlotConfig.set(
    sites="all", layout="individual", show_stats=True, show_1to1=True, font_size=12, title_size=14
)
OUT = ANALYSIS / "output/tables/adama_summary_20260910"
FIG = ANALYSIS / "output/plots/adama_summary_20260910"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED = "#22252A", "#666B72"
ADDIS = SITES["Addis_Ababa"]["color"]
ADAMA, BISHOFTU = "#C49442", "#7A4FA3"  # Existing deck colors for non-config sites.
BLUE, GREY = "#2C6E9E", "#8F8C84"
SOURCES = []


def read(path, **kwargs):
    path = Path(path)
    SOURCES.append(str(path))
    return pd.read_csv(path, **kwargs)


def save(fig, name):
    fig.savefig(FIG / (name + ".png"), bbox_inches="tight")
    plt.close(fig)


root = davis_root()
hips = read(root / "DAVIS/CSU_AMOD/csu_amod_HIPS_Batch_54.csv", encoding="cp1252")
ftir = read(root / "DAVIS/CSU_AMOD/csu_amod_FTIR_Batch_54.csv", encoding="cp1252")
carbon = read(root / "DAVIS/Adama TOR/Carbon_concs_Batch54.csv")
ptfe = ftir.pivot(
    index=["FilterId", "SampleDate", "Volume_m3"], columns="Parameter", values="Concentration_ug_m3"
).reset_index()
ptfe["date"] = pd.to_datetime(ptfe.SampleDate).dt.normalize()
quartz = carbon.pivot(
    index=["FilterId", "SampleDate", "Volume_liters"],
    columns="Parameter",
    values="Concentration_ug_m3",
).reset_index()
quartz["date"] = pd.to_datetime(quartz.SampleDate).dt.normalize()
q = ptfe.merge(
    hips[["FilterId", "Fabs", "tau", "Uncertainty", "DepositArea", "Volume"]],
    on="FilterId",
    validate="one_to_one",
)
assert np.allclose(q.Volume_m3, q.Volume)
q = q.merge(quartz, on="date", validate="one_to_one", suffixes=("_ptfe", "_quartz"))
q = q.sort_values("date").reset_index(drop=True)
q["start_offset_min"] = (
    pd.to_datetime(q.SampleDate_quartz) - pd.to_datetime(q.SampleDate_ptfe)
).dt.total_seconds() / 60
q["volume_ratio"] = q.Volume_m3 / (q.Volume_liters / 1000)
q["pairing_flag"] = ""
q.loc[q.start_offset_min.abs() > 5, "pairing_flag"] = "start offset >5 min"
q.loc[(q.volume_ratio - 1).abs() > 0.2, "pairing_flag"] += "volume mismatch >20%"
q = apply_exclusion_flags(q, "Adama")
q = get_clean_data(q)
assert len(q) == 5
q["ec_ratio_tor"] = q.EC_ftir / q.ECTR
q["ec_ratio_tot"] = q.EC_ftir / q.ECTT
q["hips_ec"] = q.Fabs / MAC_VALUE
q["oc_ec"] = q.OCTR / q.ECTR
q["oc_fabs_tor"] = q.OCTR / q.Fabs
q["oc_fabs_ftir"] = q.OC_ftir / q.Fabs
q["char_soot"] = (q.EC1 - q.OPTR) / (q.EC2 + q.EC3)
q["implied_mac_tor"] = q.Fabs / q.ECTR
q["implied_mac_tot"] = q.Fabs / q.ECTT
q["implied_mac_ftir"] = q.Fabs / q.EC_ftir
q.to_csv(OUT / "adama_pairs.csv", index=False)
q.to_json(OUT / "adama_pairs.json", orient="records", date_format="iso", indent=2)
flagged = q.pairing_flag.ne("")

# The standard evaluation loader's existing local output avoids re-reading all
# 2,700 spectral columns. Independently verify its HIPS and volume joins below.
ev = read(ANALYSIS / "output/tables/ann_weekly_20260910/addis_evaluation.csv")
h = read(PATHS.spartan_hips_primary, encoding="cp1252")
h = h.drop_duplicates("FilterId")
ev = ev.merge(
    h[["FilterId", "tau", "DepositArea", "Uncertainty", "Fabs", "Volume"]],
    left_on="ExternalFilterId",
    right_on="FilterId",
    validate="one_to_one",
    suffixes=("", "_source"),
)
assert np.allclose(ev.Fabs, ev.Fabs_source)
assert np.allclose(ev.Volume_m3, ev.Volume)
ev["date"] = pd.to_datetime(ev["date"])
ev = apply_exclusion_flags(ev, "Addis_Ababa")
ev = get_clean_data(ev)
ev["has_ec_pair"] = np.isfinite(ev.EC_deployed_ugm3)
a = ev.loc[ev.has_ec_pair].copy()
assert len(ev) == 239 and len(a) == 190
a["hips_ec"] = a.Fabs / MAC_VALUE
sigma_x = a.Uncertainty.median() / MAC_VALUE
sigma_y = 0.531  # Existing AIRSpec held-out TOR RMSE, an approximate error model.
stats = calculate_regression_stats(
    a.hips_ec, a.EC_deployed_ugm3, errors_in_variables=True, sigma_x=sigma_x, sigma_y=sigma_y
)
a["ols_residual"] = a.EC_deployed_ugm3 - (stats["intercept"] + stats["slope"] * a.hips_ec)
a["volume_display_group"] = np.where(a.Volume_m3 < 6, "<6 m³", "6–8 m³")
# This is a display stratum, never an exclusion or an inferred sampling regime.
volume_counts = a.groupby("volume_display_group").agg(
    n=("FilterId", "size"),
    volume_min=("Volume_m3", "min"),
    volume_max=("Volume_m3", "max"),
    fabs_min=("Fabs", "min"),
    fabs_max=("Fabs", "max"),
)
volume_counts.to_csv(OUT / "addis_volume_counts.csv")
ev.to_csv(OUT / "addis_hips_population.csv", index=False)
a.to_csv(OUT / "addis_ec_pairs.csv", index=False)
b = h.loc[h.Site.eq("ETBI") & h.FilterType.eq("PM2.5") & h.Fabs.notna()].copy()
b = get_clean_data(apply_exclusion_flags(b, "Bishoftu"))
assert len(b) == 26


def panel(ax, x, y, xlabel, ylabel, color, fit=False, stat=None, lim=None):
    crossplot_on_axes(
        ax, x, y, xlabel, ylabel, color=color, fit_line=fit, stats_box=False, stats=stat
    )
    ax.collections[0].set_sizes(np.full(len(x), 38 if len(x) > 10 else 72))
    ax.collections[0].set_linewidths(0.5)
    ax.collections[0].set_alpha(0.75)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    for line in ax.lines:
        if line.get_label() == "Best fit":
            line.set_color(INK)
    if lim is not None:
        ax.set(xlim=(0, lim), ylim=(0, lim))
        for line in ax.lines:
            if line.get_label() == "1:1 line":
                line.set_data([0, lim], [0, lim])
    ax.set_aspect("equal", adjustable="box")


def pair_flags(ax, x, y, labels=True):
    ax.scatter(
        np.asarray(x)[flagged],
        np.asarray(y)[flagged],
        s=155,
        facecolors="none",
        edgecolors=MUTED,
        linewidths=1.7,
        zorder=5,
    )
    if labels:
        for i, (xx, yy) in enumerate(zip(x, y)):
            dx, dy = [(7, 7), (7, -15), (7, 7), (-32, -16), (7, 7)][i]
            ax.annotate(
                q.date.iloc[i].strftime("%d Jul"),
                (xx, yy),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
                color=MUTED,
            )


# Concentration and deposit optical depth answer different loading questions.
fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.9), layout="constrained")
for ax, col, xlabel, bins in [
    (axs[0], "Fabs", "HIPS absorption, Fabs (Mm⁻¹)", np.arange(0, 101, 5)),
    (axs[1], "tau", "HIPS filter optical depth, τ", np.arange(0, 1.71, 0.1)),
]:
    ax.hist(
        ev[col],
        bins=bins,
        density=True,
        color=ADDIS,
        alpha=0.28,
        label=f"Addis (n={len(ev)}, median {ev[col].median():.2f})",
    )
    ax.hist(
        b[col],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color=BISHOFTU,
        label=f"Bishoftu (n={len(b)}, median {b[col].median():.2f})",
    )
    for i, val in enumerate(q[col]):
        ax.axvline(val, color=ADAMA, linewidth=1.5, alpha=0.7)
    ax.plot([], [], color=ADAMA, label=f"Adama: 5 individual dates, median {q[col].median():.2f}")
    style_axes(ax, xlabel, "Density", show_legend=False)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
axs[0].set_title("Adama absorption overlaps Addis", loc="left")
axs[1].set_title("Adama filters have lower optical depth", loc="left")
save(fig, "context")

# Preserve the original Addis comparison and add the same HIPS axis for Adama.
fig, axs = plt.subplots(1, 2, figsize=(12.2, 5.2), layout="constrained")
panel(
    axs[0],
    a.hips_ec,
    a.EC_deployed_ugm3,
    "HIPS Fabs / MAC (µg/m³)",
    "Deployed FTIR EC (µg/m³)",
    ADDIS,
    True,
    stats,
    13,
)
axs[0].set_title("Addis: 190 pairs", loc="left")
axs[0].text(
    0.04,
    0.97,
    f"OLS {stats['slope']:.2f}x {stats['intercept']:+.2f}\n"
    f"Deming {stats['deming_slope']:.2f}x {stats['deming_intercept']:+.2f}\n"
    f"R² = {stats['r_squared']:.2f}, λ ≈ {stats['deming_lambda']:.2f}",
    transform=axs[0].transAxes,
    va="top",
    fontsize=10,
    bbox=dict(fc="white", ec="none", alpha=0.9),
)
panel(
    axs[1],
    q.hips_ec,
    q.EC_ftir,
    "HIPS Fabs / MAC (µg/m³)",
    "Deployed FTIR EC (µg/m³)",
    ADAMA,
    lim=13,
)
pair_flags(axs[1], q.hips_ec, q.EC_ftir, labels=False)
axs[1].set_title("Adama: 5 pairs, July 2024", loc="left")
axs[1].text(
    0.04,
    0.97,
    "Same axis limits and MAC = "
    + str(MAC_VALUE)
    + " m²/g\nNo fitted line for the five Adama dates",
    transform=axs[1].transAxes,
    va="top",
    fontsize=10,
)
save(fig, "same_reference")

fig, axs = plt.subplots(1, 3, figsize=(14.1, 4.8), layout="constrained")
for ax, x, y, xlabel, ylabel, title in [
    (axs[0], q.hips_ec, q.EC_ftir, "HIPS Fabs / MAC", "FTIR EC", "FTIR versus HIPS"),
    (axs[1], q.ECTR, q.EC_ftir, "Quartz TOR EC", "FTIR EC", "FTIR versus quartz"),
    (axs[2], q.ECTR, q.hips_ec, "Quartz TOR EC", "HIPS Fabs / MAC", "HIPS versus quartz"),
]:
    panel(ax, x, y, xlabel + " (µg/m³)", ylabel + " (µg/m³)", ADAMA, lim=6)
    pair_flags(ax, x, y)
    ax.set_title(title, fontsize=13, loc="left")
save(fig, "three_crossplots")

fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.1), layout="constrained")
panel(
    axs[0],
    a.hips_ec,
    a.EC_deployed_ugm3,
    "HIPS Fabs / MAC (µg/m³)",
    "Deployed FTIR EC (µg/m³)",
    ADDIS,
    True,
    stats,
    13,
)
points = axs[0].collections[0]
points.set_array(a.Volume_m3.to_numpy())
points.set_cmap("viridis")
points.set_norm(Normalize(2, 7.4))
points.set_alpha(0.9)
fig.colorbar(points, ax=axs[0], label="Sampled air volume (m³)", shrink=0.78, pad=0.025)
axs[0].set_title("Same crossplot, colored by volume", loc="left")
crossplot_on_axes(
    axs[1],
    a.Volume_m3,
    a.ols_residual,
    "Sampled air volume (m³)",
    "FTIR residual from pooled OLS (µg/m³)",
    color=BLUE,
    one_to_one=False,
    equal_axes=False,
    fit_line=False,
    stats_box=False,
)
axs[1].collections[0].set_sizes(np.full(len(a), 35))
axs[1].axhline(0, color=GREY, linewidth=1)
residual_limit = float(np.ceil(a.ols_residual.abs().max() + 0.7))
axs[1].set(xlim=(1.6, 7.8), ylim=(-residual_limit, residual_limit))
if axs[1].get_legend() is not None:
    axs[1].get_legend().remove()
axs[1].set_title("Residuals at the observed volumes", loc="left")
low = a.loc[a.Volume_m3.idxmin()]
axs[1].annotate(
    low.ExternalFilterId + "\n2.07 m³",
    (low.Volume_m3, low.ols_residual),
    xytext=(9, 12),
    textcoords="offset points",
    fontsize=10,
)
axs[1].text(
    0.04,
    0.96,
    "189 pairs at 6.78–7.36 m³\n1 pair at 2.07 m³",
    va="top",
    transform=axs[1].transAxes,
    fontsize=11,
)
save(fig, "volume_crossplot")

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
for frame, label, color in [(ev, "All HIPS filters", GREY), (a, "With deployed FTIR EC", ADDIS)]:
    axs[0].scatter(
        frame.date, frame.Volume_m3, s=26, color=color, alpha=0.7, label=f"{label} (n={len(frame)})"
    )
style_axes(axs[0], "Sample date", "Sampled air volume (m³)", show_legend=False)
axs[0].legend(frameon=False, fontsize=10)
axs[0].set_title("The 190-pair subset ends in September 2024", loc="left")
axs[0].tick_params(axis="x", rotation=25)
crossplot_on_axes(
    axs[1],
    a.tau,
    a.ols_residual,
    "HIPS filter optical depth, τ",
    "FTIR residual from pooled OLS (µg/m³)",
    color=ADDIS,
    one_to_one=False,
    equal_axes=False,
    fit_line=False,
    stats_box=False,
)
axs[1].collections[0].set_sizes(np.full(len(a), 35))
axs[1].axhline(0, color=GREY, lw=1)
axs[1].set(ylim=(-residual_limit, residual_limit))
if axs[1].get_legend() is not None:
    axs[1].get_legend().remove()
rs = float(spearmanr(a.tau, a.ols_residual).statistic)
axs[1].text(
    0.04,
    0.97,
    f"Spearman r = {rs:+.2f} (r² = {rs**2:.2f})\nDescriptive loading screen",
    va="top",
    transform=axs[1].transAxes,
    fontsize=10,
)
axs[1].set_title("Optical depth complements air volume", loc="left")
save(fig, "volume_coverage")

# Convert the legacy model loadings with PTFE volume before dividing by quartz
# concentration. A ratio of masses from unequal volumes is not concentration agreement.
old = read(REPO / "research/spartan_ec_2026_06_16/tables/adama_ec_calibration_comparison.csv")
old = q[["FilterId_ptfe", "Volume_m3", "ECTR", "date", "pairing_flag"]].merge(
    old.drop(columns=["date", "ECTR"]),
    left_on="FilterId_ptfe",
    right_on="FilterId",
    validate="one_to_one",
)
for col in ["EC_general", "EC_biomass_tool", "EC_biomass_local_sig"]:
    old[col + "_ratio_conc"] = old[col] / old.Volume_m3 / old.ECTR
assert np.allclose(old.EC_general_ratio_conc, q.ec_ratio_tor, rtol=1e-5)
old.to_csv(OUT / "legacy_calibrations_concentration_ratios.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
x = np.arange(5)
for offset, col, label, color in [
    (-0.16, "EC_general", "Deployed (set 26)", BLUE),
    (0, "EC_biomass_tool", "Biomass tool", ADAMA),
    (0.16, "EC_biomass_local_sig", "Biomass local rebuild", GREY),
]:
    yy = old[col + "_ratio_conc"]
    axs[0].scatter(x + offset, yy, color=color, s=58, label=label, zorder=3)
    axs[0].scatter((x + offset)[flagged], yy[flagged], s=125, facecolors="none", edgecolors=MUTED)
axs[0].axhline(1, color=INK, lw=1)
axs[0].set(xticks=x, xticklabels=q.date.dt.strftime("%d Jul"), ylim=(0, 3.5))
style_axes(axs[0], "July 2024", "FTIR EC / quartz TOR EC (concentration)", show_legend=False)
axs[0].legend(fontsize=9, frameon=False, loc="upper left")
axs[0].set_title("Deployed median = " + f"{q.ec_ratio_tor.median():.2f}", loc="left")
axs[1].bar(x, q.char_soot, color=ADAMA, width=0.55)
axs[1].axhline(1, color=INK, lw=1, ls=":")
axs[1].set(xticks=x, xticklabels=q.date.dt.strftime("%d Jul"), ylim=(0, 1.12))
style_axes(axs[1], "July 2024", "(EC1 − OPTR) / (EC2 + EC3)", show_legend=False)
axs[1].set_title("All five thermal fraction ratios are below 1", loc="left")
save(fig, "calibration_char")

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
axs[0].fill_between(
    x, q.Fabs / MAC_VALUE, q.Fabs / 6, color=ADAMA, alpha=0.18, label="HIPS / MAC 6–10"
)
for col, label, color, marker in [
    ("ECTR", "Quartz TOR", INK, "o"),
    ("ECTT", "Quartz TOT", GREY, "s"),
    ("EC_ftir", "FTIR set 26", BLUE, "^"),
]:
    axs[0].plot(x, q[col], marker=marker, color=color, label=label)
    axs[0].scatter(x[flagged], q.loc[flagged, col], s=130, facecolors="none", edgecolors=MUTED)
axs[0].set(xticks=x, xticklabels=q.date.dt.strftime("%d Jul"), ylim=(0, 10))
style_axes(axs[0], "July 2024", "EC or EC-equivalent (µg/m³)", show_legend=False)
axs[0].legend(fontsize=9, frameon=False)
axs[0].set_title("EC values and HIPS conversion sensitivity", loc="left")
for i, (col, label, color) in enumerate(
    [
        ("implied_mac_tor", "HIPS / TOR", INK),
        ("implied_mac_tot", "HIPS / TOT", GREY),
        ("implied_mac_ftir", "HIPS / FTIR", BLUE),
    ]
):
    xx = i + (x - 2) * 0.055
    axs[1].scatter(xx, q[col], color=color, s=60)
    axs[1].scatter(xx[flagged], q.loc[flagged, col], s=135, facecolors="none", edgecolors=MUTED)
    med = q.loc[~flagged, col].median()
    axs[1].hlines(med, i - 0.26, i + 0.26, color=color, lw=2)
    axs[1].text(i + 0.28, med, f"{med:.1f}", va="center", fontsize=10)
axs[1].axhline(MAC_VALUE, color=GREY, ls=":", label=f"Assumed MAC = {MAC_VALUE}")
axs[1].set(
    xticks=range(3),
    xticklabels=["HIPS / TOR", "HIPS / TOT", "HIPS / FTIR"],
    ylim=(0, 30),
    xlim=(-0.5, 2.7),
)
style_axes(axs[1], "EC definition", "Implied MAC (m²/g)", show_legend=False)
axs[1].legend(fontsize=9, frameon=False)
axs[1].set_title("Bars show medians of the 3 unflagged pairs", loc="left")
save(fig, "three_method_mac")

models = read(
    REPO / "research/ftir_ec_phase3/output/tables/ftir44/adama_three_method_with_phase3_models.csv"
)
models = models.sort_values("date").reset_index(drop=True)
assert np.allclose(models.ECTR, q.ECTR)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
mc = [
    ("EC_ftir", "Deployed", BLUE),
    ("Locked: lowest-OC/EC 800 + AIRSpec (k=5)", "Locked 800", BISHOFTU),
    ("Sweep winner: lowest-OC/EC 440 + AIRSpec (k=8)", "Winner 440", ADAMA),
    ("EC_TOT-target, 800-intersection", "TOT target", GREY),
]
model_summary = []
for ax, ref, title in [
    (axs[0], "ECTR", "Reference: quartz TOR EC"),
    (axs[1], "ECTT", "Reference: quartz TOT EC"),
]:
    for i, (col, label, color) in enumerate(mc):
        vals = models[col] / models[ref]
        xx = i + (x - 2) * 0.055
        ax.scatter(xx, vals, color=color, s=60)
        ax.scatter(xx[flagged], vals[flagged], s=135, facecolors="none", edgecolors=MUTED)
        med = vals[~flagged].median()
        ax.hlines(med, i - 0.25, i + 0.25, color=color, lw=2)
        ax.text(i + 0.28, med, f"{med:.2f}", fontsize=10, va="center", color=color)
        model_summary.append(
            dict(model=label, reference=ref, median_unflagged=med, median_all=vals.median())
        )
    ax.axhline(1, color=INK, ls=":", lw=1)
    ax.set(xticks=range(4), xticklabels=[v[1] for v in mc], ylim=(0, 3.6), xlim=(-0.5, 3.7))
    style_axes(ax, "Calibration", "FTIR EC / thermal EC", show_legend=False)
    ax.set_title(title, loc="left")
save(fig, "locked_models")
pd.DataFrame(model_summary).to_csv(OUT / "locked_model_ratios.csv", index=False)

# Recompute ratio-context sites from the original local database definitions.
# These ratio domains require positive denominators, retained as explicit flags.
db = PATHS.ftir_dir / "local_db/tables"


def daily(filename, params):
    d = read(db / filename, usecols=["Site", "SampleDate", "Parameter", "Value"])
    d = d.loc[d.Parameter.isin(params)]
    return d.groupby(["Site", "SampleDate", "Parameter"]).Value.median().unstack("Parameter")


imp = daily("results_tor.csv", ["OC", "EC"]).join(daily("results_hips.csv", ["fAbs"]), how="outer")
imp = imp.join(daily("results_grav.csv", ["PM2.5"]), how="outer")
imp["oc_ec"] = imp.OC / imp.EC
imp["oc_fabs"] = (imp.OC / 1000) / imp.fAbs
imp["fabs_pm"] = imp.fAbs / (imp["PM2.5"] / 1000)
sp = load_filter_data()
sitevalues = []
for site, spec in SITES.items():
    z = sp.loc[
        sp.Site.eq(spec["code"])
        & sp.Parameter.isin(["OC_ftir", "EC_ftir", "HIPS_Fabs", "ChemSpec_Filter_PM2.5_mass"])
    ].copy()
    z["fid"] = z.FilterId.map(base_filter_id)
    z = z.drop_duplicates(["fid", "Parameter"])
    wide = z.pivot(index="fid", columns="Parameter", values="Concentration").reset_index()
    # All selected values in this canonical dataset are already µg/m³ or Mm⁻¹.
    meta = z.drop_duplicates("fid").set_index("fid")
    wide["date"] = pd.to_datetime(wide.fid.map(meta.SampleDate), errors="coerce")
    wide["filter_id"] = wide.fid.map(meta.FilterId)
    wide = get_clean_data(apply_exclusion_flags(wide, site))
    for ratio, num, den in [
        ("oc_ec", "OC_ftir", "EC_ftir"),
        ("oc_fabs", "OC_ftir", "HIPS_Fabs"),
        ("fabs_pm", "HIPS_Fabs", "ChemSpec_Filter_PM2.5_mass"),
    ]:
        good = np.isfinite(wide[num]) & np.isfinite(wide[den]) & wide[num].gt(0) & wide[den].gt(0)
        wide["ratio_eligible_" + ratio] = good
        sitevalues.append(
            dict(
                site="Addis" if site == "Addis_Ababa" else "Pasadena" if site == "JPL" else site,
                ratio=ratio,
                value=(wide.loc[good, num] / wide.loc[good, den]).median(),
                n=int(good.sum()),
                color=spec["color"],
                basis="HIPS/gravimetry" if ratio == "fabs_pm" else "FTIR",
            )
        )
sitevalues += [
    dict(site="Adama", ratio="oc_ec", value=q.oc_ec.median(), n=5, color=ADAMA, basis="TOR"),
    dict(
        site="Adama", ratio="oc_fabs", value=q.oc_fabs_tor.median(), n=5, color=ADAMA, basis="TOR"
    ),
]
bm = read(
    Path(PATHS.spartan_hips_primary).parents[1] / "DAVIS/SPARTAN FTIR pulls/ETBI/ETBI_filters.csv",
    encoding="utf-8-sig",
)
bm.columns = [c.strip('\ufeff"') for c in bm.columns]
bm["mass_conc"] = pd.to_numeric(bm.MassCollectedOnFilter) / pd.to_numeric(bm.SampleVolume_m3)
assert bm.mass_conc.median() < 1000, "Review mass units before plotting"
bpm = b.merge(
    bm[["ExternalFilterId", "mass_conc"]],
    left_on="FilterId",
    right_on="ExternalFilterId",
    validate="one_to_one",
)
bpm["ratio_eligible"] = bpm.Fabs.gt(0) & bpm.mass_conc.gt(0) & np.isfinite(bpm.mass_conc)
bpm.to_csv(OUT / "bishoftu_mass_pairs.csv", index=False)
valid = bpm.loc[bpm.ratio_eligible]
sitevalues.append(
    dict(
        site="Bishoftu",
        ratio="fabs_pm",
        value=(valid.Fabs / valid.mass_conc).median(),
        n=len(valid),
        color=BISHOFTU,
        basis="gravimetry",
    )
)
sv = pd.DataFrame(sitevalues)
sv.to_csv(OUT / "site_ratio_medians.csv", index=False)
for col, xlabel in [
    ("oc_ec", "OC / EC"),
    ("oc_fabs", "OC / Fabs (µg/m³ per Mm⁻¹)"),
    ("fabs_pm", "Fabs / PM₂.₅ (Mm⁻¹ per µg/m³)"),
]:
    imp["eligible_" + col] = np.isfinite(imp[col]) & imp[col].gt(0)
    good = imp.loc[imp["eligible_" + col], col]
    gr = good.groupby(level=0).agg(["median", "size"])
    gr = gr.loc[gr["size"] >= 100]
    gr.to_csv(OUT / ("improve_" + col + "_site_medians.csv"))
    vals = gr["median"]
    rows = sv.loc[sv.ratio.eq(col)].sort_values("value")
    fig, ax = plt.subplots(figsize=(12.6, 4.35), layout="constrained")
    jitter = np.random.default_rng(7).uniform(0.85, 1.25, len(vals))
    ax.scatter(vals, jitter, color="#C9C6BF", s=20, alpha=0.8)
    ax.text(
        0.01,
        0.97,
        f"{len(vals)} IMPROVE site medians (≥100 valid days each)",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color=MUTED,
    )
    for i, r in enumerate(rows.itertuples()):
        yy = 0.46 if i % 2 == 0 else 0.10
        ax.scatter(r.value, 0.63, s=125, color=r.color, edgecolors="white", zorder=4)
        ax.plot([r.value, r.value], [yy + 0.1, 0.58], color=r.color, lw=1)
        pct = int(round((vals <= r.value).mean() * 100))
        ordinal = "th" if 10 <= pct % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(pct % 10, "th")
        rank = (
            "below all" if pct == 0 else "above all" if pct == 100 else f"{pct}{ordinal} percentile"
        )
        ax.text(
            r.value,
            yy,
            f"{r.site}\n{r.value:.2f}\n{rank}",
            ha="center",
            va="top",
            fontsize=10,
            color=r.color,
            fontweight="bold",
        )
    ax.set_ylim(-0.3, 1.5)
    ax.set_xlim(left=0)
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_xlabel(xlabel)
    if col == "oc_fabs":
        ax.text(
            0.99,
            0.97,
            f"Adama FTIR-based ratio: {q.oc_fabs_ftir.median():.2f}\nAdama marker uses quartz TOR OC",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=MUTED,
        )
    if col == "fabs_pm":
        ax.text(
            0.99,
            0.97,
            "Adama: paired PM₂.₅ mass unavailable",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=MUTED,
        )
    save(fig, "ratio_" + col)

summary = dict(
    n_adama=5,
    n_addis_hips=len(ev),
    n_addis_ec=len(a),
    n_bishoftu=len(b),
    fabs_medians=dict(
        Addis=float(ev.Fabs.median()), Bishoftu=float(b.Fabs.median()), Adama=float(q.Fabs.median())
    ),
    tau_medians=dict(
        Addis=float(ev.tau.median()), Bishoftu=float(b.tau.median()), Adama=float(q.tau.median())
    ),
    volume_medians=dict(
        Addis=float(ev.Volume_m3.median()),
        Bishoftu=float(b.Volume.median()),
        Adama=float(q.Volume_m3.median()),
    ),
    addis_pair_dates=[str(a.date.min().date()), str(a.date.max().date())],
    addis_hips_dates=[str(ev.date.min().date()), str(ev.date.max().date())],
    addis_fit={k: float(v) for k, v in stats.items() if isinstance(v, (int, float, np.number))},
    residual_tau_spearman=rs,
    adama_ftirtor_median=float(q.ec_ratio_tor.median()),
    adama_ftirtor_range=[float(q.ec_ratio_tor.min()), float(q.ec_ratio_tor.max())],
    adama_ftirtor_unflagged_range=[
        float(q.loc[~flagged, "ec_ratio_tor"].min()),
        float(q.loc[~flagged, "ec_ratio_tor"].max()),
    ],
    adama_quartz_ec_range=[float(q.ECTR.min()), float(q.ECTR.max())],
    adama_oc_ec_median=float(q.oc_ec.median()),
    adama_implied_mac_tor_unflagged=float(q.loc[~flagged, "implied_mac_tor"].median()),
    adama_implied_mac_tot_unflagged=float(q.loc[~flagged, "implied_mac_tot"].median()),
    sigma_x=float(sigma_x),
    sigma_y=sigma_y,
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
(OUT / "sources.json").write_text(
    json.dumps(
        [
            dict(
                path=p,
                size=Path(p).stat().st_size,
                sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            )
            for p in sorted(set(SOURCES))
        ],
        indent=2,
    )
)
print(json.dumps(summary, indent=2))
