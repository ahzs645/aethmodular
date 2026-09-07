# %% [markdown]
# # ftir_53 — committee backup figures: does the diagnosis generalize, and two lesson fixes
#
# ## tl;dr
#
# Companion to ftir_51 (the priority figure packet). It exports the figures the
# 2026-09-01 committee deck lacks, drawn from committed tables only (no live explorer
# needed), in the deck style (title-free, RGB PNG, settings line per figure) into
# `output/plots/ftir53/` with a `MANIFEST.md` and a zip:
#
# 1. **Five-site intercepts and slopes** (York/EIV, per-deployed-line blank quadratic,
#    from ftir_47). The same locked calibration gives a near-zero intercept at Bishoftu,
#    Beijing and Pasadena and a large one at Addis (and Delhi); the slope anomalies are
#    Delhi and Pasadena. This is the "does it generalize?" backup slide.
# 2. **Bishoftu replication**: the 26 accepted filters and the 14 unseen January to
#    February filters (locked before the raw-HIPS reconstruction, ftir run_locked_
#    reconstruction_confirmation) on one crossplot with Addis behind them.
# 3. **Protocol swing by variant** (lesson 2 fix): the intercept moves by under
#    0.25 µg/m³ across protocols for baseline-corrected cohorts and by up to 2.4 for raw
#    ones, so "protocol is not the driver" is a statement about corrected calibrations.
# 4. **Kiremt/Dry across all seven variants** (lesson 3 fix): the ratio is 1.8 to 2.6
#    over every variant, 2.0 to 2.2 for the four the deck plots.
# 5. **The mass term by site** (ftir_46, speaker-notes backup for the year-end decision):
#    HIPS absorption per unit filter mass at fixed FTIR EC, inside the Addis loading
#    range: Addis and Bishoftu carry two to three times the pooled non-Ethiopian sites.
#
# ## Context & Methods
#
# The deck's slides 6 to 8 argue from Addis alone. The cross-site evaluation
# (`calibration_explorer/CROSS_SITE_EVALUATION_2026-08-22.md`, York re-fit in
# `OFFSET_ADJUDICATION_2026-08-23.md`, per-line blank correction in ftir_47) is the
# evidence that the calibration is not being tuned to one site: it is applied unchanged
# to five SPARTAN sites and fails in a site-specific way. Every number here is read
# from a committed table; nothing is fitted except the OLS lines drawn on the Bishoftu
# crossplot, whose coefficients are printed beside the York numbers they should match.
#
# ### Key assumptions
#
# - Reference axis is HIPS Fabs/10 (MAC 10) everywhere, per the deck.
# - The five-site fits use the per-deployed-line quadratic blank line (ftir_47); the
#   deployed-line numbers are within 0.03 µg/m³ of them at every site.
# - The 14-filter Bishoftu holdout uses reconstructed Fabs (dated-schedule blank line),
#   not official processed HIPS; that caveat belongs on the slide.
# - Protocol and plausibility tables are ftir_51's exports (`output/tables/ftir51/`),
#   so the numbers match appendix A3 and A5 of the deck exactly.

# %%
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown

PLOTS = Path("output/plots/ftir53")
TABLES = Path("output/tables/ftir53")
PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

INK, GREY, BLUE, PURPLE, ACCENT = "#1F1F1F", "#8F8C84", "#2C6E9E", "#7A4FA3", "#B23327"
AMBER, GREEN = "#C8862B", "#4E8A5B"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E8E6E1", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "white",
                     "savefig.dpi": 200, "savefig.facecolor": "white"})
MANIFEST = []


def save(fig, name, settings):
    path = PLOTS / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path)
    if im.mode != "RGB":          # PowerPoint chokes on RGBA
        im.convert("RGB").save(path)
    MANIFEST.append((name, settings))
    print(f"saved {path}")


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    return b, a


# %% [markdown]
# ## 1. Five sites, one calibration: where the intercept and slope anomalies are

# %%
york = pd.read_csv("output/tables/ftir47/five_target_york_by_variant.csv")
SITES = ["Addis", "Delhi", "Beijing", "Pasadena", "Bishoftu"]
CONFIGS = {"locked ocec-800 AIRSpec k=5": ("Locked lowest-OC/EC 800 + AIRSpec (k = 5)", BLUE, "o"),
           "winner ocec-440 AIRSpec k=8": ("Dense-sweep winner 440 + AIRSpec (k = 8)", PURPLE, "s")}
lq = york[york.variant == "line_quad"].set_index(["config", "site"])
dep = york[york.variant == "deployed"].set_index(["config", "site"])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
ypos = np.arange(len(SITES))[::-1]
for j, (cfg, (label, colour, marker)) in enumerate(CONFIGS.items()):
    off = 0.18 if j == 0 else -0.18
    for ax, col, se in ((axes[0], "intercept", "intercept_se"), (axes[1], "slope", "slope_se")):
        vals = [lq.loc[(cfg, s), col] for s in SITES]
        errs = [lq.loc[(cfg, s), se] for s in SITES]
        ax.errorbar(vals, ypos + off, xerr=errs, fmt=marker, color=colour, ms=6.5,
                    capsize=3, lw=1.2, label=label)
axes[0].axvline(0, color=INK, lw=0.9)
axes[1].axvline(1, color=INK, lw=0.9)
axes[0].set(yticks=ypos, yticklabels=[f"{s}\n(n = {int(lq.loc[(list(CONFIGS)[0], s), 'n_used'])})"
                                       for s in SITES],
            xlabel="York intercept vs HIPS Fabs/10 (µg/m³) ± 1 se")
axes[1].set(xlabel="York slope ± 1 se")
for ax in axes:
    ax.grid(axis="y", visible=False)
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.5, 1.04))
fig.tight_layout()
five_site_table = (lq.reset_index()[["config", "site", "n_used", "slope", "slope_se",
                                     "intercept", "intercept_se", "frac_below_own_line_R1"]]
                   .sort_values(["config", "site"]))
five_site_table.to_csv(TABLES / "five_site_york_line_quad.csv", index=False)
display(five_site_table.round(3))
save(fig, "01_five_site_intercepts_and_slopes.png",
     "York/EIV fits (sigma_x from HIPS_Uncertainty, lambda*), x = Fabs/10 (MAC 10), per-deployed-line "
     "quadratic blank line (ftir_47), Option A site-held-out protocol, all matched filters per site; "
     "two calibrations applied unchanged to five SPARTAN sites.")

# %% [markdown]
# The two calibrations agree on the pattern: Addis is the only site whose intercept is
# certainly negative; Beijing's is slightly positive; Bishoftu's and Pasadena's are
# within one or two standard errors of zero; Delhi's is Addis-sized but poorly
# determined. The slope anomalies are Delhi (1.8x) and Pasadena (2 to 3x), and both
# survive the blank-line correction (the earlier "Pasadena dissolves to 0.9x" was a
# pooled-lot artifact; see ftir_47).

# %% [markdown]
# ## 2. Bishoftu: the accepted 26 and the 14 filters the calibration never saw

# %%
pred = pd.read_csv("output/tables/ftir46/explorer_predictions.csv")
WIN = "winner 440 + AIRSpec k=8"
addis = pred[(pred.config == WIN) & (pred.site == "addis")]
etbi = pred[(pred.config == WIN) & (pred.site == "etbi")]
hold = pd.read_csv("output/tables/variation_closure/locked_reconstruction_predictions.csv")
hold = hold[(hold.config == "addis_winner_k8") & (hold.target == "etbi_reconstructed_holdout")]
summary = pd.read_csv("output/tables/variation_closure/locked_reconstruction_summary.csv")
hs = summary[(summary.config == "addis_winner_k8") & (summary.target == "etbi_reconstructed_holdout")].iloc[0]

groups = [("Addis, 239 filters", addis.Fabs / 10, addis.f_x, GREY, 0.45, 12),
          ("Bishoftu, accepted 26 (Oct to Dec 2025)", etbi.Fabs / 10, etbi.f_x, BLUE, 0.9, 30),
          ("Bishoftu, unseen 14 (Jan to Feb 2026)",
           hold.observed_bc_mac10_ugm3, hold.predicted_ec_ugm3, ACCENT, 0.95, 34)]
fig, ax = plt.subplots(figsize=(6.4, 5.6))
lim = 9.5
ax.plot([0, lim], [0, lim], ls=":", color=GREY, lw=1.1)
ax.axhline(0, color="#555", lw=0.8)
rows = []
for label, x, y, colour, alpha, size in groups:
    b, a = ols(x, y)
    ax.scatter(x, y, s=size, color=colour, alpha=alpha, edgecolor="none", label=label)
    xx = np.linspace(0, lim, 2)
    ax.plot(xx, b * xx + a, color=colour, lw=1.6)
    rows.append({"group": label, "n": len(x), "ols_slope": b, "ols_intercept": a,
                 "mean_residual": float(np.mean(y - x))})
bish = pd.DataFrame(rows)
bish.to_csv(TABLES / "bishoftu_replication_fits.csv", index=False)
txt = "\n".join(f"{r.group.split(',')[0]}: {r.ols_slope:.2f}x {r.ols_intercept:+.2f}, mean resid {r.mean_residual:+.2f}"
                for r in bish.itertuples())
ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", ec="#DDD"))
ax.set(xlim=(0, lim), ylim=(min(-1.0, float(min(g[2].min() for g in groups)) - 0.3), lim),
       xlabel="HIPS EC-equivalent, Fabs/10 (µg/m³)", ylabel="FTIR-predicted EC (µg/m³)")
ax.legend(loc="lower right", frameon=False, fontsize=8, bbox_to_anchor=(0.98, 0.02))
fig.tight_layout()
display(bish.round(3))
print(f"holdout summary (locked confirmation): OLS {hs.ols_slope:.2f}x {hs.ols_intercept:+.2f}, "
      f"mean bias {hs.mean_bias:+.2f}, n = {int(hs.n)}, reference = {hs.reference_source}")
save(fig, "02_bishoftu_replication.png",
     "Dense-sweep winner (lowest-OC/EC 440, AIRSpec, k = 8, Option A) applied unchanged to Addis (239), "
     "the 26 accepted Bishoftu filters (official HIPS) and the 14 January to February 2026 Bishoftu "
     "filters (Fabs reconstructed from raw HIPS via the dated-schedule blank line; configs locked before "
     "reconstruction). x = Fabs/10, MAC 10; OLS lines per group.")

# %% [markdown]
# The Addis offset is not a property of the calibration: the same model applied 45 km
# away lands on or above the 1:1 line for both the accepted filters and the filters it
# had never seen. The holdout's flatter slope (0.66x) is on 14 dry-season filters with
# reconstructed Fabs and should be read as "near-zero intercept replicates", not as a
# slope estimate.

# %% [markdown]
# ## 3. Lesson 2, stated precisely: protocol swing is a raw-spectra problem

# %%
cv = pd.read_csv("output/tables/ftir51/cv_protocol_comparison.csv")
sw = (cv.groupby("variant")["Deming int."].agg(lambda s: s.max() - s.min())
      .rename("intercept_swing").reset_index())
order = ["network_raw", "ocec800_raw", "analogs_selcorr", "ocec800_corr"]
order = [v for v in order if v in set(sw.variant)] + [v for v in sw.variant if v not in order]
sw = sw.set_index("variant").loc[order].reset_index()
sw["corrected"] = sw.variant.str.contains("corr") & ~sw.variant.str.contains("selcorr")
sw.to_csv(TABLES / "protocol_intercept_swing.csv", index=False)

fig, ax = plt.subplots(figsize=(7.2, 3.6))
y = np.arange(len(sw))[::-1]
for i, r in sw.iterrows():
    sub = cv[cv.variant == r.variant]
    ax.plot(sub["Deming int."], [y[i]] * len(sub), color=GREY, lw=1.2, zorder=1)
    for _, p in sub.iterrows():
        mk = {"Option A": "o", "Option B": "s", "Option B2": "^"}.get(p.protocol, "o")
        ax.scatter(p["Deming int."], y[i], marker=mk, s=42, zorder=2,
                   color=BLUE if r.corrected else AMBER, edgecolor="white", lw=0.6)
    ax.text(sub["Deming int."].max() + 0.15, y[i], f"swing {r.intercept_swing:.2f}",
            va="center", fontsize=8.5, color=INK)
ax.axvline(0, color=INK, lw=0.9)
labels = {"network_raw": "Entire network, raw", "ocec800_raw": "Lowest-OC/EC 800, raw",
          "analogs_selcorr": "Analogs 500, selected corrected, calibrated raw",
          "ocec800_corr": "Lowest-OC/EC 800, baseline-corrected"}
ax.set(yticks=y, yticklabels=[labels.get(v, v) for v in sw.variant],
       xlabel="Addis Deming intercept, MAC 10, fixed 190 filters (µg/m³)")
ax.grid(axis="y", visible=False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, label="Option A: site-grouped 5-fold"),
                   Line2D([], [], marker="s", ls="", color=INK, label="Option B: interleaved 10-fold, within 5%"),
                   Line2D([], [], marker="^", ls="", color=INK, label="Option B2: interleaved 10-fold, first major min")],
          loc="upper right", frameon=False, fontsize=8)
ax.set_xlim(right=1.5)
fig.tight_layout()
display(sw.round(2))
save(fig, "03_protocol_swing_by_variant.png",
     "Deming intercept (MAC 10, fixed 190 Addis filters, training lot all) under three CV protocols per "
     "cohort, from ftir_51's cv_protocol_comparison (deck appendix A3). Amber = raw spectra, blue = "
     "baseline-corrected.")

# %% [markdown]
# ## 4. Lesson 3, stated precisely: Kiremt/Dry across all seven variants

# %%
pl = pd.read_csv("output/tables/ftir51/plausibility_markers.csv")
pl["corrected"] = pl.variant.str.contains("corr") & ~pl.variant.str.contains("selcorr")
pl = pl.sort_values("Kiremt/Dry", ascending=True).reset_index(drop=True)
FOUR = {"network_raw", "analogs_selcorr", "ocec800_raw", "ocec800_corr"}
fig, ax = plt.subplots(figsize=(7.2, 3.8))
y = np.arange(len(pl))
band = pl[pl.variant.isin(FOUR)]["Kiremt/Dry"]
ax.axvspan(band.min(), band.max(), color="#EEF3F7", zorder=0)
ax.axvline(1, color=INK, lw=0.9)
for i, r in pl.iterrows():
    ax.scatter(r["Kiremt/Dry"], y[i], s=70 if r.variant in FOUR else 44,
               color=BLUE if r.corrected else AMBER,
               edgecolor=INK if r.variant in FOUR else "white", lw=0.9, zorder=3)
    ax.text(r["Kiremt/Dry"] + 0.04, y[i], f"{r['Kiremt/Dry']:.2f}  (k = {int(r.k)})",
            va="center", fontsize=8.5)
ax.set(yticks=y, yticklabels=pl.variant, xlim=(0.8, pl["Kiremt/Dry"].max() + 0.6),
       xlabel="Kiremt / Dry ratio of median predicted EC, all 239 dated Addis filters")
ax.grid(axis="y", visible=False)
ax.text(band.min(), len(pl) - 0.4, f"the four variants on the deck: {band.min():.2f} to {band.max():.2f}",
        fontsize=8.5, color=INK)
fig.tight_layout()
pl[["variant", "k", "Dry med.", "Belg med.", "Kiremt med.", "Kiremt/Dry", "corrected"]].to_csv(
    TABLES / "kiremt_dry_all_variants.csv", index=False)
display(pl[["variant", "k", "Kiremt/Dry"]].round(2))
save(fig, "04_kiremt_dry_all_variants.png",
     "Kiremt/Dry ratio of median predicted EC for every variant in ftir_51's plausibility table "
     "(Option A, MAC 10, training lot all, all 239 dated evaluation filters). Outlined markers are "
     "the four variants plotted on the deck's slide 8.")

# %% [markdown]
# ## 5. Speaker-notes backup: the mass term by site (ftir_46)

# %%
ms = pd.read_csv("output/tables/ftir46/cross_site_mass_slopes.csv")
ms = ms[(ms.config == "locked 800 + AIRSpec (rule k)") & (ms.framing == "per deposit")]
order = ["Addis Ababa", "Bishoftu", "Beijing", "Delhi", "Pasadena", "pooled: 4 non-Addis sites"]
ms = ms.set_index("site").loc[order].reset_index()
fig, ax = plt.subplots(figsize=(7.0, 3.6))
y = np.arange(len(ms))[::-1]
for i, r in ms.iterrows():
    eth = r.site in ("Addis Ababa", "Bishoftu")
    ax.errorbar(r.mass_coef_addis_range, y[i],
                xerr=[[r.mass_coef_addis_range - r.mass_coef_addis_range_lo],
                      [r.mass_coef_addis_range_hi - r.mass_coef_addis_range]],
                fmt="o", color=ACCENT if eth else BLUE, ms=6.5, capsize=3, lw=1.2)
    ax.text(r.mass_coef_addis_range_hi + 0.004, y[i], f"n = {int(r.n_addis_range)}",
            va="center", fontsize=8.5, color=GREY)
ax.axvline(0, color=INK, lw=0.9)
ax.set(yticks=y, yticklabels=ms.site.str.replace("pooled: 4 non-Addis sites", "Pooled, four non-Ethiopian sites"),
       xlabel="HIPS EC-equivalent per µg of filter mass at fixed FTIR EC (µg/µg), inside the Addis loading range, 95% CI")
ax.grid(axis="y", visible=False)
fig.tight_layout()
ms[["site", "n_addis_range", "mass_coef_addis_range", "mass_coef_addis_range_lo",
    "mass_coef_addis_range_hi"]].to_csv(TABLES / "mass_term_by_site.csv", index=False)
display(ms[["site", "n_addis_range", "mass_coef_addis_range", "mass_coef_addis_range_lo",
            "mass_coef_addis_range_hi"]].round(3))
save(fig, "05_mass_term_by_site.png",
     "Coefficient on gravimetric mass per deposit in Fabs ~ f(X) + mass, locked lowest-OC/EC 800 + "
     "AIRSpec f(X), filters restricted to the Addis loading range, per-filter bootstrap 95% CI (ftir_46).")

# %% [markdown]
# ## Manifest and packet

# %%
lines = ["# ftir_53 committee backup figures", "",
         "Each figure is title-free (the deck sets the title). Settings line per figure:", ""]
for name, settings in MANIFEST:
    lines += [f"- `{name}`: {settings}"]
(PLOTS / "MANIFEST.md").write_text("\n".join(lines) + "\n")
with zipfile.ZipFile(PLOTS / "ftir53_committee_backup_figures.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name, _ in MANIFEST:
        z.write(PLOTS / name, name)
    z.write(PLOTS / "MANIFEST.md", "MANIFEST.md")
display(Markdown("\n".join(lines)))

# %% [markdown]
# ## Suggested slide copy (claim titles, no em dashes)
#
# **Backup slide A7 (figure 01): "The same calibration fails only where the aerosol is different"**
# Caption: York fits of FTIR-predicted EC against HIPS Fabs/10 at five SPARTAN sites, one
# locked calibration applied unchanged. Addis is the only site with a certainly negative
# intercept; Bishoftu, Beijing and Pasadena sit near zero; the slope anomalies are Delhi and
# Pasadena.
#
# **Backup slide A8 (figure 02): "Forty-five kilometres away, the offset is gone, on filters
# the model never saw"** Caption: the dense-sweep winner at Addis, at the 26 accepted
# Bishoftu filters, and at 14 Bishoftu filters analysed after the configuration was locked.
# Reconstructed Fabs for the 14; dry season only.
#
# **Slide 8, lesson 2 rewrite:** "For baseline-corrected calibrations the cross-validation
# protocol moves the intercept by under 0.25 µg/m³; on raw spectra it moves it by up to
# 2.4. Lot 248 and lot 251 give the same answer." (figure 03 as backup)
#
# **Slide 8, lesson 3 rewrite:** "Rainy-season EC is 1.8 to 2.6 times the dry season across
# all seven variants, 2.0 to 2.2 for the four shown; no variant predicts negative EC."
# (figure 04 as backup)
#
# **Speaker note for slide 11 (figure 05):** the year-end decision has a concrete
# experiment attached: at fixed FTIR EC, HIPS absorption rises with filter mass at every
# site, and the Ethiopian sites carry two to three times the pooled rate. A HIPS loading
# test on non-absorbing deposits separates the instrument share from the aerosol share.
#
# ## Takeaways
#
# - The cross-site pattern is the deck's missing evidence that the work is diagnosis, not
#   tuning; one backup slide carries it.
# - Lessons 2 and 3 are right in spirit and wrong in wording; the corrected sentences are
#   above and the figures back them from the deck's own appendix tables.
#
# ## Limits
#
# - Bishoftu is 26 + 14 dry-season filters; the 14 use reconstructed Fabs.
# - Delhi's intercept is poorly determined (se 0.8); do not call it "Addis-sized" without the se.
# - Figure 05 is a within-project result from today (ftir_46); keep it in notes, not on a slide.
