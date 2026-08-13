# %% [markdown]
# # ftir_31 — every deck figure, remade in one executed notebook
#
# ## tl;dr
#
# (filled in by the finalize step after execution)
#
# ## Context & Methods
#
# The 13 Aug 2026 briefing decks (Ann 12 Aug, Satoshi 13 Aug) embed figures whose
# generators are scattered across `build_deck_figures.py`, `build_protocol_variants.py`
# and the ftir_12/16/17 notebooks. This notebook regenerates **every deck figure in one
# place**, inline, so each image in the decks has a single executed provenance record.
#
# Regeneration strategy, per figure family:
#
# - **by_protocol set** (setup matrix, six-setup crossplots, residual-vs-D², MAC slope
#   pivot, selection curves, intercept ladder, bootstrap, cohort sweep — both protocols):
#   re-run `build_protocol_variants.main()`, which reads only committed ftir_21/22/23
#   tables.
# - **Deck-root set** (OC/EC filtering strip, combined setup matrix, the three AIRSpec
#   explainer slides, intercept ladder): re-run the `build_deck_figures.py` builders
#   (Drive `local_db` for the pool OC/EC; the cached explainer baselines for AIRSpec).
# - **Implied-MAC bridge, deployed crossplot, seasonal spectra**: rebuilt here directly
#   from the committed per-filter tables (`improve_implied_mac_bridge.csv`, phase-2
#   `addis_calibration_predictions.csv`) and the local corrected-spectra cache, with the
#   headline numbers asserted.
# - **Peak-center and Adama/ETBI context**: rebuilt from committed stats tables plus the
#   SPARTAN HIPS export; the per-filter peak-center *histogram* remains ftir_12's output
#   (regenerate with `build_notebooks.py 12`) — here its committed group stats are drawn
#   as a dot-and-IQR panel, stats-faithful to the deck claim.
#
# The July-17 charcoal panels used in the Ann deck's appendix are page extracts from an
# external PDF deck, not repo figures — they have no notebook to remake them in and are
# listed in the manifest as such.

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

sys.path.insert(0, "scripts")
sys.path.insert(0, str(Path("..") / "ftir_hips_chem" / "scripts"))

OUT = Path("output/plots/ftir31")
OUT.mkdir(parents=True, exist_ok=True)
DECK = Path("output/plots/deck")

GREY, BLUE, PURPLE, ACCENT = "#8F8C84", "#2C6E9E", "#7A4FA3", "#B23327"

def show(path, title):
    print(f"[remade] {title}: {path}")
    display(Image(filename=str(path)))

# %% [markdown]
# ### 1. The by_protocol figure set — one cell per figure, committed tables only
#
# Shared inputs for the nine per-protocol builders (`build_protocol_variants`), read
# once. Each figure below is generated **independently** in its own cell — both protocol
# folders are written each time; the site-held-out version (the one the decks use) is
# displayed first, then its calibration-app twin — both component-protocol variants
# of every calibration figure, independently.

# %%
import build_protocol_variants as bpv

T21, T22, T23 = bpv.T21, bpv.T22, bpv.T23
predictions = pd.read_csv(T21 / "addis_predictions_by_mode.csv")
metrics = pd.read_csv(T21 / "addis_metrics_by_mode.csv")
draws = pd.read_csv(T22 / "bootstrap_draws_by_mode.csv")
residuals = pd.read_csv(T22 / "addis_residuals_by_mode.csv")
sweep = pd.read_csv(T22 / "cohort_size_sweep_by_mode.csv")
sel_curves = pd.read_csv(T23 / "selection_curves.csv")
sel_decisions = pd.read_csv(T23 / "selection_decisions.csv")
BP = DECK / "by_protocol"

def both_modes(fn, *tables):
    """Run one builder for both component protocols; return {mode: path}."""
    out = {}
    for mode in bpv.MODES:
        rel = fn(mode, *tables)
        out[mode] = BP / bpv.MODE_DIR[mode] / pathlib.PurePath(rel).name
    return out

import pathlib

# %% [markdown]
# #### 1.1 Calibration setup matrix

# %%
_p = both_modes(bpv.fig_setup_matrix, metrics)
show(_p["site_heldout"], "setup matrix — site-held-out")
show(_p["app"], "setup matrix — calibration-app variant")

# %% [markdown]
# #### 1.2 Six-setup Addis crossplots

# %%
_p = both_modes(bpv.fig_crossplots, predictions, metrics)
show(_p["site_heldout"], "six-setup crossplots — site-held-out")
show(_p["app"], "six-setup crossplots — calibration-app variant")

# %% [markdown]
# #### 1.3 Residual vs Mahalanobis D²

# %%
_p = both_modes(bpv.fig_residual_vs_d2, residuals)
show(_p["site_heldout"], "residual vs D² — site-held-out")
show(_p["app"], "residual vs D² — calibration-app variant")

# %% [markdown]
# #### 1.4 MAC slope pivot

# %%
_p = both_modes(bpv.fig_mac_slope_pivot, metrics)
show(_p["site_heldout"], "MAC slope pivot — site-held-out")
show(_p["app"], "MAC slope pivot — calibration-app variant")

# %% [markdown]
# #### 1.5 Component-selection curves

# %%
_p = both_modes(bpv.fig_component_selection, sel_curves, sel_decisions)
show(_p["site_heldout"], "selection curves — site-held-out")
show(_p["app"], "selection curves — calibration-app variant")

# %% [markdown]
# #### 1.6 Intercept–slope ladder

# %%
_p = both_modes(bpv.fig_intercept_ladder, metrics)
show(_p["site_heldout"], "intercept-slope ladder — site-held-out")
show(_p["app"], "intercept-slope ladder — calibration-app variant")

# %% [markdown]
# #### 1.7 Bootstrap intercept CIs

# %%
_p = both_modes(bpv.fig_bootstrap, draws)
show(_p["site_heldout"], "bootstrap intercept CIs — site-held-out")
show(_p["app"], "bootstrap intercept CIs — calibration-app variant")

# %% [markdown]
# #### 1.8 Cohort-size sweep

# %%
_p = both_modes(bpv.fig_cohort_sweep, sweep)
show(_p["site_heldout"], "cohort-size sweep — site-held-out")
show(_p["app"], "cohort-size sweep — calibration-app variant")

# %% [markdown]
# #### 1.9 MAC effect, all setups

# %%
_p = both_modes(bpv.fig_mac_effect, predictions, metrics)
show(_p["site_heldout"], "MAC effect (all setups) — site-held-out")
show(_p["app"], "MAC effect (all setups) — calibration-app variant")

# %% [markdown]
# ### 2. The deck-root figures — one cell per figure
#
# These re-run the `build_deck_figures.py` builders (Drive `local_db` for the pool
# OC/EC; the cached explainer baselines for the AIRSpec slides).

# %%
import build_deck_figures as bdf

# %% [markdown]
# #### 2.1 OC/EC filtering strip

# %%
bdf.fig_filtering_strip()
show(DECK / "filtering_by_ocec.png", "OC/EC filtering strip")

# %% [markdown]
# #### 2.2 Combined setup matrix (both intercept columns)

# %%
bdf.fig_setup_matrix()
show(DECK / "calibration_setup_matrix.png", "combined setup matrix")

# %% [markdown]
# #### 2.3 AIRSpec explainer 1 — one real filter and its baseline

# %%
bdf.fig_airspec_1_baseline()
show(DECK / "airspec_1_baseline.png", "AIRSpec 1 — baseline")

# %% [markdown]
# #### 2.4 AIRSpec explainer 2 — after subtraction

# %%
bdf.fig_airspec_2_corrected()
show(DECK / "airspec_2_corrected.png", "AIRSpec 2 — corrected")

# %% [markdown]
# #### 2.5 AIRSpec explainer 3 — the background gap

# %%
bdf.fig_airspec_3_background_gap()
show(DECK / "airspec_3_background_gap.png", "AIRSpec 3 — background gap")

# %% [markdown]
# #### 2.6 Intercept ladder

# %%
bdf.fig_intercept_ladder()
show(DECK / "intercept_ladder.png", "intercept ladder")

# %% [markdown]
# ### 3. The implied-MAC bridge, from the committed 151,843-filter table
#
# Rebuilt directly from `improve_implied_mac_bridge.csv`: implied MAC = Fabs / TOR-EC,
# binned by OC/EC decile, with the Addis-like OC/EC ≤ 2.27 region shaded. The headline
# numbers are asserted before drawing.

# %%
bridge = pd.read_csv("output/tables/ftir16/improve_implied_mac_bridge.csv")
bridge = bridge[(bridge.EC_ugm3 > 0) & (bridge.Fabs > 0)].copy()
assert len(bridge) == 151_843, len(bridge)
med = bridge.implied_MAC.median()
sub = bridge[bridge.OC_EC <= 2.27]
assert round(med, 2) == 11.96, med
assert (len(sub), round(sub.implied_MAC.median(), 2)) == (6503, 10.05)

dec = pd.qcut(bridge.OC_EC, 10)
g = bridge.groupby(dec, observed=True)
mid = g.OC_EC.median()
q1, q2, q3 = (g.implied_MAC.quantile(q) for q in (0.25, 0.5, 0.75))
sub_x, sub_mac = sub.OC_EC.median(), sub.implied_MAC.median()

from matplotlib.ticker import FixedLocator, NullFormatter

fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.fill_between(mid, q1, q3, alpha=0.25, color=BLUE, label="IQR (per OC/EC decile)")
ax.plot(mid, q2, "o-", color=BLUE, label="Median implied MAC (decile)")
ax.plot(sub_x, sub_mac, "*", color=PURPLE, ms=17, zorder=4,
        label=f"Addis-like subset (OC/EC ≤ 2.27, n = {len(sub):,}): {sub_mac:.2f}")
ax.axhline(10, color=ACCENT, ls=":", lw=1.5)
ax.axhline(6, color=ACCENT, ls="--", lw=1.5)
xmin = min(mid.min(), sub_x) * 0.55
xmax = mid.max() * 1.2
ax.set_xscale("log")
ax.set_xlim(xmin, xmax)
ax.set_ylim(4, 38)
ax.axvspan(xmin, 2.27, color=PURPLE, alpha=0.08, zorder=0)
ax.text(np.sqrt(xmin * 2.27), 4.6, "Addis-like\nOC/EC ≤ 2.27", color=PURPLE,
        ha="center", va="bottom", fontsize=9)
ax.text(xmax * 0.97, 10.3, "MAC = 10", color=ACCENT, ha="right", fontsize=9)
ax.text(xmax * 0.97, 6.3, "MAC = 6", color=ACCENT, ha="right", fontsize=9)
ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10, 20]))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("TOR OC/EC ratio (log scale)")
ax.set_ylabel("Implied MAC = Fabs / TOR EC (m²/g)")
ax.set_title(f"IMPROVE bridge: MAC = 10 is centered at Addis-like composition "
             f"(n = {len(bridge):,})")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "improve_implied_mac_curve.png", dpi=180)
plt.close(fig)
show(OUT / "improve_implied_mac_curve.png", "implied-MAC bridge")

# %% [markdown]
# ### 4. The deployed crossplot, from committed phase-2 predictions
#
# The fixed 190-filter cohort is the complete-rows subset of
# `addis_calibration_predictions.csv`. Both MAC panels are refit here and the deck's
# headline equation asserted.

# %%
pred = pd.read_csv(Path("..") / "ftir_hips_chem" / "output" / "tables"
                   / "pls_calibration_phase2" / "addis_calibration_predictions.csv")
fixed = pred.dropna(axis=0, how="any").copy()
assert len(fixed) == 190, len(fixed)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
for ax, mac in zip(axes, (10, 6)):
    x = fixed.Fabs / mac
    y = fixed.EC_deployed_ugm3
    slope, inter = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    if mac == 10:
        assert (round(slope, 2), round(inter, 2)) == (1.90, -4.17), (slope, inter)
    else:
        assert round(slope, 2) == 1.14, slope
    ax.scatter(x, y, s=14, color="#5a7d9a", alpha=0.65, edgecolor="none")
    hi = float(x.max()) * 1.05
    ax.plot([0, hi], [inter, inter + slope * hi], color=ACCENT, lw=2)
    ax.plot([0, hi], [0, hi], color=GREY, ls="--", lw=1)
    ax.set_title(f"Deployed SPARTAN FTIR EC — MAC = {mac}")
    ax.set_xlabel(f"HIPS EC-equivalent, Fabs/{mac} (µg/m³)")
    ax.text(0.03, 0.95, f"y = {slope:.2f}x {inter:+.2f}\nR² = {r2:.3f}; n = {len(fixed)}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="white", ec=GREY, lw=0.5))
axes[0].set_ylabel("Deployed FTIR EC (µg/m³)")
fig.suptitle("The deployed calibration at Addis — intercept identical at both MACs")
fig.tight_layout()
fig.savefig(OUT / "deployed_alldata_crossplot.png", dpi=180)
plt.close(fig)
show(OUT / "deployed_alldata_crossplot.png", "deployed crossplot")

# %% [markdown]
# ### 5. Seasonal corrected spectra — npz cache joined to committed seasons
#
# The corrected ETAD spectra cache carries `media_id`; the phase-2 predictions table
# carries each MediaId's season label (dry_feb convention). Median ± 10–90 % per season.

# %%
z = np.load("output/corrected/etad_corrected_df6.npz", allow_pickle=True)
wn, corr, media = z["wn"], z["corrected"], z["media_id"]
season_of = dict(zip(pred.MediaId, pred.season))
mask = np.array([m in season_of for m in media])
labels = np.array([season_of.get(m, "") for m in media])
print(f"spectra joined to a season: {mask.sum()} of {len(media)}")

fig, ax = plt.subplots(figsize=(10.5, 5.2))
colors = {"Dry (Oct-Feb)": "#D9822B", "Belg (Mar-May)": "#3F7A56",
          "Kiremt (Jun-Sep)": BLUE}
for season, color in colors.items():
    sel = corr[mask & (labels == season)]
    if not len(sel):
        continue
    ax.fill_between(wn, np.percentile(sel, 10, axis=0), np.percentile(sel, 90, axis=0),
                    color=color, alpha=0.15)
    ax.plot(wn, np.median(sel, axis=0), color=color, lw=1.6,
            label=f"{season}  (n = {len(sel)})")
ax.invert_xaxis()
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_ylabel("Baseline-corrected absorbance")
ax.set_title("Addis corrected spectra by Ethiopian season — loading moves, shape does not")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "addis_spectra_by_season_corrected.png", dpi=180)
plt.close(fig)
show(OUT / "addis_spectra_by_season_corrected.png", "seasonal corrected spectra")

# %% [markdown]
# ### 6. The 1600-band peak centers — committed group stats, drawn
#
# Stats-faithful remake: per-group median and IQR from `peak_center_1600_stats.csv`.
# The per-filter histogram version is ftir_12's figure (`build_notebooks.py 12`).

# %%
pk = pd.read_csv("output/tables/ftir12/peak_center_1600_stats.csv")
addis = pk[pk.group.str.contains("Addis", case=False)].iloc[0]
assert 1617 <= addis.center_median <= 1619.5, addis.center_median

pk = pk.sort_values("center_median").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.5, 3.9))
for y, row in pk.iterrows():
    is_addis = "addis" in row.group.lower()
    c = ACCENT if is_addis else GREY
    ax.plot([row.center_p25, row.center_p75], [y, y], color=c, lw=6, alpha=0.55,
            solid_capstyle="round", zorder=2)
    ax.plot(row.center_median, y, "o", color=c, ms=9, zorder=3)
    ax.annotate(f"{row.center_median:.1f}", (row.center_median, y), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=9, color=c)
ax.axvline(1633, color=BLUE, ls=":", lw=1.2, zorder=1)
ax.set_yticks(range(len(pk)),
              [f"{g}  (n = {int(n)})" for g, n in zip(pk.group, pk.n_used)])
ax.text(1633.5, -0.55, "every IMPROVE cohort ≥ 1633", color=BLUE, fontsize=9,
        va="center", ha="left")
ax.set_xlim(pk.center_p25.min() - 3, pk.center_p75.max() + 3)
ax.set_ylim(-0.8, len(pk) - 0.4)
ax.set_xlabel("1600-band peak center (cm⁻¹) — median dot, IQR bar")
ax.set_title("Addis peaks at 1617–1619 cm⁻¹; every IMPROVE cohort sits higher")
fig.tight_layout()
fig.savefig(OUT / "peak_center_1600_stats_panel.png", dpi=180)
plt.close(fig)
show(OUT / "peak_center_1600_stats_panel.png", "1600-band peak centers (stats panel)")

# %% [markdown]
# ### 7. Adama & Bishoftu context — SPARTAN HIPS export + committed Adama TOR

# %%
import data_paths
hips = pd.read_csv(Path(data_paths.maia_data_root()) / "Spartan"
                   / "SPARTAN_HIPS_Batch1-51.v2.csv")
site_col = next(c for c in hips.columns if "site" in c.lower())
fabs_col = next(c for c in hips.columns if "fabs" in c.lower())
etad = pd.to_numeric(hips.loc[hips[site_col] == "ETAD", fabs_col], errors="coerce").dropna()
etbi = pd.to_numeric(hips.loc[hips[site_col] == "ETBI", fabs_col], errors="coerce").dropna()
assert round(etad.median(), 1) == 47.1, etad.median()
assert round(etbi.median(), 1) == 26.9, etbi.median()

adama = pd.read_csv("output/tables/ftir16/adama_batch54_ocec.csv")
ratios = adama.OC_EC_TR
assert 4.5 <= ratios.min() and ratios.max() <= 7.3

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
bins = np.arange(0, 95, 5)
axes[0].hist(etad, bins=bins, color="#C97B6D", alpha=0.8,
             label=f"ETAD (n = {len(etad)}, median {etad.median():.1f})")
axes[0].hist(etbi, bins=bins, color=BLUE, alpha=0.7,
             label=f"ETBI (n = {len(etbi)}, median {etbi.median():.1f})")
axes[0].set_xlabel("HIPS Fabs (Mm⁻¹)")
axes[0].set_ylabel("Filters")
axes[0].set_title("Bishoftu (ETBI) absorbs less than Addis,\nfar above IMPROVE levels")
axes[0].legend(fontsize=9)
axes[1].scatter(ratios, np.zeros(len(ratios)), s=90, color=ACCENT, zorder=3)
axes[1].axvspan(2.27, ratios.max() + 0.5, color=GREY, alpha=0.12)
axes[1].axvline(5.54, color=GREY, ls="--", lw=1.2)
axes[1].text(5.6, 0.25, "IMPROVE pool median 5.54", fontsize=9, color="#5E6369")
axes[1].axvline(2.27, color=PURPLE, ls=":", lw=1.2)
axes[1].text(2.32, -0.3, "lowest-OC/EC cut 2.27", fontsize=9, color=PURPLE)
axes[1].set_ylim(-0.6, 0.6)
axes[1].set_yticks([])
axes[1].set_xlabel("TOR OC/EC ratio (TR basis)")
axes[1].set_title(f"Adama Batch-54 (n = {len(adama)}) sits at the IMPROVE median,\n"
                  "not in the low-OC/EC tail")
fig.tight_layout()
fig.savefig(OUT / "adama_etbi_context.png", dpi=180)
plt.close(fig)
show(OUT / "adama_etbi_context.png", "Adama/ETBI context")

# %% [markdown]
# ### 8. Manifest — where every deck image now traces

# %%
manifest = pd.DataFrame([
    ("calibration_setup_matrix", "by_protocol (both)", "this notebook §1.1–1.9 (one cell each)"),
    ("crossplots_all_setups", "by_protocol (both)", "this notebook §1"),
    ("residual_vs_d2", "by_protocol (both)", "this notebook §1"),
    ("mac_slope_pivot", "by_protocol (both)", "this notebook §1"),
    ("component_selection / selection curves", "by_protocol (both)", "this notebook §1 (ftir_23 tables)"),
    ("intercept ladder / bootstrap / cohort sweep", "by_protocol (both)", "this notebook §1"),
    ("filtering_by_ocec", "deck root", "this notebook §2 → build_deck_figures"),
    ("airspec_1/2/3", "deck root", "this notebook §2 (cached explainer baselines)"),
    ("improve_implied_mac_curve", "ftir31", "this notebook §3 (committed bridge table)"),
    ("deployed_alldata_crossplot", "ftir31", "this notebook §4 (committed predictions)"),
    ("addis_spectra_by_season (corrected)", "ftir31", "this notebook §5 (npz + committed seasons)"),
    ("peak_center_1600 (stats panel)", "ftir31", "this notebook §6; histogram = ftir_12"),
    ("adama_etbi_context", "ftir31", "this notebook §7"),
    ("July-17 charcoal panels (Ann appendix)", "external PDF", "not a repo figure — no notebook can remake it"),
], columns=["figure", "written to", "provenance"])
manifest.to_csv(OUT / "figure_manifest.csv", index=False)
manifest

# %% [markdown]
# ## Takeaways
#
# (filled in by the finalize step after execution)
