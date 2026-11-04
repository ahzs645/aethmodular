"""Build season_composition_20261103.ipynb from run_season_composition_20261103.py outputs.

Run from research/ftir_hips_chem:
    ../../.venv/bin/python workflows/run_season_composition_20261103.py --bundle <local copy of the 22 Oct bundle>
    ../../.venv/bin/python workflows/build_season_composition_notebook.py
    ../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace season_composition_20261103.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# Addis FTIR EC by season: season or composition? (3 Nov 2026)

Follows `transfer_diagnosis_20261102.ipynb` and reuses its fits, so there is no new calibration here. Models are PLS1 on
12,393 IMPROVE TOR EC filters at k 18 (k 12 for the OC/EC-balanced model), applied unchanged to Addis.

**Yardstick.** Each Addis filter is compared with what IMPROVE TOR would read at the same HIPS loading and OC/BC
(`exp_tor_hips` from the 2 Nov work), not with HIPS/10 at slope 1. A ratio of 1 means "reads like TOR would".

**Calendar.** `local_2026_10` (belg_feb): Bega Oct–Jan, Belg Feb–May, Kiremt Jun–Sep.

**Questions**
1. Is there a seasonal pattern once loading is accounted for?
2. Does it follow composition, and specifically the organic "stand-in" that IMPROVE models learn?
3. Can the organic measure be trusted, given that FTIR OC comes from the same spectrum as FTIR EC?
4. Does the Addis composition relation carry over to Bishoftu?"""))

C.append(code("""import sys
sys.path.insert(0, './scripts')
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display
from plotting import PlotConfig  # applies the default white style

T = Path('output/tables/season_composition_20261103')
FIG = Path('output/plots/season_composition_20261103'); FIG.mkdir(parents=True, exist_ok=True)
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
SEASONS = ['Bega', 'Belg', 'Kiremt']
SCOL = {'Bega': '#E67E22', 'Belg': '#27AE60', 'Kiremt': '#3498DB'}  # config.ETHIOPIA_SEASONS hues
MAIN = ['published', 'raw k18', 'anchored k18', 'vibes k18', 'spline k18']
print(json.dumps(json.loads((T / 'receipt.json').read_text()), indent=1))"""))

C.append(md("""## 1. The season pattern survives loading

Median FTIR EC relative to the TOR yardstick, by season. `vs_tor_loading_adjusted` also removes the within-Addis
log-log trend with HIPS loading (Kiremt filters are heavier: median 40 µg vs 31 µg in Bega).
The order Bega < Belg < Kiremt holds for every model, and also within each third of HIPS loading (second table)."""))
C.append(code("""m = pd.read_csv(T / 'season_medians.csv')
display(m.pivot_table(index='model', columns='season', values=['vs_tor', 'vs_tor_loading_adjusted'], sort=False).round(2))
t = pd.read_csv(T / 'loading_tertiles.csv')
display(t[t.model.isin(MAIN)].pivot_table(index=['model', 'tertile'], columns='season', values='vs_tor', sort=False).round(2))
display(t[t.model == 'published'].pivot_table(index='tertile', columns='season', values='n', sort=False).rename_axis('n filters (published)'))"""))
C.append(code("""fig, ax = plt.subplots(figsize=(8, 4.5))
mm = m[m.model.isin(MAIN)]
x = np.arange(len(MAIN)); w = 0.26
for i, s in enumerate(SEASONS):
    v = mm[mm.season == s].set_index('model').reindex(MAIN).vs_tor_loading_adjusted
    ax.bar(x + (i - 1) * w, v, w, color=SCOL[s], label=s)
ax.axhline(1, color='k', lw=.6, ls=':')
ax.set_xticks(x, MAIN)
ax.set(ylabel='median FTIR EC / TOR yardstick\\n(after the loading trend)', title='Addis: same seasonal order for every model')
ax.legend(frameon=False, ncol=3)
fig.tight_layout(); fig.savefig(FIG / 'fig1_season_by_model.png', dpi=200)"""))

C.append(md("""## 2. What changes between seasons

Kiremt carries more organic matter per unit BC by every measure: FTIR OC/BC 1.74 vs 1.02 in Bega, and residual mass/BC
2.9 vs 1.5. Residual mass is PM2.5 − 1.375 SO4 − (2.20 Al + 2.49 Si + 2.42 Fe) − HIPS BC; it doesn't use the FTIR spectrum.
It is a rough organic proxy because it also holds nitrate, extra ammonium, Ca/Ti/Na dust and water."""))
C.append(code("""display(pd.read_csv(T / 'composition.csv').round(2))"""))

C.append(md("""## 3. IMPROVE: FTIR over-reads EC as organics rise, measured with independent TOR OC

This is the organic stand-in from the 2 Nov work, measured directly. In IMPROVE (out-of-site predictions, TOR EC 2–150 µg,
HIPS > 5 µg), FTIR EC / TOR EC rises with TOR OC at fixed loading for **every** baseline. The coefficient is
0.29–0.35 per log unit when loading is measured by TOR EC and 0.15–0.17 when it is measured by HIPS. The OC/EC-balanced
model roughly halves it, as expected if it is learned from the training mix."""))
C.append(code("""s = pd.read_csv(T / 'improve_organic_slope.csv')
display(s.pivot_table(index='model', columns='covariates', values='coef_loc', sort=False).round(3))
b = pd.read_csv(T / 'improve_ocec_bins.csv')
fig, ax = plt.subplots(figsize=(7, 4.5))
order = ['(0.0, 1.5]', '(1.5, 2.5]', '(2.5, 4.0]', '(4.0, 6.0]', '(6.0, 10.0]', '(10.0, 100.0]']
for nm, g in b.groupby('model'):
    g = g.set_index('ocec_bin').reindex(order)
    ax.plot(range(len(order)), g.ftir_over_tor, marker='o', label=nm, lw=2 if nm in ('raw', 'anchored_balanced') else 1)
ax.axhline(1, color='k', lw=.6, ls=':')
ax.set_xticks(range(len(order)), ['<1.5', '1.5–2.5', '2.5–4', '4–6', '6–10', '>10'])
ax.set(xlabel='IMPROVE TOR OC/EC', ylabel='median FTIR EC / TOR EC (out of site)', title='IMPROVE: FTIR EC tracks OC/EC')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'fig2_improve_ocec.png', dpi=200)
display(b.pivot_table(index='ocec_bin', columns='model', values='n', sort=False).iloc[:, :1].rename(columns=lambda c: 'n'))"""))

C.append(md("""## 4. Addis: how much of the season effect do the organic measures absorb?

OLS of log(FTIR EC / TOR yardstick) on season dummies plus covariates. The table shows Belg and Kiremt relative to Bega in %.

* **Residual mass**, which doesn't use the spectrum, roughly halves the Kiremt effect (about +40 % → +20 %). Its
  coefficient (0.26–0.32 for published, raw and anchored) falls inside the IMPROVE range from section 3.
* **FTIR OC** absorbs all of it, but needs a coefficient of 0.7–0.8, two to five times the IMPROVE value. The likeliest
  reason is that FTIR OC and FTIR EC are read from the same spectrum and share an error that rises in Kiremt. So FTIR OC
  "explaining" the season is partly circular.
* Adding nsK and the filter lot barely changes the result. A Kiremt effect of about +20 % remains with the independent
  covariates (CI excludes 0 for every model)."""))
C.append(code("""e = pd.read_csv(T / 'season_effects.csv')
cols = ['n', 'Belg_pct', 'Kiremt_pct', 'Kiremt_lo', 'Kiremt_hi', 'Kiremt_p', 'coef_lrm', 'coef_loc', 'r2']
display(e[e.model.isin(MAIN + ['anchored_balanced k12'])].set_index(['model', 'covariates'])[cols].round(2))"""))
C.append(code("""sets = ['+ loading', '+ loading + residual mass', '+ loading + residual mass + nsK + lot', '+ loading + FTIR OC']
show = ['published', 'raw k18', 'anchored k18', 'vibes k18']
fig, ax = plt.subplots(figsize=(9, 4.8))
for j, mdl in enumerate(show):
    g = e[(e.model == mdl)].set_index('covariates').reindex(sets)
    xs = j + np.linspace(-0.3, 0.3, len(sets))
    for xi, (lab, r) in zip(xs, g.iterrows()):
        c = {'+ loading': '#555', '+ loading + residual mass': '#2e86c1',
             '+ loading + residual mass + nsK + lot': '#1b4f72', '+ loading + FTIR OC': '#c0392b'}[lab]
        ax.errorbar(xi, r.Kiremt_pct, yerr=[[r.Kiremt_pct - r.Kiremt_lo], [r.Kiremt_hi - r.Kiremt_pct]], fmt='o', color=c,
                    label=lab if j == 0 else None, capsize=3)
ax.axhline(0, color='k', lw=.6)
ax.set_xticks(range(len(show)), show)
ax.set(ylabel='Kiremt vs Bega, % (95 % CI)', title='Kiremt over-read: independent organics explain about half')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'fig3_kiremt_effect.png', dpi=200)"""))
C.append(code("""from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
a = pd.read_csv(T / 'addis_filters.csv')
a = a[a.rm_over_bc > 0]
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for k, (col, lab) in enumerate([('rm_over_bc', 'residual mass / HIPS BC (independent of FTIR)'),
                                ('oc_over_bc', 'FTIR OC / HIPS BC (same spectrum as FTIR EC)')]):
    for s in SEASONS:
        g = a[a.season == s]
        ax[k].scatter(g[col], g['raw k18'] / g.exp_tor_hips, s=14, alpha=.7, color=SCOL[s], label=s)
    ax[k].set_xscale('log'); ax[k].set_yscale('log')
    for axis, ticks in ((ax[k].xaxis, [0.5, 1, 2, 4, 8]), (ax[k].yaxis, [0.25, 0.5, 0.75, 1, 1.5])):
        axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
    ax[k].axhline(1, color='k', lw=.6, ls=':')
    ax[k].set(xlabel=lab, ylabel='raw k18 FTIR EC / TOR yardstick')
    ax[k].set_box_aspect(1)
ax[0].legend(frameon=False)
fig.suptitle('Addis, per filter')
fig.tight_layout(); fig.savefig(FIG / 'fig4_addis_organics.png', dpi=200)"""))

C.append(md("""## 5. Leave one season out

Fit `log ratio ~ log HIPS + organic proxy` on two seasons and predict the third. With residual mass the held-out level is
reproduced to within about 0.1. Bega comes out lower than predicted and Kiremt higher, which matches the remaining
season effect in section 4."""))
C.append(code("""l = pd.read_csv(T / 'leave_season_out.csv')
display(l[l.model.isin(MAIN)].pivot_table(index=['model', 'proxy'], columns='held_out', values=['observed', 'predicted'], sort=False).round(2))"""))

C.append(md("""## 6. Bishoftu: a test site the models were never fitted to, but Bega only

Bishoftu (ETBI) has 21 usable filters, all from Nov–Dec 2025, so the seasonal question can't be tested there. Applying the
Addis composition relation over-predicts Bishoftu for published, raw and anchored (observed about 0.62–0.71, predicted
0.91–1.08). Bishoftu's Bega under-read is therefore larger than its composition implies, whereas spline and VIBES, already low,
are predicted about right. More Bishoftu months are needed before this means much."""))
C.append(code("""display(pd.read_csv(T / 'bishoftu.csv').round(2))"""))

C.append(md("""## 7. Which spectral windows carry the Kiremt over-read?

The k 18 PLS prediction is linear, so each filter's FTIR EC / TOR yardstick splits exactly into window shares plus an
intercept share. `kiremt_minus_bega` is the difference of season means; `kiremt_adjusted` is the Kiremt coefficient
after log HIPS and log residual mass. Column sums equal the total.

For raw and PTFE-anchored, the Kiremt excess comes from the **organic bands** (aliphatic C–H 3000–2800, 2000–1800 and
C=O/C=C 1800–1500). It is partly cancelled by **2800–2000 cm⁻¹**, where the BC continuum sits and there are almost no organic
bands. That window reads Kiremt *lower* than Bega. Relative to expected EC, then, Kiremt filters carry less continuum and more
organic-band signal. That is the organic stand-in again, and it means a continuum-based EC estimate should show a weaker
season effect. The O–H window (4000–3400) contributes little, so this doesn't support the 3500–4000 cm⁻¹ lead from the
PR #15 review. VIBES and spline remove the continuum, so their small totals are spread differently.

PLS window shares are indicative only: neighbouring channels are correlated and their coefficients offset each other."""))
C.append(code("""w = pd.read_csv(T / 'season_windows.csv')
display(w.pivot_table(index='window', columns='model', values=['kiremt_minus_bega', 'kiremt_adjusted'], sort=False).round(3))
g = w[(w.model.isin(['raw', 'anchored'])) & ~w.window.isin(['TOTAL', 'intercept'])]
fig, ax = plt.subplots(figsize=(9, 4.5))
wins = list(dict.fromkeys(g.window)); x = np.arange(len(wins))
for i, (mdl, c) in enumerate([('raw', '#1f77b4'), ('anchored', '#2ca02c')]):
    v = g[g.model == mdl].set_index('window').reindex(wins).kiremt_adjusted
    ax.bar(x + (i - .5) * .38, v, .38, color=c, label=f'{mdl} k18')
ax.axhline(0, color='k', lw=.6)
ax.set_xticks(x, wins, rotation=30, ha='right')
ax.set(ylabel='Kiremt − Bega share of FTIR EC / TOR yardstick\\n(after loading and residual mass)',
       title='Organic bands push Kiremt up; the continuum window pulls it down')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / 'fig5_season_windows.png', dpi=200)"""))

C.append(md("""## 8. Conclusions

1. **The season effect is real.** Against the TOR yardstick, after loading, Addis reads about 0.87 in Bega, 0.97 in Belg
   and 1.13–1.19 in Kiremt, for every baseline. It holds within loading thirds and within lot 251.
2. **About half is the organic stand-in.** IMPROVE models over-read EC as OC/EC rises (shown with independent TOR OC).
   Kiremt has 1.7–1.9× more organic matter per BC than Bega, and an independent residual-mass proxy with the IMPROVE-sized
   coefficient removes about half of the Kiremt effect.
3. **The rest is most likely organic too, but measured poorly.** Only FTIR OC absorbs it, with an implausibly steep
   coefficient, so part of it may be an error FTIR EC and FTIR OC share. But the window split (section 7) puts the remaining
   Kiremt excess in organic bands (C–H, C=O), not in O–H/water. Residual mass is a noisy organic proxy (it holds nitrate
   and water). Noise in a covariate weakens its coefficient and leaves some of the effect assigned to season, so "half" is a
   lower bound for the organic share.
4. **Changing the baseline will not remove it,** as long as the model reads EC through organic bands. Spline and VIBES show
   the same order. The OC/EC-balanced model halves IMPROVE's organic sensitivity but still has a Kiremt effect of +22–27 %.
   The continuum window (2800–2000 cm⁻¹) reads Kiremt lower, which is the reason to expect a continuum-protected
   background to help.

**Next tests.** (a) Repeat with the continuum-protected VIBES runs when they land (the workflow picks up any new `… k18`
column in `network_filters.csv`). (b) A continuum-only EC estimate from 2800–2000 cm⁻¹ by season. (c) Independent OM from
SPARTAN's full reconstruction (with nitrate, ammonium and Ca) instead of the residual-mass approximation. (d) More Bishoftu
months."""))

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "season_composition_20261103.ipynb")
print("wrote season_composition_20261103.ipynb")
