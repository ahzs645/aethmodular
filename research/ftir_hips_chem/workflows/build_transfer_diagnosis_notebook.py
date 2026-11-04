"""Build transfer_diagnosis_20261102.ipynb from run_transfer_diagnosis_20261102.py outputs.

Run from research/ftir_hips_chem:
    ../../.venv/bin/python workflows/run_transfer_diagnosis_20261102.py --bundle <local copy> --stages fit tables inject
    ../../.venv/bin/python workflows/run_intercept_fixes_20261103.py --bundle <local copy> --stages attrib fit tables aeth hips stress   # section 8
    ../../.venv/bin/python workflows/run_hips_response_20261103.py --bundle <local copy> --adama <batch 54 files>   # section 8d
    ../../.venv/bin/python workflows/build_transfer_diagnosis_notebook.py
    ../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace transfer_diagnosis_20261102.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# Why FTIR EC does not transfer to Addis: sites, filters, spectral regions and the baseline (2 Nov 2026)

Every number here comes from `workflows/run_transfer_diagnosis_20261102.py`, which reads the 22 Oct decision bundle:
12,393 IMPROVE calibration filters, 233 Addis filters and 3,517 SPARTAN network filters, each as raw, spline (AIRSpec)
and VIBES (87 blanks, rank 70) spectra. Everything is **per filter** (µg). HIPS BC = Fabs × volume / 10.

Models are PLS1 on IMPROVE TOR EC. IMPROVE numbers are **out of site** (5-fold GroupKFold by site). Addis and network numbers
come from the full-pool model. Nothing is tuned on Addis or on SPARTAN HIPS.

**Questions**
1. Which SPARTAN sites behave like Addis, and is Addis really unusual?
2. How much of the Addis slope and intercept against HIPS is in the reference (TOR vs HIPS), and how much is the FTIR model?
3. What does baselining remove, and does it matter?
4. Which part of the spectrum carries the Addis shortfall, and which Addis filters are worst?
5. What can be done, within Ethiopia, with the data we have?"""))

C.append(code("""import sys
sys.path.insert(0, './scripts')
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display
from plotting import PlotConfig  # applies the default white style

T = Path('output/tables/transfer_diagnosis_20261102')
FIG = Path('output/plots/transfer_diagnosis_20261102'); FIG.mkdir(parents=True, exist_ok=True)
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
COL = {'published': '#000000', 'raw': '#1f77b4', 'spline': '#9467bd', 'vibes': '#d62728', 'anchored': '#2ca02c',
       'anchored_noCH': '#17becf', 'anchored_balanced': '#bcbd22', 'vibes_deflate': '#ff7f0e', 'vibes_anchorhp': '#8c564b'}
LABEL = {'published': 'SPARTAN published', 'raw': 'raw', 'spline': 'spline (AIRSpec)', 'vibes': 'VIBES 87 r70',
         'anchored': 'PTFE-anchored', 'anchored_noCH': 'anchored, C–H masked', 'anchored_balanced': 'anchored, OC/EC-balanced',
         'vibes_deflate': 'VIBES, smooth-deflated', 'vibes_anchorhp': 'VIBES on anchored (high-pass)'}
def mid(b):
    lo, hi = b.strip('(]').split(','); return (float(lo) + float(hi)) / 2
print(json.loads((T / 'fit_receipt.json').read_text()))"""))

C.append(md("""## 1. Sites: Addis is not alone, and its median ratio is ordinary

SPARTAN's published FTIR EC against HIPS, site by site. The median ratio at Addis is about 0.9, which is unremarkable.
What is unusual is the **gain**: within Addis, FTIR EC rises about twice as fast as HIPS, giving a large negative
intercept. Several other heavily loaded sites show the same pattern (Bujumbura, Dhaka, Santiago, Bandung, Delhi,
Mexico City). Lightly loaded sites show slopes below 1."""))
C.append(code("""s = pd.read_csv(T / 'sites_published.csv')
display(s.round(2))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ax[0].scatter(s.hips_ug_p90, s.ols_slope, s=np.sqrt(s.n) * 6, c='#888', alpha=.7)
for _, r in s.iterrows():
    ax[0].annotate(r.code, (r.hips_ug_p90, r.ols_slope), fontsize=8, xytext=(3, 2), textcoords='offset points',
                   color='#d35400' if r.code.startswith('ET') else '#333')
ax[0].axhline(1, color='k', lw=.6, ls=':')
ax[0].set(xlabel='HIPS BC, 90th percentile per site (µg per filter)', ylabel='within-site OLS slope, published FTIR EC vs HIPS',
          title='Steep within-site slopes come with heavy loading')
ax[1].scatter(s.hips_ug_median, s.pub_over_hips, s=np.sqrt(s.n) * 6, c='#888', alpha=.7)
for _, r in s.iterrows():
    ax[1].annotate(r.code, (r.hips_ug_median, r.pub_over_hips), fontsize=8, xytext=(3, 2), textcoords='offset points',
                   color='#d35400' if r.code.startswith('ET') else '#333')
ax[1].axhline(1, color='k', lw=.6, ls=':')
ax[1].set(xlabel='HIPS BC, median per site (µg per filter)', ylabel='median published FTIR EC / HIPS BC',
          title='…but the median ratio at Addis is ordinary')
fig.tight_layout(); fig.savefig(FIG / 'fig1_sites_published.png', dpi=200)"""))

C.append(md("""## 2. The reference: TOR / HIPS rises at high loading inside IMPROVE

FTIR is calibrated to TOR, so the fair expectation at a given loading is what TOR would read, not HIPS. In IMPROVE, TOR/HIPS is
about 0.95 from 2–20 µg and climbs to about 1.15 at 20–45 µg and about 1.4 above 45 µg. Low-OC/EC filters, which look like
Addis, behave the same way. Most filters above 30 µg have OP = 0 (the laser saturates), so TOR may itself be high on dark
filters. Either way, a model that perfectly reproduced TOR would show Addis's convex FTIR-vs-HIPS shape."""))
C.append(code("""r = pd.read_csv(T / 'improve_tor_over_hips.csv')
piv = r.pivot_table(index='hips_bin', columns='subset', values='median', sort=False)
cnt = r.pivot_table(index='hips_bin', columns='subset', values='n', sort=False)
display(piv.round(2)); display(cnt.astype(int))
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for sub, c in [('all', '#444'), ('OC/EC<2.5', '#d35400'), ('OC/EC>=5', '#2e86c1'), ('OP=0', '#999')]:
    d = r[r.subset == sub]; ax.plot([mid(b) for b in d.hips_bin], d['median'], 'o-', color=c, label=sub)
ax.axhline(1, color='k', lw=.6, ls=':'); ax.set_xscale('log')
ax.set(xlabel='HIPS BC (µg per filter)', ylabel='median TOR EC / HIPS BC', title='IMPROVE: TOR reads above HIPS on dark filters')
ax.axvspan(20, 60, color='#f5cba7', alpha=.35, lw=0, label='Addis p5–p95')
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / 'fig2_improve_tor_hips.png', dpi=200)"""))

C.append(md("""## 3. Same loading, same composition: Addis against IMPROVE's own heavy low-OC/EC filters

Out-of-site predictions for IMPROVE filters with TOR OC/EC < 2.5, binned by HIPS loading, next to Addis from the same model.
If Addis is "just a heavy, low-OC/EC filter", the two curves should overlap.

* **raw** (SPARTAN's deployed input) and **PTFE-anchored**: Addis reaches 75–100 % of the IMPROVE curve and shares its rise with loading.
* **spline** and **VIBES**: Addis sits far below anything IMPROVE shows at that loading. This part really is specific to Addis, and it depends on the baseline."""))
C.append(code("""R = pd.read_csv(T / 'ratios_by_loading.csv')
models = [m for m in ['raw', 'spline', 'vibes', 'anchored', 'vibes_deflate', 'vibes_anchorhp'] if m in set(R.model)]
fig, axs = plt.subplots(1, len(models), figsize=(4 * len(models), 4.2), sharey=True)
for ax, m in zip(axs, models):
    for what, ls, c in [('IMPROVE low OC/EC FTIR/HIPS', '-', '#d35400'), ('IMPROVE FTIR/HIPS', '-', '#aaa'), ('Addis FTIR/HIPS', '--', COL[m])]:
        d = R[(R.model == m) & (R.k == 18) & (R.what == what) & (R.n >= 5)]
        ax.plot([mid(b) for b in d.hips_bin], d['median'], 'o' + ls, color=c, label=what.replace(' FTIR/HIPS', ''))
    d = R[(R.model == 'published') & (R.what == 'Addis FTIR/HIPS') & (R.n >= 5)]
    ax.plot([mid(b) for b in d.hips_bin], d['median'], 'k:', label='Addis published')
    ax.axhline(1, color='k', lw=.5, ls=':'); ax.set_xscale('log'); ax.set_title(f'{LABEL[m]}, k = 18')
    ax.set_xlabel('HIPS BC (µg per filter)')
axs[0].set_ylabel('median FTIR EC / HIPS BC'); axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'fig3_matched_loading.png', dpi=200)
tab = R[(R.what.isin(['IMPROVE out-of-site MAE (µg)', 'Addis OLS slope vs HIPS', 'Addis OLS intercept vs HIPS (µg)']))]
display(tab.pivot_table(index=['model', 'k'], columns='what', values='median').round(2))"""))

C.append(md("""## 4. Baselining removes the most direct EC signal: the broad IR continuum

Black carbon absorbs across the mid-IR, so loaded filters carry a smooth continuum that grows with loading (raw panel). The
spline removes all of it by construction. The current VIBES background, whose blank components span smooth ramps and whose
pinball loss pushes the background to the spectrum's floor, removes almost all of it.

Per µg of HIPS, Addis has about the same continuum as IMPROVE at 1880–2050 cm⁻¹, where absorption dominates, but less at
3900 cm⁻¹, where scattering adds. That fits black-carbon absorption."""))
C.append(code("""cb = pd.read_csv(T / 'continuum_by_loading.csv')
preps = [p for p in ['raw', 'vibes', 'spline', 'anchored', 'vibes_deflate', 'vibes_anchorhp'] if p in set(cb.prep)]
fig, axs = plt.subplots(1, len(preps), figsize=(4 * len(preps), 4), sharey=False)
for ax, prep in zip(axs, preps):
    for (win, grp), d in cb[(cb.prep == prep) & (cb.n >= 5)].groupby(['window', 'group']):
        if grp == 'IMPROVE lot 248': continue
        ax.plot([mid(b) for b in d.hips_bin], d['median'] * 1e3, 'o-' if grp == 'Addis' else 's--',
                color={'3900': '#1f77b4', '2600': '#ff7f0e', '2050': '#2ca02c', '1880': '#d62728'}[str(win)],
                label=f'{win} {grp}', alpha=1 if grp == 'Addis' else .6)
    ax.set_xscale('log'); ax.set_title(prep); ax.set_xlabel('HIPS BC (µg per filter)')
axs[0].set_ylabel('median absorbance ×1000 in band-free window'); axs[0].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(FIG / 'fig4_continuum.png', dpi=200)"""))

C.append(md("""### 4b. Two direct checks of VIBES and of the alternative

**Injection test.** A smooth continuum (0.05 absorbance at 2000 cm⁻¹) is added to 16 light IMPROVE filters, and real VIBES
is rerun with the saved 87-blank rank-70 background. The production background returns **under 1 %** of a BC-like ν¹
continuum. Taking the smooth directions out of the blank components returns all of it, but leaves about 16× more residual
on light filters, because lot-to-lot PTFE differences live in the same smooth shapes.

**Noise floor.** On light filters, blank-to-blank continuum scatter is worth **3–6 µg EC** with lot-mean subtraction. The IMPROVE median filter
holds 3.7 µg, so an IMPROVE-trained model rightly treats the continuum as noise and leans on organic bands. Addis filters
hold about 34 µg, so there the continuum is the strongest EC signal. The **PTFE-anchored** background fits the blank-component
scores only on the high-passed PTFE bands. It cuts the scatter to 1.1–1.9 µg, removes the lot offset and keeps the continuum
(`scripts/continuum_anchor.py`)."""))
C.append(code("""inj = pd.read_csv(T / 'vibes_injection.csv')
display(inj.groupby(['variant', 'shape']).value.median().unstack().round(4))
noise = pd.read_csv(T / 'continuum_noise.csv')
display(noise.pivot_table(index=['window', 'lot'], columns='prep', values='noise_ug').round(1).rename(columns=lambda c: c + ' (µg EC equiv.)'))"""))

C.append(md("""## 5. Where in the spectrum the Addis shortfall sits

Each model's EC prediction is split into wavenumber windows (coefficient × centred absorbance, per µg of HIPS). The chart
shows Addis minus matched IMPROVE filters (low OC/EC, HIPS 20–60 µg, mostly BIRM1, PHOE1, ATLA1, PITT1, PUSO1), at k = 12.
Negative means Addis gets less EC from that window.

In every baseline the largest deficit is the **aliphatic C–H stretch (3000–2800 cm⁻¹)**: the models use organic C–H as a
stand-in for EC, and Addis has less C–H per unit of BC than even IMPROVE's urban low-OC/EC filters. Only raw and anchored
recover part of it, from the continuum windows (2800–2400 and 4000–3700 cm⁻¹)."""))
C.append(code("""wa = pd.read_csv(T / 'window_attribution.csv')
piv = wa.pivot_table(index=['hi', 'lo'], columns='model', values='difference', sort=False).sort_index(ascending=False)
display(piv.round(3)); print('sum:', piv.sum().round(2).to_dict())
fig, ax = plt.subplots(figsize=(14, 4))
labels = [f'{h}–{l}' for h, l in piv.index]; x = np.arange(len(labels)); ms = list(piv.columns); wdt = .8 / len(ms)
for j, m in enumerate(ms):
    ax.bar(x + (j - (len(ms) - 1) / 2) * wdt, piv[m].fillna(0), wdt, color=COL[m], label=LABEL[m])
ax.axhline(0, color='k', lw=.6); ax.set_xticks(x, labels, rotation=60, fontsize=8)
ax.set(ylabel='Addis − matched IMPROVE\\ncontribution to FTIR/HIPS', xlabel='window (cm⁻¹)')
ax.legend(frameon=False, ncol=4, fontsize=8); fig.tight_layout(); fig.savefig(FIG / 'fig5_window_attribution.png', dpi=200)"""))

C.append(md("""## 6. Every site against the TOR yardstick

The yardstick for each filter is the median TOR/HIPS of IMPROVE filters at the same loading and OC/BC, from a small
gradient-boosted fit. The table shows each site's median FTIR / (HIPS × yardstick): 1.0 means the site reads the way TOR would
read it. At Addis, Bishoftu and Dhaka the OC/BC is below IMPROVE's range for a third to half of the filters, so their yardstick
is an extrapolation. Treat their numbers as ±20 %.

* **Under-read by every model:** Bishoftu (ETBI), Jaipur (INJA), Bujumbura (BIBU), Tarapacá (CLTA), Dhaka (BDDU), Santiago (CLST).
* **Spline collapses (about 0.4)** at exactly ETAD, ETBI and BDDU, the low-OC/BC sites.
* **Anchored versus VIBES** gains most at ETAD (0.67→0.87), CLST (0.70→0.94), ZAPR (0.84→0.99) and ETBI/INJA (+0.04 to +0.10)."""))
C.append(code("""sv = pd.read_csv(T / 'sites_vs_tor_yardstick.csv').set_index('code')
cols = [c for c in ['published', 'raw k18', 'spline k18', 'vibes k18', 'anchored k18', 'anchored_noCH k18', 'anchored_balanced k12',
                    'vibes_deflate k18', 'vibes_anchorhp k18'] if c in sv.columns]
sv = sv[sv.n >= 15].sort_values('vibes k18')
display(sv.round(2))
fig, ax = plt.subplots(figsize=(9, 9))
im_ = ax.imshow(np.log2(sv[cols].to_numpy()), cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(cols)), cols, rotation=40, ha='right', fontsize=8); ax.set_yticks(range(len(sv)), sv.index, fontsize=8)
for i in range(len(sv)):
    for j, c in enumerate(cols):
        ax.text(j, i, f'{sv[c].iloc[i]:.2f}', ha='center', va='center', fontsize=7)
cb_ = fig.colorbar(im_, ax=ax, shrink=.6); cb_.set_label('log2(FTIR / TOR-equivalent HIPS)')
ax.set_title('FTIR / (HIPS × IMPROVE TOR/HIPS at same loading & OC/BC)')
fig.tight_layout(); fig.savefig(FIG / 'fig6_sites_yardstick.png', dpi=200)
nf = pd.read_csv(T / 'network_filters.csv')
a = nf[nf.code == 'ETAD']
print('Addis by season (ratio to yardstick):')
display(a.groupby('season')[cols].apply(lambda d: (d.div(a.loc[d.index, 'exp_tor_hips'], axis=0)).median()).round(2))"""))

C.append(md("""## 7. Within Addis: which filters, and what they share

Residual of the anchored k18 model after removing the within-Addis loading trend. Under-read filters have **less IR-visible
material per unit of HIPS absorption**: low OC/BC, low PM/BC and low K/BC. Dust and sulfate do not matter. The residual
correlates with absorbance per µg HIPS at every wavenumber (ρ ≈ 0.8 above 1400 cm⁻¹). It is seasonal: Oct–Jan lowest,
Jun–Sep highest. The two worst filters, ETAD-0037 and ETAD-0139, are already listed in `docs/open-items.md` as bad scans, so
the same residual works as a QC flag."""))
C.append(code("""ar = pd.read_csv(T / 'addis_residuals.csv')
bc = ar.fabs_Mm1 / 10
chem = {c: ar[c] / bc for c in ['oc_ftir_ugm3', 'pm25_ugm3', 'k_xrf_ugm3', 'si_ugm3', 'al_ugm3', 'fe_ugm3', 'so4_ugm3']}
print({k.replace('_ugm3', '') + '/BC': round(ar.resid.corr(v, method='spearman'), 2) for k, v in chem.items()})
rs = pd.read_csv(T / 'addis_residual_spectrum.csv')
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(rs.wn, rs.rho, color='#2ca02c'); ax[0].invert_xaxis(); ax[0].axhline(0, color='k', lw=.5)
ax[0].set(xlabel='wavenumber (cm⁻¹)', ylabel='ρ(residual, absorbance per µg HIPS)', title='Addis residual vs spectrum (anchored)')
mo = ar.groupby('month').resid.agg(['median', 'size'])
ax[1].bar(mo.index, mo['median'], color=['#2e86c1' if m in (6, 7, 8, 9) else '#d35400' if m in (10, 11, 12, 1, 2) else '#999' for m in mo.index])
ax[1].axhline(0, color='k', lw=.5); ax[1].set(xlabel='month', ylabel='median log residual', title='Addis residual by month')
fig.tight_layout(); fig.savefig(FIG / 'fig7_addis_residuals.png', dpi=200)
display(ar.sort_values('resid')[['filter_id', 'date', 'season', 'hips_ug', 'resid']].head(10).round(2))"""))

C.append(md("""## What this means

1. **Most of the Addis slope and intercept against HIPS is not an FTIR failure.** IMPROVE's own heavy, low-OC/EC filters show
   the same convex FTIR-vs-HIPS shape, because TOR rises faster than HIPS above about 20 µg per filter. Judged against "slope 1,
   intercept 0 against HIPS", no TOR-trained model can pass at Addis. The fair test is FTIR / (HIPS × IMPROVE TOR/HIPS at the
   same loading and composition), section 6.
2. **What is left is a level shortfall that depends on the baseline.** Against that yardstick SPARTAN's published (raw)
   product is roughly right on average at Addis (Dry about 0.8, Kiremt about 1.2). VIBES reads about 0.7 and spline about 0.4.
3. **The mechanism.** IMPROVE filters are too light for the broad black-carbon continuum to rise above blank-to-blank PTFE
   noise, so IMPROVE-trained models infer EC from organic C–H and O–H bands. Addis has 9× the loading and the lowest OC/EC in
   either network, so the proxy under-reads there, and most of all in the Dry season. Spline and VIBES remove the continuum
   that could have corrected it.
4. **Not only Ethiopia.** Bishoftu, Dhaka, Jaipur, Bujumbura, Santiago and Tarapacá show the same under-read. Korea, Beijing,
   Bandung and Delhi read high.

## What we can do, within Ethiopia, with the data we have

* **Replace the VIBES background with a continuum-preserving one.** The PTFE-anchored background here fits IMPROVE best out of
  site (MAE 1.12 µg, against 1.16 raw and 1.15 VIBES). It moves Addis from 0.67 to 0.87 of the TOR yardstick, and Santiago
  and Pretoria to about 1. Next, build the same idea into VIBES itself (below).
* **Reduce reliance on the organic proxy.** Masking C–H costs nothing in IMPROVE and helps a little. OC/EC-balanced weights
  close most of the Addis level gap but worsen overall IMPROVE accuracy (MAE 1.8 µg), so use them as a sensitivity bound, not
  the product.
* **Report Addis against the TOR yardstick, by season,** and stop quoting slope and intercept against HIPS as the failure
  metric. Keep HIPS as a cross-check.
* **Use the Addis city TOR filters as a one- or two-parameter check, not a calibration set.** Ten filters cannot support a
  local PLS. They can test whether the continuum term (or the anchored model's level) is right at Addis loading. Before
  reading them, write down what counts as a pass (`docs/research-questions-2026-10.md`).
* **Use Bishoftu (ETBI) as the in-country holdout.** It is the lowest-reading site in the network and is untouched by any fit.
* **QC flag:** the within-Addis residual against absorbance per µg HIPS (section 7) picks out the known bad scans first.

## Colab rerun: real VIBES with continuum-protected backgrounds (3 Nov 2026)

`workflows/vibes_continuum_20261102.py` reran real VIBES (tau 0.1, PB loss, maxiter 10000) on all 16,143 spectra with two
backgrounds. That is 32,286 fits on one Colab G4 VM (48 vCPUs, about 960 fits/min, about 34 min). None failed. VIBES is a
CPU convex solve, so the VM's GPU sat idle; the VM was chosen for its cores.

* **`vibes_anchorhp`**: VIBES on the PTFE-anchored spectrum, with the blank components high-passed so it removes only the
  remaining sharp blank structure. IMPROVE out-of-site MAE is 1.10 µg (production VIBES 1.15). Addis reads 0.86 of the TOR
  yardstick (production VIBES 0.67). It matches the linear anchored background almost exactly, so the gain comes from
  anchoring and VIBES' positivity adds little on top.
* **`vibes_deflate`**: VIBES with the smooth directions (1, ν, ν²) projected out of the blank components. This is worse
  (IMPROVE MAE 1.34 µg, Addis 0.54): with nothing to model the smooth PTFE background, lot and thickness differences flood
  into the spectrum. Protecting the continuum only works if the PTFE background is still predicted from the PTFE bands.

**Recommendation:** use the anchored background (or `vibes_anchorhp`) as the continuum-preserving baseline from now on, in
place of production VIBES. Sections 3–6 above include both variants."""))

C.append(md("""## 8. The intercept that is left (3 Nov 2026)

`workflows/run_intercept_fixes_20261103.py` (stages attrib, fit, tables, aeth → `output/tables/intercept_fixes_20261103/`).
On `vibes_anchorhp`, Addis still reads 1.86x HIPS − 4.80 µg/m³. Three questions:

1. Where does it come from? Addis HIPS spans only 3.4–6.8 µg/m³, so the intercept mirrors the slope:
   intercept ≈ −(slope − ratio) × mean HIPS. The question becomes which windows give the excess day-to-day gain.
2. Can a label-free change to the calibration remove it? Six candidates, all trained on IMPROVE TOR EC only. Addis HIPS is
   never used to fit or choose. Two of them see the unlabeled Addis spectra.
3. Is it the FTIR at all? The Addis MA350 aethalometer (IR BCc over each filter's 24 h) is a third instrument on its own
   inlet, so it can say whether FTIR or HIPS follows Addis's day-to-day BC."""))
C.append(code("""I = Path('output/tables/intercept_fixes_20261103')
g = pd.read_csv(I / 'window_gain.csv')
piv = g[g.k == 18].pivot_table(index='window', columns='model', values='slope_hips_bc', sort=False)
display(piv.round(2)); print('total OLS slope vs HIPS:', piv.sum().round(2).to_dict())
sw = pd.read_csv(I / 'summary.csv')
cols = ['model', 'k', 'improve_mae', 'improve_heavy_lowocec_ratio', 'slope_hips_bc', 'intercept_hips_bc', 'ratio_tor_equiv',
        'intercept_aeth_rel', 'loglog_aeth']
display(sw[sw.k.isin([10, 18]) & (sw.model != 'base')][cols].round(3))"""))
C.append(code("""MS = {'vibes': ('#d62728', 'VIBES (before)'), 'vibes_anchorhp_full': ('#8c564b', 'VIBES anchored (after)'), 'epo': ('#e377c2', 'EPO'),
      'dipls_g10': ('#1f9e89', 'di-PLS, Addis target'), 'mask_org': ('#17becf', 'organic bands masked'), 'cont_only': ('#9467bd', 'continuum only')}
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for m, (c, lab) in MS.items():
    d = sw[sw.model == m]
    for ax, col in zip(axs, ['intercept_hips_bc', 'intercept_aeth_rel', 'improve_mae']):
        ax.plot(d.k, d[col], color=c, label=lab, lw=2 if m in ('vibes', 'vibes_anchorhp_full') else 1.2)
for ax, t in zip(axs, ['Deming intercept vs HIPS (µg/m³)', 'intercept vs aethalometer (share of mean)', 'IMPROVE out-of-site MAE (µg)']):
    ax.set(xlabel='PLS k', title=t); ax.axvline(18, color='k', lw=.5, ls=':')
axs[0].axhline(0, color='k', lw=.6); axs[1].axhline(0, color='k', lw=.6); axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'fig8_intercept_fixes.png', dpi=200)"""))
C.append(code("""# aeth_ir: MA350 IR BCc per filter (gallery/data/export_aethalometer.py), joined by the workflow
a = pd.read_csv(I / 'addis_predictions.csv')   # µg/m³; 'base' = VIBES anchored on the channel-pair grid
a = a[~a.excluded & (a.aeth_ir > 0) & a.hips_bc.notna()]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
for ax, (col, lab) in zip(axs, [('hips_bc', 'HIPS BC (MAC 10)'), ('vibes k18', 'FTIR EC, VIBES (before)'), ('base k18', 'FTIR EC, VIBES anchored (after)')]):
    ax.scatter(a.aeth_ir, a[col], s=10, alpha=.7)
    s_, i_ = np.polyfit(a.aeth_ir, a[col], 1); xx = np.array([0, 20]); ax.plot(xx, s_ * xx + i_, 'm--', lw=1)
    ax.plot(xx, xx, color='grey', lw=.6, ls=':'); ax.set(title=f'{lab}: {s_:.2f}x {i_:+.2f}', xlabel='aethalometer IR BCc (µg/m³)', xlim=(0, 20), ylim=(0, 20))
axs[0].set_ylabel('µg/m³'); fig.tight_layout(); fig.savefig(FIG / 'fig9_aethalometer.png', dpi=200)
display(pd.read_csv(I / 'aeth_sites.csv')[['code', 'n', 'hips_ug', 'y', 'intercept', 'intercept_rel', 'loglog', 'r2', 'rho_ratio_vs_aeth']].round(3))"""))
C.append(md("""**What this says**

* **Gain, not offset.** Production VIBES takes most of its extra Addis gain from the PTFE bend and the edge of the C–F
  doublet (600–500 and 1300–1250 cm⁻¹). On the anchored background, neighbouring windows trade large shares (O–H/N–H,
  C–H and 1600–1500 against 4000–3700 and 3400–3100), because the continuum is read as differences between windows.
* **No label-free retraining flattens the HIPS slope.** Low k and continuum-only make the intercept smaller only by giving up
  level. Every candidate that gets the level right (EPO, di-PLS, importance weights) reads 1.9–2.2x HIPS with an intercept of
  −5 to −6 µg/m³. EPO moves the level most (Addis 1.05 of the TOR yardstick, IMPROVE heavy low-OC/EC 0.93) but costs IMPROVE
  accuracy (MAE 1.24 µg against 1.10). Importance weights overshoot on about 830 effective filters.
* **Against the aethalometer the anchored FTIR has no intercept.** It is proportional (Deming intercept +0.04 µg/m³, 1 % of
  the mean; log-log slope 1.00), the same as SPARTAN's published FTIR. Production VIBES is not (−1.0 µg/m³, −28 %, log-log
  1.28): that intercept was real and the anchored background removed it.
* **HIPS is the compressed one at Addis.** HIPS = 0.27 × aethalometer + 2.8 µg/m³ (log-log 0.44), and its ratio to the
  aethalometer falls with the aethalometer (ρ −0.92). FTIR's ratio doesn't (ρ 0.00). A noisy aethalometer would pull both
  ratios down together, so the compression is in HIPS. It is mild at Delhi, Beijing and Pasadena (0.73–0.77). Its cause
  is taken up in 8b and 8c.

**So:** the −5 µg/m³ against HIPS is mostly the reference, and it shouldn't be tuned away in the FTIR. Next steps:

* judge Addis on the aethalometer's intercept and linearity, and on the TOR yardstick's level;
* test HIPS linearity at heavy loading directly (IMPROVE TOR vs HIPS above 30 µg with OP > 0, or HIPS on dilution or
  half-loaded duplicates);
* then decide whether EPO's level gain is worth its IMPROVE cost.

Caveat: the aethalometer days are legacy daily aggregates matched by overlap (unverified 9 AM bins, checked against HIPS on
27 Oct), so they are noisier than the filters, but timing noise cannot make one instrument's ratio fall and not the other's."""))

C.append(md("""### 8b. First test: is it filter loading? (stage `hips`)

Sampled volume is known exactly, so at a fixed BC concentration a filter that drew more air carries more BC. Regress
log(HIPS/FTIR) on log concentration and log volume, within sites. The volume coefficient is then a pure loading effect, free of
the errors-in-variables bias a concentration regressor carries. Sites are grouped by their median HIPS, never by the filter's
own HIPS. A saturation curve HIPS = true / (1 + true/Ls) is fitted where the volume effect vanishes, on heavy sites with
Ethiopia left out. Addis is then compared with HIPS linearised by it."""))
C.append(code("""display(pd.read_csv(I / 'hips_volume_test.csv').round(3))
display(pd.read_csv(I / 'hips_saturation.csv').round(1))
hs = pd.read_csv(I / 'hips_sites_loglog.csv').sort_values('median_hips_ug')
hg = pd.read_csv(I / 'addis_vs_linearised_hips.csv'); hg = hg[np.isfinite(hg.Ls) & (hg.Ls >= 45)]
Ls = json.loads((I / 'hips_receipt.json').read_text())['Ls_used']
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
ok = hs.r_ftir_ug >= .6
axs[0].scatter(hs.median_hips_ug[ok], hs.rma_vs_ftir_ug[ok], color='#8c564b', s=18, label='HIPS vs anchored FTIR (per site)')
axs[0].scatter(hs.median_hips_ug, hs.loglog_vs_aeth, color='#e0a800', s=40, label='HIPS vs aethalometer')
xx = np.linspace(.8, 40, 100); axs[0].plot(xx, 1 - xx / Ls, color='grey', label=f'predicted, Ls {Ls:.0f} µg')
for c in ('ETAD', 'INDH', 'BDDU', 'CHTS'):
    r = hs[hs.code == c].iloc[0]; axs[0].annotate(c, (r.median_hips_ug, r.rma_vs_ftir_ug), fontsize=8)
axs[0].set(xscale='log', xlabel='site median HIPS (µg per filter)', ylabel='log-log slope of HIPS', ylim=(.2, 1.6)); axs[0].axhline(1, color='k', lw=.5)
axs[0].legend(frameon=False, fontsize=8)
for m, c in [('SPARTAN published', '#444'), ('VIBES k18 (before)', '#d62728'), ('VIBES anchored k18 (after)', '#8c564b'), ('EPO k18', '#e377c2')]:
    d = hg[hg.model == m]; axs[1].plot(d.Ls, d.intercept, color=c, label=m)
axs[1].axhline(0, color='k', lw=.5); axs[1].axvline(Ls, color='grey', ls=':')
axs[1].set(xscale='log', xlabel='saturation constant Ls (µg per filter)', ylabel='Addis Deming intercept vs linearised HIPS (µg/m³)')
axs[1].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / 'fig10_hips_saturation.png', dpi=200)
print(hg[hg.Ls.round(0) == round(Ls)][['model', 'slope', 'intercept', 'hips_c_loglog_aeth']].round(3))"""))
C.append(md("""**What this shows, and its limit.** At sites with a median of 14 µg per filter or more, more air at the same
concentration goes with a lower HIPS/FTIR (−0.25 ± 0.04), and a curve fitted without Ethiopia gives Ls ≈ 64 µg per filter
(80 %: about 49–134), which lands on Addis's compression. Section 8c shows that this overstates the case: most of the
volume signal comes from sites where volume isn't independent of concentration. Keep Ls as a what-if, not a measurement."""))

C.append(md("""### 8c. Stress tests, and where we go from here (stage `stress`)

1. Is volume really independent of concentration? Not at Delhi or Bhaktapur. Split the heavy sites, and use PM2.5 as an
   independent concentration control, with site × quarter effects.
2. A better test: HIPS optical depth τ (before any MAC) against aethalometer BC per filter at the four sites with both, using
   raw HIPS from `Filter Data/unified_filter_dataset.pkl`. Fit on the other sites and predict Addis.
3. The Addis level, read from each anchor.
4. Does the season effect survive with the aethalometer taken out?"""))
C.append(code("""display(pd.read_csv(I / 'hips_volume_by_site.csv').round(2))
display(pd.read_csv(I / 'hips_volume_stress.csv').drop(columns='sites').round(3))
hold = pd.read_csv(I / 'hips_tau_holdout.csv'); display(hold.round(3))
tp = pd.read_csv(I / 'hips_tau_points.csv')
fig, ax = plt.subplots(figsize=(6.2, 6.2))
for c, col in {'USPA': '#1f77b4', 'CHTS': '#2ca02c', 'INDH': '#9467bd', 'ETAD': '#d35400'}.items():
    s_ = tp[tp.code == c]; ax.scatter(s_.aeth_ug, s_.tau, s=10, alpha=.7, color=col, label=c)
h3 = hold[hold.trained_on == hold.trained_on.iloc[0]].set_index('model')
xx = np.linspace(0, 140, 200); a_, ti = h3.loc['saturating with loading', ['p1', 'p2']]
ax.plot(xx, a_ * xx / (1 + a_ * xx / ti), 'k', label='saturating, Addis held out')
ax.plot(xx[xx < 62], h3.loc['proportional', 'p1'] * xx[xx < 62], 'k--', lw=1, label='proportional, Addis held out')
ax.set(xlabel='aethalometer BC (µg per filter)', ylabel='HIPS optical depth τ', xlim=(0, 140), ylim=(0, 1.6)); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'fig11_hips_tau_curve.png', dpi=200)
display(pd.read_csv(I / 'aeth_bridge.csv').round(2)); display(pd.read_csv(I / 'addis_true_over_hips.csv').round(2))
display(pd.read_csv(I / 'addis_season_vs_aeth.csv').round(3))
print(json.loads((I / 'stress_receipt.json').read_text()))"""))
C.append(md("""**What this says**

* **The saturation reading doesn't hold as first stated.** Where volume is independent of concentration (Abu Dhabi, Santiago,
  Pretoria, Johannesburg, Mexico City), the strictest control gives −0.08 ± 0.08 against anchored FTIR and −0.01 ± 0.06
  against published, where Ls 64 predicts about −0.25. The strong signal comes from Delhi and Bhaktapur, where low-volume
  filters fall in different sampling periods. Johannesburg and Pretoria are heavy but show no compression at all.
* **HIPS is non-linear against an independent measure, and that does hold.** With the aethalometer on x, τ bends over across
  the four sites. A curve fitted without Addis predicts Addis to −11 %, against +38 % for a straight line, and the quadratic
  now beats the linear fit at Addis (ΔAIC −10.7). That updates `hips_offset_narrative.ipynb`, whose linear result used FTIR EC
  on x. The Addis "optical floor" could be the top of one shared curve.
* **Loading or concentration: unresolved.** Loading wins two training sets and concentration the third, and the fit is
  sensitive to the sites used (τ∞ 1.1–3.7). Delhi, the only site with a different volume, has 24 filters here.
* **Not dust.** Addis soil gives about 0.2 Mm⁻¹ at 633 nm, against the 28 Mm⁻¹ offset.
* **The season effect is in FTIR.** With the aethalometer taken out, anchored FTIR reads Bega −14 % and Kiremt +8 % against
  Belg (± 4 %), and HIPS −2 % and −3 %.
* **The level is not settled, and may be worse.** If HIPS is low at Addis (about ×1.4–1.9 on the two usable curves), FTIR's
  shortfall is larger than FTIR/HIPS suggests. FTIR/aethalometer is 0.54 at Addis, 0.76 Pasadena, 0.68 Delhi and 1.20 Beijing, so
  the aethalometer can't set a single level either.

**Where we go from here**

No new data needed:
1. Replace production VIBES with the anchored background at Addis (team decision).
2. Stop using slope and intercept against raw HIPS as the Addis failure metric; use the aethalometer's intercept and the
   TOR yardstick's level. Recheck the k-sweep and the poster's Addis fit.
3. Carry the season result onto the aethalometer.
4. Don't ship a retrained model; keep EPO as a level sensitivity bound.

Needs someone else:
5. Ask UC Davis for HIPS's τ limit and flagging policy (43 % of Addis filters exceed τ 1).
6. Paired Addis filters at half and full volume on the same days would separate loading from concentration.
7. The Addis city TOR filters are the only direct test of the level. Settle the pass rule before reading them (proposal in
   the gallery at `#xc-proposal` and in `docs/open-items.md`)."""))

C.append(md("""### 8d. Flow, the whole-network HIPS curve and the TOR test (`workflows/run_hips_response_20261103.py`)

**Flow.** The MA350 target is Flow2:Flow1 = 1:3 (75:25). Beijing (65:35) and Pasadena (71:29) were moved towards it from about 55:45,
Delhi climbs through its record with no step, and Addis MA350-0238 stayed at 55:45 throughout. BCc, the dual-spot compensated record,
is what carries the data across this. It holds up. Inside Addis, adjusted for loading, it about halves the flow effect seen on the
uncompensated spots (published FTIR −10 % on BC1, −3 % on BCc; anchored −15 %, −9 %; HIPS −11 %, −8 %), and BCc/BC1 itself moves −7 %
(−10 to −5) with no filter involved. Across the fixes HIPS/BCc doesn't move: Pasadena +1 % (−12 to +17), Delhi 0 % (−12 to +14),
Beijing −1 % (−18 to +19). An earlier version of this note gave about ±7 % inside Addis and 15 % at Pasadena. The first was computed
on BC1, not BCc, and the second was a raw median with no loading adjustment. HIPS stays compressed on BCc, BC1 and BC2. The absolute
level stays open for a calibration reason, not a flow one: the MA350 cross-section isn't a filter MAC, and the scale differs by site.

**HIPS across both networks.** Within sites, on filters binned on the geometric mean of the two optical depths, HIPS is flat to
τ ≈ 0.3 and then falls behind TOR and FTIR, in IMPROVE and SPARTAN alike. Fitted to IMPROVE TOR alone: τ0 = 0.27 (80 % 0.25–0.29).

**TOR.** No Addis city filters have arrived. The five Adama pairs fail both gates of the proposed rule, so it returns "not eligible"."""))
C.append(code("""display(pd.read_csv(I / 'aeth_flow_by_site.csv').round(2)); display(pd.read_csv(I / 'aeth_fix_pasadena.csv').round(2))
fc = pd.read_csv(I / 'aeth_flow_coef.csv')
display(fc[fc.controls == 'loading'].pivot(index='code', columns=['y', 'x'], values='pct_over_range').round(1))
print(json.loads((I / 'aeth_flow_tests.json').read_text()))
cu = pd.read_csv(I / 'hips_tau_curves.csv')
fig, ax = plt.subplots(figsize=(8, 4.6))
for c, g in cu[cu.n >= 5].groupby('curve'):
    ax.plot(np.sqrt(g.lo.clip(lower=.05) * g.hi), g.ratio, marker='o', ms=3, label=c, lw=2.4 if c == 'IMPROVE: HIPS / TOR' else 1.2)
ax.set(xscale='log', xlabel='optical depth τ', ylabel='HIPS ÷ reference, relative to τ 0.2–0.3'); ax.axhline(1, color='k', lw=.5); ax.axvline(.3, color='k', ls=':', lw=.8)
ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(FIG / 'fig12_hips_network_curve.png', dpi=200)
display(pd.read_csv(I / 'hips_response_fit.csv').round(3)); display(pd.read_csv(I / 'addis_vs_hips_response.csv').round(3)); display(pd.read_csv(I / 'hips_site_response.csv').head(12).round(2))
display(pd.read_csv(I / 'baseline_scoreboard.csv').round(2)); display(pd.read_csv(I / 'adama_tor_test.csv').round(2)); display(pd.read_csv(I / 'adama_tor_summary.csv').round(2))
display(pd.read_csv(I / 'pass_rule_simulation.csv').round(2))"""))
C.append(md("""**What this adds**

* **Working assumption for HIPS:** proportional below τ 0.3; use the IMPROVE TOR response or flag it above that; treat it as a lower
  bound above τ 1. Every Addis filter is above 0.3 and 42 % are above 1.
* **The correction helps without seeing Addis.** The anchored intercept goes from −4.87 to −0.97 µg/m³ (−1.89 on the low-OC/EC fit),
  and HIPS against the aethalometer from log-log 0.44 to 0.65–0.77. It overshoots at Johannesburg, Pretoria and Mexico City.
* **Anchored is not a clean win.** It removes a real intercept, but on Adama (lot 245) it reads 1.31 of TOR against 0.96 for
  production VIBES, and it's little better on IMPROVE's heavy Addis-like filters (0.77 against 0.73). The Adama excess looks like a lot
  offset: raw reads 2.23 there, and on IMPROVE's matching filters all baselines agree.
* **EPO is better than I first said.** It's closest to TOR on both direct checks (0.91 and 1.02), at an overall error of 1.24 µg
  against 1.10. Both are options in the gallery's FTIR baseline switch.
* **The Addis level is open.** HIPS looks ×1.3–1.7 low, but Adama's Fabs/TOR is about 16 m²/g rather than 10, on five filters from
  another site. Those pull in opposite directions.
* **The pass rule, simulated with 10 filters:** a true level of 1.0 passes 93 %, 0.6 fails 100 %, and 0.8 is inconclusive 89 %. Twenty
  filters take the 1.0 pass rate to 100 % but leave 0.8 inconclusive (92 %), because a level on a band edge stays unresolved by design.
  The rule now fails high as well as low, since anchored read 1.31 on Adama. It's a proposal for the team, not an agreed rule."""))

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "transfer_diagnosis_20261102.ipynb")
print("wrote transfer_diagnosis_20261102.ipynb")
