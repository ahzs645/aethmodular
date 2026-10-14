"""Build ksweep_20261014.ipynb from the outputs of run_ksweep_20261014.py.

Run from research/ftir_hips_chem, then execute:
    uv run --no-sync python workflows/build_ksweep_notebook.py
    uv run --no-sync jupyter nbconvert --to notebook --execute --inplace ksweep_20261014.ipynb
"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []
cells.append(md("""# Does the Addis intercept follow PLS k, and can IMPROVE see it? (14 Oct 2026)

Test 1 from the 14 Oct intercept plan. The 8 Oct scratch tests suggested the Addis intercept keeps falling as the
number of PLS components **k** grows, while held-out IMPROVE R² stays flat from k ≈ 8, so IMPROVE validation cannot
see it. They also suggested the Addis residual tracks K⁺ and organic-rich samples. This notebook reads the repo workflow
that reruns that properly, and adds the deciding check: **do smoke-like IMPROVE filters drift with k the way Addis does?**

* **Workflow:** `workflows/run_ksweep_20261014.py` → `output/tables/ksweep_20261014/` (about 4 min, local).
* **Baselines:** spline (AIRSpec); VIBES 87-blank bank (4000–500, and cut to 4000–1425); the two minus-LABE1 banks
  (287 lot-251 blanks; 365 labeled blanks).
* **Cohorts:** the standard all-IMPROVE lot-251 model, and the 500 seasonal analogs (`oct1` region) for All Addis, Dry, Belg, Kiremt.
* **Splits:** 100 site-disjoint 80/20 splits per cohort; k = 1–25 from one k = 25 fit per split (the first k NIPALS
  components *are* the k-component model; checked against direct refits to 1e-15).
* **Smoke-like IMPROVE subsets** (fixed from IMPROVE data alone, before any Addis readout): non-soil K = K − 0.6 Fe ≥ p90,
  TOR OC/EC ≥ p90, the phase-2 smoke cohort; contrast: non-soil K ≤ median.
* **Addis readout:** Deming vs HIPS Fabs / MAC 10 with σx from HIPS_Uncertainty and the σy proxy 0.531, the same as every
  Addis panel so far. **IMPROVE readout:** OLS vs TOR EC and vs IMPROVE HIPS Fabs / 10 on held-out sites. The Addis λ is not
  reused for IMPROVE, because its σx (0.31 µg/m³) is the size of the IMPROVE smoke-subset mean.

Nothing is tuned on Addis."""))
cells.append(code("""import sys
sys.path.insert(0, './scripts')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from plotting import PlotConfig  # applies the default white style

T = Path('output/tables/ksweep_20261014')
FIG = Path('output/plots/ksweep_20261014')
FIG.mkdir(parents=True, exist_ok=True)

curves = pd.read_csv(T / 'curves.csv')
pooled = pd.read_csv(T / 'pooled.csv')
gap = pd.read_csv(T / 'reference_gap.csv').set_index('set')
proto = pd.read_csv(T / 'protocol_k.csv')
summary = json.loads((T / 'summary.json').read_text())
parts = pd.concat([pd.read_csv(p) for p in sorted((T / 'parts').glob('*.csv')) if not p.stem.endswith('_pooled')],
                  ignore_index=True)

BASE = {'AIRSpec': 'Spline', 'VIBES-full-r70': 'VIBES 87 (4000–500)', 'VIBES-full-cut-r70': 'VIBES 87 cut 1425',
        'VIBES-lot251-x1': 'VIBES 287 lot-251 −LABE1', 'VIBES-allblanks-x1': 'VIBES 365 −LABE1'}
COLOR = {'AIRSpec': '#1f77b4', 'VIBES-full-r70': '#d62728', 'VIBES-full-cut-r70': '#ff7f0e',
         'VIBES-lot251-x1': '#9467bd', 'VIBES-allblanks-x1': '#8c564b'}
GROUPS = ['All Addis', 'Dry (Oct-Feb)', 'Belg (Mar-May)', 'Kiremt (Jun-Sep)']
SUBSETS = summary['subsets']
KS = [2, 4, 8, 11, 15, 20, 25]
print('subset sizes:', summary['subset_sizes'], '| nsK coverage', round(summary['nsK_coverage'], 4))"""))
cells.append(md("""## 1. The standard model: the Addis intercept falls with k for every baseline

Median over 100 site-disjoint splits; the band is the IQR. The dotted vertical line is the k the frozen protocol picks
(1-SE rule on site-grouped CV)."""))
cells.append(code("""std = curves[curves.cohort.eq('standard')]
ad = std[std.readout.eq('addis')]
pk_std = proto[proto.cohort.eq('standard')].set_index('baseline').protocol_k
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
for ax, g in zip(axes, GROUPS):
    for b, lab in BASE.items():
        d = ad[ad.baseline.eq(b) & ad.set.eq(g)].sort_values('k')
        ax.plot(d.k, d.intercept, color=COLOR[b], label=lab, lw=1.8)
        ax.fill_between(d.k, d.intercept_q25, d.intercept_q75, color=COLOR[b], alpha=0.12, lw=0)
        ax.axvline(pk_std[b], color=COLOR[b], ls=':', lw=1)
    ax.axhline(0, color='k', lw=0.7)
    ax.set_title(g); ax.set_xlabel('PLS components k')
axes[0].set_ylabel('Addis Deming intercept (µg/m³)\\nFTIR EC vs Fabs / 10')
axes[0].legend(fontsize=8, loc='lower left')
fig.tight_layout(); fig.savefig(FIG / 'addis_intercept_vs_k_standard.png', dpi=150); plt.show()

tbl = ad[ad.k.isin(KS)].pivot_table(index=['set', 'baseline'], columns='k', values='intercept').reindex(GROUPS, level=0)
tbl.rename(index=BASE, level=1).round(2).to_csv(FIG / 'addis_intercept_vs_k_standard.csv')
display(tbl.rename(index=BASE, level=1).round(2))"""))
cells.append(md("""## 2. What IMPROVE validation sees over the same k range

Held-out IMPROVE R² vs TOR, standard model. It reaches its plateau by k ≈ 8–11, and from there on it cannot tell
k = 11 from k = 25, even though the Addis VIBES intercept moves by several µg/m³ over that range."""))
cells.append(code("""im = std[std.readout.eq('improve')]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for b, lab in BASE.items():
    d = im[im.baseline.eq(b) & im.set.eq('all')].sort_values('k')
    axes[0].plot(d.k, d.R2, color=COLOR[b], label=lab)
    a = ad[ad.baseline.eq(b) & ad.set.eq('All Addis')].sort_values('k')
    axes[1].plot(d.R2.to_numpy()[:len(a)], a.intercept, 'o-', ms=3, color=COLOR[b], label=lab)
axes[0].set(xlabel='PLS components k', ylabel='held-out IMPROVE R² (vs TOR)', title='IMPROVE validation')
axes[1].set(xlabel='held-out IMPROVE R² (vs TOR)', ylabel='All Addis intercept (µg/m³)',
            title='Same splits: validation vs Addis intercept')
axes[1].axhline(0, color='k', lw=0.7); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'improve_r2_vs_k.png', dpi=150); plt.show()
display(im[im.set.eq('all') & im.k.isin(KS)].pivot(index='baseline', columns='k', values='R2').rename(index=BASE).round(2))"""))
cells.append(md("""## 3. The deciding check: do smoke-like IMPROVE filters drift with k?

The populations differ about 10× in concentration (Addis Fabs/10 ≈ 4.7 µg/m³; IMPROVE subsets 0.1–0.5), so intercepts are
shown **relative to each population's mean reference**, with OLS on both sides for a like-for-like comparison.
The dashed line in the right panel is the subset's **TOR EC vs HIPS Fabs/10** line (no FTIR involved), i.e. where a model
that reproduced TOR exactly would land."""))
cells.append(code("""raw = parts[parts.cohort.eq('standard')]
a = raw[raw.readout.eq('addis') & raw.set.eq('All Addis')].assign(rel=lambda d: d.ols_intercept / d.x_mean)
addis_rel = a.groupby(['baseline', 'k']).rel.median().unstack()

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
for row, b in enumerate(['AIRSpec', 'VIBES-full-r70']):
    ax = axes[row, 0]
    ax.plot(addis_rel.columns, addis_rel.loc[b], color='k', lw=2.4, label='Addis (vs Fabs/10)')
    ax.set_title(f'{BASE[b]}: Addis'); ax.set_ylabel('OLS intercept / mean reference')
    for col, (panel, ref) in enumerate([('intercept_rel', 'vs TOR EC'), ('hips_intercept_rel', 'vs HIPS Fabs / 10')], start=1):
        ax = axes[row, col]
        for i, (s, lab) in enumerate(SUBSETS.items()):
            d = im[im.baseline.eq(b) & im.set.eq(s)].sort_values('k')
            line, = ax.plot(d.k, d[panel], color=f'C{i}', label=lab)
            if col == 2:
                ax.axhline(gap.loc[s, 'intercept_rel'], color=line.get_color(), ls='--', lw=1)
        ax.plot(addis_rel.columns, addis_rel.loc[b], color='k', lw=2.4, alpha=0.35, label='Addis (for scale)')
        ax.set_title(f'{BASE[b]}: held-out IMPROVE {ref}')
for ax in axes.ravel():
    ax.axhline(0, color='k', lw=0.7)
for ax in axes[1]:
    ax.set_xlabel('PLS components k')
axes[0, 2].legend(fontsize=7.5, loc='lower left')
fig.tight_layout(); fig.savefig(FIG / 'smoke_subsets_vs_addis.png', dpi=150); plt.show()

sub = im[im.k.isin(KS) & im.baseline.isin(['AIRSpec', 'VIBES-full-r70'])]
display(sub.pivot_table(index=['set', 'baseline'], columns='k', values='hips_intercept_rel').rename(index=BASE, level=1).round(2))
display(addis_rel[KS].rename(index=BASE).round(2))
display(gap.round(3))"""))
cells.append(md("""**Pooled out-of-fold version** (each filter's prediction averaged over the splits that held its site out).
It gives one curve per subset, so there is no split noise. Same story."""))
cells.append(code("""pp = pooled[pooled.cohort.eq('standard') & pooled.k.isin(KS)].assign(
    rel_tor=lambda d: d.intercept / d.x_mean, rel_hips=lambda d: d.hips_intercept / d.hips_x_mean)
display(pp[pp.baseline.isin(['AIRSpec', 'VIBES-full-r70'])]
        .pivot_table(index=['set', 'baseline'], columns='k', values=['rel_tor', 'rel_hips'])
        .rename(index=BASE, level=1).round(2))"""))
cells.append(md("""## 4. Seasonal analog cohorts: k is not the main story there

For the 500-analog cohorts, VIBES is already far off at k = 2–4, and the curves are not monotone. The analog failure is
present from the first components, so for those cohorts the problem is what the analogs and the VIBES spectra carry,
not how many components the model uses."""))
cells.append(code("""an = curves[curves.cohort.str.startswith('analog') & curves.readout.eq('addis')]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
for ax, g in zip(axes, GROUPS):
    for b, lab in BASE.items():
        d = an[an.baseline.eq(b) & an.cohort.eq(f'analog|{g}')].sort_values('k')
        ax.plot(d.k, d.intercept, color=COLOR[b], label=lab)
        ax.fill_between(d.k, d.intercept_q25, d.intercept_q75, color=COLOR[b], alpha=0.10, lw=0)
    ax.axhline(0, color='k', lw=0.7); ax.set_title(f'{g} analogs'); ax.set_xlabel('PLS components k')
axes[0].set_ylabel('Addis Deming intercept (µg/m³)'); axes[0].legend(fontsize=8, loc='lower left')
fig.tight_layout(); fig.savefig(FIG / 'addis_intercept_vs_k_analogs.png', dpi=150); plt.show()
display(an[an.k.isin(KS)].pivot_table(index=['cohort', 'baseline'], columns='k', values='intercept').rename(index=BASE, level=1).round(2))
display(proto.pivot(index='baseline', columns='cohort', values='protocol_k').rename(index=BASE))"""))
cells.append(md("""## Takeaways

1. **The k-dependence is real and holds in every season** (standard model, 100 site-disjoint splits). From k = 2 to k = 20
   the All-Addis intercept goes −0.1 → −1.6 µg/m³ for spline and −0.3 → −7.0 for VIBES 87. The minus-LABE1 banks behave
   like VIBES 87, so a clean blank bank does not remove it. Kiremt is worst and Dry mildest. The VIBES 4000–1425 cut is in between.
2. **IMPROVE validation is blind to it.** Held-out R² plateaus by k ≈ 8–11, and the held-out IMPROVE intercept vs TOR stays near
   zero for every subset and every k.
3. **Smoke-like IMPROVE filters do not reproduce the Addis drift.** Against TOR their intercept is flat in k. Against HIPS/10
   they do go negative, to about −0.3 of the mean, but they level off by k ≈ 7 (Addis VIBES keeps falling to k ≈ 18),
   and they behave the same for spline and VIBES. The plateau sits in the range of each subset's own TOR-vs-HIPS line
   (dashed, −0.17 to −0.53), so it reads as the model converging onto TOR, which itself sits below HIPS/10 in smoky
   IMPROVE (MAC 7.8–8.2 vs 9.6 network-wide). That is a reference gap, not a calibration failure.
4. **So the answer to the deciding question is "no", with one caveat.** The spline Addis intercept at high k (about −0.3 of the mean)
   is the same size as that IMPROVE smoke reference gap, which is consistent with "spline predicts TOR-like EC, and TOR reads
   low against HIPS in smoke". The additional VIBES drift (to about −1.0 of the mean) has no counterpart in any IMPROVE subset.
   That part is an Addis-only response shift, and k cannot be chosen on IMPROVE to avoid it.
5. **For the seasonal analogs, k is not the lever.** VIBES analog models are off at k = 2–4 already.
6. **The 7 Oct "Where we stand" conclusion that blames PTFE should be rewritten** around points 1–4.

Caveats: the IMPROVE smoke subsets are about 10× cleaner than Addis, so a loading-dependent effect could still hide here (see test 2,
the HIPS R1 blank-domain check). The subset thresholds were fixed before looking at Addis, but they are only one choice;
nsK per µg EC was not tried."""))
nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'ksweep_20261014.ipynb')
