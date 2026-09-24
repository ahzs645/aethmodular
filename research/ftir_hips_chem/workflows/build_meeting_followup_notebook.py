"""Build the notebook for the 17 Sep 2026 meeting follow-up (updated after the 23 Sep meeting).

Reads the saved tables written by run_meeting_followup_20260917.py and
run_meeting_grid_20260917.py; it does not refit. Conventions from the 23 Sep
meeting with Ann (docs/naming-conventions.md): Deming only (no OLS), "calibration
set" / "cross-validation" / "test set (Addis)", "spline baseline" for AIRSpec.

    uv run --no-sync python research/ftir_hips_chem/workflows/build_meeting_followup_notebook.py
    cd research/ftir_hips_chem && uv run --no-sync jupyter nbconvert --execute --to notebook \\
      --output meeting_followup_20260917_executed.ipynb --output-dir notebooks/archive/executed \\
      meeting_followup_20260917.ipynb
"""

from pathlib import Path
import nbformat as nbf

AREA = Path(__file__).resolve().parents[1]
DEST = AREA / "meeting_followup_20260917.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = r'''import sys
# For notebooks inside research/ftir_hips_chem/:
sys.path.insert(0, './scripts')
sys.path.insert(0, '../ftir_ec_phase3/scripts')

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Markdown

from config import MAC_VALUE, ETHIOPIA_SEASONS
from plotting import PlotConfig

PlotConfig.set(sites='Addis_Ababa', layout='individual', show_stats=True, show_1to1=True)
T = Path('output/tables/meeting_followup_20260917')
FIG = Path('output/plots/meeting_followup_20260917')
FIG.mkdir(parents=True, exist_ok=True)

summary = json.loads((T / 'summary.json').read_text())
stats = pd.read_csv(T / 'panel_stats.csv')
analogs = pd.read_csv(T / 'analog_predictions.csv')
addis = pd.read_csv(T / 'addis_predictions.csv')
rep = pd.read_csv(T / 'repeated_splits.csv')
stitched = pd.read_csv(T / 'stitched_repeats.csv')
op = pd.read_csv(T / 'op_summary.csv')
repro = pd.read_csv(T / 'meeting_reproduction.csv')
grid = pd.read_json(T / 'category_grid.jsonl', lines=True)
METHODS = summary.get('methods', ['AIRSpec', 'VIBES'])
NAME = {'AIRSpec': 'Spline baseline', 'VIBES': 'VIBES baseline (4000–1425)',
        'VIBES-full': 'VIBES baseline (4000–500)', 'VIBES-full-cut': 'VIBES baseline (fit 4000–500, used 4000–1425)'}
SEASONS = list(ETHIOPIA_SEASONS)
GROUPS = ['All Addis'] + SEASONS
COL = {**{s: v['color'] for s, v in ETHIOPIA_SEASONS.items()}, 'All Addis': '#5b6470'}
XLAB = 'HIPS Fabs / MAC 10 (µg/m³)'

def st(**q):
    m = np.ones(len(stats), bool)
    for k, v in q.items():
        m &= stats[k].eq(v).to_numpy()
    rows = stats.loc[m]
    return rows.iloc[0] if len(rows) else None

def panel(ax, x, y, xl, yl, s, color, title=None):
    """1:1 comparison panel: square equal axes, 1:1 line, Deming fit only (no OLS)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    lo = min(0, np.nanmin(np.r_[x[ok], y[ok]])); hi = np.nanmax(np.r_[x[ok], y[ok]]) * 1.05
    ax.scatter(x[ok], y[ok], s=18, color=color, alpha=0.7, edgecolors='white', linewidths=0.4)
    ax.plot([lo, hi], [lo, hi], ls='--', color='#9aa5b1', lw=1)
    if s is not None:
        xx = np.array([lo, hi]); ax.plot(xx, s.deming_slope * xx + s.deming_intercept, color='#c026d3', lw=2)
        ax.text(0.04, 0.96, f"Deming y = {s.deming_slope:.2f}x {s.deming_intercept:+.2f}\nR² {s.R2:.2f} · n {int(s.n)}",
                transform=ax.transAxes, va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    if title: ax.set_title(title, fontsize=10)

def q(method, g, col):
    s = rep.loc[rep.method.eq(method) & rep.group.eq(g), col]
    return s.median(), s.quantile(.1), s.quantile(.9)

def band(t):
    return f"{t[0]:.2f} [{t[1]:.2f}–{t[2]:.2f}]"
print(summary['notes'][0]); print(f"baselines: {', '.join(NAME.get(m, m) for m in METHODS)}; λ* = {summary['lambda']:.2f}; grid rows = {len(grid)}")
'''

TLDR = r'''rows = []
for g in SEASONS:
    rows.append({'Season': g,
                 'Test set (Addis): FTIR vs Fabs/10': band(q('AIRSpec', g, 'addis_slope')),
                 'IMPROVE cross-validation: FTIR vs Fabs/10': band(q('AIRSpec', g, 'ftir_vs_fabs_slope')),
                 'IMPROVE cross-validation: TOR vs Fabs/10': band(q('AIRSpec', g, 'tor_vs_fabs_slope')),
                 'IMPROVE cross-validation: FTIR vs TOR': band(q('AIRSpec', g, 'ftir_vs_tor_slope')),
                 'PLS factors': band(q('AIRSpec', g, 'k'))})
display(pd.DataFrame(rows).set_index('Season'))
A = {g: {c: q('AIRSpec', g, c)[0] for c in ['addis_slope', 'ftir_vs_fabs_slope', 'tor_vs_fabs_slope', 'ftir_vs_tor_slope']} for g in SEASONS}
pairs = '; '.join(f"{g.split()[0]} **{A[g]['ftir_vs_fabs_slope']:.2f}** vs **{A[g]['addis_slope']:.2f}**" for g in SEASONS)
rho = op.loc[op.method.eq('pool'), 'spearman_op_vs_log_mac'].iloc[0]
dep = st(fit_id='deployed|SPARTAN|current SPARTAN calibration', population='addis', eval_group='All Addis')
sti = st(fit_id='stitched|AIRSpec|season calibrations', population='addis', eval_group='All Addis')
one = st(fit_id='allimprove|AIRSpec|all-IMPROVE lot 251', population='addis', eval_group='All Addis')
sr = stitched.loc[stitched.method.eq('AIRSpec')]
dry, belg, kir = SEASONS
display(Markdown(f"""
**Starting point.** With the calibration SPARTAN uses today, the Addis test set gives Deming
**{dep.deming_slope:.2f}x {dep.deming_intercept:+.2f}** against HIPS Fabs/10 (n {int(dep.n)}).

**Per-season calibration, stitched back together, does not fix the intercept.** Calibrating each
season on its own spectral analogs (spline baseline) and predicting each season's Addis filters
with its own calibration gives, for all Addis, **{sti.deming_slope:.2f}x {sti.deming_intercept:+.2f}**
(median over 100 cross-validation splits: slope {sr.addis_slope.median():.2f}, intercept
{sr.addis_intercept.median():+.2f}), against **{one.deming_slope:.2f}x {one.deming_intercept:+.2f}** for one
all-IMPROVE model. Each season's own line is shallow; the season calibrations sit at different
levels, so putting them together steepens the combined line and pushes the intercept further from zero.

**Satoshi's check.** In IMPROVE cross-validation, the season's analogs are mispredicted against
HIPS in the same direction as the Addis test set, a little less strongly: {pairs}. For the Dry
analogs TOR EC vs Fabs/10 is **{A[dry]['tor_vs_fabs_slope']:.2f}** (TOR low for its absorption), so part of
the Dry bias is in the references; in Belg/Kiremt TOR vs Fabs/10 is above 1 and the shortfall is FTIR
under-reading TOR. OP does not explain the TOR/HIPS spread (Spearman ρ = **{rho:.2f}** over the pool).

Deming weights: Addis λ* = {summary['lambda']:.2f}; IMPROVE panels use each calibration's cross-validation
RMSE against a {100 * summary['ref_rel_uncertainty']:.1f} % reference uncertainty (ETAD HIPS). All Addis
numbers are against HIPS Fabs/MAC 10, an optical proxy, not thermal EC.
"""))
'''

STARTING = r'''def season_crossplot(fit_id, title, ax):
    p = addis.loc[addis.fit_id.eq(fit_id)]
    xx = np.array([0, 9])
    for g in SEASONS:
        d = p.loc[p.season.eq(g)]; s = st(fit_id=fit_id, population='addis', eval_group=g)
        ax.scatter(d.Fabs / MAC_VALUE, d.ftir_ec_ugm3, s=20, color=COL[g], alpha=0.75, edgecolors='white', linewidths=0.4,
                   label=f"{g}: {s.deming_slope:.2f}x {s.deming_intercept:+.2f}" if s is not None else g)
        if s is not None: ax.plot(xx, s.deming_slope * xx + s.deming_intercept, color=COL[g])
    s = st(fit_id=fit_id, population='addis', eval_group='All Addis')
    ax.plot(xx, s.deming_slope * xx + s.deming_intercept, color='k', lw=2, label=f"all Addis: {s.deming_slope:.2f}x {s.deming_intercept:+.2f} (R² {s.R2:.2f})")
    lo = min(0, np.nanmin(p.ftir_ec_ugm3)); ax.plot([lo, 9], [lo, 9], ls='--', color='#9aa5b1')
    ax.set_xlim(lo, 9); ax.set_ylim(lo, 9); ax.set_aspect('equal')
    ax.set_xlabel(XLAB); ax.set_ylabel('FTIR EC (µg/m³)'); ax.set_title(title, fontsize=10); ax.legend(fontsize=8, loc='upper left')

fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
season_crossplot('deployed|SPARTAN|current SPARTAN calibration', 'Starting point: current SPARTAN calibration', axes[0])
season_crossplot('allimprove|AIRSpec|all-IMPROVE lot 251', 'One all-IMPROVE model, spline baseline', axes[1])
season_crossplot('stitched|AIRSpec|season calibrations', 'Own-season calibrations, stitched (spline baseline)', axes[2])
fig.suptitle('Addis test set: FTIR EC vs HIPS Fabs / MAC 10, Deming fits by season', y=1.01)
fig.tight_layout(); fig.savefig(FIG / 'starting_point_and_stitched.png', bbox_inches='tight'); plt.show()
'''

FOUR = r'''def four_panel(method, family):
    fig, axes = plt.subplots(len(GROUPS), 4, figsize=(18, 4.4 * len(GROUPS)))
    for r, g in enumerate(GROUPS):
        if family == 'seasonal':
            fid = f'seasonal|{method}|locked|{g}'
            a = analogs.loc[analogs.fit_id.eq(fid) & analogs.role.eq('test')]
            sq = lambda p: st(fit_id=fid, population='analog_test', panel=p)
            sa = st(fit_id=fid, population='addis', eval_group=g)
        else:
            fid = f'allimprove|{method}|{family}'
            a = analogs.loc[analogs.fit_id.eq(f'{fid}|{g}') & analogs.role.eq('test')]
            sq = lambda p: st(fit_id=fid, group=g, population='analog_test', panel=p)
            sa = st(fit_id=fid, group=g, population='addis', eval_group=g)
        p = addis.loc[addis.fit_id.eq(fid)]
        if g != 'All Addis':
            p = p.loc[p.season.eq(g)]
        c = COL[g]
        panel(axes[r, 0], a.fabs_Mm1 / MAC_VALUE, a.tor_ec_ugm3, XLAB, 'TOR EC (µg/m³)', sq('tor_vs_fabs'), c, f'{g} · IMPROVE cross-validation: TOR EC vs HIPS')
        panel(axes[r, 1], a.tor_ec_ugm3, a.ftir_ec_ugm3, 'TOR EC (µg/m³)', 'FTIR EC (µg/m³)', sq('ftir_vs_tor'), c, f'{g} · IMPROVE cross-validation: FTIR vs TOR')
        panel(axes[r, 2], a.fabs_Mm1 / MAC_VALUE, a.ftir_ec_ugm3, XLAB, 'FTIR EC (µg/m³)', sq('ftir_vs_fabs'), c, f'{g} · IMPROVE cross-validation: FTIR vs HIPS')
        panel(axes[r, 3], p.Fabs / MAC_VALUE, p.ftir_ec_ugm3, XLAB, 'FTIR EC (µg/m³)', sa, c, f'{g} · Test set (Addis): FTIR vs HIPS')
    label = 'own-season calibration' if family == 'seasonal' else family
    fig.suptitle(f'{NAME.get(method, method)} · {label}: IMPROVE analogs from sites held out of the calibration set, and the Addis test set', y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / f'four_panel_{method}_{family.replace(" ", "_")}.png', bbox_inches='tight')
    plt.show()

four_panel('AIRSpec', 'seasonal')
'''

STRIP = r'''measures = [('addis_slope', 'Test set (Addis): FTIR vs Fabs/10', '#111827'), ('ftir_vs_fabs_slope', 'IMPROVE CV: FTIR vs Fabs/10', '#c026d3'),
            ('tor_vs_fabs_slope', 'IMPROVE CV: TOR vs Fabs/10', '#0f766e'), ('ftir_vs_tor_slope', 'IMPROVE CV: FTIR vs TOR', '#2171b5')]
fig, axes = plt.subplots(1, len(METHODS), figsize=(8 * len(METHODS), 7), sharey=True, squeeze=False)
for ax, method in zip(axes[0], METHODS):
    y, labels, ticks = 0, [], []
    for g in GROUPS:
        for key, lab, c in measures:
            m, lo, hi = q(method, g, key)
            ax.plot([lo, hi], [y, y], color=c, lw=3, alpha=.45); ax.plot(m, y, 'o', color=c)
            labels.append(f'{g.split()[0]} · {lab}'); ticks.append(y); y += 1
        y += .8
    ax.axvline(1, color='#9aa5b1', ls='--'); ax.set_xlim(0, 3)
    ax.set_title(f'{NAME.get(method, method)}: Deming slope, median and 10–90 % of 100 splits', fontsize=10)
    ax.set_xlabel('Deming slope (1 = agreement)')
axes[0][0].set_yticks(ticks, labels); axes[0][0].invert_yaxis()
fig.tight_layout(); fig.savefig(FIG / 'repeated_splits.png', bbox_inches='tight'); plt.show()
'''

MAC = r'''refs = pd.read_csv(T / 'improve_references.csv')
mem = pd.read_csv(T / 'analog_membership.csv').query("mask == 'locked'").merge(refs, on='filter_id', suffixes=('', '_r'))
ok = lambda d: d.loc[(d.tor_ec_ugm3 > 0.02) & (d.fabs_Mm1 > 0)]
bins = np.geomspace(2, 60, 41)
fig, axes = plt.subplots(1, len(METHODS), figsize=(8 * len(METHODS), 5), sharey=True, squeeze=False)
for ax, method in zip(axes[0], METHODS):
    pool = ok(refs); v = pool.fabs_Mm1 / pool.tor_ec_ugm3
    ax.hist(v, bins, density=True, histtype='step', lw=3, color='#8a8f98', label=f'IMPROVE pool, median {v.median():.1f}')
    for g in SEASONS:
        d = ok(mem.loc[mem.method.eq(method) & mem.group.eq(g)]); v = d.fabs_Mm1 / d.tor_ec_ugm3
        ax.hist(v, bins, density=True, histtype='step', lw=2, color=COL[g], label=f'{g} analogs, median {v.median():.1f}')
    ax.axvline(10, color='k', ls=':'); ax.set_xscale('log'); ax.set_xlabel('Fabs / TOR EC (m²/g)')
    ax.set_title(f'Analogs selected on {NAME.get(method, method).lower()} spectra', fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'mac_envelope.png', bbox_inches='tight'); plt.show()
'''

OPC = r'''display(op.round(3))
fid = 'seasonal|AIRSpec|locked|Dry (Oct-Feb)'
a = analogs.loc[analogs.fit_id.eq(fid) & analogs.role.eq('test')]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (x, y, yl) in zip(axes, [(a.fabs_Mm1 / MAC_VALUE, a.tor_ec_ugm3, 'TOR EC (µg/m³)'), (a.fabs_Mm1 / MAC_VALUE, a.ftir_ec_ugm3, 'FTIR EC (µg/m³)')]):
    sc = ax.scatter(x, y, c=a.op_tor_frac, cmap='viridis', vmin=.1, vmax=.8, s=28)
    hi = max(x.max(), y.max()) * 1.05; ax.plot([0, hi], [0, hi], ls='--', color='#9aa5b1'); ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_aspect('equal'); ax.set_xlabel(XLAB); ax.set_ylabel(yl)
fig.colorbar(sc, ax=axes, label='OP fraction = OPTR / (EC + OPTR)')
fig.suptitle('Dry analogs in IMPROVE cross-validation, coloured by TOR OP fraction (spline baseline)')
fig.savefig(FIG / 'dry_analogs_by_op.png', bbox_inches='tight'); plt.show()
'''

SLOPEMAP = r'''fig, axes = plt.subplots(1, len(METHODS), figsize=(7.5 * len(METHODS), 7), squeeze=False)
for ax, method in zip(axes[0], METHODS):
    g = grid.loc[grid.method.eq(method) & grid['mode'].eq('site_heldout') & (grid.heldout_R2 >= .85)]
    ax.scatter(g.all_slope, g.all_intercept, s=6, color='#9aa5b1', alpha=.5, label='grid, IMPROVE CV R² ≥ 0.85')
    for s in SEASONS:
        r = rep.loc[rep.method.eq(method) & rep.group.eq(s)]
        ax.scatter(r.addis_slope, r.addis_intercept, s=8, color=COL[s], alpha=.35)
        ax.scatter(r.addis_slope.median(), r.addis_intercept.median(), s=160, marker='^', color=COL[s], edgecolors='k', label=f'{s} calibration, own season')
    sr = stitched.loc[stitched.method.eq(method)]
    ax.scatter(sr.addis_slope, sr.addis_intercept, s=8, color='#6d28d9', alpha=.35)
    ax.scatter(sr.addis_slope.median(), sr.addis_intercept.median(), s=300, marker='*', color='#6d28d9', edgecolors='white', label='seasons stitched, all Addis')
    o = st(fit_id=f'allimprove|{method}|all-IMPROVE lot 251', population='addis', eval_group='All Addis')
    ax.scatter(o.deming_slope, o.deming_intercept, s=120, marker='s', color='k', label='all-IMPROVE model')
    d = st(fit_id='deployed|SPARTAN|current SPARTAN calibration', population='addis', eval_group='All Addis')
    ax.scatter(d.deming_slope, d.deming_intercept, s=160, marker='X', color='#b91c1c', label='current SPARTAN')
    ax.add_patch(plt.Rectangle((0.8, -1), 0.4, 2, color='#0f766e', alpha=.12))
    ax.axvline(1, ls='--', color='#9aa5b1'); ax.axhline(0, ls='--', color='#9aa5b1')
    ax.set_xlim(0, 3); ax.set_ylim(-8, 2)
    ax.set_xlabel('Test set (Addis) Deming slope'); ax.set_ylabel('Test set (Addis) Deming intercept (µg/m³)')
    ax.set_title(NAME.get(method, method), fontsize=10); ax.legend(fontsize=7.5, loc='lower left')
fig.tight_layout(); fig.savefig(FIG / 'slope_intercept_map.png', bbox_inches='tight'); plt.show()
'''

FOREST = r'''cals = [('All-IMPROVE model (lot 251)', 'allimprove|{m}|all-IMPROVE lot 251', True),
        ('Analogs of all Addis', 'seasonal|{m}|locked|All Addis', False),
        ('Own-season calibration', 'season', False),
        ('Seasons stitched', 'stitched|{m}|season calibrations', False),
        ('Three seasonal sets combined', 'combined|{m}|three seasonal sets combined', False),
        ('VIP/Euclidean analogs (500)', 'combined|{m}|VIP/Euclidean analogs (500)', False)]
rows = []
for m in METHODS:
    for g in GROUPS:
        for lab, tmpl, by in cals:
            fid = f'seasonal|{m}|locked|{g}' if tmpl == 'season' else tmpl.format(m=m)
            s = st(fit_id=fid, group=g, population='addis', eval_group=g) if by else st(fit_id=fid, population='addis', eval_group=g)
            if s is not None:
                rows.append({'baseline': NAME.get(m, m), 'season': g, 'calibration': lab, 'slope': s.deming_slope, 'slope_lo': s.slope_ci_low,
                             'slope_hi': s.slope_ci_high, 'intercept': s.deming_intercept, 'int_lo': s.intercept_ci_low,
                             'int_hi': s.intercept_ci_high, 'R2': s.R2, 'n': s.n})
forest = pd.DataFrame(rows)
display(forest.round(2))
'''

SPEC = r'''from spec_curve import passes_guardrails, GUARDRAIL_TEXT
g = grid.dropna(subset=['all_intercept']).sort_values('all_intercept').reset_index(drop=True)
ok = np.array([passes_guardrails({**r, 'deming_slope': r['all_slope']}) for r in g.to_dict('records')])
groups = [('Spectral preprocessing', [(NAME.get(m, m), g.method.eq(m)) for m in METHODS]),
          ('Calibration cohort', [('Lowest OC/EC', g.category.eq('ocec')), ('Spectral analogs (correlation), all Addis', g.category.eq('corr_all')),
                                  ('… Dry season', g.category.eq('corr_dry')), ('… Belg season', g.category.eq('corr_belg')),
                                  ('… Kiremt season', g.category.eq('corr_kiremt')), ('Ethiopia-shaped smoke', g.category.eq('eth_shaped')),
                                  ('Spectral analogs (VIP-weighted distance)', g.category.eq('vip_analogs')),
                                  ('All IMPROVE filters, or smoke (906)', g.category.isin(['pool', 'smoke']))]),
          ('Calibration settings', [('Site-grouped cross-validation', g['mode'].eq('site_heldout')), ('Interleaved cross-validation', g['mode'].eq('app')),
                                    ('9 or fewer PLS factors', g.k.le(9)), ('15 or more PLS factors', g.k.ge(15)), ('Cohort under 600 filters', g.cutoff.lt(600))])]
rows = [(lab, m) for _, items in groups for lab, m in items]
fig, (ax, bx) = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={'height_ratios': [1.1, 1.2]}, sharex=True)
x = np.arange(len(g)); lo = np.quantile(g.all_intercept, .003) - .6
ax.fill_between(x, 0, g.all_intercept.clip(lower=lo), color='#2171b5', alpha=.14)
ax.plot(x, g.all_intercept.clip(lower=lo), color='#2171b5', lw=1.4)
ax.axhspan(-.5, .5, color='#0f766e', alpha=.1); ax.axhline(0, color='k', lw=1)
ax.scatter(x[ok], g.all_intercept[ok], color='#b2182b', s=10, zorder=3, label=f'passes every guardrail ({ok.sum()} of {len(g)})')
ax.set_ylim(lo, max(2.6, np.quantile(g.all_intercept, .997) + 1.9)); ax.set_ylabel('Test set (Addis) Deming intercept, MAC 10 (µg/m³)')
ax.set_title(f'Specification curve: {len(g):,} baselined Addis specifications, sorted by the intercept they produce'); ax.legend(loc='lower right')
bins = np.array_split(x, min(len(x), 800))
img = np.array([[m.to_numpy()[b].mean() for b in bins] for _, m in rows])
im = bx.imshow(img, aspect='auto', cmap='Reds', vmin=0, vmax=1, extent=[0, len(g), len(rows), 0], interpolation='nearest')
y0 = 0
for title, items in groups:
    bx.text(-0.01, y0 + 0.1, title, transform=bx.get_yaxis_transform(), ha='right', va='top', fontsize=9.5, fontweight='bold')
    y0 += len(items)
    bx.axhline(y0, color='white', lw=3)
bx.set_yticks(np.arange(len(rows)) + .5, [lab for lab, _ in rows], fontsize=8.5)
bx.set_xlabel('specification, ordered by the intercept it produces')
fig.colorbar(im, ax=bx, fraction=.02, label='share of specifications using the choice')
fig.text(.01, .002, 'Guardrails: ' + GUARDRAIL_TEXT.replace('held-out TOR R²', 'IMPROVE cross-validation R²'), fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'specification_curve.png', bbox_inches='tight'); plt.show()
'''

HEAT = r'''def heat(method, mode, metric):
    d = grid.loc[grid.method.eq(method) & grid['mode'].eq(mode)].copy()
    d['bin'] = (d.cutoff / 25).round() * 25
    if mode == 'site_heldout':
        d.loc[d.heldout_R2 < .85, metric] = np.nan
    return d.pivot_table(index='label', columns='bin', values=metric, aggfunc='mean')
combos = [(m, mode) for m in METHODS for mode in ('site_heldout', 'app')]
fig, axes = plt.subplots(len(combos), 2, figsize=(20, 4.4 * len(combos)), squeeze=False)
CV = {'site_heldout': 'site-grouped CV', 'app': 'interleaved CV'}
for r, (method, mode) in enumerate(combos):
    for c, (metric, vmin, vmax) in enumerate([('all_intercept', -2.5, 2.5), ('all_slope', 0, 2)]):
        piv = heat(method, mode, metric); ax = axes[r, c]
        im = ax.imshow(piv.values, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(piv.index)), piv.index, fontsize=9)
        step = max(1, len(piv.columns) // 12)
        ax.set_xticks(range(0, len(piv.columns), step), [int(v) for v in piv.columns[::step]], fontsize=8)
        ax.set_title(f"{NAME.get(method, method)} · {CV[mode]} · test set (Addis) {metric.replace('all_', '')}" + (' · blank = IMPROVE CV R² < 0.85' if mode == 'site_heldout' else ''), fontsize=10)
        ax.set_xlabel('Calibration cohort size (IMPROVE filters, 25-filter bins)')
        fig.colorbar(im, ax=ax, fraction=.025)
fig.tight_layout(); fig.savefig(FIG / 'category_grid_heatmap.png', bbox_inches='tight'); plt.show()
'''


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [
        md("""# FTIR EC follow-up: tasks from the 17 Sep 2026 meeting, updated for 23 Sep

Works through Ann's task list (email of 18 Sep) and the 23 Sep review: the IMPROVE-analog
bias check, TOR/HIPS for the analogs, OP, seasonality, per-season calibrations stitched
together, the category grid (heat map and specification curve), and the fixed levers.

Conventions (docs/naming-conventions.md): **calibration set** = IMPROVE filters a model is built on;
**cross-validation** = IMPROVE sites held out of the calibration set; **test set** = the Addis
filters. Regression is **Deming only**. "Spline baseline" = AIRSpec. The same results are in the
gallery's **Meeting 17 Sep follow-up** tab."""),
        code(SETUP),
        md("## tl;dr"),
        code(TLDR),
        md("""## 1. Starting point, one all-IMPROVE model, and the seasons stitched

Left: the poster's reference plot, the calibration SPARTAN uses today. Middle: one calibration
on all IMPROVE filters of the Addis lot (spline baseline). Right: each season's Addis filters
predicted by a calibration built on that season's own analogs."""),
        code(STARTING),
        md("""## 2. Where every calibration lands on slope vs intercept

Target: slope 1, intercept 0 (green box: slope 0.8–1.2, |intercept| < 1). Grey: grid
configurations passing IMPROVE cross-validation R² ≥ 0.85. Triangles: each season calibration
read out on its own season; star: the seasons stitched, read out on all Addis."""),
        code(SLOPEMAP),
        md("""## 3. IMPROVE analog check (Satoshi): analogs in cross-validation vs the Addis test set

For each season: the analogs at sites held out of the season calibration set, and the Addis
season, predicted by the **same** calibration."""),
        code(FOUR),
        code("four_panel('AIRSpec', 'all-IMPROVE lot 251')\nfor m in METHODS[1:]:\n    four_panel(m, 'seasonal')"),
        md("### 100 cross-validation splits per season"),
        code(STRIP),
        md("## 4. Where the analogs sit in the TOR EC vs HIPS envelope"),
        code(MAC),
        md("""## 5. OP (charring)

OPTR and OPTT are in the local TOR table; EC2, EC3 and OC1–OC4 need a FED portal pull."""),
        code(OPC),
        md("## 6. All calibrations, by season (table)"),
        code(FOREST),
        md("""## 7. Specification curve over the baselined grid

Rows grouped by preprocessing, cohort and calibration settings; strip colour is the share of
the specifications at that position using the choice (white none, dark red all)."""),
        code(SPEC),
        md("## 8. Category grid as a heat map"),
        code(HEAT),
        md("""## Still open

* EC2, EC3, OC1–OC4 for the IMPROVE analogs (FED portal query; site was failing).
* The scoring decision (slope, intercept, or a target range for both).
* An independent Addis EC reference: every Addis number here is against Fabs/10."""),
    ]
    nbf.write(nb, DEST)
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
