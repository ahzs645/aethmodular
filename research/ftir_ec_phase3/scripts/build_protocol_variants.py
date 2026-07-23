#!/usr/bin/env python3
"""Standalone per-protocol versions of every k-dependent phase-3 figure.

The comparison figures in ftir_21/ftir_22 overlay both calibration protocols on one set of
axes, which is right for arguing that the protocol matters. For the deck you often want the
opposite: one clean slide per protocol, so "here is the Calibration app's answer" and "here
is the site-held-out answer" can be shown in sequence.

This writes each k-dependent figure twice — once per protocol — into a folder per protocol:

    output/plots/deck/by_protocol/calibration_app/<figure>.png
    output/plots/deck/by_protocol/site_held_out/<figure>.png

Matching file names across the two folders means a deck can be assembled by pointing at one
folder and swapped wholesale by pointing at the other. Each figure also states its protocol
in its own title, so a PNG stays self-identifying once pulled out of its folder.

Everything is read from the committed ftir_21/ftir_22 tables, so no model is refitted here
and the variants cannot drift from the notebooks that produced them.

Run ftir_21 and ftir_22 first. Usage: ``python scripts/build_protocol_variants.py``
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/plots/deck/by_protocol'

MODES = ('app', 'site_heldout')
MODE_DIR = {'app': 'calibration_app', 'site_heldout': 'site_held_out'}
MODE_TITLE = {'app': 'Calibration app protocol',
              'site_heldout': 'Site-held-out protocol'}
MODE_SUB = {
    'app': 'interleaved 10-fold CV · k = first within 5% of the minimum · fitted on all filters',
    'site_heldout': 'site-grouped 5-fold CV · k = first major minimum · site-disjoint fit',
}
INK, MUTED = '#22252A', '#6B6E75'

# Colour encodes the calibration setup, matching `calibration_setup_matrix.png` and the
# deck intercept ladder — NOT the protocol. The protocol is already carried by the folder
# and by each figure's own subtitle, so spending colour on it would waste the channel.
#
# Note this is a deliberately redundant encoding: every setup is also named by its panel
# title or row label, so nothing here requires telling two hues apart. (As a strict
# categorical palette it would not pass a CVD check — deck BLUE and PURPLE are close — but
# no chart below asks the reader to discriminate them.)
GREY, BLUE, PURPLE, ACCENT = '#8F8C84', '#2C6E9E', '#7A4FA3', '#B23327'
SETUP_COLOUR = {
    'Entire IMPROVE network (13,010, no selection)': GREY,
    'Biomass-smoke (906)': GREY,
    'Ethiopia-shaped smoke (300)': BLUE,
    'Spectral analogs (locked 500)': BLUE,
    'Lowest-OC/EC (800)': PURPLE,
    'Lowest-OC/EC + AIRSpec (800)': ACCENT,
}
# The ftir_15-derived figures are both the lowest-OC/EC cohort, split by preprocessing —
# so they take that family's two matrix colours.
SPECTRA_COLOUR = {'raw': PURPLE, 'AIRSpec df1=6': ACCENT}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.edgecolor': '#BBBBBB', 'axes.linewidth': .8,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED})

T21 = ROOT / 'output/tables/ftir21'
T22 = ROOT / 'output/tables/ftir22'
T23 = ROOT / 'output/tables/ftir23'

COHORT_ORDER = [
    'Entire IMPROVE network (13,010, no selection)',
    'Biomass-smoke (906)',
    'Ethiopia-shaped smoke (300)',
    'Spectral analogs (locked 500)',
    'Lowest-OC/EC (800)',
    'Lowest-OC/EC + AIRSpec (800)',
]
FAILS_TOR = {'Ethiopia-shaped smoke (300)', 'Spectral analogs (locked 500)'}


def _stamp(fig, figure_title, mode):
    fig.suptitle(f'{figure_title}\n{MODE_TITLE[mode]} — {MODE_SUB[mode]}',
                 y=1.0, fontsize=12.5, fontweight='bold', color=INK)


def _save(fig, stem, mode):
    directory = OUT / MODE_DIR[mode]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{stem}.png'
    fig.savefig(path, dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(path.relative_to(OUT))


def _fit_line(ax, slope, intercept, hi, colour):
    ax.plot([0, hi], [intercept, slope * hi + intercept], color=colour, lw=1.8)


# Row content for the setup matrix: (setup, filter applied, samples, selection axis).
SETUP_ROWS = [
    ('Entire IMPROVE network', 'no selection at all', '13,010',
     'none — 1 model for all sites'),
    ('Biomass-smoke', 'Katie-George smoke ratios', '906', 'wildfire days only'),
    ('Ethiopia-shaped smoke', 'nearest Addis in CH /\ncarbonyl / 1600 features', '300',
     'spectral shape'),
    ('Spectral analogs', 'Mahalanobis + Q +\nVIP-weighted mismatch', '500',
     'score-space distance'),
    ('Lowest-OC/EC', 'lowest TOR OC/EC ratio', '800', 'composition (OC/EC ≤ 2.27)'),
    ('Lowest-OC/EC + AIRSpec', 'same, on baselined spectra', '800',
     'composition + baseline'),
]


def fig_setup_matrix(mode, metrics):
    """The calibration setup matrix carrying only this protocol's intercept column.

    The combined deck version (`build_deck_figures.py`) shows both protocols side by side;
    this is the single-protocol slide for the matching folder.
    """
    rows = metrics[(metrics['mode'] == mode) & (metrics['MAC_m2_g'] == 10)
                   & (metrics['evaluation_set'] == 'fixed phase-2 cohort')]
    rows = rows.set_index('cohort')

    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(-9, 100)
    ax.axis('off')
    cols_x = [2, 25, 48, 59, 90]
    headers = ['Calibration setup', 'Filter applied to the pool', 'Samples',
               'Selection axis', 'Addis intercept']
    ax.text(0, 97, 'Every setup is the same PLS math on a differently-filtered training set',
            fontsize=14, fontweight='bold', color=INK)
    ax.text(0, 91.5, f'{MODE_TITLE[mode]} — {MODE_SUB[mode]}', fontsize=9.5, color=MUTED)
    for x, head in zip(cols_x, headers):
        ax.text(x, 85, head.upper(), fontsize=8.5, color=MUTED, fontweight='bold',
                ha='right' if x >= 90 else 'left')
    ax.plot([0, 100], [82, 82], color='#DDDDDD', lw=1)

    row_h, box_h = 13.2, 11.0
    for i, (name, filt, samples, axis) in enumerate(SETUP_ROWS):
        cohort = COHORT_ORDER[i]
        colour = SETUP_COLOUR[cohort]
        y = 74 - i * row_h
        ax.add_patch(FancyBboxPatch((0.5, y - box_h / 2), 99, box_h,
                                    boxstyle='round,pad=0.2,rounding_size=1.5',
                                    linewidth=0, facecolor=colour, alpha=.09))
        ax.plot([1.4, 1.4], [y - box_h / 2 + 1, y + box_h / 2 - 1], color=colour, lw=3,
                solid_capstyle='round')
        ax.text(cols_x[0], y, name, fontsize=11, fontweight='bold', color=INK, va='center')
        ax.text(cols_x[1], y, filt, fontsize=9, color=MUTED, va='center')
        ax.text(cols_x[2] + 4, y, samples, fontsize=10.5, color=INK, va='center',
                ha='right', fontfamily='monospace')
        ax.text(cols_x[3], y, axis + ('  ⚠ fails TOR' if cohort in FAILS_TOR else ''),
                fontsize=9, color=MUTED, va='center')
        intercept = float(rows.loc[cohort, 'intercept'])
        k = int(rows.loc[cohort, 'k'])
        ax.text(cols_x[4], y, f'{intercept:.2f}'.replace('-', '−'), fontsize=11.5,
                color=colour, va='center', ha='right', fontweight='bold',
                fontfamily='monospace')
        ax.text(cols_x[4] + 8, y, f'k={k}', fontsize=8.5, color=MUTED, va='center',
                ha='right', fontfamily='monospace')
    ax.text(0, -6, 'Intercepts at MAC = 10 on the fixed 190-filter cohort (ftir_21). '
                   '⚠ = no held-out TOR skill once whole sites are held out.',
            fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return _save(fig, 'calibration_setup_matrix', mode)


def fig_crossplots(mode, predictions, metrics):
    """The six-setup Addis crossplot grid, one protocol only (cf. ftir_21)."""
    fabs = predictions['Fabs'].to_numpy(float)
    x = fabs / 10
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 9.0))
    for ax, cohort in zip(axes.flat, COHORT_ORDER):
        prediction = predictions[f'{cohort} [{mode}]'].to_numpy(float)
        row = metrics[(metrics['cohort'] == cohort) & (metrics['mode'] == mode)
                      & (metrics['MAC_m2_g'] == 10)
                      & (metrics['evaluation_set'] == 'fixed phase-2 cohort')].iloc[0]
        hi = float(max(np.nanmax(x), np.nanmax(prediction))) * 1.06
        lo = min(0.0, float(np.nanmin(prediction))) * 1.05
        ax.plot([0, hi], [0, hi], '--', color='0.6', lw=1, zorder=1)
        ax.axhline(0, color='0.85', lw=.8, zorder=0)
        colour = SETUP_COLOUR[cohort]
        ax.scatter(x, prediction, s=16, alpha=.45, color=colour, lw=0)
        _fit_line(ax, row['slope'], row['intercept'], hi, colour)
        title = cohort + ('  ⚠' if cohort in FAILS_TOR else '')
        ax.set_title(title.replace(' (', '\n('), fontsize=10, color=INK)
        ax.text(.04, .96, f"k = {int(row['k'])}\ny = {row['slope']:.2f}x "
                          f"{row['intercept']:+.2f}\nR² = {row['R2']:.2f}  "
                          f"RMSE = {row['RMSE']:.2f}",
                transform=ax.transAxes, va='top', fontsize=8.6,
                bbox=dict(facecolor='white', edgecolor='0.85', alpha=.93))
        ax.set_xlim(0, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
    for ax in axes[1]:
        ax.set_xlabel('HIPS EC-equivalent, Fabs/10 (µg/m³)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Predicted FTIR EC (µg/m³)')
    _stamp(fig, 'Addis calibration, every setup (MAC = 10, fixed 190-filter cohort)', mode)
    fig.tight_layout()
    return _save(fig, 'crossplots_all_setups', mode)


def fig_intercept_ladder(mode, metrics):
    """Addis intercept and slope per setup, one protocol only (cf. the deck ladder)."""
    rows = metrics[(metrics['mode'] == mode) & (metrics['MAC_m2_g'] == 10)
                   & (metrics['evaluation_set'] == 'fixed phase-2 cohort')]
    rows = rows.set_index('cohort').reindex(COHORT_ORDER)
    fig, (ax_i, ax_s) = plt.subplots(1, 2, figsize=(13.2, 4.8))
    y = np.arange(len(COHORT_ORDER))[::-1]
    colours = [SETUP_COLOUR[c] for c in rows.index]
    for ax, column, reference, label in (
            (ax_i, 'intercept', 0.0, 'Addis intercept (µg/m³) — zero is the target'),
            (ax_s, 'slope', 1.0, 'Addis slope — 1.0 is self-consistent')):
        values = rows[column].to_numpy(float)
        ax.hlines(y, reference, values, color=colours, lw=5, alpha=.3, capstyle='round')
        ax.scatter(values, y, color=colours, s=115, zorder=3)
        for yi, value, colour in zip(y, values, colours):
            ax.annotate(f'{value:+.2f}' if column == 'intercept' else f'{value:.2f}',
                        (value, yi), textcoords='offset points',
                        xytext=(0, 11), ha='center', fontsize=9, color=colour,
                        fontweight='bold', fontfamily='monospace')
        ax.axvline(reference, color=INK, lw=1)
        ax.set_xlabel(label)
        ax.grid(axis='x', color='0.93', lw=.7)
        ax.set_axisbelow(True)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
    ax_i.set_yticks(y)
    ax_i.set_yticklabels([c + ('  ⚠' if c in FAILS_TOR else '') for c in rows.index],
                         fontsize=9.5)
    ax_s.set_yticks([])
    _stamp(fig, 'Addis intercept and slope by calibration setup (MAC = 10)', mode)
    fig.tight_layout()
    return _save(fig, 'intercept_slope_ladder', mode)


def fig_mac_effect(mode, predictions, metrics):
    """The MAC 10-vs-6 pivot per setup, one protocol only (cf. ftir_19/ftir_22)."""
    fabs = predictions['Fabs'].to_numpy(float)
    x10, x6 = fabs / 10, fabs / 6
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 9.0))
    for ax, cohort in zip(axes.flat, COHORT_ORDER):
        colour = SETUP_COLOUR[cohort]
        prediction = predictions[f'{cohort} [{mode}]'].to_numpy(float)
        row10 = metrics[(metrics['cohort'] == cohort) & (metrics['mode'] == mode)
                        & (metrics['MAC_m2_g'] == 10)
                        & (metrics['evaluation_set'] == 'fixed phase-2 cohort')].iloc[0]
        row6 = metrics[(metrics['cohort'] == cohort) & (metrics['mode'] == mode)
                       & (metrics['MAC_m2_g'] == 6)
                       & (metrics['evaluation_set'] == 'fixed phase-2 cohort')].iloc[0]
        hi = float(max(np.nanmax(x6), np.nanmax(prediction))) * 1.05
        lo = min(0.0, float(row10['intercept'])) - .8
        ax.plot([0, hi], [0, hi], '--', color='0.6', lw=1, zorder=1)
        ax.axhline(0, color='0.85', lw=.9, zorder=0)
        ax.scatter(x10, prediction, s=15, alpha=.42, color=colour, lw=0)
        ax.scatter(x6, prediction, s=15, alpha=.5, facecolors='none',
                   edgecolors=colour, lw=.6)
        _fit_line(ax, row10['slope'], row10['intercept'], hi, colour)
        ax.plot([0, hi], [row6['intercept'], row6['slope'] * hi + row6['intercept']],
                '--', color=colour, lw=1.6)
        ax.scatter([0], [row10['intercept']], marker='D', s=44, color=INK, zorder=5)
        ax.set_title((cohort + ('  ⚠' if cohort in FAILS_TOR else '')).replace(' (', '\n('),
                     fontsize=10, color=INK)
        ax.text(.04, .96, f"k = {int(row10['k'])}\nintercept {row10['intercept']:+.2f} "
                          f"(both MACs)\nslope {row10['slope']:.2f} → {row6['slope']:.2f} "
                          f"@ MAC 6",
                transform=ax.transAxes, va='top', fontsize=8.4,
                bbox=dict(facecolor='white', edgecolor='0.85', alpha=.93))
        ax.set_xlim(0, hi)
        ax.set_ylim(lo, hi)
    for ax in axes[1]:
        ax.set_xlabel('HIPS EC-equivalent, Fabs/MAC (µg/m³)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Predicted FTIR EC (µg/m³)')
    _stamp(fig, 'The MAC fix applied to HIPS — filled/solid MAC 10, open/dashed MAC 6', mode)
    fig.tight_layout()
    return _save(fig, 'mac_effect_all_setups', mode)


def fig_mac_slope_pivot(mode, metrics):
    """Slope dumbbells MAC 10 -> 6 per setup, one protocol only (cf. ftir_19).

    Mirrors the committed `mac_slope_pivot.png` layout: filled dot = MAC 10, open dot =
    MAC 6, both slope values printed at their own marker, and the MAC-invariant intercept
    in a fixed column on the right.
    """
    rows10 = metrics[(metrics['mode'] == mode) & (metrics['MAC_m2_g'] == 10)
                     & (metrics['evaluation_set'] == 'fixed phase-2 cohort')
                     ].set_index('cohort')
    rows6 = metrics[(metrics['mode'] == mode) & (metrics['MAC_m2_g'] == 6)
                    & (metrics['evaluation_set'] == 'fixed phase-2 cohort')
                    ].set_index('cohort')
    order = COHORT_ORDER[::-1]
    slope_max = float(rows10.loc[COHORT_ORDER, 'slope'].max())
    intercept_x = slope_max + .62
    x_right = intercept_x + 1.05

    fig, ax = plt.subplots(figsize=(10.9, 5.2))
    for yi, cohort in enumerate(order):
        colour = SETUP_COLOUR[cohort]
        fails = cohort in FAILS_TOR
        alpha = .45 if fails else 1.0
        slope10 = float(rows10.loc[cohort, 'slope'])
        slope6 = float(rows6.loc[cohort, 'slope'])
        ax.plot([slope6, slope10], [yi, yi], color=colour, lw=2.4, alpha=.3 * alpha,
                zorder=2)
        ax.scatter([slope10], [yi], s=110, color=colour, alpha=alpha, zorder=3)
        ax.scatter([slope6], [yi], s=110, facecolors='white', edgecolors=colour, lw=1.8,
                   alpha=alpha, zorder=3)
        label = cohort + ('  ⚠ fails TOR' if fails else '')
        ax.text(-.06, yi, label, ha='right', va='center', fontsize=9.5,
                color=MUTED if fails else INK)
        for slope, side in ((slope6, 'right'), (slope10, 'left')):
            offset = -.11 if side == 'right' else .11
            near_one = abs(slope - 1) <= .1 and not fails
            ax.text(slope + offset, yi, f'{slope:.2f}', ha=side, va='center', fontsize=10,
                    color=MUTED if fails else INK, alpha=alpha, zorder=4,
                    fontweight='bold' if near_one else 'normal', fontfamily='monospace',
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.4))
        ax.text(intercept_x, yi, f"{float(rows10.loc[cohort, 'intercept']):+.2f}",
                va='center', ha='left', fontsize=10, color=MUTED, fontfamily='monospace')
        ax.text(x_right - .04, yi, f"k={int(rows10.loc[cohort, 'k'])}", va='center',
                ha='right', fontsize=8.5, color=MUTED, fontfamily='monospace')
    ax.axvline(1, color=INK, lw=1, zorder=1)

    top = len(order) - .3
    ax.text(1, top, 'self-consistent\n(slope = 1)', fontsize=9, color=INK, ha='center',
            va='bottom')
    ax.text(intercept_x, top, 'INTERCEPT\n(same at both MACs)', fontsize=8.5, color=MUTED,
            ha='left', va='bottom', fontweight='bold')
    ax.scatter([], [], s=110, color=INK, label='slope @ MAC 10  (filled)')
    ax.scatter([], [], s=110, facecolors='white', edgecolors=INK, lw=1.8,
               label='slope @ MAC 6  (open)')
    ax.legend(loc='upper center', frameon=False, fontsize=9.5, ncol=2,
              bbox_to_anchor=(.42, -.12))
    ax.set_yticks([])
    ax.set_ylim(-.55, len(order) + .55)
    ax.set_xlim(0, x_right)
    ax.set_xticks(np.arange(0, slope_max + .5, .5))
    ax.set_xlabel('Addis slope, predicted EC vs Fabs/MAC (fixed cohort, n = 190)')
    _stamp(fig, 'Every slope moves ×0.6 from MAC 10 to MAC 6 — every intercept stays put',
           mode)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_bounds(0, slope_max + .25)
    fig.tight_layout()
    return _save(fig, 'mac_slope_pivot', mode)


def fig_component_selection(mode, curves, decisions):
    """The CV curve and the rule that chose k, one protocol only (cf. ftir_23).

    Draws each rule's own machinery: the Calibration app's within-5% acceptance band, or
    the site-held-out protocol's ±1 SE ribbon with every local minimum marked.
    """
    part_curves = curves[curves['mode'] == mode]
    part_decisions = decisions[decisions['mode'] == mode].set_index('cohort')

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.4))
    for ax, cohort in zip(axes.flat, COHORT_ORDER):
        curve = part_curves[part_curves['cohort'] == cohort].sort_values('n_components')
        row = part_decisions.loc[cohort]
        colour = SETUP_COLOUR[cohort]
        components = curve['n_components'].to_numpy(int)
        values = curve['rmsecv'].to_numpy(float)

        if mode == 'site_heldout':
            se = curve['rmse_se'].to_numpy(float)
            ax.fill_between(components, values - se, values + se, color=colour,
                            alpha=.13, lw=0)
        ax.plot(components, values, '-', color=colour, lw=1.7, zorder=3)
        ax.axhline(float(row['threshold_rmsecv']), color=MUTED, lw=1, ls=':', zorder=2)
        if mode == 'app':
            ax.axhspan(values.min(), float(row['threshold_rmsecv']), color=MUTED,
                       alpha=.09, lw=0)
        else:
            for minimum in (int(m) for m in str(row['local_minima']).split()
                            if m and m != 'nan'):
                index = list(components).index(minimum)
                ax.scatter([minimum], [values[index]], s=26, facecolors='white',
                           edgecolors=colour, lw=1.1, zorder=4)

        chosen = int(row['k'])
        chosen_index = list(components).index(chosen)
        ax.scatter([chosen], [values[chosen_index]], s=150, color=colour, zorder=6,
                   edgecolors='white', lw=1.6)
        ax.annotate(f'k = {chosen}', (chosen, values[chosen_index]),
                    textcoords='offset points', xytext=(0, 17), ha='center',
                    fontsize=10.5, fontweight='bold', color=colour, zorder=7,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
        if int(row['global_min_k']) != chosen:
            ax.text(.98, .04, f"curve bottoms at k = {int(row['global_min_k'])}",
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8.2,
                    color=MUTED)

        title = cohort + ('  ⚠' if cohort in FAILS_TOR else '')
        ax.set_title(title.replace(' (', '\n('), fontsize=10, color=INK)
        ax.set_xlim(0, 31)
        ax.margins(y=.14)
        ax.grid(axis='y', color='0.94', lw=.7)
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel('n PLS components')
    for ax in axes[:, 0]:
        ax.set_ylabel('RMSECV (µg/filter)')
    rule = ('shaded: within 5% of the minimum'
            if mode == 'app' else 'ribbon: ±1 SE · open dots: local minima')
    _stamp(fig, f"How this protocol chooses k — dotted line: acceptance threshold · {rule}",
           mode)
    fig.tight_layout()
    return _save(fig, 'component_selection', mode)


def fig_bootstrap(mode, draws):
    """Site-cluster bootstrap of the Addis intercept/slope, one protocol only (cf. ftir_15)."""
    part = draws[draws['mode'] == mode]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 2.9))
    spectra_order = ['raw', 'AIRSpec df1=6']
    for ax, metric, reference in ((axes[0], 'intercept', 0.0), (axes[1], 'slope', 1.0)):
        for offset, spectra in enumerate(spectra_order):
            group = part[part['spectra'] == spectra]
            colour = SPECTRA_COLOUR[spectra]
            low, high = np.percentile(group[metric], [2.5, 97.5])
            ax.plot([low, high], [offset, offset], color=colour, lw=4, alpha=.42,
                    solid_capstyle='round')
            ax.scatter([group[metric].median()], [offset], color=colour, s=100, zorder=3)
            ax.annotate(f'[{low:+.2f}, {high:+.2f}]', (high, offset),
                        textcoords='offset points', xytext=(10, 0), va='center',
                        fontsize=8.8, color=MUTED, fontfamily='monospace')
        ax.axvline(reference, color=INK, lw=1)
        ax.set_yticks(range(len(spectra_order)))
        ax.set_yticklabels([f"{s}\n(k={int(part[part['spectra'] == s]['k'].iloc[0])})"
                            for s in spectra_order], fontsize=9)
        ax.set_xlabel(f'Addis {metric} — 95% CI over resampled training sites')
        ax.set_ylim(-.7, len(spectra_order) - .3)   # two rows only; keep them close
        ax.margins(x=.28)
        ax.grid(axis='x', color='0.93', lw=.7)
        ax.set_axisbelow(True)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
    _stamp(fig, 'Site-cluster bootstrap of the Addis regression (B = 200)', mode)
    fig.tight_layout()
    return _save(fig, 'bootstrap_intercept_ci', mode)


def fig_residual_vs_d2(mode, residuals):
    """Addis residual vs extrapolation distance, one protocol only (cf. ftir_15)."""
    part = residuals[residuals['mode'] == mode]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    for ax, spectra in zip(axes, ['raw', 'AIRSpec df1=6']):
        colour = SPECTRA_COLOUR[spectra]
        frame = part[part['spectra'] == spectra].dropna(subset=['D2', 'residual_ugm3'])
        ax.scatter(frame['D2'], frame['residual_ugm3'], s=16, alpha=.5, color=colour, lw=0)
        ax.axhline(0, color=INK, lw=1)
        ax.set_xscale('log')
        r = frame['D2'].corr(frame['residual_ugm3'])
        ax.set_title(f"{spectra} spectra  (k = {int(frame['k'].iloc[0])})",
                     fontsize=10.5, color=INK)
        ax.annotate(f'r = {r:+.2f}', xy=(.97, .06), xycoords='axes fraction', ha='right',
                    fontsize=11, color=colour, fontweight='bold')
        ax.set_xlabel('D² from the calibration score cloud (log)')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel('Addis residual: FTIR EC − Fabs/10 (µg/m³)')
    _stamp(fig, 'Does the prediction residual track distance from the calibration cloud?',
           mode)
    fig.tight_layout()
    return _save(fig, 'residual_vs_d2', mode)


def fig_cohort_sweep(mode, sweep):
    """Lowest-OC/EC cohort-size sweep, one protocol only (cf. ftir_11)."""
    part = sweep[sweep['mode'] == mode].sort_values('cohort_size')
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    colour = PURPLE          # all three sizes are the lowest-OC/EC family
    for ax, metric, reference, title in zip(
            axes, ('intercept', 'slope', 'RMSE'), (0.0, 1.0, None),
            ('Addis intercept (µg/m³)', 'Addis slope', 'Addis RMSE (µg/m³)')):
        ax.plot(part['cohort_size'], part[metric], '-o', color=colour, lw=1.9, ms=9)
        for _, row in part.iterrows():
            ax.annotate(f"k={int(row['k'])}", (row['cohort_size'], row[metric]),
                        textcoords='offset points', xytext=(0, 10), ha='center',
                        fontsize=8.5, color=colour)
        if reference is not None:
            ax.axhline(reference, color=INK, lw=1, ls=':')
        ax.set_xscale('log')
        ax.set_xticks(part['cohort_size'])
        ax.set_xticklabels([str(int(s)) for s in part['cohort_size']])
        ax.set_xlabel('lowest-OC/EC cohort size')
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.grid(axis='y', color='0.93', lw=.7)
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    _stamp(fig, 'Lowest-OC/EC cohort-size sweep (raw spectra, MAC = 10)', mode)
    fig.tight_layout()
    return _save(fig, 'cohort_size_sweep', mode)


def main():
    predictions = pd.read_csv(T21 / 'addis_predictions_by_mode.csv')
    metrics = pd.read_csv(T21 / 'addis_metrics_by_mode.csv')
    draws = pd.read_csv(T22 / 'bootstrap_draws_by_mode.csv')
    residuals = pd.read_csv(T22 / 'addis_residuals_by_mode.csv')
    sweep = pd.read_csv(T22 / 'cohort_size_sweep_by_mode.csv')
    selection_curves = pd.read_csv(T23 / 'selection_curves.csv')
    selection_decisions = pd.read_csv(T23 / 'selection_decisions.csv')

    # Remove the previous flat layout (<figure>__<mode>.png) so the folder has one scheme.
    for stale in list(OUT.glob('*__*.png')) + list(OUT.glob('index.csv')):
        stale.unlink()

    written = []
    for mode in MODES:
        written.append(fig_setup_matrix(mode, metrics))
        written.append(fig_component_selection(mode, selection_curves,
                                               selection_decisions))
        written.append(fig_crossplots(mode, predictions, metrics))
        written.append(fig_intercept_ladder(mode, metrics))
        written.append(fig_mac_effect(mode, predictions, metrics))
        written.append(fig_mac_slope_pivot(mode, metrics))
        written.append(fig_bootstrap(mode, draws))
        written.append(fig_residual_vs_d2(mode, residuals))
        written.append(fig_cohort_sweep(mode, sweep))

    index = pd.DataFrame({'path': written})
    index['protocol'] = index['path'].str.split('/').str[0]
    index['figure'] = index['path'].str.split('/').str[1]
    index.to_csv(OUT / 'index.csv', index=False)
    print(f'wrote {len(written)} figures under {OUT.relative_to(ROOT)}/')
    for folder in MODE_DIR.values():
        print(f'  {folder}/')
        for name in sorted(p.name for p in (OUT / folder).glob('*.png')):
            print('    ', name)


if __name__ == '__main__':
    main()
