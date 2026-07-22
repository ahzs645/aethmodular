#!/usr/bin/env python3
"""Presentation figure utility: how each calibration setup filters its samples.

Reusable across deck refreshes. Reads the committed cohort/metric tables plus the
IMPROVE pool OC/EC values (via FTIRTransferPaths), writes standalone PNGs under
output/plots/deck/. Not part of the numbered analysis chain — purely presentational.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str((Path(__file__).resolve().parents[2] / 'ftir_hips_chem' / 'scripts')))
from pls_transfer import FTIRTransferPaths  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/plots/deck'
OUT.mkdir(parents=True, exist_ok=True)
PATHS = FTIRTransferPaths.defaults()

INK = '#22252A'
MUTED = '#6B6E75'
ACCENT = '#B23327'
BLUE = '#2C6E9E'
PURPLE = '#7A4FA3'
GREY = '#B9B6AD'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.edgecolor': '#BBBBBB', 'axes.linewidth': .8,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED})


def pool_ocec():
    """Full lot-248/251 pool TOR OC/EC ratios (one per eligible filter)."""
    sim = pd.read_csv(ROOT.parent / 'ftir_hips_chem/output/tables/pls_transfer/'
                      'improve_full_pool_addis_similarity.csv')
    sim['date'] = pd.to_datetime(sim['SampleDate'], format='mixed',
                                 errors='coerce').dt.normalize()
    tor = pd.read_csv(PATHS.ftir_dir / 'local_db/tables/results_tor.csv',
                      usecols=['Site', 'SampleDate', 'Parameter', 'Value'])
    tor = tor[tor['Parameter'].isin(['EC', 'OC'])].copy()
    tor['date'] = pd.to_datetime(tor['SampleDate'], format='mixed',
                                 errors='coerce').dt.normalize()
    tor = tor.drop_duplicates(['Site', 'date', 'Parameter'])
    wide = tor.pivot_table(index=['Site', 'date'], columns='Parameter',
                           values='Value', aggfunc='first').reset_index()
    merged = sim.merge(wide, on=['Site', 'date'], how='left').query('EC > 0 and OC > 0')
    merged['OC_EC'] = merged['OC'] / merged['EC']
    return merged['OC_EC'].to_numpy()


def fig_filtering_strip():
    """Full pool OC/EC distribution + where each calibration cohort concentrates."""
    ocec = pool_ocec()
    cohort = pd.read_csv(ROOT / 'output/tables/ftir11/lowest_ocec_800_cohort.csv')
    addis_ocec = 1.0  # Addis TOR OC/EC ~1 (below the entire IMPROVE range; meeting result)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    bins = np.geomspace(max(ocec.min(), .02), ocec.max(), 80)
    ax.hist(ocec, bins=bins, color=GREY, alpha=.7,
            label=f'Full IMPROVE pool  (n = {len(ocec):,})')
    ax.hist(cohort['OC_EC_ratio'], bins=bins, color=PURPLE, alpha=.85,
            label='Lowest-OC/EC cohort  (n = 800)')
    cut = cohort['OC_EC_ratio'].max()
    ax.axvline(cut, color=PURPLE, lw=1.3, ls='--')
    ax.annotate(f'filter cut\nOC/EC ≤ {cut:.2f}', (cut, ax.get_ylim()[1] * .82),
                xytext=(8, 0), textcoords='offset points', color=PURPLE, fontsize=9,
                va='center')
    ax.axvspan(bins[0], addis_ocec, color=ACCENT, alpha=.07)
    ax.annotate('Addis composition\n(OC/EC ≈ 1, below the\nentire IMPROVE range)',
                (addis_ocec, ax.get_ylim()[1] * .5), xytext=(-6, 0),
                textcoords='offset points', color=ACCENT, fontsize=9, ha='right', va='center')
    ax.set_xscale('log')
    ax.set_xlabel('TOR OC/EC ratio  (log scale)')
    ax.set_ylabel('IMPROVE filters')
    ax.set_title('Filtering the calibration by composition pulls it toward Addis',
                 fontsize=13, fontweight='bold', color=INK, loc='left')
    ax.legend(frameon=False, fontsize=9.5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'filtering_by_ocec.png', dpi=190, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def fig_setup_matrix():
    """A schematic 'how each calibration was built' matrix figure."""
    setups = [
        ('Deployed SPARTAN', 'Whole IMPROVE network', '~13,000', 'none — 1 model for all sites',
         '−4.17', GREY),
        ('Biomass-smoke', 'Katie-George smoke ratios', '906', 'wildfire days only',
         '−6.91', GREY),
        ('Ethiopia-shaped smoke', 'nearest Addis in CH /\ncarbonyl / 1600 features', '300',
         'spectral shape', '−3.69', BLUE),
        ('Spectral analogs', 'Mahalanobis + Q +\nVIP-weighted mismatch', '400', 'score-space distance',
         'fails TOR test', BLUE),
        ('Lowest-OC/EC', 'lowest TOR OC/EC ratio', '800', 'composition (OC/EC ≤ 2.27)',
         '−3.22', PURPLE),
        ('Lowest-OC/EC + AIRSpec', 'same, on baselined spectra', '800',
         'composition + baseline', '−1.62', ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 100)
    ax.axis('off')
    cols_x = [2, 26, 49, 60, 84]
    headers = ['Calibration setup', 'Filter applied to the pool', 'Samples',
               'Selection axis', 'Addis intercept']
    ax.text(0, 96, 'Every setup is the same PLS math on a differently-filtered training set',
            fontsize=14, fontweight='bold', color=INK)
    for x, head in zip(cols_x, headers):
        ax.text(x, 88, head.upper(), fontsize=8.5, color=MUTED, fontweight='bold')
    ax.plot([0, 100], [85, 85], color='#DDDDDD', lw=1)
    row_h = 13.2
    box_h = 11.0
    for i, (name, filt, n, axis, intercept, colour) in enumerate(setups):
        y = 77 - i * row_h                      # vertical center of the row
        box = FancyBboxPatch((0.5, y - box_h / 2), 99, box_h,
                             boxstyle='round,pad=0.2,rounding_size=1.5',
                             linewidth=0, facecolor=colour, alpha=.09)
        ax.add_patch(box)
        ax.plot([1.4, 1.4], [y - box_h / 2 + 1, y + box_h / 2 - 1], color=colour, lw=3,
                solid_capstyle='round')
        ax.text(cols_x[0], y, name, fontsize=11, fontweight='bold', color=INK, va='center')
        ax.text(cols_x[1], y, filt, fontsize=9, color=MUTED, va='center')
        ax.text(cols_x[2] + 4, y, n, fontsize=10.5, color=INK, va='center', ha='right',
                fontfamily='monospace')
        ax.text(cols_x[3], y, axis, fontsize=9, color=MUTED, va='center')
        weight = 'bold' if colour in (PURPLE, ACCENT) else 'normal'
        col = colour if colour in (PURPLE, ACCENT) else INK
        ax.text(cols_x[4] + 8, y, intercept, fontsize=11, color=col, va='center',
                ha='right', fontweight=weight, fontfamily='monospace')
    fig.tight_layout()
    fig.savefig(OUT / 'calibration_setup_matrix.png', dpi=190, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def airspec_spectra():
    """Raw spectra + fitted AIRSpec baselines for Addis and the lowest-OC/EC cohort.

    Cached under output/corrected/ because baselining ~1,000 spectra takes ~30 s.
    """
    cache = ROOT / 'output/corrected/deck_airspec_explainer.npz'
    if cache.exists():
        return dict(np.load(cache))

    from phase3_common import load_addis_evaluation
    from airspec_baseline import airspec_baseline_matrix

    addis_eval, X_addis, wn = load_addis_evaluation()
    wcols = addis_eval.attrs['wcols']
    cohort = pd.read_csv(ROOT / 'output/tables/ftir11/lowest_ocec_800_cohort.csv')
    pool = pd.read_csv(PATHS.ftir_dir / 'local_db/spectra_248_251.csv',
                       dtype={c: np.float32 for c in wcols}).set_index('AnalysisId')
    pool = pool[~pool.index.duplicated()]
    ids = [a for a in cohort['AnalysisId'].astype(int) if a in pool.index]
    X_cohort = pool.loc[ids, wcols].to_numpy(float)

    base_addis, corr_addis = airspec_baseline_matrix(wn, X_addis, 6, 4, n_jobs=6)
    base_cohort, _ = airspec_baseline_matrix(wn, X_cohort, 6, 4, n_jobs=6)
    # float32 and only the arrays the figure reads (the cohort enters via one band only).
    data = dict(wn=wn, X_addis=X_addis.astype(np.float32),
                base_addis=base_addis.astype(np.float32),
                corr_addis=corr_addis.astype(np.float32),
                base_cohort=base_cohort.astype(np.float32))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    return data


def fig_airspec_explainer():
    """What the second half of 'Lowest-OC/EC + AIRSpec' actually does to a spectrum."""
    d = airspec_spectra()
    wn = d['wn']
    valid = np.isfinite(d['base_addis'][0])          # strict 1425-4000 baseline domain
    ch = int(np.argmin(np.abs(wn - 2920)))

    # A representative Addis filter: median corrected CH-band height.
    ch_heights = d['corr_addis'][:, ch]
    pick = int(np.argsort(ch_heights)[len(ch_heights) // 2])
    raw, base, corr = (d['X_addis'][pick], d['base_addis'][pick], d['corr_addis'][pick])
    share = base[ch] / raw[ch]

    fig = plt.figure(figsize=(12.4, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=.42, wspace=.22)
    ax_raw = fig.add_subplot(gs[0, :])
    ax_corr = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    # (a) the raw spectrum and the baseline that gets subtracted
    ax_raw.fill_between(wn[valid], base[valid], raw[valid], color=ACCENT, alpha=.13,
                        lw=0, label='removed as baseline')
    ax_raw.plot(wn[valid], raw[valid], color=INK, lw=1.5, label='raw Addis spectrum')
    ax_raw.plot(wn[valid], base[valid], color=ACCENT, lw=1.6, ls='--',
                label='AIRSpec baseline (df1 = 6)')
    ax_raw.annotate(f'{share:.0%} of the raw absorbance\nat the CH band is baseline',
                    xy=(2920, base[ch]), xytext=(2300, raw[ch] * 1.12),
                    fontsize=9.5, color=ACCENT, ha='left', va='center',
                    arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.1,
                                    connectionstyle='arc3,rad=-.18'))
    ax_raw.set_ylabel('Absorbance')
    ax_raw.set_title('AIRSpec fits a smooth baseline under each spectrum and subtracts it',
                     fontsize=12, fontweight='bold', color=INK, loc='left')
    ax_raw.legend(frameon=False, fontsize=9.5, loc='lower left')

    # (b) what the calibration actually sees afterwards
    BANDS = [(3200, 'broad O–H / N–H', (0, 10), 'center'),
             (2920, 'CH', (14, 12), 'left'),
             (1720, 'carbonyl', (-2, 24), 'center'),
             (1620, '1600 band', (6, 7), 'center')]
    ax_corr.axhline(0, color=MUTED, lw=.9)
    ax_corr.plot(wn[valid], corr[valid], color=PURPLE, lw=1.5)
    for centre, label, offset, align in BANDS:
        j = int(np.argmin(np.abs(wn - centre)))
        ax_corr.annotate(label, xy=(centre, corr[j]), xytext=offset,
                         textcoords='offset points', ha=align, fontsize=8.5,
                         color=MUTED)
    ax_corr.set_ylabel('Corrected absorbance')
    ax_corr.set_title('What is left: band absorbance on a zero baseline',
                      fontsize=11, fontweight='bold', color=INK, loc='left')
    ax_corr.annotate(f'note the scale — this panel spans\n{corr[valid].max():.2f} '
                     f'absorbance, a tenth of the one above',
                     xy=(.97, .93), xycoords='axes fraction', ha='right', va='top',
                     fontsize=8.5, color=MUTED, style='italic')

    for ax in (ax_raw, ax_corr):
        ax.set_xlim(4000, 1390)                      # IR convention: descending
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    # (c) why it matters: Addis sits on a systematically higher background
    addis_base, cohort_base = d['base_addis'][:, ch], d['base_cohort'][:, ch]
    bins = np.linspace(0, float(np.percentile(np.r_[addis_base, cohort_base], 99.5)), 46)
    ax_hist.hist(cohort_base, bins=bins, color=PURPLE, alpha=.75,
                 label=f'Lowest-OC/EC cohort  (n = {len(cohort_base)})')
    ax_hist.hist(addis_base, bins=bins, color=ACCENT, alpha=.75,
                 label=f'Addis  (n = {len(addis_base)})')
    for values, colour in ((cohort_base, PURPLE), (addis_base, ACCENT)):
        ax_hist.axvline(np.median(values), color=colour, lw=1.3, ls='--')
    ax_hist.annotate(f'median baseline {np.median(cohort_base):.3f} (cohort)\n'
                     f'vs {np.median(addis_base):.3f} (Addis)',
                     xy=(.97, .55), xycoords='axes fraction',
                     ha='right', va='top', fontsize=9, color=MUTED)
    ax_hist.set_xlabel('Baseline absorbance at the CH band (2920 cm⁻¹)')
    ax_hist.set_ylabel('Filters')
    ax_hist.set_title('Addis rides a higher background than its calibration cohort',
                      fontsize=11, fontweight='bold', color=INK, loc='left')
    ax_hist.legend(fontsize=9, frameon=True, facecolor='white', edgecolor='none',
                   framealpha=.92)
    for spine in ('top', 'right'):
        ax_hist.spines[spine].set_visible(False)

    fig.suptitle('The "+ AIRSpec" half: a raw-spectra model can fit the background — '
                 'baselining leaves only the bands\n'
                 'removing it moves the Addis intercept from −3.22 to −1.62 µg/m³ '
                 '(and the slope from 1.59 to 0.86 at MAC = 10)',
                 fontsize=13, fontweight='bold', color=INK, y=1.035)
    fig.savefig(OUT / 'airspec_explainer.png', dpi=190, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def fig_intercept_ladder():
    """Addis intercept across setups — how filtering walks it toward zero."""
    m = pd.read_csv(ROOT / 'output/tables/ftir11/addis_metrics.csv')
    m = m[(m['MAC_m2_g'] == 10) & (m['cohort'] == 'fixed phase-2 cohort')]
    corrected = pd.read_csv(ROOT / 'output/tables/ftir13/addis_metrics_corrected.csv')
    corrected = corrected[(corrected['MAC_m2_g'] == 10)]

    def val(df, key, col='intercept'):
        row = df[df['model'].str.contains(key, case=False, na=False)]
        return float(row[col].iloc[0]) if len(row) else np.nan

    rows = [
        ('Deployed SPARTAN', -4.17, GREY),
        ('Biomass-smoke (906)', -6.91, GREY),
        ('Ethiopia-shaped smoke', -3.69, BLUE),
        ('Lowest-OC/EC (800)', val(m, 'lowest-OCEC 800'), PURPLE),
        ('Lowest-OC/EC + AIRSpec', -1.62, ACCENT),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colours = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    y = np.arange(len(rows))[::-1]
    ax.hlines(y, vals, 0, color=colours, lw=6, alpha=.35, capstyle='round')
    ax.scatter(vals, y, color=colours, s=130, zorder=3)
    for yi, (lab, v, c) in zip(y, rows):
        ax.text(v - .15, yi, f'{v:+.2f}', va='center', ha='right', fontsize=10.5,
                color=c, fontweight='bold', fontfamily='monospace')
        ax.text(.15, yi, lab, va='center', ha='left', fontsize=10.5, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.text(0, len(rows) - .35, 'target\n(zero)', fontsize=8.5, color=MUTED, ha='center')
    ax.set_yticks([])
    ax.set_xlim(-7.6, 3.2)
    ax.set_xlabel('Addis EC intercept  (µg/m³, MAC = 10) — closer to zero is better')
    ax.set_title('Each filtering step walks the intercept toward zero',
                 fontsize=13, fontweight='bold', color=INK, loc='left')
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'intercept_ladder.png', dpi=190, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    fig_setup_matrix()
    fig_filtering_strip()
    fig_airspec_explainer()
    fig_intercept_ladder()
    print('wrote', *[p.name for p in sorted(OUT.glob('*.png'))])
