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


def _mode_intercepts():
    """Addis intercepts per cohort under both protocols, from ftir_21's committed table.

    Returns {cohort: {'app': float, 'site_heldout': float}} at MAC = 10 on the fixed
    190-filter cohort. Absent if ftir_21 has not been run yet.
    """
    path = ROOT / 'output/tables/ftir21/addis_metrics_by_mode.csv'
    if not path.exists():
        return {}
    m = pd.read_csv(path)
    m = m[(m['MAC_m2_g'] == 10) & (m['evaluation_set'] == 'fixed phase-2 cohort')]
    return {cohort: dict(zip(rows['mode'], rows['intercept']))
            for cohort, rows in m.groupby('cohort')}


def fig_setup_matrix():
    """A schematic 'how each calibration was built' matrix figure.

    The Addis intercept is protocol-conditional (ftir_21), so the matrix carries **both**
    columns: the Calibration app's protocol and the site-held-out protocol. Row 1 is
    labelled as the entire IMPROVE network, which is what the deployed model is trained on;
    the only SPARTAN spectra in this project are ETAD's and they are evaluation-only.
    """
    by_mode = _mode_intercepts()

    def pair(cohort, fallback_app, fallback_heldout):
        got = by_mode.get(cohort, {})
        app = got.get('app')
        heldout = got.get('site_heldout')
        return (f'{app:.2f}'.replace('-', '−') if app is not None else fallback_app,
                f'{heldout:.2f}'.replace('-', '−') if heldout is not None
                else fallback_heldout)

    setups = [
        ('Entire IMPROVE network', 'no selection at all', '13,010',
         'none — 1 model for all sites',
         *pair('Entire IMPROVE network (13,010, no selection)', '−4.05', '−3.44'), GREY),
        ('Biomass-smoke', 'Katie-George smoke ratios', '906', 'wildfire days only',
         *pair('Biomass-smoke (906)', '−6.35', '−0.99'), GREY),
        ('Ethiopia-shaped smoke', 'nearest Addis in CH /\ncarbonyl / 1600 features', '300',
         'spectral shape  ⚠ fails TOR',
         *pair('Ethiopia-shaped smoke (300)', '−3.44', '−3.67'), BLUE),
        ('Spectral analogs', 'Mahalanobis + Q +\nVIP-weighted mismatch', '500',
         'score-space  ⚠ fails TOR',
         *pair('Spectral analogs (locked 500)', '−6.12', '−6.35'), BLUE),
        ('Lowest-OC/EC', 'lowest TOR OC/EC ratio', '800', 'composition (OC/EC ≤ 2.27)',
         *pair('Lowest-OC/EC (800)', '−4.59', '−3.22'), PURPLE),
        ('Lowest-OC/EC + AIRSpec', 'same, on baselined spectra', '800',
         'composition + baseline',
         *pair('Lowest-OC/EC + AIRSpec (800)', '−1.65', '−1.62'), ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(-9, 100)
    ax.axis('off')
    cols_x = [2, 24, 45, 55, 80, 93]
    headers = ['Calibration setup', 'Filter applied to the pool', 'Samples',
               'Selection axis', 'Calibration app', 'Site-held-out']
    ax.text(0, 97, 'Every setup is the same PLS math on a differently-filtered training set',
            fontsize=14, fontweight='bold', color=INK)
    ax.text(0, 91.5, 'Addis intercept depends on the calibration protocol as well as the '
                     'cohort — both are shown (MAC = 10)',
            fontsize=9.5, color=MUTED)
    for x, head in zip(cols_x, headers):
        ax.text(x, 85, head.upper(), fontsize=8.5, color=MUTED, fontweight='bold',
                ha='right' if x >= 80 else 'left')
    ax.text(86.5, 90, 'ADDIS INTERCEPT', fontsize=8.5, color=INK, fontweight='bold',
            ha='center')
    ax.plot([78, 95], [88, 88], color='#CCCCCC', lw=.8)
    ax.plot([0, 100], [82, 82], color='#DDDDDD', lw=1)
    row_h = 13.2
    box_h = 11.0
    for i, (name, filt, n, axis, app_val, heldout_val, colour) in enumerate(setups):
        y = 74 - i * row_h                      # vertical center of the row
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
        emphasised = colour in (PURPLE, ACCENT)
        ax.text(cols_x[4], y, app_val, fontsize=10.5, color=MUTED, va='center',
                ha='right', fontfamily='monospace')
        ax.text(cols_x[5], y, heldout_val, fontsize=11,
                color=colour if emphasised else INK, va='center', ha='right',
                fontweight='bold' if emphasised else 'normal', fontfamily='monospace')
    ax.text(0, -6, 'Intercepts from ftir_21 (fixed 190-filter cohort). ⚠ = no held-out TOR '
                   'skill once whole sites are held out.', fontsize=8.5, color=MUTED)
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


def _airspec_representative():
    """Shared inputs for the three AIRSpec slides.

    The displayed filter is one real spectrum, never an average — AIRSpec fits a
    baseline to each spectrum individually. "Representative" is defined explicitly:
    the filter whose worst-case percentile across the four labelled bands *and* its
    baseline height is closest to the Addis median. (Ranking on CH alone is not
    enough: the median-CH filter sits at the 92nd percentile of O–H.)
    """
    d = airspec_spectra()
    wn = d['wn']
    valid = np.isfinite(d['base_addis'][0])          # strict 1425-4000 baseline domain
    ch = int(np.argmin(np.abs(wn - 2920)))
    corr, base = d['corr_addis'], d['base_addis']
    diagnostics = [corr[:, int(np.argmin(np.abs(wn - centre)))].astype(float)
                   for centre in (3200, 2920, 1720, 1620)] + [base[:, ch].astype(float)]

    def percentiles(values):                          # average-rank percentile, 0-100
        order = values.argsort().argsort() + .5
        return order / len(values) * 100

    ranks = np.column_stack([percentiles(v) for v in diagnostics])
    worst = np.abs(ranks - 50).max(axis=1)
    pick = int(np.argmin(worst))
    return d, wn, valid, ch, pick, float(worst[pick])


def _spectrum_axes(ax):
    ax.set_xlim(4000, 1390)                          # IR convention: descending
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def fig_airspec_1_baseline():
    """Slide 1: what the AIRSpec correction removes from a raw spectrum."""
    d, wn, valid, ch, pick, spread = _airspec_representative()
    raw, base = d['X_addis'][pick], d['base_addis'][pick]
    share = base[ch] / raw[ch]
    n_addis = len(d['X_addis'])

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    # Every Addis spectrum faintly behind, so the highlighted one is visibly one of many.
    ax.plot(wn[valid], d['X_addis'][:, valid].T, color=MUTED, lw=.5, alpha=.06)
    ax.plot([], [], color=MUTED, lw=1, alpha=.45,
            label=f'the other {n_addis - 1} Addis spectra')
    ax.fill_between(wn[valid], base[valid], raw[valid], color=ACCENT, alpha=.13, lw=0,
                    label='removed as baseline')
    ax.plot(wn[valid], raw[valid], color=INK, lw=1.6, label='this raw Addis spectrum')
    ax.plot(wn[valid], base[valid], color=ACCENT, lw=1.7, ls='--',
            label='its AIRSpec baseline (df1 = 6)')
    ax.annotate(f'{share:.0%} of the raw absorbance\nat the CH band is baseline',
                xy=(2920, base[ch]), xytext=(2300, raw[ch] * 1.12),
                fontsize=10.5, color=ACCENT, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.2,
                                connectionstyle='arc3,rad=-.18'))
    ax.annotate(f'One real filter, not an average — the baseline is fit to each\n'
                f'spectrum separately. This one is within {spread:.0f} percentile points '
                f'of the\nAddis median on every labelled band and on baseline height.',
                xy=(.985, .97), xycoords='axes fraction', ha='right', va='top',
                fontsize=9, color=MUTED, style='italic')
    ax.set_ylabel('Absorbance')
    ax.legend(frameon=False, fontsize=10, loc='lower left')
    _spectrum_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / 'airspec_1_baseline.png', dpi=190, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def fig_airspec_2_corrected():
    """Slide 2: the band absorbance that survives the correction."""
    d, wn, valid, ch, pick, spread = _airspec_representative()
    corr = d['corr_addis'][pick]
    raw_peak = float(d['X_addis'][pick][valid].max())

    BANDS = [(3200, 'broad O–H / N–H', (0, 10), 'center'),
             (2920, 'CH', (14, 12), 'left'),
             (1720, 'carbonyl', (-6, 22), 'center'),
             (1620, '1600 band', (26, -12), 'left')]
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    ax.axhline(0, color=MUTED, lw=.9)
    ax.plot(wn[valid], corr[valid], color=PURPLE, lw=1.7)
    for centre, label, offset, align in BANDS:
        j = int(np.argmin(np.abs(wn - centre)))
        ax.annotate(label, xy=(centre, corr[j]), xytext=offset,
                    textcoords='offset points', ha=align, fontsize=10, color=MUTED)
    ax.set_ylabel('Corrected absorbance')
    ax.annotate(f'note the scale: this whole plot spans {corr[valid].max():.2f} '
                f'absorbance —\nthe raw spectrum reached {raw_peak:.2f}',
                xy=(.97, .93), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, color=MUTED, style='italic')
    ax.annotate(f'same single filter as the previous slide — within {spread:.0f} '
                f'percentile points\nof the Addis median on each band labelled here',
                xy=(.97, .70), xycoords='axes fraction', ha='right', va='top',
                fontsize=9, color=MUTED, style='italic')
    _spectrum_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / 'airspec_2_corrected.png', dpi=190, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def fig_airspec_3_background_gap():
    """Slide 3: why removing the background changes the Addis answer."""
    d, wn, valid, ch, _, _ = _airspec_representative()
    addis_base, cohort_base = d['base_addis'][:, ch], d['base_cohort'][:, ch]

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    bins = np.linspace(0, float(np.percentile(np.r_[addis_base, cohort_base], 99.5)), 46)
    ax.hist(cohort_base, bins=bins, color=PURPLE, alpha=.75,
            label=f'Lowest-OC/EC calibration cohort  (n = {len(cohort_base)})')
    ax.hist(addis_base, bins=bins, color=ACCENT, alpha=.75,
            label=f'Addis  (n = {len(addis_base)})')
    for values, colour, name in ((cohort_base, PURPLE, 'cohort'),
                                 (addis_base, ACCENT, 'Addis')):
        median = float(np.median(values))
        ax.axvline(median, color=colour, lw=1.4, ls='--')
        ax.annotate(f'{name} median {median:.3f}', xy=(median, ax.get_ylim()[1] * .96),
                    xytext=(6, 0), textcoords='offset points', color=colour, fontsize=10,
                    va='top')
    ax.set_xlabel('AIRSpec baseline absorbance at the CH band (2920 cm⁻¹)')
    ax.set_ylabel('Filters')
    ax.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='none',
              framealpha=.92, loc='center right')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'airspec_3_background_gap.png', dpi=190, bbox_inches='tight',
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
    fig_airspec_1_baseline()
    fig_airspec_2_corrected()
    fig_airspec_3_background_gap()
    fig_intercept_ladder()
    print('wrote', *[p.name for p in sorted(OUT.glob('*.png'))])
