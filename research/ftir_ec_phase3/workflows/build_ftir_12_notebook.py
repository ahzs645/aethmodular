#!/usr/bin/env python3
"""Build the ftir_12 ~1600 cm-1 band identity notebook (phase 3)."""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ftir_12_band_1600_identity.ipynb"


def build_notebook():
    cells = []

    def md(text):
        cells.append(new_markdown_cell(text.strip()))

    def code(text):
        cells.append(new_code_cell(text.strip()))

    code(
        """
import sys
# Phase-3 notebooks reuse the phase-2 modules in research/ftir_hips_chem/scripts.
sys.path.insert(0, '../ftir_hips_chem/scripts')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

from config import ETHIOPIA_SEASONS, season_for_month
from plotting import PlotConfig
from pls_transfer import (
    FTIRTransferPaths, load_current_pls_model,
    local_continuum_peak_height, offset_correct,
)

PlotConfig.set(sites='all', layout='individual', show_stats=True, show_1to1=True)
PATHS = FTIRTransferPaths.defaults()
P3_TABLES = Path('output/tables/ftir_12')
P3_PLOTS = Path('output/plots/ftir_12')
for directory in (P3_TABLES, P3_PLOTS):
    directory.mkdir(parents=True, exist_ok=True)
"""
    )

    md(
        """
# What is the elevated ~1600 cm⁻¹ band in Addis spectra: carboxylate, amine, or nitro?

## tl;dr

TLDR_PLACEHOLDER

## Context & Methods

At the July 2026 meeting Satoshi noted the ~1600 cm⁻¹ feature that is relatively stronger in
Addis than in IMPROVE smoke could be **carboxylate** (COO⁻ asymmetric stretch, typically
1570–1610 cm⁻¹, with a symmetric partner near 1400 cm⁻¹ and *no* accompanying N–H stretch),
**amine/ammonium** (N–H bend 1590–1650 cm⁻¹, accompanied by N–H stretch absorption in
3100–3400 cm⁻¹), or **nitro/aromatic** groups near 1500–1560 cm⁻¹. This notebook applies three
diagnostics that only need band arithmetic:

1. **Peak-center statistics** of the continuum-subtracted 1550–1650 cm⁻¹ feature per group;
2. **Partner-band co-variation** of the 1600 band against the ~1400 partner, the 3100–3400
   N–H/O–H window, carbonyl, and CH;
3. **Median-spectrum overlays** in the 1300–1900 and 2600–3600 cm⁻¹ windows.

Groups: Addis (239 filters), the current 906-filter IMPROVE smoke calibration cohort, the
`ftir_11` lowest-OC/EC cohort (800), and a random 800-filter IMPROVE pool sample.

### Key Assumptions

- Spectra are raw Teflon spectra with only a constant 3900–4000 cm⁻¹ offset correction for
  overlays; peak heights use local linear continua, which is robust to broad baseline drift
  but **not** a substitute for AIRSpec baselining.
- The ~1400 cm⁻¹ symmetric-carboxylate window (1380–1440) and the 1515–1560 nitro window sit
  at or below the region the lab normally trusts (>1500 cm⁻¹, AIRSpec segment 2 ends at
  1425); results there are directional only.
- Band assignments are generic organic-aerosol ranges, not compound identifications.
"""
    )

    md("## Data\n\n### 1. Load the four spectral groups")

    code(
        """
raw_etad = pd.read_csv(PATHS.etad_dir / 'ETAD_FTIR_spectra.csv')
wcols = sorted([c for c in raw_etad.columns if c not in ('SampleAnalysisId', 'MediaId')],
               key=lambda value: -float(value))
wn = np.array([float(c) for c in wcols])
X_addis = raw_etad.groupby('MediaId', as_index=False)[wcols].mean()[wcols].to_numpy(float)
X_addis = X_addis[~np.isnan(X_addis).any(axis=1)]

current = load_current_pls_model(PATHS.ftir_dir, 'EC')
assert np.allclose(current.wavenumbers, wn)
X_smoke = current.X

ocec_split = pd.read_csv('output/tables/ftir_11/locked_ocec_train_test_split.csv')
similarity = pd.read_csv(
    '../ftir_hips_chem/output/tables/pls_transfer/improve_full_pool_addis_similarity.csv')
rng = np.random.default_rng(7)
random_ids = set(int(v) for v in similarity['AnalysisId'].sample(
    n=800, random_state=7))
ocec_ids = set(int(v) for v in ocec_split['AnalysisId'])

needed = ocec_ids | random_ids
parts = []
for chunk in pd.read_csv(PATHS.ftir_dir / 'local_db/spectra_248_251.csv', chunksize=750):
    keep = chunk['AnalysisId'].astype(int).isin(needed)
    if keep.any():
        parts.append(chunk.loc[keep, ['AnalysisId'] + wcols])
pool_spectra = pd.concat(parts, ignore_index=True).drop_duplicates('AnalysisId')
pool_ids = pool_spectra['AnalysisId'].astype(int)
X_ocec = pool_spectra.loc[pool_ids.isin(ocec_ids), wcols].to_numpy(float)
X_random = pool_spectra.loc[pool_ids.isin(random_ids), wcols].to_numpy(float)

groups = {
    'Addis': X_addis,
    'IMPROVE smoke (906)': X_smoke,
    'Lowest-OC/EC cohort (800)': X_ocec,
    'Random IMPROVE pool (800)': X_random,
}
group_audit = pd.DataFrame([{'group': name, 'n_spectra': len(X)}
                            for name, X in groups.items()])
group_audit.to_csv(P3_TABLES / 'group_audit.csv', index=False)
display(group_audit)
"""
    )

    md("### 2. Band features and 1550–1650 cm⁻¹ peak centers")

    code(
        """
BANDS = {
    # name: (peak_band, left_continuum, right_continuum)
    'band1600': ((1550, 1650), (1750, 1800), (1500, 1530)),
    'carbonyl': ((1650, 1775), (1800, 1900), (1500, 1550)),
    'CH': ((2800, 3000), (3050, 3150), (2650, 2750)),
    'NH_OH_3100_3400': ((3100, 3400), (3500, 3600), (2650, 2750)),
    'sym1400_partner': ((1380, 1440), (1470, 1500), (1330, 1360)),
    'nitro_1520_1560': ((1515, 1560), (1570, 1600), (1490, 1510)),
}

def peak_center_1600(X):
    peak_mask = (wn >= 1550) & (wn <= 1650)
    left_mask = (wn >= 1750) & (wn <= 1800)
    right_mask = (wn >= 1500) & (wn <= 1530)
    left_x, right_x = wn[left_mask].mean(), wn[right_mask].mean()
    left_y = X[:, left_mask].mean(axis=1)
    right_y = X[:, right_mask].mean(axis=1)
    slope = (right_y - left_y) / (right_x - left_x)
    continuum = left_y[:, None] + slope[:, None] * (wn[peak_mask][None, :] - left_x)
    excess = X[:, peak_mask] - continuum
    centers = wn[peak_mask][np.argmax(excess, axis=1)]
    heights = excess.max(axis=1)
    return centers, heights

features = {}
centers = {}
for name, X in groups.items():
    table = pd.DataFrame({
        band: local_continuum_peak_height(X, wn, *bounds)
        for band, bounds in BANDS.items()
    })
    c, h = peak_center_1600(X)
    table['peak_center_1600'] = c
    features[name] = table
    # Only trust peak centers when the feature clearly exists.
    centers[name] = c[h > np.nanpercentile(h, 25)]

summary = pd.concat(
    [table.describe().loc[['25%', '50%', '75%']].T.add_prefix(f'{name} ')
     for name, table in features.items()], axis=1)
summary.to_csv(P3_TABLES / 'band_feature_quantiles.csv')

center_stats = pd.DataFrame([{
    'group': name,
    'n_used': len(values),
    'center_p25': np.percentile(values, 25),
    'center_median': np.median(values),
    'center_p75': np.percentile(values, 75),
} for name, values in centers.items()])
center_stats.to_csv(P3_TABLES / 'peak_center_1600_stats.csv', index=False)
display(center_stats)
u_stat, p_value = mannwhitneyu(centers['Addis'], centers['IMPROVE smoke (906)'])
print(f'Addis vs smoke peak-center Mann-Whitney p = {p_value:.2e}')
"""
    )

    md("## Results\n\n### 3. Partner-band co-variation within each group")

    code(
        """
PARTNERS = ['sym1400_partner', 'NH_OH_3100_3400', 'carbonyl', 'CH', 'nitro_1520_1560']
rows = []
for name, table in features.items():
    finite = table.dropna()
    for partner in PARTNERS:
        rows.append({
            'group': name,
            'partner_band': partner,
            'n': len(finite),
            'pearson_r': pearsonr(finite['band1600'], finite[partner]).statistic,
            'spearman_r': spearmanr(finite['band1600'], finite[partner]).statistic,
        })
covariation = pd.DataFrame(rows)
covariation.to_csv(P3_TABLES / 'band1600_covariation.csv', index=False)
display(covariation.pivot(index='partner_band', columns='group', values='spearman_r').round(3))
"""
    )

    md("### 4. Median-spectrum overlays in the diagnostic windows")

    code(
        """
COLORS = {'Addis': '#C0392B', 'IMPROVE smoke (906)': '#7F8C8D',
          'Lowest-OC/EC cohort (800)': '#8E44AD', 'Random IMPROVE pool (800)': '#BDC3C7'}
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
for name, X in groups.items():
    corrected = offset_correct(X, wn)
    median = np.nanmedian(corrected, axis=0)
    scale = np.nanmax(median[(wn >= 1500) & (wn <= 1900)])
    for ax, (low, high) in zip(axes, [(1300, 1900), (2600, 3600)]):
        window = (wn >= low) & (wn <= high)
        ax.plot(wn[window], median[window] / scale, lw=1.4,
                color=COLORS[name], label=name)
for ax, (low, high) in zip(axes, [(1300, 1900), (2600, 3600)]):
    ax.set_xlim(high, low)
    ax.set_xlabel('Wavenumber (cm⁻¹)')
axes[0].axvspan(1550, 1650, color='#F1C40F', alpha=.12)
axes[0].axvspan(1380, 1440, color='#16A085', alpha=.10)
axes[0].annotate('~1600 band', (1600, axes[0].get_ylim()[1]*.95), ha='center', fontsize=8)
axes[0].annotate('~1400 partner\\n(below trusted range)', (1410, axes[0].get_ylim()[1]*.78),
                 ha='center', fontsize=7)
axes[1].axvspan(3100, 3400, color='#2980B9', alpha=.10)
axes[1].annotate('N-H / O-H stretch window', (3250, axes[1].get_ylim()[1]*.95),
                 ha='center', fontsize=8)
axes[0].set_ylabel('Median absorbance (normalized to 1500–1900 max)')
axes[0].legend(fontsize=8)
fig.suptitle('Diagnostic windows: normalized median spectra by group', y=1.02)
fig.tight_layout()
fig.savefig(P3_PLOTS / 'median_spectra_diagnostic_windows.png', dpi=180, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8.0, 4.4))
positions = np.arange(len(centers))
ax.violinplot([centers[name] for name in centers], positions=positions,
              showmedians=True, widths=.8)
ax.set_xticks(positions, [name.replace(' (', '\\n(') for name in centers], fontsize=8)
ax.axhspan(1570, 1610, color='#16A085', alpha=.10)
ax.text(len(centers) - .5, 1590, 'carboxylate\\nCOO⁻ asym', fontsize=7,
        ha='right', color='#16A085')
ax.axhspan(1610, 1650, color='#2980B9', alpha=.08)
ax.text(len(centers) - .5, 1630, 'amine N-H bend /\\naromatic', fontsize=7,
        ha='right', color='#2980B9')
ax.set(ylabel='1550–1650 cm⁻¹ peak center (cm⁻¹)',
       title='Where does the ~1600 band actually peak?')
fig.tight_layout()
fig.savefig(P3_PLOTS / 'peak_center_1600_by_group.png', dpi=180, bbox_inches='tight')
plt.show()
"""
    )

    md("### 5. Ratio context: how anomalous is the Addis 1600 band?")

    code(
        """
ratio_rows = []
for name, table in features.items():
    ch = table['CH'].where(table['CH'] > 1e-4)
    carbonyl = table['carbonyl'].where(table['carbonyl'] > 1e-4)
    ratio_rows.append({
        'group': name,
        'band1600_to_CH_median': float((table['band1600'] / ch).median()),
        'band1600_to_carbonyl_median': float((table['band1600'] / carbonyl).median()),
        'NH3100_3400_to_band1600_median': float(
            (table['NH_OH_3100_3400'] / table['band1600'].where(table['band1600'] > 1e-4))
            .median()),
    })
ratios = pd.DataFrame(ratio_rows)
ratios.to_csv(P3_TABLES / 'band1600_ratio_context.csv', index=False)
display(ratios)
"""
    )

    md("## Takeaways\n\nTAKEAWAYS_PLACEHOLDER")

    notebook = new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    })
    nbf.write(notebook, OUTPUT)
    print(f'Wrote {OUTPUT}')


if __name__ == '__main__':
    build_notebook()
