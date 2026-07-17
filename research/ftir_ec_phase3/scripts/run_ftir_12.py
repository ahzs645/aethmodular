# %% [markdown]
# # ftir_12 — What is the ~1600 cm⁻¹ band in Addis: carboxylate, amine, nitro, or water?
#
# ## tl;dr
#
# (filled in by the finalize step after execution)
#
# ## Context & Methods
#
# The meeting left the identity of the elevated ~1600 cm⁻¹ feature in Addis spectra open:
# Satoshi suggested **carboxylate** (asymmetric COO⁻, ~1570–1620, with a symmetric partner near
# 1400 and no N–H companion), against **amine** (~1620–1650 bend, with N–H stretch 3100–3400),
# **nitro** (asym ~1520–1560), or the always-present **liquid-water bend** (~1635).
#
# Diagnostics on offset-corrected raw spectra (3900–4000 cm⁻¹ mean subtracted — phase-2
# convention, *not* AIRSpec baselining):
#
# 1. **Peak-center distribution** of the 1550–1680 cm⁻¹ local-continuum maximum, per group.
# 2. **Covariation** of the 1600-band height with candidate companion bands.
# 3. **Ratio context** (1600/CH, 1600/carbonyl) across Addis, the 906-sample smoke cohort, the
#    lowest-OCEC 800 cohort (ftir_11), and a random 800-sample pool draw.
# 4. **Addis-only covariation with HIPS Fabs and deployed FTIR EC** — does the band track the
#    absorbing material?
#
# Sub-1500 cm⁻¹ windows (symmetric-COO⁻ 1380–1420, nitro 1520–1560 partially) sit in
# Teflon-dominated, un-baselined territory: those two rows are indicative only and are revisited
# on AIRSpec-corrected spectra in ftir_13.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path('scripts').resolve()))
sys.path.insert(0, str((Path('..') / 'ftir_hips_chem' / 'scripts').resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import pearsonr, spearmanr

from phase3_common import PATHS, load_addis_evaluation, load_pool_metadata, load_pool_spectra
from config import season_for_month
from pls_transfer import (
    load_current_pls_model, offset_correct, local_continuum_peak_height,
    ftir_source_band_features,
)

TABLE_DIR = Path('output/tables/ftir12')
PLOT_DIR = Path('output/plots/ftir12')
for directory in (TABLE_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260717
BAND_WINDOW = (1550, 1680)
HEIGHT_FLOOR = 5e-4  # exclude spectra whose 1600 band is indistinguishable from noise

# %% [markdown]
# ## Data
#
# ### 1. Load the four spectra groups

# %%
etad_eval, X_etad_hips, wn = load_addis_evaluation(season_for_month)
wcols = etad_eval.attrs['wcols']

# All complete Addis spectra (not only HIPS-matched) for band statistics.
raw_etad = pd.read_csv(PATHS.etad_dir / 'ETAD_FTIR_spectra.csv')
etad_all = raw_etad.groupby('MediaId', as_index=False)[wcols].mean()
etad_all = etad_all[etad_all[wcols].notna().all(axis=1)]
X_addis = etad_all[wcols].to_numpy(float)

smoke = load_current_pls_model(PATHS.ftir_dir, 'EC')
assert np.allclose(smoke.wavenumbers, wn)

lowest_ocec = pd.read_csv('output/tables/ftir11/lowest_ocec_800_cohort.csv')
pool = load_pool_metadata()
rng = np.random.default_rng(RANDOM_SEED)
random_ids = pool['AnalysisId'].sample(n=800, random_state=RANDOM_SEED).astype(int).tolist()
spectra = load_pool_spectra(
    sorted(set(lowest_ocec['AnalysisId'].astype(int)) | set(random_ids)), wcols
).set_index('AnalysisId')

groups = {
    'Addis': X_addis,
    'IMPROVE smoke (906)': smoke.X,
    'Lowest-OCEC 800': spectra.loc[
        [i for i in lowest_ocec['AnalysisId'].astype(int) if i in spectra.index], wcols
    ].to_numpy(float),
    'Random pool 800': spectra.loc[
        [i for i in random_ids if i in spectra.index], wcols
    ].to_numpy(float),
}
group_audit = pd.DataFrame([{'group': k, 'n_spectra': len(v)} for k, v in groups.items()])
group_audit.to_csv(TABLE_DIR / 'group_audit.csv', index=False)
display(group_audit)

# %% [markdown]
# ### 2. Band features per group
#
# Heights are local-continuum peak heights on offset-corrected spectra; the peak *center* is the
# wavenumber of the maximum continuum-relative signal inside 1550–1680 cm⁻¹.

# %%
def band_center(X, wavenumbers, peak_band=BAND_WINDOW,
                left=(1750, 1800), right=(1500, 1530)):
    X = np.asarray(X, float)
    peak_mask = (wavenumbers >= peak_band[0]) & (wavenumbers <= peak_band[1])
    left_mask = (wavenumbers >= left[0]) & (wavenumbers <= left[1])
    right_mask = (wavenumbers >= right[0]) & (wavenumbers <= right[1])
    left_x, right_x = wavenumbers[left_mask].mean(), wavenumbers[right_mask].mean()
    left_y, right_y = X[:, left_mask].mean(axis=1), X[:, right_mask].mean(axis=1)
    slope = (right_y - left_y) / (right_x - left_x)
    continuum = left_y[:, None] + slope[:, None] * (wavenumbers[peak_mask][None, :] - left_x)
    relative = X[:, peak_mask] - continuum
    argmax = np.argmax(relative, axis=1)
    center = wavenumbers[peak_mask][argmax]
    height = relative.max(axis=1)
    # An argmax on the window boundary is a leaking neighbor band (e.g. the carbonyl
    # shoulder at the high edge), not a resolved 1600-band peak.
    edge_hit = (argmax <= 1) | (argmax >= peak_mask.sum() - 2)
    return center, height, edge_hit


def group_features(X):
    corrected = offset_correct(X, wn)
    features = ftir_source_band_features(corrected, wn)
    center, height_1600, edge_hit = band_center(corrected, wn)
    features['band1600_center'] = center
    features['band1600_height'] = height_1600
    features['band1600_center_is_edge_hit'] = edge_hit
    features['NH_OH_3100_3400'] = local_continuum_peak_height(
        corrected, wn, (3100, 3400), (3450, 3600), (2650, 2750))
    # Sub-1500 windows: un-baselined Teflon territory — indicative only.
    features['sym1400_partner'] = local_continuum_peak_height(
        corrected, wn, (1380, 1420), (1425, 1445), (1330, 1360))
    features['nitro_1520_1560'] = local_continuum_peak_height(
        corrected, wn, (1520, 1560), (1570, 1600), (1490, 1510))
    return features

features = {name: group_features(X) for name, X in groups.items()}

# %% [markdown]
# ## Results
#
# ### 3. Peak-center distributions: Addis sits below every IMPROVE cohort

# %%
center_rows = []
for name, frame in features.items():
    used = frame[frame['band1600_height'].gt(HEIGHT_FLOOR)
                 & ~frame['band1600_center_is_edge_hit']]
    center_rows.append({
        'group': name, 'n_total': len(frame), 'n_used': len(used),
        'n_edge_hits': int((frame['band1600_height'].gt(HEIGHT_FLOOR)
                            & frame['band1600_center_is_edge_hit']).sum()),
        'center_p25': used['band1600_center'].quantile(.25),
        'center_median': used['band1600_center'].median(),
        'center_p75': used['band1600_center'].quantile(.75),
    })
center_stats = pd.DataFrame(center_rows)
center_stats.to_csv(TABLE_DIR / 'peak_center_1600_stats.csv', index=False)
display(center_stats)

fig, ax = plt.subplots(figsize=(9.5, 5.2))
colors = {'Addis': '#C0392B', 'IMPROVE smoke (906)': '#7F8C8D',
          'Lowest-OCEC 800': '#27AE60', 'Random pool 800': '#2980B9'}
for name, frame in features.items():
    used = frame[frame['band1600_height'].gt(HEIGHT_FLOOR)
                 & ~frame['band1600_center_is_edge_hit']]
    ax.hist(used['band1600_center'], bins=np.arange(1550, 1682, 4), density=True,
            histtype='step', lw=1.9, color=colors[name],
            label=f'{name} (n={len(used)})')
for position, label in [(1585, 'COO⁻ asym'), (1635, 'H₂O bend'), (1650, 'amine/amide')]:
    ax.axvline(position, color='0.75', lw=1, ls=':')
    ax.text(position, ax.get_ylim()[1] * .97, label, rotation=90, va='top',
            ha='right', fontsize=8, color='0.4')
ax.set(xlabel='1600-band peak center (cm⁻¹)', ylabel='Density',
       title='Addis 1600-band peaks sit lower than any IMPROVE cohort')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'peak_center_1600_by_group.png', dpi=180, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4. Companion-band covariation

# %%
covariation_rows = []
partners = ['carbonyl_peak', 'CH_peak', 'NH_OH_3100_3400', 'sym1400_partner', 'nitro_1520_1560']
for name, frame in features.items():
    for partner in partners:
        valid = frame[['band1600_height', partner]].dropna()
        covariation_rows.append({
            'group': name, 'partner_band': partner, 'n': len(valid),
            'pearson_r': pearsonr(valid['band1600_height'], valid[partner])[0],
            'spearman_r': spearmanr(valid['band1600_height'], valid[partner])[0],
            'sub_1500_unreliable': partner in ('sym1400_partner', 'nitro_1520_1560'),
        })
covariation = pd.DataFrame(covariation_rows)
covariation.to_csv(TABLE_DIR / 'band1600_covariation.csv', index=False)
display(covariation[covariation['group'].eq('Addis')])

# %% [markdown]
# ### 5. Ratio context across groups

# %%
ratio_rows = []
for name, frame in features.items():
    ch = frame['CH_peak'].where(frame['CH_peak'] > 1e-4)
    carbonyl = frame['carbonyl_peak'].where(frame['carbonyl_peak'] > 1e-4)
    ratio_rows.append({
        'group': name,
        'band1600_to_CH_median': float((frame['band1600_height'] / ch).median()),
        'band1600_to_carbonyl_median': float((frame['band1600_height'] / carbonyl).median()),
        'NH_OH_to_band1600_median': float(
            (frame['NH_OH_3100_3400'] / frame['band1600_height']
             .where(frame['band1600_height'] > HEIGHT_FLOOR)).median()),
    })
ratios = pd.DataFrame(ratio_rows)
ratios.to_csv(TABLE_DIR / 'band1600_ratio_context.csv', index=False)
display(ratios)

# %% [markdown]
# ### 6. Does the Addis 1600 band track the absorbing material?

# %%
addis_features_hips = group_features(X_etad_hips)
addis_features_hips['Fabs'] = etad_eval['Fabs'].to_numpy(float)
addis_features_hips['EC_deployed_ugm3'] = etad_eval['EC_deployed_ugm3'].to_numpy(float)
addis_features_hips['volume_m3'] = etad_eval['SampleVolume_m3'].to_numpy(float)
# Put HIPS on a per-filter loading basis (tau) for a fair per-spectrum comparison.
addis_features_hips['HIPS_tau'] = (
    addis_features_hips['Fabs'] * addis_features_hips['volume_m3'] / 100.0)

tracking_rows = []
for target in ('HIPS_tau', 'Fabs', 'EC_deployed_ugm3'):
    for band in ('band1600_height', 'carbonyl_peak', 'CH_peak'):
        valid = addis_features_hips[[band, target]].dropna()
        tracking_rows.append({
            'target': target, 'band': band, 'n': len(valid),
            'pearson_r': pearsonr(valid[band], valid[target])[0],
            'spearman_r': spearmanr(valid[band], valid[target])[0],
        })
tracking = pd.DataFrame(tracking_rows)
tracking.to_csv(TABLE_DIR / 'addis_band_vs_absorption.csv', index=False)
display(tracking)

# %% [markdown]
# ### 7. Median diagnostic-window spectra

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
for name, X in groups.items():
    corrected = offset_correct(np.asarray(X, float), wn)
    median = np.nanmedian(corrected, axis=0)
    scale = np.nanmedian(
        features[name]['CH_peak'].where(features[name]['CH_peak'] > 1e-4))
    for ax, (low, high) in zip(axes, [(1500, 1800), (2600, 3600)]):
        mask = (wn >= low) & (wn <= high)
        ax.plot(wn[mask], median[mask] / scale, lw=1.6, color=colors[name], label=name)
axes[0].axvspan(*BAND_WINDOW, color='0.9')
axes[0].set(xlim=(1800, 1500), xlabel='Wavenumber (cm⁻¹)',
            ylabel='Median absorbance / median CH height',
            title='1600-band window (shaded)')
axes[1].set(xlim=(3600, 2600), xlabel='Wavenumber (cm⁻¹)',
            title='CH and N–H/O–H region')
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'median_spectra_diagnostic_windows.png', dpi=180, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Takeaways
#
# (filled in by the finalize step after execution)
