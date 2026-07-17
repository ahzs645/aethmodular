#!/usr/bin/env python3
"""Build the ftir_11 low-OC/EC cohort calibration notebook (phase 3)."""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ftir_11_ocec_ratio_cohort.ipynb"


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
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupShuffleSplit

from config import SITES, ETHIOPIA_SEASONS, season_for_month
from data_matching import load_filter_data
from plotting import PlotConfig
from pls_transfer import (
    FTIRTransferPaths, load_current_pls_model, regression_metrics,
    component_cv_curve, select_first_major_minimum, vip_scores,
    vip_overlap_summary, summarize_vip_bands,
)

PlotConfig.set(sites='all', layout='individual', show_stats=True, show_1to1=True)
PATHS = FTIRTransferPaths.defaults()
P2_TABLES = Path('../ftir_hips_chem/output/tables')
TABLE_DIR = Path('output/tables/ftir_11')
PLOT_DIR = Path('output/plots/ftir_11')
CAL_DIR = TABLE_DIR / 'calibrations'
for directory in (TABLE_DIR, PLOT_DIR, CAL_DIR):
    directory.mkdir(parents=True, exist_ok=True)
"""
    )

    md(
        """
# Train FTIR EC on the lowest-OC/EC IMPROVE samples (Ann's cohort idea)

## tl;dr

TLDR_PLACEHOLDER

## Context & Methods

At the July 2026 meeting, Addis Ababa was shown to sit **below the OC/EC ratio of every
IMPROVE site**, and Ann proposed the direct test of the OC/EC hypothesis: instead of
smoke-classified days or spectral analogs, select the IMPROVE filters with the **lowest
TOR OC/EC ratios** and rebuild the TOR EC calibration from them.

Protocol matches `ftir_10` exactly: cohort selection never sees Addis HIPS values or any
regression outcome; a site-held-out TOR test is locked **before** fitting
(`GroupShuffleSplit`, 20 % of sites, seed 20260717); components are chosen with the
first-major-minimum rule on a site-held-out CV curve; Addis evaluation uses HIPS
EC-equivalent (Fabs/MAC) on the x-axis at MAC = 10 with MAC = 6 as sensitivity, on the
same fixed common Addis cohort as `ftir_10`.

### Key Assumptions

- TOR OC/EC concentration ratios equal loading ratios (shared volume cancels).
- Size-matched random cohorts (same n, same disjoint-site test set, same component count)
  are the null against which OC/EC selection must show an advantage.
- HIPS/MAC comparisons remain diagnostic; no independent Addis TOR EC exists.
"""
    )

    md("## Data\n\n### 1. Addis spectra, HIPS, and deployed EC")

    code(
        """
raw_etad = pd.read_csv(PATHS.etad_dir / 'ETAD_FTIR_spectra.csv')
etad_meta = pd.read_csv(PATHS.etad_dir / 'ETAD_metadata.csv')
wcols = sorted([c for c in raw_etad.columns if c not in ('SampleAnalysisId', 'MediaId')],
               key=lambda value: -float(value))
wn = np.array([float(c) for c in wcols])
etad_spectra = raw_etad.groupby('MediaId', as_index=False)[wcols].mean()

hips = pd.read_csv(PATHS.spartan_hips_primary, encoding='cp1252')
hips_etad = (hips[hips['Site'].eq('ETAD')][['FilterId', 'Fabs', 'tau', 'DepositArea', 'Volume']]
             .drop_duplicates('FilterId'))

filter_data = load_filter_data()
deployed = (filter_data[(filter_data['Site'].eq('ETAD')) &
                        (filter_data['Parameter'].eq('EC_ftir'))]
            [['FilterId', 'Concentration']]
            .drop_duplicates('FilterId')
            .rename(columns={'Concentration': 'EC_deployed_ugm3'}))

etad = (etad_spectra.merge(etad_meta, on='MediaId', how='left', validate='one_to_one')
        .merge(hips_etad.rename(columns={'FilterId': 'ExternalFilterId'}),
               on='ExternalFilterId', how='left', validate='one_to_one')
        .merge(deployed, left_on='ExternalFilterId', right_on='FilterId',
               how='left', validate='one_to_one'))
etad['SamplingStartDate'] = pd.to_datetime(etad['SamplingStartDate'], errors='coerce')
etad['season'] = etad['SamplingStartDate'].dt.month.map(season_for_month)
etad['has_complete_spectrum'] = etad[wcols].notna().all(axis=1)
etad_eval = etad[etad['has_complete_spectrum'] & etad['Fabs'].notna() &
                 etad['SampleVolume_m3'].gt(0)].copy()
X_etad = etad_eval[wcols].to_numpy(float)
print(f'Addis evaluation spectra with HIPS: n={len(etad_eval)}')
"""
    )

    md("### 2. TOR OC/EC ratios for the full lot-248/251 pool")

    code(
        """
similarity = pd.read_csv(P2_TABLES / 'pls_transfer/improve_full_pool_addis_similarity.csv')
similarity['date'] = pd.to_datetime(similarity['SampleDate'], format='mixed',
                                    errors='coerce').dt.normalize()

tor = pd.read_csv(
    PATHS.ftir_dir / 'local_db/tables/results_tor.csv',
    usecols=['Site', 'SampleDate', 'Parameter', 'Value', 'AverageFlowRate', 'ElapsedTime'])
tor = tor[tor['Parameter'].isin(['EC', 'OC'])].copy()
tor['date'] = pd.to_datetime(tor['SampleDate'], format='mixed', errors='coerce').dt.normalize()
tor = tor.drop_duplicates(['Site', 'date', 'Parameter'])
tor_wide = (tor.pivot_table(index=['Site', 'date'],
                            columns='Parameter',
                            values=['Value', 'AverageFlowRate', 'ElapsedTime'],
                            aggfunc='first'))
tor_wide.columns = [f'{a}_{b}' for a, b in tor_wide.columns]
tor_wide = tor_wide.reset_index()
tor_wide['TOR_EC_loading_ug'] = (
    tor_wide['Value_EC'] * (tor_wide['AverageFlowRate_EC'] / 1000 *
                            tor_wide['ElapsedTime_EC']) / 1000)
tor_wide['OC_EC_ratio'] = tor_wide['Value_OC'] / tor_wide['Value_EC']

candidates = (similarity.merge(
    tor_wide[['Site', 'date', 'Value_EC', 'Value_OC', 'TOR_EC_loading_ug', 'OC_EC_ratio']],
    on=['Site', 'date'], how='left', validate='many_to_one')
    .query('TOR_EC_loading_ug > 0 and Value_OC > 0 and Value_EC > 0')
    .sort_values('OC_EC_ratio')
    .drop_duplicates('FilterId')
    .reset_index(drop=True))

pool_audit = pd.DataFrame([{
    'pool_spectra': len(similarity),
    'candidates_with_positive_TOR_OC_and_EC': len(candidates),
    'OC_EC_ratio_p05': candidates['OC_EC_ratio'].quantile(.05),
    'OC_EC_ratio_median': candidates['OC_EC_ratio'].median(),
    'OC_EC_ratio_p95': candidates['OC_EC_ratio'].quantile(.95),
}])
pool_audit.to_csv(TABLE_DIR / 'ocec_pool_audit.csv', index=False)
display(pool_audit)
"""
    )

    md("### 3. Select the lowest-OC/EC cohort and lock a disjoint-site TOR test")

    code(
        """
N_COHORT = 800
cohort = candidates.head(N_COHORT).copy()
cohort_threshold = float(cohort['OC_EC_ratio'].max())

splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=20260717)
train_position, test_position = next(splitter.split(cohort, groups=cohort['Site']))
cohort_train = cohort.iloc[train_position].copy()
cohort_test = cohort.iloc[test_position].copy()
assert set(cohort_train['Site']).isdisjoint(set(cohort_test['Site']))

# Size-matched random null cohorts drawn from all candidates outside the test sites.
rng = np.random.default_rng(42)
test_sites = set(cohort_test['Site'])
random_source = candidates[~candidates['Site'].isin(test_sites)].reset_index(drop=True)
N_RANDOM = 10
random_cohorts = [random_source.iloc[
    rng.choice(len(random_source), size=len(cohort_train), replace=False)].copy()
    for _ in range(N_RANDOM)]

split_export = pd.concat([
    cohort_train[['AnalysisId', 'FilterId', 'Site', 'OC_EC_ratio']].assign(split='train'),
    cohort_test[['AnalysisId', 'FilterId', 'Site', 'OC_EC_ratio']].assign(
        split='site-held-out TOR test'),
])
split_export.to_csv(TABLE_DIR / 'locked_ocec_train_test_split.csv', index=False)
print(f'Cohort OC/EC ratio range: {cohort.OC_EC_ratio.min():.3f}-{cohort_threshold:.3f} '
      f'(pool median {candidates.OC_EC_ratio.median():.2f})')
print(f'Locked split: train={len(cohort_train)} across {cohort_train.Site.nunique()} sites; '
      f'test={len(cohort_test)} across {cohort_test.Site.nunique()} disjoint sites; '
      f'{N_RANDOM} random null cohorts of n={len(cohort_train)}')

fig, ax = plt.subplots(figsize=(8.5, 4.6))
bins = np.geomspace(candidates['OC_EC_ratio'].min(), candidates['OC_EC_ratio'].max(), 70)
ax.hist(candidates['OC_EC_ratio'], bins=bins, color='#7F8C8D', alpha=.55,
        label=f'All eligible IMPROVE (n={len(candidates)})')
ax.hist(cohort['OC_EC_ratio'], bins=bins, color='#8E44AD', alpha=.75,
        label=f'Lowest-OC/EC cohort (n={N_COHORT})')
ax.axvline(cohort_threshold, color='#8E44AD', lw=1.2, ls='--',
           label=f'Cohort threshold OC/EC={cohort_threshold:.2f}')
ax.set_xscale('log')
ax.set(xlabel='TOR OC/EC ratio', ylabel='Filters',
       title='Addis sits below the entire IMPROVE OC/EC range; this cohort is the closest edge')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'ocec_cohort_selection.png', dpi=180, bbox_inches='tight')
plt.show()
"""
    )

    md("### 4. Fetch spectra for the cohort, test set, and random nulls")

    code(
        """
needed_frames = [cohort_train, cohort_test] + random_cohorts
needed_ids = set(int(v) for frame in needed_frames for v in frame['AnalysisId'])
spectra_parts = []
pool_path = PATHS.ftir_dir / 'local_db/spectra_248_251.csv'
for chunk in pd.read_csv(pool_path, chunksize=750):
    keep = chunk['AnalysisId'].astype(int).isin(needed_ids)
    if keep.any():
        spectra_parts.append(chunk.loc[keep, ['AnalysisId'] + wcols])
spectra = (pd.concat(spectra_parts, ignore_index=True)
           .drop_duplicates('AnalysisId')
           .set_index('AnalysisId'))
print(f'Fetched {len(spectra)} of {len(needed_ids)} needed spectra')

def spectra_for(frame):
    return spectra.loc[frame['AnalysisId'].astype(int)][wcols].to_numpy(float)

X_train = spectra_for(cohort_train)
X_test = spectra_for(cohort_test)
y_train = cohort_train['TOR_EC_loading_ug'].to_numpy(float)
y_test = cohort_test['TOR_EC_loading_ug'].to_numpy(float)
"""
    )

    md("## Results\n\n### 5. Fit with the first-major-minimum rule; random nulls at the same size and k")

    code(
        """
curve = component_cv_curve(X_train, y_train, range(1, 41),
                           groups=cohort_train['Site'].to_numpy(),
                           n_splits=5, random_state=42)
first_k, curve = select_first_major_minimum(curve)
global_k = int(curve['global_minimum_components'].iloc[0])
curve['model'] = 'Lowest-OC/EC cohort'
curve.to_csv(TABLE_DIR / 'ocec_component_cv_curve.csv', index=False)

ocec_first = PLSRegression(n_components=first_k, scale=False).fit(X_train, y_train)
ocec_global = PLSRegression(n_components=global_k, scale=False).fit(X_train, y_train)
print(f'Lowest-OC/EC cohort: first-major k={first_k}, global-min k={global_k}')

random_models = []
for index, frame in enumerate(random_cohorts):
    model = PLSRegression(n_components=first_k, scale=False).fit(
        spectra_for(frame), frame['TOR_EC_loading_ug'].to_numpy(float))
    random_models.append(model)

tor_holdout = pd.DataFrame(
    [{'model': f'Lowest-OC/EC—first major (k={first_k})',
      **regression_metrics(y_test, ocec_first.predict(X_test).ravel())},
     {'model': f'Lowest-OC/EC—global minimum sensitivity (k={global_k})',
      **regression_metrics(y_test, ocec_global.predict(X_test).ravel())}] +
    [{'model': f'Random size-matched #{index + 1} (k={first_k})',
      **regression_metrics(y_test, model.predict(X_test).ravel())}
     for index, model in enumerate(random_models)])
tor_holdout.to_csv(TABLE_DIR / 'site_held_out_tor_metrics.csv', index=False)
display(tor_holdout.head(4))
random_rows = tor_holdout['model'].str.startswith('Random')
print('Random-null held-out TOR RMSE range: '
      f"{tor_holdout.loc[random_rows, 'RMSE'].min():.2f}-"
      f"{tor_holdout.loc[random_rows, 'RMSE'].max():.2f} ug/filter")
"""
    )

    md("### 6. Addis evaluation on the fixed ftir_10 common cohort")

    code(
        """
volume = etad_eval['SampleVolume_m3'].to_numpy(float)
etad_eval['EC_ocec_first_ugm3'] = ocec_first.predict(X_etad).ravel() / volume
etad_eval['EC_ocec_global_ugm3'] = ocec_global.predict(X_etad).ravel() / volume
for index, model in enumerate(random_models):
    etad_eval[f'EC_random_{index + 1}_ugm3'] = model.predict(X_etad).ravel() / volume

prior = pd.read_csv(P2_TABLES / 'pls_calibration_phase2/addis_calibration_predictions.csv')
prior_columns = [c for c in prior.columns if c.startswith('EC_')]
merged = etad_eval.merge(prior[['ExternalFilterId'] + prior_columns],
                         on='ExternalFilterId', how='left', validate='one_to_one',
                         suffixes=('', '_prior'))
common_mask = merged[prior_columns + ['Fabs', 'EC_ocec_first_ugm3']].notna().all(axis=1)
print(f'Fixed common Addis cohort n={int(common_mask.sum())} (ftir_10 used the same filters)')

model_columns = {
    'Deployed SPARTAN FTIR EC': 'EC_deployed_ugm3',
    'Smoke IMPROVE (906)': 'EC_smoke_906_ugm3',
    'Ethiopia-shaped smoke (300)': 'EC_smoke_shape_300_ugm3',
    'Locked analog raw—first major': 'EC_analog_raw_first_ugm3',
    f'Lowest-OC/EC—first major (k={first_k})': 'EC_ocec_first_ugm3',
    f'Lowest-OC/EC—global sensitivity (k={global_k})': 'EC_ocec_global_ugm3',
}
comparison_rows = []
for mac in (6.0, 10.0):
    hips_ec = merged['Fabs'].to_numpy(float) / mac
    for model_name, column in model_columns.items():
        source = merged[column] if column in merged else merged[f'{column}_prior']
        comparison_rows.append({
            'model': model_name, 'MAC_m2_g': mac, 'cohort': 'fixed common cohort',
            **regression_metrics(hips_ec[common_mask],
                                 source[common_mask].to_numpy(float)),
        })
    for index in range(len(random_models)):
        comparison_rows.append({
            'model': f'Random size-matched #{index + 1}', 'MAC_m2_g': mac,
            'cohort': 'fixed common cohort',
            **regression_metrics(hips_ec[common_mask],
                                 merged.loc[common_mask,
                                            f'EC_random_{index + 1}_ugm3'].to_numpy(float)),
        })
addis_metrics = pd.DataFrame(comparison_rows)
addis_metrics.to_csv(TABLE_DIR / 'addis_ocec_calibration_metrics.csv', index=False)
merged.loc[:, ['MediaId', 'ExternalFilterId', 'SamplingStartDate', 'season', 'Fabs',
               'EC_deployed_ugm3', 'EC_ocec_first_ugm3', 'EC_ocec_global_ugm3'] +
           [f'EC_random_{index + 1}_ugm3' for index in range(len(random_models))]].to_csv(
    TABLE_DIR / 'addis_ocec_predictions.csv', index=False)

headline = addis_metrics[(addis_metrics['MAC_m2_g'].eq(10.0))]
display(headline[~headline['model'].str.startswith('Random')]
        [['model', 'n', 'slope', 'intercept', 'R2', 'RMSE', 'bias']])
random_intercepts = headline.loc[headline['model'].str.startswith('Random'), 'intercept']
print(f'Random-null Addis intercept range at MAC=10: '
      f'{random_intercepts.min():.2f} to {random_intercepts.max():.2f}')
"""
    )

    md("### 7. Which spectral features does the low-OC/EC model use?")

    code(
        """
current = load_current_pls_model(PATHS.ftir_dir, 'EC')
assert np.allclose(current.wavenumbers, wn)
vip_ocec = vip_scores(ocec_first)
vip_current = vip_scores(current.model)
profiles = pd.read_csv(P2_TABLES / 'pls_transfer/improve_addis_hips_vip_profiles.csv')
assert np.allclose(profiles['wavenumber_cm-1'].to_numpy(), wn)
vip_addis_hips = profiles['Addis_HIPS_VIP'].to_numpy(float)

overlaps = pd.DataFrame([
    {'pair': 'low-OC/EC vs current 906-smoke EC',
     **vip_overlap_summary(vip_ocec, vip_current)},
    {'pair': 'low-OC/EC vs Addis-only HIPS (ftir_08)',
     **vip_overlap_summary(vip_ocec, vip_addis_hips)},
    {'pair': 'current 906-smoke EC vs Addis-only HIPS (ftir_08)',
     **vip_overlap_summary(vip_current, vip_addis_hips)},
])
overlaps.to_csv(TABLE_DIR / 'ocec_vip_overlaps.csv', index=False)
display(overlaps[['pair', 'spearman_r', 'top_n_overlap', 'important_jaccard']])

bands = summarize_vip_bands(wn, vip_ocec).assign(model='Lowest-OC/EC')
bands.to_csv(TABLE_DIR / 'ocec_vip_band_summary.csv', index=False)
pd.DataFrame({'wavenumber_cm-1': wn, 'lowest_OCEC_VIP': vip_ocec,
              'current_EC_VIP': vip_current,
              'Addis_HIPS_VIP': vip_addis_hips}).to_csv(
    TABLE_DIR / 'ocec_vip_profiles.csv', index=False)

fig, ax = plt.subplots(figsize=(10.5, 5.2))
ax.plot(wn, vip_current, lw=1.1, color='#7F8C8D', label='Current 906-smoke EC')
ax.plot(wn, vip_addis_hips, lw=1.1, color='#E67E22', label='Addis-only HIPS (ftir_08)')
ax.plot(wn, vip_ocec, lw=1.3, color='#8E44AD', label=f'Lowest-OC/EC (k={first_k})')
ax.axhline(1.0, color='0.6', lw=.8, ls=':')
ax.set_xlim(4000, 400)
ax.set(xlabel='Wavenumber (cm⁻¹)', ylabel='VIP',
       title='Does OC/EC selection move the calibration toward Addis-relevant features?')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'ocec_vip_profiles.png', dpi=180, bbox_inches='tight')
plt.show()
"""
    )

    md("### 8. Crossplots on the fixed common cohort")

    code(
        """
common = merged.loc[common_mask].copy()
x = common['Fabs'].to_numpy(float) / 10.0
random_columns = [f'EC_random_{index + 1}_ugm3' for index in range(len(random_models))]
median_random = common[random_columns].median(axis=1)

panels = [
    ('Deployed SPARTAN', common['EC_deployed_ugm3'].to_numpy(float)),
    (f'Lowest-OC/EC—first major (k={first_k})', common['EC_ocec_first_ugm3'].to_numpy(float)),
    (f'Lowest-OC/EC—global (k={global_k})', common['EC_ocec_global_ugm3'].to_numpy(float)),
    ('Median of 10 random nulls', median_random.to_numpy(float)),
]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), sharex=True, sharey=True)
for ax, (label, y) in zip(axes.flat, panels):
    stats = regression_metrics(x, y)
    ax.scatter(x, y, s=24, alpha=.55, color='#34495E')
    lo, hi = min(0, x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], '--', color='0.55', lw=1)
    fit_x = np.array([x.min(), x.max()])
    ax.plot(fit_x, stats['slope'] * fit_x + stats['intercept'], color='#C0392B', lw=1.6)
    ax.set_title(label, fontsize=11)
    ax.text(.04, .96,
            f'y={stats["slope"]:.2f}x {stats["intercept"]:+.2f}\\n'
            f'R²={stats["R2"]:.3f}; RMSE={stats["RMSE"]:.2f}\\nn={stats["n"]}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='0.8', alpha=.9))
for ax in axes[-1]:
    ax.set_xlabel('HIPS EC-equivalent, MAC=10 (µg/m³)')
for ax in axes[:, 0]:
    ax.set_ylabel('FTIR EC (µg/m³)')
fig.suptitle('Lowest-OC/EC cohort vs deployed and random nulls (fixed Addis cohort)', y=1.0)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'ocec_addis_crossplots.png', dpi=180, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(curve['n_components'], curve['rmse_mean'], lw=1.5, color='#8E44AD')
ax.fill_between(curve['n_components'], curve['rmse_mean'] - curve['rmse_se'],
                curve['rmse_mean'] + curve['rmse_se'], color='#8E44AD', alpha=.15)
selected = curve[curve['selected_first_major_minimum']].iloc[0]
ax.scatter([selected['n_components']], [selected['rmse_mean']], color='#8E44AD', s=55,
           zorder=4, label=f'first major minimum k={first_k}')
ax.axvline(global_k, color='0.6', lw=.9, ls=':', label=f'global minimum k={global_k}')
ax.set(xlabel='PLS components', ylabel='Site-held-out CV RMSE (µg/filter)',
       title='Component selection, lowest-OC/EC cohort')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'ocec_component_cv_curve.png', dpi=180, bbox_inches='tight')
plt.show()

export = pd.concat([
    pd.DataFrame({'Wavenumber': [0.0],
                  'b': [float(np.asarray(ocec_first._y_mean).reshape(-1)[0] -
                        np.asarray(ocec_first._x_mean) @
                        (ocec_first.x_rotations_ @ ocec_first.y_loadings_.T).reshape(-1))]}),
    pd.DataFrame({'Wavenumber': wn,
                  'b': (ocec_first.x_rotations_ @ ocec_first.y_loadings_.T).reshape(-1)}),
], ignore_index=True)
export.to_csv(CAL_DIR / 'ec_lowest_ocec_800_first_major.csv', index=False)
print('Exported calibration coefficients to', CAL_DIR / 'ec_lowest_ocec_800_first_major.csv')
"""
    )

    md(
        """
## Takeaways

TAKEAWAYS_PLACEHOLDER
"""
    )

    notebook = new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    })
    nbf.write(notebook, OUTPUT)
    print(f'Wrote {OUTPUT}')


if __name__ == '__main__':
    build_notebook()
