# %% [markdown]
# # ftir_11 — Ann's OC/EC-ratio cohort: calibrate TOR EC on the lowest-OCEC IMPROVE samples
#
# ## tl;dr
#
# (filled in by the finalize step after execution)
#
# ## Context & Methods
#
# The July 2026 meeting produced a specific untested strategy from Ann: Addis Ababa sits at the
# extreme low end of the IMPROVE OC/EC ratio distribution, so instead of selecting calibration
# samples by smoke classification or spectral similarity, select the IMPROVE lot-248/251 samples
# with the **lowest TOR OC/EC ratios** and rebuild the TOR EC calibration on them. This notebook
# runs that strategy under the locked phase-2 protocol:
#
# - cohort selection never sees Addis HIPS values or any regression outcome;
# - a site-disjoint 20% TOR test is split off **before** fitting (same seed as ftir_10);
# - components use the first-major-minimum rule on a site-grouped CV curve;
# - Addis evaluation uses HIPS EC-equivalent (MAC = 10 primary, 6 sensitivity) on the same
#   fixed 190-filter cohort as ftir_10, plus all available pairs.
#
# Cohort sizes N = 400, 800, 1600 probe Ann's "800 or however many" suggestion; five
# size-matched random cohorts (N = 800) control for sample count alone.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path('scripts').resolve()))
sys.path.insert(0, str((Path('..') / 'ftir_hips_chem' / 'scripts').resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupShuffleSplit

from config import season_for_month
from phase3_common import (
    PHASE2_TABLES, load_addis_evaluation, load_tor_loadings,
    load_pool_metadata, load_pool_spectra,
)
from pls_transfer import (
    component_cv_curve, select_first_major_minimum, regression_metrics,
)

TABLE_DIR = Path('output/tables/ftir11')
PLOT_DIR = Path('output/plots/ftir11')
for directory in (TABLE_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

COHORT_SIZES = (400, 800, 1600)
PRIMARY_N = 800
N_RANDOM_CONTROLS = 5
SPLIT_SEED = 20260717  # identical to ftir_10's locked analog split

# %% [markdown]
# ## Data
#
# ### 1. Addis evaluation table and the fixed phase-2 comparison cohort

# %%
etad_eval, X_etad, wn = load_addis_evaluation(season_for_month)
wcols = etad_eval.attrs['wcols']
volume = etad_eval['SampleVolume_m3'].to_numpy(float)

phase2_predictions = pd.read_csv(
    PHASE2_TABLES / 'pls_calibration_phase2' / 'addis_calibration_predictions.csv')
fixed_media = set(phase2_predictions.dropna(axis=0, how='any')['MediaId'])
fixed_mask = etad_eval['MediaId'].isin(fixed_media).to_numpy()
print(f'Addis available pairs n={len(etad_eval)}; fixed phase-2 cohort n={fixed_mask.sum()}')

# %% [markdown]
# ### 2. Match the 13k pool to TOR EC and OC, and audit eligibility

# %%
pool = load_pool_metadata()
tor = load_tor_loadings()
pool = pool.merge(tor, on=['Site', 'date'], how='left', validate='many_to_one')

eligible = (
    pool['TOR_EC_loading_ug'].gt(0)
    & pool['TOR_EC_ugm3'].gt(0)
    & pool['TOR_OC_ugm3'].gt(0)
    & pool['OC_EC_ratio'].notna()
)
pool_eligible = (pool[eligible]
                 .sort_values('OC_EC_ratio')
                 .drop_duplicates('FilterId')
                 .reset_index(drop=True))

eligibility_audit = pd.DataFrame([{
    'pool_spectra': len(pool),
    'with_positive_TOR_EC_loading': int(pool['TOR_EC_loading_ug'].gt(0).sum()),
    'eligible_with_OC_and_EC': int(eligible.sum()),
    'unique_filters_eligible': len(pool_eligible),
    'OCEC_p01': float(pool_eligible['OC_EC_ratio'].quantile(.01)),
    'OCEC_p05': float(pool_eligible['OC_EC_ratio'].quantile(.05)),
    'OCEC_median': float(pool_eligible['OC_EC_ratio'].median()),
    'OCEC_p95': float(pool_eligible['OC_EC_ratio'].quantile(.95)),
}])
eligibility_audit.to_csv(TABLE_DIR / 'ocec_pool_eligibility_audit.csv', index=False)
display(eligibility_audit)

# %% [markdown]
# ### 3. Select the lowest-OCEC cohorts and size-matched random controls

# %%
cohorts = {}
for size in COHORT_SIZES:
    selection = pool_eligible.head(size).copy()
    cohorts[f'lowest-OCEC {size}'] = selection

rng = np.random.default_rng(SPLIT_SEED)
for draw in range(N_RANDOM_CONTROLS):
    index = rng.choice(len(pool_eligible), size=PRIMARY_N, replace=False)
    cohorts[f'random {PRIMARY_N} #{draw + 1}'] = pool_eligible.iloc[index].copy()

composition_rows = []
for name, frame in cohorts.items():
    composition_rows.append({
        'cohort': name, 'n': len(frame), 'sites': frame['Site'].nunique(),
        'OCEC_min': float(frame['OC_EC_ratio'].min()),
        'OCEC_max': float(frame['OC_EC_ratio'].max()),
        'OCEC_median': float(frame['OC_EC_ratio'].median()),
        'top_sites': ', '.join(frame['Site'].value_counts().head(5).index),
    })
composition = pd.DataFrame(composition_rows)
composition.to_csv(TABLE_DIR / 'cohort_composition.csv', index=False)
cohorts[f'lowest-OCEC {PRIMARY_N}'][
    ['AnalysisId', 'FilterId', 'Site', 'SampleDate', 'OC_EC_ratio',
     'TOR_EC_loading_ug', 'TOR_EC_ugm3', 'TOR_OC_ugm3']
].to_csv(TABLE_DIR / 'lowest_ocec_800_cohort.csv', index=False)
display(composition)

# %% [markdown]
# ### 4. Fetch spectra for every cohort in one chunked pass

# %%
needed_ids = sorted(set().union(*[set(f['AnalysisId'].astype(int)) for f in cohorts.values()]))
spectra = load_pool_spectra(needed_ids, wcols).set_index('AnalysisId')
print(f'Fetched {len(spectra)} of {len(needed_ids)} needed spectra')

# %% [markdown]
# ## Results
#
# ### 5. Fit each cohort under the locked site-held-out protocol

# %%
def fit_cohort(name, frame, k_override=None):
    frame = frame[frame['AnalysisId'].astype(int).isin(spectra.index)].copy()
    X = spectra.loc[frame['AnalysisId'].astype(int), wcols].to_numpy(float)
    y = frame['TOR_EC_loading_ug'].to_numpy(float)
    sites = frame['Site'].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SPLIT_SEED)
    train_pos, test_pos = next(splitter.split(frame, groups=sites))
    assert set(sites[train_pos]).isdisjoint(sites[test_pos])

    if k_override is None:
        curve = component_cv_curve(
            X[train_pos], y[train_pos], range(1, 41),
            groups=sites[train_pos], n_splits=5, random_state=42)
        k, curve = select_first_major_minimum(curve)
        curve['model'] = name
    else:
        k, curve = int(k_override), None
    model = PLSRegression(n_components=k, scale=False).fit(X[train_pos], y[train_pos])

    heldout = regression_metrics(y[test_pos], model.predict(X[test_pos]).ravel())
    addis_ugm3 = model.predict(X_etad).ravel() / volume
    return {
        'name': name, 'model': model, 'k': k, 'curve': curve,
        'n_train': len(train_pos), 'n_test': len(test_pos),
        'train_sites': int(pd.Series(sites[train_pos]).nunique()),
        'test_sites': int(pd.Series(sites[test_pos]).nunique()),
        'heldout': heldout, 'addis_ugm3': addis_ugm3,
    }

fits = {}
for size in COHORT_SIZES:
    name = f'lowest-OCEC {size}'
    fits[name] = fit_cohort(name, cohorts[name])
    print(f"{name}: k={fits[name]['k']}, held-out TOR RMSE={fits[name]['heldout']['RMSE']:.2f}")

primary_k = fits[f'lowest-OCEC {PRIMARY_N}']['k']
for draw in range(N_RANDOM_CONTROLS):
    name = f'random {PRIMARY_N} #{draw + 1}'
    fits[name] = fit_cohort(name, cohorts[name], k_override=primary_k)

# %% [markdown]
# ### 6. Held-out TOR metrics and Addis HIPS comparison for every cohort

# %%
heldout_rows, addis_rows = [], []
hips = etad_eval['Fabs'].to_numpy(float)
for name, fit in fits.items():
    heldout_rows.append({
        'model': name, 'n_components': fit['k'], 'n_train': fit['n_train'],
        'n_test': fit['n_test'], 'train_sites': fit['train_sites'],
        'test_sites': fit['test_sites'], **fit['heldout'],
    })
    for mac in (10.0, 6.0):
        for cohort_name, mask in [('available pairs', np.ones(len(etad_eval), bool)),
                                  ('fixed phase-2 cohort', fixed_mask)]:
            addis_rows.append({
                'model': name, 'MAC_m2_g': mac, 'cohort': cohort_name,
                **regression_metrics(hips[mask] / mac, fit['addis_ugm3'][mask]),
            })

deployed = etad_eval['EC_deployed_ugm3'].to_numpy(float)
for mac in (10.0, 6.0):
    for cohort_name, mask in [('available pairs', etad_eval['EC_deployed_ugm3'].notna().to_numpy()),
                              ('fixed phase-2 cohort', fixed_mask)]:
        addis_rows.append({
            'model': 'Deployed SPARTAN FTIR EC', 'MAC_m2_g': mac, 'cohort': cohort_name,
            **regression_metrics(hips[mask] / mac, deployed[mask]),
        })

heldout_metrics = pd.DataFrame(heldout_rows)
addis_metrics = pd.DataFrame(addis_rows)
heldout_metrics.to_csv(TABLE_DIR / 'site_held_out_tor_metrics.csv', index=False)
addis_metrics.to_csv(TABLE_DIR / 'addis_metrics.csv', index=False)

random_names = [f'random {PRIMARY_N} #{d + 1}' for d in range(N_RANDOM_CONTROLS)]
random_addis = addis_metrics[
    addis_metrics['model'].isin(random_names)
    & addis_metrics['MAC_m2_g'].eq(10) & addis_metrics['cohort'].eq('fixed phase-2 cohort')]
random_summary = pd.DataFrame([{
    'random_intercept_min': random_addis['intercept'].min(),
    'random_intercept_max': random_addis['intercept'].max(),
    'random_RMSE_min': random_addis['RMSE'].min(),
    'random_RMSE_max': random_addis['RMSE'].max(),
}])
random_summary.to_csv(TABLE_DIR / 'random_control_summary.csv', index=False)

display(heldout_metrics[['model', 'n_components', 'n_test', 'test_sites', 'slope', 'R2', 'RMSE']])
display(addis_metrics[(addis_metrics['MAC_m2_g'].eq(10))
                      & (addis_metrics['cohort'].eq('fixed phase-2 cohort'))]
        [['model', 'n', 'slope', 'intercept', 'R2', 'RMSE', 'bias']])

# %% [markdown]
# ### 7. Export Addis predictions and calibration coefficients

# %%
prediction_export = etad_eval[['MediaId', 'ExternalFilterId', 'SamplingStartDate',
                               'season', 'Fabs', 'EC_deployed_ugm3']].copy()
for size in COHORT_SIZES:
    prediction_export[f'EC_lowest_ocec_{size}_ugm3'] = fits[f'lowest-OCEC {size}']['addis_ugm3']
prediction_export['in_fixed_phase2_cohort'] = fixed_mask
prediction_export.to_csv(TABLE_DIR / 'addis_predictions.csv', index=False)

def export_coefficients(fit, path):
    model = fit['model']
    coefficient = (model.x_rotations_ @ model.y_loadings_.T).reshape(-1)
    intercept = float(np.asarray(model._y_mean).reshape(-1)[0]
                      - np.asarray(model._x_mean) @ coefficient)
    table = pd.concat([
        pd.DataFrame({'Wavenumber': [0.0], 'b': [intercept]}),
        pd.DataFrame({'Wavenumber': wn, 'b': coefficient}),
    ], ignore_index=True)
    table.to_csv(path, index=False)

export_coefficients(fits[f'lowest-OCEC {PRIMARY_N}'],
                    TABLE_DIR / f'ec_lowest_ocec_{PRIMARY_N}_first_major.csv')

# %% [markdown]
# ### 8. Visualize the cohort cut, CV curves, and Addis crossplots

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
axes[0].hist(pool_eligible['OC_EC_ratio'].clip(upper=25), bins=120, color='#7F8C8D', alpha=.8)
for size, color in zip(COHORT_SIZES, ('#27AE60', '#2980B9', '#8E44AD')):
    cut = float(cohorts[f'lowest-OCEC {size}']['OC_EC_ratio'].max())
    axes[0].axvline(cut, color=color, lw=1.6, label=f'N={size} cut: OC/EC ≤ {cut:.2f}')
axes[0].set(xlabel='TOR OC/EC ratio (clipped at 25)', ylabel='Eligible pool filters',
            title='Lowest-OCEC cohort cuts inside the IMPROVE pool')
axes[0].legend(fontsize=8)

for size, color in zip(COHORT_SIZES, ('#27AE60', '#2980B9', '#8E44AD')):
    curve = fits[f'lowest-OCEC {size}']['curve']
    axes[1].plot(curve['n_components'], curve['rmse_mean'], color=color, lw=1.5,
                 label=f'N={size} (k={fits[f"lowest-OCEC {size}"]["k"]})')
    chosen = curve[curve['selected_first_major_minimum']].iloc[0]
    axes[1].scatter([chosen['n_components']], [chosen['rmse_mean']], color=color, s=55, zorder=4)
axes[1].set(xlabel='PLS components', ylabel='Site-held-out CV RMSE (µg/filter)',
            title='Component selection per cohort size')
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'cohort_cut_and_cv_curves.png', dpi=180, bbox_inches='tight')
plt.show()

# %%
plot_models = [('Deployed SPARTAN', deployed)] + [
    (f'Lowest-OCEC {size} (k={fits[f"lowest-OCEC {size}"]["k"]})',
     fits[f'lowest-OCEC {size}']['addis_ugm3'])
    for size in COHORT_SIZES
]
x = hips[fixed_mask] / 10.0
fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharex=True, sharey=True)
for ax, (label, values) in zip(axes.flat, plot_models):
    y = np.asarray(values, float)[fixed_mask]
    stats = regression_metrics(x, y)
    ax.scatter(x, y, s=22, alpha=.55, color='#34495E')
    lo, hi = min(0, np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))
    ax.plot([lo, hi], [lo, hi], '--', color='0.55', lw=1)
    fit_x = np.array([np.nanmin(x), np.nanmax(x)])
    ax.plot(fit_x, stats['slope'] * fit_x + stats['intercept'], color='#C0392B', lw=1.6)
    ax.set_title(label, fontsize=10)
    ax.text(.04, .96, f'y={stats["slope"]:.2f}x {stats["intercept"]:+.2f}\n'
                      f'R²={stats["R2"]:.3f}; RMSE={stats["RMSE"]:.2f}\nn={stats["n"]}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='0.8', alpha=.9))
    ax.set_xlabel('HIPS EC-equivalent, MAC=10 (µg/m³)')
axes[0].set_ylabel('FTIR EC (µg/m³)')
fig.suptitle('OC/EC-ratio cohorts on the fixed phase-2 Addis cohort', y=1.02)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'addis_crossplots_ocec_cohorts.png', dpi=180, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Takeaways
#
# (filled in by the finalize step after execution)
