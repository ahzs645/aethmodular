# %% [markdown]
# # ftir_15 — How solid is the −1.6 intercept, where does the corrected scatter come from,
# # and does a hybrid OC/EC + spectral cohort do better?
#
# ## tl;dr
#
# (filled in by the finalize step after execution)
#
# ## Context & Methods
#
# ftir_13 left three open questions this notebook answers with the same locked protocol
# (fixed 190-filter Addis cohort, seed-20260717 site-disjoint splits, MAC = 10 primary):
#
# 1. **Uncertainty.** The headline intercepts (−3.22 raw, −1.62 AIRSpec-corrected for the
#    lowest-OCEC-800 cohort) are single numbers from single fits. A **site-cluster bootstrap**
#    (B = 200: resample training *sites* with replacement, refit at the locked component count,
#    predict the fixed Addis cohort) puts percentile intervals on intercept, slope, and RMSE.
# 2. **Scatter.** AIRSpec correction improved the intercept but worsened Addis R² (0.77→0.66).
#    Projecting the Addis spectra into the corrected model's score space gives per-filter
#    Mahalanobis distance (D²) and spectral residual (Q); correlating prediction residuals with
#    D², Q, loading, and season localizes the added scatter.
# 3. **Hybrid cohort.** The meeting discussed combining composition and spectral similarity:
#    from the 2,000 lowest-OC/EC IMPROVE filters, keep the 800 whose **corrected spectra** are
#    nearest the median corrected Addis spectrum (VIP-weighted, using the corrected OCEC-800
#    model's VIP — no Addis HIPS values touch the selection), then refit under the locked
#    protocol.

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
from scipy.stats import pearsonr, spearmanr

from config import season_for_month, ETHIOPIA_SEASONS
from phase3_common import (
    PATHS, load_addis_evaluation, load_tor_loadings, load_pool_metadata,
)
from pls_transfer import (
    component_cv_curve, select_first_major_minimum, regression_metrics,
    vip_scores, score_metric, project_scores, mahalanobis_distance_squared,
    spectral_q_residual,
)

TABLE_DIR = Path('output/tables/ftir15')
PLOT_DIR = Path('output/plots/ftir15')
for directory in (TABLE_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SPLIT_SEED = 20260717
B_BOOT = 200
N_HYBRID = 800
N_OCEC_POOL = 2000

# %% [markdown]
# ## Data
#
# ### 1. Rebuild the two locked OCEC-800 models exactly as in ftir_11 / ftir_13

# %%
etad_eval, X_etad_raw, wn_full = load_addis_evaluation(season_for_month)
wcols = etad_eval.attrs['wcols']
volume = etad_eval['SampleVolume_m3'].to_numpy(float)
hips = etad_eval['Fabs'].to_numpy(float)

phase2_predictions = pd.read_csv(
    Path('..') / 'ftir_hips_chem' / 'output' / 'tables' / 'pls_calibration_phase2'
    / 'addis_calibration_predictions.csv')
fixed_media = set(phase2_predictions.dropna(axis=0, how='any')['MediaId'])
fixed_mask = etad_eval['MediaId'].isin(fixed_media).to_numpy()

pool_npz = np.load('output/corrected/improve_pool_corrected_df6.npz', allow_pickle=True)
wn_corr = pool_npz['wn'].astype(float)
pool_row_for_id = {int(a): i for i, a in enumerate(pool_npz['analysis_id'].astype(int))}

etad_npz = np.load('output/corrected/etad_corrected_df6.npz', allow_pickle=True)
etad_corr_media = pd.DataFrame(etad_npz['corrected'].astype(float))
etad_corr_media['MediaId'] = etad_npz['media_id'].astype(int)
X_addis_corr = (etad_corr_media.groupby('MediaId').mean()
                .loc[etad_eval['MediaId'].astype(int)].to_numpy(float))

ocec = pd.read_csv('output/tables/ftir11/lowest_ocec_800_cohort.csv')
ocec = ocec[ocec['AnalysisId'].astype(int).isin(pool_row_for_id)].copy()
y_ocec = ocec['TOR_EC_loading_ug'].to_numpy(float)
sites_ocec = ocec['Site'].to_numpy()

from phase3_common import load_pool_spectra
raw_spectra = load_pool_spectra(ocec['AnalysisId'].astype(int), wcols).set_index('AnalysisId')
X_ocec_raw = raw_spectra.loc[ocec['AnalysisId'].astype(int), wcols].to_numpy(float)
X_ocec_corr = pool_npz['corrected'][
    [pool_row_for_id[int(a)] for a in ocec['AnalysisId']]].astype(float)

splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SPLIT_SEED)
train_pos, test_pos = next(splitter.split(ocec, groups=sites_ocec))
assert set(sites_ocec[train_pos]).isdisjoint(sites_ocec[test_pos])

MODELS = {
    'raw (k=6)': dict(X=X_ocec_raw, X_addis=X_etad_raw, k=6),
    'AIRSpec df1=6 (k=5)': dict(X=X_ocec_corr, X_addis=X_addis_corr, k=5),
}
for label, spec in MODELS.items():
    model = PLSRegression(n_components=spec['k'], scale=False).fit(
        spec['X'][train_pos], y_ocec[train_pos])
    spec['model'] = model
    spec['addis_ugm3'] = model.predict(spec['X_addis']).ravel() / volume
    point = regression_metrics(hips[fixed_mask] / 10, spec['addis_ugm3'][fixed_mask])
    spec['point'] = point
    print(f"{label}: point intercept {point['intercept']:.2f}, slope {point['slope']:.2f}")

# %% [markdown]
# ## Results
#
# ### 2. Site-cluster bootstrap of the Addis regression

# %%
rng = np.random.default_rng(SPLIT_SEED)
train_sites = np.unique(sites_ocec[train_pos])
site_rows = {site: train_pos[sites_ocec[train_pos] == site] for site in train_sites}

bootstrap_rows = []
for label, spec in MODELS.items():
    for b in range(B_BOOT):
        chosen = rng.choice(train_sites, size=len(train_sites), replace=True)
        rows = np.concatenate([site_rows[site] for site in chosen])
        model = PLSRegression(n_components=spec['k'], scale=False).fit(
            spec['X'][rows], y_ocec[rows])
        predicted = model.predict(spec['X_addis']).ravel() / volume
        bootstrap_rows.append({
            'model': label, 'draw': b,
            **regression_metrics(hips[fixed_mask] / 10, predicted[fixed_mask]),
        })
bootstrap = pd.DataFrame(bootstrap_rows)
bootstrap.to_csv(TABLE_DIR / 'addis_bootstrap_draws.csv', index=False)

ci_rows = []
for label, group in bootstrap.groupby('model'):
    for metric in ('intercept', 'slope', 'RMSE'):
        low, mid, high = group[metric].quantile([.025, .5, .975])
        ci_rows.append({'model': label, 'metric': metric,
                        'point_estimate': MODELS[label]['point'][metric],
                        'boot_median': mid, 'ci_2.5': low, 'ci_97.5': high})
ci = pd.DataFrame(ci_rows)
ci.to_csv(TABLE_DIR / 'addis_bootstrap_ci.csv', index=False)
display(ci)

fig, ax = plt.subplots(figsize=(8.5, 5.0))
for offset, (label, group) in enumerate(bootstrap.groupby('model')):
    ax.hist(group['intercept'], bins=30, alpha=.6,
            label=f'{label} (point {MODELS[label]["point"]["intercept"]:.2f})')
ax.axvline(0, color='0.3', lw=1.2, ls='--')
ax.axvline(-4.17, color='#7F8C8D', lw=1.2, ls=':', label='deployed SPARTAN (−4.17)')
ax.set(xlabel='Addis intercept, MAC=10 (µg/m³)', ylabel='Bootstrap draws',
       title=f'Site-cluster bootstrap (B={B_BOOT}) of the Addis intercept',
       ylim=(0, None))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'bootstrap_intercepts.png', dpi=180, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 3. Where the corrected model's Addis scatter lives: D², Q, loading, season

# %%
diag_rows, diagnostics = [], {}
for label, spec in MODELS.items():
    model = spec['model']
    _, inverse_ss = score_metric(model)
    scores = project_scores(model, spec['X_addis'])
    d2 = mahalanobis_distance_squared(scores, inverse_ss)
    q = spectral_q_residual(model, spec['X_addis'])
    residual = spec['addis_ugm3'] - hips / 10
    # Diagnostics use all 239 HIPS pairs, not only the fixed 190-filter cohort.
    frame = pd.DataFrame({
        'MediaId': etad_eval['MediaId'].to_numpy(), 'season': etad_eval['season'].to_numpy(),
        'Fabs': hips, 'residual_ugm3': residual, 'D2': d2, 'Q': q,
    })
    diagnostics[label] = frame
    for driver in ('D2', 'Q', 'Fabs'):
        diag_rows.append({
            'model': label, 'driver': driver,
            'pearson_r_vs_residual': pearsonr(frame[driver], frame['residual_ugm3'])[0],
            'spearman_r_vs_residual': spearmanr(frame[driver], frame['residual_ugm3'])[0],
            'pearson_r_vs_abs_residual': pearsonr(frame[driver],
                                                  frame['residual_ugm3'].abs())[0],
        })
residual_drivers = pd.DataFrame(diag_rows)
residual_drivers.to_csv(TABLE_DIR / 'addis_residual_drivers.csv', index=False)
display(residual_drivers)

seasonal = (pd.concat([frame.assign(model=label) for label, frame in diagnostics.items()])
            .groupby(['model', 'season'])
            .agg(n=('residual_ugm3', 'size'), mean_residual=('residual_ugm3', 'mean'),
                 sd_residual=('residual_ugm3', 'std'), median_D2=('D2', 'median'),
                 median_Q=('Q', 'median'))
            .reset_index())
seasonal.to_csv(TABLE_DIR / 'addis_residuals_by_season.csv', index=False)
display(seasonal)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
for ax, (label, frame) in zip(axes.flat, diagnostics.items()):
    for season, info in ETHIOPIA_SEASONS.items():
        keep = frame['season'].eq(season)
        if keep.any():
            ax.scatter(frame.loc[keep, 'D2'], frame.loc[keep, 'residual_ugm3'],
                       s=22, alpha=.6, color=info['color'], label=season)
    ax.axhline(0, color='0.8', lw=.9)
    ax.set(xlabel='Mahalanobis D² in model score space', title=label, xlim=(0, None))
axes[0].set_ylabel('Addis residual: FTIR EC − HIPS/10 (µg/m³)')
axes[0].legend(fontsize=8)
fig.suptitle('Prediction residual vs distance from the calibration cloud', y=1.02)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'residual_vs_D2_by_season.png', dpi=180, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4. Hybrid cohort: OC/EC pre-filter + corrected-spectrum similarity

# %%
pool_meta = load_pool_metadata().merge(load_tor_loadings(), on=['Site', 'date'],
                                       how='left', validate='many_to_one')
eligible = (pool_meta['TOR_EC_loading_ug'].gt(0) & pool_meta['TOR_EC_ugm3'].gt(0)
            & pool_meta['TOR_OC_ugm3'].gt(0) & pool_meta['OC_EC_ratio'].notna()
            & pool_meta['AnalysisId'].astype(int).isin(pool_row_for_id))
candidates = (pool_meta[eligible].sort_values('OC_EC_ratio')
              .drop_duplicates('FilterId').head(N_OCEC_POOL).copy())

corrected_model = MODELS['AIRSpec df1=6 (k=5)']['model']
vip_weight = vip_scores(corrected_model) ** 2
vip_weight = vip_weight / vip_weight.sum()
addis_target = np.nanmedian(X_addis_corr, axis=0)

candidate_rows = [pool_row_for_id[int(a)] for a in candidates['AnalysisId']]
X_candidates = pool_npz['corrected'][candidate_rows].astype(float)
candidates['vip_weighted_rmse_to_Addis'] = np.sqrt(
    ((X_candidates - addis_target) ** 2) @ vip_weight)
hybrid = candidates.nsmallest(N_HYBRID, 'vip_weighted_rmse_to_Addis').copy()
hybrid[['AnalysisId', 'FilterId', 'Site', 'OC_EC_ratio',
        'vip_weighted_rmse_to_Addis']].to_csv(TABLE_DIR / 'hybrid_cohort.csv', index=False)
print(f'Hybrid cohort: {len(hybrid)} filters, {hybrid.Site.nunique()} sites, '
      f'OC/EC {hybrid.OC_EC_ratio.min():.2f}–{hybrid.OC_EC_ratio.max():.2f}')

X_hybrid = pool_npz['corrected'][
    [pool_row_for_id[int(a)] for a in hybrid['AnalysisId']]].astype(float)
y_hybrid = hybrid['TOR_EC_loading_ug'].to_numpy(float)
sites_hybrid = hybrid['Site'].to_numpy()

h_train, h_test = next(GroupShuffleSplit(
    n_splits=1, test_size=.20, random_state=SPLIT_SEED).split(hybrid, groups=sites_hybrid))
assert set(sites_hybrid[h_train]).isdisjoint(sites_hybrid[h_test])
curve = component_cv_curve(X_hybrid[h_train], y_hybrid[h_train], range(1, 41),
                           groups=sites_hybrid[h_train], n_splits=5, random_state=42)
k_hybrid, curve = select_first_major_minimum(curve)
curve.to_csv(TABLE_DIR / 'hybrid_component_cv_curve.csv', index=False)
hybrid_model = PLSRegression(n_components=k_hybrid, scale=False).fit(
    X_hybrid[h_train], y_hybrid[h_train])
hybrid_heldout = regression_metrics(
    y_hybrid[h_test], hybrid_model.predict(X_hybrid[h_test]).ravel())
hybrid_addis = hybrid_model.predict(X_addis_corr).ravel() / volume

comparison_rows = [{
    'model': 'Hybrid (low-OC/EC ∩ spectral, corrected)', 'k': k_hybrid,
    'heldout_TOR_R2': hybrid_heldout['R2'], 'heldout_TOR_slope': hybrid_heldout['slope'],
    'heldout_TOR_RMSE': hybrid_heldout['RMSE'],
    **{f'addis_{key}': value for key, value in regression_metrics(
        hips[fixed_mask] / 10, hybrid_addis[fixed_mask]).items()},
}]
ftir13 = pd.read_csv('output/tables/ftir13/addis_metrics_corrected.csv')
reference = ftir13[(ftir13['MAC_m2_g'].eq(10))
                   & ftir13['model'].eq('lowest-OCEC 800, AIRSpec df1=6')].iloc[0]
comparison_rows.append({
    'model': 'lowest-OCEC 800 (AIRSpec df1=6, ftir_13)', 'k': 5,
    'heldout_TOR_R2': .904, 'heldout_TOR_slope': 1.011, 'heldout_TOR_RMSE': 3.859,
    **{f'addis_{key}': reference[key]
       for key in ('n', 'slope', 'intercept', 'R2', 'RMSE', 'bias')},
})
hybrid_comparison = pd.DataFrame(comparison_rows)
hybrid_comparison.to_csv(TABLE_DIR / 'hybrid_vs_ocec_metrics.csv', index=False)
display(hybrid_comparison)

# %%
x = hips[fixed_mask] / 10
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True, sharey=True)
panels = [('Lowest-OCEC 800 (AIRSpec df1=6)',
           MODELS['AIRSpec df1=6 (k=5)']['addis_ugm3'][fixed_mask]),
          (f'Hybrid cohort (k={k_hybrid})', hybrid_addis[fixed_mask])]
y_min, y_max = 0.0, 0.0
for ax, (label, y) in zip(axes.flat, panels):
    y_min, y_max = min(y_min, np.nanmin(y)), max(y_max, np.nanmax(y))
    stats = regression_metrics(x, y)
    ax.scatter(x, y, s=22, alpha=.55, color='#34495E')
    hi = max(np.nanmax(x), np.nanmax(y))
    ax.plot([0, hi], [0, hi], '--', color='0.55', lw=1)
    fit_x = np.array([np.nanmin(x), np.nanmax(x)])
    ax.plot(fit_x, stats['slope'] * fit_x + stats['intercept'], color='#C0392B', lw=1.6)
    ax.set_title(label, fontsize=10)
    ax.text(.04, .96, f'y={stats["slope"]:.2f}x {stats["intercept"]:+.2f}\n'
                      f'R²={stats["R2"]:.3f}; RMSE={stats["RMSE"]:.2f}\nn={stats["n"]}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='0.8', alpha=.9))
    ax.set_xlabel('HIPS EC-equivalent, MAC=10 (µg/m³)')
axes[0].set_xlim(0, 1.04 * np.nanmax(x))
axes[0].set_ylim(min(0, 1.05 * y_min), 1.04 * y_max)
if y_min < 0:
    for ax in axes.flat:
        ax.axhline(0, color='0.85', lw=.8, zorder=0)
axes[0].set_ylabel('FTIR EC (µg/m³)')
fig.tight_layout()
fig.savefig(PLOT_DIR / 'hybrid_vs_ocec_crossplots.png', dpi=180, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Takeaways
#
# (filled in by the finalize step after execution)
