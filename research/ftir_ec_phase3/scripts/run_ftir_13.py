# %% [markdown]
# # ftir_13 — Do real AIRSpec (EDF 6–8) baselines change the Addis story?
#
# ## tl;dr
#
# (filled in by the finalize step after execution)
#
# ## Context & Methods
#
# Every phase-2 negative result carried the caveat "constant offset correction is not AIRSpec
# baselining." That caveat is now discharged: `scripts/airspec_baseline.py` is a Python port of
# the APRLssb segmented smoothing-spline baseline (R `smooth.spline`, all-knots, df-matched),
# validated against the actual R run on the 319 ETAD scans to a worst per-spectrum deviation of
# **6×10⁻⁷ absorbance** (`output/airspec_port_report.md`). It was applied to all 13,634
# lot-248/251 IMPROVE pool spectra and all ETAD scans at DF1 = 6 and DF1 = 8 (DF2 = 4),
# covering Satoshi's "EDF 6–8" guidance; corrected spectra span 1425–4000 cm⁻¹.
#
# Three questions, all under the phase-2/3 locked protocol (site-disjoint TOR test split with
# the ftir_10 seed, first-major-minimum component rule, Addis HIPS never used for selection):
#
# 1. Does the **lowest-OCEC 800** TOR calibration (ftir_11, the best cohort so far) improve on
#    AIRSpec-corrected spectra?
# 2. Does the **906-sample smoke** calibration's Addis intercept (−6.91 raw) move?
# 3. Does the **IMPROVE HIPS → Addis transfer gap** (ftir_08: median predicted Fabs 15.0 vs
#    47.1 observed) shrink when both sides are baselined?
#
# Plus one ftir_12 follow-up: the 1600-band peak center and the (now-baselined) nitro window
# on corrected Addis spectra.

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
    PATHS, load_addis_evaluation, load_tor_loadings, load_pool_metadata,
)
from pls_transfer import (
    load_current_pls_model, component_cv_curve, select_first_major_minimum,
    regression_metrics,
)

TABLE_DIR = Path('output/tables/ftir13')
PLOT_DIR = Path('output/plots/ftir13')
for directory in (TABLE_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SPLIT_SEED = 20260717
CORRECTED_DIR = Path('output/corrected')

# %% [markdown]
# ## Data
#
# ### 1. Load corrected spectra and align the grids

# %%
pool_corrected = {
    df1: np.load(CORRECTED_DIR / f'improve_pool_corrected_df{df1}.npz', allow_pickle=True)
    for df1 in (6, 8)
}
etad_corrected = {
    df1: np.load(CORRECTED_DIR / f'etad_corrected_df{df1}.npz', allow_pickle=True)
    for df1 in (6, 8)
}
wn_corrected = pool_corrected[6]['wn'].astype(float)
for df1 in (6, 8):
    assert np.allclose(pool_corrected[df1]['wn'].astype(float), wn_corrected)
    assert np.allclose(etad_corrected[df1]['wn'].astype(float), wn_corrected)

pool_row_for_id = {
    int(analysis_id): row
    for row, analysis_id in enumerate(pool_corrected[6]['analysis_id'].astype(int))
}

# Average corrected ETAD scans per physical filter (MediaId), as everywhere else.
etad_media = {}
for df1 in (6, 8):
    media = pd.DataFrame(etad_corrected[df1]['corrected'].astype(float))
    media['MediaId'] = etad_corrected[df1]['media_id'].astype(int)
    etad_media[df1] = media.groupby('MediaId').mean()

etad_eval, X_etad_raw, wn_full = load_addis_evaluation(season_for_month)
volume = etad_eval['SampleVolume_m3'].to_numpy(float)
X_addis = {
    df1: etad_media[df1].loc[etad_eval['MediaId'].astype(int)].to_numpy(float)
    for df1 in (6, 8)
}

phase2_predictions = pd.read_csv(
    Path('..') / 'ftir_hips_chem' / 'output' / 'tables' / 'pls_calibration_phase2'
    / 'addis_calibration_predictions.csv')
fixed_media = set(phase2_predictions.dropna(axis=0, how='any')['MediaId'])
fixed_mask = etad_eval['MediaId'].isin(fixed_media).to_numpy()
print(f'Corrected grid: {wn_corrected.size} wavenumbers '
      f'({wn_corrected.min():.0f}–{wn_corrected.max():.0f} cm⁻¹); '
      f'Addis eval n={len(etad_eval)}, fixed cohort n={fixed_mask.sum()}')

# %% [markdown]
# ### 2. Rebuild the calibration cohorts on the corrected grid

# %%
def corrected_pool_rows(analysis_ids, df1):
    rows = [pool_row_for_id[int(v)] for v in analysis_ids]
    return pool_corrected[df1]['corrected'][rows].astype(float)

ocec = pd.read_csv('output/tables/ftir11/lowest_ocec_800_cohort.csv')
ocec = ocec[ocec['AnalysisId'].astype(int).isin(pool_row_for_id)].copy()

smoke = load_current_pls_model(PATHS.ftir_dir, 'EC')
smoke_in_pool = np.array([int(v) in pool_row_for_id for v in smoke.analysis_ids])
smoke_ids = smoke.analysis_ids[smoke_in_pool]
smoke_audit = pd.DataFrame([{
    'ocec_cohort_n': len(ocec),
    'smoke_cohort_n': len(smoke.analysis_ids),
    'smoke_found_in_pool': int(smoke_in_pool.sum()),
}])
smoke_audit.to_csv(TABLE_DIR / 'cohort_alignment_audit.csv', index=False)
display(smoke_audit)

# %% [markdown]
# ## Results
#
# ### 3. Question 1 — the lowest-OCEC 800 cohort on corrected spectra

# %%
def locked_fit(X, y, sites, label):
    splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SPLIT_SEED)
    train_pos, test_pos = next(splitter.split(X, groups=sites))
    assert set(sites[train_pos]).isdisjoint(sites[test_pos])
    curve = component_cv_curve(X[train_pos], y[train_pos], range(1, 41),
                               groups=sites[train_pos], n_splits=5, random_state=42)
    k, curve = select_first_major_minimum(curve)
    curve['model'] = label
    model = PLSRegression(n_components=k, scale=False).fit(X[train_pos], y[train_pos])
    heldout = regression_metrics(y[test_pos], model.predict(X[test_pos]).ravel())
    return model, k, curve, heldout

ocec_y = ocec['TOR_EC_loading_ug'].to_numpy(float)
ocec_sites = ocec['Site'].to_numpy()
ocec_fits = {}
for df1 in (6, 8):
    X = corrected_pool_rows(ocec['AnalysisId'], df1)
    ocec_fits[df1] = locked_fit(X, ocec_y, ocec_sites, f'lowest-OCEC 800, AIRSpec df1={df1}')
    model, k, _, heldout = ocec_fits[df1]
    print(f'OCEC-800 df1={df1}: k={k}, held-out TOR RMSE={heldout["RMSE"]:.2f}, '
          f'R2={heldout["R2"]:.3f}, slope={heldout["slope"]:.3f}')

# %% [markdown]
# ### 4. Question 2 — the 906-sample smoke calibration on corrected spectra

# %%
smoke_fits = {}
smoke_y = smoke.y[smoke_in_pool]
smoke_sites = smoke.sites[smoke_in_pool]
for df1 in (6, 8):
    X = corrected_pool_rows(smoke_ids, df1)
    curve = component_cv_curve(X, smoke_y, range(1, 41),
                               groups=smoke_sites, n_splits=5, random_state=42)
    k, curve = select_first_major_minimum(curve)
    model = PLSRegression(n_components=k, scale=False).fit(X, smoke_y)
    smoke_fits[df1] = (model, k)
    print(f'Smoke-906 df1={df1}: k={k}')

# %% [markdown]
# ### 5. Addis metrics for every corrected model, against the raw baselines

# %%
addis_rows = []
raw_ftir11 = pd.read_csv('output/tables/ftir11/addis_metrics.csv')
for _, row in raw_ftir11[
    raw_ftir11['model'].isin(['lowest-OCEC 800', 'Deployed SPARTAN FTIR EC'])
    & raw_ftir11['cohort'].eq('fixed phase-2 cohort')
].iterrows():
    addis_rows.append({'model': row['model'] + ' (raw spectra)',
                       'MAC_m2_g': row['MAC_m2_g'],
                       **{key: row[key] for key in
                          ('n', 'slope', 'intercept', 'R2', 'RMSE', 'bias')}})

hips = etad_eval['Fabs'].to_numpy(float)
prediction_export = etad_eval[['MediaId', 'ExternalFilterId', 'Fabs']].copy()
for df1 in (6, 8):
    predictions = {
        f'lowest-OCEC 800, AIRSpec df1={df1}':
            ocec_fits[df1][0].predict(X_addis[df1]).ravel() / volume,
        f'Smoke 906, AIRSpec df1={df1}':
            smoke_fits[df1][0].predict(X_addis[df1]).ravel() / volume,
    }
    for label, values in predictions.items():
        prediction_export[label] = values
        for mac in (10.0, 6.0):
            addis_rows.append({'model': label, 'MAC_m2_g': mac,
                               **regression_metrics(hips[fixed_mask] / mac,
                                                    values[fixed_mask])})
addis_metrics = pd.DataFrame(addis_rows)
addis_metrics.to_csv(TABLE_DIR / 'addis_metrics_corrected.csv', index=False)
prediction_export.to_csv(TABLE_DIR / 'addis_predictions_corrected.csv', index=False)
display(addis_metrics[addis_metrics['MAC_m2_g'].eq(10)]
        [['model', 'n', 'slope', 'intercept', 'R2', 'RMSE']])

# %% [markdown]
# ### 6. Question 3 — the IMPROVE HIPS → Addis transfer on corrected spectra

# %%
arrays = np.load(PATHS.ftir_dir / 'apps/apps_data.npz', allow_pickle=True)
calibration_rows = pd.DataFrame({
    'AnalysisId': arrays['EC_id'].astype(int),
    'FilterId': arrays['EC_fid'].astype(int),
    'Site': arrays['EC_site'].astype(str),
})
improve_hips_raw = pd.read_csv(
    PATHS.ftir_dir / 'local_db/tables/results_hips.csv',
    usecols=['MatchedFilterId', 'Parameter', 'Value', 'AverageFlowRate',
             'ElapsedTime', 'SampleDepositArea'])
improve_hips = (improve_hips_raw[improve_hips_raw['Parameter'].str.casefold().eq('fabs')]
                .drop_duplicates('MatchedFilterId'))
improve_join = calibration_rows.merge(
    improve_hips[['MatchedFilterId', 'Value', 'AverageFlowRate',
                  'ElapsedTime', 'SampleDepositArea']],
    left_on='FilterId', right_on='MatchedFilterId', how='left', validate='one_to_one')
improve_join['SampleVolume_m3'] = (
    improve_join['AverageFlowRate'] / 1000.0 * improve_join['ElapsedTime'])
improve_join['HIPS_tau'] = (
    improve_join['Value'] * improve_join['SampleVolume_m3'] /
    (100.0 * improve_join['SampleDepositArea']))
hips_eligible = (improve_join['Value'].notna() & improve_join['SampleVolume_m3'].gt(0)
                 & improve_join['SampleDepositArea'].gt(0) & improve_join['HIPS_tau'].notna()
                 & improve_join['AnalysisId'].isin(pool_row_for_id))

# SPARTAN's own tau uses the HIPS-file Volume: tau = Fabs × Volume / (100 × DepositArea).
tau_to_fabs = (100.0 * etad_eval['DepositArea'].to_numpy(float)
               / etad_eval['Volume'].to_numpy(float))

transfer_rows = []
for df1 in (6, 8):
    X = corrected_pool_rows(improve_join.loc[hips_eligible, 'AnalysisId'], df1)
    y_tau = improve_join.loc[hips_eligible, 'HIPS_tau'].to_numpy(float)
    sites_h = improve_join.loc[hips_eligible, 'Site'].to_numpy()
    model, k, _, heldout = locked_fit(X, y_tau, sites_h, f'IMPROVE HIPS tau df1={df1}')
    fabs_predicted = model.predict(X_addis[df1]).ravel() * tau_to_fabs
    transfer_rows.append({
        'model': f'IMPROVE HIPS tau, AIRSpec df1={df1}', 'n_train_pool': int(len(y_tau)),
        'k': k, 'heldout_tau_R2': heldout['R2'], 'heldout_tau_slope': heldout['slope'],
        'addis_n': len(etad_eval),
        'addis_median_Fabs_predicted': float(np.median(fabs_predicted)),
        'addis_median_Fabs_observed': float(np.median(hips)),
        **{f'addis_{key}': value for key, value in
           regression_metrics(hips, fabs_predicted).items() if key != 'n'},
    })
transfer = pd.DataFrame(transfer_rows)
transfer.to_csv(TABLE_DIR / 'hips_transfer_corrected.csv', index=False)
display(transfer[['model', 'k', 'heldout_tau_R2', 'addis_median_Fabs_predicted',
                  'addis_median_Fabs_observed', 'addis_bias', 'addis_RMSE']])

# %% [markdown]
# ### 7. ftir_12 follow-up on corrected Addis spectra: 1600-band center and nitro window

# %%
from phase3_common import band_center  # same definition ftir_12 uses
from pls_transfer import local_continuum_peak_height
from scipy.stats import pearsonr, spearmanr

# Both DF1 values share DF2 = 4, and everything below 1820 cm⁻¹ comes from segment 2 —
# so the 1500–1700 cm⁻¹ diagnostics are identical for df1 = 6 and 8 by construction.
band_rows = []
for df1 in (6, 8):
    corrected = X_addis[df1]
    center, height, edge = band_center(corrected, wn_corrected)
    ch = local_continuum_peak_height(
        corrected, wn_corrected, (2800, 3000), (3050, 3150), (2650, 2750))
    ring_1520_1560 = local_continuum_peak_height(
        corrected, wn_corrected, (1520, 1560), (1570, 1600), (1490, 1510))
    usable = (height > 5e-4) & ~edge
    normalized = usable & (ch > 1e-4)
    band_rows.append({
        'spectra': f'AIRSpec df1={df1}', 'n_usable': int(usable.sum()),
        'center_p25': float(np.quantile(center[usable], .25)),
        'center_median': float(np.median(center[usable])),
        'center_p75': float(np.quantile(center[usable], .75)),
        # Raw heights share a loading common cause; the CH-normalized rank correlation
        # asks whether the 1520–1560 feature rises with the 1600 band beyond loading.
        'r_heights_1600_vs_1520_1560': float(
            pearsonr(height[usable], ring_1520_1560[usable])[0]),
        'spearman_CHnorm_1600_vs_1520_1560': float(
            spearmanr(height[normalized] / ch[normalized],
                      ring_1520_1560[normalized] / ch[normalized])[0]),
        'n_normalized': int(normalized.sum()),
    })
band_followup = pd.DataFrame(band_rows)
band_followup.to_csv(TABLE_DIR / 'band1600_on_corrected.csv', index=False)
display(band_followup)

# %% [markdown]
# ### 8. Crossplots on the fixed cohort: raw-spectra models vs AIRSpec models

# %%
x = hips[fixed_mask] / 10.0
panels = [
    ('Deployed SPARTAN (raw)', etad_eval['EC_deployed_ugm3'].to_numpy(float)[fixed_mask]),
    ('Lowest-OCEC 800 (raw, ftir_11)',
     pd.read_csv('output/tables/ftir11/addis_predictions.csv')['EC_lowest_ocec_800_ugm3']
     .to_numpy(float)[fixed_mask]),
    ('Lowest-OCEC 800 (AIRSpec df1=6)',
     prediction_export['lowest-OCEC 800, AIRSpec df1=6'].to_numpy(float)[fixed_mask]),
    ('Smoke 906 (AIRSpec df1=6)',
     prediction_export['Smoke 906, AIRSpec df1=6'].to_numpy(float)[fixed_mask]),
]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharex=True, sharey=True)
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
# Anchor at the origin: x (HIPS) is non-negative by construction; y drops below zero
# only when a calibration genuinely predicts negative EC, in which case a zero line
# marks the origin instead of hiding the negative predictions.
axes[0].set_xlim(0, 1.04 * np.nanmax(x))
axes[0].set_ylim(min(0, 1.05 * y_min), 1.04 * y_max)
if y_min < 0:
    for ax in axes.flat:
        ax.axhline(0, color='0.85', lw=.8, zorder=0)
axes[0].set_ylabel('FTIR EC (µg/m³)')
fig.suptitle('AIRSpec-corrected calibrations on the fixed Addis cohort', y=1.02)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'addis_crossplots_airspec.png', dpi=180, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Takeaways
#
# (filled in by the finalize step after execution)
