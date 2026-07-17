#!/usr/bin/env python3
"""Build the ftir_13 AIRSpec-corrected calibration comparison notebook (phase 3)."""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ftir_13_airspec_corrected_calibrations.ipynb"


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

from config import season_for_month
from data_matching import load_filter_data
from plotting import PlotConfig
from pls_transfer import (
    FTIRTransferPaths, load_current_pls_model, regression_metrics,
    component_cv_curve, select_first_major_minimum, vip_scores, vip_overlap_summary,
)

PlotConfig.set(sites='all', layout='individual', show_stats=True, show_1to1=True)
PATHS = FTIRTransferPaths.defaults()
P2_TABLES = Path('../ftir_hips_chem/output/tables')
TABLE_DIR = Path('output/tables/ftir_13')
PLOT_DIR = Path('output/plots/ftir_13')
CORRECTED = Path('output/corrected')
for directory in (TABLE_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
"""
    )

    md(
        """
# Do AIRSpec-baselined spectra change the Addis EC story? (EDF 6–8)

## tl;dr

TLDR_PLACEHOLDER

## Context & Methods

Satoshi's concrete instruction from the July 2026 meeting: run the real AIRSpec smoothing-spline
baseline (EDF ≈ 6–8 in the functional-group region) instead of the constant-offset shortcut that
`ftir_10` showed performs poorly. R is no longer available on this machine, so
`scripts/airspec_baseline.py` ports the exact APRLssb algorithm (Kuzmiakova, Dillner, Takahama
2016) to Python and is **validated against the surviving R output** for the ETAD spectra
(`spectra_baselined_AIRSPEC.csv`, DF1 = 6, DF2 = 4) — see `output/airspec_port_report.md`.
The validated port was then batch-applied to the full IMPROVE lot-248/251 pool (13,634 spectra)
and the Addis filters at DF1 = 6 and DF1 = 8 (DF2 = 4), covering the 1425–4000 cm⁻¹ region.

This notebook rebuilds, on identically-corrected spectra:

1. the **906-filter smoke EC calibration** (the meeting-described cohort), and
2. the **`ftir_11` lowest-OC/EC cohort calibration** with the *same locked site-held-out
   TOR split*,

then evaluates both on AIRSpec-corrected Addis spectra against HIPS EC-equivalent, exactly as
in `ftir_10`/`ftir_11` (fixed 190-filter cohort, MAC = 10 headline, MAC = 6 sensitivity).

### Key Assumptions

- The Python port stands in for AIRSpec/APRLssb; its validation error against the R ground
  truth is documented and orders of magnitude below spectral signal.
- Baseline correction restricts the usable region to 1425–4000 cm⁻¹ (2 spline segments,
  mean-stitched); anything below 1425 cm⁻¹ is out of scope by construction.
- Cohort membership and train/test splits are frozen from `ftir_10`/`ftir_11` — only the
  spectra change, so any metric shift is attributable to the baseline treatment.
- Addis replicate scans are corrected individually and then averaged per filter
  (correct-then-average), the transpose of the raw pipeline's average-then-correct; the
  smoothing spline is near-linear in the data for a fixed weight pattern, so the difference
  is second-order and is absorbed into the validation caveats.
"""
    )

    md("## Data\n\n### 1. Load corrected spectra and frozen cohorts")

    code(
        """
pool6 = np.load(CORRECTED / 'improve_pool_corrected_df6.npz', allow_pickle=True)
pool8 = np.load(CORRECTED / 'improve_pool_corrected_df8.npz', allow_pickle=True)
etad6_raw = np.load(CORRECTED / 'etad_corrected_df6.npz', allow_pickle=True)
etad8_raw = np.load(CORRECTED / 'etad_corrected_df8.npz', allow_pickle=True)
wn_region = pool6['wn'].astype(float)
assert np.allclose(wn_region, etad6_raw['wn'].astype(float))

pool_index = {int(a): i for i, a in enumerate(pool6['analysis_id'])}

def etad_by_media(npz):
    # ETAD corrections are per replicate scan; average corrected scans per MediaId
    # (correct-then-average, the transpose of the raw pipeline's average-then-correct).
    frame = pd.DataFrame(npz['corrected'].astype(float))
    frame.insert(0, 'MediaId', npz['media_id'].astype(int))
    return frame.groupby('MediaId').mean()

etad6_media = etad_by_media(etad6_raw)
etad8_media = etad_by_media(etad8_raw)
etad_index = {int(m): i for i, m in enumerate(etad6_media.index)}
print(f'Corrected pool: {pool6["corrected"].shape[0]} spectra × {len(wn_region)} wavenumbers '
      f'({wn_region.min():.0f}–{wn_region.max():.0f} cm⁻¹); '
      f'Addis corrected: {len(etad6_media)} filters '
      f'(from {etad6_raw["corrected"].shape[0]} scans)')

# Frozen Addis evaluation cohort (same construction as ftir_10/11).
raw_etad = pd.read_csv(PATHS.etad_dir / 'ETAD_FTIR_spectra.csv')
etad_meta = pd.read_csv(PATHS.etad_dir / 'ETAD_metadata.csv')
hips = pd.read_csv(PATHS.spartan_hips_primary, encoding='cp1252')
hips_etad = (hips[hips['Site'].eq('ETAD')][['FilterId', 'Fabs']]
             .drop_duplicates('FilterId')
             .rename(columns={'FilterId': 'ExternalFilterId'}))
etad_eval = (etad_meta.merge(hips_etad, on='ExternalFilterId', how='left',
                             validate='one_to_one'))
etad_eval = etad_eval[etad_eval['Fabs'].notna() & etad_eval['SampleVolume_m3'].gt(0) &
                      etad_eval['MediaId'].astype(int).isin(etad_index)].copy()
X_addis6 = etad6_media.loc[etad_eval['MediaId'].astype(int)].to_numpy(float)
X_addis8 = etad8_media.loc[etad_eval['MediaId'].astype(int)].to_numpy(float)
keep = ~np.isnan(X_addis6).any(axis=1)
etad_eval, X_addis6, X_addis8 = etad_eval[keep], X_addis6[keep], X_addis8[keep]
print(f'Addis evaluation (corrected + HIPS): n={len(etad_eval)}')

# Frozen cohorts.
current = load_current_pls_model(PATHS.ftir_dir, 'EC')
smoke_ids = current.analysis_ids.astype(int)
ocec_split = pd.read_csv('output/tables/ftir_11/locked_ocec_train_test_split.csv')
"""
    )

    md("### 2. Assemble corrected training matrices for both cohorts")

    code(
        """
def corrected_rows(analysis_ids, pool):
    rows, kept = [], []
    corrected = pool['corrected']
    for position, analysis_id in enumerate(analysis_ids):
        row = pool_index.get(int(analysis_id))
        if row is not None:
            rows.append(corrected[row])
            kept.append(position)
    return np.asarray(rows, dtype=float), np.asarray(kept, dtype=int)

X_smoke6, smoke_kept = corrected_rows(smoke_ids, pool6)
X_smoke8, _ = corrected_rows(smoke_ids, pool8)
y_smoke = current.y[smoke_kept]
sites_smoke = current.sites[smoke_kept]
finite_smoke = ~np.isnan(X_smoke6).any(axis=1)
X_smoke6, X_smoke8 = X_smoke6[finite_smoke], X_smoke8[finite_smoke]
y_smoke, sites_smoke = y_smoke[finite_smoke], sites_smoke[finite_smoke]

train_frame = ocec_split[ocec_split['split'].eq('train')].reset_index(drop=True)
test_frame = ocec_split[ocec_split['split'].ne('train')].reset_index(drop=True)
ocec_tor = pd.read_csv('output/tables/ftir_11/locked_ocec_train_test_split.csv')
ratio_lookup = pd.read_csv('output/tables/ftir_11/ocec_pool_audit.csv')

# TOR loadings come from ftir_11's candidate construction; rebuild the loading join.
similarity = pd.read_csv(P2_TABLES / 'pls_transfer/improve_full_pool_addis_similarity.csv')
similarity['date'] = pd.to_datetime(similarity['SampleDate'], format='mixed',
                                    errors='coerce').dt.normalize()
tor = pd.read_csv(
    PATHS.ftir_dir / 'local_db/tables/results_tor.csv',
    usecols=['Site', 'SampleDate', 'Parameter', 'Value', 'AverageFlowRate', 'ElapsedTime'])
tor_ec = tor[tor['Parameter'].eq('EC')].copy()
tor_ec['date'] = pd.to_datetime(tor_ec['SampleDate'], format='mixed',
                                errors='coerce').dt.normalize()
tor_ec = tor_ec.drop_duplicates(['Site', 'date'])
tor_ec['TOR_EC_loading_ug'] = (
    tor_ec['Value'] * (tor_ec['AverageFlowRate'] / 1000 * tor_ec['ElapsedTime']) / 1000)
loading = (similarity[['AnalysisId', 'Site', 'date']]
           .merge(tor_ec[['Site', 'date', 'TOR_EC_loading_ug']], on=['Site', 'date'],
                  how='left', validate='many_to_one')
           .set_index('AnalysisId')['TOR_EC_loading_ug'])

def cohort_matrices(frame, pool):
    X, kept = corrected_rows(frame['AnalysisId'].to_numpy(), pool)
    subset = frame.iloc[kept].copy()
    subset['TOR_EC_loading_ug'] = loading.reindex(subset['AnalysisId'].astype(int)).to_numpy()
    finite = (~np.isnan(X).any(axis=1)) & subset['TOR_EC_loading_ug'].gt(0).to_numpy()
    return X[finite], subset[finite]

X_train6, train6 = cohort_matrices(train_frame, pool6)
X_train8, train8 = cohort_matrices(train_frame, pool8)
X_test6, test6 = cohort_matrices(test_frame, pool6)
X_test8, test8 = cohort_matrices(test_frame, pool8)
assembly_audit = pd.DataFrame([
    {'cohort': 'smoke 906 corrected', 'n': len(y_smoke)},
    {'cohort': 'lowest-OC/EC train corrected', 'n': len(train6)},
    {'cohort': 'lowest-OC/EC site-held-out test corrected', 'n': len(test6)},
])
assembly_audit.to_csv(TABLE_DIR / 'corrected_cohort_assembly_audit.csv', index=False)
display(assembly_audit)
"""
    )

    md("## Results\n\n### 3. Refit both calibrations on corrected spectra (EDF 6 and 8)")

    code(
        """
def fit_first_major(X, y, sites, label):
    curve = component_cv_curve(X, y, range(1, 41), groups=sites, n_splits=5, random_state=42)
    k, curve = select_first_major_minimum(curve)
    curve['model'] = label
    model = PLSRegression(n_components=k, scale=False).fit(X, y)
    return model, k, curve

smoke6_model, smoke6_k, smoke6_curve = fit_first_major(
    X_smoke6, y_smoke, sites_smoke, 'smoke906 EDF6')
smoke8_model, smoke8_k, smoke8_curve = fit_first_major(
    X_smoke8, y_smoke, sites_smoke, 'smoke906 EDF8')
ocec6_model, ocec6_k, ocec6_curve = fit_first_major(
    X_train6, train6['TOR_EC_loading_ug'].to_numpy(float),
    train6['Site'].to_numpy(), 'lowOCEC EDF6')
ocec8_model, ocec8_k, ocec8_curve = fit_first_major(
    X_train8, train8['TOR_EC_loading_ug'].to_numpy(float),
    train8['Site'].to_numpy(), 'lowOCEC EDF8')
pd.concat([smoke6_curve, smoke8_curve, ocec6_curve, ocec8_curve]).to_csv(
    TABLE_DIR / 'corrected_component_cv_curves.csv', index=False)

tor_holdout = pd.DataFrame([
    {'model': f'Lowest-OC/EC corrected EDF6 (k={ocec6_k})',
     **regression_metrics(test6['TOR_EC_loading_ug'],
                          ocec6_model.predict(X_test6).ravel())},
    {'model': f'Lowest-OC/EC corrected EDF8 (k={ocec8_k})',
     **regression_metrics(test8['TOR_EC_loading_ug'],
                          ocec8_model.predict(X_test8).ravel())},
])
tor_holdout.to_csv(TABLE_DIR / 'corrected_site_held_out_tor_metrics.csv', index=False)
print(f'k: smoke EDF6={smoke6_k}, smoke EDF8={smoke8_k}, '
      f'OCEC EDF6={ocec6_k}, OCEC EDF8={ocec8_k}')
display(tor_holdout)
"""
    )

    md("### 4. Addis evaluation on corrected spectra")

    code(
        """
volume = etad_eval['SampleVolume_m3'].to_numpy(float)
etad_eval['EC_smoke_corr6_ugm3'] = smoke6_model.predict(X_addis6).ravel() / volume
etad_eval['EC_smoke_corr8_ugm3'] = smoke8_model.predict(X_addis8).ravel() / volume
etad_eval['EC_ocec_corr6_ugm3'] = ocec6_model.predict(X_addis6).ravel() / volume
etad_eval['EC_ocec_corr8_ugm3'] = ocec8_model.predict(X_addis8).ravel() / volume

prior = pd.read_csv(P2_TABLES / 'pls_calibration_phase2/addis_calibration_predictions.csv')
ocec_prior = pd.read_csv('output/tables/ftir_11/addis_ocec_predictions.csv')
merged = (etad_eval.merge(prior[['ExternalFilterId', 'EC_deployed_ugm3',
                                 'EC_smoke_906_ugm3']],
                          on='ExternalFilterId', how='left', validate='one_to_one')
          .merge(ocec_prior[['ExternalFilterId', 'EC_ocec_first_ugm3']],
                 on='ExternalFilterId', how='left', validate='one_to_one'))

model_columns = {
    'Deployed SPARTAN FTIR EC (raw)': 'EC_deployed_ugm3',
    'Smoke 906 (raw, ftir_10)': 'EC_smoke_906_ugm3',
    f'Smoke 906 corrected EDF6 (k={smoke6_k})': 'EC_smoke_corr6_ugm3',
    f'Smoke 906 corrected EDF8 (k={smoke8_k})': 'EC_smoke_corr8_ugm3',
    'Lowest-OC/EC (raw, ftir_11)': 'EC_ocec_first_ugm3',
    f'Lowest-OC/EC corrected EDF6 (k={ocec6_k})': 'EC_ocec_corr6_ugm3',
    f'Lowest-OC/EC corrected EDF8 (k={ocec8_k})': 'EC_ocec_corr8_ugm3',
}
common_mask = merged[list(model_columns.values()) + ['Fabs']].notna().all(axis=1)
print(f'Fixed common corrected cohort n={int(common_mask.sum())}')

rows = []
for mac in (6.0, 10.0):
    hips_ec = merged['Fabs'].to_numpy(float) / mac
    for name, column in model_columns.items():
        rows.append({'model': name, 'MAC_m2_g': mac,
                     **regression_metrics(hips_ec[common_mask],
                                          merged.loc[common_mask, column].to_numpy(float))})
metrics = pd.DataFrame(rows)
metrics.to_csv(TABLE_DIR / 'addis_corrected_calibration_metrics.csv', index=False)
merged[['MediaId', 'ExternalFilterId', 'Fabs'] + list(model_columns.values())].to_csv(
    TABLE_DIR / 'addis_corrected_predictions.csv', index=False)
display(metrics[metrics['MAC_m2_g'].eq(10.0)]
        [['model', 'n', 'slope', 'intercept', 'R2', 'RMSE', 'bias']])
"""
    )

    md("### 5. Crossplots and corrected-VIP comparison")

    code(
        """
common = merged.loc[common_mask].copy()
x = common['Fabs'].to_numpy(float) / 10.0
panels = [
    ('Smoke 906 (raw)', 'EC_smoke_906_ugm3'),
    (f'Smoke 906 corrected EDF6', 'EC_smoke_corr6_ugm3'),
    ('Lowest-OC/EC (raw)', 'EC_ocec_first_ugm3'),
    (f'Lowest-OC/EC corrected EDF6', 'EC_ocec_corr6_ugm3'),
]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), sharex=True, sharey=True)
for ax, (label, column) in zip(axes.flat, panels):
    y = common[column].to_numpy(float)
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
fig.suptitle('Raw vs AIRSpec-corrected calibrations on one fixed Addis cohort', y=1.0)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'addis_corrected_crossplots.png', dpi=180, bbox_inches='tight')
plt.show()

vip_corr = pd.DataFrame({
    'wavenumber_cm-1': wn_region,
    'smoke906_corrected_EDF6_VIP': vip_scores(smoke6_model),
    'lowest_OCEC_corrected_EDF6_VIP': vip_scores(ocec6_model),
})
vip_corr.to_csv(TABLE_DIR / 'corrected_vip_profiles.csv', index=False)
overlap = vip_overlap_summary(vip_scores(smoke6_model), vip_scores(ocec6_model))
pd.DataFrame([overlap]).to_csv(TABLE_DIR / 'corrected_vip_overlap.csv', index=False)
print('Corrected smoke vs corrected low-OC/EC VIP overlap:',
      {key: round(value, 3) for key, value in overlap.items()})

fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.plot(wn_region, vip_corr['smoke906_corrected_EDF6_VIP'], lw=1.1, color='#7F8C8D',
        label=f'Smoke 906 corrected (k={smoke6_k})')
ax.plot(wn_region, vip_corr['lowest_OCEC_corrected_EDF6_VIP'], lw=1.3, color='#8E44AD',
        label=f'Lowest-OC/EC corrected (k={ocec6_k})')
ax.axhline(1.0, color='0.6', lw=.8, ls=':')
ax.set_xlim(4000, 1400)
ax.set(xlabel='Wavenumber (cm⁻¹)', ylabel='VIP',
       title='VIP on AIRSpec-corrected spectra (EDF 6)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'corrected_vip_profiles.png', dpi=180, bbox_inches='tight')
plt.show()
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
