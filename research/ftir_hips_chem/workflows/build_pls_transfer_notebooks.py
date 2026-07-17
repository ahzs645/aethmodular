"""Build the three reader-facing notebooks for the July 2026 FTIR transfer study."""

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


SETUP = r"""import sys
# For notebooks inside research/ftir_hips_chem/:
sys.path.insert(0, './scripts')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression

# Config + data
from config import (
    SITES, PROCESSED_SITES_DIR, FILTER_DATA_PATH,
    AERONET_DATA_DIR, WEATHER_DATA_DIR, MAC_VALUE,
)

# Exclusions (the IMPROVE RDS cohort already carries its model exclusions; no new
# sample is silently removed here—eligibility and missing-reference rows are audited.)
from outliers import (
    EXCLUDED_SAMPLES, MANUAL_OUTLIERS,
    apply_exclusion_flags, apply_threshold_flags,
    get_clean_data, print_exclusion_summary,
)

# Data loading / matching
from data_matching import (
    load_aethalometer_data, load_filter_data,
    match_aeth_filter_data, match_all_parameters,
)
from etad_factors import load_etad_factor_contributions, match_etad_factors

# Plotting—importing applies the repository white-background default.
from plotting import PlotConfig, crossplots, timeseries, distributions, comparisons
from plotting.utils import calculate_regression_stats

from pls_transfer import (
    FTIRTransferPaths, load_current_pls_model, vip_scores,
    select_components_cv, nested_cv_predictions, regression_metrics,
    score_metric, project_scores, mahalanobis_distance_squared,
    pairwise_score_distance_squared, spectral_q_residual, offset_correct,
    vip_overlap_summary, summarize_vip_bands, spaced_peak_table,
)

PlotConfig.set(sites='all', layout='individual', show_stats=True, show_1to1=True)
PATHS = FTIRTransferPaths.defaults()
TABLE_DIR = Path('output/tables/pls_transfer')
PLOT_DIR = Path('output/plots/pls_transfer')
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
"""


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_model_diagnostics():
    cells = [code(SETUP)]
    cells += [
        md("""
        # Does the current IMPROVE EC calibration lean on OC-related spectral features?

        ## tl;dr

        _This summary is populated after the notebook's first validated execution._

        ## Context & Methods

        This notebook reconstructs the current July 7 EC and OC PLS models from their exported
        RDS components, verifies the Python reconstruction against the R fitted values, and
        computes VIP using Eq. (14) in Takahama et al. (2019). “OC-related” is defined operationally:
        a wavenumber is OC-important when the current OC model has VIP ≥ 1 there. This avoids
        claiming a unique chemical assignment where overlapping FTIR bands make one uncertain.

        ### Key Assumptions

        - The checked-in NPZ/CSV exports are lossless representations of the named RDS files.
        - VIP is interpreted post hoc; it does not prove that a band is uniquely causal.
        - The EC and OC models use the same 2722-point wavenumber grid.
        """),
        code("""
        source_inventory = PATHS.validate()
        display(source_inventory)
        if not source_inventory['exists'].all():
            raise FileNotFoundError(source_inventory.loc[~source_inventory['exists'], 'path'].tolist())

        ec = load_current_pls_model(PATHS.ftir_dir, 'EC')
        oc = load_current_pls_model(PATHS.ftir_dir, 'OC')
        assert np.allclose(ec.wavenumbers, oc.wavenumbers)

        reconstruction = pd.DataFrame([
            {'model': 'EC', 'n': len(ec.y), 'components': ec.chosen_n_components,
             'max_abs_difference_vs_R': ec.r_prediction_max_abs_error,
             'median_abs_difference_vs_R': ec.r_prediction_median_abs_error},
            {'model': 'OC', 'n': len(oc.y), 'components': oc.chosen_n_components,
             'max_abs_difference_vs_R': oc.r_prediction_max_abs_error,
             'median_abs_difference_vs_R': oc.r_prediction_median_abs_error},
        ])
        reconstruction.to_csv(TABLE_DIR / 'current_model_reconstruction_check.csv', index=False)
        display(reconstruction)
        """),
        md("""
        ## Results

        ### 1. VIP overlap directly tests whether EC relies on the same predictors as OC
        """),
        code("""
        ec_vip = vip_scores(ec.model)
        oc_vip = vip_scores(oc.model)
        overlap = vip_overlap_summary(ec_vip, oc_vip, threshold=1.0, top_n=200)
        overlap_table = pd.DataFrame([overlap])
        overlap_table.to_csv(TABLE_DIR / 'ec_oc_vip_overlap.csv', index=False)

        vip_profile = pd.DataFrame({
            'wavenumber_cm-1': ec.wavenumbers,
            'EC_VIP': ec_vip,
            'OC_VIP': oc_vip,
            'EC_important': ec_vip >= 1.0,
            'OC_important': oc_vip >= 1.0,
        })
        vip_profile.to_csv(TABLE_DIR / 'current_ec_oc_vip_profiles.csv', index=False)
        display(overlap_table)
        """),
        code("""
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        axes[0].plot(ec.wavenumbers, ec_vip, color=SITES['Addis_Ababa']['color'], lw=1.2, label='EC VIP')
        axes[0].plot(ec.wavenumbers, oc_vip, color='#3498DB', lw=1.0, alpha=0.8, label='OC VIP')
        axes[0].axhline(1.0, color='0.35', ls='--', lw=1, label='VIP = 1')
        axes[0].set_ylabel('VIP')
        axes[0].set_title('Current IMPROVE EC and OC models emphasize many of the same wavenumbers')
        axes[0].legend(ncol=3)

        shared = (ec_vip >= 1.0) & (oc_vip >= 1.0)
        axes[1].fill_between(ec.wavenumbers, 0, ec_vip, where=shared,
                             color='#8E44AD', alpha=0.6, label='VIP ≥ 1 in both models')
        axes[1].plot(ec.wavenumbers, ec_vip, color='0.25', lw=0.8)
        axes[1].axhline(1.0, color='0.45', ls='--', lw=1)
        axes[1].set_xlabel('Wavenumber (cm⁻¹)')
        axes[1].set_ylabel('EC VIP')
        axes[1].legend()
        for ax in axes:
            ax.set_xlim(ec.wavenumbers.max(), ec.wavenumbers.min())
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'current_ec_oc_vip_overlap.png', dpi=180, bbox_inches='tight')
        plt.show()
        """),
        md("""
        ### 2. Broad-band summaries are safer than assigning individual overlapping peaks
        """),
        code("""
        ec_bands = summarize_vip_bands(ec.wavenumbers, ec_vip).assign(model='EC')
        oc_bands = summarize_vip_bands(oc.wavenumbers, oc_vip).assign(model='OC')
        band_summary = pd.concat([ec_bands, oc_bands], ignore_index=True)
        band_summary.to_csv(TABLE_DIR / 'current_model_vip_band_summary.csv', index=False)

        peak_score = ec_vip * oc_vip
        shared_peaks = spaced_peak_table(ec.wavenumbers, peak_score, n_peaks=20)
        shared_peaks['EC_VIP'] = ec_vip[shared_peaks['index']]
        shared_peaks['OC_VIP'] = oc_vip[shared_peaks['index']]
        shared_peaks.to_csv(TABLE_DIR / 'ec_oc_shared_vip_peaks.csv', index=False)
        display(band_summary.pivot(index='band', columns='model', values='vip2_mass_fraction'))
        display(shared_peaks.head(12))
        """),
        md("""
        ## Takeaways

        _This interpretation is populated after execution and review of the saved overlap and band tables._
        """),
    ]
    return notebook(cells)


ETAD_LOAD = r"""
# One spectrum per physical filter: average replicate scans before model fitting/CV.
raw_etad = pd.read_csv(PATHS.etad_dir / 'ETAD_FTIR_spectra.csv')
etad_meta = pd.read_csv(PATHS.etad_dir / 'ETAD_metadata.csv')
wcols = sorted([c for c in raw_etad.columns if c not in ('SampleAnalysisId', 'MediaId')],
               key=lambda value: -float(value))
assert np.allclose(np.array([float(c) for c in wcols]), wn)
etad_spectra = raw_etad.groupby('MediaId', as_index=False)[wcols].mean()

# The primary file is newer and has 239 ETAD HIPS results versus 190 in the user-listed backup.
hips_primary = pd.read_csv(PATHS.spartan_hips_primary, encoding='cp1252')
hips_backup = pd.read_csv(PATHS.spartan_hips_backup, encoding='cp1252')
hips_fields = ['FilterId', 'Fabs', 'tau', 'DepositArea', 'Volume']
primary_etad = hips_primary[hips_primary['Site'] == 'ETAD'][hips_fields].drop_duplicates('FilterId')
backup_etad = hips_backup[hips_backup['Site'] == 'ETAD'][hips_fields].drop_duplicates('FilterId')
source_compare = primary_etad.merge(backup_etad, on='FilterId', suffixes=('_primary', '_backup'))
source_audit = pd.DataFrame([{
    'primary_nonnull': int(primary_etad['Fabs'].notna().sum()),
    'backup_nonnull': int(backup_etad['Fabs'].notna().sum()),
    'overlap_n': int(source_compare[['Fabs_primary', 'Fabs_backup']].dropna().shape[0]),
    'overlap_max_abs_difference': float(np.nanmax(np.abs(source_compare['Fabs_primary'] - source_compare['Fabs_backup']))),
    'primary_tau_formula_max_abs_difference': float(np.nanmax(np.abs(
        primary_etad['Fabs'] * primary_etad['Volume'] /
        (100.0 * primary_etad['DepositArea']) - primary_etad['tau']))),
}])

etad = (etad_spectra.merge(etad_meta, on='MediaId', how='left')
        .merge(primary_etad.rename(columns={'FilterId': 'ExternalFilterId'}),
               on='ExternalFilterId', how='left'))
etad['has_spectrum'] = etad[wcols].notna().all(axis=1)
etad['has_hips'] = etad['Fabs'].notna()
etad['positive_volume'] = etad['SampleVolume_m3'].gt(0)
etad['positive_deposit_area'] = etad['DepositArea'].gt(0)
etad_eval = etad[etad['has_spectrum'] & etad['has_hips'] & etad['positive_volume'] & etad['positive_deposit_area']].copy()
etad_eval['HIPS_tau'] = (
    etad_eval['Fabs'] * etad_eval['SampleVolume_m3'] / (100.0 * etad_eval['DepositArea'])
)
X_etad = etad_eval[wcols].to_numpy(float)
y_etad = etad_eval['Fabs'].to_numpy(float)
y_etad_tau = etad_eval['HIPS_tau'].to_numpy(float)
"""


IMPROVE_HIPS_LOAD = r"""
arrays = np.load(PATHS.ftir_dir / 'apps/apps_data.npz', allow_pickle=True)
wn = arrays['wn'].astype(float)
calibration_rows = pd.DataFrame({
    'row': np.arange(len(arrays['EC_id'])),
    'AnalysisId': arrays['EC_id'].astype(int),
    'FilterId': arrays['EC_fid'].astype(int),
    'Site': arrays['EC_site'].astype(str),
})
improve_hips_raw = pd.read_csv(
    PATHS.ftir_dir / 'local_db/tables/results_hips.csv',
    usecols=['MatchedFilterId', 'Parameter', 'Value', 'AverageFlowRate',
             'ElapsedTime', 'SampleDepositArea'],
)
improve_hips = (improve_hips_raw[improve_hips_raw['Parameter'].str.casefold() == 'fabs']
                .drop_duplicates('MatchedFilterId'))
improve_join = calibration_rows.merge(
    improve_hips[['MatchedFilterId', 'Value', 'AverageFlowRate',
                  'ElapsedTime', 'SampleDepositArea']],
    left_on='FilterId', right_on='MatchedFilterId', how='left', validate='one_to_one')
improve_join['SampleVolume_m3'] = (
    improve_join['AverageFlowRate'] / 1000.0 * improve_join['ElapsedTime']
)
improve_join['HIPS_tau'] = (
    improve_join['Value'] * improve_join['SampleVolume_m3'] /
    (100.0 * improve_join['SampleDepositArea'])
)
eligible = (improve_join['Value'].notna() & improve_join['SampleVolume_m3'].gt(0) &
            improve_join['SampleDepositArea'].gt(0) & improve_join['HIPS_tau'].notna())
X_improve_hips = arrays['EC_X'][improve_join.loc[eligible, 'row'].to_numpy()].astype(float)
y_improve_fabs = improve_join.loc[eligible, 'Value'].to_numpy(float)
y_improve_hips = improve_join.loc[eligible, 'HIPS_tau'].to_numpy(float)
site_improve_hips = improve_join.loc[eligible, 'Site'].to_numpy()
"""


def build_hips_transfer():
    cells = [code(SETUP)]
    cells += [
        md("""
        # Train on IMPROVE HIPS, predict Addis HIPS, then compare Addis-only VIP

        ## tl;dr

        _This summary is populated after the notebook's first validated execution._

        ## Context & Methods

        The response is HIPS filter optical depth, reconstructed as
        `tau = Fabs × sampled volume / (100 × deposit area)`. This puts the HIPS reference on a
        filter-loading basis comparable to FTIR absorbance; predictions are converted back to Fabs
        for stakeholder-facing evaluation. The result remains independent of the unresolved
        MAC = 6 versus 10 question.
        The IMPROVE component count is selected by holding out entire IMPROVE sites. The Addis-only
        result uses nested cross-validation and averages replicate FTIR scans before any split.

        ### Key Assumptions

        - IMPROVE and SPARTAN `Fabs` are comparable HIPS outputs; protocol differences remain a caveat.
        - The raw 2722-point spectra share a grid and are modeled with `scale=False`, matching the R models.
        - HIPS is an independent optical comparator, not absolute EC ground truth.
        """),
        md("## Data"),
        code(IMPROVE_HIPS_LOAD),
        code(ETAD_LOAD),
        code("""
        data_audit = pd.DataFrame([
            {'cohort': 'IMPROVE spectra available', 'n': len(calibration_rows)},
            {'cohort': 'IMPROVE matched to HIPS Fabs + volume + area', 'n': len(y_improve_hips)},
            {'cohort': 'Addis unique filters with spectra', 'n': len(etad_spectra)},
            {'cohort': 'Addis spectra + HIPS + positive volume', 'n': len(etad_eval)},
        ])
        data_audit.to_csv(TABLE_DIR / 'hips_transfer_data_audit.csv', index=False)
        source_audit.to_csv(TABLE_DIR / 'spartan_hips_source_reconciliation.csv', index=False)
        display(data_audit)
        display(source_audit)
        """),
        md("""
        ## Results

        ### 1. IMPROVE HIPS model: choose complexity by holding out whole sites
        """),
        code("""
        improve_k, improve_curve = select_components_cv(
            X_improve_hips, y_improve_hips, range(1, 26),
            groups=site_improve_hips, n_splits=5, random_state=42)
        improve_curve.to_csv(TABLE_DIR / 'improve_hips_group_cv_curve.csv', index=False)
        improve_model = PLSRegression(n_components=improve_k, scale=False).fit(X_improve_hips, y_improve_hips)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(improve_curve['n_components'], improve_curve['rmse_mean'],
                    yerr=improve_curve['rmse_sd'], color='#34495E', marker='o', ms=4)
        ax.axvline(improve_k, color=SITES['Addis_Ababa']['color'], ls='--', label=f'chosen k = {improve_k}')
        ax.set(xlabel='PLS components', ylabel='Site-held-out RMSE (HIPS optical depth τ)',
               title='IMPROVE HIPS component choice is based on geographic holdout')
        ax.legend(); fig.tight_layout()
        fig.savefig(PLOT_DIR / 'improve_hips_component_selection.png', dpi=180, bbox_inches='tight')
        plt.show()
        print('Chosen IMPROVE HIPS components:', improve_k)
        """),
        md("### 2. Transfer the IMPROVE HIPS model to Addis and benchmark an Addis-only model"),
        code("""
        transfer_tau = improve_model.predict(X_etad).ravel()
        transfer_prediction = (
            transfer_tau * 100.0 * etad_eval['DepositArea'].to_numpy() /
            etad_eval['SampleVolume_m3'].to_numpy()
        )
        transfer_metrics = regression_metrics(y_etad, transfer_prediction)

        addis_nested_tau, addis_nested_folds = nested_cv_predictions(
            X_etad, y_etad_tau, range(1, 16), outer_splits=5, inner_splits=4, random_state=42)
        addis_nested_prediction = (
            addis_nested_tau * 100.0 * etad_eval['DepositArea'].to_numpy() /
            etad_eval['SampleVolume_m3'].to_numpy()
        )
        addis_cv_metrics = regression_metrics(y_etad, addis_nested_prediction)
        addis_k, addis_curve = select_components_cv(
            X_etad, y_etad_tau, range(1, 16), n_splits=5, random_state=42)
        addis_model = PLSRegression(n_components=addis_k, scale=False).fit(X_etad, y_etad_tau)

        # Deliberately unit-mismatched sensitivity: direct Fabs ignores sample volume/area.
        naive_k, naive_curve = select_components_cv(
            X_improve_hips, y_improve_fabs, range(1, 26),
            groups=site_improve_hips, n_splits=5, random_state=42)
        naive_model = PLSRegression(n_components=naive_k, scale=False).fit(X_improve_hips, y_improve_fabs)
        naive_prediction = naive_model.predict(X_etad).ravel()
        naive_metrics = regression_metrics(y_etad, naive_prediction)

        metric_table = pd.DataFrame([
            {'model': 'IMPROVE HIPS → Addis external test', **transfer_metrics,
             'components': improve_k, 'validation': 'external site transfer'},
            {'model': 'Addis-only HIPS', **addis_cv_metrics,
             'components': addis_k, 'validation': 'nested 5-fold CV'},
            {'model': 'Naive direct-Fabs sensitivity (unit mismatched)', **naive_metrics,
             'components': naive_k, 'validation': 'external site transfer; diagnostic only'},
        ])
        metric_table.to_csv(TABLE_DIR / 'hips_transfer_model_metrics.csv', index=False)
        addis_nested_folds.to_csv(TABLE_DIR / 'addis_nested_cv_component_choices.csv', index=False)
        addis_curve.to_csv(TABLE_DIR / 'addis_hips_cv_curve.csv', index=False)
        transfer_predictions = etad_eval[['MediaId', 'ExternalFilterId', 'Fabs', 'HIPS_tau',
                                          'SampleVolume_m3', 'DepositArea']].copy()
        transfer_predictions['IMPROVE_HIPS_transfer_prediction'] = transfer_prediction
        transfer_predictions['Addis_HIPS_nested_CV_prediction'] = addis_nested_prediction
        transfer_predictions['naive_direct_Fabs_prediction'] = naive_prediction
        transfer_predictions.to_csv(TABLE_DIR / 'addis_hips_transfer_predictions.csv', index=False)
        display(metric_table)
        """),
        code("""
        def crossplot(ax, observed, predicted, title, color):
            metrics = regression_metrics(observed, predicted)
            lo = min(0.0, np.nanmin(observed), np.nanmin(predicted))
            hi = max(np.nanmax(observed), np.nanmax(predicted)) * 1.05
            ax.scatter(observed, predicted, s=32, alpha=0.6, color=color,
                       edgecolors='white', linewidths=0.3)
            ax.plot([lo, hi], [lo, hi], '--', color='0.45', lw=1)
            xline = np.array([lo, hi])
            ax.plot(xline, metrics['slope'] * xline + metrics['intercept'], color=color, lw=1.8)
            ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect='equal', xlabel='Observed HIPS Fabs',
                   ylabel='Predicted HIPS Fabs', title=title)
            ax.text(0.04, 0.96,
                    f"y = {metrics['slope']:.2f}x {metrics['intercept']:+.2f}\\n"
                    f"R² = {metrics['R2']:.3f}   RMSE = {metrics['RMSE']:.2f}\\n"
                    f"n = {metrics['n']}", transform=ax.transAxes, va='top',
                    bbox={'facecolor': 'white', 'edgecolor': '0.75', 'alpha': 0.9})

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        crossplot(axes[0], y_etad, transfer_prediction,
                  'IMPROVE HIPS model transferred to Addis', '#C0392B')
        crossplot(axes[1], y_etad, addis_nested_prediction,
                  'Addis-only HIPS model (nested CV)', SITES['Addis_Ababa']['color'])
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'improve_to_addis_hips_transfer.png', dpi=180, bbox_inches='tight')
        plt.show()
        """),
        md("### 3. Compare the two HIPS models' VIP profiles"),
        code("""
        improve_vip = vip_scores(improve_model)
        addis_vip = vip_scores(addis_model)
        vip_compare = vip_overlap_summary(improve_vip, addis_vip, threshold=1.0, top_n=200)
        vip_compare_table = pd.DataFrame([vip_compare])
        vip_compare_table.to_csv(TABLE_DIR / 'improve_addis_hips_vip_overlap.csv', index=False)

        hips_vip_profile = pd.DataFrame({
            'wavenumber_cm-1': wn,
            'IMPROVE_HIPS_VIP': improve_vip,
            'Addis_HIPS_VIP': addis_vip,
        })
        hips_vip_profile.to_csv(TABLE_DIR / 'improve_addis_hips_vip_profiles.csv', index=False)
        hips_bands = pd.concat([
            summarize_vip_bands(wn, improve_vip).assign(model='IMPROVE HIPS'),
            summarize_vip_bands(wn, addis_vip).assign(model='Addis HIPS'),
        ], ignore_index=True)
        hips_bands.to_csv(TABLE_DIR / 'improve_addis_hips_vip_bands.csv', index=False)
        display(vip_compare_table)
        display(hips_bands.pivot(index='band', columns='model', values='vip2_mass_fraction'))

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(wn, improve_vip, color='#34495E', lw=1.1, label=f'IMPROVE HIPS (k={improve_k})')
        ax.plot(wn, addis_vip, color=SITES['Addis_Ababa']['color'], lw=1.1,
                label=f'Addis HIPS (k={addis_k})')
        ax.axhline(1, color='0.4', ls='--', lw=1)
        ax.set_xlim(wn.max(), wn.min())
        ax.set(xlabel='Wavenumber (cm⁻¹)', ylabel='VIP',
               title='The HIPS models rely on different spectral structures when VIP overlap is low')
        ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
        fig.savefig(PLOT_DIR / 'improve_addis_hips_vip_comparison.png', dpi=180, bbox_inches='tight')
        plt.show()
        """),
        md("""
        ## Takeaways

        _This interpretation is populated after execution and review of external-transfer,
        nested-CV, and VIP-overlap results._
        """),
    ]
    return notebook(cells)


def build_analog_selection():
    cells = [code(SETUP)]
    cells += [
        md("""
        # Find IMPROVE analogs for Addis in HIPS-relevant PLS space and rebuild TOR EC

        ## tl;dr

        _This summary is populated after the notebook's first validated execution._

        ## Context & Methods

        This notebook searches all lot-248/251 IMPROVE spectra, not only smoke-classified days.
        Similarity is evaluated three ways: score-space Mahalanobis distance in the IMPROVE HIPS
        PLS model, residual spectral magnitude Q, and a baseline-offset-corrected VIP-weighted
        spectral mismatch. The top analogs with TOR EC references are then used to build an
        exploratory TOR calibration.

        ### Key Assumptions

        - Mahalanobis distance and Q are applicability diagnostics, not proof of low prediction error.
        - Analog selection is supervised by a HIPS model, so comparisons to HIPS are descriptive and
          not an independent validation of the selected TOR calibration.
        - Offset correction in 3900–4000 cm⁻¹ is a sensitivity approximation; it is not a replacement
          for a fully validated AIRSpec EDF 6–8 processing run on all 13,000 spectra.
        """),
        md("## Data and reference model"),
        code(IMPROVE_HIPS_LOAD),
        code(ETAD_LOAD),
        code("""
        improve_k, improve_curve = select_components_cv(
            X_improve_hips, y_improve_hips, range(1, 26),
            groups=site_improve_hips, n_splits=5, random_state=42)
        improve_model = PLSRegression(n_components=improve_k, scale=False).fit(X_improve_hips, y_improve_hips)
        improve_vip = vip_scores(improve_model)
        calibration_scores, inverse_score_ss = score_metric(improve_model)
        etad_scores = project_scores(improve_model, X_etad)
        etad_score_center = np.median(etad_scores, axis=0)
        etad_spectrum_target = np.nanmedian(offset_correct(X_etad, wn), axis=0)
        vip_weights = improve_vip**2 / np.sum(improve_vip**2)
        print(f'IMPROVE HIPS model: n={len(y_improve_hips)}, k={improve_k}; Addis evaluation n={len(etad_eval)}')
        """),
        md("""
        ## Results

        ### 1. Screen the complete 13k lot-matched pool in bounded chunks
        """),
        code("""
        pool_path = PATHS.ftir_dir / 'local_db/spectra_248_251.csv'
        metadata_cols = ['AnalysisId', 'FilterId', 'SampleDate', 'Site']
        pool_parts = []
        for chunk_number, chunk in enumerate(pd.read_csv(pool_path, chunksize=750), start=1):
            X_chunk = chunk[wcols].to_numpy(float)
            scores = project_scores(improve_model, X_chunk)
            q = spectral_q_residual(improve_model, X_chunk)
            d_center = pairwise_score_distance_squared(scores, etad_score_center, inverse_score_ss)
            corrected = offset_correct(X_chunk, wn)
            vip_rmse = np.sqrt(((corrected - etad_spectrum_target) ** 2) @ vip_weights)
            part = chunk[metadata_cols].copy()
            for component in range(scores.shape[1]):
                part[f'score_{component + 1}'] = scores[:, component]
            part['D2_to_Addis_centroid'] = d_center
            part['Q_residual'] = q
            part['VIP_weighted_spectral_RMSE'] = vip_rmse
            pool_parts.append(part)
        pool = pd.concat(pool_parts, ignore_index=True)

        score_columns = [f'score_{component + 1}' for component in range(improve_k)]
        pool_scores = pool[score_columns].to_numpy(float)
        # Whiten with the calibration score metric, then find each pool sample's nearest Addis filter.
        metric_root = np.linalg.cholesky(inverse_score_ss)
        nearest = cdist(pool_scores @ metric_root, etad_scores @ metric_root,
                        metric='sqeuclidean').min(axis=1)
        pool['D2_to_nearest_Addis'] = nearest
        pool['D2_leverage'] = mahalanobis_distance_squared(pool_scores, inverse_score_ss)
        pool['analog_rank_score'] = (
            pool['D2_to_nearest_Addis'].rank(pct=True) +
            pool['VIP_weighted_spectral_RMSE'].rank(pct=True)
        ) / 2
        pool.to_csv(TABLE_DIR / 'improve_full_pool_addis_similarity.csv', index=False)
        print(f'Full pool screened: {len(pool):,} spectra; unique filters={pool.FilterId.nunique():,}')
        display(pool.nsmallest(10, 'analog_rank_score')[metadata_cols + [
            'D2_to_nearest_Addis', 'Q_residual', 'VIP_weighted_spectral_RMSE', 'analog_rank_score']])
        """),
        md("### 2. Restrict only by explicit TOR-reference eligibility, then select the closest analogs"),
        code("""
        tor = pd.read_csv(
            PATHS.ftir_dir / 'local_db/tables/results_tor.csv',
            usecols=['Site', 'SampleDate', 'Parameter', 'Value', 'AverageFlowRate', 'ElapsedTime'],
        )
        tor_ec = tor[tor['Parameter'] == 'EC'].copy()
        tor_ec['date'] = pd.to_datetime(tor_ec['SampleDate'], errors='coerce').dt.normalize()
        tor_ec = tor_ec.drop_duplicates(['Site', 'date'])
        tor_ec['TOR_EC_loading_ug'] = (
            tor_ec['Value'] * (tor_ec['AverageFlowRate'] / 1000 * tor_ec['ElapsedTime']) / 1000
        )
        candidate_pool = pool.copy()
        candidate_pool['date'] = pd.to_datetime(candidate_pool['SampleDate'], errors='coerce').dt.normalize()
        candidate_pool = candidate_pool.merge(
            tor_ec[['Site', 'date', 'TOR_EC_loading_ug']],
            on=['Site', 'date'], how='left', validate='many_to_one')
        candidate_pool['eligible_TOR_reference'] = candidate_pool['TOR_EC_loading_ug'].gt(0)
        eligible_candidates = (candidate_pool[candidate_pool['eligible_TOR_reference']]
                               .sort_values('analog_rank_score')
                               .drop_duplicates('FilterId'))
        N_ANALOGS = min(400, len(eligible_candidates))
        analogs = eligible_candidates.head(N_ANALOGS).copy()
        analogs['analog_rank'] = np.arange(1, len(analogs) + 1)
        analogs.to_csv(TABLE_DIR / 'selected_improve_addis_analogs.csv', index=False)

        # Size-matched random cohorts quantify whether analog selection adds value beyond
        # simply refitting on any 400 lot-matched TOR samples.
        RANDOM_REPEATS = 10
        random_cohorts = []
        for repeat in range(RANDOM_REPEATS):
            cohort = eligible_candidates.sample(N_ANALOGS, random_state=1000 + repeat).copy()
            cohort['random_repeat'] = repeat + 1
            random_cohorts.append(cohort)
        random_cohort_index = pd.concat(random_cohorts, ignore_index=True)
        random_cohort_index[['random_repeat', 'AnalysisId', 'FilterId']].to_csv(
            TABLE_DIR / 'random_size_matched_cohort_index.csv', index=False)

        audit = pd.DataFrame([{
            'full_pool_spectra': len(pool),
            'full_pool_unique_filters': pool['FilterId'].nunique(),
            'TOR_eligible_unique_filters': eligible_candidates['FilterId'].nunique(),
            'selected_analogs': len(analogs),
            'selection_rule': 'mean percentile rank of nearest-Addis D2 and VIP-weighted RMSE',
        }])
        audit.to_csv(TABLE_DIR / 'analog_selection_audit.csv', index=False)
        display(audit)
        display(analogs[['analog_rank', 'AnalysisId', 'FilterId', 'Site', 'SampleDate',
                         'TOR_EC_loading_ug', 'D2_to_nearest_Addis',
                         'VIP_weighted_spectral_RMSE']].head(20))

        calibration_d2 = mahalanobis_distance_squared(calibration_scores, inverse_score_ss)
        etad_d2 = mahalanobis_distance_squared(etad_scores, inverse_score_ss)
        calibration_q = spectral_q_residual(improve_model, X_improve_hips)
        etad_q = spectral_q_residual(improve_model, X_etad)
        domain_rows = []
        for cohort, d2_values, q_values in [
            ('IMPROVE HIPS calibration', calibration_d2, calibration_q),
            ('Addis', etad_d2, etad_q),
            ('Full lot-248/251 pool', pool['D2_leverage'].to_numpy(), pool['Q_residual'].to_numpy()),
            ('Selected analogs', analogs['D2_leverage'].to_numpy(), analogs['Q_residual'].to_numpy()),
        ]:
            domain_rows.append({
                'cohort': cohort, 'n': len(d2_values),
                'D2_median': float(np.median(d2_values)), 'D2_p95': float(np.percentile(d2_values, 95)),
                'Q_median': float(np.median(q_values)), 'Q_p95': float(np.percentile(q_values, 95)),
            })
        domain_summary = pd.DataFrame(domain_rows)
        domain_summary.to_csv(TABLE_DIR / 'score_and_q_domain_summary.csv', index=False)
        display(domain_summary)
        """),
        md("### 3. Retrieve only selected spectra and fit the exploratory analog TOR model"),
        code("""
        selected_ids = set(analogs['AnalysisId'].astype(int)) | set(random_cohort_index['AnalysisId'].astype(int))
        selected_parts = []
        for chunk in pd.read_csv(pool_path, chunksize=750):
            keep = chunk['AnalysisId'].astype(int).isin(selected_ids)
            if keep.any():
                selected_parts.append(chunk.loc[keep, ['AnalysisId'] + wcols])
        selected_spectra = pd.concat(selected_parts, ignore_index=True).drop_duplicates('AnalysisId')
        analog_training = analogs.merge(selected_spectra, on='AnalysisId', how='inner', validate='one_to_one')
        X_analog = analog_training[wcols].to_numpy(float)
        y_analog = analog_training['TOR_EC_loading_ug'].to_numpy(float)

        analog_k, analog_curve = select_components_cv(
            X_analog, y_analog, range(1, 41), n_splits=5, random_state=42)
        analog_model = PLSRegression(n_components=analog_k, scale=False).fit(X_analog, y_analog)
        analog_curve.to_csv(TABLE_DIR / 'analog_tor_component_cv_curve.csv', index=False)
        print(f'Analog TOR calibration: n={len(y_analog)}, chosen k={analog_k}, '
              f'CV RMSE={analog_curve.rmse_mean.min():.3f} µg loading')

        random_results = []
        for repeat, cohort in random_cohort_index.groupby('random_repeat'):
            random_training = cohort.merge(selected_spectra, on='AnalysisId', how='inner', validate='one_to_one')
            random_model = PLSRegression(n_components=analog_k, scale=False).fit(
                random_training[wcols].to_numpy(float), random_training['TOR_EC_loading_ug'].to_numpy(float))
            random_ec = random_model.predict(X_etad).ravel() / etad_eval['SampleVolume_m3'].to_numpy()
            random_results.append({'random_repeat': int(repeat),
                                   **regression_metrics(etad_eval['Fabs'].to_numpy(float) / 10.0, random_ec)})
        random_baseline = pd.DataFrame(random_results)
        random_baseline.to_csv(TABLE_DIR / 'random_size_matched_tor_transfer_metrics_mac10.csv', index=False)
        display(random_baseline.describe().loc[['min', '50%', 'max'], ['slope', 'intercept', 'R2', 'RMSE', 'bias']])
        """),
        md("### 4. Compare current and analog TOR calibrations on Addis, with MAC sensitivity"),
        code("""
        current_ec = load_current_pls_model(PATHS.ftir_dir, 'EC')
        assert np.allclose(current_ec.wavenumbers, wn)
        etad_eval['EC_current_ugm3'] = current_ec.model.predict(X_etad).ravel() / etad_eval['SampleVolume_m3'].to_numpy()
        etad_eval['EC_analog_ugm3'] = analog_model.predict(X_etad).ravel() / etad_eval['SampleVolume_m3'].to_numpy()

        comparison_rows = []
        for mac in (6.0, 10.0):
            hips_ec = etad_eval['Fabs'].to_numpy(float) / mac
            for label, column in [('Current IMPROVE TOR EC', 'EC_current_ugm3'),
                                  ('Analog-selected TOR EC', 'EC_analog_ugm3')]:
                comparison_rows.append({'model': label, 'MAC_m2_g': mac,
                                        **regression_metrics(hips_ec, etad_eval[column])})
        tor_comparison = pd.DataFrame(comparison_rows)
        tor_comparison.to_csv(TABLE_DIR / 'addis_tor_calibration_comparison.csv', index=False)
        etad_eval[['MediaId', 'ExternalFilterId', 'Fabs', 'SampleVolume_m3',
                   'EC_current_ugm3', 'EC_analog_ugm3']].to_csv(
                       TABLE_DIR / 'addis_current_vs_analog_tor_predictions.csv', index=False)
        display(tor_comparison)
        """),
        md("### 5. Identify the HIPS-relevant spectral gaps and the full-pool samples that carry them"),
        code("""
        improve_median = np.nanmedian(offset_correct(X_improve_hips, wn), axis=0)
        addis_median = np.nanmedian(offset_correct(X_etad, wn), axis=0)
        gap = addis_median - improve_median
        peak_score = np.abs(gap) * improve_vip
        missing_peaks = spaced_peak_table(wn, peak_score, n_peaks=20, min_separation_cm1=20)
        idx = missing_peaks['index'].to_numpy(int)
        missing_peaks['Addis_median_offset_corrected'] = addis_median[idx]
        missing_peaks['IMPROVE_calibration_median_offset_corrected'] = improve_median[idx]
        missing_peaks['Addis_minus_IMPROVE'] = gap[idx]
        missing_peaks['IMPROVE_HIPS_VIP'] = improve_vip[idx]
        missing_peaks.to_csv(TABLE_DIR / 'addis_hips_relevant_missing_peaks.csv', index=False)
        display(missing_peaks.head(12))

        analog_vip = vip_scores(analog_model)
        vip_model_overlap = pd.DataFrame([
            {'comparison': 'Analog TOR vs IMPROVE HIPS',
             **vip_overlap_summary(analog_vip, improve_vip, threshold=1, top_n=200)},
            {'comparison': 'Current TOR vs IMPROVE HIPS',
             **vip_overlap_summary(vip_scores(current_ec.model), improve_vip, threshold=1, top_n=200)},
        ])
        vip_model_overlap.to_csv(TABLE_DIR / 'analog_current_tor_vs_hips_vip_overlap.csv', index=False)
        display(vip_model_overlap)
        """),
        code("""
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))
        axes[0].hist(mahalanobis_distance_squared(calibration_scores, inverse_score_ss),
                     bins=40, density=True, alpha=0.6, label='IMPROVE HIPS calibration')
        axes[0].hist(mahalanobis_distance_squared(etad_scores, inverse_score_ss),
                     bins=40, density=True, alpha=0.6, label='Addis')
        axes[0].set(xlabel='Score-space D² / leverage', ylabel='Density',
                    title='Addis position in IMPROVE HIPS score space')
        axes[0].legend(fontsize=8)

        axes[1].plot(wn, addis_median, color=SITES['Addis_Ababa']['color'], lw=1.4,
                     label='Addis median')
        axes[1].plot(wn, improve_median, color='#34495E', lw=1.1,
                     label='IMPROVE HIPS calibration median')
        top_analog_ids = set(analogs.head(50)['AnalysisId'].astype(int))
        top_analog_spectra = analog_training[analog_training['AnalysisId'].astype(int).isin(top_analog_ids)][wcols].to_numpy(float)
        axes[1].plot(wn, np.nanmedian(offset_correct(top_analog_spectra, wn), axis=0),
                     color='#27AE60', lw=1.1, label='Top-50 full-pool analog median')
        axes[1].set_xlim(4000, 1500)
        axes[1].set(xlabel='Wavenumber (cm⁻¹)', ylabel='Offset-corrected absorbance',
                    title='Full-pool analogs recover part of the Addis spectral shape')
        axes[1].legend(fontsize=8)

        mac = 10.0
        hips_ec = etad_eval['Fabs'].to_numpy(float) / mac
        axes[2].scatter(hips_ec, etad_eval['EC_current_ugm3'], s=24, alpha=0.5,
                        color='#34495E', label='Current TOR model')
        axes[2].scatter(hips_ec, etad_eval['EC_analog_ugm3'], s=24, alpha=0.5,
                        color='#27AE60', label='Analog TOR model')
        hi = max(hips_ec.max(), etad_eval['EC_current_ugm3'].max(), etad_eval['EC_analog_ugm3'].max())
        lo = min(0, etad_eval['EC_current_ugm3'].min(), etad_eval['EC_analog_ugm3'].min())
        axes[2].plot([lo, hi], [lo, hi], '--', color='0.5')
        axes[2].set(xlim=(lo, hi), ylim=(lo, hi), xlabel='HIPS EC-equivalent, MAC=10 (µg/m³)',
                    ylabel='FTIR TOR-equivalent EC (µg/m³)',
                    title='Analog selection changes the Addis EC projection')
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'full_pool_analog_selection_summary.png', dpi=180, bbox_inches='tight')
        plt.show()
        """),
        md("""
        ## Takeaways

        _This interpretation is populated after execution and review of the analog audit,
        MAC sensitivity table, and full-pool spectral-gap results._
        """),
    ]
    return notebook(cells)


def main():
    outputs = {
        "ftir_07_current_pls_vip_diagnostics.ipynb": build_model_diagnostics(),
        "ftir_08_hips_transfer_and_vip.ipynb": build_hips_transfer(),
        "ftir_09_improve_analog_selection.ipynb": build_analog_selection(),
    }
    for name, nb in outputs.items():
        destination = ROOT / name
        nbf.write(nb, destination)
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
