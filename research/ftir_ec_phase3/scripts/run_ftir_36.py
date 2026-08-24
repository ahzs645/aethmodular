# %% [markdown]
# # ftir_36 — Leakage-safe domain-invariant PLS transfer
#
# ## tl;dr
#
# (filled in by the finalize step after execution)
#
# ## Context & Methods
#
# This is the preregistered follow-up identified by ftir_35. Domain-invariant PLS
# (di-PLS; Nikzad-Langerodi et al., 2018, DOI 10.1021/acs.analchem.8b00498)
# aligns source and target latent distributions while fitting responses only in the
# labeled source domain.
#
# ### Key assumptions and leakage controls
#
# - Three source configurations were frozen before this run: the Addis winner, the
#   common two-city candidate, and the screened Delhi winner.
# - The ordinary PLS component count is selected only within the 80% IMPROVE source
#   training partition using site-grouped CV.
# - The domain-penalty multiplier is selected by treating held-out IMPROVE sites as
#   unlabeled pseudo-targets. The untouched 20% IMPROVE sites are an outer validation.
# - Addis/Delhi spectra may enter as unlabeled target covariates. Their HIPS values are
#   not read until the protocol JSON has been written.
# - Adjacent channels are averaged in fixed blocks of eight before any fitting. This
#   reduces 2,002–2,722 channels to 251–341 features, making the published repeated
#   covariance eigendecomposition tractable. Every comparator receives the same bins.
# - Final evaluation is on the 14 Addis and 26 Delhi reconstructed filters that were
#   never used in the five-site screen. The larger augmented target clouds are used only
#   to estimate unlabeled target means/covariances.

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, './scripts')

from phase3_common import PATHS, PHASE2_TABLES, load_pool_metadata, load_pool_spectra, load_tor_loadings
from calibration_modes import protocol_train_mask
from domain_invariant_pls import block_average, fit_domain_invariant_pls
from pls_transfer import component_cv_curve, select_first_major_minimum

ROOT = Path('.')
REPO = Path.cwd().parents[1]
TARGETS = REPO / 'calibration_explorer/targets'
OUT = ROOT / 'output/tables/ftir36'
OUT.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 8
MAX_COMPONENTS = 20
PSEUDO_TARGET_FOLDS = 3
MULTIPLIERS = np.array([0.0, 0.1, 0.3, 1.0, 3.0])
SAVGOL = dict(window_length=11, polyorder=2, deriv=2)

CONFIGS = {
    'addis_winner': {'cohort': 'ocec', 'cutoff': 440, 'spectra': 'airspec'},
    'common_candidate': {'cohort': 'analogs', 'cutoff': 440, 'spectra': 'deriv2'},
    'delhi_winner': {'cohort': 'analogs', 'cutoff': 530, 'spectra': 'deriv2'},
}
TARGET_SPECS = {
    'addis_holdout': {
        'unlabeled': 'addis_augmented',
        'evaluation': 'addis_reconstructed_holdout',
    },
    'delhi_holdout': {
        'unlabeled': 'indh_augmented',
        'evaluation': 'indh_reconstructed_holdout',
    },
}


def metrics(observed, predicted):
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    slope, intercept = np.polyfit(observed, predicted, 1)
    fitted = slope * observed + intercept
    ss_total = np.sum((predicted - predicted.mean()) ** 2)
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'R2': float(1 - np.sum((predicted - fitted) ** 2) / ss_total)
        if ss_total > 0 else np.nan,
        'RMSE_1to1': float(np.sqrt(np.mean((predicted - observed) ** 2))),
        'mean_bias': float(np.mean(predicted - observed)),
    }


def bootstrap_rmse_change(observed, baseline, adapted, *, seed, n_boot=5000):
    """Paired percentile interval for percent RMSE change on the locked filters."""
    observed = np.asarray(observed, float)
    baseline = np.asarray(baseline, float)
    adapted = np.asarray(adapted, float)
    rng = np.random.default_rng(seed)
    changes = np.empty(n_boot)
    for index in range(n_boot):
        take = rng.integers(0, len(observed), len(observed))
        base_rmse = np.sqrt(np.mean((baseline[take] - observed[take]) ** 2))
        adapted_rmse = np.sqrt(np.mean((adapted[take] - observed[take]) ** 2))
        changes[index] = 100 * (adapted_rmse / base_rmse - 1)
    return float(np.percentile(changes, 2.5)), float(np.percentile(changes, 97.5))


def load_training_frame():
    metadata = load_pool_metadata().merge(
        load_tor_loadings(), on=['Site', 'date'], how='left', validate='many_to_one')
    pool = metadata.query('TOR_EC_loading_ug > 0').drop_duplicates('FilterId').copy()
    pool['AnalysisId'] = pool['AnalysisId'].astype(int)
    return metadata, pool.drop_duplicates('AnalysisId').set_index('AnalysisId')


def resolve_ids(name, spec, metadata, pool):
    if spec['cohort'] == 'ocec':
        eligible = (metadata['TOR_EC_loading_ug'].gt(0)
                    & metadata['TOR_EC_ugm3'].gt(0)
                    & metadata['TOR_OC_ugm3'].gt(0)
                    & metadata['OC_EC_ratio'].notna())
        ranked = (metadata[eligible].sort_values('OC_EC_ratio')
                  .drop_duplicates('FilterId')['AnalysisId'].astype(int).to_numpy())
    else:
        cached = np.load(REPO / 'calibration_explorer/cache/analog_corrected_ranking.npz')
        ranked = cached['ids'].astype(int)
    ranked = ranked[np.isin(ranked, pool.index.to_numpy())]
    ids = np.array(list(dict.fromkeys(ranked[:spec['cutoff']])), dtype=int)
    if len(ids) != spec['cutoff']:
        raise ValueError(f'{name}: resolved {len(ids)} rather than {spec["cutoff"]} rows')
    return ids


def source_representation(ids, spectra, pool, corrected):
    if spectra == 'airspec':
        row = {int(value): i for i, value in enumerate(corrected['analysis_id'].astype(int))}
        if not set(ids) <= set(row):
            raise ValueError('AIRSpec cache does not cover the resolved cohort')
        X = corrected['corrected'][[row[int(value)] for value in ids]].astype(float)
    else:
        columns = pd.read_csv(
            PATHS.ftir_dir / 'local_db/spectra_248_251.csv', nrows=0).columns
        wcols = sorted(
            [value for value in columns if str(value).replace('.', '', 1).isdigit()],
            key=lambda value: -float(value))
        raw = load_pool_spectra(ids, wcols).set_index('AnalysisId').loc[ids]
        X = savgol_filter(raw[wcols].to_numpy(float), axis=1, **SAVGOL)
    X = block_average(X, BLOCK_SIZE)
    y = pool.loc[ids, 'TOR_EC_loading_ug'].to_numpy(float)
    groups = pool.loc[ids, 'Site'].astype(str).to_numpy()
    return X, y, groups


def load_unlabeled_target(name, spectra):
    folder = TARGETS / name
    path = folder / ('spectra_corrected.csv' if spectra == 'airspec' else 'spectra.csv')
    frame = pd.read_csv(path)
    identifiers = frame.iloc[:, 0].astype(str).to_numpy()
    X = frame.iloc[:, 1:].to_numpy(float)
    if spectra == 'deriv2':
        X = savgol_filter(X, axis=1, **SAVGOL)
    return identifiers, block_average(X, BLOCK_SIZE)


def select_components(X, y, groups, train_mask):
    curve = component_cv_curve(
        X[train_mask], y[train_mask], range(1, MAX_COMPONENTS + 1),
        groups=groups[train_mask], n_splits=5, random_state=42,
    )
    selected, annotated = select_first_major_minimum(curve)
    return int(selected), 'first local minimum within one SE of global minimum', annotated


def select_multiplier(X, y, groups, source_train, n_components):
    positions = np.flatnonzero(source_train)
    split = GroupKFold(n_splits=PSEUDO_TARGET_FOLDS)
    predictions = {float(value): np.full(len(positions), np.nan) for value in MULTIPLIERS}
    for inner_train, pseudo_target in split.split(X[positions], groups=groups[positions]):
        train_pos = positions[inner_train]
        target_pos = positions[pseudo_target]
        heuristic = fit_domain_invariant_pls(
            X[train_pos], y[train_pos], X[target_pos],
            n_components=n_components, heuristic=True,
        )
        for multiplier in MULTIPLIERS:
            fitted = fit_domain_invariant_pls(
                X[train_pos], y[train_pos], X[target_pos],
                n_components=n_components,
                lambdas=heuristic.lambdas * multiplier,
            )
            predictions[float(multiplier)][pseudo_target] = fitted.predict(X[target_pos])
    rows = []
    observed = y[positions]
    for multiplier in MULTIPLIERS:
        predicted = predictions[float(multiplier)]
        if not np.isfinite(predicted).all():
            raise ValueError('pseudo-target predictions are incomplete')
        rows.append({'multiplier': float(multiplier),
                     'pseudo_target_RMSE': float(np.sqrt(np.mean((predicted - observed) ** 2)))})
    table = pd.DataFrame(rows).sort_values(['pseudo_target_RMSE', 'multiplier'])
    return float(table.iloc[0]['multiplier']), table.sort_values('multiplier')


def predict_variants(X_train, y_train, X_domain, X_eval, n_components, multiplier):
    ordinary = PLSRegression(n_components=n_components, scale=False).fit(X_train, y_train)
    zero = fit_domain_invariant_pls(
        X_train, y_train, X_domain, n_components=n_components, lambdas=0)
    heuristic = fit_domain_invariant_pls(
        X_train, y_train, X_domain, n_components=n_components, heuristic=True)
    selected = fit_domain_invariant_pls(
        X_train, y_train, X_domain, n_components=n_components,
        lambdas=heuristic.lambdas * multiplier,
    )
    return {
        'ordinary_pls': ordinary.predict(X_eval).ravel(),
        'target_centered_lambda0': zero.predict(X_eval),
        'dipls_target_centered_heuristic': heuristic.predict(X_eval),
        'dipls_target_centered_selected': selected.predict(X_eval),
        'dipls_source_centered_heuristic': heuristic.predict(
            X_eval, centering='source'),
        'dipls_source_centered_selected': selected.predict(
            X_eval, centering='source'),
    }, heuristic.lambdas


# %% [markdown]
# ## Source-only selection
#
# No target response file has been opened at this point.

# %%
metadata, pool = load_training_frame()
corrected = np.load(ROOT / 'output/corrected/improve_pool_corrected_df6.npz', allow_pickle=True)
source_data = {}
selection_rows = []
lambda_curves = []

for config_name, spec in CONFIGS.items():
    ids = resolve_ids(config_name, spec, metadata, pool)
    X, y, groups = source_representation(ids, spec['spectra'], pool, corrected)
    source_train = protocol_train_mask('site_heldout', X, y, groups)
    n_components, component_reason, component_curve = select_components(
        X, y, groups, source_train)
    multiplier, multiplier_curve = select_multiplier(
        X, y, groups, source_train, n_components)
    source_data[config_name] = {
        'spec': spec, 'ids': ids, 'X': X, 'y': y, 'groups': groups,
        'source_train': source_train, 'n_components': n_components,
        'multiplier': multiplier,
    }
    selection_rows.append({
        'config': config_name, **spec, 'n_source': len(ids),
        'n_source_train': int(source_train.sum()),
        'n_source_outer_holdout': int((~source_train).sum()),
        'n_source_sites': int(pd.Series(groups).nunique()),
        'n_features_binned': X.shape[1], 'n_components': n_components,
        'component_rule': component_reason,
        'selected_multiplier': multiplier,
    })
    multiplier_curve.insert(0, 'config', config_name)
    lambda_curves.append(multiplier_curve)
    component_curve.assign(config=config_name).to_csv(
        OUT / f'{config_name}_component_curve.csv', index=False)

selection = pd.DataFrame(selection_rows)
lambda_selection = pd.concat(lambda_curves, ignore_index=True)
selection.to_csv(OUT / 'frozen_protocol.csv', index=False)
lambda_selection.to_csv(OUT / 'lambda_selection_curves.csv', index=False)
(OUT / 'frozen_protocol.json').write_text(json.dumps({
    'block_size': BLOCK_SIZE,
    'max_components': MAX_COMPONENTS,
    'pseudo_target_folds': PSEUDO_TARGET_FOLDS,
    'candidate_multipliers': MULTIPLIERS.tolist(),
    'target_labels_opened': False,
    'configurations': selection.to_dict(orient='records'),
}, indent=2))

display(selection)
display(lambda_selection.pivot(index='multiplier', columns='config', values='pseudo_target_RMSE').round(3))

# %% [markdown]
# ## Untouched IMPROVE-site validation
#
# This is the first use of the 20% outer source holdout. It checks whether the chosen
# domain penalty improves transfer between unseen IMPROVE sites before any Addis/Delhi
# label is opened.

# %%
source_validation_rows = []
for config_name, data in source_data.items():
    train = data['source_train']
    variants, heuristic_lambdas = predict_variants(
        data['X'][train], data['y'][train], data['X'][~train], data['X'][~train],
        data['n_components'], data['multiplier'])
    for variant, predicted in variants.items():
        source_validation_rows.append({
            'config': config_name, 'variant': variant,
            'n': int((~train).sum()), **metrics(data['y'][~train], predicted),
            'median_heuristic_lambda': float(np.median(heuristic_lambdas)),
        })

source_validation = pd.DataFrame(source_validation_rows)
source_validation.to_csv(OUT / 'outer_source_validation.csv', index=False)
display(source_validation.round(3))

# %% [markdown]
# ## Locked target reveal
#
# The protocol is now frozen on disk. The following cell is the first code that reads an
# Addis or Delhi `reference.csv`. Predictions use HIPS only for final scoring.

# %%
target_summary_rows = []
target_prediction_rows = []

for config_name, data in source_data.items():
    train = data['source_train']
    spectra_kind = data['spec']['spectra']
    for target_name, target_spec in TARGET_SPECS.items():
        _, X_domain = load_unlabeled_target(target_spec['unlabeled'], spectra_kind)
        eval_ids, X_eval = load_unlabeled_target(target_spec['evaluation'], spectra_kind)
        reference = pd.read_csv(TARGETS / target_spec['evaluation'] / 'reference.csv')
        id_column = reference.columns[0]
        reference[id_column] = reference[id_column].astype(str)
        reference = reference.set_index(id_column).loc[eval_ids].reset_index()
        if not (reference['Volume_m3'] > 0).all():
            raise ValueError(f'{target_name}: invalid sample volume')
        observed = reference['Fabs'].to_numpy(float) / 10.0
        variants, heuristic_lambdas = predict_variants(
            data['X'][train], data['y'][train], X_domain, X_eval,
            data['n_components'], data['multiplier'])
        for variant, predicted_loading in variants.items():
            predicted = predicted_loading / reference['Volume_m3'].to_numpy(float)
            target_summary_rows.append({
                'config': config_name, 'target': target_name, 'variant': variant,
                'n': len(reference), 'n_unlabeled_domain': len(X_domain),
                'n_components': data['n_components'],
                'selected_multiplier': data['multiplier'],
                'median_heuristic_lambda': float(np.median(heuristic_lambdas)),
                **metrics(observed, predicted),
            })
            for filter_id, obs, pred in zip(eval_ids, observed, predicted):
                target_prediction_rows.append({
                    'config': config_name, 'target': target_name, 'variant': variant,
                    'ExternalFilterId': filter_id, 'observed_bc_mac10_ugm3': obs,
                    'predicted_ec_ugm3': pred, 'residual_ugm3': pred - obs,
                })

target_summary = pd.DataFrame(target_summary_rows)
target_predictions = pd.DataFrame(target_prediction_rows)
target_summary.to_csv(OUT / 'locked_target_summary.csv', index=False)
target_predictions.to_csv(OUT / 'locked_target_predictions.csv', index=False)
(OUT / 'run_manifest.json').write_text(json.dumps({
    'block_size': BLOCK_SIZE,
    'max_components': MAX_COMPONENTS,
    'pseudo_target_folds': PSEUDO_TARGET_FOLDS,
    'candidate_multipliers': MULTIPLIERS.tolist(),
    'target_labels_opened': True,
    'frozen_protocol': 'frozen_protocol.json',
    'configurations': selection.to_dict(orient='records'),
}, indent=2))

display(target_summary.round(3))

# %% [markdown]
# ## Results
#
# The comparison below isolates three effects: ordinary binned PLS, target-mean centering
# without a covariance penalty (`lambda0`), and the actual domain-invariant penalty. The
# heuristic row is the paper/library default; the selected row uses only the pseudo-target
# source-site rule above. Source-centering is a controlled sensitivity using the same frozen
# weights and multiplier; it tests whether covariance alignment helps after disabling the
# target-mean shift that proved inappropriate for filter-loading responses.

# %%
pivot = target_summary.pivot_table(
    index=['config', 'target'], columns='variant',
    values=['slope', 'intercept', 'R2', 'RMSE_1to1', 'mean_bias'])
display(pivot.round(3))

improvements = []
for group_index, ((config_name, target_name), group) in enumerate(
        target_summary.groupby(['config', 'target'])):
    ordinary = group.set_index('variant').loc['ordinary_pls']
    indexed = group.set_index('variant')
    for centering, variant in (
        ('target', 'dipls_target_centered_selected'),
        ('source', 'dipls_source_centered_selected'),
    ):
        selected = indexed.loc[variant]
        paired = target_predictions[
            target_predictions['config'].eq(config_name)
            & target_predictions['target'].eq(target_name)
            & target_predictions['variant'].isin(['ordinary_pls', variant])
        ].pivot(index='ExternalFilterId', columns='variant',
                values=['observed_bc_mac10_ugm3', 'predicted_ec_ugm3'])
        observed = paired['observed_bc_mac10_ugm3']['ordinary_pls'].to_numpy(float)
        baseline_prediction = paired['predicted_ec_ugm3']['ordinary_pls'].to_numpy(float)
        adapted_prediction = paired['predicted_ec_ugm3'][variant].to_numpy(float)
        change_low, change_high = bootstrap_rmse_change(
            observed, baseline_prediction, adapted_prediction,
            seed=20260824 + 10 * group_index + (centering == 'source'))
        improvements.append({
            'config': config_name, 'target': target_name, 'centering': centering,
            'selected_multiplier': selected['selected_multiplier'],
            'ordinary_RMSE': ordinary['RMSE_1to1'],
            'dipls_RMSE': selected['RMSE_1to1'],
            'RMSE_change_pct': 100 * (
                selected['RMSE_1to1'] / ordinary['RMSE_1to1'] - 1),
            'RMSE_change_ci_low': change_low,
            'RMSE_change_ci_high': change_high,
            'ordinary_slope': ordinary['slope'], 'dipls_slope': selected['slope'],
            'ordinary_intercept': ordinary['intercept'],
            'dipls_intercept': selected['intercept'],
        })
improvements = pd.DataFrame(improvements)
improvements.to_csv(OUT / 'dipls_vs_pls.csv', index=False)
display(improvements.round(3))

# %% [markdown]
# ## Takeaways
#
# (filled in by the finalize step after execution)
