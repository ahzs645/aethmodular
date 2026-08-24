# %% [markdown]
# # ftir_35 — Variation closure: what survived new HIPS holdouts and independent axes?
#
# ## tl;dr
#
# (filled in by the finalize step after execution)
#
# ## Context & Methods
#
# This notebook consolidates the audit requested after the five-site sweep. It does not
# select another winner. It reads the committed outputs of six reproducible analyses:
#
# 1. exact re-audit of all 61,635 five-site grid rows;
# 2. locked predictions on 54 reconstructed, never-screened HIPS targets;
# 3. calibration-set-specific HIPS blank-line, loading, and instrument-epoch diagnostics;
# 4. the pooled dry-season × HIPS interaction (ftir_24);
# 5. independent AERONET and MA350 optical checks; and
# 6. potassium/dust chemistry tests with loading and month controlled.
#
# Reconstructed HIPS values remain provisional. They reproduce the shipped HIPS equation
# and use the calibration line active on the analysis date, but they have not passed the
# production pipeline's MDL/comment/uncertainty QC.

# %%
from pathlib import Path

import pandas as pd
from IPython.display import display

ROOT = Path('.')
VC = ROOT / 'output/tables/variation_closure'

grid_best = pd.read_csv(VC / 'five_site_grid_best_by_target.csv')
grid_joint = pd.read_csv(VC / 'five_site_grid_addis_etbi_joint.csv')
locked = pd.read_csv(VC / 'locked_reconstruction_summary.csv')
hips = pd.read_csv(VC / 'hips_epoch_loading_summary.csv')
loading = pd.read_csv(VC / 'hips_loading_correlations.csv')
season = pd.read_csv(ROOT / 'output/tables/ftir24/season_interactions.csv')
aeronet = pd.read_csv(ROOT / 'output/tables/aeronet/three_wavelength_summary.csv')
ma350 = pd.read_csv(VC / 'four_site_ma350_summary.csv')
chemistry = pd.read_csv(VC / 'residual_chemistry_tests.csv')

assert len(grid_joint[grid_joint['estimator'].eq('deming')]) == 1
assert len(grid_joint[grid_joint['estimator'].eq('ols')]) == 1
assert locked[locked['target'].str.endswith('_holdout')]['n'].sum() == 4 * (14 + 14 + 26)

# %% [markdown]
# ## Data quality and grid audit
#
# The saved JSONL contains 71,263 rows in total. The exact five-site launch contributes
# 61,635 rows: 12,327 unique configuration×k readouts at each of five targets. There are no
# duplicate target keys in that launch subset. `heldout_R2` belongs to the IMPROVE TOR
# calibration test; `target_R2` belongs to the SPARTAN city crossplot. The earlier Delhi
# prose fused them, and the per-site table mixed Addis OLS with Delhi Deming.

# %%
audited = grid_best[
    grid_best['estimator'].eq('deming')
    & grid_best['minimum_target_R2'].eq(0.7)
][['target', 'n_eligible', 'cohort', 'cutoff', 'selection_space', 'spectra', 'k',
   'target_slope', 'target_intercept', 'target_R2', 'heldout_R2', 'extrap_pct']]
display(audited.round(3))

joint = grid_joint[grid_joint['estimator'].eq('deming')][[
    'cohort', 'cutoff', 'selection_space', 'spectra', 'k',
    'target_slope_addis', 'target_intercept_addis', 'target_R2_addis',
    'target_slope_delhi', 'target_intercept_delhi', 'target_R2_indh',
]]
display(joint.round(3))

# %% [markdown]
# ## Locked HIPS confirmation
#
# Configurations were frozen before the 54 values were reconstructed. This is a target-side
# holdout from selection, not an independent analytical reference: the target still uses the
# HIPS equation and lot blank line.

# %%
holdout = locked[
    locked['target'].str.endswith('_holdout')
    & locked['config'].isin(['addis_winner_k8', 'delhi_winner_k20',
                              'common_candidate_k20'])
][['config', 'target', 'n', 'ols_slope', 'ols_intercept', 'R2', 'RMSE',
   'mean_bias', 'extrap_pct', 'slope_ci_low', 'slope_ci_high']]
display(holdout.round(3))

# %% [markdown]
# ## HIPS-side variations
#
# Blank-line form and the raw-gain instrument epoch do not explain the Addis intercept.
# Calibration-set-specific refits are essential here: lot 251 itself used three deployed
# lines over time, so a lot-pooled blank regression would mix instrument states.

# %%
hips_overall = hips[
    hips['grouping'].eq('overall')
    & hips['config'].isin(['addis_winner_k8', 'delhi_winner_k20'])
][['config', 'target', 'reference_variant', 'n', 'ols_slope', 'ols_intercept',
   'R2', 'york_slope', 'york_intercept', 'below_blank_r1_pct']]
display(hips_overall.round(3))

gain = loading[
    loading['predictor'].eq('instrument_gain')
    & loading['config'].isin(['addis_winner_k8', 'delhi_winner_k20'])
]
display(gain.round(4))

# %% [markdown]
# ## Season interaction
#
# Separate season means reproduce ftir_15, but the pooled interaction adds a new result:
# both raw and AIRSpec models have a significantly flatter dry-season slope. The corrected
# model's mean bias is season-stable only because its intercept and slope move together; it
# is not season-invariant in calibration geometry. Both February conventions agree.

# %%
season_primary = season[
    season['scope'].eq('all') & season['MAC'].eq(10)
][['model', 'convention', 'n', 'wet_slope', 'wet_intercept', 'dry_slope',
   'dry_intercept', 'dry_slope_delta', 'dry_slope_delta_p',
   'dry_intercept_delta', 'dry_intercept_delta_p', 'joint_p']]
display(season_primary.round(4))

# %% [markdown]
# ## Independent optical axes
#
# AERONET is exploratory because the available inversion exports are Level 1.5 and contain
# no U27 field. MA350 IR-880 is independent of both FTIR and HIPS calibration and more
# directly localizes the additive mismatch.

# %%
aeronet_primary = aeronet[
    aeronet['subset'].eq('AOD440_ge_0.4')
][['city', 'n', 'AAE_440_675_median', 'AAE_675_870_median',
   'AAE_curvature_median', 'source_level_values']]
display(aeronet_primary.round(3))

ma350_exact = ma350[ma350['match'].eq('exact')][[
    'site', 'reference', 'n', 'ols_slope', 'ols_intercept',
    'intercept_ci_low', 'intercept_ci_high', 'R2',
    'deming_equal_error_slope', 'deming_equal_error_intercept',
]]
display(ma350_exact.round(3))

# %% [markdown]
# ## Chemistry attribution
#
# The partial tests residualize both chemistry and calibration residual on HIPS loading and
# cyclic month terms, then use a 5,000-permutation two-sided test with within-site BH-FDR.
# Potassium/organic axes survive at Addis; the joint mineral-dust axis does not.

# %%
chem_focus = chemistry[
    ((chemistry['config'].eq('addis_winner_k8')
      & chemistry['target'].eq('addis_augmented'))
     | (chemistry['config'].eq('delhi_winner_k20')
        & chemistry['target'].eq('indh_augmented')))
][['config', 'target', 'predictor', 'n', 'partial_r', 'permutation_p', 'fdr_q']]
display(chem_focus.round(4))

# %% [markdown]
# ## Takeaways
#
# (filled in by the finalize step after execution)
