"""Build the CLI-executable common-test, routing and Addis-readiness notebook."""

from pathlib import Path
import nbformat as nbf

AREA = Path(__file__).resolve().parents[1]
DEST = AREA / "vibes_followup_experiments.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def main():
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook.cells = [
        code("""import sys
# For notebooks inside research/ftir_hips_chem/:
sys.path.insert(0, './scripts')
sys.path.insert(0, './workflows')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Markdown

# Canonical research configuration, loaders, exclusions and plotting style.
from config import (SITES, PROCESSED_SITES_DIR, FILTER_DATA_PATH,
                    AERONET_DATA_DIR, WEATHER_DATA_DIR, MAC_VALUE)
from outliers import (EXCLUDED_SAMPLES, MANUAL_OUTLIERS, apply_exclusion_flags,
                      apply_threshold_flags, get_clean_data, print_exclusion_summary)
from data_matching import (load_aethalometer_data, load_filter_data,
                           match_aeth_filter_data, match_all_parameters)
from etad_factors import load_etad_factor_contributions, match_etad_factors
from aeronet import load_aeronet, aeronet_dir, COLS as AERONET_COLS
from improve_io import load_improve_clean
from plotting import PlotConfig, crossplots, timeseries, distributions, comparisons
from plotting.utils import calculate_regression_stats

from vibes_followup_studies import (SOURCE, OUTPUT, load_inputs, common_test,
                                    routing_study, addis_gate, write_manifest)

PlotConfig.set(sites='all', layout='individual', show_stats=True, show_1to1=True)
FIGURES = Path('output/plots/vibes_followup_notebook')
FIGURES.mkdir(parents=True, exist_ok=True)
"""),
        md("""# AIRSpec / VIBES: common-test selection and spectrum-only routing

## tl;dr

On the same **2,327 IMPROVE test filters**, the full-pool AIRSpec model has
RMSE **3.224 µg/filter**. The best of the four 500-filter analog models has
RMSE **4.837** (VIBES with the 1800–2500 and >3500 selection cuts); the
historical lowest-OC/EC models have **16.759** (AIRSpec) and **13.098**
(VIBES) when extended to the full test population. The spectrum-only router
has RMSE **6.095** versus AIRSpec **6.093** in nested training-site folds, and
**3.168** versus **3.224** on the previously examined outer sites; its paired
outer-site interval includes zero. These are exploratory comparisons, not a
fresh confirmatory validation. The Addis reference gate finds **no independent
thermal EC matches**, so it reports no Addis accuracy metric. No original
correction or exclusion flag is changed.

## Context & Methods

- Source: completed 2026-09-21 full Colab run, `output/tables/vibes_colab_cloud/
  persistent_results/full-83dcf32e86bc0f09/`.
- Population: 10,066 eligible training and 2,327 disjoint-site test IMPROVE
  physical filters, plus 253 unlabeled Addis target spectra. TOR EC is in
  **µg/filter**. Saved methods share the original 2,002-channel grid.
- Lowest-OC/EC membership is fixed historical selection using carbon labels;
  it is not a prediction-time classifier.
- Analog membership is selected from **training sites only** by mean Pearson
  similarity to Addis target spectra corrected by AIRSpec. All variants choose
  500 physical filters. Each mask changes similarity ranking only; every PLS
  model still uses all 2,002 channels. The full-pool component counts (AIRSpec
  6, VIBES 7) are fixed for analog fits. No outer-test EC is used in selection.
- The router uses 16 spectrum-only regional mean/SD features from the two
  corrected spectra. Its labels come from grouped out-of-fold training-site
  predictions. An outer grouped loop evaluates the training procedure; the
  final router is then fit using all training sites and scored on the previously
  examined original test sites. No test site, TOR loading, HIPS or location enters
  the routing features. No router threshold or feature set is optimized here.
  The nested folds share the already fitted blank-based correction; only the
  **EC calibration and router** are refit per fold, so this is conditional on
  saved preprocessing, not an end-to-end independent baseline test.
- Paired 95% intervals resample entire test sites (2,000 draws) and are
  conditional on fitted models; they do not cover model development or the
  many exploratory contrasts.

### Key Assumptions

The completed run's case ledger, paired corrected arrays, and split remain
unaltered. This notebook refits PLS from those **saved corrected arrays**; it
does not rerun AIRSpec or VIBES baseline correction. The fixed Addis spectra are
used only as unlabeled selection targets. Repeated exploration of the old test
set limits inference about any apparent improvement.
"""),
        md("""## Data · freeze and reconcile the completed run"""),
        code("""inputs = load_inputs()
write_manifest(inputs)
print('Source signature:', inputs['manifest']['signature'])
print('Cases:', len(inputs['cases']), '| train:', len(inputs['train']),
      '| common test:', len(inputs['test']), '| Addis targets:', len(inputs['targets']))
print('Train/test sites:', inputs['cases'].Site.iloc[inputs['train']].nunique(),
      inputs['cases'].Site.iloc[inputs['test']].nunique())
print('Input hashes written to', OUTPUT / 'input_manifest.json')
"""),
        md("""## Results · common-test calibration selections

The original restricted cohort was previously evaluated on only 137 of these
test filters. Here it is refit with its original 625 training members and scored
on **all 2,327** test filters, exactly like the full-pool and new analog fits.
Original saved predictions must reproduce on their original shared rows.
This is a new site-restricted AIRSpec-based analog selection, not the exact
September 10 selection population, so membership-change counts will differ.
"""),
        code("""selection_metrics, selection_predictions, selected_filters = common_test(inputs)
display(selection_metrics[['model','method','n_train','n_train_sites','components',
                           'n','RMSE','MAE','bias','predictive_R2',
                           'delta_RMSE_vs_full_AIRSpec','delta_RMSE_CI_low',
                           'delta_RMSE_CI_high']].round(3))
overlap = pd.read_json(OUTPUT / 'selection_overlap.json').T
display(overlap)
display(pd.read_csv(OUTPUT / 'restricted_membership_sensitivity.csv').round(3))
"""),
        code("""models = list(selection_metrics.model.drop_duplicates())
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
x = np.arange(len(models))
for offset, method in [(-.18, 'AIRSpec'), (.18, 'VIBES')]:
    part = selection_metrics.set_index(['model','method']).loc[[(m,method) for m in models]]
    axes[0].bar(x+offset, part.RMSE, width=.35,
                color='#44749d' if method=='AIRSpec' else '#7e54a8', label=method)
axes[0].set_xticks(x, ['Full pool','Lowest OC/EC','Analog full','Exclude CO₂',
                        'CO₂ + >3600','CO₂ + >3500'], rotation=35, ha='right')
axes[0].set_ylabel('RMSE vs TOR EC (µg/filter)')
axes[0].set_title('Same 2,327 outer-site IMPROVE filters')
axes[0].legend(frameon=False)
analog = overlap.loc[[m for m in models if m.startswith('analog_')]]
axes[1].bar(np.arange(len(analog)), analog.replaced_vs_unmasked,
            color='#617f76')
axes[1].set_xticks(np.arange(len(analog)), ['No cut','1800–2500',
                                            '+ >3600','+ >3500'], rotation=25, ha='right')
axes[1].set_ylabel('Replaced of 500 training filters')
axes[1].set_title('Analog membership change vs unmasked selection')
fig.suptitle('Selection masks affect analog ranking; all PLS fits retain 2,002 channels')
fig.savefig(FIGURES / 'common_test_selection.png', bbox_inches='tight')
plt.show()
"""),
        md("""The restricted model's earlier 137-filter result does not
generalize to all outer-site filters: it has far larger error on the 2,190
filters outside historical membership. That is a population shift result,
not proof of a particular chemical mechanism. The analog-model contrasts are
also retrospective; inspect paired site intervals and membership before
interpreting small differences. A masked analog model is distinct from
deleting spectral channels in PLS."""),
        md("""## Results · spectrum-only routing

The nested training-site rows estimate how the **router-training procedure**
behaves when entire training sites are left out. The final row uses the original
outer sites, which have been inspected in earlier work. We report both because
the second alone could reward decisions suggested by previous test charts.
"""),
        code("""routing_metrics, routing_predictions = routing_study(inputs)
display(routing_metrics[['population','method','n','n_sites','selected_VIBES',
                         'RMSE','MAE','bias','predictive_R2',
                         'delta_RMSE_vs_AIRSpec','delta_RMSE_CI_low',
                         'delta_RMSE_CI_high']].round(3))
display(pd.read_csv(OUTPUT / 'routing_largest_VIBES_errors.csv').head(3))
"""),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
populations = ['training_site_nested','prior_inspected_test']
labels = ['Nested training-site folds','Previously inspected outer sites']
for i, population in enumerate(populations):
    part = routing_metrics.set_index(['population','method'])
    for j, method in enumerate(['AIRSpec','VIBES','routed']):
        value = part.loc[(population,method),'RMSE']
        ax.bar(i + (j-1)*.22, value, width=.2,
               color={'AIRSpec':'#44749d','VIBES':'#7e54a8','routed':'#507e69'}[method],
               label=method if i==0 else None)
ax.set_xticks(range(2), labels)
ax.set_ylabel('RMSE vs TOR EC (µg/filter)')
ax.set_title('Spectrum-only routing compared with both fixed methods')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / 'routing_comparison.png', bbox_inches='tight')
plt.show()
"""),
        md("""The nested VIBES RMSE is dominated by a severe TONT1 prediction
(`improve:2008808`, about −1148 versus TOR 5.61 µg/filter). It remains in all
scores; the router selected AIRSpec for that filter. On the previously examined
outer sites the router's RMSE difference from AIRSpec is −0.056 µg/filter,
with a paired site-bootstrap interval of −0.188 to 0.090. This does not
establish a benefit, and the old test has already been studied."""),
        md("""## Results · independent Addis reference gate

This is a **readiness check**, not an accuracy estimate. The 253 Addis targets
have frozen predictions but no independently measured, identity-confirmed TOR
EC in the completed run. A future reference file may be passed as
`AETH_ADDIS_VALIDATION_CSV`; it must contain unique `sample_id`, distinct
`ptfe_filter_id` and `quartz_filter_id`, `TOR_EC_loading_ug`, `reference_method`,
`reference_source`, `authoritative_match`, and `sampling_equivalent`. The gate
rejects unconfirmed identity or sampling equivalence. It does not automatically
score newly unblinded values; the already frozen validation protocol governs
that separate analysis.
"""),
        code("""addis_status = addis_gate(inputs)
display(addis_status)
"""),
        md("""## Takeaways

Use the executed common-test and routing tables above for observed differences;
these are exploratory because the original test set was already examined.
The frozen Addis predictions are ready to compare only after an authoritative
thermal EC crosswalk and comparable quartz/PTFE sampling are supplied. HIPS/MAC
and public ChemSpec EC do not supply that independent reference.

### Reproduction

From the repository root:

```sh
uv run --locked --no-sync python research/ftir_hips_chem/workflows/build_vibes_followup_notebook.py
uv run --locked --no-sync jupyter nbconvert --execute --to notebook \\
  --ExecutePreprocessor.timeout=3600 \\
  --output vibes_followup_experiments_executed.ipynb \\
  --output-dir research/ftir_hips_chem/notebooks/archive/executed \\
  research/ftir_hips_chem/vibes_followup_experiments.ipynb
```

Outputs: `output/tables/vibes_followup_notebook/` and
`output/plots/vibes_followup_notebook/`. The input manifest contains source
SHA-256 hashes and the original run signature. Rebuilding the notebook clears
its outputs; keep the separately executed copy for the audited readout.
"""),
    ]
    nbf.validate(notebook)
    nbf.write(notebook, DEST)
    print(DEST)


if __name__ == "__main__":
    main()
