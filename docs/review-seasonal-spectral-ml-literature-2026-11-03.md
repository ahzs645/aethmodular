# Seasonal whole-spectrum ML: literature and interpretation

Date: 2026-11-03

This note supports the existing-data experiment in
[`run_seasonal_spectral_ml.py`](../research/ftir_hips_chem/workflows/run_seasonal_spectral_ml.py)
and its reusable
[`seasonal_spectral_ml.py`](../research/ftir_hips_chem/scripts/seasonal_spectral_ml.py)
module. It records methodological evidence, not new experimental results.

Whole-spectrum models can identify reproducible spectral patterns associated
with season or FTIR-minus-HIPS disagreement. A predictive interval is not an
identified chemical species, emission source, or independently validated EC
correction. Random forest (RF) is a useful comparison with established
chemometric methods; the literature does not establish that it must outperform
partial least squares (PLS) on these filters.

## Primary evidence

### 1. FTIR OC/EC prediction can transfer, with exceptions

Reggente, Dillner and Takahama (2016), *Atmospheric Measurement Techniques* **9**,
441–454, trained PLS with 526 filters from seven IMPROVE sites in 2011 and tested
more than 2,000 filters from 17 sites in 2013. Transfer succeeded at the shared
sites and nine of eleven new sites; two sites required another calibration.
They also examined distances from new spectra to the calibration population
in PLS score space. This supports explicit transfer evaluation and calibration
support diagnostics. Spectral proximity alone does not validate Ethiopian EC.

[Predicting TOR measurements from infrared spectra: extending predictions to different years and sites](https://doi.org/10.5194/amt-9-441-2016).

### 2. Spectrum-defined regimes have an aerosol EC precedent

Weakley et al. (2018), *Aerosol Science and Technology* **52**(6), 642–654,
separated atypical Elizabeth, New Jersey spectra using PLS discriminant
analysis, then fitted separate TOR-calibrated EC models. Reported test
performance improved from R² 0.760 to 0.886 and median absolute deviation
26.7% to 19.8%. Their diesel interpretation also used OC/EC, TOR fractions,
charring and site context. The study supports investigating spectral regimes;
it does not establish a calendar-specific Ethiopian correction or identify
sources from importance scores alone.

[Multilevel modeling for elemental carbon](https://doi.org/10.1080/02786826.2018.1439571).

### 3. RF is a candidate spectral regressor, not a guaranteed winner

Ghasemi and Tavakoli (2013), *Analytical Methods* **5**, 1863–1871, compared RF
with PLS, nonlinear support-vector regression and bagged trees on four real
UV/visible and near-infrared datasets, including corn and pharmaceutical
tablets. They reported comparable accuracy, computational efficiency and noise
robustness for RF. These are different measurement systems from aerosol PTFE
FTIR. Their findings motivate a matched comparison, not an assumed ranking.

[Application of random forest regression to spectral multivariate calibration](https://doi.org/10.1039/C3AY26338J).

### 4. Correlation changes the meaning of feature importance

Strobl et al. (2008), *BMC Bioinformatics* **9**, 307, used simulations and
peptide-binding data to show how correlated predictors distort ordinary RF
importance through both tree selection and permutation. Conditional
permutation better measured additional predictive information, without
eliminating every correlation-related preference. Importance also depended on
the number of candidate splitting variables and stability across random seeds.
Adjacent FTIR channels therefore cannot be interpreted as independently
competing chemical measurements. The current within-season shuffle conditions
only on season, not on loading or all correlated channels.

[Conditional variable importance for random forests](https://doi.org/10.1186/1471-2105-9-307).

### 5. Evaluate spectral intervals jointly

Gregorutti, Michel and Saint-Pierre (2015), *Computational Statistics & Data
Analysis* **90**, 15–35, developed grouped permutation importance, with theory
for additive models, simulations and a functional-data application. Group
importance generally is not the sum of individual-variable importances.
Shuffling an interval with one shared row permutation preserves relationships
within that interval. It can still create unusual combinations with the
remaining spectrum. Removing an interval and refitting answers a different
question: how well can the remaining inputs replace its predictive information?

[Grouped variable importance with random forests](https://doi.org/10.1016/j.csda.2015.04.002);
[author manuscript](https://arxiv.org/abs/1411.4170).

### 6. Validation must reflect temporal and site dependence

Roberts et al. (2017), *Ecography* **40**, 913–929, demonstrated through
simulations and case studies that random cross-validation can underestimate
error in structured data, even when residual dependence appears absent.
Blocking should match the intended prediction task; it can also create
extrapolation. The present forward October–September cycles test temporal
transfer. They do not test transfer to a new Ethiopian site. A sparse later
cycle must remain separate from a test cycle with all twelve sampled months.

[Cross-validation strategies for structured data](https://doi.org/10.1111/ecog.02881).

### 7. Model selection belongs inside validation

Varma and Simon (2006), *BMC Bioinformatics* **7**, 91, demonstrated optimistic
error estimates when tuning and performance reporting reused the same
cross-validation results, including with simulated null data. Nested
validation reduced this bias. The current experiment instead fixes model
settings and intervals in advance, so it does not need an inner tuning loop.
Choosing a new configuration after inspecting these test results would change
that status. Learned centering, scaling, encoding and latent components still
belong entirely inside each training fold.

[Bias in error estimation when using cross-validation for model selection](https://doi.org/10.1186/1471-2105-7-91).

### 8. Model explanations do not establish physical causation

Janzing, Minorics and Blöbaum (2020), *Proceedings of Machine Learning Research*
**108**, 2907–2916, distinguish observational conditioning from interventions
when defining Shapley feature relevance. The choice of reference distribution
changes the attribution question. This matters if SHAP is subsequently added:
explaining a fitted model's output does not identify a causal aerosol source.
The present experiment uses permutation and refitting, not SHAP; the same
distinction between predictive attribution and physical explanation applies.

[Feature relevance quantification in explainable AI: A causal problem](https://proceedings.mlr.press/v108/janzing20a.html).

## Consequences for this experiment

The fixed comparisons are RF, eight-component PCA followed by ridge or logistic
regression, and eight-component PLS regression. They are configurations to
compare, not optimized representatives of every algorithm. Raw full spectra,
common-window raw and signed corrected spectra, the removed baseline, and
derivatives are distinct representations. Their algebraic relationship makes
agreement between representations supporting context, not independent evidence.

Interpret results with these boundaries:

- Season classification tests whether spectral differences recur in another
  sampled cycle; it does not explain their origin.
- Disagreement contains FTIR EC computed from the spectrum. Spectral prediction
  can partly recover that existing calculation. Separate EC and HIPS prediction
  scores help expose this coupling.
- The loading/lot benchmark includes HIPS itself, alongside PM₂.₅, volume and
  lot. Because HIPS is also subtracted in the target, describe this explicitly
  as a HIPS-conditioned comparison.
- Importance measures held-out model reliance. Correlated substitutes can make
  a useful interval appear unimportant; negative importance is retained.
- Permutation-repeat spread measures Monte Carlo variability, not population
  uncertainty. A best-scoring configuration on one complete held-out cycle is
  an exploratory result, not proof of a general algorithm winner.
- Candidate band names are spectral neighborhoods, not exclusive species
  assignments or source fractions. Interpretation must remain consistent with
  the existing blank, repeat, baseline and chemistry evidence.
