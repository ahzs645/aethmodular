# Research additions: calibration transfer and optical comparators

**Reviewed 2 November 2026.** This note connects primary research to experiments
that can use the existing spectra, chemistry, metadata, and calibration exports.
It does not require new Ethiopian sampling. It documents methods and their
evidence requirements; it does not claim that a new EC correction has been
validated or that every proposed experiment has been run.

## Scientific decision

Evaluate three questions separately:

1. Does preprocessing preserve measurable spectral information and remove
   reproducible substrate/background effects?
2. Does the complete preprocessing plus calibration pipeline predict independent
   thermal-optical measurements on held-out calibration-domain filters?
3. Does its transfer to Addis or Bishoftu change disagreement with an independently
   measured optical quantity, and how much does that conclusion depend on optical
   conversion assumptions?

A favorable answer to question 3 is not a substitute for question 2, and question
2 does not establish Ethiopian EC accuracy. The SPARTAN public ChemSpec EC feed
is FTIR-derived and is not an independent thermal-optical reference. See the
repository's [current scientific position](current-research-summary.md) and
[measurement-provenance rules](../AGENTS.md).

The literature supports composition-aware calibration and explicit applicability
diagnostics. It does not establish a universal Ethiopian multiplier, a uniquely
EC-specific 1617 cm⁻¹ peak, or a universal optical mass absorption cross-section.

## Twelve primary sources and what they justify testing

Titles below are shortened where needed. Each DOI resolves to the original
publication. Full articles were inspected for selected Copernicus entries;
other entries were verified through publisher-indexed abstracts or official
author records, as identified in the last column.

| ID | Primary source | Verified finding relevant here | Experiment and limit | Evidence inspected |
|---|---|---|---|---|
| T01 | Dillner & Takahama (2015), **Predicting TOR measurements from infrared spectra: elemental carbon**, [10.5194/amt-8-4013-2015](https://doi.org/10.5194/amt-8-4013-2015) | Calibrated PTFE spectra against collocated quartz TOR EC. The study included 794 samples from seven IMPROVE sites and compared spectral pretreatments and low-loading calibration approaches. | Preserve a paired source-TOR benchmark for every preprocessing variant; report low-loading performance separately. Its IMPROVE performance does not establish Ethiopian accuracy. | Publisher abstract and methods summary. |
| T02 | Reggente, Dillner & Takahama (2016), **Extending TOR predictions to different years and sites**, [10.5194/amt-9-441-2016](https://doi.org/10.5194/amt-9-441-2016) | A 2011 calibration was tested on later-year samples and new sites; two of eleven unseen sites required a different calibration. Applicability used Mahalanobis distance in the fitted PLS score space. | Add actual-model score-space distance alongside spectral residual Q; validate its ability to flag source-domain error. Generic spectral novelty is a different diagnostic. | Publisher abstract; PDF retrieval timed out. |
| T03 | Weakley et al. (2018), **CSN multilevel modeling for elemental carbon**, [10.1080/02786826.2018.1439571](https://doi.org/10.1080/02786826.2018.1439571) | The pooled CSN calibration underperformed for an atypical Elizabeth, NJ population. Classification plus dedicated calibrations improved the reported overall test performance. Different spectral predictors contributed in the atypical and typical groups. | Evaluate source-regime submodels using independent calibration labels and group-held-out validation. A country label alone is not an aerosol regime, and a spectral coefficient is not a unique source attribution. | Publisher-indexed abstract; full-text request returned 403. |
| T04 | Weakley, Takahama & Dillner (2018), **TOR-equivalent carbon from FRM/FEM filters**, [10.1080/02786826.2018.1504161](https://doi.org/10.1080/02786826.2018.1504161) | Applied CSN calibrations to FRM filters on a mass-per-filter-area basis, exploiting collocated sampling configurations. | Audit volume, deposit area, units, substrate, and preprocessing before transferring coefficients. This is a successful specific transfer example, not proof that all PTFE lots or samplers are interchangeable. | Publisher-indexed abstract; full-text request returned 403. |
| T05 | Debus et al. (2022), **Major PM species from a single filter type**, [10.5194/amt-15-2685-2022](https://doi.org/10.5194/amt-15-2685-2022) | Combined composition-based selection with multilevel calibration over a large monitoring network. The paper discusses broad graphitic absorption, PTFE scattering, substrate changes, and weak or overlapping EC spectral information. | Compare broad-signal preservation and local bands; keep a composition representation separate from amplitude/loading. This paper is by Debus et al., not Yazdani et al. | Full publisher article, including Sections 1 and 2.5. |
| T06 | Takahama, Ruggeri & Dillner (2016), **Sparse selection of relevant infrared absorption bands**, [10.5194/amt-9-3429-2016](https://doi.org/10.5194/amt-9-3429-2016) | Evaluated four sparse calibration approaches for functional groups and TOR-calibrated carbon, making predictor regions more interpretable. | Run prespecified region ablation or sparse models inside source training folds, then compare identical held-out filters. Predictive importance is not chemical specificity; correlated bands can substitute for one another. | Publisher abstract and article record. |
| T07 | Parks et al. (2021), **Quantifying diesel elemental and organic carbon by mid-infrared spectrometry**, [10.1080/02786826.2021.1917764](https://doi.org/10.1080/02786826.2021.1917764) | Used reflection-mode spectra of diesel PM on quartz, thermal-optical calibration, and independent mine samples. | Include broad and narrow carbon-like perturbations in algorithm stress tests. Do not copy the quartz/reflection calibration or its coefficients into PTFE/transmission processing. | [NIOSH publication record](https://stacks.cdc.gov/view/cdc/215469) and [PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/34732970/). |
| T08 | White et al. (2025), **Absorption photometry of patterned deposits on IMPROVE PTFE filters**, [10.1080/10962247.2024.2442634](https://doi.org/10.1080/10962247.2024.2442634) | Nonuniform particle deposits violate the uniform-deposit interpretation of transmission. The model predicts an underestimate of summed particle absorption, potentially larger for heavily absorbing deposits. | Test loading and optical-depth dependence, and the direction of the proposed effect. Do not apply an IMPROVE screen geometry or fitted correction to SPARTAN without supporting sampler evidence. | Publisher record, [PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/39699252/), and the author's [2023 precursor presentation](https://vista.cira.colostate.edu/Improve/wp-content/uploads/2023/11/06-HIPS-Update-White.pdf). |
| T09 | Yus-Díez et al. (2025), **Filter-photometer artifacts in soot and dust measurements**, [10.5194/amt-18-3073-2025](https://doi.org/10.5194/amt-18-3073-2025) | Laboratory and ambient comparisons with a photothermal reference showed size-, composition-, and wavelength-dependent filter-photometer artifacts for AE33, CLAP, and MAAP instruments. | Retain comparator uncertainty and independent optical source labels. These results motivate sensitivity analysis; their numerical instrument corrections do not transfer to HIPS or the Addis MA350. | Full publisher article. |
| T10 | Zhao et al. (2021), **Equivalent black carbon using variable mass absorption cross-section**, [10.5194/amt-14-1319-2021](https://doi.org/10.5194/amt-14-1319-2021) | Investigated variation of the optical-to-mass conversion with particle size and coating assumptions at 880 nm. | Report a wavelength-explicit MAC scenario sweep and the algebraic crossover MAC. This study supplies a mechanism for variability, not an Ethiopia-specific 633-nm MAC estimate. | Full publisher article. |
| T11 | Pileci et al. (2021), **Co-located refractory black carbon and elemental carbon at European sites**, [10.5194/amt-14-1379-2021](https://doi.org/10.5194/amt-14-1379-2021) | Compared distinct rBC and EC measurement techniques across several campaigns; systematic differences persisted despite strong linear relationships. | Keep EC, rBC, and optical eBC distinct in names and claims. Agreement or correlation does not determine which technique is unbiased. | Publisher abstract and full article sections. |
| T12 | Chan et al. (2019), **Carbon-measurement intercomparison among three North American networks**, [10.5194/amt-12-4543-2019](https://doi.org/10.5194/amt-12-4543-2019) | Co-located observations differed with thermal protocol, analyzer, and charring correction. | Preserve the thermal protocol and instrument in calibration provenance; do not silently combine TOR/TOT labels or assume a universal conversion. | Publisher abstract and full article sections. |

The study populations, substrates, instruments, and wavelengths differ. The
reference registry is a map of testable mechanisms and validation practices,
not a set of correction factors to pool.

## Experiments using existing data

### E1. Separate the two sides of the disagreement

**Available with:** signed/raw spectral metrics, matched current FTIR EC, direct
HIPS absorption, volume, PM2.5, dates, lots, and source flags.

For each prespecified spectral metric, report its association with:

- FTIR EC alone;
- measured HIPS absorption alone, in Mm⁻¹;
- FTIR EC minus optical eBC under the stated MAC.

Use the same eligible physical filters and the same available metadata controls
for all three analyses. Retain signed associations alongside squared strengths.
Use time-blocked evaluation for predictive screens; fit scaling and tuning only
within training quarters. Include prespecified sensitivity results for flagged
acquisitions, without deleting them from the record.

**Interpretation:** a metric that predicts the difference mainly because it
reconstructs the existing FTIR estimator is useful for auditing that estimator,
but it does not establish measurement error. A replicated association with
independent absorption provides different evidence, still subject to optical
artifacts and aerosol composition. This decomposition is an analysis proposal
motivated by the measurement dependencies in T01–T03 and T08–T12, rather than a
published Ethiopian correction method.

### E2. Test whether Ethiopia conclusions survive MAC assumptions

**Available with:** direct HIPS absorption and matched FTIR EC. No refitting of
spectra is required.

For positive, explicitly chosen 633-nm MAC scenarios \(m\), calculate

\[
\mathrm{eBC}_{633}(m)=b_{\mathrm{abs},633}/m,
\qquad
\Delta(m)=\mathrm{EC}_{\mathrm{FTIR}}-\mathrm{eBC}_{633}(m).
\]

With absorption in Mm⁻¹ and MAC in m²/g, the result is numerically in µg/m³.
State that the scenario grid is analyst-chosen and is not an uncertainty
distribution or a set of Ethiopian calibration estimates. Report counts,
median signed difference, median absolute difference, and sign changes by site
and cohort. Keep direct HIPS and public extension products separate.

For EC greater than zero, also report

\[
m_* = b_{\mathrm{abs},633}/\mathrm{EC}_{\mathrm{FTIR}}.
\]

This is the **MAC needed for algebraic agreement** on that filter. It is not a
measured MAC, because its denominator is the estimator under investigation.
Mark nonpositive or missing EC explicitly. Avoid interpreting ratios close to
the EC detection limit.

Do not substitute a literature value quoted at 550 or 880 nm directly into a
633-nm calculation. If a wavelength conversion is explored, state its assumed
absorption Ångström exponent and separate that assumption from measurement.
In particular, [Bond & Bergstrom's 2006 investigative review](https://doi.org/10.1080/02786820500421521)
reports 7.5 ± 1.2 m²/g at 550 nm for uncoated particles. That is not a validated
7.5 m²/g value at 633 nm. T09–T11 justify testing assumptions rather than
declaring MAC 10 to be known truth.

### E3. Test the direction and loading dependence of deposit effects

**Available with:** absorption, sample volume, and preferably deposit area;
raw reflectance/transmittance and sampler geometry allow a stronger test.

Calculate optical loading \(b_{\rm abs}V\), and, only with documented area
\(A\), optical-depth proxy \(b_{\rm abs}V/A\) with consistent units. Compare
disagreement against both concentration and loading while retaining lots and
dates. If two filters have similar ambient concentrations but different sample
volumes, they help separate the two explanations.

T08 predicts that uniform-deposit inversion underestimates absorption when
deposition is sufficiently nonuniform. Any proposed application must have that
direction and become plausible at the relevant deposited optical depth. It
cannot be used as an arbitrary additive term chosen to erase a positive
intercept. Without geometry or raw optical records, this remains a mechanism
screen, not a reconstruction of true absorption.

### E4. Add actual-calibration PLS applicability next to descriptive novelty

**Available with:** the original calibration exports or a reconstructed source
model that has passed its comparison against the original predictions.

Reuse the existing implementation rather than creating another distance helper:

- [pls_transfer.py](../research/ftir_hips_chem/scripts/pls_transfer.py):
  `load_current_pls_model`, `score_metric`, `project_scores`,
  `mahalanobis_distance_squared`, and `spectral_q_residual`.
- [test_pls_transfer.py](../tests/test_pls_transfer.py): synthetic checks for the
  quadratic form, residual behavior, and related PLS calculations.

The EC loader expects these exports under the resolved FTIR data directory:

```text
apps/apps_data.npz
notebook/extracted/EC_samples.csv
notebook/extracted/EC_stats_chosen.csv
notebook/extracted/EC_fitted_long.csv
```

It reconstructs the selected model and compares its fitted predictions against
the R export before returning the model. These inputs are not tracked in the
checkout, but a subsequent read-only inventory of the existing external data
archive confirmed all four files: `apps_data.npz` (18,017,058 bytes),
`EC_samples.csv` (57,703 bytes), `EC_stats_chosen.csv` (286 bytes), and
`EC_fitted_long.csv` (1,943,802 bytes). This establishes input availability, not
a newly completed reconstruction or target applicability result. Reuse the
existing data resolver and run the reconstruction check before analysis.
Do not fabricate calibration distances from a five-site descriptive PCA.

**Normalization detail:** `score_metric` returns \((T^\mathsf{T}T)^{-1}\), so
the existing quadratic form is leverage-scaled. For centered scores and the
ordinary unbiased sample covariance, conventional Mahalanobis squared distance
is \((n_{\rm train}-1)\) times this quantity. Both can be valid, but thresholds,
labels, and comparisons must use the same convention. A shrinkage covariance is
a further method choice, not the same scaling transformation.

Report both score-space distance and spectral residual Q: a sample can be close
in retained scores yet have substantial unmodeled spectral structure. Learn
reference distributions and thresholds from source training/validation data;
join Ethiopian optical labels afterward. Passing the applicability screen does
not establish that the conditional relationship between spectra and EC transfers.

**Existing analysis distinction:**
[ftir_39](../research/ftir_ec_phase3/ftir_39_pls_applicability_domain.ipynb) evaluates
PLS candidates using a separate source-fitted PCA domain model. Its
[`applicability_domain_analysis`](../research/ftir_ec_phase3/scripts/theory_test_suite.py)
explicitly fits a scaler, PCA, and shrinkage covariance to source training
spectra, with source-site holdouts defining reference percentiles. This is a
useful source-only spectral-domain check. It is not T02's actual PLS-score
distance. Retain the two names and interpretations separately.

### E5. Test spectral regions by source-validated ablation

**Available with:** source spectra and independent source TOR labels, plus the
target spectra in exactly the same representation.

Freeze a small set of regions before the next run: broad 1500–4000 cm⁻¹ signal,
1500–1800 cm⁻¹ shape, 1700–1750 cm⁻¹ carbonyl region, and 2850–2950 cm⁻¹ C–H
region. Compare full-region, leave-one-region-out, and selected-region models.
Use the same source outer folds and target cohort for every variant. Select
component count and any sparse penalty within inner training folds. Preserve
scan grouping at physical-filter level.

T05–T07 motivate broad and local features. A synthetic peak-recovery test can
screen preprocessing today, while full EC performance comparison requires
matched recalibration. Occluding a region only at prediction time measures
perturbation sensitivity of the existing model; retraining without that region
measures information redundancy. Label these as different experiments.

### E6. Test composition regimes before geographic corrections

**Available with:** source spectra, independent source TOR, target spectra, and
independent chemical tracers where already measured.

Use a source-fitted composition representation to propose a small number of
regimes. Keep raw amplitude/loading information available in a separate block.
Evaluate a common calibration and regime-specific alternatives using held-out
source sites and years. Only then reveal Ethiopian optical comparisons. T03 and
T05 support this design; they do not guarantee that Addis aerosol matches any
published regime.

When testing Ethiopian mechanisms, favor independent XRF/IC measurements over
FTIR-derived OC, EC, or spectral factors as corroboration. Report what is
measured and what is derived. A classifier trained to separate countries can
identify filter lots, instruments, or analytical runs instead of composition.
For Addis–Bishoftu contrasts, restrict a matched analysis to observed overlap in
months, loading, and known lots, while separately showing the full cohorts.
Missing metadata are an explicit matching limitation.

### E7. Preserve the validation hierarchy when comparing pipelines

For each baseline/calibration combination, retain one row per physical filter
and a separate scan-level record. Report these outcome families independently:

| Outcome | What a favorable result establishes | What it does not establish |
|---|---|---|
| Signed blank residual and replicate stability | Better behavior for known background/replicate comparisons | Preservation of all aerosol signal |
| Injected narrow and broad signal recovery | Sensitivity to specified numerical perturbations | Recovery of real EC mass |
| Held-out source TOR prediction | Accuracy for the stated thermal protocol and source population | Accuracy for Ethiopian filters |
| Actual-model PLS distance and residual Q | Proximity to the stated calibration representation | Correct target response relationship |
| Ethiopian HIPS agreement and MAC sensitivity | Stability of an optical comparison under stated assumptions | Independent target EC accuracy |

Do not train a residual correction on HIPS-derived eBC and then describe improved
agreement with those same HIPS values as EC validation. Do not use every sample
to learn baseline banks, scaling, feature selection, or calibration parameters
before cross-validation. These are design requirements implied by the distinct
measurements and validation populations, rather than additional laboratory
funding requirements.

## Execution order and completion conditions

1. **Run E1–E3 on the current, source-labelled cohort.** They need existing
   chemistry/optics and a finite prespecified set of spectral metrics. Save
   input hashes, filter membership, units, scenarios, and exclusion flags.
2. **Run numerical baseline preservation controls.** Use the companion baseline
   investigation for signed blanks, real replicate differences, and broad/narrow
   perturbations. Keep baseline choice independent of the target optical labels.
3. **Reuse the confirmed source-model exports and run E4.** Complete the existing
   R/Python reconstruction check and preserve its receipt. If a future runtime
   cannot access those exports, retain descriptive novelty and record the named
   missing inputs rather than generating a substitute metric with a stronger
   label.
4. **Run E5 and E6 only with matched source calibration inputs.** Compare complete
   pipelines with a frozen source/target protocol. A candidate progresses if it
   improves the intended baseline behavior without unacceptable source-domain
   degradation; target optical results are reported as a separate consequence.
5. **Publish E7's separate outcomes.** Record which experiments ran, the failures
   as well as favorable results, and the remaining identifiability limits.

Recovering and reusing an existing calibration export is an access/provenance
task. It should not be confused with a need to fund new sampling. New Ethiopian
reference measurements would answer a different question, but the research and
tests above can substantially narrow the current problem before that becomes
possible.
