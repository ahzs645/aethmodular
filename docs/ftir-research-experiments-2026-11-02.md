# FTIR research additions and existing-data experiments

**2 November 2026 — research synthesis and executed diagnostics**

## Main result

The literature supports concrete tests with the spectra, blanks, archived
calibration exports and chemistry records already available. This investigation
connects **22 distinct research publications, including one preprint**, plus a
separately identified review, to four executed experiment families. The evidence
supports improving diagnosis and deciding which complete pipelines deserve
validation. It does not select a validated Ethiopian EC correction.

The most useful new findings are:

1. A correction can produce very clean blank spectra while removing broad
   constructed aerosol-like information. Narrow-feature recovery and blank
   cleanliness are insufficient selection criteria.
2. The apparent Addis–Bishoftu contrast changes substantially after collection
   dates are aligned. With MAC 10, the shared-window median FTIR-minus-optical
   differences are −1.393 and −1.266 µg/m³, respectively.
3. The component removed near 1617 cm⁻¹ is most strongly associated with the
   reported FTIR-derived EC product itself. Its association with the disagreement
   does not identify that component as an error.
4. Actual historical PLS applicability and acquisition quality detect different
   problems. The nearly flat ETAD-0037 scan and both ETAD-0139 scans pass the
   archived-model distance screens.

Every numerical result in this document comes from the linked executed
workflows and their hashed evidence tables. Literature findings are identified
separately. Existing source values, calibration products and exclusion decisions
were preserved.

## Research worth adding

The two annotated research notes contain bibliographic details, DOI links,
access level, exact relevance, proposed tests and limits:

- [Baseline and background methods: 11 primary publications](research-baseline-methods-2026-11-02.md).
- [Calibration transfer and optical interpretation: 12 primary publications](research-transfer-and-optics-2026-11-02.md).

Parks et al. appears in both notes, so there are 22 distinct research
publications rather than 23. Of those, 21 are journal articles and one is the
VIBES preprint. Bond & Bergstrom (2006) is an additional, explicitly identified
investigative review. Some papers were available in full text; others were
checked through author/publisher abstracts and metadata. The notes disclose
that distinction instead of implying that every complete paper was accessible.

| Research family | Key references | What the project can test |
|---|---|---|
| Aerosol baseline correction and reproducible pipelines | Kuzmiakova et al. 2016; Reggente et al. 2019 AIRSpec | Fitting regions, signed blanks, repeat consistency, and complete preprocessing/calibration provenance. |
| Broad carbon-related information | Parks et al. 2021; Debus et al. 2022; Pollard et al. 1990 | Retention of broad continua and multiple spectral regions. Quartz/reflection and historical PTFE coefficients are not imported. |
| Adaptive smooth baselines | Baek et al. 2015 arPLS; Zhang et al. 2010 airPLS | Width/amplitude stress tests and explicit convergence. These methods are distinct from AIRSpec and from one another. |
| Explicit background models | Martens & Stark 1991; Solheim et al. 2022 | Additive blank models, protected analyte bands, conditioning and signal/background overlap. |
| Probabilistic interference removal | Kamper et al. 2026 VIBES, **preprint** | The same held-out blank and signal-preservation tests, using its actual trained bank and optimizer. |
| Atmospheric water and CO₂ | Ahmed et al. 2024 | Sequential-repeat subtraction only when acquisition metadata establish immediately successive stable-sample measurements. |
| FTIR EC transfer and spectral selection | Dillner & Takahama 2015; Reggente et al. 2016; Weakley et al. 2018; Takahama et al. 2016 | Actual-model PLS applicability, matched source-TOR recalibration, grouped validation and region ablation. |
| Optical comparator behavior | White et al. 2025; Yus-Díez et al. 2025; Zhao et al. 2021 | Loading effects and wavelength-specific MAC assumptions; instrument-specific corrections cannot be borrowed without justification. |
| Carbon definitions and protocols | Pileci et al. 2021; Chan et al. 2019 | Keep optical eBC, refractory BC and protocol-specific EC distinct; preserve reference provenance. |

## 1. Executed baseline challenge

Full methods and results: [baseline-challenge-results-2026-11-02.md](baseline-challenge-results-2026-11-02.md).

The experiment uses 37 physical field blanks from 39 native scans. It adds six
known shapes at two amplitudes, compares seven method settings, and holds each
physical blank out of any learned background bank. A second evaluation holds
out the entire known filter lot. The complete ledger contains **4,116
method-specific injection records**; these are not independent field samples.

There are **77 nonconverged method-specific injection records**, all retained.
The comparison below uses identical successful cases across all seven methods:
373 challenges per method across 34 physical blanks. Medians pool the two
amplitudes for each shape.

| Method | Narrow 1617 addition, sigma 8 cm⁻¹ | Broad 1617 addition, sigma 120 cm⁻¹ | Broad linear continuum |
|---|---:|---:|---:|
| Native spline | 100.0% | **33.8%** | ~0% |
| Whittaker arPLS, lambda 10⁴ | 97.5% | 78.5% | ~0% |
| Whittaker arPLS, lambda 10⁶ | 100.0% | 99.7% | ~0% |
| Whittaker arPLS, lambda 10⁸ | 100.0% | 100.5% | ~0% |
| Independent blank mean | 100.0% | 100.0% | 100.0% |
| Blank PCs plus constant/trend | 94.4% | **22.5%** | ~0% |
| Same model, protected bands | 100.0% | **62.8%** | ~0% |

These percentages describe **recovery of artificial additions**, not actual EC
recovery. Width is Gaussian standard deviation, not FWHM. The full table also
includes the intermediate 1617 width, C–H and O–H additions, shape distortion,
and amplitude-specific outcomes.

The blank-PCA methods have the smallest held-out blank RMS, about 0.000080 and
0.000096 absorbance, compared with 0.000569 for the spline on the shared
34-blank success cohort. Yet they lose substantial broad signal. Conversely,
blank-mean subtraction preserves an added signal by algebra, but its median
blank RMS is much larger, about 0.013315. Neither endpoint alone identifies a
successful correction.

All flexible methods tested remove the linear continuum. For an additive
constant/trend fit, that is an identifiability limit: the constructed continuum
lies in the nuisance space. Protecting the 1500–1800 interval does not protect a
signal that also extends across the fitting channels outside it.

## 2. Executed optical sensitivity and Ethiopia date alignment

Full methods and results: [optical-sensitivity-results-2026-11-02.md](optical-sensitivity-results-2026-11-02.md).

The current public ledger contains 5,069 physical-filter records at 38 sites.
The experiment uses **2,500 eligible direct-HIPS pairs**; 873 eligible
public-HIPS extension pairs remain separately labeled and are not treated as
direct measured Fabs.

For the two Ethiopian sites, illustrative MAC scenarios at the HIPS wavelength
of **633 nm** produce:

| Assumed MAC, m²/g | Addis median EC minus optical BC, µg/m³ (n=232) | Bishoftu median, µg/m³ (n=21) |
|---|---:|---:|
| 7.5 | −1.710 | −2.065 |
| 10 | −0.329 | −1.266 |
| 12.5 | +0.465 | −0.712 |
| 15 | +1.063 | −0.291 |

These MACs are scenarios, not fitted estimates or recommendations. In
particular, the frequently quoted 7.5 m²/g value in Bond & Bergstrom's review
refers to 550 nm and cannot be imported at 633 nm without a wavelength model.
The experiment does not optimize MAC to force agreement.

More revealing is the common collection-date window, **5 November–29 December
2025**:

| Comparison at MAC 10 | Addis | Bishoftu |
|---|---:|---:|
| Directly paired physical filters | 12 | 21 |
| Median FTIR EC minus optical BC | **−1.393 µg/m³** | **−1.266 µg/m³** |
| Filters with FTIR EC below optical BC | 12 of 12 | 21 of 21 |

The difference between the full-period site medians is therefore sensitive to
which dates are represented. This is a common date range, **not collocation,
same-lot matching, shared acquisition runs, or proof of a shared bias**. The
33-filter panel is exported for follow-up inspection.

Per-filter `Fabs / FTIR EC` and denominator-threshold sensitivity are retained
as algebraic effective ratios. They are not independently measured MACs.

## 3. Executed response decomposition

Full methods and results: [spectral-response-results-2026-11-02.md](spectral-response-results-2026-11-02.md).

Each indicator is compared with three different responses: reported FTIR EC,
measured direct HIPS absorption, and EC minus HIPS/10. Every comparison below
uses the same 232 physical filters and identical controls: PM2.5, sampled
volume, monthly sine/cosine, collection time, and lot. Neither carbon/optical
response is itself a control.

| Indicator | Partial rank r² with FTIR EC | With measured HIPS Fabs | With EC minus HIPS/10 |
|---|---:|---:|---:|
| Raw 1617 | 0.716 | 0.197 | 0.364 |
| Corrected 1617 | 0.041 | 0.019 | 0.072 |
| Removed 1617 | **0.707** | **0.233** | **0.339** |
| Raw broad contrast near 1600 | 0.124 | 0.015 | 0.203 |
| Raw C–H contrast | 0.529 | 0.258 | 0.143 |

The strongest removed-component association is with the FTIR-derived product
itself. That is consistent with recovering behavior of an estimator already
based on spectra. It does not determine whether the removed component is
background, real aerosol absorption, or a mixture. Signed correlations and all
eight screened indicators remain in the evidence tables.

These adjusted associations are not directly interchangeable with the earlier
screen's r² values: the earlier screen included optical concentration among its
controls and used a different loading/volume specification.

## 4. Executed historical PLS applicability screen

Full methods and results: [calibration-domain-results-2026-11-02.md](calibration-domain-results-2026-11-02.md).

The existing exports reconstruct one archived raw-spectrum EC PLS model:
**906 independent IMPROVE filters, 140 sites, 31 components, 2,722 channels**.
The source collection dates span 17 May 2022–30 April 2023. Native and source
grids match exactly at the export's float32 precision. No interpolation,
baseline replacement, normalization or target outcome fitting occurs.

Reconstructed fitted values reproduce the saved R values with a maximum
absolute difference of about 5.08×10⁻⁵ in the exported response units. This
verifies reconstruction; it is not a new held-out accuracy result.

The workflow reuses the existing PLS score and residual helpers, while naming
their normalization precisely. For 906 training filters, covariance-normalized
squared Mahalanobis distance is **905 times the leverage-scaled score distance**.
Q is the summed squared full-grid spectral reconstruction residual.

Among the **253 native Addis PM2.5 physical-filter means**:

- 80 (31.6%) exceed either source training 95th-percentile D²/Q reference.
- 18 (7.1%) exceed either 99th-percentile reference; those 18 exceed Q.
- ETAD-0037 and both ETAD-0139 acquisitions pass both screens despite their
  independently identified acquisition problems.

The thresholds are descriptive **in-sample training percentiles**, not
validated quality limits, error probabilities, or automatic exclusion rules.
The archived model is not asserted to be the current public FTIR EC product.
Its original per-filter held-out predictions and fold memberships were not
provided; historical aggregate CV values remain explicitly labeled as such.

The new domain shortlist prioritizes ETAD-0308, ETAD-0006, ETAD-0008,
ETAD-0004 and ETAD-0022 for their high Q values. ETAD-0070 adds a repeat-level
question: its three scan D² values are approximately 676.6, 114.8 and 130.8,
while the mean spectrum has D² 201.0. The mean conceals some scan-level
extremeness. These are review priorities, not newly excluded filters.

## What is ready and what requires another experiment

| Work | Status and next requirement |
|---|---|
| Signed blank and signal-recovery benchmark | Executed; reusable workflow, complete failure records, common-case summaries and figure. |
| Known-lot background transfer | Executed on 28 known-lot blanks; only two known lots and unequal banks limit interpretation. |
| Direct-HIPS MAC and collection-date sensitivity | Executed; 33-filter Ethiopian shared-window panel exported. |
| Response decomposition | Executed on 232 matched current direct-HIPS filters. |
| Actual archived-model PLS D²/Q | Executed with exact preprocessing/grid verification and calibration-specific normalization. |
| VIBES in the new injection benchmark | Protocol specified; the additive-PCA results are not VIBES results. |
| airPLS comparison | Literature reviewed; not executed as another algorithm in this benchmark. |
| Ahmed sequential atmospheric correction | Requires existing acquisition timestamps/background IDs establishing immediately successive stable repeats. Arbitrary repeated scans do not meet that requirement. |
| New matched preprocessing/TOR calibration comparison | Still a separate experiment; source calibration exports are now located, so this does not inherently require new Ethiopian sample collection. |
| Ethiopian EC accuracy | Not established by these optical and spectral diagnostics. |

## Recommended next order

First, preserve the scan-level review queue and inspect its acquisition records.
Next, compare a small frozen set of complete raw/spline/alternative-background
calibration pipelines using the existing source TOR data, with physical-filter
and source-site holdouts. Retain width-dependent signal preservation, blank
transfer, repeat consistency and domain distance as separate outcomes. Only then
compare the frozen Ethiopian predictions across the same filters and dates.

The important outcome of this research is a better-defined experiment:
**can a pipeline preserve the useful broad information, control the background,
and retain source-reference performance when the complete calibration is
evaluated consistently?** Existing data already allow substantial testing of
that question. A smaller Ethiopian optical discrepancy alone cannot answer it.

## Reproduce

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/run_baseline_challenge.py \
  --native-dir "$AETHMODULAR_ETAD_NATIVE_DIR" --jobs 4
uv run python research/ftir_hips_chem/workflows/run_optical_sensitivity.py
uv run python research/ftir_hips_chem/workflows/run_spectral_response_screen.py \
  --native-dir "$AETHMODULAR_ETAD_NATIVE_DIR"
uv run python research/ftir_hips_chem/workflows/run_calibration_domain_screen.py \
  --ftir-dir "$AETHMODULAR_FTIR_DIR" --native-dir "$AETHMODULAR_ETAD_NATIVE_DIR"
```

Each workflow writes to its own directory under
`research/ftir_hips_chem/output/tables/`, with input/code hashes and explicit
cohort membership. Raw native and calibration source files remain external to
version control. The standalone evidence package contains the generated tables,
receipts, research notes and scientific figure.

## Verification

The focused suite passes **51 tests**, covering the new workflows and their
portable cohort/baseline dependencies. Checks include physical-filter identity,
held-out blank independence, protected-band behavior, failure retention,
common-case comparisons, unit conversions, optical-source separation,
zero-denominator handling, response-free nuisance adjustment, PLS distance
normalization, exact grids, and chronology. Ruff passes on the new code and
tests. All four generated receipts' code hashes match the delivered files.

These implementation checks support reproducibility. They do not turn the
constructed-signal experiments, source training references, or optical
comparisons into independent Ethiopian EC validation.
