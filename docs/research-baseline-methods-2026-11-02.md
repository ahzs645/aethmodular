# Research additions for testing FTIR baseline and background assumptions

Literature checked: 2 November 2026. Scope: methods that can be investigated with
existing spectra, field blanks, repeat scans, metadata and existing calibration
data. This is a research and experiment specification; numerical results from
new runs belong in the run receipt and its companion results report.

## What this research changes

The most useful question is whether a method can remove a held-out background
while retaining a known added signal. A small blank residual alone cannot answer
that question. The aerosol spline paper explicitly describes background
separation as ill-posed [B1], and a subsequent diesel-aerosol study uses a broad
infrared slope as an EC predictor [B4]. A correction that makes a spectrum look
flat can therefore remove information worth retaining.

Three method families deserve distinct tests: sample-specific spline or penalized
smooth baselines; background models learned from independent blanks; and
corrections for specific atmospheric interferences. Each makes different
assumptions. None is an independently validated Ethiopian EC correction merely
because it improves the appearance of a spectrum or agreement with HIPS.

The existing native audit establishes the available Addis test population:
319 scans of 296 physical filters, including **39 scans of 37 physical field
blanks**. Repeated scans of a blank cannot count as separate training and test
filters. See [the native audit](review-native-addis-baseline-2026-11-02.md) for
input hashes, identities, cohort reconciliation and acquisition-review flags.

## Verified primary-source reading list

The claims in this table are summaries of the cited research, not new results
from aethmodular. “Full text” means the accessible article body or PDF was
examined; “abstract” limits the extraction to the authors' abstract. Citation
dates use the publication year, which can differ from an online-posting date.

| ID | Verified publication and access | Finding relevant to this project | Proposed test and limit |
|---|---|---|---|
| **B1** | Kuzmiakova, A., Dillner, A. M., and Takahama, S. (2016). *An automated baseline correction protocol for infrared spectra of atmospheric aerosols collected on polytetrafluoroethylene (Teflon) filters*. Atmospheric Measurement Techniques 9, 2615–2631. [DOI: 10.5194/amt-9-2615-2016](https://doi.org/10.5194/amt-9-2615-2016). Full text. | Fits smoothing splines through defined background regions and evaluates blank behavior, aerosol applications and matched TOR predictions. Discusses broad aerosol features, imperfect blank subtraction and positive bias introduced by forcing nonnegativity. | Retain the native spline as a reference; test fitting boundaries as well as degrees of freedom. Preserve signed residuals. Its IMPROVE validation does not establish transfer to Ethiopia. |
| **B2** | Reggente, M., Höhn, R., and Takahama, S. (2019). *An open platform for Aerosol InfraRed Spectroscopy analysis – AIRSpec*. Atmospheric Measurement Techniques 12, 2313–2329. [DOI: 10.5194/amt-12-2313-2019](https://doi.org/10.5194/amt-12-2313-2019). Full text. | Connects baseline correction, functional-group fitting and multivariate calibration, with parameter files and diagnostic outputs supporting reproducibility. | Save the actual grid, masks, algorithm version, parameters, corrected spectra and removed components for every run. AIRSpec is a framework; its name does not identify a unique preprocessing/calibration version. |
| **B3** | Takahama, S., Johnson, A., and Russell, L. M. (2013). *Quantification of Carboxylic and Carbonyl Functional Groups in Organic Aerosol Infrared Absorbance Spectra*. Aerosol Science and Technology 47, 310–325. [DOI: 10.1080/02786826.2012.752065](https://doi.org/10.1080/02786826.2012.752065). Publisher article and abstract. | Develops functional-group interpretation using constrained line shapes and calibrated absorptivities for selected compounds and mixtures. | Add multiple-band composition contrasts and line-shape checks. A feature around 1617 cm⁻¹ alone cannot uniquely identify a compound, source or charcoal contribution. Functional-group fitting is not an EC reference. |
| **B4** | Parks, D. A., Griffiths, P. R., Weakley, A. T., and Miller, A. L. (2021). *Quantifying elemental and organic carbon in diesel particulate matter by mid-infrared spectrometry*. Aerosol Science and Technology 55, 1014–1027. [DOI: 10.1080/02786826.2021.1917764](https://doi.org/10.1080/02786826.2021.1917764); [author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC8563026/). Full article content retrieved through search; direct PMC opening was intermittently challenged. | Uses a broad sloping feature, including an integrated 4000–3796 cm⁻¹ interval, to predict EC in diesel particulate matter. The measurement is reflection-mode FTIR on quartz filters with thermal-optical calibration. | Test whether baseline methods destroy broad ramps or curved continua. Do not transfer the published coefficient, normalization or quantitative performance to PTFE transmission spectra or Ethiopian ambient mixtures. |
| **B5** | Pollard, M., Jaklevic, J., and Howes, J. (1990). *Analysis of Carbon Particle Deposition on Teflon Filters Using Fourier Transform Infrared Spectroscopy*. Aerosol Science and Technology 12, 182–193. [DOI: 10.1080/02786829008959338](https://doi.org/10.1080/02786829008959338). Abstract and opening PDF text. | Compared PTFE transmission absorbance in the 650–666 cm⁻¹ interval with thermal-combustion EC from concurrently collected quartz filters. | Add that interval as an exploratory lower-wavenumber diagnostic when its transmission and background behavior are usable. It lies outside the current spline baseline domain; the historical calibration cannot be imported as a current EC formula. |
| **B6** | Baek, S.-J., Park, A., Ahn, Y.-J., and Choo, J. (2015). *Baseline correction using asymmetrically reweighted penalized least squares smoothing*. Analyst 140, 250–257. [DOI: 10.1039/C4AN01061B](https://doi.org/10.1039/C4AN01061B); [PubMed](https://pubmed.ncbi.nlm.nih.gov/25382860/). Abstract. | Introduces arPLS, using a generalized logistic weighting scheme that adapts to estimated noise while fitting a penalized smooth baseline. Demonstrations include simulated and measured Raman spectra. | Use arPLS as a neutral comparison, with smoothing and convergence recorded. Test broad-signal retention explicitly; successful Raman peak recovery does not validate aerosol FTIR background removal. |
| **B7** | Zhang, Z.-M., Chen, S., and Liang, Y.-Z. (2010). *Baseline correction using adaptive iteratively reweighted penalized least squares*. Analyst 135, 1138–1146. [DOI: 10.1039/B922045C](https://doi.org/10.1039/B922045C); [PubMed](https://pubmed.ncbi.nlm.nih.gov/20419267/). Abstract and publisher search text. | Introduces airPLS, an iterative residual-dependent weighting method for baseline estimation without manually identifying peaks. | A later alternative to arPLS if the first benchmark identifies a specific failure. airPLS and arPLS are different algorithms; neither is AIRSpec. An algorithm's automated weights do not remove the need to choose and validate smoothing. |
| **B8** | Martens, H., and Stark, E. (1991). *Extended multiplicative signal correction and spectral interference subtraction: new preprocessing methods for near infrared spectroscopy*. Journal of Pharmaceutical and Biomedical Analysis 9, 625–635. [DOI: 10.1016/0731-7085(91)80188-F](https://doi.org/10.1016/0731-7085(91)80188-F); [PubMed](https://pubmed.ncbi.nlm.nih.gov/1790182/). Abstract. | Separates modeled path-length/scattering effects from known spectral interferences in a multivariate preprocessing framework. | Test additive blank-basis and low-order nuisance fits as an EMSC-inspired alternative. Full EMSC scaling changes concentration information; keep the scaling coefficient or retrain a model designed for that representation. |
| **B9** | Solheim, J. H., Zimmermann, B., Tafintseva, V., Dzurendová, S., Shapaval, V., and Kohler, A. (2022). *The Use of Constituent Spectra and Weighting in Extended Multiplicative Signal Correction in Infrared Spectroscopy*. Molecules 27, 1900. [DOI: 10.3390/molecules27061900](https://doi.org/10.3390/molecules27061900); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8948808/). Full text. | Distinguishes analyte and interference constituents, discusses orthogonality, and demonstrates how weighting and normalization choices change the result. | Compare full-window versus protected-window background fits; report conditioning and signal–background overlap. A background component resembling an analyte cannot be declared harmless simply because it came from PCA. |
| **B10** | Ahmed, W., Osborne, E. L., Veluthandath, A. V., and Murugan, G. S. (2024). *A Rapid and Simplified Approach to Correct Atmospheric Absorptions in Infrared Spectra*. Analytical Chemistry 96, 18052–18060. [DOI: 10.1021/acs.analchem.4c03594](https://doi.org/10.1021/acs.analchem.4c03594); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11561874/). Full text. | Estimates atmospheric interference from the difference between two immediately successive sample spectra, then optimizes its subtraction by spectral-length minimization. Strong sample peaks can bias the optimization window. | Use only repeats whose acquisition timing and sample stability support the method. Estimate water correction outside a protected 1500–1800 cm⁻¹ region and evaluate the protected region afterward. Arbitrary repeats, especially flat/normal pairs, are not pure atmospheric references. |
| **B11** | Kamper, F., Krymova, E., Dillner, A. M., Baccarini, A., Dudani, N., Nenes, A., Obozinski, G., and Takahama, S. (2026). *Towards Universal Calibration for Infrared Spectroscopy with Probabilistic Interference Removal*. ChemRxiv, version 1, 21 September 2026. [DOI: 10.26434/chemrxiv.15009188/v1](https://doi.org/10.26434/chemrxiv.15009188/v1). **Preprint; abstract and metadata.** | Presents VIBES: blank-informed covariance, probabilistic analyte/background separation and estimated regularization. Reports demonstrations involving sharp and broad signals. | Evaluate the vendored implementation with a frozen blank bank, physical-filter holdouts, rank sensitivity and signal additions. The preprint does not demonstrate Ethiopian EC accuracy. Its landing page advertises code at [Zenodo DOI 10.5281/zenodo.22831848](https://doi.org/10.5281/zenodo.22831848); that archive was not retrieved during this literature check. |

## Experiments with the current data

### 1. Signal-preservation curves on independent field blanks

For each held-out physical blank, apply a baseline method twice: to its measured
spectrum, and to that spectrum plus a known constructed absorbance signal. With
correction operator \(C\), blank \(b\), signal shape \(s\) and amplitude \(a\), use

\[
\Delta = C(b + a s) - C(b),\qquad
g = \frac{\langle \Delta, a s\rangle}{\langle a s, a s\rangle}.
\]

The gain \(g\) measures retention in the known signal direction. Also save
shape error \(\|\Delta-a s\|/\|a s\|\), signed integrated-area recovery and
off-feature changes. A gain near one with large shape error is not a successful
recovery. Values outside zero to one are possible and should remain visible.

Use a frozen set of narrow and broad shapes: a peak near 1617 cm⁻¹, a carbonyl
region peak, a C–H-region feature, broad humps and a sloping continuum. Sweep
width and amplitude rather than choosing one attractive example. Explicitly
define Gaussian widths as standard deviation or FWHM; do not mix conventions.
These synthetic shapes are stress tests, not measured pure-EC spectra or
estimates of actual EC loss. Their purpose follows B1 and B4, while the exact
shape grid is an analyst-designed experiment.

For learned backgrounds, neither scan of a held-out blank may enter its training
mean or PCA basis. Repeat-level outputs can be retained, but summarize by
physical blank. Fit any tuned rank or regularization within training data. Save
the held-out identities, training identities and parameters in each receipt.

Include raw spectra as an identity control and a training-blank mean subtraction
as a fixed additive control. For a fixed mean, the incremental signal is
preserved algebraically; that does **not** demonstrate that the background of an
unseen loaded filter has been removed correctly. The control tests both the
measurement code and the distinction between background removal and signal
retention.

### 2. A background-removal versus signal-retention comparison

For each method, present signed blank residuals and added-signal recovery
together. Relevant blank summaries include median residual, RMS, between-blank
spread, negative fraction and local shape around 1617 cm⁻¹. Do not clip negative
values or normalize each blank to unit area. A method can make every blank
small and still erase real broad signal.

Initial comparisons should be the existing spline baseline, arPLS, independent
blank-mean subtraction and additive blank-PCA fits. VIBES belongs in the same
evaluation design, with its actual learned background bank and optimization
recorded. A simple PCA least-squares subtraction is **not** VIBES, even though
both use blank information.

For an independent Whittaker arPLS implementation, document the objective
\(\sum_i w_i(y_i-z_i)^2+\lambda\|D_2z\|^2\), its second-difference penalty,
negative-residual noise estimate, and logistic weight update. The official
[pybaselines algorithm documentation](https://github.com/derb12/pybaselines/blob/main/docs/algorithms/whittaker.rst)
gives \(w_i=[1+\exp(2(d_i+m_- -2s_-)/s_-)]^{-1}\), where \(d=y-z\) and
\(m_-\), \(s_-\) describe the negative residuals. Its
[reference implementation](https://pybaselines.readthedocs.io/en/latest/_modules/pybaselines/whittaker.html)
uses the relative change in weights as a stopping criterion; the
[utility definition](https://pybaselines.readthedocs.io/en/latest/generated/api/pybaselines.utils.relative_difference.html)
is \(\|w_{new}-w\|_2/\|w\|_2\). These are software-source checks, separate
from the abstract-level literature extraction for B6. Record the exact standard
deviation convention, numerical guards and convergence history. Whittaker arPLS
and the repository's existing penalized-spline arPLS variant do not share an
interchangeable smoothing parameter or implementation identity.

Compare common valid wavenumbers first. The spline implementation does not
produce a corrected spectrum below 1425 cm⁻¹; its missing values must not become
zeros. Full-range tests of other methods should be reported separately.

### 3. Protected spectral windows and background identifiability

Fit a blank mean plus a small number of blank-derived components using channels
outside the protected 1500–1800 cm⁻¹ interval; predict the background inside it.
Compare with a fit using all channels. Extend protection to additional analyte
windows only as a preregistered sensitivity. Test ranks and polynomial nuisance
terms separately, retaining any fitted scaling coefficient.

This is an EMSC-inspired test of the role of fitting windows [B8–B9]. It is not
proof that the protected signal is aerosol. Background variation can extend
across the protected region, and broad aerosol signal can extend outside it.
Save the model matrix condition number or rank diagnostics. Where a candidate
signal lies in the allowed nuisance span, it is not identifiable from this fit
alone: extra flexibility may simply move that signal into the removed component.
The broad-ramp injection provides a direct check of that failure mode.

### 4. Blank-bank transfer across filter lots

Where lots are known, compare leave-one-physical-blank-out evaluation with a
stronger held-out-lot test. The latter asks whether an unseen filter lot can be
handled by backgrounds learned from other lots. Report training and test counts
per lot; missing lot is its own provenance state. Sparse or single-filter strata
should not produce a claimed reliable lot effect.

Sampling date and lot are associated in the native cohort. A lot-based change
in loaded filters is therefore not automatically a substrate cause. Use the
blank experiment to test background transfer directly, and use like-month
comparisons for loaded filters where there is actual overlap. Analytical-run
holdouts require run identifiers; the native metadata described in the earlier
audit do not provide them.

### 5. Atmospheric-interference feasibility using repeat scans

The regions identified in B10 include water-vapor absorption at 1205–2072 and
3231–4000 cm⁻¹, and CO₂ absorption at 2208–2442 cm⁻¹. These windows motivate
diagnostics, not automatic removal of everything in them. Condensed-phase aerosol
water and other sample bands can overlap these regions.

Before applying the published sequential-spectrum correction, establish that two
scans are immediately successive measurements of the unchanged filter under
comparable positioning. Require amplitude and shape consistency outside the
atmospheric windows. The current native metadata lack scan times and background
identifiers, so repeat identity alone does not establish these prerequisites.
ETAD-0139's flat/normal pair must not be used as an atmospheric reference.

If the metadata become available from existing acquisition records, estimate the
subtraction coefficient outside the disputed 1500–1800 cm⁻¹ interval and inspect
that interval independently. Report changes to unrelated sample bands and the
stability of the coefficient across plausible estimation windows. Until then,
cross-window covariance and repeat differences remain exploratory diagnostics.

### 6. Wider spectrum and multiple-band contrasts

Add separate descriptors for the broad 1500–4000 cm⁻¹ shape, 1500–1800 regional
curvature, narrow 1600–1635 local contrast, carbonyl region, C–H region and O–H
region. Keep raw, removed and signed corrected versions. B3 supports checking
combinations of bands; B4 motivates preserving the continuum. Neither licenses
assigning a specific Ethiopian source from one channel.

B5 motivates an exploratory 650–666 cm⁻¹ descriptor on usable raw or separately
background-corrected spectra. Because the interval is narrow and near atmospheric
CO₂ bending structure, assess blank and repeat behavior before interpreting it.
Do not splice it into the spline-corrected domain without an explicit method.

### 7. Complete preprocessing-and-calibration evaluation

If accessible existing IMPROVE raw spectra and TOR labels permit it, run a
matched calibration for every spectral representation. Fit preprocessing,
scaling, variable selection and calibration choices inside the allowed training
partitions. Keep physical-filter repeats together. Use site-held-out
cross-validation for calibration-set transfer, then evaluate Ethiopian samples
with the frozen pipeline. The existing training/test conventions and product
version fields still apply.

Keep three endpoints separate: background/signal tests, IMPROVE TOR prediction,
and Ethiopian optical disagreement. Choosing a baseline or its hyperparameters
to minimize Addis HIPS disagreement cannot establish independent EC accuracy.
Changes to a spectrum are not compatible by default with coefficients learned
from a different representation. No new sample collection is required to run
the available-input tests above; the limits concern what their results can
establish.

## Claims and implementation boundaries

| Evidence produced | Defensible interpretation | Interpretation it does not establish |
|---|---|---|
| Agreement with saved spline output | Numerical reproduction of that implementation | Correct physical division between substrate and aerosol |
| Small residual on held-out blanks | Removal of backgrounds represented by the blank test set | Preservation of every aerosol feature |
| Added-signal recovery | Behavior for explicitly chosen constructed shapes and amplitudes | Percentage of real EC retained or lost |
| Better repeat agreement | Reduced variability for those repeated acquisitions | Independent carbon accuracy |
| Better held-out IMPROVE TOR prediction | Better source-domain reference prediction under that split | Validity for every Ethiopian mixture |
| Smaller HIPS–FTIR difference | Closer agreement between the named products and conversion | Proof that the FTIR product is more accurate |
| Similar feature at Addis and Bishoftu | A cross-site spectral pattern worth testing | A common artifact, compound or country-level correction |

Preserve all scans, flags and measured products. Record review decisions at scan
level before computing new physical-filter averages. Label each proposed
experiment as planned, implemented, run, or blocked by a specific missing input;
this reading list alone does not assert that every method has been implemented.

Related project context:
[native Addis audit](review-native-addis-baseline-2026-11-02.md),
[multisite baseline screen](review-baseline-evidence-2026-11-02.md), and
[existing VIBES/spline comparison](pyvibes-vs-airspec.md).
