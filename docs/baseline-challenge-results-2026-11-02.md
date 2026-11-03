# Baseline signal-preservation challenge — 2 November 2026

## Result

The native spline preserves an added narrow 1617 cm⁻¹ feature but retains only
about one-third of a much broader constructed feature at the same center. An
additive blank-PCA model produces smaller held-out blank residuals while removing
even more of that broad feature. Protecting analyte windows improves that model's
retention, but does not preserve a linear continuum that lies in its nuisance
span. These are concrete failure modes for evaluating preprocessing; they do
not measure actual EC loss or establish which background is correct.

All results below were calculated from the existing native Addis export. No EC
value, optical MAC, original scan, or canonical exclusion was changed. The
seven representations were not fed into unchanged EC coefficients and called
recalibrated products.

## Why this experiment follows from the research

[Kuzmiakova et al. (2016)](https://doi.org/10.5194/amt-9-2615-2016) evaluates
background removal through aerosol and blank behavior and matched TOR models.
[Parks et al. (2021)](https://doi.org/10.1080/02786826.2021.1917764) uses a broad
infrared slope as carbon information in quartz/reflection measurements; its
coefficients are not transferable to these PTFE/transmission spectra.
[Solheim et al. (2022)](https://doi.org/10.3390/molecules27061900) distinguishes
analyte and interfering constituents and discusses ambiguous background models.
Together they motivate testing blank removal and known-signal retention as
separate endpoints. The exact constructed signals below are analyst-designed
stress tests, not published pure-EC spectra.

## Inputs and experimental design

- 319 native scans of 296 physical filters; 2,722 channels in the original grid.
- 39 blank scans aggregated to 37 independent physical field blanks.
- Common comparison domain: 2,002 native channels, strictly 1425–4000 cm⁻¹,
  spaced by approximately 1.285667 cm⁻¹. Missing spline output below this domain
  is never treated as zero.
- Six constructed shapes: Gaussians centered at 1617 cm⁻¹ with standard
  deviations 8, 40 and 120 cm⁻¹; a C–H-region Gaussian at 2920 with sigma 18;
  an O–H-region Gaussian at 3300 with sigma 150; and a broad linear continuum.
  The latter rises from 0.2 to 1.0 across the modeled domain before amplitude
  multiplication. Gaussian widths are **standard deviations, not FWHM**.
- Two nominal amplitudes: 0.002 and 0.010 absorbance.
- 481 spectra for sample-specific correction: each of 37 blanks, plus its 12
  signal additions. There are 444 distinct blank/shape/amplitude challenges.
- Three background methods are additionally evaluated with the entire known
  target lot withheld. This includes 5 lot-248 blanks and 23 lot-251 blanks;
  9 unknown-lot blanks cannot establish cross-lot independence and are excluded
  from the known-lot training and evaluation banks.
- 4,116 method-specific injection records in total. These are repeated
  evaluations of the same physical blanks, not 4,116 independent samples.
- 45 actual repeated scans of 22 physical filters yield 24 within-filter
  pairwise comparisons per method, retained as a separate outcome.

For correction operator C, measured blank b, and added signal s, recovery uses

\[
\Delta=C(b+s)-C(b),\qquad
g=\frac{\langle\Delta,s\rangle}{\langle s,s\rangle}.
\]

The tables also retain total recovery error and shape distortion. Gain can be
negative or exceed one; those values are not clipped. A gain of one alone does
not establish correct shape or removal of the underlying background.

Every physical blank is excluded from its training bank. Rank and windows were
fixed in advance of these outcomes. No HIPS or FTIR EC outcome was used for
parameter selection. Cross-lot fits use only other **known** lots, avoiding the
assumption that an unknown lot is different from the target lot.

## Seven method settings

| Setting | Definition |
|---|---|
| Native spline | Existing AIRSpec port, segment degrees of freedom 6 and 4. Its earlier R reproduction is documented separately. |
| Whittaker arPLS, lambda 10⁴ | Independent second-difference/logistic-weight implementation on the fixed native grid. |
| Whittaker arPLS, lambda 10⁶ | Same implementation, larger smoothness penalty. |
| Whittaker arPLS, lambda 10⁸ | Same implementation, larger smoothness penalty. |
| Blank mean | Subtract the mean of independent training blanks; no spectrum-specific coefficient fitting. |
| Blank PCs + trend | Mean plus three training-blank PCs, constant and linear trend; coefficients fitted to all modeled channels. |
| Blank PCs + trend, protected | Same background basis, but fit coefficients outside 1500–1800, 2200–2450, 2800–3050 and 3100–3600 cm⁻¹. |

The additive models are **EMSC-inspired**, not full EMSC and not VIBES. They do
not divide by a fitted reference coefficient or otherwise normalize loading.
Their component counts, design conditioning and bank membership are recorded.

The independent arPLS implementation follows the logistic-weight/Whittaker
formulation associated with [Baek et al. (2015)](https://doi.org/10.1039/C4AN01061B)
and checked against official pybaselines documentation. It uses population
standard deviation (`ddof=0`) for negative residuals, relative L2 weight
tolerance 0.001 and at most 100 iterations. Numerical equivalence to original
MATLAB or other arPLS implementations has not been established. This is also
different from the repository's existing **penalized-spline** arPLS neutral
cache; their lambda values are not interchangeable.

## Signal retention on identical successful-fit cases

There were 77 nonconverged method-specific injection records. All remain in the
ledger. A fair comparison uses only the same blank/shape/amplitude cases for
which all seven methods converged: **373 challenges per method across 34
physical blanks**. Each shape/amplitude cell has 23–34 blanks. These counts
are recorded separately rather than treating every cell as n=37.

The following medians pool the two amplitudes within each shape on that shared
successful cohort. Percentages are recovery of the **constructed addition**.

| Method | 1617, sigma 8 | 1617, sigma 40 | 1617, sigma 120 | C–H, sigma 18 | O–H, sigma 150 | Linear continuum |
|---|---:|---:|---:|---:|---:|---:|
| Native spline | 100.0% | 96.3% | **33.8%** | 100.0% | 93.3% | ~0% |
| arPLS, lambda 10⁴ | 97.5% | 98.8% | 78.5% | 100.0% | ~0% | ~0% |
| arPLS, lambda 10⁶ | 100.0% | 100.0% | 99.7% | 100.0% | 79.5% | ~0% |
| arPLS, lambda 10⁸ | 100.0% | 100.0% | 100.5% | 100.2% | 91.4% | ~0% |
| Blank mean | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Blank PCs + trend | 94.4% | 71.0% | **22.5%** | 94.4% | 35.3% | ~0% |
| Blank PCs + trend, protected | 100.0% | 99.9% | **62.8%** | 100.0% | 80.8% | ~0% |

The blank-mean control preserves additions algebraically:
`(b + s - mean) - (b - mean) = s`. Its perfect recovery therefore cannot select
it as the best correction. It must also remove realistic sample-specific
background variation, which this experiment does not establish.

The loss of a linear continuum in the flexible methods is similarly structural.
That continuum is indistinguishable from an allowed offset/trend. Protecting a
finite analyte window does not prevent a broad signal extending outside it from
being fitted as background. The relevant scientific question remains which
parts of the measured broad signal belong to aerosol and which to interference.

## Blank residuals do not rank signal preservation

For the **34 blanks with successful unspiked fits from every method**, median
corrected RMS absorbance is:

| Method | Median held-out blank RMS |
|---|---:|
| Native spline | 0.000569 |
| arPLS, lambda 10⁴ | 0.000532 |
| arPLS, lambda 10⁶ | 0.000843 |
| arPLS, lambda 10⁸ | 0.000707 |
| Blank mean | 0.013315 |
| Blank PCs + trend | **0.000080** |
| Blank PCs + trend, protected | **0.000096** |

The PCA-based methods have the smallest blank residuals, while the injection
table shows substantial loss of broad features. Both outcomes must remain
visible. The blank-only shared cohort is defined separately from the
injection-success intersection.

Known-lot holdout exposes a second issue. The full-window blank-PCA median
residual RMS is 0.000875 for the five held-out lot-248 blanks and 0.000346 for
the 23 held-out lot-251 blanks. Protected-window results are 0.001166 and
0.000430, respectively. The corresponding training banks contain 23 and only
5 physical blanks. These are small, unequal lot-transfer experiments, not
evidence that lot alone causes loaded-filter EC disagreement.

## Actual repeated scans

The repeat ledger retains every comparison and convergence state. No scan is
selected because its derived quantity agrees better with HIPS. ETAD-0139's
flat/normal pair remains highly discordant under all tested corrections;
correction cannot restore signal absent from an acquisition. For the native
spline the difference RMS is 0.008617 absorbance and the difference relative to
the mean-spectrum RMS is about 2.03. These amplitude diagnostics are separate
from high shape correlation in the other repeated filters.

## Interpretation and next experiments

1. Retain both blank removal and signal preservation in every baseline comparison.
2. Carry selected smoother/background candidates into **matched source-TOR
   recalibration**, preserving the broad input representations as separate
   candidates. Do not substitute a corrected spectrum into coefficients trained
   on raw spectra.
3. Add the actual VIBES model to this same physical-blank challenge when its
   fitted bank and solver are available in the execution environment. The
   additive PCA results do not stand in for it.
4. Evaluate new banks with physical-filter and known-lot holdouts. Unknown lot
   cannot establish independence. Analytical-run holdouts need acquisition IDs.
5. Keep Ethiopian optical agreement as a separate transfer diagnostic; use the
   [optical sensitivity results](optical-sensitivity-results-2026-11-02.md) and
   [response decomposition](spectral-response-results-2026-11-02.md) to avoid
   confusing a changed estimator or MAC assumption with improved EC accuracy.

## Reproduction and evidence

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/run_baseline_challenge.py \
  --native-dir "$AETHMODULAR_ETAD_NATIVE_DIR" --jobs 4
uv run pytest tests/test_baseline_challenge.py
```

The source folder contains `ETAD_FTIR_spectra.csv`, `ETAD_metadata.csv`,
`etad_spectra_lotmap.csv`, and `spectra_baselined_AIRSPEC.csv`. These raw inputs
remain external to version control. The run uses Python 3.12.14, NumPy 2.4.2
and pandas 2.3.3; the repository recommends Python 3.13.

Outputs are in `research/ftir_hips_chem/output/tables/baseline_challenge_20261102/`:

- `injection_trials.csv`: every trial, convergence and common-comparison flag.
- `injection_summary.csv`: per-method success summaries and failure counts.
- `common_converged_injection_summary.csv`: identical-case comparisons.
- `heldout_blank_metrics.csv`: signed residuals, RMS and convergence.
- `background_training_ledger.csv` and `background_model_ledger.csv`: exact
  training identities, lots, ranks, fitting channels and conditioning.
- `arpls_solver_status.csv`: all 1,443 sample-specific arPLS fit records.
- `repeat_scan_metrics.csv` and `repeat_pair_metrics.csv`: original-repeat audit.
- `physical_blanks.csv`, `manifest.json`, and the standalone PNG/SVG figure.

The manifest snapshots source/code hashes before computation. A post-run check
confirmed that its four code hashes match the delivered implementation.
