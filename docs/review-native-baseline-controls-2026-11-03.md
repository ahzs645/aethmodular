# Native baseline controls with existing data — 3 November 2026

**The existing data support a useful baseline diagnostic, but no tested method
passes every reliability check.** The native spline preserves the specified
narrow 1617 cm⁻¹ addition well. Broad smooth additions are poorly preserved by
every method tested. A local rank-8 VIBES model reduces held-out blank residuals,
but its performance degrades when a known filter lot is absent from training.
The neutral spline can converge numerically and still respond unstably to a
small added peak.

No new samples, measurements, EC products, or canonical exclusions were created.
No HIPS/EC outcome selected a method, filter, rank, boundary, or injection.

## Data and fixed comparisons

The four existing ETAD source files contain **319 scans of 296 physical filters**:
253 PM2.5, 37 field blanks and 6 PM10. The native grid has 2,722 channels spaced
1.285667 cm⁻¹. Comparable metrics use the 2,002-channel 1425–4000 cm⁻¹ window;
method fitting domains remain explicit and differ. Twenty-two physical filters
have repeat scans, producing 24 within-filter pairs.

| Method ID | Method actually evaluated |
| --- | --- |
| `spline_default` | Native repository spline, df1 = 6 / df2 = 4; adaptive segment-two minimum in 1520–1600; 20-native-channel extension. |
| `spline_anchor1500` | Same solver and settings; segment-two boundary fixed at the channel nearest 1500 cm⁻¹. |
| `spline_anchor1560` | Same solver and settings; boundary fixed at the channel nearest 1560 cm⁻¹. |
| `neutral_arpls` | Repository `pspline_arpls`, λ = 10⁶, default 50 iterations; fit ≥700 with 1100–1300 omitted. |
| `neutral_arpls_iter500` | Same objective, λ and tolerance; only the iteration ceiling increases to 500. |
| `vibes_etad_r3` | Vendored VIBES, ETAD-only physical-blank bank, fixed rank 3, PB loss, τ = 0.1, full native grid. |
| `vibes_etad_r8` | Same local-bank experiment, fixed rank 8. **Neither local VIBES arm is the completed global rank-70 product.** |

The original native spline reproduces saved R correction with maximum absolute
error **6.0143 × 10⁻⁷ AU**. This verifies implementation fidelity, not physical
correctness. [Methods and validation](../research/ftir_hips_chem/output/tables/baseline_existing_data_20261103/)

## Held-out blanks and actual-blank injections

Five deterministic folds approximately balance lots while grouping by physical
blank. Every scan and injected child of a held-out parent remains outside its
VIBES bank. Training banks contain 29–30 physical blanks; all 37 parents are
evaluated once. The local ambient correction uses all 37 blanks.

Each actual blank receives a Gaussian centered at **1617, σ = 25 cm⁻¹**, or
**2800, σ = 600 cm⁻¹**, at amplitudes 0.001, 0.01 and 0.05 AU. There are 37
independent physical parents and 222 injected cases, not 222 independent blanks.
Recovery is `corrected(parent + addition) − corrected(parent)`, integrated over
1510–1730 or 1500–3990 cm⁻¹. The Gaussian shapes are controls, not identified
ambient species or known amounts of EC.

All-finite medians are shown below, with convergence counts retained separately.
The tables also contain converged-only injection medians.

| Method | Blank median RMS, AU | Converged blank parents | Narrow recovery medians at 0.001 / 0.01 / 0.05 AU | Broad recovery medians at 0.001 / 0.01 / 0.05 AU |
| --- | ---: | ---: | --- | --- |
| Native spline | 0.000551 | 37/37 | 99.93% / 99.93% / 99.93% | 12.22% / 12.22% / 12.39% |
| Boundary 1500 | 0.000547 | 37/37 | 100.00% / 100.00% / 100.00% | 12.21% / 12.21% / 12.38% |
| Boundary 1560 | 0.000561 | 37/37 | 91.98% / 91.98% / 91.98% | 12.24% / 12.24% / 12.40% |
| Neutral, 50 iterations | 0.006675 | 31/37 | 100.00% / 100.00% / 100.00% | −91.14% / −68.68% / 38.10% |
| Neutral, 500 iterations | 0.006671 | 37/37 | 100.00% / 100.00% / 100.00% | −91.14% / −68.71% / 38.10% |
| Local VIBES, rank 3 | 0.000566 | 37/37 | 100.00% / 100.00% / 100.00% | −6.95% / 18.67% / 32.74% |
| Local VIBES, rank 8 | 0.000136 | 37/37 | 99.44% / 99.79% / 99.96% | 4.80% / 18.36% / 25.51% |

Rank 8 has smaller blank RMS than the native spline for all 37 held-out parents;
the median *paired* RMS ratio is 0.2374. This success does not establish broad
signal preservation. Negative paired recovery means adding a positive signal
caused the corrected integrated signal to decrease; it is not a measurement of
negative EC. Nearly perfect protected-peak recovery is also not evidence that
the assumed continuum is chemically correct.

Evidence: [blank metrics](../research/ftir_hips_chem/output/tables/baseline_existing_data_20261103/heldout_blank_metrics.csv),
[injection metrics](../research/ftir_hips_chem/output/tables/baseline_existing_data_20261103/injection_recovery.csv),
and [summaries](../research/ftir_hips_chem/output/tables/baseline_existing_data_20261103/injection_summary.csv).

### Convergence is different from stability

The 50-iteration neutral method misses tolerance 10⁻³ on **51/319 native scans**
(44 PM2.5, 6 blanks, 1 PM10) and **39/259 blank-parent/injection cases**.
Increasing the ceiling to 500 resolves every one of these convergence failures.
It does not resolve the signal-preservation failures.

For example, the weak narrow addition on field blank **ETAD-0231, lot 251**, has
**42.30-fold** apparent area recovery even after convergence. ETAD-0239 gives
21.54-fold recovery at the same amplitude. A median near 100% hides these tails.
The broad-signal results also remain poor after convergence. `finite_output`,
`fit_converged` and `success` are separate fields; success requires both finite
output and convergence. Finite failed estimates remain inspectable.

## Transfer between known lots

Unknown-lot blanks are omitted from this check. The bank never contains a blank
from the test lot.

| Training → test population | Native spline median RMS | Local rank 3 | Local rank 8 |
| --- | ---: | ---: | ---: |
| Lot 251, 23 blanks → lot 248, 5 blanks | 0.000661 AU | 0.012086 AU | 0.001276 AU |
| Lot 248, 5 blanks → lot 251, 23 blanks | 0.000545 AU | 0.003913 AU | Unavailable: five training blanks cannot support rank 8. |

The local bank's strong mixed-lot held-out result does not transfer reliably to
an absent lot. The small lot-248 population also limits precision. These are
ETAD controls; they do not establish performance at Bishoftu or other sites.
[Bank membership and held-out identities](../research/ftir_hips_chem/output/tables/baseline_existing_data_20261103/background_banks.csv)

## Ambient shape check: 10 existing physical filters

All blank parents choose the lowest search channel, 1520.94 cm⁻¹. To avoid
generalizing from that special shape, a separate input-only stress check selected
up to five raw-RMS order statistics in each physical-mean boundary stratum.
The 253 PM2.5 filters contain 140 low-edge and 113 middle-boundary cases, with
none above 1560. The realized selection therefore has **10 parents**, not 15.

Selected IDs: ETAD-0034, ETAD-0037, ETAD-0081, ETAD-0132, ETAD-0182, ETAD-0200,
ETAD-0206, ETAD-0221, ETAD-0278 and ETAD-0306. ETAD-0037 remains a documented
suspect acquisition, deliberately visible as the low raw-signal extreme.

At the fixed 0.01-AU addition, the native spline recovers **99.826–99.961% of the
narrow Gaussian area across all 10 parents**. This check does not support a claim
that the current spline generally destroys this specified narrow feature.
It does not test every possible feature width or shoulder shape.

Broad recovery remains poor: native-spline medians are 12.43% and 12.52% in the
two strata; local rank-8 medians are 10.20% and 2.08%; local rank-3 medians are
negative. The small selected set is a stress test, not a prevalence estimate.
The 37-blank and 10-ambient schemes must remain separate in charts and counts.
[Selection](../research/ftir_hips_chem/output/tables/baseline_existing_data_20261103/ambient_injection_selection.csv)
and [ambient recovery](../research/ftir_hips_chem/output/tables/baseline_existing_data_20261103/ambient_injection_recovery.csv).

## Repeat scans and display support

Twenty PM2.5 physical filters supply 22 repeat pairs. Results are summarized
within physical filter before taking the across-filter median.

| Representation | Median pair RMS, AU | Median relative pair RMS |
| --- | ---: | ---: |
| Raw | 0.003593 | 1.82% |
| Native spline | 0.000303 | 3.44% |
| Boundary 1500 | 0.000307 | 3.42% |
| Boundary 1560 | 0.000310 | 3.58% |
| Neutral, 500 iterations | 0.000451 | 4.95% |
| Local VIBES, rank 3 | 0.000595 | 2.68% |
| Local VIBES, rank 8 | 0.000381 | 4.25% |

Relative RMS divides by each representation's own pair-mean spectral RMS.
Background removal changes that denominator, so increased relative RMS alone
does not establish worse instrument repeatability. Show absolute difference and
mean strength together. Separate field-blank repeats, whose corrected denominator
can be near zero. [Repeat metrics](../research/ftir_hips_chem/output/tables/baseline_existing_data_20261103/replicate_filter_metrics.csv)

**Forced-zero spline tails must not appear as measured absence of signal.** The
native algorithm sets baseline equal to raw absorbance below its selected
segment-two boundary plus 20 descending channels. Its correction is therefore
exactly zero there. The last fitted channel is recorded per scan and method as
`extension_last_wavenumber`. A display-support mask is:

```python
support = shared_mask & (wavenumber >= extension_last_wavenumber - 1e-8)
```

For a physical-filter mean, display the intersection of its contributing scans'
supports, while retaining the original stored averages. Blanks end near
1495.23 cm⁻¹ under the default spline. Primary shared-window RMS and negative
fractions describe the defined pipeline output, including these forced zeros;
they are not unrestricted measurements of residual signal throughout that window.
Injection scoring deliberately counts known additions lost in a forced-zero span
as loss. Native raw data are not zero-masked by this rule.

Held-out blank/injection controls correct a mean raw physical blank. Ambient
physical-filter traces average corrections fitted to individual scans. These
operations need not commute. All signed values and failed-fit flags are retained.

## Reproduction and verification

Configure `AETHMODULAR_ETAD_NATIVE_DIR` for the existing four-file input folder
through the local environment; do not commit its value. No private locations or
remote identifiers are needed in source code.

```bash
uv sync --python 3.13 --extra explorer --extra vibes
uv run --extra explorer --extra vibes python research/ftir_hips_chem/workflows/run_existing_data_baseline_controls.py --native-dir "$AETHMODULAR_ETAD_NATIVE_DIR" --workers 4
uv run --extra explorer --extra vibes python research/ftir_hips_chem/workflows/run_ambient_baseline_injections.py --workers 4
uv run --extra explorer --extra vibes python tests/test_native_baseline_controls.py -v
uv run ruff check research/ftir_hips_chem/scripts/native_baseline_controls.py research/ftir_hips_chem/workflows/run_existing_data_baseline_controls.py research/ftir_hips_chem/workflows/run_ambient_baseline_injections.py tests/test_native_baseline_controls.py
```

Outputs are under `research/ftir_hips_chem/output/tables/baseline_existing_data_20261103/`.
The main receipt hashes source inputs, numerical code and generated artifacts;
the ambient supplement hashes and references that receipt. Signed scan and
physical-filter arrays are in `native_method_arrays.npz`; chart summaries and
identity-level ledgers are CSVs.

The recorded execution used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
pybaselines 1.2.1, CVXPY 1.9.3 and Clarabel 0.11.1; the receipt records remaining
versions. The repository recommends Python 3.13; compare numerical tolerances
and receipts when reproducing in another environment. Eight targeted unit tests
pass, covering native-grid fidelity, segment support, held-out parent identity,
paired signed recovery, neutral-method equivalence and finite nonconvergence.
Targeted Ruff checks pass. No canonical product or exclusion was changed.
