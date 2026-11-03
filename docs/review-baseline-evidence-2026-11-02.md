# Baseline and spectral evidence audit — 2 November 2026

Reviewed source: `bdb47d78fff6dd6b2468d806bb5741ed46d8bc33`. This audit distinguishes
saved historical results from calculations repeated in this review. It accompanies
the portable screening module; it is not an EC calibration release. Original ETAD
spectra and R-corrected spectra were located separately during the wider review;
the portable calculations below deliberately use only the committed gallery export.

## What deserves priority

The useful question is whether a sample's apparent spectral difference survives
changes in baseline assumptions, within comparable loading and measurement
conditions. A baseline that reduces disagreement with HIPS is not necessarily a
better chemical separation. SPARTAN ChemSpec EC repeats the FTIR product, so it
cannot decide which method is correct. Source-domain IMPROVE TOR prediction and
target-domain optical agreement answer different questions.

Several promising explanations have already been tested and revised. The next
work should preserve these revisions rather than restarting the original claims.

| Question | Existing evidence | Scope and next use |
| --- | --- | --- |
| Is the 1617 cm⁻¹ feature specific to the Addis optical offset? | The neutral-baseline comparison finds an interior feature at Addis and Bishoftu. Bishoftu did not share the same offset in that comparison. | A candidate regional feature, not an established carrier of the optical disagreement. Compare matched-load Addis and Bishoftu filters, with the same baseline and provenance. |
| Can the spline baseline change the feature? | The AIRSpec segment-two algorithm searches for a minimum in 1520–1600 cm⁻¹ and uses it as the low-frequency boundary of the excluded analyte interval. | The assumed background can overlap the feature's shoulder. Show the fitted baseline and its selected boundary per filter. Do not claim that AIRSpec can never retain a 1617 band; attenuation depends on shape and boundary selection. |
| Is the feature charcoal? | The saved adjusted association with the archived charcoal PMF fraction is partial r = 0.0027, q = 0.9754, n = 102. | This comparison does not support that attribution. It also cannot rule out all combustion influences; the factor label is itself an interpretation. |
| Does a smaller blank residual establish better EC? | The September paired AIRSpec/VIBES experiment improved held-out blank removal without an established pooled EC advantage. | Blank removal, narrow-band recovery, broad-signal preservation and EC prediction require separate checks. The September scores must not be relabelled as the later rank-70 full-range result. |
| Does a lot difference explain Addis? | The committed derivative envelope-escape export shows a lot difference pooled across seasons, but not within the dry months. This arithmetic was reproduced below. | Lot, season, loading and analysis period are confounded. Retain their cross-tabulation and compare supported within-group cases. |
| Would additional PMF factors fix the residuals? | Signed negative input cells impose a substantial nonnegative-model error floor at both ETAD and ETBI. | Investigate baseline residuals and the uncertainty model before interpreting factors as sources. Do not erase the evidence by clipping negative inputs. |

Sources: [current claim limits](current-research-summary.md),
[1617 history and its two corrections](../research/ftir_ec_phase3/BAND1617_LEAD_2026-08-23.md),
[current exported band tests](../gallery/app/public/data/research/spectral/band_1617.json),
[September paired audit](vibes-subgroup-audit-2026-09-21.md),
[October rank-70 migration](vibes-gallery-rank70-migration.md), and
[PMF saved cases](../gallery/app/public/data/research/python_pmf/python_pmf.json).

## Baseline implementation details that change the interpretation

The [AIRSpec port](../research/ftir_ec_phase3/scripts/airspec_baseline.py) implements
segments 4000–1820 and 2000–1425 cm⁻¹. Segment two finds a per-spectrum minimum
inside 1520–1600, gives the interval between that boundary and 1820 zero fitting
weight, and extends the fit by 20 native channels. It then sets the baseline
equal to the observed spectrum beyond that extension, producing zeros there.
The 20-channel extension is resolution-dependent. Running this code unchanged on
the packed 7.7 cm⁻¹ grid would not faithfully reproduce its native-grid behavior.

The [neutral baseline](../research/ftir_ec_phase3/scripts/neutral_baseline.py) instead
uses penalized-spline arPLS without that band anchor. It retains ≥700 cm⁻¹ and
excludes the 1100–1300 PTFE region. “Neutral” describes the absence of this
particular anchor; it does not establish a physically correct background.

The [VIBES method comparison](pyvibes-vs-airspec.md) explains its different premise:
a mean and principal components learned from field blanks, then a per-spectrum
fit separating blank-like background and mostly nonnegative analyte. Its
applicability depends on blank coverage and preservation of true analyte signal.
Field blanks are not guaranteed to be chemically empty.

The September source summary is not the newest preprocessing version. The
[October migration](vibes-gallery-rank70-migration.md) records full-range rank-70
correction, shared-window and full-window refits, and migration of dependent
predictions. The [October 26 lot-241 workflow](../research/ftir_hips_chem/workflows/vibes_lot241_20261026.py)
also records a blank-bank extension and paired fixed-rank controls. Its code
reports that one blank strongly altered automatic rank selection. That is a
reason to retain leave-one-physical-blank checks and rank sensitivity; the code's
description alone is not a fresh reproduction of its results.

## What is actually in the portable spectra export

The [exporter](../gallery/data/export_research_lot_spectra.py) and
[descriptor](../gallery/app/public/data/research/lot_spectra/lot_spectra.json)
define the measurements and transformations. Their counts were independently
recomputed from the descriptor.

| Site | PM2.5 physical filters | PM10 physical filters |
| --- | ---: | ---: |
| Addis (ETAD) | 247 | 6 |
| Bishoftu (ETBI) | 40 | 0 |
| Beijing (CHTS) | 191 | 9 |
| Delhi (INDH) | 214 | 0 |
| JPL/Pasadena (USPA) | 158 | 0 |
| Total | 850 | 15 |

These 865 rows are not the 4,452-analysis native network export. That separate
[inventory](spartan-network-data-pulls.md#october-19-ftir-export-receipt) records
4,270 distinct filters at 28 sites: 3,517 PM2.5, 219 PM10 and 534 field blanks.
Repeated analyses account for part of the larger analysis count. Native database
coverage and the public carbon-product cohort are not identical populations.

The packed raw traces have 454 bins, each the mean of six native channels
(approximately 7.7 cm⁻¹), and are quantized to int16 with a per-filter scale.
Their descriptor reports maximum absolute quantization error 2.23 × 10⁻⁵
absorbance. Derivative traces retain 300 points on the source 2 cm⁻¹ grid in
1450–1850 and 2800–3000. The packed spline-baseline view is clipped to zero and
normalized to positive area before binning. It is useful for shape displays,
but cannot reveal signed negative residuals or be subtracted from raw absorbance
to estimate the original baseline. No field-blank traces are included here.

All three binary files decoded with the declared lengths and finite values.
The new loader validates grid monotonicity, unique physical IDs, column lengths,
per-filter scales and byte counts and records input SHA256 hashes.

## Independently repeated calculations

### Derivative envelope-escape score

The score is the fraction of 300 derivative channels outside the committed
5th–95th percentile envelope from 916 IMPROVE spectra. Recomputing it from the
packed derivative file agrees with the saved scores except at three of 865
filters, each differing by one channel out of 300; rounding/quantization near
an envelope boundary accounts for that resolution limit.

| PM2.5 site population in this export | Median escape fraction |
| --- | ---: |
| Addis | 0.2667 |
| Bishoftu | 0.3017 |
| Beijing | 0.3167 |
| Delhi | 0.2267 |
| JPL/Pasadena | 0.4200 |

Thus this particular measure does not rank Addis as the most spectrally unusual
site. A discrepancy strongest at Addis need not coincide with the largest global
distance from IMPROVE. The envelope uses a limited spectral region and a 916-row
reference collection; it is not a direct assessment of the entire deployed EC
calibration's validity.

For the original 239-filter Addis `char12` cohort, the lot comparison reproduced
the saved values exactly at the reported precision:

| Comparison | n, lot 248 / lot 251 | Median escape, 248 / 251 | Mann–Whitney p |
| --- | ---: | ---: | ---: |
| All months | 34 / 156 | 0.2900 / 0.2567 | 0.01106 |
| Dry months only, `dry_feb` calendar | 27 / 46 | 0.2900 / 0.2883 | 0.94531 |

The 49 remaining rows in that cohort lack a lot in this exported ledger. The
within-dry comparison does not prove lots equivalent; it shows that the pooled
association should not be interpreted as an isolated lot effect.

### PMF signed-input lower bounds

For a nonnegative prediction and a negative observed cell, the smallest possible
squared scaled residual is `(observed / uncertainty)^2`. Summing these cells
therefore gives a lower bound on the total ordinary Q irrespective of factor
count. From the saved case totals, this audit independently recomputed
`sqrt(negative_q_lower_bound / cells)` and the fraction of saved ordinary Q.

| Case | Negative cells, saved fraction | Recomputed scaled-RMSE floor | Saved fit scaled RMSE | Floor share of ordinary Q |
| --- | ---: | ---: | ---: | ---: |
| ETAD 246, robust fit | 16.019% | 3.97023 | 6.35113 | 39.08% |
| ETAD 102, robust fit | 16.774% | 3.84793 | 5.26610 | 53.39% |
| ETBI 53, robust fit | 16.325% | 3.13876 | 4.91483 | 40.78% |

These are arithmetic verifications of saved aggregate results; the original PMF
matrix was not used for this calculation. Ordinary Q is separate from the robust
optimization objective. Negative observations can arise from legitimate signed
measurement noise as well as baseline error. This result identifies a modeling
constraint; it does not by itself identify the source of the negative values.

## New portable sensitivity module and controls

[portable_baseline_screen.py](../research/ftir_hips_chem/scripts/portable_baseline_screen.py)
provides:

- A validated packed-data reader with source hashes and explicit transformation
  limits.
- Signed local spectral contrasts above the line joining two flank means.
  The default 1580–1650 feature uses 1500–1540 and 1760–1800 flanks; alternative
  flanks should be reported as sensitivity comparisons.
- An exploratory asymmetric penalized smoother with two stiffness settings,
  optional protected bands, and preserved signed residuals. PTFE 1100–1300 and
  channels at/below 700 cm⁻¹ are excluded from fitting and returned as NaN.
- Known-input narrow- and broad-signal recovery controls. Each constructed
  parent has spiked and unspiked versions, and recovery is their corrected
  difference. These are not independent blanks or ambient EC standards.

The smoothness penalty uses actual wavenumber spacing in median-channel units.
Lambda 10⁴ and 10⁶ are deliberately separated sensitivity settings on this grid,
not validated physical constants. They must not be transferred to native
resolution unchanged. No HIPS measurement enters fitting or parameter choice.

On the committed packed grid, both unprotected settings converged for all 865
spectra. Approximately 12.1% and 12.8% of reported corrected cells remain
negative. Those fractions describe this experimental calculation and cannot be
compared directly to the historical PMF fractions as a method improvement:
the cohorts, spectral supports and preprocessing differ.

The constructed 1617 cm⁻¹ Gaussian (σ = 25 cm⁻¹) recovered approximately
94–98% of its added area without protection. Protecting 1500–1760 recovered
approximately 100%, as expected when that feature cannot influence the fit.
The broad Gaussian (center 2800, σ = 600 cm⁻¹) exposed a failure: the less stiff
setting retained only about 2–3% of added area, while the stiffer setting ranged
from a negative paired response to about 24% recovery over the tested amplitudes.
All these fits converged. Numerical convergence and narrow-peak recovery therefore
do not certify preservation of a broad carbon-related continuum. These smoothers
are diagnostic perturbations, not proposed production replacements.

Seven focused numerical/provenance tests pass with:

```sh
python3 tests/test_portable_baseline_screen.py -v
```

They cover affine-background invariance of signed local contrasts, negative-value
retention, grid reversal, masked-channel contamination, protected-band influence,
input/byte validation and the deliberate broad-signal failure control. The runtime
passed `python3 -m aethmodular_cli doctor` with NumPy, pandas, Matplotlib and SciPy;
Python 3.12 was used in this managed environment rather than the recommended 3.13.

## Concrete next discrimination experiment

Select Addis and Bishoftu filters with overlapping loading, season, lot and
measurement-period coverage, keeping the conflicting cases visible. Include
Addis filters with large and small optical disagreement within each group,
and Bishoftu cases with a strong 1617 feature. Contrast raw spectra, native
signed spline correction, blank-trained VIBES, derivative shape, and protected
local-band measurements. Record per-filter baseline amplitude, slope, curvature,
selected anchor, negative residuals, and removed signal in each spectral region.

Then test whether baseline sensitivity and particular bands explain optical
disagreement beyond metadata and loading in held-out time/lot groups. Fit all
transforms, scalers and thresholds using training groups only. Keep repeated
scans of a physical filter together. Similarity to Ethiopia, the HIPS residual,
and independent EC error are three different targets and should remain labelled.

For any proposed baseline change, require held-out physical blanks, weak and
strong narrow and broad additions, and a paired IMPROVE TOR benchmark under
refitted matching preprocessing. A validated Ethiopian EC correction ultimately
requires identity-confirmed independent EC measurements across the local range.
The current spectral evidence can prioritize those filters and make the
experiment far more informative; it cannot supply the missing measurement.

## Interpretation safeguards already learned

The [charcoal comparison archive](../research/ftir_etad_char/README.md) found
fuel-library rankings and apparent temperature dependent on representation.
An artificially symmetric truncated-baseline comparison could give a very high
Addis–charcoal correlation while degrading cross-laboratory recovery. Retain
different representations as robustness checks, and do not optimize processing
to maximize the preferred library match.

The [blank-line adjudication](../research/ftir_ec_phase3/OFFSET_ADJUDICATION_2026-08-23.md)
also explicitly retracts pooled-lot HIPS corrections. Blanks must be keyed to
the deployed calibration line as well as lot. The remaining small blank-line
effect does not exclude every high-loading optical artifact, and the older
document's stronger attribution to real aerosol is not established by that
test. The later current-research summary correctly preserves the competing
measurement and aerosol explanations.
