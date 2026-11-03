# Site and physical-filter evidence review — 2 November 2026

Reviewed repository input commit: `bdb47d78fff6dd6b2468d806bb5741ed46d8bc33`.
Counts and descriptive statistics below were independently recomputed from the
committed exports. They are an optical-disagreement investigation, not an
assessment of accuracy against independent thermal EC.

## Available data are broader than the original four-site table

The portable source for a network-wide screen is
[`spartan_reference.json`](../gallery/app/public/data/spartan_reference.json):
**5,069 physical filters at 38 sites**, with 57 measurement fields and nine
published exclusions. Its EC and ChemSpec EC columns are two labels for the
same FTIR-derived product. It distinguishes direct in-house HIPS Fabs, public
BC measured with HIPS, public BC estimated from an HIPS–SSR curve, and SSR
measurements. Those distinctions must survive a join or a plot.

The reusable [`portable_cohort_screen.py`](../research/ftir_hips_chem/scripts/portable_cohort_screen.py)
preserves each original field, retains all physical rows and exclusions, and
provides a pair ledger. Direct Fabs is the primary optical comparator. Public
HIPS BC extends coverage only where direct Fabs is missing. It never substitutes
SSR or counts ChemSpec EC as an independent observation. Known negative values
remain visible; exclusion and missingness are separate statuses.

The full ledger has **3,373 eligible optical pairs: 2,500 direct-Fabs pairs and
873 public-HIPS extensions**. Nine rows are flagged: seven Delhi cartridge
filters and two historical Pasadena exclusions. The module uses the exported
central-registry flags and `get_clean_data`. It does not apply placeholder
date-only aethalometer exclusions to additional filter-only rows.

| Site | Physical reference rows | Excluded | Direct Fabs + FTIR EC pairs | Public-HIPS-only extension pairs | Combined pairs |
|---|---:|---:|---:|---:|---:|
| Addis, ETAD | 247 | 0 | 232 | 14 | 246 |
| Bishoftu, ETBI | 60 | 0 | 21 | 34 | 55 |
| Delhi, INDH | 213 | 7 | 109 | 25 | 134 |
| Beijing, CHTS | 403 | 0 | 171 | 24 | 195 |
| Pasadena, USPA | 293 | 2 | 156 | 59 | 215 |

These counts differ from the older
[`filters.json`](../gallery/app/public/data/filters.json), which has 942 rows,
including 190 Addis rows and no Bishoftu rows. A notebook or chart using only
that older table cannot answer the current within-Ethiopia question completely.
Different exported tables also represent different measurement vintages.

## Optical discrepancies in the direct-Fabs population

Here optical BC means `HIPS Fabs / config.MAC_VALUE`, currently 10 m²/g.
The signed quantity is **FTIR EC minus optical BC**, in µg/m³. R² is squared
Pearson correlation. It is not predictive R² or a test of agreement with true EC.

| Site | Direct pairs | Median signed difference | Median absolute difference | Squared Pearson correlation |
|---|---:|---:|---:|---:|
| Addis | 232 | −0.329 | 1.033 | 0.753 |
| Bishoftu | 21 | −1.266 | 1.266 | 0.836 |
| Delhi | 109 | +0.370 | 0.935 | 0.913 |
| Beijing | 171 | −0.043 | 0.257 | 0.500 |
| Pasadena | 156 | +0.002 | 0.113 | 0.513 |

The current Bishoftu public FTIR product is lower than direct HIPS/10 on all
21 available pairs. Across its 55 combined pairs the median difference is
−1.05 µg/m³. Therefore the historical statement “Bishoftu has no offset” must
remain attached to the particular refitted model and cohort that produced it;
it is not a general description of the current reported FTIR product.

With x = optical BC and y = FTIR EC, the direct-pair Addis Deming slope ranges
from 2.464 to 2.193 and intercept from −6.975 to −5.670 as lambda changes from
0.25 to 4. At Bishoftu the corresponding slope is 0.883 to 0.793 and intercept
−0.855 to −0.617. These are **variance-ratio sensitivity results**, not a
determination of the correct lambda. The export provides HIPS uncertainties on
every direct pair at these five sites, but neither their precise variance
interpretation nor target-domain FTIR error is established here.

There is no universal sense in which Ethiopia is “the worst.” The answer
depends on the model, period, loading and discrepancy metric. The wider public
network also includes large absolute FTIR–optical differences at Jaipur and
Dhaka. Those sites are useful future comparison candidates, but their complete
raw spectra are not supplied by the packed five-site export reviewed here.

## Spectra and metadata that can actually be joined

The packed spectra and descriptor are in
[`research/lot_spectra/`](../gallery/app/public/data/research/lot_spectra/).
The exporter documents their numerical meaning in
[`export_research_lot_spectra.py`](../gallery/data/export_research_lot_spectra.py).
There are **865 distinct physical-filter IDs: 850 PM2.5 and 15 PM10**.

| Site | PM2.5 spectra | PM10 spectra | PM2.5 sampling period |
|---|---:|---:|---|
| Addis | 247 | 6 | 2022-12-07 to 2026-03-01 |
| Bishoftu | 40 | 0 | 2025-10-20 to 2026-02-19 |
| Delhi | 214 | 0 | 2022-06-28 to 2026-04-26 |
| Beijing | 191 | 9 | 2022-07-05 to 2024-12-08 |
| Pasadena | 158 | 0 | 2022-07-22 to 2023-11-14 |

The current packed-to-public join has no duplicate IDs, no date disagreements,
and no site-code disagreements among matched rows. Bishoftu has 39 matched
public identities and 34 optical pairs within its 40-spectrum population;
the reference's 55 pairs include later filters outside those spectra. All
247 Addis PM2.5 spectra match the public ledger, with 246 optical pairs.

Two spectra carry explicit acquisition QC code 0: `CHTS-0814` and `INDH-0265`.
Neither currently has public EC or either HIPS comparator, so excluding them
from paired calculations does not change the present pair count. They should
remain identifiable in an acquisition-review queue. Missing acquisition QC
metadata, notably all Addis rows, must not be interpreted as a demonstrated QC
pass.

### The three spectral arrays are not interchangeable

- `raw.bin` contains absorbance binned to roughly 7.7 cm⁻¹, 454 points/filter.
- `deriv.bin` contains the stored second derivative on a 2 cm⁻¹ grid in the
  1450–1850 and 2800–3000 cm⁻¹ regions, 300 points/filter.
- `airspec.bin` contains **clipped nonnegative, positive-area-normalized**
  spline-baselined spectra, binned to roughly 3.9 cm⁻¹, 668 points/filter.

Subtracting the packed AIRSpec array from packed raw absorbance does not yield
the removed baseline: grids and units differ, and negative residuals and
absolute signal scale were discarded. The packed AIRSpec display is also
unsuitable for a signed-negative-residual audit or an absolute EC recalibration.

The binary format is a float32 scale for each filter followed by an int16
filter-by-channel matrix. It is not an ordinary float32 matrix.

### Lot and season remain partly confounded

The packed Addis metadata identify 34 lot-248 and 156 lot-251 filters through
September 2024. **All 63 later Addis spectra lack lot metadata**, including five
PM10 filters. Bishoftu's 40 spectra all have lot 251. Consequently, the committed
packed metadata alone cannot establish a contemporaneous, same-lot
Addis–Bishoftu comparison. Inferring the missing Addis lot from its sampling
date would replace evidence with an assumption.

At Delhi, lot 251 covers March 2023–September 2025 and lot 253 begins on
20 September 2025. Period, composition/loading and lot changes overlap. A
between-lot difference there is not an isolated substrate effect. Beijing and
Pasadena have interleaved lots 248 and 251, making them better substrate
comparison candidates after matching loading and analysis periods.

## Useful Ethiopian filters to investigate together

These are exploratory contrasts, not proposed exclusions or declared bad
measurements. Public FTIR EC, direct HIPS, sampling date and packed spectral
metadata were joined by exact physical-filter ID.

| Filter | Sampling date | Lot | FTIR EC, µg/m³ | HIPS Fabs, Mm⁻¹ | PM2.5, µg/m³ | Spectral envelope escape |
|---|---|---:|---:|---:|---:|---:|
| ETAD-0037 | 2023-01-09 | 248 | 0.69 | 38.403 | 14.17 | 0.540 |
| ETAD-0040 | 2023-01-15 | 248 | 2.42 | 38.536 | 15.48 | 0.300 |
| ETAD-0036 | 2023-01-06 | 248 | 2.73 | 39.320 | 15.33 | 0.267 |
| ETAD-0221 | 2024-06-18 | 251 | 11.74 | 68.914 | 53.78 | 0.250 |
| ETAD-0202 | 2024-04-27 | 251 | 6.85 | 68.280 | 35.81 | 0.217 |
| ETBI-0024 | 2025-12-09 | 251 | 1.98 | 34.564 | 23.07 | 0.300 |
| ETAD-0242 | 2024-08-16 | 251 | 3.43 | 34.681 | 13.11 | 0.293 |

The first three filters provide a particularly useful comparison: same site,
same lot, nearby dates and very similar optical loading and PM2.5, but very
different FTIR EC and spectral departure. ETAD-0037 should remain in the
evaluation; removing it would erase evidence about the problem. ETAD-0221 and
ETAD-0202 have similar HIPS but different PM2.5, dates and FTIR EC, offering a
high-loading composition/season contrast. The Addis–Bishoftu pair has similar
HIPS and the same recorded lot but different years/seasons and PM2.5, so it is
an investigative contrast rather than a controlled site experiment.

The stored spectral-envelope escape score is **not an EC-failure probability**.
After separating PM10, published exclusions and explicit QC-zero spectra,
median escape is approximately 0.420 at Pasadena, 0.317 at Beijing, 0.302 at
Bishoftu, 0.267 at Addis and 0.220 at Delhi. Pasadena has the largest median
departure but much smaller optical disagreement. A threshold based on that
score alone would discard a large fraction of useful low-loading controls.

## HIPS baseline evidence must retain the deployed line

[`research/hips_optics/filters.json`](../gallery/app/public/data/research/hips_optics/filters.json)
provides 734 optical records with raw transmittance, reflectance, attenuation,
analysis date, lot and the actual calibration intercept/slope. There are **86
field blanks**: 28 Addis, 24 Beijing, 15 Delhi and 19 Pasadena. There is no
Bishoftu optical record in this particular export.

The 28 Addis field blanks belong to four deployed calibration lines:

| Lot | Intercept | Slope | Field blanks | Optical analysis dates |
|---|---:|---:|---:|---|
| 248 | 1383.2 | −2.544 | 4 | 2023-05-03 to 2023-07-20 |
| 248 | 2262.0 | −5.320 | 1 | 2023-03-17 |
| 251 | 1397.0 | −2.687 | 11 | 2024-08-08 to 2025-03-13 |
| 251 | 1416.2 | −2.783 | 12 | 2023-08-22 to 2024-04-23 |

Do not pool these by lot and fit a single correction across different hardware
calibrations. These field blanks also carry FTIR masses and are not guaranteed
to be chemically empty laboratory substrates. Their analysis date is different
from an ambient filter's sampling date and should remain a separate variable.

The optical export has a physical-type inconsistency: `ETAD-0248`,
`CHTS-0873` and `CHTS-0889` are labelled PM2.5 there but PM10 in the spectral
metadata. Do not let that optical field overwrite explicit physical-filter
type or the public PM2.5 membership during a join.

## Independent workflow review

The portable baseline review was checked for identity handling and target
leakage. It joins by physical ID, audits date and site equality, and fits scaling
and regularization inside each training fold. Outer prediction uses earlier
quarters only. Per-filter baseline fits do not pool held-out target values.

The dummy-category vocabulary is constructed from the whole descriptive frame,
but a category absent from training has a zero fitted coefficient; neither its
outcome nor its numeric magnitude enters fitting. This has no outcome-leakage
effect for the current linear pipeline, although explicitly frozen category
levels or a training-fitted encoder would make the chronological contract more
literal.

Predicting `FTIR EC − optical BC` while controlling optical BC describes
conditional FTIR variation. It does not establish that the residual is an EC
error or that a correlated baseline descriptor caused it. Missing acquisition
batch, spectral calibration identity, instrument history and missing lot data
remain important competing explanations. Forward prediction results remain
exploratory because historical observations motivated the feature set.

Five focused unittest checks pass for the cohort module: source preference and
no SSR substitution; no duplicate-product counting; preservation of exclusions
and negative values; exact linear Deming behavior; and rejection of duplicate
physical identities or unreasoned exclusions. The review did not obtain new
laboratory data or refit the upstream FTIR calibration.

### Final chronological and identity checks

Four additional tests in
[`test_portable_baseline_review.py`](../tests/test_portable_baseline_review.py)
pass using a five-quarter synthetic fixture and the actual nested Ridge
pipeline, with no mocked model fitting:

1. Each held-out physical filter receives one prediction and all its training
   records precede its collection quarter.
2. Adding 1,000 µg/m³ to every final-quarter target changes neither fitted
   predictions nor chosen penalties in any quarter.
3. Changing final-quarter covariates by orders of magnitude and adding unseen
   lot/source categories changes no earlier-quarter prediction or penalty.
4. Duplicate physical-filter IDs are rejected before chronological fitting.

These tests establish the implemented fold contract, not validity of the
scientific target. The outcome contains an existing FTIR-derived EC estimate;
spectral predictors can recover structure in that estimate. A lower forward
optical-disagreement prediction error is useful for localization and triage,
but does not independently demonstrate that its baseline is wrong or validate
an EC correction.

The direct-HIPS-only secondary screen has 167 held-out predictions versus 183
in the complete screen. Fourteen public-only records disappear, and the two
remaining direct pairs in 2026 Q1 do not meet the minimum of four test filters
per quarter. Thus those two populations are not a common-test comparison.
Within every reported population, the metadata, baseline, band and combined
models evaluate exactly the same physical filters. The known-lot analysis
also ends earlier and should not be presented as an independent replication.

The selected panel preserves useful counterexamples: ETAD-0037 has relatively
low generic-baseline sensitivity but high optical disagreement, while
ETAD-0202 has relatively high sensitivity but little disagreement. These are
relative within-site classifications, not validated defect categories. Lot,
season, optical source and missing batch metadata remain part of the planned
comparison, not properties that panel selection has controlled automatically.

### Native and portable Addis populations are different

The separate native run has 189 eligible original-product optical pairs; the
portable public-product run has 246, with **188 identities in common**.
`ETAD-0246` (2024-09-21) is native-only because its in-house EC is 9.279024
µg/m³ but the public EC field is missing. The portable run adds 58 later
filters beginning on 2024-10-09. The native spectra population itself includes
253 PM2.5 filters, while the packed export has 247; spectral availability and
product pairing are distinct denominators.

The native residual is **HIPS/10 minus FTIR EC**; the portable residual is
**FTIR EC minus HIPS/10**. Reverse the sign before comparing directions, and
do not assume that sign reversal makes the numerical products identical.
Among the 188 shared pairs, nine public EC values differ from the in-house
product by more than 0.01 µg/m³: `ETAD-0094`, `ETAD-0096` through `ETAD-0102`,
and `ETAD-0104`. The largest difference is −0.232958 µg/m³ at ETAD-0104
(public minus in-house); HIPS/10 agrees to approximately 5 × 10⁻⁸ µg/m³.
Most other product differences are consistent with two-decimal rounding.
The nine larger differences need release/volume provenance review; this check
does not establish their cause. Native and portable residuals must not be
pooled as interchangeable observations.
