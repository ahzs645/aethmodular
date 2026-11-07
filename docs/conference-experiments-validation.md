# Conference experiment validation — 7 November 2026

This record describes bounded exploratory runs of the
[conference experiment workflow](conference-experiments.md). The source
calibration, Pasadena humidity and tracked Addis filter/weather calculations
used actual supplied data. The end-to-end demonstration used artificial data and
is a separate software check. No correction or source-contribution product was
deployed from these results.

## What was run

| Question | Actual-data execution | Interpretation or remaining input |
|---|---|---|
| Humidity memory | Full supplied Pasadena minute file; seven wavelength/spot signals; later-day evaluation | No events met the declared RH-step criterion; small lag-model association, not a humidity correction. Jacros minute execution remains pending local access to its larger file. |
| Calibration coverage | 27 source-model fits with equal training counts and fixed held-out sites | High-loading restoration improved RF in this design; model and region effects were mixed. |
| Chemical regimes and episode evolution | Strict coverage audit and exploratory adjusted sulfate comparison | No demonstrated useful sulfate-dependent slope change. Episode evolution needs an independent event catalogue. |
| Predict independent chemistry | Input readiness evaluated | Blocked: verified active filter schedules and optical/observation histories are absent from the tracked exports. |
| Rain history | Daily features and forward-split support audit | Shared-event purging leaves too few independent training filters for the declared model. |
| Reference-sample design | 27 source-model fits comparing three selection strategies | Coverage selection helped RF but worsened ridge/PLS in this source split. Local specimen selection still needs candidate spectra and material review. |
| Additional measurements | Executable protocols and pair-eligibility checks; synthetic demonstration | No offline AMS, local thermal EC, PAS or SP2 data were created or acquired by this PR. |

## 1. Source calibration coverage

The native historical source archive contains 906 physical filters and 2,722
spectral columns. The existing documented TONT1 exclusion remains visible in the
ledger, leaving 905 eligible filters. A fixed seed of 20261107 selects 35 complete
sites for testing: **255 test filters and 650 training-pool filters**. Every
experimental fit uses **128 physical training filters**.

The prefit design has three regions (low EC loading, high EC loading and spectral
extremes), three retained-coverage fractions (0, 0.5 and 1), and three model
families. It completed all **27 fits**. The region boundaries are learned from
the training pool; predictive preprocessing and tuning use only each selected
training set and its inner site folds.

The fixed candidate settings are PCA8 ridge with alpha 1/10, PLS with 4/8/16
components, and RF with 100 trees and minimum leaf size 2/5. Selection uses three
inner site folds. This is a deliberately bounded coverage experiment, not a new
score for the existing calibration fitted on the full source population.

### Clearest contrast: remove and restore high-loading examples

| RF training arm | Held-out RMSE, µg/filter | Bias, µg/filter | Predicted/reference 90% span |
|---|---:|---:|---:|
| High-loading region fully depleted | 27.670 | −5.596 | 0.307 |
| High-loading coverage restored, same total training count | 22.793 | +0.823 | 1.492 |

The paired **depleted minus restored RMSE** difference is **+4.877 µg/filter**,
with a 500-draw held-out-site bootstrap interval **[+0.717, +8.239]**.

This supports the proposition that missing high-loading examples can contribute
to RF range compression in this source population. The restored prediction span
exceeds the reference span, so restoration did not simply produce a perfectly
calibrated range. Ridge and PLS high-loading comparisons also reduced aggregate
RMSE here, but their corresponding intervals cross zero; low-loading and spectral
region results are mixed.

The interval conditions on these fitted models, this training budget and this
fixed site split. It is not a local Addis confidence interval or a guarantee that
additional high-loading samples will solve transfer error. The 255 held-out
source-filter identities remain in the local run ledger, not in this PR.

## 2. Assay-selection simulation

The same training pool, 128-sample budget, source reference values and held-out
sites were used for random, spectral/loading-coverage, and 50/50 hybrid selection.
Reference values are hidden from the selection procedure and then revealed for
fitting and inner validation. Three seeds and three model families produced
another **27 completed fits**.

Mean held-out RMSE across the three seeds, in µg/filter:

| Selection strategy | PCA/ridge | PLS | RF |
|---|---:|---:|---:|
| Random representative | 20.586 | 18.979 | 21.509 |
| Spectral/loading coverage | 21.661 | 21.590 | 17.196 |
| Half representative, half challenge | 21.827 | 21.515 | 18.408 |

Coverage selection helped RF and worsened both linear model families in this
design. A sample-selection method should therefore be evaluated together with
the intended model and target population. These results do not support assuming
that space-filling acquisition always beats random acquisition.

The coverage-only selection is deterministic: repeating its seed does not create
new ridge or PLS panels. RF still varies with its estimator seed. These three
rows are descriptive averages over the declared seeds, not independent estimates
of population performance. Paired site-bootstrap tables remain available in the
local report, with the conditional uncertainty limitation retained.

The local reference-panel code separately records representative inclusion
probabilities, disjoint challenge selections, blanks and blind-repeat requests.
That planning capability was exercised with synthetic candidates. No actual
Addis filter was declared laboratory-ready merely from its spectral or filter ID.

## 3. Full-period Pasadena humidity diagnostic

The actual minute input contains **471,232 rows**, from
**2023-06-23 15:46:59 UTC through 2024-05-31 01:48:59 UTC**. All rows were supplied;
the manifest projected only the 15 necessary columns. It used the five BCc
channels plus IR spot 1 and spot 2, reconstructed onto the attenuation scale
using the declared existing sigma values. Day/hour features use
`America/Los_Angeles`, while input timestamps retain their explicit UTC basis.

There are **53 session/tape/gap segments**. Recorded RH ranges from **5.0% to
27.2%**, and the maximum valid ten-minute RH change is **3.5 percentage points**.
The prespecified criterion is **5 percentage points in ten minutes**. Consequently:

- No candidate event meets the criterion.
- There is no rising/falling matched-event estimate and no observed recovery
  curve under this protocol.
- The event stage reports insufficient data, rather than concluding that the
  MA350 has no humidity memory.
- The actual-filter-window stage is blocked by missing verified intervals and
  their parent filter catalogue.

The separate later-day predictive comparison remains estimable. It uses 70,210
training samples from 247 complete local days and 23,379 test samples from 83
later days, at the declared five-minute stride. Finite nonpositive excursions
are retained in amplitude diagnostics, including 46,540 such IR observations;
they are not censored merely because they may be instrument artifacts.

| IR attenuation model | Test RMSE, Mm⁻¹ | Prediction R² |
|---|---:|---:|
| Current state plus prior reference amplitude | 3.997247 | 0.051831 |
| Same baseline plus RH history at 5/15/30/60 minutes | 3.981927 | 0.059085 |

The RMSE gain is **0.015320 Mm⁻¹**, approximately **0.4%**, with a 300-draw test-day
bootstrap interval **[0.004408, 0.028197]** (unadjusted for multiple comparisons).
Both model baselines include total flow as well as the spot-flow ratio.
Explained variation remains low. This
is a small association conditional on the stated baseline; changing aerosols and
air masses could contribute. The spectral channels and two IR spots are related
measurements, not independent confirmation of the mechanism.

The final numbers above supersede development runs that excluded nonpositive
amplitude values or controlled only the spot-flow ratio. Retaining finite
excursions and including measured total flow corrected those input/control
contracts; event thresholds and model hyperparameters were not retuned to
improve the result. The sensor's physical location and response time, and the history behind
the input filename's “Cleaned” label, have not been independently verified.

The Jacros raw file was not executed: its 556,073,572-byte size exceeded the
available retrieval interface's 268,435,456-byte limit. No hourly export was
substituted for this minute-level question. The implemented workflow accepts a
locally supplied compatible minute file; the Pasadena result is not labelled as
an Addis result.

## 4. Adjusted Addis sulfate comparison

The tracked unified table yields 224 physical-filter records. Source review found
the independent sulfate method in the original ChemSpec file: methods **316 and
318**, with **Ion Chromatography** in the analysis description and concentration
units of µg/m³. Preparation verifies each retained sulfate value against the
method-bearing record for its physical filter. **174 filters** pass the declared
eligibility and value-provenance checks.

The experiment uses direct HIPS absorption and the disagreement
`D = FTIR EC − HIPS absorption / MAC`. It fits the existing EC-line benchmark
using training filters, then controls for PM₂.₅ concentration, PM₂.₅ areal loading,
sample-day Bole temperature/wind, filter lot and canonical `belg_feb` season.
Areal loading is derived from PM concentration × archived collection volume /
deposit area. Those metadata do not establish actual pump-active intervals.

The strict all-month analysis has **no eligible fold**. The explicitly
exploratory all-seasons comparison uses 89 training filters in the 2022–23 cycle
and 85 later test filters in the 2023–24 cycle. Training observations span
December 2022–September 2023; test observations span October 2023–July 2024.
The training cycle lacks October/November and the test cycle lacks
August/September. Both cover all three seasons, and the report marks the test
cycle incomplete.

| Model | Test RMSE of D, µg/m³ | MAE, µg/m³ |
|---|---:|---:|
| Training-fitted EC line | 0.480491 | 0.364035 |
| Adjusted baseline | 0.468729 | 0.362432 |
| Baseline plus sulfate main effect | 0.440064 | 0.333927 |
| Baseline plus sulfate main effect and EC×sulfate interaction | 0.438506 | 0.332320 |

The interaction's incremental RMSE difference versus the sulfate-main-effect
model is **−0.001558 µg/m³**, with a 300-draw paired physical-filter bootstrap
interval **[−0.006807, +0.002956]**. This does not demonstrate a useful change in
the EC-related slope. The main-effect result provides an exploratory reason to
investigate a sulfate association with disagreement.

That association cannot identify a coating mechanism, separate FTIR sensitivity
from optical behavior, or establish true EC accuracy. The bootstrap treats
physical filters as independent blocks and does not model longer temporal
dependence. Sample-day Bole weather is a coarse control rather than collection-
window or chamber meteorology. Earlier reduced-control development results are
not the adjusted estimates reported here.

## 5. Rain support and unresolved analyses

Rain and sulfate populations are prepared separately, so an unrelated sulfate
repeat conflict does not exclude a rain-only observation. The daily weather file
has 657 calendar bins, 652 with valid precipitation. The feature builder retains
missing bins and distinguishes strictly antecedent 1/3/7-day rain from sample-day
rain. Daily data cannot establish rain during actual collection.

There are **185 eligible rain-comparison filters**, initially 89 training and 96
test filters. Keeping connected rain episodes together removes 65 training
filters that share event components with the test period. **24 remain**, below
the declared minimum of 30. Neither the strict nor exploratory rain comparison
is fitted. The minimum count and event definition were not relaxed to obtain a
numerical answer.

The archived weather file lacks per-value observed/model provenance. Its metric
fields are interpreted according to [Meteostat's daily-data
documentation](https://dev.meteostat.net/api/stations/daily): °C, mm and km/h. It
must not be described as uniformly observed local rain.

The tracked recipe also emits explicit blocked entries for optical prediction of
independent chemistry and for episode evolution. Those questions need reviewed
active collection schedules/observed optical features and independently defined
pollution episodes, respectively. Future measurement readiness is represented by
protocol ledgers; no calibration, sampling equivalence, substrate compatibility
or laboratory result is fabricated to complete them.

## Reproduction and audit record

The preparation commands in the [workflow guide](conference-experiments.md)
recreate the source and tracked-filter recipes. The
[Pasadena example manifest](../research/ftir_hips_chem/workflows/conference_pasadena.example.json)
uses `${AETHMODULAR_MINUTE_CSV}` for the supplied local file and includes the full
declared settings. Set the variable to the file's location, then pass this
manifest to `run_conference_experiments.py --config` with a new output directory.

Selected input identities:

| Input | SHA-256 |
|---|---|
| Native source `apps_data.npz` | `8b2bd5b70d2f00bbbf82f410cce37bb4e1ba3be318ba175525d2e65fbfeadd53` |
| Source fixed test membership | `a707e8c1100cf0b7c831e3ff3362d7d893b946707a26cb00e30f0937b01194af` |
| Full supplied Pasadena minute CSV | `024dffbe5d9deb09e8bd58bb76add78aaac1a3a9fede454b1abc6b3c10b2832e` |
| Tracked unified filter CSV | `d960998d6921b9beef06cbcb6e982ddeeef9812f21fa6a5624f8ed5b324b5bdf` |
| Tracked ChemSpec method source | `a9f87a16ca07030aaeaf3b0d67c1c5fe3f0d209c34a3a51b365e6a8724f33020` |
| Tracked Bole daily weather | `41330ef7ce1766a52ddc1ad655ae6749bfc018df2ed437ca54e342ebfaa75334` |

Native-source preparation was checked against the completed run: spectra,
Unicode acquisition IDs, wavenumbers and both experiment parameter dictionaries
are identical. The tracked preparation and both wrapper CLIs were executed on
the reviewed files. Each actual run writes its complete input/code/output
receipts locally; only this aggregate account is added to Git.

All **130 focused scientific and related active-interval/seasonal regression
tests passed** locally. The synthetic CLI exercises all eleven operations and retains the
synthetic label throughout its report. Scoped Ruff and compilation checks pass.
The PR's dedicated GitHub workflow repeats those software checks without private
inputs. Raw minute records, spectra, filter-level outputs and conference photos
are not included in the proposed changes.
