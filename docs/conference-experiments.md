# Conference ideas as reproducible experiments

This workflow turns seven conference-inspired questions into reviewable analysis
code, data contracts and measurement plans. It extends the existing calibration,
seasonal validation and active-interval matching utilities. It does not install
an empirical correction or treat optical disagreement as independent EC error.

The implementation is under
[`conference_experiments/`](../research/ftir_hips_chem/scripts/conference_experiments/).
The [validation record](conference-experiments-validation.md) separates actual
source/field results, insufficient support and synthetic software checks.

## Questions and implementations

| Conference material | Scientific question | Operation and module | Decisive comparison |
|---|---|---|---|
| BSEC memory, `IMG_7464`; chamber transitions, `IMG_7492` | Does recent humidity history explain a response beyond current RH? | `humidity_memory`, `humidity.py` | Matched rising/falling events and later-day instantaneous versus lagged models |
| Sparse calibration coverage, `IMG_7536`; linear-tail comparison, `IMG_7538` | Does missing calibration coverage compress predicted EC loading? | `calibration_coverage`, `coverage.py` | Depleted versus restored regions at the same training count and identical held-out sites |
| Coatings/mixing state, `IMG_7549`, `IMG_7550`, `IMG_7412`; aging, `IMG_7355` | Does the optical–carbon relationship vary with independently measured composition or within independently defined events? | `regime_interactions`, `chemistry.py`; `episode_evolution`, `episodes.py` | Chemical main effect versus slope interaction; matched early/late spectral behavior |
| PMF-to-ML, `IMG_7380`, `IMG_7382` | Do optical features predict independent chemistry beyond routine conditions? | `chemistry_prediction`, `chemistry.py` | Baseline versus spectral shape versus within-window temporal features on later filters |
| Rain-event design, `IMG_7577`; transport pairing, `IMG_7344` | Does rain history explain differences within a season? | `rain_history`, `rain_evaluation`, `rain.py` | Antecedent rain and sample-day/verified active-collection rain, keeping connected events together |
| Calibration designs, `IMG_7492`, `IMG_7501` | Which reference samples are informative at a fixed assay budget? | `reference_panel`, `selection_simulation`, `selection.py` | Representative, coverage and hybrid strategies on the same source test sites |
| Offline AMS, `IMG_7339`; photoacoustics, `IMG_7477`; SP2, `IMG_7414` | Which independent measurements could separate the competing explanations? | `measurement_plan`, `colocation_eligibility`, `measurements.py` | Explicit material, method, timing, wavelength and calibration evidence |

Photo names identify the user's conference notes; the photographs and their
private links are not part of this code change. A presentation supplies a
hypothesis or design idea. Its result is not assumed to apply to this project.

## Run it

Use the locked project environment from the repository root:

```bash
uv sync --frozen
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/run_conference_experiments.py \
  --demo --output research/ftir_hips_chem/output/tables/conference/demo
```

The demonstration creates artificial spectra, reference values, minute signals,
filter chemistry, weather, event definitions and evidence strings. It exercises
all eleven operations implementing the seven questions. Its report is headed
**SYNTHETIC DEMONSTRATION**. None of its values are scientific observations.

Prepare the tracked Addis snapshot and inspect the generated manifest:

```bash
uv run python research/ftir_hips_chem/workflows/prepare_conference_inputs.py \
  --tracked --output research/ftir_hips_chem/output/tables/conference/tracked_inputs
uv run python research/ftir_hips_chem/workflows/run_conference_experiments.py \
  --config research/ftir_hips_chem/output/tables/conference/tracked_inputs/manifest.json \
  --output research/ftir_hips_chem/output/tables/conference/tracked_run
```

This preparation reviews method descriptions from the tracked ChemSpec export,
uses direct HIPS observations, preserves conflicting repeats and exclusion
flags, and keeps rain and sulfate populations separate. It records the strict
full-month validation requirement and explicitly labelled exploratory
all-season comparisons. Missing optical collection-window evidence stays
blocked. It does not manufacture sampling schedules from sample dates.
The tracked adjusted comparison includes PM concentration, PM areal loading
derived from archived volume/area, sample-day Bole temperature/wind, lot and
season. The generic APIs accept an explicit prespecified control-column list.

For the historical source calibration archive, supply its local root containing
`apps/` and `notebook/extracted/`:

```bash
uv run python research/ftir_hips_chem/workflows/prepare_conference_inputs.py \
  --source-dir /path/to/FTIR \
  --output research/ftir_hips_chem/output/tables/conference/source_inputs
uv run python research/ftir_hips_chem/workflows/run_conference_experiments.py \
  --config research/ftir_hips_chem/output/tables/conference/source_inputs/manifest.json \
  --output research/ftir_hips_chem/output/tables/conference/source_run
```

The source adapter uses the existing verified archive loader, retains the 906-row
ledger, and applies the documented single TONT1 scan exclusion for the 905-filter
experiment. Reconstructed historical fitted coefficients are discarded before
the new training/test split. Preparation and run receipts hash the original and
canonical inputs separately.

Each output directory must be new or empty. Reusing a populated directory fails
without overwriting its record. A run writes `report.md`, `receipt.json`, and one
directory per experiment containing `summary.json` and CSV result tables.
Receipts include explicit settings, input/output hashes, implementation hashes,
Git state and numerical-library versions. Generated data and results stay under
the existing ignored output tree.

Exit codes are `0` for completed, `1` for calculation failure, `2` for invalid
workflow input, and `3` for a run containing blocked or insufficient-data stages.
Completion means a calculation ran; it does not mean a hypothesis was confirmed.
Independent experiments continue if another input is unavailable or fit fails.

## Manifest and input contracts

A manifest declares `schema_version: 1`, `data_scope: "research"` or `"synthetic"`,
named datasets, and an ordered list of experiments. Each experiment declares its
`id`, `kind`, `inputs`, and `parameters`. Paths are relative to the manifest or
can use an explicit environment variable. An absent input can instead declare
`unavailable_reason`; the report retains that reason as a blocked result.

```json
{
  "schema_version": 1,
  "data_scope": "research",
  "datasets": {
    "filters": {"path": "filters.csv"},
    "weather": {"path": "weather.csv"}
  },
  "experiments": [
    {
      "id": "rain_features",
      "kind": "rain_history",
      "inputs": {"data": "filters", "weather": "weather"},
      "parameters": {"config": {"mode": "daily"}}
    },
    {
      "id": "rain_comparison",
      "kind": "rain_evaluation",
      "inputs": {"data": {"experiment": "rain_features", "table": "features"}},
      "parameters": {"test_coverage": "all_months"}
    }
  ]
}
```

Result dependencies can only refer to earlier experiments. Tables accept CSV,
Parquet or JSON records. An explicit `columns` list projects only needed columns
while retaining every row and hashing the complete input file. Spectra require an NPZ numeric `X` matrix, a non-object
Unicode `analysis_ids` array, and a metadata table with `analysis_id`. The reader
requires exact, unique row-ID agreement and explicitly aligns metadata to the
matrix. Pickled arrays and implicit positional joins are rejected. Distinct
analysis IDs may refer to the same physical filter; those scans are aggregated
once before any split, with contradictory site/reference metadata rejected.

### 1. Humidity history

Minute inputs require timezone-aware timestamps, sample-stream RH, session/tape
boundaries and named optical signal columns. Temperature, attenuation/loading,
flows and a reference optical amplitude are needed for adjusted comparisons.
Missing covariates can leave descriptive curves available while blocking a
matched comparison or predictive model. Missing timestamps and gaps are never
treated as observed minutes.

Raw MA350 input can be declared directly using `parameters.raw_ma350`:

```json
{
  "raw_ma350": {
    "bc_unit": "ng/m3",
    "include_spots": true,
    "cross_sections": {"UV": 24.069, "Blue": 19.070, "Green": 17.028, "Red": 14.091, "IR": 10.120}
  },
  "timezone": "America/Los_Angeles",
  "rh_provenance": "Reviewed MA350 Sample RH (%) field; sample-stream sensor",
  "config": {"time_col": "Time (UTC)"}
}
```

These cross sections match the existing MA350 export's attenuation reconstruction.
BC in ng/m³ multiplied by sigma and divided by 1,000 yields `b_atn` in Mm⁻¹.
This conversion is recorded in the result. It does not supply a scattering
correction or produce corrected `b_abs`. Prepared signals must already be in
Mm⁻¹ and explicitly declare `signal_quantity` as `b_atn` or `b_abs`.

The default event design requires a 5 percentage-point RH change over 10 minutes,
stable pre/post plateaus, contiguous support and a 60-minute recovery window.
Session, tape and recording gaps break an event. Matching compares opposite
directions at similar initial and late-stable RH, late-stable temperature/loading,
spot-flow ratio, total flow, prior optical amplitude, local hour and month,
without reusing matched events. The total-flow caliper is 15% by default; both
adjusted minute-model baselines also include measured total flow. Results retain rejected
candidates, matching support, channel/spot curves, and late-stable response.

Lag features use elapsed time rather than row offsets. Complete earlier local
days train the instantaneous and history models; complete later days evaluate
both on the same rows. Uncertainty resamples days. Finite negative optical
excursions remain informative for amplitude diagnostics; a logarithmic exponent
requires positive channel signals and audits nonpositive values separately.

An optional filter-window stage accepts reviewed `active_intervals`, a canonical
`filter_catalog`, `stream_site` and processing/observation evidence. It reuses the
existing interval and coverage machinery. Missing evidence blocks that stage.
An all-minute versus non-event-window mean is a selection sensitivity, not a
corrected concentration or a measured humidity bias. A raw file's name containing
“Cleaned” is not evidence that all rows are observed or all processing is known.

The MA200 humidity-response study motivates this experiment, including the
possibility of recovery and sensor-response effects. It does not authorize
transferring MA200 coefficients to the MA350:
[Düsing et al. (2019), DOI 10.5194/amt-12-5879-2019](https://amt.copernicus.org/articles/12/5879/2019/).

### 2. Calibration coverage

The response is independently measured source thermal-optical EC in **µg/filter**.
The experiment explicitly rejects an FTIR-derived reference or an incompatible
response unit. Entire test sites are selected before fitting and are identical
in every comparison. Inner model selection also separates sites.

Low/high regions use the outer training pool's 20th/80th loading percentiles.
The spectral region uses an outer-training-only spectral descriptor and its
upper 20% radius. At a fixed budget, a fully restored arm includes the region's
natural training-pool proportion; depletion removes some/all of those examples
and replaces them from a fixed complement ordering. This is a controlled
labelled-source experiment. Label-free acquisition is evaluated separately.

Ridge with PCA, PLS and random forest reuse the existing model-building utilities.
Every predictive pipeline fits preprocessing inside its current training fold.
Outputs include selected physical filters, inner folds, tuning, fixed test
membership, loading-bin errors, prediction slope/bias, and predicted versus
reference 90% spans. A larger span can overshoot; it is not automatically better.
Paired intervals resample held-out sites conditional on the fitted models.

Evidence that coverage matters in this source population guides sample
acquisition. It does not establish local transfer accuracy, determine the best
model for Addis, or justify fitting a correction to HIPS-derived equivalent BC.

### 3. Chemical regimes and within-event optical evolution

Filter input columns are `physical_id`, local `date`, `lot`, `ftir_ec_ugm3`,
`hips_fabs_Mm1`, `hips_source="direct_HIPS"`, and prespecified measured chemistry
and controls. The target disagreement is
`D = FTIR EC - direct HIPS absorption / MAC`, using the existing project MAC.
This target contains two different operational quantities; it is not true EC
error or an independent estimate of mass absorption efficiency.

`ChemicalMeasurement` requires a reviewed analyte, method, units, source evidence
and explicit `independence_verified=true`. Supported independent methods are IC,
XRF and ICP-MS. FTIR carbon, HIPS BC and FTIR/PMF factors cannot be passed off as
independent chemistry. The tracked ChemSpec source explicitly describes sulfate
as Ion Chromatography, whereas its EC/OC fields are FTIR products.

The model sequence fits an EC-line benchmark on training filters, adds PM mass,
season and lot controls, then adds the measured chemical main effect and a
centered EC-by-chemistry interaction. Centering is learned on training filters.
An interaction is compared with the chemistry-main-effect model, as well as the
controlled baseline. Complete later collection cycles are the default test;
partial-year exploratory variants are labelled in the cycle inventory. Physical
filters remain intact. Filter bootstrap intervals do not model dependence across
longer pollution episodes, so they remain exploratory.

The complementary minute-level test requires an external episode table with
`event_id`, timezone-aware `start_utc`/`end_utc`, `definition="independent"` and
`evidence`. It does not discover events by selecting extreme optical outcomes.
Overlapping episodes are rejected. Early/late comparisons match environmental
and instrument state, and events sharing a local day remain in one validation
block. Attenuation ratios yield an **attenuation exponent**, not atmospheric AAE.

Mixing-state-dependent absorption is a plausible competing explanation, not an
Addis finding established by this workflow. See the field study with independent
absorption, SP2 and coating information:
[Yuan et al. (2021), DOI 10.5194/acp-21-635-2021](https://acp.copernicus.org/articles/21/635/2021/).

### 4. Independent chemistry from the continuous record

This compares three prespecified arms: routine optical amplitude/weather/season
and lot, the same baseline plus spectral shape, and then within-window temporal
features. All arms use the same eligible physical filters and predict directly
measured chemistry. Target-derived features, FTIR products and source factors are
not acceptable independent predictors for this test.

Feature construction must follow the existing
[`active-interval matching contract`](active-interval-matching-2026-09-10.md).
The prepared feature table must include actual UTC collection bounds,
`sampling_schedule_verified`, `observation_provenance_verified`,
`processing_history_verified`, `window_evidence_id`,
`valid_observed_coverage_fraction`, `minimum_segment_valid_observed_coverage`,
`optical_quantity` and `optical_units`. Verification flags must explicitly be
true. The default valid-observed coverage gates are 0.75 overall and 0.5 in each
active segment. Invalid units, unverified inputs and incomplete windows remain
in the exclusion table. Intermittent sampling must use its active segments,
not the entire first-to-last envelope.

The baseline, shape and temporal column lists are explicit manifest settings;
they are not selected by searching test-period correlations. Train-only
preprocessing and later-cycle evaluation prevent repeat-filter and temporal
leakage. A successful result supports prediction at the validated collection
timescale. Minute-by-minute source mass and chemical-factor attribution would
require their own independent evidence.

### 5. Rain history and timing

Daily weather has local `date` and `rain_mm`. It supports rain in the preceding
1/3/7 calendar days and a separately named sample-day total. The antecedent
quantities exclude the sample day and future days. Sample-day rain never becomes
rain during actual collection. Missing/negative precipitation remains missing,
not dry weather; incomplete time-since-rain histories are censored.

Hourly data require explicit one-hour accumulation start/end timestamps plus
reviewed active collection intervals. The existing interval validator checks
source-file hashes, physical identities, local/UTC clocks and overlaps. Inactive
gaps are excluded. Partial hourly bins do not get a fabricated uniform rainfall
allocation. Without sufficient information, their actual-collection total stays
unavailable.

The rain comparison uses the same disagreement benchmark and concentration/season/lot
controls as the chemical-regime test. Connected rain episodes are kept together;
any training filters sharing a test event are purged. If too few training
filters survive, the comparison returns insufficient data. It does not change
the event definition or minimum count to force a regression. Transport-defined
source/background pairing remains a future input-dependent extension: this PR
does not infer trajectories from rain or sample dates.

### 6. Reference-sample design

The candidate matrix uses the same explicit physical identities and spectra
contract. Selection sees spectra, known lot/season/site strata, explicit QC flags
and an optional **pre-assay** loading proxy. Unknown reference EC and disagreement
outcomes do not select samples. The default loading proxy is spectral L2 norm,
which is not EC mass.

A panel contains disjoint representative and challenge subsets. The
representative subset records inclusion probabilities for all eligible
specimens; a small budget falls back to simple random sampling so no stratum
gets zero selection probability. The challenge subset covers low/high loading
proxies, spectral extremes and space-filling coverage. It is not used to estimate
population accuracy without a representative evaluation design.

Blanks and blind repeat assays have their own budget and readiness ledger.
Unavailable blanks remain explicit gaps. Replicate FTIR scans do not become new
reference assays. Every selected specimen still requires substrate, remaining
material and storage review. For a local study, lock a representative evaluation
subset before inspecting its reference outcomes and keep it out of subsequent
calibration fitting; challenge samples can guide model development separately.

The source simulation compares random, coverage and hybrid acquisition at equal
assay counts on identical held-out sites. Selection is made before reference
labels are revealed to each model. The simulation tests the design in the source
population, and can show that a strategy helps one model but harms another.
It is not a new local reference panel or a guarantee about the next assay batch.

### 7. Independent follow-up measurements

`measurement_plan` creates a requirement ledger for explicitly identified
candidates. Optional evidence rows contain `physical_id`, `method`,
`requirement`, `verified`, and an evidence citation. A check passes only with
explicit true verification and a nonempty citation. No plan contains new EC
observations, even if every protocol requirement is ready.

| Method | Evidence required before measurement | What it could distinguish |
|---|---|---|
| Offline AMS | Substrate compatibility, material remaining, storage, extraction blanks, recovery, method and identity | Independent organic composition associated with unusual spectra or disagreement |
| Thermal-optical EC | Compatible substrate, laboratory protocol, blanks, detection limit, uncertainty, identity and sampling equivalence | Local FTIR EC calibration/transfer accuracy |
| Photoacoustic absorption | Independent calibration and uncertainty, inlet losses, size fraction, wavelength, clocks and active intervals | Behavior of filter-based optics against an independent absorption method |
| SP2 refractory BC | Facility feasibility, mass calibration, detection-size range, inlet losses, size fraction, clocks and active intervals | Variation in absorption per measured refractory BC |

Archived PTFE does not automatically support thermal EC. SP2 and conventional
photoacoustic comparisons require appropriate airborne sampling; an archived
filter is not a substitute. A laboratory may need new paired collection rather
than an aliquot of old material. The offline AMS result is organic-composition
evidence and does not replace independent thermal EC.

`colocation_eligibility` requires independent method labels, matching size
fractions, explicit sampling/calibration evidence and timezone-aware intervals.
The common interval must cover at least 90% of **each** paired interval by
default. Optical comparison wavelengths must agree exactly; any interpolation
requires a separate reviewed product. Absorption-to-SP2 comparisons retain the
absorption wavelength. EC, equivalent BC and refractory BC remain distinct.

The output is an eligibility/readiness assessment. A method-agreement analysis,
uncertainty propagation or MAC estimate would follow only after the independent
measurements are acquired and their calibration evidence reviewed. The Clemson
handout remains a concrete facility lead, not a booked collaboration; the
[official SP2 facility page](https://cecas.clemson.edu/airqualitylab/sp2/) describes
the request route.

## Review and verification

The tests exercise the scientific failure modes: shuffled/mismatched spectral
IDs, repeated physical filters, held-out outcome perturbations, train-only
preprocessing and feature selection, exact budgets, gaps/tape boundaries,
elapsed-time lags, retained negative amplitude excursions, same-RH support,
independent method labels, unverified collection windows, future-rain leakage,
missing precipitation, shared rain episodes and incompatible paired intervals.
The synthetic CLI then exercises serialization and cross-stage table inputs.

```bash
uv run pytest tests/test_conference*.py \
  tests/test_active_interval_matching.py tests/test_seasonal_ml_checks.py \
  tests/test_seasonal_spectral_ml.py -q
```

The dedicated GitHub workflow runs those checks and the synthetic demonstration
without private inputs. Real source/field results are described in the separate
validation record; source/filter identities and raw data are not added to the PR.
