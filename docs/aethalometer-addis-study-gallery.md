# Addis study in the gallery

The **Addis study** page brings the temporal framework of
[Shlipak et al. (2026)](https://doi.org/10.1021/acsestair.6c00140) into the existing
gallery using its saved **Jacros MA350-0238 hourly IR BC** record. It shows what
can be inspected now and distinguishes those observations from the paper's
published Central and Jacros results.

Open **Investigations → Shlipak et al. (2026) · the saved Jacros record → Addis
study comparison**, or **Aethalometer → Addis study**. The direct route is
`#tab=aethalometer&aep=addis-study&net=focus`. The public Investigations Areas
view also includes a curated Addis study card; the other instrument pages
retain their existing visibility.

## What has been brought through

| View | What it shows | Interpretation |
|---|---|---|
| Study record and coverage | Daily or monthly BC and retained-hour coverage across the full April 2022–March 2025 calendar | Missing periods stay on the time axis and missing concentrations stay null. |
| Seasonal comparison | Available-hour BC summaries and support under the paper's `belg_feb` calendar or the repository's `dry_feb` alternative | The February choice is explicit and can be inspected without changing accepted historical results. |
| Local-hour patterns | Seasonal hourly patterns with contributing observation counts | Descriptive temporal association; event windows are included. |
| Festival traces | Meskel and Hidar Sitaten windows for the six published dates | The actual saved hours and missing hours remain visible; the two 2024 events are unavailable. |
| Published context | Reported Central/Jacros means, SDs, maxima and completeness | These describe the paper's populations, not matched agreement with this partial export. |
| Supplied materials | A source-by-source transfer review of `EC_BC.pptx` and `JPL-main.zip`, linked to existing Data quality and Sources pages | Each method keeps its source identity and implementation/readiness status. |
| Archived optical sensitivity | Eighteen saved Central/Jacros rows across nine fossil/biomass exponent pairs, selected with discrete controls | Archived ratio-of-sums results and bootstrap intervals, not new fractions from the gallery record. |
| Loading sensitivity | Analytical gain curves for the deck's three distinct constant-k candidates | An illustration over ATN1 0–60; no current concentration is corrected or refitted. |
| Historical EC/BC fits | Four fit summaries transcribed from slide 9 | FTIR-derived EC is the response; fit strength does not establish independent thermal-EC accuracy. |
| Implementation status | Available comparisons, implemented source model and missing observation inputs | No Central series or fossil/biomass fractions are invented. |

The ten figures use the gallery's shared chart/table components, tooltips,
export controls and dashboard registry. View selections are encoded in the
URL so a shared chart keeps its aggregation, calendar, event, archived alpha pair or loading candidate. Key
coverage and interpretation labels remain visible with Explain disabled.

## Observations and denominators

The page is generated from
`gallery/app/public/data/aethalometer/overview/hourly/ETAD.json` and its
`overview/index.json` metadata. In this export, **ETAD identifies the Jacros
record**, not the paper's separate Central deployment. The instrument identity
and the underlying minute-file path are declared in the output.

For the saved snapshot inspected on 9 November 2026:

| Measure | Value |
|---|---:|
| Study calendar | 1 April 2022, 00:00 to 1 April 2025, 00:00, end exclusive |
| Timezone | Africa/Addis_Ababa, UTC+3 |
| Full study calendar | 26,304 hours |
| Saved grid | 12 April 2022, 13:00 to 20 August 2024, 11:00, inclusive hour starts |
| Saved grid slots | 20,663 |
| Retained hourly observations | 18,280 |
| Retained hours / full study calendar | 69.50% |
| Retained hours / saved grid | 88.47% |
| Mean over retained hours | 8.147210 µg/m³ |
| Median over retained hours | 5.762 µg/m³ |
| Sample SD over retained hours | 7.580589 µg/m³ |
| Maximum observed hour | 73.861 µg/m³ |

These numbers are derived from the saved hourly JSON. The paper's Jacros mean
of **7.67 µg/m³** describes its longer study population. Its reported **93.9%**
completeness has an unresolved exact denominator; it is not treated as directly
comparable to either coverage fraction above.

The full calendar is retained in every monthly and daily series. In
particular, **July 2022 has only 109 of 744 hours (14.65%)**, and September
2024–March 2025 has no saved observations. Monthly statistics are available-hour
summaries without an additional completeness cutoff; the counts and coverage
are part of their interpretation.

### Processing inherited from the saved source

The upstream Overview exporter reads the Jacros minute CSV's UTC timestamps,
projects them to local standard time, and retains hourly means with at least
30 valid IR minute records. It includes finite signed minute values and does
not apply the separate Diurnal export's negative/three-SD minute screen. The
saved hourly export rounds concentrations to integer ng/m³. The new study
export converts those values to µg/m³ and does not recreate instrument QC from
raw records.

The new exporter preserves finite signed saved values and makes the inherited
rule explicit. Nonfinite or malformed input cannot silently become observed
zero. The generated provenance hashes identify the exact saved input bytes,
study profile and configuration used.

The longer `daily/ETAD-JACROS.json` file is not spliced into this hourly record.
Its daily support is not equivalent to verified hourly coverage. The spectral
`diurnal/aeth/ETAD.json` export also has a different channel-screening recipe
and an IR-only minute-count field, so it does not establish paired spectral
coverage for these IR observations.

## Seasons and events

The paper calendar (`belg_feb`) defines Bega as October–January, Belg as
February–May and Kiremt as June–September. The alternative (`dry_feb`) assigns
February to Bega. Both are read from the repository's named calendars; the
accepted configuration is not changed. Monthly calendar labels alone do not
establish measured rainfall or a causal emissions explanation.

Under `belg_feb`, available-hour seasonal means in this saved record are:

| Season | Retained hours | Full-study season hours | Mean BC (µg/m³) |
|---|---:|---:|---:|
| Bega | 5,584 | 8,856 | 6.19316 |
| Belg | 6,417 | 8,664 | 7.54273 |
| Kiremt | 6,279 | 8,784 | 10.50274 |

The seasonal and hourly summaries include event windows. They do not claim
to reproduce the paper's event-excluded climatologies.

Meskel uses **local noon on the cited date to noon the following day**, matching
the main paper's Figure 6 display window. Hidar Sitaten uses a declared
**local midnight-to-midnight display window**, also used by the supplied
archive's `Figures.ipynb`. The final paper's exact climatology exclusion
remains unresolved. All windows are end-exclusive and contain 24 hour
starts. Their means are labeled as means over available hours, with no new
minimum-hour acceptance rule. For example, Meskel 2022 has only **17/24** saved
hours. Its peak and partial trace remain inspectable without implying full
window coverage. Both 2024 festival dates occur after the saved record ends.

## What the supplied presentation and archive resolve

The [source-materials review](aethalometer-source-materials-review.md) records
slide and notebook evidence, file hashes and the current modules that already
implement related methods. The curated profile is
`research/ftir_hips_chem/workflows/profiles/ec_bc_jpl_review.json`; the exporter
hashes it and attaches its attributed summaries without changing any hourly
observation or paper benchmark.

The archive's executed optical-sensitivity table reports **5.8% biomass BC at
both sites for exponents 1.0/2.0**, and **7.7% at both sites for 1.0/1.8**.
The cleaning pipeline visibly calls the latter parameter pair; the separate
sensitivity notebook includes both. The saved statistic is
`100 × sum(Biomass BCc) / sum(IR BCc)` over finite paired rows. Its archived
bootstrap intervals use 5,000 independent row resamples, which do not preserve
time dependence or account for end-member uncertainty. Input files, paired
coverage, the wavelength pair and exact final-run lineage remain unavailable.

The deck provides three distinct k candidates: **0.011723**, **0.011846** and
**0.0129**. The gallery's analytical curve uses
`BCc / BC1 = 1 / (1 - k × ATN1)` and retains only positive denominators. This
curve is explicitly separate from observed concentration. The existing
minute-level spot-closure checks remain the route for internal instrument
diagnostics. A future calibration comparison needs raw minute fields, flow and
tape/status context, matched filter IDs and evaluation on separate time blocks.

## Rebuild and audit

```bash
uv run --no-sync aeth gallery rebuild --only addis-study
uv run --no-sync python -m pytest -q tests/test_addis_study_export.py
cd gallery/app
npm run index:charts
npm run typecheck
npm run test:source
```

`gallery/data/export_addis_study.py` writes
`gallery/app/public/data/aethalometer/addis-study.json`. It is registered in the
gallery rebuild pipeline and uses saved public exports, so regenerating this
comparison does not require the private Drive mirror. Its tests exercise
timezone boundaries, missing periods, units, support counts, seasonal choices,
event windows and input provenance.

The public Investigations catalog marks this as **Saved-export comparison**
(`diagnostic`). It contributes no inferred dependency edges. Private research
history is still omitted from public builds. Dashboard snapshots capture the
study export through the shared resource registry.

### Verification on 9 November 2026

- **95 focused Python regressions passed**, including the 29 new exporter
  cases, the upstream Overview checks, rebuild/section contracts and closeout
  workflow inputs, and 26 cases for safe failure and truthful reporting in the
  existing opt-in site-correction routine.
- **387 source-only Node tests and 91 public-data Node tests passed**, with no
  skipped or unavailable cases. The source suite includes the new seven-case
  display/data contract and public history/index checks.
- TypeScript, the public production build, Ruff and `git diff --check` passed.
  The public visibility/archive check found no private Research or ML data
  paths in compiled code and no private files in the 5,140-file archive.
- An independent calculation from the original IR JSON matched all **1,290
  statistical summary rows** and **144 event-hour values**, plus the five
  original source hashes. A further review verified all 18 archived sensitivity
  rows, the three deck candidates, four fit rows and 183 analytical response
  points; the supplied-materials addition leaves every observational summary
  and event value unchanged. This checks the saved-data derivation, not the
  paper's raw processing or its unprovided deployments.
- A managed-browser review confirmed the ten rendered views with Explain
  off, daily/monthly selection, synchronized named calendar controls, the
  incomplete Meskel 2022 label and the explicit unavailable Meskel 2024 state.
  The selected options appeared in the page URL and survived reload. Archived
  alpha-pair selection and loading-candidate selection worked, and a timeline
  PNG export completed. The public build's Investigations → Areas → Addis study
  card opens the intended study page with all ten views.

The public source and saved-export checks do not establish a full private-data
rebuild, exact study reproduction, independent target EC accuracy or mobile
browser acceptance.

## Next data-dependent steps

1. **Extend the identified hourly deployments.** Obtain the remaining Jacros
   observations through March 2025 and a separately identified Central record,
   including instrument serials, deployment intervals and the actual QC
   evidence. Then run the named comparison command on full-calendar and
   overlapping-hour populations.
2. **Connect the implemented optical source model to verified paired inputs.**
   Supply corrected absorption at a declared wavelength pair, or an explicitly
   documented attenuation proxy, with independent paired-hour support. Preserve
   invalid values, raw fractions, clipping flags and sensitivity to the fossil
   and biomass exponents. AAE class frequencies are not fractions of BC mass.
3. **Resolve the supporting methods before exact reproduction.** Confirm the
   paper's instrument-QC thresholds, wavelength/correction recipe, completeness
   denominator, source-fraction aggregation and event exclusions. Meteorology
   and trajectory/PSCF comparisons require their own input and provenance work.

The paper's MA350 measurements had no size-selective inlet. Optical BC in that
sampled aerosol stream does not establish PM2.5 thermal EC or independent
validation of the current Addis FTIR product. The existing
[paper-comparison guide](aethalometer-paper-comparison.md) documents the source
model and stricter deployment-aware comparison workflow.
